"""BLIND ABLATION -- how much does the video actually contribute to Acc@QA?

Diagnostic measurement, not a tuning run. Runs the recorded val_confirm_e2e
configuration (all 12 frozen hyperparameters, cerberus_mode="none",
ranking_mode="ppr", codec_conf_source="packet_size", K=4, Method D
half_width_s=2.2) through four arms that vary ONLY the context given to the
answerer and (for arm C only) the system prompt:

  A_FULL          real captions for the 4 retrieved frames, default system prompt
  B_BLIND_STRICT  empty context, default system prompt
  C_BLIND_FAIR    empty context, system prompt with the "insufficient evidence"
                  instructions removed and a forced-choice instruction added
  D_SHUFFLE       real captions, but from a DIFFERENT (deranged-mapped) video

Arm A is captioned once; B/C/D reuse arm A's captions and do no captioning of
their own. The user-side ANSWER/REASON prompt (build_mc_prompt, imported
verbatim from scripts/val_confirm_e2e_eval.py) is byte-identical across all
four arms -- only context and (arm C) system_prompt vary.

Writes every artifact under tuning/blind_ablation/. Never opens or modifies
the six protected files listed in the task, and never touches the NExT-GQA
test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.aria as aria  # noqa: E402
import iris.query as iris_query  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.query_reformulation import parse_mc_answer, format_mc_label  # noqa: E402
from iris.retrieval_entry import retrieve_for_question  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402

from part3_tune import load_frozen_state  # noqa: E402
from val_confirm_e2e_eval import (  # noqa: E402
    make_e2e_config, ensure_indexes_e2e, load_val_confirm_questions,
    build_arg_parser, build_mc_prompt, INDEX_CACHE_DIR,
    PER_QUESTION_CSV as RECORDED_PER_QUESTION_CSV,
)

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)

OUT_DIR = REPO / "tuning" / "blind_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHUFFLE_SEED = 20260726
BOOTSTRAP_SEED = 20260726
N_BOOTSTRAP = 2000

REFUSAL_RE = re.compile(
    r"insufficient|cannot determine|not enough|unable to|no evidence|don't have access|do not have access",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Arm C system prompt: same three "insufficient evidence" constraints removed,
# forced-choice instruction added. User-side build_mc_prompt is untouched --
# only this system prompt and the empty context differ from arm B.
# ---------------------------------------------------------------------------
ARM_C_SYSTEM_PROMPT = (
    "You are ARIA, a video understanding assistant.\n\n"
    "You are answering a multiple-choice question about a video. You do not "
    "have access to the video, its frames, or any retrieval context.\n\n"
    "Choose the single most plausible option based on the question and the "
    "answer options alone.\n\n"
    "You must pick exactly one option. Do not abstain and do not say the "
    "evidence is insufficient -- a best guess is required.\n\n"
    "Answer in clear natural language.\n\n"
)

PER_Q_FIELDNAMES = [
    "video", "qid", "type", "question", "arm", "raw_answer",
    "pred_answer_idx", "gold_answer_idx", "acc_qa", "parse_failed", "refused",
    "context_char_len", "answer_ms",
    # extras beyond the required minimum, kept for audit / fairness-test use
    "user_prompt", "iop", "iou", "acc_gqa_unverified",
]


def build_derangement(items: list[str], seed: int) -> dict[str, str]:
    """Sattolo's algorithm: produces a single n-cycle over `items`, which is
    a derangement (no fixed points) for any n > 1, with no rejection loop
    needed. Deterministic given `seed`."""
    rng = random.Random(seed)
    n = len(items)
    idxs = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(0, i)
        idxs[i], idxs[j] = idxs[j], idxs[i]
    mapping = {items[i]: items[idxs[i]] for i in range(n)}
    assert all(k != v for k, v in mapping.items()), "derangement has a fixed point"
    assert sorted(mapping.values()) == sorted(items), "mapping is not a bijection"
    return mapping


def make_row(vid, qid, qtype, question, arm, raw_answer, pred_idx, gold_idx,
             context_len, answer_ms, user_prompt, iop=None, iou=None, acc_gqa=None):
    parse_failed = pred_idx is None
    acc_qa = int(pred_idx is not None and pred_idx == gold_idx)
    refused = bool(REFUSAL_RE.search(raw_answer or ""))
    return {
        "video": vid, "qid": qid, "type": qtype, "question": question, "arm": arm,
        "raw_answer": raw_answer, "pred_answer_idx": pred_idx, "gold_answer_idx": gold_idx,
        "acc_qa": acc_qa, "parse_failed": parse_failed, "refused": refused,
        "context_char_len": context_len, "answer_ms": round(answer_ms, 2),
        "user_prompt": user_prompt, "iop": iop, "iou": iou, "acc_gqa_unverified": acc_gqa,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_Q_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bootstrap_delta(rows_a: list[dict], rows_b: list[dict], seed: int = BOOTSTRAP_SEED,
                     n_resamples: int = N_BOOTSTRAP, subset_keys: set | None = None):
    """Paired Acc@QA delta (a - b), bootstrapped over VIDEO clusters (not
    questions). subset_keys, if given, restricts to a set of (video, qid)
    pairs (e.g. the IoP>=0.5 grounded subset)."""
    acc_a = {(r["video"], r["qid"]): r["acc_qa"] for r in rows_a}
    acc_b = {(r["video"], r["qid"]): r["acc_qa"] for r in rows_b}
    keys = set(acc_a) & set(acc_b)
    if subset_keys is not None:
        keys &= subset_keys
    video_to_qids: dict[str, list[str]] = defaultdict(list)
    for (v, q) in keys:
        video_to_qids[v].append(q)
    videos = sorted(video_to_qids)
    if not videos:
        return {"n_questions": 0, "observed_delta": None, "ci_low": None, "ci_high": None}

    def delta_for(video_list):
        a_correct = a_total = b_correct = b_total = 0
        for v in video_list:
            for qid in video_to_qids[v]:
                a_correct += acc_a[(v, qid)]
                b_correct += acc_b[(v, qid)]
                a_total += 1
                b_total += 1
        if a_total == 0:
            return None
        return a_correct / a_total - b_correct / b_total

    observed = delta_for(videos)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_resamples):
        sample = [rng.choice(videos) for _ in range(len(videos))]
        d = delta_for(sample)
        if d is not None:
            deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (None, None)
    n_q = sum(len(v) for v in video_to_qids.values())
    return {
        "n_questions": n_q, "n_videos": len(videos),
        "observed_delta": observed,
        "ci_low": float(lo) if lo is not None else None,
        "ci_high": float(hi) if hi is not None else None,
        "n_resamples_used": len(deltas),
    }


def arm_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    n_correct = sum(r["acc_qa"] for r in rows)
    n_parsed = sum(1 for r in rows if not r["parse_failed"])
    n_correct_parsed = sum(r["acc_qa"] for r in rows if not r["parse_failed"])
    n_refused = sum(1 for r in rows if r["refused"])
    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    type_breakdown = {
        t: {"n": len(rs), "acc_qa": sum(x["acc_qa"] for x in rs) / len(rs)}
        for t, rs in by_type.items()
    }
    letter_dist: dict[str, int] = defaultdict(int)
    for r in rows:
        letter = format_mc_label(r["pred_answer_idx"]) if r["pred_answer_idx"] is not None else "PARSE_FAIL"
        letter_dist[letter] += 1
    return {
        "n": n,
        "acc_qa_all": n_correct / n if n else 0.0,
        "acc_qa_parsed_only": n_correct_parsed / n_parsed if n_parsed else None,
        "n_parsed": n_parsed,
        "parse_failure_rate": 1 - n_parsed / n if n else None,
        "refusal_rate": n_refused / n if n else None,
        "refusal_regex": REFUSAL_RE.pattern,
        "type_breakdown": type_breakdown,
        "answer_letter_distribution": dict(letter_dist),
    }


def main() -> None:
    t_script_start = time.perf_counter()
    print("[setup] loading frozen state and building config (identical to recorded val_confirm run)", flush=True)
    frozen = load_frozen_state()["frozen"]
    args_ns = build_arg_parser().parse_args([])  # defaults == the recorded baseline's flags (none passed)
    cfg = make_e2e_config(frozen, args_ns)
    print(f"[setup] cfg: cerberus_mode={cfg.cerberus_mode} ranking_mode={cfg.ranking_mode} "
          f"codec_conf_source={cfg.codec_conf_source} l2_retrieve_top_k={cfg.l2_retrieve_top_k} "
          f"answerer_model={cfg.answerer_model} answerer_endpoint={cfg.answerer_endpoint}", flush=True)

    questions = load_val_confirm_questions()
    assert len(questions) == 639, f"expected 639 val_confirm questions, got {len(questions)}"
    video_ids = sorted({q["video"] for q in questions})
    assert len(video_ids) == 112, f"expected 112 videos, got {len(video_ids)}"

    print(f"[setup] {len(questions)} questions across {len(video_ids)} videos; reusing existing index cache "
          f"({INDEX_CACHE_DIR}) -- expecting 0 fresh ingests", flush=True)
    index_paths, n_fresh, n_hits = ensure_indexes_e2e(video_ids, cfg)
    print(f"[ingest] fresh_ingests={n_fresh} cache_hits={n_hits}", flush=True)
    if n_fresh != 0:
        raise SystemExit(
            f"[FATAL] {n_fresh} videos required fresh ingest -- expected all 112 to be cache hits "
            "under the recorded config-hash. Aborting rather than silently rebuilding the index cache."
        )
    if n_hits != 112:
        raise SystemExit(f"[FATAL] expected 112 cache hits, got {n_hits}. Aborting.")

    # ------------------------------------------------------------------
    # ARM A -- real captions, default system prompt. Captioning happens
    # here ONLY. Also builds captions_dump.json and the per-(video,qid)
    # context_text cache that arm D reuses (no recaptioning for B/C/D).
    # ------------------------------------------------------------------
    print("[arm A] starting full pipeline (retrieval + captioning + answerer)...", flush=True)
    index_cache: dict = {}
    armA_rows: list[dict] = []
    captions_dump: dict = {}
    context_cache: dict[tuple[str, str], str] = {}
    first_qid_of_video: dict[str, str] = {}
    half_width_s = float(frozen["span_method_half_width_s"])

    # ------------------------------------------------------------------
    # Checkpoint: one JSON line per completed question, appended and
    # flushed immediately, so a killed/interrupted run can resume without
    # redoing already-scored questions (captioning + answerer calls are
    # the expensive part; this makes that work durable).
    # ------------------------------------------------------------------
    checkpoint_path = OUT_DIR / "arm_A_checkpoint.jsonl"
    done_keys: set[tuple[str, str]] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                vid, qid = rec["video"], rec["qid"]
                armA_rows.append(rec["row"])
                context_cache[(vid, qid)] = rec["context_text"]
                captions_dump.setdefault(vid, {})[qid] = rec["frame_captions"]
                first_qid_of_video.setdefault(vid, qid)
                done_keys.add((vid, qid))
        print(f"[arm A] resuming: {len(done_keys)} questions already completed per checkpoint "
              f"({checkpoint_path})", flush=True)
    checkpoint_f = open(checkpoint_path, "a")

    t0_arm = time.perf_counter()
    n_processed_this_run = 0
    for i, q in enumerate(questions, 1):
        vid, qid = q["video"], q["qid"]
        if (vid, qid) in done_keys:
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(index_paths[vid])
        index = index_cache[vid]

        retrieved_frames, plan, telemetry = retrieve_for_question(
            q["question"], index, cfg, type_code=q.get("type"), family=q.get("family"),
        )
        query_embedding, _ = iris_query._call_embed_query(q["question"], cfg)
        pred_span, used_clip_anchor = predicted_span_from_frames_peak(
            retrieved_frames, query_embedding, half_width_s=half_width_s,
            duration_s=q["duration"],
        )
        # question-blind captioning -- exactly reproduces the recorded run,
        # no question= / choices= kwargs.
        iris_query._ensure_captions(index, retrieved_frames, cfg)
        cache_obj = iris_query.wrapper_init_l1_cache(cfg)
        iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
        context_text = cache_obj.as_context_text()

        prompt = build_mc_prompt(q)
        t1 = time.perf_counter()
        raw_answer = aria.generate(prompt=prompt, context=context_text, config=cfg)
        answer_ms = (time.perf_counter() - t1) * 1000
        pred_idx = parse_mc_answer(raw_answer)
        gold_idx = q["gold_answer_idx"]

        gold_tuples = [(g[0], g[1]) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)
        acc_qa_bool = pred_idx is not None and pred_idx == gold_idx
        acc_gqa = bool(acc_qa_bool and iop >= 0.5)

        row = make_row(vid, qid, q.get("type"), q["question"], "A_FULL", raw_answer,
                        pred_idx, gold_idx, len(context_text), answer_ms, prompt,
                        iop=round(iop, 5), iou=round(iou, 5), acc_gqa=acc_gqa)
        armA_rows.append(row)

        frame_captions = {}
        for f in retrieved_frames:
            cap = f.get("caption")
            cap_text = cap.get("semantic_caption") if isinstance(cap, dict) else cap
            frame_captions[str(f["frame_idx"])] = cap_text
        captions_dump.setdefault(vid, {})[qid] = frame_captions
        context_cache[(vid, qid)] = context_text
        first_qid_of_video.setdefault(vid, qid)

        checkpoint_f.write(json.dumps({
            "video": vid, "qid": qid, "context_text": context_text,
            "frame_captions": frame_captions, "row": row,
        }) + "\n")
        checkpoint_f.flush()
        n_processed_this_run += 1

        if i % 50 == 0 or i == len(questions):
            n_correct_far = sum(r["acc_qa"] for r in armA_rows)
            n_done_far = len(armA_rows)
            print(f"[arm A] {n_done_far}/{len(questions)} acc_qa_so_far={n_correct_far/n_done_far:.4f}", flush=True)

    checkpoint_f.close()
    print(f"[arm A] done: {n_processed_this_run} newly processed this run, "
          f"{len(armA_rows)}/{len(questions)} total, in {time.perf_counter()-t0_arm:.1f}s", flush=True)
    assert len(armA_rows) == len(questions), f"expected {len(questions)} arm A rows, got {len(armA_rows)}"
    write_csv(OUT_DIR / "arm_A_full_per_question.csv", armA_rows)
    with open(OUT_DIR / "captions_dump.json", "w") as f:
        json.dump(captions_dump, f, indent=2)

    # ------------------------------------------------------------------
    # REPRODUCTION GATE
    # ------------------------------------------------------------------
    n_correct_A = sum(r["acc_qa"] for r in armA_rows)
    n_A = len(armA_rows)
    gate_pass = (n_correct_A == 349 and n_A == 639)
    print(f"[GATE] Arm A Acc@QA = {n_correct_A}/{n_A} = {n_correct_A/n_A:.4f} "
          f"(target 349/639 = 0.5462) -- {'PASS' if gate_pass else 'FAIL'}", flush=True)

    gate_report = {"n_correct": n_correct_A, "n_total": n_A, "target_correct": 349,
                   "target_total": 639, "pass": gate_pass}
    if not gate_pass:
        # Diff against the recorded per-question csv.
        recorded = {}
        with open(RECORDED_PER_QUESTION_CSV) as f:
            for r in csv.DictReader(f):
                recorded[(r["video"], r["qid"])] = r
        flips = []
        for r in armA_rows:
            key = (r["video"], r["qid"])
            rec = recorded.get(key)
            if rec is None:
                continue
            rec_acc = rec["acc_qa"] in ("True", "1", "true")
            if bool(r["acc_qa"]) != rec_acc:
                flips.append({
                    "video": r["video"], "qid": r["qid"],
                    "recorded_acc_qa": rec_acc, "new_acc_qa": bool(r["acc_qa"]),
                    "recorded_pred": rec.get("pred_answer_idx"), "new_pred": r["pred_answer_idx"],
                    "new_raw_answer": r["raw_answer"],
                })
        gate_report["flips"] = flips
        gate_report["n_flips"] = len(flips)
        with open(OUT_DIR / "reproduction_gate_FAILED.json", "w") as f:
            json.dump(gate_report, f, indent=2)
        print(f"[GATE FAIL] {len(flips)} questions flipped vs recorded run. "
              f"Details in {OUT_DIR / 'reproduction_gate_FAILED.json'}. STOPPING per task instructions.", flush=True)
        raise SystemExit(1)

    with open(OUT_DIR / "reproduction_gate.json", "w") as f:
        json.dump(gate_report, f, indent=2)

    # ------------------------------------------------------------------
    # Build the shuffle map for arm D (seeded derangement over 112 videos)
    # ------------------------------------------------------------------
    shuffle_map = build_derangement(video_ids, SHUFFLE_SEED)
    with open(OUT_DIR / "shuffle_map.json", "w") as f:
        json.dump(shuffle_map, f, indent=2, sort_keys=True)
    print(f"[arm D] shuffle_map built (seed={SHUFFLE_SEED}), verified fixed-point-free derangement over "
          f"{len(shuffle_map)} videos", flush=True)

    # ------------------------------------------------------------------
    # ARMS B, C, D -- no captioning, no retrieval. B/C use empty context;
    # D reuses arm A's context_text for the mapped video's first question.
    # ------------------------------------------------------------------
    armB_rows, armC_rows, armD_rows = [], [], []
    backend = aria.get_backend(cfg)

    bcd_checkpoint_path = OUT_DIR / "arm_BCD_checkpoint.jsonl"
    bcd_done_keys: set[tuple[str, str]] = set()
    if bcd_checkpoint_path.exists():
        with open(bcd_checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                armB_rows.append(rec["row_b"])
                armC_rows.append(rec["row_c"])
                armD_rows.append(rec["row_d"])
                bcd_done_keys.add((rec["video"], rec["qid"]))
        print(f"[arms B/C/D] resuming: {len(bcd_done_keys)} questions already completed per checkpoint "
              f"({bcd_checkpoint_path})", flush=True)
    bcd_checkpoint_f = open(bcd_checkpoint_path, "a")

    t0_bcd = time.perf_counter()
    n_bcd_processed_this_run = 0
    for i, q in enumerate(questions, 1):
        vid, qid = q["video"], q["qid"]
        if (vid, qid) in bcd_done_keys:
            continue
        gold_idx = q["gold_answer_idx"]
        prompt = build_mc_prompt(q)  # byte-identical to arm A's prompt for this qid

        # Arm B: empty context, default system prompt.
        t1 = time.perf_counter()
        raw_b = aria.generate(prompt=prompt, context="", config=cfg)
        ms_b = (time.perf_counter() - t1) * 1000
        pred_b = parse_mc_answer(raw_b)
        row_b = make_row(vid, qid, q.get("type"), q["question"], "B_BLIND_STRICT",
                          raw_b, pred_b, gold_idx, 0, ms_b, prompt)
        armB_rows.append(row_b)

        # Arm C: empty context, overridden system prompt (forced choice).
        t1 = time.perf_counter()
        raw_c = backend.generate(prompt=prompt, context="", system_prompt=ARM_C_SYSTEM_PROMPT,
                                  seed=42, keep_alive=0)
        ms_c = (time.perf_counter() - t1) * 1000
        pred_c = parse_mc_answer(raw_c)
        row_c = make_row(vid, qid, q.get("type"), q["question"], "C_BLIND_FAIR",
                          raw_c, pred_c, gold_idx, 0, ms_c, prompt)
        armC_rows.append(row_c)

        # Arm D: real captions from a DIFFERENT (deranged-mapped) video.
        mapped_vid = shuffle_map[vid]
        d_context = context_cache[(mapped_vid, first_qid_of_video[mapped_vid])]
        t1 = time.perf_counter()
        raw_d = aria.generate(prompt=prompt, context=d_context, config=cfg)
        ms_d = (time.perf_counter() - t1) * 1000
        pred_d = parse_mc_answer(raw_d)
        row_d = make_row(vid, qid, q.get("type"), q["question"], "D_SHUFFLE",
                          raw_d, pred_d, gold_idx, len(d_context), ms_d, prompt)
        armD_rows.append(row_d)

        bcd_checkpoint_f.write(json.dumps({
            "video": vid, "qid": qid, "row_b": row_b, "row_c": row_c, "row_d": row_d,
        }) + "\n")
        bcd_checkpoint_f.flush()
        n_bcd_processed_this_run += 1

        if len(armB_rows) % 100 == 0 or len(armB_rows) == len(questions):
            print(f"[arms B/C/D] {len(armB_rows)}/{len(questions)} done", flush=True)

    bcd_checkpoint_f.close()
    print(f"[arms B/C/D] done: {n_bcd_processed_this_run} newly processed this run, "
          f"{len(armB_rows)}/{len(questions)} total, in {time.perf_counter()-t0_bcd:.1f}s", flush=True)
    assert len(armB_rows) == len(armC_rows) == len(armD_rows) == len(questions)
    write_csv(OUT_DIR / "arm_B_blind_strict_per_question.csv", armB_rows)
    write_csv(OUT_DIR / "arm_C_blind_fair_per_question.csv", armC_rows)
    write_csv(OUT_DIR / "arm_D_shuffle_per_question.csv", armD_rows)

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------
    print("[stats] computing per-arm metrics and bootstrap deltas...", flush=True)
    recorded_iop: dict[tuple[str, str], float] = {}
    with open(RECORDED_PER_QUESTION_CSV) as f:
        for r in csv.DictReader(f):
            recorded_iop[(r["video"], r["qid"])] = float(r["iop"])

    grounded_keys = {k for k, v in recorded_iop.items() if v >= 0.5}
    ungrounded_keys = {k for k, v in recorded_iop.items() if v < 0.5}
    zero_keys = {k for k, v in recorded_iop.items() if v == 0.0}
    assert len(grounded_keys) == 198, len(grounded_keys)
    assert len(ungrounded_keys) == 441, len(ungrounded_keys)
    assert len(zero_keys) == 344, len(zero_keys)

    report = {
        "reproduction_gate": gate_report,
        "arms": {
            "A_FULL": arm_metrics(armA_rows),
            "B_BLIND_STRICT": arm_metrics(armB_rows),
            "C_BLIND_FAIR": arm_metrics(armC_rows),
            "D_SHUFFLE": arm_metrics(armD_rows),
        },
        "paired_deltas": {
            "A_minus_C": bootstrap_delta(armA_rows, armC_rows),
            "A_minus_D": bootstrap_delta(armA_rows, armD_rows),
            "D_minus_C": bootstrap_delta(armD_rows, armC_rows),
            "A_minus_B": bootstrap_delta(armA_rows, armB_rows),
        },
        "A_minus_C_subsets": {
            "grounded_iop_ge_0.5": bootstrap_delta(armA_rows, armC_rows, subset_keys=grounded_keys),
            "ungrounded_iop_lt_0.5": bootstrap_delta(armA_rows, armC_rows, subset_keys=ungrounded_keys),
            "zero_overlap_iop_eq_0": bootstrap_delta(armA_rows, armC_rows, subset_keys=zero_keys),
        },
        "bootstrap_config": {"seed": BOOTSTRAP_SEED, "n_resamples": N_BOOTSTRAP, "cluster": "video", "ci": "95% percentile"},
        "shuffle_config": {"seed": SHUFFLE_SEED, "n_videos": len(video_ids)},
        "total_script_wall_s": time.perf_counter() - t_script_start,
    }
    with open(OUT_DIR / "statistical_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[stats] A-C (headline) delta:", report["paired_deltas"]["A_minus_C"], flush=True)
    print("BLIND_ABLATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
