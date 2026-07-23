"""P-NOW-A held-out Acc@GQA — SECOND and FINAL touch of the TEST half.

Pre-registered in eval_results/P_NOWA_accgqa_prereg.md. Runs the frozen config
(graph_mode=flat, l2_retrieve_top_k=12, half_width=2.2, peak_source=
clip_in_ppr_top8, ppr_lambda=0.5, ppr_damping=0.5, motion_similarity_mode=
action_score) over the 27 TEST videos (eval_results/test_videos.txt, 120
questions) ONLY, for all three arms (proposed / uniform / random), with LLM
MC-answering via the seated llama-server (same path as pillar2_grounded_qa.py
and A6).

SEAM CHECK: the proposed-arm grounding aggregate (mIoP, IoP@0.5) must
reproduce eval_results/P_NOWA_test_raw.json (300d857) exactly, since
retrieval and span construction are unchanged between the two test touches.
A mismatch is written into the output under "seam_check" with pass=false;
do not commit raw output if that flag is false.

Single-cell mode -- do not add a grid here. This script is meant to run
exactly once against TEST. If TEST is ever run again, it is burned and must
be labelled as such per the pre-registration.

VERIFY: python scripts/pnowa_test_accgqa_run.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.aria as aria
import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved
from eval.mc_scorer import build_mc_prompt, parse_mc_answer, LETTERS
from eval.span import predict_span, _pick_by_clip_similarity
from scripts.pillar2_grounded_qa import (
    iou, iop, uniform_ts, bootstrap_paired_differences,
    preflight_backend, git_provenance, config_hash,
)

DATA_DIR = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
GQA_JSON = DATA_DIR / "gsub_val.json"
VAL_CSV = DATA_DIR / "val.csv"
VAL_VIDEO_LIST = REPO / "eval_results" / "val_videos.txt"
TEST_VIDEO_LIST = REPO / "eval_results" / "test_videos.txt"

# FROZEN -- must match 300d857 / P_NOWA_test_raw.json exactly.
TOP_K = 12
HALF_WIDTH = 2.2
SPAN_MODE = "ppr_peak"
PEAK_SOURCE = "clip_in_ppr_top8"
NUM_BOOT = 1000

# SEAM CHECK targets -- from eval_results/P_NOWA_test_raw.json (300d857).
SEAM_MIOP_EXPECTED = 0.3349
SEAM_IOP05_EXPECTED = 0.3500
SEAM_TOLERANCE = 1e-3


def _load_test_rows() -> list[dict]:
    val_videos = {l.strip() for l in open(VAL_VIDEO_LIST, encoding="utf-8") if l.strip()}
    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    assert not (val_videos & test_videos), "val/test video overlap -- split is corrupt"

    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(open(VAL_CSV, encoding="utf-8")))
    for r in rows:
        r["family"] = r["type"][0]

    grounded = [
        r for r in rows
        if r["video"] in test_videos
        and r["video"] in gsub
        and r["qid"] in gsub[r["video"]]["location"]
    ]
    used_videos = {r["video"] for r in grounded}
    leaked_val_videos = used_videos & val_videos
    assert not leaked_val_videos, f"VAL videos leaked into TEST run: {sorted(leaked_val_videos)}"
    return grounded


def video_clustered_bootstrap(values_by_video: dict[str, list[float]], n_boot: int = NUM_BOOT, seed: int = 20260722):
    import random
    videos = sorted(values_by_video.keys())
    rng = random.Random(seed)

    def _mean_of_sample(sample_videos):
        vals = []
        for v in sample_videos:
            vals.extend(values_by_video[v])
        return sum(vals) / len(vals) if vals else float("nan")

    point = _mean_of_sample(videos)
    boots = []
    for _ in range(n_boot):
        sample = [videos[rng.randrange(len(videos))] for _ in range(len(videos))]
        boots.append(_mean_of_sample(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def _caption_lines(frames_or_dicts) -> list[str]:
    lines = []
    for i, f in enumerate(frames_or_dicts, 1):
        if isinstance(f, dict):
            cap_val = f.get("caption") or ""
            ts = f.get("timestamp", 0.0)
        else:
            cap_val = f.caption or ""
            ts = f.timestamp
        cap = cap_val.get("semantic_caption") if isinstance(cap_val, dict) else cap_val
        cap = cap or "[CAPTION_FAILED]"
        lines.append(f"[Frame {i} @ {ts:.1f}s] {cap}")
    return lines


def main() -> None:
    print("=== P-NOW-A HELD-OUT Acc@GQA -- SECOND AND FINAL TEST-HALF TOUCH ===")

    cfg_proposed = IRISConfig(
        cerberus_mode="v2",
        l2_retrieve_top_k=TOP_K,
        ranking_mode="ppr",
        ppr_lambda=0.5,
        ppr_damping=0.5,
        candidate_thresh=0.08,
        l1_w_action=0.60,
        l1_w_query=0.25,
        l1_w_persist=0.15,
        l1_w_pagerank=0.0,
        l1_w_entropy=0.0,
        l1_w_hessian=0.0,
        l1_w_recency=0.0,
    )
    assert cfg_proposed.graph_mode == "flat"
    assert cfg_proposed.motion_similarity_mode == "action_score"

    print(f"\n[LLM] Seating {cfg_proposed.answerer_model} via llama-server at {cfg_proposed.answerer_endpoint}...")
    backend = aria.LlamaServerBackend(
        endpoint=cfg_proposed.answerer_endpoint,
        text_model=cfg_proposed.answerer_model,
    )
    aria.set_backend(backend)
    preflight_backend(backend)
    aria.run_diagnostics()

    grounded_rows = _load_test_rows()
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    n_videos = len(set(r["video"] for r in grounded_rows))
    print(f"[DATA] TEST grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in {r["video"] for r in grounded_rows}}

    print("\n[INDEX] Loading cached flat indexes...")
    loaded_indexes = {}
    for row in grounded_rows:
        vid = row["video"]
        if vid in loaded_indexes:
            continue
        try:
            loaded_indexes[vid] = iris_ingest.load_index(CACHE_DIR / vid)
        except Exception as e:
            print(f"  ERR loading {vid}: {e}")
            loaded_indexes[vid] = None

    question_results = []
    video_to_questions: dict[str, list[dict]] = {}
    per_video_iop: dict[str, list[float]] = {}
    per_video_iop05: dict[str, list[float]] = {}

    n_clip_fallback = 0
    n_clip_total = 0

    import random
    rand_gen = random.Random(20260710)

    print("\n[EVAL] Running queries against LLM (3 arms)...")
    for idx, row in enumerate(grounded_rows, 1):
        vid = row["video"]
        qid = str(row["qid"])
        family = row["family"]
        question = row["question"]
        opts = {k: row[k] for k in ["a0", "a1", "a2", "a3", "a4"]}
        gold_idx = int(row["answer"])
        gold_letter = LETTERS[gold_idx]

        index = loaded_indexes.get(vid)
        if index is None:
            continue

        gold_spans = gsub[vid]["location"][qid]
        duration = duration_by_vid.get(vid) or max(f.timestamp for f in index.frames)

        print(f"[{idx}/{len(grounded_rows)}] QID={qid} Video={vid} Family={family} Query: {question[:50]}...")

        # ─── A. Proposed Arm ───
        emb = _embed_query(question, cfg_proposed)
        retrieved_proposed = _build_retrieved(index, emb, cfg_proposed)

        n_clip_total += 1
        if _pick_by_clip_similarity(retrieved_proposed, emb) is None:
            n_clip_fallback += 1

        prompt_prop, context_prop = build_mc_prompt(question, opts, "\n".join(_caption_lines(retrieved_proposed)))
        raw_prop = aria.generate(prompt=prompt_prop, context=context_prop)
        parsed_prop = parse_mc_answer(raw_prop, opts)
        choice_prop = parsed_prop.parsed_letter

        span_prop = predict_span(
            retrieved_proposed, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration, peak_source=PEAK_SOURCE, query_embedding=emb,
        )
        iop_prop = iop(span_prop, gold_spans)
        iou_prop = iou(span_prop, gold_spans)
        acc_qa_prop = float(choice_prop == gold_letter)
        acc_gqa_prop = float(acc_qa_prop and iop_prop >= 0.5)

        per_video_iop.setdefault(vid, []).append(iop_prop)
        per_video_iop05.setdefault(vid, []).append(1.0 if iop_prop >= 0.5 else 0.0)

        # ─── B. Uniform Arm ───
        ts_uniform_targets = uniform_ts(duration, TOP_K)
        retrieved_uniform = []
        for target_t in ts_uniform_targets:
            closest_fr = min(index.frames, key=lambda fr: abs(fr.timestamp - target_t))
            retrieved_uniform.append(closest_fr)

        prompt_unif, context_unif = build_mc_prompt(question, opts, "\n".join(_caption_lines(retrieved_uniform)))
        raw_unif = aria.generate(prompt=prompt_unif, context=context_unif)
        parsed_unif = parse_mc_answer(raw_unif, opts)
        choice_unif = parsed_unif.parsed_letter

        span_unif = predict_span(retrieved_uniform, mode="minmax")
        iop_unif = iop(span_unif, gold_spans)
        iou_unif = iou(span_unif, gold_spans)
        acc_qa_unif = float(choice_unif == gold_letter)
        acc_gqa_unif = float(acc_qa_unif and iop_unif >= 0.5)

        # ─── C. Random Arm ───
        frames_pool = list(index.frames)
        q_rand = random.Random(int(qid) + 20260710)
        retrieved_random = q_rand.sample(frames_pool, min(len(frames_pool), TOP_K))
        retrieved_random.sort(key=lambda fr: fr.timestamp)

        prompt_rand, context_rand = build_mc_prompt(question, opts, "\n".join(_caption_lines(retrieved_random)))
        raw_rand = aria.generate(prompt=prompt_rand, context=context_rand)
        parsed_rand = parse_mc_answer(raw_rand, opts)
        choice_rand = parsed_rand.parsed_letter

        span_rand = predict_span(retrieved_random, mode="minmax")
        iop_rand = iop(span_rand, gold_spans)
        iou_rand = iou(span_rand, gold_spans)
        acc_qa_rand = float(choice_rand == gold_letter)
        acc_gqa_rand = float(acc_qa_rand and iop_rand >= 0.5)

        res_dict = {
            "qid": qid, "video": vid, "family": family, "question": question, "gold": gold_letter,
            "proposed_choice": choice_prop, "proposed_acc_qa": acc_qa_prop, "proposed_iop": iop_prop,
            "proposed_iou": iou_prop, "proposed_iop_05": float(iop_prop >= 0.5),
            "proposed_iou_05": float(iou_prop >= 0.5), "proposed_acc_gqa": acc_gqa_prop,
            "proposed_raw_response": parsed_prop.raw_response, "proposed_parse_path": parsed_prop.parse_path,
            "uniform_choice": choice_unif, "uniform_acc_qa": acc_qa_unif, "uniform_iop": iop_unif,
            "uniform_iou": iou_unif, "uniform_iop_05": float(iop_unif >= 0.5),
            "uniform_iou_05": float(iou_unif >= 0.5), "uniform_acc_gqa": acc_gqa_unif,
            "uniform_raw_response": parsed_unif.raw_response, "uniform_parse_path": parsed_unif.parse_path,
            "random_choice": choice_rand, "random_acc_qa": acc_qa_rand, "random_iop": iop_rand,
            "random_iou": iou_rand, "random_iop_05": float(iop_rand >= 0.5),
            "random_iou_05": float(iou_rand >= 0.5), "random_acc_gqa": acc_gqa_rand,
            "random_raw_response": parsed_rand.raw_response, "random_parse_path": parsed_rand.parse_path,
        }
        question_results.append(res_dict)
        video_to_questions.setdefault(vid, []).append(res_dict)

    # ── Aggregate ──────────────────────────────────────────────────────────
    n = len(question_results)
    fallback_rate = n_clip_fallback / n_clip_total if n_clip_total else 0.0
    print(f"\n[CLIP-ANCHOR] fallback_rate={fallback_rate:.4%} ({n_clip_fallback}/{n_clip_total})")

    def mean_val(arm: str, key: str) -> float:
        return float(np.mean([q[f"{arm}_{key}"] for q in question_results]))

    metric_keys = ["acc_gqa", "acc_qa", "iop", "iop_05", "iou", "iou_05"]
    metrics = {arm: {k: mean_val(arm, k) for k in metric_keys} for arm in ["proposed", "uniform", "random"]}

    # P(correct|grounded): among proposed-arm grounded (IoP>=0.5) questions, fraction correct.
    grounded_qs = [q for q in question_results if q["proposed_iop_05"] == 1.0]
    n_grounded = len(grounded_qs)
    p_correct_given_grounded = (
        float(np.mean([q["proposed_acc_qa"] for q in grounded_qs])) if n_grounded else float("nan")
    )

    print(f"\n[BOOTSTRAP] Running {NUM_BOOT} resamples for paired differences...")
    ci_results = bootstrap_paired_differences(video_to_questions, metric_keys, num_resamples=NUM_BOOT)

    acc_gqa_point, acc_gqa_lo, acc_gqa_hi = video_clustered_bootstrap(
        {vid: [q["proposed_acc_gqa"] for q in qs] for vid, qs in video_to_questions.items()}
    )
    acc_qa_point, acc_qa_lo, acc_qa_hi = video_clustered_bootstrap(
        {vid: [q["proposed_acc_qa"] for q in qs] for vid, qs in video_to_questions.items()}
    )

    # SEAM CHECK -- proposed-arm grounding aggregate must reproduce 300d857 exactly.
    seam_miop, seam_miop_lo, seam_miop_hi = video_clustered_bootstrap(per_video_iop)
    seam_iop05, seam_iop05_lo, seam_iop05_hi = video_clustered_bootstrap(per_video_iop05)
    seam_pass = (
        abs(seam_miop - SEAM_MIOP_EXPECTED) <= SEAM_TOLERANCE
        and abs(seam_iop05 - SEAM_IOP05_EXPECTED) <= SEAM_TOLERANCE
    )
    print(f"\n[SEAM CHECK] mIoP={seam_miop:.4f} (expected {SEAM_MIOP_EXPECTED}), "
          f"IoP@0.5={seam_iop05:.4f} (expected {SEAM_IOP05_EXPECTED}) -> {'PASS' if seam_pass else 'FAIL'}")

    # Parser path histogram (proposed arm).
    parse_path_hist: dict[str, int] = {}
    for q in question_results:
        p = q["proposed_parse_path"]
        parse_path_hist[p] = parse_path_hist.get(p, 0) + 1

    git_commit, git_dirty = git_provenance(REPO)
    provenance = {
        "backend_class": type(backend).__name__,
        "endpoint": backend.endpoint,
        "model": backend.text_model,
        "temperature": backend.temperature,
        "cache_prompt": backend.cache_prompt,
        "span_mode": SPAN_MODE,
        "span_half_width": HALF_WIDTH,
        "span_peak_source": PEAK_SOURCE,
        "ppr_lambda": cfg_proposed.ppr_lambda,
        "ppr_damping": cfg_proposed.ppr_damping,
        "graph_mode": cfg_proposed.graph_mode,
        "motion_similarity_mode": cfg_proposed.motion_similarity_mode,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash(cfg_proposed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": n,
        "n_videos": n_videos,
        "top_k": TOP_K,
        "num_boot": NUM_BOOT,
        "clip_anchor_fallback_rate": fallback_rate,
    }

    out = {
        "provenance": provenance,
        "seam_check": {
            "pass": seam_pass,
            "miop": seam_miop, "miop_ci": [seam_miop_lo, seam_miop_hi], "miop_expected": SEAM_MIOP_EXPECTED,
            "iop_05": seam_iop05, "iop_05_ci": [seam_iop05_lo, seam_iop05_hi], "iop_05_expected": SEAM_IOP05_EXPECTED,
        },
        "primary": {
            "acc_gqa": {"point": acc_gqa_point, "ci_lo": acc_gqa_lo, "ci_hi": acc_gqa_hi},
        },
        "secondary": {
            "acc_qa": {"point": acc_qa_point, "ci_lo": acc_qa_lo, "ci_hi": acc_qa_hi},
            "p_correct_given_grounded": p_correct_given_grounded,
            "n_grounded": n_grounded,
        },
        "overall_metrics": metrics,
        "bootstrap_paired_differences": ci_results,
        "parse_path_histogram": parse_path_hist,
        "results": question_results,
    }

    out_path = REPO / "eval_results" / "P_NOWA_accgqa_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[EXPORT] Saved raw metrics to: {out_path}")

    print("\n=== RESULT ===")
    print(f"  Acc@GQA (proposed) = {acc_gqa_point:.4f}  95% CI=[{acc_gqa_lo:.4f}, {acc_gqa_hi:.4f}]")
    print(f"  Acc@QA  (proposed) = {acc_qa_point:.4f}  95% CI=[{acc_qa_lo:.4f}, {acc_qa_hi:.4f}]")
    print(f"  P(correct|grounded) = {p_correct_given_grounded:.4f}  (n_grounded={n_grounded})")
    print(f"  n = {n} questions / {n_videos} videos")
    print(f"  SEAM CHECK: {'PASS' if seam_pass else 'FAIL -- DO NOT COMMIT, SURFACE TO USER'}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
