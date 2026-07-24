"""END-TO-END VAL_CONFIRM RUN -- full pipeline, verification layer off.

First run in the tuning phase producing the project's actual target
metric (Acc@GQA), not a retrieval-only proxy. Every prior number
(mIoP/mIoU/IoP@0.5) came from scripts/part3_tune.py, which never calls
the captioner, answerer, or Cerberus. This script uses the real pipeline
(captioner + answerer via iris.aria.generate(), the same call path
scripts/nextqa_single_video_eval.py exercises for one video), extended
to the full val_confirm split, all 12 currently-frozen hyperparameters
applied live from tuning/frozen_state.json, and Cerberus verification
explicitly disabled (cerberus_mode="none") -- an explicit choice, not an
oversight.

val_confirm is the held-out split (per split_manifest.json) never used
by any hyperparameter family above -- specifically meant to catch
overfitting to val_tune.

Deliberately NOT run through part3_tune.py's run_family()/FAMILIES
machinery (this isn't a grid sweep, no family selection) -- but reuses
its ingest/index-cache/config machinery pattern and eval/metrics.py's
predicted_span_from_frames_peak (Method D) and the canonical
nextgqa_metrics.py the same way part3_tune.py does, so results are
directly comparable to every prior family's numbers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import iris.aria as aria  # noqa: E402
import iris.query as iris_query  # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402
from iris.query_reformulation import parse_mc_answer, format_mc_label  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402

from part3_tune import (  # noqa: E402
    INGEST_RELEVANT_KEYS, load_frozen_state, TUNING_DIR,
)

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)

VIDEO_DIR = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"
# Dedicated fresh cache dir -- NOT tuning/index_cache/ or index_cache_scenespans/
# or any other cache used by any hyperparameter family. Must not exist before
# this run starts (checked in main()).
INDEX_CACHE_DIR = TUNING_DIR / "index_cache_val_confirm_e2e"

PER_QUESTION_CSV = TUNING_DIR / "val_confirm_e2e_per_question.csv"
REPORT_PATH = TUNING_DIR / "val_confirm_e2e_report.md"

PER_Q_FIELDNAMES = [
    "video", "qid", "type", "question",
    "pred_answer_idx", "pred_answer_label", "gold_answer_idx", "gold_answer_label",
    "acc_qa", "pred_span_start", "pred_span_end", "gold_spans", "iop", "iou",
    "acc_gqa_unverified", "used_clip_anchor", "raw_answer_nonempty",
    "retrieval_span_ms", "caption_answer_ms",
]


def build_mc_prompt(row: dict) -> str:
    return f"""You are answering a NExT-QA multiple-choice video question.

Use only the provided retrieved frame evidence. Do not use outside knowledge if
the evidence is insufficient.

Question:
{row["question"]}

Options:
A. {row["choices"][0]}
B. {row["choices"][1]}
C. {row["choices"][2]}
D. {row["choices"][3]}
E. {row["choices"][4]}

Return exactly this format:
ANSWER: <A|B|C|D|E>
REASON: <one short sentence grounded in the frame evidence>
"""


def make_e2e_config(frozen: dict) -> IRISConfig:
    """Same construction as part3_tune.py's make_config, but explicit
    about every field the task spec calls out (not silently relying on
    IRISConfig() defaults for anything the task named)."""
    cfg = IRISConfig()
    cfg.cerberus_mode = "none"
    cfg.ranking_mode = frozen.get("ranking_mode", "ppr")
    cfg.codec_conf_source = frozen.get("codec_conf_source", "packet_size")
    cfg.codec_conf_pictype_norm = True
    for key in ("retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
                "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
                "luma_entropy_weight", "persistence_threshold", "max_prominence"):
        setattr(cfg, key, frozen[key])
    return cfg


def ingest_config_hash(cfg: IRISConfig) -> str:
    payload = {k: getattr(cfg, k) for k in INGEST_RELEVANT_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_val_confirm_questions() -> list[dict]:
    split = json.loads((REPO / "split_manifest.json").read_text())
    confirm_videos = set(split["confirm_videos"])
    rows = list(csv.DictReader(open(REPO / "eval" / "data" / "nextqa" / "val.csv", newline="", encoding="utf-8")))
    gsub = json.loads((REPO / "eval" / "data" / "nextqa" / "gsub_val.json").read_text())
    out = []
    for r in rows:
        vid = r["video"]
        if vid not in confirm_videos:
            continue
        qid = r["qid"]
        vpath = VIDEO_DIR / f"{vid}.mp4"
        if not vpath.exists():
            continue
        gold = gsub.get(vid, {}).get("location", {}).get(qid)
        if not gold:
            continue
        out.append({
            "video": vid, "qid": qid, "question": r["question"], "type": r.get("type"),
            "choices": [r["a0"], r["a1"], r["a2"], r["a3"], r["a4"]],
            "gold_answer_idx": int(r["answer"]),
            "gold_spans": gold, "duration": gsub[vid]["duration"],
        })
    return out


def ensure_indexes_e2e(video_ids: list[str], cfg: IRISConfig, n_workers: int = 8) -> tuple[dict[str, str], int, int]:
    """Same shape as part3_tune.ensure_indexes but pointed at the fresh
    val_confirm_e2e cache dir. Returns (paths, n_fresh_ingests, n_cache_hits)."""
    h = ingest_config_hash(cfg)
    paths = {}
    todo = []
    n_cache_hits = 0
    for vid in video_ids:
        p = INDEX_CACHE_DIR / f"{vid}__{h}"
        if p.with_suffix(p.suffix + ".npz").exists():
            paths[vid] = str(p)
            n_cache_hits += 1
        else:
            todo.append(vid)

    if todo:
        print(f"[ingest] {len(todo)}/{len(video_ids)} videos need ingest under config-hash {h}", flush=True)

        def _do(vid: str):
            vpath = VIDEO_DIR / f"{vid}.mp4"
            idx = iris_ingest.ingest(str(vpath), cfg)
            out_path = INDEX_CACHE_DIR / f"{vid}__{h}"
            iris_ingest.save_index(idx, str(out_path))
            return vid

        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_do, vid): vid for vid in todo}
            for fut in as_completed(futs):
                vid = futs[fut]
                try:
                    fut.result()
                    paths[vid] = str(INDEX_CACHE_DIR / f"{vid}__{h}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[ingest FAIL] {vid}: {type(exc).__name__}: {exc}", flush=True)
                done += 1
                if done % 20 == 0:
                    print(f"[ingest] {done}/{len(todo)} done", flush=True)
    return paths, len(todo), n_cache_hits


def smoke_test_backend(cfg: IRISConfig) -> str:
    """Confirm the answerer backend is reachable before starting the full
    loop, per the task's explicit requirement not to discover this
    partway through."""
    raw = aria.generate(
        prompt="Return exactly this format:\nANSWER: <A|B|C|D|E>\nREASON: one sentence.\n\n"
               "Question: What color is the sky on a clear day?\nOptions:\nA. red\nB. blue\nC. green\nD. purple\nE. black",
        context="No frame evidence provided for this smoke test.",
        config=cfg,
    )
    if not raw or not raw.strip():
        raise RuntimeError("Smoke test: backend returned empty response")
    return raw


def main() -> None:
    state = load_frozen_state()
    frozen = state["frozen"]
    print(f"[setup] frozen hyperparameters read live from tuning/frozen_state.json: {frozen}", flush=True)

    required_keys = ["retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
                      "span_method", "span_method_half_width_s", "peak_distance", "peak_prominence",
                      "packet_size_weight", "motion_weight", "luma_entropy_weight",
                      "persistence_threshold", "max_prominence"]
    missing = [k for k in required_keys if k not in frozen]
    if missing:
        raise SystemExit(f"[setup] frozen_state.json missing expected keys: {missing}")

    if INDEX_CACHE_DIR.exists() and any(INDEX_CACHE_DIR.iterdir()):
        raise SystemExit(f"[setup] {INDEX_CACHE_DIR} already exists and is non-empty -- "
                          "this run must not reuse any prior cache. Aborting.")
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = make_e2e_config(frozen)
    print(f"[setup] config: cerberus_mode={cfg.cerberus_mode} ranking_mode={cfg.ranking_mode} "
          f"codec_conf_source={cfg.codec_conf_source} codec_conf_pictype_norm={cfg.codec_conf_pictype_norm} "
          f"answerer_backend={cfg.answerer_backend} answerer_endpoint={cfg.answerer_endpoint} "
          f"answerer_model={cfg.answerer_model}", flush=True)

    print("[setup] smoke-testing answerer backend before the full loop...", flush=True)
    smoke_raw = smoke_test_backend(cfg)
    print(f"[setup] smoke test OK, backend reachable. Raw response: {smoke_raw[:200]!r}", flush=True)

    half_width_s = float(frozen["span_method_half_width_s"])
    assert frozen["span_method"] == "D", f"expected span_method=D, got {frozen['span_method']!r}"

    questions = load_val_confirm_questions()
    video_ids = sorted({q["video"] for q in questions})
    print(f"[setup] val_confirm questions loaded: {len(questions)} usable (nominal 113 videos) "
          f"across {len(video_ids)} usable videos", flush=True)

    index_paths, n_fresh, n_hits = ensure_indexes_e2e(video_ids, cfg)
    print(f"[ingest] fresh_ingests={n_fresh} cache_hits={n_hits} "
          f"({'ALL FRESH -- OK' if n_hits == 0 else 'WARNING: cache hits found in a supposedly-fresh dir'})", flush=True)

    index_cache: dict = {}
    rows_out = []
    retrieval_ms_list, caption_answer_ms_list = [], []
    n_scored = 0
    n_answer_nonempty_sample = 0
    correct_qa = 0
    correct_gqa = 0
    iops, ious = [], []

    csv_f = open(PER_QUESTION_CSV, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=PER_Q_FIELDNAMES)
    csv_w.writeheader()

    t_run_start = time.perf_counter()
    for i, q in enumerate(questions, 1):
        vid = q["video"]
        if vid not in index_paths:
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(index_paths[vid])
        index = index_cache[vid]

        t0 = time.perf_counter()
        try:
            query_embedding, _ = iris_query._call_embed_query(q["question"], cfg)
            retrieved_frames, _ = iris_query._retrieve_with_l1(index, query_embedding, cfg)
            pred_span, used_clip_anchor = predicted_span_from_frames_peak(
                retrieved_frames, query_embedding, half_width_s=half_width_s,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL retrieval] video={vid} qid={q['qid']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        t_retrieval_span = (time.perf_counter() - t0) * 1000
        retrieval_ms_list.append(t_retrieval_span)

        t1 = time.perf_counter()
        try:
            decoded_for_captions = iris_query._ensure_captions(index, retrieved_frames, cfg)
        except TypeError:
            decoded_for_captions = iris_query._ensure_captions(index, retrieved_frames)
        cache_obj = iris_query.wrapper_init_l1_cache(cfg)
        iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
        context_text = cache_obj.as_context_text()

        prompt = build_mc_prompt(q)
        raw_answer = aria.generate(prompt=prompt, context=context_text, config=cfg)
        pred_idx = parse_mc_answer(raw_answer)
        t_caption_answer = (time.perf_counter() - t1) * 1000
        caption_answer_ms_list.append(t_caption_answer)

        gold_idx = q["gold_answer_idx"]
        acc_qa = nextgqa_metrics.acc_qa(pred_idx, gold_idx)
        gold_tuples = [(g[0], g[1]) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)
        acc_gqa = bool(acc_qa and iop >= 0.5)

        n_scored += 1
        correct_qa += int(acc_qa)
        correct_gqa += int(acc_gqa)
        iops.append(iop)
        ious.append(iou)
        if raw_answer and raw_answer.strip():
            n_answer_nonempty_sample += 1

        rows_out.append({
            "video": vid, "qid": q["qid"], "type": q.get("type"), "question": q["question"],
            "pred_answer_idx": pred_idx, "pred_answer_label": format_mc_label(pred_idx),
            "gold_answer_idx": gold_idx, "gold_answer_label": format_mc_label(gold_idx),
            "acc_qa": acc_qa, "pred_span_start": round(pred_span[0], 3), "pred_span_end": round(pred_span[1], 3),
            "gold_spans": json.dumps(q["gold_spans"]), "iop": round(iop, 5), "iou": round(iou, 5),
            "acc_gqa_unverified": acc_gqa, "used_clip_anchor": used_clip_anchor,
            "raw_answer_nonempty": bool(raw_answer and raw_answer.strip()),
            "retrieval_span_ms": round(t_retrieval_span, 2), "caption_answer_ms": round(t_caption_answer, 2),
        })
        csv_w.writerow(rows_out[-1])
        csv_f.flush()

        if i % 25 == 0 or i == len(questions):
            print(f"[{i}/{len(questions)}] scored={n_scored} Acc@QA_so_far={correct_qa/n_scored:.4f} "
                  f"Acc@GQA_so_far={correct_gqa/n_scored:.4f} mIoP_so_far={sum(iops)/len(iops):.4f}", flush=True)

    csv_f.close()
    total_wall_s = time.perf_counter() - t_run_start

    n = n_scored
    metrics = {
        "n_scored": n,
        "n_nominal_videos": 113,
        "n_usable_videos": len(video_ids),
        "n_fresh_ingests": n_fresh,
        "n_cache_hits": n_hits,
        "Acc@QA": correct_qa / n if n else 0.0,
        "Acc@GQA_unverified": correct_gqa / n if n else 0.0,
        "mIoP": sum(iops) / n if n else 0.0,
        "mIoU": sum(ious) / n if n else 0.0,
        "IoP@0.3": sum(1 for x in iops if x >= 0.3) / n if n else 0.0,
        "IoP@0.5": sum(1 for x in iops if x >= 0.5) / n if n else 0.0,
        "IoU@0.3": sum(1 for x in ious if x >= 0.3) / n if n else 0.0,
        "IoU@0.5": sum(1 for x in ious if x >= 0.5) / n if n else 0.0,
        "median_retrieval_span_ms": statistics.median(retrieval_ms_list) if retrieval_ms_list else 0.0,
        "p95_retrieval_span_ms": (statistics.quantiles(retrieval_ms_list, n=20)[18] if len(retrieval_ms_list) >= 20 else max(retrieval_ms_list, default=0.0)),
        "median_caption_answer_ms": statistics.median(caption_answer_ms_list) if caption_answer_ms_list else 0.0,
        "p95_caption_answer_ms": (statistics.quantiles(caption_answer_ms_list, n=20)[18] if len(caption_answer_ms_list) >= 20 else max(caption_answer_ms_list, default=0.0)),
        "total_wall_s": total_wall_s,
    }
    print("VAL_CONFIRM_E2E_METRICS_JSON=" + json.dumps(metrics), flush=True)
    print("VAL_CONFIRM_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
