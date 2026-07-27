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

import argparse
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
from iris.retrieval_entry import retrieve_for_question  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402

from answerer_provenance import capture_answerer_provenance  # noqa: E402
from caption_dump_io import (  # noqa: E402
    apply_caption_load, load_caption_dump, record_caption_dump, write_caption_dump,
)
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


def output_paths_for_mode(query_mode: str, traversal_mode: str) -> tuple[Path, Path]:
    """The frozen baseline run (query_mode="none") keeps its historical
    filenames; every reformulation arm writes to its own suffixed pair so a
    non-baseline run can never silently overwrite the recorded held-out
    baseline artifacts."""
    if query_mode == "none" and traversal_mode == "none":
        return PER_QUESTION_CSV, REPORT_PATH
    suffix = f"_{query_mode}_{traversal_mode}"
    return (
        TUNING_DIR / f"val_confirm_e2e_per_question{suffix}.csv",
        TUNING_DIR / f"val_confirm_e2e_report{suffix}.md",
    )


PER_Q_FIELDNAMES = [
    "video", "qid", "type", "question",
    "pred_answer_idx", "pred_answer_label", "gold_answer_idx", "gold_answer_label",
    "acc_qa", "pred_span_start", "pred_span_end", "gold_spans", "iop", "iou",
    "acc_gqa_unverified", "used_clip_anchor", "raw_answer_nonempty",
    "retrieval_span_ms", "caption_answer_ms",
    # Query-reformulation telemetry (blank/none for the frozen baseline arm).
    "query_mode", "traversal_mode", "relation", "relation_source",
    "fallback_reason", "num_ppr_calls", "context_frame_count",
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


def make_e2e_config(frozen: dict, args: argparse.Namespace | None = None) -> IRISConfig:
    """Same construction as part3_tune.py's make_config, but explicit
    about every field the task spec calls out (not silently relying on
    IRISConfig() defaults for anything the task named).

    Query reformulation is opt-in and explicit: with no --query-mode the
    config is byte-identical to the pre-existing frozen baseline
    (query_reformulation_mode="none", temporal_traversal_mode="none"), which
    routes through retrieve_for_question's "none" branch -- the same
    _call_embed_query + _build_retrieved pair this script called directly
    before the flag existed.
    """
    cfg = IRISConfig()
    cfg.cerberus_mode = "none"
    cfg.ranking_mode = frozen.get("ranking_mode", "ppr")
    cfg.codec_conf_source = frozen.get("codec_conf_source", "packet_size")
    cfg.codec_conf_pictype_norm = True
    for key in ("retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
                "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
                "luma_entropy_weight", "persistence_threshold", "max_prominence"):
        setattr(cfg, key, frozen[key])

    if args is not None:
        cfg.query_reformulation_mode = args.query_mode
        cfg.temporal_traversal_mode = args.temporal_traversal_mode
        cfg.temporal_context_seconds = args.temporal_context_seconds
        cfg.temporal_scene_hops = args.temporal_scene_hops
        cfg.max_context_frames = args.max_context_frames
        cfg.multi_query_combine = args.multi_query_combine
        cfg.max_retrieval_queries = (
            min(args.max_queries, 3) if args.query_mode == "structured_v2" else args.max_queries
        )
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--query-mode", dest="query_mode", default="none",
        choices=["none", "legacy", "structured_v2"],
        help="Retrieval query mode, routed through "
             "iris.retrieval_entry.retrieve_for_question. 'none' (default) is "
             "the frozen verbatim-question baseline.",
    )
    p.add_argument(
        "--temporal-traversal-mode", dest="temporal_traversal_mode", default="none",
        choices=["none", "legacy_symmetric", "directional"],
        help="Temporal context expansion mode (default: none).",
    )
    p.add_argument("--temporal-context-seconds", dest="temporal_context_seconds", type=float, default=4.0)
    p.add_argument("--temporal-scene-hops", dest="temporal_scene_hops", type=int, default=1)
    p.add_argument("--max-context-frames", dest="max_context_frames", type=int, default=8)
    p.add_argument("--multi-query-combine", dest="multi_query_combine", default="weighted_max",
                   choices=["weighted_max", "logsumexp"])
    p.add_argument("--max-queries", dest="max_queries", type=int, default=3,
                   help="Retrieval query budget (hard-capped to 3 under structured_v2).")
    p.add_argument(
        "--caption-dump", dest="caption_dump", default=None,
        help="Write every generated caption to this path as "
             "{video: {qid: {frame_idx: caption}}} (same schema as "
             "tuning/blind_ablation/captions_dump.json). Opt-in; omitting "
             "this flag leaves behaviour unchanged.",
    )
    p.add_argument(
        "--caption-load", dest="caption_load", default=None,
        help="Load captions from a dump written by --caption-dump and skip "
             "captioning entirely. Errors loudly on any (video, qid, "
             "frame_idx) miss rather than silently captioning it live. "
             "Opt-in; omitting this flag leaves behaviour unchanged.",
    )
    p.add_argument(
        "--repeat", dest="repeat", type=int, default=1,
        help="Run the answer stage this many times over identical retrieval "
             "and (with --caption-load) identical captions, reporting "
             "mean/min/max/flip-count Acc@QA and Acc@GQA instead of a single "
             "point estimate. Default 1 preserves current single-run "
             "behaviour exactly.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    per_question_csv, _report_path = output_paths_for_mode(
        args.query_mode, args.temporal_traversal_mode
    )
    if per_question_csv.exists():
        raise SystemExit(
            f"[setup] {per_question_csv} already exists -- refusing to overwrite a "
            "recorded held-out artifact. Move or delete it first if this rerun is intended."
        )
    print(f"[setup] query_mode={args.query_mode} "
          f"temporal_traversal_mode={args.temporal_traversal_mode} -> {per_question_csv.name}", flush=True)

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

    cfg = make_e2e_config(frozen, args)
    print(f"[setup] config: cerberus_mode={cfg.cerberus_mode} ranking_mode={cfg.ranking_mode} "
          f"codec_conf_source={cfg.codec_conf_source} codec_conf_pictype_norm={cfg.codec_conf_pictype_norm} "
          f"answerer_backend={cfg.answerer_backend} answerer_endpoint={cfg.answerer_endpoint} "
          f"answerer_model={cfg.answerer_model}", flush=True)

    print("[setup] smoke-testing answerer backend before the full loop...", flush=True)
    smoke_raw = smoke_test_backend(cfg)
    print(f"[setup] smoke test OK, backend reachable. Raw response: {smoke_raw[:200]!r}", flush=True)

    from urllib.parse import urlparse
    parsed_endpoint = urlparse(cfg.answerer_endpoint)
    provenance = capture_answerer_provenance(
        endpoint=cfg.answerer_endpoint,
        port=parsed_endpoint.port or 8091,
        gguf_path=None,
        gguf_expected_sha256=None,
        sampler_params={
            "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": cfg.answerer_seed, "cache_prompt": False,
        },
    )
    env_suffix = (
        "" if (args.query_mode == "none" and args.temporal_traversal_mode == "none")
        else f"_{args.query_mode}_{args.temporal_traversal_mode}"
    )
    env_path = TUNING_DIR / f"val_confirm_e2e_environment{env_suffix}.json"
    env_path.write_text(json.dumps({"answerer_provenance": provenance}, indent=2))
    print(f"[setup] wrote answerer provenance to {env_path}", flush=True)

    half_width_s = float(frozen["span_method_half_width_s"])
    assert frozen["span_method"] == "D", f"expected span_method=D, got {frozen['span_method']!r}"

    questions = load_val_confirm_questions()
    video_ids = sorted({q["video"] for q in questions})
    print(f"[setup] val_confirm questions loaded: {len(questions)} usable (nominal 113 videos) "
          f"across {len(video_ids)} usable videos", flush=True)

    index_paths, n_fresh, n_hits = ensure_indexes_e2e(video_ids, cfg)
    print(f"[ingest] fresh_ingests={n_fresh} cache_hits={n_hits} "
          f"({'ALL FRESH -- OK' if n_hits == 0 else 'WARNING: cache hits found in a supposedly-fresh dir'})", flush=True)

    if args.repeat > 1 and not args.caption_load:
        raise SystemExit(
            "[setup] --repeat > 1 requires --caption-load so the caption stage is "
            "frozen and only the answerer varies across repeats -- otherwise a "
            "flip could come from either the captioner or the answerer and "
            "would be unattributable."
        )
    caption_load_dump = load_caption_dump(args.caption_load) if args.caption_load else None
    caption_dump_accumulator: dict = {} if args.caption_dump else None

    index_cache: dict = {}
    retrieval_ms_list = []
    n_answer_nonempty_sample = 0
    records: list[dict] = []  # one entry per successfully-retrieved-and-captioned question

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
            retrieved_frames, plan, telemetry = retrieve_for_question(
                q["question"], index, cfg,
                type_code=q.get("type"), family=q.get("family"),
            )
            # Span construction (frozen Method D) is deliberately anchored on
            # the VERBATIM question embedding in every query mode. Method D is
            # a frozen span parameter, not part of the variable under test, so
            # letting the anchoring embedding change with query_mode would vary
            # two things at once and make an mIoP delta unattributable.
            query_embedding, _ = iris_query._call_embed_query(q["question"], cfg)
            pred_span, used_clip_anchor = predicted_span_from_frames_peak(
                retrieved_frames, query_embedding, half_width_s=half_width_s,
                duration_s=q["duration"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL retrieval] video={vid} qid={q['qid']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        t_retrieval_span = (time.perf_counter() - t0) * 1000
        retrieval_ms_list.append(t_retrieval_span)

        t1 = time.perf_counter()
        if caption_load_dump is not None:
            apply_caption_load(index, retrieved_frames, vid, q["qid"], caption_load_dump)
        else:
            try:
                iris_query._ensure_captions(index, retrieved_frames, cfg)
            except TypeError:
                iris_query._ensure_captions(index, retrieved_frames)
        if caption_dump_accumulator is not None:
            record_caption_dump(retrieved_frames, vid, q["qid"], caption_dump_accumulator)
        cache_obj = iris_query.wrapper_init_l1_cache(cfg)
        iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
        context_text = cache_obj.as_context_text()
        t_caption_ms = (time.perf_counter() - t1) * 1000

        gold_idx = q["gold_answer_idx"]
        gold_tuples = [(g[0], g[1]) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)

        records.append({
            "video": vid, "qid": q["qid"], "type": q.get("type"), "question": q["question"],
            "choices": q["choices"],
            "gold_answer_idx": gold_idx, "gold_spans": q["gold_spans"],
            "pred_span": pred_span, "used_clip_anchor": used_clip_anchor,
            "iop": iop, "iou": iou, "context_text": context_text,
            "retrieval_span_ms": t_retrieval_span, "caption_ms": t_caption_ms,
            "query_mode": cfg.query_reformulation_mode, "traversal_mode": cfg.temporal_traversal_mode,
            "relation": getattr(plan, "relation", None), "relation_source": getattr(plan, "relation_source", None),
            "fallback_reason": getattr(plan, "fallback_reason", None),
            "num_ppr_calls": telemetry.get("num_ppr_calls"), "context_frame_count": len(retrieved_frames),
        })

        if i % 25 == 0 or i == len(questions):
            print(f"[{i}/{len(questions)}] retrieval+captioning complete for {len(records)} questions so far", flush=True)

    if caption_dump_accumulator is not None:
        write_caption_dump(args.caption_dump, caption_dump_accumulator)
        print(f"[setup] wrote caption dump to {args.caption_dump}", flush=True)

    def _answer_and_score_one(rec: dict) -> dict:
        prompt = build_mc_prompt(rec)
        t_ans = time.perf_counter()
        raw_answer = aria.generate(prompt=prompt, context=rec["context_text"], config=cfg)
        answer_ms = (time.perf_counter() - t_ans) * 1000
        pred_idx = parse_mc_answer(raw_answer)
        acc_qa = nextgqa_metrics.acc_qa(pred_idx, rec["gold_answer_idx"])
        acc_gqa = bool(acc_qa and rec["iop"] >= 0.5)
        return {
            "video": rec["video"], "qid": rec["qid"], "type": rec["type"], "question": rec["question"],
            "pred_answer_idx": pred_idx, "pred_answer_label": format_mc_label(pred_idx),
            "gold_answer_idx": rec["gold_answer_idx"], "gold_answer_label": format_mc_label(rec["gold_answer_idx"]),
            "acc_qa": acc_qa, "pred_span_start": round(rec["pred_span"][0], 3), "pred_span_end": round(rec["pred_span"][1], 3),
            "gold_spans": json.dumps(rec["gold_spans"]), "iop": round(rec["iop"], 5), "iou": round(rec["iou"], 5),
            "acc_gqa_unverified": acc_gqa, "used_clip_anchor": rec["used_clip_anchor"],
            "raw_answer_nonempty": bool(raw_answer and raw_answer.strip()),
            "retrieval_span_ms": round(rec["retrieval_span_ms"], 2),
            "caption_answer_ms": round(rec["caption_ms"] + answer_ms, 2),
            "query_mode": rec["query_mode"], "traversal_mode": rec["traversal_mode"],
            "relation": rec["relation"], "relation_source": rec["relation_source"],
            "fallback_reason": rec["fallback_reason"], "num_ppr_calls": rec["num_ppr_calls"],
            "context_frame_count": rec["context_frame_count"],
            "_answer_ms": answer_ms, "_raw_answer_nonempty": bool(raw_answer and raw_answer.strip()),
        }

    def _write_pass_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PER_Q_FIELDNAMES)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in PER_Q_FIELDNAMES})

    n_repeats = max(args.repeat, 1)
    all_passes: list[list[dict]] = []
    for rep in range(n_repeats):
        pass_rows = [_answer_and_score_one(rec) for rec in records]
        all_passes.append(pass_rows)
        n_scored = len(pass_rows)
        correct_qa = sum(r["acc_qa"] for r in pass_rows)
        correct_gqa = sum(r["acc_gqa_unverified"] for r in pass_rows)
        n_answer_nonempty_sample += sum(1 for r in pass_rows if r["_raw_answer_nonempty"])
        print(f"[repeat {rep + 1}/{n_repeats}] scored={n_scored} "
              f"Acc@QA={correct_qa / n_scored if n_scored else 0.0:.4f} "
              f"Acc@GQA={correct_gqa / n_scored if n_scored else 0.0:.4f}", flush=True)
        pass_csv_path = per_question_csv if n_repeats == 1 else (
            per_question_csv.with_name(per_question_csv.stem + f"_repeat{rep + 1}" + per_question_csv.suffix)
        )
        _write_pass_csv(pass_csv_path, pass_rows)

    total_wall_s = time.perf_counter() - t_run_start

    n = len(records)
    iops = [r["iop"] for r in records]
    ious = [r["iou"] for r in records]
    caption_answer_ms_list = [row["caption_answer_ms"] for row in all_passes[0]] if all_passes else []

    # Grounding metrics are deterministic (retrieval + span construction do
    # not vary across repeats) -- reported once, not once per repeat.
    metrics = {
        "n_scored": n,
        "n_repeats": n_repeats,
        "n_nominal_videos": 113,
        "n_usable_videos": len(video_ids),
        "n_fresh_ingests": n_fresh,
        "n_cache_hits": n_hits,
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
        "minicpm_truncation_stats": aria.get_minicpm_truncation_stats(),
    }

    if n_repeats == 1:
        pass_rows = all_passes[0] if all_passes else []
        correct_qa = sum(r["acc_qa"] for r in pass_rows)
        correct_gqa = sum(r["acc_gqa_unverified"] for r in pass_rows)
        metrics["Acc@QA"] = correct_qa / n if n else 0.0
        metrics["Acc@GQA_unverified"] = correct_gqa / n if n else 0.0
    else:
        acc_qa_per_rep = [sum(r["acc_qa"] for r in p) / n if n else 0.0 for p in all_passes]
        acc_gqa_per_rep = [sum(r["acc_gqa_unverified"] for r in p) / n if n else 0.0 for p in all_passes]

        def _flip_count(key: str) -> int:
            flips = 0
            for qidx in range(n):
                values = {all_passes[rep][qidx][key] for rep in range(n_repeats)}
                if len(values) > 1:
                    flips += 1
            return flips

        repeat_summary = {
            "n_repeats": n_repeats,
            "Acc@QA": {
                "mean": statistics.mean(acc_qa_per_rep), "min": min(acc_qa_per_rep), "max": max(acc_qa_per_rep),
                "per_repeat": acc_qa_per_rep, "n_flipped_questions": _flip_count("acc_qa"),
            },
            "Acc@GQA_unverified": {
                "mean": statistics.mean(acc_gqa_per_rep), "min": min(acc_gqa_per_rep), "max": max(acc_gqa_per_rep),
                "per_repeat": acc_gqa_per_rep, "n_flipped_questions": _flip_count("acc_gqa_unverified"),
            },
            "grounding_metrics_reported_once": {
                "mIoP": metrics["mIoP"], "mIoU": metrics["mIoU"],
                "IoP@0.5": metrics["IoP@0.5"], "IoU@0.5": metrics["IoU@0.5"],
            },
        }
        env_suffix_for_summary = (
            "" if (args.query_mode == "none" and args.temporal_traversal_mode == "none")
            else f"_{args.query_mode}_{args.temporal_traversal_mode}"
        )
        summary_path = TUNING_DIR / f"val_confirm_e2e_repeat_summary{env_suffix_for_summary}.json"
        summary_path.write_text(json.dumps(repeat_summary, indent=2))
        print(f"[repeat] wrote variance summary to {summary_path}", flush=True)
        metrics["repeat_summary"] = repeat_summary

    print("VAL_CONFIRM_E2E_METRICS_JSON=" + json.dumps(metrics), flush=True)
    print("VAL_CONFIRM_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
