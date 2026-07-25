"""Controlled val_tune ablation for Structured Query Reformulation V2.

Varies EXACTLY ONE thing across arms -- the query reformulation / temporal
traversal mode -- and holds every other knob at the values read live from
tuning/frozen_state.json. Retrieval for every arm goes through the single
canonical entry point iris.retrieval_entry.retrieve_for_question, so no arm
gets a bespoke retrieval path.

Arms:
  A  query_mode=none          traversal=none              (frozen raw baseline)
  B  query_mode=legacy        traversal=none
  C  query_mode=legacy        traversal=legacy_symmetric
  D  query_mode=structured_v2 traversal=none
  E  query_mode=structured_v2 traversal=directional
  F  collapses onto E by construction -- batched embeddings and the single
     PPR solve are NOT optional in this codebase's structured_v2 path
     (iris/retrieval_entry.py::_retrieve_structured_v2 always calls
     _call_embed_queries once and _multi_query_retrieve once). There is no
     config that turns them off, so a separate F arm would be a relabelled
     copy of E. It is reported as collapsed, never as an independent arm.

Fairness notes (both deliberate, both load-bearing):

1. Span construction is frozen Method D (span_method_half_width_s from
   frozen_state.json) and is anchored on the VERBATIM question embedding in
   every arm. Method D is a frozen span parameter, not the variable under
   test; letting the anchoring embedding change with query_mode would vary
   two things at once and make any mIoP delta unattributable.

2. Question text and NExT-QA type codes come from the dataset row
   (eval/data/nextqa/val.csv) only -- never from answer choices. Gold spans
   are used ONLY for scoring, never for retrieval or anchor selection.

Reuses part3_tune.py's load_val_tune_questions / make_config /
ensure_indexes / load_frozen_state rather than re-deriving the split or the
ingest hash.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import iris.query as iris_query  # noqa: E402
from iris.retrieval_entry import retrieve_for_question  # noqa: E402
from iris.query_reformulation import build_query_plan_v2, all_embedding_texts  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402

from part3_tune import (  # noqa: E402
    load_val_tune_questions, make_config, ensure_indexes, load_frozen_state,
    INDEX_CACHE_DIR, TUNING_DIR,
)

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)

VAL_CSV = REPO / "eval" / "data" / "nextqa" / "val.csv"
RAW_DIR = TUNING_DIR / "query_reformulation_v2_raw"
ABLATION_CSV = TUNING_DIR / "query_reformulation_v2_ablation.csv"
REPORT_MD = TUNING_DIR / "query_reformulation_v2_report.md"
PARSER_AUDIT_CSV = TUNING_DIR / "query_reformulation_v2_parser_audit.csv"

# Ingest-relevant frozen keys, forwarded verbatim into make_config so the
# ingest hash (and therefore the index cache) matches the frozen state.
FROZEN_CONFIG_KEYS = [
    "retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
    "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
    "luma_entropy_weight", "persistence_threshold", "max_prominence",
]

ARMS = {
    "A": {"query_mode": "none", "traversal": "none", "label": "raw baseline (frozen)"},
    "B": {"query_mode": "legacy", "traversal": "none", "label": "legacy, no expansion"},
    "C": {"query_mode": "legacy", "traversal": "legacy_symmetric", "label": "legacy + symmetric expansion"},
    "D": {"query_mode": "structured_v2", "traversal": "none", "label": "structured_v2, no traversal"},
    "E": {"query_mode": "structured_v2", "traversal": "directional", "label": "structured_v2 + directional"},
}

TYPE_CODES = ["CW", "CH", "TN", "TP", "TC"]
GOLD_AT_K = 4


def load_type_codes() -> dict[tuple[str, str], str]:
    """(video, qid) -> NExT-QA type code, from the dataset row only."""
    with open(VAL_CSV, newline="", encoding="utf-8") as fh:
        return {(r["video"], r["qid"]): r["type"] for r in csv.DictReader(fh)}


def family_of(type_code: str | None) -> str | None:
    """NExT-QA family is the type code's leading letter: C(ausal)/T(emporal)/
    D(escriptive). Used only by the legacy reformulator's family gate."""
    if not type_code:
        return None
    return type_code[0].upper() if type_code[0].upper() in {"C", "T", "D"} else None


def in_any_gold(ts: float, gold_spans: list) -> bool:
    return any(float(g[0]) <= ts <= float(g[1]) for g in gold_spans)


def duration_bucket(gold_spans: list) -> str:
    """Bucket by total gold-span duration (the localization target's width)."""
    width = max((float(g[1]) - float(g[0])) for g in gold_spans)
    if width < 2.0:
        return "<2s"
    if width < 5.0:
        return "2-5s"
    if width < 10.0:
        return "5-10s"
    return ">=10s"


def make_arm_config(frozen: dict, query_mode: str, traversal: str, args) -> object:
    overrides = {k: frozen[k] for k in FROZEN_CONFIG_KEYS}
    cfg = make_config(overrides)
    cfg.query_reformulation_mode = query_mode
    cfg.temporal_traversal_mode = traversal
    cfg.temporal_context_seconds = args.temporal_context_seconds
    cfg.temporal_scene_hops = args.temporal_scene_hops
    cfg.max_context_frames = args.max_context_frames
    cfg.multi_query_combine = args.multi_query_combine
    cfg.max_retrieval_queries = 3 if query_mode == "structured_v2" else args.legacy_max_queries
    cfg.query_trace_enabled = True
    return cfg


def run_arm(arm: str, questions: list[dict], index_paths: dict[str, str],
            index_cache: dict, frozen: dict, args) -> tuple[list[dict], dict]:
    spec = ARMS[arm]
    cfg = make_arm_config(frozen, spec["query_mode"], spec["traversal"], args)
    half_width_s = float(frozen["span_method_half_width_s"])

    rows: list[dict] = []
    n_fail = 0
    t_arm0 = time.perf_counter()

    for i, q in enumerate(questions, 1):
        vid = q["video"]
        if vid not in index_paths:
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(index_paths[vid])
        index = index_cache[vid]

        trace: dict = {}
        t0 = time.perf_counter()
        try:
            frames, plan, telemetry = retrieve_for_question(
                q["question"], index, cfg,
                type_code=q["type_code"], family=q["family"], trace=trace,
            )
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [FAIL retrieval] arm={arm} video={vid} qid={q['qid']}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        # Span anchor embedding: verbatim question in EVERY arm (see module docstring).
        try:
            query_embedding, _ = iris_query._call_embed_query(q["question"], cfg)
        except Exception:  # noqa: BLE001
            query_embedding = None
        pred_span, used_clip_anchor = predicted_span_from_frames_peak(
            frames, query_embedding, half_width_s=half_width_s,
        )

        gold_tuples = [(float(g[0]), float(g[1])) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)

        top_k_frames = frames[:GOLD_AT_K]
        gold_at_4 = any(in_any_gold(float(f["timestamp"]), q["gold_spans"]) for f in top_k_frames)
        anchor_at_1 = bool(frames) and in_any_gold(float(frames[0]["timestamp"]), q["gold_spans"])

        plan_t = (telemetry.get("plan") or {})
        rows.append({
            "arm": arm, "video": vid, "qid": q["qid"], "type_code": q["type_code"],
            "question": q["question"],
            "gold_spans": json.dumps(q["gold_spans"]),
            "duration_bucket": q["duration_bucket"],
            "pred_span_start": round(float(pred_span[0]), 4),
            "pred_span_end": round(float(pred_span[1]), 4),
            "iop": round(float(iop), 6), "iou": round(float(iou), 6),
            "gold_frame_at_4": bool(gold_at_4), "anchor_in_gold_at_1": bool(anchor_at_1),
            "used_clip_anchor": bool(used_clip_anchor),
            "n_frames_returned": len(frames),
            "retrieved_timestamps": json.dumps([round(float(f["timestamp"]), 3) for f in frames]),
            "zero_width_span": float(pred_span[0]) == float(pred_span[1]),
            "invalid_span": not (float(pred_span[1]) >= float(pred_span[0])),
            "retrieval_ms": round(retrieval_ms, 3),
            "num_ppr_calls": telemetry.get("num_ppr_calls"),
            "num_clip_texts_embedded": telemetry.get("num_clip_texts_embedded"),
            "embed_cache_hits": telemetry.get("embed_cache_hits"),
            "embed_cache_misses": telemetry.get("embed_cache_misses"),
            # structured_v2 parser telemetry (empty for arms A/B/C)
            "relation": plan_t.get("relation"),
            "relation_source": plan_t.get("relation_source"),
            "fallback_reason": plan_t.get("fallback_reason"),
            "parser_confidence": plan_t.get("parser_confidence"),
            "position_prior": json.dumps(plan_t.get("position_prior")) if plan_t else None,
            "occurrence_selector": plan_t.get("occurrence_selector"),
            "temporal_direction": plan_t.get("temporal_direction"),
            "anchor_queries": json.dumps(plan_t.get("anchor_queries")) if plan_t else None,
            "target_query": plan_t.get("target_query"),
        })

        if i % 250 == 0:
            done = len(rows)
            print(f"  [arm {arm}] {i}/{len(questions)} scored={done} "
                  f"gold@4_so_far={sum(r['gold_frame_at_4'] for r in rows)/max(done,1):.4f} "
                  f"mIoP_so_far={sum(r['iop'] for r in rows)/max(done,1):.4f}", flush=True)

    wall_s = time.perf_counter() - t_arm0
    return rows, {"arm": arm, "wall_s": wall_s, "n_fail": n_fail,
                  "query_mode": spec["query_mode"], "traversal": spec["traversal"]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    iops = [r["iop"] for r in rows]
    ious = [r["iou"] for r in rows]
    lat = [r["retrieval_ms"] for r in rows]
    ppr = [r["num_ppr_calls"] for r in rows if r["num_ppr_calls"] is not None]
    nframes = [r["n_frames_returned"] for r in rows]
    return {
        "n": n,
        "gold_frame_at_4": sum(r["gold_frame_at_4"] for r in rows) / n,
        "anchor_in_gold_at_1": sum(r["anchor_in_gold_at_1"] for r in rows) / n,
        "mIoP": sum(iops) / n,
        "mIoU": sum(ious) / n,
        "IoP@0.3": sum(1 for x in iops if x >= 0.3) / n,
        "IoP@0.5": sum(1 for x in iops if x >= 0.5) / n,
        "IoU@0.3": sum(1 for x in ious if x >= 0.3) / n,
        "IoU@0.5": sum(1 for x in ious if x >= 0.5) / n,
        "zero_width_count": sum(r["zero_width_span"] for r in rows),
        "zero_width_rate": sum(r["zero_width_span"] for r in rows) / n,
        "invalid_span_count": sum(r["invalid_span"] for r in rows),
        "mean_pred_span_width": sum(r["pred_span_end"] - r["pred_span_start"] for r in rows) / n,
        "median_retrieval_ms": statistics.median(lat),
        "p95_retrieval_ms": (statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else max(lat, default=0.0)),
        "mean_ppr_calls": (sum(ppr) / len(ppr)) if ppr else None,
        "max_ppr_calls": max(ppr) if ppr else None,
        "avg_context_frames": sum(nframes) / n,
        "max_context_frames_observed": max(nframes),
    }


def paired_bootstrap(rows_a: list[dict], rows_x: list[dict], metric: str,
                      n_boot: int = 2000, seed: int = 12345) -> dict:
    """Paired bootstrap over the questions BOTH arms scored. Returns the arm
    delta (x - a) with a percentile CI."""
    by_a = {(r["video"], r["qid"]): r for r in rows_a}
    by_x = {(r["video"], r["qid"]): r for r in rows_x}
    keys = sorted(set(by_a) & set(by_x))
    if not keys:
        return {}
    da = [float(by_a[k][metric]) for k in keys]
    dx = [float(by_x[k][metric]) for k in keys]
    n = len(keys)
    obs = sum(dx) / n - sum(da) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(dx[i] for i in idx) / n - sum(da[i] for i in idx) / n)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot) - 1]
    return {"n_paired": n, "delta": obs, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0)}


def bootstrap_ci_abs(rows: list[dict], metric: str, n_boot: int = 2000, seed: int = 12345) -> dict:
    vals = [float(r[metric]) for r in rows]
    if not vals:
        return {}
    n = len(vals)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"mean": sum(vals) / n,
            "ci_lo": means[int(0.025 * n_boot)],
            "ci_hi": means[int(0.975 * n_boot) - 1]}


def parser_diagnostics(questions: list[dict], frozen: dict, args) -> dict:
    """Step 4: build a QueryPlanV2 for EVERY val_tune question and histogram
    the parser's behavior. Pure parsing -- no retrieval, no GPU, no index."""
    cfg = make_arm_config(frozen, "structured_v2", "directional", args)
    rel_source = Counter()
    fallback = Counter()
    relation = Counter()
    n_position_prior = 0
    n_occurrence = 0
    embed_counts = Counter()
    per_type_fallback = defaultdict(Counter)
    over_budget = 0
    plans = []

    for q in questions:
        plan = build_query_plan_v2(q["question"], type_code=q["type_code"],
                                   family=q["family"], config=cfg)
        texts = list(all_embedding_texts(plan))
        rel_source[plan.relation_source] += 1
        relation[plan.relation] += 1
        fallback[plan.fallback_reason or "<none>"] += 1
        per_type_fallback[q["type_code"]][plan.fallback_reason or "<none>"] += 1
        if plan.position_prior is not None:
            n_position_prior += 1
        if plan.occurrence_selector is not None:
            n_occurrence += 1
        embed_counts[len(texts)] += 1
        if len(texts) > 3:
            over_budget += 1
        plans.append((q, plan, texts))

    return {
        "n_questions": len(questions),
        "relation_source": dict(rel_source),
        "relation": dict(relation),
        "fallback_reason": dict(fallback),
        "n_position_prior": n_position_prior,
        "n_occurrence_selector": n_occurrence,
        "embedding_count_histogram": {str(k): v for k, v in sorted(embed_counts.items())},
        "over_embedding_budget": over_budget,
        "per_type_fallback": {k: dict(v) for k, v in per_type_fallback.items()},
        "_plans": plans,
    }


def write_parser_audit(plans, path: Path, per_type: int = 20, seed: int = 4242) -> int:
    """Stratified sample balanced across CW/CH/TN/TP/TC for human hand-check."""
    by_type = defaultdict(list)
    for q, plan, texts in plans:
        by_type[q["type_code"]].append((q, plan, texts))
    rng = random.Random(seed)
    picked = []
    for tc in TYPE_CODES:
        pool = by_type.get(tc, [])
        rng.shuffle(pool)
        picked.extend(pool[:per_type])
    fields = ["type_code", "video", "qid", "question", "relation", "relation_source",
              "temporal_direction", "parser_confidence", "fallback_reason",
              "anchor_clause", "target_clause", "position_prior", "occurrence_selector",
              "generated_prompts", "corrections_applied", "aliases_applied"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for q, plan, texts in picked:
            w.writerow({
                "type_code": q["type_code"], "video": q["video"], "qid": q["qid"],
                "question": q["question"], "relation": plan.relation,
                "relation_source": plan.relation_source,
                "temporal_direction": plan.temporal_direction,
                "parser_confidence": plan.parser_confidence,
                "fallback_reason": plan.fallback_reason or "",
                "anchor_clause": " | ".join(plan.anchor_queries),
                "target_clause": plan.target_query or "",
                "position_prior": json.dumps(plan.position_prior),
                "occurrence_selector": plan.occurrence_selector or "",
                "generated_prompts": json.dumps(texts),
                "corrections_applied": ";".join(plan.corrections_applied),
                "aliases_applied": ";".join(plan.aliases_applied),
            })
    return len(picked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="A,C,E", help="Comma-separated arms to run (default A,C,E).")
    p.add_argument("--limit", type=int, default=None, help="Smoke-test on the first N questions.")
    p.add_argument("--limit-videos", type=int, default=None, help="Smoke-test on the first N videos.")
    p.add_argument("--temporal-context-seconds", dest="temporal_context_seconds", type=float, default=4.0)
    p.add_argument("--temporal-scene-hops", dest="temporal_scene_hops", type=int, default=1)
    p.add_argument("--max-context-frames", dest="max_context_frames", type=int, default=8)
    p.add_argument("--multi-query-combine", dest="multi_query_combine", default="weighted_max",
                   choices=["weighted_max", "logsumexp"])
    p.add_argument("--legacy-max-queries", dest="legacy_max_queries", type=int, default=5)
    p.add_argument("--parser-only", action="store_true",
                   help="Run Step 4 parser diagnostics only (no retrieval, no GPU).")
    p.add_argument("--out-prefix", default="", help="Filename prefix for smoke-test outputs.")
    args = p.parse_args()

    frozen = load_frozen_state()["frozen"]
    print(f"[setup] frozen (live from tuning/frozen_state.json): {frozen}", flush=True)

    type_codes = load_type_codes()
    questions = load_val_tune_questions()
    for q in questions:
        tc = type_codes.get((q["video"], q["qid"]))
        q["type_code"] = tc
        q["family"] = family_of(tc)
        q["duration_bucket"] = duration_bucket(q["gold_spans"])

    if args.limit_videos:
        keep = sorted({q["video"] for q in questions})[: args.limit_videos]
        questions = [q for q in questions if q["video"] in set(keep)]
    if args.limit:
        questions = questions[: args.limit]

    video_ids = sorted({q["video"] for q in questions})
    print(f"[setup] val_tune: {len(questions)} questions across {len(video_ids)} videos", flush=True)
    print(f"[setup] type distribution: {dict(Counter(q['type_code'] for q in questions))}", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 4 parser diagnostics (cheap, always run; may explain everything)
    print("\n[parser] building QueryPlanV2 for every question (no retrieval)...", flush=True)
    diag = parser_diagnostics(questions, frozen, args)
    plans = diag.pop("_plans")
    print("[parser] relation_source histogram: " + json.dumps(diag["relation_source"]), flush=True)
    print("[parser] fallback_reason histogram: " + json.dumps(diag["fallback_reason"]), flush=True)
    print("[parser] relation histogram:        " + json.dumps(diag["relation"]), flush=True)
    print(f"[parser] position_prior set: {diag['n_position_prior']}/{diag['n_questions']}  "
          f"occurrence_selector set: {diag['n_occurrence_selector']}/{diag['n_questions']}", flush=True)
    print("[parser] embeddings/question histogram: " + json.dumps(diag["embedding_count_histogram"])
          + f"  (over budget: {diag['over_embedding_budget']})", flush=True)

    prefix = args.out_prefix
    diag_path = TUNING_DIR / f"{prefix}query_reformulation_v2_parser_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2))
    audit_path = TUNING_DIR / f"{prefix}query_reformulation_v2_parser_audit.csv"
    n_audit = write_parser_audit(plans, audit_path)
    print(f"[parser] wrote {diag_path.name} and {audit_path.name} ({n_audit} sampled questions)", flush=True)

    if args.parser_only:
        print("PARSER_ONLY_COMPLETE", flush=True)
        return

    cfg_for_ingest = make_arm_config(frozen, "none", "none", args)
    index_paths = ensure_indexes(video_ids, cfg_for_ingest)
    print(f"[setup] index cache: {len(index_paths)}/{len(video_ids)} videos available", flush=True)

    # Warm the CLIP text encoder BEFORE any arm is timed. The first
    # _call_embed_query in a process pays a one-off model load (~1.8s observed),
    # which otherwise lands entirely in whichever arm runs first and makes its
    # p95 meaningless for the latency acceptance check.
    warm_cfg = make_arm_config(frozen, "none", "none", args)
    t_warm = time.perf_counter()
    iris_query._call_embed_query("warmup", warm_cfg)
    print(f"[setup] CLIP text-encoder warmup: {(time.perf_counter()-t_warm)*1000:.1f} ms "
          "(excluded from all arm latencies)", flush=True)

    index_cache: dict = {}
    arm_rows: dict[str, list[dict]] = {}
    arm_meta: dict[str, dict] = {}

    for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            print(f"[skip] unknown arm {arm}", flush=True)
            continue
        spec = ARMS[arm]
        print(f"\n[arm {arm}] {spec['label']}: query_mode={spec['query_mode']} "
              f"traversal={spec['traversal']}", flush=True)
        rows, meta = run_arm(arm, questions, index_paths, index_cache, frozen, args)
        arm_rows[arm] = rows
        arm_meta[arm] = meta
        s = summarize(rows)
        print(f"[arm {arm}] DONE n={s['n']} gold@4={s['gold_frame_at_4']:.4f} "
              f"anchor@1={s['anchor_in_gold_at_1']:.4f} mIoP={s['mIoP']:.4f} mIoU={s['mIoU']:.4f} "
              f"IoP@0.5={s['IoP@0.5']:.4f} zero_width={s['zero_width_count']} "
              f"median_ms={s['median_retrieval_ms']:.2f} p95_ms={s['p95_retrieval_ms']:.2f} "
              f"wall={meta['wall_s']:.1f}s", flush=True)

        raw_path = RAW_DIR / f"{prefix}arm_{arm}_per_question.jsonl"
        with open(raw_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"[arm {arm}] raw -> {raw_path.relative_to(REPO)}", flush=True)

    write_outputs(arm_rows, arm_meta, diag, frozen, args, prefix)
    print("\nABLATION_COMPLETE", flush=True)


def write_outputs(arm_rows, arm_meta, diag, frozen, args, prefix: str) -> None:
    ablation_csv = TUNING_DIR / f"{prefix}query_reformulation_v2_ablation.csv"
    report_md = TUNING_DIR / f"{prefix}query_reformulation_v2_report.md"

    # ---- one row per arm x metric
    fields = ["arm", "query_mode", "traversal", "scope", "metric", "value", "n"]
    with open(ablation_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for arm, rows in arm_rows.items():
            meta = arm_meta[arm]
            base = {"arm": arm, "query_mode": meta["query_mode"], "traversal": meta["traversal"]}
            for k, v in summarize(rows).items():
                w.writerow({**base, "scope": "overall", "metric": k, "value": v, "n": len(rows)})
            for tc in TYPE_CODES:
                sub = [r for r in rows if r["type_code"] == tc]
                if not sub:
                    continue
                for k, v in summarize(sub).items():
                    w.writerow({**base, "scope": f"type={tc}", "metric": k, "value": v, "n": len(sub)})
            for b in ["<2s", "2-5s", "5-10s", ">=10s"]:
                sub = [r for r in rows if r["duration_bucket"] == b]
                if not sub:
                    continue
                for k, v in summarize(sub).items():
                    w.writerow({**base, "scope": f"gold_width={b}", "metric": k, "value": v, "n": len(sub)})

    # ---- report
    L = []
    L.append("# Structured Query Reformulation V2 -- val_tune ablation\n")
    L.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  ")
    L.append(f"Arms run: {', '.join(sorted(arm_rows))}  ")
    L.append(f"Frozen state (live from tuning/frozen_state.json): `{json.dumps(frozen)}`\n")
    L.append("Arm F collapses onto E: batched embeddings and the single PPR solve are not "
             "optional in the structured_v2 path (`_retrieve_structured_v2` always calls "
             "`_call_embed_queries` once and `_multi_query_retrieve` once), so a separate "
             "F arm would be a relabelled copy of E.\n")

    L.append("## Headline metrics\n")
    hdr = ["arm", "mode", "traversal", "n", "gold@4", "anchor@1", "mIoP", "mIoU",
           "IoP@0.3", "IoP@0.5", "zero-width", "mean span (s)", "med ms", "p95 ms", "PPR/q", "ctx frames"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for arm in sorted(arm_rows):
        s = summarize(arm_rows[arm]); m = arm_meta[arm]
        L.append("| " + " | ".join([
            arm, m["query_mode"], m["traversal"], str(s["n"]),
            f"{s['gold_frame_at_4']:.4f}", f"{s['anchor_in_gold_at_1']:.4f}",
            f"{s['mIoP']:.4f}", f"{s['mIoU']:.4f}", f"{s['IoP@0.3']:.4f}", f"{s['IoP@0.5']:.4f}",
            f"{s['zero_width_count']}", f"{s['mean_pred_span_width']:.2f}",
            f"{s['median_retrieval_ms']:.2f}", f"{s['p95_retrieval_ms']:.2f}",
            ("n/a" if s["mean_ppr_calls"] is None else f"{s['mean_ppr_calls']:.2f}"),
            f"{s['avg_context_frames']:.2f}",
        ]) + " |")

    if "A" in arm_rows:
        L.append("\n## Deltas vs arm A (paired bootstrap, 2000 resamples, seed 12345)\n")
        L.append("| arm | metric | delta | 95% CI | significant |")
        L.append("|---|---|---|---|---|")
        for arm in sorted(arm_rows):
            if arm == "A":
                continue
            for metric in ["gold_frame_at_4", "anchor_in_gold_at_1", "iop", "iou"]:
                bs = paired_bootstrap(arm_rows["A"], arm_rows[arm], metric)
                if not bs:
                    continue
                L.append(f"| {arm} | {metric} | {bs['delta']:+.4f} | "
                         f"[{bs['ci_lo']:+.4f}, {bs['ci_hi']:+.4f}] | "
                         f"{'YES' if bs['significant'] else 'no'} |")

        L.append("\n## Gained / lost vs arm A (gold_frame_at_4)\n")
        by_a = {(r["video"], r["qid"]): r for r in arm_rows["A"]}
        for arm in sorted(arm_rows):
            if arm == "A":
                continue
            by_x = {(r["video"], r["qid"]): r for r in arm_rows[arm]}
            keys = sorted(set(by_a) & set(by_x))
            gained = [k for k in keys if by_x[k]["gold_frame_at_4"] and not by_a[k]["gold_frame_at_4"]]
            lost = [k for k in keys if by_a[k]["gold_frame_at_4"] and not by_x[k]["gold_frame_at_4"]]
            L.append(f"\n**Arm {arm}**: gained {len(gained)}, lost {len(lost)}, net {len(gained)-len(lost):+d} "
                     f"(paired n={len(keys)})\n")
            L.append(f"- gained qids: `{json.dumps([f'{v}:{q}' for v, q in gained])}`")
            L.append(f"- lost qids: `{json.dumps([f'{v}:{q}' for v, q in lost])}`")

    L.append("\n## Per-type breakdown\n")
    L.append("| arm | type | n | gold@4 | anchor@1 | mIoP | mIoU | IoP@0.5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for arm in sorted(arm_rows):
        for tc in TYPE_CODES:
            sub = [r for r in arm_rows[arm] if r["type_code"] == tc]
            if not sub:
                continue
            s = summarize(sub)
            L.append(f"| {arm} | {tc} | {s['n']} | {s['gold_frame_at_4']:.4f} | "
                     f"{s['anchor_in_gold_at_1']:.4f} | {s['mIoP']:.4f} | {s['mIoU']:.4f} | {s['IoP@0.5']:.4f} |")

    L.append("\n## Gold-span duration buckets\n")
    L.append("| arm | bucket | n | gold@4 | mIoP | mIoU |")
    L.append("|---|---|---|---|---|---|")
    for arm in sorted(arm_rows):
        for b in ["<2s", "2-5s", "5-10s", ">=10s"]:
            sub = [r for r in arm_rows[arm] if r["duration_bucket"] == b]
            if not sub:
                continue
            s = summarize(sub)
            L.append(f"| {arm} | {b} | {s['n']} | {s['gold_frame_at_4']:.4f} | "
                     f"{s['mIoP']:.4f} | {s['mIoU']:.4f} |")

    L.append("\n## Parser diagnostics (Step 4)\n")
    L.append(f"- questions parsed: {diag['n_questions']}")
    L.append(f"- `relation_source`: `{json.dumps(diag['relation_source'])}`")
    L.append(f"- `relation`: `{json.dumps(diag['relation'])}`")
    L.append(f"- `fallback_reason`: `{json.dumps(diag['fallback_reason'])}`")
    L.append(f"- position_prior set: {diag['n_position_prior']}/{diag['n_questions']}")
    L.append(f"- occurrence_selector set: {diag['n_occurrence_selector']}/{diag['n_questions']}")
    L.append(f"- embeddings/question: `{json.dumps(diag['embedding_count_histogram'])}` "
             f"(over 3-budget: {diag['over_embedding_budget']})")
    L.append(f"- per-type fallback: `{json.dumps(diag['per_type_fallback'])}`")

    report_md.write_text("\n".join(L) + "\n")
    print(f"\n[out] {ablation_csv.relative_to(REPO)}", flush=True)
    print(f"[out] {report_md.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
