"""Part 3: sequential, one-family-at-a-time hyperparameter tuning on val_tune.

Retrieval-only evaluation: mIoP/IoP@0.5 depend only on which frames L2/PPR
retrieves (-> predicted span = [min,max] timestamp of retrieved frames),
not on captioning/answering/verification. So this harness calls embedding +
retrieval directly (iris.query._call_embed_query / _retrieve_with_l1) and
never touches the captioner, answerer, or Cerberus at all -- this is a
retrieval/grounding-quality sweep, not an answer-quality sweep (matches the
task's own selection metric: val_tune mIoP primary, IoP@0.5 tie-break).

Families are frozen strictly in order (retrieval_strategy -> ppr_lambda ->
ppr_damping -> l2_retrieve_top_k -> peak_distance/prominence). State
(previously frozen values) persists in tuning/frozen_state.json so each
family invocation picks up exactly where the previous one left off.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest  # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402
# predicted_span_from_frames is IRIS-specific (how retrieved frames become a
# span) -- not a duplicate of an official metric, stays here. IoP/IoU scoring
# itself is loaded from the canonical, registry-designated module (Part 2c
# consolidation: eval/metrics.py's independent IoP/IoU reimplementation is
# retired from this harness in favor of the one validated module).
from eval.metrics import predicted_span_from_frames, predicted_span_from_frames_peak  # noqa: E402

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)


def best_over_gold_spans(gold_spans: list[list[float]], pred_span: tuple[float, float]) -> tuple[float, float]:
    """Adapter to the canonical nextgqa_metrics.py's (pred_s, pred_e, gold_tuples)
    signature, preserving this harness's existing (gold_spans, pred_span) -> (iou, iop) call shape."""
    gold_tuples = [(g[0], g[1]) for g in gold_spans]
    iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
    iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)
    return iou, iop

VIDEO_DIR = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"
INDEX_CACHE_DIR = REPO / "tuning" / "index_cache"
INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TUNING_DIR = REPO / "tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)
FROZEN_STATE_PATH = TUNING_DIR / "frozen_state.json"
ALL_TRIALS_PATH = TUNING_DIR / "all_trials.csv"

DEFAULTS = {
    "retrieval_strategy": "hybrid",
    "ppr_lambda": 0.5,
    "ppr_damping": 0.5,
    "l2_retrieve_top_k": 5,
    "peak_distance": 5,
    "peak_prominence": 0.05,
    "packet_size_weight": 0.5,
    "motion_weight": 0.3,
    "luma_entropy_weight": 0.2,
}

# packet_size_weight/motion_weight/luma_entropy_weight feed action_score
# (iris/action_score.py's score_all(), ~lines 100-112), which determines
# is_peak at ingest time -- these MUST be ingest-relevant, same class of
# bug as the pre-commit-60202b5 scene_id omission and the pre-commit-
# 5d7c2b1 stale Method D default: without this, ingest_config_hash()
# can't distinguish weight combos and the harness would silently reuse a
# cached index built under a different combo's ingest output.
INGEST_RELEVANT_KEYS = ["retrieval_strategy", "l2_retrieve_top_k", "peak_distance", "peak_prominence", "peak_order",
                        "packet_size_weight", "motion_weight", "luma_entropy_weight"]

# Families whose grid values are tuples of multiple config fields rather
# than a single scalar -- maps family name to the ordered list of
# IRISConfig field names each tuple position overrides.
TUPLE_FAMILIES = {
    "peak_dist_prom": ["peak_distance", "peak_prominence"],
    "action_score_weights": ["packet_size_weight", "motion_weight", "luma_entropy_weight"],
}

FAMILIES = {
    "retrieval_strategy": ["peak_only", "top_k_action", "peak_neighbors", "hybrid"],
    "ppr_lambda": [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00],
    "ppr_damping": [0.50, 0.65, 0.80, 0.85, 0.90],
    "l2_retrieve_top_k": [4, 8, 12, 16],
    # (packet_size_weight, motion_weight, luma_entropy_weight) -- action_score
    # divides by the weight sum, so only the ratio between the three matters;
    # these five combos are not redundant with any rescaled equivalent.
    # Centered on the default (0.5, 0.3, 0.2) with each channel taken to a
    # dominant extreme plus an equal-weighting control.
    "action_score_weights": [(0.5, 0.3, 0.2), (0.8, 0.1, 0.1), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (0.34, 0.33, 0.33)],
    # find_peaks(distance=..., prominence=...) small joint grid: distance
    # controls min samples between selected peaks, prominence the minimum
    # peak salience relative to neighboring troughs. Centered on the
    # defaults (5, 0.05) with a narrower/wider distance and a stricter/looser
    # prominence, kept small (4 combos, not the full 3x3=9) since this
    # family requires a full re-ingest per combo just like family 1/4.
    "peak_dist_prom": [(3, 0.03), (5, 0.05), (8, 0.05), (5, 0.10)],
}
# action_score_weights is the more upstream parameter (shapes the curve
# find_peaks operates on) and belongs before peak_dist_prom in principle,
# even though peak_dist_prom already ran first in practice (commit
# 5d7c2b1) -- see tuning/selection_decisions.md's Family 5 limitations
# note for why that ordering slip cost little (Family 5 landed on a clean
# negative result, so no non-default value is conditionally at risk).
FAMILY_ORDER = ["retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k", "action_score_weights", "peak_dist_prom"]


def load_frozen_state() -> dict:
    if FROZEN_STATE_PATH.exists():
        return json.loads(FROZEN_STATE_PATH.read_text())
    return {"frozen": {}, "completed_families": []}


def save_frozen_state(state: dict) -> None:
    FROZEN_STATE_PATH.write_text(json.dumps(state, indent=2))


# Part 3e (tuning/family2_k_span_method_report.md, the 140-cell K x lambda x
# span-method sweep) superseded Part 3c's Method B pick: Method B's raw mIoP
# win was substantially inflated by zero-width spans (29.8% at the frozen
# K=4), an IoP-metric artifact rather than genuine localization, while
# Method D (fixed-width window centred on the CLIP-similarity peak frame)
# is essentially invariant to both K and lambda across the whole grid and
# has 0.00% zero-width incidence everywhere. Method D was adopted as the
# default span construction for Family 5 onward and the eventual official
# test run. half_width_s is read live from tuning/frozen_state.json's
# frozen block (falling back to 2.2, the provisional constant used
# throughout Part 3c/3d/3e, only if the state file predates this key).
SPAN_METHOD_D_HALF_WIDTH_S = load_frozen_state()["frozen"].get("span_method_half_width_s", 2.2)


def default_predicted_span(retrieved_frames: list[dict], query_embedding=None) -> tuple[float, float]:
    span, _used_clip_anchor = predicted_span_from_frames_peak(
        retrieved_frames, query_embedding,
        half_width_s=SPAN_METHOD_D_HALF_WIDTH_S,
    )
    return span


def make_config(overrides: dict) -> IRISConfig:
    cfg = IRISConfig()
    cfg.cerberus_mode = "none"
    for k, v in {**DEFAULTS, **overrides}.items():
        setattr(cfg, k, v)
    return cfg


def ingest_config_hash(cfg: IRISConfig) -> str:
    payload = {k: getattr(cfg, k) for k in INGEST_RELEVANT_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_val_tune_questions() -> list[dict]:
    split = json.loads((REPO / "split_manifest.json").read_text())
    tune_videos = set(split["tune_videos"])
    rows = list(csv.DictReader(open(REPO / "eval" / "data" / "nextqa" / "val.csv", newline="", encoding="utf-8")))
    gsub = json.loads((REPO / "eval" / "data" / "nextqa" / "gsub_val.json").read_text())
    out = []
    for r in rows:
        vid = r["video"]
        if vid not in tune_videos:
            continue
        qid = r["qid"]
        vpath = VIDEO_DIR / f"{vid}.mp4"
        if not vpath.exists():
            continue
        gold = gsub.get(vid, {}).get("location", {}).get(qid)
        if not gold:
            continue
        out.append({
            "video": vid, "qid": qid, "question": r["question"],
            "choices": [r["a0"], r["a1"], r["a2"], r["a3"], r["a4"]],
            "gold_spans": gold, "duration": gsub[vid]["duration"],
        })
    return out


def ensure_indexes(video_ids: list[str], cfg: IRISConfig, n_workers: int = 8) -> dict[str, str]:
    """Ingest (or load from cache) every needed video under this config's
    ingest-relevant hyperparameters. Returns video_id -> cached index path."""
    h = ingest_config_hash(cfg)
    paths = {}
    todo = []
    for vid in video_ids:
        p = INDEX_CACHE_DIR / f"{vid}__{h}"
        # save_index() -> np.savez() silently appends ".npz" to whatever path
        # it's given, so the on-disk file is "{vid}__{h}.npz", not "{vid}__{h}".
        # Must check for that exact filename or every rerun re-ingests
        # everything it already has cached.
        if p.with_suffix(p.suffix + ".npz").exists():
            paths[vid] = str(p)
        else:
            todo.append(vid)

    if todo:
        print(f"    [ingest] {len(todo)}/{len(video_ids)} videos need ingest under config-hash {h}", flush=True)

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
                    print(f"    [ingest FAIL] {vid}: {type(exc).__name__}: {exc}", flush=True)
                done += 1
                if done % 50 == 0:
                    print(f"    [ingest] {done}/{len(todo)} done", flush=True)
    return paths


def evaluate_config(cfg: IRISConfig, questions: list[dict], index_paths: dict[str, str],
                     index_cache: dict[str, object]) -> dict:
    """Retrieval-only pass: embed each question, retrieve, derive predicted
    span, score against gold. No captioner/answerer/Cerberus involved."""
    from iris.query import _call_embed_query, _retrieve_with_l1

    ious, iops, retrieval_ms = [], [], []
    n_scored = 0
    for q in questions:
        vid = q["video"]
        if vid not in index_paths:
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(index_paths[vid])
        index = index_cache[vid]

        t0 = time.perf_counter()
        try:
            query_embedding, _ = _call_embed_query(q["question"], cfg)
            retrieved_frames, _ = _retrieve_with_l1(index, query_embedding, cfg)
        except Exception:  # noqa: BLE001
            continue
        dt_ms = (time.perf_counter() - t0) * 1000
        retrieval_ms.append(dt_ms)

        pred_span = default_predicted_span(retrieved_frames, query_embedding)
        iou, iop = best_over_gold_spans(q["gold_spans"], pred_span)
        ious.append(iou)
        iops.append(iop)
        n_scored += 1

    import statistics
    return {
        "n_questions_scored": n_scored,
        "mIoP": (sum(iops) / len(iops)) if iops else 0.0,
        "mIoU": (sum(ious) / len(ious)) if ious else 0.0,
        "IoP@0.3": (sum(1 for x in iops if x >= 0.3) / len(iops)) if iops else 0.0,
        "IoP@0.5": (sum(1 for x in iops if x >= 0.5) / len(iops)) if iops else 0.0,
        "IoU@0.3": (sum(1 for x in ious if x >= 0.3) / len(ious)) if ious else 0.0,
        "IoU@0.5": (sum(1 for x in ious if x >= 0.5) / len(ious)) if ious else 0.0,
        "median_retrieval_ms": statistics.median(retrieval_ms) if retrieval_ms else 0.0,
        "p95_retrieval_ms": (statistics.quantiles(retrieval_ms, n=20)[18] if len(retrieval_ms) >= 20 else max(retrieval_ms, default=0.0)),
    }


ALL_TRIALS_FIELDNAMES = [
    "family", "trial_value", "config_hash", "n_questions_scored", "mIoP", "mIoU",
    "IoP@0.3", "IoP@0.5", "IoU@0.3", "IoU@0.5", "median_retrieval_ms", "p95_retrieval_ms",
    "wall_s", "selected",
    # action_score_weights-only diagnostics (blank for every other family's rows):
    # these three are the actual mechanism packet_size_weight/motion_weight/
    # luma_entropy_weight act through (is_peak / L1_PEAK tier admission),
    # not just the downstream mIoP/IoU numbers.
    "avg_peak_frame_count", "peak_fraction", "gold_peak_coverage_rate",
]


def _migrate_all_trials_schema() -> None:
    """One-time widen of an existing all_trials.csv written under the old
    (pre-diagnostic-columns) header to ALL_TRIALS_FIELDNAMES, preserving
    every prior row (diagnostic columns blank for families that never
    computed them). No-op if the file doesn't exist yet or is already
    on the current schema."""
    if not ALL_TRIALS_PATH.exists():
        return
    with open(ALL_TRIALS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == ALL_TRIALS_FIELDNAMES:
            return
        rows = list(reader)
    with open(ALL_TRIALS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_TRIALS_FIELDNAMES, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_trial_row(row: dict) -> None:
    _migrate_all_trials_schema()
    write_header = not ALL_TRIALS_PATH.exists()
    with open(ALL_TRIALS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_TRIALS_FIELDNAMES, restval="")
        if write_header:
            w.writeheader()
        w.writerow(row)


def compute_action_score_diagnostics(questions: list[dict], index_cache: dict[str, object]) -> dict:
    """The actual mechanism packet_size_weight/motion_weight/luma_entropy_weight
    act through: is_peak / L1_PEAK tier admission at ingest time (NOT
    retrieval ranking directly -- persistence_value's gamma-weighted term
    only fires under ranking_mode="legacy", which this project doesn't
    use). Computed from index_cache, which evaluate_config() already
    populated with every video's freshly-ingested index for this trial's
    config-hash."""
    peak_counts, total_frames_list = [], []
    for idx in index_cache.values():
        peak_counts.append(sum(1 for f in idx.frames if f.is_peak))
        total_frames_list.append(len(idx.frames))

    covered, n_q = 0, 0
    for q in questions:
        vid = q["video"]
        if vid not in index_cache:
            continue
        idx = index_cache[vid]
        n_q += 1
        gold_spans = q["gold_spans"]
        if any(f.is_peak and any(g[0] <= f.timestamp <= g[1] for g in gold_spans) for f in idx.frames):
            covered += 1

    total_peak = sum(peak_counts)
    total_frames = sum(total_frames_list)
    return {
        "avg_peak_frame_count": (total_peak / len(peak_counts)) if peak_counts else 0.0,
        "peak_fraction": (total_peak / total_frames) if total_frames else 0.0,
        "gold_peak_coverage_rate": (covered / n_q) if n_q else 0.0,
    }


def select_best(trials: list[dict], default_value) -> dict:
    """mIoP primary; IoP@0.5 tie-break if mIoP diff < 0.005; then lower
    median retrieval latency; then prefer the default/simpler value."""
    best = None
    for t in trials:
        if best is None:
            best = t
            continue
        d = t["metrics"]["mIoP"] - best["metrics"]["mIoP"]
        if d > 0.005:
            best = t
        elif abs(d) <= 0.005:
            if t["metrics"]["IoP@0.5"] > best["metrics"]["IoP@0.5"] + 1e-9:
                best = t
            elif abs(t["metrics"]["IoP@0.5"] - best["metrics"]["IoP@0.5"]) < 1e-9:
                if t["metrics"]["median_retrieval_ms"] < best["metrics"]["median_retrieval_ms"] - 1e-6:
                    best = t
                elif abs(t["metrics"]["median_retrieval_ms"] - best["metrics"]["median_retrieval_ms"]) < 1e-6:
                    if t["value"] == default_value:
                        best = t
    return best


def run_family(family: str, questions: list[dict]) -> None:
    state = load_frozen_state()
    if family in state["completed_families"]:
        print(f"[family={family}] already completed, skipping (frozen value: {state['frozen'].get(family)})")
        return

    grid = FAMILIES[family]
    frozen_so_far = state["frozen"]
    print(f"[family={family}] grid={grid} frozen_so_far={frozen_so_far}", flush=True)

    trials = []
    index_cache: dict[str, object] = {}
    video_ids = sorted({q["video"] for q in questions})

    for val in grid:
        t_start = time.perf_counter()
        if family in TUPLE_FAMILIES:
            field_names = TUPLE_FAMILIES[family]
            overrides = {**frozen_so_far, **dict(zip(field_names, val))}
            label = ",".join(f"{fn}={v}" for fn, v in zip(field_names, val))
        else:
            overrides = {**frozen_so_far, family: val}
            label = str(val)

        cfg = make_config(overrides)
        print(f"  [trial] {family}={label} (codec_conf_source={cfg.codec_conf_source})", flush=True)
        index_paths = ensure_indexes(video_ids, cfg)
        index_cache.clear()  # different config-hash -> different cached objects; avoid cross-trial staleness
        metrics = evaluate_config(cfg, questions, index_paths, index_cache)
        wall_s = time.perf_counter() - t_start

        diagnostics = {}
        if family == "action_score_weights":
            diagnostics = compute_action_score_diagnostics(questions, index_cache)
            print(f"    [diag] avg_peak_frame_count={diagnostics['avg_peak_frame_count']:.2f} "
                  f"peak_fraction={diagnostics['peak_fraction']:.4f} "
                  f"gold_peak_coverage_rate={diagnostics['gold_peak_coverage_rate']:.4f}", flush=True)

        print(f"    -> mIoP={metrics['mIoP']:.4f} IoP@0.5={metrics['IoP@0.5']:.4f} "
              f"mIoU={metrics['mIoU']:.4f} n={metrics['n_questions_scored']} wall={wall_s:.0f}s", flush=True)

        trials.append({"value": val, "label": label, "metrics": metrics, "diagnostics": diagnostics,
                        "cfg_hash": ingest_config_hash(cfg), "wall_s": wall_s})
        row = {
            "family": family, "trial_value": label, "config_hash": ingest_config_hash(cfg),
            "n_questions_scored": metrics["n_questions_scored"], "mIoP": round(metrics["mIoP"], 5),
            "mIoU": round(metrics["mIoU"], 5), "IoP@0.3": round(metrics["IoP@0.3"], 5),
            "IoP@0.5": round(metrics["IoP@0.5"], 5), "IoU@0.3": round(metrics["IoU@0.3"], 5),
            "IoU@0.5": round(metrics["IoU@0.5"], 5), "median_retrieval_ms": round(metrics["median_retrieval_ms"], 2),
            "p95_retrieval_ms": round(metrics["p95_retrieval_ms"], 2), "wall_s": round(wall_s, 1),
            "selected": "",
        }
        if diagnostics:
            row["avg_peak_frame_count"] = round(diagnostics["avg_peak_frame_count"], 3)
            row["peak_fraction"] = round(diagnostics["peak_fraction"], 5)
            row["gold_peak_coverage_rate"] = round(diagnostics["gold_peak_coverage_rate"], 5)
        append_trial_row(row)

    if family in TUPLE_FAMILIES:
        default_val = tuple(DEFAULTS[fn] for fn in TUPLE_FAMILIES[family])
    else:
        default_val = DEFAULTS.get(family)
    best = select_best(trials, default_val)

    print(f"[family={family}] SELECTED: {best['label']} (mIoP={best['metrics']['mIoP']:.4f})", flush=True)

    if family in TUPLE_FAMILIES:
        for fn, v in zip(TUPLE_FAMILIES[family], best["value"]):
            frozen_so_far[fn] = v
    else:
        frozen_so_far[family] = best["value"]
    state["frozen"] = frozen_so_far
    state["completed_families"].append(family)
    state[f"selection_detail_{family}"] = {"selected": best["label"], "all_trials": [
        {"value": t["label"], "mIoP": t["metrics"]["mIoP"], "IoP@0.5": t["metrics"]["IoP@0.5"],
         "median_retrieval_ms": t["metrics"]["median_retrieval_ms"], **t["diagnostics"]} for t in trials
    ]}
    save_frozen_state(state)

    has_diagnostics = family == "action_score_weights"
    with open(TUNING_DIR / "selection_decisions.md", "a") as f:
        f.write(f"\n## Family: {family}\n\n")
        f.write(f"Grid: {grid}\n\n")
        if has_diagnostics:
            f.write("| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored | avg_peak_frame_count | peak_fraction | gold_peak_coverage_rate |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for t in trials:
                mark = " **<- selected**" if t is best else ""
                d = t["diagnostics"]
                f.write(f"| {t['label']}{mark} | {t['metrics']['mIoP']:.4f} | {t['metrics']['IoP@0.5']:.4f} | "
                        f"{t['metrics']['mIoU']:.4f} | {t['metrics']['median_retrieval_ms']:.1f} | {t['metrics']['n_questions_scored']} | "
                        f"{d['avg_peak_frame_count']:.2f} | {d['peak_fraction']:.4f} | {d['gold_peak_coverage_rate']:.4f} |\n")
        else:
            f.write("| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |\n")
            f.write("|---|---|---|---|---|---|\n")
            for t in trials:
                mark = " **<- selected**" if t is best else ""
                f.write(f"| {t['label']}{mark} | {t['metrics']['mIoP']:.4f} | {t['metrics']['IoP@0.5']:.4f} | "
                        f"{t['metrics']['mIoU']:.4f} | {t['metrics']['median_retrieval_ms']:.1f} | {t['metrics']['n_questions_scored']} |\n")
        f.write(f"\nSelected **{family} = {best['label']}** "
                f"(mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).\n")


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else None
    questions = load_val_tune_questions()
    print(f"[setup] val_tune questions loaded: {len(questions)} across {len(set(q['video'] for q in questions))} videos", flush=True)

    if family:
        run_family(family, questions)
    else:
        for fam in FAMILY_ORDER:
            run_family(fam, questions)

    state = load_frozen_state()
    if len(state["completed_families"]) == len(FAMILY_ORDER):
        print("HYPERPARAMETER_TUNING_COMPLETE (all families frozen; val_confirm evaluation still pending)")


if __name__ == "__main__":
    main()
