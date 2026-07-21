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
from eval.metrics import best_over_gold_spans, predicted_span_from_frames  # noqa: E402

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
}

INGEST_RELEVANT_KEYS = ["retrieval_strategy", "l2_retrieve_top_k", "peak_distance", "peak_prominence", "peak_order"]

FAMILIES = {
    "retrieval_strategy": ["peak_only", "top_k_action", "peak_neighbors", "hybrid"],
    "ppr_lambda": [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00],
    "ppr_damping": [0.50, 0.65, 0.80, 0.85, 0.90],
    "l2_retrieve_top_k": [4, 8, 12, 16],
    # find_peaks(distance=..., prominence=...) small joint grid: distance
    # controls min samples between selected peaks, prominence the minimum
    # peak salience relative to neighboring troughs. Centered on the
    # defaults (5, 0.05) with a narrower/wider distance and a stricter/looser
    # prominence, kept small (4 combos, not the full 3x3=9) since this
    # family requires a full re-ingest per combo just like family 1/4.
    "peak_dist_prom": [(3, 0.03), (5, 0.05), (8, 0.05), (5, 0.10)],
}
FAMILY_ORDER = ["retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k", "peak_dist_prom"]


def load_frozen_state() -> dict:
    if FROZEN_STATE_PATH.exists():
        return json.loads(FROZEN_STATE_PATH.read_text())
    return {"frozen": {}, "completed_families": []}


def save_frozen_state(state: dict) -> None:
    FROZEN_STATE_PATH.write_text(json.dumps(state, indent=2))


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

        timestamps = [f["timestamp"] for f in retrieved_frames]
        pred_span = predicted_span_from_frames(timestamps)
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


def append_trial_row(row: dict) -> None:
    write_header = not ALL_TRIALS_PATH.exists()
    fieldnames = ["family", "trial_value", "config_hash", "n_questions_scored", "mIoP", "mIoU",
                  "IoP@0.3", "IoP@0.5", "IoU@0.3", "IoU@0.5", "median_retrieval_ms", "p95_retrieval_ms",
                  "wall_s", "selected"]
    with open(ALL_TRIALS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


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
        if family == "peak_dist_prom":
            overrides = {**frozen_so_far, "peak_distance": val[0], "peak_prominence": val[1]}
            label = f"distance={val[0]},prominence={val[1]}"
        else:
            overrides = {**frozen_so_far, family: val}
            label = str(val)

        cfg = make_config(overrides)
        print(f"  [trial] {family}={label}", flush=True)
        index_paths = ensure_indexes(video_ids, cfg)
        index_cache.clear()  # different config-hash -> different cached objects; avoid cross-trial staleness
        metrics = evaluate_config(cfg, questions, index_paths, index_cache)
        wall_s = time.perf_counter() - t_start
        print(f"    -> mIoP={metrics['mIoP']:.4f} IoP@0.5={metrics['IoP@0.5']:.4f} "
              f"mIoU={metrics['mIoU']:.4f} n={metrics['n_questions_scored']} wall={wall_s:.0f}s", flush=True)

        trials.append({"value": val, "label": label, "metrics": metrics, "cfg_hash": ingest_config_hash(cfg), "wall_s": wall_s})
        append_trial_row({
            "family": family, "trial_value": label, "config_hash": ingest_config_hash(cfg),
            "n_questions_scored": metrics["n_questions_scored"], "mIoP": round(metrics["mIoP"], 5),
            "mIoU": round(metrics["mIoU"], 5), "IoP@0.3": round(metrics["IoP@0.3"], 5),
            "IoP@0.5": round(metrics["IoP@0.5"], 5), "IoU@0.3": round(metrics["IoU@0.3"], 5),
            "IoU@0.5": round(metrics["IoU@0.5"], 5), "median_retrieval_ms": round(metrics["median_retrieval_ms"], 2),
            "p95_retrieval_ms": round(metrics["p95_retrieval_ms"], 2), "wall_s": round(wall_s, 1),
            "selected": "",
        })

    default_val = DEFAULTS["peak_distance"], DEFAULTS["peak_prominence"] if family == "peak_dist_prom" else DEFAULTS.get(family)
    best = select_best(trials, default_val if family != "peak_dist_prom" else (DEFAULTS["peak_distance"], DEFAULTS["peak_prominence"]))

    # Mark selected row in the CSV (rewrite with the selected flag for this family's rows)
    import pandas as _pd  # noqa
    print(f"[family={family}] SELECTED: {best['label']} (mIoP={best['metrics']['mIoP']:.4f})", flush=True)

    if family == "peak_dist_prom":
        frozen_so_far["peak_distance"] = best["value"][0]
        frozen_so_far["peak_prominence"] = best["value"][1]
    else:
        frozen_so_far[family] = best["value"]
    state["frozen"] = frozen_so_far
    state["completed_families"].append(family)
    state[f"selection_detail_{family}"] = {"selected": best["label"], "all_trials": [
        {"value": t["label"], "mIoP": t["metrics"]["mIoP"], "IoP@0.5": t["metrics"]["IoP@0.5"],
         "median_retrieval_ms": t["metrics"]["median_retrieval_ms"]} for t in trials
    ]}
    save_frozen_state(state)

    with open(TUNING_DIR / "selection_decisions.md", "a") as f:
        f.write(f"\n## Family: {family}\n\n")
        f.write(f"Grid: {grid}\n\n")
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
