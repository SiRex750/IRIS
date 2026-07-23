"""Part 3e follow-up: percentile bootstrap CIs on mIoP for each (K, span
method)'s top 2-3 lambda candidates (by raw mIoP from
tuning/lambda_k_span_method_comparison.csv), plus a cross-K comparison of
each method's single best (K, lambda) cell against its K=4 counterpart.

Re-scores only the (K, lambda) pairs needed for the candidate set (reusing
the same cached indexes as part3e_lambda_k_span_method_comparison.py --
no re-ingest, grouped by K to load each K's index cache once), capturing
the full per-question IoP array so the bootstrap resamples over the true
n=2685 question set.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import part3_tune as pt  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.query import _call_embed_query, _retrieve_with_l1  # noqa: E402
from eval.metrics import (  # noqa: E402
    best_over_gold_spans, predicted_span_from_frames,
    predicted_span_from_frames_clustered, predicted_span_from_frames_scene,
    predicted_span_from_frames_peak,
)

FRESH_CACHE_DIR = REPO / "tuning" / "index_cache_scenespans"
GAP_THRESHOLD_S = 3.0
TAIL_TRIM_PCT = 20.0
HALF_WIDTH_S = 2.2
N_RESAMPLES = 1000
SEED = 20260723
K_GRID = [4, 5, 8, 12, 16]
METHODS = ["A", "B", "C", "D"]


def top_candidates(csv_path: Path, n: int = 3) -> dict[tuple[int, str], list[float]]:
    rows = list(csv.DictReader(open(csv_path)))
    by_km: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for r in rows:
        key = (int(r["K"]), r["method"])
        by_km.setdefault(key, []).append((float(r["lambda"]), float(r["mIoP"])))
    out = {}
    for key, lst in by_km.items():
        lst.sort(key=lambda x: -x[1])
        out[key] = [lam for lam, _ in lst[:n]]
    return out


def iop_for_cell(cfg, index_cache: dict, questions: list[dict]) -> dict[str, list[float]]:
    per_method: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}
    for q in questions:
        vid = q["video"]
        if vid not in index_cache:
            continue
        index = index_cache[vid]
        try:
            qe, _ = _call_embed_query(q["question"], cfg)
            frames, _ = _retrieve_with_l1(index, qe, cfg)
        except Exception:
            continue
        if not frames:
            continue
        gold_spans = q["gold_spans"]
        timestamps = [f["timestamp"] for f in frames]

        span_a = predicted_span_from_frames(timestamps)
        span_b = predicted_span_from_frames_clustered(frames, GAP_THRESHOLD_S, TAIL_TRIM_PCT, query_embedding=qe)
        span_c, _ = predicted_span_from_frames_scene(frames, index.scene_spans, query_embedding=qe)
        span_d, _ = predicted_span_from_frames_peak(frames, qe, half_width_s=HALF_WIDTH_S, duration_s=q.get("duration"))

        for method, span in (("A", span_a), ("B", span_b), ("C", span_c), ("D", span_d)):
            _, iop = best_over_gold_spans(gold_spans, span)
            per_method[method].append(iop)
    return per_method


def percentile_bootstrap(values: list[float], n_resamples: int, rng: random.Random) -> tuple[float, float, float]:
    n = len(values)
    point = sum(values) / n
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return point, lo, hi


def main():
    questions = pt.load_val_tune_questions()
    video_ids = sorted({q["video"] for q in questions})

    candidates = top_candidates(REPO / "tuning" / "lambda_k_span_method_comparison.csv", n=3)
    needed_by_k: dict[int, set[float]] = defaultdict(set)
    for (K, method), lams in candidates.items():
        for lam in lams:
            needed_by_k[K].add(lam)
    print(f"[setup] (K, lambda) pairs needed: "
          f"{sum(len(v) for v in needed_by_k.values())} across {len(needed_by_k)} K values", flush=True)

    frozen = json.loads((REPO / "tuning" / "frozen_state.json").read_text())["frozen"]
    base = {"retrieval_strategy": frozen["retrieval_strategy"], "ppr_damping": frozen["ppr_damping"]}

    iop_by_k_lambda: dict[tuple[int, float], dict[str, list[float]]] = {}
    for K in K_GRID:
        if K not in needed_by_k:
            continue
        ingest_cfg = pt.make_config({**base, "l2_retrieve_top_k": K, "ppr_lambda": sorted(needed_by_k[K])[0]})
        h = pt.ingest_config_hash(ingest_cfg)
        index_cache = {}
        for vid in video_ids:
            p = FRESH_CACHE_DIR / f"{vid}__{h}"
            if p.with_suffix(p.suffix + ".npz").exists():
                index_cache[vid] = iris_ingest.load_index(str(p))
        print(f"[setup K={K}] loaded {len(index_cache)}/{len(video_ids)} cached indexes (hash {h})", flush=True)

        for lam in sorted(needed_by_k[K]):
            cfg = pt.make_config({**base, "l2_retrieve_top_k": K, "ppr_lambda": lam})
            iop_by_k_lambda[(K, lam)] = iop_for_cell(cfg, index_cache, questions)
            n = len(iop_by_k_lambda[(K, lam)]["A"])
            print(f"[scored] K={K} lambda={lam} n={n}", flush=True)
        index_cache.clear()

    rng = random.Random(SEED)
    results = []
    for K in K_GRID:
        for method in METHODS:
            lams = candidates.get((K, method))
            if not lams:
                continue
            print(f"\n=== K={K} Method {method} ===", flush=True)
            stats = {}
            for lam in lams:
                values = iop_by_k_lambda[(K, lam)][method]
                point, lo, hi = percentile_bootstrap(values, N_RESAMPLES, rng)
                stats[lam] = (point, lo, hi)
                print(f"  lambda={lam}: mIoP={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  n={len(values)}", flush=True)
                results.append({"K": K, "method": method, "lambda": lam, "mIoP": round(point, 5),
                                 "ci_lo": round(lo, 5), "ci_hi": round(hi, 5), "n": len(values)})
            best_lam = lams[0]
            for other_lam in lams[1:]:
                best_lo, best_hi = stats[best_lam][1], stats[best_lam][2]
                other_lo, other_hi = stats[other_lam][1], stats[other_lam][2]
                overlap = not (best_hi < other_lo or other_hi < best_lo)
                print(f"  best(lambda={best_lam}) vs lambda={other_lam}: CI overlap = {overlap}", flush=True)
                results.append({"K": K, "method": method, "comparison": f"{best_lam}_vs_{other_lam}",
                                 "ci_overlap": overlap})

    # Cross-K comparison: each method's single best (K, lambda) cell vs its
    # K=4 best cell -- is any K clearly better than the currently-frozen K=4?
    print("\n=== Cross-K: best cell per method vs K=4's best cell ===", flush=True)
    for method in METHODS:
        best_overall = None
        for K in K_GRID:
            lams = candidates.get((K, method), [])
            if not lams:
                continue
            top_lam = lams[0]
            point, lo, hi = percentile_bootstrap(iop_by_k_lambda[(K, top_lam)][method], N_RESAMPLES, rng)
            if best_overall is None or point > best_overall[1]:
                best_overall = (K, point, lo, hi, top_lam)
        k4_lams = candidates.get((4, method), [])
        if not k4_lams or best_overall is None:
            continue
        k4_lam = k4_lams[0]
        k4_point, k4_lo, k4_hi = percentile_bootstrap(iop_by_k_lambda[(4, k4_lam)][method], N_RESAMPLES, rng)
        K, point, lo, hi, top_lam = best_overall
        overlap = not (hi < k4_lo or k4_hi < lo)
        print(f"  Method {method}: best K={K} lambda={top_lam} mIoP={point:.4f} CI=[{lo:.4f},{hi:.4f}]  "
              f"vs K=4 lambda={k4_lam} mIoP={k4_point:.4f} CI=[{k4_lo:.4f},{k4_hi:.4f}]  "
              f"overlap={overlap}", flush=True)
        results.append({"cross_K_comparison": method, "best_K": K, "best_lambda": top_lam,
                         "best_mIoP": round(point, 5), "best_ci_lo": round(lo, 5), "best_ci_hi": round(hi, 5),
                         "k4_lambda": k4_lam, "k4_mIoP": round(k4_point, 5),
                         "k4_ci_lo": round(k4_lo, 5), "k4_ci_hi": round(k4_hi, 5), "ci_overlap": overlap})

    with open(REPO / "tuning" / "lambda_k_bootstrap_ci.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBOOTSTRAP_CI_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
