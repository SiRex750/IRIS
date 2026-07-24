"""P_FUNNEL diagnostic: nested coverage/selection funnel on VAL.

Implements eval_results/P_funnel_prereg.md exactly. Read-only -- selects
nothing, changes no default, no frozen config value. Reuses
pnowa_width_topk_sweep.py::_load_val_rows() verbatim (including its
test-leak assertion) rather than writing a new loader, and reads the
frozen config from that module's BASE dict rather than hardcoding it.

VERIFY (guard only -- exercises assertion 1, does not write output):
    python scripts/p_funnel_diagnostic.py --top_k 8
VERIFY (full grid -- all four top_k values, writes eval_results/P_funnel_raw.json):
    python scripts/p_funnel_diagnostic.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.grounding_scorer import load_indexes
from eval.span import predict_span, _pick_by_clip_similarity, _frame_embedding, _frame_timestamp
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved
from scripts.pillar2_grounded_qa import git_provenance, config_hash
from scripts.pnowa_width_topk_sweep import (
    _load_val_rows, BASE, GQA_JSON, CACHE_DIR, VAL_VIDEO_LIST, TEST_VIDEO_LIST,
)

TOP_KS = [8, 12, 16, 24]
SPAN_MODE = "ppr_peak"
HALF_WIDTH = 2.2
PEAK_SOURCE = "clip_in_ppr_top8"
SEED = 20260710
NUM_BOOT = 1000

# Prereg prediction, checked as a hard assertion at top_k=8 (4 dp).
EXPECTED_PEAK_IN_GOLD_K8 = 0.3227

OUT_JSON = REPO / "eval_results" / "P_funnel_raw.json"


def _in_gold(ts: float, gold_spans: list[list[float]]) -> bool:
    return any(float(s) <= ts <= float(e) for s, e in gold_spans)


def _best_gold_rank(retrieved: list[dict], emb, gold_spans: list[list[float]]) -> int | None:
    """1-based rank, within `retrieved` ordered by the same raw CLIP cosine
    similarity clip_in_ppr_top8 uses, of the highest-ranked in-gold frame.
    None when no retrieved frame is in-gold."""
    q = np.asarray(emb, dtype=np.float64).flatten()
    q_norm = float(np.linalg.norm(q))
    scored = []
    for f in retrieved:
        e = _frame_embedding(f)
        sim = -np.inf
        if e is not None and q_norm >= 1e-8:
            ev = np.asarray(e, dtype=np.float64).flatten()
            e_norm = float(np.linalg.norm(ev))
            if e_norm >= 1e-8:
                sim = float(np.dot(ev, q) / (e_norm * q_norm))
        scored.append((sim, _frame_timestamp(f)))
    scored.sort(key=lambda x: x[0], reverse=True)
    for rank, (_, ts) in enumerate(scored, start=1):
        if _in_gold(ts, gold_spans):
            return rank
    return None


def run_top_k(top_k: int, grounded_rows: list[dict], gsub: dict,
              duration_by_vid: dict[str, float], counters: dict) -> list[dict]:
    cfg = IRISConfig(**BASE, l2_retrieve_top_k=top_k)
    loaded = load_indexes(grounded_rows, CACHE_DIR)

    for vid in {r["video"] for r in grounded_rows}:
        if loaded.get(vid) is None:
            counters["index_load_failures"] += 1

    per_question = []
    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        index = loaded.get(vid)
        if index is None:
            continue
        gold_spans = gsub[vid]["location"][qid]

        emb = _embed_query(row["question"], cfg)
        retrieved = _build_retrieved(index, emb, cfg)

        counters["n_clip_total"] += 1
        if _pick_by_clip_similarity(retrieved, emb) is None:
            counters["n_clip_fallback"] += 1

        if len(retrieved) != top_k:
            counters["len_mismatches"] += 1

        index_coverage = 1 if any(_in_gold(fr.timestamp, gold_spans) for fr in index.frames) else 0
        pool_coverage = 1 if any(_in_gold(r["timestamp"], gold_spans) for r in retrieved) else 0

        span, t_peak = predict_span(
            retrieved, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb, return_peak=True,
        )
        peak_in_gold = 1 if (t_peak is not None and _in_gold(t_peak, gold_spans)) else 0

        if index_coverage < pool_coverage or pool_coverage < peak_in_gold:
            counters["nesting_violations"].append((vid, qid))

        best_rank = _best_gold_rank(retrieved, emb, gold_spans) if pool_coverage else None

        per_question.append({
            "video": vid, "qid": qid, "top_k": top_k,
            "index_coverage": index_coverage,
            "pool_coverage": pool_coverage,
            "peak_in_gold": peak_in_gold,
            "best_gold_rank": best_rank,
        })
    return per_question


def _cluster_bootstrap(video_to_vals: dict[str, list[float]], seed: int, num_boot: int) -> dict:
    vids = list(video_to_vals.keys())
    n_vids = len(vids)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(num_boot):
        resampled_vids = rng.choice(vids, size=n_vids, replace=True)
        vals = []
        for v in resampled_vids:
            vals.extend(video_to_vals[v])
        if vals:
            means.append(float(np.mean(vals)))
    arr = np.array(means)
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P_FUNNEL nested coverage/selection diagnostic (VAL only)")
    parser.add_argument(
        "--top_k", type=int, nargs="*", default=None,
        help="Restrict to specific top_k value(s) (subset of [8,12,16,24]). "
             "Default runs the full grid. A partial/guard run does NOT write "
             "eval_results/P_funnel_raw.json.",
    )
    args = parser.parse_args()
    top_ks_to_run = args.top_k if args.top_k else TOP_KS
    for tk in top_ks_to_run:
        assert tk in TOP_KS, f"top_k={tk} is not one of the pre-registered values {TOP_KS}"

    grounded_rows = _load_val_rows()
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    n_videos = len(set(r["video"] for r in grounded_rows))
    print(f"[DATA] VAL grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    used_videos = {r["video"] for r in grounded_rows}
    leaked = used_videos & test_videos
    assert not leaked, f"TEST videos present in P_FUNNEL run: {sorted(leaked)}"
    print(f"[GUARD] zero test videos present: OK ({len(used_videos)} val videos, {len(test_videos)} test videos on file)")

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in used_videos}

    counters = {
        "n_clip_total": 0, "n_clip_fallback": 0,
        "len_mismatches": 0, "index_load_failures": 0,
        "nesting_violations": [],
    }

    all_rows: list[dict] = []
    for top_k in top_ks_to_run:
        rows = run_top_k(top_k, grounded_rows, gsub, duration_by_vid, counters)
        all_rows.extend(rows)

        pig_rate = sum(r["peak_in_gold"] for r in rows) / len(rows) if rows else float("nan")
        pool_rate = sum(r["pool_coverage"] for r in rows) / len(rows) if rows else float("nan")
        idx_rate = sum(r["index_coverage"] for r in rows) / len(rows) if rows else float("nan")
        print(f"[CELL] top_k={top_k:>3} index_coverage={idx_rate:.4f} "
              f"pool_coverage={pool_rate:.4f} peak_in_gold={pig_rate:.4f} n={len(rows)}", flush=True)

        if top_k == 8:
            if round(pig_rate, 4) != EXPECTED_PEAK_IN_GOLD_K8:
                print(
                    f"FATAL: assertion 1 failed -- computed peak_in_gold at top_k=8 = "
                    f"{pig_rate:.4f}, expected {EXPECTED_PEAK_IN_GOLD_K8:.4f} (4dp). "
                    f"STOPPING -- do not continue.", file=sys.stderr,
                )
                sys.exit(1)
            print(f"[GUARD] assertion 1 PASSED: peak_in_gold@top_k=8 = {pig_rate:.4f} == {EXPECTED_PEAK_IN_GOLD_K8:.4f}")

    # ── assertions 2-6 ───────────────────────────────────────────────────────
    fallback_rate = counters["n_clip_fallback"] / counters["n_clip_total"] if counters["n_clip_total"] else 0.0
    print(f"[GUARD] CLIP-anchor fallback rate: {fallback_rate:.4%} ({counters['n_clip_fallback']}/{counters['n_clip_total']})")
    assert counters["n_clip_fallback"] == 0, (
        f"CLIP-anchor fallback fired {counters['n_clip_fallback']} times -- "
        f"peak_source=clip_in_ppr_top8 silently degraded to ppr_score."
    )

    print(f"[GUARD] len(retrieved) != top_k mismatches: {counters['len_mismatches']}")
    assert counters["len_mismatches"] == 0, f"{counters['len_mismatches']} questions had len(retrieved) != top_k"

    print(f"[GUARD] index load failures: {counters['index_load_failures']}")
    assert counters["index_load_failures"] == 0, f"{counters['index_load_failures']} index(es) failed to load"

    print(f"[GUARD] nesting violations (index_coverage >= pool_coverage >= peak_in_gold): {len(counters['nesting_violations'])}")
    assert not counters["nesting_violations"], (
        f"{len(counters['nesting_violations'])} nesting violations: {counters['nesting_violations']}"
    )

    print("[GUARD] all acceptance assertions passed" + (" (partial run)" if set(top_ks_to_run) != set(TOP_KS) else ""))

    if set(top_ks_to_run) != set(TOP_KS):
        print(f"[STOP] partial/guard run (top_k={top_ks_to_run}) -- NOT writing {OUT_JSON.name}.")
        return

    # ── full grid only below: bootstrap + provenance + write output ─────────
    video_to_pool: dict[str, list[float]] = {}
    video_to_peak: dict[str, list[float]] = {}
    video_to_idx: dict[str, list[float]] = {}
    for r in all_rows:
        video_to_pool.setdefault(r["video"], []).append(r["pool_coverage"])
        video_to_peak.setdefault(r["video"], []).append(r["peak_in_gold"])
        video_to_idx.setdefault(r["video"], []).append(r["index_coverage"])

    video_to_headroom = {
        v: [p - k for p, k in zip(video_to_pool[v], video_to_peak[v])]
        for v in video_to_pool
    }

    boot = {
        "index_coverage": _cluster_bootstrap(video_to_idx, SEED, NUM_BOOT),
        "pool_coverage": _cluster_bootstrap(video_to_pool, SEED, NUM_BOOT),
        "peak_in_gold": _cluster_bootstrap(video_to_peak, SEED, NUM_BOOT),
        "selection_headroom": _cluster_bootstrap(video_to_headroom, SEED, NUM_BOOT),
    }
    print("\n[BOOTSTRAP] pooled across top_ks run, seed=%d, num_boot=%d" % (SEED, NUM_BOOT))
    for k, v in boot.items():
        print(f"  {k}: mean={v['mean']:.4f} CI=[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    git_commit, git_dirty = git_provenance(REPO)
    cfg8 = IRISConfig(**BASE, l2_retrieve_top_k=8)
    provenance = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash(cfg8),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(grounded_rows),
        "n_videos": n_videos,
        "num_boot": NUM_BOOT,
        "seed": SEED,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "provenance": provenance,
            "bootstrap": boot,
            "per_question": all_rows,
        }, fh, indent=2)
    print(f"\n[LOG] raw dump written to {OUT_JSON}")


if __name__ == "__main__":
    main()
