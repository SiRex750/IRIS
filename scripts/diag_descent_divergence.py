"""Phase 6, tasks 2c-iii/2c-iv: divergence + gate-firing + cross-scene edge
SPARSITY sweep DIAGNOSTICS on real, distinct-scene NExT-GQA data. Measurement
only -- no production behavior change. Decides whether PPR descent (2c-ii)
earns its place, and what descent-graph structure the grounding gate should
actually test.

NOTE: eval/data/nextqa/index_cache/*.npz predates scene_id assignment (all
cached frames have scene_id=-1, empty _scene_centroids) -- reload cannot
retroactively assign scene_id, that requires the packet curve which only
exists during a real ingest() decode. So this script ingests fresh with
graph_mode="scene_sparse" and caches into its own dir (index_cache_ssparse/)
so re-runs of this diagnostic are fast without touching the shared
index_cache/ that other phase6 scripts depend on.

VERIFY: python scripts/diag_descent_divergence.py
        python scripts/diag_descent_divergence.py --sweep-crossscene
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace as _replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
import iris.scene_retrieval as scene_retrieval
from iris.iris_config import IRISConfig

DATA_DIR   = REPO / "eval" / "data" / "nextqa"
FLAT_DIR   = DATA_DIR / "NExTVideo_flat"
CACHE_DIR  = DATA_DIR / "index_cache_ssparse"  # this script's own cache (scene_id-assigned)
DEV_JSONL  = DATA_DIR / "dev_100.jsonl"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_QUESTIONS = 50
PRODUCTION_TAU = 0.05

CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
    graph_mode="scene_sparse",
    scene_shortcut_margin=PRODUCTION_TAU,
    scene_diag=True,
)

# 2c-iv: cross-scene edge sparsity modes under sweep. "A_all" is the 2c-ii
# baseline (current production default) -- everything else is compared
# against it.
SWEEP_MODES = [
    ("A_all",      {"scene_crossscene_mode": "all"}),
    ("B_p50",      {"scene_crossscene_mode": "threshold", "scene_crossscene_threshold_pctile": 50.0}),
    ("B_p75",      {"scene_crossscene_mode": "threshold", "scene_crossscene_threshold_pctile": 75.0}),
    ("B_p90",      {"scene_crossscene_mode": "threshold", "scene_crossscene_threshold_pctile": 90.0}),
    ("C_rep_only", {"scene_crossscene_mode": "rep_only"}),
]


def bucket_histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v <= e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = [f"<={e}" for e in edges] + [f">{edges[-1]}"]
    return list(zip(labels, counts))


def load_indices(dev_rows: list[dict]) -> tuple[dict, list[str]]:
    distinct_vids = sorted({r["video"] for r in dev_rows})
    print(f"Distinct videos among these questions: {len(distinct_vids)}")

    cached_by_vid: dict = {}
    failed_ingest = []
    for i, vid in enumerate(distinct_vids, 1):
        cache_path = CACHE_DIR / vid
        npz = Path(str(cache_path) + ".npz")
        if npz.exists():
            try:
                idx = iris_ingest.load_index(cache_path)
                if idx._scene_centroids:
                    cached_by_vid[vid] = idx
                    print(f"[{i:2d}/{len(distinct_vids)}] cache hit  {vid}  N={len(idx.frames)}")
                    continue
            except Exception:
                pass  # fall through to re-ingest

        video_path = FLAT_DIR / f"{vid}.mp4"
        if not video_path.exists():
            failed_ingest.append(f"{vid} (video not found)")
            continue
        try:
            idx = iris_ingest.ingest(str(video_path), config=CFG)
            iris_ingest.save_index(idx, cache_path)
            cached_by_vid[vid] = idx
            print(f"[{i:2d}/{len(distinct_vids)}] ingested   {vid}  N={len(idx.frames)}")
        except Exception as e:
            failed_ingest.append(f"{vid} (ingest error: {str(e)[:100]})")
        sys.stdout.flush()

    print(f"Loaded/ingested: {len(cached_by_vid)}  failed: {len(failed_ingest)}")
    if failed_ingest:
        print(f"  failed: {failed_ingest}")
    print()
    return cached_by_vid, distinct_vids


def restated_tau_table(margins: list[float]) -> None:
    """Report-not-tune: what tau would need to be to fire (shortcut) on
    ~X% of queries, derived from the OBSERVED margin distribution."""
    if not margins:
        print("  N/A (no finite margins observed)")
        return
    sorted_margins = sorted(margins)
    print("  tau restated from observed margin distribution (report-not-tune):")
    for shortcut_pct in [10, 25, 50, 75, 90]:
        # shortcut fires when margin > tau, i.e. tau = the (100-shortcut_pct)th percentile
        # of margins gives roughly shortcut_pct% of margins strictly above it.
        pctile = 100 - shortcut_pct
        idx = min(len(sorted_margins) - 1, max(0, round(pctile / 100 * (len(sorted_margins) - 1))))
        tau_val = sorted_margins[idx]
        print(f"    tau ~= {tau_val:.4f}  ->  ~{shortcut_pct}% of queries would shortcut")
    print(f"  current production tau={PRODUCTION_TAU} sits at the "
          f"{sum(1 for m in margins if m <= PRODUCTION_TAU) / len(margins):.1%} percentile "
          f"of this distribution (fraction of margins <= tau).")


def main_baseline(dev_rows: list[dict], cached_by_vid: dict) -> None:
    scene_retrieval.SCENE_DIAG_RECORDS.clear()

    questions_processed = 0
    questions_skipped = 0
    for row in dev_rows:
        vid = row["video"]
        idx = cached_by_vid.get(vid)
        if idx is None or not idx._scene_centroids:
            questions_skipped += 1
            continue

        emb = iris_query._embed_query(row["question"], CFG)
        try:
            iris_query._build_retrieved(idx, emb, CFG)
        except Exception as e:
            print(f"  SKIP qid={row['qid']} video={vid}: {str(e)[:120]}")
            questions_skipped += 1
            continue
        questions_processed += 1

    print(f"Questions processed: {questions_processed}  skipped: {questions_skipped}")
    print()

    records = scene_retrieval.SCENE_DIAG_RECORDS
    if not records:
        print("=== NO RECORDS -- nothing to report ===")
        return

    n_shortcut = sum(1 for r in records if r["branch"] == "shortcut")
    n_descend = sum(1 for r in records if r["branch"] == "descend")
    n_total = len(records)
    print("=== BRANCH DISTRIBUTION (production tau={:.2f}) ===".format(PRODUCTION_TAU))
    print(f"  shortcut : {n_shortcut} / {n_total}  ({n_shortcut / n_total:.1%})")
    print(f"  descend  : {n_descend} / {n_total}  ({n_descend / n_total:.1%})")
    print()

    margins = [r["margin"] for r in records if r["margin"] != float("inf")]
    inf_margins = sum(1 for r in records if r["margin"] == float("inf"))
    edges = [0.0, 0.01, 0.02, PRODUCTION_TAU, 0.1, 0.2, 0.5, 1.0]
    hist = bucket_histogram(margins, edges)
    print("=== MARGIN HISTOGRAM (finite margins only; tau={:.2f} marked) ===".format(PRODUCTION_TAU))
    for label, count in hist:
        print(f"  {label}: {count}")
    print(f"  inf (single shortlisted scene, no cross-scene comparison possible): {inf_margins}")
    if margins:
        print(f"  min={min(margins):.4f}  median={statistics.median(margins):.4f}  max={max(margins):.4f}  mean={statistics.mean(margins):.4f}")
    print()

    descend_records = [r for r in records if r["branch"] == "descend"]
    print(f"=== DESCEND-CASE DIVERGENCE (n={len(descend_records)}) ===")
    if descend_records:
        jaccards = [r["jaccard"] for r in descend_records]
        top1_changed_frac = sum(1 for r in descend_records if r["top1_changed"]) / len(descend_records)
        rank_disps = [r["mean_rank_displacement"] for r in descend_records if r["mean_rank_displacement"] is not None]

        print(f"  mean jaccard overlap   : {statistics.mean(jaccards):.4f}")
        print(f"  median jaccard overlap : {statistics.median(jaccards):.4f}")
        print(f"  min/max jaccard        : {min(jaccards):.4f} / {max(jaccards):.4f}")
        print(f"  %top1_changed          : {top1_changed_frac:.1%}")
        if rank_disps:
            print(f"  mean rank-displacement (shared frames) : {statistics.mean(rank_disps):.4f}")
            print(f"  median rank-displacement (shared frames): {statistics.median(rank_disps):.4f}")
        else:
            print("  mean rank-displacement: N/A (no descend case had any shared frames)")
    else:
        print("  (no descend cases fired -- see decision rule below)")
    print()

    print("=== POOL DENSITY (cross_scene_edges_added / C(post_pull_pool, 2)) ===")
    if descend_records:
        densities = [r["pool_density"] for r in descend_records]
        print(f"  mean density   : {statistics.mean(densities):.4f}")
        print(f"  median density : {statistics.median(densities):.4f}")
        print(f"  min/max        : {min(densities):.4f} / {max(densities):.4f}")
    else:
        print("  N/A (no descend cases)")
    print()

    print("=== RESTATED TAU (report-not-tune) ===")
    restated_tau_table(margins)
    print()

    print("=== DECISION RULE (report only, not acted on) ===")
    if n_descend == 0:
        print(f"  Gate fired 0% ({n_descend}/{n_total}) even on distinct-scene videos.")
        print(f"  tau={PRODUCTION_TAU} is mis-scaled to the real margin distribution -- see restated tau above.")
    elif descend_records:
        mean_jaccard = statistics.mean(r["jaccard"] for r in descend_records)
        top1_changed_frac = sum(1 for r in descend_records if r["top1_changed"]) / len(descend_records)
        if mean_jaccard > 0.9 and top1_changed_frac < 0.1:
            print(f"  Descend jaccard~{mean_jaccard:.3f} (~1.0), top1_changed~{top1_changed_frac:.1%} (~0%).")
            print("  PPR is cosmetic on this data -- recommend dropping PPR, ship coarse+max-sim (2c-i alone).")
        else:
            print(f"  Descend jaccard={mean_jaccard:.3f}, top1_changed={top1_changed_frac:.1%} -- meaningful divergence.")
            print("  PPR earns a place -- proceed to the grounding gate to test whether the divergence is BETTER.")


def main_sweep(dev_rows: list[dict], cached_by_vid: dict) -> None:
    per_mode_records: dict[str, list[dict]] = {name: [] for name, _ in SWEEP_MODES}

    questions_processed = 0
    questions_skipped = 0
    for row in dev_rows:
        vid = row["video"]
        idx = cached_by_vid.get(vid)
        if idx is None or not idx._scene_centroids:
            questions_skipped += 1
            continue

        emb = iris_query._embed_query(row["question"], CFG)

        mode_results: dict[str, dict] = {}
        for mode_name, overrides in SWEEP_MODES:
            cfg_variant = _replace(CFG, **overrides)
            scene_retrieval.SCENE_DIAG_RECORDS.clear()
            try:
                iris_query._build_retrieved(idx, emb, cfg_variant)
            except Exception as e:
                print(f"  SKIP qid={row['qid']} video={vid} mode={mode_name}: {str(e)[:100]}")
                continue
            if not scene_retrieval.SCENE_DIAG_RECORDS:
                continue
            mode_results[mode_name] = scene_retrieval.SCENE_DIAG_RECORDS[-1]

        if "A_all" not in mode_results:
            questions_skipped += 1
            continue

        a_idxs = set(mode_results["A_all"]["result_frame_idxs"])
        for mode_name, rec in mode_results.items():
            this_idxs = set(rec["result_frame_idxs"])
            union = a_idxs | this_idxs
            rec["jaccard_vs_A"] = (len(a_idxs & this_idxs) / len(union)) if union else 1.0
            per_mode_records[mode_name].append(rec)

        questions_processed += 1

    print(f"Questions processed: {questions_processed}  skipped: {questions_skipped}")
    print()

    if questions_processed == 0:
        print("=== NO RECORDS -- nothing to report ===")
        return

    only_descend = {
        name: [r for r in recs if r["branch"] == "descend"]
        for name, recs in per_mode_records.items()
    }

    print("=== CROSS-SCENE EDGE SPARSITY SWEEP (n={} questions, all descend at tau={}) ===".format(
        questions_processed, PRODUCTION_TAU
    ))
    hdr = (
        f"{'mode':>11} | {'n':>4} | {'mean_density':>12} | {'mean_jaccard':>12} | "
        f"{'med_jaccard':>11} | {'%top1_chg':>9} | {'mean_rankdisp':>13} | {'jac_vs_A_mean':>13} | {'jac_vs_A_med':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for mode_name, _ in SWEEP_MODES:
        recs = only_descend[mode_name]
        if not recs:
            print(f"{mode_name:>11} | {'0':>4} | (no descend records)")
            continue
        densities = [r["pool_density"] for r in recs]
        jaccards = [r["jaccard"] for r in recs]
        top1_frac = sum(1 for r in recs if r["top1_changed"]) / len(recs)
        rank_disps = [r["mean_rank_displacement"] for r in recs if r["mean_rank_displacement"] is not None]
        jac_a = [r["jaccard_vs_A"] for r in recs]
        mean_rd = statistics.mean(rank_disps) if rank_disps else float("nan")
        print(
            f"{mode_name:>11} | {len(recs):>4} | {statistics.mean(densities):>12.4f} | "
            f"{statistics.mean(jaccards):>12.4f} | {statistics.median(jaccards):>11.4f} | "
            f"{top1_frac:>8.1%} | {mean_rd:>13.4f} | {statistics.mean(jac_a):>13.4f} | {statistics.median(jac_a):>12.4f}"
        )
    print()

    # ── restated tau (mode-independent -- margin doesn't depend on crossscene mode) ──
    all_margins = [r["margin"] for r in only_descend["A_all"] if r["margin"] != float("inf")]
    print("=== RESTATED TAU (report-not-tune; margin is crossscene-mode-independent) ===")
    restated_tau_table(all_margins)
    print()

    # ── decision (report only, do not pick) ─────────────────────────────────
    print("=== DECISION (report only, not picked) ===")
    baseline_density = statistics.mean(r["pool_density"] for r in only_descend["A_all"]) if only_descend["A_all"] else 0.0
    print(f"  A_all baseline mean pool density: {baseline_density:.4f}")
    found_sparse_nonflat = False
    for mode_name, _ in SWEEP_MODES[1:]:
        recs = only_descend[mode_name]
        if not recs:
            continue
        mean_density = statistics.mean(r["pool_density"] for r in recs)
        mean_jaccard = statistics.mean(r["jaccard"] for r in recs)
        top1_frac = sum(1 for r in recs if r["top1_changed"]) / len(recs)
        well_below = mean_density < 0.5 * baseline_density
        non_cosmetic = mean_jaccard < 0.9 or top1_frac >= 0.1
        verdict = "CANDIDATE (sparse + non-cosmetic)" if (well_below and non_cosmetic) else (
            "sparse but cosmetic (jaccard~1.0)" if well_below else "not meaningfully sparser than A"
        )
        print(f"  {mode_name}: density={mean_density:.4f} (baseline*{mean_density/baseline_density if baseline_density else float('nan'):.2f}), "
              f"jaccard={mean_jaccard:.4f}, top1_changed={top1_frac:.1%}  -> {verdict}")
        if well_below and non_cosmetic:
            found_sparse_nonflat = True
    print()
    if found_sparse_nonflat:
        print("  >= 1 mode drops density well below A_all while keeping PPR non-cosmetic.")
        print("  That mode is the real scene-sparse structure -- run the grounding gate on it.")
    else:
        print("  No sparser mode keeps PPR non-cosmetic (either cosmetic-at-any-density, or only")
        print("  A_all's near-flat density keeps it non-cosmetic). Honest finding: the hierarchy's")
        print("  value looks like candidate-restriction (which scenes/frames enter the pool), not")
        print("  scene-scoped diffusion once inside it. Grounding gate should test THAT framing.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-crossscene", action="store_true",
                         help="run the 2c-iv cross-scene edge sparsity sweep instead of the 2c-iii baseline")
    args = parser.parse_args()

    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")][:N_QUESTIONS]
    print(f"Loaded first {len(dev_rows)} questions from {DEV_JSONL.name}")

    cached_by_vid, _ = load_indices(dev_rows)

    if args.sweep_crossscene:
        main_sweep(dev_rows, cached_by_vid)
    else:
        main_baseline(dev_rows, cached_by_vid)


if __name__ == "__main__":
    main()
