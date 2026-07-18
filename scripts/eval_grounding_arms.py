"""Phase 6, task 2c-iv: the Pillar-3 QUALITY GATE.

Caption-free temporal grounding (FiW/IoP vs NExT-GQA gold spans -- the
contamination-proof forward metric; NOT QA accuracy, which is GATE_BLIND-dead)
across FOUR arms on the SAME question set:

  1. flat                       -- production ppr, graph_mode=flat (the
                                    previously-validated baseline, FiW>uniform).
  2. scene_sparse / rep_only    -- the shipped scene_sparse default (2c-iv).
  3. scene_sparse / threshold@75
  4. scene_sparse / threshold@90

THE GATE: scene_sparse(rep_only) FiW >= flat FiW => hierarchy is
quality-neutral-or-better, justified on cost/scale grounds alone.
If BELOW flat => failing invariant. Report and STOP; do not tune to recover.

Also reports: realized shortcut rate at the restated tau=0.015, and grounding
of shortcut-branch vs descend-branch questions separately (does the max-sim
short-circuit hold grounding on its own, or is it riding on descent?).

--max-n: lift the question-set cap from the first 50 dev_100 questions to ALL
of dev_100 (the largest reachable N given grounded+both-cached availability),
and add a per-family (C/T/D) breakdown -- the confirmation run for whether the
N=33 (all-family-C) gate pass generalizes, in particular to family T (temporal),
which a scene-boundary method could plausibly hurt even if it helps causal (C).

VERIFY: python scripts/eval_grounding_arms.py
        python scripts/eval_grounding_arms.py --max-n
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.scene_retrieval as scene_retrieval
from iris.iris_config import IRISConfig
from eval.grounding_scorer import frames_in_window, iop, load_indexes, uniform_ts
from eval.span import FROZEN_HALF_WIDTH_SECONDS, predict_span

# Predicted-span mode for the retrieval arms below (eval/span.py). half_width
# reads the single frozen source (eval.span.FROZEN_HALF_WIDTH_SECONDS = 2.2s,
# duration-anchor method, DECISIONS.md 2026-07-18) rather than a local literal.
SPAN_MODE = "ppr_peak"
SPAN_HALF_WIDTH: float | None = FROZEN_HALF_WIDTH_SECONDS

DATA_DIR       = REPO / "eval" / "data" / "nextqa"
FLAT_CACHE     = DATA_DIR / "index_cache"            # flat-mode ingests
SSPARSE_CACHE  = DATA_DIR / "index_cache_ssparse"     # scene_sparse-mode ingests (scene_id assigned)
DEV_JSONL      = DATA_DIR / "dev_100.jsonl"
GQA_JSON       = DATA_DIR / "gsub_val.json"

N_QUESTIONS_DEFAULT = 50  # first-50-question batch used in 2c-iii/2c-iv diagnostics
TOP_K = 8
TAU = 0.015  # restated tau (2c-iv step 0), report-not-tune

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=TOP_K,
)

ARMS = [
    ("flat",              dict(BASE, graph_mode="flat")),
    ("scene_sparse/rep",  dict(BASE, graph_mode="scene_sparse", scene_shortcut_margin=TAU, scene_crossscene_mode="rep_only")),
    ("scene_sparse/t75",  dict(BASE, graph_mode="scene_sparse", scene_shortcut_margin=TAU, scene_crossscene_mode="threshold", scene_crossscene_threshold_pctile=75.0)),
    ("scene_sparse/t90",  dict(BASE, graph_mode="scene_sparse", scene_shortcut_margin=TAU, scene_crossscene_mode="threshold", scene_crossscene_threshold_pctile=90.0)),
]


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "  N/A  "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", action="store_true",
                         help="use ALL of dev_100.jsonl (not just the first 50) -- the largest "
                              "reachable N given grounded+both-cached availability")
    args = parser.parse_args()

    if not GQA_JSON.exists():
        print(f"FATAL: {GQA_JSON} not found.", file=sys.stderr)
        sys.exit(1)

    all_dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    dev_rows = all_dev_rows if args.max_n else all_dev_rows[:N_QUESTIONS_DEFAULT]
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))

    flat_cached = {p.stem for p in FLAT_CACHE.glob("*.npz")}
    ssparse_cached = {p.stem for p in SSPARSE_CACHE.glob("*.npz")}
    both_cached = flat_cached & ssparse_cached

    grounded_rows = [
        r for r in dev_rows
        if r["video"] in both_cached
        and r["video"] in gsub
        and str(r["qid"]) in gsub[r["video"]]["location"]
    ]

    label = "ALL dev_100" if args.max_n else f"first-{N_QUESTIONS_DEFAULT} dev_100"
    print(f"{label} questions: {len(dev_rows)}")
    print(f"grounded (in gsub) AND cached in BOTH flat+scene_sparse caches: {len(grounded_rows)}")
    fam_counts: dict = {}
    for r in grounded_rows:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    print(f"  by family: {fam_counts}")
    print(f"REALIZED N (used for all 4 arms) = {len(grounded_rows)}")
    print()

    if not grounded_rows:
        print("FATAL: no grounded+cached questions.", file=sys.stderr)
        sys.exit(2)

    # Fresh index objects per arm -- flat arm needs the FULLY-CONNECTED flat
    # graph; scene_sparse arms need the block-diagonal scene_sparse graph with
    # scene_id assigned. These are NOT interchangeable (flat's production PPR
    # path runs over the whole graph with no node_subset -- if handed a
    # block-diagonal graph it would silently understate the flat baseline).
    flat_idxs = load_indexes(grounded_rows, FLAT_CACHE)
    ssparse_idxs = load_indexes(grounded_rows, SSPARSE_CACHE)

    from iris.query import _embed_query, _build_retrieved

    # duration lookup for the uniform floor
    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in {r["video"] for r in grounded_rows}}

    family_by_question: dict[tuple, str] = {(r["video"], str(r["qid"])): r["family"] for r in grounded_rows}
    families = sorted(set(family_by_question.values()))

    per_arm_pq: dict[str, dict] = {}          # arm -> (vid,qid) -> fiw
    per_arm_pq_iop: dict[str, dict] = {}      # arm -> (vid,qid) -> iop
    per_arm_agg: dict[str, dict] = {}
    branch_by_question: dict[tuple, str] = {}  # (video,qid) -> "shortcut"/"descend", from rep_only arm

    for arm_name, cfg_kwargs in ARMS:
        cfg = IRISConfig(**cfg_kwargs)
        idxs = flat_idxs if cfg_kwargs.get("graph_mode") == "flat" else ssparse_idxs

        if "rep" in arm_name:
            scene_retrieval.SCENE_DIAG_RECORDS.clear()
            cfg = IRISConfig(**{**cfg_kwargs, "scene_diag": True})

        fiw_list, iop_list = [], []
        pq: dict[tuple, float] = {}
        pq_iop: dict[tuple, float] = {}
        for row in grounded_rows:
            vid, qid = row["video"], str(row["qid"])
            index = idxs.get(vid)
            if index is None:
                continue
            gold_spans = gsub[vid]["location"][qid]
            emb = _embed_query(row["question"], cfg)
            retrieved = _build_retrieved(index, emb, cfg)
            ts = [f["timestamp"] for f in retrieved]
            span = predict_span(
                retrieved, mode=SPAN_MODE, half_width=SPAN_HALF_WIDTH,
                duration=duration_by_vid.get(vid),
            )
            fiw_val = frames_in_window(ts, gold_spans)
            iop_val = iop(span, gold_spans)
            fiw_list.append(fiw_val)
            iop_list.append(iop_val)
            pq[(vid, qid)] = fiw_val
            pq_iop[(vid, qid)] = iop_val

            if "rep" in arm_name and scene_retrieval.SCENE_DIAG_RECORDS:
                branch_by_question[(vid, qid)] = scene_retrieval.SCENE_DIAG_RECORDS[-1]["branch"]

        per_arm_pq[arm_name] = pq
        per_arm_pq_iop[arm_name] = pq_iop
        per_arm_agg[arm_name] = {
            "fiw": statistics.mean(fiw_list) if fiw_list else None,
            "iop": statistics.mean(iop_list) if iop_list else None,
            "n": len(fiw_list),
        }

    # ── uniform floor ────────────────────────────────────────────────────────
    fiw_u, iop_u = [], []
    pq_u: dict[tuple, float] = {}
    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        gold_spans = gsub[vid]["location"][qid]
        ts = uniform_ts(duration_by_vid[vid], TOP_K)
        # No retrieval score for synthetic uniform timestamps -- minmax is the
        # correct construction here, not a fallback of convenience.
        span_u = predict_span([{"timestamp": t} for t in ts], mode="minmax")
        fiw_val = frames_in_window(ts, gold_spans)
        fiw_u.append(fiw_val)
        iop_u.append(iop(span_u, gold_spans))
        pq_u[(vid, qid)] = fiw_val
    uniform_agg = {"fiw": statistics.mean(fiw_u), "iop": statistics.mean(iop_u), "n": len(fiw_u)}
    per_arm_pq["uniform"] = pq_u

    # ── report ───────────────────────────────────────────────────────────────
    print("=== 4-ARM GROUNDING TABLE (FiW = fraction of retrieved timestamps in gold window) ===")
    hdr = f"{'arm':<20} | {'FiW':>8} | {'IoP':>8} | {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for arm_name, _ in ARMS:
        a = per_arm_agg[arm_name]
        print(f"{arm_name:<20} | {_fmt(a['fiw']):>8} | {_fmt(a['iop']):>8} | {a['n']:>4}")
    print(f"{'uniform floor':<20} | {_fmt(uniform_agg['fiw']):>8} | {_fmt(uniform_agg['iop']):>8} | {uniform_agg['n']:>4}")
    print()

    # ── per-family breakdown (the T row is load-bearing) ────────────────────
    print("=== PER-FAMILY GROUNDING (C=causal, T=temporal, D=descriptive) ===")
    fhdr = f"{'arm':<20} | " + " | ".join(f"{fam:>16}" for fam in families) + " |  n_total"
    print(fhdr)
    print("-" * len(fhdr))
    for arm_name, _ in ARMS + [("uniform", None)]:
        pq = per_arm_pq[arm_name]
        cells = []
        for fam in families:
            keys = [k for k, f in family_by_question.items() if f == fam and k in pq]
            if keys:
                mean_fiw = statistics.mean(pq[k] for k in keys)
                cells.append(f"{mean_fiw:>10.4f} (n={len(keys):>2})")
            else:
                cells.append(f"{'N/A':>16}")
        n_total = sum(1 for k in pq if k in family_by_question)
        print(f"{arm_name:<20} | " + " | ".join(cells) + f" | {n_total:>7}")
    print()

    if "T" in families:
        print("=== T-FAMILY (temporal) CALLOUT -- the load-bearing number ===")
        flat_t = [per_arm_pq["flat"][k] for k, f in family_by_question.items() if f == "T" and k in per_arm_pq["flat"]]
        for arm_name, _ in ARMS[1:]:
            arm_t = [per_arm_pq[arm_name][k] for k, f in family_by_question.items() if f == "T" and k in per_arm_pq[arm_name]]
            if flat_t and arm_t:
                print(f"  {arm_name:<20} T-FiW={statistics.mean(arm_t):.4f}  flat T-FiW={statistics.mean(flat_t):.4f}  "
                      f"delta={statistics.mean(arm_t) - statistics.mean(flat_t):+.4f}  n={len(arm_t)}")
        print()
    else:
        print("=== T-FAMILY CALLOUT === N/A -- no family-T questions reached both caches at this N.")
        print()

    flat_fiw = per_arm_agg["flat"]["fiw"]
    rep_fiw = per_arm_agg["scene_sparse/rep"]["fiw"]
    t75_fiw = per_arm_agg["scene_sparse/t75"]["fiw"]
    t90_fiw = per_arm_agg["scene_sparse/t90"]["fiw"]
    uniform_fiw = uniform_agg["fiw"]

    print("=== ARM vs FLAT vs UNIFORM FLOOR ===")
    for arm_name, _ in ARMS:
        fiw = per_arm_agg[arm_name]["fiw"]
        if fiw is None:
            continue
        vs_flat = fiw - flat_fiw
        vs_unif = fiw - uniform_fiw
        print(f"  {arm_name:<20} FiW={fiw:.4f}  vs_flat={vs_flat:+.4f}  vs_uniform={vs_unif:+.4f}")
    print()

    # ── THE GATE ─────────────────────────────────────────────────────────────
    print("=== THE GATE ===")
    if rep_fiw is None or flat_fiw is None:
        print("  N/A -- missing data for flat or scene_sparse/rep_only arm.")
    elif rep_fiw >= flat_fiw:
        print(f"  PASS: scene_sparse(rep_only) FiW={rep_fiw:.4f} >= flat FiW={flat_fiw:.4f}")
        print("  Hierarchy is quality-neutral-or-better; justified on cost/scale grounds.")
    else:
        print(f"  FAIL: scene_sparse(rep_only) FiW={rep_fiw:.4f} < flat FiW={flat_fiw:.4f}")
        print("  This is a failing invariant. STOP. Surfacing as-is -- do NOT tune to recover.")
    print()

    # ── per-family failing invariant check -- do NOT average this away ──────
    print("=== PER-FAMILY GATE CHECK (report only -- a per-family drop is NOT averaged away) ===")
    any_family_fail = False
    for fam in families:
        flat_fam = [per_arm_pq["flat"][k] for k, f in family_by_question.items() if f == fam and k in per_arm_pq["flat"]]
        if not flat_fam:
            continue
        flat_fam_mean = statistics.mean(flat_fam)
        for arm_name, _ in ARMS[1:]:
            arm_fam = [per_arm_pq[arm_name][k] for k, f in family_by_question.items() if f == fam and k in per_arm_pq[arm_name]]
            if not arm_fam:
                continue
            arm_fam_mean = statistics.mean(arm_fam)
            status = "OK" if arm_fam_mean >= flat_fam_mean else "BELOW FLAT"
            if status == "BELOW FLAT":
                any_family_fail = True
            print(f"  family={fam}  {arm_name:<20} FiW={arm_fam_mean:.4f}  vs flat={flat_fam_mean:.4f}  -> {status}")
    if any_family_fail:
        print("  >= 1 (family, arm) pair fell BELOW flat -- surfaced above, NOT averaged away by the aggregate PASS.")
    else:
        print("  No (family, arm) pair fell below flat -- the aggregate PASS holds per-family too.")
    print()

    # ── shortcut vs descend breakdown (rep_only arm) ────────────────────────
    print(f"=== REALIZED SHORTCUT RATE (rep_only arm, tau={TAU}) ===")
    branches = list(branch_by_question.values())
    n_shortcut = sum(1 for b in branches if b == "shortcut")
    n_descend = sum(1 for b in branches if b == "descend")
    n_branch_total = len(branches)
    if n_branch_total:
        print(f"  shortcut: {n_shortcut}/{n_branch_total} ({n_shortcut/n_branch_total:.1%})")
        print(f"  descend : {n_descend}/{n_branch_total} ({n_descend/n_branch_total:.1%})")

        rep_pq = per_arm_pq["scene_sparse/rep"]
        shortcut_fiws = [rep_pq[k] for k, b in branch_by_question.items() if b == "shortcut" and k in rep_pq]
        descend_fiws = [rep_pq[k] for k, b in branch_by_question.items() if b == "descend" and k in rep_pq]
        print(f"  mean FiW | shortcut cases (n={len(shortcut_fiws)}): {statistics.mean(shortcut_fiws):.4f}" if shortcut_fiws else "  mean FiW | shortcut cases: N/A (none fired)")
        print(f"  mean FiW | descend  cases (n={len(descend_fiws)}): {statistics.mean(descend_fiws):.4f}" if descend_fiws else "  mean FiW | descend cases: N/A")
    else:
        print("  N/A -- no branch records captured.")


if __name__ == "__main__":
    main()
