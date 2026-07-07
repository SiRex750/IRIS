"""NExT-GQA caption-free grounding ablation.

No LLM, no captions, no aria.generate.
Scores retrieved-frame temporal overlap with NExT-GQA gold spans.

Arms: lambda in [1.0, 0.5, 0.0] (PPR) + uniform baseline.
Per-top_k table: arm | FiW(all) | FiW(C) | FiW(T) | IoP(all) | n
Paired bootstrap 95% CIs for three contrasts at each top_k.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris.iris_config import IRISConfig
from eval.grounding_scorer import load_indexes, score_grounding_arm

DATA_DIR  = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
GQA_JSON  = DATA_DIR / "gsub_val.json"

ARMS   = [1.0, 0.5, 0.0]
TOP_KS = [8, 12]

BASE_CFG = dict(
    ranking_mode="ppr",
    ppr_damping=0.5,
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
)


N_BOOTSTRAP = 10_000
RNG_SEED    = 42


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "  N/A "


def _bootstrap_paired_ci(
    pq_a: dict[tuple[str, str], float],
    pq_b: dict[tuple[str, str], float],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Paired bootstrap CI for mean(FiW_a - FiW_b) over shared questions.

    Returns {"mean": float, "ci_lo": float, "ci_hi": float, "p_pos": float, "n": int}.
    p_pos = fraction of bootstrap samples where mean(delta) > 0.
    """
    shared = sorted(set(pq_a) & set(pq_b))
    if not shared:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "p_pos": None, "n": 0}
    diffs = np.array([pq_a[k] - pq_b[k] for k in shared], dtype=np.float64)
    rng   = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    return {
        "mean":  float(diffs.mean()),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "p_pos": float((boot_means > 0).mean()),
        "n":     n,
    }


def main() -> None:
    if not GQA_JSON.exists():
        print(f"FATAL: {GQA_JSON} not found. Run STEP 1 to download gsub_val.json.",
              file=sys.stderr)
        sys.exit(1)

    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    gsub     = json.load(open(GQA_JSON, encoding="utf-8"))

    cached_vids = {p.stem for p in CACHE_DIR.glob("*.npz")}

    # Filter to grounded-AND-cached: video cached AND str(qid) in gsub[video]["location"]
    grounded_rows = [
        r for r in dev_rows
        if r["video"] in cached_vids
        and r["video"] in gsub
        and str(r["qid"]) in gsub[r["video"]]["location"]
    ]

    print(f"dev_100 total:          {len(dev_rows)}")
    print(f"cached-video subset:    {len([r for r in dev_rows if r['video'] in cached_vids])}")
    print(f"grounded-AND-cached:    {len(grounded_rows)}")
    fam_counts = {}
    for r in grounded_rows:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    print(f"  by family: {fam_counts}")
    print()

    # D must be zero — hard stop if not.
    if fam_counts.get("D", 0) != 0:
        print(
            f"ERROR: D-family count = {fam_counts['D']} (expected 0). "
            "Join is wrong — aborting.",
            file=sys.stderr,
        )
        sys.exit(2)

    for top_k in TOP_KS:
        print(f"=== TOP_K={top_k} ===")

        hdr = f"{'arm':<12} | {'FiW(all)':>8} | {'FiW(C)':>7} | {'FiW(T)':>7} | {'IoP(all)':>8} | {'n':>4}"
        print(hdr)
        print("-" * len(hdr))

        # Load fresh indexes once per top_k — all arms share the same objects so
        # there is no reload overhead, and PPR node-state mutations (last_retrieval_score,
        # retrieval_contributions) from one arm cannot bleed into the next because the
        # scoring path reads only codec_conf and node.embedding, which are never mutated.
        idxs = load_indexes(grounded_rows, CACHE_DIR)

        # PPR arms — collect per_question for bootstrap
        pq: dict[float, dict] = {}
        for lam in ARMS:
            cfg = IRISConfig(l2_retrieve_top_k=top_k, ppr_lambda=lam, **BASE_CFG)
            stats = score_grounding_arm(grounded_rows, CACHE_DIR, cfg, gsub, loaded=idxs)
            ov  = stats["overall"]
            byf = stats["by_family"]
            pq[lam] = stats["per_question"]
            label = f"lam={lam}"
            print(
                f"{label:<12} | {_fmt(ov['fiw']):>8} | "
                f"{_fmt(byf.get('C', {}).get('fiw')):>7} | "
                f"{_fmt(byf.get('T', {}).get('fiw')):>7} | "
                f"{_fmt(ov['iop']):>8} | "
                f"{ov['n']:>4}"
            )

        # Uniform baseline (also uses idxs — no retrieval happens, but loaded is accepted
        # for interface consistency and avoids a redundant reload)
        cfg_unif = IRISConfig(l2_retrieve_top_k=top_k, ppr_lambda=0.5, **BASE_CFG)
        stats_u = score_grounding_arm(
            grounded_rows, CACHE_DIR, cfg_unif, gsub, arm_name="uniform", loaded=idxs
        )
        ov_u  = stats_u["overall"]
        byf_u = stats_u["by_family"]
        pq_unif = stats_u["per_question"]
        print(
            f"{'uniform':<12} | {_fmt(ov_u['fiw']):>8} | "
            f"{_fmt(byf_u.get('C', {}).get('fiw')):>7} | "
            f"{_fmt(byf_u.get('T', {}).get('fiw')):>7} | "
            f"{_fmt(ov_u['iop']):>8} | "
            f"{ov_u['n']:>4}"
        )
        print()

        # ── Paired bootstrap CIs ──────────────────────────────────────────────
        contrasts = [
            ("lam=1.0 - lam=0.5", pq[1.0], pq[0.5]),
            ("lam=1.0 - uniform", pq[1.0], pq_unif),
            ("lam=0.5 - uniform", pq[0.5], pq_unif),
        ]
        print(f"  Paired bootstrap CIs (n_boot={N_BOOTSTRAP}, seed={RNG_SEED}):")
        ci_hdr = f"  {'contrast':<22} | {'mean_d':>8} | {'95% CI':^17} | {'p(d>0)':>8} | {'n':>4}"
        print(ci_hdr)
        print("  " + "-" * (len(ci_hdr) - 2))
        for label, pa, pb in contrasts:
            r = _bootstrap_paired_ci(pa, pb)
            if r["mean"] is None:
                print(f"  {label:<22} | {'N/A':>8} | {'N/A':^17} | {'N/A':>8} | {r['n']:>4}")
            else:
                ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                print(
                    f"  {label:<22} | {r['mean']:>+8.4f} | {ci_str:^17} | "
                    f"{r['p_pos']:>8.4f} | {r['n']:>4}"
                )
        print()


if __name__ == "__main__":
    main()
