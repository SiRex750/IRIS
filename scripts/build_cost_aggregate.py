#!/usr/bin/env python3
"""Re-derive eval_results/build_cost_final.json's computed fields from the six
raw per-round timing files in eval_results/build_cost_final_raw/.

This is an independent re-derivation for verification purposes. It does NOT
read or write eval_results/build_cost_final.json; it writes a separate output
file (default: eval_results/build_cost_final_rederived.json).

Usage:
    python scripts/build_cost_aggregate.py [--out PATH]
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "eval_results" / "build_cost_final_raw"

RAW_FILES = {
    "old": ["old_r1.json", "old_r2.json", "old_r3.json"],
    "new": ["new_r1.json", "new_r2.json", "new_r3.json"],
}

THRESHOLD_PCT = 2.0  # matches old_arm_cross_round_verdict.threshold_pct in the original artifact


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def spread_pct_of_median(values):
    """(max-min)/median * 100, matching the original artifact's
    *_cross_round_spread_pct_of_median fields (verified against old/new arms)."""
    return (max(values) - min(values)) / statistics.median(values) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "eval_results" / "build_cost_final_rederived.json"),
    )
    args = ap.parse_args()
    out_path = Path(args.out)

    if out_path.resolve() == (REPO_ROOT / "eval_results" / "build_cost_final.json").resolve():
        print("Refusing to overwrite build_cost_final.json", file=sys.stderr)
        sys.exit(1)

    raw = {}
    input_hashes = {}
    for arm, fnames in RAW_FILES.items():
        raw[arm] = []
        for fname in fnames:
            fpath = RAW_DIR / fname
            data = json.loads(fpath.read_text())
            raw[arm].append(data)
            input_hashes[fname] = sha256_of(fpath)

    # Per-round medians, as recorded by each raw file (median of that round's per_iter_sec).
    old_round_medians = [r["median_per_iter_sec"] for r in raw["old"]]
    new_round_medians = [r["median_per_iter_sec"] for r in raw["new"]]

    old_spread_pct = spread_pct_of_median(old_round_medians)
    new_spread_pct = spread_pct_of_median(new_round_medians)

    cross_round_medians = {
        "old_round_medians_sec": old_round_medians,
        "new_round_medians_sec": new_round_medians,
        "old_cross_round_spread_pct_of_median": old_spread_pct,
        "new_cross_round_spread_pct_of_median": new_spread_pct,
    }

    # Pooled: flatten all per_iter_sec values across the 3 rounds of an arm, take the median.
    old_pooled = [x for r in raw["old"] for x in r["per_iter_sec"]]
    new_pooled = [x for r in raw["new"] for x in r["per_iter_sec"]]
    old_pooled_median = statistics.median(old_pooled)
    new_pooled_median = statistics.median(new_pooled)

    pooled = {
        "old_pooled_n": len(old_pooled),
        "new_pooled_n": len(new_pooled),
        "old_pooled_median_sec": old_pooled_median,
        "new_pooled_median_sec": new_pooled_median,
        "ratio_pooled_medians": old_pooled_median / new_pooled_median,
    }

    # Per-round ratios: old round median / new round median, for each of the 3 rounds.
    per_round_ratios = [
        old_round_medians[i] / new_round_medians[i] for i in range(len(old_round_medians))
    ]

    ratio_summary = {
        "per_round_ratios": per_round_ratios,
        "min_per_round_ratio": min(per_round_ratios),
        "max_per_round_ratio": max(per_round_ratios),
        "ratio_from_pooled_medians": pooled["ratio_pooled_medians"],
    }

    old_arm_cross_round_verdict = {
        "threshold_pct": THRESHOLD_PCT,
        "observed_spread_pct": old_spread_pct,
        "varies_materially_within_this_run": old_spread_pct > THRESHOLD_PCT,
    }

    git_head = git("rev-parse", "HEAD")
    dirty_count = len([l for l in git("status", "--porcelain").splitlines() if l.strip()])

    output = {
        "provenance": {
            "script": "scripts/build_cost_aggregate.py",
            "purpose": "independent re-derivation of eval_results/build_cost_final.json for verification; does not overwrite it",
            "input_files_sha256": input_hashes,
            "git_head": git_head,
            "dirty_file_count": dirty_count,
        },
        "cross_round_medians": cross_round_medians,
        "pooled": pooled,
        "ratio_summary": ratio_summary,
        "old_arm_cross_round_verdict": old_arm_cross_round_verdict,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
