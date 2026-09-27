"""Tuning-half-only re-run of the block_diagonal grounding gate, part (b).

Why this exists: part (b) of scripts/blockdiag_grounding_gate.py selects questions by
index-cache membership only, so its 526 questions included the 27 held-out validation
videos -- after the held-out split had been declared closed (registration commit
30d557b, 2026-07-23; gate commit 3d83b2c, 2026-07-27). This re-run restricts part (b)
to the 59-video tuning half (eval_results/val_videos.txt) and asserts that no held-out
video (eval_results/test_videos.txt) is loaded or scored.

It reuses the original script's functions unchanged, so the retrieval pipeline, config
shape (TAU=0.015, rep_only, top_k=8, half_width=2.2) and bit-identity checks are exactly
those of the committed gate. Only the question set differs.

Writes NEW files only; refuses to overwrite anything. Never touches the committed
blockdiag_grounding_gate_result.{json,md}. Read-only on both index caches (sha256 of
every cache file before and after the run, asserted unchanged). Part (a) is not re-run:
it was already restricted to the tuning half.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import blockdiag_grounding_gate as g  # noqa: E402  (original gate; its functions are reused as-is)

OUT_JSON = REPO / "eval_results" / "blockdiag_grounding_gate_tuning_only.json"
OUT_MD = REPO / "eval_results" / "blockdiag_grounding_gate_tuning_only.md"

for p in (OUT_JSON, OUT_MD):
    assert p not in (g.OUT_JSON, g.OUT_MD), "would overwrite the committed gate artifact"
    if p.exists():
        sys.exit(f"Refusing to overwrite existing {p}")

VAL_VIDEOS = {l.strip() for l in open(g.VAL_VIDEO_LIST, encoding="utf-8") if l.strip()}
HELDOUT_VIDEOS = {l.strip() for l in open(g.TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
assert VAL_VIDEOS and HELDOUT_VIDEOS, "split lists are empty"
assert not (VAL_VIDEOS & HELDOUT_VIDEOS), "tuning/held-out video overlap -- split is corrupt"

_original_loader = g._load_grounded_ssparse_rows


def _tuning_only_loader():
    rows, gsub = _original_loader()
    kept = [r for r in rows if r["video"] in VAL_VIDEOS]
    kept_videos = {r["video"] for r in kept}
    assert not (kept_videos & HELDOUT_VIDEOS), "held-out video leaked into the tuning-only run"
    return kept, gsub


# run_scene_sparse_equivalence() looks this name up in the module at call time,
# so replacing the module attribute restricts the question set without editing the original.
g._load_grounded_ssparse_rows = _tuning_only_loader


def main() -> None:
    print("Hashing cache dirs (pre-run, read-only)...")
    flat_before = g.dir_sha256(g.FLAT_CACHE)
    ssparse_before = g.dir_sha256(g.SSPARSE_CACHE)

    print("\n=== Part (b), tuning half only: fully_connected vs block_diagonal, scene_sparse ===")
    res = g.run_scene_sparse_equivalence()
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches_sample"}, indent=2))

    print("\nHashing cache dirs (post-run, asserting unchanged)...")
    flat_after = g.dir_sha256(g.FLAT_CACHE)
    ssparse_after = g.dir_sha256(g.SSPARSE_CACHE)
    flat_ok = flat_before == flat_after
    ssparse_ok = ssparse_before == ssparse_after

    gate_pass = res["bit_identical"] and flat_ok and ssparse_ok
    result = {
        "outcome": "GATE_PASS" if gate_pass else "GATE_FAIL",
        "scope": "part (b) only, restricted to the tuning half (eval_results/val_videos.txt); "
                 "no held-out video loaded or scored",
        "supersedes_for_citation": "part (b) of eval_results/blockdiag_grounding_gate_result.json "
                                   "(which included the 27 held-out videos)",
        "n_tuning_videos_in_split_list": len(VAL_VIDEOS),
        "part_b_scene_sparse_equivalence_tuning_only": res,
        "flat_cache_untouched": flat_ok,
        "ssparse_cache_untouched": ssparse_ok,
        "flat_cache_sha256_before": flat_before,
        "flat_cache_sha256_after": flat_after,
        "ssparse_cache_sha256_before": ssparse_before,
        "ssparse_cache_sha256_after": ssparse_after,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Block-diagonal grounding gate -- part (b), tuning half only",
        "",
        f"**Outcome: {result['outcome']}**",
        "",
        "Re-run of part (b) of `scripts/blockdiag_grounding_gate.py` restricted to the 59-video tuning",
        "half. The original part (b) selected by cache membership and so included the 27 held-out",
        "videos after that split was declared closed; this run loads and scores none of them.",
        "The original script's functions are reused unchanged; only the question set differs.",
        "",
        f"- N questions: {res['n_questions']}, videos: {res['n_videos']}",
        f"- peak_in_gold rate -- fully_connected: {res['peak_in_gold_rate_fully_connected']}, "
        f"block_diagonal: {res['peak_in_gold_rate_block_diagonal']}",
        f"- mIoP -- fully_connected: {res['mIoP_fully_connected']}, "
        f"block_diagonal: {res['mIoP_block_diagonal']}",
        f"- Per-question bit-identical (retrieved order, peak, span, IoP, peak_in_gold): "
        f"{res['bit_identical']} ({res['n_mismatches']} mismatches)",
        "",
        "## Cache integrity",
        "",
        f"- `eval/data/nextqa/index_cache` sha256 unchanged: {flat_ok}",
        f"- `eval/data/nextqa/index_cache_ssparse` sha256 unchanged: {ssparse_ok}",
        "",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}\n\n=== GATE RESULT: {result['outcome']} ===")
    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
