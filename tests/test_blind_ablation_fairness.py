"""Fairness checks for the blind-ablation artifacts under tuning/blind_ablation/.

These are integration-style tests over the artifacts scripts/blind_ablation_eval.py
produces (not unit tests of the script's internals): they assert the four arms
actually differed only in the ways the task spec allows, and that the test
split was never touched.

Owner: Track B (diagnostic ablation)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "tuning" / "blind_ablation"
SCRIPT_PATH = REPO / "scripts" / "blind_ablation_eval.py"

ARM_CSVS = {
    "A": OUT_DIR / "arm_A_full_per_question.csv",
    "B": OUT_DIR / "arm_B_blind_strict_per_question.csv",
    "C": OUT_DIR / "arm_C_blind_fair_per_question.csv",
    "D": OUT_DIR / "arm_D_shuffle_per_question.csv",
}


def _artifacts_present() -> bool:
    return all(p.exists() for p in ARM_CSVS.values()) and (OUT_DIR / "shuffle_map.json").exists()


pytestmark = pytest.mark.skipif(
    not _artifacts_present(),
    reason="blind_ablation artifacts not present -- run scripts/blind_ablation_eval.py first",
)


def _load_rows(path: Path) -> dict[tuple[str, str], dict]:
    with open(path, newline="") as f:
        return {(r["video"], r["qid"]): r for r in csv.DictReader(f)}


def test_arms_b_and_c_received_zero_length_context():
    rows_b = _load_rows(ARM_CSVS["B"])
    rows_c = _load_rows(ARM_CSVS["C"])
    bad_b = [k for k, r in rows_b.items() if int(r["context_char_len"]) != 0]
    bad_c = [k for k, r in rows_c.items() if int(r["context_char_len"]) != 0]
    assert not bad_b, f"arm B had non-empty context for: {bad_b[:5]}"
    assert not bad_c, f"arm C had non-empty context for: {bad_c[:5]}"
    assert len(rows_b) > 0 and len(rows_c) > 0


def test_shuffle_map_is_a_true_derangement():
    mapping = json.loads((OUT_DIR / "shuffle_map.json").read_text())
    assert len(mapping) == 112, f"expected 112 videos in shuffle_map, got {len(mapping)}"
    fixed_points = [k for k, v in mapping.items() if k == v]
    assert not fixed_points, f"shuffle_map has fixed points (video mapped to itself): {fixed_points}"
    assert sorted(mapping.values()) == sorted(mapping.keys()), "shuffle_map is not a bijection over the video set"


def test_four_arms_saw_byte_identical_user_prompt_per_qid():
    rows_a = _load_rows(ARM_CSVS["A"])
    rows_b = _load_rows(ARM_CSVS["B"])
    rows_c = _load_rows(ARM_CSVS["C"])
    rows_d = _load_rows(ARM_CSVS["D"])
    keys = set(rows_a) & set(rows_b) & set(rows_c) & set(rows_d)
    assert len(keys) == 639, f"expected 639 shared (video, qid) keys, got {len(keys)}"
    mismatches = []
    for k in keys:
        prompts = {rows_a[k]["user_prompt"], rows_b[k]["user_prompt"],
                   rows_c[k]["user_prompt"], rows_d[k]["user_prompt"]}
        if len(prompts) != 1:
            mismatches.append(k)
    assert not mismatches, f"user_prompt differed across arms for: {mismatches[:5]}"


def test_no_test_split_file_referenced_or_opened():
    source = SCRIPT_PATH.read_text()
    forbidden = ["gsub_test", "test.csv", "NExT-GQA/test", "nextqa/test"]
    hits = [tok for tok in forbidden if tok in source]
    assert not hits, f"scripts/blind_ablation_eval.py references forbidden test-split token(s): {hits}"

    test_split_candidates = [
        REPO / "eval" / "data" / "nextqa" / "gsub_test.json",
        REPO / "eval" / "data" / "nextqa" / "test.csv",
    ]
    for p in test_split_candidates:
        if p.exists():
            # We only assert the ablation script never opens it (static check
            # above); if it happens to exist on disk from unrelated setup we
            # don't assert its mtime here since other processes may touch it.
            pass


def test_arms_a_and_d_used_nonempty_context():
    rows_a = _load_rows(ARM_CSVS["A"])
    rows_d = _load_rows(ARM_CSVS["D"])
    empty_a = [k for k, r in rows_a.items() if int(r["context_char_len"]) == 0]
    empty_d = [k for k, r in rows_d.items() if int(r["context_char_len"]) == 0]
    assert not empty_a, f"arm A (real captions) had empty context for: {empty_a[:5]}"
    assert not empty_d, f"arm D (shuffled captions) had empty context for: {empty_d[:5]}"
