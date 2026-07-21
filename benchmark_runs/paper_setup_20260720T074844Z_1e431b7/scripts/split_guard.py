"""Leakage guard: import and call before any tuning or test-set command runs.

Not wired into iris/ (setup-only artifact). Intended usage from future command scripts:

    from split_guard import guard_tuning_command, guard_official_test_command
    guard_tuning_command(split_name, config_path)
    guard_official_test_command(split_name)
"""
import json
import os
import sys

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "split_manifest.json")

TUNING_SPLITS = {"val_tune", "val_confirm"}
OFFICIAL_TEST_SPLIT = "official_test"


def _load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def guard_tuning_command(split_name: str, config_path: str = ""):
    """Reject any tuning invocation that targets the official test split."""
    if split_name == OFFICIAL_TEST_SPLIT:
        print(
            f"REJECTED: tuning command may not target split='{split_name}'. "
            f"Tuning is restricted to {sorted(TUNING_SPLITS)}.",
            file=sys.stderr,
        )
        sys.exit(2)
    if "test" in config_path.lower() or "official" in config_path.lower():
        print(
            f"REJECTED: config path '{config_path}' looks like a test-split path; "
            "tuning commands must not reference test paths.",
            file=sys.stderr,
        )
        sys.exit(2)
    manifest = _load_manifest()
    if split_name not in manifest["partitions"]:
        print(f"REJECTED: unknown split '{split_name}'.", file=sys.stderr)
        sys.exit(2)


def guard_question_id(split_name: str, question_id: str):
    manifest = _load_manifest()
    test_qids = set(manifest["partitions"][OFFICIAL_TEST_SPLIT]["question_ids"])
    if split_name in TUNING_SPLITS and question_id in test_qids:
        print(f"REJECTED: question_id '{question_id}' belongs to official_test, not {split_name}.", file=sys.stderr)
        sys.exit(2)


def guard_official_test_command(split_name: str):
    """Reject the official-test command if it is accidentally pointed at a validation split."""
    if split_name != OFFICIAL_TEST_SPLIT:
        print(
            f"REJECTED: official-test command was supplied split='{split_name}'. "
            f"It must be exactly '{OFFICIAL_TEST_SPLIT}'. Refusing to silently substitute a validation split.",
            file=sys.stderr,
        )
        sys.exit(2)
    manifest = _load_manifest()
    ot = manifest["partitions"][OFFICIAL_TEST_SPLIT]
    if ot["video_count"] == 0:
        print(
            "REJECTED: official_test partition is unpopulated (0 videos) -- official NExT-GQA test "
            "annotations were not verifiable locally at setup time. See setup_failures.jsonl. "
            "The official test command cannot run until this is resolved.",
            file=sys.stderr,
        )
        sys.exit(3)


if __name__ == "__main__":
    # smoke self-test, no video/model involved
    guard_tuning_command("val_tune", "configs/sweep/peak_rule/ppr_score_legacy.json")
    print("OK: val_tune tuning call accepted")
    try:
        guard_tuning_command("official_test", "configs/final.json")
        print("FAIL: should have rejected")
        sys.exit(1)
    except SystemExit as e:
        assert e.code == 2
        print("OK: tuning-on-test-split correctly rejected")
    try:
        guard_official_test_command("val_confirm")
        print("FAIL: should have rejected")
        sys.exit(1)
    except SystemExit as e:
        assert e.code == 2
        print("OK: official-test-command-on-validation-split correctly rejected")
    try:
        guard_official_test_command("official_test")
        print("FAIL: should have rejected (unpopulated test partition)")
        sys.exit(1)
    except SystemExit as e:
        assert e.code == 3
        print("OK: official-test-command correctly refuses to run against unpopulated test partition")
