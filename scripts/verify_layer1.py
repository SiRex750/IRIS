"""Offline verification of Cerberus layer 1 (deterministic metadata checking).

Loads diag_v2_capture.jsonl, hand-parses the 6 known deterministically-
checkable metadata claims (frame_idx + field + stated_value) into
MetadataClaim objects, verifies each against the real VIRAT_S_000102 index,
and prints source_text | field | stated | stored | verdict.

This is a fixture, not a general prose parser: the claim contract (typed
JSON from ARIA) makes parsing metadata claims out of prose unnecessary in
production. The 6 claims below were located using the same METADATA_RE
taxonomy regex as scripts/diag_v3_visual_scoping.py's --ceiling mode.

Also prints an AnswerClaims JSON round-trip (serialize -> parse -> equality)
for one hand-built example containing all four claim types.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
from iris.cerberus_layers import verify_metadata_claim
from iris.claim_contract import (
    AbsenceClaim,
    AnswerClaims,
    GlobalClaim,
    MetadataClaim,
    VisualClaim,
)

CAPTURE_JSONL = REPO / "diag_v2_capture.jsonl"
VIRAT_INDEX = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_000102.npz"

# Hand-parsed from diag_v2_capture.jsonl's "real" claims matching the
# METADATA_RE taxonomy regex (action score|persistence|selection reason).
# Each tuple: (frame_idx, field, stated_value, source_text)
FIXTURES = [
    (24993, "action_score", 1.00,
     "Specifically, Frame 24993 shows a parking lot with an action score of "
     "1.00 and persistence of 1.00, indicating high confidence in the "
     "presence of a car."),
    (24993, "persistence", 1.00,
     "Specifically, Frame 24993 shows a parking lot with an action score of "
     "1.00 and persistence of 1.00, indicating high confidence in the "
     "presence of a car."),
    (25069, "action_score", 0.97,
     "Specifically: Frame 25069 shows a car parked in a parking lot with an "
     "Action Score of 0.97 and Persistence of 0.96."),
    (25069, "persistence", 0.96,
     "Specifically: Frame 25069 shows a car parked in a parking lot with an "
     "Action Score of 0.97 and Persistence of 0.96."),
    (23760, "persistence", 0.00,
     "Additionally, frame 23760 shows a low persistence score of 0.00, "
     "indicating that the object (in this case, the car) was not detected "
     "in the previous frame, suggesting it may have moved or disappeared "
     "from view."),
    (2148, "timestamp_sec", 71.7,
     "For example: Frame 2148 (Timestamp: 71.7s) shows a group of people "
     "walking down a street with an Action Score of 1.00, indicating high "
     "confidence in detecting people."),
]


def run_fixtures() -> None:
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: {CAPTURE_JSONL} missing", file=sys.stderr)
        sys.exit(1)

    # Confirm the fixture source_texts actually appear in the capture, so this
    # stays honest to "known" claims rather than invented ones.
    capture_text = CAPTURE_JSONL.read_text(encoding="utf-8")
    for frame_idx, fld, stated, source_text in FIXTURES:
        if source_text not in capture_text:
            print(f"WARNING: source_text not found verbatim in capture: {source_text!r}")

    index = iris_ingest.load_index(VIRAT_INDEX)
    frame_map = {f.frame_idx: f for f in index.frames}

    print(f"{'source_text':<90} | {'field':<14} | {'stated':>8} | {'stored':>8} | {'verdict'}")
    print("-" * 140)

    for frame_idx, fld, stated, source_text in FIXTURES:
        claim = MetadataClaim(
            frame_idx=frame_idx,
            field=fld,
            stated_value=stated,
            source_text=source_text,
        )
        verdict = verify_metadata_claim(claim, index)

        fr = frame_map.get(frame_idx)
        attr = {"action_score": "action_score", "persistence": "persistence_value",
                "timestamp_sec": "timestamp"}[fld]
        stored = getattr(fr, attr) if fr is not None else None

        disp_source = (source_text[:87] + "...") if len(source_text) > 90 else source_text
        stored_disp = f"{stored:.4f}" if stored is not None else "N/A"
        print(f"{disp_source:<90} | {fld:<14} | {stated:>8.2f} | {stored_disp:>8} | {verdict}")

    print()


def run_roundtrip() -> None:
    print("=== AnswerClaims JSON round-trip ===")
    original = AnswerClaims(
        query="Is anyone loading a vehicle?",
        claims=[
            VisualClaim(
                frame_idx=23760,
                assertion="a car is parked near the entrance",
                is_core=True,
            ),
            MetadataClaim(
                frame_idx=23760,
                field="persistence",
                stated_value=0.0,
                source_text="frame 23760 shows a low persistence score of 0.00",
            ),
            AbsenceClaim(
                event="someone loading a vehicle",
                is_core=False,
            ),
            GlobalClaim(
                text="Overall the parking lot appears static across the clip.",
            ),
        ],
    )

    serialized = original.to_json()
    parsed = AnswerClaims.from_json(serialized)

    print(serialized)
    print()
    print(f"round-trip equality: {original == parsed}")


def main() -> None:
    run_fixtures()
    run_roundtrip()


if __name__ == "__main__":
    main()
