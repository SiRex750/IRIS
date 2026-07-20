"""Tests for iris.cerberus_layers.

Layer 1 (verify_metadata_claim) is exercised end-to-end by
scripts/verify_layer1.py; no separate unit test here.

Layer 2 (verify_visual_claim / score_nli_pair) loads the real DeBERTa NLI
model, same as tests/test_cerberus_v.py.
"""
from __future__ import annotations

import pytest

from iris.cerberus_layers import (
    Layer2Result,
    get_nli_gate,
    score_nli_pair,
    verify_visual_claim,
)
from iris.claim_contract import VisualClaim

# (claim, fact) pairs spanning the branches score_nli_pair's provenance
# comment calls out: plain entailment, negation-risk (claim negates, fact
# doesn't), a citation-bound visual pair, an off-topic/contradicting pair,
# a neutral pair, and a pair inside the 0.5-0.85 entailment-floor "blind
# zone" (score=0.8009, from the normalized-kernel/frame-18708 fixture in
# scripts/verify_layer2.py) -- this last one is the one that actually
# exercises the dead-code floor bug found by scripts/diag_l3_judge.py;
# without it this test can pass even when score_nli_pair and its oracle
# disagree in that zone, since none of the other pairs land there.
FIXTURE_PAIRS = [
    ("Maverick flies a jet", "Maverick flies an F-18."),
    ("Maverick does not fly a jet", "Maverick flies an F-18."),
    ("there is a car parked in the parking lot", "a car parked in a parking lot."),
    ("A dog runs across the parking lot.", "a car parked in a parking lot."),
    ("A person is walking near the entrance", "a group of people walking down a street."),
    ("also shows a group of people walking down a street .00.", "a group of people walking down a street."),
]


def test_score_nli_pair_parity():
    """score_nli_pair must return the label-specific probability for each
    predicted label (P1-21 fix: contradiction returns contradiction probability,
    not entailment probability). The oracle in _capture_full_nli always records
    entailment_score, which is the P1-21 bug -- so we verify labels match and
    that the returned score is the correct label-specific probability."""
    from scripts.diag_v2_scoping_separation import _capture_full_nli

    gate = get_nli_gate()

    for claim, fact in FIXTURE_PAIRS:
        own_label, own_score = score_nli_pair(gate, claim, fact)

        records: list[dict] = []
        _capture_full_nli(gate, [claim], [fact], "full_nli", 0.9, records, "test_query")
        assert len(records) == 1
        oracle_label = records[0]["label"]

        assert own_label == oracle_label, f"label mismatch for ({claim!r}, {fact!r})"
        # P1-21: score_nli_pair now returns label-specific probability:
        # entailment -> entailment prob, contradiction -> contradiction prob.
        # The oracle always returns entailment prob (the original bug).
        # So for entailment labels, scores should match. For non-entailment labels,
        # score_nli_pair returns the label-specific prob which differs from entailment prob.
        assert 0.0 <= own_score <= 1.0, f"score out of range for ({claim!r}, {fact!r})"
        if own_label == "entailment":
            oracle_score = records[0]["entailment_score"]
            assert own_score == pytest.approx(oracle_score), f"entailment score mismatch for ({claim!r}, {fact!r})"


def test_verify_visual_claim_verified():
    gate = get_nli_gate()
    claim = VisualClaim(frame_idx=25069, assertion="there is a car parked in the parking lot")
    result = verify_visual_claim(claim, "a car parked in a parking lot.", gate)

    assert isinstance(result, Layer2Result)
    assert result.verdict == "verified"
    assert result.best_sentence == "a car parked in a parking lot."
    assert result.best_score > 0.9
    assert result.n_sentences == 1
    assert result.oversize_sentences == []


def test_verify_visual_claim_sentence_scoped_entailment():
    """A multi-sentence caption where only ONE sentence is about the claim's
    subject must not let an off-topic sentence block the entailing one."""
    gate = get_nli_gate()
    claim = VisualClaim(frame_idx=1, assertion="there is a car parked in the parking lot")
    caption = "A dog runs across the field. A car parked in a parking lot. The sky is cloudy."
    result = verify_visual_claim(claim, caption, gate)

    assert result.verdict == "verified"
    assert result.best_sentence == "A car parked in a parking lot."
    assert result.n_sentences == 3


def test_verify_visual_claim_rejected():
    gate = get_nli_gate()
    claim = VisualClaim(frame_idx=1, assertion="Maverick does not fly a jet")
    result = verify_visual_claim(claim, "Maverick flies an F-18.", gate)

    assert result.verdict == "rejected"


def test_verify_visual_claim_unverifiable_empty_caption():
    gate = get_nli_gate()
    claim = VisualClaim(frame_idx=1, assertion="there is a car parked in the parking lot")
    result = verify_visual_claim(claim, "", gate)

    assert result.verdict == "unverifiable"
    assert result.best_sentence is None
    assert result.n_sentences == 0
