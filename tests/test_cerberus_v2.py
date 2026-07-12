"""Tests for the Cerberus v2 switchover: config flag, layer 3 (bounded
absence), the verify_answer router, and the answer badge.

Badge computation (_compute_badge) is pure logic over ClaimVerdict labels --
tested without loading the NLI model. Router dispatch and layer 3 need the
real DeBERTa NLI model, same as tests/test_cerberus_layers.py.
"""
from __future__ import annotations

import pytest

from iris.cerberus_layers import (
    ClaimVerdict,
    Evidence,
    Layer3Result,
    _compute_badge,
    _decompose_disjunction,
    get_nli_gate,
    verify_absence_claim,
    verify_answer,
)
from iris.claim_contract import (
    AbsenceClaim,
    AnswerClaims,
    GlobalClaim,
    MetadataClaim,
    VisualClaim,
)
from iris.iris_config import IRISConfig
from iris.types import FrameRecord, IRISIndex


# ── config ───────────────────────────────────────────────────────────────

def test_cerberus_mode_default_legacy():
    cfg = IRISConfig()
    assert cfg.cerberus_mode == "legacy"
    cfg.validate()  # must not raise


def test_cerberus_mode_v2_valid():
    cfg = IRISConfig(cerberus_mode="v2")
    cfg.validate()  # must not raise


def test_cerberus_mode_invalid_rejected():
    cfg = IRISConfig(cerberus_mode="bogus")
    with pytest.raises(AssertionError):
        cfg.validate()


# ── badge (pure logic, no NLI) ──────────────────────────────────────────

def _v(label: str) -> ClaimVerdict:
    return ClaimVerdict(claim=VisualClaim(frame_idx=1, assertion="x"), label=label)


def test_badge_verified_when_core_passes_and_nothing_else_present():
    core = _v("verified")
    assert _compute_badge(core, [core]) == "verified"


def test_badge_unverified_when_core_unverifiable():
    core = _v("unverifiable")
    assert _compute_badge(core, [core]) == "unverified"


def test_badge_unverified_when_core_rejected():
    core = _v("rejected")
    assert _compute_badge(core, [core]) == "unverified"


def test_badge_flagged_when_any_rejection_present():
    core = _v("verified")
    other = _v("rejected")
    assert _compute_badge(core, [core, other]) == "flagged"


def test_badge_partial_when_noncore_unverifiable_remains():
    core = _v("verified_absent")
    other = _v("unverifiable")
    assert _compute_badge(core, [core, other]) == "partial"


def test_badge_metadata_checked_fail_counts_as_rejection():
    core = _v("verified")
    meta_fail = ClaimVerdict(claim=MetadataClaim(frame_idx=1, field="action_score",
                                                   stated_value=0.5, source_text="x"),
                              label="checked_fail")
    assert _compute_badge(core, [core, meta_fail]) == "flagged"


def test_badge_metadata_checked_pass_does_not_block_verified():
    core = _v("verified")
    meta_pass = ClaimVerdict(claim=MetadataClaim(frame_idx=1, field="action_score",
                                                   stated_value=0.5, source_text="x"),
                              label="checked_pass")
    assert _compute_badge(core, [core, meta_pass]) == "verified"


# ── layer 3 / router (real NLI) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def gate():
    return get_nli_gate()


def test_verify_absence_claim_verified_absent(gate):
    claim = AbsenceClaim(event="There is a fire or smoke.", is_core=True)
    captions = [(1, "a car parked in a parking lot."), (2, "a red brick building.")]
    result = verify_absence_claim(claim, captions, gate)

    assert isinstance(result, Layer3Result)
    assert result.verdict == "verified_absent"
    assert result.n_frames_checked == 2
    assert "verified-absent over the 2 frames checked" in result.phrasing


def test_verify_absence_claim_rejected_when_event_found(gate):
    claim = AbsenceClaim(event="A car is parked in the parking lot.", is_core=True)
    captions = [(1, "a car parked in a parking lot."), (2, "a red brick building.")]
    result = verify_absence_claim(claim, captions, gate)

    assert result.verdict == "rejected"
    assert result.rejecting_frame_idx == 1
    assert result.n_frames_checked == 2


def test_propositionalize_event_wraps_bare_fragment(gate):
    from iris.cerberus_layers import _propositionalize_event

    nlp = gate._get_spacy()
    assert _propositionalize_event(nlp, "a person entering the building") == (
        "There is a person entering the building."
    )
    # already a full proposition -- passed through unchanged
    assert _propositionalize_event(nlp, "A person is entering the building.") == (
        "A person is entering the building."
    )


def test_decompose_disjunction_noun_coordination(gate):
    nlp = gate._get_spacy()
    assert _decompose_disjunction(nlp, "There is a fire or smoke.") == [
        "There is a fire.", "There is smoke.",
    ]


def test_decompose_disjunction_verb_coordination(gate):
    nlp = gate._get_spacy()
    assert _decompose_disjunction(nlp, "Someone is loading or unloading a vehicle.") == [
        "Someone is loading a vehicle.", "Someone is unloading a vehicle.",
    ]


def test_decompose_disjunction_no_op_on_atomic_proposition(gate):
    nlp = gate._get_spacy()
    assert _decompose_disjunction(nlp, "A person is entering the building.") == [
        "A person is entering the building.",
    ]


def test_verify_absence_claim_disjunction_rejects_on_either_disjunct(gate):
    """Tightening direction: 'fire or smoke' must reject if EITHER is found,
    even though a whole-hypothesis pass would call the mismatched disjunct
    (smoke, when only fire is present) merely 'not entailed'."""
    claim = AbsenceClaim(event="There is a fire or smoke.", is_core=True)
    captions = [(1, "There is a large fire burning in the parking lot.")]
    result = verify_absence_claim(claim, captions, gate)

    assert result.verdict == "rejected"
    assert result.rejecting_frame_idx == 1
    assert "fire" in result.phrasing.lower()


def test_verify_absence_claim_wraps_bare_fragment_event(gate):
    """A non-compliant (fragment) AbsenceClaim.event is still checked, via
    the deterministic wrap -- not rejected outright, not silently dropped."""
    claim = AbsenceClaim(event="a car parked in the parking lot", is_core=True)
    captions = [(1, "a car parked in a parking lot."), (2, "a red brick building.")]
    result = verify_absence_claim(claim, captions, gate)

    assert result.verdict == "rejected"  # wrapped to "There is a car parked in the parking lot." -- still found
    assert result.rejecting_frame_idx == 1


def _minimal_index() -> IRISIndex:
    frame = FrameRecord(
        frame_idx=1, timestamp=10.0, luma_diff_energy=0.1, luma_entropy=0.1,
        motion_magnitude=0.1, action_score=0.97, persistence_value=0.96, is_peak=True,
    )
    return IRISIndex(
        video_path="synthetic.mp4", frames=[frame], index_action_score=0.9,
        stats={}, frames_processed=1, peak_count=1, skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0, config_snapshot={},
    )


def test_router_metadata_claim_checked_pass(gate):
    index = _minimal_index()
    claim = MetadataClaim(frame_idx=1, field="action_score", stated_value=0.97, source_text="x")
    answer = AnswerClaims(query="q", claims=[
        VisualClaim(frame_idx=1, assertion="there is a car parked in the parking lot", is_core=True),
        claim,
    ])
    evidence = Evidence(
        index=index,
        retrieved_frames=[{"frame_idx": 1, "caption": {"semantic_caption": "a car parked in a parking lot."}}],
        nli=gate,
    )
    verification = verify_answer(answer, evidence)
    meta_verdict = next(v for v in verification.claim_verdicts if v.claim is claim)
    assert meta_verdict.label == "checked_pass"


def test_router_visual_claim_frame_not_in_evidence(gate):
    index = _minimal_index()
    claim = VisualClaim(frame_idx=999, assertion="there is a car parked in the parking lot", is_core=True)
    answer = AnswerClaims(query="q", claims=[claim])
    evidence = Evidence(index=index, retrieved_frames=[], nli=gate)
    verification = verify_answer(answer, evidence)

    assert verification.core_claim_verdict.label == "unverifiable"
    assert "cited frame not in evidence" in verification.core_claim_verdict.reason
    assert verification.badge == "unverified"


def test_router_global_claim_unverifiable(gate):
    index = _minimal_index()
    answer = AnswerClaims(query="q", claims=[
        VisualClaim(frame_idx=1, assertion="there is a car parked in the parking lot", is_core=True),
        GlobalClaim(text="Overall the lot is static."),
    ])
    evidence = Evidence(
        index=index,
        retrieved_frames=[{"frame_idx": 1, "caption": {"semantic_caption": "a car parked in a parking lot."}}],
        nli=gate,
    )
    verification = verify_answer(answer, evidence)
    global_verdict = next(v for v in verification.claim_verdicts if isinstance(v.claim, GlobalClaim))
    assert global_verdict.label == "unverifiable"
    # core (visual) claim entails -> verified; the unverifiable non-core GlobalClaim
    # demotes the badge to "partial" per Decision 1, not "verified".
    assert verification.badge == "partial"


def test_router_full_answer_badge_verified(gate):
    index = _minimal_index()
    answer = AnswerClaims(query="q", claims=[
        VisualClaim(frame_idx=1, assertion="there is a car parked in the parking lot", is_core=True),
        MetadataClaim(frame_idx=1, field="action_score", stated_value=0.97, source_text="x"),
        AbsenceClaim(event="a fire or smoke", is_core=False),
    ])
    evidence = Evidence(
        index=index,
        retrieved_frames=[{"frame_idx": 1, "caption": {"semantic_caption": "a car parked in a parking lot."}}],
        nli=gate,
    )
    verification = verify_answer(answer, evidence)
    assert verification.core_claim_verdict.label == "verified"
    assert verification.badge == "verified"
