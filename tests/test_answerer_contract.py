"""Tests for the v2 answerer's compliance taxonomy: typed claim-contract
exceptions (iris.claim_contract), classification + corrective retry
(iris.query._generate_answer_claims_v2), and the compliance_failed
short-circuit in iris.query._query_v2.

Hermetic: no DeBERTa NLI model load. iris.aria's backend is swapped for a
ScriptedV2Backend and iris.cerberus_layers.get_nli_gate / verify_answer are
monkeypatched for the _query_v2 orchestration tests.
"""
from __future__ import annotations

import json

import pytest

import iris.aria as aria
import iris.query as iris_query
from iris.aria import LLMBackend
from iris.cerberus_layers import AnswerVerification, ClaimVerdict
from iris.claim_contract import (
    AbsenceClaim,
    AnswerClaims,
    ClaimFieldShapeError,
    CoreClaimInvariantError,
    GlobalClaim,
    MetadataClaim,
    UnknownClaimTypeError,
    VisualClaim,
    claim_from_wire_dict,
)
from iris.query import _extract_json_object, _generate_answer_claims_v2, _query_v2


# ── scripted backend ─────────────────────────────────────────────────────


class ScriptedV2Backend(LLMBackend):
    """Returns a scripted list of raw strings in call order; records every
    call's args so correctives can be asserted."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
        self.calls.append({
            "prompt": prompt, "context": context, "model": model,
            "system_prompt": system_prompt, "response_format": response_format,
            "max_tokens": max_tokens, "schema_format": schema_format,
        })
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def _restore_backend():
    prior = aria.get_backend()
    yield
    aria.set_backend(prior)


# ── JSON fixtures ────────────────────────────────────────────────────────


def _claims_json(claims: list[dict], query: str = "Is anyone present?") -> str:
    return json.dumps({"query": query, "claims": claims})


VALID_JSON = _claims_json(
    [{"type": "visual", "frame_idx": 1, "assertion": "a person is present", "is_core": True}]
)
MALFORMED_JSON = "this is { not valid json"
NO_CORE_JSON = _claims_json(
    [{"type": "visual", "frame_idx": 1, "assertion": "a person is present", "is_core": False}]
)
TWO_CORE_JSON = _claims_json(
    [
        {"type": "visual", "frame_idx": 1, "assertion": "a person is present", "is_core": True},
        {"type": "visual", "frame_idx": 2, "assertion": "a car is parked", "is_core": True},
    ]
)
BAD_METADATA_JSON = _claims_json(
    [
        {"type": "visual", "frame_idx": 1, "assertion": "a person is present", "is_core": True},
        {"type": "metadata", "frame_idx": 1, "field": "bogus_field", "stated_value": 1.0,
         "source_text": "x"},
    ]
)
UNKNOWN_TYPE_JSON = _claims_json(
    [
        {"type": "bogus", "frame_idx": 1},
        {"type": "visual", "frame_idx": 1, "assertion": "a person is present", "is_core": True},
    ]
)
MISSING_CLAIMS_KEY_JSON = json.dumps({"query": "Is anyone present?"})
CLAIM_MISSING_FIELD_JSON = _claims_json(
    [{"type": "visual", "frame_idx": 1, "is_core": True}]
)


# ── _extract_json_object ─────────────────────────────────────────────────


def test_extract_bare_object():
    assert _extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced():
    raw = '```json\n{"a": 1}\n```'
    assert _extract_json_object(raw) == '{"a": 1}'


def test_extract_bare_fenced():
    raw = '```\n{"a": 1}\n```'
    assert _extract_json_object(raw) == '{"a": 1}'


def test_extract_prose_then_object():
    raw = 'Here is the answer: {"a": 1}'
    assert _extract_json_object(raw) == '{"a": 1}'


def test_extract_nested_balanced_braces():
    raw = '{"a": {"b": 1}}'
    assert _extract_json_object(raw) == '{"a": {"b": 1}}'


def test_extract_trailing_text_after_object():
    raw = '{"a": 1} some trailing text'
    assert _extract_json_object(raw) == '{"a": 1}'


def test_extract_no_brace_fallback():
    raw = "no braces here"
    assert _extract_json_object(raw) == "no braces here"


# ── _generate_answer_claims_v2 ───────────────────────────────────────────


def _set_scripted(responses: list[str]) -> ScriptedV2Backend:
    backend = ScriptedV2Backend(responses)
    aria.set_backend(backend)
    return backend


def test_valid_attempt_1():
    _set_scripted([VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert claims is not None
    assert raw == VALID_JSON
    assert failed is False
    assert n == 1
    assert labels == []


def test_malformed_json_then_valid():
    _set_scripted([MALFORMED_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert claims is not None
    assert failed is False
    assert n == 2
    assert labels == ["json_decode"]


def test_is_core_zero_then_valid():
    backend = _set_scripted([NO_CORE_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["is_core_invariant"]
    retry_prompt = backend.calls[1]["prompt"]
    assert "is_core" in retry_prompt
    assert "not valid JSON" not in retry_prompt


def test_is_core_two_then_valid():
    _set_scripted([TWO_CORE_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["is_core_invariant"]


def test_bad_metadata_field_then_valid():
    _set_scripted([BAD_METADATA_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["bad_metadata_field"]


def test_unknown_claim_type_then_valid():
    _set_scripted([UNKNOWN_TYPE_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["unknown_claim_type"]


def test_missing_claims_key_then_valid():
    _set_scripted([MISSING_CLAIMS_KEY_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["schema_shape"]


def test_claim_missing_required_field_then_valid():
    _set_scripted([CLAIM_MISSING_FIELD_JSON, VALID_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert failed is False
    assert n == 2
    assert labels == ["claim_field_shape"]


def test_invalid_twice():
    _set_scripted([MALFORMED_JSON, NO_CORE_JSON])
    claims, raw, failed, n, labels = _generate_answer_claims_v2("q", "ctx")
    assert claims is None
    assert raw == NO_CORE_JSON
    assert failed is True
    assert n == 2
    assert labels == ["json_decode", "is_core_invariant"]


# ── _extract_caption_text ────────────────────────────────────────────────


from iris.cerberus_layers import _extract_caption_text


def test_extract_caption_text_dict_with_semantic_caption():
    assert _extract_caption_text({"semantic_caption": "a car is parked"}) == "a car is parked"


def test_extract_caption_text_raw_string():
    assert _extract_caption_text("a car is parked") == "a car is parked"


def test_extract_caption_text_none():
    assert _extract_caption_text(None) is None


def test_extract_caption_text_empty_string():
    assert _extract_caption_text("") is None


def test_extract_caption_text_caption_failed_sentinel():
    assert _extract_caption_text("[CAPTION_FAILED]") is None


def test_extract_caption_text_dict_without_semantic_caption():
    assert _extract_caption_text({"other_key": "x"}) is None


# ── _query_v2 orchestration (hermetic) ───────────────────────────────────


class _FakeIndex:
    frames_processed = 10
    peak_count = 3
    skipped_frames_ratio = 0.5
    storage_reduction_factor = 2.0


_RETRIEVED_FRAMES = [
    {
        "frame_idx": 1, "timestamp": 0.0, "luma_diff_energy": 0.0,
        "action_score": 0.1, "persistence_value": 0.0, "is_peak": False,
        "clip_embedding": None, "luma_entropy": 0.0, "caption": None,
        "pagerank_score": 0.0, "last_retrieval_score": 0.0,
        "retrieval_contributions": {},
    }
]


def _patch_query_plumbing(monkeypatch):
    monkeypatch.setattr(iris_query, "_embed_query", lambda question, config: None)
    monkeypatch.setattr(iris_query, "_build_retrieved", lambda index, emb, config: list(_RETRIEVED_FRAMES))
    monkeypatch.setattr(iris_query, "_ensure_captions", lambda index, frames: 0)


def test_query_v2_compliance_failed_short_circuits(monkeypatch):
    from iris import iris_config

    _patch_query_plumbing(monkeypatch)
    _set_scripted([MALFORMED_JSON, MALFORMED_JSON])

    def _gate_should_not_be_called():
        raise AssertionError("get_nli_gate must not be called when compliance_failed")

    monkeypatch.setattr("iris.cerberus_layers.get_nli_gate", lambda: _gate_should_not_be_called())

    result = _query_v2("q", _FakeIndex(), iris_config.IRISConfig(cerberus_mode="v2"))

    assert result["compliance_failed"] is True
    assert result["badge"] == "unverified"
    assert result["claim_verdicts"] == []
    assert result["core_claim_verdict"] is None
    assert result["compliance_failure_labels"] == ["json_decode", "json_decode"]


def test_query_v2_compliant_invokes_verify_answer(monkeypatch):
    from iris import iris_config

    _patch_query_plumbing(monkeypatch)
    _set_scripted([VALID_JSON])

    stub_gate = object()
    monkeypatch.setattr("iris.cerberus_layers.get_nli_gate", lambda: stub_gate)

    core_verdict = ClaimVerdict(claim=VisualClaim(frame_idx=1, assertion="a person is present", is_core=True),
                                 label="verified")
    stub_verification = AnswerVerification(
        claim_verdicts=[core_verdict], badge="verified", core_claim_verdict=core_verdict,
    )
    calls = []

    def _stub_verify_answer(answer_claims, evidence):
        calls.append((answer_claims, evidence))
        return stub_verification

    monkeypatch.setattr("iris.cerberus_layers.verify_answer", _stub_verify_answer)

    result = _query_v2("q", _FakeIndex(), iris_config.IRISConfig(cerberus_mode="v2"))

    assert len(calls) == 1
    assert result["compliance_failed"] is False
    assert result["badge"] == "verified"
    assert result["claim_verdicts"] == [core_verdict]
    assert result["core_claim_verdict"] == core_verdict
    assert result["compliance_failure_labels"] == []


# ── AnswerClaims.from_wire / claim_from_wire_dict (task 3/4, sentinel wire schema) ──

# every wire claim field is REQUIRED (task 4) -- tests build a fully
# sentinel-filled object and override only the fields under test, matching
# what the grammar actually produces (never a partial dict).
_SENTINEL_CLAIM = {
    "frame_idx": -1, "assertion": "", "is_core": False,
    "field": "none", "stated_value": -1, "source_text": "",
    "event": "", "text": "",
}


def _wire_claim(claim_type: str, **overrides) -> dict:
    d = {"claim_type": claim_type, **_SENTINEL_CLAIM}
    d.update(overrides)
    return d


def test_from_wire_visual_happy_path():
    claim, noise = claim_from_wire_dict(_wire_claim("visual", frame_idx=1, assertion="a car is parked", is_core=True))
    assert isinstance(claim, VisualClaim)
    assert claim.frame_idx == 1
    assert claim.assertion == "a car is parked"
    assert claim.is_core is True
    assert noise == []


def test_from_wire_metadata_happy_path():
    claim, noise = claim_from_wire_dict(_wire_claim("metadata", frame_idx=1, field="action_score",
                                                      stated_value=0.5, source_text="x"))
    assert isinstance(claim, MetadataClaim)
    assert claim.frame_idx == 1
    assert claim.field == "action_score"
    assert claim.stated_value == 0.5
    assert claim.source_text == "x"
    assert noise == []


def test_from_wire_absence_happy_path():
    claim, noise = claim_from_wire_dict(_wire_claim("absence", event="a person running", is_core=False))
    assert isinstance(claim, AbsenceClaim)
    assert claim.event == "a person running"
    assert claim.is_core is False
    assert noise == []


def test_from_wire_global_happy_path():
    claim, noise = claim_from_wire_dict(_wire_claim("global", text="Overall static."))
    assert isinstance(claim, GlobalClaim)
    assert claim.text == "Overall static."
    assert noise == []


def test_from_wire_visual_sentinel_valued_required_field_raises():
    # assertion left at its sentinel ("") -- named explicitly in the message.
    with pytest.raises(ClaimFieldShapeError, match="assertion"):
        claim_from_wire_dict(_wire_claim("visual", frame_idx=1, is_core=True))


def test_from_wire_metadata_sentinel_valued_required_fields_raises():
    # stated_value/source_text left at sentinel -- both named.
    with pytest.raises(ClaimFieldShapeError, match="stated_value"):
        claim_from_wire_dict(_wire_claim("metadata", frame_idx=1, field="action_score"))


def test_from_wire_metadata_stated_value_sentinel_alone_raises():
    # -1 is a syntactically valid "number" per the schema but IS the
    # stated_value sentinel -- must not be accepted as a real value.
    with pytest.raises(ClaimFieldShapeError, match="stated_value"):
        claim_from_wire_dict(_wire_claim("metadata", frame_idx=1, field="action_score", source_text="x"))


def test_from_wire_absence_event_sentinel_raises():
    with pytest.raises(ClaimFieldShapeError, match="event"):
        claim_from_wire_dict(_wire_claim("absence", is_core=True))


def test_from_wire_absence_missing_is_core_key_raises():
    # is_core literally absent (not just False) -- the one case sentinel-
    # mapping doesn't cover, since False is itself a legitimate real value.
    d = _wire_claim("absence", event="a person running")
    del d["is_core"]
    with pytest.raises(ClaimFieldShapeError, match="is_core"):
        claim_from_wire_dict(d)


def test_from_wire_global_text_sentinel_raises():
    with pytest.raises(ClaimFieldShapeError, match="text"):
        claim_from_wire_dict(_wire_claim("global"))


def test_from_wire_unknown_claim_type():
    with pytest.raises(UnknownClaimTypeError):
        claim_from_wire_dict(_wire_claim("bogus", text="x"))


def test_from_wire_irrelevant_real_values_ignored_but_counted():
    # a visual claim carrying REAL (non-sentinel) values for other variants'
    # fields too -- flat schema can't forbid this; claim_from_wire_dict must
    # ignore the noise for construction but report it for the harness.
    claim, noise = claim_from_wire_dict(_wire_claim(
        "visual", frame_idx=1, assertion="a car is parked", is_core=True,
        field="action_score", stated_value=0.9, source_text="irrelevant",
        event="irrelevant", text="irrelevant",
    ))
    assert isinstance(claim, VisualClaim)
    assert claim.frame_idx == 1
    assert claim.assertion == "a car is parked"
    assert claim.is_core is True
    assert set(noise) == {"field", "stated_value", "source_text", "event", "text"}


def test_from_wire_sentinel_valued_irrelevant_fields_not_counted_as_noise():
    # irrelevant fields left at their sentinel (the expected case) must NOT
    # be counted as noise -- only REAL values on irrelevant fields are noise.
    claim, noise = claim_from_wire_dict(_wire_claim("visual", frame_idx=1, assertion="a car is parked", is_core=True))
    assert noise == []


def test_answerclaims_from_wire_happy_path():
    d = {
        "query": "Is anyone loading a vehicle?",
        "claims": [
            _wire_claim("visual", frame_idx=1, assertion="a car is parked", is_core=True),
            _wire_claim("metadata", frame_idx=1, field="persistence", stated_value=0.0, source_text="x"),
            _wire_claim("absence", event="someone loading a vehicle", is_core=False),
            _wire_claim("global", text="Overall static."),
        ],
    }
    ac = AnswerClaims.from_wire(d)
    assert ac.query == "Is anyone loading a vehicle?"
    assert len(ac.claims) == 4
    assert ac.field_noise == []


def test_answerclaims_from_wire_field_noise_aggregated_across_claims():
    d = {
        "query": "q",
        "claims": [
            _wire_claim("visual", frame_idx=1, assertion="x", is_core=True, event="noise1"),
            _wire_claim("global", text="y", frame_idx=99),
        ],
    }
    ac = AnswerClaims.from_wire(d)
    assert set(ac.field_noise) == {"event", "frame_idx"}


def test_answerclaims_from_wire_core_count_zero_raises():
    d = {
        "query": "q",
        "claims": [_wire_claim("visual", frame_idx=1, assertion="x", is_core=False)],
    }
    with pytest.raises(CoreClaimInvariantError):
        AnswerClaims.from_wire(d)


def test_answerclaims_from_wire_core_count_two_raises():
    d = {
        "query": "q",
        "claims": [
            _wire_claim("visual", frame_idx=1, assertion="x", is_core=True),
            _wire_claim("visual", frame_idx=2, assertion="y", is_core=True),
        ],
    }
    with pytest.raises(CoreClaimInvariantError):
        AnswerClaims.from_wire(d)
