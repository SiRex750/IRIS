"""ARIA claim contract — the typed claim schema Cerberus verifies against.

Schema only: this module defines the four claim types ARIA's answers decompose
into, and their (de)serialization. No prompt/generation changes and no verifier
logic live here (see iris.cerberus_layers for layer 1).

Claim types:
    VisualClaim    — plain visual-language assertion about a frame's content,
                      checked by Cerberus-V's NLI layer (layer 2/3, not yet built).
    MetadataClaim  — an arithmetic assertion about a stored pipeline metric
                      (action_score, persistence, timestamp_sec) for one frame,
                      checked deterministically against the index (layer 1).
    AbsenceClaim   — "event X does not appear in the evidence". Schema only this
                      task: verified later only as bounded ("verified-absent over
                      the checked evidence"), never as a universal negative.
                      event MUST be a FULL DECLARATIVE PROPOSITION ("A person is
                      entering the building."), not a bare noun phrase or gerund
                      fragment ("a person entering the building") -- layer 3
                      (iris.cerberus_layers.verify_absence_claim) scores it as an
                      NLI hypothesis, and standard NLI hypotheses are propositions
                      with a finite verb, not fragments. A deterministic guard in
                      verify_absence_claim wraps any fragment it receives
                      ("There is " + event + "."), so a non-compliant event still
                      gets checked -- but the contract itself asks for the
                      well-formed version.
    GlobalClaim    — a claim not anchored to any single frame (e.g. a summary
                      or aggregate statement). Schema only this task.

AnswerClaims bundles one query's full claim set. Per Decision 1, exactly one
claim among the Visual/Absence claims must be marked as the core claim (the
one the answer's badge is judged against); GlobalClaim and MetadataClaim are
never core.

ARIA will eventually be prompted to emit exactly this JSON shape:

    {
      "query": "Is anyone loading a vehicle?",
      "claims": [
        {"type": "visual", "frame_idx": 23760, "assertion":
            "a car is parked near the entrance", "is_core": true},
        {"type": "metadata", "frame_idx": 23760, "field": "persistence",
            "stated_value": 0.0, "source_text":
            "frame 23760 shows a low persistence score of 0.00"},
        {"type": "absence", "event": "Someone is loading a vehicle.",
            "is_core": false},
        {"type": "global", "text":
            "Overall the parking lot appears static across the clip."}
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Union

METADATA_FIELDS = ("action_score", "persistence", "timestamp_sec")


class ClaimContractError(Exception):
    """Base for all classifiable claim-contract failures. taxonomy_label is
    read by iris.query._generate_answer_claims_v2 to pick a corrective
    message -- this module is the single classification authority."""
    taxonomy_label = "other"


class MalformedJSONError(ClaimContractError, ValueError):
    taxonomy_label = "json_decode"


class SchemaShapeError(ClaimContractError, ValueError):
    taxonomy_label = "schema_shape"


class UnknownClaimTypeError(ClaimContractError, ValueError):
    taxonomy_label = "unknown_claim_type"


class BadMetadataFieldError(ClaimContractError, ValueError):
    taxonomy_label = "bad_metadata_field"


class ClaimFieldShapeError(ClaimContractError, ValueError):
    taxonomy_label = "claim_field_shape"


class CoreClaimInvariantError(ClaimContractError, ValueError):
    taxonomy_label = "is_core_invariant"


@dataclass
class VisualClaim:
    frame_idx: int
    assertion: str
    is_core: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"type": "visual", **asdict(self)}


@dataclass
class MetadataClaim:
    frame_idx: int
    field: str
    stated_value: float
    source_text: str

    def __post_init__(self) -> None:
        if self.field not in METADATA_FIELDS:
            raise BadMetadataFieldError(
                f"MetadataClaim.field must be one of {METADATA_FIELDS}, got {self.field!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "metadata", **asdict(self)}


@dataclass
class AbsenceClaim:
    """event must be a full declarative proposition ("A person is entering
    the building."), not a bare noun phrase or gerund fragment ("a person
    entering the building") -- see the module docstring."""
    event: str
    is_core: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"type": "absence", **asdict(self)}


@dataclass
class GlobalClaim:
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "global", **asdict(self)}


Claim = Union[VisualClaim, MetadataClaim, AbsenceClaim, GlobalClaim]

_TYPE_TO_CLS = {
    "visual": VisualClaim,
    "metadata": MetadataClaim,
    "absence": AbsenceClaim,
    "global": GlobalClaim,
}


def claim_from_dict(d: dict[str, Any]) -> Claim:
    cls = _TYPE_TO_CLS.get(d.get("type"))
    if cls is None:
        raise UnknownClaimTypeError(f"Unknown claim type: {d.get('type')!r}")
    payload = {k: v for k, v in d.items() if k != "type"}
    try:
        return cls(**payload)
    except TypeError as e:
        raise ClaimFieldShapeError(
            f"{cls.__name__} field shape mismatch: {e}"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Flat wire schema (task 3) -- schema-constrained decoding (Ollama format=<schema>,
# GBNF/XGrammar) SIGSEGVs or silently falls back to unconstrained text on
# anyOf/oneOf discriminated unions, and Qwen/Hermes-style PEG template paths have
# a trailing-quote bug on enum/const. AnswerClaims is exactly a 4-way union, so
# the WIRE shape (what the grammar constrains) must be a single FLAT object type
# carrying every variant's fields, discriminated by a plain string enum
# ("claim_type") -- not the nested {"type": "visual", ...} / {"type":
# "metadata", ...} shape from_dict/from_json use. from_wire below is the
# adapter from this flat wire shape back to the same typed dataclasses and the
# same task-1 exception taxonomy; from_json/claim_from_dict (the anyOf-free
# but non-constrained json-mode path) are UNCHANGED -- byte-identical.
#
# property -> which variant(s) use it (its "real" fields):
#   claim_type    all (discriminator; required on every claim)
#   frame_idx     visual, metadata
#   assertion     visual
#   is_core       visual, absence (metadata/global: always False, its own
#                 natural sentinel -- never nullable, never tracked as noise)
#   field         metadata
#   stated_value  metadata
#   source_text   metadata
#   event         absence
#   text          global
#
# SENTINEL CONVENTION (task 4): every property is now REQUIRED on every claim
# object -- the grammar CAN enforce presence (no per-variant conditionality
# left to need anyOf for), but it still cannot enforce that a property is only
# filled when relevant to that claim's variant. A model fills every property
# NOT used by its claim_type with a fixed sentinel so the grammar always sees
# a complete object:
#   frame_idx     -1              (never a real frame index)
#   stated_value  -1              (chosen over 0.0 -- 0.0 is a legitimate
#                                  real persistence/action_score reading)
#   field         "none"          (added to the wire-only enum below;
#                                  METADATA_FIELDS itself is untouched --
#                                  MetadataClaim.field still only accepts the
#                                  3 real values)
#   assertion, source_text, event, text   ""    (empty string)
# Sentinels are SINGLE-TYPED per field (integer/-1, string/"", string/"none")
# -- never a nullable type union like ["integer","null"], which is the SAME
# documented SIGSEGV class as anyOf/oneOf, just spelled differently.
_WIRE_FIELD_ENUM = (*METADATA_FIELDS, "none")

SENTINEL_FRAME_IDX = -1
SENTINEL_STATED_VALUE = -1
SENTINEL_FIELD = "none"
SENTINEL_STR = ""

# field -> its sentinel value, used both to build the flat wire object and to
# map sentinel-valued properties back to None before variant validation.
# is_core deliberately excluded: False is a legitimate real value (a
# non-core visual/absence claim, or the fixed value for metadata/global), so
# it is never sentinel-mapped or counted as noise.
_SENTINEL_VALUES: dict[str, Any] = {
    "frame_idx": SENTINEL_FRAME_IDX,
    "assertion": SENTINEL_STR,
    "field": SENTINEL_FIELD,
    "stated_value": SENTINEL_STATED_VALUE,
    "source_text": SENTINEL_STR,
    "event": SENTINEL_STR,
    "text": SENTINEL_STR,
}

ANSWER_CLAIMS_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query", "claims"],
    "properties": {
        "query": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim_type", "frame_idx", "assertion", "is_core",
                    "field", "stated_value", "source_text", "event", "text",
                ],
                "properties": {
                    "claim_type": {"type": "string", "enum": ["visual", "metadata", "absence", "global"]},
                    "frame_idx": {"type": "integer"},
                    "assertion": {"type": "string"},
                    "is_core": {"type": "boolean"},
                    "field": {"type": "string", "enum": list(_WIRE_FIELD_ENUM)},
                    "stated_value": {"type": "number"},
                    "source_text": {"type": "string"},
                    "event": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
        },
    },
}

# variant -> its REQUIRED (real, non-sentinel) wire properties beyond claim_type
_WIRE_REQUIRED_FIELDS = {
    "visual": ("frame_idx", "assertion", "is_core"),
    "metadata": ("frame_idx", "field", "stated_value", "source_text"),
    "absence": ("event", "is_core"),
    "global": ("text",),
}


def claim_from_wire_dict(d: dict[str, Any]) -> tuple[Claim, list[str]]:
    """Adapter: one flat wire-schema claim object -> a typed Claim dataclass.
    Dispatches on claim_type, maps sentinel-valued properties to None, then
    validates that variant's required properties hold REAL (non-sentinel,
    non-missing) values -- a required property left at its sentinel is
    ClaimFieldShapeError, named explicitly, same as a property missing
    outright (the grammar requires presence, not realness, of every
    property, so both are genuine variant-completeness failures).

    Returns (claim, irrelevant_fields_filled) -- the second element lists any
    OTHER variant's fields that were filled with a real (non-sentinel) value
    on this claim (noise the flat schema can't forbid without anyOf; counted
    for the harness, never itself an error)."""
    claim_type = d.get("claim_type")
    required = _WIRE_REQUIRED_FIELDS.get(claim_type)
    if required is None:
        raise UnknownClaimTypeError(f"Unknown claim_type: {claim_type!r}")

    mapped = dict(d)
    for key, sentinel in _SENTINEL_VALUES.items():
        if mapped.get(key) == sentinel:
            mapped[key] = None

    bad = []
    for k in required:
        if k == "is_core":
            if "is_core" not in d:
                bad.append(k)
            continue
        if mapped.get(k) is None:
            bad.append(k)
    if bad:
        quoted = ", ".join(f"{n!r}" for n in bad)
        raise ClaimFieldShapeError(
            f"{claim_type} claim requires a real {quoted}; got sentinel/missing."
        )

    irrelevant_fields_filled = [
        k for k in _SENTINEL_VALUES if k not in required and mapped.get(k) is not None
    ]

    if claim_type == "visual":
        claim = VisualClaim(frame_idx=mapped["frame_idx"], assertion=mapped["assertion"], is_core=d["is_core"])
    elif claim_type == "metadata":
        claim = MetadataClaim(frame_idx=mapped["frame_idx"], field=mapped["field"],
                               stated_value=mapped["stated_value"], source_text=mapped["source_text"])
    elif claim_type == "absence":
        claim = AbsenceClaim(event=mapped["event"], is_core=d["is_core"])
    else:
        claim = GlobalClaim(text=mapped["text"])
    return claim, irrelevant_fields_filled


@dataclass
class AnswerClaims:
    """One query's full decomposed claim set.

    Invariant (Decision 1): exactly one claim among the Visual/Absence claims
    must have is_core=True — that claim is what the answer's badge is judged
    against. MetadataClaim and GlobalClaim are never core and carry no
    is_core field.

    field_noise (task 4): flat list of irrelevant-variant wire field names
    filled with real values across all claims -- populated only by from_wire
    (the sentinel-schema path); from_json/from_dict leave it [] (the nested
    shape has no such noise to report).
    """
    query: str
    claims: list[Claim] = field(default_factory=list)
    field_noise: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        core_count = sum(
            1
            for c in self.claims
            if isinstance(c, (VisualClaim, AbsenceClaim)) and c.is_core
        )
        if core_count != 1:
            raise CoreClaimInvariantError(
                f"AnswerClaims requires exactly one is_core=True claim among "
                f"Visual/Absence claims, found {core_count}"
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "query": self.query,
                "claims": [c.to_dict() for c in self.claims],
            }
        )

    @staticmethod
    def from_json(s: str) -> "AnswerClaims":
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise MalformedJSONError(f"Not valid JSON: {e}") from e
        try:
            raw_claims = d["claims"]
            query = d["query"]
        except KeyError as e:
            raise SchemaShapeError(f"Missing required key: {e}") from e
        claims = [claim_from_dict(c) for c in raw_claims]
        return AnswerClaims(query=query, claims=claims)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> "AnswerClaims":
        """Adapter from the flat, grammar-constrained wire shape
        (ANSWER_CLAIMS_WIRE_SCHEMA) to AnswerClaims. Takes an already-parsed
        dict (schema-constrained decoding guarantees syntactically valid
        JSON, so there is no MalformedJSONError path here -- a caller with a
        raw string should json.loads() it first). Missing top-level query/
        claims -> SchemaShapeError; per-claim dispatch and validation is
        claim_from_wire_dict's job, so the same typed exception taxonomy
        (UnknownClaimTypeError, ClaimFieldShapeError, CoreClaimInvariantError
        via normal construction) applies unchanged."""
        try:
            raw_claims = d["claims"]
            query = d["query"]
        except KeyError as e:
            raise SchemaShapeError(f"Missing required key: {e}") from e
        claims: list[Claim] = []
        field_noise: list[str] = []
        for c in raw_claims:
            claim, noise = claim_from_wire_dict(c)
            claims.append(claim)
            field_noise.extend(noise)
        return cls(query=query, claims=claims, field_noise=field_noise)
