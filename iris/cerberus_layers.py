"""Cerberus layers — verification of ARIA's typed claims (iris.claim_contract).

Layer 1: deterministic metadata verification.
Layer 2: frame-bound, sentence-scoped visual NLI verification.
Layer 3 (absence bounding) is not implemented here.

Ground truth for layer 1 comes from IRISIndex.frames (iris.types.FrameRecord),
the persisted index store — the same values as_context_text() surfaces to
ARIA's prompt via CachedFrame (l1_elysium.py). CachedFrame itself is NOT used
as ground truth here: it is populated from the _build_retrieved() dict, whose
docstring (query.py) states motion-geometry keys are deliberately dropped and
default to 0.0 on that path. Reading FrameRecord directly avoids silently
treating a dropped/defaulted field as a real stored value.

Layer 2 (verify_visual_claim) scores one VisualClaim against ONE caption
(the caller-supplied text for the exact frame the claim cites) — no
similarity scoping, no cross-frame voting, no lemma filter. The caption is
sentencized and the claim is scored against each sentence independently, so
a single off-topic sentence in a multi-sentence caption can't veto an
entailing one. Caption fetching/wiring to the lazy-caption path is not this
module's job; callers pass caption_text in.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from iris.claim_contract import (
    AbsenceClaim,
    AnswerClaims,
    Claim,
    GlobalClaim,
    MetadataClaim,
    VisualClaim,
)
from iris.types import IRISIndex

logger = logging.getLogger(__name__)

TIMESTAMP_TOLERANCE_SEC = 0.05
OVERSIZE_SENTENCE_TOKENS = 60

_FIELD_TO_ATTR = {
    "action_score": "action_score",
    "persistence": "persistence_value",
    "timestamp_sec": "timestamp",
}


def verify_metadata_claim(claim: MetadataClaim, index: IRISIndex) -> str:
    """Verify one MetadataClaim against the persisted index.

    Returns one of: "checked_pass", "checked_fail", "field_unavailable".
    """
    frame = next((f for f in index.frames if f.frame_idx == claim.frame_idx), None)
    if frame is None:
        return "field_unavailable"

    attr = _FIELD_TO_ATTR[claim.field]
    stored = getattr(frame, attr, None)
    if stored is None:
        return "field_unavailable"

    if claim.field == "timestamp_sec":
        passed = abs(stored - claim.stated_value) <= TIMESTAMP_TOLERANCE_SEC
    else:
        passed = round(stored, 2) == round(claim.stated_value, 2)

    return "checked_pass" if passed else "checked_fail"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — frame-bound, sentence-scoped visual NLI
# ─────────────────────────────────────────────────────────────────────────────


def get_nli_gate() -> Any:
    """Load (or reuse) a CerberusV instance with spaCy + the DeBERTa NLI
    model warmed up. Same lazy-loader pattern as
    scripts/diag_v3_visual_scoping.py:_get_nli_gate."""
    from iris.cerberus_v import CerberusV

    gate = CerberusV()
    gate._get_spacy()
    gate._get_nli_model()
    return gate


def score_nli_pair(gate: Any, claim: str, fact: str) -> tuple[str, float]:
    """Per-pair NLI scoring: argmax label, negation-aware 0.5/0.85 threshold,
    geo (GPE/LOC) check.

    PROVENANCE: copied verbatim from iris/cerberus_v.py's
    CerberusV._full_nli inner per-pair loop (lines computing `label`,
    `entailment_score`, the negation-risk threshold flip, and the geo
    check). That logic is not importable as a standalone function there —
    it lives inline inside the batched forward-pass loop — so this is a
    literal copy, identical to scripts/diag_v3_visual_scoping.py's
    _score_kernel_pair (which itself already lifted it verbatim once for
    the same reason). Any future change to that inner loop's semantics must
    be mirrored here; tests/test_cerberus_layers.py::test_score_nli_pair_parity
    scores fixture pairs through both this function and CerberusV.verify's
    full path and asserts identical (label, score).
    """
    import torch
    import torch.nn.functional as F

    nlp = gate._get_spacy()
    tokenizer, model = gate._get_nli_model()
    device = model.device

    inputs = tokenizer([fact], [claim], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        pred = int(torch.argmax(logits, dim=-1).cpu().item())
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]

    label = "neutral"
    id2label = getattr(model.config, "id2label", {}) or {}
    raw_label = str(id2label.get(pred, "")).lower()
    if "entail" in raw_label:
        label = "entailment"
    elif "contrad" in raw_label:
        label = "contradiction"
    elif "neutral" in raw_label:
        label = "neutral"
    else:
        if pred == 2:
            label = "entailment"
        elif pred == 0:
            label = "contradiction"

    entail_idx = 2
    for idx, lbl in id2label.items():
        if "entail" in str(lbl).lower():
            entail_idx = idx
            break
    entailment_score = probs[entail_idx] if entail_idx < len(probs) else probs[-1]

    claim_doc = nlp(claim)
    has_negation_claim = any(t.dep_ == "neg" or t.lower_ == "no" for t in claim_doc)
    fact_doc = nlp(fact)
    has_negation_fact = any(t.dep_ == "neg" or t.lower_ == "no" for t in fact_doc)
    negation_high_risk = has_negation_claim and not has_negation_fact

    threshold = 0.5 if negation_high_risk else 0.85
    # BUG (found via scripts/diag_l3_judge.py A2, same defect as
    # iris/cerberus_v.py's _full_nli -- this function is a verbatim copy):
    # `and negation_high_risk` made the 0.85 branch of the ternary above
    # dead code -- this check only ever fired when negation_high_risk was
    # True, where threshold was already forced to 0.5. Non-negation
    # entailment pairs were NEVER floored. Fixed: drop the extra conjunct.
    if label == "entailment" and entailment_score <= threshold:
        label = "neutral"

    if label == "entailment":
        claim_gpes = {ent.text.lower().strip() for ent in claim_doc.ents if ent.label_ in ("GPE", "LOC")}
        if claim_gpes and not any(gpe in fact.lower() for gpe in claim_gpes):
            label = "neutral"

    return label, entailment_score


def _sentencize(gate: Any, caption_text: str) -> list[str]:
    """spaCy-sentencized caption, stripped of empties, with newlines/whitespace
    runs collapsed to a single space before splitting (same normalization as
    scripts/diag_v4_moondream_captions.py:_split_sentences)."""
    nlp = gate._get_spacy()
    normalized = re.sub(r"\s+", " ", caption_text).strip()
    if not normalized:
        return []
    doc = nlp(normalized)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


@dataclass
class Layer2Result:
    verdict: str  # "verified" | "rejected" | "unverifiable"
    best_sentence: str | None
    best_score: float
    n_sentences: int
    oversize_sentences: list[str] = field(default_factory=list)
    # Explicit per-sentence diagnostics for the within-frame spurious-
    # rejection watch metric (scripts/smoke_v2.py): a "rejected" verdict
    # means zero sentences entailed (entailment wins the verdict whenever
    # any sentence entails -- see the aggregation rule below), so
    # any_entailed is always False when verdict=="rejected" BY
    # CONSTRUCTION. It is reported anyway, honestly, rather than assumed --
    # a future change to the aggregation rule would otherwise silently
    # invalidate that invariant.
    any_entailed: bool = False
    best_contradiction_sentence: str | None = None
    best_contradiction_score: float | None = None


def verify_visual_claim(claim: VisualClaim, caption_text: str, nli: Any) -> Layer2Result:
    """Score claim.assertion against EACH sentence of caption_text (the
    caption for the exact frame the claim cites — caller-supplied, layer 2
    does not fetch it) using the per-pair NLI semantics in score_nli_pair.

    Verdict: any sentence entails -> verified (best entailing sentence +
    score); elif any sentence contradicts -> rejected; else unverifiable.

    Envelope guard: sentences over OVERSIZE_SENTENCE_TOKENS tokens are still
    scored, but counted into oversize_sentences and logged — a metric, not
    a gate.
    """
    sentences = _sentencize(nli, caption_text)
    tokenizer, _model = nli._get_nli_model()

    oversize: list[str] = []
    scored: list[tuple[str, str, float]] = []  # (sentence, label, score)
    for sentence in sentences:
        n_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if n_tokens > OVERSIZE_SENTENCE_TOKENS:
            oversize.append(sentence)
            logger.warning(
                "layer2: sentence exceeds %d tokens (%d) for frame_idx=%s: %r",
                OVERSIZE_SENTENCE_TOKENS, n_tokens, claim.frame_idx, sentence,
            )
        label, score = score_nli_pair(nli, claim.assertion, sentence)
        scored.append((sentence, label, score))

    if not scored:
        return Layer2Result(
            verdict="unverifiable", best_sentence=None, best_score=0.0,
            n_sentences=0, oversize_sentences=oversize,
        )

    entailed = [s for s in scored if s[1] == "entailment"]
    contradicted = [s for s in scored if s[1] == "contradiction"]
    best_contradiction = max(contradicted, key=lambda s: s[2]) if contradicted else None

    if entailed:
        best = max(entailed, key=lambda s: s[2])
        verdict = "verified"
    elif contradicted:
        best = max(contradicted, key=lambda s: s[2])
        verdict = "rejected"
    else:
        best = max(scored, key=lambda s: s[2])
        verdict = "unverifiable"

    return Layer2Result(
        verdict=verdict,
        best_sentence=best[0],
        best_score=best[2],
        n_sentences=len(sentences),
        oversize_sentences=oversize,
        any_entailed=bool(entailed),
        best_contradiction_sentence=best_contradiction[0] if best_contradiction else None,
        best_contradiction_score=best_contradiction[2] if best_contradiction else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — bounded absence (Decision 2)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Layer3Result:
    verdict: str  # "verified_absent" | "rejected"
    n_frames_checked: int
    rejecting_frame_idx: int | None = None
    rejecting_sentence: str | None = None
    rejecting_score: float | None = None
    phrasing: str = ""


# Penn Treebank finite-verb tags (spaCy's tagger uses this tagset): a clause
# has a finite verb iff one of its tokens carries one of these. VBG (gerund/
# present participle) and VBN (past participle) are deliberately excluded --
# "a person entering the building" has a VBG root and nothing else, so it
# reads as a bare fragment despite "entering" looking verb-like.
_FINITE_VERB_TAGS = {"VBZ", "VBP", "VBD", "MD"}


def _has_finite_verb(nlp: Any, text: str) -> bool:
    doc = nlp(text)
    return any(t.tag_ in _FINITE_VERB_TAGS for t in doc)


def _propositionalize_event(nlp: Any, event: str) -> str:
    """Deterministic guard, ONE rule: if event has no finite verb (a bare
    noun phrase / gerund fragment, e.g. "a person entering the building"),
    wrap it into a minimal declarative proposition -- "There is " + event +
    "." -- before it is used as an NLI hypothesis. Standard NLI hypotheses
    are full propositions; a fragment hypothesis is out-of-distribution for
    the model in a way that (per scripts/stack_e2e.log's traced findings)
    correlates with spurious high-confidence entailment against unrelated
    sentences. Template, not a model call, not tunable. Logs a warning
    naming the original fragment -- a non-compliant AbsenceClaim.event is
    still checked, just made visible, not silently reshaped without a trace.
    """
    if _has_finite_verb(nlp, event):
        return event
    wrapped = f"There is {event}."
    logger.warning(
        "layer3: AbsenceClaim.event has no finite verb (bare fragment), "
        "wrapping as a proposition: %r -> %r", event, wrapped,
    )
    return wrapped


_DISJUNCTION_WORDS = {"or", "and"}


def _find_top_level_conj(doc: Any):
    """Find a top-level 'X or/and Y' coordination: a token in dep_="conj"
    whose head has a "cc" sibling with lemma "or"/"and", where the head
    itself is either the sentence ROOT or a direct dependent of the ROOT
    (covers both "A [is loading] or [unloading] B" -- head is ROOT -- and
    "There is a [fire] or [smoke]" -- head is the ROOT's attr). Returns
    (head, cc, conj) or None if no such coordination exists."""
    for tok in doc:
        if tok.dep_ != "conj":
            continue
        head = tok.head
        cc = next((c for c in head.children if c.dep_ == "cc" and c.lemma_.lower() in _DISJUNCTION_WORDS), None)
        if cc is None:
            continue
        if head.dep_ == "ROOT" or head.head.dep_ == "ROOT":
            return head, cc, tok
    return None


def _decompose_disjunction(nlp: Any, event: str) -> list[str]:
    """Split a proposition on its top-level 'or'/'and' into atomic
    propositions -- "There is a fire or smoke." -> ["There is a fire.",
    "There is smoke."]; "Someone is loading or unloading a vehicle." ->
    ["Someone is loading a vehicle.", "Someone is unloading a vehicle."].
    An event with no top-level disjunction/conjunction returns unchanged
    as a single-element list. Deterministic, dependency-parse-driven, no
    model call.

    Two reconstruction cases, distinguished by the coordinated tokens'
    POS: VERB/AUX coordination (the shared object attaches under the
    SECOND conjunct in spaCy's parse -- "unloading" carries the "vehicle"
    dobj -- so it is borrowed for the first conjunct too) vs NOUN
    coordination (nothing to borrow; reuse whatever trailing material
    follows the whole coordinated phrase, e.g. "has been left behind.").
    """
    doc = nlp(event.strip())
    found = _find_top_level_conj(doc)
    if found is None:
        return [event.strip()]
    head, cc, conj = found

    leading_full = doc[:head.i].text
    leading_full = f"{leading_full} " if leading_full else ""
    # For the SECOND conjunct, drop a determiner that belongs specifically
    # to the head noun (e.g. "a" in "a fire or smoke" -> "There is smoke.",
    # not "There is a smoke.").
    det = next((c for c in head.children if c.dep_ == "det" and c.i == head.i - 1), None)
    leading_for_conj = doc[:det.i].text if det is not None else leading_full.strip()
    leading_for_conj = f"{leading_for_conj} " if leading_for_conj else ""

    if head.pos_ in ("VERB", "AUX") and conj.pos_ in ("VERB", "AUX"):
        borrowed = [t for t in conj.subtree if t.i != conj.i]
        if borrowed:
            lo, hi = min(t.i for t in borrowed), max(t.i for t in borrowed)
            tail = f" {doc[lo:hi + 1].text}."
        else:
            tail = "."
        part1 = f"{leading_full}{head.text}{tail}"
        part2 = f"{leading_full}{conj.text}{tail}"
    else:
        conj_end = max(t.i for t in conj.subtree)
        trailing = doc[conj_end + 1:].text
        sep = " " if trailing and not trailing.startswith((".", ",", "!", "?")) else ""
        part1 = f"{leading_full}{head.text}{sep}{trailing}".strip()
        part2 = f"{leading_for_conj}{conj.text}{sep}{trailing}".strip()

    parts = [re.sub(r"\s+([.,!?])", r"\1", re.sub(r"\s{2,}", " ", p)).strip() for p in (part1, part2)]
    return parts


def verify_absence_claim(
    claim: AbsenceClaim,
    all_retrieved_captions: list[tuple[int, str]],
    nli: Any,
) -> Layer3Result:
    """Bounded absence (approved Decision 2): run claim.event -- contracted
    to be a full declarative proposition phrasing the POSITIVE event (see
    aria._SYSTEM_PROMPT_V2 and claim_contract.AbsenceClaim), deterministically
    propositionalized via _propositionalize_event if it isn't -- through
    layer 2's per-sentence machinery against EVERY retrieved frame's caption
    (an absence claim carries no citation to bind to, unlike VisualClaim).

    DISJUNCTION DECOMPOSITION (scripts/diag_l3_judge.py A1: the compound
    "There is a fire or smoke." scored spurious high-confidence entailment
    against topically unrelated sentences in a way its atomic decomposition
    never did -- 0/516 pairs in that diagnostic's matrix): the proposition
    is split into atomic propositions via _decompose_disjunction before
    scoring. Absence requires ALL atomic propositions absent; ANY
    entailment (of any atomic proposition, against any frame) rejects the
    whole claim -- a tightening, not a loosening: "no fire or smoke" is
    only true if there is neither.

    No entailment anywhere -> verified_absent; any entailment -> rejected,
    recording the highest-scoring (frame, atomic proposition).

    phrasing is generated FROM this result, not left to answer prose: a
    verified_absent claim is only honest if the reader knows it was checked
    against a bounded set of frames, not the whole video -- so the "verified-
    absent over the N frames checked" wording is a hard part of the contract.
    """
    n = len(all_retrieved_captions)
    nlp = nli._get_spacy()

    hypothesis = _propositionalize_event(nlp, claim.event)
    atomic_props = _decompose_disjunction(nlp, hypothesis)
    if len(atomic_props) > 1:
        logger.info(
            "layer3: AbsenceClaim.event decomposed into %d atomic propositions: %r -> %r",
            len(atomic_props), hypothesis, atomic_props,
        )

    best_entailment: tuple[int, str, float, str] | None = None

    for prop in atomic_props:
        for frame_idx, caption_text in all_retrieved_captions:
            pseudo_claim = VisualClaim(frame_idx=frame_idx, assertion=prop)
            result = verify_visual_claim(pseudo_claim, caption_text, nli)
            if result.verdict == "verified":
                if best_entailment is None or result.best_score > best_entailment[2]:
                    best_entailment = (frame_idx, result.best_sentence, result.best_score, prop)

    if best_entailment is not None:
        frame_idx, sentence, score, prop = best_entailment
        return Layer3Result(
            verdict="rejected",
            n_frames_checked=n,
            rejecting_frame_idx=frame_idx,
            rejecting_sentence=sentence,
            rejecting_score=score,
            phrasing=(
                f"rejected: {prop!r} (atomic proposition from {claim.event!r}) found at frame {frame_idx} "
                f"({sentence!r}, score={score:.4f})"
            ),
        )

    return Layer3Result(
        verdict="verified_absent",
        n_frames_checked=n,
        phrasing=f"verified-absent over the {n} frames checked",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router + answer badge (Cerberus v2)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Evidence:
    """Everything verify_answer needs for one query, bundled so its call
    signature stays verify_answer(answer_claims, evidence) per the v2
    contract -- no separate index/nli params to thread through."""
    index: IRISIndex
    retrieved_frames: list[dict]  # post-lazy-caption frame dicts (iris.query._build_retrieved shape)
    nli: Any  # a CerberusV gate, e.g. from get_nli_gate()


@dataclass
class ClaimVerdict:
    claim: Claim
    label: str  # verified|rejected|unverifiable|verified_absent|checked_pass|checked_fail|field_unavailable
    reason: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class AnswerVerification:
    claim_verdicts: list[ClaimVerdict]
    badge: str  # verified | flagged | partial | unverified
    core_claim_verdict: ClaimVerdict


def _extract_caption_text(caption_val: Any) -> str | None:
    """Same extraction/rejection rule as l1_elysium.L1ElysiumCache._frame_to_nli_fact,
    applied to a retrieved-frame dict's "caption" value instead of a CachedFrame."""
    if not caption_val:
        return None
    if isinstance(caption_val, dict):
        semantic_caption = caption_val.get("semantic_caption")
    else:
        semantic_caption = caption_val
    if not semantic_caption or semantic_caption == "[CAPTION_FAILED]":
        return None
    return semantic_caption


def _verify_visual(claim: VisualClaim, evidence: Evidence) -> ClaimVerdict:
    frame_dict = next(
        (f for f in evidence.retrieved_frames if f.get("frame_idx") == claim.frame_idx), None
    )
    if frame_dict is None:
        # Never fetch a new frame to satisfy a claim -- only score against
        # what retrieval + lazy-captioning already surfaced to ARIA.
        return ClaimVerdict(claim=claim, label="unverifiable", reason="cited frame not in evidence")

    caption_text = _extract_caption_text(frame_dict.get("caption"))
    if caption_text is None:
        return ClaimVerdict(claim=claim, label="unverifiable", reason="no caption available for cited frame")

    result = verify_visual_claim(claim, caption_text, evidence.nli)
    return ClaimVerdict(
        claim=claim,
        label=result.verdict,
        reason=f"best_sentence={result.best_sentence!r} score={result.best_score:.4f}",
        detail={
            "best_sentence": result.best_sentence,
            "best_score": result.best_score,
            "n_sentences": result.n_sentences,
            "oversize_sentences": result.oversize_sentences,
            "any_entailed": result.any_entailed,
            "best_contradiction_sentence": result.best_contradiction_sentence,
            "best_contradiction_score": result.best_contradiction_score,
        },
    )


def _verify_metadata(claim: MetadataClaim, evidence: Evidence) -> ClaimVerdict:
    label = verify_metadata_claim(claim, evidence.index)
    return ClaimVerdict(
        claim=claim, label=label,
        reason=f"field={claim.field} stated_value={claim.stated_value}",
    )


def _verify_absence(claim: AbsenceClaim, evidence: Evidence) -> ClaimVerdict:
    captions: list[tuple[int, str]] = []
    for f in evidence.retrieved_frames:
        text = _extract_caption_text(f.get("caption"))
        if text is not None:
            captions.append((f["frame_idx"], text))

    result = verify_absence_claim(claim, captions, evidence.nli)
    return ClaimVerdict(
        claim=claim,
        label=result.verdict,
        reason=result.phrasing,
        detail={
            "n_frames_checked": result.n_frames_checked,
            "rejecting_frame_idx": result.rejecting_frame_idx,
            "rejecting_sentence": result.rejecting_sentence,
            "rejecting_score": result.rejecting_score,
        },
    )


def _verify_global(claim: GlobalClaim, evidence: Evidence) -> ClaimVerdict:
    return ClaimVerdict(claim=claim, label="unverifiable", reason="global claims are not checkable")


# Badge computation (approved Decision 1): a claim's label falls into exactly
# one of three classes. checked_fail counts as a rejection (a factually wrong
# metadata claim is as much a red flag as a contradicted visual one);
# field_unavailable counts as unverifiable, same as a visual/absence claim
# with no checkable evidence.
_PASS_LABELS = {"verified", "verified_absent", "checked_pass"}
_REJECT_LABELS = {"rejected", "checked_fail"}


def _label_class(label: str) -> str:
    if label in _PASS_LABELS:
        return "pass"
    if label in _REJECT_LABELS:
        return "reject"
    return "unverifiable"  # "unverifiable", "field_unavailable"


def _compute_badge(core_verdict: ClaimVerdict, verdicts: list[ClaimVerdict]) -> str:
    """verified  <- core claim passes AND zero rejections anywhere.
    flagged   <- any rejection anywhere (core or non-core).
    partial   <- core passes, zero rejections, but an unverifiable non-core
                 claim remains (still counts as a pass for the badge).
    unverified <- core claim does not pass (unverifiable, or itself rejected --
                 a rejected core is "unverified" here, not "flagged", since
                 the core claim itself has no verified answer to flag)."""
    if _label_class(core_verdict.label) != "pass":
        return "unverified"
    if any(_label_class(v.label) == "reject" for v in verdicts):
        return "flagged"
    if any(_label_class(v.label) == "unverifiable" for v in verdicts if v is not core_verdict):
        return "partial"
    return "verified"


def verify_answer(answer_claims: AnswerClaims, evidence: Evidence) -> AnswerVerification:
    """Router: dispatches each claim in answer_claims to its layer, then
    computes the answer badge from the resulting per-claim verdicts.

    MetadataClaim -> layer 1 (verify_metadata_claim, ground truth from
                     evidence.index).
    VisualClaim   -> layer 2 (verify_visual_claim) against its cited frame's
                     caption, looked up in evidence.retrieved_frames -- a
                     citation to a frame outside that set is unverifiable,
                     never fetched.
    AbsenceClaim  -> layer 3 (verify_absence_claim) against every retrieved
                     frame's caption.
    GlobalClaim   -> unverifiable, surfaced with a label (not checkable by
                     any layer).
    """
    verdicts: list[ClaimVerdict] = []
    core_verdict: ClaimVerdict | None = None

    for claim in answer_claims.claims:
        if isinstance(claim, VisualClaim):
            v = _verify_visual(claim, evidence)
        elif isinstance(claim, MetadataClaim):
            v = _verify_metadata(claim, evidence)
        elif isinstance(claim, AbsenceClaim):
            v = _verify_absence(claim, evidence)
        elif isinstance(claim, GlobalClaim):
            v = _verify_global(claim, evidence)
        else:
            raise TypeError(f"Unknown claim type: {type(claim)!r}")

        verdicts.append(v)
        if isinstance(claim, (VisualClaim, AbsenceClaim)) and claim.is_core:
            core_verdict = v

    if core_verdict is None:
        # AnswerClaims.__post_init__ guarantees exactly one is_core claim
        # among Visual/Absence claims; reaching here means that invariant
        # was bypassed (e.g. claims list mutated after construction).
        raise ValueError("AnswerClaims has no core claim -- invariant violated")

    badge = _compute_badge(core_verdict, verdicts)
    return AnswerVerification(claim_verdicts=verdicts, badge=badge, core_claim_verdict=core_verdict)
