"""scripts/diag_l3_judge.py — Layer-3 judge-rule diagnostic. READ-ONLY: makes
no changes to iris/. Investigates the STOP condition raised in
scripts/verify_stack_e2e.py (stack_e2e_v2.log): "There is a fire or
smoke." scored spurious entailment (0.4584 - 0.9483) against topically
unrelated sentences, in BOTH caption arms, even after propositionalizing
the AbsenceClaim.event contract.

PART A1 -- DISJUNCTION MATRIX: scores the DECOMPOSED atomic hypotheses --
"There is a fire." / "There is smoke." (from "There is a fire or smoke.")
and "Someone is loading a vehicle." / "Someone is unloading a vehicle."
(from "Someone is loading or unloading a vehicle.") -- against: the 3 exact
failing premises from stack_e2e_v2.log, every distinct BLIP caption_only
(diag_v2_capture.jsonl), and every spaCy-sentencized Moondream pass1
sentence. Prints every ENTAILMENT-labeled pair with its score.

PART A2 -- FLOOR IMPACT: re-derives the verdict every already-scored pair
would get under a CORRECTED entailment floor (0.85 for non-negation pairs,
0.5 for negation-risk pairs -- the threshold iris.cerberus_layers.
score_nli_pair / iris.cerberus_v.CerberusV._full_nli already COMPUTE but,
per a dead-code defect, never actually apply unless negation_high_risk is
True), for: layer-2 fixtures (canonical + normalized kernels + Moondream-8,
via scripts/verify_layer2.py's own fixture generation, reused not
re-derived), the fabricated-claim regression (BLIP + Moondream), and the
stack_e2e absence verdicts (re-derived live for the 7 negative queries,
same fixtures as scripts/verify_stack_e2e.py). Prints before/after verdict
deltas.

The floor recompute needs no extra NLI forward passes: for any pair whose
CURRENT label is not "entailment", the floor cannot change it (it only ever
downgrades entailment -> neutral). For a pair labeled "entailment", if it
was already negation-high-risk the existing (buggy) code already correctly
checked it against 0.5 -- floor recompute is a no-op there too. The ONLY
pairs the floor can change are entailment-labeled, non-negation-risk pairs,
which the current code never checked against anything -- see
_apply_corrected_floor.

VERIFY:
    python scripts/diag_l3_judge.py 2>&1 | tee diag_l3_judge.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.query as iris_query
from iris.cerberus_layers import _sentencize, get_nli_gate, score_nli_pair
from scripts.demo_cctv_query import CFG, _load_index
from scripts.diag_v2_scoping_separation import CAPTURE_JSONL, FABRICATED_CLAIMS, _load_capture
from scripts.diag_v4_moondream_captions import _caption_with_moondream, _check_ollama, _fetch_frames_pil, _image_to_b64_jpeg
from scripts.verify_layer2 import (
    CANONICAL_ASSERTION,
    CANONICAL_FRAME_IDX,
    _load_pass1,
    _query_pools,
    _regenerate_normalized_kernels,
)
from scripts.verify_stack_e2e import NEGATIVE_SPECS

ENTAILMENT_FLOOR = 0.85

# ── the 3 exact failing (hypothesis, premise) pairs from stack_e2e_v2.log ──
FAILING_PAIRS = [
    ("There is a fire or smoke.", "a parking lot",
     "BLIP frame 1263, score=0.6495"),
    ("There is a fire or smoke.", "The scene conveys a sense of activity and life around the building.",
     "MOONDREAM frame 228, score=0.9483"),
    ("Someone is loading or unloading a vehicle.",
     "The image shows a parking lot with several cars parked in it, including one car that appears to be "
     "in the process of being towed away by two men dressed in dark clothing and carrying a large object "
     "on their shoulders.",
     "MOONDREAM frame 24469, score=0.9486"),
]

# ── hand-decomposed atomic hypotheses (task-specified, not the algorithmic
# decomposer -- this diagnostic predates/validates that decomposer) ──
DECOMPOSED_HYPOTHESES = [
    "There is a fire.",
    "There is smoke.",
    "Someone is loading a vehicle.",
    "Someone is unloading a vehicle.",
]


# ─────────────────────────────────────────────────────────────────────────────
# A1 — disjunction matrix
# ─────────────────────────────────────────────────────────────────────────────

def _distinct_blip_captions() -> list[str]:
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: {CAPTURE_JSONL} missing", file=sys.stderr)
        sys.exit(1)
    records = _load_capture()
    return sorted({r["caption_only"] for r in records if r["caption_only"] is not None})


def _distinct_moondream_sentences(gate) -> list[str]:
    rows = _load_pass1()
    sentences: set[str] = set()
    for r in rows:
        for s in _sentencize(gate, r["moondream_caption"]):
            sentences.add(s)
    return sorted(sentences)


def run_disjunction_matrix(gate) -> None:
    print("=" * 100)
    print("A1 -- DISJUNCTION MATRIX")
    print("=" * 100)

    premises: list[tuple[str, str]] = []  # (premise_text, source_label)
    for hyp, premise, label in FAILING_PAIRS:
        premises.append((premise, f"FAILING PAIR ({label})"))
    for cap in _distinct_blip_captions():
        premises.append((cap, "BLIP caption_only"))
    for sent in _distinct_moondream_sentences(gate):
        premises.append((sent, "Moondream pass1 sentence"))

    print(f"  n hypotheses: {len(DECOMPOSED_HYPOTHESES)}   n premises: {len(premises)}   "
          f"n pairs: {len(DECOMPOSED_HYPOTHESES) * len(premises)}")
    print()

    hits = []
    for hyp in DECOMPOSED_HYPOTHESES:
        for premise, source in premises:
            label, score = score_nli_pair(gate, hyp, premise)
            if label == "entailment":
                hits.append((hyp, premise, source, score))

    print(f"  entailment-labeled pairs: {len(hits)}")
    print()
    hdr = f"  {'hypothesis':<42} | {'score':>6} | {'source':<30} | {'premise':<70}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for hyp, premise, source, score in sorted(hits, key=lambda h: -h[3]):
        premise_disp = (premise[:67] + "...") if len(premise) > 70 else premise
        print(f"  {hyp:<42} | {score:>6.4f} | {source:<30} | {premise_disp:<70}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# A2 — floor impact
# ─────────────────────────────────────────────────────────────────────────────

def _apply_corrected_floor(nlp, claim: str, fact: str, label: str, score: float,
                            floor: float = ENTAILMENT_FLOOR) -> str:
    """Re-derive the label a CORRECTED score_nli_pair would produce, from an
    already-computed (label, score) pair, with NO extra NLI forward pass.
    See module docstring for why this needs no re-scoring."""
    if label != "entailment":
        return label
    claim_doc = nlp(claim)
    fact_doc = nlp(fact)
    has_negation_claim = any(t.dep_ == "neg" or t.lower_ == "no" for t in claim_doc)
    has_negation_fact = any(t.dep_ == "neg" or t.lower_ == "no" for t in fact_doc)
    negation_high_risk = has_negation_claim and not has_negation_fact
    if negation_high_risk:
        return label  # existing code already correctly floors this case at 0.5
    return "neutral" if score <= floor else "entailment"


def _sentence_pairs(gate, hypothesis: str, premise_text: str) -> list[tuple[str, str, float]]:
    """(sentence, label, score) for every sentence of premise_text scored
    against hypothesis via the CURRENT score_nli_pair -- the same per-
    sentence data verify_visual_claim aggregates internally, exposed raw."""
    sentences = _sentencize(gate, premise_text)
    return [(s, *score_nli_pair(gate, hypothesis, s)) for s in sentences]


def _aggregate(pairs: list[tuple[str, str, float]]) -> str:
    labels = [lbl for _, lbl, _ in pairs]
    if any(l == "entailment" for l in labels):
        return "verified"
    if any(l == "contradiction" for l in labels):
        return "rejected"
    return "unverifiable"


def _aggregate_floored(nlp, hypothesis: str, pairs: list[tuple[str, str, float]]) -> str:
    floored = [(s, _apply_corrected_floor(nlp, hypothesis, s, lbl, sc), sc) for s, lbl, sc in pairs]
    return _aggregate(floored)


def _print_delta_table(rows: list[tuple]) -> None:
    hdr = f"  {'item':<55} | {'before':<14} | {'after':<14} | {'delta'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_delta = 0
    for label, before, after in rows:
        delta = "CHANGED" if before != after else ""
        if delta:
            n_delta += 1
        label_disp = (label[:52] + "...") if len(label) > 55 else label
        print(f"  {label_disp:<55} | {before:<14} | {after:<14} | {delta}")
    print(f"  ({n_delta}/{len(rows)} changed)")
    print()


def run_floor_impact_layer2(gate) -> None:
    print("=" * 100)
    print("A2.1 -- FLOOR IMPACT: layer-2 fixtures (canonical + normalized kernels)")
    print("=" * 100)

    records = _load_capture()
    blip_caption_25069 = None
    from scripts.diag_v3_visual_scoping import _fact_header
    for r in records:
        if r["fact_text"] is None:
            continue
        fi, _ts = _fact_header(r["fact_text"])
        if fi == CANONICAL_FRAME_IDX:
            blip_caption_25069 = r["caption_only"]
            break

    rows = []
    label, score = score_nli_pair(gate, CANONICAL_ASSERTION, blip_caption_25069)
    floored = _apply_corrected_floor(gate._get_spacy(), CANONICAL_ASSERTION, blip_caption_25069, label, score)
    rows.append((f"canonical (score={score:.4f})", label, floored))

    kernel_fixtures = _regenerate_normalized_kernels(gate, records)
    for normalized, frame_idx, caption in kernel_fixtures:
        label, score = score_nli_pair(gate, normalized, caption)
        floored = _apply_corrected_floor(gate._get_spacy(), normalized, caption, label, score)
        rows.append((f"kernel frame={frame_idx} (score={score:.4f}) {normalized[:30]!r}", label, floored))

    pass1_rows = _load_pass1()
    moondream_by_frame = {r["frame_idx"]: r["moondream_caption"] for r in pass1_rows}
    for normalized, frame_idx, _blip_caption in kernel_fixtures:
        moon_caption = moondream_by_frame.get(frame_idx)
        if moon_caption is None:
            continue
        pairs = _sentence_pairs(gate, normalized, moon_caption)
        before = _aggregate(pairs)
        after = _aggregate_floored(gate._get_spacy(), normalized, pairs)
        rows.append((f"Moondream kernel frame={frame_idx} {normalized[:30]!r}", before, after))

    _print_delta_table(rows)


def run_floor_impact_fabricated(idx, gate) -> None:
    print("=" * 100)
    print("A2.2 -- FLOOR IMPACT: fabricated-claim regression (BLIP + Moondream)")
    print("=" * 100)

    records = _load_capture()
    pools = _query_pools(records)
    n_before_verified = 0
    n_after_verified = 0
    n_pairs = 0
    for claim_text in FABRICATED_CLAIMS:
        for query, frame_caps in pools.items():
            for frame_idx, caption in frame_caps:
                n_pairs += 1
                pairs = _sentence_pairs(gate, claim_text, caption)
                before = _aggregate(pairs)
                after = _aggregate_floored(gate._get_spacy(), claim_text, pairs)
                if before == "verified":
                    n_before_verified += 1
                if after == "verified":
                    n_after_verified += 1

    pass1_rows = _load_pass1()
    for claim_text in FABRICATED_CLAIMS:
        for row in pass1_rows:
            n_pairs += 1
            pairs = _sentence_pairs(gate, claim_text, row["moondream_caption"])
            before = _aggregate(pairs)
            after = _aggregate_floored(gate._get_spacy(), claim_text, pairs)
            if before == "verified":
                n_before_verified += 1
            if after == "verified":
                n_after_verified += 1

    print(f"  n (fabricated claim, frame) bindings scored (BLIP + Moondream): {n_pairs}")
    print(f"  verified BEFORE floor: {n_before_verified} (expected 0)")
    print(f"  verified AFTER floor:  {n_after_verified} (expected 0)")
    print()


def run_floor_impact_stack_e2e(idx, gate) -> None:
    print("=" * 100)
    print("A2.3 -- FLOOR IMPACT: stack_e2e absence verdicts (7 negative queries, live re-derive)")
    print("=" * 100)

    _check_ollama()
    rows = []
    for question, event in NEGATIVE_SPECS:
        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)

        blip_pairs_all = []
        for f in retrieved:
            cap = f.get("caption")
            text = cap.get("semantic_caption") if isinstance(cap, dict) else cap
            if not text:
                continue
            blip_pairs_all.extend(_sentence_pairs(gate, event, text))
        before = _aggregate(blip_pairs_all)
        after = _aggregate_floored(gate._get_spacy(), event, blip_pairs_all)
        before_verdict = "rejected" if before == "verified" else "verified_absent"
        after_verdict = "rejected" if after == "verified" else "verified_absent"
        rows.append((f"BLIP  {question}", before_verdict, after_verdict))

        frame_idxs = [f["frame_idx"] for f in retrieved]
        pil_by_frame = _fetch_frames_pil(idx, frame_idxs)
        moon_pairs_all = []
        for frame_idx in frame_idxs:
            b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
            caption = _caption_with_moondream(b64)
            moon_pairs_all.extend(_sentence_pairs(gate, event, caption))
        before = _aggregate(moon_pairs_all)
        after = _aggregate_floored(gate._get_spacy(), event, moon_pairs_all)
        before_verdict = "rejected" if before == "verified" else "verified_absent"
        after_verdict = "rejected" if after == "verified" else "verified_absent"
        rows.append((f"MOON  {question}", before_verdict, after_verdict))

    _print_delta_table(rows)


def main() -> None:
    gate = get_nli_gate()
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print()

    run_disjunction_matrix(gate)
    run_floor_impact_layer2(gate)
    run_floor_impact_fabricated(idx, gate)
    run_floor_impact_stack_e2e(idx, gate)


if __name__ == "__main__":
    main()
