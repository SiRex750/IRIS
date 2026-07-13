"""scripts/verify_stack_e2e.py — validates the FULL Cerberus v2 verification
stack (router + layers 1/2/3 + answer badge) with HAND-BUILT AnswerClaims
fixtures, bypassing ARIA entirely. Retrieval, lazy captioning, and the NLI
gate are all real; only the claim-generation step is replaced by fixtures a
compliant answerer SHOULD emit, so a stack bug can't hide behind ARIA's own
(separately-measured, often noncompliant -- see smoke_v2.log) JSON output.

QUERY SET: the same 11 queries as scripts/smoke_v2.py (virat_query_smoke's
3 + demo_cctv_query's 4 POSITIVE + 4 NEGATIVE). Polarity assignment (which
queries get a core VisualClaim vs a core AbsenceClaim) is by GROUND TRUTH
in VIRAT_S_000102's actual content, not by source-list membership, per
human review of the captured captions (both arms) and the source video:

  POSITIVE (4, core VisualClaim) -- demo_cctv_query.POSITIVE_QUERIES:
    "Is there a car in the parking lot?"
    "Is a person visible in the scene?"
    "Are there vehicles parked?"
    "Is anyone walking?"

  NEGATIVE (7, core AbsenceClaim) -- none of these events occur anywhere in
  either caption arm for this video:
    "Is anyone loading or unloading a vehicle?"        (smoke #1)
    "Did a person enter the building?"                 (smoke #2 -- human
        video review: early window shows 3 people standing in conversation;
        late window shows a man in grey walking FROM the building area, out
        of camera view -- movement AWAY, not entry; any entry predates the
        recording. Confirmed negative.)
    "Is there an unattended bag or object left behind?" (smoke #3)
    "Is anyone loading a vehicle?"
    "Is there an unattended bag?"
    "Is anyone running or fleeing?"
    "Is there a fire or smoke?"

For each positive query, the "known-correct frame" is chosen dynamically:
the first frame in that query's REAL retrieved pool (BLIP arm) whose
caption matches a query-appropriate keyword pattern -- not a hardcoded
frame_idx, so the fixture stays grounded in whatever retrieval actually
returns. The VisualClaim assertion is a light paraphrase of that frame's
BLIP caption (see _assertion_from_caption). Every fixture also sprinkles a
non-core MetadataClaim (real action_score off the same frame) and a
non-core GlobalClaim, to exercise all four routing branches every query.

Each fixture (same claims) is run through verify_answer against TWO
evidence arms for the SAME retrieved frame set: BLIP captions (already
lazily cached via iris.query._ensure_captions) and live Moondream captions
(fetched fresh via Ollama for these exact frames -- not the frozen 55-frame
diag_v4 sample, which may not overlap this run's retrieval). Scoring the
same assertion against two independently-worded caption sources is a
stronger test of layer 2's sentence-scoping than self-entailment against
the caption the assertion was paraphrased from.

Finally reruns the fabricated-claim regression (scripts/diag_v2_scoping_
separation.FABRICATED_CLAIMS) against every distinct frame retrieved this
run, both arms -- must stay 0, same invariant as scripts/verify_layer2.py.

VERIFY:
    python scripts/verify_stack_e2e.py 2>&1 | tee stack_e2e.log

STOP: any negative query's core AbsenceClaim verdict comes back "rejected"
in either arm -- that means layer 3 found the positive-phrased event
somewhere in that query's evidence, which on these 11 queries would be a
genuine finding to trace (why does a caption describe an event we already
confirmed absent?), not a bug to suppress. Also STOP on any fabricated
claim verifying (same invariant as verify_layer2.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.query as iris_query
from iris.cerberus_layers import Evidence, get_nli_gate, verify_answer, verify_visual_claim
from iris.claim_contract import AbsenceClaim, AnswerClaims, GlobalClaim, MetadataClaim, VisualClaim
from scripts.demo_cctv_query import CFG, NEGATIVE_QUERIES, POSITIVE_QUERIES, _load_index
from scripts.diag_v2_scoping_separation import FABRICATED_CLAIMS
from scripts.diag_v4_moondream_captions import _caption_with_moondream, _check_ollama, _fetch_frames_pil, _image_to_b64_jpeg
from scripts.virat_query_smoke import QUERIES as SMOKE_QUERIES

# ─────────────────────────────────────────────────────────────────────────────
# Query polarity assignment (see module docstring for the ground-truth read)
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE_SPECS = [
    ("Is there a car in the parking lot?", re.compile(r"\bcars?\b|\bvehicles?\b", re.IGNORECASE)),
    ("Is a person visible in the scene?", re.compile(r"\bpersons?\b|\bpeople\b", re.IGNORECASE)),
    ("Are there vehicles parked?", re.compile(r"\bcars?\b|\bvehicles?\b", re.IGNORECASE)),
    ("Is anyone walking?", re.compile(r"\bwalk(?:ing|s)?\b", re.IGNORECASE)),
]
assert [q for q, _ in POSITIVE_SPECS] == POSITIVE_QUERIES, "POSITIVE_SPECS must track demo_cctv_query.POSITIVE_QUERIES verbatim"

NEGATIVE_SPECS = [
    ("Is anyone loading or unloading a vehicle?", "Someone is loading or unloading a vehicle."),
    ("Did a person enter the building?", "A person is entering the building."),
    ("Is there an unattended bag or object left behind?", "An unattended bag or object has been left behind."),
    ("Is anyone loading a vehicle?", "Someone is loading a vehicle."),
    ("Is there an unattended bag?", "There is an unattended bag."),
    ("Is anyone running or fleeing?", "Someone is running or fleeing."),
    ("Is there a fire or smoke?", "There is a fire or smoke."),
]
assert {q for q, _ in NEGATIVE_SPECS} == set(SMOKE_QUERIES) | set(NEGATIVE_QUERIES), (
    "NEGATIVE_SPECS must cover exactly virat_query_smoke.QUERIES + demo_cctv_query.NEGATIVE_QUERIES"
)


def _assertion_from_caption(caption_text: str) -> str:
    """Light paraphrase of a caption into a plain visual-language assertion --
    same style as the canonical fixture used throughout scripts/verify_layer2.py
    ("a car parked in a parking lot." -> "there is a car parked in the parking lot")."""
    text = caption_text.strip().rstrip(".")
    if text.lower().startswith("there is") or text.lower().startswith("there are"):
        return text
    return f"there is {text}"


def _pick_frame(retrieved_frames: list[dict], pattern: "re.Pattern") -> dict | None:
    for f in retrieved_frames:
        caption_val = f.get("caption")
        text = caption_val.get("semantic_caption") if isinstance(caption_val, dict) else caption_val
        if text and pattern.search(text):
            return f
    return None


def _caption_text(frame_dict: dict) -> str | None:
    caption_val = frame_dict.get("caption")
    if isinstance(caption_val, dict):
        return caption_val.get("semantic_caption")
    return caption_val


# ─────────────────────────────────────────────────────────────────────────────
# Fixture construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_positive_fixture(question: str, pattern: "re.Pattern", retrieved_frames: list[dict]) -> AnswerClaims:
    frame = _pick_frame(retrieved_frames, pattern)
    if frame is None:
        print(f"FATAL: no retrieved frame for {question!r} matches pattern {pattern.pattern!r} -- "
              f"cannot build a grounded fixture. Retrieved captions:", file=sys.stderr)
        for f in retrieved_frames:
            print(f"    frame_idx={f['frame_idx']} caption={_caption_text(f)!r}", file=sys.stderr)
        sys.exit(1)

    assertion = _assertion_from_caption(_caption_text(frame))
    claims = [
        VisualClaim(frame_idx=frame["frame_idx"], assertion=assertion, is_core=True),
        MetadataClaim(
            frame_idx=frame["frame_idx"], field="action_score",
            stated_value=round(frame.get("action_score", 0.0), 2),
            source_text=f"frame {frame['frame_idx']} action score {frame.get('action_score', 0.0):.2f}",
        ),
        GlobalClaim(text="Overall the scene is a static outdoor parking-lot/walkway area."),
    ]
    return AnswerClaims(query=question, claims=claims)


def _build_negative_fixture(question: str, event: str, retrieved_frames: list[dict]) -> AnswerClaims:
    claims: list = [AbsenceClaim(event=event, is_core=True)]
    if retrieved_frames:
        top = retrieved_frames[0]
        cap_text = _caption_text(top)
        if cap_text:
            claims.insert(0, VisualClaim(
                frame_idx=top["frame_idx"], assertion=_assertion_from_caption(cap_text), is_core=False,
            ))
            claims.append(MetadataClaim(
                frame_idx=top["frame_idx"], field="action_score",
                stated_value=round(top.get("action_score", 0.0), 2),
                source_text=f"frame {top['frame_idx']} action score {top.get('action_score', 0.0):.2f}",
            ))
    claims.append(GlobalClaim(text="Overall the scene is a static outdoor parking-lot/walkway area."))
    return AnswerClaims(query=question, claims=claims)


# ─────────────────────────────────────────────────────────────────────────────
# Moondream arm: live captions for exactly this run's retrieved frames
# ─────────────────────────────────────────────────────────────────────────────

_MOONDREAM_CACHE: dict[int, str] = {}


def _moondream_captions_for(idx, frame_idxs: list[int]) -> dict[int, str]:
    """Live Moondream captions for frame_idxs, memoized across the whole run
    (_MOONDREAM_CACHE) so a frame retrieved by multiple queries -- common,
    e.g. frame 25069 -- is only captioned once via Ollama, not once per query
    plus again in the final fabricated-regression rerun."""
    missing = [fi for fi in frame_idxs if fi not in _MOONDREAM_CACHE]
    if missing:
        print(f"  fetching {len(missing)} frame(s) via seek for live Moondream captioning...")
        pil_by_frame = _fetch_frames_pil(idx, missing)
        for i, frame_idx in enumerate(missing):
            b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
            caption = _caption_with_moondream(b64)
            _MOONDREAM_CACHE[frame_idx] = caption
            print(f"    [{i + 1}/{len(missing)}] frame_idx={frame_idx} moondream={caption!r}")
    return {fi: _MOONDREAM_CACHE[fi] for fi in frame_idxs if fi in _MOONDREAM_CACHE}


def _retrieved_with_moondream(retrieved_frames: list[dict], moondream_by_frame: dict[int, str]) -> list[dict]:
    out = []
    for f in retrieved_frames:
        g = dict(f)
        cap = moondream_by_frame.get(f["frame_idx"])
        g["caption"] = {"semantic_caption": cap} if cap is not None else None
        out.append(g)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_claim_table(claim_verdicts: list) -> None:
    hdr = f"  {'type':<10} | {'core':<5} | {'label':<16} | {'reason':<75}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for v in claim_verdicts:
        claim_type = type(v.claim).__name__
        is_core = getattr(v.claim, "is_core", False)
        reason_disp = (v.reason[:72] + "...") if len(v.reason) > 75 else v.reason
        print(f"  {claim_type:<10} | {str(is_core):<5} | {v.label:<16} | {reason_disp:<75}")
    print()


def run_arm(answer_claims: AnswerClaims, evidence: Evidence, arm_label: str, polarity: str,
            stop_findings: list) -> str:
    verification = verify_answer(answer_claims, evidence)
    print(f"-- {arm_label} ARM --")
    _print_claim_table(verification.claim_verdicts)
    print(f"  core_claim_verdict: label={verification.core_claim_verdict.label}  "
          f"reason={verification.core_claim_verdict.reason}")
    print(f"  BADGE: {verification.badge}")
    print()

    if polarity == "negative" and verification.core_claim_verdict.label == "rejected":
        stop_findings.append((answer_claims.query, arm_label, verification.core_claim_verdict))

    return verification.badge


def main() -> None:
    _check_ollama()
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print()

    gate = get_nli_gate()
    stop_findings: list = []
    badge_tally: dict[str, dict[str, int]] = {"positive": {}, "negative": {}}
    all_frame_idxs: set[int] = set()
    all_retrieved_by_query: dict[str, list[dict]] = {}

    query_specs: list[tuple[str, str, object]] = (
        [(q, "positive", pat) for q, pat in POSITIVE_SPECS]
        + [(q, "negative", ev) for q, ev in NEGATIVE_SPECS]
    )
    # Preserve the original 11-query presentation order (smoke, positive, negative)
    order = SMOKE_QUERIES + POSITIVE_QUERIES + NEGATIVE_QUERIES
    query_specs.sort(key=lambda t: order.index(t[0]))

    for question, polarity, spec in query_specs:
        print("=" * 100)
        print(f"[{polarity.upper()}] QUERY: {question!r}")
        print("=" * 100)

        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)
        all_retrieved_by_query[question] = retrieved
        all_frame_idxs.update(f["frame_idx"] for f in retrieved)

        if polarity == "positive":
            answer_claims = _build_positive_fixture(question, spec, retrieved)
        else:
            answer_claims = _build_negative_fixture(question, spec, retrieved)

        print("-- FIXTURE --")
        for c in answer_claims.claims:
            print(f"  {c.to_dict()}")
        print()

        blip_evidence = Evidence(index=idx, retrieved_frames=retrieved, nli=gate)
        blip_badge = run_arm(answer_claims, blip_evidence, "BLIP", polarity, stop_findings)

        moon_by_frame = _moondream_captions_for(idx, [f["frame_idx"] for f in retrieved])
        moon_retrieved = _retrieved_with_moondream(retrieved, moon_by_frame)
        moon_evidence = Evidence(index=idx, retrieved_frames=moon_retrieved, nli=gate)
        moon_badge = run_arm(answer_claims, moon_evidence, "MOONDREAM", polarity, stop_findings)

        badge_tally[polarity].setdefault(f"BLIP:{blip_badge}", 0)
        badge_tally[polarity][f"BLIP:{blip_badge}"] += 1
        badge_tally[polarity].setdefault(f"MOONDREAM:{moon_badge}", 0)
        badge_tally[polarity][f"MOONDREAM:{moon_badge}"] += 1

    # ── fabricated-claim regression rerun, this run's live frame pool ───────
    print("=" * 100)
    print("FABRICATED REGRESSION RERUN (this run's retrieved frame pool, both arms)")
    print("=" * 100)

    frame_idxs_sorted = sorted(all_frame_idxs)
    blip_caption_by_frame: dict[int, str | None] = {}
    for frames in all_retrieved_by_query.values():
        for f in frames:
            blip_caption_by_frame[f["frame_idx"]] = _caption_text(f)

    moon_caption_by_frame = _moondream_captions_for(idx, frame_idxs_sorted)

    fab_hits = []
    n_pairs = 0
    for claim_text in FABRICATED_CLAIMS:
        for frame_idx in frame_idxs_sorted:
            for arm_label, cap in (("BLIP", blip_caption_by_frame.get(frame_idx)),
                                    ("MOONDREAM", moon_caption_by_frame.get(frame_idx))):
                if not cap:
                    continue
                n_pairs += 1
                claim = VisualClaim(frame_idx=frame_idx, assertion=claim_text)
                result = verify_visual_claim(claim, cap, gate)
                if result.verdict == "verified":
                    fab_hits.append((claim_text, frame_idx, arm_label, cap, result))

    print(f"  n (fabricated claim, frame, arm) bindings scored: {n_pairs}")
    print(f"  verified: {len(fab_hits)} (expected 0)")
    print()
    if fab_hits:
        print("  ** RED FLAG: fabricated claim(s) verified **", file=sys.stderr)
        for claim_text, frame_idx, arm_label, cap, result in fab_hits:
            print(f"    claim={claim_text!r} frame_idx={frame_idx} arm={arm_label} caption={cap!r}")
            print(f"    best_sentence={result.best_sentence!r} score={result.best_score:.4f}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  positive query badges: {badge_tally['positive']}")
    print(f"  negative query badges: {badge_tally['negative']}")
    print(f"  fabricated regression verified: {len(fab_hits)} (expected 0)")
    print(f"  negative-query core-rejected findings: {len(stop_findings)} (expected 0)")
    print()

    hard_stop = False
    if stop_findings:
        hard_stop = True
        print("STOP: negative query core AbsenceClaim was REJECTED (event found where confirmed absent):",
              file=sys.stderr)
        for query, arm_label, verdict in stop_findings:
            print(f"    query={query!r} arm={arm_label} reason={verdict.reason}", file=sys.stderr)
    if fab_hits:
        hard_stop = True
        print("STOP: fabricated-claim regression found a verified fabricated claim.", file=sys.stderr)

    if hard_stop:
        sys.exit(1)
    print("All STOP conditions clear.")


if __name__ == "__main__":
    main()
