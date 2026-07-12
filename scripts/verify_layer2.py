"""Offline verification of Cerberus layer 2 (frame-bound, sentence-scoped
visual NLI) against frozen artifacts (diag_v2_capture.jsonl and
diag_v4_moondream_captions_pass1.jsonl). No live pipeline / ARIA / retrieval
calls -- purely replays already-captured claims and captions through
iris.cerberus_layers.verify_visual_claim.

Fixtures:
  (a) CANONICAL RECOVERY: VisualClaim(frame_idx=25069, "there is a car
      parked in the parking lot", is_core=True) against that frame's
      BLIP caption_only. EXPECT verified, score ~0.97.
  (b) NORMALIZED KERNELS: real claims are citation-resolved and kernel-
      normalized exactly as scripts/diag_v3_visual_scoping.py's
      --kernel-normalize mode (_resolve_citations + _decompose_claim +
      _normalize_kernel, reused verbatim -- not re-derived here), then each
      normalized kernel is bound to its cited frame's caption_only and run
      through verify_visual_claim. --kernel-normalize established 4 such
      kernels entail; this fixture regenerates them at run time (not
      hardcoded) and checks layer 2 recovers exactly the same 4.
  (c) FABRICATED REGRESSION: all 6 FABRICATED_CLAIMS (they carry no frame
      citation), bound EXHAUSTIVELY to every frame in every query's
      retrieved pool, against BLIP caption_only. STANDING METRIC: EXPECT 0
      verified across all bindings.
  (d) MOONDREAM ARM: (a)-(c) repeated with captions from
      diag_v4_moondream_captions_pass1.jsonl (multi-sentence premises --
      this is what actually exercises sentence-scoping, since BLIP captions
      are single-sentence). Fabricated regression must still be 0. The
      normalized-kernel entailment count is reported against diag_v4
      --split-premise's own bound-NLI-replay entailment count as a
      consistency check on layer 2's per-pair semantics -- divergence is
      printed, not reconciled.

AbsenceClaims are OUT OF SCOPE: they route to layer 3 (not built), so no
absence claim is run through layer 2 here.

VERIFY:
    python scripts/verify_layer2.py 2>&1 | tee layer2_verify.log

STOP conditions (nonzero exit):
  - any fabricated claim verifies (arm (c) or (d))
  - the per-pair parity test (tests/test_cerberus_layers.py) is not assumed
    passing by this script -- run pytest separately; this script only
    re-affirms the fabricated-regression invariant it depends on.
  - layer-2 Moondream normalized-kernel entailment count differs from
    diag_v4 --split-premise's own bound-NLI-replay count by more than +/-1
    (surfaced as a FATAL, not tuned around).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris.cerberus_layers import Layer2Result, get_nli_gate, score_nli_pair, verify_visual_claim
from iris.claim_contract import VisualClaim
from scripts.diag_v2_scoping_separation import (
    CAPTURE_JSONL,
    FABRICATED_CLAIMS,
    _group_by_claim,
    _load_capture,
)
from scripts.diag_v3_visual_scoping import (
    _classify_claim,
    _decompose_claim,
    _fact_header,
    _normalize_kernel,
    _resolve_citations,
)

PASS1_JSONL = REPO / "diag_v4_moondream_captions_pass1.jsonl"

# Established by --kernel-normalize (scripts/diag_v3_visual_scoping.py):
# originally 4 normalized kernels entailed against BLIP captions, vs 0 under
# raw citation-binding. Revised to 3 after the entailment-floor fix
# (scripts/diag_l3_judge.py A2 / iris/cerberus_layers.py score_nli_pair):
# the frame=18708 kernel scored 0.8009, below the 0.85 floor the code always
# documented but (per a dead-code conjunct) never actually applied to
# non-negation pairs -- that pair now correctly demotes to unverifiable.
# This script's expected recovery count for arm (b).
EXPECTED_NORMALIZED_ENTAILMENTS = 3

# Established by `python scripts/diag_v4_moondream_captions.py --split-premise`
# (pass 1): "entailments under split-premise bound-NLI replay: 8". Layer 2's
# arm (d.b) count is compared against this, not re-derived -- that script
# remains the source of truth for its own number.
DIAG_V4_SPLIT_PREMISE_BASELINE = 8
DIAG_V4_SPLIT_PREMISE_TOLERANCE = 1

CANONICAL_FRAME_IDX = 25069
CANONICAL_ASSERTION = "there is a car parked in the parking lot"
CANONICAL_EXPECTED_SCORE = 0.97
CANONICAL_SCORE_TOLERANCE = 0.03


def _load_pass1() -> list[dict]:
    if not PASS1_JSONL.exists():
        print(f"FATAL: {PASS1_JSONL} missing", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(PASS1_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _print_table(rows: list[tuple]) -> None:
    hdr = (f"  {'assertion':<55} | {'frame':>7} | {'verdict':<12} | "
           f"{'best_sentence':<50} | {'score':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for assertion, frame_idx, verdict, best_sentence, score in rows:
        a_disp = (assertion[:52] + "...") if len(assertion) > 55 else assertion
        s_disp = best_sentence or ""
        s_disp = (s_disp[:47] + "...") if len(s_disp) > 50 else s_disp
        print(f"  {a_disp:<55} | {frame_idx:>7} | {verdict:<12} | {s_disp:<50} | {score:>6.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# (b) regenerate the normalized-kernel fixtures (BLIP arm)
# ─────────────────────────────────────────────────────────────────────────────

def _regenerate_normalized_kernels(gate, records: list[dict]) -> list[tuple[str, int, str]]:
    """Returns [(normalized_kernel, cited_frame_idx, blip_caption_only), ...]
    for every citable real claim -- one row per claim, bound to whichever
    cited frame scores highest (argmax entailment_score over cited rows),
    exactly matching --kernel-normalize's own replay/argmax selection (most
    claims have exactly one citation; 2 of 16 in this capture have two).
    Does NOT pre-filter by entailment; callers score each row again through
    verify_visual_claim and observe which entail."""
    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    out = []
    for key, rows in real_groups.items():
        claim_text = key[1]
        cited_rows = _resolve_citations(claim_text, rows)
        if not cited_rows:
            continue
        stripped, _fields, is_mangled = _decompose_claim(claim_text)
        if is_mangled:
            continue
        normalized = _normalize_kernel(stripped)

        scored = []
        for row in cited_rows:
            label, score = score_nli_pair(gate, normalized, row["caption_only"])
            scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        top_row = scored[0][1]

        frame_idx, _ts = _fact_header(top_row["fact_text"])
        out.append((normalized, frame_idx, top_row["caption_only"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# (c) fabricated regression -- exhaustive binding to every frame in every
# query's pool
# ─────────────────────────────────────────────────────────────────────────────

def _query_pools(records: list[dict]) -> dict[str, list[tuple[int, str]]]:
    """query -> distinct [(frame_idx, caption_only), ...] from that query's
    retrieved fact pool (BLIP arm)."""
    pools: dict[str, dict[int, str]] = {}
    for r in records:
        if r["fact_text"] is None:
            continue
        frame_idx, _ts = _fact_header(r["fact_text"])
        pools.setdefault(r["query"], {})[frame_idx] = r["caption_only"]
    return {q: sorted(fm.items()) for q, fm in pools.items()}


def run_fabricated_regression(gate, pools: dict[str, list[tuple[int, str]]], caption_label: str) -> list[tuple]:
    print("=" * 100)
    print(f"(c) FABRICATED REGRESSION -- {caption_label}")
    print("=" * 100)

    verified_hits = []
    table_rows = []
    n_pairs = 0
    for claim_text in FABRICATED_CLAIMS:
        for query, frame_caps in pools.items():
            for frame_idx, caption in frame_caps:
                n_pairs += 1
                claim = VisualClaim(frame_idx=frame_idx, assertion=claim_text)
                result = verify_visual_claim(claim, caption, gate)
                if result.verdict == "verified":
                    verified_hits.append((claim_text, query, frame_idx, caption, result))
                    table_rows.append((claim_text, frame_idx, result.verdict, result.best_sentence, result.best_score))

    print(f"  n (fabricated claim, frame) bindings scored: {n_pairs}")
    print(f"  verified: {len(verified_hits)}")
    print()

    if verified_hits:
        print("  ** RED FLAG: fabricated claim(s) verified by layer 2 **", file=sys.stderr)
        print("  ** RED FLAG: fabricated claim(s) verified by layer 2 **")
        _print_table(table_rows)
        for claim_text, query, frame_idx, caption, result in verified_hits:
            print(f"    claim={claim_text!r}")
            print(f"    query={query!r} frame_idx={frame_idx}")
            print(f"    caption={caption!r}")
            print(f"    best_sentence={result.best_sentence!r} score={result.best_score:.4f}")
            print()

    return verified_hits


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: {CAPTURE_JSONL} missing", file=sys.stderr)
        sys.exit(1)

    records = _load_capture()
    pass1_rows = _load_pass1()
    moondream_by_frame = {r["frame_idx"]: r["moondream_caption"] for r in pass1_rows}

    gate = get_nli_gate()

    all_hard_failures: list[str] = []

    # ── (a) CANONICAL RECOVERY, BLIP arm ────────────────────────────────────
    print("=" * 100)
    print("(a) CANONICAL RECOVERY -- BLIP caption_only")
    print("=" * 100)

    blip_caption_25069 = None
    for r in records:
        if r["fact_text"] is None:
            continue
        frame_idx, _ts = _fact_header(r["fact_text"])
        if frame_idx == CANONICAL_FRAME_IDX:
            blip_caption_25069 = r["caption_only"]
            break
    if blip_caption_25069 is None:
        print(f"FATAL: no BLIP caption_only found for frame_idx={CANONICAL_FRAME_IDX}", file=sys.stderr)
        sys.exit(1)

    canonical_claim = VisualClaim(frame_idx=CANONICAL_FRAME_IDX, assertion=CANONICAL_ASSERTION, is_core=True)
    canonical_result = verify_visual_claim(canonical_claim, blip_caption_25069, gate)
    _print_table([(CANONICAL_ASSERTION, CANONICAL_FRAME_IDX, canonical_result.verdict,
                   canonical_result.best_sentence, canonical_result.best_score)])

    if canonical_result.verdict != "verified":
        all_hard_failures.append(
            f"(a) canonical recovery FAILED: verdict={canonical_result.verdict} (expected verified)")
    elif abs(canonical_result.best_score - CANONICAL_EXPECTED_SCORE) > CANONICAL_SCORE_TOLERANCE:
        print(f"  ** WARNING: score {canonical_result.best_score:.4f} outside expected "
              f"~{CANONICAL_EXPECTED_SCORE} +/- {CANONICAL_SCORE_TOLERANCE} band **")
    print()

    # ── (b) NORMALIZED KERNELS, BLIP arm ────────────────────────────────────
    print("=" * 100)
    print("(b) NORMALIZED KERNELS -- BLIP caption_only")
    print("=" * 100)

    kernel_fixtures = _regenerate_normalized_kernels(gate, records)
    print(f"  n citable, unmangled normalized kernels: {len(kernel_fixtures)}")
    print()

    kernel_table = []
    kernel_entailed = []
    for normalized, frame_idx, caption in kernel_fixtures:
        claim = VisualClaim(frame_idx=frame_idx, assertion=normalized)
        result = verify_visual_claim(claim, caption, gate)
        kernel_table.append((normalized, frame_idx, result.verdict, result.best_sentence, result.best_score))
        if result.verdict == "verified":
            kernel_entailed.append((normalized, frame_idx, result))

    _print_table(kernel_table)
    print(f"  normalized kernels verified: {len(kernel_entailed)} (expected {EXPECTED_NORMALIZED_ENTAILMENTS})")
    if len(kernel_entailed) != EXPECTED_NORMALIZED_ENTAILMENTS:
        print(f"  ** WARNING: recovered count differs from --kernel-normalize's established "
              f"{EXPECTED_NORMALIZED_ENTAILMENTS} **")
    print()

    # ── (c) FABRICATED REGRESSION, BLIP arm ─────────────────────────────────
    pools = _query_pools(records)
    blip_hits = run_fabricated_regression(gate, pools, "BLIP caption_only")
    if blip_hits:
        all_hard_failures.append(f"(c) BLIP arm: {len(blip_hits)} fabricated claim(s) verified")

    # ── (d) MOONDREAM ARM: (a)-(c) repeated ─────────────────────────────────
    print("=" * 100)
    print("(d) MOONDREAM ARM")
    print("=" * 100)

    # (a) canonical recovery, moondream caption
    print("-" * 100)
    print("(d.a) CANONICAL RECOVERY -- Moondream caption")
    print("-" * 100)
    moon_caption_25069 = moondream_by_frame.get(CANONICAL_FRAME_IDX)
    if moon_caption_25069 is None:
        print(f"FATAL: no Moondream caption for frame_idx={CANONICAL_FRAME_IDX}", file=sys.stderr)
        sys.exit(1)
    moon_canonical_result = verify_visual_claim(canonical_claim, moon_caption_25069, gate)
    _print_table([(CANONICAL_ASSERTION, CANONICAL_FRAME_IDX, moon_canonical_result.verdict,
                   moon_canonical_result.best_sentence, moon_canonical_result.best_score)])
    print(f"  n sentences in Moondream premise: {moon_canonical_result.n_sentences}  "
          f"(BLIP premise: 1 -- this is the sentence-scoping exercise)")
    if moon_canonical_result.oversize_sentences:
        print(f"  oversize sentences (>60 tokens): {len(moon_canonical_result.oversize_sentences)}")
    print()

    # (b) normalized kernels, moondream captions
    print("-" * 100)
    print("(d.b) NORMALIZED KERNELS -- Moondream captions")
    print("-" * 100)
    moon_kernel_table = []
    moon_kernel_entailed = []
    missing_moon_frames = []
    for normalized, frame_idx, _blip_caption in kernel_fixtures:
        moon_caption = moondream_by_frame.get(frame_idx)
        if moon_caption is None:
            missing_moon_frames.append(frame_idx)
            continue
        claim = VisualClaim(frame_idx=frame_idx, assertion=normalized)
        result = verify_visual_claim(claim, moon_caption, gate)
        moon_kernel_table.append((normalized, frame_idx, result.verdict, result.best_sentence, result.best_score))
        if result.verdict == "verified":
            moon_kernel_entailed.append((normalized, frame_idx, result))

    if missing_moon_frames:
        print(f"  NOTE: {len(missing_moon_frames)} cited frame_idx have no Moondream caption "
              f"(not in the 55-frame pass1 sample): {missing_moon_frames}")
    _print_table(moon_kernel_table)
    print(f"  normalized kernels verified (Moondream): {len(moon_kernel_entailed)}")
    print()

    # (c) fabricated regression, moondream captions -- bound to every distinct
    # Moondream frame (the pass1 sample is not query-scoped; matches
    # diag_v4's own fabricated-probe binding, which iterates all v4_rows).
    print("-" * 100)
    print("(d.c) FABRICATED REGRESSION -- Moondream captions")
    print("-" * 100)
    moon_verified = []
    moon_table = []
    n_pairs = 0
    for claim_text in FABRICATED_CLAIMS:
        for row in pass1_rows:
            n_pairs += 1
            claim = VisualClaim(frame_idx=row["frame_idx"], assertion=claim_text)
            result = verify_visual_claim(claim, row["moondream_caption"], gate)
            if result.verdict == "verified":
                moon_verified.append((claim_text, row["frame_idx"], row["moondream_caption"], result))
                moon_table.append((claim_text, row["frame_idx"], result.verdict, result.best_sentence, result.best_score))

    print(f"  n (fabricated claim, frame) bindings scored: {n_pairs}")
    print(f"  verified: {len(moon_verified)}")
    if moon_verified:
        print("  ** RED FLAG: fabricated claim(s) verified by layer 2 (Moondream arm) **", file=sys.stderr)
        print("  ** RED FLAG: fabricated claim(s) verified by layer 2 (Moondream arm) **")
        _print_table(moon_table)
        all_hard_failures.append(f"(d.c) Moondream arm: {len(moon_verified)} fabricated claim(s) verified")
    print()

    # ── consistency check vs diag_v4 --split-premise ────────────────────────
    print("=" * 100)
    print("CONSISTENCY CHECK: layer-2 Moondream normalized-kernel entailments "
          "vs diag_v4 --split-premise bound-NLI-replay entailment count")
    print("=" * 100)
    diff = len(moon_kernel_entailed) - DIAG_V4_SPLIT_PREMISE_BASELINE
    print(f"  layer 2 (this script), Moondream normalized-kernel entailments: {len(moon_kernel_entailed)}")
    print(f"  diag_v4 --split-premise (pass 1) bound-NLI-replay entailments:  {DIAG_V4_SPLIT_PREMISE_BASELINE}")
    print(f"  diff: {diff:+d} (tolerance: +/-{DIAG_V4_SPLIT_PREMISE_TOLERANCE})")
    if abs(diff) > DIAG_V4_SPLIT_PREMISE_TOLERANCE:
        print("  ** DIVERGENCE: per-pair semantics differ from diag_v4's -- surfacing, not tuning. **")
        all_hard_failures.append(
            f"consistency check: layer-2 Moondream entailments ({len(moon_kernel_entailed)}) diverge from "
            f"diag_v4 --split-premise baseline ({DIAG_V4_SPLIT_PREMISE_BASELINE}) by {diff:+d} "
            f"(tolerance +/-{DIAG_V4_SPLIT_PREMISE_TOLERANCE})")
    else:
        print("  within tolerance.")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  (a) canonical recovery (BLIP):       {canonical_result.verdict} (score={canonical_result.best_score:.4f})")
    print(f"  (a) canonical recovery (Moondream):  {moon_canonical_result.verdict} (score={moon_canonical_result.best_score:.4f})")
    print(f"  (b) normalized kernels verified (BLIP):      {len(kernel_entailed)}/{len(kernel_fixtures)} "
          f"(expected {EXPECTED_NORMALIZED_ENTAILMENTS})")
    print(f"  (b) normalized kernels verified (Moondream): {len(moon_kernel_entailed)}/{len(kernel_fixtures)}")
    print(f"  (c) fabricated verified (BLIP):      {len(blip_hits)} (expected 0)")
    print(f"  (d.c) fabricated verified (Moondream): {len(moon_verified)} (expected 0)")
    print()

    if all_hard_failures:
        print("STOP conditions triggered:", file=sys.stderr)
        for msg in all_hard_failures:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1)

    print("All STOP conditions clear.")


if __name__ == "__main__":
    main()
