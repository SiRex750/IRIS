"""scripts/smoke_v2.py — Cerberus v2 switchover smoke test.

Runs virat_query_smoke.py's 3 smoke queries + demo_cctv_query.py's 8-query
polarity suite (4 POSITIVE + 4 NEGATIVE) through iris.query.query with
cerberus_mode="v2" -- the full v2 orchestration (ARIA AnswerClaims JSON
contract + strict parse/retry, iris.cerberus_layers router, answer badge).
Reuses those scripts' query lists and index-loading as-is; no new query
set, no mechanism change to legacy (cerberus_mode="legacy" is a completely
separate, untouched branch in iris.query.query).

Per query prints: raw answer, compliance (attempts / compliance_failed),
per-claim verdict table (claim type, label, reason), badge, and for every
claim verdict labeled "rejected": whether ANY sentence entailed and the
best-contradiction sentence text -- the within-frame spurious-rejection
watch metric (see iris.cerberus_layers.Layer2Result's any_entailed /
best_contradiction_sentence fields).

EXPECT: positive queries with correct answers no longer suppressed
(car/vehicles/person queries badge verified); negative queries reach
verified_absent where correct; compliance rate reported honestly whatever
it is.

VERIFY:
    python scripts/smoke_v2.py 2>&1 | tee smoke_v2.log
    python scripts/verify_layer2.py 2>&1 | tee layer2_regression.log   (unchanged; fabricated must stay 0)

STOP: compliance rate 0 after format=json attempts (surfaced below as a
nonzero exit -- a model-choice question, not something this script tunes
around).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.query as iris_query
from scripts.demo_cctv_query import CFG, NEGATIVE_QUERIES, POSITIVE_QUERIES, _load_index
from scripts.virat_query_smoke import QUERIES as SMOKE_QUERIES

CFG_V2 = dataclasses.replace(CFG, cerberus_mode="v2")


def _print_claim_table(claim_verdicts: list) -> None:
    hdr = f"  {'type':<10} | {'label':<16} | {'reason':<70}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for v in claim_verdicts:
        claim_type = type(v.claim).__name__
        reason_disp = (v.reason[:67] + "...") if len(v.reason) > 70 else v.reason
        print(f"  {claim_type:<10} | {v.label:<16} | {reason_disp:<70}")
    print()


def _print_rejection_watch(claim_verdicts: list) -> None:
    rejected = [v for v in claim_verdicts if v.label == "rejected"]
    if not rejected:
        return
    print("  -- WITHIN-FRAME SPURIOUS-REJECTION WATCH (rejected claims) --")
    for v in rejected:
        claim_desc = getattr(v.claim, "assertion", None) or getattr(v.claim, "event", None) or repr(v.claim)
        any_entailed = v.detail.get("any_entailed")
        best_contra_sentence = v.detail.get("best_contradiction_sentence") or v.detail.get("rejecting_sentence")
        print(f"    claim: {claim_desc!r}")
        print(f"      any_sentence_entailed: {any_entailed}")
        print(f"      best_contradiction_sentence: {best_contra_sentence!r}")
    print()


def run_query(idx, question: str, tag: str, tally: dict) -> None:
    print("=" * 100)
    print(f"[{tag}] QUERY: {question!r}")
    print("=" * 100)

    result = iris_query.query(question, idx, CFG_V2)

    print("-- RAW ANSWER --")
    print(result["raw_answer"])
    print()

    print(f"-- COMPLIANCE: attempts={result['n_llm_attempts']}  compliance_failed={result['compliance_failed']} --")
    print()

    tally["n_queries"] += 1
    if result["compliance_failed"]:
        tally["n_compliance_failed"] += 1
        print("-- BADGE: unverified (compliance_failed -- no claim verdicts) --")
        print()
        tally["badges"]["unverified"] += 1
        return

    print("-- PER-CLAIM VERDICTS --")
    _print_claim_table(result["claim_verdicts"])
    _print_rejection_watch(result["claim_verdicts"])

    print(f"-- BADGE: {result['badge']} --")
    print()

    tally["badges"][result["badge"]] += 1


def main() -> None:
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print(f"cerberus_mode={CFG_V2.cerberus_mode}")
    print()

    tally = {
        "n_queries": 0,
        "n_compliance_failed": 0,
        "badges": {"verified": 0, "flagged": 0, "partial": 0, "unverified": 0},
    }

    for question in SMOKE_QUERIES:
        run_query(idx, question, "SMOKE", tally)

    for question in POSITIVE_QUERIES:
        run_query(idx, question, "POSITIVE", tally)

    for question in NEGATIVE_QUERIES:
        run_query(idx, question, "NEGATIVE/ABSENCE", tally)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    n = tally["n_queries"]
    n_fail = tally["n_compliance_failed"]
    compliance_rate = (n - n_fail) / n if n else float("nan")
    print(f"  n queries: {n}")
    print(f"  compliance_failed: {n_fail}/{n}  (compliance rate = {compliance_rate:.4f})")
    print(f"  badges: {tally['badges']}")
    print()

    if compliance_rate == 0.0:
        print("STOP: compliance rate is 0 after format=json attempts. Surfacing -- this is a model/", file=sys.stderr)
        print("prompt-compliance question, not something to tune around here.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
