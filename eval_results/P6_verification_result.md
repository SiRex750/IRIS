# P6 — Verification (Cerberus v2) — NEGATIVE RESULT (2026-07-22)
Scope: cerberus_v.py as it exists on branch siddanth/peak-source-a6-p1. origin/main carries a
divergent cerberus v2 line that was NOT tested here.
Discrimination check, n=21 via iris.query.query(cerberus_mode="v2"), seated defaults, read-only.
Strata: A grounded (n=8), B ungrounded (n=8), C constructed-absence (n=5); grounding labels from A6.

FINDINGS
1. Badge does NOT discriminate grounding: mean best_score A=0.867 vs B=0.846. An ungrounded answer
   describes the wrong frames faithfully and badges "verified". Verification cannot detect retrieval
   error — the dominant failure mode (A6: 61% hard-miss).
2. AbsenceClaim fired 0/21, including all 5 constructed-absence cases.
3. Badge tracks CLAIM SHAPE, not truth: single tight VisualClaim -> verified ~0.98; 5 claims or a
   compound interpretive claim -> unverified ~0.001.
4. FALSE-REJECT of correct grounded answers: a true, grounded answer scored unverifiable at 0.034
   because the claim bundled a visual fact with an interpretive tail no caption sentence entails.
5. Probe-design correction: the 3 "rubber-stamped" absence cases were NOT fabrications — the model
   declined the absent-entity presupposition and described the real scene. The verifier was correct;
   the probe was flawed. Recorded so the error is not re-derived.

ROOT CAUSE: CerberusV all-or-nothing gate — is_verified = len(rejected)==0 and len(unverifiable)==0.
One interpretive or surplus claim flips the whole answer. Independently corroborated by the team's
own open-issues list ("Cerberus over-rejects correct answers", parked via cerberus_mode="none").

CONCLUSION: calibrated-abstention is NOT supported by the current machinery. Do NOT build a
risk-coverage harness on this badge and do NOT tune thresholds to manufacture separation.
Smallest fix worth trying: score the CORE claim only. Even a perfect faithfulness gate cannot catch
retrieval error, so the realistic scope is fabrication/abstention, never correctness.
