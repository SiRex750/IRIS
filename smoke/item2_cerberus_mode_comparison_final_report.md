# Item 2: cerberus_mode="legacy" vs "v2" Abstention Comparison

## Verifying v2 is functionally complete (task's explicit prerequisite)
Read `_query_v2` (`iris/query.py`) end-to-end before running anything: identical retrieval/
captioning to legacy, its own `AnswerClaims` JSON-contract generation with one corrective retry,
`iris.cerberus_layers.verify_answer` for verification, badge in
`{verified, partial/partially_verified, flagged, unverified}`. Confirmed via the live run:
**`compliance_failed=False` on 12/12 questions** — the JSON claim-contract path never failed to
parse in this sample, so it is not a half-implemented path silently degrading results.

## Method
Same 3 videos / 12 real questions, same cached indexes, `cerberus_mode="legacy"` then
`cerberus_mode="v2"`, both through the real `iris.query.query()` path. Does not change the
default `cerberus_mode` (still `"legacy"` in `iris_config.py`).

## Results (`smoke/cerberus_mode_comparison.json`)

| | legacy (default) | v2 |
|---|---|---|
| Abstention rate | 10/12 = 0.833 | 2/12 = 0.167 |
| Errors | 0 | 0 |

**Delta: -0.667** (v2 abstains on 5x fewer questions than legacy in this sample).
v2 badge distribution: `verified`: 10, `unverified`: 2, `partial`/`flagged`: 0 (neither appeared
in this sample — the two-state legacy comparison happened to have a clean analog here, though
`flagged`'s "legacy would fully abstain, v2 keeps the core answer with a caveat" case, called out
as the interesting nuance in the comparison script's docstring, did not get exercised by this
particular 12-question sample).

(Legacy's abstention rate here, 10/12, differs slightly from the original smoke run's 11/12 on
the same questions — expected run-to-run LLM variance given item 1's documented residual
nondeterminism, not a contradiction.)

## Interpretation

This is a large, consistent effect, not noise: legacy's all-or-nothing gate (`iris/query.py`
`wrapper_cerberus_gate`, `is_verified = len(rejected)==0 and len(unverifiable)==0` over ALL split
claims) is dramatically stricter than v2's core-claim-focused badge system in practice, on real
questions against real (often terse/generic) captions. This matches the task's hypothesis and the
mechanism identified from reading the code: one vague or unsupported clause anywhere in a
multi-sentence legacy answer kills the whole response, while v2 only requires the *core* claim to
be entailed.

## Recommendation

**Do not flip the default from this smoke sample alone** — 12 questions across 3 videos is not
enough to certify precision isn't being traded away for the abstention-rate improvement (v2 could
be verifying claims legacy would have correctly rejected as unsupported; a lower abstention rate
is only good news if the *newly non-abstained* answers are actually correct, which needs a real
accuracy check against gold answers, currently blocked on the same official-dataset access gap
noted throughout this repo's benchmark-setup work). What this comparison does establish cleanly:
v2 is functionally complete and reliable (0 compliance failures, 0 errors) and the abstention-rate
gap between the two modes is large enough to be worth resolving with a real accuracy-scored
comparison before deciding which mode should be the default.
