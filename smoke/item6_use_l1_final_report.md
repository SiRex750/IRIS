# Item 6: use_l1 A/B Comparison — Results

See `item6_use_l1_investigation.md` for why no blocking reason was found for `use_l1=False`
(no commit history, no code comment, zero test coverage, and the "novel contributions" writeup
assumes L1 is active). This file reports the actual comparison run before any default change.

## Method
Same 3 videos / 12 questions as the rest of this smoke suite, reusing cached indexes (no
re-ingest). `use_l1=False` (current default) vs `use_l1=True`, both through the real
`iris.query.query()` path, `cerberus_mode="legacy"`.

## Results (`smoke/use_l1_comparison.json`)

| | use_l1=False (current default) | use_l1=True |
|---|---|---|
| Abstention rate | 11/12 = 0.917 | 11/12 = 0.917 |
| Mean total latency | 52.48s | 51.32s |
| Errors | 0 | 0 |

**Retrieved frames differed in 12/12 questions** between the two modes — L1 substantively changes
*which* frames get retrieved (confirmed via `query_telemetry.l1_consulted=True`,
`l1_hit=True` in every `use_l1=True` row), it is not a no-op.

## Interpretation

L1 is not dormant-but-harmless — it actively changes retrieval on every question in this sample.
But in this 12-question sample, that change had **no measurable effect on the final abstention
rate** (both 11/12) and **no meaningful latency difference** (~1s, within run-to-run noise given
the answerer dominates total latency). This is consistent with either of two explanations that
this sample size cannot distinguish:
1. L1's different frame selection doesn't change downstream answer quality much on these
   particular questions (CerberusV's strict gate abstains regardless of which 5 frames are
   fed to it, given how sparse/generic the captions tend to be — see the broader smoke findings).
2. 12 questions is too small a sample to detect a real quality delta that would show up at
   official-dataset scale.

## Recommendation

**Do not flip the default based on this sample alone.** The result is genuinely ambiguous (real
behavioral change, no measurable quality/latency signal either way at n=12) rather than a clear
"L1 helps" or "L1 doesn't matter." Two concrete next steps, not done here:
- Run the same comparison at official-dataset scale with real Acc@GQA/IoP/IoU scoring (blocked on
  the still-unresolved official NExT-GQA dataset access, see the earlier paper-setup snapshot).
- Add the zero-test-coverage gap identified in the investigation: at minimum a smoke-level
  regression test exercising `use_l1=True` so future changes to L1 don't silently break the path
  that's already dormant.

No config default was changed by this investigation.
