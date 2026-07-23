# P-NOW-A — pre-declared val/test split (2026-07-22). DECLARED BEFORE ANY MEASUREMENT.
Pool: 526 grounded questions = gsub_val.json ∩ 86 cached videos (eval/data/nextqa/index_cache).
No new ingest. Split is VIDEO-LEVEL — no video appears in both halves (questions share videos, so
question-level splitting would leak).

VAL  = the 59 videos covered by A6_mixed_raw.json  -> 406 questions.  All tuning happens here.
TEST = the 27 remaining videos                      -> 120 questions.  Untouched by any prior tuning.

Rationale: half_width=2.2 and every prior grounding number were selected on 64 questions drawn from
the 59 val videos. Only the 27 untouched videos give a genuinely held-out estimate. Asymmetric ratio
is deliberate: tuning needs power, the reported number needs cleanliness.

Comparability (measured before declaring): family C/T 56/44 vs 62/38; gold median 4.90s vs 5.10s;
<=2.5s 17.0% vs 20.8%; video duration median 30s vs 27s. Mean gold-span differs (10.15 vs 7.47) due
to the val pool's long tail; medians match.

PRE-REGISTERED BIASES (declared now, do not re-interpret later):
- TEST has fewer multi-interval questions (6.7% vs 11.3%) -> slightly EASIER.
- TEST has more short golds (20.8% vs 17.0%) -> slightly HARDER.
- These partly offset; net direction unknown.

PRE-REGISTERED EXPECTATION: the held-out number will likely be LOWER than the in-sample 0.3426.
half_width=2.2 was selected in-sample on N=64. A drop is the number becoming real, NOT a regression.

PRECISION: TEST is 120 questions / 27 videos. Expect ~+/-0.08-0.11 CI on IoP@0.5 (video-clustered
bootstrap, 1000 resamples). Accepted as the cost of a clean estimate.

PROTOCOL:
1. All sweeps and parameter selection on VAL only.
2. Freeze every parameter.
3. ONE run on TEST. That number is reported. No re-tuning after seeing it. If TEST is run more than
   once, it is burned and must be labelled as such.
