# C2 — Caption truncation and quality audit (offline, from committed data)

Source: `tuning/blind_ablation/captions_dump.json` (SHA-256
`f4f4d3505e0fae1241f30b0f6539ada99c9d9718f4667b64befbf5f2b1ad328e`, protected,
unmodified by this task — verified in `protected_hashes_before/after.txt`).
Schema: `{video: {qid: {frame_idx: caption}}}`. This is the recorded blind
ablation run's real captioning output (arm A, question-blind), 112 videos /
639 questions. The captioner was **not** re-run for this audit; this is a
static read of already-committed data. Full numbers: `C2_caption_stats.json`.

## 1. Empty / whitespace-only captions

**0 / 2556 (0.0%)**. No caption in this dump is empty, whitespace-only, or
literally missing.

## 2. `[CAPTION_FAILED]` occurrences

**0 / 2556 (0.0%)**. No caption in this dump contains the `[CAPTION_FAILED]`
sentinel `iris/aria.py::MiniCPMCaptioner.caption()` returns on a second
truncation.

**Read carefully**: this does NOT mean truncation never happens in general —
it means it did not happen anywhere in *this specific 639-question recorded
run*. The 0% empty-caption rate here is good news for that run specifically,
but the underlying risk (silent `[CAPTION_FAILED]` on any future run) is real
and exactly why `get_minicpm_truncation_stats()` is now wired into the e2e
harness going forward (see below) — this audit alone cannot rule out
truncation on the 990-video official test run, which is roughly 8.8x the
video count of this sample.

## 3. Caption length distribution and truncation-risk proximity

| | chars | tokens (≈chars/4) |
|---|---|---|
| min | 80 | 20.0 |
| p25 | 199 | 49.75 |
| median | 238 | 59.5 |
| p75 | 281 | 70.25 |
| p95 | 361 | 90.25 |
| max | 569 | 142.25 |

The `num_predict` ceilings are 400 (first attempt) and 700 (retry) tokens.
The longest caption observed anywhere in this dump is ~142 tokens — **under
36% of the first-attempt ceiling**. Zero captions fall within 50 tokens of
either ceiling. There is no truncation-risk signal in this population: every
caption that was generated completed well within budget.

This is consistent with (2)'s 0% `[CAPTION_FAILED]` rate — the two numbers
corroborate each other rather than being independent claims.

## 4. Per-video caption cache behavior

1665 distinct `(video, frame_idx)` pairs appear across the 2556 total caption
entries; 535 of those pairs are referenced by more than one question (i.e.
the same frame was retrieved for >1 question in the same video). Of those 535
repeat-use pairs, **534 (99.8%) reused byte-identical caption text** across
every qid that touched them — confirming the per-video caption cache
(`FrameRecord.caption`, keyed by `frame_idx`) is working as designed within a
single run.

One outlier: video `8254300526`, frame `869` — qid 3 and qid 8 got an
identical caption, but qid 7 got a differently-worded (same-content,
paraphrased) caption for the exact same frame. This is a single instance out
of 535 (0.2%), most consistent with a cache miss or re-caption race rather
than the caching mechanism itself being broken.

Cost implication: question-aware captioning (captioning per-question instead
of per-frame) would multiply captioning calls from 1665 (today, one per
distinct frame) up toward 2556 (one per frame *per question that retrieves
it*) — roughly a 1.5x increase in captioner calls for this val_confirm
population, scaling up further at official-test scale in proportion to how
much frame reuse the official 990-video/5553-question set has.

## 5. Qualitative read — 10 randomly sampled captions (seed=42)

Sampled captions (full text in `C2_caption_stats.json`) are itemized,
generic scene descriptions: "A person riding a bicycle on a road... Trees
lining the side of the path... Pole with blue and white markings..." None of
the 10 samples reference the question being asked, the multiple-choice
options, or any specific temporal/causal relation — they read as what a
dense-captioning model would produce for any viewer of that frame, not
targeted evidence for a specific gold answer. This matches the module
docstring in `scripts/blind_ablation_eval.py` describing this run's
captioning as deliberately "question-blind."

This is directly relevant to the standing open finding in project memory
(`tuning/family2-tuning-state`) that answering conditional on correct
grounding (IoP≥0.5) only lifts accuracy 61.1% vs 51.7% unconditional — a
+9.4pp visual lift that is smaller than it should be if captions carried
strong question-relevant detail. Generic scene captions are a plausible
contributor to that gap, though this audit cannot itself attribute causality
(no question-aware caption arm exists in this dump to compare against).

## Wiring `get_minicpm_truncation_stats()` into the harness (code change)

`iris/aria.py::get_minicpm_truncation_stats()` existed but was never called by
any eval script — the truncation counters were invisible in every prior
report. `scripts/val_confirm_e2e_eval.py::main()` now calls it once at the end
of every run and includes the result under `metrics["minicpm_truncation_stats"]`
in the printed `VAL_CONFIRM_E2E_METRICS_JSON=` line, so truncation rate is
visible going forward without needing a separate audit like this one.
Test: `tests/test_c2_truncation_stats_wiring.py` (seeds the module-level
counters, mocks the rest of the pipeline, asserts the final metrics dict
carries them through with the correct truncation rate). Confirmed to fail
before this wiring (`AttributeError` — the pre-fix harness has no such call)
and pass after.

## Bottom line

- Empty-caption rate in the recorded run: **0%** — prior Acc@QA/Acc@GQA
  numbers for this specific 639-question run were **not** computed with any
  blank/`[CAPTION_FAILED]` evidence.
- Truncation-risk: **none observed** — max caption length is ~36% of the
  first-attempt ceiling.
- This is a retrospective finding about one already-recorded run, not a
  guarantee for the 990-video official test run, which has not yet been
  audited (and cannot be, since it hasn't run) — that's what the newly-wired
  counters are for.
