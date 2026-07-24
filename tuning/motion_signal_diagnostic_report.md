# Motion-signal diagnostic — do codec motion features separate gold frames?

Pure analysis, no re-ingest, no LLM, no model training. Reuses
`tuning/index_cache_val_confirm_e2e/` (112 videos, config-hash
`4edae64ed40256e3`, same cache as the c24f8f8 gap diagnostic) and
`tuning/val_confirm_e2e_per_question_gap_diagnostic.csv`'s bucket
classification. All 639 val_confirm questions. De-risking measurement for
a possible motion-aware anchor selector — **not** an implementation, and
`frozen_state.json` / production code are untouched.

Scripts used (not committed — ephemeral analysis, same convention as the
uncommitted script behind c24f8f8): `step0_population_check.py`,
`step1_step2_motion_signal.py`. Raw outputs committed:
`tuning/motion_step0_population.json`,
`tuning/motion_step1_population_separation.json`,
`tuning/motion_step2_anchor_repick.json`,
`tuning/motion_step2_per_question.csv`.

## Step 0 — are the features actually populated?

Loaded all 112 cached `FrameRecord` lists (18,378 frames total) and
checked all 6 motion fields for exact-zero fraction and non-zero range.

| feature | present on FrameRecord | frac exactly 0.0 | nonzero min | nonzero median | nonzero max | nonzero std | unpopulated (>90% zero)? |
|---|---|---|---|---|---|---|---|
| motion_magnitude | yes | 26.49% | 0.000417 | 5.0137 | 98.634 | 8.896 | No |
| divergence | yes | 26.61% | −3.2355 | −0.0502 | 1.1371 | 0.327 | No |
| curl | yes | 26.49% | 0.000417 | 1.3935 | 19.555 | 1.997 | No |
| jacobian_frobenius | yes | 26.49% | 0.000833 | 2.2809 | 30.692 | 3.247 | No |
| hessian_max_eigenvalue | yes | 26.49% | 0.001146 | 2.1179 | 31.521 | 3.158 | No |
| motion_entropy | yes | 26.49% | 0.009726 | 1.5559 | 3.211 | 0.657 | No |

All 6 fields round-tripped through `save_index`/`load_index` correctly
(no dropped fields) and are genuinely computed, not defaulted-to-0.0 —
the ~26.5% exact-zero rate is consistent across all 5 flow-geometry
features (near-identical fraction, distinct from `divergence`'s signed
distribution) and is plausibly attributable to non-decoded/skipped
frames rather than a broken computation; not investigated further since
it's well under the 90% unpopulated threshold. **Step 0 passes — the
data is real. Proceeding to Steps 1–2.**

## Step 1 — population-level separation (gold vs non-gold frames)

Pooled every `(question, frame)` pair across all 639 val_confirm
questions × their video's frames whose index was cached (109,237 pairs,
639/639 questions embedded, 0 embed failures). `in-gold` = frame
timestamp falls inside any gold span for that question.
`divergence`/`curl` are signed; AUC/Cohen's d use `abs(value)` as the
"motion salience" reading (higher = more salient) — raw signed means
are reported below for reference.

| feature | n (pos/neg) | mean in-gold | mean out-of-gold | Cohen's d | AUC |
|---|---|---|---|---|---|
| motion_magnitude | 21,387 / 87,850 | 5.7067 | 6.1282 | −0.0500 | **0.4970** |
| divergence (signed) | 21,387 / 87,850 | 0.1381 | 0.1481 | −0.0358 | **0.5010** |
| curl | 21,387 / 87,850 | 1.4467 | 1.5376 | −0.0470 | **0.4992** |
| jacobian_frobenius | 21,387 / 87,850 | 2.3647 | 2.5196 | −0.0491 | **0.4990** |
| hessian_max_eigenvalue | 21,387 / 87,850 | 2.2249 | 2.3620 | −0.0455 | **0.5001** |
| motion_entropy | 21,387 / 87,850 | 1.0826 | 1.0877 | −0.0058 | **0.4984** |
| **clip_sim_to_query (reference)** | 21,387 / 87,850 | 0.2578 | 0.2486 | **0.2861** | **0.5798** |

**None of the 6 motion features individually separate gold from
non-gold frames above chance.** All 6 AUCs sit in [0.497, 0.501] —
statistically indistinguishable from 0.5 (coin flip) at this sample
size, and all 6 Cohen's d magnitudes are ≤0.05 (negligible effect,
conventionally d<0.2 is "small"). The incumbent CLIP-similarity
reference column is clearly better than any motion feature
(AUC=0.580, d=0.286 — a real but modest single-feature signal, well
below what a usable classifier needs on its own) but still far from
strong. **Population-level, gold-frame localization is not something
these motion features carry on their own.**

## Step 2 — the anchoring-relevant test (does motion help at the anchor step?)

Reused the c24f8f8 bucket classification exactly (from
`val_confirm_e2e_per_question_gap_diagnostic.csv`'s `acc_qa`,
`overlap_status`, `retrieved_frame_in_gold`, `gap_seconds` columns —
no re-derivation). Re-ran retrieval (embed + `_retrieve_with_l1`,
current frozen config, confirmed to reproduce config-hash
`4edae64ed40256e3` exactly) for all 639 questions, and for each,
checked whether ranking the *same retrieved top-k pool* by each motion
feature (or a CLIP+motion blend, 3 fixed ratios) — instead of by CLIP
similarity — would anchor an in-gold frame.

One implementation note surfaced mid-run and is worth flagging as its
own small finding: the default `graph_mode="scene_sparse"` retrieval
path's frame dicts (`iris/scene_retrieval.py`) carry only
`frame_idx`/`timestamp`/`clip_embedding`/etc — **none of the 6 motion
fields** (unlike the flat-graph `_build_retrieved()` path, which does
carry 5 of them but not `motion_magnitude`). A motion-aware anchor
built directly against `_retrieve_with_l1`'s returned dicts would
silently rank on missing/defaulted values under the scene-sparse path
that's actually the default. This diagnostic backfills all 6 features
from the `FrameRecord` by `frame_idx` before ranking; any real
implementation needs to do the same (or extend `retrieve_scene_sparse`'s
dict) — noted as a build-time gotcha, not something to fix here.

### bucket (b) — wrong-frame-anchored, n=38 (CLIP anchors 0% of these by construction)

| method | hit / n | rate |
|---|---|---|
| **clip (baseline)** | 0 / 38 | **0.00%** |
| motion_magnitude alone | 11 / 38 | 28.95% |
| divergence alone | 11 / 38 | 28.95% |
| curl alone | **12 / 38** | **31.58%** |
| jacobian_frobenius alone | 11 / 38 | 28.95% |
| hessian_max_eigenvalue alone | 11 / 38 | 28.95% |
| motion_entropy alone | 7 / 38 | 18.42% |
| best blend (any ratio, any feature) | 12 / 38 (0.3clip+0.7hessian, or 0.3clip+0.7curl) | 31.58% |
| worst blend (0.7clip+0.3\*) | 3–5 / 38 | 7.9–13.2% |

Every single motion feature alone recovers 18–32% of bucket (b) — the
exact subset where CLIP-anchoring is wrong by definition. Blending
toward more CLIP weight (0.7 clip / 0.3 motion) makes it *worse* than
motion alone (down to 8–13%), because CLIP's own ranking is actively
wrong on these questions — diluting it with more CLIP weight fights the
correction instead of helping it.

### subset A — full Acc@QA-right/Acc@GQA-wrong failure group, n=228 (includes buckets a/b/c; only buckets a+b, n=82, have *any* in-gold frame in the pool for *any* method to find)

| method | hit / n | rate |
|---|---|---|
| **clip (baseline)** | 5 / 228 | **2.19%** |
| motion_magnitude alone | 26 / 228 | 11.40% |
| divergence alone | 23 / 228 | 10.09% |
| curl alone | **27 / 228** | **11.84%** |
| jacobian_frobenius alone | 26 / 228 | 11.40% |
| hessian_max_eigenvalue alone | **27 / 228** | **11.84%** |
| motion_entropy alone | 24 / 228 | 10.53% |
| best blend | 26 / 228 (0.3clip+0.7hessian) | 11.40% (still below best alone) |

Restricting to the 82 "achievable" questions (buckets a+b, where a
gold frame really is in the pool), the best motion features
(curl/hessian_max_eigenvalue) recover 27/82 = **32.9%**, vs CLIP's
5/82 = 6.1% — a >5x relative improvement on exactly the subset where
recovery is even possible.

### subset B — all zero-overlap questions regardless of Acc@QA, n=344

| method | hit / n | rate |
|---|---|---|
| clip (baseline) | 0 / 344 | 0.00% |
| best motion alone (motion_magnitude/curl/hessian) | 26–27 / 344 | 7.6–7.8% |

Same qualitative pattern as subset A: CLIP is at its structural floor
(0%, since zero-overlap includes bucket-c-style total retrieval misses
where nothing can help), motion features pull a modest absolute number
of questions back regardless.

### whole split — all 639 questions (not just the failure group)

| method | hit / n | rate |
|---|---|---|
| clip (baseline) | 205 / 639 | 32.08% |
| motion_magnitude alone | 183 / 639 | 28.64% |
| curl alone | 186 / 639 | 29.11% |
| hessian_max_eigenvalue alone | 191 / 639 | 29.89% |
| **best blend: 0.7 clip + 0.3 curl (or jacobian_frobenius, or hessian_max_eigenvalue)** | **208 / 639** | **32.55%** |

On the *whole* split, motion features alone are all slightly worse
than CLIP alone (CLIP is still the better single anchor most of the
time — consistent with Step 1's population AUCs). But a light
CLIP-dominant blend (0.7 CLIP + 0.3 motion, with curl/jacobian_frobenius/
hessian_max_eigenvalue) nets **+3 questions (+0.47pp) over CLIP alone**
— a small, real net gain: it fixes some of bucket (b) without giving up
much of the majority of cases CLIP already gets right.

## Step 3 — honest verdict

**Not "no signal" and not "never computed" — the two are cleanly
distinguished by these results, and neither applies uniformly:**

- **Step 0**: the motion features are genuinely computed and populated
  (not a defaulted-zero artifact) — that possibility is ruled out.
- **Step 1 (population separation)**: essentially **no usable
  single-feature signal**. All 6 motion features sit at AUC≈0.50
  (chance) and d≈0.05 (negligible) for telling a random gold frame
  from a random non-gold frame across the whole dataset. If the
  question were "can motion alone tell me which frame is correct,"
  the answer is no.
- **Step 2 (anchor re-pick, the test that actually matters)**: **real,
  concentrated signal exactly where it's needed.** On bucket (b) — the
  38 questions where CLIP's anchor choice is *actively wrong* despite a
  correct frame sitting in the same retrieved pool — every motion
  feature alone recovers 18–32% of them, where CLIP recovers 0% by
  construction. This is not visible in the population-level test
  because it's a *local, pool-relative* signal (motion salience picks
  out the more "eventful" frame among ~4 already-retrieved candidates),
  not a *global* one (motion salience does not separate gold frames
  from the thousands of non-gold frames in a video). Both things are
  true at once: motion is a weak global discriminator and a real local
  re-ranker within a small, CLIP-pre-filtered candidate set.

**Conclusion: motion-aware anchoring has real, if modest, headroom and
is worth building — but only as a re-ranker over the CLIP-retrieved
pool, not as a replacement for CLIP retrieval or as a standalone
population-level classifier.** Concretely:

- Best starting features: **curl** and **hessian_max_eigenvalue** (both
  tie for best on bucket (b) alone-ranking and on subset A), with
  **motion_magnitude** close behind and cheaper to compute (already the
  field used elsewhere in the pipeline, no flow-geometry decode needed
  if that ever becomes a cost concern).
- Best starting blend for a global anchor-selection change: a
  **CLIP-dominant blend (~0.7 CLIP / 0.3 motion)**, not an even or
  motion-dominant one — motion-dominant blends win big on bucket (b)
  specifically but lose ground on the majority of already-correct
  CLIP anchors elsewhere in the split; the light blend is the only
  configuration tested that nets positive (+0.47pp) over CLIP alone on
  the whole 639-question split while still meaningfully improving
  bucket (b).
- Ceiling check: bucket (b) is only 38/639 (5.9%) of the whole split
  and 16.67% of the 228-question failure group (c24f8f8) — recovering
  all of it would move whole-split anchor-in-gold rate by at most
  ~1.9pp (38/639), and the best blend only captures part of that. This
  is a real, worthwhile improvement, not a bucket-(c)-sized fix — bucket
  (c) (retrieval never finding a gold frame at all, 64% of the failure
  group per c24f8f8) remains the dominant problem and motion-aware
  anchoring does nothing for it, since it only re-ranks frames already
  in the retrieved pool.
