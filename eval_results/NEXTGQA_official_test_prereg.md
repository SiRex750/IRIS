# Official NExT-GQA test-set Acc@GQA on the scene-sparse complete-block graph — pre-registration

**Registered: 2026-09-27. Status: DRAFT until committed. Nothing in this document may change after
its commit; any deviation during the run is recorded as a deviation, not by editing this file.**

## 1. Why this run exists

The paper's construction result (§4: ~234x faster, ~6.8x smaller, bit-identical) is measured on the
complete-block scene-sparse graph, but no accuracy number in the paper is measured on that graph:
§6.1's Acc@GQA (0.1667) uses a flat dense graph, and §6.2/§7.1 use the tiered `hierarchical_sparse`
graph (paper §3.5, §8). This run measures grounded accuracy on the configuration §4 builds cheaply,
on a split no IRIS experiment has touched, and pairs it with the flat configuration on the same
questions so the effect of the cheap construction on accuracy is measured directly.

## 2. Data and provenance

- **Split:** the OFFICIAL NExT-GQA test split — 5,553 questions over 990 videos (`test.csv`,
  `gsub_test.json`). Source of record: the official release, github.com/doc-doc/NExT-GQA.
- **Naming hazard, stated so nobody trips on it:** `eval_results/test_videos.txt` in this repo is our
  held-out carve of the VALIDATION split, not the official test split. It is unrelated to this run.
- **If videos or annotations come from a mirror** (e.g. the Hugging Face repackaging
  `shuzhig/nextgqa`), before any ingest: record the mirror and revision; sha256-compare
  `test.csv` and `gsub_test.json` against the official release; assert every test `video_id` has a
  video file. Any annotation mismatch -> STOP and report; do not proceed on mirror annotations.
- **Disjointness guard (assert, fail loud):** no official-test `video_id` appears in `val.csv`,
  `gsub_val.json`, `eval_results/val_videos.txt`, or `eval_results/test_videos.txt`.
- **Videos are retained until the run finishes.** Lazy captioning re-decodes source video
  (`iris.query._ensure_captions`); deleting a video before its questions are answered makes a
  caption miss unrecoverable.

## 3. Frozen configuration

Everything not listed takes its `IRISConfig` dataclass default as of the run commit; the run records
the full resolved config and its hash per arm.

**Ingest (shared by all arms — one ingest, one set of survivors):** `IRISConfig` defaults:
salience weights (0.5, 0.3, 0.2), `candidate_thresh` 0.08, `salient_thresh` 0.35, adaptive
thresholds on, `retrieval_strategy="hybrid"`, alpha 0.4, beta 0.3, `scene_segmentation="codec"`.
**`captioner_backend="minicpm"`** — set explicitly, overriding the dataclass default (moondream), to
match the captioner recorded in the NExT-GQA caches behind §6.1. New cache directories only; no
existing cache is read or written.

**Arm S — scene-sparse, complete-block (PRIMARY):** `graph_mode="scene_sparse"`,
`graph_edge_mode="block_diagonal"` (bit-identical to `fully_connected` per the §4.1 gates),
`scene_shortlist_width=0` (auto: max(4, ceil(sqrt(S)))), `scene_shortcut_margin=0.015`,
`scene_neighbor_window=30`, `scene_crossscene_mode="rep_only"`.

**Arm F — flat (the §6.1 configuration, as reference):** `graph_mode="flat"`.

**Shared answer path (both graph arms), identical to §6.1:** `cerberus_mode="v2"`,
`l2_retrieve_top_k=12`, `ranking_mode="ppr"`, `ppr_lambda=0.5`, `ppr_damping=0.5`,
`motion_similarity_mode="action_score"`, `l1_w_action=0.60`, `l1_w_query=0.25`,
`l1_w_persist=0.15` (other l1 weights 0.0); span `predict_span(mode="ppr_peak", half_width=2.2,
peak_source="clip_in_ppr_top8")`; MC prompt and parser `eval.mc_scorer.build_mc_prompt` /
`parse_mc_answer`.

**Arm U — uniform:** `uniform_ts(duration, 12)` exactly as in `scripts/pnowa_test_accgqa_run.py`,
answered through the same prompt and answerer.

**Answerer:** `granite4:micro` via llama-server (`aria.LlamaServerBackend`), temperature 0,
`cache_prompt=false` (verified sent in every request payload, `iris/aria.py`), server run with
`--parallel 1`. Build: the determinism-gated binary pinned at commit 505c2f7 (b10099); the run
records the build string the server reports.

**Scoring:** official NExT-GQA convention — IoP as maximum overlap over gold spans (the vendored
official scorer used in §6.1); Acc@GQA = answer correct AND IoP >= 0.5.

The random-frame arm of §6.1 is not run: it separated from the proposed arm there and adds cost
without informing the question this run asks. This is the only run of these arms on this split; no
arm is added later.

## 4. Metrics

**PRIMARY:** Acc@GQA of Arm S over all 5,553 questions.

**SECONDARY (all pre-declared):**
- paired difference Acc@GQA(S) - Acc@GQA(F), same questions, with a video-clustered bootstrap 95% CI
  (B = 10,000, seed 20260927);
- paired difference Acc@QA(S) - Acc@QA(F), same bootstrap;
- Acc@QA(S) - Acc@QA(U), same bootstrap;
- per arm: Acc@QA, mIoP, IoP@0.5, mIoU, option-parse failure rate;
- per-type breakdown (causal / temporal / descriptive) for S and F, reported descriptively only.

## 5. Predictions (stated before any result)

- Acc@GQA(S) lands in the weakly-supervised band, roughly 0.14–0.19.
- The paired Acc@GQA(S) - Acc@GQA(F) CI includes zero: the cheap construction does not measurably
  change grounded accuracy.
- Acc@QA(S) > Acc@QA(U), with a CI excluding zero at this n.

## 6. How each outcome will be reported (fixed now)

- **S-F CI includes zero:** "no detected change in grounded accuracy from the cheap construction at
  n = 5,553", with the CI width stated — not "equivalent".
- **S-F CI excludes zero, S lower:** the paper states that the cheap construction lowers grounded
  accuracy on this benchmark, by the measured amount, in §6 and §8. No tuning of Arm S follows.
- **S-F CI excludes zero, S higher:** reported as observed; no mechanism claimed beyond the
  measurement.
- **Whatever Acc@GQA(S) is, that is the number.** No configuration change, re-run, or subset
  selection after results are seen. Any later run on this split is disclosed as a second touch.

## 7. Failure handling (fixed now)

- Videos that fail ingest: counted and listed. Their questions are scored as **incorrect** in the
  primary metric for every arm (conservative; no survivorship). A secondary table excludes them.
- Option-parse failures count as incorrect (as in §6.1) and are reported per arm.
- Caption misses whose video cannot be decoded: counted per arm; the frame carries the existing
  `[CAPTION_FAILED]` marker, as in the §6.1 harness.

## 8. Run integrity guards (assert, fail loud)

1. **Registration precedes the run, verifiably:** the run script reads this file, records its
   sha256, and refuses to start unless the commit that added this file is an ancestor of `HEAD`.
2. **Clean tree:** zero tracked-dirty files at run start; commit hash recorded.
3. **Determinism spot-check before the full run:** the first 20 questions of Arm F are answered
   twice; raw answerer outputs must be byte-identical, else STOP.
4. **Disjointness** (§2) and **mirror checks** (§2) pass.
5. **Per-arm resolved config hash** and the llama-server build string are written to the output.

## 9. Outputs

`eval_results/NEXTGQA_official_test_raw.json` (per-question rows for every arm, including raw
answerer text) and `eval_results/NEXTGQA_official_test_result.md`, both with provenance (commit,
dirty count, config hashes, prereg sha256, server build, data source and hashes, timings).

## 10. Consequences for the paper, declared in advance

- §6.1 gains a directly comparable full-test row for Arm S, and Arm F measured on the same questions.
- §8's "construction saving is not measured on the evaluated configuration" is updated for NExT-GQA
  by the S-F result, whichever way it goes. It remains open for MLVU (§6.2), which uses the tiered
  graph.
- The held-out validation result (0.1667, §6.1) is kept and labelled as a different split; it is not
  replaced or averaged with this one.
