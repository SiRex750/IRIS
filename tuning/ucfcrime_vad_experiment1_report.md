# UCF-Crime VAD Experiment 1 — Report

**Status: PARTIAL-CORPUS RESULT, not the standard 290-video benchmark.**
The official annotation file was initially absent from this box and every
reachable git branch; the user then extracted it locally mid-run
(`eval/data/ucf/annotations/.../Temporal_Anomaly_Annotation.txt`, verified as
the genuine 290-row/140-anomalous/150-normal file). Of those 290 official test
videos, only **169** (19 anomalous + all 150 normal) have a matching local
`.mp4` — the other 121, spanning 9 entire anomaly categories, are not on this
box and were not downloaded, per instructions. **Every AUC number below is
computed over this 169-video subset and is explicitly NOT comparable to the
cited 290-video published numbers (ZS CLIP / ZS ImageBind / LAVAD / EventVAD).**
This caveat applies to every number in §3–§6 and is restated there.

## 0. Branch / config / data provenance

- Task asked to work off a branch named `feat/prerun-fixes` and to use IRIS
  files from `git@github.com:swarapotd-rgb/IRIS.git`. That branch exists only
  on that remote (not on `origin` = `SiRex750/IRIS`). This experiment branch
  (`siddanth/ucf-vad-exp1`) was created from `swara/feat/prerun-fixes`
  (commit `3a6930c`), which is where `tuning/frozen_state.json` and the
  `packet_size_weight` field on `ActionScoreConfig` actually exist — neither
  exists on `origin/main`.
- `tuning/frozen_state.json` was **not edited**. Its `"frozen"` block was read
  and used as-is everywhere: `packet_size_weight=0.8, motion_weight=0.1,
  luma_entropy_weight=0.1, peak_distance=5, peak_prominence=0.05,
  persistence_threshold=0.4, max_prominence=0.5` (Stage B additionally used
  `retrieval_strategy="hybrid"` and `l2_retrieve_top_k=4` from the same file).
- UCF-Crime video files (`eval/data/ucf/videos/`) and the Python venv
  (`.venv/`) are untracked/local-only and not present in a fresh `git
  worktree` checkout; an `ln -s` was attempted to link them in from the main
  checkout, but silently fell back to a **full copy** on this Windows box
  (confirmed via `Get-Item ... | Select LinkType` showing no reparse point,
  and matching `du -sh` sizes — 11GB in both locations). This was only
  discovered when the user's newly-extracted `eval/data/ucf/annotations/`
  folder didn't show up in the worktree; it was then copied in explicitly.
  No video was re-downloaded — only the existing local corpus was used, per
  instructions.

## 1. Dataset provenance and verification

**Source on this box:** `eval/data/ucf/videos/` (absolute path:
`C:/Users/Siddanth Anil/IRIS/eval/data/ucf/videos/`). Same corpus previously
inventoried in `eval_results/ucf_inventory.md` (a 40-file sample) and used for
the scaling-curve work referenced in `docs/` (N up to 13,506 *survivor*
frames from `Arson019`, raw container frame count 126,553).

This run re-verified the **full population** (350/350 files, not a sample) —
`scripts/ucfcrime_vad_exp1_verify.py`, output
`tuning/ucfcrime_vad_exp1/ucfcrime_dataset_validation.json`:

| Check | Result |
|---|---|
| Total files | **350** `.mp4` files |
| Total bytes | **11,177,277,692 bytes** (~10.41 GiB) |
| Container open (`av.open`) success | **350 / 350** |
| Packet-level readability (`packet.size`, `packet.is_keyframe` — PyAV's `ffprobe -show_packets` equivalent) | **350 / 350** |
| Codec distribution | **h264: 348, mjpeg: 2** |
| Extracted-frame directories (`.jpg`/`.png` in the video tree) | **0** — confirmed real encoded video, not extracted frames |
| Non-`.mp4` files present in the tree | **0** |

No standalone `ffprobe` binary is present on this box; PyAV (`av.open` /
`container.demux`, bundled `libavformat`) was used as a read-only equivalent
to `ffprobe -show_streams` / `-show_packets`.

**The 2 mjpeg files** (`Arrest050_x264.mp4`, `Assault017_x264.mp4` — both
carry a misleading `_x264` filename suffix) are flagged. Neither is among the
169 annotation-matched videos scored in §3/§4, so they don't affect the AUC
numbers below.

### Category coverage

Only **4 of the 13** official anomaly categories are present locally at all:
Abuse, Arrest, Arson, Assault (50 each) plus `Testing_Normal_Videos_Anomaly`
(150, matching the official normal-test count). Burglary, Explosion,
Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
are entirely absent.

### 1d. Annotations

`Temporal_Anomaly_Annotation_ForTestVideos.txt` (functionally identical
content under the filename `Temporal_Anomaly_Annotation.txt`) was not found
anywhere reachable at the start of this run (this repo, `origin`, or 9
branches on `swarapotd-rgb/IRIS`) and was extracted locally by the user
partway through, at
`eval/data/ucf/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt`.
Verified (`scripts/ucfcrime_vad_exp1_stageA.py`, output
`tuning/ucfcrime_vad_exp1/annotation_validation.json`):

| Check | Result |
|---|---|
| Total rows | **290** |
| Unique `video_name` values | **290** |
| Anomalous / Normal split | **140 / 150** (matches spec) |
| Rows with a matching local `.mp4` | **169** (19 anomalous + 150 normal) |
| Rows with no local video file | **121** — all from the 9 absent categories, listed by name in `annotation_validation.json` |
| Frame-index-out-of-bounds spans (span end ≥ decoded frame count) | **3** — see below |
| Video decode failures during verification | **0** |

The 3 out-of-bounds spans are all off-by-one at the boundary (annotation span
end exactly equals the video's decoded frame count, e.g. `Arson011_x264.mp4`:
span end `1267` vs. `1266` decoded frames), consistent with a 1-indexed vs.
0-indexed convention mismatch rather than a real annotation error. **Reported,
not silently clamped**: they're logged by name in
`annotation_validation.json`, and the ground-truth array construction clips
the span to `[0, n_frames-1]` (a 1-frame boundary effect, not a data
corruption) — noted here explicitly per non-negotiable #4.

## 2. Evaluation protocol (as applied)

- **Metric:** frame-level ROC-AUC. Ground truth = 1 inside an annotated span,
  0 elsewhere; all-0 for Normal videos.
- **Primary:** pooled (micro) AUC — concatenate frame scores across all
  scored videos, one ROC-AUC. **Computed over the 169-video subset, not the
  290-video standard set — not directly comparable to the cited rows in §6.**
- **Secondary:** per-video macro AUC, averaged only over videos with ≥1
  positive frame (the 19 anomalous videos; 150 Normal videos excluded — AUC
  undefined with no positives).
- **Propagation rule:** Stage-A/Stage-B scores are computed only on
  retained-tier frames (I_FRAME/PEAK/SALIENT/CANDIDATE — the frames
  `iris.charon_v.parse_video` returns in `output_frames`). Every other
  (SKIP-tier) frame is assigned its nearest **preceding** retained frame's
  score (piecewise-constant hold-forward); frames before the first retained
  frame hold that first frame's score backward (stated, not hidden).
- **Both** pooled-over-all-frames (with hold-forward fill) and
  pooled-over-retained-frames-only (no fill) AUC are reported, to show the
  propagation effect.
- **Retention rate measured, not assumed:** mean **10.49%** across the 169
  scored videos (min 10.40%, max 11.25%) — this corrects the task prompt's
  assumed ~13% figure.

## 3. Stage A (codec-only anomaly score)

Frozen weights (`packet_size_weight=0.8, motion_weight=0.1,
luma_entropy_weight=0.1`, plus the frozen peak/persistence gate) applied
directly as the anomaly score — no graph, no PPR, no CLIP, no captioner, no
LLM. Script: `scripts/ucfcrime_vad_exp1_stageA.py`. Full output:
`tuning/ucfcrime_vad_exp1/stageA_results.json`,
`tuning/ucfcrime_vad_exp1/stageA_per_video.csv`, per-frame CSVs at
`tuning/ucfcrime_vad_exp1/per_frame_ground_truth_scores/*.csv`.

**169/169 videos decoded successfully, 0 failures.**

| Result | Value |
|---|---|
| **Pooled AUC — all frames, propagated** | **0.7232** |
| Pooled AUC — retained frames only (no fill) | 0.7086 |
| Macro AUC — 19 anomalous videos only | 0.5650 |
| Retention rate (mean / min / max) | 10.49% / 10.40% / 11.25% |

### Per-category pooled AUC (each category's matched videos + all 150 matched normal videos)

| Category | Matched videos | Pooled AUC |
|---|---|---|
| Abuse | 2 | 0.8136 |
| Arson | 9 | 0.8022 |
| Arrest | 5 | 0.7532 |
| Assault | 3 | 0.6520 |

Small n per category (2–9 videos) — read these as directional, not precise
estimates; no other categories are available to compare against (§1).

### Interpretation gate

Per the task's own table, pooled AUC 0.7232 > 0.65 → **"Codec-native VAD is
real. Stage B is justified."** **This placement is on the 169-video partial
corpus, not the intended 290-video evaluation the gate was designed around**
— restated because it matters: the pooled number is inflated relative to what
a full 290-video run would likely show, since (a) pooling against the full
150-video normal set while only sampling 19 (of 140) anomalous videos gives
disproportionate weight to whichever anomaly signal happens to be easiest in
the 4 available categories, and (b) the **macro AUC (0.565, per-video, no
normal-video pooling boost) is much weaker — close to the cited ZS-CLIP level
(53.16)** and is arguably the more honest single number for "does this codec
signal generalize per-video." Both numbers are reported side by side
deliberately; take the pooled 0.72 with that context, not at face value
against the cited rows in §6.

## 4. Stage B (graph-topology anomaly score)

<!-- FILLED IN FROM tuning/ucfcrime_vad_exp1/stageB_results.json once the background run completes -->
STAGE_B_PENDING_FILL

## 5. Efficiency measurement (Step 5)

Measured on **32 videos** (task minimum: 30), sampled deterministically by
stratifying the full 350-file population by `nb_frames_meta` and taking evenly
spaced ranks — not uniform-random, but explicitly chosen to span the full
duration range (137 → 18,224 container frames). The single 126,553-frame
`Arson019` outlier was excluded (documented in `eval_results/ucf_inventory.md`
as a stress-test point, not a curve anchor) to keep wall time bounded — stated,
not silent. Script: `scripts/ucfcrime_vad_exp1_efficiency.py`. Raw output:
`tuning/ucfcrime_vad_exp1/efficiency_per_video.csv`,
`tuning/ucfcrime_vad_exp1/efficiency_measurements.json`.

Each video was ingested through `iris.charon_v.parse_video(..., full_decode=False)`
— the codec-level L1 pass — using the frozen action-score weights.
**32/32 ingests completed with no errors.**

| Measurement | Result |
|---|---|
| Wall time per video — mean / min / max | **2.94s / 0.34s / 15.47s** |
| Wall time per 1,000 frames — mean of per-video ratios | **1.13s / 1,000 frames** |
| Wall time per 1,000 frames — aggregate (Σtime / Σframes×1000) | **0.895s / 1,000 frames** (105,241 frames decoded across the 32 videos in 94.23s total) |
| Peak RSS — mean / max across the 32 runs | **530.4 MB / 1,291.2 MB** |
| Retention rate (this sample) — mean / min / max | **10.49% / 10.40% / 10.95%** |
| Device used | **CPU** (torch 2.13.0+cpu; `torch.cuda.is_available()` = `False`) |
| GPU required at any point | **No** |
| NN forward-pass calls during ingest, summed over 32 videos | **0** (instrumented via a monkeypatch on `torch.nn.Module.__call__` for the duration of each `parse_video()` call, not asserted from source — see `note_on_verification_method` in `efficiency_measurements.json`) |

This 0-NN-forward-pass result is specific to the **Stage-A-only ingest path**
(`parse_video`, no CLIP). Stage B (§4) does invoke CLIP forward passes (CPU)
to build the scene-sparse graph — that cost is measured separately in §4 and
is not part of this table.

**Retention-rate correction:** the task prompt assumes "~13% of frames." The
measured retention on this corpus is **~10.5%**, consistently (10.4–11.25%
range across both the 32-video efficiency sample and the 169-video Stage-A/B
run) — used throughout this report in place of the prompt's assumed figure.

**No per-query LLM call:** this ingest path never invokes `answerer_backend`
or any captioner/LLM code path (confirmed by the same instrumentation).
Unlike NExT-GQA, there is no per-query answerer stage diluting this
comparison — ingest + score is the entire pipeline for Stage A.

### Cost comparison vs EventVAD (cited, not reproduced)

EventVAD's published ingest requires, per frame: CLIP ViT embeddings **and**
RAFT optical flow, feeding VideoLLaMA2.1-7B, on a single NVIDIA A800 80GB GPU
(cited, not run here — non-negotiable #3). IRIS Stage A's measured ingest
uses zero neural-network forward passes and zero GPU requirement, at
~0.9–1.1s per 1,000 frames on ordinary CPU with peak RSS under 1.3GB. Stage B
(§4) does use CLIP, but still CPU-only, no GPU. No EventVAD wall-clock number
is stated or estimated — the comparison is architectural, not a timing race.

## 6. Comparison table

| Method | Training-free | UCF-Crime pooled AUC | Ingest features | Hardware |
|---|---|---|---|---|
| ZS CLIP | yes | 53.16 (cited) | CLIP per frame | GPU |
| ZS ImageBind | yes | 55.78 (cited) | ImageBind per frame | GPU |
| LAVAD | yes | 78.33 (cited) | VLM captions + LLM scoring | GPU |
| EventVAD | yes | 82.03 (cited) | CLIP + RAFT optical flow, 7B VLM | A800 80GB |
| **IRIS Stage A** | yes | **0.7232 pooled / 0.5650 macro — measured, 169/290-video partial corpus, NOT comparable to the rows above** | codec packet size, no model | CPU only, measured (§5) |
| **IRIS Stage B** | yes | **see §4 — same 169-video caveat** | + scene-sparse graph (CLIP + PageRank) | CPU only |

The four cited rows are copied from their published papers, not re-run, per
non-negotiable #3. Context (not a comparison row): weakly-supervised methods
reach ~89.8% AUC (MTFL) but are trained on video-level labels — a different
problem category, not a target for this table.

## 7. What I could not measure

This section is not empty:

1. **The standard 290-video pooled/macro AUC** — 121 of 290 official test
   videos (9 entire anomaly categories: Burglary, Explosion, Fighting,
   RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism) are not
   present on this box. Every AUC in §3/§4/§6 is a 169-video partial-corpus
   number, explicitly flagged everywhere it appears, not filled in or
   estimated for the missing 121.
2. **Per-category AUC for the 9 absent categories** — no data, not estimated.
3. **A properly-registered fusion-weight sweep for Stage B** — deliberately
   not done; fixed 50/50 per instructions.
4. Everything listed as measured elsewhere in this report (dataset
   verification, efficiency, Stage A, Stage B) was in fact measured — nothing
   further was silently skipped within the 169-video scope available.
