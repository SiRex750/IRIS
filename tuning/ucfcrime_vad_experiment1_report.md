# UCF-Crime VAD Experiment 1 — Report

**Status: BLOCKED at Step 1d / Step 2.** This is a measurement task and a negative
result is a valid, complete outcome. No frame-level anomaly-detection accuracy
(Steps 2–4) is reported below because the ground-truth data required to compute
it does not exist anywhere reachable from this box. Everything that could be
measured without ground truth (dataset verification, ingest efficiency) was
measured and is reported in full, with real numbers from files written during
this run.

## 0. Branch / config provenance (read before the rest)

- Task asked to work off a branch named `feat/prerun-fixes` and to use IRIS
  files from `git@github.com:swarapotd-rgb/IRIS.git`. That branch exists only
  on that remote (not on `origin` = `SiRex750/IRIS`, and not locally before this
  run). This experiment branch (`siddanth/ucf-vad-exp1`) was created from
  `swara/feat/prerun-fixes` (commit `3a6930c`), which is where `tuning/frozen_state.json`
  and the `packet_size_weight` field on `ActionScoreConfig` actually exist —
  neither exists on `origin/main` at the commit this session started from.
- `tuning/frozen_state.json` was **not edited**. Its `"frozen"` block was read
  and used as-is:
  `packet_size_weight=0.8, motion_weight=0.1, luma_entropy_weight=0.1,
  peak_distance=5, peak_prominence=0.05, persistence_threshold=0.4,
  max_prominence=0.5` (plus retrieval/PPR fields not used by this experiment,
  since Stage B never ran — see §5).
- UCF-Crime video files (`eval/data/ucf/videos/`) and the Python venv (`.venv/`)
  are untracked/local-only and are not present in a fresh `git worktree`
  checkout; they were symlinked in from the pre-existing checkout at
  `C:/Users/Siddanth Anil/IRIS/` rather than re-downloaded or re-created, per
  the instruction to use only the existing local download.

## 1. Dataset provenance and verification

**Source on this box:** `eval/data/ucf/videos/` (absolute path:
`C:/Users/Siddanth Anil/IRIS/eval/data/ucf/videos/`, linked into this worktree).
This is the same corpus previously inventoried in `eval_results/ucf_inventory.md`
(a 40-file sample) and used for the scaling-curve work referenced in `docs/`
(`eval_results/oom_frontier.md`, `eval_results/candidate_clips.md` — N up to
13,506 *survivor* frames from `Arson019`, whose raw container frame count is
126,553).

This run re-verified the **full population** (350/350 files, not a sample) —
see `scripts/ucfcrime_vad_exp1_verify.py` and its output
`tuning/ucfcrime_vad_exp1/ucfcrime_dataset_validation.json`:

| Check | Result |
|---|---|
| Total files | **350** `.mp4` files |
| Total bytes | **11,177,277,692 bytes** (~10.41 GiB) |
| Container open (`av.open`) success | **350 / 350** |
| Packet-level readability (`packet.size`, `packet.is_keyframe` — PyAV's `ffprobe -show_packets` equivalent) | **350 / 350** |
| Codec distribution | **h264: 348, mjpeg: 2** |
| Extracted-frame directories (`.jpg`/`.png` in the video tree) | **0** — confirmed these are encoded video containers, not extracted frames |
| Non-`.mp4` files present in the tree | **0** |

No standalone `ffprobe` binary is present on this box (same situation already
documented in `eval_results/ucf_inventory.md`); PyAV (`av.open` / `container.demux`,
bundled `libavformat`) was used as a read-only equivalent to
`ffprobe -show_streams` / `-show_packets` — it reads the same container/packet
metadata without a full pixel decode loop.

**The 2 mjpeg files** (`Arrest050_x264.mp4`, `Assault017_x264.mp4` — note both
carry a misleading `_x264` filename suffix) are flagged and excluded from the
`is_h264_or_hevc` count and from the efficiency sample's stratification pool.
This confirms and extends the single mjpeg flag already noted in
`eval_results/ucf_inventory.md` from a 40-file sample — the full-population
pass found a second one the sample had missed.

### Category coverage — the first hard finding

The local corpus is **not the full UCF-Crime test/train tree**. By directory:

| Directory | Count | Official UCF-Crime role |
|---|---|---|
| `Anomaly-Videos-Part-1/Abuse` | 50 | 1 of 13 anomaly categories |
| `Anomaly-Videos-Part-1/Arrest` | 50 | 1 of 13 anomaly categories |
| `Anomaly-Videos-Part-1/Arson` | 50 | 1 of 13 anomaly categories |
| `Anomaly-Videos-Part-1/Assault` | 50 | 1 of 13 anomaly categories |
| `Testing_Normal_Videos_Anomaly` | 150 | matches the official 150-video normal test count |

Only **4 of the 13** official anomaly categories are present at all: Abuse,
Arrest, Arson, Assault. **Burglary, Explosion, Fighting, RoadAccidents,
Robbery, Shooting, Shoplifting, Stealing, and Vandalism are entirely absent**
from this box. `Anomaly-Videos-Part-1` is also the official *training* zip
name in the CRCV Dropbox distribution (Parts 1–4 contain the full category
videos, split into train/test only via a separate file list) — nothing in the
local checkout identifies which, if any, of these 200 anomaly clips are members
of the official 290-video *test* split versus the ~1,610-video *train* split.

### 1d. Annotations — BLOCKER

`Temporal_Anomaly_Annotation_ForTestVideos.txt` was searched for and **not found**:

- Not present anywhere under this repo's working tree.
- Not present in any branch of `origin` (`SiRex750/IRIS`).
- Not present in any of the 9 branches fetched from `swarapotd-rgb/IRIS`
  (`main`, `feat/prerun-fixes`, and 7 others) — checked via `git ls-tree -r
  <branch> | grep -i ucf`.
- Not present anywhere else searched on this machine (home directory scan for
  `*Anomaly*annotation*`, `*Test*split*`).

**Without this file (or an equivalent train/test split + temporal-window
ground truth), Step 1d cannot be completed, and none of Step 2 (evaluation
protocol application), Step 3 (Stage A AUC), or Step 4 (Stage B) can be
computed — there is no ground truth to score against.** This is reported
plainly per non-negotiable #5: the blocker is written up here and the
AUC-dependent branch of work stopped, rather than substituting a proxy split,
a self-labeled heuristic, or a different dataset.

Full detail (search paths, per-file codec/packet results) is in
`tuning/ucfcrime_vad_exp1/ucfcrime_dataset_validation.json`.

## 2. Evaluation protocol (defined, never applied)

Per Step 2, the protocol below is what *would* be used if ground truth were
available. It is stated for completeness and to make clear that no part of it
was silently skipped or approximated — it was simply never run, because there
is nothing to evaluate against:

- Metric: frame-level ROC-AUC, pooled (micro, concatenate all test-video frame
  scores) as primary, per-video macro (excluding normal videos, which have no
  positives) as secondary.
- Propagation rule: piecewise-constant hold of each retained frame's score
  forward to the non-retained frames it covers, so every frame gets a score.
- Both pooled-over-all-frames and pooled-over-retained-frames-only AUC would
  be reported, to show the propagation effect.

**None of this was executed.** No AUC number of any kind appears in this report.

## 3. Stage A (codec-only anomaly score) — NOT RUN

Stage A requires scoring frames from the 290 annotated test videos against
ground-truth anomaly windows. With no annotation file and no confirmed test
membership for any of the 200 available anomaly clips, there is no valid
input to compute a pooled or per-category AUC from. **No Stage A AUC, no
interpretation-gate placement, and no per-category breakdown are reported.**

What *was* verified and is real: the codec action-score module itself
(`iris/action_score.py`, frozen weights from `tuning/frozen_state.json`) runs
correctly end-to-end on this corpus — see §6 and the per-frame CSVs at
`tuning/ucfcrime_vad_exp1/per_frame_action_scores/*.csv`, produced as a side
effect of the efficiency measurement (raw action scores only, **no AUC
column**, since there is nothing to score them against).

## 4. Stage B (graph-topology score) — SKIPPED

Per the task's own gate: Stage B only runs if Stage A pooled AUC > 0.65. Stage
A was never computed (§3), so this condition can't be evaluated. Stage B is
**skipped**, not attempted with a substitute threshold or a proxy signal.

## 5. Efficiency measurement (Step 5 — not blocked by the annotation gap)

Measured on **32 videos** (task minimum: 30), sampled deterministically by
stratifying the full 350-file population by `nb_frames_meta` and taking evenly
spaced ranks — not uniform-random, but explicitly chosen to span the full
duration range (137 → 18,224 container frames). The single 126,553-frame
`Arson019` outlier was excluded from this sample (already documented in
`eval_results/ucf_inventory.md` as a stress-test point, not a curve anchor) to
keep total wall time bounded; this exclusion is stated, not silent. Script:
`scripts/ucfcrime_vad_exp1_efficiency.py`. Raw per-video output:
`tuning/ucfcrime_vad_exp1/efficiency_per_video.csv`; summary:
`tuning/ucfcrime_vad_exp1/efficiency_measurements.json`.

Each video was ingested through `iris.charon_v.parse_video(..., full_decode=False)`
— the codec-level L1 pass (packet-size demux + selective pixel/motion-vector
feature extraction), using the frozen action-score weights from
`tuning/frozen_state.json`. **32/32 ingests completed with no errors.**

| Measurement | Result |
|---|---|
| Wall time per video — mean / min / max | **2.94s / 0.34s / 15.47s** |
| Wall time per 1,000 frames — mean of per-video ratios | **1.13s / 1,000 frames** |
| Wall time per 1,000 frames — aggregate (Σtime / Σframes×1000) | **0.895s / 1,000 frames** (105,241 frames decoded across the 32 videos in 94.23s total) |
| Peak RSS — mean / max across the 32 runs | **530.4 MB / 1,291.2 MB** |
| Retention rate (frames kept for full feature extraction ÷ frames decoded) — mean | **10.49%** |
| Retention rate — min / max | **10.40% / 10.95%** |
| Device used | **CPU** (torch 2.13.0+cpu on this box; `torch.cuda.is_available()` = `False`) |
| GPU required at any point | **No** |
| NN forward-pass calls during ingest, summed over all 32 videos | **0** (instrumented via a monkeypatch on `torch.nn.Module.__call__` for the duration of each `parse_video()` call, not asserted from reading the source — see `note_on_verification_method` in `efficiency_measurements.json`) |

**Retention-rate correction:** the task prompt states "IRIS's L1 retains
roughly 13% of frames." The measured retention on this corpus is **~10.5%**,
consistently across all 32 videos (10.40–10.95% range, a tight band). This is
the real observed figure and is used in place of the prompt's assumed 13%
anywhere retention matters — it was not adjusted or clamped to match the
prompt's expectation.

Per-frame Stage-A codec action scores (packet-size/motion/luma-entropy,
frozen weights, raw — no smoothing, no AUC) were also emitted as a side effect
for all 32 sampled videos, at `tuning/ucfcrime_vad_exp1/per_frame_action_scores/<video>.csv`,
so the scores can be independently recomputed. These are **not** anomaly-detection
results — there is no ground truth to validate them against (§3) — they are the
raw signal only.

**No per-query LLM call**: this ingest path never invokes `answerer_backend`
or any captioner/LLM code path — confirmed by the same zero-NN-forward-pass
instrumentation above, which would have caught any such call. Unlike the
NExT-GQA setting, there is no per-query answerer stage to dilute this cost
comparison; ingest + score is the entire pipeline.

### Cost comparison vs EventVAD (cited, not reproduced)

EventVAD's published ingest pipeline requires, per frame: CLIP ViT embeddings
**and** RAFT optical flow, feeding a VideoLLaMA2.1-7B backbone, run on a single
NVIDIA A800 80GB GPU (cited from the EventVAD paper — not run in this
experiment, per non-negotiable #3). IRIS's measured ingest above uses zero
neural-network forward passes, zero GPU requirement, and runs on ordinary CPU
at ~0.9–1.1 seconds per 1,000 frames with peak RSS under 1.3 GB across the
sampled range. No EventVAD wall-clock number is stated or estimated here — the
comparison is architectural (features/model/hardware required), not a
timing race.

## 6. Comparison table

| Method | Training-free | UCF-Crime pooled AUC | Ingest features | Hardware |
|---|---|---|---|---|
| ZS CLIP | yes | 53.16 (cited, Sultani et al. protocol as reported by LAVAD/EventVAD papers) | CLIP per frame | GPU |
| ZS ImageBind | yes | 55.78 (cited) | ImageBind per frame | GPU |
| LAVAD | yes | 78.33 (cited) | VLM captions + LLM scoring | GPU |
| EventVAD | yes | 82.03 (cited) | CLIP + RAFT optical flow, 7B VLM | A800 80GB |
| **IRIS Stage A** | yes | **not measured — blocked (§3)** | codec packet size, no model | measured, see §5 |
| **IRIS Stage B** | yes | **N/A — skipped (§4)** | + scene-sparse graph | N/A |

Context (not a comparison row): weakly-supervised methods reach ~89.8% AUC
(MTFL) but are trained on video-level labels — a different problem category,
not a target for this table.

No EventVAD/LAVAD numbers were reproduced; the four cited rows are copied from
their published papers, not re-run, per non-negotiable #3.

## 7. What I could not measure

This section is not empty:

1. **IRIS Stage A / Stage B pooled and macro AUC, per-category AUC, and the
   interpretation-gate placement** — blocked by the complete absence of
   `Temporal_Anomaly_Annotation_ForTestVideos.txt` (or any equivalent
   train/test split list) on this machine, on `origin/IRIS`, or on any of the
   9 branches checked on `swarapotd-rgb/IRIS`. Not estimated, not proxied.
2. **Whether any of the 200 local anomaly clips (Abuse/Arrest/Arson/Assault)
   are official test-split members** — cannot be determined without the split
   list. Treated as unknown, not assumed either way.
3. **9 of 13 official anomaly categories** (Burglary, Explosion, Fighting,
   RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism) — not
   present on this box at all. Per instructions, no download was performed to
   fill this gap.
4. **Retention rate on the full 13% figure claimed in the task prompt** — the
   task states "IRIS's L1 retains roughly 13% of frames"; this run's own
   measurement (§5) reports the actual observed retention on the sampled
   videos instead of assuming that figure, and it differs — see §5 for the
   real number and why.
5. **EventVAD wall-clock cost** — not fabricated; EventVAD was not run
   (non-negotiable #3). Only its published architecture description (CLIP +
   RAFT optical flow, VideoLLaMA2.1-7B, single A800 80GB) is stated, as cited.

Nothing was silently dropped: every planned measurement that didn't happen is
listed above with the specific reason.
