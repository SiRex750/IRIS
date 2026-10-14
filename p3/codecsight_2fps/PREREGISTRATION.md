# Paper 3 — Pre-registration: CodecSight-matched 2 fps arm (CCTV)
Date: 2026-10-14. Approved in review on 2026-10-14 with three edits (§1 earlier 2 FPS arm, §4 truth limitation,
date). Written before any encode, RAFT run or metric of this arm. The only
script run so far is `inventory.py`, which decodes the raw VIRAT files to count frames and reads container metadata.
It reads no pixel value, motion vector or flow value. Repository HEAD when written: 9a8bb72 (branch sonu/p3).
Collection and analysis code are written after this file is committed; their commit hashes are recorded in the
results.

What this is: **CodecSight's documented policy in its own regime.** The CodecSight simulation
(`p3/codecsight_sim/PREREGISTRATION.md`, 9f5b5a3) applied the policy to native-rate encodes (24–30 fps, 16-frame
refresh = 0.53–0.67 s) and said plainly that this is not CodecSight's setting (its §2, "The gap"). Here we re-encode at
2 fps with a 16-frame GOP (I-frame every 8 s), which by our reading of v4 (16-frame GOP = 8 s GOP interval) is the
stream CodecSight's policy runs on. It is still not CodecSight: no VLM is run and no task accuracy is measured.

Question: in that regime, does the documented policy stay fresh at every encoder setting, or does it reuse tokens
whose content has truly moved, and does that change with the encoder?

## 1. Clips and source-frame windows
**Clips:** the 27 eligible clips of the confirmatory test (`p3/virat_confirm/eligibility_v0.csv`, 7 scenes).
Eligibility is inherited; §5 adds a 2 fps motion count that sets the denominator of the verdicts.

**Frames analysed before at native rate:** frames 0–299 of every clip (exploratory test, `p3/virat`) and frames
300–599 (confirmatory test, `p3/virat_confirm`, start 300). Frames 600 and later have never been decoded for any
analysis. (The CodecSight simulation and the post hoc matched comparison reused the saved confirmatory data, frames
300–599.)

**Frames already sampled at 2 FPS:** the CodecSight simulation's exploratory 2 FPS arm
(`p3/codecsight_sim/PREREGISTRATION.md` §7, variants a and b) already sampled frames 300–599 of these 27 clips at
2 FPS (every 12th or 15th frame). It used native-rate vectors (frame s−1 → s), not 0.5 s vectors, and truth summed
over every native frame. Its results are known to us (`p3/codecsight_sim/verdicts.md`). In those results, the
policy's median STALE_MOV per arm was 0.22–0.74 in variant a and 0.03–0.62 in variant b. The 16 clips whose windows
overlap frames 300–599 (table below) therefore include frames that were analysed at 2 FPS spacing before, with a
different vector signal.

**Sampling step k:** every 12th source frame for the 23.97 fps clips (20 clips, h264 sources, 0.5005 s apart) and
every 15th for the 30 fps clips (7 clips, mpeg4 sources, 0.5 s apart). k = round(fps / 2).

**Window rule (fixed here, applied by `inventory.py`).** A window of G full GOPs is 16G sampled frames,
s_j = start + j·k for j = 0..16G−1, spanning (16G − 1)·k + 1 source frames.
1. Unseen: start = 600, and G = the largest value in 2..5 that fits in the clip. 5 GOPs = 80 sampled frames = 40 s,
   which is CodecSight's request window (v4 §5).
2. Otherwise, end-aligned with 2 GOPs: start = n − ((32 − 1)·k + 1), the latest start, so the overlap with analysed
   frames is as small as possible.
3. Clips shorter than one 2-GOP span (373 frames at k = 12, 466 at k = 15) are excluded. **No clip is excluded:**
   the shortest clip has 493 frames.

**Result (`inventory.txt`):**
- 11 clips have unseen windows (start 600): 9 with 5 GOPs (80 sampled frames, 40 s), and
  VIRAT_S_000207_00 and VIRAT_S_010003_13 with 3 GOPs (48 frames, 24 s).
- **16 clips cannot fit 2 GOPs after frame 599.** Their 2-GOP windows (32 sampled frames, 16 s) overlap frames
  analysed before at native rate:

| clip | window (source frames) | in exploratory 0–299 | in confirmatory 300–599 | sampled frames in analysed ranges |
|---|---|---|---|---|
| VIRAT_S_000207_03_000556_000590 | 554–1019 | – | 554–599 | 4 / 32 |
| VIRAT_S_000207_04_000902_000934 | 472–937 | – | 472–599 | 9 / 32 |
| VIRAT_S_010000_06_000728_000762 | 393–765 | – | 393–599 | 18 / 32 |
| VIRAT_S_010000_07_000827_000860 | 242–614 | 242–299 | 300–599 | 30 / 32 |
| VIRAT_S_010002_02_000174_000204 | 267–639 | 267–299 | 300–599 | 28 / 32 |
| VIRAT_S_010003_01_000111_000137 | 161–533 | 161–299 | 300–533 | 32 / 32 |
| VIRAT_S_010003_02_000165_000202 | 469–841 | – | 469–599 | 11 / 32 |
| VIRAT_S_010003_03_000219_000259 | 586–958 | – | 586–599 | 2 / 32 |
| VIRAT_S_010003_05_000499_000523 | 164–536 | 164–299 | 300–536 | 32 / 32 |
| VIRAT_S_010003_06_000526_000560 | 301–673 | – | 301–599 | 25 / 32 |
| VIRAT_S_010003_07_000608_000636 | 122–494 | 122–299 | 300–494 | 32 / 32 |
| VIRAT_S_010003_10_000901_000934 | 265–637 | 265–299 | 300–599 | 28 / 32 |
| VIRAT_S_010004_03_000239_000277 | 517–889 | – | 517–599 | 7 / 32 |
| VIRAT_S_010005_00_000048_000075 | 175–547 | 175–299 | 300–547 | 32 / 32 |
| VIRAT_S_010005_02_000177_000203 | 243–615 | 243–299 | 300–599 | 30 / 32 |
| VIRAT_S_010005_04_000299_000323 | 120–492 | 120–299 | 300–492 | 32 / 32 |

  In total, 352 of the 1,328 sampled frames lie in ranges analysed before. Five clips are entirely inside them.
  Those frames were analysed as native-rate encodes, with native-rate vectors and 1/24–1/30 s truth. The 2 fps
  encodes, their 0.5 s vectors and 0.5 s truth are new. The scenes and the native-rate results on these frames are
  known to us. This is stated wherever these clips are reported. The 11 unseen-window clips are also reported as a
  subgroup (descriptive, no verdict).
- The windows differ in length (2, 3 or 5 GOPs). Every metric is a per-clip ratio, so clips are comparable. The GOP
  count is reported beside each clip.

## 2. Encodes (2 fps, 16-frame GOP)
- **Input:** the sampled source frames s_j as decoded yuv420p planes (`run_virat.decode_source`, display order,
  no RGB round trip), handed to the encoder as consecutive frames of a 2 fps stream (stream rate 2/1; for the
  23.97 fps clips the true spacing is 0.5005 s). The source indices and a sha256 of the sampled frames are recorded.
- **GOP:** keyint 16 (I-frame every 16 frames = 8 s), no scene-cut I-frames.
  - x264: `keyint=16:min-keyint=16:scenecut=0` (as `lib/encode.build_options`). x264 clamps min-keyint to
    keyint/2 + 1, so its SEI will show keyint_min = 9. With scenecut = 0 this has no effect (as at keyint 30 in
    `p3/sweep/SETTINGS.md`).
  - NVENC: `g=16`. Adaptive scene-cut I-frames need rc-lookahead, which is off in these arms. We also pass
    `no-scenecut=1`. This needs one new optional knob in `build_options` (default None = previous behaviour, as
    every knob added after the first validation).
  - Every encode is checked: I-frames exactly at 0, 16, 32, … A mismatch is logged as a check mismatch and listed.
    The refresh rule (§4) follows the actual I-frames in any case.
- **Arms:** the 9 arms of the confirmatory test, with the same knobs except keyint (and the NVENC scene-cut flag
  above):
  - x264 CRF 12 and CRF 45; x264 QP 24, bf 0; and matched-B x264 QP 24, bf 2 (`x264_qp24_bf2_pb1`: b-adapt 0,
    no pyramid, pbratio 1.0). All x264 arms use preset medium, ref = 1, threads = 1, sliced-threads = 0.
  - NVENC QP 18, 23 and 45; NVENC QP 28 bf 0; and NVENC QP 28 bf 2 B=P (`nvenc_qp28_bf2_bqeq`: b_qfactor 1.0,
    b_qoffset 0, b_ref_mode disabled). All NVENC arms use constqp, preset p4, refs 1.
- **Checks, as `run_virat.check`:** slices per frame = 1; x264 SEI threads 1, sliced_threads 0, scenecut 0,
  keyint 16, ref 1, bframes (b_adapt, b_pyramid for bf 2); no B-frames in bf 0 arms; decoded QP as `QP_REQ`. Also
  logged: kbps, PSNR-Y, I/P/B counts.
- 27 clips × 9 arms = 243 encodes, 11,952 encoded frames (108 x264 encodes on the CPU, 135 NVENC encodes on the RTX
  4050).

## 3. Vectors and truth
**Vectors.** `lib/mvs.extract_mvs` with PyAV's default threading, i.e. no frame threading ("single-threaded", as in
the confirmatory test and `p3/sweep/DEVIATIONS.md` 1). Painting onto the 4×4 cell grid and q = floor(4|MV|)
(255 = no vector) are as in `run_virat.cell_arrays`. The npz layout is the confirmatory one, so
`p3/codecsight_sim/simulate.py`'s loaders read it unchanged. B-frame vectors are used naively, as before.

**Truth (T).** torchvision `raft_large` (DEFAULT weights, the confirmatory checkpoint, sha256 ff5fadd5…), 12
updates, float32, batch 1. The input is the decoded source frames s_{j−1} → s_j as rgb24, at 0.5 s spacing and
full 1280×720. Per pair j = 1..n−1, the flow is forward-splatted onto frame j's 4×4 grid
(`lib/gt_grid.GTGrid(flow, no bad mask).cell_gt()`), as in the confirmatory test. A cell is valid at j if
splat_count ≥ 8 (cover ≥ 0.5). RAFT is an estimate; it is called "RAFT-estimated" everywhere.

**RAFT at 0.5 s: checks, computed from RAFT, the source and the x264 CRF 12 encode only, before any policy, gate
or stale quantity.** Motion over 0.5 s is 12–15 times larger than over one native frame, with more occlusion and blur.
RAFT's all-pairs correlation has no hard search limit, so large motion is not cut off. It may still be wrong.
- **R-chain (gate for every verdict).** For the first 4 sampled pairs after each I-frame (j = 16g + 1..16g + 4), RAFT
  is also run at native rate over the k source pairs between s_{j−1} and s_j. The native flows are composed along
  each pixel's path: F ← F + f_i(x + F), sampled bilinearly. A pixel whose path leaves the frame is dropped.
  Native-rate RAFT on these clips passed V0 in the confirmatory test (26/27 clips), so the chain carries that check
  to the 0.5 s direct flow D.
  - Per clip: the median relative endpoint error r = |D − F| / |F| over pixels with |F| > 2 px, pooled over the clip's
    chain pairs.
  - **PASS if r < 0.20 in ≥ 75% of the clips counted in §5** (ceil(0.75 N)).
  - The median absolute |D − F| is also reported.
  - Chaining adds its own drift over 12–15 steps, so a failure means at least one of the two estimates is
    unreliable at this spacing. We do not say which one.
- **V0-2fps (mirror of 45f3e98; reported, see interpretation).** x264 CRF 12, every P-frame of the 2 fps stream
  (d = 1). Cells must be valid, have |RAFT| > 2 px, have a vector, and have texture ≥ the 75th percentile of cell
  texture (`analyze_virat.cell_texture`) over all cells of the sampled frames 1..n−1 of all 27 windows.
  - Per clip: the median |MV − RAFT|, with float vectors re-extracted.
  - PASS if < 1.5 px in ≥ 75% of the counted clips.
  - The same check is also reported on cells with 2 < |RAFT| ≤ 16 px. x264 medium searches ±16 px around its
    predicted vector, so larger 0.5 s motions can be missed by the encoder, not by RAFT.
- **Forward–backward consistency (robustness, not a gate).** RAFT is also run backward (s_j → s_{j−1}). The
  cell-level check of `p3/review_fixes2/i08b_fb_consistency.py`: |f + b_w|² < 0.01(|f|² + |b_w|²) + 0.5.
  - Reported: consistent share of valid cells and of moving cells (|T| > 2 px). At native rate it was 0.9995 and
    0.82 (medians).
  - All verdict metrics are recomputed with inconsistent cell-frames set to invalid. In the summed truth this drops
    the sum, by the finite-terms rule.
- Also reported: the share of moving cells with |T| > 16, 32 and 64 px.

**If RAFT fails at 0.5 s:**

| R-chain | V0-2fps | consequence |
|---|---|---|
| PASS | PASS | verdicts as §6 |
| PASS | FAIL | verdicts as §6. The paper also reports that x264 CRF 12 vectors disagree with RAFT at 0.5 s on textured moving cells. This is a statement about the vectors (e.g. motion beyond the search range); the ≤ 16 px split is shown beside it. |
| FAIL | any | **no verdict.** Every number is reported as descriptive, under the heading "RAFT unreliable at 0.5 s". The paper says CodecSight's regime could not be tested with our truth. |

This departs from the earlier pre-registrations, where V0 alone gated the verdicts. At 0.5 s V0 is confounded by
the encoder's search range, and R-chain is not. **This choice needs approval.**

## 4. Policy, gate and metrics
**Policy (P): exactly §2 of `p3/codecsight_sim/PREREGISTRATION.md`** (9f5b5a3), as implemented in `simulate.py`,
native-rate path. Here "native" is the 2 fps stream.
- Signal: MV only.
- Cells: painted 4×4 cells.
- Tokens: the derived InternVL3 16×16 token grid (80 × 45 px on 1280×720), with overlap membership.
- A token is dynamic if max |MV| ≥ τ over its cells, or if any cell has no vector.
- The active set is the union of dynamic tokens since the last refresh.
- Refresh at every I-frame and 16 frames after the previous refresh. With keyint 16 these are the same frames:
  0, 16, 32, … (8 s apart).
- Reused = not a refresh frame and not active.
- τ = 0.5 px primary (v4 default), τ = 1 px secondary, τ = 0.25 px (v3 default) descriptive.

**Truth for the policy:** S_j(c) = Σ_{i=r+1..j} T_i(c), summed at the fixed cell since the last refresh r.
- **Limitation.** S is summed at a fixed cell, not along the content's path (as in 9f5b5a3 §3), and here over up to
  15 steps of 0.5 s (7.5 s; the refresh period is 16 frames = 8 s). At native rate the same sum spanned 0.53–0.67 s. Content
  moves farther from the cell over 8 s, so this is a much coarser approximation: it can miss motion that leaves
  the cell, count motion that enters it, and cancel motion that reverses. The forward–backward recomputation (§3)
  is the robustness check for this. It drops every sum that contains an inconsistent 0.5 s step.
- Validity uses the per-frame cover rule plus the finite-terms rule.
- A token is valid if ≥ half its cells are valid. It has truly moved if some valid cell has |S| > 2 px.
- Scored frames: j = 1..n−1 of the 2 fps stream.
- REUSE, STALE and STALE_MOV are defined as in that file's §4. B-frame-only ratios are reported for the bf 2 arms.

**Bare per-cell gate (descriptive):** as in that file's §4.
- Reuse if the cell has a vector and |MV| < τ.
- Per-frame truth: |T_j| > 2 px.
- Same frames and τ.
- Reported: REUSE, STALE and STALE_MOV.

**Common-truth metrics (descriptive; definitions of `p3/codecsight_sim/posthoc_matched/PLAN.md` §2–4,
e311c83):**
- Cell level, for P (derived, cell state from the token holding the cell centre), G1 (per-cell gate, no
  accumulation) and G2 (derived token gate, no accumulation), at τ 0.5 and 1.
- One truth for all three: the sum since the cell was last computed.
- Reported: REUSE_c and STALE_c.
- Also **STALE_c / MOVE_c**, where MOVE_c = valid cell-frames with |T_j| > 2 px / valid cell-frames (per-frame cover
  rule, scored frames). MOVE_c is the same for every method on an encode.
  - The ratio is stale cell-frames per unit of this-frame motion. It puts clips with different amounts of motion on
    one scale.
  - It is not a share. It can exceed 1, because STALE_c uses summed truth.
- The matched-reuse interpolation is not part of this arm.

**Summaries:** per-arm medians over clips. For K1–K4: the median per-clip paired difference, with 95% bootstrap CIs.
Method and seed are as in that file's §4: rng 20261006; draws in the order (1) scene indices (10000, n_scenes),
(2) clip indices (10000, n_clips), over the clips counted in §5; scene-cluster and clip-level. Intervals at τ 0.5
and 1.

## 5. Clips counted for verdicts
- **2 fps motion count**, computed from RAFT only, before any policy or gate metric: valid cell-frames with
  |T_j| > 2 px over j = 1..n−1. This is the confirmatory test's eligibility count, on the 2 fps pairs.
- A clip is counted if this is ≥ 2,000, the confirmatory threshold. Percent thresholds in §6 are of the counted
  clips (N).
- If N < 10, everything is descriptive.
- Clips not counted are listed and still reported.

## 6. Predictions (fixed before any number of this arm is seen)
Primary configuration: derived token grid, τ = 0.5 px, all scored frames. Verdict metric: policy STALE_MOV.
τ = 1 px is evaluated the same way and reported as secondary. Mirrors of K1–K4 and KILL-A of
`p3/codecsight_sim/PREREGISTRATION.md` §6:

| key | mirrors | comparison (per clip) | margin | required count |
|---|---|---|---|---|
| F1 | K1 / C1 | STALE_MOV(x264 CRF 45) − STALE_MOV(x264 CRF 12) | > 0.01 | ≥ 80% of N |
| F2 | K2 / C2 | STALE_MOV(NVENC QP 45) − STALE_MOV(NVENC QP 18) | > 0.01 | ≥ 75% of N |
| F3 | K3 / C3 | STALE_MOV(x264 QP 24 bf 2) − STALE_MOV(x264 QP 24 bf 0) | > 0.01 | ≥ 80% of N |
| F4 | K4 / C4 | STALE_MOV(NVENC QP 28 B=P) − STALE_MOV(NVENC QP 28 bf 0) | > 0.01 | ≥ 75% of N |
| KILL-A2 | KILL-A | all 9 arms: median STALE_MOV < 0.01 **and** median REUSE > 0.20 | – | 9/9 arm-medians |

With N = 27 the counts are 22, 21, 22, 21.

**Kill rules, as before:**
- F1 and F2 both fail: in this regime the quality effect does not carry over to the policy.
- Exactly one of them fails: the effect is encoder-specific.
- F3 or F4 fails: the B-frame claim does not hold in this regime for that encoder.

**Decision rule, on the 9 CCTV arms at the primary configuration (only if R-chain passes):**
- **"The policy stays fresh in its own regime"** if KILL-A2 fires.
- **"It leaks"** if, for at least one arm, median STALE_MOV > 0.01 and median REUSE > 0.20. The leaking arms are named.
- **Neither:** no arm leaks, but some arm has median REUSE ≤ 0.20. That arm is reported as "policy reuses little".
  Its low STALE_MOV is not counted as freshness. The two named outcomes exclude each other.

**What each outcome means for the paper:**
- **Fresh (KILL-A2 fires).** The paper states that CodecSight's documented policy, run as CodecSight runs it
  (2 fps, 8 s GOP, τ 0.5), absorbs the encoder effect on our CCTV data. It reuses more than 20% of tokens with under
  1% of truly-moved tokens stale, at every setting we tried.
  - The paper's encoder-dependence claim is then about per-cell MV gates and about MV reuse at native rate. It does
    not claim that CodecSight's deployment is unsafe.
  - F1–F4 still say whether the residual stale share moves with the encoder.
- **Leaks.** In CodecSight's own regime, at least one standard encoder setting makes the documented policy reuse
  content that has truly moved.
  - The paper names the arms and reports F1–F4. If they pass, the leak tracks the quantiser or B-frames, i.e. the
    encoder effect reaches the deployment CodecSight targets. If they fail, the leak exists but is not shown to
    depend on the encoder.
  - The paper does not claim an accuracy loss: no VLM is run.
- **Neither.** At 0.5 s spacing the union fills most of the token grid before the refresh for some arm, so the
  policy is safe there by reusing little. The paper reports this as no evidence either way for that arm: neither
  absorption nor leak is shown.
- **No verdict (R-chain fails).** The paper says this regime could not be tested with RAFT truth, and keeps the
  native-rate results with their stated gap.

Nothing here changes a verdict of the main pre-registration, the confirmatory test or the CodecSight simulation.

**Descriptive:**
- the 11 unseen-window clips as a subgroup (F1–F4 counts and KILL-A2 medians);
- per-scene tables;
- τ = 0.25 px;
- the bare gate and the common-truth metrics;
- B-frame-only ratios;
- the share of tokens active before each refresh;
- token-frames lost to the finite-terms rule;
- FB-consistent recomputation;
- bitrate and PSNR per arm.

## 7. Hardware and runtime estimate
NVENC and RAFT need this laptop's RTX 4050 (6 GB; driver 610.74). Confirmatory RAFT at 1280×720 peaked at 2.6 GB.
Per-unit costs come from `p3/virat_confirm/manifest.json` (native rate, same machine). Counts come from
`inventory.txt`.

| step | work | unit cost | estimate |
|---|---|---|---|
| decode sources (CPU path, GPU path, RGB for RAFT) | 3 × 28,565 frames (decode from frame 0 to window end) | ~300 frames/s | ~5 min CPU, ~2 min wall |
| x264 encodes | 108 encodes, 5,312 frames | 0.134 s per sampled frame over the 4 arms; ×2 allowance for 0.5 s motion | ~6 min CPU, ~1 min wall (7 workers) |
| NVENC encodes | 135 encodes, 6,640 frames | ~0.003 s/frame + session start | ~2 min GPU |
| MV extraction | 243 encodes, 11,952 frames | 0.0075 s/frame | ~1.5 min CPU |
| RAFT forward, 2 fps | 1,301 pairs | 0.61 s/pair | ~13 min GPU |
| RAFT backward, 2 fps | 1,301 pairs | 0.61 s/pair | ~13 min GPU |
| RAFT native, R-chain | 4,308 pairs | 0.61 s/pair | ~44 min GPU |
| chain composition, splats | 4,308 + 2,602 flow fields | numpy | ~5 min CPU |
| analysis (policy, gate, common truth, checks) | 243 small encodes | – | < 10 min CPU |

**Total: about 1.5 h wall**, about 70 min of it RAFT on the GPU. The encode phase overlaps CPU and GPU. Disk:
under 1 GB of data and encodes. The x264 per-frame cost at 2 fps is the least certain number; even at 4× it adds
under 5 minutes.

## 8. Rules
- Fixed by this file. No tuning after results.
- Bugs: fix, record in `p3/codecsight_2fps/DEVIATIONS.md`, rerun everything, report both runs.
- Per-clip results are primary; per-scene results are also shown.
- "x% of N" means at least ceil(x·N). A missing value counts as not meeting the condition and is listed.
- Data, encodes and caches are never committed.
