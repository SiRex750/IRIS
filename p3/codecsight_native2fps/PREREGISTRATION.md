# Paper 3 — Pre-registration: CodecSight's regime, reading B (native-rate encode, 2 fps sampling), CCTV
Date: 2026-10-17. Approved in review on 2026-10-17 (§5 counting rule on native pairs confirmed). Written before any encode, RAFT run or metric of this arm. The only script
run so far is `inventory.py` (this directory), which reads `p3/codecsight_2fps/inventory.csv` and derives window
lengths, I-frame positions and pair counts from frame counts. It decodes nothing and reads no pixel value, motion
vector or flow value. Repository HEAD when written: e774ae8 (branch sonu/p3). Collection and analysis code are written
after this file is committed; their commit hashes are recorded in the results.

What this is: **a second reading of CodecSight's regime**, tested with the documented policy of
`p3/codecsight_sim/PREREGISTRATION.md` (9f5b5a3). It is still not CodecSight: no VLM is run and no task accuracy is
measured. Nothing here changes a verdict of any earlier pre-registration.

## 1. Source and the two readings
Paper: Zou et al., "CodecSight", arXiv:2604.06036 (CC BY 4.0). v4 (15 Sep 2026, current) is followed. Quotes are v4:
- §5: "Videos are reencoded with an 8 s GOP interval"
- §5: "inference samples frames at 2 FPS"
- §6: "16-frame GOP"
- §6: "MV threshold of 0.5"

The paper states an 8 s GOP interval and a 16-frame GOP, and says samples are taken at 2 FPS. It does not say at what
frame rate the re-encode runs. Two readings fit the text:
- **Reading A (cc2eaf9).** The re-encode is at 2 fps: the encoder sees only the sampled frames, 16 frames = 8 s,
  vectors span 0.5 s. Collected and analysed in `p3/codecsight_2fps/`. Its pre-registered gate (R-chain) FAILED
  (1/22 counted clips), so every number of reading A is descriptive and **there is no verdict**. This file does not
  revise that outcome and does not use reading A's numbers as evidence.
- **Reading B (this file).** The re-encode keeps the native frame rate, with an I-frame every 8 s
  (192 frames at 23.97 fps, 240 at 30 fps). Inference samples every 12th (15th) decoded frame, i.e. 2 FPS. "16-frame
  GOP" then counts sampled frames: an I-frame every 16 sampled frames. Vectors are native-rate (1/24–1/30 s). This is
  the stream a camera or transcoder at native rate would produce, with the sampling done at the decoder.
- Neither reading is established by the paper. Reading B is tested because reading A's truth could not be validated
  and reading B's can: native-rate RAFT passed V0 in the confirmatory test (26/27 clips).
- **τ.** v4's default is 0.5 px (primary here). v3's default was 0.25 px (reported, descriptive).

**What is already known to us.** The CodecSight simulation's exploratory 2 FPS arm (9f5b5a3 §7, variants a and b)
used the same policy and the same two variants on saved native-rate encodes (keyint 250), frames 300–599. Its results
are known (`p3/codecsight_sim/verdicts.md`): median STALE_MOV per arm 0.22–0.74 (a) and 0.03–0.62 (b). So for variant
(a) the outcome of KILL-A is largely foreseeable from that arm. What is new here: GOP-matched encodes (I-frame every
8 s, aligned with the sampled frames), windows reaching frames 600 and later for 11 clips, a V0 gate on these frames,
and pre-registered verdicts for each variant.

## 2. Data
- **Clips and windows: exactly cc2eaf9's.** The same 27 eligible clips (`p3/virat_confirm/eligibility_v0.csv`) and the
  same source-frame windows (`p3/codecsight_2fps/inventory.csv`; window rule of cc2eaf9 §1: frames 600+ with up to
  5 GOPs where they fit, else end-aligned 2 GOPs). The sampled frames are the same s_j = start + j·k,
  j = 0..16G−1, k = 12 (20 clips) or 15 (7 clips). The native window runs from s_0 to s_{16G−1}: every native frame in it
  is encoded and given truth.
- **Result (`inventory.txt`).** 11 unseen-window clips (9 with 5 GOPs, 2 with 3) and 16 end-aligned 2-GOP clips.
  16,914 native frames per arm, 1,328 sampled frames, 16,887 native pairs.
- **Overlap with frames analysed before, disclosed as in cc2eaf9 §1.** 16 clips' windows lie partly in frames 0–299
  (exploratory `p3/virat`) or 300–599 (confirmatory `p3/virat_confirm` and the CodecSight simulation): 352 of 1,328
  sampled frames. The encodes (keyint 192/240, windows re-cut) and the 8 s GOP are new. The scenes and native-rate
  results on these frames are known to us. The 11 unseen-window clips are also reported as a subgroup (descriptive,
  no verdict).
- Window lengths differ (2, 3 or 5 GOPs); every metric is a per-clip ratio. The GOP count is reported per clip.

## 3. Encodes (native frame rate, 8 s GOP)
- **Input:** the native frames of the window as decoded yuv420p planes (`run_virat.decode_source`, display order, no
  RGB round trip), at the source frame rate (23.97 or 30 fps). Source indices and a sha256 are recorded.
- **GOP:** keyint = 8 s = 16k frames (192 at k = 12, 240 at k = 15), min-keyint the same, no scene-cut I-frames.
  - x264: `keyint=K:min-keyint=K:scenecut=0`. x264 clamps min-keyint to K/2 + 1, so its SEI shows
    keyint_min = K/2 + 1; with scenecut = 0 this has no effect (as in cc2eaf9 §2).
  - NVENC: `g=K` and `no-scenecut=1` (the knob added for cc2eaf9, default None).
  - Every encode is checked: I-frames exactly at 0, K, 2K, … (so exactly at sampled positions 0, 16, 32, …). A
    mismatch is a check mismatch and is listed.
- **Arms and controls: the 9 arms of the confirmatory test**, same knobs except keyint: x264 CRF 12 and CRF 45; x264
  QP 24 bf 0; matched-B x264 QP 24 bf 2 (`x264_qp24_bf2_pb1`); NVENC QP 18, 23, 45; NVENC QP 28 bf 0; NVENC QP 28 bf 2
  B=P (`nvenc_qp28_bf2_bqeq`). All x264 arms: preset medium, ref 1, threads 1, sliced-threads 0. All NVENC arms:
  constqp, preset p4, refs 1, rc-lookahead off.
- **Checks:** as cc2eaf9 §2 (slices per frame 1, SEI fields, no B-frames in bf 0 arms, decoded QP as requested, no
  unconsumed options). Logged: kbps, PSNR-Y, I/P/B counts. At native rate the CRF arms are QP-equivalent to the
  confirmatory CRF arms (the issue of `p3/codecsight_2fps/DEVIATIONS.md` 1 does not arise).
- 27 × 9 = 243 encodes (108 x264, 135 NVENC), 152,226 encoded frames.

## 4. Vectors, policy, truth
**Vectors.** `lib/mvs.extract_mvs`, PyAV default threading (single-threaded, as the confirmatory test). Painting to the
4×4 cell grid and q = floor(4|MV|) as `run_virat.cell_arrays`; the npz layout is the confirmatory one. B-frame vectors
are used naively, as before. The policy at a sampled frame s_j uses the vectors of frame s_j itself (frame s_j − 1 →
s_j at native rate). Skipped frames have vectors too, used only in variant (b).

**Policy P: exactly §2 of 9f5b5a3**, as implemented in `simulate.py`, applied to the sampled frames:
- signal MV only; 4×4 cells; derived InternVL3 16×16 token grid (80 × 45 px on 1280×720) with overlap membership;
- a token is dynamic if max |MV| ≥ τ over its cells or any cell has no vector;
- active set = union of dynamic tokens since the last refresh;
- refresh at every I-frame and 16 sampled frames after the previous refresh. With keyint = 16k these coincide:
  sampled positions 0, 16, 32, … (8 s apart);
- reused = not a refresh frame and not active;
- τ = 0.5 px primary, 1 px secondary, 0.25 px descriptive.

**Two pre-registered variants, each with its own verdicts:**
- **(a) Sampled vectors only.** The dynamic set at s_j comes from frame s_j's vectors; the union accumulates over
  sampled frames only.
- **(b) Union over skipped frames.** As (a), and at s_j the union also includes every token dynamic (same rule, same τ,
  each frame's own vectors, no-vector cells included) in any skipped native frame since the previous sampled frame
  (s_{j−1}+1 … s_j−1). At a refresh frame the active set is cleared as in (a): skipped frames before a refresh do not
  carry over. Because I-frames sit at sampled positions, no skipped frame is an I-frame; this is asserted.

**Truth.** torchvision `raft_large` (DEFAULT weights, sha256 ff5fadd5…), 12 updates, float32, batch 1, decoded source
frames as rgb24, full 1280×720. RAFT is run on **every native pair** i−1 → i, i = 1..L−1, of every window (L = native
frames of the window). Each flow is forward-splatted onto frame i's 4×4 grid (`lib/gt_grid.GTGrid(flow, no bad
mask).cell_gt()`); a cell is valid at i if splat_count ≥ 8 (cover ≥ 0.5). RAFT is an estimate; it is called
"RAFT-estimated" everywhere. At native rate this is the confirmatory truth; no composition and no 2 fps RAFT are used,
so reading A's R-chain is not needed and is not run.
- **Sum for the policy:** S_j(c) = Σ_{i=r+1..s_j} T_i(c), summed at the fixed cell since the last refresh r (native
  index), over every native frame including the skipped ones: up to 191 steps at k = 12 and 239 at k = 15.
- **Limitation (as cc2eaf9 §4, stated again).** S is summed at a fixed cell, not along the content's path. Over the
  8 s refresh period content moves far from the cell, so S can miss motion that leaves the cell, count motion that
  enters it, and cancel motion that reverses. It is also a sum of up to 239 separate RAFT estimates, so any small
  per-frame RAFT bias adds up. A token "truly moved" threshold of |S| > 2 px therefore may be crossed by estimator
  drift. The 9f5b5a3 §7 arm used the same sum (at keyint 250). A drift report (§6) shows how |S| grows with lag, but
  it gates nothing.
- Validity: the per-frame cover rule plus the finite-terms rule (every term finite). A token is valid if ≥ half its
  cells are valid; it has truly moved if some valid cell has |S| > 2 px. Scored frames: the sampled frames j = 1..n−1.
  REUSE, STALE and STALE_MOV are defined as in 9f5b5a3 §4. B-frame-only ratios are reported for the bf 2 arms.

**Gate: V0 at native rate on these frames, exactly as d62753f** (V0 of 45f3e98). x264 CRF 12, every native P-frame
(d = 1), cells valid, |RAFT| > 2 px, a vector present, texture ≥ the 75th percentile of cell texture
(`analyze_virat.cell_texture`) over all cells of native frames 1..L−1 of all 27 windows. Per clip, the median
|MV − RAFT| (float vectors re-extracted). **PASS if < 1.5 px in ≥ ceil(0.75 N) of the counted clips** (§5). A clip
with no V0 cell does not pass. **If V0 fails, every number of this arm is descriptive** under the heading "RAFT
unreliable", for both variants. V0 is computed from RAFT, the source and the CRF 12 encode only, before any policy
metric.

## 5. Clips counted for verdicts
- **Counting rule (RAFT only, before any policy or gate metric):** valid cell-frames with |T_i| > 2 px, over all native
  pairs i = 1..L−1 of the window. A clip is counted if this is ≥ 2,000, the confirmatory eligibility threshold. This
  is the same count as the confirmatory test, on these windows; it differs from cc2eaf9 §5, which counted on its
  scored 2 fps pairs. (Counting on the sampled-frame pairs only would use 1/12–1/15 of the pairs and make the 2,000
  threshold a different test. This choice was approved.)
- Percent thresholds are of the counted clips (N). If N < 10, everything is descriptive. Clips not counted are listed and
  still reported. N is the same for both variants.
- A missing value (empty denominator) counts as not meeting a condition and is listed.

## 6. Predictions (fixed before any number of this arm is seen)
Primary configuration: derived token grid, τ = 0.5 px, all scored (sampled) frames, verdict metric policy STALE_MOV.
τ = 1 px is evaluated the same way and reported as secondary. The predictions below are evaluated **separately for
variant (a) and variant (b)**; each variant has its own F1–F4, KILL-A and decision. Mirrors of K1–K4 and KILL-A of
9f5b5a3 §6, and of F1–F4 and KILL-A2 of cc2eaf9 §6:

| key | comparison (per clip) | margin | required count |
|---|---|---|---|
| F1 | STALE_MOV(x264 CRF 45) − STALE_MOV(x264 CRF 12) | > 0.01 | ≥ 80% of N |
| F2 | STALE_MOV(NVENC QP 45) − STALE_MOV(NVENC QP 18) | > 0.01 | ≥ 75% of N |
| F3 | STALE_MOV(x264 QP 24 bf 2) − STALE_MOV(x264 QP 24 bf 0) | > 0.01 | ≥ 80% of N |
| F4 | STALE_MOV(NVENC QP 28 B=P) − STALE_MOV(NVENC QP 28 bf 0) | > 0.01 | ≥ 75% of N |
| KILL-A | all 9 arms: median STALE_MOV < 0.01 **and** median REUSE > 0.20 | – | 9/9 arm-medians |

With N = 27 the counts are 22, 21, 22, 21. "x% of N" means at least ceil(x·N).

**Kill rules, as before, per variant:** F1 and F2 both fail: the quality effect does not carry to the policy in this
regime. Exactly one fails: the effect is encoder-specific. F3 or F4 fails: the B-frame claim does not hold in this
regime for that encoder.

**Decision rule, on the 9 arms at the primary configuration (only if V0 passes), per variant:**
- **"The policy stays fresh"** if KILL-A fires.
- **"It leaks"** if, for at least one arm, median STALE_MOV > 0.01 and median REUSE > 0.20. The leaking arms are named.
- **Neither:** no arm leaks, but some arm has median REUSE ≤ 0.20 ("policy reuses little"; its low STALE_MOV is not
  counted as freshness). The three outcomes exclude each other.

**Two variants, no multiplicity correction, no designated primary.** The paper reports both. If their decisions
differ, the paper says the result depends on whether the vectors of skipped frames are read, and does not choose the
more favourable one. The 16 verdict-bearing comparisons (2 variants × 4 + 2 KILL-A) are not adjusted.

**What each outcome means for the paper (per variant):**
- **Fresh.** Under reading B (native-rate encode, 8 s GOP, 2 FPS sampling, τ 0.5) the documented policy reuses more
  than 20% of tokens with under 1% of truly-moved tokens stale at every setting tried. The paper's encoder-dependence
  claim is then about per-cell MV gates and native-rate reuse, not about this deployment. F1–F4 still say whether the
  residual stale share moves with the encoder.
- **Leaks.** At least one standard encoder setting makes the policy reuse content that has truly moved under reading
  B. The paper names the arms and reports F1–F4. If they pass, the leak tracks the quantiser or B-frames; if they
  fail, the leak is not shown to depend on the encoder. No accuracy loss is claimed.
- **Neither.** Safe by reusing little for some arm: no evidence either way for that arm.
- **V0 fails.** The regime could not be tested with RAFT truth even at native rate on these frames; all numbers are
  descriptive.
- Reading A and reading B are reported side by side. Reading A has no verdict. Only reading B (if V0 passes) supports
  a verdict statement, and the paper says that it is a statement about reading B, not about CodecSight's deployment,
  whose encode frame rate the paper does not give.

**Descriptive (no verdicts):**
- the 11 unseen-window clips as a subgroup (F1–F4 counts and KILL-A medians, per variant);
- per-scene tables; τ = 0.25 px;
- the bare per-cell gate and the common-truth metrics (STALE_c, REUSE_c, STALE_c / MOVE_c) as defined in cc2eaf9 §4,
  on the sampled frames;
- B-frame-only ratios; the share of tokens active before each refresh; token-frames lost to the finite-terms rule;
- bitrate and PSNR per arm;
- **drift report:** the median and 95th percentile of |S| over valid cells, by lag since the last refresh (1, 12, 48, 96,
  192, 239 native frames), per clip; a gate for nothing;
- reading B against reading A at the same sampled source frames, per arm (medians only);
- paired bootstrap CIs for F1–F4 as cc2eaf9 §4: rng 20261006, draws (1) scene indices (10000, n_scenes), (2) clip
  indices (10000, n_clips), over the counted clips, scene-cluster and clip-level, at τ 0.5 and 1.
- Forward–backward consistency is not run in this arm (native-rate FB was reported earlier at 0.9995 / 0.82).

## 7. Hardware and runtime estimate
RTX 4050 (6 GB; driver 610.74); RAFT at 1280×720 peaked at 2.6 GB. Counts from `inventory.txt`; unit costs from
`p3/virat_confirm/manifest.json` and cc2eaf9 §7.

| step | work | estimate |
|---|---|---|
| RAFT, forward, every native pair | **16,887 pairs × 0.61 s** | **172 min (2.9 h) GPU** |
| decode (CPU path, GPU path, RGB for RAFT) | 3 × 28,565 frames (decode from frame 0 to window end) | ~5 min |
| x264 encodes | 108 encodes, 67,656 frames | ~38 min CPU (~6–10 min wall on 7 workers) |
| NVENC encodes | 135 encodes, 84,570 frames | ~4 min GPU |
| MV extraction | 243 encodes, 152,226 frames | ~19 min CPU |
| analysis (policy ×2 variants, gate, V0) | – | < 15 min CPU |

**Total: about 3.5 h wall**, 2.9 h of it RAFT. Encodes, decode and MV extraction overlap with RAFT only partly. Disk:
about 8 GB for the per-clip RAFT cell grids and 243 native encodes (never committed). The least certain number is the
x264 cost per native frame (assumed 0.034 s, a quarter of cc2eaf9's 0.134 s per sampled frame over four arms); even at
4× it adds under 40 min of CPU wall, mostly hidden behind RAFT.

## 8. Rules
- Fixed by this file. No tuning after results.
- Bugs: fix, record in `p3/codecsight_native2fps/DEVIATIONS.md`, rerun everything, report both runs.
- Per-clip results are primary; per-scene results are also shown.
- Data, encodes and caches are never committed.
