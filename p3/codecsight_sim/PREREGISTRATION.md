# Paper 3 — Pre-registration: CodecSight policy simulation (saved data only)
Date: 2026-10-07. Approved in review on 2026-10-07 with four edits (§1/§2 gap, §7 2 FPS label and variant (b), §4 bare gate, §4 summaries and CIs). Written before any metric of this simulation was computed. The only
script run so far is `inventory.py`, which lists files, frame counts, frame types and I-frame positions (it reads
no motion vector or flow value). Repository HEAD when written: 0e22626 (branch sonu/p3). Analysis code is written
after this file is committed; its commit hash is recorded in the results.

What this is: **CodecSight's documented policy applied to our data.** It is not CodecSight. No VLM is run, no
encode is made, no task accuracy is measured. The question is the same as the main paper's: when encoder
settings change, does a codec-MV reuse policy reuse more content that has truly moved? Here the policy is the one
CodecSight documents (MV-only selection, 2x2 token grouping, accumulation within a GOP), not the bare per-cell gate.

## 1. Source: what the paper says
Paper: Zou et al., "CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference",
arXiv:2604.06036. License CC BY 4.0. Quotes are copied verbatim from the arXiv HTML. Equations are copied from the
HTML's LaTeX (alt text).
- **v3** (9 Apr 2026, titled "CodecSight", system design as in v1–v2; v1 is titled "CoStream") is the version
  with the M = V + αR signal.
- **v4** (15 Sep 2026, current) removes α. It keeps MV-only as the default and makes residuals an optional
  "residual rescue" rule. It also changes the default τ (0.25 in v3, 0.5 in v4).
- Both versions were read in full for the items below. Where they differ, both are given. The policy follows
  **v4** (the current version) wherever they differ, except for the signal, where both agree in effect: MV only.

| item | v3 (9 Apr 2026) | v4 (15 Sep 2026, current) |
|---|---|---|
| Motion signal | §3.3.1, Eq. (3): $M_{t}(i)=V_{t}(i)+\alpha R_{t}(i)$, "where $\alpha$ controls the relative contributions of motion displacement and residual error." | No α. §3.3.1: "By default, CodecSight uses $V_{t}(i)$ alone to determine candidate patches for pruning." Residuals enter only through the optional rescue rule, Eq. (4). |
| Default α = 0 | §3.3.1: "Accordingly, our default hardware-decoded implementation sets $\alpha=0$ and uses motion vectors alone as the pruning signal." | §6.2.4: "We therefore use MV-only pruning by default and expose residual rescue as an optional quality knob for deployments that favor higher accuracy." |
| Dynamic rule | §3.3.2, Eq. (4): $\text{dynamic}(i)=M_{t}(i)\geq\tau$ | §3.3.2, Eq. (3): $D_{t}^{\mathrm{MV}}(i)=\mathbb{I}\!\left[V_{t}(i)\geq\tau_{\mathrm{MV}}\right]$; Eq. (5): $D_{t}(i)=D_{t}^{\mathrm{MV}}(i)\lor D_{t}^{\mathrm{res}}(i)\lor D_{t}^{\mathrm{intra}}(i)$ |
| Block → patch | §3.3.1: "we resample the block-level motion and residual maps onto the patch grid" (rule not given) | §3.3.1: "Each patch receives the maximum motion-vector magnitude among blocks whose projected footprints overlap it" |
| Intra regions | not specified | §3.3.2: "Patches overlapping these regions are retained because the absence of an inter-frame motion vector must not be interpreted as low motion." |
| 2x2 grouping | Fig. 9 caption (§3.3): "Each $(2\times 2)$ group of neighboring patches is projected into a vision token. A token is retained if any of its constituent patches are marked dynamic, otherwise it will be discarded." | Fig. 8 caption (§3.3.2): "Illustration of the MV-guided pruning policy for a VLM with $(2\times 2)$ spatial projection groups. A group is retained if any constituent patch is selected." |
| Union rule | §3.3.2: "the active set of a P-frame is defined as the union of its own detections and those of all preceding P-frames since the last I-frame. Thus, once a patch is marked dynamic, it remains active until the next I-frame resets the mask." | §3.3.2: "For each P-frame, the accumulated mask retains patch locations selected in the current frame or any preceding P-frame since the most recent I-frame. Once selected, a location remains active until the next I-frame resets the accumulation." |
| I-frames | §3.3.2: "I-frames are always fully encoded and provide the reference visual context for subsequent P-frames." | §3.3.2: "CodecSight retains all patches in I-frames, which provide no inter-frame motion vectors for motion-based selection." |
| Default τ | §6: "an MV threshold of 0.25 pixel"; §6.3.2: "We therefore use MV=0.25 in the remaining experiments" | §6: "the 8 s stride, 16-frame GOP, and MV threshold of 0.5 are selected from the sweeps in §6.3" |
| Sampling | §5: "Our evaluation uses a 40-second sliding window sampled at 2 FPS" | §5: "inference samples frames at 2 FPS" |
| GOP | §6: "a GOP size of 16 frames" | §5: "videos are re-encoded with an 8 s GOP interval"; §6: "16-frame GOP" |
| Window / stride | §6: "a stride of 20% of the window size" (40 s window, §5) | §5: "Each request contains a 40 s window (80 frames), and windows advance by 8 s (16 frames)." |
| VLM | §5: InternVL3 (InternViT 300M, Qwen2.5-14B) and Qwen3-VL (Qwen-ViT 600M, Qwen3-32B), Table 2 | §5, Table 2: LLaVA-OneVision (SigLIP ViT-SO 400M, Qwen2-7B), InternVL3 (InternViT 300M, Qwen2.5-14B), Qwen3-VL (Qwen-ViT 600M, Qwen3-32B) |
| Input resolution | §2.2: "each $448\times 448$ frame requires 256 tokens" (InternVL3) | §2.2: "each $448\times 448$ frame produces 256 visual tokens" (InternVL3) |
| B-frames | not specified (H.264 I/P GOP described) | not specified (I-frame + P-frames, §2.4) |

Not specified in either version, so these are **our choices** (given in §2):
- the resize mode (whole frame to one 448×448 input, or tiles)
- the token grid for Qwen3-VL and LLaVA-OV
- the H.264 block-to-patch rule in v3
- whether the 16-frame GOP counts sampled or source frames, and whether the re-encode is at 2 fps. v4 states both
  an "8 s GOP interval" (§5) and a "16-frame GOP" (§6). 16 frames = 8 s implies an encoded stream at 2 frames per
  second. **Our inference** (not stated in the paper) is that CodecSight re-encodes at 2 fps, or counts GOP length
  in sampled frames; see the gap in §2.
- B-frame handling
- what a pruned token's location "holds" for the model.

## 2. Policy (CodecSight's documented policy applied to our data)
- **Signal:** motion vectors only (α = 0 in v3, the MV-only default in v4). No residuals.
- **Cells:** the main paper's 4×4-px cell grid. A cell takes the vector of the codec block painted onto it, with
  painting exactly as in the main sweep (`pilot/run_pilot.py:paint`): block top-left from rint(dst_topleft),
  future-referencing blocks painted first, past-referencing blocks override. A cell with no vector (intra, or not
  covered) is a no-vector cell.
- **Token footprint, primary (derived):** both versions give InternVL3 256 tokens for a 448×448 frame, which is a
  16×16 token grid. *Our choice:* the whole source frame is resized to one 448×448 input (no tiling, aspect not
  kept), so a token covers W/16 × H/16 source pixels. That is Sintel 1024×436 → 64 × 27.25 px; CCTV 1280×720 →
  80 × 45 px. The token rectangle is [j·W/16, (j+1)·W/16) × [i·H/16, (i+1)·H/16).
  - A token's cells are all cells whose 4×4 footprint overlaps its rectangle with positive area (v4 §3.3.1
    overlap rule). A boundary cell can belong to two tokens.
  - No patch size is needed: the 256-token count fixes the grid. Patch size 14 with 2×2 grouping gives the same
    grid, but the paper does not state the patch size.
- **Token footprint, sensitivity (descriptive):** square 16, 32 and 64 px tokens, tiled from (0, 0) on the 4-px
  grid; the last row and column are partial.
- **Block-to-token rule:** at frame t, a token is dynamic if, over its cells, max |MV| ≥ τ, or any cell is a
  no-vector cell (v4 Eq. (3), (5) and the intra rule).
- **τ:**
  - 0.5 px primary. This is v4's default; v3's default is 0.25 px.
  - 1 px secondary.
  - 0.25 px (v3 default) reported, descriptive only.
  - The CCTV stored quantisation (q = floor(4|MV|)) makes all three exact.
- **Union:** within a refresh period, the active set at frame t is the union of the dynamic sets of every frame
  after the last refresh, up to and including t. An active token is recomputed at t. Once active, a token stays
  active until the next refresh.
- **Refresh:** at every I-frame, and 16 frames after the previous refresh, whichever comes first (the counter
  restarts at every refresh). At a refresh frame every token is computed and the active set is cleared.
  - This 16-frame period stands in for CodecSight's 16-frame GOP. Our encodes use keyint 250, so I-frames are
    rare. From the inventory: Sintel I-frame at 0 only (one MPEG-4 encode, ambush_4, also at 3 and 4); CCTV
    I-frames at 0 and 250 (or 0 only for clips under 251 frames).
  - Refresh positions are listed per encode in `inventory.txt`. Sintel: 0, 16, 32, 48. CCTV: 0, 16, …, 240, then
    250, 266, 282, 298.
- **Primary rate:** native, every frame sampled.
- **The gap, stated plainly:** CodecSight samples at 2 FPS with a 16-frame (8 s) GOP. Our data are 24 fps
  (Sintel) and 23.97 or 30 fps (CCTV: 180 and 63 encodes). So:
  - A 16-frame refresh period is 0.53–0.67 s here, against 8 s in CodecSight.
  - v4 gives both an 8 s GOP and a 16-frame GOP, so 16 frames = 8 s implies an encoded stream at 2 frames per
    second. Our inference is that CodecSight re-encodes at 2 fps (or counts GOP length in sampled frames). If so,
    its motion vectors span 0.5 s, against 1/24–1/30 s here. We cannot reproduce 0.5 s vectors without new encodes.
  - **The direction of the resulting bias is unknown.**
    - Longer-span vectors see more of the motion that accumulates before a refresh, which would make the policy
      safer than in our simulation.
    - They are also less accurate (larger search, more intra and mismatched blocks) and are applied over an 8 s
      refresh period, not 0.53–0.67 s, which would make it less safe.
    - We do not predict which effect wins.
  - The primary simulation is therefore not CodecSight's setting. It is the documented policy at native rate with
    a 16-frame refresh. The 2 FPS arm (§7) is a different deployment again, not CodecSight's setup.
- **B-frames:** the same naive handling as the main paper. Every non-I frame (P, or B; P-frames at distance > 1 in
  the bf 2 arms included) uses its raw vectors, not divided by reference distance, painted as above. Frames are
  processed in display order.
- **What "reused" means here:** a token that is not active at a non-refresh frame t is pruned. *Our choice:* the
  model's latest representation of that location is the one computed at the last refresh, so we call it reused.
  KV refresh across LLM windows (v3 §3.4, v4 §3.4) is out of scope.

## 3. Truth
- **Per-frame, per-cell ground-truth vector g_t(c)** (frame t−1 → t, on frame t's 4-px grid), codec-independent:
  - Sintel: `lib/gt_grid.GTGrid(flow_into(t), bad_mask_into(t)).cell_gt()` on the final pass, as in the main
    sweep. Per-frame validity: gt_cover ≥ 0.5. The per-cell field is rebuilt from the Sintel .flo files, because the
    saved block files hold per-block ground truth only. No encode is involved.
  - CCTV: `raft.npz` `splat_t` of the confirmatory collection (row i is t = i + 1). Per-frame validity:
    splat_count ≥ 8 (cover ≥ 0.5), as in the confirmatory test.
- **Summed displacement since the last refresh r, at frame t > r:** S_t(c) = Σ_{k=r+1..t} g_k(c), summed at the
  fixed cell c.
  - **This is an approximation.** It is not the displacement along the content's trajectory: content that moves
    out of c is no longer what is summed. It can understate or overstate true travel, and it cancels for motion
    that reverses.
- **Cell valid at t:** valid at t, and g_k(c) is finite for every k in r+1..t. The share of token-frames lost to
  the "finite" part is reported.
- **Token valid at t:** at least half of its cells are valid at t.
- **Token truly moved at t:** valid, and some valid cell has |S_t(c)| > 2 px.
- **Token STALE at t:** reused at t and truly moved at t.
- **Frame range:** frames t = 1..n−1 are scored. Frame 0 has no ground truth; it is a refresh frame and is not
  scored. Refresh frames t > 0 are scored as computed.

## 4. Metrics (per encode = arm × sequence or clip; reported side by side)
- REUSE = reused valid token-frames / valid token-frames. I-frames and refresh frames count as computed.
- STALE = stale token-frames / valid token-frames.
- STALE_MOV = stale token-frames / truly-moved token-frames. Empty denominator → missing.
- **Frame group:**
  - Primary: all scored frames of the encode. The union couples frames, so a frame-type subset is not a separate
    policy.
  - Descriptive: for the bf 2 arms, the same ratios over B-frames only.
- **Also reported, descriptive:** valid token-frames, truly-moved token-frames, the share of tokens active at the
  frame before each refresh, and the token-frames lost to the finite-terms rule.
- **Bare per-cell gate beside the policy (descriptive).** The main paper's rule on the same encodes, frames
  (t = 1..n−1) and τ, with the same naive vectors and painting:
  - A cell is reused at t if it has a vector and |MV| < τ. A no-vector cell, and every cell of an I-frame, is
    recomputed.
  - Truth is per-frame: |g_t(c)| > 2 px, not summed. Cell validity is the per-frame rule of §3.
  - Reported per encode: reuse share (reused valid cells / valid cells), STALE (stale / valid cells) and
    STALE_MOV (stale / valid cells with |g_t(c)| > 2 px).
  - These values differ from the main sweep's numbers, which scored only d = 1 P-frames or B-frames, at τ = 1 px.
- **Summaries and intervals (descriptive, not verdicts):**
  - Per-arm medians over units of REUSE, STALE and STALE_MOV, for the policy and the bare gate.
  - For each S1–K4 comparison of §6, the median per-unit paired difference in that prediction's metric, with a 95%
    bootstrap CI. For S1r, it is the median per-sequence rho. If that interval is degenerate, the count of
    sequences with rho = 1 is reported instead of a CI.
  - Method and seed handling as in `p3/review_fixes3/r4_bootstrap_ci.py`:
    - percentile intervals (numpy.percentile 2.5 / 97.5, linear interpolation), 10,000 resamples
    - one generator, `numpy.random.default_rng(20261006)`, drawn in the order (1) Sintel sequence indices
      (10000, 23), (2) CCTV scene indices (10000, 7), (3) CCTV clip indices (10000, 27)
    - the same draws are used for every comparison within a dataset
  - Sintel resamples the 23 sequences. CCTV uses the scene-cluster bootstrap (7 scenes drawn with replacement,
    all eligible clips of each drawn scene kept) and, beside it, a clip-level bootstrap over the 27 clips.
  - Intervals are computed at the primary configuration, and at τ = 1 px as secondary.

## 5. Data (no new encodes; inventory in `inventory.txt` / `inventory.csv`)
**Sintel, final pass, 23 sequences (20–50 frames, 24 fps).** Saved block files `p3/sweep/blocks/final/` (same as
p3/sweep_rerun for every arm used, see p3/sweep_rerun/COMPARISON.md):
- x264 CRF 12, 18, 23, 28, 33, 38, 45
- NVENC QP 18, 23, 28, 33, 38, 45
- x264 QP 24 bf 0 and bf 2 (`x264_qp24_bf2_pb1`)
- NVENC QP 28 bf 0 and B=P (`nvenc_qp28_bf2_bqeq`)
- ~41 dB trio: `x264_crf23`, `nvenc_qp28`, `mpeg4_q4` (as matched in p3/review_fixes3)

That is 17 arms, 391 encodes, none missing.

**CCTV.** The 27 eligible clips of the confirmatory test (`p3/virat_confirm/eligibility_v0.csv`; V0 PASS 26/27)
× its 9 arms (x264 CRF 12, 45; x264 QP 24 bf 0, bf 2; NVENC QP 18, 23, 45; NVENC QP 28 bf 0, B=P), 193–300
frames, with RAFT. 243 encodes, none missing. Eligibility is inherited, not recomputed.

## 6. Predictions (fixed before any number of this simulation is seen)
Primary configuration: derived token footprint, τ = 0.5 px, native rate. Verdicts are taken there only. The same
rules are evaluated at τ = 1 px and reported as secondary.

They mirror the main paper's predictions with the same comparisons, thresholds and counting rules, using each
dataset's verdict metric:
- Sintel: STALE, the analogue of stale/valid.
- CCTV: STALE_MOV.

| key | mirrors | comparison (per sequence / clip) | margin | required count |
|---|---|---|---|---|
| S1x | P1 | Sintel STALE(x264 CRF 45) − STALE(x264 CRF 12) | > 0.001 | ≥ 18/23 |
| S1n | P1 | Sintel STALE(NVENC QP 45) − STALE(NVENC QP 18) | > 0.001 | ≥ 18/23 |
| S1r | P1 (Spearman) | Sintel Spearman(CRF, STALE) over the 7 x264 CRF levels | ≥ 0.8 | ≥ 16/23 |
| S2x | P2 | Sintel STALE(x264 QP 24 bf 2) − STALE(x264 QP 24 bf 0) | > 0.001 | ≥ 18/23 |
| S2n | P2 | Sintel STALE(NVENC QP 28 B=P) − STALE(NVENC QP 28 bf 0) | > 0.001 | ≥ 16/23 |
| K1 | C1 | CCTV STALE_MOV(x264 CRF 45) − STALE_MOV(x264 CRF 12) | > 0.01 | ≥ 80% = 22/27 |
| K2 | C2 | CCTV STALE_MOV(NVENC QP 45) − STALE_MOV(NVENC QP 18) | > 0.01 | ≥ 75% = 21/27 |
| K3 | C3 | CCTV STALE_MOV(x264 QP 24 bf 2) − STALE_MOV(x264 QP 24 bf 0) | > 0.01 | ≥ 80% = 22/27 |
| K4 | C4 | CCTV STALE_MOV(NVENC QP 28 B=P) − STALE_MOV(NVENC QP 28 bf 0) | > 0.01 | ≥ 75% = 21/27 |
| KILL-A | new | every arm, both datasets: median STALE_MOV < 0.01 **and** median REUSE > 0.20 | see text | all 26 arm-medians |

**Not mirrored, and why:**
- P1 MPEG-4 (q 2 / q 31 are not in the data list).
- P2b (this policy has no scaled-vector variant).
- P2c and P3 (their arms are not in the data list).
- C5 (FLIP is not defined for this policy).

The ~41 dB trio and the NVENC QP 23 CCTV arm are descriptive.

**Kill rules, as in the main paper:**
- S1x and S1n both fail: under this policy the quantiser does not change reuse safety on Sintel.
- One of them fails: the claim is encoder-specific.
- S2x or S2n fails: the B-frame claim does not survive this policy for that encoder, reported as a null.
- K1 and K2 both fail: the quality effect under this policy does not carry to CCTV.
- K3 or K4 fails: the B-frame claim under this policy is Sintel-only for that encoder.

**KILL-A, the absorption condition.** "If under this policy STALE_MOV stays below 1% at every setting while REUSE
stays above 20%, CodecSight's extra rules absorb the problem, and the paper says so." Operationalised:
- Fires if, at the primary configuration, for every one of the 17 Sintel arms and every one of the 9 CCTV arms,
  the median over units of STALE_MOV is < 0.01 and the median REUSE is > 0.20.
- If it fires, the paper states that CodecSight's documented policy absorbs the encoder effect on our data,
  whatever S1–K4 say. S1–K4 still say whether the residual stale share moves with the encoder.
- If some arm has median REUSE ≤ 0.20, that arm is reported as "policy reuses little". Its low STALE_MOV is not
  counted as absorption, and KILL-A does not fire.

## 7. Exploratory (labelled as such; no verdicts)
**CCTV at 2 FPS: a deployment that decodes a native-rate stream and samples it at 2 FPS.** This is not
CodecSight's setup, which (by our inference, §2) encodes at 2 fps.
- Sampled frames: s_j = j·k, with k = 12 for 23.97 fps clips and k = 15 for 30 fps clips (17–25 sampled frames
  per clip, see the inventory).
- Refresh: at the first sampled frame at or after an I-frame (our choice; an unsampled I-frame still resets the
  GOP), and 16 sampled frames after the previous refresh. Refresh patterns: [0, 192], [0, 192, 252] or
  [0, 240, 255].
- Truth: S_t(c) is summed over every native frame since the refresh, including the skipped frames, with the same
  validity rules. Only sampled frames are scored.
- Variant (a): at a sampled frame the policy uses that frame's own codec vectors (frame s−1 → s, native rate). The
  union accumulates over sampled frames only.
- Variant (b): as (a), but the union at each sampled frame also includes every token that was dynamic (same rule
  and τ) in any skipped native frame since the previous sampled frame.
- Both variants are descriptive. REUSE, STALE and STALE_MOV are reported per arm and clip.

Also descriptive: footprints 16, 32 and 64 px; τ = 0.25 px.

## 8. Rules
- Fixed by this file. No tuning after results.
- Bugs: fix, record in `p3/codecsight_sim/DEVIATIONS.md`, rerun everything, report both runs.
- Per-sequence and per-clip results are primary; per-scene results are also shown for CCTV.
- "x% of N" means at least ceil(x·N). A missing value counts as not meeting the condition, and is listed.
- Nothing in this simulation changes a verdict of the main pre-registration or of the confirmatory test.
