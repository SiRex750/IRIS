## Reading guide

- **Headline group:** P-frame vectors with d = 1 (the whole of every bf 0 arm). For ref 2/3 the group is "d1(assumed)", because d is unknown there and read as 1.
- **Gate:** reuse if |MV| < τ. Flip rate is measured against x264 CRF 12, on the same display frame and cell.
- **Mean-of-4 caveat.** For ratio metrics (precision, false-static ÷ zero-MV, stale ÷ reused), the mean of four blends two regimes:
  - ambush_7 and bandage_2 have large static areas (pixel static share 0.79 / 0.56). Reuse precision there is 0.80–0.99.
  - alley_1 and ambush_5 are almost entirely moving (static share 0.0015 / 0.0059). Reuse precision there is 0.004–0.06, because there is almost nothing static to reuse.

  So the per-sequence columns are the ones to read, and a mean near 0.5 describes none of the four.

## What moved the gate (exploratory; τ = 1 unless noted)

**Which factor moves it most.** The quantiser. Along x264 CRF 12 → 45 (mean of 4):
- flip rate goes 0 → 0.316
- stale ÷ reused goes 0.091 → 0.276
- median EPE goes 0.39 → 2.01 px

Stale ÷ reused per sequence, CRF 12 → 45:

| sequence | CRF 12 | CRF 45 |
|---|---|---|
| alley_1 | 0.008 | 0.121 |
| ambush_5 | 0.351 | 0.868 |
| ambush_7 | 0.002 | 0.051 |
| bandage_2 | 0.004 | 0.064 |

The NVENC QP 18 → 45 series moves about as much: flip 0.079 → 0.352, stale 0.089 → 0.259. mpeg4 moves less across its whole q range (flip 0.247 → 0.316, stale 0.229 → 0.268), but it starts high.

**Encoder.** Compared near matched PSNR (x264 CRF 23 at 42.1 dB, NVENC QP 28 at 41.6 dB, mpeg4 q 4 at 41.5 dB, all mean of 4):

| | x264 CRF 23 | NVENC QP 28 | mpeg4 q 4 |
|---|---|---|---|
| flip rate | 0.084 | 0.123 | 0.259 |
| stale ÷ reused | 0.109 | 0.144 | 0.235 |
| median EPE | 0.65 | 0.46 | 1.08 |

NVENC has the lower EPE but the higher stale rate. That's a single matched point, not a curve.

**Other x264 settings at CRF 23:**
- **preset:** ultrafast raises stale to 0.184 (alley_1 0.169 vs 0.012 at medium) and flip to 0.122, at 2–3× the bitrate. Veryslow is within about 0.01 of medium (stale 0.120 vs 0.109, flip 0.088 vs 0.084).
- **ref 2/3 (read as d=1):** negligible change. Median EPE +0.02 px, stale +0.01, flip +0.003.
- **keyint 30:** negligible change (stale −0.006, flip −0.002).

The pilot can't see how many ref 2/3 blocks actually used t−2. The small change is consistent with x264 mostly choosing t−1 on this content, unlike the synthetic shift in validation A.

**Steady or inverted-U.** Using the GT metrics (flip rate is 0 at CRF 12 by construction):
- **Mean of 4, x264 CRF series:** EPE, EPE > 3 px, and false-static ÷ zero-MV rise steadily; precision falls steadily. Stale ÷ reused is flat from CRF 12 to 18 (0.091, 0.090), then rises. Recall is U-shaped (0.56 → 0.54 → 0.67).
- **Per sequence, it is not uniformly steady:**
  - **ambush_7 EPE is inverted-U** under x264: 0.53 → 0.90 at CRF 28 → 0.25 at CRF 45. At high CRF the static background turns into exact-zero vectors (zero-MV share 0.13 → 0.64), and zero is its true motion. Its recall is U-shaped (0.42 → 0.81) for the same reason. At low CRF, x264 codes small non-zero vectors on a background that is 79% static.
  - **Under NVENC, ambush_7 EPE falls steadily** (0.42 → 0.23).
  - **alley_1 false-static ÷ zero-MV peaks at CRF 38** (0.95), then drops to 0.905.
  - **Stale ÷ reused and precision are the steadiest metrics.** Under x264, stale is monotone in 3 of 4 sequences; ambush_5 dips 0.351 → 0.341 at CRF 18. Precision falls steadily in every sequence and encoder except alley_1 under x264 and NVENC, where it first edges up (for example 0.015 → 0.018) at values that are near zero anyway.
- The per-sequence shape table is below.

**B-frames and ref>1 vs QP (ANVIL).** ref>1 barely moves anything (above). B-frames move the gate about as much as a large QP step:
- **x264 CRF 23:** bf2's B vectors vs the bf 0 d=1 vectors: stale 0.187 vs 0.109, flip 0.168 vs 0.084. That's about the gap between CRF 23 and CRF 33 on stale (0.204), and between CRF 33 and 38 on flip (0.135 / 0.192).
- **NVENC QP 28:** bf2's B vectors vs bf 0: stale 0.266 vs 0.144, flip 0.205 vs 0.123. That's at the level of QP 38–45 on stale (0.212–0.259) and about QP 38 on flip (0.213).

The naive vs scaled split separates two things:
- **EPE gap: mostly reference distance.**
  - x264 B vectors: naive 2.47 → scaled 0.70 (mean of 4). Per sequence, alley_1 2.38 → 0.18, ambush_7 0.26 → 0.24.
  - x264 P-frames with d = 2–3: naive 3.52 → scaled 0.79.
  - The exception is ambush_5: scaled B EPE is 2.26, vs 1.35 for bf 0. That's where the steady-motion assumption behind scaling is least likely to hold.
- **Gate gap: mostly not reference distance.**
  - For B vectors, scaling moves stale only 0.187 → 0.171 (x264) and 0.266 → 0.233 (NVENC). Flip doesn't move at all (0.168 → 0.171; 0.205 → 0.206).
  - So the B-frame gate shift looks like it comes from the B vectors themselves.
  - Those vectors are coded coarser: mean B-frame QP is 27.3 vs 23.9 for P (x264), and 36 vs 28 for NVENC. The B-frame comparison is therefore confounded with QP.
  - For P-frames at d = 2–3, the gate gap *is* mostly reference distance: stale naive 0.234 → scaled 0.119, vs 0.109 at bf 0.

## Failures, odd things, runtime

- **Failures:** none. All 100 encode × sequence runs succeeded. Runtimes:
  - step 0 (static scan over 1041 flow files): not separately timed
  - pilot: **306 s**, of which 12 s was loading GT and images once
  - summary: seconds
  - Per run: about 3.3 s. ref 2/3 and B-frame arms took about 1 s, because compare.py's `gt_src` runs only for known-d=1 P vectors. Veryslow took 3.7 s (only 50 frames).
- **x264 adaptive B placement** (`b_adapt=1`, not disabled by the spec) gives different frame patterns per sequence. For example, ambush_5 is `…BPBPPBPBPBPPPBPBPPPPPPBP`, and bandage_2 has many P-P runs.
  - In x264's bf2 arm, the "d1" P-frames are therefore the high-motion frames: in ambush_5, 8 frames with GT 7–28 px. Hence precision 0, and stale ÷ reused = 1.0 (2196 of 2197 reused cells). That's a selection effect, not a measurement error.
- **NVENC bf2 ends in `…BPP`,** so its d1 group is one frame per sequence. On ambush_5 that frame (49) has GT motion of about 55 px while NVENC's vectors are about 0 (EPE 55 px). compare.py's source-position GT agrees (53 px), so this is not an artifact of the frame-t grid.
- **Different QP scales:**
  - NVENC constqp puts I-frames at QP 22 and B-frames at 36 when QP is 28.
  - mpeg4's exported QP is 2×q (the MPEG-2 qscale convention), e.g. q 4 → 8.0.
  - The "mean P QP" column shows these as the decoder exports them.
- **ambush_5 is the hardest sequence.**
  - 13.6% of its in-frame cells are excluded (gt_cover < 0.5), vs about 2% elsewhere.
  - MV coverage is 64.5% at CRF 12 (many intra blocks).
  - EPE > 3 px covers 29–75% of its area.
- **Low-QP noise-chasing on ambush_7:** at CRF 12 only 13% of valid area has zero MV, on a sequence where 79% of pixels are static, so reuse recall is 0.42.
- **Scaled vectors can beat bf 0 d=1 on EPE:** alley_1 B scaled 0.18 and P d>1 scaled 0.08, vs 0.23 at bf 0. One possible reason is that dividing a quarter-pel vector by d gives finer effective precision. These are also different frames.
- **ref 2/3 are only 1–2% smaller** than ref 1 at CRF 23. Ultrafast is 2–3× the bitrate of medium.
- **mpeg4's flip rate is high even at q 2** (0.25). It uses a different block structure (16×16/8×8, half-pel) from the x264 reference.
- **x264 version:** the build reports "x264 - core 165" with no revision string.
