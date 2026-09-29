## Reading guide

- Every number is for one sequence. Columns are static-heavy (S: ambush_7, bandage_2), then moving (M: alley_1, ambush_5).
- **Headline gate metrics (τ = 1 px):**
  - stale ÷ valid: reused cells whose GT motion is > 2 px
  - missed reuse ÷ valid: GT-static cells (|GT| < 0.5 px) that the gate recomputes

  Missed reuse only carries information on the static-heavy pair. On alley_1 and ambush_5 there is almost nothing static to miss (≤ 0.005 of valid area in every arm).
- Flip rate is given two ways: vs x264 CRF 12, and vs the encoder's own best setting (NVENC QP 18, mpeg4 q 2).

## What moved the gate (exploratory; τ = 1)

**Quantiser moves stale reuse most.** Stale ÷ valid, lowest to highest QP:

| sequence | x264 CRF 12 → 45 | NVENC QP 18 → 45 | mpeg4 q 2 → 31 |
|---|---|---|---|
| ambush_7 (S) | 0.0007 → 0.031 | 0.0009 → 0.052 | 0.015 → 0.043 |
| bandage_2 (S) | 0.0022 → 0.029 | 0.0035 → 0.034 | 0.025 → 0.040 |
| alley_1 (M) | 0.0006 → 0.014 | 0.0007 → 0.045 | 0.018 → 0.036 |
| ambush_5 (M) | 0.009 → 0.259 | 0.014 → 0.423 | 0.201 → 0.311 |

Missed reuse moves the other way and is large on ambush_7:
- x264 CRF 12 → 45: 0.457 → 0.212
- NVENC QP 18 → 45: 0.439 → 0.107
- bandage_2: 0.04–0.06 throughout

So higher QP trades missed reuse for stale reuse on the static-heavy sequences.

**Other factors, same metrics (x264 CRF 23 unless noted):**
- **ref 2/3 (read as d=1):** stale moves ≤ 0.0009, missed ≤ 0.018. Negligible.
- **keyint 30:** stale ≤ 0.0008. Negligible.
- **Search range** (me=umh, merange 16/32/64 vs medium's hex/16): stale ≤ 0.0023. Negligible (details below).
- **Preset:** ultrafast raises ambush_5 stale 0.015 → 0.058 and moves ambush_7 missed by 0.076. Veryslow is close to medium.
- **Encoder, compared on the PSNR row** (x264 CRF 23 / NVENC QP 28 / mpeg4 q 4):

  | | ambush_7 | bandage_2 | alley_1 | ambush_5 |
  |---|---|---|---|---|
  | stale ÷ valid | 0.0010 / 0.0029 / 0.017 | 0.0039 / 0.011 / 0.026 | 0.0006 / 0.0042 / 0.023 | 0.015 / 0.068 / 0.226 |
  | missed ÷ valid | 0.449 / 0.305 / 0.168 | – | – | – |

  x264 has the least stale reuse and the most missed reuse.
- **mpeg4's flip rate vs x264 CRF 12 is mostly an encoder difference.** At q 4 it is 0.09–0.32 vs x264, but only 0.02–0.09 vs mpeg4 q 2.

**Steady or inverted-U** (GT metrics, per sequence):
- **Stale ÷ valid rises steadily with QP** in all 4 sequences for NVENC and mpeg4. For x264 it's steady on bandage_2 and ambush_5, with tiny early dips on ambush_7 (0.00070 → 0.00067 at CRF 18) and alley_1 (at CRF 23).
- **Missed reuse on ambush_7 falls with QP:** steadily for NVENC; for x264 with small bumps.
- **bandage_2's missed reuse is a mild ∩ under x264:** 0.042 → 0.062 at CRF 38 → 0.053.
- **x264's EPE on ambush_7 is ∩:** 0.49 → 0.85 at CRF 28 → 0.26. NVENC and mpeg4 are flat or falling there.
- **Why ambush_7 behaves this way** (block level, GT-static blocks, x264 CRF 23):
  - 51% of the area gets |MV| ≥ 1 px (median 2.75, p90 16 px)
  - 26% gets a non-zero sub-pixel vector
  - 23% gets exactly zero
  - 13% of P-frame area has no vector at all

  At CRF 45, 77% is exactly zero. At NVENC QP 28 it is 67%. That is what drives both the missed reuse and the ∩ in EPE. I have not looked into why x264 puts multi-pixel vectors on static background at low and mid CRF.

**Search range (the ambush_5 question).**
- ambush_5's stale cells are mostly slow. At CRF 23 the speed bins are 2–8 / 8–16 / 16–32 / >32 px = 11094 / 2251 / 3030 / 1230, i.e. 63% at 2–8 px and 7% above 32 px.
- me=umh with merange 16/32/64 gives stale ÷ valid 0.0130 / 0.0130 / 0.0126, vs 0.0149 with hex/16.
- So the ambush_5 stale reuse isn't explained by motion beyond the search range.
- The "35%" in the brief is stale ÷ reused (v1 0.351, v2 0.329 at CRF 12). As a share of valid area it is 0.9%.

**B-frames vs QP (ANVIL), with QP matched.** stale ÷ valid, bf 0 d=1 → B naive (B scaled in brackets):

| | ambush_7 | bandage_2 | alley_1 | ambush_5 |
|---|---|---|---|---|
| x264 --qp 24, B QP = P QP = 24 [extra arm] | 0.0011 → 0.013 (0.013) | 0.0037 → 0.039 (0.040) | 0.0007 → 0.018 (0.019) | 0.020 → 0.066 (0.069) |
| NVENC QP 28, B QP = P QP = 28 | 0.0029 → 0.011 (0.011) | 0.011 → 0.035 (0.036) | 0.0042 → 0.018 (0.018) | 0.068 → 0.088 (0.089) |
| NVENC QP 28, default B QP 36 | 0.0029 → 0.014 | 0.011 → 0.030 | 0.0042 → 0.016 | 0.068 → 0.181 |
| x264 CRF 23, default (B QP +1.6 to +5.4) | 0.0010 → 0.0082 | 0.0039 → 0.032 | 0.0006 → 0.013 | 0.015 → 0.047 |

- **With QP matched, B-frame stale reuse is still 3–27× the bf 0 level in x264,** and 1.3–4.3× in NVENC. For x264 that is roughly CRF 38–45 territory: bandage_2 0.039 vs 0.029 at CRF 45; alley_1 0.018 vs 0.014.
- **Scaling by reference distance changes stale ÷ valid by ≤ 0.0032,** so the gate shift is not a reference-distance effect.
- **Matching the QP does matter for NVENC on ambush_5:** 0.181 → 0.088.
- **B-frames also cut missed reuse** (ambush_7 0.457 → 0.183 under x264 CQP 24). B-frames push the gate toward reuse.
- **EPE shows the opposite pattern.** The naive B-vs-bf0 gap is mostly reference distance: x264 CQP 24 B EPE median is 2.2 px naive vs 0.19 scaled on alley_1, and 7.0 vs 2.2 on ambush_5. Scaled B EPE is at or below bf 0, except on ambush_5 (2.2 vs 1.15).
- **Median EPE says the B vectors are fine, yet the gate gets more stale.** The stale cells are a tail that median EPE doesn't see.
- **P-frames at d = 3 in the B arms** are different: there, scaling does close most of the gate gap (x264 CQP 24 ambush_5 stale 0.012 naive → 0.022 scaled vs 0.020 at bf 0; missed 0.574 → 0.414 vs 0.457 on ambush_7).

## v1 → v2: what the slice fix changed (x264 table below)

- **CRF 12–23:** bitrate −1 to −6%, EPE median slightly lower (for example ambush_7 CRF 23: 0.88 → 0.78), flip within ±0.005. Stale ÷ valid changes are small in absolute terms.
- **CRF 45: larger.**
  - bitrate −16 to −28%
  - alley_1 stale 0.029 → 0.014, zero-MV 0.134 → 0.024
  - ambush_5 stale 0.374 → 0.259
  - bandage_2 flip 0.22 → 0.14, zero-MV 0.43 → 0.54

  Slicing mattered most at low bitrate.
- **mpeg4 also changed.** v1's mpeg4 ran 16 video packets per frame (see oddities). At q 2, ambush_5 EPE 2.84 → 1.73 and stale 0.266 → 0.201. Other mpeg4 changes are smaller (bitrate −1 to −8%).

## Failures, odd things, runtime

- **Failures:** none. 136/136 encode × sequence runs succeeded.
- **Runtime:** 507 s (8.5 min), plus 11 s to load GT. Single-threaded x264 roughly doubled per-run time at low CRF compared with v1.
- **Threading and slices:** every x264 SEI reads `threads=1 sliced_threads=0` with no `slices` field. The bitstream has 1 slice per frame for every x264 and NVENC encode, and 1 video packet per frame for mpeg4.
- **mpeg4 was sliced in v1 too.** At PyAV's default thread count, the mpegvideo encoder writes one resync-marked video packet per thread: 16 per frame on this machine (15 markers counted). v2 passes `threads=1` to mpeg4 as well. This extends the prompt's x264 fix to mpeg4 via the same additive knob.
- **x264 `pbratio=1.0` under CRF is a no-op.** The pb1 arm is byte-identical to the default-B arm on all four sequences. With mbtree=1 (CRF default), x264 ignores pbratio and its SEI omits `pb_ratio`. Decoded B QP stays 1.6–5.4 above P.
  - The extra constant-QP pair (`x264 --qp 24`, bf 0 and bf 2 with pbratio 1.0, not in the prompt) gives decoded I/P/B = 21/24/24 exactly. It's labelled [extra] everywhere.
- **NVENC B QP:** `b_qfactor=1, b_qoffset=0` gives decoded B = P = 28. The default is B 36.
- **The last frame of every b-adapt=0 / NVENC bf2 stream** is a P right after a P (`…BPP`), so each B arm has a 1-frame d=1 group. It's in results.csv and left out of the tables.
- **The final frame of NVENC bf2 streams mixes 28 and 36 QP.** That frame's QP map mixes the P and B QP (28 and 36); every other frame is uniform. This is why the default-B arm shows mean P QP 28.09–28.19.
- **x264 --qp 24 bf 0** costs 1.2–1.5× the bitrate of CRF 23, at +1.3–2.4 dB.
- **`p3/lib/encode.py` has uncommitted additive knobs:** threads, b_adapt, pbratio, me, merange, qp for libx264, b_qfactor/b_qoffset, and mpeg4 threads. With them unset, the default options are byte-for-byte the old ones. The diff is saved in manifest.json.
