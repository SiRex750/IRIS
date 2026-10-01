# Texture check: ambush_7, x264 CRF 12 / 23 / 45 (exploratory, report only)

**Setup**
- **Cells:** 4×4 px on frame t's grid, P-frames 1–49 (d = 1). Only valid cells count (gt_cover ≥ 0.5): 1,334,699 per encode.
- **Texture** = mean luma-gradient magnitude of the source frame t over the cell. Luma comes from the same RGB→yuv420p conversion the encoder input used; the gradient is `np.gradient`.
- **Quartile edges** come from the final-pass source frames over valid cells: 0.35 / 0.62 / 2.02. They are held fixed for every encode, including the clean pass.
  - The clean pass is also shown with its own edges (0.94 / 2.46 / 5.44), because its texture distribution is very different.
- **Gate** as in pilot v2, τ = 1: reuse if the cell has a vector and |MV| < 1. GT-static means |GT| < 0.5. Stale means reused and |GT| > 2.
- **Encodes and GT:**
  - Final-pass encodes are the committed-run v2 bitstreams.
  - Clean-pass encodes were made here with identical knobs. Their options are in `texture_meta.json`; the SEI shows threads=1 / sliced_threads=0, with 1 slice per frame.
  - GT is the same for both passes.
- **Sanity check:** the "all" rows reproduce v2's missed ÷ valid (0.457 / 0.449 / 0.212) and stale ÷ valid.

Columns:
- **static share:** GT-static cells ÷ valid cells in the quartile
- **missed:** missed reuse ÷ valid
- **stale:** stale ÷ valid
- **static |MV|≥1:** share of GT-static cells with |MV| ≥ 1 px
- **static no-vec:** share of GT-static cells with no vector (intra)

## Final pass (fixed edges)

| CRF | quartile | valid cells | static share | **missed** | **stale** | **static \|MV\|≥1** | static no-vec |
|---|---|---|---|---|---|---|---|
| 12 | Q1 (lowest) | 333,598 | 0.969 | **0.702** | 0.0010 | **0.418** | 0.306 |
| 12 | Q2 | 333,200 | 0.953 | 0.648 | 0.0003 | 0.387 | 0.293 |
| 12 | Q3 | 334,226 | 0.737 | 0.417 | 0.0009 | 0.278 | 0.287 |
| 12 | Q4 (highest) | 333,675 | 0.513 | 0.061 | 0.0006 | 0.074 | 0.044 |
| 23 | Q1 | 333,598 | 0.969 | **0.674** | 0.0008 | **0.552** | 0.144 |
| 23 | Q2 | 333,200 | 0.953 | 0.636 | 0.0007 | 0.546 | 0.121 |
| 23 | Q3 | 334,226 | 0.737 | 0.424 | 0.0014 | 0.453 | 0.123 |
| 23 | Q4 | 333,675 | 0.513 | 0.061 | 0.0009 | 0.086 | 0.034 |
| 45 | Q1 | 333,598 | 0.969 | 0.268 | 0.0073 | 0.154 | 0.123 |
| 45 | Q2 | 333,200 | 0.953 | 0.285 | 0.0094 | 0.176 | 0.123 |
| 45 | Q3 | 334,226 | 0.737 | 0.244 | 0.0411 | 0.215 | 0.116 |
| 45 | Q4 | 333,675 | 0.513 | 0.051 | 0.0658 | 0.076 | 0.024 |

## Clean pass, same fixed edges (cells redistribute: Q1 has 97,712, Q4 has 747,890)

| CRF | quartile | valid cells | static share | **missed** | **stale** | **static \|MV\|≥1** | static no-vec |
|---|---|---|---|---|---|---|---|
| 12 | Q1 | 97,712 | 0.972 | **0.218** | 0.0001 | **0.104** | 0.120 |
| 12 | Q2 | 130,017 | 0.957 | 0.194 | 0.0002 | 0.099 | 0.104 |
| 12 | Q3 | 359,080 | 0.854 | 0.135 | 0.0007 | 0.069 | 0.089 |
| 12 | Q4 | 747,890 | 0.712 | 0.046 | 0.0009 | 0.020 | 0.045 |
| 23 | Q1 | 97,712 | 0.972 | **0.150** | 0.0003 | **0.104** | 0.050 |
| 23 | Q2 | 130,017 | 0.957 | 0.143 | 0.0010 | 0.110 | 0.040 |
| 23 | Q3 | 359,080 | 0.854 | 0.112 | 0.0015 | 0.100 | 0.032 |
| 23 | Q4 | 747,890 | 0.712 | 0.039 | 0.0014 | 0.036 | 0.020 |
| 45 | Q1 | 97,712 | 0.972 | 0.259 | 0.0080 | 0.228 | 0.038 |
| 45 | Q2 | 130,017 | 0.957 | 0.266 | 0.0088 | 0.241 | 0.036 |
| 45 | Q3 | 359,080 | 0.854 | 0.227 | 0.0235 | 0.233 | 0.032 |
| 45 | Q4 | 747,890 | 0.712 | 0.157 | 0.0530 | 0.199 | 0.022 |

## Clean pass, its own quartile edges (secondary)

| CRF | missed Q1 / Q2 / Q3 / Q4 | stale Q1 / Q2 / Q3 / Q4 | static \|MV\|≥1 Q1 / Q2 / Q3 / Q4 |
|---|---|---|---|
| 12 | 0.195 / 0.109 / 0.056 / 0.028 | 0.0003 / 0.0008 / 0.0008 / 0.0012 | 0.097 / 0.056 / 0.024 / 0.010 |
| 23 | 0.144 / 0.093 / 0.048 / 0.023 | 0.0009 / 0.0015 / 0.0013 / 0.0014 | 0.110 / 0.086 / 0.045 / 0.018 |
| 45 | 0.263 / 0.204 / 0.168 / 0.141 | 0.0095 / 0.0323 / 0.0588 / 0.0492 | 0.240 / 0.223 / 0.211 / 0.183 |

## Overall (all valid cells)

| CRF | missed, final → clean | stale, final → clean | static \|MV\|≥1, final → clean | static no-vec, final → clean | bitrate kb/s, final → clean |
|---|---|---|---|---|---|
| 12 | 0.457 → 0.097 | 0.0007 → 0.0007 | 0.321 → 0.051 | 0.255 → 0.071 | 4628 → 11878 |
| 23 | 0.449 → 0.077 | 0.0010 → 0.0013 | 0.452 → 0.069 | 0.114 → 0.028 | 1049 → 2515 |
| 45 | 0.212 → 0.194 | 0.031 → 0.037 | 0.162 → 0.217 | 0.105 → 0.028 | 91 → 76 |

## Reading (exploratory)

**Missed reuse on the final pass is concentrated in low texture.**
- At CRF 12 and 23 it falls steadily across the quartiles: 0.70 / 0.65 / 0.42 / 0.06 (CRF 12) and 0.67 / 0.64 / 0.42 / 0.06 (CRF 23).
- Part of that is composition: Q1 and Q2 are about 96% GT-static, Q4 only 51%. But the pattern holds within GT-static cells too:
  - In Q1, 42–55% of static cells carry |MV| ≥ 1 px and 14–31% have no vector.
  - In Q4, 7–9% carry |MV| ≥ 1 px and 3–4% have no vector.
- The concentration is not unique to the lowest quartile. Q1 and Q2 are similar, and the drop comes at Q3–Q4.
- Final-pass texture is very low overall (the upper edge of Q2 is 0.62 grey levels per pixel).

**The clean pass removes most of the missed reuse at CRF 12–23:** overall 0.457 → 0.097 and 0.449 → 0.077. This is more than a shift in the texture distribution:
- At the same fixed texture level, clean Q1 misses 0.22 / 0.15 vs 0.70 / 0.67 for final.
- The share of static cells with |MV| ≥ 1 drops from 0.42–0.55 to about 0.10.

Low-gradient final-pass content therefore behaves differently from low-gradient clean content. This check does not identify which final-pass effect is responsible (Sintel's final pass adds blur and atmospheric rendering to the clean pass).

**CRF 45 is different.**
- Final and clean missed reuse are close (0.21 vs 0.19), and the texture gradient is much flatter.
- Stale reuse rises with texture in both passes (final Q4 0.066, clean Q4 0.053 with fixed edges), where there was almost none at CRF 12–23.
- The clean pass costs 2.4–2.6× the bitrate at CRF 12–23, and slightly less at CRF 45.
