# Paper 3 — setup and validation report

Date: 2026-09-29. Environment: Python 3.13, PyAV 17.1.0 (libavcodec 62.28), numpy 2.2.6, RTX 4050 Laptop (NVENC).
Scope: tools built and validated only. No sweep was run. Everything written is under `p3/`. `iris/*.py` and the datasets were only read.

Layout:
```
p3/lib/        flo.py  encode.py  mvs.py  compare.py
p3/validate/   inventory.py  check_tools.py  val_A_shift.py  val_BC_alley.py
p3/results/    inventory.json  check_tools.json  val_A.json  val_BC.json  (+ .log)
p3/work/       encoded .mp4 files and per-block .npz dumps (scratch; *.mp4/*.npz are gitignored)
```
To rerun: `python p3/validate/<script>.py` (inventory takes the VIRAT roots as arguments).

---

## 1. Data inventory

### MPI-Sintel — `C:\Users\akash\Documents\datasets\MPI-Sintel`
- `training/` holds `albedo clean final flow flow_viz invalid occlusions`. `test/` has 12 sequences (clean and final only).
- **23 training sequences, 1064 frames, 1041 `.flo` files.** Every frame is 1024×436.
- The count rule holds for every sequence: `final = clean = invalid = N` and `flow = occlusions = N−1`.
- Frames per sequence: 50 for most of them. The exceptions are ambush_2 (21), ambush_4 (33), ambush_6 (20) and market_6 (40).
- Index convention (verified in §2.1): `flow/frame_k.flo` is the flow from image k to image k+1, defined on image k's pixels. In 0-based display order d, the flow d−1→d is `frame_{d:04d}.flo`, masked by `occlusions/frame_{d:04d}.png` ∪ `invalid/frame_{d:04d}.png`.
- The occlusion and invalid PNGs are stored as **0/255**, not the 0/1 the README describes. The reader treats any value >0 as set.

### VIRAT — `C:\Users\akash\Documents\datasets\VIRAT`
Note: this folder was **empty** when I first listed it at the start of the session. By the time I ran the inventory it held 54 clips under `CCTV 01\` (newest write 18:01:17; inventory ran at 18:06). The numbers below are from that final state. All 54 files open and decode.

| group | n | codec / profile | resolution | fps | duration (min–max, total) | bitrate (file) | B-frames in first 300 |
|---|---|---|---|---|---|---|---|
| `VIRAT_S_0100xx_*` | 45 | h264 Main | 1280×720 | 23.97 | 13–300 s, 2595 s | 0.50–1.88 Mb/s | **none** (I 90, P 13410) |
| `VIRAT_S_0002xx_*` | 8 | mpeg4 Simple | 1280×720 | 30.00 | 31–156 s, 581 s | 8.37–8.45 Mb/s | **none** (I 84, P 2316) |
| `VIRAT_S_000002.mp4` | 1 | h264 High | **1920×1080** | 29.97 | 303 s | 16.95 Mb/s | **yes**: I 2 / P 100 / B 198 |

The per-clip numbers (codec, profile, pix_fmt, size, fps, duration, header frame count, stream and file bitrate, per-type counts) are in `results/inventory.json`. `VIRAT_S_000002.mp4` doesn't follow the usual `_NN_start_end` naming and is the only clip with B-frames. Treat it as an outlier.

There's also an in-repo copy at `Iris/test_data/Virat`: 4 mpeg4 Simple clips, 1280×720, 30 fps, about 8.4 Mb/s, no B-frames. I inventoried it as well, labelled separately.

---

## 2. Tool checks (`validate/check_tools.py` → `results/check_tools.json`)

### 2.1 `flo.py`
- `read_flo` returns float32 (436, 1024, 2). The tag is checked, and so is the payload length.
- **Index and sign check.** I warped image 2 back onto image 1 with `flow/frame_0001.flo` (alley_1 clean, nearest-neighbour, occluded and invalid pixels excluded). Mean L1 photometric error:
  - with the flow as given: **3.17**
  - with zero flow: 9.16
  - with the flow negated: 12.77

  The flow file index and sign are therefore as documented. **PASS.**

### 2.2 `encode.py`: do the knobs take effect?
All runs use the first 24 frames of alley_1 final. Exact option dicts are in the JSON.

| encoder | options used | knob → bytes | monotone |
|---|---|---|---|
| libx264 | `crf, preset=medium, x264-params=bframes=0:keyint=250:min-keyint=250:scenecut=0[:ref=R]` | CRF 12/18/24/30/36 → 667 471 / 308 204 / 147 550 / 76 841 / 43 602 | ✅ (the x264 SEI confirms `crf=12.0…36.0`) |
| h264_nvenc | `rc=constqp, qp, bf, g, preset=p4[, refs]` | QP 12/18/24/30/36 → 916 952 / 499 474 / 232 735 / 119 607 / 62 185 | ✅ (decoded per-MB QP: P-frames = qp exactly) |
| mpeg4 | `qmin=q, qmax=q, bf, g` | q 2/4/8/16/31 → 691 940 / 314 072 / 143 259 / 77 235 / 59 609 | ✅ |

- **The first mpeg4 attempt failed this check, and I changed the method.** `flags=+qscale, global_quality=q·118` gave byte-identical files (611 623 B) for every q. The flag does get set, but mpegvideo's fixed-qscale path takes lambda from each input `AVFrame.quality`. PyAV can't set that field, so it stays 0 and gets clamped to q=2. Pinning `qmin=qmax=q` fixes it, and the sizes now fall monotonically. This is documented in `encode.py`.
- **NVENC constqp caveat.** I-frames come out at about 0.8×qp (for qp=24, the decoded MB QPs are {19 on I, 24 on P}). That's NVENC's default I/P offset.
- **B-frames.** `bframes=0` → `IPPP…` (0 B) for all three encoders. `bframes=2` → `IBBPBBP…` (15 B out of 24) for all three. The x264 SEI confirms `bframes=0/2`. ✅
- **keyint (x264).** `keyint=8` → I-frames at display indices 0, 8, 16, and the SEI reports `keyint=8`. ✅
- **ref (x264).** `ref=1` → the SEI reports `ref=1`. ✅ For NVENC `refs=1` I couldn't check it independently. Its behaviour in §3 is consistent with a single reference.

### 2.3 `mvs.py`
- Uses `codec_context.options={"flags2": "+export_mvs"}` and `side_data["MOTION_VECTORS"].to_ndarray()`, copied field by field. There are no per-vector Python loops.
- For each frame (display order, pts-checked) it returns: `index, pts, pict_type, pkt_size` (packet matched by pts) and `mvs` with fields `source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale`.
- **Convention, checked against the probe and FFmpeg's `add_mb`:**
  - `dst` is the block **centre** in the current frame.
  - `src = dst + motion/motion_scale`, with C integer truncation. Example from the probe: `motion_x=7, scale=4, dst_x=8 → src_x=9`.
  - So motion reference→current is `dst − src = −motion/scale`. The library uses the sub-pel form.
  - In §3, the integer `dst−src` and the sub-pel form gave identical errors on integer shifts.
- **The decoder does not export which reference picture a block used.** "Reference = t−1" is guaranteed only for P-frames with a single reference and no B-frames. See §3 for why this matters.
- PyAV 17 returns `pict_type` as an int. `mvs._pict_name` maps it to I/P/B.

### 2.4 `compare.py`
- Scope: P-frames d ≥ 1, blocks with `source=−1`, reference assumed to be d−1.
- For each block: codec motion vs the mean GT flow of `frame_{d:04d}.flo` over the block's footprint at the **src** position in frame d−1. Means are computed with summed-area tables.
- A block is excluded if its src or dst footprint is not fully inside 1024×436. H.264 pads to 448 rows, so the last MB row is partial.
- Pixels that are occluded or invalid are excluded. A block is kept only if at least 50% of its pixels are valid.
- Outputs per block: EPE, `zero_mv` flag, and gate flags for τ ∈ {0.5, 1, 2}:
  - `mv_moving@τ` = |MV| > τ
  - `gt_moving@τ` = |GT| > τ
  - `epe_ok@τ` = EPE ≤ τ

  (The brief didn't define the gate precisely. This is my reading, and it's easy to change.)

### 2.5 Per-macroblock QP via `export_side_data=+venc_params`: **YES**
PyAV exposes `VideoEncParams` with `qp_map()`, which returns an absolute QP per MB. Example: `libx264_crf_24.mp4`, frame 1 (P) → a 28×64 map, min/median/max = 18 / 26 / 34, top-left 4×4 = `[[25,25,23,23],[30,24,24,24],[27,27,27,27],[28,28,26,26]]`. The frame-level `qp` field holds the PPS init_qp (20 here). The actual QP is init_qp plus each block's `delta_qp`, which `qp_map()` already applies. It's available through `extract_mvs(path, want_qp=True)`.

---

## 3. Validation A: synthetic shift (`val_A_shift.py` → `results/val_A.json`)

**Setup:**
- Base image: alley_1 final `frame_0001.png`.
- 10 frames, each a 960×384 window whose origin moves by −(dx, dy) per frame. Scene content therefore moves by exactly +(dx, dy) px per frame.
- Only interior blocks count: dst footprint ≥ 32 px from every border, P-frames, `source=−1`. That's about 10–11k blocks per shift.
- **Primary config:** libx264 CRF 12, preset medium, bframes 0, keyint 250, **ref=1** (SEI: `ref=1 bframes=0`). Picture types `IPPPPPPPPP`.

| shift (dx,dy) | median recovered (dx,dy) | **median error px** | error if the sign were flipped | blocks with error <0.5 px | per-frame median error |
|---|---|---|---|---|---|
| (4, 0) | (4.0, 0.0) | **0.00** | 8.00 | 99.97% | 0.00–0.00 |
| (0, 3) | (0.0, 3.0) | **0.00** | 6.00 | 99.89% | 0.00–0.00 |
| (−5, 2) | (−5.0, 2.0) | **0.00** | 10.77 | 99.81% | 0.00–0.00 |
| (3, −4) | (3.0, −4.0) | **0.00** | 10.00 | 99.85% | 0.00–0.00 |

The sign is correct on both axes, including negative dx and negative dy. The magnitude is exact. `dst − src` (integer) and `−motion/scale` (sub-pel) agree. No sign was flipped to make this match: the sign comes straight from FFmpeg's definition `src = dst + motion/scale`, and the probe confirmed it before I ran the test.

**Secondary configs** (reported, not gating):
- **h264_nvenc** QP 15, bf 0, refs 1: median error 0.00 on all four shifts, ≥99.9% of blocks within 0.5 px.
- **mpeg4** q 2, bf 0: median error 0.00 on all four shifts, 94–98% of blocks within 0.5 px.
- **libx264 with the default ref (preset medium → ref=3)** gives the wrong reference:

  | shift | median recovered | median error | share of blocks at 1× / **2×** / 3× the shift |
  |---|---|---|---|
  | (4,0) | (4,0) | 0.00 | 100% / 0% / 0% |
  | (0,3) | (0,6) | 3.00 | 35% / **65%** / 0.1% |
  | (−5,2) | (−10,4) | 5.39 | 36% / **64%** / 0.1% |
  | (3,−4) | (6,−8) | 5.00 | 34% / **65%** / 0.1% |

  Roughly two-thirds of the blocks predict from t−2, and nothing in the exported MV marks this. **Any comparison against GT must use a single reference frame (x264 `ref=1`, NVENC `refs=1`) with bframes 0.** Otherwise the "reference = t−1" assumption in `compare.py` is false for most blocks. This finding should be carried into the sweep design.

**A: PASS.**

---

## 4. Validation B: real GT on alley_1 (`val_BC_alley.py` → `results/val_BC.json`)

**Setup:**
- libx264 CRF 12, medium, bframes 0, keyint 250. All 50 frames: 1 I and 49 P.
- Kept means inside the image and ≥50% of pixels valid. Excluded blocks: 5 360 outside the image and 1 302 below the valid threshold (final, ref 1).
- **Textured** = the block's mean luma-gradient magnitude in the top quartile (threshold 7.98 for final).
- **Moving** = |mean GT| > 1 px.

| run | subset | blocks kept | **median EPE** | mean EPE | p90 EPE | **zero-MV fraction** |
|---|---|---|---|---|---|---|
| **final, ref=1** (primary) | all | 179 274 | **0.226** | 0.555 | 0.983 | **0.0012** |
| | moving (GT > 1 px) | 163 939 | 0.228 | 0.563 | 1.005 | 0.0008 |
| | textured ∧ moving | 39 313 | **0.186** | 0.598 | 1.212 | 0.0002 |
| clean, ref=1 | all | 175 916 | 0.171 | 0.494 | 0.932 | 0.0012 |
| | textured ∧ moving | 38 834 | 0.159 | 0.553 | 1.263 | 0.0002 |
| final, default ref (3) | all | 179 771 | 0.560 | 1.230 | 2.729 | 0.0009 |
| | textured ∧ moving | 39 461 | 0.742 | 1.320 | 2.771 | 0.0002 |

- As expected, the median EPE is small: 0.19 px on textured moving blocks. Clean is lower than final (no motion blur or fog). The default ref (3) is about 2.5× worse, which is consistent with §3.
- Gate results for final, ref=1, over all kept blocks:

  | τ | GT moving | MV moving | MV and GT agree | EPE ≤ τ |
  |---|---|---|---|---|
  | 0.5 | 99.8% | 99.3% | 99.2% | 78.8% |
  | 1 | 91.4% | 88.4% | 88.4% | 90.2% |
  | 2 | 22.6% | 26.7% | 94.7% | 95.2% |

- **Caveat:** alley_1 is almost entirely moving (camera motion; GT > 0.5 px on 99.8% of blocks). So the zero-MV fraction and the τ = 0.5 gate say little about static-region behaviour on this sequence. That needs a sequence with real static regions.

## 5. Validation C: speed (alley_1, 50 frames, 1024×436, single run, laptop)

| stage | frames/s |
|---|---|
| PNG load (PIL) | 46–94 |
| **encode** libx264 CRF 12 medium bf0 | **97–107** |
| encode libx264 CRF 12 medium bf2 | 75 |
| encode h264_nvenc QP 15 | 132 |
| encode mpeg4 q 2 | 362 |
| **MV extraction** (h264 decode + export_mvs + to_ndarray) | **596–1102** |
| MV + per-MB QP extraction | 499–811 |
| MV extraction, mpeg4 stream | 1258 |
| GT comparison (`compare_sequence`, includes .flo/mask reads) | 13–18 |

These are single-run timings, so they vary by roughly ±30% between runs. Encode is timed on frames already in memory. Reading the GT files is now the slowest stage, not the codec.

---

## Notes and deviations
1. **VIRAT appeared during the session** (§1). The inventory reflects its state at 18:06.
2. **mpeg4 fixed quantiser** is implemented as `qmin=qmax=q`, not `flags=+qscale/global_quality`. The latter is a no-op through PyAV (§2.2).
3. **Single reference is required** for MV-vs-GT comparison (§3). B-frame (`source=+1`) blocks aren't compared: Sintel ships only forward flow, and the reference picture is unknown.
4. The **gate definition** (§2.4) is my interpretation of "gate decisions for τ".
5. The one inventory bug I found (PyAV int `pict_type` read as "no B-frames") was fixed before the numbers above were produced.

VALIDATION: PASS — Synthetic shifts are recovered exactly (median error 0.00 px, correct sign on both axes, ≥99.8% of blocks within 0.5 px with x264 ref=1). Real Sintel GT gives a median EPE of 0.23 px (0.19 px on textured moving blocks) on alley_1. One condition applies: comparisons must use a single reference frame, because x264's default ref=3 silently sends about 65% of blocks to t−2.
