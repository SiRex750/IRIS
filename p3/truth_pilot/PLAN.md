# Paper 3 — Truth pilot: go/no-go for two long-gap ground-truth candidates (DOT, CoTracker3 online)
Date: 2026-10-19. Written and committed **before anything in this directory is run**: no candidate is installed, no
candidate output exists. Repository HEAD when written: 8aed967 (branch sonu/p3). Code is written after this file is
committed; its commit hash is recorded with the results.

**Why this exists.** The RAFT-based truth of reading B **failed** its pre-registered drift gate
(`p3/codecsight_native2fps/PREREGISTRATION.md`, ed606f4 §3; `gates.md`: 0/25 counted clips with false-move rate
≤ 0.01). Reading A's RAFT truth also failed (R-chain, `p3/codecsight_2fps/checks.md`). This pilot **chooses a
reference to replace the failed RAFT-based truth**. It is a pilot of truth candidates only. It produces no policy
number, no V0, no encode metric and no STALE / REUSE value, and it changes no verdict of any earlier
pre-registration. If neither candidate passes, the RAFT-based truth stays failed and the arms stay descriptive.

**Deadline.** If this pilot is not finished by 2026-11-08, the paper stays as it is.

## 1. Candidates (fixed)
| | DOT | CoTracker3 online |
|---|---|---|
| Paper | Le Moing et al., "Dense Optical Tracking", CVPR 2024 | Karaev et al., "CoTracker3", 2024 (Meta) |
| Code | github.com/16lemoing/dot (MIT; its CoTracker component is CC BY-NC, RAFT BSD-3) | github.com/facebookresearch/co-tracker (CC BY-NC 4.0) |
| Weights | DOT demo defaults: `cvo_raft_patch_8.pth` (estimator), `movi_f_raft_patch_4_alpha.pth` (refiner), `movi_f_cotracker2_patch_4_wind_8.pth` (tracker) | `scaled_online.pth` (HF `facebook/cotracker3`, cc-by-nc-4.0) |
| Training data | Kubric (CVO, MOVi-F) | Kubric + pseudo-labelled real videos |
| Mode | `tracks_from_first_to_every_other_frame`, `num_tracks` 8192 (demo default) | online predictor, queries at the first frame, chunks of `2·step` frames advanced by `step` |
| Resolution | **480×856** (DOT demo default). Fallback **360×640** if 480×856 runs out of memory on any period; the fallback then applies to every clip and period, and both attempts are reported | native 1280×720 frames in; the model resizes to its fixed **384×512** internally (anisotropic for 16:9) and returns native coordinates |

Licences: both candidates are non-commercial (CC BY-NC); this is accepted for this research use. PointOdyssey is
CC BY-NC-SA 4.0; neither checkpoint above lists it as training data, and it is not downloaded. Repo commit hashes and
checkpoint sha256 are recorded with the results. Code and weights live outside the repository and are never committed.

Hardware: RTX 4050 Laptop, 6 GB (about 4.3 GB free with the desktop running), driver 610.74. VRAM fit has not been
measured when this is written; peak VRAM is recorded per run.

## 2. Clips (fixed)
The first 3 unseen-window clips (rule `unseen`, `win_first` = 600) in sorted order of
`p3/codecsight_2fps/inventory.csv`:
1. VIRAT_S_000206_07_001501_001600 (30 fps, k 15, keyint 240, 5 GOPs, native frames 600–1785)
2. VIRAT_S_000206_08_001618_001712 (30 fps, k 15, keyint 240, 5 GOPs, native frames 600–1785)
3. VIRAT_S_000207_00_000000_000045 (30 fps, k 15, keyint 240, 3 GOPs, native frames 600–1305)

**Only truth is computed on them**: no policy, no gate of any earlier file, no stale metric, no encode. Native RAFT is
the existing `p3/codecsight_native2fps/data/<clip>/raft.npz` (not recomputed).

Disclosed: all three are 30 fps clips from scenes 000206/000207; two share a scene; no 23.97 fps clip is included.
Their RAFT drift-gate rates are already known (0.248, 0.251, 0.235, `gates.md`).

**Refresh periods.** Window-relative native frames, refresh r = 0, 240, 480, …; a period runs r … min(r + 239, L − 1).
Clips 1–2: 5 periods; clip 3: 3 periods (the last ends at frame 705). Each candidate is run on each period separately,
with the period's first frame r as the query/reference frame. Frames are the decoded source frames as rgb24 (PyAV,
display order, as `collect.raft_clip`).

**Query cells.** The cells (cy, cx) of the 180×320 4×4-cell grid with cy ≡ 0 and cx ≡ 0 (mod 4): 45 × 80 = 3,600 cells,
the same for both candidates and both gates. Query point = cell centre (4cx + 1.5, 4cy + 1.5) in native pixels.
CoTracker3 tracks all 3,600 queries jointly; if that runs out of memory, 4 batches of 900 for every clip (reported).
DOT: its dense flow from frame r is sampled bilinearly at the query point mapped to the inference resolution
(x' = (x + 0.5)·W'/1280 − 0.5, same for y), and the displacement is scaled back to native pixels. Candidate
displacement D(c, t) = position at t − position at r, native px. Visibility / occlusion outputs are not used.

## 3. Gate 1 — drift
Over each refresh period, at the sampled frames s = r + 15m (m ≥ 1, s ≤ end of period), exactly the cell-frames of
the failed RAFT drift gate (ed606f4 §3), restricted to the query cells:
- a cell-frame (s, c) is **quiet** if c is valid (splat_count ≥ 8, finite; `gates.load_truth`) at every native step
  r+1 … s and its per-frame native RAFT motion |T_i(c)| **never exceeds 0.5 px** (≤ 0.5) at those steps;
- it is a **false move** if the candidate says it moved more than 2 px: |D(c, s)| > 2 px.
- Rate per clip = false moves / quiet cell-frames, over all periods. **Pass if ≤ 0.01 on all 3 clips.** A clip with no
  quiet cell-frame does not pass.

## 4. Gate 2 — agreement on short spans
Spans of ≤ 16 native frames from each period start: (r, r + d), d = 1 … 16.
- **Native RAFT chained displacement** R(c, d): start at the query point at frame r, and for t = r+1 … r+d add
  `flow_tm1` of pair t−1 → t (cell-mean RAFT, indexed on frame t−1's grid) sampled bilinearly in cell coordinates
  at the current position. The chain is dropped if it leaves the image or meets a non-finite term.
- Points **moving > 2 px**: |R(c, d)| > 2 px.
- Error = |D(c, r + d) − R(c, d)|, native px. Per clip: the median over all (period, query cell, d) moving points.
  **Pass if < 1.0 px on all 3 clips.** A clip with no moving point does not pass.

## 5. Decision and report
- A candidate **passes only if both gates pass** (all 3 clips each). Both candidates are reported either way, even if
  one fails early. Passing selects that candidate as the reference to pre-register for the truth of reading B;
  if both pass, both are reported and the choice is made in that later pre-registration, not here.
- Reported: per clip per candidate, Gate 1 rate (with counts) and Gate 2 median (with counts), and runtime (candidate
  inference seconds per clip, total wall time, peak VRAM). Nothing else.
- Fixed by this file; no tuning after results. Bugs: fix, record in `p3/truth_pilot/DEVIATIONS.md`, rerun everything,
  report both runs. Candidate outputs, code and weights are never committed.
