# Paper 3 figure captions (draft)

All figures are made by `p3/figures/make_figures.py` from saved data only: no new encodes, MV extraction,
RAFT runs or metrics. Before plotting, the script asserts that the medians it plots equal the values quoted
in the paper (output: `make_figures.log`).

Shared counting rule. Frames are split into 4x4-px cells. A cell is **reused** when its codec motion vector
has |MV| < tau (tau = 1 px). In Sintel, a cell is **valid** when at least half of its area has
ground-truth (GT) flow (occluded and invalid pixels excluded, GT forward-splatted onto the frame's grid). A
**stale** cell is a valid, reused cell whose GT |flow| is > 2 px. The metric is computed per sequence (or
per clip) and pooled over that sequence's frames. Medians are taken across sequences (or clips); values
are never pooled across sequences.

## Fig. B1: B-frames at matched QP (Sintel)

**Draft caption.** B-frames raise stale reuse at matched quantiser. Each grey line is one of the 23 MPI-Sintel
training sequences (FINAL pass). The black line joins the medians. Left: x264 at QP 24, without B-frames
(bf 0) vs with 2 B-frames (pb-ratio 1, so B-frames are also coded at QP 24). Median 0.55% -> 4.4%; 23/23
sequences increase. Right: NVENC at QP 28, without B-frames vs with B-frames at the P-frame QP (B = P).
Median 3.3% -> 7.0%; 22/23 sequences increase. y axis: stale cells / valid cells, tau = 1 px, log scale.
Counting rule: bf 0 counts the P-frames that reference the previous frame (d = 1). The B-frame arm counts
only the B-frames. Their vectors are used as decoded ("naive"): they are not rescaled for reference
distance or direction, which is what a decoder-side reuse gate would see.

- Data: `p3/sweep/results.csv`; pass = final, tau = 1.0. Encodes `x264_qp24` (group d1) vs
  `x264_qp24_bf2_pb1` (group B_naive); `nvenc_qp28` (d1) vs `nvenc_qp28_bf2_bqeq` (B_naive). Column
  `stale_of_valid`.
- Metric: stale cells / valid cells per sequence, as defined above.
- Note: the scaled-vector variant (B_scaled) differs by a median of 0.0009 (x264) and 0.0005 (NVENC) from
  B_naive (review_fixes3 R4, P2b), so the choice does not change the figure.

## Fig. B2: CCTV stale reuse vs quality (VIRAT, confirmatory test)

**Draft caption.** Stale reuse on real surveillance video rises at low quality. Data: the confirmatory CCTV
test (VIRAT ground, source frames 300-599, pre-registration d62753f). It has 27 eligible clips from 7 scenes
(eligible = at least 2000 moving cell-frames). Each coloured line is one scene: the median over that scene's
eligible clips (n per scene in the legend; scene 010002 has one clip). The dashed black line is the median
over all 27 clips. Left: x264 CRF 12 vs CRF 45, 5.5% -> 24.9%. Right: NVENC constant QP 18, 23 and 45,
4.9% -> 10.1% from QP 18 to QP 45. All encodes are without B-frames. y axis: STALE_MOV at tau = 1 px.
Counting rule: VIRAT has no GT, so motion is estimated with RAFT (torchvision raft_large). STALE_MOV is the
share of moving cells (|RAFT flow| > 2 px) that the codec MV marks as reused (|MV| < 1 px). P-frames
referencing the previous frame only (d = 1).

- Data: `p3/virat_confirm/results.csv`; group d1, tau = 1.0; arms `x264_crf12`, `x264_crf45`, `nvenc_qp18`,
  `nvenc_qp23`, `nvenc_qp45`; column `stale_mov`. Eligibility from the `eligible` column (27 clips).
- Metric: STALE_MOV per clip. Then the median per scene (coloured lines) and the median over the 27 clips
  (black line).
- Note: RAFT flow is an estimate, not ground truth (V0 check: median |MV - RAFT| < 1.5 px on 26/27 eligible
  clips).

## Fig. B3: example frame (illustrative)

**Draft caption.** Illustrative example: where stale reuse happens. MPI-Sintel `shaman_2` (FINAL pass),
frame 40 (`frame_0040.png`), shown in grey. Orange cells are stale: reused (|MV| < 1 px) although GT
|flow| > 2 px. Blue cells are correctly reused: reused, with GT |flow| < 0.5 px. Left: x264 CRF 12 (0 stale,
6990 correctly reused cells). Right: x264 CRF 45 (3572 stale, 9264 correctly reused cells), bf 0, the same
frame. Reused cells with GT |flow| between 0.5 and 2 px, and cells without a valid GT, are left unmarked.
This is one frame chosen by a fixed rule, not a summary statistic. Rule: `shaman_2` is the sequence whose
x264 CRF 45 stale/valid (0.131) is the median of the 23 sequences. Frame 40 is the P-frame (d = 1) whose
stale-cell count at CRF 45 is the median of that sequence's 49 P-frames.

- Data: saved block tables `p3/sweep/blocks/final/x264_crf{12,45}__shaman_2.parquet` (decoded MVs from the
  main sweep) and the Sintel GT flow / occlusion / invalid masks. Cells are rebuilt with the main sweep's
  rules (`p3/pilot_v2/run_pilot_v2.run_one`: block paint onto 4x4 cells, GT cover >= 0.5, tau = 1 px).
- Check: before drawing, the script sums the rebuilt per-frame counts over the whole sequence and asserts
  that they equal `results.csv` for both encodes (stale, correctly reused, reused and valid cells; all four
  match exactly).
