# POST HOC — matched-reuse comparison: CodecSight policy vs gates without accumulation
**POST HOC. Not pre-registered. It changes no verdict** of `p3/codecsight_sim/PREREGISTRATION.md` (9f5b5a3), of the
main pre-registration or of the confirmatory test. Written 2026-10-11 before any number of this analysis was computed.
Repository HEAD when written: 70e2a5d (branch sonu/p3). Saved data only (same inputs and loaders as
`p3/codecsight_sim/simulate.py`); no encode, no MV extraction, no RAFT.

## Question
Is the CodecSight policy (P) fresher than a gate without accumulation at the same reuse, when every method is
scored against the same truth?

## 1. Common pieces (unchanged from the pre-registration)
- Cells: 4x4 px grid, painted vectors exactly as `simulate.py` (Sintel: painted float |MV|; CCTV: q = floor(4|MV|),
  255 = no vector). A cell is **dynamic at t** if it has no vector or |MV| >= tau (CCTV: q >= 4 tau).
- Token footprints: derived (InternVL3 16x16 grid: Sintel 64 x 27.25 px, CCTV 80 x 45 px), and square 16, 32,
  64 px tiled from (0, 0). Token membership for the dynamic rule: every cell whose footprint overlaps the token with
  positive area (`simulate.token_matrix`). A token is **dynamic at t** if any member cell is dynamic (max rule plus
  no-vector rule).
- tau: 0.25, 0.5, 1 px.
- Refresh (all methods): every I-frame, and 16 frames after the previous refresh (`simulate.native_refresh`).
  Every unit is computed at a refresh frame. Native rate, every frame.
- Scored frames: t = 1..n-1. Per-frame cell GT g_t(c) and cover rule as in pre-registration §3 (Sintel
  gt_cover >= 0.5; CCTV splat_count >= 8).

## 2. Methods
- **G1, per-cell gate, no accumulation.** Unit = cell. Computed at t if t is a refresh frame or the cell is dynamic
  at t. Otherwise reused.
- **G2, token gate, no accumulation.** Unit = token. Computed at t if t is a refresh frame or the token is dynamic
  at t. Otherwise reused.
- **P, pre-registered policy.** Unit = token. Active set = union of dynamic tokens of every frame after the last
  refresh up to and including t. Computed at t if t is a refresh frame or the token is active. Otherwise reused.
- G2 and P: footprints derived, 16, 32, 64 px x tau 0.25, 0.5, 1 (12 configurations each). G1: tau 0.25, 0.5, 1
  (3 configurations).
- **Cell state for token methods (our choice).** Derived tokens overlap, so a boundary cell can belong to two
  tokens. For cell-level scoring each cell takes the computed/reused state of the one token that contains its
  centre (4c + 2 px on each axis; token index floor((4c + 2) * 16 / L) for derived, floor((4c + 2) / s) for square
  s). For square tokens this is the only token that contains the cell. The dynamic rule still uses overlap
  membership.

## 3. One truth for all methods
- **Last computed** L_t(c): the last frame <= t at which c's unit was computed (a refresh frame counts). For a
  cell computed at t, L_t(c) = t.
- **Summed displacement** S_t(c) = sum of g_k(c) for k = L_t(c)+1 .. t, at the fixed cell (empty sum = 0 for a cell
  computed at t). Same approximation as pre-registration §3.
- **Cell valid at t:** cover rule holds at t, and every g_k(c) in that sum is finite (finite-terms rule). A computed
  cell is valid iff its cover rule holds at t.
- **Stale at t:** reused at t, valid, and |S_t(c)| > 2 px.
- Validity depends on L, so the valid denominator can differ between methods on the same encode. Both counts are
  written to the output.

## 4. Metrics (cell level, per encode = arm x sequence/clip, scored frames 1..n-1)
- REUSE_c = reused valid cell-frames / valid cell-frames.
- STALE_c = stale valid cell-frames / valid cell-frames.

## 5. Matched-reuse comparison
- Per unit and per P configuration (footprint f, tau): take G2's four points (REUSE_c, STALE_c) at the same tau and
  footprints derived, 16, 32, 64. Points with equal REUSE_c are merged (mean STALE_c). If P's REUSE_c lies within
  [min, max] of G2's REUSE_c, G2's STALE_c is linearly interpolated there (piecewise-linear through the points
  sorted by REUSE_c; no extrapolation). Report D = STALE_c(P) - interpolated STALE_c(G2).
  - If G2 has a single distinct REUSE_c and P equals it, D uses that point. Otherwise out of range: no D, counted.
- Per arm and P configuration: median D over units with a D, the number of units with D < 0 (P lower), D = 0, and
  the number out of range.
- The same against G1: G1's three points (tau 0.25, 0.5, 1) per unit, interpolated at P's REUSE_c with the same
  rules. Where P's REUSE_c is outside G1's range, the count out of range is reported, and the table says so.
- Primary table: P at derived / tau 0.5 against G2 at tau 0.5 and against G1; all 12 P configurations are
  reported as well. None of these is a verdict.

## 6. Data
- CCTV: the 27 eligible clips x 9 arms of pre-registration §5 (243 encodes).
- Sintel: 23 sequences x 17 arms (391 encodes).

## 7. Validation before results (stop if either fails)
- (a) G1 with per-frame truth (last computed = t-1 for a reused cell, so S = g_t; refresh = I-frames only, because
  the pre-registered bare gate has no 16-frame refresh and recomputes only I-frame cells) must reproduce
  `g_valid`, `g_reused`, `g_moved`, `g_stale` of `p3/codecsight_sim/per_unit.csv` exactly, for every arm and the
  three tau, on 3 CCTV clips and 3 Sintel sequences (first three in sorted order). g_moved is checked as valid
  cells with |g_t| > 2 px.
- (b) P at derived / tau 0.5 with token-level counting and the pre-registered truth (S summed since the last
  refresh; token valid if >= half its member cells are valid; moved if a valid member cell has |S| > 2 px;
  reused = not refresh and not active) must reproduce `valid_tf`, `reused_tf`, `stale_tf` of `per_unit.csv`
  (config primary) exactly, for every arm on the same units.
- The truth accumulator and the reuse masks used in the main run are the same functions exercised in (a) and (b);
  (a) and (b) only change the reset mask (every frame; refresh frames) and the counting level.

## 8. Outputs (folder `p3/codecsight_sim/posthoc_matched/`; every file starts with POST HOC)
- `posthoc.py`, `validation.txt`, `per_unit_posthoc.csv`, `matched.md` (matched-reuse tables, per-arm medians of
  REUSE_c and STALE_c per method/configuration), `fig_matched.png`: STALE_c against REUSE_c, one panel per CCTV
  arm (x264 CRF 12, CRF 45, NVENC QP 18, QP 45), one point per method/configuration at the per-arm median over the
  27 clips of each coordinate, G1, G2 and P distinguished by marker, labelled by tau and token size.
- Runtime is logged.
