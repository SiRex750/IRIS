# Item 6: B-frame cells with no past-pointing vector (supplementary)

## Handling, from the code


How the code handles them (p3/pilot_v2/run_pilot_v2.py run_one, used unchanged by the Sintel sweep; the CCTV runner
p3/virat/run_virat.py cell_arrays copies the same mapping):
    for sgn in (1, -1):  # future first, past overrides
        ... paint(...) ; cmv[ci] = mv[sel][bi] ; cdir[ci] = sgn
    has = cdir != 0 ; reuse = has & (|use| < tau)
  - a cell covered by a past-pointing block (FFmpeg source = -1) uses that vector, whether or not a future-pointing
    block also covers it (a bi-predicted block is exported as one past and one future entry, so the past one wins);
  - a cell covered ONLY by future-pointing blocks (source = +1) uses the FUTURE vector: B_naive takes its raw length
    |mv|; B_scaled uses -mv / d_fut (run_one: `use = cmv / cd * where(cdir == 1, -1, 1)`). It is gated like any
    other cell - not recomputed;
  - a cell with no vector at all (intra block, or not covered) has cdir = 0 -> has = False -> never reused (recompute).
CCTV: mv_q holds the past vector if present, else the future one (cell_dir -1 / +1 / 0), gated as |MV| < tau
(naive); the confirmatory B arms use naive vectors only.

## How often (B-frames only; tau = 1, B naive)

| data | B arm | B-frames | future-only / valid, median | min | max | no vector / valid, median | future-only / valid, pooled | share of stale cells that are future-only, median | pooled |
|---|---|---|---|---|---|---|---|---|---|
| Sintel final | x264_crf23_bf2 | 680 | 0.3013 | 0.1871 | 0.4295 | 0.0226 | 0.3000 | 0.0792 | 0.0893 |
| Sintel final | x264_qp24_bf2_pb1 | 680 | 0.2367 | 0.1509 | 0.3719 | 0.0291 | 0.2453 | 0.0846 | 0.0832 |
| Sintel final | nvenc_qp28_bf2 | 680 | 0.1155 | 0.0329 | 0.2860 | 0.0235 | 0.1324 | 0.0698 | 0.0868 |
| Sintel final | nvenc_qp28_bf2_bqeq | 680 | 0.2191 | 0.1013 | 0.3823 | 0.0611 | 0.2248 | 0.0679 | 0.0731 |
| Sintel clean | x264_qp24_bf2_pb1 | 680 | 0.2337 | 0.1611 | 0.3589 | 0.0183 | 0.2392 | 0.0346 | 0.0491 |
| Sintel clean | nvenc_qp28_bf2_bqeq | 680 | 0.2139 | 0.1454 | 0.3070 | 0.0285 | 0.2195 | 0.0429 | 0.0505 |
| CCTV confirmatory (27 eligible) | x264_qp24_bf2_pb1 | 5089 | 0.1497 | 0.1029 | 0.3111 | 0.0009 | 0.1602 | 0.1734 | 0.1687 |
| CCTV confirmatory (27 eligible) | nvenc_qp28_bf2_bqeq | 5089 | 0.0340 | 0.0113 | 0.1583 | 0.0009 | 0.0450 | 0.1671 | 0.1630 |

Self-check: recomputed valid and B_naive stale cells equal results.csv in 238/238 arm x sequence/clip rows.
