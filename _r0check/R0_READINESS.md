# R0 readiness check — 2026-08-04

Read-only verification. No files moved/renamed/copied/deleted, no venv changes,
no experiment scripts run, worktrees left in place.

## Q1 — Do the committed ground-truth CSVs contain per-frame gold labels?

**YES.**

Header (verbatim, `per_frame_ground_truth_scores\*.csv`):
```
frame_idx,ground_truth,action_score_propagated,is_retained_tier
```

`ground_truth` is a per-frame 0/1 gold anomaly flag. Verified by value, not by name:

- `Arson011_x264.csv` — 1267 lines (1 header + 1266 rows). ground_truth: 0 -> 409 rows, 1 -> 857 rows.
  Transition at frame 150 (`149,0,...` then `150,1,...`). Last 1 at frame_idx 1265.
- `Normal_Videos_003_x264.csv` — 2823 lines (1 header + 2822 rows). ground_truth: 0 -> 2822 rows. No 1s.

169 such CSVs exist (19 anomalous + 150 normal).

The missing `Temporal_Anomaly_Annotation.txt` is **not needed** to recover gold labels
for these 169 videos. It would be needed only for videos outside this set.

## Q2 — Are the 19 annotated anomalous videos present locally?

**YES — 19 / 19 present** in `C:\Users\akash\Downloads\Anomaly-Videos-Part-1`.

No filename mismatch: CSV basenames already carry the `_x264` suffix
(`Arson011_x264.csv` <-> `Arson011_x264.mp4`).

## Blocking gap for pooled AUC

The 150 **Normal** videos are ABSENT. `Normal_Videos_*` search over C:\Users\akash
returns 161 hits, all CSVs (11 action-score + 150 ground-truth), zero .mp4.
Downloads contains only `Anomaly-Videos-Part-1` (Abuse/Arrest/Arson/Assault, 200 files, ~6.0 GB).

Gold labels exist for 169 videos; video exists for 19 of them.

## Decode cross-check (Arson011)

| Source | Value |
|---|---|
| PyAV packets (size>0) | 1266 |
| PyAV declared `s.frames` | 1266 |
| CSV data rows | 1266 |
| stageA_per_video `n_frames_decoded` | 1266 |
| stageA_per_video `n_positive_frames` | 857 |
| ground_truth=1 rows counted here | 857 |

All match. The flagged off-by-one is in the annotation source, not the CSV:
`annotation_validation.json` records Arson011 span `{680, 1267}` against
`decoded_n_frames: 1266`. Same pattern for Arson016 (`{1000,1796}` vs 1795) and
Assault006 (`{1185,8096}` vs 8096). Arson011 has `n_spans: 2`, which is why
ground_truth=1 is not one contiguous block.

## Annotation file

`Temporal_Anomaly*` under C:\Users\akash -> NO MATCHES. Still absent.
The Downloads extraction contains only .mp4 files, no annotation text.
