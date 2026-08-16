# MLVU four-arm scene-segmentation ablation (IRIS)

Identical subset as the codec baseline (`eval_results/MLVU_codec_baseline.{json,md}`):
n_per_task=25, max_duration_s=600.0, seed=42, 145 unique videos, 150 sampled
questions across AR/AO/AC/NQA/PQA/TR (ER + generation tasks VS/SSC excluded
per spec). The same MC prompt (question + numbered options + instructions) drives
both retrieval and the answerer in every arm, unchanged from the baseline
mechanism — a deliberate choice so the codec arm can be checked for exact
reproduction (see GUARD (b) below).
Same granite4:micro answerer (temperature=0.0).

- repo HEAD: `77f625badae67ad24fc57a9e986ef24e0ea6fe4b` (dirty=True — untracked
  scratch/experiment files from other tracks present, none touching
  ingest/query/config)
- Reuse structure: `iris.charon_v.parse_video` (decode + codec parse) and
  `_demux_packet_curve` run ONCE per video, shared across all four arms;
  only scene_id assignment / graph build / retrieval / answer are re-run
  per arm via `iris.ingest._build_index_from_records`.
- Wall clock: ~9h15m (12:18 PM → 9:39 PM), 600/600 work items, 0 ingest
  failures, single retry-wrapper attempt (no restarts needed).

## GUARDS

- **(a) Question set identical across all four arms**: TRUE by construction
  (one shared sampled pool, hash `35ab29f354455a48`, drives all four arms) —
  and confirmed by the checkpoint: every (arm, task, question_id) triple
  present, 150 per arm, 600 total, 0 missing.
- **(b) codec arm reproduces the committed baseline EXACTLY**: **PASSED.**
  `codec_reproduces_baseline: true`, `codec_vs_baseline_diff: {}`. Codec-arm
  M-Avg = 0.340 (baseline: 0.340); per-task n and accuracy match the
  baseline exactly for all six tasks. The ablation is not contaminated.

## Per-task accuracy × 4 arms + M-Avg × 4 arms

| Arm | AR | AO | AC | NQA | PQA | TR | **M-Avg** |
|---|---|---|---|---|---|---|---|
| codec | 0.320 | 0.160 | 0.160 | 0.280 | 0.400 | 0.720 | **0.340** |
| fixed_count | 0.360 | 0.320 | 0.240 | 0.280 | 0.440 | 0.720 | **0.393** |
| fixed_time_matched | 0.280 | 0.160 | 0.160 | 0.280 | 0.360 | 0.680 | **0.320** |
| fixed_seconds (60s) | 0.320 | 0.280 | 0.240 | 0.160 | 0.480 | 0.640 | **0.353** |

(N=25/task/arm, N=150/arm, N=600 total.)

## codec − fixed_time_matched (per task)

| Task | Δ (codec − fixed_time_matched) |
|---|---|
| **AR** (called out) | **+0.040** |
| AO | +0.000 |
| AC | +0.000 |
| **NQA** (called out) | **+0.000** |
| PQA | +0.040 |
| TR | +0.040 |
| **M-Avg** | **+0.020** |

AR moves (+0.040, codec ahead) in the direction where a real boundary-time
vs boundary-placement effect would be expected to show up; NQA shows no
movement at all (+0.000) — needle-retrieval accuracy is apparently
insensitive to whether scene boundaries are equal-time vs equal-survivor-count
once N is held fixed. AO/AC are flat as expected (answerer-floored).

## codec − fixed_count (per task)

| Task | Δ (codec − fixed_count) |
|---|---|
| **AR** (called out) | **−0.040** |
| AO | −0.160 |
| AC | −0.080 |
| **NQA** (called out) | **+0.000** |
| PQA | −0.040 |
| TR | +0.000 |
| **M-Avg** | **−0.053** |

fixed_count (equal survivor-count buckets, same N as codec) beats codec on
every task except AR and PQA — most sharply on AO (−0.160) and AC (−0.080).
NQA is flat here too. Since fixed_time_matched (equal *time* buckets, same N)
does *not* reproduce this advantage (its M-Avg is below codec, not above),
the fixed_count arm's advantage looks like it comes specifically from
*where survivors land relative to boundaries* (equal survivor-count
placement), not just from decoupling boundary count from codec valleys.

## Video-clustered bootstrap 95% CI: codec vs fixed_time_matched M-Avg delta

- Point estimate (observed): **+0.0200** (codec − fixed_time_matched)
- Video-clustered bootstrap (seed=42, B=10000, resampling videos with
  replacement, n=123 unique videos shared by the codec+fixed_time_matched
  question pool, M-Avg recomputed per resample as the unweighted mean over
  tasks with n>0 in that resample):
- **95% CI: [−0.0299, +0.0741]**
- **CI includes 0** — the codec vs fixed_time_matched M-Avg delta is not
  statistically distinguishable from zero at video-cluster resolution with
  this sample size (N=150 questions / 123 videos). Directionally positive
  (codec ahead) but not a confirmed effect.

## Excluded tasks

- ER: egocentric (excluded per spec)
- SSC: generation task (excluded per spec)
- VS: generation task (excluded per spec)
