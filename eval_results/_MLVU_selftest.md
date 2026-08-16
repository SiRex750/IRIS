# MLVU codec-mode baseline (IRIS)

- repo HEAD: `15a7d7929c40ff4669849740bd88f2ec413aa39c` (dirty=True)
- config hash: `e67562fb00850bd1`
- n_per_task=3, max_duration_s=600.0, seed=42
- scene_segmentation=`codec`, graph_mode=`scene_sparse`

**M-Avg: 1.000**  (unweighted mean of per-task accuracy, N=18)

Overall option-parse failure rate: 0.0%  (threshold: 10%, abort if exceeded)
Ingest failures: 0

## Per-task results

| Task | Name | N | Correct | Accuracy | Random floor | Above floor | Parse-fail rate | Skipped (length) | Skipped (no video) |
|---|---|---|---|---|---|---|---|---|---|
| AR | Anomaly Recognition | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |
| AO | Action Order | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |
| AC | Action Count | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |
| NQA | Needle QA | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |
| PQA | Plot QA | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |
| TR | Topic Reasoning | 3 | 3 | 1.000 | 0.250 | True | 0.0% | 0 | 0 |

## Excluded tasks

- ER: egocentric (excluded per spec)
- SSC: generation task (excluded per spec)
- VS: generation task (excluded per spec)
