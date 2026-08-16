# MLVU codec-mode baseline (IRIS)

- repo HEAD: `<unavailable: Command '['git', 'rev-parse', 'HEAD']' returned non-zero exit status 3221225794.>` (dirty=None)
- config hash: `e67562fb00850bd1`
- n_per_task=25, max_duration_s=600.0, seed=42
- scene_segmentation=`codec`, graph_mode=`scene_sparse`

**M-Avg: 0.340**  (unweighted mean of per-task accuracy, N=150)

Overall option-parse failure rate: 3.3%  (threshold: 10%, abort if exceeded)
Ingest failures: 0

## Per-task results

| Task | Name | N | Correct | Accuracy | Random floor | Above floor | Parse-fail rate | Skipped (length) | Skipped (no video) |
|---|---|---|---|---|---|---|---|---|---|
| AR | Anomaly Recognition | 25 | 8 | 0.320 | 0.250 | True | 0.0% | 0 | 175 |
| AO | Action Order | 25 | 4 | 0.160 | 0.250 | False | 4.0% | 0 | 234 |
| AC | Action Count | 25 | 4 | 0.160 | 0.250 | False | 0.0% | 0 | 181 |
| NQA | Needle QA | 25 | 7 | 0.280 | 0.250 | True | 12.0% | 0 | 291 |
| PQA | Plot QA | 25 | 10 | 0.400 | 0.250 | True | 0.0% | 0 | 428 |
| TR | Topic Reasoning | 25 | 18 | 0.720 | 0.250 | True | 4.0% | 0 | 214 |

## Excluded tasks

- ER: egocentric (excluded per spec)
- SSC: generation task (excluded per spec)
- VS: generation task (excluded per spec)
