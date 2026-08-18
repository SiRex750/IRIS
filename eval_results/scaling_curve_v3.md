# Query-latency scaling exponents with confidence intervals (ledger C1.5)

## Verdict: **PRINT-SAFE**

- flat headline (N>=1102) 95% CI on k: (2.020474224515457, 2.1254813814709306)
- scene_sparse headline (N>=1102) 95% CI on k: (0.8005433482385389, 0.884257833906474)
- CIs overlap: False; scene_sparse CI upper bound < 1.0: True

## Survivor census

- Population (UCF Anomaly-Part-1 + Testing_Normal_Videos_Anomaly): 350 clips (container-frame proxy only, NOT survivor N: min=137, median=2021, max=126553)
- Measured survivor N (actual fit support): n=31 clips, min=24, median=333, max=13506
- Histogram by decade of N:
  - 10-99: 4
  - 100-999: 20
  - 1000-9999: 6
  - 10000-99999: 1
- Excluded (graph_edge_mode mismatch, not fully_connected): ['Normal_Videos_924', 'Normal_Videos_935']

## Bin occupancy (target >=5 clips/bin)

| target N | n clips | meets target | clips |
|---:|---:|:---:|---|
| 1102 | 7 | yes | Arrest016, Assault042, Arrest030, Normal_Videos_925, Arrest039, Arson042, Abuse036 |
| 2963 | 2 | NO -- shortfall | Abuse042, Normal_Videos_940 |
| 6559 | 2 | NO -- shortfall | Arrest047, VIRAT_S_040001_01_000448_001101 |
| 13506 | 1 | NO -- shortfall | Arson019 |
| 91 | 5 | yes | Normal_Videos_289, Normal_Videos_908, Assault036, Abuse023, Assault045 |
| 212 | 13 | yes | Abuse025, Arrest031, Abuse037, Normal_Videos_891, Normal_Videos_168, Arrest022, Arson050, Normal_Videos_758, Arson040, Arrest007, Arrest036, Abuse043, Arson002 |

## Fits (clip-level bootstrap, seed=42, B=10000)

### flat

| fit range | n_points | k | 95% CI | R^2 |
|---|---:|---:|---|---:|
| full_range | 17 | 1.980 | [1.951, 2.006] | 0.9978 |
| full_range_excl_shortcut_violations | 15 | 1.980 | [1.946, 2.010] | 0.9982 |
| N_ge_1102_headline | 4 | 2.063 | [2.020, 2.125] | 0.9994 |
| N_ge_1102_minus_subnoisefloor | 4 | 2.063 | [2.020, 2.125] | 0.9994 |

**Censored-as-lower-bound** (n_completions=17, n_censored_substituted_at_cap=2): k=2.597 (R^2=0.9195) -- k here is a LOWER BOUND on the true exponent (censored points' true latency is >= the substituted cap value), not a point estimate; no bootstrap CI reported for a bound.

Censored (timed_out) points, excluded from completions-only fits: ['Arrest047(N=6559)', 'Arson019(N=13506)']

### scene_sparse

| fit range | n_points | k | 95% CI | R^2 |
|---|---:|---:|---|---:|
| full_range | 19 | 0.497 | [0.461, 0.530] | 0.8980 |
| full_range_excl_shortcut_violations | 17 | 0.503 | [0.462, 0.539] | 0.8968 |
| N_ge_1102_headline | 6 | 0.834 | [0.801, 0.884] | 0.9830 |
| N_ge_1102_minus_subnoisefloor | 6 | 0.834 | [0.801, 0.884] | 0.9830 |

**Censored-as-lower-bound** (n_completions=19, n_censored_substituted_at_cap=0): k=0.497 (R^2=0.8980) -- k here is a LOWER BOUND on the true exponent (censored points' true latency is >= the substituted cap value), not a point estimate; no bootstrap CI reported for a bound.

## Guards

- shortcut_pct guard: FAIL -- see violations (20 scene_sparse completions checked)
  - VIOLATIONS: [{'label': 'Assault036', 'shortcut_pct': 20.0}, {'label': 'Abuse037', 'shortcut_pct': 10.0}]
- salience weights: ledger S6: the corpus this fit rests on (existing v2/textquery corpus + this task's census/latency extension) was built at salience weights (0.5, 0.3, 0.2) throughout (verified via config_snapshot on every index_cache .npz used) -- NOT the frozen production default (0.8, 0.1, 0.1). This is internally consistent (every clip in this fit uses the same weights, so N-vs-latency comparisons within the fit are fair), but the survivor N values here are NOT comparable to any retention/survivor-count number computed elsewhere under the frozen weights -- stated inline per ledger S6, not excluded, since excluding would leave no corpus to fit.

## Censoring table

| label | arm | N | outcome | cap (s) |
|---|---|---:|---|---:|
| Arrest047 | flat | 6559 | timed_out | 3600.0 |
| Arson019 | flat | 13506 | timed_out | 3600.0 |

## Provenance

- git HEAD: `9c66393d1624ebae5443f62ddbe8f5fb8dc12b7f` (dirty: True, 94 changed files)
- query_source: text, n_queries_per_clip: 50
- watchdog: wall_time_cap_sec=3600.0, mem_cap_bytes=29000000000
- bootstrap: n_boot=10000, seed=42, resample WITHIN each N-bucket, refit, repeat (clip-level resampling, not bucket-mean resampling)
- harness config hashes:
  - `scripts/scaling_curve_ci.py`: `79c22304e65b31aaffca61348514eeb9ffa8c015`
  - `scripts/_scaling_curve_ci_census_worker.py`: `b8c6669f0e82115eeacc88917be64d82fef62ae5`
  - `scripts/scaling_curve_v2.py`: `16e9e40823a52bba5bfb0b6e9cefcc1176e19c7a`
  - `scripts/_scaling_curve_v2_worker.py`: `b6baa5b919350a97efc585277bdbca137ade912d`
  - `scripts/scaling_curve_v3_report.py`: `de6589518df3b67a0df775888f80da9959e42d01`

Full raw data: `eval_results/scaling_curve_v3.json`.
