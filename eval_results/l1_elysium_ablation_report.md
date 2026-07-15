# Phase 1 - L1 Elysium Frame Admission Ablation Report

Evaluating Variant C (Full IRIS) vs Variant A (LRU Baseline) and Variant B (Action Only) across the evaluated VIRAT dataset (8 videos, 24 queries).

## Budget: 1% Retention

| Metric | Variant A (LRU) | Variant B (Action Only) | Variant C (Full IRIS) |
| :--- | :---: | :---: | :---: |
| **Cache Hit Rate** | 12.50% | 58.33% | 25.00% |
| **Eviction Regret Rate** | 87.50% | 41.67% | 75.00% |

### Paired Bootstrap CIs (1000 resamples)
- **Hit Rate Contrast (C vs. A):** Mean diff = +0.1250 | 95% CI = [-0.0844, +0.3333]
- **Hit Rate Contrast (C vs. B):** Mean diff = -0.3333 | 95% CI = [-0.5417, -0.1240]
- **Eviction Regret Contrast (C vs. A):** Mean diff = -0.1250 | 95% CI = [-0.3333, +0.0844]
- **Eviction Regret Contrast (C vs. B):** Mean diff = +0.3333 | 95% CI = [+0.1240, +0.5417]

## Budget: 2% Retention

| Metric | Variant A (LRU) | Variant B (Action Only) | Variant C (Full IRIS) |
| :--- | :---: | :---: | :---: |
| **Cache Hit Rate** | 12.50% | 62.50% | 41.67% |
| **Eviction Regret Rate** | 87.50% | 37.50% | 58.33% |

### Paired Bootstrap CIs (1000 resamples)
- **Hit Rate Contrast (C vs. A):** Mean diff = +0.2917 | 95% CI = [+0.0417, +0.5000]
- **Hit Rate Contrast (C vs. B):** Mean diff = -0.2083 | 95% CI = [-0.3750, -0.0417]
- **Eviction Regret Contrast (C vs. A):** Mean diff = -0.2917 | 95% CI = [-0.5000, -0.0417]
- **Eviction Regret Contrast (C vs. B):** Mean diff = +0.2083 | 95% CI = [+0.0417, +0.3750]

## Budget: 5% Retention

| Metric | Variant A (LRU) | Variant B (Action Only) | Variant C (Full IRIS) |
| :--- | :---: | :---: | :---: |
| **Cache Hit Rate** | 62.50% | 83.33% | 79.17% |
| **Eviction Regret Rate** | 37.50% | 16.67% | 20.83% |

### Paired Bootstrap CIs (1000 resamples)
- **Hit Rate Contrast (C vs. A):** Mean diff = +0.1667 | 95% CI = [-0.0833, +0.4167]
- **Hit Rate Contrast (C vs. B):** Mean diff = -0.0417 | 95% CI = [-0.1250, +0.0000]
- **Eviction Regret Contrast (C vs. A):** Mean diff = -0.1667 | 95% CI = [-0.4167, +0.0833]
- **Eviction Regret Contrast (C vs. B):** Mean diff = +0.0417 | 95% CI = [+0.0000, +0.1250]

> [!CAUTION]
> **STOP Condition Triggered at 5% budget!**
> Variant C's performance is not statistically distinguishable from Variant B.
> - Hit Rate contrast CI [-0.1250, +0.0000] crosses zero.
> - Eviction Regret contrast CI [+0.0000, +0.1250] crosses zero.

## Budget: 10% Retention

| Metric | Variant A (LRU) | Variant B (Action Only) | Variant C (Full IRIS) |
| :--- | :---: | :---: | :---: |
| **Cache Hit Rate** | 83.33% | 87.50% | 95.83% |
| **Eviction Regret Rate** | 16.67% | 12.50% | 4.17% |

### Paired Bootstrap CIs (1000 resamples)
- **Hit Rate Contrast (C vs. A):** Mean diff = +0.1250 | 95% CI = [+0.0000, +0.2917]
- **Hit Rate Contrast (C vs. B):** Mean diff = +0.0833 | 95% CI = [+0.0000, +0.2083]
- **Eviction Regret Contrast (C vs. A):** Mean diff = -0.1250 | 95% CI = [-0.2917, +0.0000]
- **Eviction Regret Contrast (C vs. B):** Mean diff = -0.0833 | 95% CI = [-0.2083, +0.0000]

> [!CAUTION]
> **STOP Condition Triggered at 10% budget!**
> Variant C's performance is not statistically distinguishable from Variant B.
> - Hit Rate contrast CI [+0.0000, +0.2083] crosses zero.
> - Eviction Regret contrast CI [-0.2083, +0.0000] crosses zero.

## Final Verdict

**[RED]** The STOP condition was triggered. The complex motion geometry signals do not show a statistically significant improvement over Variant B (Action Only) at extreme low-retention budgets. We report this failure as-is without tuning thresholds or parameters.