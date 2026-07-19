# Preliminary Grounded VideoQA Benchmark (Pillar 2)

Working tree was clean at run time (`git_dirty=false`).

**Provenance:** backend=`LlamaServerBackend` | model=`granite4:micro` | temperature=`0.0` | cache_prompt=`False` | endpoint=`http://127.0.0.1:8091/v1` | span_mode=`minmax` | span_half_width=`2.2` | git_commit=`bcf940e9cbe25799ec3fba1e8fd94e482ddc6d07` | git_dirty=`False` | timestamp_utc=`2026-07-19T23:00:24.517969+00:00`

Evaluating the end-to-end IRIS pipeline (`charon_v` -> L1 admission Variant B -> L2 PPR retrieval -> LlamaServerBackend `granite4:micro` backend) on the NExT-GQA test subset ($n=64$ questions, 59 unique videos).

## 1. Quantitative Performance Summary

| Metric | Proposed System | Uniform Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Acc@GQA** (Grounded QA) | 10.94% | 4.69% | 3.12% |
| **Acc@QA** (QA Accuracy) | 53.12% | 21.88% | 29.69% |
| **mIoP** (Mean IoP) | 0.2744 | 0.2553 | 0.2563 |
| **IoP@0.5** (Grounding Recall) | 18.75% | 15.62% | 14.06% |
| **mIoU** (Mean IoU) | 0.2334 | 0.2460 | 0.2322 |
| **IoU@0.5** (Grounding Overlap) | 14.06% | 14.06% | 14.06% |

*Note: Acc@GQA represents the percentage of questions that are both answered accurately (Acc@QA) and grounded correctly (IoP $\ge$ 0.5).*

## 2. Statistical Rigor (Paired Bootstrap CIs)

Calculated over 1000 video-level cluster bootstrap resamples. 95% Confidence Intervals (CI) gate each paired difference.

### A. Proposed vs. Uniform Baseline (Proposed - Uniform)
*   **Acc@GQA Diff:** +0.0626 (95% CI: `[+0.0149, +0.1290]`)
*   **Acc@QA Diff:**  +0.3113 (95% CI: `[+0.1691, +0.4678]`)
*   **IoP@0.5 Diff:**  +0.0312 (95% CI: `[-0.0164, +0.0938]`)
*   **mIoP Diff:**     +0.0188 (95% CI: `[-0.0204, +0.0615]`)
*   **IoU@0.5 Diff:**  -0.0003 (95% CI: `[-0.0952, +0.0909]`)
*   **mIoU Diff:**     -0.0123 (95% CI: `[-0.0496, +0.0223]`)

### B. Proposed vs. Random Baseline (Proposed - Random)
*   **Acc@GQA Diff:** +0.0777 (95% CI: `[+0.0156, +0.1501]`)
*   **Acc@QA Diff:**  +0.2343 (95% CI: `[+0.0952, +0.3871]`)
*   **IoP@0.5 Diff:**  +0.0463 (95% CI: `[-0.0159, +0.1148]`)
*   **mIoP Diff:**     +0.0176 (95% CI: `[-0.0268, +0.0644]`)
*   **IoU@0.5 Diff:**  -0.0014 (95% CI: `[-0.0923, +0.0834]`)
*   **mIoU Diff:**     +0.0016 (95% CI: `[-0.0369, +0.0393]`)

## 3. Discussion and Scientific Context

### The "Trust Gap"
The scientific literature notes a massive **Trust Gap** on Grounded VideoQA benchmarks: SOTA closed-ended QA models achieve up to ~69% Acc@QA, but drop to a meager **16% Acc@GQA** when forced to localize the visual evidence that justifies their answer. Human baselines achieve 82% Acc@QA / Acc@GQA, proving that contemporary models rely on language priors rather than visual evidence.

### Competitive Status with 2B Agentic Frontier
With our proposed system achieving **10.94% Acc@GQA** on this evaluation:
*   We outperform simple baseline samplers (Uniform and Random) by a statistically significant margin.
*   Measured Acc@GQA: 10.94%.
