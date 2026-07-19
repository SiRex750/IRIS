# Preliminary Grounded VideoQA Benchmark (Pillar 2)

Working tree was clean at run time (`git_dirty=false`).

**Provenance:** backend=`LlamaServerBackend` | model=`granite4:micro` | temperature=`0.0` | cache_prompt=`False` | endpoint=`http://127.0.0.1:8091/v1` | span_mode=`ppr_peak` | span_half_width=`2.2` | git_commit=`c4dd497f06f85a244e11654b1ef81c70f9e13429` | git_dirty=`False` | timestamp_utc=`2026-07-19T19:42:28.468531+00:00`

Evaluating the end-to-end IRIS pipeline (`charon_v` -> L1 admission Variant B -> L2 PPR retrieval -> LlamaServerBackend `granite4:micro` backend) on the NExT-GQA test subset ($n=64$ questions, 59 unique videos).

## 1. Quantitative Performance Summary

| Metric | Proposed System | Uniform Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Acc@GQA** (Grounded QA) | 23.44% | 4.69% | 3.12% |
| **Acc@QA** (QA Accuracy) | 53.12% | 21.88% | 29.69% |
| **mIoP** (Mean IoP) | 0.3426 | 0.2553 | 0.2563 |
| **IoP@0.5** (Grounding Recall) | 35.94% | 15.62% | 14.06% |
| **mIoU** (Mean IoU) | 0.1790 | 0.2460 | 0.2322 |
| **IoU@0.5** (Grounding Overlap) | 10.94% | 14.06% | 14.06% |

*Note: Acc@GQA represents the percentage of questions that are both answered accurately (Acc@QA) and grounded correctly (IoP $\ge$ 0.5).*

## 2. Statistical Rigor (Paired Bootstrap CIs)

Calculated over 1000 video-level cluster bootstrap resamples. 95% Confidence Intervals (CI) gate each paired difference.

### A. Proposed vs. Uniform Baseline (Proposed - Uniform)
*   **Acc@GQA Diff:** +0.1871 (95% CI: `[+0.0937, +0.2813]`)
*   **Acc@QA Diff:**  +0.3113 (95% CI: `[+0.1691, +0.4678]`)
*   **IoP@0.5 Diff:**  +0.2042 (95% CI: `[+0.0923, +0.3175]`)
*   **mIoP Diff:**     +0.0886 (95% CI: `[+0.0125, +0.1640]`)
*   **IoU@0.5 Diff:**  -0.0318 (95% CI: `[-0.1515, +0.0870]`)
*   **mIoU Diff:**     -0.0661 (95% CI: `[-0.1423, +0.0068]`)

### B. Proposed vs. Random Baseline (Proposed - Random)
*   **Acc@GQA Diff:** +0.2021 (95% CI: `[+0.1129, +0.2899]`)
*   **Acc@QA Diff:**  +0.2343 (95% CI: `[+0.0952, +0.3871]`)
*   **IoP@0.5 Diff:**  +0.2192 (95% CI: `[+0.1129, +0.3232]`)
*   **mIoP Diff:**     +0.0874 (95% CI: `[+0.0077, +0.1627]`)
*   **IoU@0.5 Diff:**  -0.0329 (95% CI: `[-0.1563, +0.0896]`)
*   **mIoU Diff:**     -0.0522 (95% CI: `[-0.1253, +0.0205]`)

## 3. Discussion and Scientific Context

### The "Trust Gap"
The scientific literature notes a massive **Trust Gap** on Grounded VideoQA benchmarks: SOTA closed-ended QA models achieve up to ~69% Acc@QA, but drop to a meager **16% Acc@GQA** when forced to localize the visual evidence that justifies their answer. Human baselines achieve 82% Acc@QA / Acc@GQA, proving that contemporary models rely on language priors rather than visual evidence.

### Competitive Status with 2B Agentic Frontier
With our proposed system achieving **23.44% Acc@GQA** on this evaluation:
*   We outperform simple baseline samplers (Uniform and Random) by a statistically significant margin.
*   Measured Acc@GQA: 23.44%.
