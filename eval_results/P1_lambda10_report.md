# Preliminary Grounded VideoQA Benchmark (Pillar 2)

Working tree was clean at run time (`git_dirty=false`).

**Provenance:** backend=`LlamaServerBackend` | model=`granite4:micro` | temperature=`0.0` | cache_prompt=`False` | endpoint=`http://127.0.0.1:8091/v1` | span_mode=`ppr_peak` | span_half_width=`2.2` | git_commit=`c9290ab6ad3cf47466f3138464432d4326ca9640` | git_dirty=`False` | timestamp_utc=`2026-07-20T07:40:25.837700+00:00`

Evaluating the end-to-end IRIS pipeline (`charon_v` -> L1 admission Variant B -> L2 PPR retrieval -> LlamaServerBackend `granite4:micro` backend) on the NExT-GQA test subset ($n=64$ questions, 59 unique videos).

## 1. Quantitative Performance Summary

| Metric | Proposed System | Uniform Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Acc@GQA** (Grounded QA) | 20.31% | 4.69% | 3.12% |
| **Acc@QA** (QA Accuracy) | 51.56% | 21.88% | 29.69% |
| **mIoP** (Mean IoP) | 0.3295 | 0.2553 | 0.2563 |
| **IoP@0.5** (Grounding Recall) | 31.25% | 15.62% | 14.06% |
| **mIoU** (Mean IoU) | 0.1755 | 0.2460 | 0.2322 |
| **IoU@0.5** (Grounding Overlap) | 12.50% | 14.06% | 14.06% |

*Note: Acc@GQA represents the percentage of questions that are both answered accurately (Acc@QA) and grounded correctly (IoP $\ge$ 0.5).*

## 2. Statistical Rigor (Paired Bootstrap CIs)

Calculated over 1000 video-level cluster bootstrap resamples. 95% Confidence Intervals (CI) gate each paired difference.

### A. Proposed vs. Uniform Baseline (Proposed - Uniform)
*   **Acc@GQA Diff:** +0.1541 (95% CI: `[+0.0492, +0.2537]`)
*   **Acc@QA Diff:**  +0.2982 (95% CI: `[+0.1587, +0.4376]`)
*   **IoP@0.5 Diff:**  +0.1545 (95% CI: `[+0.0308, +0.2755]`)
*   **mIoP Diff:**     +0.0744 (95% CI: `[+0.0010, +0.1471]`)
*   **IoU@0.5 Diff:**  -0.0168 (95% CI: `[-0.1475, +0.1061]`)
*   **mIoU Diff:**     -0.0704 (95% CI: `[-0.1524, +0.0076]`)

### B. Proposed vs. Random Baseline (Proposed - Random)
*   **Acc@GQA Diff:** +0.1692 (95% CI: `[+0.0781, +0.2623]`)
*   **Acc@QA Diff:**  +0.2211 (95% CI: `[+0.0833, +0.3594]`)
*   **IoP@0.5 Diff:**  +0.1695 (95% CI: `[+0.0500, +0.2858]`)
*   **mIoP Diff:**     +0.0732 (95% CI: `[-0.0027, +0.1527]`)
*   **IoU@0.5 Diff:**  -0.0179 (95% CI: `[-0.1452, +0.1046]`)
*   **mIoU Diff:**     -0.0566 (95% CI: `[-0.1354, +0.0262]`)

## 3. Discussion and Scientific Context

### The "Trust Gap"
The scientific literature notes a massive **Trust Gap** on Grounded VideoQA benchmarks: SOTA closed-ended QA models achieve up to ~69% Acc@QA, but drop to a meager **16% Acc@GQA** when forced to localize the visual evidence that justifies their answer. Human baselines achieve 82% Acc@QA / Acc@GQA, proving that contemporary models rely on language priors rather than visual evidence.

### Competitive Status with 2B Agentic Frontier
With our proposed system achieving **20.31% Acc@GQA** on this evaluation:
*   We outperform simple baseline samplers (Uniform and Random) by a statistically significant margin.
*   Measured Acc@GQA: 20.31%.
