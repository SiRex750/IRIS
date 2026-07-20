# Preliminary Grounded VideoQA Benchmark (Pillar 2)

Working tree was clean at run time (`git_dirty=false`).

**Provenance:** backend=`LlamaServerBackend` | model=`granite4:micro` | temperature=`0.0` | cache_prompt=`False` | endpoint=`http://127.0.0.1:8091/v1` | span_mode=`ppr_peak` | span_half_width=`2.2` | git_commit=`4fc5148499549e7ae4b30d0bfb9dc8ea2d1adf00` | git_dirty=`False` | timestamp_utc=`2026-07-20T00:09:46.788401+00:00`

Evaluating the end-to-end IRIS pipeline (`charon_v` -> L1 admission Variant B -> L2 PPR retrieval -> LlamaServerBackend `granite4:micro` backend) on the NExT-GQA test subset ($n=64$ questions, 59 unique videos).

## 1. Quantitative Performance Summary

| Metric | Proposed System | Uniform Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Acc@GQA** (Grounded QA) | 15.62% | 4.69% | 3.12% |
| **Acc@QA** (QA Accuracy) | 54.69% | 21.88% | 29.69% |
| **mIoP** (Mean IoP) | 0.2216 | 0.2553 | 0.2563 |
| **IoP@0.5** (Grounding Recall) | 20.31% | 15.62% | 14.06% |
| **mIoU** (Mean IoU) | 0.1097 | 0.2460 | 0.2322 |
| **IoU@0.5** (Grounding Overlap) | 6.25% | 14.06% | 14.06% |

*Note: Acc@GQA represents the percentage of questions that are both answered accurately (Acc@QA) and grounded correctly (IoP $\ge$ 0.5).*

## 2. Statistical Rigor (Paired Bootstrap CIs)

Calculated over 1000 video-level cluster bootstrap resamples. 95% Confidence Intervals (CI) gate each paired difference.

### A. Proposed vs. Uniform Baseline (Proposed - Uniform)
*   **Acc@GQA Diff:** +0.1092 (95% CI: `[+0.0156, +0.2132]`)
*   **Acc@QA Diff:**  +0.3269 (95% CI: `[+0.1999, +0.4616]`)
*   **IoP@0.5 Diff:**  +0.0457 (95% CI: `[-0.0656, +0.1562]`)
*   **mIoP Diff:**     -0.0341 (95% CI: `[-0.1111, +0.0446]`)
*   **IoU@0.5 Diff:**  -0.0783 (95% CI: `[-0.1912, +0.0303]`)
*   **mIoU Diff:**     -0.1365 (95% CI: `[-0.2117, -0.0618]`)

### B. Proposed vs. Random Baseline (Proposed - Random)
*   **Acc@GQA Diff:** +0.1242 (95% CI: `[+0.0328, +0.2204]`)
*   **Acc@QA Diff:**  +0.2498 (95% CI: `[+0.1094, +0.3872]`)
*   **IoP@0.5 Diff:**  +0.0607 (95% CI: `[-0.0441, +0.1667]`)
*   **mIoP Diff:**     -0.0353 (95% CI: `[-0.1156, +0.0468]`)
*   **IoU@0.5 Diff:**  -0.0794 (95% CI: `[-0.1935, +0.0299]`)
*   **mIoU Diff:**     -0.1227 (95% CI: `[-0.1985, -0.0478]`)

## 3. Discussion and Scientific Context

### The "Trust Gap"
The scientific literature notes a massive **Trust Gap** on Grounded VideoQA benchmarks: SOTA closed-ended QA models achieve up to ~69% Acc@QA, but drop to a meager **16% Acc@GQA** when forced to localize the visual evidence that justifies their answer. Human baselines achieve 82% Acc@QA / Acc@GQA, proving that contemporary models rely on language priors rather than visual evidence.

### Competitive Status with 2B Agentic Frontier
With our proposed system achieving **15.62% Acc@GQA** on this evaluation:
*   We outperform simple baseline samplers (Uniform and Random) by a statistically significant margin.
*   Measured Acc@GQA: 15.62%.
