# Scientific Contract / Preregistration
# Remote-GPU Video Benchmark on NExT-GQA

## Metadata
*   **Evaluation Date**: 2026-07-10
*   **Dataset**: NExT-GQA
*   **Target Split**: Validation Subset (89 unique videos, 257 QA-grounded event spans)
*   **Primary Endpoint**: WindowCoverage@K at 10% retention (retention ratio = 0.10)
*   **Methods Compared**:
    1.  `uniform`
    2.  `random` (30 seeds, `0..29`)
    3.  `iframe_prior`
    4.  `scene_change`
    5.  `luma_difference`
    6.  `optical_flow_farneback`
    7.  `clip_kmeans_diversity` (MiniBatchKMeans, Hungarian unique assignment)
    8.  `clip_query_topk`
    9.  `packet_size_only`
    10. `proposed_model_selection` (Selection-only Proposed model)
    11. `proposed_system` (Full Proposed System with graph retrieval)

## Primary Hypothesis
The **proposed model** (selection-only) and the **proposed system** (with graph retrieval) will outperform matched-budget frame-selection baselines on the temporal evidence window coverage metric.

## Claim Gate Criteria
The claim **"the proposed model/system provides superior matched-budget frame selection"** is valid if and only if:
1.  All methods run successfully on the same set of videos.
2.  The proposed selector beats the strongest baseline on `WindowCoverage@K` at 10% retention.
3.  The 95% confidence interval for that improvement (generated with 10,000 paired video-clustered bootstrap replicates, seed `20260710`) excludes zero.
4.  The absolute improvement is at least 3.0 percentage points.
5.  All timings, memory utilization, and failure modes are disclosed.

## Frozen Hyperparameters
*   **CLIP Backbone**: `ViT-B/32`
*   **Farneback Flow**: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0 (resized long side to 320px)
*   **MiniBatchKMeans**: n_init=10, max_iter=100, reassignment_ratio=0
*   **PPR Damping**: 0.85
