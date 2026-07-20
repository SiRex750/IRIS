# Publication-Ready Benchmark Summary Report (T0 Reproduction)

## Experiment Overview
CPU-only reproduction of the existing 89-video / 255-event retrieval experiment. All execution settings and threads were restricted to CPU-only execution.

## Proposed Method Specification
The proposed method evaluated in this benchmark is the **“Sparse hierarchical spatiotemporal graph with codec-semantic Personalized PageRank retrieval.”** (represented as `proposed_sparse_hybrid_ppr`).

## Primary Endpoint: Recall@5

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Precision@5 | Window Coverage |
|---|---|---|---|---|---|---|
| semantic_ppr_sparse | 0.2627 | 0.5882 | 0.7373 | 0.4117 | 0.2675 | 1.0000 |
| legacy_sparse_direct | 0.2196 | 0.5725 | 0.6863 | 0.3818 | 0.2227 | 1.0000 |
| dense_hybrid_ppr | 0.2588 | 0.5569 | 0.7059 | 0.4024 | 0.2384 | 1.0000 |
| proposed_sparse_hybrid_ppr | 0.2235 | 0.5451 | 0.6824 | 0.3739 | 0.2314 | 1.0000 |
| proposed_model_selection | 0.2235 | 0.5294 | 0.6902 | 0.3743 | 0.2110 | 1.0000 |
| clip_query_topk | 0.3176 | 0.5098 | 0.6235 | 0.4148 | 0.2878 | 0.9804 |
| codec_ppr_sparse | 0.2353 | 0.4941 | 0.6314 | 0.3554 | 0.2110 | 1.0000 |
| packet_size_only | 0.1882 | 0.4941 | 0.7059 | 0.3379 | 0.1922 | 1.0000 |
| luma_difference | 0.2078 | 0.4745 | 0.6392 | 0.3438 | 0.2008 | 1.0000 |
| optical_flow_farneback | 0.1882 | 0.1882 | 0.1961 | 0.2017 | 0.1859 | 0.6824 |
| scene_change | 0.0510 | 0.1804 | 0.1922 | 0.1146 | 0.1184 | 1.0000 |
| uniform | 0.0510 | 0.1804 | 0.1961 | 0.1226 | 0.1310 | 1.0000 |
| iframe_prior | 0.0510 | 0.1765 | 0.1922 | 0.1095 | 0.1122 | 1.0000 |
| random | 0.1037 | 0.1762 | 0.2088 | 0.1533 | 0.1452 | 0.9997 |
| clip_kmeans_diversity | 0.0706 | 0.1725 | 0.2078 | 0.1364 | 0.1349 | 1.0000 |

## Paired Method Comparisons (Proposed Graph minus competitors)

| Comparison | Point Difference | 95% Paired CI | Prob > 0 | Shared Videos | Shared Events |
|---|---|---|---|---|---|
| proposed_sparse_hybrid_ppr_minus_random | 0.3689 | [0.2772, 0.4553] | 1.0000 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_uniform | 0.3647 | [0.2734, 0.4512] | 1.0000 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_clip_query_topk | 0.0353 | [-0.0512, 0.1206] | 0.7709 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_optical_flow_farneback | 0.3569 | [0.2738, 0.4382] | 1.0000 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_proposed_model_selection | 0.0157 | [-0.0635, 0.0951] | 0.6275 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_legacy_sparse_direct | -0.0275 | [-0.1020, 0.0474] | 0.2194 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_semantic_ppr_sparse | -0.0431 | [-0.1177, 0.0308] | 0.1131 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_codec_ppr_sparse | 0.0510 | [-0.0080, 0.1085] | 0.9444 | 89 | 255 |
| proposed_sparse_hybrid_ppr_minus_dense_hybrid_ppr | -0.0118 | [-0.0720, 0.0465] | 0.3281 | 89 | 255 |

## Publishability Assessment
Classification: **PAPER-SUPPORTING PRELIMINARY RESULT**

### Advantages Claimed:
- **proposed_sparse_hybrid_ppr_minus_random**: Recall@5 point difference of 0.3689 (95% CI: [0.2772, 0.4553]). This difference is **statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_uniform**: Recall@5 point difference of 0.3647 (95% CI: [0.2734, 0.4512]). This difference is **statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_clip_query_topk**: Recall@5 point difference of 0.0353 (95% CI: [-0.0512, 0.1206]). This difference is **not statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_optical_flow_farneback**: Recall@5 point difference of 0.3569 (95% CI: [0.2738, 0.4382]). This difference is **statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_proposed_model_selection**: Recall@5 point difference of 0.0157 (95% CI: [-0.0635, 0.0951]). This difference is **not statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_legacy_sparse_direct**: Recall@5 point difference of -0.0275 (95% CI: [-0.1020, 0.0474]). This difference is **not statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_semantic_ppr_sparse**: Recall@5 point difference of -0.0431 (95% CI: [-0.1177, 0.0308]). This difference is **not statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_codec_ppr_sparse**: Recall@5 point difference of 0.0510 (95% CI: [-0.0080, 0.1085]). This difference is **not statistically significant**.
- **proposed_sparse_hybrid_ppr_minus_dense_hybrid_ppr**: Recall@5 point difference of -0.0118 (95% CI: [-0.0720, 0.0465]). This difference is **not statistically significant**.

### Superiority Assessment:
There is **no statistically significant difference** in performance between the proposed method (**proposed_sparse_hybrid_ppr**) and the legacy graph retrieval (**legacy_sparse_direct**), as the 95% confidence interval [-0.1020, 0.0474] contains zero. Therefore, we do **not** claim superiority over the legacy method.
