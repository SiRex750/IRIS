# Publication-Ready Benchmark Summary Report (T0 Reproduction)

## Experiment Overview
CPU-only reproduction of the existing 89-video / 255-event retrieval experiment. All execution settings and threads were restricted to CPU-only execution.

## Primary Endpoint: Recall@5

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | mAP | Precision@5 | Window Coverage |
|---|---|---|---|---|---|---|---|
| proposed_system | 0.2275 | 0.5686 | 0.7020 | 0.3888 | 0.2388 | 0.2212 | 1.0000 |
| hybrid_ppr | 0.2784 | 0.5608 | 0.7137 | 0.4085 | 0.2933 | 0.2698 | 1.0000 |
| semantic_only_ppr | 0.2627 | 0.5373 | 0.7098 | 0.3937 | 0.2946 | 0.2604 | 1.0000 |
| proposed_model_selection | 0.2235 | 0.5294 | 0.6902 | 0.3743 | 0.2375 | 0.2110 | 1.0000 |
| codec_only_ppr | 0.2275 | 0.5137 | 0.7059 | 0.3693 | 0.2284 | 0.1906 | 1.0000 |
| clip_query_topk | 0.3176 | 0.5098 | 0.6235 | 0.4148 | 0.3054 | 0.2878 | 0.9804 |
| packet_size_only | 0.1882 | 0.4941 | 0.7059 | 0.3379 | 0.2445 | 0.1922 | 1.0000 |
| luma_difference | 0.2078 | 0.4745 | 0.6392 | 0.3438 | 0.2514 | 0.2008 | 1.0000 |
| optical_flow_farneback | 0.1882 | 0.1882 | 0.1961 | 0.2017 | 0.2571 | 0.1859 | 0.6824 |
| scene_change | 0.0510 | 0.1804 | 0.1922 | 0.1146 | 0.2959 | 0.1184 | 1.0000 |
| uniform | 0.0510 | 0.1804 | 0.1961 | 0.1226 | 0.3025 | 0.1310 | 1.0000 |
| iframe_prior | 0.0510 | 0.1765 | 0.1922 | 0.1095 | 0.2940 | 0.1122 | 1.0000 |
| random | 0.1037 | 0.1762 | 0.2088 | 0.1533 | 0.3124 | 0.1452 | 0.9997 |
| clip_kmeans_diversity | 0.0706 | 0.1725 | 0.2078 | 0.1364 | 0.3149 | 0.1349 | 1.0000 |

## Paired Method Comparisons (Graph minus competitors)

| Comparison | Point Difference | 95% Paired CI | Prob > 0 | Shared Videos | Shared Events |
|---|---|---|---|---|---|
| proposed_system_minus_random | 0.3924 | [0.3171, 0.4679] | 1.0000 | 89 | 255 |
| proposed_system_minus_uniform | 0.3882 | [0.3123, 0.4643] | 1.0000 | 89 | 255 |
| proposed_system_minus_clip_query_topk | 0.0588 | [-0.0200, 0.1412] | 0.9181 | 89 | 255 |
| proposed_system_minus_optical_flow_farneback | 0.3804 | [0.3031, 0.4556] | 1.0000 | 89 | 255 |
| proposed_system_minus_proposed_model_selection | 0.0392 | [0.0040, 0.0751] | 0.9853 | 89 | 255 |
| proposed_system_minus_semantic_only_ppr | 0.0314 | [-0.0388, 0.1032] | 0.7947 | 89 | 255 |
| proposed_system_minus_hybrid_ppr | 0.0078 | [-0.0664, 0.0827] | 0.5608 | 89 | 255 |

## Publishability Assessment
Classification: **PAPER-SUPPORTING PRELIMINARY RESULT**

### Advantages Claimed:
- **proposed_system_minus_random**: Recall@5 point difference of 0.3924 (95% CI: [0.3171, 0.4679]). This difference is **statistically significant**.
- **proposed_system_minus_uniform**: Recall@5 point difference of 0.3882 (95% CI: [0.3123, 0.4643]). This difference is **statistically significant**.
- **proposed_system_minus_clip_query_topk**: Recall@5 point difference of 0.0588 (95% CI: [-0.0200, 0.1412]). This difference is **not statistically significant**.
- **proposed_system_minus_optical_flow_farneback**: Recall@5 point difference of 0.3804 (95% CI: [0.3031, 0.4556]). This difference is **statistically significant**.
- **proposed_system_minus_proposed_model_selection**: Recall@5 point difference of 0.0392 (95% CI: [0.0040, 0.0751]). This difference is **statistically significant**.
- **proposed_system_minus_semantic_only_ppr**: Recall@5 point difference of 0.0314 (95% CI: [-0.0388, 0.1032]). This difference is **not statistically significant**.
- **proposed_system_minus_hybrid_ppr**: Recall@5 point difference of 0.0078 (95% CI: [-0.0664, 0.0827]). This difference is **not statistically significant**.
