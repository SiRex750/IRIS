# Retrieval Evaluator Integrity Gate Report

| Gate | Status | Evidence |
|---|---|---|
| 1. Live Git provenance captured | **Passed** | Captured Commit 7b38c30f on Branch main. |
| 2. No unresolved merge conflicts | **Passed** | Stash and resolve cleanly verified. |
| 3. Production ingest used | **Passed** | Executed ingest() successfully in 19.03s. |
| 4. Production NMS used | **Passed** | Dynamic audit assertion passed. |
| 5. Adaptive budget honestly reported | **Passed** | Admitted budget K=48 (11.940% retention) mapped per-video. |
| 6. Proposed evaluator matches production order | **Passed** | Asserted order equivalence across all queries. |
| 7. Complete nonzero embeddings | **Passed** | Dynamic audit assertion passed. |
| 8. Variable codec confidence where expected | **Passed** | Audited 45 unique codec_conf values. |
| 9. Graph tiers verified | **Passed** | Audited tier distribution: {'L1_PEAK': 26, 'L2_SALIENT': 22}. |
| 10. Edge families verified | **Passed** | Audited edge families: ['motion_neighbor', 'hierarchy_peak_salient', 'semantic_salient', 'temporal']. |
| 11. No PPR fallback | **Passed** | Dynamic audit assertion passed. |
| 12. No result padding | **Passed** | Dynamic audit assertion passed. |
| 13. Same admitted IDs across ranking arms | **Passed** | Dynamic audit assertion passed. |
| 14. Graph unchanged across queries | **Passed** | Dynamic audit assertion passed. |
| 15. Independent latency measurement | **Passed** | Independent warm-up and 5 timed repetitions evaluated. |
| 16. Compilation passed | **Passed** | Dynamic audit assertion passed. |
| 17. Targeted tests passed | **Passed** | pytest suite returned successfully with exit code 0. |
| 18. Smoke produced all required artifacts | **Passed** | Verified presence of all 11 files. |
