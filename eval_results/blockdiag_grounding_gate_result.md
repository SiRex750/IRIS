# Block-diagonal grounding acceptance gate

**Outcome: GATE_PASS**

## Part (a) -- frozen flat VAL cell (unaffected by this change, reconfirmed)

- graph_mode=flat, top_k=8, half_width=2.2, n=406 questions / 59 videos
- Rerun peak_in_gold rate: 0.3226600985221675
- Matches committed 0.3227 exactly (rounded to 4dp): True
- Note: flat's code path never sets node_groups, so graph_edge_mode="block_diagonal" is UNREACHABLE from this config (the new branch raises ValueError without node_groups). This cell cannot exercise block_diagonal at all -- it is reconfirmed here only to prove the fix left it untouched.

## Part (b) -- fully_connected vs block_diagonal, scene_sparse, real pipeline

- N questions: 526, videos: 86
- peak_in_gold rate -- fully_connected: 0.31749049429657794, block_diagonal: 0.31749049429657794
- mIoP -- fully_connected: 0.31402667038753285, block_diagonal: 0.31402667038753285
- Per-question bit-identical (retrieved order, peak, span, IoP, peak_in_gold): True (0 mismatches)
- No committed grounding number was ever produced with graph_mode=scene_sparse + graph_edge_mode=fully_connected (existing scene_sparse committed arms use the default hierarchical_sparse edge formula, which block_diagonal does not target). This is therefore an equivalence proof between the two build paths this fix actually concerns, not a rerun of a pre-existing committed number.

## Cache integrity

- `eval/data/nextqa/index_cache` sha256 unchanged: True
- `eval/data/nextqa/index_cache_ssparse` sha256 unchanged: True

