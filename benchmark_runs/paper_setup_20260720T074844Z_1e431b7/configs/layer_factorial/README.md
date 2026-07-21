# 2^3 Layer Factorial (F000-F111)

Canonical replacements as specified by the task, mapped against what actually exists in code
(see `canonical_pipeline.md`, `method_registry.json`):

- **L1 ON** = production admission (`retrieval_strategy="hybrid"`, default `IRISConfig`). IMPLEMENTED.
- **L1 OFF** = uniform budget-matched admission at the same K. **NOT_IMPLEMENTED** — no `retrieval_strategy` value does uniform sampling today (see `method_registry.json::L1_uniform_budget_matched`).
- **L2 ON** = production `graph_mode="scene_sparse"` + `ranking_mode="ppr"` + configured peak rule (`ppr_score_legacy`, itself just top-1 PPR output). IMPLEMENTED.
- **L2 OFF** = direct CLIP cosine ranking over the exact same survivor pool: `ranking_mode="legacy"`, `alpha=1.0, beta=gamma=delta=0.0`. IMPLEMENTED.
- **L3 ON** = legacy answerer + CerberusV verification + abstention (production default). IMPLEMENTED.
- **L3 OFF** = same captions/answerer, no verifier/abstention gate. **NOT_IMPLEMENTED** — no config flag bypasses `wrapper_cerberus_gate` while keeping the answerer call identical (see `method_registry.json::L3_no_verifier`).

## Cell-by-cell executability today

| Cell | L1 | L2 | L3 | Executable today? |
|---|---|---|---|---|
| F111 | ON | ON | ON | Yes — this is `PROPOSED_FULL` |
| F110 | ON | ON | OFF | No — blocked on L3 OFF |
| F101 | ON | OFF | ON | Yes |
| F100 | ON | OFF | OFF | No — blocked on L3 OFF |
| F011 | OFF | ON | ON | No — blocked on L1 OFF |
| F010 | OFF | ON | OFF | No — blocked on L1 OFF and L3 OFF |
| F001 | OFF | OFF | ON | No — blocked on L1 OFF |
| F000 | OFF | OFF | OFF | No — blocked on L1 OFF and L3 OFF |

Each cell still gets its own immutable `configs/layer_factorial/F###.json` file below (per task
instruction "for every cell create a separate immutable configuration file"), with an explicit
`"executable": true/false` and `"blocked_by"` field so a future runner refuses to silently run a
blocked cell. Static checks below (same K / model revisions / query text / answer normalization /
evaluator / timeout / failure accounting) are checked across the cells that share configuration,
not executed.
