# Lambda ("ppr_lambda") Semantics — Proven, Not Assumed

Source: `iris/l2_asphodel.py::L2Asphodel.retrieve_ppr`, ~line 1273. Config field: `ppr_lambda`
(`IRISConfig`, default `0.5`). Verified two independent ways: (1) direct code read, (2)
`scripts/lambda_endpoint_tests.py` — 10/10 PASS against real production code on a synthetic
3-node graph (no video, no model, no dataset). Report: `lambda_endpoint_tests.txt`.

## The formula, as written in code

```python
sem_rank   = _rank_pct(raw_semantic_cosine_similarity)   # rank-percentile in [0,1]
codec_rank = _rank_pct(raw_codec_conf)                   # rank-percentile in [0,1]
seed_raw[nid] = max(0.0, lambda_ * sem_rank[nid] + (1.0 - lambda_) * codec_rank[nid])
seed = seed_raw / sum(seed_raw.values())                 # sum-normalized before PPR personalization
```

This is **`lambda * semantic + (1 - lambda) * codec`**, not the reverse. Both signals are
rank-percentile transformed before blending — the blend operates on ranks, not raw cosine
similarity / raw `codec_conf` values.

## Endpoint identities (confirmed by test, not by name)

- **`ppr_lambda = 1.0` -> semantic-only.** `seed_raw` reduces to `sem_rank`; node ranking order
  under lambda=1.0 exactly matches the `sem_rank` ranking order. `codec_rank` has zero influence.
- **`ppr_lambda = 0.0` -> codec-only.** `seed_raw` reduces to `codec_rank`; node ranking order
  under lambda=0.0 exactly matches the `codec_rank` ranking order. `sem_rank` has zero influence.
- **`ppr_lambda = 0.5`** (the `IRISConfig` default) is an equal-weight rank-space blend: each
  node's seed contribution is `0.5*sem_rank + 0.5*codec_rank`, before sum-normalization.

## No endpoint silently enables another signal

Test confirms `sem_rank` and `codec_rank` per-node values are byte-identical across the
lambda=0.0/0.5/1.0 runs — only the blend weight changes, never the underlying signal
computation. Node order and PPR scores are deterministic (confirmed: two independent builds of
the same synthetic graph at lambda=0.5 produce identical retrieved-node order).

## Internal explicit names (avoids relying on the ambiguous "lambda" name alone)

| Concept | Config field | Value at semantic-only endpoint | Value at codec-only endpoint |
|---|---|---|---|
| `semantic_weight` | `ppr_lambda` | `1.0` | `0.0` |
| `codec_weight` | `1.0 - ppr_lambda` | `0.0` | `1.0` |

No config field is renamed or inverted by this setup task — `ppr_lambda` continues to mean
exactly what the code above computes. `semantic_weight`/`codec_weight` are documentation aliases
only, introduced here for unambiguous reference in method-registry / sweep-config docs.

## Backward translation

Any existing saved config or report that states "lambda=X" should be read as
`semantic_weight=X`, `codec_weight=1-X`, matching the formula above. No historical config value
needs to be renumbered — the formula direction is confirmed to match what `ppr_lambda` has always
meant in this codebase; there is no silent inversion to correct.

## Scope note

No validation videos were queried to produce this document — all evidence comes from (a) reading
`iris/l2_asphodel.py` directly and (b) running `retrieve_ppr` against a hand-built 3-node
synthetic `networkx` graph with fabricated embeddings/`codec_conf` values.
