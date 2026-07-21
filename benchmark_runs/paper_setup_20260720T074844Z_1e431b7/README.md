# IRIS Paper-Setup Snapshot — 2026-07-20T07:48:44Z @ 1e431b7

SETUP ONLY. No video was decoded, no model was queried, no real accuracy/grounding/latency
result was produced. See `setup_report.md` for the full account and `PRE_BENCHMARK_SETUP_READY.json`
(if present) / `setup_failures.jsonl` (blockers) for gate status.

Start here:
1. `setup_report.md` — what's ready, what isn't, exact next command.
2. `canonical_pipeline.md` — the real production call graph, including a major correction to the
   assumed architecture (no CLIP-peak-reranking/predicted-span stage exists in code today).
3. `dataset_manifest.json` + `setup_failures.jsonl` — official NExT-GQA dataset status (blocked).
4. `lambda_semantics.md` / `ppr_formula.md` — proven (not assumed) PPR personalization formula.
5. `fairness_contract.md`, `method_registry.json`, `configs/layer_factorial/` — comparison rules.

This directory is intended to be immutable after setup completion; do not edit files here except
through a new dated `paper_setup_*` snapshot.
