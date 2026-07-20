# Item 6: Why does `use_l1` default to False?

## Investigation method
- `git log --all --oneline -S "use_l1" -- iris/iris_config.py` → **no matches**. The `use_l1` field
  does not exist in any committed history at all — it is part of this session's pre-existing
  uncommitted working-tree diff (`git diff HEAD -- iris/iris_config.py` shows it as an added line).
  There is no commit to `git blame`, no PR description, no commit message to consult.
- No comment in `iris_config.py` near the field explains the default (just a capacity-comment for
  the following `l1_capacity` field, nothing about `use_l1` itself).
- `grep -rn "use_l1"` across `iris/`: only 3 files reference it —
  `iris_config.py` (declaration), `ingest.py` (gates L1 cache construction at ingest time),
  `query.py` (`_retrieve_with_l1` gates whether L1 is consulted as an actual retrieval mechanism
  vs. used only as ARIA's context-text formatter).
- **Zero test coverage**: no test file references `use_l1` at all — there is no regression test
  exercising `use_l1=True`, which is itself a signal that the True path has not been validated.
- `eval/ablation_plan.md` ("L1 Elysium Ablation Plan") formally proposes an ablation study
  (LRU / uniform-weight / full-weight variants) but does not explain the production default, and
  does not appear to have been executed (no results file found alongside it).
- `IRIS_novel_contributions.md` describes the L1 Elysium composite `keep_score` (7-signal weighted
  eviction) as a headline "novel contribution" of the system, described unconditionally as
  something IRIS "does" — with no caveat that it is disabled by default in the shipped config.

## Conclusion
**No blocking reason (bug, cost, or correctness issue) is documented anywhere in the repository
for `use_l1=False`.** This looks like an unfinished wiring gap, not a deliberate tradeoff: the
formal ablation plan and the "novel contribution" writeup both assume L1 is an active, evaluated
part of the pipeline, but the default config never turns it on and no test exercises the True path.

Per the task instruction, since no blocking reason was found, the next step is an actual benchmark
run comparing `use_l1=True` vs `use_l1=False` on the same 3 videos / 12 questions, measuring
retrieval-quality and latency deltas, **before** touching the default. That run is scripted at
`smoke_scripts/compare_use_l1.py` (uses the cached indexes, no re-ingest) — pending in the queue
behind the live-inference diagnostics for item 1, to avoid resource contention on the shared local
Ollama server during this investigation.
