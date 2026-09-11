# Build cost — final measurement, v2 (re-run after defer_recompute commit)

Re-execution of the A.5.8 interleaved build-cost protocol, identical in every
respect (protocol, cache, parameters, `scripts/build_cost_final_probe.py`
unmodified) to the run behind `eval_results/build_cost_final.md`. The only
intended difference: the tree is now committed.

## Provenance

- **Commit**: `3a2c1db9794f48a7f5d520b40eb87aa4bf0c6ce7` —
  "perf(ingest): defer_recompute=True in _build_graph, behaviour-preserving"
  (branch `siddanth/peak-source-a6-p1`)
- **Dirty count at commit time**: 133 (`git status --porcelain`: 1 modified +
  132 untracked paths — `paper/IRIS_paper_draft.md` plus scratch under
  `eval_results/`, `scripts/`, `NExT-GQA/`, `eval/metrics_official.py`).
  `iris/` itself was clean (the commit captured exactly the intended diff,
  nothing else).
- Clip: VIRAT_S_040001_01_000448_001101, N=4,892 survivor frames / 528 scenes
  (unchanged).
- Config: `graph_mode=scene_sparse`; arm-specific `graph_edge_mode`
  (unchanged).
- Guard: `edge_count == 23,571` asserted on every build — **passed on all
  159** (9 old + 150 new).

## Round 1

Machine state before old-loop: cpu_percent=4.0%, available RAM=18.58 GB / 33.46 GB total, 16 logical / 8 physical CPUs.
Machine state before new-loop: cpu_percent=9.2%, available RAM=18.97 GB.

| arm | median | min | max | total_loop_sec |
|---|---|---|---|---|
| old (n=3) | 49.875771 | 49.538214 | 50.129131 | 149.545623 |
| new (n=50) | 0.212052 | 0.121100 | 0.226067 | 9.245701 |

- **ratio (median old / median new) = 235.206**

## Round 2

Machine state before old-loop: cpu_percent=7.7%, available RAM=18.57 GB.
Machine state before new-loop: cpu_percent=8.2%, available RAM=19.30 GB.

| arm | median | min | max | total_loop_sec |
|---|---|---|---|---|
| old (n=3) | 49.612377 | 48.582846 | 49.920515 | 148.117258 |
| new (n=50) | 0.212026 | 0.119973 | 0.228370 | 9.266114 |

- **ratio (median old / median new) = 233.992**

## Round 3

Machine state before old-loop: cpu_percent=9.1%, available RAM=19.28 GB.
Machine state before new-loop: cpu_percent=11.9%, available RAM=18.90 GB.

| arm | median | min | max | total_loop_sec |
|---|---|---|---|---|
| old (n=3) | 49.712318 | 48.755108 | 50.146276 | 148.616234 |
| new (n=50) | 0.212039 | 0.121950 | 0.230859 | 9.263618 |

- **ratio (median old / median new) = 234.449**

## Cross-round stability

| arm | round-1 median | round-2 median | round-3 median | spread as % of median |
|---|---|---|---|---|
| old | 49.875771 | 49.612377 | 49.712318 | **0.530%** |
| new | 0.212052 | 0.212026 | 0.212039 | **0.012%** |

Original run: old spread 0.177%, new spread 0.242%. This re-run: old spread
0.530%, new spread 0.012%. Both are still small relative to the ~30–40%
cross-session swings documented in A.5.8's motivation, and neither arm shows
a directional drift across rounds — but the dense (old) arm's spread nearly
tripled (0.177% → 0.530%) and the fast (new) arm's spread shrank by ~20×
(0.242% → 0.012%) relative to the original run. Available RAM was slightly
lower throughout this run (18.57–19.30 GB free) than the original
(19.78–19.96 GB free); nothing else in the machine snapshots points to a
specific cause.

## Ratio: pooled medians

- old pooled median (n=9 builds across 3 rounds) = **49.712318 sec**
- new pooled median (n=150 builds across 3 rounds) = **0.212039 sec**
- **ratio from pooled medians = 234.449×**

## Ratio summary

- per-round ratios: [235.206, 233.992, 234.449]
- min per-round ratio: 233.992
- max per-round ratio: 235.206
- ratio from pooled medians: 234.449

**Agreement among the four ratio estimates (3 per-round + 1 pooled): 0.518%
of their mean** (original run: 0.4%). All four still agree tightly, just
outside the exact 0.4% figure A.5.8 reports for the original run.

## Comparison to the original run (`build_cost_final.md`)

| quantity | original | this re-run (v2) |
|---|---|---|
| old pooled median (sec) | 50.195124 | 49.712318 |
| new pooled median (sec) | 0.214939 | 0.212039 |
| pooled ratio | 233.532× | 234.449× |
| old cross-round spread | 0.177% | 0.530% |
| new cross-round spread | 0.242% | 0.012% |
| 4-estimate agreement | 0.4% | 0.518% |
| edge_count guard | 159/159 passed | 159/159 passed |

Both the wall-clock figures and the ratio moved by roughly 1% or less
between the two runs (old pooled median −0.96%, new pooled median −1.35%,
pooled ratio +0.39%) — consistent with normal machine-state variation
between sessions, not with a behavioral change (the code under test between
the two runs is identical; only the working-tree commit state differs, and
`defer_recompute` was already `True` in both). No decision is made here
about which figure set (original vs. v2) is primary — both are reported as
measured. Raw per-round, per-arm outputs are in
`eval_results/build_cost_final_v2_raw/`.
