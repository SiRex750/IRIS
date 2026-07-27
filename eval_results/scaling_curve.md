# Query-latency scaling curve: flat vs scene_sparse across N

Extends `scripts/virat_latency_ab.py` (single-N VIRAT result,
`eval_results/virat_latency_N4892_result.md`) to a CURVE across N using UCF-Crime
clips + the reused VIRAT point. Harness: same seeded-CLIP-embedding-query
methodology, same `N_WARMUP=3` / `N_REPS=5` / `TOP_K=8` constants from
`scripts/latency_ab.py`, same query seed `20260726`, 50 queries per clip. Flat and
scene_sparse run as **separate processes per clip** so a flat failure cannot take
down the scene_sparse measurement. Git commit at run start: `ccf764f`.

**Not committed yet** — this is the raw report for review before `git add`/commit.

## Critical methodology caveat: the flat watchdog is NOT applied to the reused VIRAT point

The 7 new UCF measurements wrap the flat arm (graph build + the full 50-query timed
loop) in a **480s wall-time / 29GB memory watchdog**, matching the committed
`scripts/virat_smoke_flat.py` pattern. The **reused VIRAT N=4,892 point had no such
cap** — `scripts/virat_latency_ab.py` ran flat to completion unbounded, and on the
observed per-query cost that run's flat arm took on the order of **30–50 minutes**
wall time (400 timed executions × ~7.9s). Read the table below with this in mind:
VIRAT (N=4,892) shows a flat SUCCESS not because N=4,892 is "easier" than UCF's
N=2,963 (which FAILED), but because the VIRAT run was given unlimited time and the
UCF runs were not. The flat-infeasibility "frontier" reported here is a **function of
the 480s budget we chose**, not an absolute hardware limit — extending the budget
would likely let more UCF points succeed too, just very slowly.

## Clip selection

Targets given: ~250, ~800, ~2k, ~5k (reused VIRAT), ~10k, ~30k, ~80k, ~126k raw
container frames, from `eval_results/ucf_inventory.json`/`.md`. `Assault017_x264.mp4`
excluded per instruction (mjpeg, not h264).

**Critical finding discovered during this run, not knowable from ffprobe alone:**
raw container frame count is a poor proxy for the actual scaling variable, `N` =
post-ingest survivor count. UCF's frame-similarity skip ratio is consistently
**~89.3–89.6%** across every clip tested (vs VIRAT's 75%), so survivor N came in far
below the raw-frame targets:

| clip | raw frames (ffprobe) | target N | **actual N (survivors)** | skip ratio |
|---|---|---|---|---|
| Normal_Videos_881 | 224 | ~250 | **24 — EXCLUDED** | 0.893 |
| Normal_Videos_289 | 863 | ~800 | **91** | 0.895 |
| Abuse025 | 2,031 | ~2k | **212** | 0.896 |
| Arrest016 | 10,586 | ~10k | **1,102** | 0.896 |
| Abuse042 | 28,476 | ~30k | **2,963** | 0.896 |
| Arrest047 | 63,060 | ~80k | **6,559** | 0.896 |
| Arson019 | 126,553 | ~126k | **13,506** | 0.893 |
| VIRAT (reused) | — | ~5k | **4,892** | 0.75 |

`Normal_Videos_881` (N=24) is **excluded from the curve**: the harness requires
`N_Q=50` unique seeded queries sampled without replacement from survivor CLIP
embeddings, and 24 < 50. This is a genuine harness boundary condition, not an error —
recorded in `scaling_curve_raw.json` with `excluded: true`.

The achieved spread is still real and useful — **91 to 13,506 survivors**, roughly two
orders of magnitude — just centered lower than the raw-frame targets implied, because
the raw-frame targets were the only information available pre-ingest (per instructions:
"closest available" from `ucf_inventory`, which only has ffprobe data).

## Results table

N = survivor count (post frame-skip). `median_total_retrieval_s` in seconds.

| N (survivors) | source | flat | scene_sparse | scene_sparse shortcut% |
|---:|---|---|---|---:|
| 24 | UCF Normal_Videos_881 | **excluded** (N < N_Q=50) | excluded | — |
| 91 | UCF Normal_Videos_289 | 0.002222 | 0.000810 | 4.0% |
| 212 | UCF Abuse025 | 0.012263 | 0.000934 | 24.0% |
| 1,102 | UCF Arrest016 | 0.470872 | 0.003194 | 10.0% |
| 2,963 | UCF Abuse042 | **FAILED** (wall_timeout @480s, peak 3.79GB) | 0.008239 | 12.0% |
| 4,892 | VIRAT (reused, uncapped) | 7.944805 | 0.007198 | 0.0% |
| 6,559 | UCF Arrest047 | **FAILED** (wall_timeout @480s, peak 15.89GB) | 0.010940 | 18.0% |
| 13,506 | UCF Arson019 | **FAILED** (wall_timeout @480s, peak 26.74GB) | 0.020269 | 8.0% |

### Flat-infeasibility frontier

Flat build+query wall time crossed the 480s budget somewhere between N=1,102
(succeeded, 0.4709s median × 400 executions ≈ 188s total, comfortably under budget)
and N=2,963 (failed). Peak RSS at failure climbs with N (3.79GB → 15.89GB → 26.74GB),
approaching but not yet exceeding the 29GB memory cap at N=13,506 — **wall time, not
memory, was the binding constraint in every observed failure**. Extrapolating flat's
measured exponent (below) suggests memory would eventually bind at even larger N, but
that was not reached here.

### Growth exponents (median total_retrieval_s vs N, log-log slope)

**flat** (successful points only: N=91, 212, 1,102, 4,892 — note the last is the
uncapped VIRAT point on a different video):
- 91→212: k=2.02
- 212→1,102: k=2.21
- 1,102→4,892: k=1.90
- Consistent with **O(N²)**, as expected for PPR over an `N(N-1)/2`-edge fully-connected
  graph. This matches the mechanism identified in the original VAL-scale result
  (`eval_results/P_latency_result.md`).

**scene_sparse** (all 7 points, N=91 through 13,506):
- 91→212: k=0.17
- 212→1,102: k=0.75
- 1,102→2,963: k=0.96
- 2,963→4,892: k=−0.27 (VIRAT point — see caveat below)
- 4,892→6,559: k=1.43
- 6,559→13,506: k=0.85
- Overall (first→last): **k≈0.64**
- Noisy point-to-point (including one non-monotonic dip at the VIRAT point), but
  overall clearly **sub-quadratic and sub-linear-to-linear-ish**, far better than
  flat's ~2.0. The dip and noise are expected: scene_sparse's cost depends on
  scene-graph structure (num_scenes, shortcut/descend branch mix), which differs by
  video content, not just by N — mixing UCF and VIRAT points on one curve introduces
  this confound, called out explicitly rather than smoothed over.

## Component breakdown and branch fire rate

Full per-clip component breakdown (`scene_centroid_rank_s`, `subgraph_induction_s`,
`cross_scene_edges_s`, `scene_ppr_s`, `flat_ppr_s`) and per-clip PPR node/edge counts
are in `eval_results/scaling_curve_raw.json`. Shortcut fire rate ranges 0–24% across
clips (no clear trend with N in this sample), always reported per clip as required.

## Assertions (from scaling_curve_raw.json provenance block)

- `nextqa_cache_sha256_before == nextqa_cache_sha256_after`: **true**
  (`c6a522716ca41f3...`, 87 files, verified before the run and again after all 7
  ingests + all timing arms completed).
- Each non-excluded clip: flat and scene_sparse used the **identical seeded query
  set** (same `query_seed=20260726`, same `n_queries=50` asserted in both arms'
  output).
- Graph built once per mode, outside the timed loop, in every clip (ingest builds
  scene_sparse once and saves it; the flat arm loads the cached index once and builds
  flat once, before entering the query loop).
- Shortcut fire rate reported per clip (table above).
- `eval/data/ucf/index_cache/` received 7 new `.npz` files (one per non-excluded clip
  plus the excluded one); `eval/data/virat/index_cache/` and
  `eval/data/nextqa/index_cache/` untouched by this run (VIRAT point was read-only
  cited, not re-run).

## Caveats (carried forward from the N=4,892 run, restated here)

1. **Synthetic seeded CLIP-embedding queries, not text queries.** `query_embed_s` is
   `null` across every clip and both modes — all 50 queries per clip are L2-normalized
   samples of survivor CLIP embeddings (seed `20260726`), not `_embed_query` text
   encodings. This curve measures retrieval **graph mechanics** only.
2. **Flat graph built in-memory from the same loaded scene_sparse frames**, not from a
   separate flat ingest, in every clip including the reused VIRAT point. Flagged again
   here, still not independently confirmed to be neutral between modes.
3. **Ingest cost (claim B) is separate and NOT addressed by this curve.** Per-clip
   ingest wall time is recorded in `scaling_curve_raw.json` for reference
   (8s at N=24 up to 1,285s / ~21min at N=13,506) but this document is about
   per-query retrieval latency after ingest, not ingest throughput.
4. **New methodology caveat from this run** (see boxed section above): the flat arm's
   480s/29GB watchdog is a **budget choice specific to the 7 new UCF measurements**,
   not applied to the reused, uncapped VIRAT point. Do not read "VIRAT flat succeeded,
   UCF flat at similar-or-smaller N failed" as evidence that VIRAT's graph is
   structurally easier — it was simply allowed to run far longer.
5. **Mixed-dataset curve.** The 7 UCF points and the 1 VIRAT point come from different
   videos with different scene structure, resolution (UCF 320×240 vs VIRAT's native
   res), and codec history. Growth-exponent fits above are directional, not a clean
   controlled experiment — most visible in the non-monotonic scene_sparse dip at the
   VIRAT point.
6. **`Normal_Videos_881` excluded, not zero-cost.** Its ingest (8.0s, 24 survivors) is
   recorded in the raw JSON but no latency numbers exist for it — do not treat its
   absence from the results table as "N=24 is instant."
