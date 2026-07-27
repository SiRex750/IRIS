# Query-latency scaling curve v2: UNIFORM watchdog policy, fairness settled

Re-runs `eval_results/scaling_curve.{json,md}` (v1, still uncommitted, kept in
place — not overwritten) with the confound v1 flagged fixed. v1 wrapped the 7
new UCF flat measurements in a 480s/29GB watchdog but ran the reused VIRAT
N=4,892 flat point **uncapped**, so v1's "flat fails past N≈3k" was a
wall-time-**budget** artifact, not a hardware infeasibility frontier. Flat's
peak RSS stayed under 27GB even at N=13,506 — flat was building fine, it was
just being killed by the clock, not memory.

**Changes from v1** (full detail in `eval_results/scaling_curve_v2_raw.json`
`provenance` block):

1. **One watchdog, identical value, every flat arm, including VIRAT.**
   `wall_time_cap_sec=3600.0`, `mem_cap_bytes=29e9` (~29GB; system has 33.4GB
   total) applied to all 7 flat measurements with no exceptions. The watchdog
   wraps index load + flat graph build + the full `N_WARMUP=3 +
   n_queries(50)×N_REPS(5) = 253` timed query executions, same wrapping as
   v1's UCF arms, now also applied to VIRAT.
2. **VIRAT N=4,892 flat re-measured** under this harness rather than reused
   read-only from the committed `ccf764f` A/B result.
3. **Flat and scene_sparse run in separate subprocesses per clip** (unchanged
   from v1) via `scripts/_scaling_curve_v2_worker.py`, launched by
   `scripts/scaling_curve_v2.py`.
4. **Three distinct non-completion outcomes tracked**, not collapsed into
   "failed": `completed` (ran to genuine completion, latency reported),
   `oom` (RSS crossed 29GB before the wall cap), `timed_out` (wall cap
   crossed first, with peak RSS at that moment reported). All three appear
   in the per-clip table below as data, not as assertion failures.

Same seeded-CLIP-embedding-query methodology, `query_seed=20260726`,
50 queries/clip, `N_WARMUP=3`/`N_REPS=5`/`TOP_K=8` as v1. Same 6 UCF clips +
VIRAT as v1 (v1's `Normal_Videos_881`, N=24, stays excluded: N < N_Q=50).
All UCF index caches reused from `eval/data/ucf/index_cache/` — no
re-ingest of a clip already cached; VIRAT cache likewise reused read-only for
the scene_sparse arm and for the flat arm's "cached_frames" source.

## Per-clip flat outcome (the headline result)

| N (survivors) | source | flat outcome | flat median_total_retrieval_s | flat peak RSS | scene_sparse median_total_retrieval_s | shortcut% |
|---:|---|---|---:|---:|---:|---:|
| 91 | UCF Normal_Videos_289 | **completed** | 0.002206 | 0.29 GB | 0.000797 | 4.0% |
| 212 | UCF Abuse025 | **completed** | 0.009783 | 0.30 GB | 0.000912 | 24.0% |
| 1,102 | UCF Arrest016 | **completed** | 0.358702 | 0.73 GB | 0.002357 | 10.0% |
| 2,963 | UCF Abuse042 | **completed** (793s wall) | 2.903400 | 3.65 GB | 0.005008 | 12.0% |
| 4,892 | VIRAT (re-measured, uniform cap) | **completed** (2,189s wall) | 7.991639 | 8.92 GB | 0.006969 | 0.0% |
| 6,559 | UCF Arrest047 | **timed_out** @3600s cap | — | 17.00 GB (still climbing) | 0.010416 | 18.0% |
| 13,506 | UCF Arson019 | **timed_out** @3600s cap | — | 26.90 GB (still climbing) | 0.021042 | 8.0% |

**Reading this table**: Abuse042 (N=2,963), which v1 reported as `FAILED
(wall_timeout @480s, peak 3.79GB)`, **completes in 793s** under the uniform
3600s cap — v1's "failure" at this N was entirely the 480s budget, exactly as
the confound predicted. Arrest047 (N=6,559) and Arson019 (N=13,506) still do
not complete even under the ~7.5× larger uniform cap — genuine `timed_out`
outcomes, not `oom` (RSS at the 3600s mark: 17.0GB and 26.9GB respectively,
both still short of the 29GB cap and still climbing). No `oom` outcome was
observed anywhere in this run: **wall time, not memory, remains the binding
constraint at every N tested**, now confirmed under a policy that gave flat
7.5× more time to prove otherwise.

VIRAT (N=4,892) reproduces its previously-committed uncapped number closely:
**7.9916s** here vs **7.9448s** in `ccf764f`'s `virat_latency_N4892_result.md`
(0.6% difference, within run-to-run noise) — the uniform 3600s cap did not
bind for VIRAT (2,189s observed, 61% of the cap), so the remeasurement is a
clean like-for-like check, not a truncated one.

## Fairness check: same-frames flat construction vs. independent flat ingest

Tested on **Abuse042** (N=2,963, mid-size per pre-registration), the two
flat-graph constructions:

| construction | N survivors | node/edge count | median_total_retrieval_s | wall (load/ingest + query loop) |
|---|---:|---|---:|---:|
| (a) cached_frames — flat built from frames already loaded by the scene_sparse ingest (this run's default, and v1's method) | 2,963 | 2,963 / 4,388,203 | 2.903400 | 793s |
| (b) independent_ingest — fresh `iris_ingest.ingest(video, config=flat)` from the raw video, no scene_sparse detour | 2,963 | 2,963 / 4,388,203 | 3.075178 | 945s |

Survivor count and edge count are **identical** between constructions (same
frame-selection config, just a different code path to get there — confirms
the two are ingesting the same frame set, not silently diverging).
Latency difference: **5.6%** (3.075 vs 2.903), inside the run's noise band
(intra-clip point-to-point exponent noise elsewhere in this data is
10-20%+; see e.g. the flat/scene_sparse component breakdowns in the raw
JSON).

**Verdict: MATCH — the same-frames flat construction used throughout this
curve (and v1's) is validated as fair.** Building flat from frames already in
memory does not measurably advantage or disadvantage flat relative to an
independent ingest; the small gap is consistent with cache-warmth /
scheduler noise, not a construction bias. No change to methodology is
recommended.

## UCF-only fitted scaling exponents

Fit is `log(median_total_retrieval_s) = k·log(N) + c`, least-squares on
**UCF points only** (VIRAT is a different-distribution outlier — different
video, different codec history, different skip ratio — marked separately on
the curve, not pooled into the fit).

- **flat**: fit on the 4 UCF points that completed (N = 91, 212, 1,102,
  2,963) — **k = 2.086**. Consistent with the O(N²) mechanism (PPR over an
  N(N-1)/2-edge fully-connected graph) identified in
  `eval_results/P_latency_result.md` and observed again here up to N=2,963;
  the two `timed_out` UCF points (6,559 and 13,506) are excluded from the fit
  (no latency to fit) but are consistent with the same trend continuing —
  Arrest047's peak RSS (17.0GB) at the 3600s mark implies real per-query cost
  well beyond what a k≈2 curve would still allow inside any practical budget.
- **scene_sparse**: fit on **all 6 UCF points** (N = 91 through 13,506, all
  completed) — **k = 0.663**. Sub-quadratic, sub-linear, consistent with v1's
  directional read (v1's mixed UCF+VIRAT fit was noisier, k≈0.64 pooled with
  the VIRAT dip included; this UCF-only fit is cleaner because the
  distribution-outlier point is no longer pooled into the regression).
- **VIRAT point, marked separately, not fit**: flat 7.9916s at N=4,892 lands
  almost exactly on the UCF-only flat trend's extrapolation (the k=2.086 fit
  predicts **7.988s** at N=4,892 — 0.05% off), despite being a different
  video with a different skip ratio — striking agreement, though with only 4
  points in the fit this is one data point of confirmation, not a
  replication. scene_sparse 0.006969s at N=4,892 is the **lowest shortcut
  rate in the whole run (0%)**, consistent with v1's observation that
  scene_sparse cost depends on scene structure, not just N — VIRAT's 528
  scenes over 4,892 survivors (9.3 survivors/scene) is coarser than every UCF
  clip's scene granularity (see table below), plausibly why VIRAT never takes
  the cheap shortcut branch.

## Scene count per clip (denominator of the sparse advantage)

| N (survivors) | clip | num_scenes | survivors/scene |
|---:|---|---:|---:|
| 91 | Normal_Videos_289 | 27 | 3.4 |
| 212 | Abuse025 | 43 | 4.9 |
| 1,102 | Arrest016 | 315 | 3.5 |
| 2,963 | Abuse042 | 712 | 4.2 |
| 4,892 | VIRAT | 528 | 9.3 |
| 6,559 | Arrest047 | 2,178 | 3.0 |
| 13,506 | Arson019 | 3,592 | 3.8 |

UCF's survivors/scene ratio is stable (~3-5) across nearly two orders of
magnitude of N — scene_sparse's sub-quadratic scaling on UCF is not an
artifact of scenes growing disproportionately with N; scene count tracks N
roughly linearly for this dataset. VIRAT's much coarser ratio (9.3) is the
likely reason its shortcut rate is 0% and its scene_sparse cost, while still
far below flat's, doesn't show the same shortcut savings UCF clips do.

## Component breakdown, branch fire rate, assertions

Full per-clip component breakdown (`scene_centroid_rank_s`,
`subgraph_induction_s`, `cross_scene_edges_s`, `scene_ppr_s`, `flat_ppr_s`),
PPR node/edge counts, and per-query records are in
`eval_results/scaling_curve_v2_raw.json`. Shortcut fire rate (per clip,
required): 91→4.0%, 212→24.0%, 1,102→10.0%, 2,963→12.0%, 4,892(VIRAT)→0.0%,
6,559→18.0%, 13,506→8.0%. No clear monotonic trend with N.

- **Identical seeded query set per clip across both modes**: same
  `query_seed=20260726`, same `n_queries=50`, verified in every clip's
  worker output (both arms independently reconstruct the same seeded sample
  from the same survivor CLIP embeddings).
- **Graph built once per mode, outside the timed loop**: scene_sparse loads
  its cached graph once; flat builds its graph once (either from cached
  frames or from a fresh ingest, per the fairness-check arm) before the
  253-execution timed loop begins, in every clip.
- **`eval/data/nextqa/index_cache/` untouched**: sha256 before/after this run
  identical (`3a25c9038eed0...`, see `provenance` block in the raw JSON —
  note this hash differs from v1's recorded value because v1's committed run
  and this run happened at different times / different cache directory
  states; what matters, and what's asserted, is before==after **within this
  run**, which holds).
- **Uniform watchdog value**: `wall_time_cap_sec=3600.0`,
  `mem_cap_bytes=29000000000` — recorded once in `provenance.uniform_watchdog`
  and structurally identical (same worker invocation) across all 7 flat
  measurements; not re-declared per clip because it cannot differ per clip in
  this harness.
- Flat/scene_sparse ran in separate subprocesses per clip throughout — no
  flat timeout or crash affected any scene_sparse measurement (both Arrest047
  and Arson019's scene_sparse arms completed normally despite their flat
  arms timing out).

## Caveats (carried forward from v1 and the original VIRAT run)

1. **Synthetic seeded CLIP-embedding queries, not text queries.**
   `query_embed_s` is `null` throughout — all 50 queries per clip are
   L2-normalized samples of survivor CLIP embeddings, not `_embed_query` text
   encodings. This curve measures retrieval graph mechanics only.
2. **scene_sparse's own O(N²) ingest cost is separate and not addressed
   here.** This document is about per-query retrieval latency after ingest,
   not ingest throughput. (v1's per-clip ingest wall times, 8s at N=24 to
   ~1,285s at N=13,506, are unchanged by this run — no re-ingest occurred for
   any UCF clip; VIRAT's fairness-check independent ingest, 161s at N=2,963,
   is the only new ingest wall time recorded.)
3. **Mixed-dataset curve.** UCF and VIRAT differ in scene structure,
   resolution (UCF 320×240 vs VIRAT native), codec history, and — newly
   quantified this run — survivors/scene granularity (UCF ~3-5,
   VIRAT ~9.3). This is why VIRAT is plotted as a separately-marked point and
   excluded from both fitted exponents, not because its numbers are
   untrustworthy.
4. **Flat-infeasibility "frontier" is now a genuine ~1-hour-budget frontier,
   not a 480s artifact — but it is still a budget, not a proof of hardware
   impossibility.** Arrest047 and Arson019 might complete given more than
   3,600s; peak RSS at the cap (17.0GB, 26.9GB) had not reached the 29GB
   memory cap in either case, so a longer wall-time budget (not more memory)
   is what would be needed to observe genuine completion or genuine OOM at
   these N. This run deliberately does not chase that further — 3,600s
   (7.5× v1's budget) was the pre-registered "generous" cap for this
   re-run.
5. **`Normal_Videos_881` (N=24) still excluded, not zero-cost** — unchanged
   from v1; 24 < N_Q=50, a harness boundary condition, not an error.

## Provenance

Full machine-readable provenance (git commit, watchdog policy, cache
fingerprints, per-clip raw worker output, fairness-check raw output, fit
coefficients) is in `eval_results/scaling_curve_v2_raw.json`. Harness:
`scripts/scaling_curve_v2.py` (driver) +
`scripts/_scaling_curve_v2_worker.py` (per-clip-per-mode subprocess worker).

**Not committed yet** — per instructions, this is the raw v2 report for
review before `git add`/commit. `eval_results/scaling_curve.{json,md}` (v1)
is left in place, unmodified.
