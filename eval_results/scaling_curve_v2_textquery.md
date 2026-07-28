# Query-latency scaling curve v2, text-query variant — prediction did NOT hold

Re-runs `eval_results/scaling_curve_v2.md`'s flat/scene_sparse curve
(same 7 clips, same `N_WARMUP=3`/`N_REPS=5`/`TOP_K=8`, same uniform
`wall_time_cap_sec=3600.0`/`mem_cap_bytes=29e9` watchdog) with
`--query-source text` instead of v2's default `--query-source synthetic`.
Full methodology and pre-registered prediction: `eval_results/scaling_curve_v2_textquery_NOTES.md`.

**The pre-registered prediction (exponents invariant to query type,
staying ~2.09 flat / ~0.66 scene_sparse) did not hold.** Both exponents moved
outside noise, and scene_sparse's shift has an identified, deterministic
mechanism: real text queries never trigger scene_sparse's cheap "shortcut"
branch. Per the pre-registration, this is reported as a finding, not tuned
away.

## Fitted exponents (UCF-only fit, `log(median_total_retrieval_s) = k·log(N) + c`)

| arm | v2 (synthetic) k | v2 R² | textquery k | textquery R² | Δk (relative) |
|---|---:|---:|---:|---:|---:|
| flat | 2.086 | 0.9988 | **1.926** | 0.9991 | −7.7% |
| scene_sparse | 0.663 | 0.9683 | **0.537** | 0.9133 | **−19.0%** |

flat: fit on the same 4 UCF points that completed within the 3600s cap
(N = 91, 212, 1,102, 2,963) — Arrest047 and Arson019 timed out in both runs.
scene_sparse: fit on all 6 UCF points (N = 91 through 13,506), all completed
in both runs.

## Per-clip results

| N (survivors) | source | flat outcome | flat median_total_retrieval_s (v2 → textquery) | scene_sparse median_total_retrieval_s (v2 → textquery) | scene_sparse shortcut% (v2 → textquery) |
|---:|---|---|---:|---:|---:|
| 91 | UCF Normal_Videos_289 | completed | 0.002206 → 0.003780 | 0.000797 → 0.001435 | 4.0% → **0.0%** |
| 212 | UCF Abuse025 | completed | 0.009783 → 0.015694 | 0.000912 → 0.003884 | 24.0% → **0.0%** |
| 1,102 | UCF Arrest016 | completed | 0.358702 → 0.403426 | 0.002357 → 0.003895 | 10.0% → **0.0%** |
| 2,963 | UCF Abuse042 | completed | 2.903400 → 2.976174 | 0.005008 → 0.007937 | 12.0% → **0.0%** |
| 4,892 | VIRAT (cross-dataset, not in fit) | completed | 7.991639 → **7.939508** | 0.006969 → 0.010092 | 0.0% → 0.0% |
| 6,559 | UCF Arrest047 | timed_out @3600s (both) | — | 0.010416 → 0.017372 | 18.0% → **0.0%** |
| 13,506 | UCF Arson019 | timed_out @3600s (both) | — | 0.021042 → 0.028964 | 8.0% → **0.0%** |

Flat wall times / outcomes: identical shape to v2. Arrest047 and Arson019
still time out at the same 3600s cap under text queries (Arson019 peak RSS
26.94GB vs v2's 26.90GB — essentially unchanged memory profile). VIRAT
(4,892, not part of the UCF fit) reproduces almost exactly: 7.9395s here vs
7.9916s in v2 (0.65% difference, within run-to-run noise) — flat's raw
per-query cost at fixed N is close to invariant; it's the **fitted slope**
across N that shifted, driven mostly by the smallest points (N=91: +71%
relative; N=212: +60%; N=1,102: +12.5%; N=2,963: +2.5% — a converging
pattern, not simple noise scatter, since flat's R² stayed ≥0.999 in both
runs: both are clean power laws, just with different exponents).

## The scene_sparse finding: shortcut branch never fires under text queries

`branch_fire_rate.shortcut_pct` is **0.0% for every single clip** under
`query_source=text`, vs. 4.0%–24.0% under `query_source=synthetic` (v2).
This is deterministic, not noisy: scene_sparse's shortcut branch fires when
a query lands close enough to an existing scene centroid to skip full
subgraph induction + PPR. A `synthetic` query IS a sampled survivor CLIP
*image* embedding — by construction it is near-identical to some frame's
embedding in the graph, so it frequently lands inside a scene's shortcut
margin. A generic real-world text query ("a person running") has no reason
to sit that close to any specific frame's embedding, so it always falls
through to the more expensive `descend` path (`scene_centroid_rank_s` +
`subgraph_induction_s` + `cross_scene_edges_s` + `scene_ppr_s`, all populated
every time — see `median_total_retrieval_s_descend` in the raw JSON for
every clip, `median_total_retrieval_s_shortcut` is `null` throughout).

This raises scene_sparse's per-query cost at every N (see table above,
2nd-to-last column), and because the shortcut/descend cost gap is
proportionally larger at small N (small graphs have few scenes, so the
shortcut savings matter more relative to fixed overhead) than at large N,
it flattens the fitted log-log slope — hence k dropping from 0.663 to 0.537.
**Query type is not a nuisance variable for scene_sparse — it directly gates
which code path runs.** The synthetic-query curve was, in this specific
sense, measuring an easier workload than real queries will produce in
deployment.

## What is NOT called into question

- Both arms remain **clean power laws** (flat R²=0.999, scene_sparse
  R²=0.913 — still a good fit, just noisier than v2's 0.968).
- **flat still scales far worse than scene_sparse** at every N tested;
  scene_sparse's real-N ordering versus flat is unchanged.
- VIRAT's near-exact reproduction (0.65% diff) shows flat's absolute
  per-query cost at a fixed, moderate N is not meaningfully sensitive to
  query type — the effect here is specifically on the **fitted slope**
  (small-N leverage) and, for scene_sparse, on **branch selection**.
- The wall-time frontier (flat timing out at N=6,559/13,506 under the
  uniform 3600s cap) is unchanged.

## Provenance

Raw data: `eval_results/scaling_curve_v2_textquery_raw.json`. Harness:
`scripts/scaling_curve_v2_textquery.py` + `scripts/scaling_curve_v2.py` +
`scripts/_scaling_curve_v2_worker.py` (`--query-source text`). Text queries:
10 fixed surveillance-domain phrases (see NOTES), cycled to 50/clip,
CLIP ViT-B/32 text-encoded via `iris.query._embed_query` (same model used at
ingest) inside the timed region every rep — `query_embed_s` is populated
throughout this run (~0.017s median, consistent across clips and both graph
modes, confirming the text-embedding cost itself IS the constant predicted
pre-registration — it's the *downstream* branch-selection effect that was
not predicted).

Arson019's flat arm (the last of the 7) was launched via a detached Windows
Scheduled Task after the interactive-session background process was twice
reclaimed mid-run during idle gaps between messages (diagnosed as
session/job-object teardown, not OS sleep — no `Kernel-Power` sleep/resume
events in the System log across either kill window). Same worker script,
same CLI args, same watchdog policy; only the launch mechanism differs, and
its result (timed_out @3600.17s, peak RSS 26.94GB) is consistent with the
other 6 clips' pattern and with v2's Arson019 flat result (timed_out
@3600s, peak RSS 26.90GB).
