# Scaling curve v2, text-query variant — pre-registration + result

## What this run changes vs. v2

`eval_results/scaling_curve_v2.md` reported 50 queries/clip built as seeded
L2-normalized samples of survivor CLIP *image* embeddings
(`scripts/_scaling_curve_v2_worker.py:seeded_queries`) — `query_embed_s` was
null throughout. A reviewer reads this as "no real query was ever run
through the scaling experiment."

This run adds `--query-source text` (opt-in; default stays `synthetic`,
mirroring the `graph_edge_mode` flag pattern) to
`scripts/_scaling_curve_v2_worker.py`. Under `text`, each clip's 50 queries
are the fixed list `TEXT_QUERIES` (10 surveillance-domain phrases: "a person
running", "an unattended bag", "two people fighting", "a vehicle entering
the scene", "a person falling", "someone climbing over a fence", "a person
carrying a weapon", "a crowd gathering suddenly", "someone breaking a
window", "a person loitering near a doorway"), cycled 5x to reach 50/clip —
same query count as the synthetic run, so per-clip timings stay comparable.
Each query is encoded through `iris.query._embed_query`, which calls the
SAME CLIP ViT-B/32 text encoder (`iris._clip.get_clip_model`) used to
produce the image embeddings at ingest, L2-normalized, same dtype/device.
The embed call happens *inside* the timed region on every rep (not cached),
mirroring `scripts/latency_ab.py`'s `_timed_query` — so `query_embed_s` is
populated this time.

Same 7 clips, same `query_seed=20260726` constant (unused by the text path,
kept for provenance parity), same `N_Q=50`, `N_WARMUP=3`, `N_REPS=5`,
`TOP_K=8`, same uniform `WALL_CAP_SEC=3600` / `MEM_CAP_BYTES=29e9` watchdog
as v2. Driver: `scripts/scaling_curve_v2_textquery.py` (does not re-run v2's
fairness check — independent_ingest vs cached_frames is orthogonal to query
source and was already settled in v2).

## Pre-registered prediction (written BEFORE reading the result)

Fitted exponents stay ~2.09 (flat) / ~0.66 (scene_sparse), matching v2's
`k=2.086` (flat, UCF-only) and `k=0.663` (scene_sparse, UCF-only). The text
query adds a constant ~0.016s embedding cost to BOTH arms (confirmed in a
smoke test: N=91 flat `query_embed_s`=0.0170s, N=91 scene_sparse
`query_embed_s`=0.0167s — same CLIP text encoder call regardless of graph
mode, as expected) and must not change the slope — invariance IS the result
that kills the objection.

**If either exponent moves outside noise, that means query type was
silently affecting graph mechanics — a finding, not something to tune
away.**

## Result: prediction did NOT hold — STOP condition triggered

Per the pre-registration: **the prediction was invariance; the exponents
moved outside noise; this is reported as a finding, not tuned away.**

### Fitted exponents, UCF-only log-log OLS, fit range stated

| arm | full range (n, R²) | N≥1,102 (n, R²) | reported |
|---|---|---|---|
| flat | 1.926 (n=4, R²=.9991) | 2.020 (n=2, R²=1.0, trivial) | **~2 (quadratic)** — flat censors large-N (Arrest047/Arson019 time out), so the exponent rests on the low-N range plus the large-N (VIRAT) invariance check below, not on a clean wide-range fit. |
| scene_sparse | 0.537 (n=6, R²=.913) | 0.817 (n=4, R²=.996) | **PRIMARY ~0.82**, from the clean large-N-only fit. The 0.54 full-range number is shown for comparison and labeled small-N-sensitive — see mechanism below. |

The n=2 flat restricted fit is trivially R²=1.0 (a line through two points)
and is not evidence of quadratic behavior on its own; it's reported to show
that restricting to the large-N range, where the small-N slope leverage
described below is absent, still lands near 2.0.

### Mechanism — two effects, kept separate

1. **Level shift (range-wide):** `branch_fire_rate.shortcut_pct` = **0.0%
   for every one of the 7 clips** under `query_source=text`, vs. 4.0%–24.0%
   under `query_source=synthetic` (v2). This is deterministic, not noise:
   v2's queries were literally sampled survivor CLIP *image* embeddings, so
   they frequently landed inside a scene centroid's shortcut margin and
   skipped full subgraph induction + PPR. Real text queries never do —
   every one falls through to the expensive `descend` path
   (`median_total_retrieval_s_descend` populated at every N;
   `median_total_retrieval_s_shortcut` is `null` throughout). This raises
   scene_sparse's per-query cost at every N, uniformly across the range.
2. **Slope leverage (small-N only):** a small additive per-query overhead
   (~1–6ms) flattens both fitted exponents specifically because N=91 and
   N=212 are included in the full-range fit. At N=91 the shift is
   sub-2ms — jitter-plausible on its own — but at N=212 it is >2ms and
   follows the same direction consistently across clips, which is why it
   reads as systematic rather than scatter (flat's full-range R² stays
   ≥0.999 both runs, i.e. it's a clean power law with a different exponent,
   not fit degradation). This additive term matters proportionally more at
   small N, which is exactly the leverage point of a log-log OLS fit —
   hence the full-range vs. N≥1,102 divergence in both arms.

**Query type is not a nuisance variable for scene_sparse — it directly
gates which code path runs.** The synthetic-query curve was, in this
specific sense, measuring an easier workload than real queries will
produce in deployment.

### query_embed_s

Held as pre-registered: ~0.017s, constant across every clip and both graph
modes (range 0.0163s–0.0180s across all 14 arm-clip combinations), reported
**outside** `total_retrieval_s` (confirmed by inspection —
`total_retrieval_s` is smaller than `query_embed_s` at small N, e.g.
Normal_Videos_289 scene_sparse: `total_retrieval_s`=0.00143s <
`query_embed_s`=0.01684s — the two are recorded as separate timed regions,
not nested). That part of the prediction held; what was not predicted is
the downstream effect of query *content* on scene_sparse's branch
selection.

### Framing: tractability divergence, not a clean exponent ratio

This comparison is **unmatched-N**: flat censors the range where
scene_sparse still runs (flat times out past N≈6.5k; scene_sparse is
tractable through N≈13.5k at ~29ms median). Report this as a **tractability
divergence** — flat exceeds the 1-hour wall-time budget past N~6,559,
scene_sparse remains tractable through N~13,506 — not as a ratio of two
comparable power-law exponents over the same range. Do **not** extrapolate
scene_sparse's fit past N~13,506: the two largest points (N=6,559 → 0.0174s,
N=13,506 → 0.0290s) show mild upward curvature relative to the N≥1,102
fitted line, consistent with the shortcut-branch level shift interacting
with growing scene count, not yet characterized past this range.

flat's absolute per-query cost at a fixed, moderate N is close to
query-invariant: VIRAT (N=4,892, cross-dataset, not in the UCF fit)
reproduces v2 within 0.65% (7.9395s vs 7.9916s), and Abuse042 (N=2,963, the
largest UCF fit point) reproduces within 2.5% (2.9762s vs 2.9034s) — both
consistent with the converging per-N relative-difference pattern (N=91:
+71%, N=212: +60%, N=1,102: +12.5%, N=2,963: +2.5%, VIRAT N=4,892: +0.65%),
i.e. the small-N additive overhead's relative contribution shrinking as N
grows, not run-to-run noise growing or shrinking arbitrarily.

## What is NOT called into question

- Both arms remain **clean power laws** in their respective fit ranges
  (flat full-range R²=0.999; scene_sparse N≥1,102 R²=.996).
- **flat still scales far worse than scene_sparse** at every N tested;
  scene_sparse's real-N ordering versus flat is unchanged.
- The wall-time frontier is unchanged in shape, only relabeled as a
  tractability divergence rather than a same-range comparison: flat exceeds
  the 1-hour budget past N~6,559 (Arrest047, Arson019 both time out at
  3600s under text queries, same as v2); scene_sparse remains tractable
  through N~13,506 at ~29ms median.

## Open for a follow-up run (#2 — NOT started here)

- Per-bucket repeats to put a confidence interval on the fitted exponent,
  rather than a single log-log OLS point estimate per arm.
- Characterize the small-N noise floor directly (is the N=91/N=212
  additive overhead reproducible run-to-run, or itself noisy at that
  scale?).
- Any reported exponent must state its fit range explicitly (full-range vs.
  N≥1,102) — do not quote a bare "k=" number without it, per the divergence
  documented above.

## Provenance

Raw data: `eval_results/scaling_curve_v2_textquery_raw.json`. Harness:
`scripts/scaling_curve_v2_textquery.py` + `scripts/scaling_curve_v2.py` +
`scripts/_scaling_curve_v2_worker.py` (`--query-source text`). Text queries:
10 fixed surveillance-domain phrases (see above), cycled to 50/clip,
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

This was a retrieval-only run: no LLM in the timed path (`iris.query`'s
retrieval stack only; the `LlamaBackend`/`minicpm-v4.6` DIAGNOSTICS line
recorded per-clip in the raw JSON reflects the ingest-time captioner
config, not anything invoked during query timing).

- git SHA (HEAD at commit time): `378daa7be2c4412bcd7cd29dc32d48af2d083ba7`
- `git status --porcelain` at commit time (this run's own outputs plus
  unrelated in-flight work in the tree, none of it staged by this commit):
  ```
   M scripts/_scaling_curve_v2_worker.py
   M scripts/scaling_curve_v2.py
  ?? NExT-GQA/
  ?? eval_results/_detach_test.ps1
  ?? eval_results/_v2_textquery_arson019_flat_done.marker
  ?? eval_results/_v2_textquery_arson019_flat_stdout.log
  ?? eval_results/_v2_textquery_arson019_ss_stdout.log
  ?? eval_results/_v2_textquery_stdout.log
  ?? eval_results/_v2_textquery_tmp/
  ?? eval_results/oom_frontier.json
  ?? eval_results/oom_frontier.md
  ?? eval_results/oomfrontier_flat_Arson019.json
  ?? eval_results/oomfrontier_flat_Arson019_clean.json
  ?? eval_results/oomfrontier_flat_Normal_Videos_924.json
  ?? eval_results/oomfrontier_flat_Normal_Videos_935.json
  ?? eval_results/oomfrontier_scenesparse_Normal_Videos_924.json
  ?? eval_results/oomfrontier_scenesparse_Normal_Videos_935.json
  ?? eval_results/scaling_curve_v2_textquery.md
  ?? eval_results/scaling_curve_v2_textquery_NOTES.md
  ?? eval_results/scaling_curve_v2_textquery_raw.json
  ?? eval_results/ucf_inventory.json
  ?? eval_results/ucf_inventory.md
  ?? scripts/oomfrontier_blockdiag_build_probe.py
  ?? scripts/oomfrontier_flat_build.py
  ?? scripts/oomfrontier_ingest_scenesparse.py
  ?? scripts/scaling_curve_v2_textquery.py
  ```
- Harness config hash (`git hash-object` of the three harness files, since
  they are new/modified and not yet committed as of the run):
  - `scripts/scaling_curve_v2_textquery.py`: `471bb48294a58ace7e84ea2db1006025d07ad446`
  - `scripts/scaling_curve_v2.py`: `16e9e40823a52bba5bfb0b6e9cefcc1176e19c7a`
  - `scripts/_scaling_curve_v2_worker.py`: `b6baa5b919350a97efc585277bdbca137ade912d`
- Machine/env: Windows-11-10.0.26200-SP0, Python 3.12.10 (MSC v.1943 64-bit
  AMD64), backend `LlamaBackend`/`llama3.2:3b` + `MiniCPMCaptioner`/
  `minicpm-v4.6` at ingest (not invoked during query timing — see above).

Full breakdown: `eval_results/scaling_curve_v2_textquery.md`. Raw data:
`eval_results/scaling_curve_v2_textquery_raw.json`.
