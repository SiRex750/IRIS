# Geometry-Gate Plan: skip unconsumed motion-geometry math in charon_v ingest

Status: INVESTIGATION + DESIGN ONLY. No code changed. No re-ingest run.

Trigger: `build_cost_3way.md` found codec EXTRACTION on the VIRAT N=4,892 clip
costs 323s — the largest of all four arms — dominated by
`compute_motion_geometry` inside `charon_v.parse_video`. This plan determines
whether that cost is safe to gate off.

---

## 1. What `compute_motion_geometry` computes (`iris/charon_v.py:21-106`)

Called once per non-SKIP frame (`charon_v.py:364`), over a `(grid_h, grid_w) =
(height // 16, width // 16)` macroblock grid built from the frame's raw H.264
motion vectors (NOT a full per-pixel grid — the task framing overstates
resolution slightly, but the grid still costs real `np.gradient` calls per
frame at 528 scenes / 4,892 frames).

Per frame, in order:
1. Bin motion vectors into per-macroblock mean `U`, `V` (cheap, O(num_mvs)).
2. First-order gradients: `np.gradient(U)`, `np.gradient(V)` → `U_y,U_x,V_y,V_x`
   (2 gradient calls).
3. **divergence** = mean(|U_x + V_y|) — cheap given step 2.
4. **curl** = mean(|V_x - U_y|) — cheap given step 2.
5. **jacobian_frobenius** = mean(sqrt(U_x²+U_y²+V_x²+V_y²)) — cheap given step 2.
6. `M = sqrt(U²+V²)` (motion-magnitude field, cheap).
7. Second-order gradients for the Hessian: `np.gradient(M)` →`M_y,M_x`, then
   `np.gradient(M_y)`, `np.gradient(M_x)` → `M_yy,M_yx,M_xy,M_xx` (**3 more
   gradient calls** — this is the single most expensive block in the function,
   more `np.gradient` invocations than divergence/curl/jacobian combined).
8. **hessian_max_eigenvalue** = mean(0.5*(|trace| + sqrt(max(0, diff²+4·M_yx·M_xy))))
   — depends on step 7.
9. **motion_entropy** = Shannon entropy of a 10-bin histogram of `M` — cheap,
   reuses `M` from step 6, independent of steps 2/7.

Relative cost ranking (most → least expensive): **hessian_max_eigenvalue**
(3 extra gradient calls) > divergence/curl/jacobian_frobenius (2 gradient
calls, shared) > motion_entropy (histogram only, no extra gradients).
Exact per-quantity wall-time split was not profiled in this investigation
(no code was run); this ranking is read directly off the gradient call count
and should be confirmed with a `cProfile`/`line_profiler` pass before
committing to the "expected speedup" numbers in §5.

---

## 2. KEY FINDING — where each quantity actually goes

Traced from `compute_motion_geometry`'s output dict through every place that
name appears in the repo.

| Quantity | (a) persisted to .npz/FrameRecord | (b) read by graph build / edge weighting | (c) read by retrieval/PPR | (d) read by L1 keep_score (ARIA-answer path) |
|---|---|---|---|---|
| divergence | **YES** (`FrameRecord.divergence`, `ingest.py:340`) | NO — dead (see below) | NO | NO — not in `keep_score` formula |
| curl | **YES** (`ingest.py:341`) | NO — dead | NO | NO |
| jacobian_frobenius | **YES** (`ingest.py:342`) | NO — dead | NO | NO |
| hessian_max_eigenvalue | **YES** (`ingest.py:343`) | NO — dead | NO | **YES** — `cached_frame.py:135` `w_hessian * self.motion.hessian_max_eigenvalue` |
| motion_entropy | **YES** (`ingest.py:344`) | NO — dead | NO | **YES** — `cached_frame.py:134` `w_entropy * self.motion.motion_entropy` |

Two genuinely different downstream consumers exist, and they disagree:

**(b)/(c) — the L2 graph / PPR path** (`iris/l2_asphodel.py`, exercised by
`scripts/blockdiag_grounding_gate.py` and the NExT-GQA `peak_in_gold` metric):
`_build_graph`'s `feature_records` dict (`iris/ingest.py:117-137`) bundles all
5 geometry values only *inside* `refined_motion_tensor`
(`ingest.py:125-132`) — it never emits them as top-level `divergence` /
`curl` / ... keys. `L2Asphodel.add_frame_node`'s `get_val(feature_record,
"divergence", 0.0)` (`l2_asphodel.py:765-769`) therefore always falls back to
`0.0`, regardless of `motion_similarity_mode`. Even when
`motion_similarity_mode == "geometry_6d"` (`_motion_similarity`,
`l2_asphodel.py:276-308`), the 6-D vector it builds is `[motion_magnitude,
0.0, 0.0, 0.0, 0.0, 0.0]` for every node — the branch is live code but
operates on dead data. **On this path, all 5 geometry quantities are
computed and then dropped.**

**(d) — the L1 Elysium / ARIA-answer path** (`iris/query.py`,
`iris/pipeline.py` → `wrapper_populate_cache` → `CachedFrame` →
`keep_score`): this path does NOT go through `_build_graph`'s
`feature_records` dict. It is populated straight from the per-frame dict
(originating in `FrameRecord`, which — per point (a) — carries the *real*
computed values), via `FrameMotionDescriptor(divergence=frame.get(...),
curl=frame.get(...), ...)` (`pipeline.py:247-256`, `query.py` equivalents).
`CachedFrame.keep_score` (`cached_frame.py:101-137`) then reads
`self.motion.motion_entropy` and `self.motion.hessian_max_eigenvalue`
unconditionally — not gated by `motion_similarity_mode`, and with non-zero
default weights `l1_w_entropy=0.10`, `l1_w_hessian=0.10`
(`iris_config.py:114-115`). These two values feed the eviction ranking that
decides which frames survive in L1's active context and therefore what text
reaches ARIA's prompt (`as_context_text()` → `aria.generate(...)`,
`pipeline.py:707-708`, `query.py:634`).

`FrameRecord`'s own docstring (`types.py:32-34`) states this explicitly:
*"Motion geometry — carried for exact L1 keep-score / eviction parity. Only
motion_entropy + hessian_max_eigenvalue enter keep_score today, but all five
populate FrameMotionDescriptor in wrapper_populate_cache."* This is a
deliberate, documented design, not an accident.

`divergence`, `curl`, `jacobian_frobenius` are carried into
`FrameMotionDescriptor` (and `CachedFrame.build_motion_embedding`) but are
**never read by `keep_score`**, and `build_motion_embedding` /
`L1ElysiumCache.query(query_motion_embedding=...)` (the only place that would
use all 6 dims) is invoked nowhere in the live pipeline — its only caller
repo-wide is the standalone diagnostic `scripts/compression_analysis.py`.
So for these 3 quantities, (d) is also effectively "computed and dropped" in
production, even though the L1 code technically has a (currently-unused)
consumer path (`l1_elysium.py:96-146`, dual-vector retrieval).

**Bottom line for step 2**: divergence, curl, jacobian_frobenius are dead on
every currently-exercised path. hessian_max_eigenvalue and motion_entropy are
dead on the graph/PPR/grounding path, but genuinely live on the L1
keep_score/ARIA-answer path — a path the `peak_in_gold` acceptance guard
(§7) never exercises and therefore cannot certify.

---

## 3. What the live motion term (`beta*motion`) actually needs

`L2Asphodel._motion_similarity` default branch (`motion_similarity_mode ==
"action_score"`, the config default — `l2_asphodel.py:131`, `iris_config.py:42`):

```python
if max_score_range == 0.0:
    return 1.0
return max(0.0, 1.0 - (abs(node_u.action_score - node_v.action_score) / max_score_range))
```

(`l2_asphodel.py:310-312`, also the fallback inside the dead `geometry_6d`
branch, `l2_asphodel.py:305-308`). This uses only `action_score`, a scalar
computed by `ActionScoreModule` from `luma_diff_energy`, `motion_magnitude`,
and `luma_entropy` (`ingest.py` §1 scoring step) — all three are byproducts
of the demux/motion-vector-extraction pass that `charon_v.parse_video` does
regardless (`motion_magnitude` itself is computed directly from
`motion_vectors` at `charon_v.py:359-360`, independent of
`compute_motion_geometry`). **The live weight formula needs no differential
geometry at all** — not even motion magnitude requires
`compute_motion_geometry`'s grid; `motion_magnitude` is computed upstream of
that function's call site.

---

## 4. Cross-check against the two prior findings

**Prior finding 1 (geometry_6d UNTESTED, two wiring bugs)** —
`DECISIONS.md` "FINDING 3", 2026-07-22 — **HOLDS, confirmed verbatim in
current code**:
- Bug (a): `ingest.py::_build_graph` (current lines ~117-142, was ~117-142 in
  the finding) bundles geometry only inside `refined_motion_tensor`. Verified
  above at `ingest.py:125-137`.
- Bug (b): `motion_similarity_mode` is read once at `L2Asphodel.__init__`
  (`l2_asphodel.py:147`) from the config passed at construction time — frozen
  per graph instance, matching the finding's "frozen at build time" claim.
No code changes since that finding altered either bug.

**Prior finding 2 (live edge weights = alpha*semantic + beta*motion only)**
— confirmed by `build_cost_3way.md` and the `_edge_weight` formula
(`l2_asphodel.py:329-330`: `fully_connected` mode returns exactly
`self.alpha * semantic + self.beta * motion`, no gamma/persistence/geometry
term). **HOLDS.**

**Discrepancy / addition found in this investigation, not present in either
prior finding**: neither prior finding mentions the L1 Elysium keep_score
consumer. That path is real, live-by-default (non-zero weights), and reads 2
of the 5 geometry quantities. This does not contradict either prior finding
(both were scoped to the graph/PPR path) but it means "geometry is fully
unconsumed" is **true for the grounding/PPR path and false in general** —
the correct scope qualifier that was implicit in prior findings needs to be
made explicit before gating.

---

## 5. Proposed gate design

Two independent, differently-scoped cuts:

**Cut A — unconditional, safe on every currently-exercised path:**
Stop computing `divergence`, `curl`, `jacobian_frobenius`. Skip the
first-order-gradient block (`U_y,U_x,V_y,V_x` and the div/curl/jac_norm
arithmetic) entirely. Still compute `M`, the Hessian block, and entropy.
No consumer anywhere in the repo reads these 3 values today (graph path:
dead by wiring bug; L1 path: not in `keep_score`; dual-vector query: never
invoked in production). Zero behavior change on any exercised path.

**Cut B — conditional on whether the L1/ARIA-answer path is in scope for
this ingest run:** Stop computing `hessian_max_eigenvalue` and
`motion_entropy` (skip the second-order-gradient block and the histogram)
when the caller does not intend to run `query.py`'s ARIA-answer flow against
this index — e.g., grounding-only evals like
`scripts/blockdiag_grounding_gate.py` / the NExT-GQA `peak_in_gold` harness.
Gate this behind an explicit new config flag (e.g.
`IRISConfig.compute_l1_geometry: bool`, default `True` to preserve today's
behavior everywhere by default) rather than piggy-backing on
`motion_similarity_mode`, since that flag is already proven to be
disconnected from the L1 path and reusing it would conflate two unrelated
gates. When `False`, `FrameRecord.hessian_max_eigenvalue` /
`.motion_entropy` should serialize as `0.0` (matching the function's existing
early-return default for empty motion vectors, so no new "missing field"
state is introduced), and `keep_score`'s corresponding terms simply evaluate
to 0 for that run — callers who need L1/ARIA behavior must not set the flag
to `False`.

**Expected speedup:** Cut A removes 2 of 5 `np.gradient` calls per frame; Cut
B (when applicable) removes the remaining 3 plus the histogram — the larger
share per §1's cost ranking. Applying both on a grounding-only run should
remove effectively all of `compute_motion_geometry`'s cost, leaving
`charon_v.parse_video`'s extraction time close to the ~72s pixel_diff decode
floor reported in `build_cost_3way.md` for the VIRAT clip (this is a
directional estimate from the gradient-call count, not a measurement —
confirm with a profiled before/after run in the implementation phase).
Cut A alone, applied everywhere (including L1/ARIA runs), is a smaller,
strictly-safe speedup with no scope conditions attached.

---

## 6. Identity test specification

Two separate identity tests are needed, matching the two consumers in §2.

**Test 1 — graph/PPR path (covers Cut A + Cut B together, since neither
Cut affects this path):**
On an existing cached clip (e.g. one of the UCF/VIRAT clips already ingested
under `eval_results/` — reuse the same clip(s) `build_cost_3way.py` used, no
new video acquisition needed), run ingest twice: once with today's code
(baseline), once with the gate applied (geometry computation skipped per
Cuts A+B). Assert, with **zero tolerance**:
- `L2Asphodel.export_graph_data()`'s `edges` list is identical: same
  `(source, target, weight, edge_type, semantic_weight, motion_weight,
  temporal_weight)` tuples, bit-for-bit (`weight` compared with `==`, not
  `np.isclose`, since the formula never reads geometry on this path so the
  floats should be produced by the exact same arithmetic).
- Every `FrameRecord` field the graph path reads (`action_score`,
  `persistence_value`, `luma_diff_energy`, `motion_magnitude`,
  `luma_entropy`, `packet_size`, `pict_type`, `codec_conf`, `scene_id`,
  `pagerank_score`) is unchanged.
- `retrieve_ppr(...)` returns the same ranked frame list and the same
  `retrieval_contributions` dict for a fixed query embedding.
Model this on `tests/test_geom_fields.py::test_geometry_survives_roundtrip`
and `tests/test_ppr_retrieve.py` (both already exist) — extend/parametrize
rather than writing new fixtures from scratch.

**Test 2 — L1 keep_score path (only exercised if Cut B is applied to a run
that also intends to use L1; must FAIL loudly, not silently pass, if Cut B
is misapplied to such a run):**
With Cut A only: assert `CachedFrame.keep_score(...)` is bit-identical
before/after for a batch of frames (model on
`tests/test_l1_elysium.py::test_eviction_respects_keep_score_not_just_insertion_order`),
since Cut A never touches the 2 fields `keep_score` reads.
With Cut B applied to a run where `compute_l1_geometry=False`: explicitly
assert `keep_score` **changes** (both `motion_entropy` and
`hessian_max_eigenvalue` terms go to 0) — this is the documented,
accepted behavior change for that opt-in mode, not a bug. The test should
also assert that `IRISConfig.compute_l1_geometry=True` (the default)
reproduces the current baseline `keep_score` exactly, so the default
zero-config behavior is provably unchanged.

If Test 1 shows any divergence for Cut A alone, Cut A is not safe and the
finding in §2 (col (b)/(c)) is wrong — stop and re-investigate before
implementing anything.

---

## 7. Acceptance guard

Per the pre-registered NExT-GQA VAL cell (`DECISIONS.md`, `peak_in_gold ==
0.3227`, `scripts/blockdiag_grounding_gate.py`, `ranking_mode="ppr"`,
`codec_conf_source="packet_size"`): rerun that exact harness with Cut A (and
Cut B, since this harness never touches L1) applied, and assert
`peak_in_gold_rate == 0.3227` exactly (no tolerance) alongside the existing
per-question bit-identical assertions the script already makes (retrieved
order, peak, span, IoP — `blockdiag_grounding_gate.py:359`).

**Caveat, stated explicitly so it isn't lost**: this guard, by construction,
never exercises L1 or `keep_score`. Passing it certifies Cut A and Cut B are
safe *for the grounding/PPR/peak_in_gold metric* — it does **not** certify
that Cut B is safe for the ARIA-answer/query.py pipeline. Test 2 (§6) is the
only check that covers that path; the acceptance guard alone is not
sufficient evidence to ship Cut B unconditionally.

---

## 8. RISK CALL

Split verdict, because the two consumers disagree:

- **divergence, curl, jacobian_frobenius → clean "remove dead computation."**
  Provably unconsumed by every path currently exercised in this repo (graph
  build: broken wiring, confirmed unchanged since the 2026-07-22 finding;
  L1 keep_score: never read; dual-vector L1 query: never invoked outside a
  standalone script). Safe, unconditional win. This is Cut A.

- **hessian_max_eigenvalue, motion_entropy → "consumed somewhere subtle,"
  harder, must preserve by default.** These are read unconditionally by
  `CachedFrame.keep_score` in the L1/ARIA-answer pipeline, with non-zero
  default weights, and `FrameRecord`'s own docstring documents this as
  intentional ("carried for exact L1 keep-score / eviction parity"). Gating
  these off is only safe for callers that don't exercise that pipeline
  (e.g., grounding-only NExT-GQA evals) and must be an explicit, off-by-default
  opt-in (Cut B), not a blanket default change, or it silently changes which
  frames ARIA sees in its answer-generation context.

Recommended sequencing for implementation (not part of this investigation):
ship Cut A first (strictly safe, no config surface needed beyond deleting
dead arithmetic), validate with Test 1 + the acceptance guard, then propose
Cut B as a separate, explicitly-scoped follow-up gated by a new config flag,
validated by Test 2 in addition to Test 1 + the acceptance guard.
