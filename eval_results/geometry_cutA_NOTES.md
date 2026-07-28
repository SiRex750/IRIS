# Geometry Cut A — Notes (dead-code removal, NOT a speedup)

Source: investigation in `eval_results/geometry_gate_plan.md`, gate script
`scripts/geometry_cutA_identity_gate.py`, results in
`geometry_cutA_identity_gate.{json,md}`, raw run log
`geometry_cutA_run.log`. Code change: `iris/charon_v.py` (`compute_full_geometry`
param, default `False`), `iris/iris_config.py` (new `compute_full_geometry: bool
= False` field), `iris/ingest.py` (wired through to `parse_video`).

## What this is

Removes computation of 3 motion-geometry quantities — `divergence`, `curl`,
`jacobian_frobenius` — proven consumed by **nothing** on any currently-exercised
path:
- graph/PPR build (`L2Asphodel`) reads none of them (dead by an existing
  wiring bug, unrelated to this change — see `geometry_gate_plan.md` §2).
- L1/ARIA `keep_score` (`cached_frame.py`) reads only `hessian_max_eigenvalue`
  and `motion_entropy` — never these 3.

## What this is NOT

This is **dead-code removal, not an optimization**. Measured speedup on the
VIRAT N=4,892 clip (528 scenes, the only clip gated):

- extraction time: 362.685s (full geometry) → 358.706s (Cut A) — **3.979s
  delta, 1.1% of full-geometry extraction time.**

The extraction cost lives in the demux and/or the retained (L1-consumed)
Hessian/entropy computation — not in the 3 quantities this cut removes.
**Do not cite Cut A as a performance improvement.**

## Verification (identity gate, GATE_PASS — see geometry_cutA_identity_gate.json)

All checked bit-identical between full-geometry and Cut-A re-ingests of the
same VIRAT clip:

- **Graph identity**: edges (23,571 in both), all weight/semantic/motion/
  temporal/edge_type fields, 0 mismatches, 0 only-in-one-side.
- **PPR**: ranking order and scores bit-identical (top-10 identical list).
- **Cut-B fields untouched**: `hessian_max_eigenvalue` and `motion_entropy`
  bit-identical across both runs (0 mismatches) — confirming this change
  does not touch the L1-consumed quantities.
- **Schema preserved**: `divergence`/`curl`/`jacobian_frobenius` still present
  on every frame record under Cut A, serialized as sentinel `0.0` (matching
  the function's existing empty-input default) rather than omitted — no
  missing-field/KeyError risk downstream.
- **peak_in_gold path**: covered by the graph/PPR identity check above (Cut A
  never touches the fields that path reads); not independently re-run here.

## Default flag

`IRISConfig.compute_full_geometry` defaults to `False` (new default, this
change). Confirmed safe as an unconditional default-off because no consumer
anywhere in the repo reads the 3 gated quantities today. `hessian_max_eigenvalue`
and `motion_entropy` (Cut B) are always computed regardless of this flag —
untouched, still on by default, still feed `keep_score`.

Existing callers updated: `iris/ingest.py` passes
`compute_full_geometry=getattr(config, "compute_full_geometry", False)`
through to `charon_v.parse_video`. Existing unit tests
(`tests/test_audit_fixes.py`, `tests/test_motion_geometry.py`) that assert on
`divergence`/`curl` values now pass `compute_full_geometry=True` explicitly,
since they test that computation directly.

## Not done here (Cut B, out of scope)

`hessian_max_eigenvalue` and `motion_entropy` remain fully computed,
unconditionally, on every path — they are live inputs to L1 `keep_score`.
Gating those off (Cut B) was investigated in `geometry_gate_plan.md` §5-8 but
is a separate, harder, explicitly-scoped follow-up (would change ARIA-answer
behavior for callers who opt in) — not implemented or gated in this change.
