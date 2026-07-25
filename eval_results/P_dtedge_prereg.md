# P_DTEDGE pre-registration
Read-only ABLATION. VAL only (59 videos / 406 Qs). Selects nothing, seats
nothing. Measures whether wiring directed temporal edges into the E2E
retrieval path moves anchor selection on the temporal-question subset.

## Hypothesis
Directional PPR bias (forward for "after", backward for "before") improves
peak_in_gold on temporal questions specifically, because it gives the anchor
a query-conditional reason to prefer the correct in-pool frame over an
equally-CLIP-similar frame on the wrong side of the reference event. Predict:
positive effect on the temporal subset, ~zero effect on non-temporal
questions (which have no direction to exploit).

## Question subset
Partition the 406 VAL questions into TEMPORAL vs NON-TEMPORAL using the
EXISTING query-reformulation code's own classification (do not invent a new
classifier — reuse whatever the single-video path already uses to detect
before/after/causal). Report the subset sizes. The primary read is on the
temporal subset; non-temporal is the negative control.

## Arms (both on the frozen acceptance cell, top_k=8)
  UNDIRECTED (baseline) — current E2E path, directed edges OFF.
  DIRECTED — the same path with directional temporal bias wired in.
Everything else identical: flat / ranking_mode=ppr / codec_conf_source=
packet_size / ppr_lambda=0.5 / ppr_damping=0.5 / span_mode=ppr_peak /
half_width=2.2 / peak_source=clip_in_ppr_top8. index_cache/ LOAD ONLY.

## Primary read (video-clustered bootstrap, 1000 resamples, seed recorded)
  peak_in_gold: DIRECTED − UNDIRECTED, on the temporal subset, with CI.
  Same paired delta on the non-temporal subset (control — expect ~0).
  Report mIoP / IoP@0.5 deltas alongside for context.

## Fire-rate assertion (silent-fallback rule)
The directional reformulation MUST count how often it actually fires (detects
a direction and applies bias) vs falls back to undirected. Report the fire
rate on the temporal subset. If a "directed" run silently runs undirected on
some question class, that is a silent fallback and the result is void — the
counter must make it visible. Assert the fire rate is reported, not assumed.

## Pre-registered gate (declared before running)
On the temporal-subset peak_in_gold delta, CI-lower:
  > 0      -> directed edges recover selection on temporal questions;
              worth building and validating on a FRESH split.
  crosses 0 -> no separated effect at this n; report as a tie, build nothing.
  Non-temporal control must NOT show a large effect; if it does, the change
  is doing something other than exploiting direction and the temporal read
  is confounded — report that and stop.

## STOP condition
This is a VAL diagnostic. It selects nothing, changes no frozen default. Any
seating of directed edges is a SEPARATE experiment on a NEW split — the test
half is BURNED.

## Wiring constraints (for the implementation step, NOT this one)
  - Do the wiring on a branch that leaves the frozen acceptance config
    intact. After wiring, the acceptance test (peak_in_gold 0.3227 at the
    frozen cell, directed edges OFF) MUST still pass EXACTLY — that proves
    the UNDIRECTED arm is byte-identical to the measured baseline.
  - Directed edges must be a flag OFF by default; the baseline path must be
    provably unchanged.

## Limitations (disclose)
1. VAL only, n=406, temporal subset smaller. Test half burned.
2. Uses the existing reformulation classifier; its temporal-detection recall
   bounds what this can measure.
3. Conditional on ranking_mode="ppr" throughout.
4. A null here does not rule out directed edges under a better temporal
   classifier or on a domain with more temporal questions (VIRAT/CCTV).
