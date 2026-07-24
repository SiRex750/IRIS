# P_SELCEILING pre-registration
Read-only diagnostic. VAL only (59 videos / 406 questions). Selects nothing,
changes no default. Follows P_funnel (commit c50fa8f): selection is the
binding constraint (registered top_k=8 selection_headroom 0.336, CI
[0.278, 0.394]).

## Question
Of the questions where the gold frame IS in the retrieved pool but the
current CLIP anchor MISSES it (the recoverable set: pool_coverage=1 AND
peak_in_gold=0), how many would a query-conditional CAPTION signal recover?
This is a proxy-CEILING for in-pool caption reranking. It does NOT build,
tune, or seat a reranker.

## Config
Frozen acceptance cell, top_k=8 ONLY, read from the same BASE dict the
funnel used. flat / ranking_mode=ppr / codec_conf_source=packet_size /
ppr_lambda=0.5 / ppr_damping=0.5 / span_mode=ppr_peak / half_width=2.2 /
peak_source=clip_in_ppr_top8. index_cache/ LOAD ONLY — never regenerate.

## Signal (existing captions only — NO re-ingest, NO new captions)
For each recoverable-set question, take the query-blind captions already in
the loaded index for the top_k=8 pool frames. Score each caption against the
QUESTION text with a fixed, declared-in-advance text-similarity function
(specify exactly which — e.g. the same CLIP text encoder already loaded,
cosine over caption-text vs question-text; or a stated lexical baseline).
Pick the top-scoring frame. Record whether it is in gold.

## Reads (all video-clustered bootstrap, 1000 resamples, seed recorded)
- recoverable_set size (n and fraction of all questions)
- recovered_fraction = P(caption-top frame in gold | recoverable set), with CI
- for context, the same measure over the WHOLE pool, not just recoverable set
- caption availability: fraction of pool frames that actually have a caption
  in the index (assert and report — if captions are sparse this ceiling is
  understated; a zero-caption frame must be counted, not silently skipped)

## Pre-registered gate (declared BEFORE running)
On recovered_fraction over the recoverable set, read on the CI-lower:
  <= 0.10  -> a caption signal recovers too little; in-pool caption reranking
             is NOT worth a fresh-split experiment. Selection stays a stated
             ceiling in the paper, not a lever.
  >= 0.30  -> caption reranking is worth building and validating on a NEW
             split. Becomes the highest-EV P2/P3-adjacent experiment.
  otherwise -> INCONCLUSIVE. Report as such; build nothing.

## STOP condition
Selects nothing. Changes no default, threshold, or frozen value. Any real
reranker is a SEPARATE experiment on a NEW split — the current test half is
BURNED, and a selection rule tuned on val cannot be validated held-out here.

## Limitations (disclose)
1. Uses EXISTING query-blind captions scored against the question. A purpose-
   built reranker could do better OR worse; this is one proxy, not the ceiling
   of all signals.
2. The text-similarity function is itself a choice; the gate is conditional on
   the one declared here.
3. VAL only, n=406, recoverable subset smaller. Test half burned.
4. Conditional on ranking_mode="ppr" throughout, never compared vs legacy.
5. If the same CLIP text encoder is used to score captions, note the shared-
   encoder confound: it partially re-uses the signal that built the pool. A
   lexical or different-model baseline alongside it would separate that — state
   whether one is included.
