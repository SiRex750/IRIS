# P_FUNNEL pre-registration
Read-only diagnostic. VAL only (59 videos / 406 questions). Selects nothing.

## Quantities
Per grounded VAL question, three nested point-in-span membership tests:
- index_coverage: any frame timestamp in the loaded index falls inside any gold span
- pool_coverage:  any retrieved top-k timestamp falls inside any gold span
- peak_in_gold:   the clip_in_ppr_top8 anchor timestamp falls inside any gold span
Strictly nested: index_coverage >= pool_coverage >= peak_in_gold.
Also best_gold_rank = 1-based rank within the retrieved pool, ordered by the
same CLIP score clip_in_ppr_top8 uses, of the highest-ranked in-gold frame;
null when pool_coverage = 0.

## Config (frozen; only top_k varies)
flat / ranking_mode=ppr / codec_conf_source=packet_size / ppr_lambda=0.5 /
ppr_damping=0.5 / span_mode=ppr_peak / half_width=2.2 /
peak_source=clip_in_ppr_top8. top_k in [8, 12, 16, 24].

## Prediction
index_coverage high but not 1.0. pool_coverage rises with top_k while
peak_in_gold is already known flat across 8->24 (0.3227 / 0.3276 / 0.3251 /
0.3251). If that pattern holds, selection rather than retrieval is the
binding constraint at k >= 8.

## Interpretation rule (fixed before any result is seen)
Read at top_k=8, on the CI, not the point estimate.
A. selection headroom = pool_coverage - peak_in_gold
   CI upper <= 0.05 -> anchor near-optimal given the pool; do NOT build
                       caption reranking
   CI lower >= 0.15 -> material headroom; in-pool caption reranking is the
                       highest-EV cheap experiment
   otherwise        -> INCONCLUSIVE; take no action either way
B. index_coverage < 0.95 -> ingest sampling rate structurally forecloses
   (1 - index_coverage) of questions at ANY retrieval or embedding quality;
   denser sampling is proven necessary and P3 becomes critical path.
   >= 0.95 -> do not cite this run as support for densification.
C. best_gold_rank concentrated at ranks 2-4 -> a different query-conditional
   signal plausibly recovers those questions. Concentrated at the tail of k
   -> the pool ordering itself is the problem.

## STOP condition
This run selects nothing. No frozen value, default, or threshold may change
as a result. Any implied config change is a separate pre-registered
experiment on a NEW split — the current test half is BURNED.

## Limitations
1. index_coverage measures this cache's frozen sampling rate, not CLIP.
2. Conditional on ranking_mode="ppr", never compared against "legacy".
3. VAL only; test half burned.
4. best_gold_rank ranks on the same CLIP score the anchor uses, so it bounds
   headroom reachable by a DIFFERENT signal only.
5. Unaffected by the union-vs-max-per-span IoP convention issue — these are
   membership tests, not IoP.
