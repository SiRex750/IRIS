# Latency A/B: flat vs scene_sparse — pre-registration
VAL only (406 Qs / 59 videos). TEST untouched. graph_mode passed explicitly per arm; NO default
change. Measurement, not selection.
SETUP: CPU only, --threads 8 (9800X3D, 8 physical cores), machine idle. 3 warm-up queries
discarded per video. Each query 5x, report MEDIAN. Index load time measured SEPARATELY and
excluded from per-query numbers.
PREDICTION: scene_sparse descend is FASTER. PPR runs on a subgraph (shortlist_width =
max(4, ceil(sqrt(num_scenes))) -> 4-6 of 16-35 scenes, plus a 30-frame anchor window, so roughly
25-50% of nodes) drawn from an already block-diagonal graph. PPR cost scales with edges, which
scale superlinearly with node count, so the saving should exceed the node-count ratio.
RISK: add_cross_scene_edges computes semantic/motion/temporal similarity per candidate pair AT
QUERY TIME. If it evaluates many pairs it can eat the PPR saving. The component timers attribute this.
SCALING (the result that matters for the CPU/edge claim): median per-query latency vs frame count
per video, per arm, with the trend. A widening scene_sparse advantage extrapolates to CCTV-length
footage; a flat or narrowing one does not.
REPORTING: shortcut and descend queries reported SEPARATELY as well as pooled — shortcut skips the
graph entirely and pooling hides the mechanism.
STOP/FINDING: if scene_sparse is slower at equal accuracy, that is a clean negative. Report it; do
NOT tune shortlist_width or tau to rescue it (no held-out set remains).
