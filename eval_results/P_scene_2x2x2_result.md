# Scene-sparse mechanism experiment — RESULT (VAL, 406 Qs / 59 videos, top_k=8)
HONESTY NOTE: designed as 2x2x2 but executed as 2x2. The geometry_6d axis never ran (see below),
so 4 of 8 cells are duplicates. Do NOT cite this as eight independent conditions.

FINDING 1 — scene_sparse ~= flat, both span modes, no detectable effect:
  peak_anchored: peak_in_gold -0.0025 [-0.0588,+0.0477]; mIoP +0.0096 [-0.0295,+0.0460];
                 IoP@0.5 -0.0049 [-0.0601,+0.0461]; mIoU +0.0041 [-0.0222,+0.0267]
  minmax:        mIoP +0.0059 [-0.0111,+0.0212]; IoP@0.5 0.0000 [-0.0244,+0.0243]
  Valid comparison: both arms ran action_score. DESCEND fired 94.1% of queries, so the graph,
  add_cross_scene_edges and P1-11 were genuinely exercised — this is not an inert-path null.
  Prediction P1 (scene_sparse wins under minmax via temporal-clustering width artifact) FALSIFIED:
  the width signature is present in mIoU direction but far too small to matter.

FINDING 2 — span construction is the ONLY lever with CIs excluding zero:
  peak_anchored - minmax (flat): mIoP +0.0284 [+0.0022,+0.0568];
  IoP@0.5 +0.1034 [+0.0541,+0.1538]; mIoU -0.0867 [-0.1276,-0.0511].
  The mIoU loss is significant and must be reported as a TRADE, not a clean win.

FINDING 3 — geometry_6d UNTESTED (not a null). Two independent wiring bugs, both verified:
  (a) ingest.py::_build_graph (~117-142) bundles the 5 geometry values only inside
      refined_motion_tensor, never as top-level dict keys, so add_frame_node's
      get_val(..., "divergence", 0.0) always takes the 0.0 default. .npz/FrameRecord carry
      real nonzero values; the drop is at dict construction.
  (b) motion_similarity_mode is an instance attribute fixed at graph-build time from
      config_snapshot (frozen at ingest, predates the flag), so a loaded index's graph is
      always "action_score" regardless of query-time cfg.
  Branch counters: entered=0, fallback=0 — never entered.
  DECISION: not fixing. Low EV (pool composition shown irrelevant: peak_in_gold flat 0.3227->0.3251
  across top_k 8->24) and the fix touches _build_graph, the measured path behind A6/P1/P-NOW-A.
  Refer the geometry axis to the team, where it is natively wired and runs at N=2,685.

SILENT-FALLBACK PATTERN (4th instance): eval_grounding_arms ppr_score spans; CLIP zero-vector
anchor fallback; teammate's Method C scene_id -> Method A (94.38%); geometry_6d never entered.
STANDING RULE: any path with a fallback branch must assert its fallback rate == 0 and fail loudly.
