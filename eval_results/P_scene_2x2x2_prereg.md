# Scene-sparse mechanism experiment — pre-registration (VAL only, 406 Qs / 59 videos)
PURPOSE: explain the teammate's reported scene_sparse > flat result by decomposing it into
its possible causes. This is a MECHANISM experiment, NOT config selection — the held-out test
half is BURNED (used once at 50dc236/300d857), so nothing here can be adopted as tuned.

DESIGN: 2 x 2 x 2, all cells on the same 406 val questions, top_k=8 fixed.
  graph_mode           : flat | scene_sparse
  motion_similarity_mode: action_score | geometry_6d
  span                 : minmax | peak_anchored (half_width=2.2, clip_in_ppr_top8)
Anchor cell (already measured, must reproduce): flat / action_score / peak_anchored
  = mIoP 0.3126, IoP@0.5 0.3202.

PREDICTIONS:
P1. Under MINMAX spans, scene_sparse > flat. Mechanism: scenes are temporally contiguous, so
    scene-restricted retrieval yields temporally clustered frames -> narrower min-max envelope
    -> mechanically higher IoP. Same artifact already identified for lambda=0.25 and for the
    team's top_k effect.
P2. Under PEAK_ANCHORED spans, scene_sparse ~= flat (CI crosses zero). Width is constant, so
    the artifact cannot express itself; and the K-sweep showed the candidate set is not the
    constraint (peak_in_gold flat 0.3227->0.3251 across top_k 8->24).
P3. geometry_6d: no strong prior. It changes graph edge weights, hence pool membership, but
    peak selection is CLIP-based within the pool. Expect small effects on peak_in_gold.
IF P1 AND P2 BOTH HOLD: the teammate's scene_sparse advantage is a span-width artifact, and
  the unifying finding is that every apparent retrieval win under min-max is span-width driven.
IF scene_sparse > flat UNDER PEAK_ANCHORED: scene structure is a real retrieval lever — the
  first one found in this project. Report as a genuine positive; do NOT tune it (no clean
  test set remains).

PRIMARY: peak_in_gold (the ceiling quantity). SECONDARY: mIoP, IoP@0.5, mIoU, IoU@0.5.
All paired, video-clustered bootstrap CIs (1000 resamples).

DECLARED CONFOUND: P1-11 edge_type label merging is reachable ONLY via the scene_sparse descend
path, so it cannot be separated from graph_mode in this design. If scene_sparse wins under
peak_anchored, P1-11 remains an unexcluded alternative cause.
