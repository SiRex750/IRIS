# Scene-sparse vs flat A/B — pre-registration (2026-07-22)
Harness: scripts/eval_grounding_arms.py --max-n at 76660ca (add_cross_scene_edges +
retrieve_ppr graph_override restored; flat path verified byte-identical).
N = 64 (dev_100 grounded ∩ both caches), C:36 / T:28. Grounding-only, CLIP, no answerer.
Arms: flat | scene_sparse/rep [PRIMARY] | /t75, /t90 [SENSITIVITY]. Uniform floor.

PRIMARY: scene_sparse/rep − flat, paired per-question, video-clustered bootstrap CI,
on mIoP, IoP@0.5, and peak_in_gold.
PREDICTION: rep >= flat, mechanism = anchor-temporal-neighbor pull recovering SHORT events.
  Gains, if any, CONCENTRATED IN T-FAMILY (the 28 temporal questions). A flat/negative
  T-family result falsifies the mechanism regardless of the aggregate.
KNOWN DEGENERACY: crossscene_mode/pctile are read ONLY on the descend branch, so rep/t75/t90
  are identical computations whenever shortcut fires. Report shortcut/descend rate per arm;
  interpret the three modes only on descend-branch questions.
CONFIRM-DON'T-ARGMAX: rep is primary; do NOT switch the claim to whichever mode scores highest.
STOP/FINDING: if rep − flat CI includes 0 or is negative (aggregate AND T-family), retrieval
  STRUCTURE is not the lever; the ceiling is representational (CLIP short-event blindness).
CAVEAT: N=64 in-sample, no held-out split. Result is indicative, not bankable.
