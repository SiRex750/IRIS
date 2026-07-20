# Scene-sparse retrieval A/B — pre-registration (2026-07-19)
Harness: scripts/eval_grounding_arms.py, clip_in_ppr_top8 span engaged (query_embedding threaded, d0eba5a).
Grounding-only (answerer-free). N = grounded ∩ cached-in-BOTH flat+scene_sparse, --max-n (largest reachable).
Arms: flat (production PPR) vs scene_sparse/rep [PRIMARY] vs scene_sparse/t75, /t90 [SENSITIVITY]. Uniform floor.

PRIMARY: scene_sparse/rep − flat, paired per-question, video-clustered bootstrap 95% CI, on mIoP and IoP@0.5.
PREDICTION: scene_sparse/rep ≥ flat, mechanism = anchor-temporal-neighbor pull recovers short events
  (A6: CLIP misses short events, 5.8s misses vs 18.7s hits). Gain, if any, concentrated in the TEMPORAL family.
STOP/FINDING: if rep − flat CI includes 0 or is negative → retrieval STRUCTURE is not the lever; ceiling is
  representation-limited (CLIP can't localize short events regardless of graph). Flat stays, clean negative.
DISCIPLINE:
- Confirm-don't-argmax: rep is the pre-registered primary; t75/t90 are sensitivity — do NOT switch the claim
  to whichever scores highest.
- No held-out split → N in-sample; any win is a dev_200 candidate, NOT bankable.
- Defaults only (scene_shortcut_margin=TAU=0.015, top_k=8, ppr_lambda=0.5, damping=0.5) — not tuned.
- SEAM CHECK: flat now uses clip_in_ppr_top8; flat mIoP should be ~0.30–0.35 (up from the ~0.26 ppr_score
  regime) and match A6 per-question on overlapping questions. flat mIoP ~0.26 means the fix didn't engage → STOP.
