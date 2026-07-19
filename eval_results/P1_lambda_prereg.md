# P1 / Fix 2 — ppr_lambda sweep pre-registration (2026-07-19)
Seed (verified l2_asphodel.py:736): lambda*sem_rank + (1-lambda)*codec_rank.
lambda=1.0 = pure semantic (codec off); lambda=0.0 = pure codec.
CORRECTION: roadmap v9 pre-registered lambda*=0; that is BACKWARDS. Mechanism
(codec-at-query-time negative -> downweight codec -> raise lambda) predicts lambda*=1.0.
Prediction: proposed mIoP, IoP@0.5, Acc@QA ordered lambda=1.0 >= 0.5 >= 0.0;
lambda=1.0 - lambda=0.0 CI-separated >0 on at least Acc@QA.
Attenuation caveat (valid outcome): grounding may be FLAT (clip_in_ppr_top8 rescues
the peak regardless of seed); flat grounding + monotone Acc@QA is coherent.
STOP conditions: if lambda=0.0 does not underperform lambda=1.0, codec-negative fails
at production level -> surface, do not tune. Non-monotone/lambda*!=1.0 -> investigate.
Confirm-don't-argmax: no held-out split; test endpoints {0.0, 1.0}, anchor to mechanism.
Anchor: lambda=0.5 already measured (A6_mixed_raw.json). Damping fixed at 0.5.
