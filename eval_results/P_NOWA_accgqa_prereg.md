# Held-out Acc@GQA — pre-registration (2026-07-22)

SECOND-TOUCH DISCLOSURE: the test half (27 videos / 120 questions) was used once at 300d857 for
grounding. This is a second run on the same half. Justification: the config is FROZEN and
IDENTICAL — no selection occurs; we measure additional metrics on the same held-out set. Declared
before any result is seen. MUST be disclosed in the paper. After this run, no further test-half
measurement without a new split.

LAST TOUCH: all three arms (proposed / uniform / random) run now, because a later run for the
baselines would constitute a third touch.

FROZEN CONFIG (must match 300d857 exactly): graph_mode=flat, l2_retrieve_top_k=12,
half_width=2.2, span_peak_source=clip_in_ppr_top8, ppr_lambda=0.5, ppr_damping=0.5,
motion_similarity_mode=action_score; answerer granite4:micro via llama-server b9976, temp=0,
cache_prompt=false, --parallel 1; MC prompt + parser path (same as A6).

PRIMARY: Acc@GQA (correct AND IoP>=0.5), proposed arm.

SECONDARY: Acc@QA; P(correct|grounded) with n_grounded; proposed-vs-uniform and
proposed-vs-random Acc@QA differences.

PREDICTION: Acc@GQA ~= IoP@0.5 (0.3500) x P(correct|grounded) (~0.65 from A6, N=64) ~= 0.22,
with a wide CI at n=120. Acc@QA proposed > uniform, replicating the val/N=64 result (+0.31).

NO TUNING AFTER THIS RESULT. Whatever Acc@GQA comes back as, that is the number.
