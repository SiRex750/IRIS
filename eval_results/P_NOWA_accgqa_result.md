# Held-out Acc@GQA — RESULT (dc59de3, second and final test-half touch)
SEAM CHECK PASSED: grounding reproduced EXACTLY across both test touches (mIoP 0.3349,
IoP@0.5 0.3500) — both runs measured the identical system and compose into one held-out result.

PRIMARY: Acc@GQA = 0.1667 [0.088, 0.243], n=120 / 27 videos.
SECONDARY: Acc@QA 0.375 [0.275,0.477]; P(correct|grounded) 0.476 (n_grounded=42);
P(correct|ungrounded) 0.321 (25/78). Parser 120/120 clean_leading, zero failures.
Decomposition: 42/120 grounded, 45/120 correct, 20/120 both. 0.350 x 0.476 = 0.167.

CORRECTION 1 — THE ANSWERER IS NOT NEGLIGIBLE. Held-out P(correct|grounded)=0.476 vs the in-sample
0.65 (A6, N=64). It sits at the FLOOR of A6's CI [0.45,0.84], so A6 is not contradicted, but the
conclusion drawn from it does not survive: perfect grounding -> Acc@GQA 0.476 (+0.31 headroom);
perfect answerer -> 0.350 (+0.18). Grounding still has more room, but P2 (answerer diagnostic)
moves from CONFIRMED-DEFERRED back to LIVE. Constraint: no test half remains, so answerer work
cannot be validated held-out without a new split.

CORRECTION 2 — THE PROPOSED-VS-UNIFORM Acc@QA CLAIM IS DOWNGRADED. Held out: +0.120 [0.000,0.248]
(CI touches zero) vs the in-sample +0.31 [+0.16,+0.45]. It survives against random: +0.127
[+0.036,+0.229]. Honest form: "retrieval beats RANDOM frame sampling on answer accuracy; against
UNIFORM sampling the advantage is positive but not statistically separated at n=120."

REPLICATED: the grounded-vs-ungrounded correctness gap held at ~15pt in both samples (65 vs 46
in-sample; 47.6 vs 32.1 held out). Levels dropped, the gap held — supports the faithfulness story
independently of absolute levels.

PLACEMENT (indicative only — ours is n=120 from a val-derived split with a ~8pt CI; published
figures are full-test): 0.1667 is BELOW MUPA-2B 0.287 with the CI excluding it, and in the same
band as SeViLA 0.166 / LangRepo 0.171 / FrozenBiLM+NG+ 0.175, above FrozenBiLM 0.158 and
Temp[CLIP] 0.147 — while running a sub-2B answerer on CPU.

TEST HALF IS NOW BURNED. No further test-half measurement without a new split.
