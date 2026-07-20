# P1 / Fix 2 — ppr_lambda sweep RESULT (2026-07-19)
Runs: λ=0.0 P1_lambda00_raw.json (c9290ab); λ=0.5 A6_mixed_raw.json (c4dd497); λ=1.0 P1_lambda10_raw.json (2766b00). Baseline (uniform/random) invariance confirmed identical to A6 across all λ.

## Sweep (proposed arm)
| λ   | mIoP  | IoP@0.5 | Acc@QA | Acc@GQA |
|-----|-------|---------|--------|---------|
| 0.0 | 0.222 | 0.203   | 0.547  | 0.156   |
| 0.5 | 0.343 | 0.359   | 0.531  | 0.234   |
| 1.0 | 0.330 | 0.313   | 0.516  | 0.203   |

## Paired diffs (per-question, video-clustered 95% CI)
- λ=0.5 − λ=0.0: mIoP +0.121 [+0.050,+0.196]*, IoP@0.5 +0.156 [+0.061,+0.254]*
- λ=1.0 − λ=0.0: mIoP +0.108 [+0.023,+0.195]*, IoP@0.5 +0.109 [−0.015,+0.226]
- λ=0.5 − λ=1.0: mIoP +0.013 [−0.053,+0.080], IoP@0.5 +0.047 [−0.045,+0.143]  (n.s. — tied)
- Acc@QA: flat across all λ (all CIs span 0).

## Verdict
- Pre-registered λ*=1.0 FALSIFIED. λ=0.5 (default) and λ=1.0 are statistically indistinguishable; λ=0.0 (pure codec) is significantly worse on grounding only.
- Shape: plateau over λ∈[0.5,1.0], collapse at λ→0. NOT monotone.
- λ is a CLOSED lever — default 0.5 is on the optimal plateau; no improvement available. Fix 2 = no gain. Keep λ=0.5.
- Codec effect REFINED: a domination effect (over-weighted query-blind codec_conf seed evicts good frames from the top-8 that clip-peak can't recover), not a per-signal negative. Balanced codec is benign. Consistent with the peak-source fix carrying localization.
- Discipline: confirm-don't-argmax honored; N=64 in-sample, no held-out split; λ not tuned.
