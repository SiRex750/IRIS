# A6 — Pinned Grounded-VideoQA Measurement (2026-07-19)

Runs (both git_dirty=false, granite4:micro @ llama-server b9976, temp=0, cache_prompt=false, --parallel 1, N=64 in-sample, 59 videos, num_boot=1000):
- Mixed (production):   span_mode=ppr_peak, peak_source=clip_in_ppr_top8, half_width=2.2 @ c4dd497 -> A6_mixed_raw.json (committed bcf940e)
- All-minmax (fork):    span_mode=minmax @ bcf940e -> A6_allminmax_raw.json (committed bcf940e)

## Determinism / provenance
- Answerer determinism REPLICATED across a fresh run: proposed Acc@QA = 0.53125 byte-identical in both runs (temp=0, cache_prompt=false, single slot).
- All-minmax reproduced expected span-independent values exactly (uniform iop 0.25532, acc_qa 0.21875; random iop 0.25632, acc_qa 0.29688) and proposed-minmax iop 0.27440 (cross-checks the half_width report). No seam.

## Primary metrics (proposed, mixed)
- mIoP 0.3426 | IoP@0.5 35.94% (23/64) | mIoU 0.1790 | IoU@0.5 10.94% | Acc@QA 53.13% (34/64) | Acc@GQA 23.44% (15/64)
- Parser: 64/64 clean_leading, zero fallbacks/failures.

## P(correct | grounded)  [replaces the quarantined 36% placeholder]
- 15/23 = 65.2%, video-clustered bootstrap 95% CI [0.450, 0.840] (n_grounded=23 over 59 videos).
- The answerer is competent when handed correct evidence -> NOT the binding constraint at ~36% grounding. P2 (answerer diagnostic) stays deferred.
- Acc@GQA measured 23.4% vs ~13% projected; projection was pessimistic (used the stale 36%).

## Grounding gap (faithfulness signal, feeds P6)
- 34 correct total; 15 correct AND grounded; 19 correct but UNGROUNDED.
- P(correct | ungrounded) = 19/41 = 46.3% (>> 20% MC chance) -> language-prior shortcut answering.

## Baseline-span comparability fork — RESOLVED
proposed - uniform:
| metric  | mixed diff [95% CI]            | all-minmax diff [95% CI]        | survives? |
|---------|-------------------------------|---------------------------------|-----------|
| mIoP    | +0.087 [+0.008, +0.168]       | +0.019 [-0.021, +0.063]         | NO        |
| IoP@0.5 | +0.203 [+0.091, +0.313]       | +0.031 [-0.031, +0.097]         | NO        |
- Decomposition of mixed +0.087: span construction (clip-peak vs minmax) +0.068; retrieval frame selection +0.019 (n.s.).
- Conclusion: DO NOT claim IRIS retrieval localizes better than uniform. Grounding gains are carried by clip-in-PPR-top8 peak selection (span construction), not frame selection.

## Span-independent retrieval win (Acc@QA; span mode irrelevant to the answer)
- proposed - uniform Acc@QA +0.3125 [+0.162, +0.452]; proposed - random Acc@QA +0.2344 [+0.094, +0.377].
- Clean, unconfounded evidence that IRIS retrieval delivers better answerer evidence than uniform/random sampling.
