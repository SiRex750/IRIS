# top_k x half_width sweep - pre-registration
Grid: top_k {8, 12, 16, 24} x half_width {2.2, 1.5, 1.0, 0.75}. Grounding-only (CLIP, no answerer).
Report per cell: peak_in_gold rate, mIoP, IoP@0.5, mIoU, IoU@0.5.

PRIMARY for top_k: peak_in_gold rate. Rationale: mIoP is hard-capped by the peak-in-gold rate (proof: IoP>=0.5 implies peak in gold for contiguous gold), so peak_in_gold is the ceiling and mIoP is a lagging proxy.

PREDICTION (top_k): peak_in_gold rises with K then plateaus. The team's K-negative does NOT transfer - it is a min-max span-width artifact; under fixed-width spans K cannot widen the span.

PREDICTION (half_width): mIoP rises monotonically as W narrows, asymptoting at the peak-in-gold rate and NEVER exceeding it; mIoU falls monotonically. There is NO interior optimum for IoP.

SELECTION RULE (declared before seeing results, to avoid the Method B trap): pick the NARROWEST half_width whose mIoU stays >= 0.179 (the current measured value). We do NOT trade away IoU we already hold; mIoU is already below MUPA-2B's 27.2. Width is chosen by this floor, NOT by argmax.

STOP/FINDING: if peak_in_gold is flat in K, the candidate set is not the constraint and the remaining loss is representational (CLIP short-event blindness) -> re-ingest territory, surface it.

CAVEAT: N=64 in-sample, no held-out split. Any width chosen here is PROVISIONAL and must be re-derived on the P-NOW-A val split before it is claimed.
