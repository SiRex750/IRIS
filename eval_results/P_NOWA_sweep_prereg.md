# P-NOW-A width x top_k sweep — pre-registration. VAL ONLY (406 Qs / 59 videos). TEST UNTOUCHED.
PRIMARY for top_k: peak_in_gold rate. Rationale: mIoP is hard-capped by the peak-in-gold rate
  (IoP>=0.5 implies the peak is inside gold for contiguous gold), so peak_in_gold IS the ceiling and
  mIoP is a lagging proxy.
PREDICTION (top_k): peak_in_gold rises with K then plateaus. The team's K-negative does NOT transfer
  — theirs is a min-max span-width artifact; under fixed-width spans K cannot widen the span.
PREDICTION (half_width): mIoP rises weakly as W narrows; mIoU falls as W narrows and may PEAK near
  the gold median. Val gold median is 4.90s and the current window is 4.4s, i.e. already narrower
  than typical gold — so the width lever is expected to be WEAK, benefiting mainly the 17.9% of
  questions with gold <=2.5s. A flat width result is a valid finding, not a failure.
SELECTION RULE (declared before seeing results): reference cell = (top_k=8, half_width=2.2), the
  current default. Choose the config maximizing mIoP subject to mIoU >= the reference cell's mIoU.
  We do not trade away IoU we already hold; mIoU is already below MUPA-2B's 27.2. Width is chosen by
  this constraint, NOT by unconstrained argmax.
STOP/FINDING: if peak_in_gold is flat in K, the candidate set is not the constraint and the residual
  loss is representational (CLIP short-event blindness) -> re-ingest territory. Surface it.
TEST PROTOCOL: the winning config is frozen and run ONCE on the 27 test videos. No re-tuning after.
