# P-NOW-A — held-out grounding result
Split declared at 50dc236 BEFORE measurement. VAL 59 videos / 406 Qs; TEST 27 videos / 120 Qs.

VAL SWEEP (top_k x half_width): top_k is FLAT — peak_in_gold 0.3227 / 0.3276 / 0.3251 / 0.3251
across top_k 8/12/16/24. The candidate set is NOT the constraint; residual loss is representational
(CLIP short-event blindness) = re-ingest territory.

Width is a TRADE, not a gain: at top_k=8, narrowing 2.2 -> 0.75 gives mIoP +0.014 and mIoU -0.074.
No width improves both. The pre-registered mIoU floor correctly rejected the narrow cells.

SELECTION (pre-registered rule) picked top_k=12, half_width=2.2 — but the val CIs show this is
NOISE: vs top_k=8, mIoP +0.0068 [-0.0132,+0.0264]; IoP@0.5 +0.0000 [-0.0289,+0.0264];
mIoU +0.0029 [-0.0095,+0.0145]. The "tuned" config is effectively the default. Reported as such.

HELD-OUT TEST (300d857, one run, frozen config):
  peak_in_gold 0.3667 [0.2871,0.4454] | mIoP 0.3349 [0.2486,0.4235]
  IoP@0.5 0.3500 [0.2627,0.4435] | mIoU 0.1855 [0.1402,0.2283] | IoU@0.5 0.1333 [0.0847,0.1803]

NOTE: TEST came in ABOVE VAL (0.3349 vs 0.3126). The pre-registered drop did NOT occur. Most likely
the declared bias — TEST has fewer multi-interval questions (6.7% vs 11.3%), which are structurally
uncapturable. CIs overlap heavily, so the honest read is that val and test are CONSISTENT, not that
test is better.
