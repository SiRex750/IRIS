# Block-diagonal grounding gate -- part (b), tuning half only

**Outcome: GATE_PASS**

Re-run of part (b) of `scripts/blockdiag_grounding_gate.py` restricted to the 59-video tuning
half. The original part (b) selected by cache membership and so included the 27 held-out
videos after that split was declared closed; this run loads and scores none of them.
The original script's functions are reused unchanged; only the question set differs.

- N questions: 406, videos: 59
- peak_in_gold rate -- fully_connected: 0.31773399014778325, block_diagonal: 0.31773399014778325
- mIoP -- fully_connected: 0.3160237077304346, block_diagonal: 0.3160237077304346
- Per-question bit-identical (retrieved order, peak, span, IoP, peak_in_gold): True (0 mismatches)

## Cache integrity

- `eval/data/nextqa/index_cache` sha256 unchanged: True
- `eval/data/nextqa/index_cache_ssparse` sha256 unchanged: True

