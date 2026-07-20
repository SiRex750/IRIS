# NExT-GQA Metric Definitions

See `metric_registry.json` for the machine-readable registry and `scripts/nextgqa_metrics.py` for
implementations; `scripts/test_nextgqa_metrics_synthetic.py` / `synthetic_metric_test_report.txt`
for the 27 handwritten synthetic checks (exact match, no overlap, partial overlap, pred-inside-gold,
gold-inside-pred, boundary, equality at 0.3/0.5, multi-gold-span max, empty prediction, reversed span,
zero-duration span, out-of-bounds span, frame-to-seconds conversion).

IoP = intersection(pred, gold) / duration(pred). Zero-duration prediction -> IoP = 0.0.

IoU = intersection(pred, gold) / union(pred, gold). Zero-measure union -> IoU = 0.0.

Acc@GQA = (predicted MC answer == gold answer) AND (IoP >= 0.5).

Multiple gold spans: score the predicted span against **each** gold span independently and take the
**maximum**. Gold spans are never concatenated/unioned into a single interval before scoring (that
would inflate IoP/IoU for predictions that straddle two disjoint gold intervals).

## Diagnostic vs. official

`Temporal Hit@1/5/10`, `MRR`, `Peak-in-gold`, `Candidate-contains-gold`, `Candidate-to-peak loss`,
`Survivor coverage ceiling`, `Window coverage` are retrieval diagnostics over discrete frame
timestamps. None of them is an official NExT-GQA metric. In particular, **Temporal Hit@K must never
be reported as, or confused with, an official "Recall@K"** — NExT-GQA's official protocol scores
continuous IoU/IoP over spans, not discrete top-K hit/miss over frames.

## Known discrepancy with existing repo code

`eval/grounding_scorer.py::iop()` already exists and computes IoP against the **union** of all gold
spans collapsed into a single predicted-span comparison, rather than the official max-over-each-span
protocol. This setup task does not modify that production file (out of scope), but the discrepancy is
recorded here and in `metric_registry.json` as something that must be reconciled — pick one contract
and delete the other — before any metric number is reported as "official NExT-GQA IoP".
