# Part 3 Hyperparameter Tuning -- Selection Decisions

Selection metric: val_tune mIoP (primary), IoP@0.5 tie-break (<0.005 mIoP diff), then lower median retrieval latency, then simpler/default config. Every trial runs with cerberus_mode="none" throughout (retrieval-only evaluation -- no captioner/answerer/Cerberus involved, since mIoP/IoP depend only on retrieved frame timestamps).

## Family: retrieval_strategy

Grid: ['peak_only', 'top_k_action', 'peak_neighbors', 'hybrid']

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| peak_only | 0.2639 | 0.1903 | 0.2136 | 3.7 | 2685 |
| top_k_action | 0.2532 | 0.1832 | 0.1959 | 3.4 | 2685 |
| peak_neighbors | 0.2648 | 0.1926 | 0.2110 | 3.7 | 2685 |
| hybrid **<- selected** | 0.2775 | 0.2153 | 0.2047 | 4.0 | 2685 |

Selected **retrieval_strategy = hybrid** (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).
