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

## Family: ppr_lambda

Grid: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| 0.0 | 0.2707 | 0.2257 | 0.1785 | 3.8 | 2685 |
| 0.1 | 0.2725 | 0.2276 | 0.1834 | 3.9 | 2685 |
| 0.25 **<- selected** | 0.2777 | 0.2287 | 0.1937 | 4.1 | 2685 |
| 0.5 | 0.2775 | 0.2153 | 0.2047 | 4.0 | 2685 |
| 0.75 | 0.2794 | 0.2127 | 0.2086 | 3.8 | 2685 |
| 0.9 | 0.2794 | 0.2104 | 0.2089 | 3.8 | 2685 |
| 1.0 | 0.2796 | 0.2086 | 0.2075 | 3.8 | 2685 |

Selected **ppr_lambda = 0.25** (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).

### Family 2 supplementary analysis

**PPR blend direction (verified from code, not the config comment):**
`iris/l2_asphodel.py:retrieve_ppr` computes
`seed_raw = lambda_ * sem_rank + (1 - lambda_) * codec_rank`, so lambda=1.0
is semantic-only and lambda=0.0 is codec-only -- confirmed directly from
the arithmetic, matching the config comment and prior sessions' assumption.

**Metric disagreement (mIoU picks a different winner than mIoP/IoP@0.5):**
mIoU increases nearly monotonically with lambda (0.1785 at 0.0 -> 0.2089 at
0.9, dipping slightly to 0.2075 at 1.0), i.e. mIoU alone would favor
lambda~0.9, not 0.25. IoP@0.3 shows the same high-lambda-favoring pattern
(0.3248 at 0.0 -> 0.3322 at 1.0). IoP@0.5 shows the opposite pattern,
falling from 0.2257 at lambda=0.0 down to 0.2086 at lambda=1.0, peaking at
0.2287 at lambda=0.25 -- this is the decisive tie-break metric per the
selection rule, and it disagrees with what mIoU/IoP@0.3 alone would pick.
raw mIoP is nearly flat across the whole grid (0.2707 to 0.2796, a 0.0089
spread) and does not decisively prefer any value on its own -- it is the
IoP@0.5 tie-break that actually decides this family.

**Replication check against the prior 89-video Recall@5/Hit@5 benchmark
(different dataset, different code state, different metric):** that
benchmark found lambda=0.5 underperformed lambda=1.0 (semantic-only), i.e.
"more semantic weight wins." That direction does NOT clearly replicate
here. Raw mIoP alone shows a whisper-thin trend in the same direction
(0.2707 at lambda=0 up to 0.2796 at lambda=1, only +0.0089 / +3.3%
relative, mostly within adjacent-value noise), but the actual pre-registered
selection rule -- which exists specifically to resolve cases this close --
is decided by IoP@0.5, and IoP@0.5 moves in the OPPOSITE direction,
favoring lower lambda (more codec weight). The winning configuration on
this dataset/metric (lambda=0.25, 75% codec-weighted) is the near-opposite
of "semantic-only wins." This is reported as a genuine, replicated-in-the-
opposite-direction finding, not suppressed to match the earlier result.

**Mechanistic read:** NExT-GQA gold spans are usually short, precise
temporal windows -- the exact moment an event happens, not a general topic
region. codec_conf is a motion/visual-activity confidence signal computed
at ingest; weighting the PPR seed toward it (low lambda) biases retrieval
toward frames at genuine motion/activity peaks, producing tighter predicted
spans that more often clear the strict IoP@0.5 bar. Pure codec-only
(lambda=0.0) lacks any semantic disambiguation, so it can lock onto a
salient-but-question-irrelevant motion event -- costing some precision
relative to 0.25. Pure semantic-heavy (lambda>=0.75) retrieves frames that
are topically/textually relevant to the question but temporally more
diffuse (CLIP similarity matches objects/scene content broadly, not event
timing), which spreads the predicted span wider -- this raises mIoU
(better union-normalized overlap on average, less over/undershoot) and the
looser IoP@0.3, but lowers IoP@0.5 because a wider, less temporally-precise
span less often satisfies the strict threshold. lambda=0.25 -- mostly
codec-driven with a modest semantic nudge to pick the *right* event, not
just *an* eventful moment -- is the sweet spot for the strict, primary-tie-
break metric this task selects on.
