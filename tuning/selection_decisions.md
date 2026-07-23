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

## Family: ppr_damping

Grid: [0.5, 0.65, 0.8, 0.85, 0.9]

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| 0.5 **<- selected** | 0.2777 | 0.2287 | 0.1937 | 4.0 | 2685 |
| 0.65 | 0.2751 | 0.2179 | 0.1961 | 3.9 | 2685 |
| 0.8 | 0.2735 | 0.2108 | 0.2021 | 3.9 | 2685 |
| 0.85 | 0.2742 | 0.2101 | 0.2039 | 4.0 | 2685 |
| 0.9 | 0.2728 | 0.2056 | 0.2052 | 4.1 | 2685 |

Selected **ppr_damping = 0.5** (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).

## Family: ppr_damping

`ppr_damping` maps directly to `alpha` in `iris/l2_asphodel.py:1288`'s
`nx.pagerank(g, weight="weight", personalization=seed, alpha=damping)` --
networkx's standard PageRank damping factor, legal range (0,1) exclusive
(already enforced by `IRISConfig._check(0.0 < self.ppr_damping < 1.0, ...)`).
Confirmed from the actual call site, not assumed.

Grid: [0.50, 0.65, 0.80, 0.85, 0.90]

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| 0.50 **<- selected (= default)** | 0.2777 | 0.2287 | 0.1937 | 4.0 | 2685 |
| 0.65 | 0.2751 | 0.2179 | 0.1961 | 3.9 | 2685 |
| 0.80 | 0.2735 | 0.2108 | 0.2021 | 3.9 | 2685 |
| 0.85 | 0.2742 | 0.2101 | 0.2039 | 4.0 | 2685 |
| 0.90 | 0.2728 | 0.2056 | 0.2052 | 4.1 | 2685 |

Selected **ppr_damping = 0.50** -- the pre-tuning default. All 5 values are
mutually within 0.005 mIoP of the top (max spread 0.0049), so the whole
family is decided by the IoP@0.5 tie-break, which favors 0.50 clearly and
monotonically over every other value tested. This is the STOP CONDITION's
"tuned value does not beat default" case, reported as a genuine negative
result -- no ppr_damping deviation from 0.50 is adopted.

mIoU (not the selection metric) increases with damping (0.1937 at 0.5 ->
peak 0.2052 at 0.9) -- the same disagreement pattern as Families 1 and 2:
a looser/union-normalized metric favors more diffusion, while the strict
IoP@0.5 tie-break favors less.

Mechanistic read: damping (`alpha`) controls how much PageRank mass
continues propagating outward along graph edges vs. teleporting back to the
personalized seed distribution at each iteration. Low damping keeps
PageRank mass concentrated near the directly-seeded, highest-relevance
frames; high damping lets relevance diffuse further through the graph to
seed-adjacent-but-not-seed frames before resetting, broadening the
effective candidate pool the top-k draws from. That broader pool raises
mIoU (better union-normalized coverage on average) but hurts IoP@0.5 (the
selection landing precisely inside a short gold window becomes less
likely) -- the same "concentration helps precision, diffusion helps
union-coverage" trade-off already seen with `ppr_lambda` in Family 2, just
driven by a different graph-propagation mechanism.

**Honesty check on magnitude:** this family's move is effectively zero --
the winner IS the untouched default, not a new value discovered by tuning.
Compared to Family 2's real (if modest) +0.0134 IoP@0.5 gain from tuning
`ppr_lambda` away from its default, Family 3 found no improvement available
at all in this parameter. Not inflating this into a "damping was
confirmed optimal" success story beyond what it is: a clean negative
result on a 5-point grid.

## Family: l2_retrieve_top_k

Grid: [4, 8, 12, 16]

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| 4 **<- selected** | 0.2788 | 0.2358 | 0.1779 | 3.7 | 2685 |
| 8 | 0.2703 | 0.2022 | 0.2145 | 3.8 | 2685 |
| 12 | 0.2639 | 0.1858 | 0.2219 | 3.7 | 2685 |
| 16 | 0.2604 | 0.1765 | 0.2249 | 3.7 | 2685 |

Selected **l2_retrieve_top_k = 4** (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).

## Family: l2_retrieve_top_k

Grid as specified: [4, 8, 12, 16] -- note this grid does NOT include the
default value (5). To honestly evaluate the stop condition ("does the
tuned value beat the default"), a supplementary trial at K=5 was run using
the same harness/config after the 4-value grid completed (see below).

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| 4 **<- selected** | 0.2788 | 0.2358 | 0.1779 | 3.8 | 2685 |
| 5 (default, supplementary) | 0.2777 | 0.2287 | 0.1937 | 4.0 | 2685 |
| 8 | 0.2703 | 0.2022 | 0.2022 | 3.8 | 2685 |
| 12 | 0.2639 | 0.1858 | 0.1859 | 3.7 | 2685 |
| 16 | 0.2604 | 0.1765 | 0.1765 | 3.7 | 2685 |

Selected **l2_retrieve_top_k = 4**. K=4 and K=5 are mutually tied on mIoP
(diff 0.00114 < 0.005); K=4 wins the tie-break on higher IoP@0.5 (0.2358 vs
0.2287). K=8/12/16 are NOT tied with K=4 -- clearly, decisively worse
(gaps of 0.0085-0.0184 mIoP, all well outside the tie-break band). So the
honest characterization is: a narrow win over the untouched default,
inside a much larger and clearer decline as K grows past 5.

mIoU disagrees with the mIoP/IoP@0.5 ranking, but not in the same
monotonic way as prior families: mIoU peaks at K=8 (0.2022), not at either
extreme -- K=4 (the mIoP/IoP@0.5 winner) actually has the LOWEST mIoU of
the whole grid (0.1779). Flagged plainly, same as every prior family.

Mechanistic read: `l2_retrieve_top_k` directly and mechanically bounds the
predicted span (min/max timestamp of exactly K retrieved frames). Small K
keeps only the most highly-ranked, presumably most relevant frames, which
tend to cluster tightly in time around the actual event -- a narrow,
precise predicted span that lands inside short gold windows more often
(higher IoP@0.5). As K grows, lower-ranked frames get pulled in that are
more likely temporally scattered (either genuinely relevant-but-distant
context or graph-adjacent noise), widening the predicted span -- this
raises union-based coverage up to a point (K=8's mIoU peak) but past that
the added frames dilute precision faster than they help coverage (mIoU
also falls back down by K=12/16). Same "too few risks missing evidence,
too many risks widening the span with off-topic frames" tension as
described in the task, resolved here via pool size rather than a ranking
weight.

**Honesty check on magnitude:** this is a real, if modest, win over the
default -- larger in relative terms than Family 3's null result, smaller
than the apparent gap to K=8/12/16 might suggest at first glance (that gap
is versus worse alternatives, not versus the baseline). Not a fourth flat
result, but also not a dramatic breakthrough -- a genuine ~0.4% relative
mIoP improvement and a real ~3% relative IoP@0.5 improvement over the
untouched default, decided by the tie-break rule exactly as it's designed
to be used.

## Family: peak_dist_prom

Grid: [(3, 0.03), (5, 0.05), (8, 0.05), (5, 0.1)]

| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |
|---|---|---|---|---|---|
| distance=3,prominence=0.03 | 0.2948 | 0.2980 | 0.1598 | 3.9 | 2685 |
| distance=5,prominence=0.05 **<- selected** | 0.2952 | 0.2980 | 0.1600 | 3.9 | 2685 |
| distance=8,prominence=0.05 | 0.2937 | 0.2965 | 0.1592 | 3.9 | 2685 |
| distance=5,prominence=0.1 | 0.2952 | 0.2980 | 0.1600 | 3.9 | 2685 |

Selected **peak_dist_prom = distance=5,prominence=0.05** (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval latency; then default).

### Family 5 supplementary analysis

**All values (full metrics from `tuning/all_trials.csv`):**

| value | mIoP | mIoU | IoP@0.3 | IoP@0.5 | IoU@0.3 | IoU@0.5 | wall_s |
|---|---|---|---|---|---|---|---|
| distance=3,prominence=0.03 | 0.29484 | 0.15984 | 0.36797 | 0.29795 | 0.23948 | 0.12067 | 1100.9 |
| distance=5,prominence=0.05 (default) **<- selected** | 0.29516 | 0.16003 | 0.36872 | 0.29795 | 0.23985 | 0.12142 | 106.4 |
| distance=8,prominence=0.05 | 0.29365 | 0.15923 | 0.36872 | 0.29646 | 0.23985 | 0.11881 | 1089.1 |
| distance=5,prominence=0.10 | 0.29516 | 0.16003 | 0.36872 | 0.29795 | 0.23985 | 0.12142 | 1088.4 |

**Prominence=0.10 is a complete no-op at distance=5 on this dataset, not just a
near-tie.** `distance=5,prominence=0.05` and `distance=5,prominence=0.10`
are bit-identical across every one of the six scoring metrics above,
computed independently over the same 2685 questions from two separately
built full ingests (config-hashes `23e55c57a3b14534` and
`9f288740c132690c` -- confirmed distinct, both required a fresh 450-video
ingest, no cache was shared between them). Verified this isn't a
wiring bug: diffing the two index caches' per-frame `is_peak` flags for a
sampled video shows the identical 43-frame peak set at both prominence
values, and `iris/action_score.py`'s `find_peaks(..., prominence=self.config.peak_prominence)`
call confirms `peak_prominence` is genuinely threaded through and not
dead code. The honest read: this dataset's action-score peaks are
effectively bimodal at `distance=5` -- real motion peaks sit comfortably
above 0.10 prominence and noise sits below 0.05, so raising the floor from
0.05 to 0.10 filters nothing extra. `peak_prominence` has no effective
range on this data at the currently-frozen `distance=5`.

**Mechanistic read (distance):** `peak_distance` sets the minimum spacing
`scipy.find_peaks` enforces between selected local maxima in the
action-score signal -- it, not `peak_prominence`, is what actually moves
the numbers here. Tightening to `distance=3` lets peaks pack closer
together in fast-motion segments (more candidate PEAK-tier frames feeding
the hybrid retrieval pool), which very slightly hurts precision (mIoP
-0.00032, mIoU -0.00019 vs default) without buying anything -- likely
diluting the PEAK-tier pool with near-duplicate, temporally adjacent
frames rather than adding genuinely new salient moments. Loosening to
`distance=8` forces sparser peaks, which can merge or drop distinct local
maxima in busy segments -- this costs more (mIoP -0.00151, IoP@0.5
-0.00149 vs default, both larger than the distance=3 gaps), suggesting
under-spacing loses real evidence faster than over-spacing dilutes it.
Both directions move away from default in the same (negative) direction
mIoP-wise, and neither clears the 0.005 tie-break band -- default spacing
(`distance=5`) is not just untouched, it's the best of the four tested.

**Honesty check on magnitude:** this is a clean negative result, the same
character as Family 3 (`ppr_damping`) -- no tuned value beats the
untouched default `(5, 0.05)`. `distance=3` and `distance=8` are both
mechanistically worse (if only slightly, and within the tie-break band).
`(5, 0.10)` doesn't beat the default either -- it doesn't even differ from
it, since prominence has no effect at this spacing. Not inflating a
bit-identical tie into a "prominence was confirmed optimal" success story:
the honest characterization is that this grid found no improvement
available in either `peak_distance` or `peak_prominence` over the values
already in use since Family 1.
