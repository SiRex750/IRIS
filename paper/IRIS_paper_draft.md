# Structural Construction of Retrieval Graphs for Long-Video Question Answering

*Working draft — assembled 2026-09-03. Ledger tags `[C…]` are for co-author verification and are removed before submission. Open items are collected in Appendix C rather than left inline.*

---

## Abstract

Graph-structured pipelines for long-video question answering increasingly build the graph
itself with a large multimodal model, at a per-chunk cost that scales with video duration.
That this cost is query-independent and amortises across questions has been argued and
measured before; that compressed-domain signals can stand in for full decode is older
still. Our contribution is narrower: we construct the retrieval graph *structurally*, with
no generative model anywhere in the construction path, and we verify the construction
rather than benchmarking it.

Our central result is an identity rather than an improvement. A block-diagonal construction
that enumerates only within-scene node pairs is **bit-identical** to the conventional
dense-then-prune construction — identical nodes and edges, identical edge weights at zero
tolerance, and bit-identical PageRank across seeds — while visiting only the 0.2% of node
pairs it retains rather than the 99.8% that pruning discards. This makes construction
~233× faster (50.20 s → 0.215 s) and ~6.8× smaller in peak memory at a graph of 4,892
nodes, measured under an interleaved protocol over 159 builds [C2.1, C2.2] — on the
complete-block edge configuration, which is not the sparser tiered configuration our
accuracy numbers use. The same block structure separates query-latency scaling: over a
31-clip corpus under real text queries, dense graphs scale with graph size at exponent
2.06 (95% CI [2.02, 2.13]) and scene-sparse graphs at 0.83 ([0.80, 0.88]) — and past a
certain size the separation becomes a tractability boundary, where dense *retrieval* fails
to return within a 3600 s budget on graphs that the sparse path answers in hundredths of a
second [C1.1, C1.3].

One of our results is that an efficiency win can fail to matter, and we report it as a
finding rather than a caveat. The retrieval speedup does not reach the user at the sizes we
measure: a captioning stage that is O(top_k) rather than O(N) accounts for 85–93% of query
time and dominates the total, so end-to-end latency does not improve despite a
retrieval-mechanics speedup of three orders of magnitude, and the crossover into a material
advantage sits near 12,000 nodes. Answer quality sits in the weakly-supervised band — Acc@GQA 0.167 on a held-out validation split, with a ~3.4B answerer on CPU — rather than at the agentic state of the art.
We report both, and we pre-registered kill criteria for two hypotheses we expected to
confirm: that codec-derived scene boundaries beat content-blind ones, and that codec-based
frame admission beats uniform sampling. Both criteria triggered. Neither where the video is
cut nor which frames are kept is doing the work — the block structure is — which is what
makes the construction cheap, verifiable, and portable across segmentation policies.

---

## Alternate short version (for a venue with a 150-word cap)

We build long-video retrieval graphs structurally — from compressed-domain signals and CLIP
embeddings, on CPU, with no neural forward passes at ingest and no frontier model — rather
than prompting a proprietary MLLM per chunk. A block-diagonal construction that enumerates
only within-scene pairs is bit-identical to dense-then-prune construction (same nodes,
edges, weights at zero tolerance, same PageRank) while visiting 0.2% of the pairs, giving
~233× faster and ~6.8× smaller builds on that configuration. Query latency separates as
graph size grows —
dense 2.06 (95% CI [2.02, 2.13]) versus sparse 0.83 ([0.80, 0.88]) — becoming a
tractability boundary where dense retrieval does not return. We are explicit about the
limits: a captioning stage dominates end-to-end time, so the retrieval speedup is not
user-facing, and accuracy sits in the weakly-supervised band. Two pre-registered negatives
show the block structure, not the codec signal, carries the result.

---

---

## 1. Introduction

Answering questions about long video requires structure. A single hour of footage at
modest sampling rates yields tens of thousands of candidate frames, and flat retrieval
over that pool becomes intractable well before it becomes inaccurate. The now-standard
response is to build an intermediate representation — a graph over frames, shots, or
scenes — and retrieve against it.

The prevailing way to build that graph is to ask a large multimodal model to write it.
EgoSG prompts a proprietary model once per fixed-duration chunk to emit a symbolic scene
description and reasons over the resulting text; Vgent extracts entities per clip with an
LVLM and links clips that share them. Both work, and both produce representations richer
than ours. But both make graph construction a model-metered operation whose cost scales
with video duration, and the symbolic line additionally places a proprietary dependency at
the base of the pipeline that its own authors report does not degrade gracefully to
open-source substitutes. For a surveillance archive, a personal video library, or any
setting where footage cannot leave the premises, that is a poor foundation.

**We should be precise about what is new here, because two of the obvious claims are not
ours to make.** That construction cost is query-independent and amortises across the
questions asked of a video is Vgent's argument, and Vgent measured the offline/online split
before we did. That compressed-domain signals can substitute for full decode is older still,
running from motion-vector video analysis in the 2010s to codec-primitive video language
models this year (§2.5). Training-free retrieval over long video is a crowded field.

What we contribute is narrower and, we think, still worth reporting. We build the graph with
no generative model anywhere in the construction path — an image encoder over admitted
frames is the only network involved. We prove, rather than assert, that a block-diagonal
construction yields the bit-identical graph a dense-then-prune construction produces, and
measure what that saves. We fit query-latency scaling exponents with confidence intervals
and locate a tractability boundary where the dense arm stops returning at all. And we report
two pre-registered hypotheses that failed and one efficiency result that does not survive a
full pipeline — which together constrain the claim more usefully than another benchmark row
would.

**For retrieval, the graph does not need to be semantically parsed. It needs to be
structurally sound and cheap.** We build a scene-sparse graph over surviving frames using
compressed-domain signals and CLIP embeddings, on CPU, with no neural forward passes during
frame selection [C5.1] and no proprietary model anywhere in the construction path.

### The construction result

Sparsifying a graph invites an obvious objection: what was lost? Our answer is nothing,
and we mean it literally. The scene-sparse graph contains only within-scene edges, so the
conventional construction — materialise the dense graph, then prune across scenes — spends
almost all of its work on edges it will discard. At N=4,892 survivors over 528 scenes, it
visits 11.96 million candidate pairs to retain 23,571: **99.8% of the pairs it examines
are thrown away.** Constructing the diagonal blocks directly visits exactly the pairs it
keeps.

The resulting graph is not an approximation of the pruned graph; it is the same graph. We
verify this with an identity gate rather than a similarity metric: identical node sets,
identical edge sets, zero mismatches on every edge attribute at tolerance 0.0, bit-identical
PageRank across all 4,892 nodes, and identical personalised-PageRank ordering and scores
across five seeds [C2.1]. A downstream gate over 526 grounded-QA questions produces zero
behavioural differences [C2.3]. Construction costs 0.215 s instead of 50.20 s (~233×) and
0.93 GB instead of 6.33 GB (~6.8×) [C2.2].

### The scaling result

The same block structure changes how query cost grows. Fitting latency against graph size
over 31 clips under real text queries, dense graphs scale with exponent 2.063 (95% CI
[2.020, 2.125]) while scene-sparse graphs scale at 0.834 ([0.801, 0.884]) — disjoint
intervals, with the sparse bound below linear [C1.1, C1.5].

Past a certain size this stops being a matter of speed. Under a uniform 3600 s and 29 GB
watchdog, dense retrieval failed to return at all on clips of 6,559 and 13,506 survivors,
with memory still climbing at the cap, while sparse retrieval completed the same clips in
0.0104 s and 0.0210 s [C1.3]. We report this as a tractability boundary rather than a
speedup ratio, because a censored measurement admits no ratio.

### Where the speedup goes, and why that is a result

We state the following before the evidence rather than in a closing limitations paragraph,
because it is part of the contribution rather than a concession attached to it.

**A retrieval speedup of three orders of magnitude does not reach the user.** Measuring the
full query path at N=4,892, the end-to-end ratio is 0.802× — the sparse arm is *slower* at
the median. Retrieval mechanics shrink from 8.535 s to 0.030 s, but a captioning and
verification stage costing 44–67 s dominates the total and is O(top_k) rather than O(N), so
it does not shrink with the graph. Sparsifying the graph makes a small term negligible
while leaving the largest term untouched.

This is an Amdahl result, and we report it as one. A great deal of recent work reports
retrieval-side speedups for long-video systems without measuring the pipeline those
retrievals sit inside. Our own retrieval ratio is three orders of magnitude and buys
nothing at the sizes we tested; the crossover into a material end-to-end advantage sits
near N≈12,000. Anyone optimising retrieval for a captioner-fronted video pipeline should
know where that line is before optimising further, and we would rather publish the number
than the ratio that flatters us.

**Accuracy sits in the weakly-supervised band, not at the state of the art.** On NExT-GQA
our held-out Acc@GQA is 0.1667 [0.088, 0.243] with a ~3.4B answerer on CPU [C4.1] —
comparable to weakly-supervised baselines and well below current agentic methods. This
paper's claim is that a graph costing 0.215 s to build does not degrade answer quality
below that band, not that it advances it.

### Negative results as contributions

We pre-registered kill criteria for two hypotheses we expected to confirm, and both
triggered. Codec-derived boundary placement does not beat content-blind placement at
matched segment count: across four segmentation strategies the differences fall within
noise, and on long videos the codec-versus-matched-count difference is +0.088 M-Avg with a
95% CI of [−0.009, +0.204]. Codec-based frame admission is statistically indistinguishable
from uniform sampling at matched budget, a result that now holds from three independent
directions [C3.2, §7.2].

These constrain our own claim, and we think they improve it. Neither *where* we place
boundaries nor *which* frames we admit is doing the work; the block structure is. A
pipeline adopting this construction inherits the cost and tractability properties without
adopting our codec signal, our segmentation, or our admission policy.

### Contributions

1. **A block-diagonal graph construction proven bit-identical** to dense-then-prune under a
   five-seed identity gate and a 526-question downstream gate, at ~233× lower wall-clock
   and ~6.8× lower peak memory [C2.1–C2.3]. The result is measured on the complete-block
   edge configuration; §3.5 and §4.3 state how it relates to the tiered configuration our
   accuracy numbers use.
2. **Query-latency scaling exponents with bootstrap confidence intervals** over a 31-clip
   corpus, establishing quadratic-versus-sublinear separation, together with a tractability
   boundary at which dense retrieval does not complete [C1.1, C1.5, C1.3].
3. **A construction path with no generative model in it.** Frame selection performs zero
   neural forward passes, verified by instrumentation rather than asserted from source
   [C5.1, C5.2]; the only network in the path is a CLIP image encoder over admitted frames.
   Full ingest costs 11.976 s per video, of which 2.845 s is selection (§3.1).
4. **An Amdahl accounting of the query path**, locating where a three-orders-of-magnitude
   retrieval speedup is absorbed, including a cache-locality cost our own sparse retrieval
   incurs (§5.3, §5.4). We regard this as a result, not a caveat.
5. **Two pre-registered negative results** delimiting which components of the pipeline are
   load-bearing (§7).

The paper is organised so that each claim arrives with its limit attached. §2 positions the
work against the symbolic-graph, graph-RAG and compressed-domain lines. §3 describes the
pipeline and, at some length, its configuration provenance. §4 establishes construction
identity before reporting construction savings. §5 reports query scaling and then the
end-to-end accounting that bounds it. §6 places accuracy in its band. §7 reports the two
failed hypotheses. §8 states what remains open, of which the largest is that our
construction result is measured on an edge configuration our accuracy numbers do not use.

---

---

## 2. Related Work

### 2.1 The token bottleneck in long-video QA

Multimodal LLMs are bounded by input token capacity, so applying them directly to long
video forces aggressive frame subsampling and the loss of temporal context that follows.
The scale of the constraint is concrete: at 1 FPS, published per-question frame budgets
range from 32 frames (InternVL3) and 180 (VideoLLaMA3) to 512 (Qwen2.5-VL) and 3,900
(Gemini Flash 2.0) — against the tens of thousands of frames in an hour of footage. Every
approach to long-video QA is, in some form, a strategy for deciding what to discard.

Broadly, three families have emerged: **subsample and hope**, which accepts the loss;
**retrieve then answer**, which selects evidence per query; and **build an intermediate
representation**, which converts the video once into a compact structure that later
queries reuse. Our work sits in the third family, and borrows from the second at query
time.

### 2.2 Graph-structured intermediate representations

The representative recent system, and our closest point of comparison, is **EgoSG**
[Taluzzi et al., 2026]. It partitions video into non-overlapping fixed-length chunks
(Δt = 60 s, processed at 1 FPS) and prompts a frontier MLLM — Gemini Flash 2.0 — to emit
a symbolic scene graph per chunk: environmental elements, dynamic objects, spatial
relations, and timestamped action hyperedges, serialized as text. Graphs are built
iteratively, with each chunk's graph produced by prompting the model to *update* the
previous one, so the final graph summarises the whole video. On HD-EPIC VQA (1,250
questions across 25 prototypes) this improves over raw-video input for most models, by
3.22 points for Gemini itself and 5.60 for Qwen2.5-VL-14B.

We differ from EgoSG on four axes, and it is worth being precise about which of them are
genuine contrasts and which are simply different problems.

**(1) What the graph is for.** EgoSG's graph is a *symbolic reasoning substrate*: text
fed to an MLLM that reasons over entities and events. Ours is an *embedding retrieval
structure*: a weighted graph over frames used to rank evidence. **These are not
substitutes** — but the difference is in where description is produced, not in which
questions are reachable. A time-bounded query — "which step did the participant perform
between 32:10 and 32:38" — is answerable under either design: EgoSG reads symbolic events
written at construction time, while we select the frames falling inside the interval and
caption them on demand. We expose no time-window query interface, however — retrieval is
CLIP-text-driven end to end (§3.4) — so this describes what the representation admits, not
a measured capability. What differs is where the cost falls. EgoSG pays a frontier-model
call per chunk during construction and then reads text cheaply for every later query; we
produce no text at construction and pay the captioning cost per query instead, at roughly
2.5 s per frame on CPU (§5.4). The asymmetry that genuinely favours a symbolic graph is a
different one: relational queries spanning the whole video — how two entities interacted
across an hour, how often some event recurred — are answerable from a serialized graph
without revisiting the footage, whereas we would have to caption exhaustively to match it.
Conversely our structure supports frame-level retrieval that a serialized graph discards.
We compare on construction cost, not on representational power.

**(2) How construction scales, and what it depends on.** EgoSG's generation costs
approximately 5.7 s per one-minute clip and scales linearly with duration, metered against
a proprietary API. Two properties compound this. First, the iterative update rule makes
construction **inherently sequential** — chunk *i*'s graph depends on chunk *i−1*'s — so it
cannot be parallelised across a long video. Second, the authors report that they attempted
open-source models for generation and found that even the latest ones struggle to produce
accurate graphs reliably, identifying this dependency explicitly as a limitation of the
approach.

Our construction has neither property. Scenes are independent, so the block structure is
embarrassingly parallel by definition, and no model — proprietary or otherwise —
participates in construction. Building the graph at N=4,892 costs 0.215 s on CPU (§4.2).

**(3) What kind of guarantee is available.** EgoSG's authors are candid that graph
generation is imperfect: a manual audit of five clips finds roughly 5% of nodes and
relations in error, 15.5% false action hyperedges, and 13.7% of objects missing, and they
note explicitly that optimising generation quality is not their focus. This is not a
criticism — it is the expected behaviour of a generative construction, and their results
hold despite it.

But it does mean no correctness guarantee is available for the constructed artifact. Ours
is a different situation: because our construction is structural, we can verify it exactly,
and we do (§4.1). **We are careful about what this guarantee covers.** It establishes that
our block-diagonal construction produces the identical graph to our own dense-then-prune
reference — a *fidelity* guarantee about construction, not a claim that the resulting graph
is semantically correct. A structural construction cannot hallucinate an object, but
neither can it recognise one.

**(4) What is measured.** EgoSG's efficiency argument is made in tokens and estimated
FLOPs, with runtime amortised over the ~22 questions per video in HD-EPIC. Ours is made in
wall-clock and resident memory for construction, and in measured query latency against
graph size. Both papers report an amortisation story — theirs over questions per video,
ours over a caption cache across a session (§5.4) — and in both cases the per-query
economics differ substantially from the per-video ones.

**Vgent** [Shen et al., NeurIPS 2025] is the closer comparator, and we treat it rather
than EgoSG as the reference point for our efficiency framing. Vgent partitions a long video
into clips of K=64 frames, uses an LVLM to extract entities from each clip's frames and
subtitles, and builds a graph whose nodes are clips and whose edges link clips sharing
merged prototype entities. Retrieval is by keyword embedding over that graph, followed by
structured sub-query verification of the retrieved clips.

Two things make it the right foil. First, it is built on the same amortisation argument we
make: graph construction is offline and query-independent, so a single build is reused
across the questions asked of that video. Second — and unusually — it reports the split
explicitly, at 20.13 s of query-independent construction and 3.93 s of query-dependent work
per minute of video, against 20.81 s per minute for the query-dependent Video-RAG baseline,
which yields a claimed 1.73× end-to-end speedup on VideoMME at roughly three questions per
video. We are therefore not the first to argue that construction cost is a separate axis
deserving its own measurement, and §1 does not claim to be.

Where we differ is what construction costs. Vgent invokes an LVLM per clip, as EgoSG invokes
Gemini per chunk; our frame *selection* runs on CPU with zero neural forward passes, and our
only construction-time network is CLIP ViT-B/32 over admitted frames — an image encoder, not
a generative captioner or an LVLM (§3.1). That is the axis on which our construction result
is a contribution, and it is narrower than "graph-based video RAG is expensive."

The comparison must be drawn at that level of care. Our selection stage costs 2.845 s per
video and our full ingest, including CLIP enrichment, costs 11.976 s (§3.1); Vgent's 20.13 s
is per *minute of video* and includes its LVLM extraction. Even the end-to-end figures are
not like-for-like, since the denominators differ, so the honest claim is about the *class* of
model each construction requires — an image encoder against a generative LVLM — rather than
about the seconds.

<!-- open item 3 -> Appendix C -->

### 2.3 Grounded video question answering

**NExT-GQA** [Xiao et al., 2023] introduced the requirement that a system not only answer
correctly but point at the evidence, reporting Acc@GQA alongside grounding metrics (mIoP,
IoP@0.5, mIoU). Its central finding — that models achieving strong answer accuracy often
ground poorly, i.e. are right for the wrong reasons — is the reason we report grounded
accuracy rather than answer accuracy alone (§6.1).

The weakly-supervised band on that benchmark is occupied by Temp[CLIP] with NG+ (16.0
Acc@GQA), SeViLA as reproduced by the benchmark authors (16.6), LangRepo (17.1), and
FrozenBiLM with NG+ (17.5). <!-- open item 4 -> Appendix C -->

Two conventions matter for reading any of these numbers. First, published figures are
computed on the full 5,553-question test set, whereas ours is a 120-question held-out
split (§6.1). Second, IoP admits more than one definition, and the union convention is
biased upward relative to the benchmark's max-per-span convention — a distinction we flag
because it determines whether grounding figures are comparable at all.

### 2.4 Agentic and grounder-based methods

A more recent line attaches an explicit temporal grounder to the QA pipeline, often with
multiple cooperating agents or roles. **MUPA** reports 28.7 Acc@GQA at 2B parameters and
30.3 at 7B; **VideoMind** reports 25.2 at 2B. These define the current accuracy frontier
on NExT-GQA, and **we do not compete with them on accuracy** — our contribution is
construction cost, and our answer quality sits a band below (§6.1). We cite them to
situate that gap honestly rather than to select a weaker comparison set.

*(Note for co-authors: MUPA's abstract misstates its own Table 1 — cite the table:
28.7 / 39.1 / 38.7, not 29.0 / 39.7.)*

### 2.5 Compressed-domain video analysis

![Figure 5](figures/fig3_edge_configs.svg)

**Figure 5.** The two edge configurations share a block structure but not an edge set. §4's construction result is measured on the upper one; §6's accuracy results use the lower one.


Using signals available in the encoded bitstream — motion vectors, residual energy, packet
size — to avoid full decode is a long-standing idea in video analysis, running from
motion-vector surrogates for optical flow through CoViAR and DMC-Net, which modelled
I-frame, motion and residual streams jointly for action recognition <!-- open item 5 -> Appendix C -->.

The idea has recently reached video language models directly. **CoPE-VideoLM**
[Sarkar et al., 2026] passes I-frames through a frozen vision encoder but converts P-frames
into compact Δ-tokens — eight per fused P-frame — using their motion vectors and residuals,
avoiding full RGB decode for most frames. It reports up to 86% lower time-to-first-token and
up to 93% fewer visual tokens across 14 benchmarks, and situates itself against earlier
codec work that discarded residuals or temporal ordering (Video-LaVIT, EMA).

**Our use of the codec is different in kind, and the distinction is what makes our negative
result compatible with their positive one.** CoPE-VideoLM uses codec primitives as a
*representation* — the motion vectors and residuals are encoded and consumed by the model,
so their information content is exploited directly, at the cost of a trained Δ-encoder and
end-to-end fine-tuning. We use codec statistics as a *selection and weighting heuristic*
over frames we then treat conventionally, training nothing. A finding that codec-derived
salience does not beat uniform sampling for *admission* (§7.2) says nothing about whether
codec primitives carry signal when *encoded*; the two claims concern different uses of the
same bitstream fields. We take no position on theirs.

We use such signals for scene
segmentation and edge weighting. Both are cheap in *marginal* terms rather than absolute
ones, and the distinction matters: edge weighting from codec statistics costs an order of
magnitude less than pixel-difference or semantic weighting on identical topology
(0.017 s vs 0.119 s and 0.122 s at N=4,892), while extracting those statistics requires a
demux pass that is not itself cheap in isolation. That pass is the one ingest already
performs for frame selection, so its cost attributable to graph construction is
approximately zero — but we state the amortisation rather than quoting the marginal figure
alone.

**We make a deliberately modest claim here, and report a negative result that constrains
it.** At matched segment count, codec-derived boundary placement performs within noise of
content-blind alternatives (§7.1), and codec-based frame admission is statistically
indistinguishable from uniform sampling at matched budget (§7.2). Compressed-domain
signals in our pipeline are a *cheap* source of structure, not an *informative* one, and
we present them as such.

### 2.6 Positioning

Against EgoSG and its family, we replace a frontier-model-generated symbolic graph with a
structurally-constructed embedding graph: far cheaper, verifiable, parallel, locally
runnable — and semantically poorer. Against grounded-QA baselines, we land in the
weakly-supervised band while training nothing and running on CPU. Against the agentic
frontier, we are a band behind on accuracy and orders of magnitude ahead on construction
cost.

The claim we make is narrow and, we think, useful: **for retrieval over long video, the
intermediate graph does not have to be expensive, and does not have to be generated.**

---

---

## 3. Method

IRIS ingests a video once into a compact index, then answers arbitrary text queries
against that index. This section describes the pipeline; §4 and §5 measure it. Throughout,
**N** denotes the number of *surviving* frames after admission, not the number of frames in
the container — the distinction matters for every scaling figure we report.

### 3.1 Ingest and frame admission

![Figure 1](figures/fig1_pipeline.svg)

**Figure 1.** The pipeline with measured per-stage cost. Construction runs once per video; query costs are medians over five real text queries at N=4,892 on the scene-sparse arm.


Ingest demuxes the video and computes, per frame, a scalar **action score** combining
luma difference, motion energy, and luma entropy at weights 0.5 / 0.3 / 0.2. Frames are
assigned to tiers by thresholding this signal (`salient_thresh` 0.35, `candidate_thresh`
0.08) with adaptive per-video thresholding enabled, and local maxima are marked as peaks
via `argrelextrema` with a window of 3. Admitted frames — the survivors — are the input to
everything downstream; production retention across the annotated corpus falls between
10.41% and 11.14% [C3.4].

**No neural network runs during ingest.** We verify this by instrumentation rather than
inspection: monkeypatching `torch.nn.Module.__call__` for the duration of every
`parse_video()` call and counting invocations across 32 videos yields exactly zero [C5.1].
Ingest costs 2.945 s mean wall time (min 0.337, max 15.474) at 530 MB mean peak RSS
(max 1.291 GB), CPU-only [C5.2].

CLIP embeddings (ViT-B/32) are computed for admitted frames after this stage and are what
the neural-forward-pass claim excludes; the claim is specific to `parse_video`.
Concretely, `ingest()` validates the container, calls `charon_v.parse_video()` for decode,
scoring and selection, and only then enters `_build_index_from_records()`, whose fifth step
computes CLIP embeddings for admitted frames alone. Captioning is deferred entirely and
`caption` remains `None` at ingest. **The 2.945 s and 530 MB figures therefore cover
`parse_video` and not CLIP enrichment**, which is why the neural-forward-pass count is zero:
the counted region ends before the only stage that would contribute one.

We flag the consequence rather than leave it implicit. Our ingest as a whole is *not*
forward-pass-free — it runs CLIP ViT-B/32 over every survivor — and the figure above is a
selection-stage cost, not an end-to-end ingest cost. Any comparison against a system whose
reported construction cost includes its neural extraction stage is therefore not
like-for-like on this figure.

Both stages have since been measured over that same frozen sample, so we can report the
end-to-end figure rather than only the selection one:

| stage | mean wall | min | max |
|---|---:|---:|---:|
| `parse_video` (selection) | 2.845 s | 0.314 s | 14.870 s |
| CLIP enrichment | 9.131 s | 0.501 s | 56.885 s |
| **full ingest** | **11.976 s** | 0.814 s | 71.754 s |

CLIP enrichment costs 26.56 ms per admitted frame and is roughly three times the selection
stage. **The honest headline for construction is therefore ~12 s per video, not ~2.9 s** —
the smaller figure describes the forward-pass-free portion only, and we report both so the
distinction cannot be lost downstream.

<!-- open item 6 -> Appendix C -->

Two provenance notes. The re-measurement reproduces `parse_video`'s wall time within noise
(2.845 s against 2.945 s, −3.4%). It does **not** reproduce the 530 MB peak RSS, reporting
1,400 MB instead, because the re-measurement warms CLIP before the loop so that every video
sees steady-state cost, leaving CLIP weights resident and inflating a sampler that reads
whole-process RSS. The 530 MB figure comes from a run in which torch was never loaded and
remains the correct one for `parse_video` in isolation; the 1,400 MB figure is an artifact of
the newer script's design and is not a memory regression.

### 3.2 Scene segmentation

Survivors are partitioned into scenes. The default rule derives boundaries from valleys in
the packet-size curve obtained directly from the container — no pixel decode, no model
call. Segmentation is exposed as a configuration axis with four settings — `codec`,
`fixed_count`, `fixed_seconds`, and `fixed_time_matched` — which are precisely the four arms
compared in §7.1.

**We present the codec rule as one cheap option, not a superior one.** §7.1 shows that at
matched segment count, boundary placement is within noise across all four rules. The
property the pipeline depends on is that survivors are partitioned into scenes at all, not
that the partition is content-adaptive.

### 3.3 The scene-sparse graph and block-diagonal construction

Let the S scenes partition the N survivors, with scene *i* holding *nᵢ* survivors. The
scene-sparse graph connects every pair of survivors within a scene and no pair across
scenes, giving an adjacency that is block-diagonal under a scene-ordered permutation.
Edges carry semantic, motion, and temporal weight components.

The edge count is therefore

&nbsp;&nbsp;&nbsp;&nbsp;|E| = Σᵢ nᵢ(nᵢ−1)/2

against N(N−1)/2 for the dense graph. The conventional construction computes the dense
graph and prunes cross-scene edges; ours enumerates the diagonal blocks directly. At
N=4,892 over S=528 scenes the two quantities are 23,571 and 11,963,386 — the pruning
construction discards **99.8%** of the pairs it examines, while block-diagonal construction
visits exactly as many pairs as it retains.

In the balanced case, where all S scenes hold n = N/S survivors, the block-diagonal pair
count is S·n(n−1)/2 ≈ N²/(2S) against N²/2 for the dense path, so the saving is the scene
count S. Two boundaries follow. A single scene (S=1) makes the two paths identical, and the
construction saves nothing. The saving grows with scene count for a fixed N, and is maximal
when scenes are equal in size: for fixed N and S, Σᵢ nᵢ² is minimised at nᵢ = N/S, so an
uneven partition with one dominant scene approaches the dense cost. At N=4,892 over S=528
the realised ratio is 23,571/11,963,386 = 0.197%, against 1/S = 0.189% for a perfectly
balanced partition of the same N and S — the VIRAT partition is close to balanced.

Because the pruned graph and the block-diagonal graph are the same object, this is a
change of construction order, not of representation. §4.1 verifies that claim empirically
rather than resting on the argument.

This section describes the **complete-block** graph. The production edge mode
(`hierarchical_sparse`, §3.5) shares the block structure but populates each block sparsely,
so |E| above is an upper bound on it rather than a description of it.

<!-- open item 7 -> Appendix C -->

### 3.4 Retrieval

A text query is encoded with the same CLIP model used at ingest and resolved in stages:

1. **Scene shortlist.** Each scene is represented by the centroid of its frame embeddings.
   The query is scored against scene centroids and only the top `w` scenes proceed, with
   `w = max(4, ⌈√S⌉)` by default.
2. **Descent.** Within shortlisted scenes, frames are ranked by personalised PageRank over
   the scene-sparse graph, seeded from the query, with damping 0.5 and a rank-space blend
   (λ = 0.5) between semantic and codec-derived rank.
3. **Shortcut.** When the leading scene's score exceeds the runner-up by more than a margin
   (default 0.015), the descent is skipped and the scene's representative frames are
   returned directly.
4. **Top-k, captioning, answering.** The top-k retrieved frames are captioned and passed to
   the answerer (`granite4:micro`, ~3.4B, Q4_K_M, CPU).

Two properties of this design are load-bearing for later sections. The **shortlist** makes
retrieval sublinear by construction — it is why §5.1's exponent is below 1 — but it is also
a hard recall gate: a scene excluded here cannot contribute a frame downstream, and we
report the consequence in §8. The **shortcut** is a latency optimisation that bypasses the
graph entirely; it is the mechanism behind the guard violations noted in §5.1. The
conjecture that it fires more often at small S is **confirmed from source**, and the
mechanism is exact.

Retrieval ranks scenes by the best per-frame CLIP similarity within each, then computes
`margin = anchor_sim − runner_up_sim` between the top two shortlisted scenes. If
`margin > τ` (`scene_shortcut_margin`, default 0.05) the exact top-k from the anchor scene
is returned directly and the union-subgraph PPR is skipped. Crucially, when only one scene
is shortlisted the margin is set to infinity — there is nothing to descend across — so the
shortcut fires deterministically. Both conditions become more likely as S falls, which is
why the two smallest clips in the corpus are the two that trip it.

The shortcut is exact rather than approximate: it returns the same top-k the full path
would, and the diagnostic records divergence against itself as trivially zero.

### 3.5 Configuration and provenance

Results in this paper are produced under three settings of `graph_edge_mode`, and we state
per experiment which was used rather than describing a single "frozen configuration":

| experiment | `graph_mode` | `graph_edge_mode` |
|---|---|---|
| §4.1 identity gate, §4.2 build savings | `scene_sparse` | `fully_connected` ↔ `block_diagonal` |
| §5.1–5.2 scaling exponents, tractability | `flat` / `scene_sparse` | `fully_connected` |
| §5.3 end-to-end, §5.4 caption stage | `scene_sparse` | `hierarchical_sparse` |
| §6.2 MLVU, §7.1 segmentation ablation | `scene_sparse` | `hierarchical_sparse` |
| §6.1 NExT-GQA grounded QA | `flat` | — (see below) |

Two clarifications keep this from reading as an inconsistency.

**All three edge modes produce within-scene-only graphs under
`graph_mode="scene_sparse"`.** `fully_connected` and `block_diagonal` connect every
intra-scene pair; `hierarchical_sparse` builds a tiered edge set (temporal at
`graph_temporal_window=1`, hierarchy parents, salient-semantic top-4, motion-neighbour
top-2) and then removes every cross-scene edge. The block structure is therefore common to
all three.

**The modes do not, however, produce the same graph within the blocks, and §4's
construction result is scoped accordingly.** `fully_connected` and `block_diagonal` fill
every intra-scene pair — 23,571 edges at N=4,892 / S=528. `hierarchical_sparse` fills a
selected subset of the same blocks — 6,284 edges on the identical clip, roughly 3.75×
sparser. Two consequences we state rather than gloss:

- **The §4 construction saving is measured on the complete-block configuration**, which is
  not the configuration that produced any accuracy number in this paper (§6 and §7.1 run
  `hierarchical_sparse`). §4.3 scopes the claim explicitly.
- **`hierarchical_sparse` pays its own quadratic pass**, which block-diagonal construction
  does not remove. `_add_motion_neighbour_edges` scans all N(N−1)/2 pairs to select two
  neighbours per node, and the salient-semantic pass scans all pairs among salient nodes.
  The tiered mode is sparse in its *output*, not in its *construction*. Making the
  construction saving apply to the evaluated configuration requires bounding those scans to
  `node_groups`, which the current code does not do (§8, future work).

**A design consequence of the prune-after-select order, worth recording.** Under
`scene_sparse`, `hierarchical_sparse` selects top-k neighbours *globally* and only then
discards cross-scene edges. A node whose two nearest motion neighbours both lie in other
scenes therefore ends with zero motion edges rather than its two best in-scene ones. The
production graph's edge count consequently depends on scene boundaries in a way we have not
characterised. <!-- open item 8 -> Appendix C -->

The
exponents in §5.1 were measured under `fully_connected` and transfer to `block_diagonal`
by the identity established in §4.1.

**Under `graph_mode="flat"` the block-diagonal path is unreachable**, because the flat
construction path never partitions nodes into groups. The §6.1 grounded-QA results are
therefore a dense-graph measurement, reported as a correctness floor rather than as a
measurement of the sparse construction.

Per-experiment configuration hashes are recorded in the artifacts and indexed in
Appendix A; those hashes, not this table, are the reproducibility anchor.

### 3.6 Edge weight formulas

Every edge carries three component scores and one scalar weight derived from them.

- **Semantic**: ReLU-clamped cosine between the two nodes' CLIP embeddings, or 0 if either
  embedding is absent or zero-norm.
- **Motion**: under the default `motion_similarity_mode="action_score"`, a normalised
  action-score gap, `max(0, 1 − |a_u − a_v| / R)`, where `R` is the per-build action-score
  range (returning 1.0 when `R = 0`). Under the optional `"geometry_6d"` mode it is instead
  a clamped cosine over a six-dimensional kinematic vector (motion magnitude, divergence,
  curl, Jacobian Frobenius norm, maximum Hessian eigenvalue, motion entropy), falling back
  to the scalar form when either vector is degenerate. All results in this paper use the
  default scalar mode.
- **Temporal**: `1 / (1 + |t_u − t_v|)` in seconds.

The scalar weight is a per-family combination of those three, floored at 1e-6:

| edge family | weight |
|---|---|
| `temporal` | 0.7·temporal + 0.3·motion |
| `semantic_salient` | 0.8·semantic + 0.2·temporal |
| `motion_neighbor` | 0.8·motion + 0.2·temporal |
| `hierarchy_peak_salient` | temporal · max(semantic, motion) |
| `hierarchy_salient_candidate` | temporal · max(motion, 0.5·semantic) |
| `fully_connected` | α·semantic + β·motion |
| default | α·semantic + β·motion + 0.1·temporal |

Two properties of this table are worth stating rather than leaving to be discovered.

**The `fully_connected` family is the only one that consults α and β, and the only one
that ignores the temporal component** — it is also the only one returned without the 1e-6
floor, so a pair with zero semantic and zero motion similarity carries weight exactly 0.
The tiered families use fixed coefficients throughout. A consequence for §5.1: α and β
affect the dense arm's edge weights and do not affect the scene-sparse arm's at all.

<!-- open item 9 -> Appendix C -->

---

---

## 4. Construction Cost

The scene-sparse graph is the same graph the conventional construction produces, built at a
fraction of the cost. That is the claim of this section — not that it is a *better* graph.
We establish the identity first and report the savings second, because a construction
speedup is uninteresting if it silently changes what is built.

One scope statement belongs here rather than in a footnote. This section concerns the
**complete-block** configuration, in which every intra-scene pair carries an edge. Our
evaluated configuration (`hierarchical_sparse`, §3.5) shares the same block structure but
populates each block sparsely — 6,284 edges against 23,571 on the same clip — and is not
the configuration measured below. §4.3 states what does and does not follow from that.

### 4.1 The identity gate

**Block-diagonal construction and dense-then-prune construction produce the same graph.** We
verify this rather than argue it.

The two paths that produce a scene-sparse graph are the reference path, which materialises
the dense N×N graph and prunes cross-scene edges, and the block-diagonal path, which
materialises only the within-scene blocks. At N=4,892 survivors over 528 scenes (VIRAT),
they agree exactly [C2.1]:

- **Edge count** 23,571, matching the theoretical prediction, the reference path, and the
  block-diagonal path.
- **Node sets** identical; zero edges present in only one graph, in either direction.
- **Edge attributes** — `weight`, `semantic_weight`, `motion_weight`, `temporal_weight`,
  `edge_type` — zero mismatches at tolerance **0.0**, not at a floating-point epsilon.
- **PageRank** bit-identical across all 4,892 nodes.
- **Personalised PageRank** top-20 ordering and exact scores identical across five seeds.

**The two paths are also indistinguishable at the end of the pipeline.** Over 526 NExT-GQA
validation questions, the `fully_connected` and `block_diagonal` construction paths produce
bit-identical retrieval — identical retrieved order, peak frame, predicted span, IoP, and
peak-in-gold on every question (peak-in-gold 0.3175 and mIoP 0.3140 under both) [C2.3]. We
run this second gate because graph identity does not by itself guarantee identical
downstream behaviour.

We are precise about what that gate establishes: it is an **equivalence proof between the
two construction paths the change concerns**, not a re-run of a previously committed
result. No committed grounding number was produced under `scene_sparse` with
`fully_connected`; our committed scene-sparse arms use the `hierarchical_sparse` edge
formula, which the block-diagonal construction does not target (§3.5).

We state this as an identity rather than an approximation because the block-diagonal
construction is not an approximation. Cross-scene edges are absent from the scene-sparse
graph by definition; the reference path computes them and then discards them. The
block-diagonal path simply never computes them.

### 4.2 Savings

![Figure 2](figures/fig2_construction.svg)

**Figure 2.** Left: the dense path enumerates every node pair and retains only the within-scene blocks. Right: cost of the two construction orders for the bit-identical graph.


**Building the identical graph block-diagonally is 233.5× faster and uses 6.80× less peak
memory than building it dense and pruning.**

| | reference (dense-then-prune) | block-diagonal | ratio |
|---|---:|---:|---:|
| wall-clock (per build) | 50.20 s | 0.215 s | **233.5×** |
| peak RSS | 6.33 GB | 0.93 GB | **6.80×** |

The two rows come from two separate measurements, because wall-clock time and peak memory
are not well measured by the same protocol.

The **wall-clock** figures come from a controlled, interleaved measurement: 159 builds on
the same clip (N=4,892 survivors, 528 scenes), arms alternated across three rounds so any
machine drift affects both equally, with load and free memory logged before each arm.
Round-level ratios were 233.15×, 234.00×, and 233.60×, and the pooled-median ratio 233.53×
— agreement within 0.4%.

The **peak RSS** figures come from five repeats per arm, each a single build in a fresh
process, with medians reported: 6,325,764,096 bytes for the reference path and
930,693,120 bytes for the block-diagonal path. We measure memory this way because a peak
RSS taken over 50 consecutive in-process builds is a peak over the loop, not over a build,
and would not describe what production does. An independently recorded post-dedup RSS
ratio (6.8039×) agrees with the median-based ratio to three significant figures. Every
build in both runs produced exactly 23,571 edges.

**Three measurement caveats travel with the table. The first two make this a conservative
figure; the third bounds how far it generalises.**

First, the fast arm is timed over 50 consecutive in-process builds. This puts the timed
quantity safely above timer resolution, but it also charges each iteration a fixed
allocation overhead of roughly 0.09 s — negligible against a 50 s build, substantial
against a 0.2 s one. A single cold build in a fresh process, which is what production
performs, measures faster (≈0.12 s). We report the looped figure because it is the
reproducible one; the effect is to *understate* the ratio.

We note that this leaves us reporting a looped timing and a cold-process memory figure. The
protocol was chosen per quantity for measurement validity, not for favourability — a looped
peak RSS would not measure a build — but we do not know which direction a looped memory
measurement would move the ratio, and we have not run one.

Second, this isolates the `_build_graph` stage with frames and embeddings already cached;
it is not a full-ingest figure. The ingest path that produces those inputs is reported
separately (§3.1): 2.945 s mean wall time and 530 MB mean peak RSS over a 32-video
stratified sample, CPU-only, with zero neural forward passes [C5.1, C5.2]. That figure
covers frame selection and excludes the CLIP enrichment that follows it, so it is a
selection-stage cost rather than a full-ingest one (§3.1).

Third, the reported ratio is a **within-session** figure. The reference arm's wall time has
varied across separate measurement sessions running the same corrected code — 61.39 s and
48.57 s in two earlier fresh-process runs, against 50.20 s here — a spread of roughly 20%
that the interleaved protocol neither explains nor reproduces (machine state was logged as
idle and stable throughout: 2.5–10.1% CPU, 19.78–19.96 GB free of 33.46 GB, across all six
launches). Interleaving establishes that the two arms drift *together* within a session; it
does not establish that the reference arm's absolute cost is stable *across* sessions. The
0.4% cross-round agreement should be read as within-session precision, not as a bound on
session-to-session variation, and the ratio inherits that variation.

One correction belongs on the record. An earlier version of this measurement reported ~220×
and ~6.7×. Both construction paths at that time performed a redundant edge-and-PageRank
pass whose result was immediately discarded; removing it changed the timing of both arms
without changing what they build. That was verified bit-identically on nodes, all five edge
fields, per-node `scene_id`, and PageRank by a dedicated equivalence run, and independently
by the identity gate of §4.1. The 159 builds of the timing protocol assert the 23,571-edge
count on every build, not full field identity.

Intermediate measurements of the corrected code gave 259.23× from a single fresh-process
probe and a 396.73× median across five cold repeats per arm, with a 5×5 pairing spread of
[224.17×, 406.67×]. That spread is dominated by the block-diagonal arm's timing noise — its
cold builds range 0.1203 s to 0.2164 s, a ~1.8× spread on a sub-quarter-second operation,
against under 1% spread on the reference arm. This is why we report the looped, interleaved
protocol above rather than any of these point measurements.

### 4.3 Why construction cost is the durable claim

**Within the configuration it is measured on, the construction saving cannot be absorbed
by a downstream stage**, because there is no downstream stage inside construction. That is
the sense in which it is the more durable of our two efficiency results.

The query-side savings do not have that property. §5.3 reports a retrieval speedup of two
to three orders of magnitude that nonetheless fails to reach the user, because a
constant-cost stage dominates the end-to-end path at the sizes we measure.

We draw the contrast deliberately. A reader should finish §4 and §5 knowing which of our
efficiency numbers survives contact with a full pipeline and which does not.

**The limit of the construction claim, stated plainly.** It is measured on the
complete-block configuration, and our reported accuracy numbers come from the tiered
configuration. The two share a block structure but not an edge set, and block-diagonal
construction does not make the tiered configuration cheaper to build: that mode selects its
neighbours by scanning all pairs and then keeping a few, so it pays a quadratic pass of its
own (§3.5). Carrying this result into the evaluated configuration means bounding those scans
to the scene partition — a change the current implementation does not make, and one that
would produce a *different* graph rather than an identically-constructed one, since global
top-k followed by cross-scene pruning is not the same as within-scene top-k. It therefore
requires accuracy validation, not an identity gate. We mark it as future work (§8) rather
than claiming it.

We would rather a reader learn this limit from us in §4 than derive it from the
configuration table in §3.5.

---

## 5. Query Scaling

### 5.1 Latency exponents

![Figure 3](figures/fig4_exponents.svg)

**Figure 3.** Fitted query-latency exponents with clip-level bootstrap confidence intervals. The headline intervals are disjoint and the scene-sparse bound lies below linear.


We fit log(latency) = k·log(N) + c per arm over a corpus of 31 measured clips drawn from
UCF-Crime (Anomaly-Part-1 and Testing_Normal_Videos), under **real CLIP text queries**
(50 queries per clip), with clip-level bootstrap confidence intervals (seed 42,
B=10,000, resampling clips within each N-bucket and refitting) [C1.1, C1.5].

| arm | fit range | n points | k | 95% CI | R² |
|---|---|---:|---:|---|---:|
| dense (flat) | N ≥ 1102 (headline) | 4 | 2.063 | [2.020, 2.125] | 0.9994 |
| scene-sparse | N ≥ 1102 (headline) | 6 | 0.834 | [0.801, 0.884] | 0.9830 |
| dense (flat) | full range | 17 | 1.980 | [1.951, 2.006] | 0.9978 |
| scene-sparse | full range | 19 | 0.497 | [0.461, 0.530] | 0.8980 |

**Over the measured range, dense query cost grows with the square of graph size and
scene-sparse query cost grows sublinearly.** The headline intervals are disjoint and the
scene-sparse upper bound lies below 1.0, which is the pre-registered bar for that
separation.

**On the two scene-sparse exponents.** We report both and take as the headline the one
*less* favourable to our claim. The full-range fit (0.497) is markedly lower than the
headline fit (0.834), giving a larger sparse-versus-dense separation than we report.
We infer that small-N clips sit close to a measurement noise floor from the shape of the
fit alone — the shallower small-N slope and the full-range fit's visibly poorer explanatory
power (R² 0.898 vs 0.983) — where fixed per-query overhead would dominate and latency would
grow more slowly than the asymptotic trend; we have not verified this by direct
measurement, and no repeated small-N runs or fitted overhead constant exist to confirm it.
We take the N ≥ 1102 fit as the headline because it is the regime the paper is about — large
graphs — and because it is the conservative choice.

**Three caveats travel with the table:**

1. **Thin support at large N.** Bin occupancy against a target of five clips per bin:
   N≈1102 has 7, N≈2963 has 2, N≈6559 has 2, N≈13506 has 1. The measured survivor-N
   distribution is heavily weighted toward small graphs (median N=333; 24 of 31 clips
   below N=1000). The headline fits rest on 4 dense and 6 sparse points.
2. **A shortcut-branch guard violation, with the mechanism now identified.** Two small-N
   clips (Assault036, N=97; Abuse037, N=188) took a scene-sparse shortcut path on 20% and
   10% of queries respectively, violating a pre-registered guard that all timed queries
   traverse the PPR path. The mechanism is the margin test of §3.4: when the top two
   shortlisted scenes are separated by more than τ=0.05, or when only one scene is
   shortlisted at all, retrieval returns the anchor scene's exact top-k and skips the
   subgraph PPR. Both conditions become more likely as scene count falls, which is why the
   corpus's two smallest clips are the two affected.

   Two consequences follow, and the second is the one that matters. The shortcut is
   **exact, not approximate** — it returns the top-k the full path would return — so no
   retrieved result is degraded by it. But it is also **faster**, since it skips PPR
   entirely, so the affected queries bias the scene-sparse arm *favourably*. The effect is
   small and we can bound it: both clips fall outside the headline range, and refitting the
   full range with both excluded raises the exponent from 0.497 to 0.503, under 2% and in
   the direction unfavourable to us. We report the violation rather than suppress it.
3. **Salience-weight provenance.** The entire corpus was built at salience weights
   (`luma_diff_weight`, `motion_weight`, `luma_entropy_weight`) = (0.5, 0.3, 0.2), verified
   per-clip from cached configuration snapshots. These are the shipped defaults:
   `IRISConfig` declares them and `configs/default_iris_config.json` overrides none of the
   three, so survivor-N values here are directly comparable to retention figures computed
   under the production configuration.

   An earlier draft described this as a *deviation* from a frozen production default of
   (0.8, 0.1, 0.1). That triple does exist, but not as an IRIS default: it appears in the
   frozen config recorded by the UCF-Crime VAD tuning experiment, as
   (`packet_size_weight`, `motion_weight`, `luma_entropy_weight`). The first of those names
   is consumed nowhere in the pipeline — the scoring stage reads `luma_diff_weight`,
   `motion_weight` and `luma_entropy_weight`, and an absent `luma_diff_weight` falls back to
   0.5. So (0.8, 0.1, 0.1) was never IRIS's production salience triple, and the earlier
   disclaimer is withdrawn rather than restated.

### 5.2 Tractability divergence

**At the largest sizes we measured, the dense arm does not return at all** — a capability
difference that the exponent separation understates. Under a uniform 3600 s wall-clock and
29 GB memory watchdog, the dense arm was censored at N=6,559 (Arrest047) and N=13,506
(Arson019), with resident memory at 17.0 GB and 26.9 GB respectively and still climbing at
the cap. The scene-sparse arm completed the same two clips in 0.0104 s and 0.0210 s [C1.3].
Unlike the latency results below, this is not subject to Amdahl dilution: a query that never
returns cannot be rescued by a fast downstream stage.

The censoring bounds what can be said quantitatively, and we say only that much. This is a
**tractability boundary, not a speedup ratio**: the dense measurements are censored, so any
ratio computed against them is a lower bound on an unknown quantity. Substituting the cap as
a floor value and refitting gives k ≥ 2.597 for the dense arm — a bound, not an estimate,
and we attach no confidence interval to it.

### 5.3 End-to-end accounting: where the speedup goes

![Figure 4](figures/fig5_e2e.svg)

**Figure 4.** Left: the end-to-end query path. Right: inside the dominant stage. Captioning is 85% of that stage in the dense arm and 93% in the scene-sparse arm; L1 retrieval costs zero at every query.


This section reports the most useful thing we measured, which is that our own headline
retrieval ratio buys nothing at the sizes we tested. We report it in full because the
literature this paper sits in reports retrieval-side speedups routinely and end-to-end
accountings rarely, and the gap between the two is large enough to change what a system
designer should optimise.

Retrieval-mechanics measurements at N=4,892 show a 1,104× ratio between arms (7.9448 s
dense vs 0.0072 s scene-sparse) [C1.4]. **This ratio does not reach the user.**

We measured the full query path — query text to final answer — at the same N, with
`granite4:micro` (~3.4B, Q4_K_M) on CPU, `l2_retrieve_top_k=30`, `cerberus_mode="legacy"`,
five real CLIP-text queries per arm plus a discarded warm-up, and the caption cache reset
before each arm to prevent cross-arm leakage. **The primary measurement** is an
instrumented run on a clean, commit-stamped checkout (commit `deeeba8`,
`tracked_dirty_count` 0), which reports median per-component costs of:

| component | dense | scene-sparse |
|---|---:|---:|
| retrieval mechanics | 8.535 | 0.030 |
| answerer (LLM) | 2.830 | 2.817 |
| captioning | 37.269 | 62.132 |
| L1 retrieval | 0.000 | 0.000 |
| Cerberus verification | 6.408 | 4.418 |

Stage medians are taken independently, so they need not sum to the reported totals of
55.64 s and 69.40 s.

Captioning accounts for 85% of the bucket in the dense arm and 93% in the scene-sparse arm.
L1 retrieval costs **exactly zero** at every query in both arms — it populates an in-memory
dictionary over already-retrieved frames and issues no I/O or model call — so the bucket is
in practice a two-way split between captioning and verification, not a three-way one.

Median totals for this run are 55.64 s (dense) and 69.40 s (scene-sparse) —
**end-to-end ratio: 0.802×** — the sparse arm was slower at the median. Per-query caption
cost ranges from 2.6 s to 69.3 s depending on how many of the top-30 frames miss the
caption cache, so with five queries per arm these medians are directionally indicative
rather than precise.

The third stage bundles two costs: fetching each cache-missing frame's pixels (seek to the
nearest keyframe, decode forward through that GOP) and the captioner forward pass itself.
We measured the split on one query at the same N (30 cache misses, 272 frames decoded, 27
distinct GOPs): 0.815 s in the decode loop against 114.6 s in captioning, so frame fetching
accounts for under 1% of the stage. This split query was measured cold-cache, with all 30
retrieved frames missing the caption cache, so its 114.6 s absolute captioning figure
exceeds the partly-warm 62.132 s median in the table above; the ratio between decode and
captioning, not the absolute captioning time, is the point. The cost is captioner inference
at fixed top_k, not frame access.

**Provenance note.** This run was made on a clean, commit-stamped checkout (commit
`deeeba8`, `tracked_dirty_count` 0) with the captioner pinned in the harness
configuration; the resolved captioner identity is not recorded in the run artifacts.

**A prior measurement, made before this bucket was decomposed, reports a lower ratio
under a different captioner.** We measured the same full query path at the same N with
the same query protocol, the caption cache reset before each arm:

| stage (median, s) | dense | scene-sparse |
|---|---:|---:|
| retrieval mechanics (incl. query embedding) | 8.086 | 0.028 |
| answerer (LLM) | 2.629 | 2.623 |
| captioning + L1 + verification (single bucket) | 39.92 | 62.80 |
| **total** | **50.91** | **65.45** |

Every cell is a median taken independently, so the total column is the median of the
per-query totals rather than the sum of the three stage medians above it. The two differ:
the dense stage medians sum to 50.635 s against a median total of 50.91 s. Both are correct
figures for what they measure; neither is a reconciliation error.

**End-to-end ratio: 0.778×** — the sparse arm was *slower* at the median in this sample.
Per-query totals span 15.49–88.05 s (dense) and 10.19–79.40 s (sparse), so the sample is
noisy; but the direction of the median is unambiguous and the mechanism is not in doubt.

This prior measurement used a different captioner from the primary run above, and
captioning is 85–93% of the dominant stage, so the two runs corroborate **direction only,
not magnitude**: both find the sparse arm slower at the median, but the closeness of
0.778× and 0.802× should not be read as agreement in magnitude.

**Provenance note on the table above.** These figures were produced with BLIP captioning,
not the MiniCPM backend the committed default configuration specifies (§3.5). The harness
was untracked at the time and the selection mechanism is unrecoverable.

**The retrieval ratio in this run is not the ratio in §5.3's first line.** The
retrieval-mechanics ratio here is approximately 291×, against the 1,104× reported in
[C1.4]. The two are measured under different query types — real CLIP text encoding here,
synthetic sampled embeddings there — and **must not be conflated.** We report both with
their methodology attached; the 1,104× figure is a retrieval-mechanics number under
synthetic queries and is labelled as such wherever it appears.

One observation matters more than the headline ratio.

**The LLM is not the bottleneck.** The answerer stage costs 2.83 s and is identical
across arms, as expected given both feed the same top-k context to the same model. The
dominant term is the third stage — dominated by captioning per the primary table above,
though the earlier measurement (the prior-measurement table) recorded it only as a single
bucket covering captioning, L1 retrieval, and Cerberus verification. It is O(top_k) rather
than O(N) and therefore does not shrink as the graph sparsifies. Sparsifying the graph makes the
retrieval step negligible while leaving the largest term untouched.

**Crossover.** The retrieval-only crossover — the graph size past which scene-sparse
retrieval mechanics cost less than dense — sits at N≈24 from the §5.1 fits. We do not
project an end-to-end crossover. Doing so requires assuming the constant stage cost L is
equal across arms, which this run's own data contradicts (39.9 s vs 62.8 s; see §5.4), and
a projection built on that assumption returns ~1.15× at N=4,892 against a measured 0.778×.
The measured tractability boundary in §5.2 is the stronger statement and we rest on it.

### 5.4 Caption-cache locality (negative result)

The sparse arm's captioning stage costs more than the dense arm's at identical top_k.
Over 20 real text queries per arm at N=4,892, with top_k verified byte-identical across
arms (30 retrieved frames on all 40 queries) and an identical query set, the paired
clip-level bootstrap difference is **+9.30 s per query, 95% CI [3.52, 16.22]** (seed 42,
B=10,000), which excludes zero.

The cause is not per-frame cost and not frame dispersion:

| | dense | scene-sparse |
|---|---:|---:|
| caption cache misses per query (mean) | 3.85 | 7.75 |
| frames decoded (mean) | 38.3 | 73.15 |
| per-frame caption wall time (mean) | 2.535 s | 2.520 s |
| verify calls per query (mean) | 1.05 | 1.00 |

Per-frame caption cost is equal across arms; the entire difference is miss count. We
tested the hypothesis that the sparse arm retrieves more temporally dispersed frames and
**rejected it**: sparse retrievals touch marginally *fewer* distinct scenes (23.8 vs
26.8) and have a *smaller* mean pairwise frame-index distance (2,732 vs 3,070), and the
correlation between dispersion and caption time is essentially zero (pooled r = −0.058,
n=40).

The mechanism is cross-query cache locality. The dense arm re-retrieves an overlapping
pool of high-salience frames largely independent of the query — its miss count falls to
zero by the sixth distinct question — whereas the sparse arm returns query-specific
frames, so successive distinct questions touch largely disjoint frame sets and the
caption cache warms slowly.

We note the double edge plainly: **the dense arm's caching advantage is a consequence of
its retrieval being less responsive to the query.** The same property that makes it cheap
to cache makes it a worse retriever. This does not make the cost imaginary — the caption
cache is a production mechanism that persists across a session — so the honest statement
of our efficiency claim is that the sparse graph retrieves far more cheaply *and*
benefits less from cross-query caption reuse.

A build-time caption prefetch over high-centrality nodes would plausibly eliminate this
cost, given a construction budget of 0.215 s. We have not implemented or measured it.

---

---

## 6. Correctness Floor

This section does not claim competitive accuracy. It claims that a graph costing 0.215 s
to build, produced without a neural forward pass and queried by a ~3.4B answerer on CPU,
does not degrade answer quality below the weakly-supervised band. That is the relevant
question for an efficiency contribution: a construction saving is uninteresting if the
resulting system cannot answer anything.

### 6.1 Grounded QA on NExT-GQA

**Split provenance.** Our evaluation pool is drawn entirely from NExT-GQA's *validation*
grounding subset (`gsub_val`), intersected with our cached videos: 526 questions over 86
videos. We partitioned this at the video level into a 59-video tuning half (406 questions)
and a **27-video held-out validation half** (120 questions). We call the latter a
*held-out val split* rather than a test split, because it is not the official NExT-GQA
test set — that set (5,553 questions over 990 videos) is untouched by any experiment in
this paper.

On the held-out val split of 120 questions over 27 videos, **Acc@GQA is 0.1667, 95% CI
[0.088, 0.243]**, with Acc@QA 0.375 [0.275, 0.477] [C4.1]. The answerer is
`granite4:micro` (~3.4B, Q4_K_M, temperature 0, CPU); the option parser succeeded on
120/120 questions with zero failures.

The result decomposes cleanly: 42 of 120 questions are correctly grounded, 45 of 120 are
answered correctly, and 20 satisfy both — that is, 0.350 × 0.476 = 0.167.

**Grounding and correctness are coupled.** P(correct | grounded) is 0.476 against
P(correct | ungrounded) of 0.321 [C4.2]. This ~15-point gap replicated across two
independent samples (65 vs 46 in-sample; 47.6 vs 32.1 held out) even though absolute
levels fell between them, which supports the coupling independently of where the levels
land.

**Neither stage is negligible.** Perfect grounding would raise Acc@GQA to 0.476 (+0.31
headroom); a perfect answerer over current grounding would give 0.350 (+0.18). Grounding
carries more headroom, but the answerer is not a rounding error, and an earlier
in-sample estimate that suggested otherwise (P(correct | grounded) = 0.65) is not
reproduced held out — the held-out value sits at the floor of that estimate's interval.

**Against uniform frame sampling, our retrieval is positive but not separated.** The
held-out advantage in answer accuracy is +0.120, 95% CI [0.000, 0.248] — the interval
touches zero. Against *random* sampling it is +0.127 [+0.036, +0.229], which does
separate. The honest statement, and the one we make: **retrieval beats random frame
sampling on answer accuracy; against uniform sampling the advantage is positive but not
statistically separated at n=120** [C4.4].

**Placement, indicatively.** Our 0.1667 falls below MUPA-2B's 0.287 with our interval
excluding it, and lands in the same band as SeViLA (0.166), LangRepo (0.171), and
FrozenBiLM+NG+ (0.175), while running a ~3.4B answerer on CPU with no training.

| method | params | Acc@GQA | mIoP | IoP@0.5 |
|---|---|---:|---:|---:|
| Temp[CLIP] NG+ | 130M | 16.0 | 25.7 | 25.5 |
| SeViLA* | 4B | 16.6 | 29.5 | 22.9 |
| LangRepo | 12B | 17.1 | 31.3 | 28.7 |
| FrozenBiLM NG+ | 1B | 17.5 | 24.2 | 23.7 |
| VideoMind-2B | 2B | 25.2 | 36.4 | 32.6 |
| MUPA-2B | 2B | 28.7 | 39.1 | 38.7 |
| MUPA-7B | 7B | 30.3 | 41.4 | 39.4 |
| **IRIS (ours)** | ~3.4B, CPU | **16.7** | — | — |

**This comparison is indicative, not a leaderboard entry, and the splits differ.** Our
figure is n=120 from a held-out *validation* split with an ~8-point interval; every
published figure in the table is computed on the full 5,553-question **test** set. These
are not the same measurement, and we place our row to situate the system rather than to
rank it. We do not claim to have beaten any method in the table, and a genuinely
comparable number would require running the official test split — which remains available
to us precisely because we never touched it.

**The held-out val split is burned.** Both permitted touches have been used, so no further
measurement on those 27 videos is possible without a new split [C4.6] — a constraint that
binds future work on the answerer, which this section identifies as a live target with
+0.18 of headroom.

**Metric convention.** NExT-GQA questions may carry multiple gold spans, and the official
scorer takes the maximum overlap over gold spans rather than their union. An earlier
version of our evaluation used a union convention. We vendored a faithful implementation
of the official scorer, gated it against a synthetic edge-case suite (9/9 cases matching,
including the zero-width-span case that yields IoP=1.0 and IoU=0.0), and confirmed the two
conventions genuinely disagree on multi-gold-span questions.

We then measured the difference on the tuning half, where re-scoring costs no touch: mIoP
0.3126 union versus 0.3113 official, mIoU 0.1486 versus 0.1586, and **IoP@0.5 identical at
0.3202 under both**. The conventions can only differ on the 11.3% of questions carrying
multiple gold spans, and no prediction in the split was zero-width, so the inflation modes
available to the union convention never fire on this data. Because Acc@GQA is keyed to
IoP@0.5, our reported Acc@GQA is convention-invariant. Grounding figures for the held-out
half were computed under the union convention and cannot be recomputed without retained
spans; the measurement above is our evidence that this does not materially affect them.

**On reporting mIoP and IoP@0.5.** We omit them from the table above. They are
measurable and now convention-checked, but this section is a correctness floor rather
than a grounding claim, the grounding-correctness coupling is carried by Acc@GQA and
the conditional probabilities, and printing a grounding magnitude from a 120-question
validation split beside full-test leaderboard figures would invite exactly the
comparison we decline to make. The dashes in the table reflect this choice, not a
missing measurement [C4.3].

### 6.2 Multi-task long-video QA on MLVU

We additionally evaluate on MLVU's multiple-choice tasks, restricted to the six
third-person tasks (the egocentric Ego Reasoning task is excluded, since our
compressed-domain signals assume a largely static camera; the generation tasks are
outside our multiple-choice harness). 150 questions, 25 per task, videos capped at 600 s,
seed 42, `scene_segmentation="codec"`, `graph_mode="scene_sparse"`.

**M-Avg: 0.340.** Zero ingest failures across 145 unique videos; overall option-parse
failure rate 3.3%.

| task | accuracy | chance | parse-fail |
|---|---:|---:|---:|
| Topic Reasoning | 0.720 | 0.250 | 4.0% |
| Plot QA | 0.400 | 0.250 | 0.0% |
| Anomaly Recognition | 0.320 | 0.250 | 0.0% |
| Needle QA | 0.280 | 0.250 | 12.0% |
| Action Order | 0.160 | 0.250 | 4.0% |
| Action Count | 0.160 | 0.250 | 0.0% |

**Two tasks fall below chance, and the cause is the answerer rather than retrieval.**
Action Order and Action Count sit at 0.160 against a 0.250 floor. Parse failure on these
tasks is 4% and 0% respectively, so the model is confidently wrong rather than unparsed.
Both tasks require properties that top-k retrieval does not supply: chronological
ordering requires reasoning over relative time, and exhaustive counting requires complete
coverage rather than the most relevant evidence.

This is the same failure mode we observe on NExT-GQA's directional-temporal questions,
where accuracy collapses on before/after items even when the correct evidence is
retrieved. We treat it as a limitation of the
answerer at this scale rather than of the graph, and note it in §8.

Needle QA's 12% parse-failure rate is the highest of the six and its accuracy (0.280)
sits closest to chance among the tasks that clear it; the two may be related, and we do
not claim a meaningful margin there.

### 6.3 What this section supports

The system answers. It answers in the band occupied by weakly-supervised methods with
comparable or larger models, using a graph built in 0.215 s on CPU with no training and
no frontier model. It does not answer as well as current agentic methods, and where it
fails — directional temporal reasoning, exhaustive counting — the failure is located in
the answerer rather than in the representation.

---


---

## 7. What Does Not Matter

The results in this section are negative. We report them because each was
pre-registered with a kill criterion fixed before the data were seen, each criterion
triggered, and together they constrain what a practitioner should spend effort on. They
also delimit our own contribution: they are the reason §4 claims cheap construction
rather than better construction.

### 7.1 Segmentation placement, once the budget is fixed

Our scene boundaries derive from compressed-domain residual peaks (§3.5). The natural
question is whether this content-adaptive placement produces a better graph than a
content-blind rule. We tested it directly.

**Design.** Four boundary strategies, holding retrieval, answerer, question set, and seed
constant so that segmentation is the only moving variable: (a) **codec** — residual peaks;
(b) **equal-count** — the same number of segments as codec produced for that video, placed
to give equal survivor counts per segment; (c) **equal-time** — the same number of
segments, placed at equal time intervals; (d) **fixed-60s** — the interval used by
comparable graph pipelines. Arms (b) and (c) are matched-count controls: they isolate
*where* boundaries fall from *how many* there are.

**Short videos (≤600 s), six MLVU tasks, 150 questions, 145 videos.**

| arm | M-Avg |
|---|---:|
| codec | 0.340 |
| equal-count | 0.393 |
| equal-time | 0.320 |
| fixed-60s | 0.353 |

Codec did not lead. The simplest matched-count control was 5.3 points ahead of it.

**Long videos (600–1800 s), AR and PQA, 45 videos.** Short videos yield few segments, so
placement has little room to matter; the long-video run tests the regime where it should.
The premise holds strongly — codec produced a median of 581 scenes per video (mean 582,
range 174–1319), against single digits in the short-video run.

| arm | AR (n=17) | PQA (n=91) | M-Avg |
|---|---:|---:|---:|
| codec | 0.412 | 0.451 | **0.431** |
| equal-count | 0.235 | 0.451 | **0.343** |

The codec-minus-equal-count difference is **+0.088 M-Avg, 95% CI [−0.009, +0.204]**
(video-clustered bootstrap, seed 42, B=10,000). The interval includes zero. Our
pre-registered criterion required it to exclude zero; the difference did move in the
predicted direction (from −0.053 on short videos to +0.088 on long), and a tie was
pre-registered as failure.

**We therefore do not claim that codec-derived boundaries produce a better graph.** The
entire difference is carried by anomaly recognition (+0.176, 95% CI [0.000, 0.353],
n=17); plot QA is exactly flat (+0.000, CI [−0.101, +0.102]). A domain-concentrated
benefit remains possible and is untested at a sample size that could resolve it.

**The finding practitioners should take:** once the segment budget is fixed, boundary
placement is within noise across four strategies spanning content-adaptive to
content-blind. This is what licenses using the cheapest available segmentation — and it
is a stronger argument for our pipeline than a codec win would have been, because it does
not depend on our particular signal being special.

### 7.2 Frame admission versus uniform sampling

A second question is whether codec signals help *select* which frames enter the graph.
Three separate tests, on the 19 annotated UCF-Crime anomaly videos, say they do not.

**Coverage metrics are uninformative without a matched control.** Uniform admission
saturates gold-window coverage (M1 = 1.0000) at every budget swept, down to 5% retention
[C3.1]. Any selector will therefore appear to "cover 99%+ of gold windows"; the metric
measures annotated-window length, not selection quality. We flag this because we
previously reported such a coverage figure ourselves, and it does not mean what it
appears to mean.

**At matched budget, codec admission is indistinguishable from uniform.** At 10.5%
retention, the paired differences are +0.256 pp (95% CI [−0.236, +0.710]) and +0.780 pp
([−0.005, +1.753]) on the two informative metrics; both span zero [C3.2]. The admitted
set sits at the label-blind identity: mean(M2 − retention) = +0.0017 (sd 0.0099) [C3.3].

**Whether the tie is an artifact of the score wasting its budget is unresolved.** A natural
objection is that the score might clump its picks onto near-duplicate frames, so that a tie
would reflect poor temporal coverage rather than an uninformative signal. A post-hoc
diagnostic in the shot-bucketing study — explicitly not pre-registered — measured temporal
dispersion at a matched budget of k=503 across arms. The codec action-score top-k arm, which
is the R0 selector at the same budget, sampled 42 distinct temporal locations, 8.4% of
budget, with a mean run length of 13.19; uniform sampled 503 distinct locations, 100%, mean
run 1.00. The mechanism is that the propagated action score is hold-forward propagated from
the retained tier, so it is a step function with roughly 9.5 frames per plateau — top-k
therefore degenerates into selecting whole plateaus from their earliest frame, spending the
budget on adjacent near-duplicate frames. **This does not dispose of the objection**:
clumping is real and is a plausible contributor to the tie, and this diagnostic cannot
separate an uninformative signal from a budget wasted on near-duplicates. Distinguishing
the two requires the plateau-dedup contrast, which has been specified but not run.

**Shot geometry adds nothing either.** Segmenting each video into shots directly from the
packet curve (zero decode) and sampling one frame per shot at its midpoint gives, against
uniform, +0.001 pp ([−0.416, +0.421]) and +0.141 pp ([−0.608, +0.972]) — half-widths that
*exclude* a 1 pp effect, making this the one properly powered contrast in the group and a
genuine null.
The corresponding shot-plus-score contrast was underpowered (half-widths 1.69 and 2.79 pp)
and we report it as such rather than as a tie.

**Power.** The annotated corpus is 19 videos, so each video is 5.3 pp of any aggregate and
the resolution floor is roughly 1 pp. No re-run at new seeds or budgets changes this; only
more labelled anomaly footage would. We did not re-run in search of significance.

**Reproducibility limitation.** The continuous admission score was not persisted to disk —
only the derived retention decision was — so the admission diagnostic could not be
extended post hoc without a re-run [C3.5]. We have since adopted the rule that whenever a
derived decision is written, the continuous score it came from is written alongside it.

### 7.3 What these negatives buy

Read together, the two results say that neither *where* we cut nor *which frames we keep*
is doing the work. What is doing the work is structural: the graph is block-diagonal, so
it is cheap to build and cheap to query, and that property is independent of how the
blocks are chosen.

This is a narrower claim than the one we set out to make, and a more portable one. A
pipeline adopting our construction does not need our codec signal, our segmentation, or
our admission policy; it needs the block structure. §9 collects this portability point
rather than restating §4.

---

---

## 8. Limitations

We have reported negative results inline where they arose (§5.3, §5.4, §7) rather than
deferring them here. This section collects the constraints that bound our claims, including
those a reader would otherwise have to assemble across sections.

**8.1 The retrieval speedup is not an end-to-end speedup.** At N=4,892 the measured
end-to-end ratio is 0.802× despite a retrieval-mechanics ratio of two to three orders of
magnitude, because a captioning and verification stage costing 44–67 s dominates and is
O(top_k) rather than O(N) (§5.3). **The construction result (§4) carries
no equivalent caveat**; the query-latency results should be read as component-level
characterisations rather than as user-facing latency claims.

**8.2 Sparse retrieval pays a caption-cache penalty.** Across a multi-query session, our
retrieval incurs +9.30 s per query [3.52, 16.22] in caption cost relative to dense
retrieval, because query-specific retrieval warms a caption cache more slowly than dense
retrieval's query-insensitive frame pool (§5.4). A build-time prefetch over high-centrality
nodes would plausibly eliminate this; we have not implemented or measured it.

**8.3 The scaling fit rests on thin support at large N.** Our measured corpus is
small-graph-heavy (median N=333; 24 of 31 clips below N=1000), and the bins that matter
most for the headline claim hold 7, 2, 2, and 1 clips respectively. The headline fits use 4
dense and 6 sparse points. The confidence intervals are tight because the fits are clean,
not because the support is dense, and additional large-N clips are the obvious way to
strengthen this — a matter of ingestion effort rather than method.

**8.4 A guard violation in the sparse arm is unexplained.** Two small-N clips took a
shortcut path on 10–20% of queries, violating a pre-registered guard (§5.1). Both lie
outside the headline range and excluding them moves the fit by under 2%, but we do not yet
have a confirmed mechanism. Our working hypothesis — that the shortcut fires more readily
at small scene counts, where scene-centroid margins are wider — is stated in §3.4 and
untested.

**8.5 Accuracy sits in the weakly-supervised band.** Held-out Acc@GQA is 0.1667
[0.088, 0.243], comparable to weakly-supervised baselines and roughly 12 points below
current agentic methods (§6.1). Our claim is that cheap construction does not degrade
answer quality below that band, not that it advances the state of the art.

**8.6 The evaluation splits are constrained.** The NExT-GQA test half is burned: both
permitted touches are used, and no further test-half measurement is possible without a new
split. This binds future work on the answerer — which §6.1 identifies as a live target with
+0.18 of headroom — more than it binds the results reported here. Our MLVU evaluation is
150 questions across six tasks with videos capped at 600 s, and against uniform frame
sampling our retrieval advantage is positive but not statistically separated at n=120.

**8.7 Directional temporal reasoning fails, and the failure is not in the graph.** On MLVU,
Action Order and Action Count fall below chance with parse-failure rates of 4% and 0%, so
the answerer is confidently wrong rather than unparsed (§6.2). The same pattern appears on
NExT-GQA's before/after questions. Both require properties top-k retrieval does not
supply — relative temporal ordering and exhaustive coverage — and neither is addressed by
changing how the graph is built.

**8.8 Scene shortlisting bounds downstream recall.** Retrieval admits only the top
⌈√S⌉-scale scenes before frame-level ranking (§3.4), so a scene excluded at that stage
cannot contribute evidence downstream. Characterising the recall-versus-cost frontier of
that gate — including whether the current width rule is recall-safe at large S — is future
work and is not evaluated here.

**8.9 The CPU-only demonstration is weaker than it could be.** Our ingest measurements were
taken on a machine without a GPU. Demonstrating CPU-only operation *by choice* on
GPU-equipped hardware would be the stronger form of the claim, and has not been done.

**8.10 Some artifacts are not fully traceable.** <!-- open item 19 -> Appendix C -->

---

### 8.10 The construction saving is not yet measured on the evaluated configuration

Our construction result (§4) is measured on the complete-block edge configuration, in which
every intra-scene pair carries an edge. Every accuracy number we report (§6, §7.1) comes
from the tiered configuration, which shares the block structure but populates each block
sparsely — 6,284 edges against 23,571 at N=4,892 / S=528.

Block-diagonal construction does not close that gap on its own. The tiered mode selects its
motion and semantic neighbours by scanning all pairs and retaining a small top-k, so it pays
a quadratic construction pass irrespective of how sparse its output is. Bounding those scans
to the scene partition would plausibly transfer the saving, but it does not produce the same
graph: selecting top-k globally and then pruning cross-scene edges is not equivalent to
selecting top-k within scene, because a node whose best neighbours all lie in other scenes
currently retains none. The change therefore needs accuracy validation against the
526-question grounding gate, not a bit-identity proof.

We flag this as the most substantive open item in the paper rather than a minor extension.
It is the difference between a construction result about a configuration we evaluate and one
about a configuration we do not.

## 9. Conclusion

We set out to test whether a long-video retrieval graph must be expensive to build. It does
not.

Our central result is an identity: block-diagonal construction produces exactly the graph
that dense construction produces and then prunes — the same nodes, the same edges, the same
weights at zero tolerance, the same PageRank — while visiting only the pairs it keeps
rather than the 99.8% it would discard. That makes construction ~233× faster and ~6.8×
smaller in memory, on CPU, with no neural forward pass during ingest and no proprietary
model anywhere in the construction path. The same block structure separates query latency
into quadratic and sublinear regimes, and at large graph sizes into a boundary where dense
retrieval does not complete at all.

We have been deliberate about what this does not establish. The retrieval speedup does not
reach the user at the sizes we measured, because a constant-cost captioning stage dominates
the end-to-end path; our sparse retrieval pays a caption-cache penalty that dense retrieval
avoids precisely because dense retrieval ignores the query; and our answer accuracy sits in
the weakly-supervised band rather than at the frontier. We report each of these where it
arose.

The negative results proved the most useful part of the work. We pre-registered kill
criteria for two hypotheses we expected to confirm — that codec-derived boundary placement
beats content-blind placement, and that codec-based admission beats uniform sampling — and
both criteria triggered. Neither where we cut the video nor which frames we keep is doing
the work. **What is doing the work is the block structure alone**, which is also what makes
the result portable: a pipeline can adopt this construction without adopting our
segmentation, our admission policy, or our compressed-domain signals.

For retrieval over long video, the intermediate graph does not have to be generated, does
not have to be clever, and does not have to be expensive. It has to be structured — and
structure, it turns out, is nearly free.

<!-- open item 20 -> Appendix C -->

---

---

## Appendix A — Artifact index

### A.1 How to read this index

Every quantitative claim in the paper maps to a named artifact in `eval_results/`, a
harness script in `scripts/`, and — where recorded — a commit and configuration hash. We
list the seed and resampling procedure for every interval we report, since none of our
confidence intervals are analytic.

We also list what is *missing*. Section A.5 records provenance gaps we know about rather
than omitting them; a reproducibility appendix that reports only the traceable subset is
not a reproducibility appendix.

### A.2 Environment

**Machine.** CPU: AMD Ryzen 7 9800X3D, 8 physical cores / 16 logical. RAM: 31.16 GB.
OS: Windows 11 (`Windows-11-10.0.26200-SP0`).

**Runtime.** Python 3.12.10. torch 2.13.0+cpu (CUDA not available). numpy 2.5.1,
networkx 3.6.1, av 18.0.0, transformers 4.57.6. clip: pip reports version 1.0; the
package exposes no `__version__` attribute, so the version string is not independently
verifiable from the module. ollama 0.33.3.

**Decoder (caption path).** PyAV 18.0.0, linked FFmpeg libraries: libavutil 60.26.102,
libavcodec 62.28.102, libavformat 62.12.102, libavdevice 62.3.102, libavfilter 11.14.102,
libswscale 9.5.102, libswresample 6.3.102. No ffmpeg binary was found on PATH, in the
virtual environment, or in the repository; all decoding in the caption path goes through
PyAV's linked libraries above.

All measurements are CPU-only.

Answerer: `granite4:micro` (~3.4B, Q4_K_M) served via llama-server, temperature 0,
`cache_prompt=false`, `--parallel 1`. Embeddings: CLIP ViT-B/32, the same model at ingest
and query time. <!-- open item 22 -> Appendix C -->

All reported measurements are CPU-only on a single machine, and the memory watchdog
described in §5.2 sits near this machine's total RAM.

This environment was captured on the same machine after the reported runs rather than
at run time (captured 2026-09-08T18:43:07Z at commit `48c082a`, 2 tracked files dirty);
the full record is in `eval_results/env_A2.json`.

### A.3 Artifact index by section

### §4 — Construction cost

| claim | artifact | provenance |
|---|---|---|
| identity gate: 23,571 edges, 0 mismatches at tol 0.0, PageRank bit-identical, PPR identical across 5 seeds | `blockdiag_identity_gate_result.{json,md}` | commit `3d83b2c` on `origin/siddanth/peak-source-a6-p1`, present in the working tree **(verified)**; outcome `GATE_PASS`; all five §4.1 claims checked against the artifact and matching; harness `scripts/blockdiag_identity_gate.py`. Nuance: the 0.0 tolerance qualifier appears in the `.md` summary prose, not as a field in the JSON, which records a field-mismatch count of 0 without an explicit tolerance value |
| build savings, wall-clock: 50.195124 s → 0.214939 s (233.532×) | `build_cost_final.{json,md}`, `build_cost_final_raw/` | harness `scripts/build_cost_final_probe.py`; 159 builds (9 old, 150 new), arms interleaved old→new across 3 rounds, each arm looped in-process; guard `edge_count == 23,571` on all 159, PASSED; machine state snapshotted before each arm |
| build savings, peak RSS: 6,325,764,096 B → 930,693,120 B (6.80×) | `build_dedup_repeats.{json,md}` | harness `scripts/blockdiag_build_probe.py`; 5 repeats per arm, one build per fresh process, 10 launches, medians reported; guard `edge_count == 23,571` on all 10, PASSED. Measured separately from wall-clock because a peak RSS over a 50-build in-process loop is a peak over the loop, not over a build (§4.2) |
| RSS ratio cross-check: 6.803870× | `build_dedup.json` (`rss_ratio_after_dedup`) | independent post-dedup figure; agrees with the median-based 6.80× to three significant figures |
| **supersession chain for the build-savings figures** | see below | recorded because no single artifact states it |

The build-savings numbers were measured four times. Only the last is reported in §4.2; the
earlier three are superseded and appear in artifacts that are still on disk, so the chain
is recorded here to prevent a reader reconciling them incorrectly.

| # | figure | artifact | status |
|---|---|---|---|
| 1 | 69.472 s → 0.316 s = 219.88× (~220×); 6.360 → 0.950 GB = 6.698× (~6.7×) | `blockdiag_gate_and_savings_summary.md`, `blockdiag_probe_{old,new}.json` | **superseded** — measured before the redundant edge-and-PageRank pass was removed from both arms |
| 2 | 61.392 s → 0.237 s = 259.23×; 6.324 → 0.930 GB = 6.804× | `build_dedup.{json,md}` | **superseded for wall-clock** — single fresh-process probe per arm; its RSS ratio survives as the cross-check above |
| 3 | 48.568233 s → 0.122421 s = 396.73×, pairing spread [224.17×, 406.67×] | `build_dedup_repeats.{json,md}` | **superseded for wall-clock** — spread dominated by the block-diagonal arm's sub-quarter-second timer noise; its **RSS medians are the reported figures** |
| 4 | 50.195124 s → 0.214939 s = 233.532× | `build_cost_final.{json,md}` | **reported in §4.2** — looped and interleaved to clear the timer floor and symmetrise machine drift; contains no RSS measurement |

Cache used by (1)–(3): `eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz`,
N=4,892 / 528 scenes, `config_snapshot` sha256 `b61eb07f19b4035708778205f0d94885cb28fafdf5e3681e795f63f34dd72bb5`;
HEAD before the dedup change `90ef59b` on `siddanth/peak-source-a6-p1`.

<!-- open item 23 -> Appendix C -->

| grounding gate: 526 questions, 0 mismatches | `blockdiag_grounding_gate_result.{json,md}` | commit `3d83b2c` on `origin/siddanth/peak-source-a6-p1`, present in the working tree **(ledger)** — contents not yet read |
| pair-visit counts: 11,963,386 dense vs 23,571 block-diagonal | `virat_smoke_N4892_{flat,scenesparse}.json` | `flat_edge_count: 11963386`, `edge_count_matches_theoretical: true`; `scene_sparse_edge_count: 23571`, `block_diagonal_exact: true` |
| ingest: 0 neural forward passes; 2.945 s / 530 MB over 32 videos | `Iris-ucfvad/tuning/ucfcrime_vad_exp1/efficiency_measurements.json` | **(ledger)**; `tuning/ucfcrime_vad_exp1/efficiency_measurements.json` at `origin/siddanth/ucf-vad-exp1` |

### §5 — Query scaling

| claim | artifact | provenance |
|---|---|---|
| exponents: dense 2.063 [2.020, 2.125]; sparse 0.834 [0.801, 0.884] | `scaling_curve_v3.{json,md}` | git HEAD `9c66393d…` (dirty, 94 changed files); committed at `90ef59b`; text queries, 50/clip; bootstrap seed 42, B=10,000, clip-level within-bucket resample + refit |
| full-range fits: dense 1.980 [1.951, 2.006]; sparse 0.497 [0.461, 0.530] | same | same |
| censoring: dense timed out at N=6,559 and N=13,506 under 3600 s / 29 GB | same | censoring table in artifact; lower-bound fit k ≥ 2.597 |
| survivor census: 31 clips, median N=333, bins of 7/2/2/1 | `_ci_survivor_census.json`, `scaling_curve_ci_census.json` | harness `scripts/_ci_survivor_census.py` |
| shortcut guard violations: Assault036 20%, Abuse037 10% | `scaling_curve_v3.md` guards section | — |
| retrieval mechanics 7.9448 s vs 0.0072 s (1,104×) | `virat_latency_N4892_{raw.json,result.md}` | run 2026-07-27; 50 **synthetic** queries, `query_seed=20260726`; shortcut fired 0/50; **see A.5.1, A.5.2**; measured under synthetic sampled embeddings, not real CLIP text encoding — real text queries give ≈291× (§5.3) |
| **end-to-end (primary) 55.64 s vs 69.40 s (0.802×)** | `e2e_stage3_decomp_raw.{json,md}` | commit `deeeba8` (`tracked_dirty_count` 0); 5 real text queries/arm + discarded warmup; caption cache reset per arm; captioner pinned in harness config, resolved identity not recorded |
| end-to-end (prior) 50.91 s vs 65.45 s (0.778×) | `e2e_speedup.{json,md}` | git HEAD `90ef59bb…` (dirty, 87 changed files); 5 real text queries/arm + discarded warmup; caption cache reset per arm |
| caption delta +9.30 s [3.52, 16.22] | `caption_stage_diagnosis.{json,md}`, `_stdout.log` | 20 queries/arm; paired clip-level bootstrap, seed 42, B=10,000; top_k byte-identical (30 frames on all 40 queries) |

### §6 — Correctness floor

| claim | artifact | provenance |
|---|---|---|
| Acc@GQA 0.1667 [0.088, 0.243], n=120/27 videos | `P_NOWA_accgqa_result.md`, `P_NOWA_accgqa_raw.json` | commit `dc59de3`; second and final test-half touch; frozen config listed in `P_NOWA_accgqa_prereg.md` |
| split: 59 val / 27 test videos, video-level | `P_NOWA_split_declaration.md` | declared 2026-07-22, before any measurement |
| P(correct \| grounded) 0.476 vs 0.321 | `P_NOWA_accgqa_result.md` | — |
| vs uniform +0.120 [0.000, 0.248]; vs random +0.127 [+0.036, +0.229] | same | all three arms run together to avoid a third touch |
| MLVU M-Avg 0.340, per-task table | `MLVU_codec_baseline.{json,md}` | config hash `e67562fb00850bd1`; seed 42; **commit not recorded — see A.5.3** |
| competitor figures | primary sources | Xiao et al. Table 3; MUPA Table 1 (**not** its abstract); LangRepo Table 5; SeViLA grounding via the benchmark paper's reproduction |

### §7 — Negative results

| claim | artifact | provenance |
|---|---|---|
| 4-arm segmentation, ≤600 s: 0.340 / 0.393 / 0.320 / 0.353 | `MLVU_ablation.{json,md}` | seed 42, 150 questions, 145 videos |
| long-video: +0.088 [−0.009, +0.204]; AR +0.176 [0.000, 0.353]; PQA +0.000 | `MLVU_ablation_long_trimmed.{json,md}` | repo HEAD `9c66393d…` (dirty); question-set hash `67647a0ca98f73f6`; video-clustered bootstrap seed 42, B=10,000; premise guard median 581 scenes/video |
| R0: uniform saturates coverage at 5%; matched-budget nulls | `Iris/r0_full/summary.md` | **verified against artifact** — `r0_full/summary.md` at `origin/sonu/audit-docs`; §7.2 figures checked: M1 = 1.0000 for uniform at every swept budget down to 5%, dM2 +0.256 [−0.236, +0.710], dM3 +0.780 [−0.005, +1.753], mean(M2 − retention) = +0.0017 (sd 0.0099) — all match; contrast is Arm A (production, `is_retained_tier`) vs Arm C (uniform) at Arm A's natural retention of 10.54% |
| T5 shot geometry null; post-hoc dispersion diagnostic (R0 arm 8.4% distinct/mean run 13.19 vs uniform 100%/1.00) | `_shotbucket/run/summary.md` @ `origin/sonu/t5-shotbucket` | verified against artifact |

### A.4 Determinism and seeds

| purpose | seed |
|---|---|
| all bootstrap resampling | 42 (B=10,000; B=3,000 for the per-type decomposition) |
| MLVU question sampling | 42 |
| VIRAT synthetic query generation | 20260726 |
| R0 bootstrap | 20260805 **verified against `r0_full/summary.md` at `origin/sonu/audit-docs`** |

Codec-mode scene assignment was verified deterministic across independent parse+build runs
(1,812 assignments, 174 distinct scenes, identical). Cache non-mutation was fingerprinted
before and after the VIRAT latency run for both the NExT-QA and VIRAT caches.

### A.5 Known provenance gaps

**A.5.1 — The dense arm in the N=4,892 latency comparison did not run its own ingest.**
The artifact records that the flat graph was built in memory from the same loaded
scene-sparse frames, to avoid a second ~13-minute ingest pass, and states plainly that this
*"has not been independently confirmed"* not to advantage either mode. Since this run
produces the 1,104× retrieval-mechanics figure, the caveat travels with that number,
which is measured under synthetic sampled embeddings, not real CLIP text encoding — real
text queries give ≈291× (§5.3).

**A.5.2 — The N=4,892 clip is mpeg4, not H.264.** The ingest log for the VIRAT clip warns
that the codec is not h264/hevc and that motion-vector export may be unavailable. This does
not affect the construction identity result (which is graph algebra and codec-independent)
or the latency measurements (which time graph mechanics). But our flagship clip is one on
which codec-derived motion signals may be degraded or absent, and a paper about
compressed-domain signals should say so rather than let a reader discover it.
<!-- open item 25 -> Appendix C -->

**A.5.3 — The MLVU baseline artifact records no commit.** Its git HEAD field contains a
captured error from a failed `git rev-parse`, and its dirty flag is null. The config hash
is present. <!-- open item 26 -> Appendix C -->

**A.5.4 — Some artifacts are still quoted from a ledger rather than read directly.** The R0
outputs, T5 outputs, and ingest efficiency measurements are reachable — R0 at
`origin/sonu/audit-docs` (`r0_full/`, `r0_smoke/`, `_r0check/`), T5 at
`origin/sonu/t5-shotbucket` (`_shotbucket/`), and the ingest efficiency measurements at
`origin/siddanth/ucf-vad-exp1` (`tuning/ucfcrime_vad_exp1/efficiency_measurements.json`,
added in commit `41d7205`). The R0 and T5 figures (§7.2, §7) have since been read from and
checked against their artifacts and are marked verified above. The identity-gate result
(§4.1) has since been read from and checked against its artifact and is marked verified
above. What remains ledger-quoted is the ingest efficiency measurement and the
grounding-gate result — those artifacts have not been read. No T6 artifact exists at all,
which is a separate matter from reachability (see item 15). <!-- open item 27 -> Appendix C -->

**A.5.5 — Most runs were made from dirty working trees.** The scaling and end-to-end
artifacts record 94 and 87 changed files respectively at run time. The changes are scratch
and evaluation outputs rather than pipeline code, but a commit hash plus "dirty" does not
uniquely identify the code that ran. <!-- open item 28 -> Appendix C -->

**A.5.6 — One headline artifact was promoted rather than re-run.** The N=4,892 latency
result document was written by promoting pre-existing artifacts (run 2026-07-27); the
harness was not re-executed at documentation time, and provenance rests on artifact
timestamps rather than a commit.

**A.5.7 — `captioner_backend` is inert, and one artifact's recorded value misdescribes
what ran.** `iris/aria.py::get_captioner()` takes no configuration argument. It ignores the
`IRISConfig` each harness constructs and performs its own `ConfigManager().get_config()`,
which loads `configs/default_iris_config.json`. That file has specified `minicpm`
unchanged across every commit examined, and none of the eight evaluation harnesses calls
`aria.set_captioner()`. Consequently the `captioner_backend` field on each script's config
object is validated but never consumed, and **every artifact in this paper was produced by
the MiniCPM captioner** — one captioner throughout, which is at least internally
consistent.

Two corrections follow. `MLVU_codec_baseline.json` records `captioner_backend:
"moondream"`; that value was never in force and should be read as an artifact of the dead
field, not as a description of the run. And because `config_hash` incorporates a field
with no runtime effect, two runs with different hashes may share an identical captioner
path. We report this rather than silently correcting it, because a reader reproducing from
the recorded configuration would otherwise be misled.

**A.5.8 — Build-cost measurement protocol.** The §4.2 figures come from an interleaved
protocol rather than a single run, and the reason is worth recording. Single-run
measurements of the same code varied between 224× and 406×: the block-diagonal arm
completes in ~0.2 s, close enough to timer resolution that its run-to-run spread dominated
the ratio, and the dense arm drifted 30–40% *across* CLI sessions while remaining tight
within any one session. The protocol fixes both — 159 builds, arms alternated across three
rounds, machine load and free memory logged before each arm, and the fast arm timed over
50 consecutive in-process iterations. Under it the dense arm varied 0.177% across rounds
and the four ratio estimates agreed within 0.4%. Raw per-round, per-arm outputs are
preserved in `eval_results/build_cost_final_raw/`.

### A.6 Reproduction

<!-- open item 29 -> Appendix C -->

---


---

## Appendix B — Pre-registration record

### B.1 Protocol

Every experiment reported in this paper that could have gone more than one way was
registered before the data were seen: the primary quantity, the decision rule, and — where
applicable — the prediction. We record them here in full, including the cases where the
prediction was wrong and the cases where the criterion killed a hypothesis we expected to
confirm.

The protocol we followed throughout:

1. Parameter selection happens on a validation split only.
2. Parameters are frozen; the frozen configuration is written down.
3. The held-out split is measured **once**. That number is reported.
4. No tuning after the result. If a split is measured more than once, it is burned and
   labelled as such.

**Registration strength is not uniform across these entries, and we say so rather than
letting the appendix imply otherwise.** Entries B.3–B.6 have standalone pre-registration
documents written and dated before their runs. Entries B.7–B.11 were registered in the
task specification that commissioned each run, with the criterion restated verbatim in the
resulting artifact — a weaker form, since specification and execution were closer together
in time. We mark each entry accordingly. The distinction does not affect whether the
criteria bound our reporting, which they did in every case, including the two where the
registered prediction failed.

### B.2 Summary

| # | test | registered | form | criterion | outcome |
|---|---|---|---|---|---|
| B.3 | NExT-GQA val/test split | 2026-07-22 | standalone | video-level split declared before any measurement | held |
| B.3a | Held-out grounding | 2026-07-22 | standalone | report whatever returns; no tuning after | **prediction not borne out**, reported |
| B.4 | Held-out Acc@GQA | 2026-07-22 | standalone | report whatever returns; no tuning after | **prediction failed**, reported |
| B.5 | Retrieval funnel | <!-- open item 30 -> Appendix C --> | standalone | CI-thresholded action rule, fixed in advance | read as registered |
| B.6 | R0 admission gate | <!-- open item 31 -> Appendix C --> | standalone | kill criterion on matched-budget control | **triggered** |
| B.7 | Scaling exponent CIs | 2026-08-15 | in-spec | CIs disjoint **and** sparse upper bound < 1.0 | **both held** |
| B.8 | Segmentation ablation | 2026-08-14 | in-spec | codec−fixed CI must exclude zero; tie counts as failure | **failed** |
| B.9 | End-to-end verification | 2026-08-15 | in-spec | if e2e materially below claim, restate around construction | **triggered** |
| B.10 | Caption-stage diagnosis | 2026-08-15 | in-spec | if paired CI includes zero, declare noise and stop | proceeded; hypothesis rejected |
| B.11 | Shot-bucketing selection | 2026-08-18 | in-spec | three-branch read: beats / ties / underpowered | **underpowered** |

*Form:* **standalone** = a dated pre-registration document written before the run;
**in-spec** = registered in the commissioning task specification, criterion restated in the
result artifact (see B.1).

### B.3 NExT-GQA validation/test split (2026-07-22)

Declared before any measurement on the pool of 526 grounded questions over 86 cached
videos. The split is **video-level** — no video appears in both halves — because questions
share videos and question-level splitting would leak. Validation: 59 videos, 406 questions,
where all tuning occurs. Test: 27 videos, 120 questions, untouched by prior tuning. The
asymmetric ratio was deliberate: tuning needs power, the reported number needs cleanliness.

Comparability was measured *before* declaring, and two biases were registered in advance
along with the direction each would push:

- Test has fewer multi-interval questions (6.7% vs 11.3%) → slightly **easier**.
- Test has more short gold spans (20.8% vs 17.0%) → slightly **harder**.
- These partly offset; net direction declared unknown.

The expectation was registered too: the held-out number would likely fall *below* the
in-sample 0.3426, because the tuned parameter had been selected in-sample on 64 questions —
and that **a drop is the number becoming real, not a regression.** Expected precision of
±0.08–0.11 was accepted in advance as the cost of a clean estimate.

### B.3a Held-out grounding (2026-07-22) — *the prediction also did not hold*

The split declaration in B.3 registered an expectation as well as a design: the held-out
grounding number would likely fall *below* the in-sample value, because the tuned
parameter had been selected in-sample.

**It did not.** Test came in above validation — mIoP 0.3349 against 0.3126.

The honest reading is the artifact's own: the confidence intervals overlap heavily, so
validation and test are **consistent**, not test-is-better. The direction is attributable
to a bias declared in advance (B.3): the test half contains fewer multi-interval questions
(6.7% vs 11.3%), which are structurally harder to capture with a single predicted span.

We record this alongside B.4 because the two together make the point that one alone would
not. Two registered predictions, neither borne out — one low, one high — both reported
without adjustment, and neither followed by a change to the frozen configuration.

### B.4 Held-out Acc@GQA (2026-07-22) — *the prediction failed*

Registered with a full frozen-configuration listing and an explicit second-touch
disclosure: the test half had been used once for grounding, and this run measured
additional metrics on the identical frozen configuration, with no selection occurring
between. All three arms (proposed, uniform, random) were run together, because running
baselines later would have constituted a third touch. The registration states: *no further
test-half measurement without a new split*, and *whatever Acc@GQA comes back as, that is
the number.*

**The registered prediction was Acc@GQA ≈ 0.22**, from IoP@0.5 (0.3500) × P(correct |
grounded) (~0.65, estimated in-sample at N=64).

**The measured result was 0.1667.**

The prediction failed because its second factor did not survive out of sample: held-out
P(correct | grounded) is 0.476, not 0.65 — sitting at the floor of the in-sample estimate's
interval, so that estimate is not contradicted, but the conclusion drawn from it does not
hold. The consequence is recorded in §6.1: the answerer is not a negligible stage, and
work on it returns to the critical path. Because the registration forbade tuning after the
result, and because the test half is now burned, that work cannot be validated held-out
without a new split.

We report this failure prominently because it is the clearest demonstration that the
registration was binding rather than decorative.

### B.5 Retrieval funnel diagnostic

A read-only diagnostic on validation only, selecting nothing. Three strictly nested
membership quantities were defined in advance (index coverage ≥ pool coverage ≥
peak-in-gold), together with an action rule keyed to the **confidence interval, not the
point estimate**:

- Selection headroom CI upper ≤ 0.05 → the anchor is near-optimal given the pool; do not
  build caption reranking.
- CI lower ≥ 0.15 → material headroom; in-pool caption reranking is the highest-value cheap
  experiment.
- Otherwise → inconclusive; take no action either way.

A stop condition was registered alongside: the run selects nothing, and no frozen value,
default, or threshold may change as a result. Five limitations were declared in advance,
including that the diagnostic is conditional on one ranking mode and that its
`best_gold_rank` quantity bounds only the headroom reachable by a *different* signal.

<!-- open item 32 -> Appendix C -->

### B.6 R0 codec-admission gate

The kill criterion — that coverage metrics are uninformative without a budget-matched
control, and that a saturating uniform arm would invalidate the coverage framing — was
registered before the run and **triggered**: uniform admission reaches the coverage ceiling
at every budget swept, down to 5% retention. The matched-budget contrast was then reported
as a null, with intervals spanning zero (§7.2).

<!-- open item 33 -> Appendix C -->

### B.7 Scaling exponent confidence intervals (2026-08-15)

Registered before fitting: the exponents would be considered printable **only if** the
dense and sparse confidence intervals were disjoint **and** the sparse interval's upper
bound fell below 1.0 (so that "sublinear" is supported rather than assumed). Bootstrap
procedure, seed, and resampling unit were fixed in advance (clip-level, within N-bucket,
refit per resample; seed 42, B=10,000).

**Both conditions held**: dense 2.063 [2.020, 2.125], sparse 0.834 [0.801, 0.884]. Had they
not, the registration specified the fallback — lead on tractability divergence and
construction cost instead of a clean exponent.

### B.8 Segmentation ablation (2026-08-14) — *criterion failed, hypothesis retired*

The hypothesis was that codec-derived boundary placement produces a better graph than
content-blind placement at matched segment count. Registered in advance:

- The comparison would be run at matched segment count, with retrieval, answerer, question
  set, and seed held constant.
- The codec-minus-fixed difference would have to **exclude zero** to support the
  hypothesis.
- **A tie counts as failure** — the burden of proof sits with the codec arm.
- A premise guard was registered: if long videos did not produce substantially more scenes
  than the short-video run, the test would be inconclusive on mechanism regardless of the
  accuracy outcome.

The premise guard **passed** decisively (median 581 scenes per video against single digits).
The primary criterion **failed**: +0.088 M-Avg, 95% CI [−0.009, +0.204]. The point estimate
moved in the predicted direction (from −0.053 on short videos), which is why the criterion
mattered — a directional move alone would have been easy to over-read.

It was further registered that this would be the **last test of the hypothesis**, and that
a failure would not be followed by re-running at new seeds, budgets, or subsets. It has not
been.

### B.9 End-to-end verification (2026-08-15) — *decision rule triggered*

Registered before measuring: if the end-to-end speedup came back materially below the
figure the project had been carrying, the efficiency claim would be **restated around
construction cost and tractability divergence rather than query speedup**, and the
retrieval-mechanics ratio would be labelled as such wherever it appeared.

The rule triggered. A prior-artifact search found no measurement backing the carried
end-to-end figure at all; the fresh measurement returned 0.778×. The paper's framing was
changed accordingly (§1, §5.3, §9). The decomposition run (commit `deeeba8`) has since been
promoted to the primary §5.3 measurement at 0.802×, and the 0.778× figure recorded here is
retained as the prior measurement.

### B.10 Caption-stage diagnosis (2026-08-15)

Registered in two stages. First a power condition: the effect would be re-measured at
n=20 per arm, and **if the paired confidence interval included zero, the discrepancy would
be declared noise and the investigation stopped.** It did not — +9.30 s [3.52, 16.22] —
so the causal analysis proceeded.

Second, an interpretation rule fixed in advance: if the effect proved real and
dispersion-driven, it would be disclosed in the paper as a genuine system cost of sparse
retrieval; if it proved an artifact of the measurement harness rather than the method, it
would be scoped as an implementation detail.

The registered leading hypothesis — frame dispersion — was **tested and rejected** (r =
−0.058). The mechanism found instead was cross-query cache locality, which is a production
behaviour rather than a harness artifact, so the disclosure rule applied (§5.4).

### B.11 Shot-bucketing selection (2026-08-18) — *underpowered branch*

Registered with a three-branch read — beats uniform / ties uniform / underpowered — and an
advance acknowledgement that the 19-video annotated corpus resolves effects of roughly one
percentage point at best.

The primary contrast landed in the **underpowered** branch, with half-widths exceeding the
effect size of interest, and was reported as such rather than as a tie. A secondary
contrast (shot geometry alone, no score) landed as a **properly powered null**, with
half-widths excluding a one-point effect. The registration's own guidance — do not re-run
at new seeds or budgets to hunt for significance, and do not report the positive lean of an
underpowered contrast as a result — was followed.

---


## Appendix C — Open items

Every bracketed editorial note from the working drafts, collected. Numbering matches the inline markers.


**02_related_work.md**

1. we expose no time-window query interface — retrieval is CLIP-text-driven end to end (§3.4) — so the interval case above describes what the representation admits, not a measured capability. Either scope the sentence that way in print or add the interface; it must not read as an evaluated result.

RESOLVED — the §2.2 time-bounded query example now carries an explicit scoping clause: "We expose no time-window query interface, however — retrieval is CLIP-text-driven end to end (§3.4) — so this describes what the representation admits, not a measured capability." Added in place, without restructuring the paragraph or altering the EgoSG comparison.

2. do not convert the Vgent and IRIS construction figures into a ratio. Theirs is seconds per minute of video; ours is a graph build at a given survivor count, and our ingest figure is per video over a 32-video sample. A comparison needs the VIRAT clip durations and a matched denominator, or it becomes the phantom 4.6× again.

RESOLVED — checked, guard held. The draft nowhere converts the Vgent and IRIS construction figures into a ratio. The two are stated side by side once (§2.2), where the text explicitly declines the comparison on the grounds that the denominators differ and frames the claim as being about the class of model each construction requires rather than the seconds. The only multiplier near Vgent in the draft is its own reported 1.73× against Video-RAG, internal to Vgent's paper. Any future edit adding a Vgent-vs-IRIS ratio must first supply the VIRAT clip durations and a matched denominator.

3. EgoSG and Vgent are two systems, not a family. One more would let §2.2 speak about a line of work rather than a pair. Unverified candidates from a literature pass: EGAgent (arXiv 2601.18157, temporally annotated entity scene graphs, egocentric), GraphVideoAgent (ACM MM 2025, entity-relation graphs, 8.2 frames average on EgoSchema/NExT-QA), and MemDreamer (arXiv 2606.07512, hierarchical graph memory). None read from source; verify scope before citing.

4. verify the two NG+ variants against Xiao et al. Table 3 directly — currently taken from the benchmark table via our own ledger.

5. cite the classical line from primary sources; the lineage above is taken from AdaCodec's related-work survey (arXiv 2606.02569) and has not been verified against the original papers


**03_method.md**

6. the salience weights behind the §3.1 run are not recoverable from its artifact.** The figures come from the VAD tuning tree, whose `frozen_config_used` block records (`packet_size_weight`, `motion_weight`, `luma_entropy_weight`) = (0.8, 0.1, 0.1). That schema is not `IRISConfig`'s — `IRISConfig` has no `packet_size_weight` — so the block is the experiment's own record of intended settings rather than a dump of the object actually passed, and it cannot tell us what the scoring stage received. If those names were passed through, `packet_size_weight` was silently dropped and `luma_diff_weight` defaulted to 0.5 while the other two took 0.1, giving an effective (0.5, 0.1, 0.1) against the production (0.5, 0.3, 0.2). The wall-clock and forward-pass results do not depend on the weights and stand either way; the recorded 10.49% mean retention does, and should not be quoted as a production figure until an ingest is re-run with the weights logged as resolved. This is the third dead config field found in this codebase, after `captioner_backend` on harness configs and `beta` outside `ranking_mode="legacy"`; Appendix A should say that recorded-but-unconsumed fields are a systematic hazard here, and that a `config_hash` covering them records noise.

7. state the weight formulas for the semantic / motion / temporal components, and whether any depends on cross-scene context (if any did, the identity result would be non-trivial in a way §4.1 should say explicitly).

8. measure the distribution of surviving motion/semantic edges per node under `scene_sparse` — cheap, and it either closes this or turns it into a finding.

9. α and β do not have a single value in this repository.** `L2Asphodel.__init__` hard-codes `alpha=0.4, beta=0.6`; `IRISConfig`'s dataclass declares `beta=0.3`; `configs/default_iris_config.json` sets `beta=0.6`; and the config in the `ucf-vad-exp1` worktree sets `beta=0.3`. Since a run adopts `IRISConfig`'s value only when a config object is passed, and the two shipped configs disagree, dense-arm edge weights differ between the two trees. Determine which tree produced the §5.1 scaling corpus before the appendix claims a single configuration. This does not affect §4.1's identity gate, where both paths share whatever β was in force.**


**04_05_construction_and_scaling.md**

10. formal statement in §3.3 — Σ_s |S_s|² vs N², with the balanced-scene case. RESOLVED. The balanced-scene case is stated in §3.3, with the S=1 and uneven-partition boundaries, and the realised 0.197% checked against 1/S = 0.189% at N=4,892, S=528.

11. state whether the noise-floor explanation was verified directly or is inferred from the fit; if inferred, say so.

RESOLVED — §5.1 now states plainly that the noise-floor account is inferred from the shape of the fit (the shallower small-N slope and the full-range fit's lower R², 0.898 vs 0.983), not verified by direct measurement. No artifact in eval_results/ records a direct noise-floor measurement — no repeated small-N runs and no fitted overhead constant exist. `scaling_curve_v2_textquery_NOTES.md` lists "characterize the small-N noise floor directly" under a follow-up run explicitly marked NOT started. The R² figures and the subsequent sentence on taking the N ≥ 1102 fit as the conservative headline are unchanged.

12. — decide which run is canonical for §5.3. The decomposition run has strictly better provenance: clean checkout, recorded commit, the configured captioner, and a component breakdown. The table above has the advantage of being the figure already circulated. They agree on direction and disagree on absolute stage times by roughly 4 s. Our recommendation is to promote the decomposition run to the primary table and retain 0.778× as the corroborating prior measurement — but this changes a headline number, so it is not our call. **RESOLVED.** The decomposition run (commit `deeeba8`, `tracked_dirty_count` 0) was promoted to the primary §5.3 table at 0.802×. The 0.778× run (git HEAD `90ef59b`, 87 dirty files, BLIP captioner, unrecoverable selection mechanism) is retained as the prior measurement. The two used different captioners and corroborate direction only, not magnitude. The promotion was propagated to §1, §8.1, and the Appendix A artifact index, which now carries rows for both runs.

13. decide whether to report the projection at all, or only the measurement plus its assumptions. Arguments both ways; a reviewer may reasonably regard a single-N extrapolation as unsupported. **RESOLVED.** The end-to-end crossover projection was CUT. Reason: it assumes L equal across arms, contradicted by the same run's data (39.9 s vs 62.8 s), and returns ~1.15× at N=4,892 against a measured 0.778×. The retrieval-only crossover at N≈24 is retained. The cut covered three locations — §5.3, §8.1, and the Appendix A artifact index — and any future session finding a crossover figure elsewhere should remove it rather than reinstate the projection.


**06_correctness_floor.md**

14. cross-reference precisely once the companion analysis is available; do not import its numbers into this paper.

RESOLVED — checked, no edit needed. The §6.2 passage names the failure mode and its locus (directional-temporal questions, answerer rather than retrieval) and imports no number, metric, or result from the companion analysis. It contains no reference to the companion work at all. If a citation becomes available before submission, add it there; do not import figures with it.


**07_negatives.md**

15. verify from the T6 artifact; report-quoted.

RESOLVED — CLAIM WITHDRAWN, NOT VERIFIED. No T6 artifact exists. A search of 200 commits
across all branches on both remotes found the figures 83.4% distinct instants, mean run
1.22, and "naive top-k 7.8% / 13.5" nowhere outside the draft itself and a document quoting
the draft. What exists is a task specification for a candidate T6 study in
CLAUDE_CODE_TASKS.md on origin/sonu/t5-shotbucket, gated on a Step 1 that was never run.

The only real measurement of this quantity is a post-hoc, explicitly not-pre-registered
dispersion diagnostic in _shotbucket/run/summary.md, and it points the other way: the codec
action-score top-k arm — the R0 selector at matched budget k=503 — sampled 42 distinct
locations (8.4%) with mean run length 13.19, against uniform's 503 (100%) and 1.00. The
withdrawn paragraph had attributed the clumped profile to a "naive top-k arm" and a
de-clumped profile to "production", inverting which arm clumps.

§7.2 has been rewritten to report the real diagnostic and to state plainly that the
clumping objection is not disposed of. The Appendix A row no longer carries the T6 claim.

16. verify from the T5 artifact; report-quoted.

RESOLVED — VERIFIED AGAINST ARTIFACT. The artifact is reachable at origin/sonu/t5-shotbucket, path _shotbucket/run/summary.md. All six cited figures match exactly: the shot-geometry contrast against uniform is the H_shot1_mid row (dM2 +0.001 [−0.416, +0.421], half-width 0.419, TIES; dM3 +0.141 [−0.608, +0.972], half-width 0.790, TIES), and the shot-plus-score half-widths 1.69 and 2.79 are the F_shot1_action primary contrast's 1.691 and 2.793, both labelled UNDERPOWERED in the artifact as in the paper. The prose now names the specific arm; the artifact's parallel two-frame-per-shot midpoint arm (I_shot2_mid) is not cited here and its dM3 is underpowered rather than a tie.

17. one sentence tying this forward to §9 — the conclusion should collect this rather than restate §4.

RESOLVED — §7.3 now ends with one added sentence: "§9 collects this portability point rather than restating §4." §9 already states the portability point (lines 1443–1445 pre-edit), so no other change was needed.


**08_09_limitations_conclusion.md**

18. decide how far to forward-reference the companion analysis. Recommendation: name the failure mode and its locus, cite nothing unpublished, and import no numbers.

RESOLVED — the recommendation was taken, and the text already complies. §8.7 names the failure mode and its locus, cites nothing unpublished, and imports no numbers. A grep of the whole draft for companion-work references ("companion", "forthcoming", "in preparation", "under review", "separate paper") returns hits only inside Appendix C's own editorial notes — none in the body, Appendix A, or Appendix B.

19. state plainly whichever of the following remain true at submission: the MLVU baseline artifact records no git commit; the salience-weight provenance question (§3, note 1) is unresolved; the R0/T5 artifacts are not reachable from all authors' machines. If they are fixed by then, delete this item rather than softening it.

20. one or two sentences of future work, once the team settles what is in scope for the companion paper. Candidates that do not depend on that decision: the shortlist recall-cost frontier (§8.8), build-time caption prefetch (§8.2), and additional large-N clips (§8.3).


**10_appendix_A_artifact_index.md**

21. state CPU model, core count, RAM, OS, Python version, and torch version. None of this is recorded in the artifacts and it must be captured before submission — a reviewer cannot interpret a wall-clock number without it. **RESOLVED.** A.2 is now populated from `eval_results/env_A2.json`.

22. llama-server build — `b9976` is recorded in one prereg; confirm it is the build used for every reported run.

23. `build_cost_final.md` records no commit hash. Stamp one, or state that the reported build figures were measured on an uncommitted tree — see A.5.5.

24. either confirm the two paths produce equivalent inputs, or state this caveat wherever 1,104× appears. Recommendation: state it — §5.3 already labels the figure as retrieval-mechanics-only, and one more clause is cheap. RESOLVED. The recommendation was taken — the caveat is stated wherever the figure appears rather than claiming path equivalence. The two paths are not equivalent: the same measurement under real CLIP text queries gives approximately 291x against 1,104x under synthetic sampled embeddings, and that gap is itself the evidence.

25. confirm what the codec path actually did on this clip, and add one sentence to §3 or §4.

26. re-stamp or re-run before submission.

27. RESOLVED, PREMISE FALSE. This item asserted a repository-history blocker preventing these artifacts from being pushed. Verified 2026-09-07 in the working repository (213 commits): the commit and 156 MB archive named in the originating team report do not exist in this tree; the largest object in history is 21.7 MB, below GitHub's 100 MB per-file limit; and `main` was already synchronised with `origin/main`. No history rewrite was needed or performed. The reachability limitation in A.5.4 is unaffected and stands as written — the R0, T5/T6, and ingest-efficiency artifacts remain unreachable from all authors' machines, which is a question of where those files live, not of repository history. See also items 19 and 33.

28. decide how to handle this. Options: re-run the headline measurements from a clean tree, or state the limitation and provide the harness config hashes — `scaling_curve_v3` already records per-script SHA-1 hashes, which is the stronger practice and should be extended to the other harnesses.

29. write once A.5.5 is decided. It should state the minimal path: which scripts to run, in what order, against which cached indices, and which results are reproducible without re-ingesting video. The `index_cache/*.npz` dependency is no longer a blocker: those caches were built at (0.5, 0.3, 0.2), which is the shipped default configuration, not a deviation from it (§3 note 1, resolved).


**11_appendix_B_prereg.md**

30. date

31. date

32. recover the registration date, and state the realised outcome against the three-way rule.

33. registration date and the criterion's exact wording, from `Iris/r0_full/`. This artifact is not reachable from all authors' machines — see note 3.
