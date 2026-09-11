# Appendix C item 3 — third graph-video system for §2.2

Verified from source (arXiv/ACM abstract pages, HTML fulltext). No figures were
converted into a ratio against IRIS's own numbers — none of the three papers
below reports a construction-cost figure to convert, so the item-2 guard is
satisfied trivially, not by omission.

---

## 1. "EGAgent" — arXiv 2601.18157

**Identifier check: the ID resolves to a real, different-titled paper that
contains EGAgent as a named subsystem, not a paper titled "EGAgent."**

- **Actual title:** *Agentic Very Long Video Understanding*
- **Authors:** Aniket Rege, Arka Sadhu, Yuliang Li, Kejie Li, Ramya Korlakai
  Vinayak, Yuning Chai, Yong Jae Lee, Hyo Jin Kim
- **Venue/date:** arXiv cs.CV; v1 submitted 26 Jan 2026, v2 5 Mar 2026, v3
  (current) 3 Jul 2026
- EGAgent is the paper's entity-scene-graph component, evaluated on
  EgoLifeQA (57.5%) and Video-MME (Long) (74.1%). If cited, the citation must
  name the paper's real title, with EGAgent introduced as its subsystem — not
  cited as if "EGAgent" were the paper.

**Graph structure.** Nodes are entities typed as person / object / location.
Edges are relationship types — talks-to, interacts-with, mentions, uses —
each annotated with a start_time/end_time interval.

**Construction.** Built by an LLM-based extractor: "For each document
*d*∈𝒟, we apply an LLM-based extractor ℱ to jointly identify entities and
their relationships." This is a neural forward pass (an LLM call) but not a
per-frame vision-model pass — it runs over documents (captions/transcripts),
not raw frames directly.

**Construction cost.** Not reported. The paper gives no latency, FLOPs, or
token figure for graph construction. The only size figure is storage, not
cost: the resulting SQLite database for the 52-hour EgoLife dataset is
"approximately 2 MB on disk."

**Framing sentence (their own).** "[An] entity graph *G*=(*V*,*E*) to capture
relationships and interactions, enabling the planning agent to query this
graph during inference" — the temporal annotations are "crucial for
reasoning about events and interactions that unfold or repeat across long
horizons." This is close to a paraphrase of EgoSG's framing as a *symbolic
reasoning substrate* queried by an agent, and the domain (egocentric,
wearable-device video) matches EgoSG's HD-EPIC setting rather than Vgent's
general long-form setting.

---

## 2. "GraphVideoAgent" — ACM MM 2025 (no arXiv ID given; resolved)

**No arXiv ID was supplied for this one; it exists.** Correct identifier:
arXiv 2501.15953, published as *Understanding Long Videos via LLM-Powered
Entity Relation Graphs*, ACM Multimedia 2025 (DOI
10.1145/3746027.3755537), where the system itself is named GraphVideoAgent.

- **Authors:** Meng Chu, Yicong Li, Tat-Seng Chua
- **Venue/date:** arXiv cs.IR, submitted 27 Jan 2025; ACM MM 2025
  (Proceedings of the 33rd ACM International Conference on Multimedia)

**Graph structure.** Nodes are entities extracted from per-frame captions via
named-entity recognition and noun-phrase chunking; each node stores frame
indices, VLM-derived visual features, caption text, and a state-change
sequence. Edges are three typed relations — spatial (from prepositions:
"in," "on," "at"), interaction (from verbs like "talk," "meet"), and action
(from verbs like "open," "close," "hold") — built via dependency parsing of
the captions (the paper uses spaCy), storing relation type, temporal
information, and the source linguistic elements.

**Construction.** Node features draw on a VLM (visual features, captioning),
but the graph structure itself — the entities and typed edges — is built by
classical NLP (NER, noun-phrase chunking, dependency parsing over already-
generated captions), not by a single generative model call over the graph
as a whole. This is a meaningfully different construction pattern from
EgoSG (one frontier-model call per chunk produces the graph directly) and
Vgent (one LVLM call per clip extracts entities): here the relation
extraction step is rule-based/parser-based, with the neural cost front-
loaded into captioning rather than graph-building per se.

**Construction cost.** Not reported. The paper reports only end-to-end frame
counts used at query time (8.2 frames average on EgoSchema, 8.1 on NExT-QA)
and accuracy deltas — no construction latency or compute figure.

**Framing sentence (their own).** The graph is meant to enable "more
structured processing of video content," used to "guide [the agent's]
search, focusing on relevant temporal segments and entity relations" — a
retrieval-guidance framing closer to Vgent's than to EgoSG's, but built by a
different mechanism than either.

---

## 3. "MemDreamer" — arXiv 2606.07512

**Identifier check: correct.**

- **Title:** *MemDreamer: Decoupling Perception and Reasoning for Long Video
  Understanding via Hierarchical Graph Memory and Agentic Retrieval
  Mechanism*
- **Authors:** Cong Chen, Guo Gan, Kaixiang Ji, ZhaoYang Zhang, Zhen Yang,
  Guangming Yao, Hao Chen, Jingdong Chen, Yi Yuan, Chunhua Shen
- **Venue/date:** arXiv cs.CV; v1 submitted 5 Jun 2026, v2 24 Jun 2026

**Graph structure.** A three-tier hierarchy plus a per-event subgraph layer.
Tier 1 (video root): one node with title, description, themes, key entities,
time range. Tier 2 (super events): narrative-phase nodes with label,
description, time range, child references, deduplicated entities. Tier 3
(macro events): leaf nodes with per-episode summary, time range, child
micro-event IDs, top-k entities. Within each macro event, a subgraph of
entity nodes (person/object/location/group) and micro-event nodes (atomic
actions with temporal extent, subject, object, action description). Edges
are typed: spatial-attribute between entities (LOCATED_IN, NEXT_TO,
PART_OF...), subject-object between entities and micro-events (PERFORMS,
RECEIVES, USES_TOOL), temporal-causal between micro-events (BEFORE, CAUSES,
PREVENTS, SUBEVENT_OF), plus cross-tier SUBEVENT_OF and inter-episode
CAUSES/LEADS_TO/CONTRASTS_WITH edges.

**Construction.** A single frontier MLLM, Gemini-3.1-Pro, processes streaming
video in 10-minute windows using "streaming adaptive segmentation" to find
semantic boundaries, extracting local subgraph structures and summaries
directly; bottom-up aggregation then clusters macro events into super events
up to the video root. This is the pattern closest to EgoSG's (one frontier
model produces the graph structure directly, chunk by chunk) among the
three candidates.

**Construction cost.** Not reported. No latency, memory, or compute figure
for graph construction is given.

**Framing sentence (their own).** The graph captures "entities, events, and
their logical relations" in a "three-tier coarse-to-fine hierarchy,"
explicitly contrasted with "flat chunk-based storage" for preserving
"long-range dependencies" — a framing about hierarchical reasoning capacity,
not specifically about egocentric video or agent query-time retrieval.

---

## Recommendation

**Add EGAgent (cite the paper by its real title, *Agentic Very Long Video
Understanding*, Rege et al. 2026, with EGAgent named as its entity-graph
subsystem) as the third system in §2.2.**

Reasoning: of the three, EGAgent is the one whose own framing most directly
echoes the axis §2.2 already uses to describe EgoSG — a symbolic graph built
so an agent can query it at inference time, in the same egocentric,
wearable-video setting as EgoSG's HD-EPIC. MemDreamer is a defensible
runner-up (its construction pattern — one frontier model producing graph
structure directly — is actually closer to EgoSG's mechanically), but its
own framing is about hierarchical narrative structure for general long
video, not about an egocentric reasoning substrate, so it doesn't tighten
the "line of work" claim as cleanly. GraphVideoAgent is the weakest fit: its
edges are built by dependency parsing over captions, not by a generative
model call over the graph, which is a different construction pattern from
EgoSG and Vgent, not a third instance of the same one.

**Exact claim §2.2 could make about EGAgent, scoped to what the source
says:**

> A third system in this line, EGAgent [Rege et al., 2026], builds an entity
> graph — nodes typed as person/object/location, edges as temporally
> annotated relations (talks-to, interacts-with, mentions, uses) — from
> egocentric video via an LLM-based extractor, so that a planning agent can
> query it at inference time. Like EgoSG, construction is generative and
> model-driven rather than structural; unlike EgoSG, the paper reports no
> construction-cost figure, only that the resulting database is compact
> (~2 MB for 52 hours of video).

This claim makes no cost comparison to IRIS and asserts nothing about
EGAgent's construction cost beyond the one figure the paper actually gives
(storage size, not compute/latency) — it does not claim EGAgent is cheap or
expensive to build.

If a cost-free citation is not wanted (i.e., if §2.2's flow specifically
needs a system that *does* report a construction-cost number the way EgoSG
and Vgent do), none of the three candidates supports that — all three are
silent on construction cost. In that case, leaving §2.2 at two systems
remains the more honest choice than stretching EGAgent, MemDreamer, or
GraphVideoAgent to imply a cost figure that isn't in their papers.

---

## Follow-up: does GraphVideoAgent's construction require a generative model?

§2.2's claim (per Appendix C item 2's resolution) is about **the class of
model** each construction requires, not about seconds. The candidate
selection above risked a selection error — citing the two systems that
plainly need a generative model (EGAgent's LLM extractor, MemDreamer's
Gemini-3.1-Pro) and setting aside the one whose *edges* come from spaCy
dependency parsing, which is not itself generative. Re-reading
GraphVideoAgent's own Sections 3.1–3.3 (arXiv 2501.15953v1, HTML) resolves
the tension.

**1. What are the nodes and edges built from — is dependency parsing
confirmed, and over what text?**
Confirmed. Section 3.2: "Our system constructs a sophisticated
multi-relational graph structure G=(V,E) that captures rich entity
interactions and their temporal evolution." Nodes are "derived through a
comprehensive extraction process that combines named entity recognition with
noun phrase chunking." Edges: "Each edge *e_ij* ∈ *E* is constructed through
dependency parsing of captions, storing relation type, temporal information,
and associated linguistic elements." Both NER/chunking and dependency
parsing run **over captions** — not over raw frames or embeddings directly.

**2. Where do those captions come from — same stage as construction, or
prior?**
Section 3.1 names the captioning model explicitly: **LaViLa**, "specifically
for egocentric video captioning," alongside EVA-CLIP-8B-plus "for
high-quality frame feature extraction." Section 3.1's own framing statement
is direct: the system "constructs a dynamic knowledge graph G=(V,E) that
captures rich entity relations and temporal dynamics **from video frame
captions**" — captions are stated as the input to graph construction, not a
downstream or unrelated artifact. Section 3.3 ties this to the actual build
step: "Initially, the system uniformly samples N frames ℱ₀ from the video to
construct a baseline knowledge graph. From these frames' captions, the LLM
generates a preliminary answer." Captioning the seed frames is therefore
part of building the *baseline* graph, not a separate prior pipeline stage
unrelated to construction — LaViLa is invoked specifically to produce the
text that NER/chunking and dependency parsing then turn into nodes and
edges.

**3. Does any part of construction require a neural forward pass?**
Yes, unambiguously — at one remove from the parsing step itself. LaViLa is a
neural video-captioning model; producing the captions that seed the baseline
graph is a generative forward pass. EVA-CLIP-8B-plus's "high-quality frame
feature extraction" is a second neural pass (an encoder, not generative) also
present in the node tuples ("VLM-derived visual features," per the item-1
report above). The dependency-parsing/NER step that actually assembles *G*
is classical NLP and does not itself call a neural network, but it operates
on text that could not exist without LaViLa's forward pass. The paper gives
no algorithm box that walls off an "offline construction" phase from the
LaViLa call, but the sentence in Section 3.3 above makes the dependency
explicit: no captions, no baseline graph.

**4. Eager or lazy captioning?**
Neither purely. Section 3.3 describes captioning the initial **uniformly
sampled** frames (N₀, reported elsewhere as 8 frames on average at query
time) to build the *baseline* graph — this is eager relative to that seed
set, not lazy in the sense of being deferred to query time, but it is not
eager over the whole video either: most frames are never captioned. The
graph is then **extended** during the iterative retrieval loop, which adds
a small number of further frames per round (3 additional frames per stage,
per the paper) and presumably captions them too, since the graph must be
updated with their entities/relations. So construction is front-loaded but
incremental: a small first pass at build time, more captioning folded into
what the paper calls retrieval but which also keeps growing the graph.
There is no separate "index built once, queried many times" split — the
graph itself keeps being written to during what §2.2 would call query time.

**Direct answer:** GraphVideoAgent's construction **does** require a
generative model — one remove from the graph-structuring code. Dependency
parsing and NER are classical and don't call a network, but they run on
LaViLa's output, and LaViLa is invoked specifically to produce the input
those steps need. §2.2's class-of-model claim survives GraphVideoAgent; the
paper does not present a counterexample of a generative-model-free graph
construction. Dropping it to avoid the appearance of a counterexample would
itself have been the selection error the prompt warned against — the source
does not actually complicate the claim, it needed the one-remove framing
stated precisely.

**Recommended role in §2.2, and exact scoped wording:**

GraphVideoAgent should be cited as a **supporting case**, not a disclosed
complication — but the supporting sentence must state the one-remove
relationship precisely, not claim GraphVideoAgent's graph-builder itself is
a generative model call (it isn't; EgoSG's and MemDreamer's are, EGAgent's
LLM extractor is closer to a direct call too, while GraphVideoAgent's own
graph-structuring step is parsing). Recommended text:

> GraphVideoAgent [Chu et al., ACM MM 2025] builds its graph from parsed
> captions rather than a single generative call: entities come from NER and
> noun-phrase chunking, and typed edges (spatial, interaction, action) from
> dependency parsing, applied to captions of an initially sampled frame set.
> Those captions are themselves produced by a captioning model, LaViLa, so a
> generative forward pass sits one step upstream of graph construction even
> though the parsing step that assembles the graph does not itself call a
> network. Construction is not fully eager: only a small sampled seed set is
> captioned before the baseline graph is built, with more captions folded in
> as the graph is extended during retrieval.

And, if §2.2 wants the sharper cross-system statement the prompt asked for,
scoped to what has actually been verified across all systems discussed
(EgoSG, Vgent, EGAgent, MemDreamer, GraphVideoAgent):

> Across this line of work, every system's graph construction depends on a
> generative model somewhere in the pipeline — directly, as a single call
> that emits the graph (EgoSG's Gemini call per chunk, MemDreamer's
> Gemini-3.1-Pro pass per streaming window), or one step removed, as a
> captioner or extractor whose output is then parsed or structured
> (Vgent's and GraphVideoAgent's per-clip/per-frame captioning, EGAgent's
> LLM extractor over documents). We defer that cost differently: no model
> participates in construction at all, and captioning — where we use it —
> happens per query, not per video.

This second form is the one that actually earns "a line of work" rather
than a list: it names the shared trait precisely (generative model
somewhere in construction, direct or one-removed) rather than asserting a
uniform mechanism the five papers don't share.
