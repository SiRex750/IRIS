# IRIS Graph Construction Limitations and Functional Fix Plan

This document lists the current limitations in the IRIS graph construction and retrieval layer, based on the live application path:

```text
api.py → iris.pipeline.run_pipeline()
       → wrapper_l2_retrieve()
       → L2Asphodel.add_frame_nodes_bulk()
       → L2Asphodel.enrich_nodes_bulk()
       → L2Asphodel.retrieve()
```

The important conclusion is that the current graph is structurally real, but not yet functionally central to retrieval. NetworkX nodes, edges, edge weights, and PageRank are computed, but the default retrieval path mostly behaves like a weighted frame ranking function.

---

## 1. PageRank is computed but not used in default retrieval

### Current behavior

`L2Asphodel._update_pagerank()` computes PageRank and writes it into each node:

```python
node.pagerank_score = score
```

However, the default retrieval function `L2Asphodel.retrieve()` ranks frames using:

```text
final_score =
    alpha * semantic_similarity
  + beta  * action_score
  + gamma * persistence_value
```

`pagerank_score` is not part of this formula.

### Why this is a limitation

The graph structure does not meaningfully affect which frames are retrieved. Edge construction and PageRank computation are expensive, but the output ranking is mostly independent of them.

### Required fix

Add PageRank into the default retrieval formula or switch the default ranking mode to `retrieve_ppr()`.

Possible corrected formula:

```text
final_score =
    alpha * semantic_similarity
  + beta  * action_score
  + gamma * persistence_value
  + delta * pagerank_score
```

Add `delta` to `IRISConfig`, validate it, and expose retrieval contributions in debug output.

---

## 2. The live API path uses legacy retrieval, not PPR

### Current behavior

The codebase contains `L2Asphodel.retrieve_ppr()`, which uses query-conditioned Personalized PageRank:

```text
seed =
    lambda * rank_percentile(semantic_similarity)
  + (1 - lambda) * rank_percentile(codec_confidence)
```

But the current API path in `pipeline.py` calls:

```python
graph.retrieve(...)
```

not:

```python
graph.retrieve_ppr(...)
```

### Why this is a limitation

The better graph-aware retrieval method exists but is not used by the application endpoint.

### Required fix

Respect `IRISConfig.ranking_mode` inside `wrapper_l2_retrieve()`:

```python
if config.ranking_mode == "ppr":
    retrieved_nodes = graph.retrieve_ppr(
        query_embedding,
        top_k=l2_retrieve_top_k,
        damping=config.ppr_damping,
        lambda_=config.ppr_lambda,
    )
else:
    retrieved_nodes = graph.retrieve(...)
```

Then set `ranking_mode = "ppr"` for the functional graph configuration.

---

## 3. Edge weights are built, but default retrieval ignores edges

### Current behavior

The graph is fully connected. Each edge weight is computed from:

```text
edge_weight =
    alpha * cosine_similarity(frame_i_embedding, frame_j_embedding)
  + beta  * motion_coherence(frame_i, frame_j)
```

where motion coherence is:

```text
1 - abs(action_score_i - action_score_j) / action_score_range
```

### Why this is a limitation

In the default `retrieve()` path, edge weights do not influence the returned frames. They only matter if PageRank/PPR is used.

### Required fix

Either:

1. Make `retrieve_ppr()` the default graph retrieval mode, or
2. Add a graph-neighborhood expansion step after initial top-k retrieval.

Example:

```text
initial seeds = top semantic/action frames
expand through graph = top weighted neighbors
rerank = seed_score + edge_support + pagerank
```

This would make graph connectivity affect retrieval instead of being decorative.

---

## 4. The graph is fully connected, which is expensive and noisy

### Current behavior

For `N` indexed frames, the graph creates roughly:

```text
N * (N - 1) / 2
```

edges.

### Why this is a limitation

Fully connected graphs become expensive for long videos and often add weak/noisy edges between unrelated frames.

### Required fix

Replace the fully connected graph with a sparse graph.

Recommended edge types:

1. Temporal edges:

```text
frame_i → frame_i+1
frame_i → frame_i+k
```

2. Semantic nearest-neighbor edges:

```text
top-k CLIP neighbors per frame
```

3. Motion/action similarity edges:

```text
top-k action/motion neighbors per frame
```

4. Optional object/entity edges after real object detection or caption parsing:

```text
frames sharing entity/object/action labels
```

Target graph complexity:

```text
O(N * k)
```

instead of:

```text
O(N²)
```

---

## 5. Motion coherence rewards similar action scores, not meaningful event relation

### Current behavior

Two frames are strongly connected if their action scores are close.

### Why this is a limitation

Frames with similar action intensity are not necessarily semantically related. For example, two unrelated high-motion moments may become connected just because both are active.

### Required fix

Use multiple edge channels instead of a single blended scalar:

```text
temporal_weight
semantic_weight
motion_weight
codec_weight
object_or_caption_weight
```

Then either:

1. Store them as separate edge attributes, or
2. Combine them using a learned/tuned formula.

Example:

```text
edge_weight =
    w_temporal * temporal_decay
  + w_semantic * clip_similarity
  + w_motion   * motion_similarity
  + w_codec    * codec_event_similarity
```

---

## 6. `query_action_score` is passed but unused

### Current behavior

`retrieve()` accepts:

```python
query_action_score: float
```

but the current scoring logic uses each node's own `action_score`, not the query action score.

### Why this is a limitation

The API suggests query-conditioned motion retrieval, but the implementation does not use that query signal.

### Required fix

Either remove the unused argument or use it correctly.

Possible usage:

```text
motion_query_similarity =
    1 - abs(node.action_score - query_action_score) / action_score_range
```

Then:

```text
final_score =
    alpha * semantic_similarity
  + beta  * motion_query_similarity
  + gamma * persistence_value
  + delta * pagerank_score
```

---

## 7. `refined_motion_tensor` is currently a placeholder

### Current behavior

The graph schema includes:

```python
refined_motion_tensor
```

but the pipeline passes:

```python
np.zeros(1, dtype=np.float32)
```

### Why this is a limitation

The graph claims to support refined motion tensors, but no real refined motion representation is being inserted.

### Required fix

Either:

1. Remove `refined_motion_tensor` until implemented, or
2. Populate it with a real motion descriptor.

Practical replacement:

```text
motion_vector =
[
  motion_magnitude,
  divergence,
  curl,
  jacobian_frobenius,
  hessian_max_eigenvalue,
  motion_entropy
]
```

Then use cosine or normalized distance over this vector for motion edges.

---

## 8. Graph nodes do not contain real semantic triples

### Current behavior

`enrich_nodes_bulk()` currently sets:

```python
node_data.triples = []
node_data.embedding = embedding
```

The graph receives CLIP embeddings, but not structured subject-verb-object triples.

### Why this is a limitation

The layer is described as a knowledge graph, but the graph is currently closer to a frame-similarity graph. It lacks explicit object/action/entity relations.

### Required fix

Extract structured triples from captions or object/action detectors.

Example node metadata:

```json
{
  "objects": ["person", "car"],
  "actions": ["walking", "entering"],
  "scene": "street",
  "caption": "A person walks toward a parked car."
}
```

Example semantic edges:

```text
frame_12 --shares_object:person--> frame_18
frame_12 --same_action:walking--> frame_20
frame_12 --temporal_next--> frame_13
```

---

## 9. Caption-derived semantics are weak and sometimes template-like

### Current behavior

The semantic side uses:

1. CLIP image embeddings
2. CLIP zero-shot labels from a fixed vocabulary
3. BLIP-generated captions

The fixed zero-shot vocabulary is narrow and includes test-video-specific phrases.

### Why this is a limitation

If captions or labels are generic, wrong, or biased toward the fixed vocabulary, semantic graph edges become unreliable.

### Required fix

Use a more general semantic extraction path:

1. Generate BLIP or VLM caption.
2. Extract objects/actions/entities from caption.
3. Optionally verify object/action detections with a detector.
4. Store extracted fields on graph nodes.
5. Build semantic edges from extracted fields, not only CLIP cosine similarity.

---

## 10. The frontend graph is not the actual L2 graph

### Current behavior

`api.py` builds `graph_data` from retrieved frames only. It creates visualization nodes and temporal/coherence edges after retrieval.

This is separate from the actual `L2Asphodel.graph`.

### Why this is a limitation

The UI can show a graph that does not reflect the graph used internally. That makes debugging misleading.

### Required fix

Expose the real L2 graph structure from `L2Asphodel`.

Add a method:

```python
def export_graph_data(self) -> dict:
    return {
        "nodes": [...],
        "edges": [...]
    }
```

Then return that from the pipeline debug payload.

The UI should render the actual L2 nodes and edges, not a reconstructed approximation.

---

## 11. Graph construction happens per request in the old API path

### Current behavior

The API calls `run_pipeline()` for every uploaded video/query pair. That means graph construction happens inside the request flow.

There is a newer ingest/query split:

```text
iris.ingest.ingest(video) → IRISIndex
iris.query.query(question, index) → answer
```

but the API still uses `pipeline.run_pipeline()`.

### Why this is a limitation

For repeated questions on the same video, the graph should be built once and reused. Rebuilding per request wastes time and prevents persistent graph refinement.

### Required fix

Move the API to the ingest/query architecture:

```text
upload video → ingest once → save IRISIndex
question → query existing IRISIndex
```

This also makes lazy captioning, graph reuse, and PPR retrieval easier to support.

---

## 12. `codec_conf` is not fully wired in the live pipeline path

### Current behavior

The newer ingest path computes `codec_conf` from packet size or action score and stores it on nodes. `retrieve_ppr()` uses this value.

The older `pipeline.py` path does not fully compute and pass `codec_conf` into the graph nodes.

### Why this is a limitation

PPR depends on codec confidence. If `codec_conf` remains at its default value, PPR loses an important signal.

### Required fix

Unify the graph-building code so both API and ingest use the same `_build_graph()` path from `iris.ingest`.

Minimum fix:

```python
feature_record["packet_size"] = f_data.get("packet_size", 0.0)
feature_record["codec_conf"] = computed_codec_conf
```

Better fix:

Retire duplicate graph-building logic in `pipeline.py` and use `ingest.py` as the single graph construction implementation.

---

## 13. Graph serialization does not preserve the live NetworkX graph

### Current behavior

`save_index()` serializes frame records and embeddings. The live graph is rebuilt on `load_index()`.

### Why this is a limitation

This is acceptable for deterministic graph construction, but it becomes a limitation if graph construction later includes expensive edges, learned edge weights, community detection, or manual corrections.

### Required fix

Short term:

Keep deterministic rebuild, but ensure all edge inputs are serialized.

Long term:

Serialize graph edge lists:

```json
{
  "source": 12,
  "target": 18,
  "weights": {
    "temporal": 0.92,
    "semantic": 0.81,
    "motion": 0.44,
    "final": 0.73
  }
}
```

---

## 14. Debug logging is too noisy during retrieval

### Current behavior

`retrieve()` prints retrieval score contributions for every node.

### Why this is a limitation

For larger videos, this pollutes logs and slows request handling.

### Required fix

Move debug printing behind a config flag:

```python
if config.debug_retrieval:
    print(...)
```

or return debug contributions only in `verbose=True` output.

---

## 15. Graph terminology overstates current capability

### Current behavior

The code and docs describe L2 as a spatiotemporal knowledge graph. In practice, the current graph is:

```text
frame nodes + similarity edges + CLIP embeddings + action scores
```

It does not yet contain durable object/action/entity relations.

### Why this is a limitation

The implementation can support retrieval, but it should not be described as a complete knowledge graph until semantic relation extraction is implemented.

### Required fix

Use accurate terminology until the semantic graph is built:

```text
Current: frame similarity graph
Target: spatiotemporal semantic knowledge graph
```

---

# Minimum Fix Set to Make the Graph Actually Functional

The smallest practical set of changes is:

## Priority 1 — Use graph-aware retrieval

Wire `ranking_mode == "ppr"` into the API path and make `retrieve_ppr()` available from `pipeline.py`.

Expected impact:

```text
Graph edges and PageRank start affecting retrieved frames.
```

## Priority 2 — Add PageRank to legacy retrieval

If legacy retrieval is kept, add `pagerank_score` to its formula:

```text
score =
    alpha * semantic_similarity
  + beta  * action_score
  + gamma * persistence_value
  + delta * pagerank_score
```

Expected impact:

```text
Graph structure affects retrieval even outside PPR mode.
```

## Priority 3 — Build a sparse graph

Replace fully connected edges with temporal, semantic-neighbor, and motion-neighbor edges.

Expected impact:

```text
Less noise, lower cost, more meaningful graph topology.
```

## Priority 4 — Populate real semantic node metadata

Extract objects, actions, scenes, and entities from captions or detectors.

Expected impact:

```text
The graph becomes a semantic graph, not just a frame similarity graph.
```

## Priority 5 — Expose the actual graph to the frontend

Stop reconstructing a separate visualization graph in `api.py`. Export the real `L2Asphodel.graph`.

Expected impact:

```text
The UI becomes useful for debugging the actual graph.
```

## Priority 6 — Move API to ingest/query split

Build graph once per video, query it many times.

Expected impact:

```text
Better latency, cleaner architecture, reusable graph state.
```

---

# Recommended Target Design

The target graph should look like this:

```text
FrameNode
  frame_idx
  timestamp
  action_score
  persistence_value
  packet_size
  codec_conf
  clip_embedding
  caption
  objects
  actions
  scene

Edges
  temporal_next
  temporal_window
  semantic_neighbor
  motion_neighbor
  shared_object
  shared_action
```

Retrieval should run as:

```text
1. Encode query with CLIP/text model.
2. Create personalization seed from:
   - semantic similarity
   - codec confidence
   - action score
   - optional object/action match
3. Run Personalized PageRank on the sparse graph.
4. Return top-k frames by PPR score.
5. Expand context with high-weight neighbors.
6. Send selected frame captions/evidence to ARIA.
```

This makes the graph a real retrieval mechanism instead of a passive container.

