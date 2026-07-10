# L1 Elysium Ablation Plan
**Track A — Owner: Swara**  
**Project: IRIS-HADES**

---

## 1. Objective

This document defines the ablation study for the L1 Elysium cache (l1_elysium.py).

The goal is to prove that each signal in the keep_score eviction formula
contributes meaningfully to retrieval quality. We do this by running three
cache variants on the same video and queries, comparing how well each
variant retrieves the right frames when asked a question.

---

## 2. The Three Variants

### Variant A — LRU Baseline
**What it does:** Evicts the oldest frame (by admission time). No codec signals used.  
**What it represents:** The simplest possible cache — "keep the newest, throw away the oldest."  
**IRISConfig settings:**
```python
IRISConfig(
    l1_w_action   = 0.00,
    l1_w_query    = 0.00,
    l1_w_persist  = 0.00,
    l1_w_pagerank = 0.00,
    l1_w_entropy  = 0.00,
    l1_w_hessian  = 0.00,
    l1_w_recency  = 1.00,
)
```
**Expected weakness:** Evicts high-action frames simply because they are old.
A dramatic event at minute 2 of a video gets thrown out by minute 5,
even if no query has been answered yet.

---

### Variant B — Action Score Only
**What it does:** Eviction is governed entirely by the continuous action_score
from action_score.py (residual energy + motion + entropy). No FrameMotionDescriptor
geometry fields (hessian_boundary, motion_entropy) are used.  
**What it represents:** Codec-guided eviction without motion geometry.  
**IRISConfig settings:**
```python
IRISConfig(
    l1_w_action   = 0.60,
    l1_w_query    = 0.25,
    l1_w_persist  = 0.15,
    l1_w_pagerank = 0.00,
    l1_w_entropy  = 0.00,
    l1_w_hessian  = 0.00,
    l1_w_recency  = 0.00,
)
```
**Expected weakness:** Cannot distinguish between two frames with similar
action_score where one has sharp motion boundaries (high hessian — likely
an object entering the frame) and one has chaotic background noise
(high entropy but low hessian — less semantically important).

---

### Variant C — Full IRIS L1 (proposed system)
**What it does:** All seven signals active. action_score + query_similarity +
persistence + pagerank + motion_entropy + hessian_boundary + recency.  
**What it represents:** The complete Track A contribution.  
**IRISConfig settings:**
```python
IRISConfig(
    l1_w_action   = 0.30,
    l1_w_query    = 0.20,
    l1_w_persist  = 0.15,
    l1_w_pagerank = 0.10,
    l1_w_entropy  = 0.10,
    l1_w_hessian  = 0.10,
    l1_w_recency  = 0.05,
)
```

---

## 3. Test Video

**File:** `mov_bbb.mp4` (Big Buck Bunny — W3Schools sample)  
**Why this video:** Contains a clear mix of scene types:
- Static background segments (low residual — good for testing that boring
  frames are correctly deprioritised)
- Sudden motion events (high residual peaks — good for testing that
  important frames survive eviction)
- Scene cuts (I-frame boundaries — good for testing reset behaviour)

---

## 4. Queries Used

Each query is designed to target a specific frame region of the video
so retrieval success can be verified by timestamp.

| Query ID | Query text | Target region |
|---|---|---|
| Q1 | "Show me the moment when fast motion begins" | High-residual peak frame |
| Q2 | "Find a static scene with no movement" | Low-residual flat region |
| Q3 | "What happens right after the scene changes?" | Post-I-frame frame |

---

## 5. Metrics

### 5.1 Cache Hit Rate
**Definition:** Percentage of queries where the target frame is still in L1
at query time (not evicted before the query arrives).
**Why it matters:** If the target frame was evicted before the query,
the cache must fall back to L2 retrieval, increasing latency.

### 5.2 Eviction Regret Rate
**Definition:** Percentage of evicted frames that were later needed by a query.

**Why it matters:** A high regret rate means the cache is making bad eviction
decisions, throwing away frames it will need again.

### 5.3 Top-1 Retrieval Precision
**Definition:** For each query, is the target frame returned as the number 1 result?
Precision at 1 = queries where target frame is rank 1, divided by total queries.

### 5.4 Frames Processed per Query
**Definition:** How many frames the pipeline scans to answer a query.
Lower is better, meaning L1 held the right frames already.

---

## 6. Expected Results

| Metric | Variant A LRU | Variant B Action only | Variant C Full IRIS |
|---|---|---|---|
| Cache hit rate | Low | Medium | High |
| Eviction regret | High | Medium | Low |
| Precision at 1 | Low | Medium | High |
| Frames scanned | High | Medium | Low |

### Reasoning for expected gap between Variant B and C

Variant B cannot protect a frame like this:
- action_score = 0.55 (moderate, not obviously important)
- hessian_max_eigenvalue = 0.91 (sharp motion boundary, object just entered frame)
- motion_entropy = 0.84 (chaotic, unpredictable motion)

Variant B sees action_score = 0.55 and treats this frame as mediocre.
Variant C sees hessian and entropy and recognises it as a semantically rich event.
When a query arrives for object entering frame, Variant C has kept it.
Variant B evicted it in favour of a frame with action_score = 0.60 but no
geometric distinctiveness.

---

## 7. How to Run the Experiment

Once the full pipeline Charon-V to action_score to L1 is connected,
instantiate each variant and feed the same frames and queries:

from iris_config import IRISConfig
from l1_elysium import L1ElysiumCache

configs = {
    LRU: IRISConfig(l1_w_action=0.0, l1_w_query=0.0, l1_w_persist=0.0,
                    l1_w_pagerank=0.0, l1_w_entropy=0.0,
                    l1_w_hessian=0.0, l1_w_recency=1.0),
    ActionOnly: IRISConfig(l1_w_action=0.60, l1_w_query=0.25, l1_w_persist=0.15,
                           l1_w_pagerank=0.0, l1_w_entropy=0.0,
                           l1_w_hessian=0.0, l1_w_recency=0.0),
    FullIRIS: IRISConfig(l1_w_action=0.30, l1_w_query=0.20, l1_w_persist=0.15,
                         l1_w_pagerank=0.10, l1_w_entropy=0.10,
                         l1_w_hessian=0.10, l1_w_recency=0.05),
}

for name, cfg in configs.items():
    cache = L1ElysiumCache(config=cfg)
    # feed frames, run queries, record metrics

---

## 8. What This Proves for the Paper

The ablation study answers the key reviewer question:
Do you actually need all seven signals, or does just action_score work?

If Variant C significantly outperforms Variant B, we prove that
FrameMotionDescriptor geometry (hessian_boundary and motion_entropy)
contributes meaningfully beyond the basic action_score alone.

If Variant B significantly outperforms Variant A, we prove that
codec-guided eviction is better than naive time-based eviction.

Both gaps together prove the full IRIS L1 design is justified.
