"""
L2 Asphodel — motion-aware video RAG graph for IRIS.

This module builds a NetworkX-based spatiotemporal scene graph (Asphodel) from
video frames. It supports a "cold-start" phase, constructing the graph structure
purely mathematically using motion signals without AI models, and later enriches
nodes with semantic text triples and embeddings from the LLM (ARIA).

Owner: Sia 
"""

from __future__ import annotations
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
import scipy.sparse
from typing import Any, List, Dict, Union, Optional


def _rank_pct(values: dict) -> dict:
    """Average-tied rank-percentile in [0, 1] over {node_id: float}."""
    keys = list(values.keys())
    n = len(keys)
    if n == 0:
        return {}
    if n == 1:
        return {keys[0]: 0.5}
    vals = np.array([values[k] for k in keys], dtype=np.float64)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
            j += 1
        avg = (i + j) / 2.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg
        i = j + 1
    rp = ranks / (n - 1)
    return {keys[idx]: float(rp[idx]) for idx in range(n)}


@dataclass
class AsphodelNode:
    """
    Schema representing a video frame node in the Asphodel spatiotemporal graph.
    
    Attributes:
        frame_idx:             Index of the frame in the decoded video sequence.
        timestamp:             Robust timestamp of the frame in seconds.
        action_score:          Motion action score from action_score.py (personalization seed).
        persistence_value:     Motion persistence score from action_score.py.
        luma_diff_energy:       Luma difference residual proxy extracted by Charon-V.
        motion_magnitude:      Magnitude of raw motion vectors.
        luma_entropy:               Luma (Y-plane) histogram entropy.
        refined_motion_tensor: The bfloat16 smoothed motion tensor from the RMR layer.
        triples:               KnowledgeTriple objects added by ARIA during enrichment.
        embedding:             CLIP embedding numpy array added by ARIA (None until enriched).
    pagerank_score:        Structural importance score computed dynamically via PageRank.
        tier:                  Hierarchical frame tier: L1_PEAK, L2_SALIENT, or L3_CANDIDATE.
        pict_type:             Codec picture type, used to mark I-frame scene resets.
        scene_id:              Monotonic scene segment id inferred from I-frames.
    """
    frame_idx:             int
    timestamp:             float
    action_score:          float
    persistence_value:     float
    luma_diff_energy:       float
    motion_magnitude:      float
    luma_entropy:               float
    refined_motion_tensor: np.ndarray
    triples:               list = field(default_factory=list)
    embedding:             np.ndarray | None = None
    pagerank_score:        float = 0.0
    packet_size:           float = 0.0
    codec_conf:            float = 0.5
    tier:                  str = "L3_CANDIDATE"
    pict_type:             str = "?"
    scene_id:              int = 0
    last_retrieval_score:  float = 0.0
    retrieval_contributions: dict = field(default_factory=dict)
    divergence:             float = 0.0
    curl:                   float = 0.0
    jacobian_frobenius:     float = 0.0
    hessian_max_eigenvalue: float = 0.0
    motion_entropy:         float = 0.0


class L2Asphodel:
    """
    L2 Layer (Asphodel) — NetworkX-based spatiotemporal scene graph manager.
    
    This class orchestrates a NetworkX Graph representation of the video scene. It:
      - Accepts new video frames as nodes using motion metrics (Cold-Start).
      - Forms a fully connected graph with edges weighted by a motion coherence metric.
      - Automatically computes PageRank scores using motion intensity as a personalization seed.
      - Enriches nodes with semantic metadata and embeddings.
      - Adjusts edge weights to a hybrid (semantic + motion) formula.
      - Exposes hybrid retrieval and exports the graph features into a Compressed Sparse Row (CSR) matrix.
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize an empty NetworkX graph and configuration weights.

        Args:
            config: A configuration object or dictionary containing alpha and beta.
                    alpha: weight for semantic cosine similarity (default 0.4)
                    beta: weight for motion coherence similarity (default 0.6)
        """
        # An undirected Graph is chosen because spatiotemporal relationships are
        # bidirectional; PageRank influence propagates mutually based on coherence.
        self.graph = nx.Graph()
        
        # Default weights injected from IRISConfig
        self.alpha: float = 0.4
        self.beta: float = 0.6
        self.gamma: float = 0.0
        self.delta: float = 0.0
        self.graph_edge_mode: str = "fully_connected"
        self.graph_temporal_window: int = 1
        self.graph_semantic_top_k: int = 4
        self.graph_motion_top_k: int = 2
        self.graph_semantic_threshold: float = 0.5
        self.debug_retrieval: bool = False
        self.salient_thresh: float = 0.35
        self.candidate_thresh: float = 0.08
        self.motion_similarity_mode: str = "action_score"

        # Parse config fields if provided (handling both IRISConfig classes and dictionaries)
        if config is not None:
            self.alpha = self._cfg(config, "alpha", self.alpha)
            self.beta = self._cfg(config, "beta", self.beta)
            self.gamma = self._cfg(config, "gamma", self.gamma)
            self.delta = self._cfg(config, "delta", self.delta)
            self.graph_edge_mode = self._cfg(config, "graph_edge_mode", self.graph_edge_mode)
            self.graph_temporal_window = int(self._cfg(config, "graph_temporal_window", self.graph_temporal_window))
            self.graph_semantic_top_k = int(self._cfg(config, "graph_semantic_top_k", self.graph_semantic_top_k))
            self.graph_motion_top_k = int(self._cfg(config, "graph_motion_top_k", self.graph_motion_top_k))
            self.graph_semantic_threshold = float(self._cfg(config, "graph_semantic_threshold", self.graph_semantic_threshold))
            self.debug_retrieval = bool(self._cfg(config, "graph_debug_retrieval", self.debug_retrieval))
            self.salient_thresh = float(self._cfg(config, "salient_thresh", self.salient_thresh))
            self.candidate_thresh = float(self._cfg(config, "candidate_thresh", self.candidate_thresh))
            self.motion_similarity_mode = self._cfg(config, "motion_similarity_mode", self.motion_similarity_mode)

    @staticmethod
    def _cfg(config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _update_all_edge_weights(self, node_groups: list[list] | None = None) -> None:
        """
        Recalculates edge weights between all nodes in the graph.

        Default mode is a sparse hierarchy:
          L1_PEAK      = peak / I-frame scene reset nodes
          L2_SALIENT   = high-action salient nodes
          L3_CANDIDATE = lower-action candidate nodes

        Edge families:
          - temporal: local neighbors in indexed-frame order
          - hierarchy_peak_salient: nearest L1 parent for each L2 node
          - hierarchy_salient_candidate: nearest L2 parent for each L3 node
          - semantic_salient: L2↔L2 semantic cross-root edges
          - motion_neighbor: nearest action/motion neighbors

        `graph_edge_mode="fully_connected"` preserves the older dense behavior.
        
        All edges store channel attributes (`semantic_weight`, `motion_weight`,
        `temporal_weight`, `edge_type`) plus the final scalar `weight` used by
        PageRank/PPR.
        """
        node_ids = list(self.graph.nodes)
        num_nodes = len(node_ids)
        self.graph.remove_edges_from(list(self.graph.edges))
        if num_nodes <= 1:
            self._refresh_scene_ids()
            return

        self._refresh_scene_ids()
        sorted_ids = sorted(
            node_ids,
            key=lambda nid: (
                self.graph.nodes[nid]["node_data"].timestamp,
                self.graph.nodes[nid]["node_data"].frame_idx,
            ),
        )
        nodes_data = [self.graph.nodes[n]["node_data"] for n in sorted_ids]

        scores = [node.action_score for node in nodes_data]
        max_score = max(scores)
        min_score = min(scores)
        max_score_range = max_score - min_score

        if self.graph_edge_mode == "fully_connected":
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    self._add_weighted_edge(sorted_ids[i], sorted_ids[j], "fully_connected", max_score_range)
        else:
            self._add_temporal_edges(sorted_ids, max_score_range)
            self._add_hierarchy_edges(sorted_ids, max_score_range)
            self._add_salient_semantic_edges(sorted_ids, max_score_range)
            self._add_motion_neighbor_edges(sorted_ids, max_score_range)

        if node_groups is not None:
            node_to_group = {}
            for gid, group in enumerate(node_groups):
                for nid in group:
                    node_to_group[nid] = gid
            edges_to_remove = []
            for u, v in self.graph.edges:
                if node_to_group.get(u) != node_to_group.get(v):
                    edges_to_remove.append((u, v))
            self.graph.remove_edges_from(edges_to_remove)

    def _refresh_scene_ids(self) -> None:
        """Infer scene segments from I-frames in temporal order."""
        sorted_ids = sorted(
            self.graph.nodes,
            key=lambda nid: (
                self.graph.nodes[nid]["node_data"].timestamp,
                self.graph.nodes[nid]["node_data"].frame_idx,
            ),
        )
        scene_id = 0
        first = True
        for nid in sorted_ids:
            node = self.graph.nodes[nid]["node_data"]
            pict = str(getattr(node, "pict_type", "?")).upper()
            if not first and pict.startswith("I"):
                scene_id += 1
            node.scene_id = scene_id
            first = False

    def _semantic_similarity(self, node_u: AsphodelNode, node_v: AsphodelNode) -> float:
        if node_u.embedding is None or node_v.embedding is None:
            return 0.0
        norm_u = np.linalg.norm(node_u.embedding)
        norm_v = np.linalg.norm(node_v.embedding)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        # L2-002: Clamp to [0, 1] — raw cosine can be negative when embeddings
        # point in opposite directions. Negative edge weights corrupt PageRank
        # (random walk interpretation requires non-negative transition probs).
        return max(0.0, float(np.dot(node_u.embedding, node_v.embedding) / (norm_u * norm_v)))

    def _motion_similarity(self, node_u: AsphodelNode, node_v: AsphodelNode, max_score_range: float) -> float:
        if self.motion_similarity_mode == "geometry_6d":
            # P1-09: Build a normalized motion feature vector from available corrected signals:
            # motion_magnitude, divergence, curl, jacobian_frobenius, hessian_max_eigenvalue, motion_entropy.
            # Divergence is signed. We compute cosine similarity over this 6-D vector.
            vu = np.array([
                node_u.motion_magnitude,
                node_u.divergence,
                node_u.curl,
                node_u.jacobian_frobenius,
                node_u.hessian_max_eigenvalue,
                node_u.motion_entropy
            ], dtype=np.float32)

            vv = np.array([
                node_v.motion_magnitude,
                node_v.divergence,
                node_v.curl,
                node_v.jacobian_frobenius,
                node_v.hessian_max_eigenvalue,
                node_v.motion_entropy
            ], dtype=np.float32)

            norm_u = float(np.linalg.norm(vu))
            norm_v = float(np.linalg.norm(vv))

            if norm_u > 1e-8 and norm_v > 1e-8:
                cosine = float(np.dot(vu, vv) / (norm_u * norm_v))
                return max(0.0, min(1.0, cosine))

            # Fallback: action_score-based similarity for nodes without motion descriptors.
            if max_score_range == 0.0:
                return 1.0
            return max(0.0, 1.0 - (abs(node_u.action_score - node_v.action_score) / max_score_range))

        if max_score_range == 0.0:
            return 1.0
        return max(0.0, 1.0 - (abs(node_u.action_score - node_v.action_score) / max_score_range))

    def _temporal_proximity(self, node_u: AsphodelNode, node_v: AsphodelNode) -> float:
        dt = abs(float(node_u.timestamp) - float(node_v.timestamp))
        return 1.0 / (1.0 + dt)

    def _edge_weight(self, semantic: float, motion: float, temporal: float, edge_type: str) -> float:
        if edge_type.startswith("hierarchy_peak_salient"):
            base = temporal * max(semantic, motion)
        elif edge_type.startswith("hierarchy_salient_candidate"):
            base = temporal * max(motion, semantic * 0.5)
        elif edge_type == "temporal":
            base = 0.7 * temporal + 0.3 * motion
        elif edge_type == "semantic_salient":
            base = 0.8 * semantic + 0.2 * temporal
        elif edge_type == "motion_neighbor":
            base = 0.8 * motion + 0.2 * temporal
        elif edge_type == "fully_connected":
            return float(self.alpha * semantic + self.beta * motion)
        else:
            base = self.alpha * semantic + self.beta * motion + 0.1 * temporal
        return max(1e-6, float(base))

    def _add_weighted_edge_to_graph(
        self,
        target_graph: nx.Graph,
        u: int,
        v: int,
        edge_type: str,
        max_score_range: float,
    ) -> str:
        """Graph-targeted edge insertion using the shared weighting helpers.

        Reads node data from *target_graph* (not self.graph), so it safely
        operates on any NetworkX graph — including temporary subgraph copies
        used in scene_sparse DESCEND retrieval — without ever swapping
        self.graph or sharing mutable state across concurrent queries.

        Args:
            target_graph:    The NetworkX graph to insert the edge into.
            u, v:            Node IDs (must already exist in target_graph).
            edge_type:       Edge family label used by _edge_weight.
            max_score_range: Pre-computed action-score range for motion normalisation.

        Returns:
            "new"       — edge was not present and has been inserted.
            "updated"   — an existing edge with lower weight was replaced.
            "unchanged" — an existing edge had equal-or-greater weight; no write.

        Raises:
            KeyError: If u or v is absent from target_graph, or if either node
                      lacks a "node_data" attribute.  We raise rather than
                      silently skip so callers get an actionable diagnostic.
        """
        if u == v:
            return "unchanged"
        if u not in target_graph.nodes:
            raise KeyError(f"Node {u!r} is not present in the target graph.")
        if v not in target_graph.nodes:
            raise KeyError(f"Node {v!r} is not present in the target graph.")
        node_u = target_graph.nodes[u].get("node_data")
        node_v = target_graph.nodes[v].get("node_data")
        if node_u is None:
            raise KeyError(f"Node {u!r} exists in target_graph but has no 'node_data' attribute.")
        if node_v is None:
            raise KeyError(f"Node {v!r} exists in target_graph but has no 'node_data' attribute.")
        semantic = self._semantic_similarity(node_u, node_v)
        motion = self._motion_similarity(node_u, node_v, max_score_range)
        temporal = self._temporal_proximity(node_u, node_v)
        weight = self._edge_weight(semantic, motion, temporal, edge_type)
        if target_graph.has_edge(u, v):
            existing = target_graph[u][v]
            # P1-11: Preserve all relationship labels.  A pair can legitimately
            # have multiple relationship types (e.g. temporal AND semantic_salient).
            # Accumulate edge_type labels as a pipe-separated set so no relationship
            # is silently discarded, while still keeping the highest weight.
            existing_types: set[str] = set(
                existing.get("edge_type", "").split("|")
            )
            existing_types.discard("")
            existing_types.add(edge_type)
            merged_type = "|".join(sorted(existing_types))
            if weight <= existing.get("weight", 0.0):
                # Weaker weight — update only the label set, not the weight.
                target_graph[u][v]["edge_type"] = merged_type
                return "unchanged"
            # Stronger weight — update weight AND label set.
            outcome = "updated"
        else:
            merged_type = edge_type
            outcome = "new"
        target_graph.add_edge(
            u,
            v,
            weight=weight,
            edge_type=merged_type,
            semantic_weight=semantic,
            motion_weight=motion,
            temporal_weight=temporal,
        )
        return outcome

    def _add_weighted_edge(self, u: int, v: int, edge_type: str, max_score_range: float) -> None:
        if u == v:
            return
        node_u = self.graph.nodes[u]["node_data"]
        node_v = self.graph.nodes[v]["node_data"]
        semantic = self._semantic_similarity(node_u, node_v)
        motion = self._motion_similarity(node_u, node_v, max_score_range)
        temporal = self._temporal_proximity(node_u, node_v)
        weight = self._edge_weight(semantic, motion, temporal, edge_type)
        if self.graph.has_edge(u, v):
            existing = self.graph[u][v]
            if weight <= existing.get("weight", 0.0):
                return
        self.graph.add_edge(
            u,
            v,
            weight=weight,
            edge_type=edge_type,
            semantic_weight=semantic,
            motion_weight=motion,
            temporal_weight=temporal,
        )

    def _add_temporal_edges(self, sorted_ids: list[int], max_score_range: float) -> None:
        window = max(0, int(self.graph_temporal_window))
        if window == 0:
            return
        for i, u in enumerate(sorted_ids):
            for j in range(i + 1, min(len(sorted_ids), i + window + 1)):
                self._add_weighted_edge(u, sorted_ids[j], "temporal", max_score_range)

    def _nearest_node(self, node: AsphodelNode, candidates: list[AsphodelNode]) -> AsphodelNode | None:
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cand: (
                abs(float(cand.timestamp) - float(node.timestamp)),
                abs(int(cand.frame_idx) - int(node.frame_idx)),
            ),
        )

    def _add_hierarchy_edges(self, sorted_ids: list[int], max_score_range: float) -> None:
        nodes = [self.graph.nodes[nid]["node_data"] for nid in sorted_ids]
        peaks = [node for node in nodes if node.tier == "L1_PEAK"]
        salient = [node for node in nodes if node.tier == "L2_SALIENT"]

        for node in salient:
            parent = self._nearest_node(node, peaks)
            if parent is not None:
                self._add_weighted_edge(parent.frame_idx, node.frame_idx, "hierarchy_peak_salient", max_score_range)

        candidate_parents = salient or peaks
        for node in nodes:
            if node.tier != "L3_CANDIDATE":
                continue
            parent = self._nearest_node(node, candidate_parents)
            if parent is not None:
                self._add_weighted_edge(parent.frame_idx, node.frame_idx, "hierarchy_salient_candidate", max_score_range)

    def _add_salient_semantic_edges(self, sorted_ids: list[int], max_score_range: float) -> None:
        top_k = max(0, int(self.graph_semantic_top_k))
        if top_k == 0:
            return
        salient_ids = [
            nid for nid in sorted_ids
            if self.graph.nodes[nid]["node_data"].tier == "L2_SALIENT"
        ]
        for u in salient_ids:
            node_u = self.graph.nodes[u]["node_data"]
            sims = []
            for v in salient_ids:
                if u == v:
                    continue
                node_v = self.graph.nodes[v]["node_data"]
                semantic = self._semantic_similarity(node_u, node_v)
                if semantic >= self.graph_semantic_threshold:
                    sims.append((semantic, v))
            sims.sort(reverse=True)
            for _, v in sims[:top_k]:
                self._add_weighted_edge(u, v, "semantic_salient", max_score_range)

    def _add_motion_neighbor_edges(self, sorted_ids: list[int], max_score_range: float) -> None:
        top_k = max(0, int(self.graph_motion_top_k))
        if top_k == 0:
            return
        for u in sorted_ids:
            node_u = self.graph.nodes[u]["node_data"]
            sims = []
            for v in sorted_ids:
                if u == v:
                    continue
                node_v = self.graph.nodes[v]["node_data"]
                sims.append((self._motion_similarity(node_u, node_v, max_score_range), v))
            sims.sort(reverse=True)
            for _, v in sims[:top_k]:
                self._add_weighted_edge(u, v, "motion_neighbor", max_score_range)

    # ── Cross-scene edge helper (scene_sparse DESCEND) ───────────────────────

    def add_cross_scene_edges(
        self,
        pool_ids: list[int],
        scene_of: dict[int, int],
        *,
        mode: str,
        threshold_percentile: float,
        scene_anchors: dict[int, int],
        graph: nx.Graph,
    ) -> int:
        """Add temporary cross-scene edges to *graph* (a subgraph copy).

        This is the scene_sparse DESCEND helper described in §2c-ii.  It
        populates *graph* — normally the induced subgraph copy built in
        scene_retrieval.py — with edges that bridge frames belonging to
        different scene segments.  The production self.graph is **never**
        written to: all edges go exclusively into the supplied *graph* argument.

        Args:
            pool_ids:             Frame IDs that make up the retrieval pool.
                                  Every ID must exist as a node in *graph*.
            scene_of:             Mapping {frame_idx: scene_id} for every pool
                                  frame.  Must cover all pool_ids.
            mode:                 Cross-scene edge strategy — one of:
                                    "all"        — every cross-scene pair.
                                    "threshold"  — pairs above a percentile
                                                   of pairwise semantic sims.
                                    "rep_only"   — one representative per
                                                   scene via scene_anchors
                                                   (production default).
            threshold_percentile: Percentile cutoff in [0, 100] used when
                                  mode="threshold"; ignored for other modes.
            scene_anchors:        {scene_id: frame_idx} mapping one
                                  representative frame per scene.  Used by
                                  mode="rep_only".
            graph:                Target NetworkX graph (must already contain
                                  all pool_ids as nodes with 'node_data').

        Returns:
            Number of genuinely *new* cross-scene edges inserted into *graph*.
            Updated edges (stronger weight replacing weaker), unchanged edges,
            same-scene pairs, and duplicate attempts are not counted.

        Raises:
            ValueError: For invalid mode or threshold_percentile.
            KeyError:   For missing nodes/scene assignments/anchor frames.
        """
        # ── Validation ────────────────────────────────────────────────────────
        _VALID_MODES = {"all", "threshold", "rep_only"}
        if mode not in _VALID_MODES:
            raise ValueError(
                f"add_cross_scene_edges: mode must be one of {sorted(_VALID_MODES)!r}, "
                f"got {mode!r}."
            )
        if not (0.0 <= threshold_percentile <= 100.0):
            raise ValueError(
                f"add_cross_scene_edges: threshold_percentile must be in [0, 100], "
                f"got {threshold_percentile!r}."
            )
        for pid in pool_ids:
            if pid not in graph.nodes:
                raise KeyError(
                    f"add_cross_scene_edges: pool_id {pid!r} is not a node in the "
                    f"supplied graph.  Ensure the subgraph copy was built correctly."
                )
            if pid not in scene_of:
                raise KeyError(
                    f"add_cross_scene_edges: pool_id {pid!r} has no entry in scene_of."
                )
        for scene_id, anchor_fid in scene_anchors.items():
            if anchor_fid in scene_of and scene_of[anchor_fid] != scene_id:
                raise ValueError(
                    f"add_cross_scene_edges: scene_anchors declares frame {anchor_fid!r} "
                    f"as the representative of scene {scene_id!r}, but scene_of says "
                    f"that frame belongs to scene {scene_of[anchor_fid]!r}."
                )

        # Deterministic ordering: sort once, generate each unordered pair once.
        sorted_ids = sorted(pool_ids)

        # Pre-compute action-score range for motion similarity (same formula as
        # the production graph build path).
        action_scores = [graph.nodes[pid]["node_data"].action_score for pid in sorted_ids]
        if len(action_scores) >= 2:
            max_score_range = max(action_scores) - min(action_scores)
        else:
            max_score_range = 0.0

        new_edges_added: int = 0

        if mode == "all":
            # ── mode="all": densest diagnostic mode ───────────────────────────
            # Every cross-scene pair receives a cross_scene_all edge via the
            # shared weighting helper.
            for i, u in enumerate(sorted_ids):
                for v in sorted_ids[i + 1:]:
                    if scene_of[u] == scene_of[v]:
                        continue
                    outcome = self._add_weighted_edge_to_graph(
                        graph, u, v, "cross_scene_all", max_score_range
                    )
                    if outcome == "new":
                        new_edges_added += 1

        elif mode == "threshold":
            # ── mode="threshold": percentile-gated semantic similarity ─────────
            # 1. Collect all eligible cross-scene pairs (stable, sorted order).
            # 2. Compute semantic similarity for each.
            # 3. Determine the percentile cutoff.
            # 4. Add pairs whose sim is finite, > 0, and >= cutoff.
            pairs: list[tuple[int, int]] = []
            sims: list[float] = []
            for i, u in enumerate(sorted_ids):
                node_u = graph.nodes[u]["node_data"]
                for v in sorted_ids[i + 1:]:
                    if scene_of[u] == scene_of[v]:
                        continue
                    node_v = graph.nodes[v]["node_data"]
                    sim = self._semantic_similarity(node_u, node_v)
                    pairs.append((u, v))
                    sims.append(sim)

            if not pairs:
                # No eligible cross-scene pairs — nothing to add.
                return 0

            sim_array = np.array(sims, dtype=np.float64)
            # Handle all-zero case: cutoff = 0 means nothing clears the
            # "> 0" guard, so we return 0 edges — safe and deterministic.
            if np.all(sim_array == 0.0):
                cutoff = 0.0
            else:
                cutoff = float(np.percentile(sim_array, threshold_percentile))

            for (u, v), sim in zip(pairs, sims):
                if not np.isfinite(sim):
                    continue
                if sim <= 0.0:
                    continue
                if sim < cutoff:
                    continue
                outcome = self._add_weighted_edge_to_graph(
                    graph, u, v, "cross_scene_threshold", max_score_range
                )
                if outcome == "new":
                    new_edges_added += 1

        else:  # mode == "rep_only"
            # ── mode="rep_only": representative-only sparse mode ───────────────
            # Connect each unordered pair of scene representatives.
            # Maximum new edges for S represented scenes = S*(S-1)/2.
            # This is the production-default mode — no all-pairs frame edges.
            sorted_scenes = sorted(scene_anchors.keys())
            for i, scene_a in enumerate(sorted_scenes):
                anchor_a = scene_anchors[scene_a]
                for scene_b in sorted_scenes[i + 1:]:
                    anchor_b = scene_anchors[scene_b]
                    # scene_a != scene_b by construction (different loop indices)
                    # Verify both anchors exist in the graph (may have been
                    # excluded from pool — skip gracefully rather than crashing).
                    if anchor_a not in graph.nodes or anchor_b not in graph.nodes:
                        continue
                    outcome = self._add_weighted_edge_to_graph(
                        graph, anchor_a, anchor_b, "cross_scene_rep", max_score_range
                    )
                    if outcome == "new":
                        new_edges_added += 1

        return new_edges_added

    # ─────────────────────────────────────────────────────────────────────────

    def _update_pagerank(self) -> None:
        """
        Executes a NetworkX PageRank calculation and updates the pagerank_score attribute of each node.
        
        Personalization Seed Logic:
        Personalized PageRank weights random walk jump targets using a customization vector. By feeding
        action_score as the personalization seed, the graph simulation biases paths to start from or
        jump back to frames with high physical motion activity. Thus, frames containing major motion spikes
        automatically acquire high graph importance (PageRank) without needing semantic understanding.
        """
        num_nodes = len(self.graph)
        if num_nodes == 0:
            return
        elif num_nodes == 1:
            idx = list(self.graph.nodes)[0]
            self.graph.nodes[idx]["node_data"].pagerank_score = 1.0
            return

        # Build personalization dictionary mapping frame_idx -> action_score
        personalization = {
            idx: float(self.graph.nodes[idx]["node_data"].action_score)
            for idx in self.graph.nodes
        }

        total_seed = sum(personalization.values())
        if total_seed > 0:
            try:
                # Use standard damping alpha=0.85 for NetworkX pagerank calculation
                pr = nx.pagerank(self.graph, weight="weight", personalization=personalization, alpha=0.85)
            except Exception:
                # Fallback to standard PageRank if personalization fails
                pr = nx.pagerank(self.graph, weight="weight", personalization=None, alpha=0.85)
        else:
            pr = nx.pagerank(self.graph, weight="weight", personalization=None, alpha=0.85)

        # Store the computed scores back to the node objects
        for idx, score in pr.items():
            self.graph.nodes[idx]["node_data"].pagerank_score = score

    def add_frame_node(self, feature_record: Any, action_score_record: Any) -> None:
        """
        Admits a new frame as a node, connects it to existing nodes, and runs PageRank.

        Args:
            feature_record: Dict or object containing:
                            - frame_idx (int)
                            - timestamp (float)
                            - luma_diff_energy (float)
                            - motion_magnitude (float)
                            - luma_entropy (float)
                            - refined_motion_tensor (np.ndarray)
            action_score_record: Dict or object containing:
                            - action_score (float)
                            - persistence_value (float)
        """
        # Universal helper to safely read from dict keys or object attributes
        def get_val(record: Any, key: str, default: Any = None) -> Any:
            if record is None:
                return default
            if isinstance(record, dict):
                return record.get(key, default)
            return getattr(record, key, default)

        frame_idx = int(get_val(feature_record, "frame_idx"))
        timestamp = float(get_val(feature_record, "timestamp", 0.0))
        luma_diff_energy = float(get_val(feature_record, "luma_diff_energy", 0.0))
        motion_magnitude = float(get_val(feature_record, "motion_magnitude", 0.0))
        luma_entropy = float(get_val(feature_record, "luma_entropy", 0.0))
        refined_motion_tensor = get_val(feature_record, "refined_motion_tensor", None)

        if refined_motion_tensor is None:
            refined_motion_tensor = np.zeros(1, dtype=np.float32)

        action_score = float(get_val(action_score_record, "action_score", 0.0))
        persistence_value = float(get_val(action_score_record, "persistence_value", 0.0))
        packet_size = float(get_val(feature_record, "packet_size", 0.0))
        codec_conf = float(get_val(feature_record, "codec_conf", 0.5))
        pict_type = str(get_val(feature_record, "pict_type", "?"))
        is_peak = bool(get_val(feature_record, "is_peak", False))
        divergence = float(get_val(feature_record, "divergence", 0.0))
        curl = float(get_val(feature_record, "curl", 0.0))
        jacobian_frobenius = float(get_val(feature_record, "jacobian_frobenius", 0.0))
        hessian_max_eigenvalue = float(get_val(feature_record, "hessian_max_eigenvalue", 0.0))
        motion_entropy = float(get_val(feature_record, "motion_entropy", 0.0))
        tier = self._assign_tier(action_score=action_score, is_peak=is_peak, pict_type=pict_type)

        # Instantiate AsphodelNode
        node_obj = AsphodelNode(
            frame_idx=frame_idx,
            timestamp=timestamp,
            action_score=action_score,
            persistence_value=persistence_value,
            luma_diff_energy=luma_diff_energy,
            motion_magnitude=motion_magnitude,
            luma_entropy=luma_entropy,
            refined_motion_tensor=refined_motion_tensor,
            triples=[],
            embedding=None,
            packet_size=packet_size,
            codec_conf=codec_conf,
            tier=tier,
            pict_type=pict_type,
            divergence=divergence,
            curl=curl,
            jacobian_frobenius=jacobian_frobenius,
            hessian_max_eigenvalue=hessian_max_eigenvalue,
            motion_entropy=motion_entropy,
        )

        # Add node to NetworkX graph
        self.graph.add_node(frame_idx, node_data=node_obj)

        # Recompute edge weights dynamically for the updated node set
        self._update_all_edge_weights()

        # Refresh PageRank scores based on the new graph structure
        self._update_pagerank()

    def _assign_tier(self, action_score: float, is_peak: bool = False, pict_type: str = "?") -> str:
        """Map codec/action signals into the peak → salient → candidate hierarchy."""
        if is_peak or str(pict_type).upper().startswith("I"):
            return "L1_PEAK"
        if action_score >= self.salient_thresh:
            return "L2_SALIENT"
        return "L3_CANDIDATE"

    def enrich_node(self, frame_idx: int, triples: list, embedding: np.ndarray) -> None:
        """
        Enriches an existing node with semantic text triples and a CLIP embedding array.
        Recalculates edge weights to other enriched nodes using the hybrid formula, and runs PageRank.

        Args:
            frame_idx: The node's frame_idx identifier.
            triples: A list of KnowledgeTriple objects extracted by ARIA.
            embedding: CLIP embedding numpy array.
        """
        if frame_idx not in self.graph.nodes:
            raise KeyError(f"Frame index {frame_idx} not found in the graph.")

        node_data = self.graph.nodes[frame_idx]["node_data"]
        node_data.triples = triples
        node_data.embedding = embedding

        # Recompute all edge weights (hybrid weights will now apply between this node and other enriched ones)
        self._update_all_edge_weights()

        # Re-run PageRank now that the transition graph has new hybrid weights
        self._update_pagerank()

    def add_frame_nodes_bulk(
        self,
        feature_records: list,
        action_score_records: list,
        node_groups: list[list] | None = None,
    ) -> None:
        """
        Batch-insert multiple frame nodes.

        Adds all nodes to the graph first, then calls _update_all_edge_weights and
        _update_pagerank ONCE — instead of once per node as add_frame_node() does.

        Complexity:
            add_frame_node() called N times: O(N * N²) = O(N³)
            add_frame_nodes_bulk() for N nodes: O(N²)  ← this method

        Args:
            feature_records:     list of feature dicts (same schema as add_frame_node)
            action_score_records: parallel list of action_score dicts
        """
        def get_val(record: Any, key: str, default: Any = None) -> Any:
            if record is None:
                return default
            if isinstance(record, dict):
                return record.get(key, default)
            return getattr(record, key, default)

        for feature_record, action_score_record in zip(feature_records, action_score_records):
            frame_idx         = int(get_val(feature_record, "frame_idx"))
            timestamp         = float(get_val(feature_record, "timestamp", 0.0))
            luma_diff_energy   = float(get_val(feature_record, "luma_diff_energy", 0.0))
            motion_magnitude  = float(get_val(feature_record, "motion_magnitude", 0.0))
            luma_entropy           = float(get_val(feature_record, "luma_entropy", 0.0))
            refined_motion_tensor = get_val(feature_record, "refined_motion_tensor", None)
            if refined_motion_tensor is None:
                refined_motion_tensor = np.zeros(1, dtype=np.float32)

            action_score    = float(get_val(action_score_record, "action_score", 0.0))
            persistence_value = float(get_val(action_score_record, "persistence_value", 0.0))
            packet_size     = float(get_val(feature_record, "packet_size", 0.0))
            codec_conf      = float(get_val(feature_record, "codec_conf", 0.5))
            pict_type       = str(get_val(feature_record, "pict_type", "?"))
            is_peak         = bool(get_val(feature_record, "is_peak", False))
            divergence             = float(get_val(feature_record, "divergence", 0.0))
            curl                   = float(get_val(feature_record, "curl", 0.0))
            jacobian_frobenius     = float(get_val(feature_record, "jacobian_frobenius", 0.0))
            hessian_max_eigenvalue = float(get_val(feature_record, "hessian_max_eigenvalue", 0.0))
            motion_entropy         = float(get_val(feature_record, "motion_entropy", 0.0))
            tier            = self._assign_tier(action_score=action_score, is_peak=is_peak, pict_type=pict_type)

            node_obj = AsphodelNode(
                frame_idx=frame_idx,
                timestamp=timestamp,
                action_score=action_score,
                persistence_value=persistence_value,
                luma_diff_energy=luma_diff_energy,
                motion_magnitude=motion_magnitude,
                luma_entropy=luma_entropy,
                refined_motion_tensor=refined_motion_tensor,
                triples=[],
                embedding=None,
                packet_size=packet_size,
                codec_conf=codec_conf,
                tier=tier,
                pict_type=pict_type,
                divergence=divergence,
                curl=curl,
                jacobian_frobenius=jacobian_frobenius,
                hessian_max_eigenvalue=hessian_max_eigenvalue,
                motion_entropy=motion_entropy,
            )
            self.graph.add_node(frame_idx, node_data=node_obj)

        # Single recompute after all nodes inserted — the key efficiency gain.
        self._update_all_edge_weights(node_groups=node_groups)
        self._update_pagerank()

    def enrich_nodes_bulk(self, enrichment_map: dict, node_groups: list[list] | None = None) -> None:
        """
        Batch-enrich multiple nodes with CLIP embeddings.

        Applies all embeddings first, then calls _update_all_edge_weights and
        _update_pagerank ONCE instead of once per node as enrich_node() does.

        Args:
            enrichment_map: {frame_idx (int): embedding (np.ndarray)}
        """
        for frame_idx, embedding in enrichment_map.items():
            if frame_idx not in self.graph.nodes:
                continue
            node_data = self.graph.nodes[frame_idx]["node_data"]
            node_data.triples = []
            node_data.embedding = embedding

        # Single recompute after all enrichments applied.
        self._update_all_edge_weights(node_groups=node_groups)
        self._update_pagerank()

    def retrieve(
        self,
        query_embedding: np.ndarray | None,
        query_action_score: float,
        top_k: int = 5
    ) -> list[AsphodelNode]:
        """
        Performs hybrid retrieval scoring of nodes based on semantics, query-motion
        similarity, persistence, and graph PageRank.
        
        Formula:
          Total_Score =
              semantic_similarity      * alpha
            + query_motion_similarity  * beta
            + persistence_value        * gamma
            + pagerank_score           * delta
          
        If query_embedding is None or nodes lack embeddings (cold-start),
        the semantic contribution becomes zero.
        
        Args:
          query_embedding:    Target query CLIP embedding (can be None).
          query_action_score: Target action score of interest.
          top_k:              Number of top matches to return.
          
        Returns:
          A list of AsphodelNode objects sorted by Total_Score in descending order.
        """
        node_ids = list(self.graph.nodes)
        if not node_ids:
            return []

        nodes_data = [self.graph.nodes[n]["node_data"] for n in node_ids]

        # Calculate max_score_range of action scores dynamically to normalize motion differences
        scores = [node.action_score for node in nodes_data]
        max_score_range = max(scores) - min(scores) if len(scores) > 1 else 0.0

        scored_nodes = []
        for node in nodes_data:
            semantic_sim = 0.0
            if node.embedding is not None and query_embedding is not None:
                norm_node = np.linalg.norm(node.embedding)
                norm_query = np.linalg.norm(query_embedding)
                if norm_node > 0.0 and norm_query > 0.0:
                    semantic_sim = float(np.dot(node.embedding, query_embedding) / (norm_node * norm_query))
            
            if max_score_range == 0.0:
                motion_query_sim = 1.0
            else:
                motion_query_sim = max(
                    0.0,
                    1.0 - (abs(node.action_score - float(query_action_score)) / max_score_range),
                )

            comp_sem = self.alpha * semantic_sim
            comp_act = self.beta * motion_query_sim
            comp_pers = self.gamma * node.persistence_value
            comp_pr = self.delta * node.pagerank_score
            final_score = comp_sem + comp_act + comp_pers + comp_pr
            
            if self.debug_retrieval:
                print(f"[DEBUG] Node {node.frame_idx} retrieval score contributions: "
                      f"semantic = {comp_sem:.4f} (sim={semantic_sim:.4f}, wt={self.alpha:.2f}), "
                      f"motion_query = {comp_act:.4f} (sim={motion_query_sim:.4f}, wt={self.beta:.2f}), "
                      f"persistence = {comp_pers:.4f} (persist={node.persistence_value:.4f}, wt={self.gamma:.2f}), "
                      f"pagerank = {comp_pr:.4f} (pr={node.pagerank_score:.4f}, wt={self.delta:.2f}) | "
                      f"final = {final_score:.4f}")
                  
            node.last_retrieval_score = final_score
            node.retrieval_contributions = {
                "semantic": comp_sem,
                "motion_query": comp_act,
                "persistence": comp_pers,
                "pagerank": comp_pr,
                "semantic_similarity": semantic_sim,
                "motion_query_similarity": motion_query_sim,
                "tier": node.tier,
                "scene_id": node.scene_id,
            }
            scored_nodes.append((node, final_score))

        # Sort descending by calculated score
        scored_nodes.sort(key=lambda item: item[1], reverse=True)
        return [node for node, score in scored_nodes[:top_k]]

    def retrieve_ppr(
        self,
        query_embedding: np.ndarray | None,
        top_k: int = 5,
        damping: float = 0.5,
        lambda_: float = 0.5,
        *,
        graph_override: "nx.Graph | None" = None,
    ) -> list[AsphodelNode]:
        """
        Codec-discounted Personalized PageRank retrieval (6.2b mechanism).

        Seed = additive rank-space blend: λ·rank_pct(sem_sim) + (1-λ)·rank_pct(codec_conf),
        ReLU-clamped, sum-normalized. codec_conf is pre-computed at ingest (per-pict-type
        normalized when config.codec_conf_pictype_norm=True) — NOT re-normalized here.

        Falls back to uniform teleport when all seeds are zero.

        SCENE-002: Returns [] immediately when query_embedding has zero norm —
        a degenerate query would make sem_rank uniform, which collapses PPR to
        unguided PageRank and yields unpredictable results. Let the fallback
        sort (action_score + luma_diff_energy) take over instead.

        graph_override: If provided, run PPR over this graph instead of
            self.graph.  Used by the scene_sparse DESCEND path to run PPR
            over a temporary induced subgraph copy without mutating the
            production graph.  Every node in the override graph must carry
            a 'node_data' attribute.  When None (the default), the method is
            fully backward-compatible with its pre-patch behaviour.
        """
        # Resolve the graph to operate on.  This single binding is the only
        # place where the choice between the production graph and the override
        # is made.  Everything below uses `g` — there is no remaining
        # self.graph reference inside the PPR computation.
        g: nx.Graph = graph_override if graph_override is not None else self.graph

        node_ids = list(g.nodes)
        if not node_ids:
            return []

        # Validate override nodes carry node_data (fail-fast; do not silently
        # degrade into AttributeError deep inside the ranking loop).
        if graph_override is not None:
            for nid in node_ids:
                if g.nodes[nid].get("node_data") is None:
                    raise KeyError(
                        f"retrieve_ppr: graph_override node {nid!r} has no "
                        f"'node_data' attribute.  All nodes in the override graph "
                        f"must carry an AsphodelNode as 'node_data'."
                    )

        # SCENE-002: Reject zero-norm query embedding before computing sem_rank
        if query_embedding is not None:
            q_norm = float(np.linalg.norm(query_embedding))
            if q_norm < 1e-8:
                # Degenerate query — PPR would just be unguided PageRank.
                # Return empty list; callers fall back to action-score sort.
                return []

        n = len(node_ids)
        teleport_fallback = False

        # Semantic similarity (ReLU-clamped cosine)
        raw_sem: dict = {}
        for nid in node_ids:
            node = g.nodes[nid]["node_data"]
            sem = 0.0
            if query_embedding is not None and node.embedding is not None:
                norm_node = np.linalg.norm(node.embedding)
                norm_query = np.linalg.norm(query_embedding)
                if norm_node > 0.0 and norm_query > 0.0:
                    sem = float(np.dot(node.embedding, query_embedding) / (norm_node * norm_query))
            raw_sem[nid] = max(0.0, sem)

        # Rank-percentile of semantic similarities
        sem_rank = _rank_pct(raw_sem)

        # Rank-percentile of pre-computed codec_conf (do NOT re-normalize the values themselves)
        raw_codec = {nid: g.nodes[nid]["node_data"].codec_conf for nid in node_ids}
        codec_rank = _rank_pct(raw_codec)

        # Additive rank-space blend, ReLU, sum-normalize
        seed_raw = {
            nid: max(0.0, lambda_ * sem_rank[nid] + (1.0 - lambda_) * codec_rank[nid])
            for nid in node_ids
        }
        total = sum(seed_raw.values())
        if total > 0.0:
            seed = {k: v / total for k, v in seed_raw.items()}
        else:
            seed = {k: 1.0 / n for k in node_ids}
            teleport_fallback = True

        try:
            pr = nx.pagerank(
                g,
                weight="weight",
                personalization=seed,
                alpha=damping,
            )
        except (nx.NetworkXError, ZeroDivisionError):
            # L2-006: Narrowed from bare except to only catch expected failures.
            # Other exceptions (memory, unexpected graph state) should propagate.
            pr = nx.pagerank(g, weight="weight", personalization=None, alpha=damping)
            teleport_fallback = True

        for nid, score in pr.items():
            node = g.nodes[nid]["node_data"]
            node.last_retrieval_score = score
            node.retrieval_contributions = {
                "sem_rank":        sem_rank[nid],
                "codec_rank":      codec_rank[nid],
                "seed":            seed[nid],
                "ppr":             score,
                "lambda":          lambda_,
                "teleport_fallback": teleport_fallback,
                "tier":            node.tier,
                "scene_id":        node.scene_id,
            }

        sorted_ids = sorted(pr, key=lambda k: pr[k], reverse=True)
        return [g.nodes[k]["node_data"] for k in sorted_ids[:top_k]]

    def export_graph_data(self, max_edges: int | None = None) -> dict:
        """Export the actual L2 graph for debugging/UI visualization.

        This returns the NetworkX graph that retrieval/PageRank used, including
        hierarchy tiers and edge-channel weights. It avoids the old UI problem
        where graph visuals were reconstructed from retrieved frames only.
        """
        nodes = []
        for nid in sorted(self.graph.nodes):
            node = self.graph.nodes[nid]["node_data"]
            nodes.append({
                "id": int(nid),
                "frame_idx": int(node.frame_idx),
                "timestamp": float(node.timestamp),
                "tier": node.tier,
                "scene_id": int(node.scene_id),
                "pict_type": node.pict_type,
                "action_score": float(node.action_score),
                "persistence_value": float(node.persistence_value),
                "luma_diff_energy": float(node.luma_diff_energy),
                "pagerank_score": float(node.pagerank_score),
                "codec_conf": float(node.codec_conf),
                "last_retrieval_score": float(getattr(node, "last_retrieval_score", 0.0)),
            })

        edge_rows = []
        for u, v, data in self.graph.edges(data=True):
            edge_rows.append({
                "source": int(u),
                "target": int(v),
                "weight": float(data.get("weight", 0.0)),
                "edge_type": data.get("edge_type", "unknown"),
                "semantic_weight": float(data.get("semantic_weight", 0.0)),
                "motion_weight": float(data.get("motion_weight", 0.0)),
                "temporal_weight": float(data.get("temporal_weight", 0.0)),
            })
        edge_rows.sort(key=lambda e: e["weight"], reverse=True)
        if max_edges is not None:
            edge_rows = edge_rows[:max_edges]

        return {
            "nodes": nodes,
            "edges": edge_rows,
            "stats": {
                "node_count": len(nodes),
                "edge_count": self.graph.number_of_edges(),
                "edge_mode": self.graph_edge_mode,
            },
        }

    def export_to_csr(self) -> scipy.sparse.csr_matrix:
        """
        Exports the node feature records into a Scipy Compressed Sparse Row (CSR) matrix.
        
        Why CSR?
        A CSR matrix is highly efficient for downstream linear algebra, vector database lookups,
        and query optimization. It stores sparse features in contiguous rows using indices and
        pointers, keeping memory consumption extremely small for long videos.
        
        Feature vector representation per node:
          [frame_idx, timestamp, action_score, persistence_value, luma_diff_energy,
           motion_magnitude, luma_entropy, pagerank_score, ...refined_motion_tensor..., ...embedding...]
          
        Missing embeddings or motion tensors are zero-padded to maintain consistent columns.
        """
        # Sort nodes by frame_idx to preserve temporal decoding sequence in matrix rows
        sorted_nodes = sorted(
            [self.graph.nodes[n]["node_data"] for n in self.graph.nodes],
            key=lambda x: x.frame_idx
        )

        if not sorted_nodes:
            return scipy.sparse.csr_matrix((0, 0))

        # Dynamic dimension inspection to determine padding sizes
        embed_dim = 0
        for node in sorted_nodes:
            if node.embedding is not None:
                embed_dim = max(embed_dim, node.embedding.flatten().shape[0])

        motion_dim = 0
        for node in sorted_nodes:
            if node.refined_motion_tensor is not None:
                motion_dim = max(motion_dim, node.refined_motion_tensor.flatten().shape[0])

        rows = []
        for node in sorted_nodes:
            # 8 baseline scalar features
            base_feats = [
                float(node.frame_idx),
                float(node.timestamp),
                float(node.action_score),
                float(node.persistence_value),
                float(node.luma_diff_energy),
                float(node.motion_magnitude),
                float(node.luma_entropy),
                float(node.pagerank_score)
            ]

            # Flatten and append motion tensor
            if motion_dim > 0:
                if node.refined_motion_tensor is not None:
                    m_flat = node.refined_motion_tensor.flatten()
                    if m_flat.shape[0] == motion_dim:
                        base_feats.extend(m_flat.tolist())
                    else:
                        padded = np.zeros(motion_dim)
                        n_val = min(motion_dim, m_flat.shape[0])
                        padded[:n_val] = m_flat[:n_val]
                        base_feats.extend(padded.tolist())
                else:
                    base_feats.extend([0.0] * motion_dim)

            # Flatten and append embedding vector (zero-padded if missing/cold-start)
            if embed_dim > 0:
                if node.embedding is not None:
                    e_flat = node.embedding.flatten()
                    if e_flat.shape[0] == embed_dim:
                        base_feats.extend(e_flat.tolist())
                    else:
                        padded = np.zeros(embed_dim)
                        n_val = min(embed_dim, e_flat.shape[0])
                        padded[:n_val] = e_flat[:n_val]
                        base_feats.extend(padded.tolist())
                else:
                    base_feats.extend([0.0] * embed_dim)

            rows.append(base_feats)

        matrix = np.array(rows, dtype=np.float32)
        return scipy.sparse.csr_matrix(matrix)


if __name__ == "__main__":
    print("=====================================================================")
    print("IRIS L2 Asphodel Scene Graph Demo")
    print("=====================================================================")

    # 1. Initialize Graph with custom weights (alpha=0.4, beta=0.6)
    config = {"alpha": 0.4, "beta": 0.6}
    asphodel = L2Asphodel(config=config)
    print("Initialized L2Asphodel graph with alpha=0.4, beta=0.6.")

    # 2. Add Peak Frames (Cold-Start Phase)
    print("\n--- Phase 1: Cold-Start (Adding Peak Frames) ---")
    mock_frames = [
        (
            {
                "frame_idx": 10,
                "timestamp": 0.4,
                "luma_diff_energy": 0.5,
                "motion_magnitude": 12.5,
                "luma_entropy": 3.2,
                "refined_motion_tensor": np.array([1.2, 2.3], dtype=np.float32)
            },
            {"action_score": 0.8, "persistence_value": 0.7}
        ),
        (
            {
                "frame_idx": 20,
                "timestamp": 0.8,
                "luma_diff_energy": 0.2,
                "motion_magnitude": 4.1,
                "luma_entropy": 1.5,
                "refined_motion_tensor": np.array([0.5, 0.7], dtype=np.float32)
            },
            {"action_score": 0.3, "persistence_value": 0.4}
        ),
        (
            {
                "frame_idx": 30,
                "timestamp": 1.2,
                "luma_diff_energy": 0.9,
                "motion_magnitude": 18.2,
                "luma_entropy": 4.8,
                "refined_motion_tensor": np.array([2.1, 3.5], dtype=np.float32)
            },
            {"action_score": 0.95, "persistence_value": 0.9}
        )
    ]

    for f_rec, a_rec in mock_frames:
        asphodel.add_frame_node(f_rec, a_rec)
        print(f"Admitted frame node {f_rec['frame_idx']} (Action Score: {a_rec['action_score']})")

    # Display Graph Node States after Cold-Start
    print("\nGraph Node States after Cold-Start:")
    for node_id in sorted(asphodel.graph.nodes):
        node = asphodel.graph.nodes[node_id]["node_data"]
        print(f"  Frame {node.frame_idx}: Action Score = {node.action_score}, PageRank Score = {node.pagerank_score:.4f}")

    # Display Edge Weights
    print("\nEdge Weights (Pure Motion Coherence):")
    for u, v, data in asphodel.graph.edges(data=True):
        print(f"  Edge ({u} <-> {v}): Weight = {data['weight']:.4f}")

    # Retrieve in Cold-Start Phase
    print("\nRetrieval in Cold-Start Phase (Query Action Score: 0.90):")
    results = asphodel.retrieve(query_embedding=None, query_action_score=0.90, top_k=2)
    for i, r in enumerate(results):
        print(f"  Match {i+1}: Frame {r.frame_idx} (Action Score: {r.action_score}, PageRank: {r.pagerank_score:.4f})")

    # 3. Node Enrichment Phase (ARIA updates)
    print("\n--- Phase 2: Enrichment (ARIA semantic updates) ---")
    emb_10 = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    emb_30 = np.array([0.15, 0.85, 0.05, 0.0], dtype=np.float32)
    triples_10 = [("person", "walks in", "room")]
    triples_30 = [("person", "opens", "door")]

    asphodel.enrich_node(frame_idx=10, triples=triples_10, embedding=emb_10)
    asphodel.enrich_node(frame_idx=30, triples=triples_30, embedding=emb_30)
    print("Enriched frame 10 and frame 30 with semantic triples and embeddings.")

    # Check updated edge weights (Edge 10 <-> 30 uses hybrid weight formula, others cold-start)
    print("\nUpdated Edge Weights after Enrichment:")
    for u, v, data in asphodel.graph.edges(data=True):
        print(f"  Edge ({u} <-> {v}): Weight = {data['weight']:.4f}")

    # Retrieve with Query Embedding (Semantic Match)
    query_emb = np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32)
    print("\nHybrid Retrieval (Query Semantics matching Frame 10, Query Action Score: 0.85):")
    results = asphodel.retrieve(query_embedding=query_emb, query_action_score=0.85, top_k=2)
    for i, r in enumerate(results):
        print(f"  Match {i+1}: Frame {r.frame_idx} (Action Score: {r.action_score}, PageRank: {r.pagerank_score:.4f})")

    # 4. Export to CSR Matrix
    csr_matrix = asphodel.export_to_csr()
    print("\n--- Phase 3: Export to CSR Matrix ---")
    print(f"CSR Matrix Shape: {csr_matrix.shape}")
    print(f"CSR Matrix representation:\n{csr_matrix.toarray()}")
    print("=====================================================================")
