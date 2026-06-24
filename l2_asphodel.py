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


@dataclass
class AsphodelNode:
    """
    Schema representing a video frame node in the Asphodel spatiotemporal graph.
    
    Attributes:
        frame_idx:             Index of the frame in the decoded video sequence.
        timestamp:             Robust timestamp of the frame in seconds.
        action_score:          Motion action score from action_score.py (personalization seed).
        persistence_value:     Motion persistence score from action_score.py.
        residual_energy:       Luma difference residual proxy extracted by Charon-V.
        motion_magnitude:      Magnitude of raw motion vectors.
        entropy:               Entropy of motion vectors/energy.
        refined_motion_tensor: The bfloat16 smoothed motion tensor from the RMR layer.
        triples:               KnowledgeTriple objects added by ARIA during enrichment.
        embedding:             CLIP embedding numpy array added by ARIA (None until enriched).
        pagerank_score:        Structural importance score computed dynamically via PageRank.
    """
    frame_idx:             int
    timestamp:             float
    action_score:          float
    persistence_value:     float
    residual_energy:       float
    motion_magnitude:      float
    entropy:               float
    refined_motion_tensor: np.ndarray
    triples:               list = field(default_factory=list)
    embedding:             np.ndarray | None = None
    pagerank_score:        float = 0.0


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

        # Parse config fields if provided (handling both IRISConfig classes and dictionaries)
        if config is not None:
            self.alpha = getattr(config, "alpha", self.alpha)
            self.beta = getattr(config, "beta", self.beta)
            self.gamma = getattr(config, "gamma", self.gamma)
            if isinstance(config, dict):
                self.alpha = config.get("alpha", self.alpha)
                self.beta = config.get("beta", self.beta)
                self.gamma = config.get("gamma", self.gamma)

    def _update_all_edge_weights(self) -> None:
        """
        Recalculates edge weights between all nodes in the graph to maintain a fully
        connected, mathematically consistent spatiotemporal graph.
        
        Formula:
          Total_Weight = (cosine_similarity(i, j) * alpha) + (motion_coherence(i, j) * beta)
          
        If either endpoint lacks an embedding (cold-start phase), the weight defaults to
        motion_coherence(i, j).
        """
        node_ids = list(self.graph.nodes)
        num_nodes = len(node_ids)
        if num_nodes <= 1:
            return

        # Fetch all node data objects to analyze the range of action scores
        nodes_data = [self.graph.nodes[n]["node_data"] for n in node_ids]
        scores = [node.action_score for node in nodes_data]

        # Dynamic Math Normalization:
        # Calculate max_score_range dynamically based on the current max and min action_score.
        # This keeps the motion coherence relative and scaled between 0.0 and 1.0.
        # If all nodes share the same action_score (or range is 0), we default coherence to 1.0.
        max_score = max(scores)
        min_score = min(scores)
        max_score_range = max_score - min_score

        # Re-populate/update the fully connected edge set
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u, v = node_ids[i], node_ids[j]
                node_u = self.graph.nodes[u]["node_data"]
                node_v = self.graph.nodes[v]["node_data"]

                # Calculate motion coherence
                if max_score_range == 0.0:
                    motion_coherence = 1.0
                else:
                    motion_coherence = 1.0 - (abs(node_u.action_score - node_v.action_score) / max_score_range)

                # Determine weight based on enrichment status
                if node_u.embedding is not None and node_v.embedding is not None:
                    # Both nodes have embeddings -> Hybrid formula
                    norm_u = np.linalg.norm(node_u.embedding)
                    norm_v = np.linalg.norm(node_v.embedding)
                    if norm_u == 0.0 or norm_v == 0.0:
                        cos_sim = 0.0
                    else:
                        cos_sim = float(np.dot(node_u.embedding, node_v.embedding) / (norm_u * norm_v))
                    
                    weight = (cos_sim * self.alpha) + (motion_coherence * self.beta)
                else:
                    # Cold-start phase
                    weight = motion_coherence

                self.graph.add_edge(u, v, weight=weight)

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
                            - residual_energy (float)
                            - motion_magnitude (float)
                            - entropy (float)
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
        residual_energy = float(get_val(feature_record, "residual_energy", 0.0))
        motion_magnitude = float(get_val(feature_record, "motion_magnitude", 0.0))
        entropy = float(get_val(feature_record, "entropy", 0.0))
        refined_motion_tensor = get_val(feature_record, "refined_motion_tensor", None)

        if refined_motion_tensor is None:
            refined_motion_tensor = np.zeros(1, dtype=np.float32)

        action_score = float(get_val(action_score_record, "action_score", 0.0))
        persistence_value = float(get_val(action_score_record, "persistence_value", 0.0))

        # Instantiate AsphodelNode
        node_obj = AsphodelNode(
            frame_idx=frame_idx,
            timestamp=timestamp,
            action_score=action_score,
            persistence_value=persistence_value,
            residual_energy=residual_energy,
            motion_magnitude=motion_magnitude,
            entropy=entropy,
            refined_motion_tensor=refined_motion_tensor,
            triples=[],
            embedding=None
        )

        # Add node to NetworkX graph
        self.graph.add_node(frame_idx, node_data=node_obj)

        # Recompute edge weights dynamically for the updated node set
        self._update_all_edge_weights()

        # Refresh PageRank scores based on the new graph structure
        self._update_pagerank()

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

    def retrieve(
        self,
        query_embedding: np.ndarray | None,
        query_action_score: float,
        top_k: int = 5
    ) -> list[AsphodelNode]:
        """
        Performs hybrid retrieval scoring of nodes based on semantics and motion coherence.
        
        Formula:
          Total_Score = (Semantic_Score * alpha) + (Motion_Score * beta)
          
        If query_embedding is None or nodes lack embeddings (cold-start),
        the scoring falls back to Motion_Score only.
        
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
            
            comp_sem = self.alpha * semantic_sim
            comp_act = self.beta * node.action_score
            comp_pers = self.gamma * node.persistence_value
            final_score = comp_sem + comp_act + comp_pers
            
            print(f"[DEBUG] Node {node.frame_idx} retrieval score contributions: "
                  f"semantic = {comp_sem:.4f} (sim={semantic_sim:.4f}, wt={self.alpha:.2f}), "
                  f"action = {comp_act:.4f} (score={node.action_score:.4f}, wt={self.beta:.2f}), "
                  f"persistence = {comp_pers:.4f} (persist={node.persistence_value:.4f}, wt={self.gamma:.2f}) | "
                  f"final = {final_score:.4f}")
                  
            node.last_retrieval_score = final_score
            node.retrieval_contributions = {
                "semantic": comp_sem,
                "action": comp_act,
                "persistence": comp_pers
            }
            scored_nodes.append((node, final_score))

        # Sort descending by calculated score
        scored_nodes.sort(key=lambda item: item[1], reverse=True)
        return [node for node, score in scored_nodes[:top_k]]

    def export_to_csr(self) -> scipy.sparse.csr_matrix:
        """
        Exports the node feature records into a Scipy Compressed Sparse Row (CSR) matrix.
        
        Why CSR?
        A CSR matrix is highly efficient for downstream linear algebra, vector database lookups,
        and query optimization. It stores sparse features in contiguous rows using indices and
        pointers, keeping memory consumption extremely small for long videos.
        
        Feature vector representation per node:
          [frame_idx, timestamp, action_score, persistence_value, residual_energy,
           motion_magnitude, entropy, pagerank_score, ...refined_motion_tensor..., ...embedding...]
          
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
                float(node.residual_energy),
                float(node.motion_magnitude),
                float(node.entropy),
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
                "residual_energy": 0.5,
                "motion_magnitude": 12.5,
                "entropy": 3.2,
                "refined_motion_tensor": np.array([1.2, 2.3], dtype=np.float32)
            },
            {"action_score": 0.8, "persistence_value": 0.7}
        ),
        (
            {
                "frame_idx": 20,
                "timestamp": 0.8,
                "residual_energy": 0.2,
                "motion_magnitude": 4.1,
                "entropy": 1.5,
                "refined_motion_tensor": np.array([0.5, 0.7], dtype=np.float32)
            },
            {"action_score": 0.3, "persistence_value": 0.4}
        ),
        (
            {
                "frame_idx": 30,
                "timestamp": 1.2,
                "residual_energy": 0.9,
                "motion_magnitude": 18.2,
                "entropy": 4.8,
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
