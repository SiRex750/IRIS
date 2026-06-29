"""
Unit tests for L2 Asphodel video RAG graph.

Owner: Track C
"""

from __future__ import annotations
import pytest
import numpy as np
import scipy.sparse
import networkx as nx

from iris.l2_asphodel import L2Asphodel, AsphodelNode


def test_init_and_config():
    """Verify L2Asphodel class initialization and config parsing."""
    # Test defaults
    graph_default = L2Asphodel()
    assert graph_default.alpha == 0.4
    assert graph_default.beta == 0.6
    assert isinstance(graph_default.graph, nx.Graph)

    # Test dictionary config
    dict_config = {"alpha": 0.25, "beta": 0.75}
    graph_dict = L2Asphodel(config=dict_config)
    assert graph_dict.alpha == 0.25
    assert graph_dict.beta == 0.75

    # Test object config
    class MockConfig:
        def __init__(self):
            self.alpha = 0.1
            self.beta = 0.9

    graph_obj = L2Asphodel(config=MockConfig())
    assert graph_obj.alpha == 0.1
    assert graph_obj.beta == 0.9


def test_add_frame_node():
    """Test that peak frames are admitted correctly as nodes and PageRank triggers."""
    asphodel = L2Asphodel()
    
    feat_1 = {
        "frame_idx": 10,
        "timestamp": 1.0,
        "luma_diff_energy": 0.8,
        "motion_magnitude": 5.0,
        "luma_entropy": 2.0,
        "refined_motion_tensor": np.array([0.1, 0.2])
    }
    score_1 = {
        "action_score": 0.9,
        "persistence_value": 0.85
    }

    asphodel.add_frame_node(feat_1, score_1)

    assert 10 in asphodel.graph.nodes
    node_data = asphodel.graph.nodes[10]["node_data"]
    assert isinstance(node_data, AsphodelNode)
    assert node_data.frame_idx == 10
    assert node_data.timestamp == 1.0
    assert node_data.action_score == 0.9
    assert node_data.pagerank_score == 1.0  # Only one node, PageRank defaults to 1.0

    # Add second node
    feat_2 = {
        "frame_idx": 20,
        "timestamp": 2.0,
        "luma_diff_energy": 0.4,
        "motion_magnitude": 2.0,
        "luma_entropy": 1.0,
        "refined_motion_tensor": np.array([0.05, 0.1])
    }
    score_2 = {
        "action_score": 0.4,
        "persistence_value": 0.3
    }

    asphodel.add_frame_node(feat_2, score_2)

    assert 20 in asphodel.graph.nodes
    assert asphodel.graph.has_edge(10, 20)
    
    # max_score_range = 0.9 - 0.4 = 0.5
    # coherence = 1.0 - abs(0.9 - 0.4) / 0.5 = 0.0
    # verify coherence edge weight
    edge_weight = asphodel.graph[10][20]["weight"]
    assert pytest.approx(edge_weight) == 0.0

    # Check pagerank updated
    pr_10 = asphodel.graph.nodes[10]["node_data"].pagerank_score
    pr_20 = asphodel.graph.nodes[20]["node_data"].pagerank_score
    # Node 10 has higher action_score (0.9 vs 0.4), personalized PageRank should reflect this
    assert pr_10 > pr_20


def test_dynamic_max_score_range_and_zero_division():
    """Verify coherence calculations handle dynamic range and edge case of zero-range correctly."""
    asphodel = L2Asphodel()

    # Case 1: Identical action scores -> range = 0 -> weight = 1.0
    feat_1 = {"frame_idx": 1, "timestamp": 0.1, "action_score": 0.5}
    score_1 = {"action_score": 0.5}
    feat_2 = {"frame_idx": 2, "timestamp": 0.2, "action_score": 0.5}
    score_2 = {"action_score": 0.5}

    asphodel.add_frame_node(feat_1, score_1)
    asphodel.add_frame_node(feat_2, score_2)
    assert asphodel.graph[1][2]["weight"] == 1.0

    # Case 2: Introduce range
    feat_3 = {"frame_idx": 3, "timestamp": 0.3, "action_score": 1.5}
    score_3 = {"action_score": 1.5} # min=0.5, max=1.5, range=1.0

    asphodel.add_frame_node(feat_3, score_3)
    
    # 1 <-> 2 coherence: 1 - abs(0.5 - 0.5)/1.0 = 1.0
    assert pytest.approx(asphodel.graph[1][2]["weight"]) == 1.0
    # 1 <-> 3 coherence: 1 - abs(0.5 - 1.5)/1.0 = 0.0
    assert pytest.approx(asphodel.graph[1][3]["weight"]) == 0.0
    # 2 <-> 3 coherence: 1 - abs(0.5 - 1.5)/1.0 = 0.0
    assert pytest.approx(asphodel.graph[2][3]["weight"]) == 0.0


def test_enrich_node_and_hybrid_weights():
    """Verify that enriching nodes transitions edge weights to the hybrid formula."""
    asphodel = L2Asphodel(config={"alpha": 0.4, "beta": 0.6})

    feat_1 = {"frame_idx": 10, "timestamp": 1.0}
    score_1 = {"action_score": 0.8}
    feat_2 = {"frame_idx": 20, "timestamp": 2.0}
    score_2 = {"action_score": 0.4}

    asphodel.add_frame_node(feat_1, score_1)
    asphodel.add_frame_node(feat_2, score_2)

    # Initial cold-start weight: coherence = 1 - (0.8 - 0.4) / 0.4 = 0.0
    assert pytest.approx(asphodel.graph[10][20]["weight"]) == 0.0

    # Enrich only one node (remains cold-start)
    asphodel.enrich_node(10, ["triple1"], np.array([1.0, 0.0]))
    assert pytest.approx(asphodel.graph[10][20]["weight"]) == 0.0

    # Enrich both nodes (triggers hybrid formula)
    # Cosine similarity between [1, 0] and [0.6, 0.8] is 0.6
    # Coherence is 0.0
    # Hybrid weight = (0.6 * 0.4) + (0.0 * 0.6) = 0.24
    asphodel.enrich_node(20, ["triple2"], np.array([0.6, 0.8]))
    assert pytest.approx(asphodel.graph[10][20]["weight"]) == 0.24


def test_retrieve():
    """Test cold-start and hybrid retrieval functionality."""
    asphodel = L2Asphodel(config={"alpha": 0.5, "beta": 0.5, "gamma": 0.0})

    # Add peak frames (min_score=0.1, max_score=0.9, range=0.8)
    asphodel.add_frame_node({"frame_idx": 1, "timestamp": 0.1}, {"action_score": 0.1})
    asphodel.add_frame_node({"frame_idx": 2, "timestamp": 0.2}, {"action_score": 0.5})
    asphodel.add_frame_node({"frame_idx": 3, "timestamp": 0.3}, {"action_score": 0.9})

    # Retrieval 1: Cold start (no embeddings)
    # Expected order: 3 (score 0.45), 2 (score 0.25), 1 (score 0.05)
    res = asphodel.retrieve(query_embedding=None, query_action_score=0.6, top_k=3)
    assert [node.frame_idx for node in res] == [3, 2, 1]

    # Retrieval 2: Hybrid. Enrich node 1 and 3, node 2 stays cold-start.
    # Query embedding: [1, 0]
    # Node 1: embedding [0.5, 0.866] (cos_sim = 0.5). Score = 0.5*0.5 + 0.5*0.1 = 0.30
    # Node 2: cold-start, semantic_sim = 0.0. Score = 0.5*0.0 + 0.5*0.5 = 0.25
    # Node 3: embedding [0.9, 0.435] (cos_sim = 0.9). Score = 0.5*0.9 + 0.5*0.9 = 0.90
    # Expected order: 3, 1, 2
    asphodel.enrich_node(1, [], np.array([0.5, 0.866]))
    asphodel.enrich_node(3, [], np.array([0.9, 0.435]))

    res_hybrid = asphodel.retrieve(query_embedding=np.array([1.0, 0.0]), query_action_score=0.9, top_k=3)
    assert [node.frame_idx for node in res_hybrid] == [3, 1, 2]


def test_export_to_csr():
    """Verify feature matrix building and Scipy CSR compression."""
    asphodel = L2Asphodel()

    # Empty graph
    empty_csr = asphodel.export_to_csr()
    assert empty_csr.shape == (0, 0)
    assert isinstance(empty_csr, scipy.sparse.csr_matrix)

    # Populated graph
    asphodel.add_frame_node(
        {
            "frame_idx": 2,
            "timestamp": 10.0,
            "luma_diff_energy": 0.5,
            "motion_magnitude": 1.5,
            "luma_entropy": 0.8,
            "refined_motion_tensor": np.array([0.2, 0.3])
        },
        {"action_score": 0.7, "persistence_value": 0.6}
    )
    asphodel.add_frame_node(
        {
            "frame_idx": 1,
            "timestamp": 5.0,
            "luma_diff_energy": 0.3,
            "motion_magnitude": 1.0,
            "luma_entropy": 0.4,
            "refined_motion_tensor": np.array([0.1, 0.15])
        },
        {"action_score": 0.4, "persistence_value": 0.3}
    )

    asphodel.enrich_node(2, [], np.array([0.5, 0.5, 0.5]))

    csr = asphodel.export_to_csr()
    
    # 8 scalars + 2 motion dimensions + 3 embedding dimensions = 13 features
    # 2 rows (frame 1, frame 2)
    assert csr.shape == (2, 13)
    assert isinstance(csr, scipy.sparse.csr_matrix)

    # Convert to dense to check details
    dense = csr.toarray()
    
    # Node at row 0 must be frame 1 (temporal sort verification)
    assert dense[0, 0] == 1.0  # frame_idx
    assert dense[0, 1] == 5.0  # timestamp
    assert dense[0, 2] == 0.4  # action_score
    assert dense[0, 8] == 0.1  # motion tensor first component
    assert np.all(dense[0, 10:] == 0.0)  # no embedding -> zero-padded

    # Node at row 1 must be frame 2
    assert dense[1, 0] == 2.0  # frame_idx
    assert dense[1, 1] == 10.0 # timestamp
    assert dense[1, 2] == 0.7  # action_score
    assert dense[1, 8] == 0.2  # motion tensor first component
    assert np.all(dense[1, 10:] == 0.5)  # embedding values
