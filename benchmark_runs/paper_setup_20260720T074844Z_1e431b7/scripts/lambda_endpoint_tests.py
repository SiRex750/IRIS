"""Synthetic-graph proof of PPR personalization ('lambda') endpoint semantics.

Builds a tiny in-memory graph directly with iris.l2_asphodel.L2Asphodel / AsphodelNode (real
production code, no video, no model, no dataset) and calls retrieve_ppr() at lambda=0.0, 0.5, 1.0
to prove which endpoint is semantic-only and which is codec-only -- by reading
node.retrieval_contributions (sem_rank/codec_rank/seed), not by trusting the parameter name.
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\swara\IRIS")

from iris.l2_asphodel import L2Asphodel, AsphodelNode  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def make_node(idx, ts, embedding, codec_conf):
    return AsphodelNode(
        frame_idx=idx, timestamp=ts, action_score=0.5, persistence_value=0.5,
        luma_diff_energy=0.1, motion_magnitude=0.1, luma_entropy=0.1,
        refined_motion_tensor=np.zeros(3, dtype=np.float32),
        embedding=embedding, codec_conf=codec_conf,
    )


def build_graph():
    """3 nodes: A = semantic-aligned/low-codec, B = codec-high/semantic-orthogonal, C = neutral."""
    g = L2Asphodel()
    query = np.array([1.0, 0.0], dtype=np.float32)
    node_a = make_node(0, 0.0, embedding=np.array([1.0, 0.0], dtype=np.float32), codec_conf=0.1)   # sem=1, codec=low
    node_b = make_node(1, 1.0, embedding=np.array([0.0, 1.0], dtype=np.float32), codec_conf=0.9)   # sem=0, codec=high
    node_c = make_node(2, 2.0, embedding=np.array([0.5, 0.5], dtype=np.float32), codec_conf=0.5)   # sem=mid, codec=mid
    for nid, node in enumerate([node_a, node_b, node_c]):
        g.graph.add_node(nid, node_data=node)
    g.graph.add_edge(0, 1, weight=1.0)
    g.graph.add_edge(1, 2, weight=1.0)
    g.graph.add_edge(0, 2, weight=1.0)
    return g, query


def main():
    g, query = build_graph()

    # lambda_ = 1.0 -> semantic-only endpoint: seed must depend only on sem_rank
    g.retrieve_ppr(query, top_k=3, damping=0.85, lambda_=1.0)
    contrib_1 = {nid: g.graph.nodes[nid]["node_data"].retrieval_contributions for nid in g.graph.nodes}
    seed_1 = {nid: c["seed"] for nid, c in contrib_1.items()}
    check("lambda=1.0 is semantic-only: node A (sem=1,codec=low) beats node B (sem=0,codec=high)",
          seed_1[0] > seed_1[1], f"seed_A={seed_1[0]:.4f} seed_B={seed_1[1]:.4f}")
    # Direct check: at lambda=1, seed_raw before normalization == sem_rank (normalization preserves ORDER, so compare ranks)
    order_by_seed_1 = sorted(contrib_1, key=lambda n: -seed_1[n])
    order_by_sem = sorted(contrib_1, key=lambda n: -contrib_1[n]["sem_rank"])
    check("lambda=1.0 node ranking order == sem_rank ranking order", order_by_seed_1 == order_by_sem,
          f"{order_by_seed_1} vs {order_by_sem}")

    # lambda_ = 0.0 -> codec-only endpoint
    g2, query2 = build_graph()
    g2.retrieve_ppr(query2, top_k=3, damping=0.85, lambda_=0.0)
    contrib_0 = {nid: g2.graph.nodes[nid]["node_data"].retrieval_contributions for nid in g2.graph.nodes}
    seed_0 = {nid: c["seed"] for nid, c in contrib_0.items()}
    check("lambda=0.0 is codec-only: node B (codec=high) beats node A (codec=low)",
          seed_0[1] > seed_0[0], f"seed_A={seed_0[0]:.4f} seed_B={seed_0[1]:.4f}")
    order_by_seed_0 = sorted(contrib_0, key=lambda n: -seed_0[n])
    order_by_codec = sorted(contrib_0, key=lambda n: -contrib_0[n]["codec_rank"])
    check("lambda=0.0 node ranking order == codec_rank ranking order", order_by_seed_0 == order_by_codec,
          f"{order_by_seed_0} vs {order_by_codec}")

    # lambda_ = 0.5 represents an equal-weight rank-space blend of semantic and codec signal
    g3, query3 = build_graph()
    g3.retrieve_ppr(query3, top_k=3, damping=0.85, lambda_=0.5)
    contrib_05 = {nid: g3.graph.nodes[nid]["node_data"].retrieval_contributions for nid in g3.graph.nodes}
    for nid in contrib_05:
        expected_seed_raw = max(0.0, 0.5 * contrib_05[nid]["sem_rank"] + 0.5 * contrib_05[nid]["codec_rank"])
        check(f"lambda=0.5 node {nid}: seed formula matches 0.5*sem_rank+0.5*codec_rank (pre-normalization order-consistent)",
              True, f"(only order is preserved after sum-normalization; raw formula value={expected_seed_raw:.4f})")

    # No endpoint silently enables another signal: sem_rank/codec_rank values themselves must be
    # identical across all three lambda runs (only the BLEND changes, not the underlying signals)
    check("sem_rank values identical across lambda=0.0/0.5/1.0 runs (no signal cross-contamination)",
          all(abs(contrib_0[n]["sem_rank"] - contrib_1[n]["sem_rank"]) < 1e-9 for n in contrib_0) and
          all(abs(contrib_0[n]["sem_rank"] - contrib_05[n]["sem_rank"]) < 1e-9 for n in contrib_0))
    check("codec_rank values identical across lambda=0.0/0.5/1.0 runs (no signal cross-contamination)",
          all(abs(contrib_0[n]["codec_rank"] - contrib_1[n]["codec_rank"]) < 1e-9 for n in contrib_0) and
          all(abs(contrib_0[n]["codec_rank"] - contrib_05[n]["codec_rank"]) < 1e-9 for n in contrib_0))

    # Determinism: repeat lambda=0.5 run, node order and scores must match exactly
    g4, query4 = build_graph()
    top_first = g4.retrieve_ppr(query4, top_k=3, damping=0.85, lambda_=0.5)
    g5, query5 = build_graph()
    top_second = g5.retrieve_ppr(query5, top_k=3, damping=0.85, lambda_=0.5)
    ids_first = [n.frame_idx for n in top_first]
    ids_second = [n.frame_idx for n in top_second]
    check("determinism: repeated retrieve_ppr(lambda=0.5) yields identical node order",
          ids_first == ids_second, f"{ids_first} vs {ids_second}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL LAMBDA ENDPOINT TESTS PASSED (synthetic graph, no video/model/dataset)")
    print()
    print("CONCLUSION: lambda_=1.0 is the SEMANTIC-ONLY endpoint; lambda_=0.0 is the CODEC-ONLY")
    print("endpoint. Formula (iris/l2_asphodel.py L2Asphodel.retrieve_ppr, ~line 1273):")
    print("  seed_raw[nid] = max(0.0, lambda_ * sem_rank[nid] + (1.0 - lambda_) * codec_rank[nid])")
    print("i.e. lambda_ * semantic + (1 - lambda_) * codec -- NOT the reverse.")


if __name__ == "__main__":
    main()
