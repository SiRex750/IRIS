import numpy as np

from iris.ingest import save_index, load_index, _build_graph
from iris.types import IRISIndex, FrameRecord


def _synthetic_index():
    frames = []
    for i in range(3):
        frames.append(FrameRecord(
            frame_idx=i, timestamp=float(i),
            luma_diff_energy=0.1 * i, luma_entropy=0.2 * i, motion_magnitude=0.0,
            action_score=0.3 * i, persistence_value=0.1 * i, is_peak=(i == 1),
            caption={"semantic_caption": f"c{i}"} if i != 2 else None,
            clip_embedding=np.full(512, 0.1 * (i + 1), dtype=np.float32),
            pagerank_score=0.0,
        ))
    snapshot = {"alpha": 0.4, "beta": 0.6, "gamma": 0.0}
    idx = IRISIndex(
        video_path="v.mp4", frames=frames, index_action_score=0.6,
        stats={"total": 10, "skipped": 6}, frames_processed=3, peak_count=1,
        skipped_frames_ratio=0.6, storage_reduction_factor=10 / 3,
        config_snapshot=snapshot,
    )
    idx._graph = _build_graph(idx.frames, snapshot)
    # pull rebuilt pagerank back onto the FrameRecords so save captures it
    for fr in idx.frames:
        fr.pagerank_score = float(idx._graph.graph.nodes[fr.frame_idx]["node_data"].pagerank_score)
    return idx


def test_roundtrip_fields(tmp_path):
    original = _synthetic_index()
    p = tmp_path / "idx"
    save_index(original, p)
    loaded = load_index(p)

    assert loaded.video_path == original.video_path
    assert loaded.frames_processed == original.frames_processed
    assert loaded.peak_count == original.peak_count
    assert loaded.config_snapshot == original.config_snapshot
    assert loaded.schema_version == original.schema_version
    assert len(loaded.frames) == 3
    for lo, og in zip(loaded.frames, original.frames):
        assert lo.frame_idx == og.frame_idx
        assert lo.is_peak == og.is_peak
        assert lo.caption == og.caption           # dict AND None cases
        assert abs(lo.action_score - og.action_score) < 1e-9
        np.testing.assert_allclose(lo.clip_embedding, og.clip_embedding, rtol=1e-6)


def test_load_rebuilds_graph_deterministically(tmp_path):
    """The graph rebuilt on load must match the graph ingest built —
    this is what makes projection+rebuild safe vs pickling the graph."""
    original = _synthetic_index()
    p = tmp_path / "idx"
    save_index(original, p)
    loaded = load_index(p)

    assert loaded._graph is not None
    assert loaded._graph.graph.number_of_nodes() == original._graph.graph.number_of_nodes() == 3
    for fi in range(3):
        pr_orig = original._graph.graph.nodes[fi]["node_data"].pagerank_score
        pr_load = loaded._graph.graph.nodes[fi]["node_data"].pagerank_score
        np.testing.assert_allclose(pr_load, pr_orig, rtol=1e-9)
