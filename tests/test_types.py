from iris.types import FrameRecord, IRISIndex


def test_framerecord_minimal_defaults():
    fr = FrameRecord(
        frame_idx=0, timestamp=0.0,
        luma_diff_energy=0.0, luma_entropy=0.0, motion_magnitude=0.0,
        action_score=0.0, persistence_value=0.0, is_peak=False,
    )
    assert fr.caption is None
    assert fr.clip_embedding is None
    assert fr.pagerank_score == 0.0


def _minimal_index_kwargs():
    return dict(
        video_path="x.mp4", frames=[], index_action_score=0.0,
        stats={}, frames_processed=0, peak_count=0,
        skipped_frames_ratio=0.0, storage_reduction_factor=0.0,
        config_snapshot={},
    )


def test_irisindex_graph_defaults_none():
    idx = IRISIndex(**_minimal_index_kwargs())
    assert idx._graph is None
    assert idx.schema_version == 1


def test_irisindex_graph_excluded_from_equality():
    a = IRISIndex(**_minimal_index_kwargs())
    b = IRISIndex(**_minimal_index_kwargs())
    assert a == b
    a._graph = object()        # differs only in _graph
    assert a == b              # compare=False -> still equal
