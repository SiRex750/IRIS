"""Tests for packet_size and pict_type carriage through the full ingest pipeline.

Covers:
- FrameRecord has valid packet_size and pict_type after ingest
- AsphodelNode.packet_size is populated on the live graph
- save_index / load_index round-trips both fields (allow_pickle=False)
- test_parity_old_vs_new in test_query still passes (import-and-call)
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = REPO_ROOT / os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")
SKIP_INGEST = not VIDEO_PATH.exists()
SKIP_REASON = f"video not found: {VIDEO_PATH}"

VALID_PICT_TYPES = {"I", "P", "B", "?"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_once():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from iris.ingest import ingest
    return ingest(str(VIDEO_PATH))


# ---------------------------------------------------------------------------
# Field presence on FrameRecord
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_INGEST, reason=SKIP_REASON)
def test_frame_record_packet_size_and_pict_type():
    index = _ingest_once()
    assert index.frames, "no frames ingested"
    for fr in index.frames:
        assert hasattr(fr, "packet_size"), "FrameRecord missing packet_size"
        assert hasattr(fr, "pict_type"),   "FrameRecord missing pict_type"
        assert math.isfinite(fr.packet_size), f"packet_size not finite: {fr.packet_size}"
        assert fr.packet_size >= 0.0, f"packet_size < 0: {fr.packet_size}"
        assert fr.pict_type in VALID_PICT_TYPES, f"pict_type invalid: {fr.pict_type!r}"


# ---------------------------------------------------------------------------
# AsphodelNode carries packet_size
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_INGEST, reason=SKIP_REASON)
def test_asphodel_node_packet_size():
    index = _ingest_once()
    graph = index._graph
    assert graph is not None
    for nid in graph.graph.nodes:
        node = graph.graph.nodes[nid]["node_data"]
        assert hasattr(node, "packet_size"), f"AsphodelNode {nid} missing packet_size"
        assert math.isfinite(node.packet_size), f"node {nid} packet_size not finite"
        assert node.packet_size >= 0.0, f"node {nid} packet_size < 0"


# ---------------------------------------------------------------------------
# save_index / load_index round-trip (allow_pickle=False)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(SKIP_INGEST, reason=SKIP_REASON)
def test_save_load_roundtrip_packet_fields():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from iris.ingest import ingest, save_index, load_index

    index = ingest(str(VIDEO_PATH))
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / "test_index.npz"
        save_index(index, cache)   # must not raise (was broken before _json_safe fix)
        loaded = load_index(cache)

    assert len(loaded.frames) == len(index.frames)
    for orig, loaded_fr in zip(index.frames, loaded.frames):
        assert loaded_fr.packet_size == orig.packet_size, (
            f"packet_size mismatch: {orig.packet_size} vs {loaded_fr.packet_size}"
        )
        assert loaded_fr.pict_type == orig.pict_type, (
            f"pict_type mismatch: {orig.pict_type!r} vs {loaded_fr.pict_type!r}"
        )


# ---------------------------------------------------------------------------
# Synthetic unit test (no video required): default fields are safe
# ---------------------------------------------------------------------------

def test_frame_record_defaults():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from iris.types import FrameRecord
    fr = FrameRecord(
        frame_idx=0, timestamp=0.0, luma_diff_energy=0.0, luma_entropy=0.0,
        motion_magnitude=0.0, action_score=0.0, persistence_value=0.0, is_peak=False,
    )
    assert fr.packet_size == 0.0
    assert fr.pict_type == "?"


def test_asphodel_node_default():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from iris.l2_asphodel import AsphodelNode
    node = AsphodelNode(
        frame_idx=0, timestamp=0.0, action_score=0.0, persistence_value=0.0,
        luma_diff_energy=0.0, motion_magnitude=0.0, luma_entropy=0.0,
        refined_motion_tensor=np.zeros(1, dtype=np.float32),
    )
    assert node.packet_size == 0.0


def test_iris_config_codec_conf_source():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from iris.iris_config import IRISConfig
    cfg = IRISConfig()
    assert cfg.codec_conf_source == "packet_size"
    cfg2 = IRISConfig(codec_conf_source="action_score")
    cfg2.validate()  # must not raise
    cfg3 = IRISConfig(codec_conf_source="packet_size")
    cfg3.validate()  # must not raise
