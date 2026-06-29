"""Production tests for codec-discounted PPR (6.2b).

Synthetic-only: no video file required. Covers:
  - IRISConfig new fields + validation
  - AsphodelNode.codec_conf default
  - per-pict-type normalization (flag on vs off)
  - retrieve_ppr lambda continuum (1.0 = sem-only, 0.0 = codec-only, 0.5 differs)
  - determinism
  - codec_conf round-trip through save/load
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from iris.iris_config import IRISConfig
from iris.l2_asphodel import L2Asphodel, AsphodelNode
from iris.types import FrameRecord, IRISIndex
import iris.ingest as iris_ingest
from iris.ingest import _rank_percentile


# ── graph factory ───────────────────────────────────────────────────────────

def _make_graph(frames: list[dict]) -> L2Asphodel:
    """Build a small L2Asphodel from synthetic dicts; codec_conf set from dict."""
    g = L2Asphodel()
    feature_records, action_records, emb_map = [], [], {}
    for f in frames:
        fi = f["frame_idx"]
        feature_records.append({
            "frame_idx":             fi,
            "timestamp":             float(f.get("timestamp", 0.0)),
            "luma_diff_energy":      0.0,
            "motion_magnitude":      0.0,
            "luma_entropy":          0.0,
            "refined_motion_tensor": np.zeros(1, dtype=np.float32),
            "packet_size":           float(f.get("packet_size", 0.0)),
            "codec_conf":            float(f.get("codec_conf", 0.5)),
        })
        action_records.append({
            "action_score":      float(f.get("action_score", 0.1)),
            "persistence_value": 0.0,
        })
        if f.get("embedding") is not None:
            emb_map[fi] = np.array(f["embedding"], dtype=np.float32)
    g.add_frame_nodes_bulk(feature_records, action_records)
    if emb_map:
        g.enrich_nodes_bulk(emb_map)
    return g


# ── helpers ─────────────────────────────────────────────────────────────────

def _compute_cc(output_frames: list[dict], pictype_norm: bool) -> dict:
    """Replicate ingest step-6.5 codec_conf computation for testing."""
    raw = {f["frame_idx"]: float(f.get("packet_size", 0.0)) for f in output_frames}
    pict = {f["frame_idx"]: str(f.get("pict_type", "?")) for f in output_frames}
    if pictype_norm:
        groups: dict = {}
        for fi, pt in pict.items():
            groups.setdefault(pt, []).append(fi)
        rp: dict = {}
        for pt, nids in groups.items():
            if len(nids) < 2:
                for fi in nids:
                    rp[fi] = 0.5
            else:
                rp.update(_rank_percentile({fi: raw[fi] for fi in nids}))
    else:
        rp = _rank_percentile(raw)
    return {fi: 0.1 + 0.9 * rp[fi] for fi in rp}


# ── Test 1: IRISConfig new fields and validation ─────────────────────────────

def test_irisconfig_ppr_defaults():
    cfg = IRISConfig()
    assert cfg.ppr_lambda == 0.5
    assert cfg.ppr_damping == 0.5
    assert cfg.codec_conf_pictype_norm is True
    cfg.validate()


def test_irisconfig_ppr_lambda_bounds():
    with pytest.raises(AssertionError):
        IRISConfig(ppr_lambda=1.01).validate()
    with pytest.raises(AssertionError):
        IRISConfig(ppr_lambda=-0.01).validate()
    IRISConfig(ppr_lambda=0.0).validate()
    IRISConfig(ppr_lambda=1.0).validate()


def test_irisconfig_ppr_damping_bounds():
    with pytest.raises(AssertionError):
        IRISConfig(ppr_damping=0.0).validate()
    with pytest.raises(AssertionError):
        IRISConfig(ppr_damping=1.0).validate()
    IRISConfig(ppr_damping=0.01).validate()
    IRISConfig(ppr_damping=0.99).validate()


# ── Test 2: AsphodelNode.codec_conf default ──────────────────────────────────

def test_asphodel_node_codec_conf_default():
    node = AsphodelNode(
        frame_idx=0, timestamp=0.0, action_score=0.0, persistence_value=0.0,
        luma_diff_energy=0.0, motion_magnitude=0.0, luma_entropy=0.0,
        refined_motion_tensor=np.zeros(1, dtype=np.float32),
    )
    assert node.codec_conf == 0.5


# ── Test 3: per-pict-type normalization ──────────────────────────────────────

def test_per_type_normalization_median_i():
    """I-frame with median I packet_size (~200) gets cc≈0.55, not driven by P-frames."""
    output_frames = [
        {"frame_idx": 0, "packet_size": 100, "pict_type": "I"},
        {"frame_idx": 1, "packet_size": 200, "pict_type": "I"},  # median I
        {"frame_idx": 2, "packet_size": 300, "pict_type": "I"},
        {"frame_idx": 3, "packet_size": 500, "pict_type": "P"},  # larger than any I
        {"frame_idx": 4, "packet_size": 400, "pict_type": "P"},
    ]
    cc_on  = _compute_cc(output_frames, pictype_norm=True)
    cc_off = _compute_cc(output_frames, pictype_norm=False)

    # flag ON: node 1 (median I) gets cc = 0.1 + 0.9*0.5 = 0.55
    assert abs(cc_on[1] - 0.55) < 1e-9, f"Expected 0.55, got {cc_on[1]}"
    # flag OFF: node 1 driven by P-frames → cc = 0.1 + 0.9*(1/4) = 0.325
    assert abs(cc_off[1] - 0.325) < 1e-9, f"Expected 0.325, got {cc_off[1]}"


def test_per_type_normalization_frac_i_top8():
    """Per-type norm drops frac_I in top-codec_conf from ~1.0 toward base rate."""
    # 4 I-frames, 12 P-frames; I packet_sizes are all large
    output_frames = []
    for i in range(4):
        output_frames.append({"frame_idx": i, "packet_size": 10000 + i*100, "pict_type": "I"})
    for i in range(12):
        output_frames.append({"frame_idx": 4 + i, "packet_size": 100 + i*50, "pict_type": "P"})

    cc_on  = _compute_cc(output_frames, pictype_norm=True)
    cc_off = _compute_cc(output_frames, pictype_norm=False)

    top4_raw = sorted(cc_off, key=lambda fi: cc_off[fi], reverse=True)[:4]
    top4_norm = sorted(cc_on,  key=lambda fi: cc_on[fi],  reverse=True)[:4]

    pict = {f["frame_idx"]: f["pict_type"] for f in output_frames}
    frac_raw  = sum(1 for fi in top4_raw  if pict[fi] == "I") / 4
    frac_norm = sum(1 for fi in top4_norm if pict[fi] == "I") / 4

    assert frac_raw == 1.0, f"C_raw top-4 should be all I, got {frac_raw}"
    assert frac_norm < frac_raw, "Per-type norm must reduce I-frame fraction in top-k"


# ── Test 4: λ continuum ────────────────────────────────────────────────────

# Graph: 5 nodes; query closest to frame 0 (sem winner); codec winner = frame 2
_FRAMES = [
    {"frame_idx": 0, "codec_conf": 0.1, "action_score": 0.1, "embedding": [1, 0, 0, 0, 0]},
    {"frame_idx": 1, "codec_conf": 0.4, "action_score": 0.3, "embedding": [0, 1, 0, 0, 0]},
    {"frame_idx": 2, "codec_conf": 1.0, "action_score": 0.9, "embedding": [0, 0, 1, 0, 0]},
    {"frame_idx": 3, "codec_conf": 0.7, "action_score": 0.7, "embedding": [0, 0, 0, 1, 0]},
    {"frame_idx": 4, "codec_conf": 0.55, "action_score": 0.5, "embedding": [0, 0, 0, 0, 1]},
]
_QUERY = np.array([1, 0, 0, 0, 0], dtype=np.float32)


def test_lambda_1_semantic_top():
    g = _make_graph(_FRAMES)
    top = g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=1.0)
    assert top[0].frame_idx == 0, f"λ=1.0 top should be frame 0 (semantic winner), got {[n.frame_idx for n in top]}"


def test_lambda_0_codec_top():
    g = _make_graph(_FRAMES)
    top = g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=0.0)
    assert top[0].frame_idx == 2, f"λ=0.0 top should be frame 2 (codec winner), got {[n.frame_idx for n in top]}"


def test_lambda_half_differs_from_extremes():
    g = _make_graph(_FRAMES)
    order_1    = tuple(n.frame_idx for n in g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=1.0))
    order_0    = tuple(n.frame_idx for n in g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=0.0))
    order_half = tuple(n.frame_idx for n in g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=0.5))
    assert order_half != order_1, "λ=0.5 ordering must differ from λ=1.0"
    assert order_half != order_0, "λ=0.5 ordering must differ from λ=0.0"


# ── Test 5: Determinism ────────────────────────────────────────────────────

def test_retrieve_ppr_determinism():
    g = _make_graph(_FRAMES)
    r1 = [n.frame_idx for n in g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=0.5)]
    r2 = [n.frame_idx for n in g.retrieve_ppr(_QUERY, top_k=5, damping=0.5, lambda_=0.5)]
    assert r1 == r2, f"Non-deterministic: {r1} vs {r2}"


# ── Test 6: codec_conf round-trip through save/load ──────────────────────────

def test_codec_conf_roundtrip():
    frames = [
        FrameRecord(
            frame_idx=0, timestamp=0.0, luma_diff_energy=0.0, luma_entropy=0.0,
            motion_magnitude=0.0, action_score=0.5, persistence_value=0.0,
            is_peak=False, codec_conf=0.75, pict_type="I", packet_size=200.0,
            clip_embedding=np.zeros(512, dtype=np.float32),
        ),
        FrameRecord(
            frame_idx=1, timestamp=0.1, luma_diff_energy=0.0, luma_entropy=0.0,
            motion_magnitude=0.0, action_score=0.3, persistence_value=0.0,
            is_peak=False, codec_conf=0.325, pict_type="P", packet_size=100.0,
            clip_embedding=np.zeros(512, dtype=np.float32),
        ),
    ]
    index = IRISIndex(
        video_path="synthetic",
        frames=frames,
        index_action_score=0.5,
        stats={"total": 2, "skipped": 0},
        frames_processed=2,
        peak_count=0,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "idx"
        iris_ingest.save_index(index, path)
        loaded = iris_ingest.load_index(path)

    # FrameRecord values preserved
    assert abs(loaded.frames[0].codec_conf - 0.75) < 1e-6
    assert abs(loaded.frames[1].codec_conf - 0.325) < 1e-6
    # Graph nodes also carry codec_conf (via _build_graph → add_frame_nodes_bulk)
    assert abs(loaded._graph.graph.nodes[0]["node_data"].codec_conf - 0.75) < 1e-6
    assert abs(loaded._graph.graph.nodes[1]["node_data"].codec_conf - 0.325) < 1e-6
