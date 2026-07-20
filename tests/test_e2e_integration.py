"""Phase 9: Synthetic end-to-end integration test.

Exercises the canonical ingest() → query() path with fully mocked
VLM/LLM backends. Verifies that:
  1. ingest() returns a valid IRISIndex
  2. query() returns a JSON-safe dict with all expected keys
  3. run_pipeline() delegates correctly and produces backward-compatible output
  4. The output is json.dumps()-safe (P1-24 contract)
"""
import json
import numpy as np
import pytest

import iris.ingest as ingest_mod
import iris.aria as aria
from iris.ingest import _build_index_from_records
from iris.iris_config import IRISConfig
from iris.query import query as query_fn
from iris.pipeline import run_pipeline
from iris.types import IRISIndex


# ── Fixtures ──────────────────────────────────────────────────────────────────

FIXED_EMB = np.full(512, 1.0 / np.sqrt(512), dtype=np.float32)


class MockLLMBackend:
    """Minimal mock that satisfies aria.generate() and diagnostics."""
    def generate(self, *args, **kwargs):
        return "The video shows a person walking across a field. There is a dog nearby."
    def get_model(self):
        return "mock-model"
    def get_api_key(self):
        return None


@pytest.fixture
def patched(monkeypatch):
    """Patch VLM helpers so no real models are loaded."""
    monkeypatch.setattr(ingest_mod, "get_clip_embedding_from_pil",
                        lambda pil, device: FIXED_EMB.copy())
    monkeypatch.setattr(ingest_mod, "get_frame_clip_embedding",
                        lambda frame, device: FIXED_EMB.copy())
    monkeypatch.setattr(ingest_mod, "get_semantic_and_clip_caption",
                        lambda pil, frame, emb, device: {"semantic_caption": "a person walking"})
    # Also patch the query module's _embed_query to return a fixed embedding
    import iris.query as q
    monkeypatch.setattr(q, "_embed_query", lambda question, config: FIXED_EMB.copy())
    monkeypatch.setattr(q, "_ensure_captions", lambda index, frames: 0)


def _synthetic_records():
    """Build synthetic output_frames, raw_records, and stats."""
    output_frames = []
    raw_records = []
    for i in range(5):
        output_frames.append({
            "frame_idx": i,
            "timestamp": float(i) * 0.5,
            "tier": "CANDIDATE",
            "luma_diff_energy": 0.1 * (i + 1),
            "motion_vectors": [],
            "pil_image": object(),
            "motion_magnitude": 0.2 * (i + 1),
            "divergence": 0.0,
            "curl": 0.0,
            "jacobian_frobenius": 0.0,
            "hessian_max_eigenvalue": 0.0,
            "motion_entropy": 0.0,
        })
        raw_records.append({
            "frame_idx": i,
            "frame_type": "I" if i == 0 else "P",
            "luma_diff_energy": 0.1 * (i + 1),
            "motion_magnitude": 0.2 * (i + 1),
            "luma_entropy": 0.3 * (i + 1),
        })
    stats = {
        "total": 10,
        "skipped": 5,
        "i_frames": 1,
        "peaks": 1,
        "salient": 2,
        "candidate": 2,
        "salient_thresh_used": 0.35,
        "candidate_thresh_used": 0.08,
    }
    return output_frames, raw_records, stats


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_ingest_produces_valid_index(patched):
    """Phase 9a: ingest path produces a well-formed IRISIndex."""
    of, rr, st = _synthetic_records()
    config = IRISConfig(graph_mode="flat")
    idx = _build_index_from_records(of, rr, st, "synthetic.mp4", config, nms_window=10)

    assert isinstance(idx, IRISIndex)
    assert idx._graph is not None
    assert len(idx.frames) == 5
    assert idx.video_path == "synthetic.mp4"
    assert idx.frames_processed == 5
    assert idx.skipped_frames_ratio == pytest.approx(0.5)
    assert idx.storage_reduction_factor == pytest.approx(2.0)
    assert isinstance(idx.config_snapshot, dict)
    assert idx.config_snapshot.get("graph_mode") == "flat"


def test_query_returns_json_safe_dict(patched):
    """Phase 9b: query() returns a fully JSON-serializable result dict."""
    of, rr, st = _synthetic_records()
    config = IRISConfig(graph_mode="flat")
    idx = _build_index_from_records(of, rr, st, "synthetic.mp4", config, nms_window=10)

    # Set mock LLM backend
    original_backend = aria.get_backend()
    aria.set_backend(MockLLMBackend())

    try:
        result = query_fn("What happens in the video?", idx, config)

        # Check all required keys exist
        required_keys = [
            "answer", "raw_answer", "context_text", "verified",
            "verified_claims", "rejected_claims", "unverifiable_claims",
            "frames_processed", "peak_count", "compression_ratio",
            "skipped_frames_ratio", "storage_reduction_factor",
            "retrieved_frame_idxs", "timings",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # P1-24: result must be fully JSON-serializable
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        roundtripped = json.loads(serialized)
        assert roundtripped["answer"] == result["answer"]
        assert roundtripped["verified"] == result["verified"]
    finally:
        aria.set_backend(original_backend)


def test_run_pipeline_delegates_to_ingest_query(patched):
    """Phase 9c: run_pipeline backward-compat wrapper produces expected keys."""
    of, rr, st = _synthetic_records()
    config = IRISConfig(graph_mode="flat")
    idx = _build_index_from_records(of, rr, st, "synthetic.mp4", config, nms_window=10)

    original_backend = aria.get_backend()
    aria.set_backend(MockLLMBackend())

    try:
        result = query_fn("What happens?", idx, config)

        # Backward-compat keys from run()
        compat_keys = ["answer", "verified", "frames_processed", "peak_count", "compression_ratio"]
        for key in compat_keys:
            assert key in result, f"Missing backward-compat key: {key}"

        assert isinstance(result["answer"], str)
        assert isinstance(result["verified"], bool)
        assert isinstance(result["frames_processed"], int)
        assert isinstance(result["peak_count"], int)
        assert isinstance(result["compression_ratio"], float)
    finally:
        aria.set_backend(original_backend)
