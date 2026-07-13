import inspect

import numpy as np
import pytest

import iris.pipeline as p
import iris.query as q
from iris.types import IRISIndex, FrameRecord


# 1. Lifted wrappers are byte-for-byte verbatim copies from pipeline.py
def test_wrappers_verbatim():
    for name in ["wrapper_init_l1_cache", "wrapper_populate_cache", "wrapper_cerberus_gate"]:
        assert inspect.getsource(getattr(q, name)) == inspect.getsource(getattr(p, name)), name


# 2. Claim split is deterministic on a fixed input (pins the regex)
def test_split_claims_deterministic():
    raw = ("**Summary:** A figure walks across the room. "
           "It then sits down near a table.\n- short\n"
           "A second person enters from the left side.")
    claims = q._split_claims(raw)
    assert claims == [
        "A figure walks across the room.",
        "It then sits down near a table.",
        "short A second person enters from the left side.",
    ]


# 3. query() wiring is hermetic: fake graph + patched aria + patched cerberus
class _FakeNode:
    def __init__(self, i):
        self.frame_idx = i
        self.timestamp = float(i)
        self.luma_diff_energy = 0.1 * i
        self.action_score = 0.5
        self.persistence_value = 0.2
        self.pagerank_score = 0.3
        self.last_retrieval_score = 0.0
        self.retrieval_contributions = {}


class _FakeGraph:
    def retrieve(self, emb, query_action_score, top_k):
        return [_FakeNode(0), _FakeNode(1)]

    def retrieve_ppr(self, emb, top_k, damping=0.5, lambda_=0.5):
        return [_FakeNode(0), _FakeNode(1)]


def _index_with_fake_graph():
    frames = [FrameRecord(
        frame_idx=i, timestamp=float(i), luma_diff_energy=0.1 * i,
        luma_entropy=0.0, motion_magnitude=0.0, action_score=0.5,
        persistence_value=0.2, is_peak=(i == 0),
        caption={"semantic_caption": f"scene {i}"},
        clip_embedding=np.zeros(512, dtype=np.float32), pagerank_score=0.3,
    ) for i in range(2)]
    idx = IRISIndex(
        video_path="v.mp4", frames=frames, index_action_score=0.5,
        stats={"total": 10, "skipped": 6}, frames_processed=4, peak_count=1,
        skipped_frames_ratio=0.6, storage_reduction_factor=2.5,
        config_snapshot={},
    )
    idx._graph = _FakeGraph()
    return idx


def test_query_wiring(monkeypatch):
    monkeypatch.setattr(q, "_embed_query", lambda question, config: np.zeros(512, dtype=np.float32))
    monkeypatch.setattr(q.aria, "generate", lambda prompt, context: "A figure moves across the frame slowly.")
    # Patch cerberus gate to a fixed verdict so no DeBERTa load
    monkeypatch.setattr(q, "wrapper_cerberus_gate",
                        lambda claims, cache, score, config: (True, list(claims), [], [], False))

    idx = _index_with_fake_graph()
    res = q.query("what happens?", idx)

    assert res["frames_processed"] == 4
    assert res["peak_count"] == 1
    assert abs(res["skipped_frames_ratio"] - 0.6) < 1e-9
    assert res["retrieved_frame_idxs"] == [0, 1]
    assert res["verified"] is True
    assert res["answer"] == "A figure moves across the frame slowly."
    assert set(res.keys()) >= {
        "answer", "raw_answer", "verified", "nli_mocked", "verified_claims",
        "rejected_claims", "unverifiable_claims", "frames_processed",
        "peak_count", "compression_ratio", "skipped_frames_ratio",
        "storage_reduction_factor", "timings",
    }


# 4. PARITY GATE — old run_pipeline vs new ingest+query on the real clip.
#    aria.generate is patched to a FIXED answer in BOTH runs to remove LLM
#    nondeterminism; real Cerberus runs in both, so identical verdicts prove
#    L1 fact pools are identical. Skips if mov_bbb.mp4 is absent.
import os
VIDEO = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="test video not on disk")
def test_parity_old_vs_new(monkeypatch):
    from iris.ingest import ingest
    from iris import pipeline

    FIXED = "A figure moves across the frame. The scene appears to show motion near the center."
    monkeypatch.setattr(pipeline.aria, "generate", lambda prompt, context: FIXED)
    monkeypatch.setattr(q.aria, "generate", lambda prompt, context: FIXED)

    question = "What is happening in this video?"

    old = pipeline.run_pipeline(VIDEO, question)
    index = ingest(VIDEO)
    new = q.query(question, index)

    # deterministic structural parity
    assert new["frames_processed"] == old["frames_processed"]
    assert new["peak_count"] == old["peak_count"]
    assert abs(new["skipped_frames_ratio"] - old["skipped_frames_ratio"]) < 1e-9
    assert abs(new["storage_reduction_factor"] - old["storage_reduction_factor"]) < 1e-9
    # verification + answer parity (identical claims -> identical real-Cerberus verdict)
    assert new["verified"] == old["verified"]
    assert sorted(new["verified_claims"]) == sorted(old["verified_claims"])
    assert sorted(new["rejected_claims"]) == sorted(old["rejected_claims"])
    assert sorted(new["unverifiable_claims"]) == sorted(old["unverifiable_claims"])
    assert new["answer"] == old["answer"]
