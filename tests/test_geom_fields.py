import numpy as np
import pytest

import iris.ingest as ingest_mod
from iris.ingest import save_index, load_index
from iris.iris_config import IRISConfig

FIXED_EMB = np.full(512, 1.0 / np.sqrt(512), dtype=np.float32)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ingest_mod, "get_clip_embedding_from_pil",
                        lambda pil, device: FIXED_EMB.copy())
    monkeypatch.setattr(ingest_mod, "get_semantic_and_clip_caption",
                        lambda pil, frame, emb, device: {"semantic_caption": "x"})


def _synthetic_with_geom():
    of, rr = [], []
    for i in range(2):
        of.append({
            "frame_idx": i, "timestamp": float(i), "tier": "CANDIDATE",
            "luma_diff_energy": 0.1 * i, "motion_vectors": [], "pil_image": object(),
            "divergence": 0.5 + i, "curl": 0.6 + i, "jacobian_frobenius": 0.7 + i,
            "hessian_max_eigenvalue": 0.8 + i, "motion_entropy": 0.9 + i,
        })
        rr.append({"frame_idx": i, "frame_type": "P",
                   "luma_diff_energy": 0.1 * i, "motion_magnitude": 0.1 * i,
                   "luma_entropy": 0.2 * i})
    stats = {"total": 5, "skipped": 3, "i_frames": 1, "peaks": 0,
             "salient": 1, "candidate": 2,
             "salient_thresh_used": 0.35, "candidate_thresh_used": 0.08}
    return of, rr, stats


def test_ingest_carries_geometry(patched):
    of, rr, st = _synthetic_with_geom()
    idx = ingest_mod._build_index_from_records(of, rr, st, "v.mp4", IRISConfig(graph_mode="flat"), 10)
    by_idx = {fr.frame_idx: fr for fr in idx.frames}
    for i in range(2):
        fr = by_idx[i]
        assert fr.divergence == pytest.approx(0.5 + i)
        assert fr.curl == pytest.approx(0.6 + i)
        assert fr.jacobian_frobenius == pytest.approx(0.7 + i)
        assert fr.hessian_max_eigenvalue == pytest.approx(0.8 + i)
        assert fr.motion_entropy == pytest.approx(0.9 + i)


def test_geometry_survives_roundtrip(patched, tmp_path):
    of, rr, st = _synthetic_with_geom()
    idx = ingest_mod._build_index_from_records(of, rr, st, "v.mp4", IRISConfig(graph_mode="flat"), 10)
    p = tmp_path / "idx"
    save_index(idx, p)
    loaded = load_index(p)
    lo = {fr.frame_idx: fr for fr in loaded.frames}
    for i in range(2):
        assert lo[i].divergence == pytest.approx(0.5 + i)
        assert lo[i].hessian_max_eigenvalue == pytest.approx(0.8 + i)
        assert lo[i].motion_entropy == pytest.approx(0.9 + i)
