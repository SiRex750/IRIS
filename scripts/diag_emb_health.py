"""Diagnostic-only, no production changes: embedding-health check on the
videoplayback scene_sparse index. Throwaway verification script -- decides
whether the 2c-i max-sim numbers are trustworthy or degenerate upstream.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    vpath = REPO_ROOT / "videoplayback.mp4"
    if not vpath.exists():
        print(f"ERROR: {vpath} not found")
        sys.exit(1)

    cfg = IRISConfig(
        ranking_mode="ppr", codec_conf_source="packet_size",
        codec_conf_pictype_norm=True, ppr_lambda=0.5, ppr_damping=0.5,
        l2_retrieve_top_k=8, graph_mode="scene_sparse",
    )

    print("Ingesting videoplayback.mp4 (graph_mode=scene_sparse)...")
    sys.stdout.flush()
    idx = iris_ingest.ingest(str(vpath), config=cfg)

    frames = [fr for fr in idx.frames if fr.clip_embedding is not None]
    emb_by_idx = {fr.frame_idx: np.asarray(fr.clip_embedding, dtype=np.float64) for fr in frames}
    matrix = np.stack([emb_by_idx[fr.frame_idx] for fr in frames])

    print()
    print("=== EMBEDDING HEALTH ===")
    print(f"num_survivors = {len(frames)}")
    print(f"embedding matrix shape = {matrix.shape}")

    rounded = np.round(matrix, 6)
    unique_rows = np.unique(rounded, axis=0)
    print(f"unique embedding vectors (rounded 1e-6) = {unique_rows.shape[0]} / {matrix.shape[0]}")

    std_axis0 = np.std(matrix, axis=0)
    print(f"std along axis 0: min={std_axis0.min():.6f} mean={std_axis0.mean():.6f} max={std_axis0.max():.6f}")

    norms = np.linalg.norm(matrix, axis=1)
    print(f"embedding norms: mean={norms.mean():.6f} p95={np.percentile(norms, 95):.6f} "
          f"min={norms.min():.6f} max={norms.max():.6f}")

    if 0 in emb_by_idx and 1416 in emb_by_idx:
        tied_cos = _cosine(emb_by_idx[0], emb_by_idx[1416])
        print(f"cosine(emb[frame_idx=0], emb[frame_idx=1416]) = {tied_cos:.6f}")
    else:
        print("cosine(emb[frame_idx=0], emb[frame_idx=1416]) = N/A (one or both frame_idx not present)")

    rng = np.random.default_rng(0)
    n = len(frames)
    sample_size = min(50, n)
    sample_positions = rng.choice(n, size=sample_size, replace=False)
    sample = matrix[sample_positions]
    sample_norms = np.linalg.norm(sample, axis=1, keepdims=True)
    sample_norms[sample_norms == 0.0] = 1.0
    normed = sample / sample_norms
    sims = normed @ normed.T
    off_diag_mask = ~np.eye(sample_size, dtype=bool)
    off_diag_vals = sims[off_diag_mask]
    print(f"pairwise cosine spread (random {sample_size}-survivor sample, off-diagonal): "
          f"mean={off_diag_vals.mean():.6f} min={off_diag_vals.min():.6f} max={off_diag_vals.max():.6f} "
          f"std={off_diag_vals.std():.6f}")


if __name__ == "__main__":
    main()
