"""Capture a canonical, hashable snapshot of ingest() output for a fixed set
of videos -- used both before and after the scene_spans persistence change
to prove parity on every existing field. Mirrors save_index's manifest dict
exactly (so it covers every serialized field) but is computed in-memory,
does not touch disk .npz files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest  # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402
from iris.ingest import _json_safe  # noqa: E402

FIXED_VIDEOS = ["6936757706", "3079724515", "8900428927", "10001787725", "10030609934"]
VIDEO_DIR = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"


def manifest_like(index) -> tuple[dict, dict]:
    """Reproduce save_index's manifest dict construction (sans scene_spans,
    which doesn't exist pre-change and is checked separately post-change),
    plus a separate dict of embedding arrays for byte-level comparison."""
    frame_dicts = []
    embeddings = {}
    for fr in index.frames:
        frame_dicts.append({
            "frame_idx": fr.frame_idx, "timestamp": fr.timestamp,
            "luma_diff_energy": fr.luma_diff_energy, "luma_entropy": fr.luma_entropy,
            "motion_magnitude": fr.motion_magnitude, "action_score": fr.action_score,
            "persistence_value": fr.persistence_value, "is_peak": fr.is_peak,
            "divergence": fr.divergence, "curl": fr.curl,
            "jacobian_frobenius": fr.jacobian_frobenius, "hessian_max_eigenvalue": fr.hessian_max_eigenvalue,
            "motion_entropy": fr.motion_entropy, "caption": fr.caption,
            "pagerank_score": fr.pagerank_score, "packet_size": fr.packet_size,
            "pict_type": fr.pict_type, "codec_conf": fr.codec_conf, "scene_id": fr.scene_id,
        })
        if fr.clip_embedding is not None:
            embeddings[f"emb_{fr.frame_idx}"] = fr.clip_embedding.tobytes()

    graph_edges = []
    nx_graph = getattr(getattr(index, "_graph", None), "graph", None)
    if nx_graph is not None:
        for u, v, data in nx_graph.edges(data=True):
            graph_edges.append({
                "source": int(u), "target": int(v), "weight": float(data.get("weight", 0.0)),
                "edge_type": data.get("edge_type", "unknown"),
                "semantic_weight": float(data.get("semantic_weight", 0.0)),
                "motion_weight": float(data.get("motion_weight", 0.0)),
                "temporal_weight": float(data.get("temporal_weight", 0.0)),
            })

    manifest = {
        "schema_version": index.schema_version, "video_path": index.video_path,
        "frames": frame_dicts, "graph_edges": graph_edges,
        "index_action_score": index.index_action_score, "stats": _json_safe(index.stats),
        "frames_processed": index.frames_processed, "peak_count": index.peak_count,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "config_snapshot": index.config_snapshot,
    }
    return manifest, embeddings


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ingest_snapshot.json"
    cfg = IRISConfig()
    cfg.cerberus_mode = "none"

    snapshots = {}
    for vid in FIXED_VIDEOS:
        vpath = VIDEO_DIR / f"{vid}.mp4"
        index = iris_ingest.ingest(str(vpath), cfg)
        manifest, embeddings = manifest_like(index)
        manifest_json = json.dumps(manifest, sort_keys=True)
        emb_bytes = b"".join(embeddings[k] for k in sorted(embeddings))
        snapshots[vid] = {
            "manifest_sha256": hashlib.sha256(manifest_json.encode()).hexdigest(),
            "embeddings_sha256": hashlib.sha256(emb_bytes).hexdigest(),
            "n_frames": len(manifest["frames"]),
            "n_graph_edges": len(manifest["graph_edges"]),
            "scene_id_distribution": sorted({f["scene_id"] for f in manifest["frames"]}),
        }
        print(f"[snapshot] {vid}: n_frames={snapshots[vid]['n_frames']} manifest_sha256={snapshots[vid]['manifest_sha256'][:16]}...", flush=True)

    Path(out_path).write_text(json.dumps(snapshots, indent=2))
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
