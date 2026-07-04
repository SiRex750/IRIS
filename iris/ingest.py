"""Phase 3 ingest spine: ingest(video_path) -> IRISIndex.

Runs ONCE per video. Wraps the existing iris.* modules (charon_v, action_score,
_clip, l2_asphodel) — does not reimplement them. Builds the L1/L2/L3 graph and
projects it into a serializable IRISIndex whose live graph is attached on
_graph. The old pipeline.py is left untouched until the new spine passes the
eval harness, then deleted.

Behavior is intended to match pipeline.run_pipeline's per-video half exactly
(parity is the Phase 3 exit criterion).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any

import json

import numpy as np

import iris.charon_v as charon_v
import iris.aria as aria
from iris.action_score import ActionScoreConfig, ActionScoreModule
from iris.l2_asphodel import L2Asphodel
from iris.types import FrameRecord, IRISIndex
from iris._clip import (
    get_clip_embedding_from_pil,
    get_frame_clip_embedding,
    get_semantic_and_clip_caption,
)


def _resolve_config(config: Any) -> Any:
    """Lifted from pipeline.run_pipeline (config resolution block)."""
    if config is not None:
        return config
    try:
        from iris.iris_config import ConfigManager
        cfg = ConfigManager().get_config()
        if cfg is None:
            from iris.iris_config import IRISConfig
            cfg = IRISConfig()
        return cfg
    except Exception:
        from iris.iris_config import IRISConfig
        return IRISConfig()


def _get(rec: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict or a dataclass/object record."""
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _rank_percentile(values: dict) -> dict:
    """Average-tied rank-percentile in [0, 1] over {node_id: float}."""
    keys = list(values.keys())
    n = len(keys)
    if n == 0:
        return {}
    if n == 1:
        return {keys[0]: 0.5}
    vals = np.array([values[k] for k in keys], dtype=np.float64)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
            j += 1
        avg = (i + j) / 2.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg
        i = j + 1
    rp = ranks / (n - 1)
    return {keys[idx]: float(rp[idx]) for idx in range(n)}


def _build_graph(records: list, config: Any) -> L2Asphodel:
    """Build + PageRank an L2Asphodel from enriched records.

    Shared by ingest() (fresh CLIP enrichment) and load_index() (rehydrated
    embeddings). Accepts dicts (ingest) or FrameRecords (load) via _get().

    Calls the public *_bulk methods directly — behavior-identical to the
    batch_* runtime shim in pipeline.wrapper_l2_retrieve, minus tuple packing.
    """
    graph = L2Asphodel(config=config)
    feature_records: list[dict] = []
    action_score_records: list[dict] = []
    enrichment_map: dict = {}
    for r in records:
        fi = int(_get(r, "frame_idx"))
        feature_records.append({
            "frame_idx":             fi,
            "timestamp":             float(_get(r, "timestamp", 0.0)),
            "luma_diff_energy":      float(_get(r, "luma_diff_energy", 0.0)),
            "motion_magnitude":      float(_get(r, "motion_magnitude", 0.0)),
            "luma_entropy":          float(_get(r, "luma_entropy", 0.0)),
            "refined_motion_tensor": np.asarray([
                float(_get(r, "motion_magnitude", 0.0)),
                float(_get(r, "divergence", 0.0)),
                float(_get(r, "curl", 0.0)),
                float(_get(r, "jacobian_frobenius", 0.0)),
                float(_get(r, "hessian_max_eigenvalue", 0.0)),
                float(_get(r, "motion_entropy", 0.0)),
            ], dtype=np.float32),
            "packet_size":           float(_get(r, "packet_size", 0.0)),
            "codec_conf":            float(_get(r, "codec_conf", 0.5)),
            "pict_type":             str(_get(r, "pict_type", "?")),
            "is_peak":               bool(_get(r, "is_peak", False)),
        })
        action_score_records.append({
            "action_score":      float(_get(r, "action_score", 0.0)),
            "persistence_value": float(_get(r, "persistence_value", 0.0)),
        })
        enrichment_map[fi] = _get(r, "clip_embedding", None)
    graph.add_frame_nodes_bulk(feature_records, action_score_records)
    graph.enrich_nodes_bulk(enrichment_map)
    return graph


def _build_index_from_records(
    output_frames: list[dict],
    raw_records: list[dict],
    stats: dict,
    video_path: str | Path,
    config: Any,
    nms_window: int,
) -> IRISIndex:
    """Pure-compute core: scoring -> NMS -> selection -> enrich -> graph ->
    project -> IRISIndex. No video read (parse_video happens in ingest()).
    Testable with synthetic records."""

    # 1. Continuous action scoring (lifted from run_pipeline)
    asc = ActionScoreConfig(
        luma_diff_weight=getattr(config, "luma_diff_weight", 0.5),
        motion_weight=getattr(config, "motion_weight", 0.3),
        luma_entropy_weight=getattr(config, "luma_entropy_weight", 0.2),
        peak_distance=getattr(config, "peak_distance", 5),
        peak_prominence=getattr(config, "peak_prominence", 0.05),
        persistence_threshold=getattr(config, "persistence_threshold", 0.4),
        max_prominence=getattr(config, "max_prominence", 0.5),
    )
    score_records = ActionScoreModule(config=asc).score_all(raw_records)
    action_scores = {r["frame_idx"]: r for r in score_records}

    # 2. NMS on peak decisions (lifted verbatim)
    if nms_window is not None and nms_window > 0:
        peak_indices = [idx for idx, si in action_scores.items() if si["is_peak"]]
        peak_indices.sort(key=lambda idx: action_scores[idx]["action_score"], reverse=True)
        accepted_peaks: set = set()
        for idx in peak_indices:
            if any(abs(idx - a) <= nms_window for a in accepted_peaks):
                action_scores[idx]["is_peak"] = False
            else:
                accepted_peaks.add(idx)

    # 3. Map scores back onto non-SKIP output frames (lifted verbatim)
    raw_map = {r["frame_idx"]: r for r in raw_records}
    for frame in output_frames:
        fi = frame["frame_idx"]
        si = action_scores.get(fi, {"action_score": 0.0, "is_peak": False, "persistence_value": 0.0})
        frame["action_score"] = si["action_score"]
        frame["is_peak"] = si["is_peak"]
        frame["persistence_value"] = si["persistence_value"]
        frame["luma_entropy"] = raw_map.get(fi, {}).get("luma_entropy", 0.0)
        # pict_type is stored as "frame_type" in charon_v raw records ('I'/'P'/'B'); default '?' for skipped.
        frame["pict_type"] = raw_map.get(fi, {}).get("frame_type", "?")

    # 4. Frame-selection strategy (config-driven, NOT query-driven) — lifted verbatim
    strategy = getattr(config, "retrieval_strategy", "hybrid")
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    if strategy == "peak_only":
        frames_to_index = [f for f in output_frames if f.get("is_peak", False)]
    elif strategy == "top_k_action":
        top_n = l2_retrieve_top_k * 3
        frames_to_index = sorted(output_frames, key=lambda x: x.get("action_score", 0.0), reverse=True)[:top_n]
    elif strategy == "peak_neighbors":
        peaks = [f for f in output_frames if f.get("is_peak", False)]
        target: set = set()
        for p in {f["frame_idx"] for f in peaks}:
            target.update(range(max(0, p - 2), p + 3))
        frames_to_index = [f for f in output_frames if f["frame_idx"] in target]
    else:  # hybrid
        frames_to_index = output_frames

    # 5. Per-video enrichment: CLIP embedding only. Captioning is lazy — moved
    #    to query time (iris.query._ensure_captions) so build cost scales with
    #    survivors embedded, not survivors captioned. caption stays None here
    #    and is populated on demand, cached by frame_idx, at most once per
    #    retrieved frame.
    device = _device()
    has_pil_cache = bool(frames_to_index) and all(
        f.get("pil_image") is not None for f in frames_to_index
    )
    if not has_pil_cache and frames_to_index:
        # Legacy path: re-open the video once and decode (no pil cache available).
        import av
        frame_map = {f["frame_idx"]: f for f in frames_to_index}
        container = av.open(str(video_path))
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in frame_map:
                f = frame_map[idx]
                clip_emb = get_frame_clip_embedding(frame, device)
                f["clip_embedding"] = clip_emb
                f["caption"] = None
        container.close()
    else:
        for f in frames_to_index:
            clip_emb = get_clip_embedding_from_pil(f["pil_image"], device)
            f["clip_embedding"] = clip_emb
            f["caption"] = None

    # 6. Build the L2 graph (edges + PageRank) once.
    graph = _build_graph(frames_to_index, config)

    # 6.5. Compute per-node codec_conf (query-independent; stored on node + FrameRecord).
    #      Source: packet_size (true demux) or action_score (proxy ablation arm).
    #      Normalization: per-pict-type (C_bytype) when codec_conf_pictype_norm=True,
    #      else global rank-percentile (C_raw baseline).
    codec_conf_source = getattr(config, "codec_conf_source", "packet_size")
    codec_conf_pictype_norm = getattr(config, "codec_conf_pictype_norm", True)

    raw_signal: dict = {}
    pict_by_fi: dict = {}
    for f in frames_to_index:
        fi = f["frame_idx"]
        if codec_conf_source == "action_score":
            raw_signal[fi] = float(f.get("action_score", 0.0))
        else:
            raw_signal[fi] = float(f.get("packet_size", 0.0))
        pict_by_fi[fi] = str(f.get("pict_type", "?"))

    if codec_conf_pictype_norm:
        groups: dict = {}
        for fi, pt in pict_by_fi.items():
            groups.setdefault(pt, []).append(fi)
        rp_map: dict = {}
        for pt, nids in groups.items():
            if len(nids) < 2:
                for fi in nids:
                    rp_map[fi] = 0.5
            else:
                sub = {fi: raw_signal[fi] for fi in nids}
                rp_map.update(_rank_percentile(sub))
    else:
        rp_map = _rank_percentile(raw_signal)

    codec_conf_map = {fi: 0.1 + 0.9 * rp_map.get(fi, 0.5) for fi in raw_signal}

    for fi, cc in codec_conf_map.items():
        if fi in graph.graph.nodes:
            graph.graph.nodes[fi]["node_data"].codec_conf = cc

    # 7. Project enriched frames into serializable FrameRecords (+ pagerank).
    frames: list[FrameRecord] = []
    for f in frames_to_index:
        fi = f["frame_idx"]
        node = graph.graph.nodes[fi]["node_data"]
        frames.append(FrameRecord(
            frame_idx=fi,
            timestamp=float(f.get("timestamp", 0.0)),
            luma_diff_energy=float(f.get("luma_diff_energy", 0.0)),
            luma_entropy=float(f.get("luma_entropy", 0.0)),
            motion_magnitude=float(f.get("motion_magnitude", 0.0)),
            action_score=float(f.get("action_score", 0.0)),
            persistence_value=float(f.get("persistence_value", 0.0)),
            is_peak=bool(f.get("is_peak", False)),
            divergence=float(f.get("divergence", 0.0)),
            curl=float(f.get("curl", 0.0)),
            jacobian_frobenius=float(f.get("jacobian_frobenius", 0.0)),
            hessian_max_eigenvalue=float(f.get("hessian_max_eigenvalue", 0.0)),
            motion_entropy=float(f.get("motion_entropy", 0.0)),
            caption=f.get("caption", None),
            clip_embedding=f.get("clip_embedding", None),
            pagerank_score=float(node.pagerank_score),
            packet_size=float(f.get("packet_size", 0.0)),
            pict_type=str(f.get("pict_type", "?")),
            codec_conf=float(codec_conf_map.get(fi, 0.5)),
        ))

    # 8. Video-level scalars (precomputed; index_action_score == old query_action_score)
    index_action_score = max((f.get("action_score", 0.0) for f in frames_to_index), default=0.5)
    frames_processed = len(output_frames)
    peak_count = len([f for f in output_frames if f.get("is_peak", False)])
    skipped_frames_ratio = float(stats["skipped"] / stats["total"]) if stats.get("total", 0) > 0 else 0.0
    storage_reduction_factor = float(stats["total"] / len(output_frames)) if len(output_frames) > 0 else 0.0

    # 9. Config provenance
    config_snapshot = asdict(config) if is_dataclass(config) else dict(getattr(config, "__dict__", {}))

    return IRISIndex(
        video_path=str(video_path),
        frames=frames,
        index_action_score=float(index_action_score),
        stats=stats,
        frames_processed=frames_processed,
        peak_count=peak_count,
        skipped_frames_ratio=skipped_frames_ratio,
        storage_reduction_factor=storage_reduction_factor,
        config_snapshot=config_snapshot,
        _graph=graph,
    )


def ingest(video_path: str | Path, config: Any = None, *, nms_window: int = 10) -> IRISIndex:
    """Run once per video. Decode + score + select + enrich + build graph."""
    config = _resolve_config(config)
    aria.run_diagnostics()
    output_frames, stats, raw_records = charon_v.parse_video(
        str(video_path),
        return_stats=True,
        return_raw=True,
        candidate_thresh=config.candidate_thresh,
        salient_thresh=config.salient_thresh,
        adaptive=getattr(config, "adaptive", True),
        visual_debug_mode=getattr(config, "visual_debug_mode", False),
    )
    return _build_index_from_records(output_frames, raw_records, stats, video_path, config, nms_window)


# ── Serialization ──────────────────────────────────────────────────────────
# Single-file .npz format: embeddings stored as float32 arrays keyed
# "emb_{frame_idx}"; everything else as one embedded JSON string under
# "__manifest__". No pickle (allow_pickle=False) — portable and safe.
# The live L2 graph is NOT serialized; load_index() rebuilds it from the
# frames via _build_graph (projection + rebuild).

def _json_safe(obj):
    """Recursively convert tuple keys to strings so json.dumps succeeds."""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_index(index: IRISIndex, path: str | Path) -> None:
    """Serialize an IRISIndex to a single .npz file (np.savez appends .npz)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    embeddings: dict[str, np.ndarray] = {}
    frame_dicts: list[dict] = []
    for fr in index.frames:
        frame_dicts.append({
            "frame_idx":         fr.frame_idx,
            "timestamp":         fr.timestamp,
            "luma_diff_energy":  fr.luma_diff_energy,
            "luma_entropy":      fr.luma_entropy,
            "motion_magnitude":  fr.motion_magnitude,
            "action_score":      fr.action_score,
            "persistence_value": fr.persistence_value,
            "is_peak":                fr.is_peak,
            "divergence":             fr.divergence,
            "curl":                   fr.curl,
            "jacobian_frobenius":     fr.jacobian_frobenius,
            "hessian_max_eigenvalue": fr.hessian_max_eigenvalue,
            "motion_entropy":         fr.motion_entropy,
            "caption":                fr.caption,
            "pagerank_score":    fr.pagerank_score,
            "packet_size":       fr.packet_size,
            "pict_type":         fr.pict_type,
            "codec_conf":        fr.codec_conf,
        })
        if fr.clip_embedding is not None:
            embeddings[f"emb_{fr.frame_idx}"] = np.asarray(fr.clip_embedding, dtype=np.float32)

    graph_edges: list[dict] = []
    graph_obj = getattr(index, "_graph", None)
    nx_graph = getattr(graph_obj, "graph", None)
    if nx_graph is not None:
        for u, v, data in nx_graph.edges(data=True):
            graph_edges.append({
                "source": int(u),
                "target": int(v),
                "weight": float(data.get("weight", 0.0)),
                "edge_type": data.get("edge_type", "unknown"),
                "semantic_weight": float(data.get("semantic_weight", 0.0)),
                "motion_weight": float(data.get("motion_weight", 0.0)),
                "temporal_weight": float(data.get("temporal_weight", 0.0)),
            })

    manifest = {
        "schema_version":           index.schema_version,
        "video_path":               index.video_path,
        "frames":                   frame_dicts,
        "graph_edges":              graph_edges,
        "index_action_score":       index.index_action_score,
        "stats":                    _json_safe(index.stats),
        "frames_processed":         index.frames_processed,
        "peak_count":               index.peak_count,
        "skipped_frames_ratio":     index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "config_snapshot":          index.config_snapshot,
    }
    arrays: dict[str, np.ndarray] = {"__manifest__": np.array(json.dumps(manifest))}
    arrays.update(embeddings)
    np.savez(path, **arrays)


def load_index(path: str | Path) -> IRISIndex:
    """Deserialize an IRISIndex and rebuild its live L2 graph from frames."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    data = np.load(path, allow_pickle=False)
    manifest = json.loads(data["__manifest__"].item())

    frames: list[FrameRecord] = []
    for d in manifest["frames"]:
        emb_key = f"emb_{d['frame_idx']}"
        emb = data[emb_key].astype(np.float32) if emb_key in data.files else None
        frames.append(FrameRecord(
            frame_idx=d["frame_idx"],
            timestamp=d["timestamp"],
            luma_diff_energy=d["luma_diff_energy"],
            luma_entropy=d["luma_entropy"],
            motion_magnitude=d["motion_magnitude"],
            action_score=d["action_score"],
            persistence_value=d["persistence_value"],
            is_peak=d["is_peak"],
            divergence=d.get("divergence", 0.0),
            curl=d.get("curl", 0.0),
            jacobian_frobenius=d.get("jacobian_frobenius", 0.0),
            hessian_max_eigenvalue=d.get("hessian_max_eigenvalue", 0.0),
            motion_entropy=d.get("motion_entropy", 0.0),
            caption=d["caption"],
            clip_embedding=emb,
            pagerank_score=d["pagerank_score"],
            packet_size=d.get("packet_size", 0.0),
            pict_type=d.get("pict_type", "?"),
            codec_conf=d.get("codec_conf", 0.5),
        ))

    index = IRISIndex(
        video_path=manifest["video_path"],
        frames=frames,
        index_action_score=manifest["index_action_score"],
        stats=manifest["stats"],
        frames_processed=manifest["frames_processed"],
        peak_count=manifest["peak_count"],
        skipped_frames_ratio=manifest["skipped_frames_ratio"],
        storage_reduction_factor=manifest["storage_reduction_factor"],
        config_snapshot=manifest["config_snapshot"],
        schema_version=manifest["schema_version"],
    )
    # Rebuild the live graph (projection + rebuild). config_snapshot is a dict;
    # L2Asphodel accepts dict configs for alpha/beta/gamma.
    index._graph = _build_graph(index.frames, index.config_snapshot)
    if manifest.get("graph_edges"):
        index._graph.graph.remove_edges_from(list(index._graph.graph.edges))
        for edge in manifest["graph_edges"]:
            source = edge["source"]
            target = edge["target"]
            if source not in index._graph.graph.nodes or target not in index._graph.graph.nodes:
                continue
            index._graph.graph.add_edge(
                source,
                target,
                weight=float(edge.get("weight", 0.0)),
                edge_type=edge.get("edge_type", "unknown"),
                semantic_weight=float(edge.get("semantic_weight", 0.0)),
                motion_weight=float(edge.get("motion_weight", 0.0)),
                temporal_weight=float(edge.get("temporal_weight", 0.0)),
            )
        index._graph._refresh_scene_ids()
        index._graph._update_pagerank()
    return index
