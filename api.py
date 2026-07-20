"""
IRIS FastAPI Backend — api.py
===============================
Wraps `run_pipeline(video_path, query, verbose=True)` from pipeline.py
and exposes it as a REST endpoint for the React frontend.

HOW TO RUN
----------
1. Install dependencies (from the IRIS project root):
       pip install fastapi "uvicorn[standard]" python-multipart

2. Start the server (from the IRIS project root):
       uvicorn api:app --reload --host 0.0.0.0 --port 8000

3. In a separate terminal, start the React frontend:
       cd iris-frontend
       npm install
       npm run dev

The frontend runs at http://localhost:5173
The API runs at http://localhost:8000
Swagger docs available at http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Ensure pipeline.py can import its sibling modules ──────────────────────────
# The IRIS modules live in the same directory as this file.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Lazy import so startup doesn't block (CLIP model loads on first call)
from iris.pipeline import run_pipeline  # noqa: E402

# ── Index cache (API-001) ──────────────────────────────────────────────────────
# Keyed by SHA-256 hash of video content so repeated questions about the same
# video reuse the pre-built IRISIndex instead of rebuilding parse/embed/graph.
import asyncio
import hashlib
from iris.ingest import ingest as iris_ingest
import iris.query as iris_query

_INDEX_CACHE: dict[str, object] = {}
_INDEX_CACHE_LOCK = asyncio.Lock()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IRIS Pipeline API",
    description=(
        "Wraps the IRIS end-to-end video-QA pipeline "
        "(Charon-V → Action Score → L2 Asphodel → L1 Elysium → ARIA → Cerberus-V)"
    ),
    version="1.1.0",
)

# Allow any localhost origin — covers Vite's fallback ports (5173, 5174, 5175…)
# and any other local dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Graph data builder ────────────────────────────────────────────────────────

def _build_graph_data(debug_info: dict, retrieved_frames: list) -> dict:
    """
    Build a graph-visualisation-friendly payload from pipeline debug data.

    Preferred path: return the real L2 graph exported by L2Asphodel.
    Fallback path: reconstruct a small graph from retrieved frames only.

    Returns {"nodes": [...], "edges": [...]}
    """
    real_graph = debug_info.get("l2_graph") if debug_info else None
    if real_graph and real_graph.get("nodes") is not None:
        nodes = []
        for node in real_graph.get("nodes", []):
            node_id = str(node.get("id", node.get("frame_idx")))
            nodes.append({
                "id": node_id,
                "frame_idx": node.get("frame_idx"),
                "timestamp": round(float(node.get("timestamp", 0.0)), 3),
                "tier": node.get("tier"),
                "scene_id": node.get("scene_id"),
                "pict_type": node.get("pict_type"),
                "action_score": round(float(node.get("action_score", 0.0)), 4),
                "persistence_value": round(float(node.get("persistence_value", 0.0)), 4),
                "luma_diff_energy": round(float(node.get("luma_diff_energy", 0.0)), 4),
                "pagerank_score": round(float(node.get("pagerank_score", 0.0)), 4),
                "codec_conf": round(float(node.get("codec_conf", 0.5)), 4),
                "last_retrieval_score": round(float(node.get("last_retrieval_score", 0.0)), 6),
            })

        edges = []
        for edge in real_graph.get("edges", []):
            weight = float(edge.get("weight", 0.0))
            edge_type = edge.get("edge_type", "unknown")
            edges.append({
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "weight": round(weight, 4),
                "label": edge_type,
                "edge_type": edge_type,
                "semantic_weight": round(float(edge.get("semantic_weight", 0.0)), 4),
                "motion_weight": round(float(edge.get("motion_weight", 0.0)), 4),
                "temporal_weight": round(float(edge.get("temporal_weight", 0.0)), 4),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": real_graph.get("stats", {}),
            "source": "l2_asphodel",
        }

    if not retrieved_frames:
        return {"nodes": [], "edges": []}

    # --- Nodes ---
    action_scores_map: dict = debug_info.get("action_scores", {}) if debug_info else {}

    nodes = []
    for f in retrieved_frames:
        idx = f.get("frame_idx", 0)
        score_info = action_scores_map.get(idx, action_scores_map.get(str(idx), {}))
        nodes.append({
            "id": str(idx),
            "frame_idx": idx,
            "timestamp": round(float(f.get("timestamp", 0.0)), 3),
            "action_score": round(float(f.get("action_score", 0.0)), 4),
            "persistence_value": round(float(f.get("persistence_value", 0.0)), 4),
            "luma_diff_energy": round(float(f.get("luma_diff_energy", 0.0)), 4),
            "luma_entropy": round(float(f.get("luma_entropy", 0.0)), 4),
            "is_peak": bool(f.get("is_peak", False)),
            "caption": (
                f.get("caption").get("semantic_caption") or f.get("caption").get("clip_label") or "—"
                if isinstance(f.get("caption"), dict)
                else f.get("caption") or "—"
            ),
            # pagerank_score may come from the score_info dict if pipeline exposes it
            "pagerank_score": round(float(score_info.get("pagerank_score", 0.0)), 4),
        })

    # --- Edges ---
    # Sort by frame_idx (temporal order) and connect neighbours
    nodes_sorted = sorted(nodes, key=lambda n: n["frame_idx"])
    scores = [n["action_score"] for n in nodes_sorted]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max(max_score - min_score, 1e-6)

    edges = []
    for i in range(len(nodes_sorted) - 1):
        u = nodes_sorted[i]
        v = nodes_sorted[i + 1]
        coherence = 1.0 - abs(u["action_score"] - v["action_score"]) / score_range
        edges.append({
            "source": u["id"],
            "target": v["id"],
            "weight": round(float(coherence), 4),
            "label": f"{coherence:.2f}",
        })

    # Also add cross-edges for nodes with very high coherence (non-adjacent)
    for i in range(len(nodes_sorted)):
        for j in range(i + 2, len(nodes_sorted)):
            u = nodes_sorted[i]
            v = nodes_sorted[j]
            coherence = 1.0 - abs(u["action_score"] - v["action_score"]) / score_range
            if coherence > 0.85:  # only strong cross-links
                edges.append({
                    "source": u["id"],
                    "target": v["id"],
                    "weight": round(float(coherence), 4),
                    "label": f"{coherence:.2f}",
                    "cross": True,
                })

    return {"nodes": nodes, "edges": edges}


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(..., description="MP4 video file to analyse"),
    query: str = Form(..., description="Natural-language question about the video"),
) -> JSONResponse:
    """
    Accept a multipart upload (video file + query string), run the full
    IRIS pipeline, and return the result dictionary as JSON.

    The returned JSON shape mirrors the dict produced by run_pipeline(),
    with an additional `graph_data` key containing the L2 Asphodel graph
    ready for frontend visualisation:
    {
        "answer":                str,
        "verified":              bool,
        "nli_mocked":            bool,
        "frames_processed":      int,
        "peak_count":            int,
        "compression_ratio":     float,
        "skipped_frames_ratio":  float,
        "storage_reduction_factor": float,
        "timings": { ... },
        "debug_info": { ... },
        "graph_data": {
            "nodes": [ { id, frame_idx, timestamp, action_score, ... } ],
            "edges": [ { source, target, weight } ]
        }
    }
    """
    # ── 1. Validate the upload ────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video format '{suffix}'. Please upload an MP4 file.",
        )

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string must not be empty.")

    # ── 2. Persist upload to a content-addressed cache path ───────────────────
    # P0-06: We used to write the upload to a NamedTemporaryFile which the
    # finally block deletes.  A later cache hit calls iris_query.query(index, ...)
    # which may need to reopen the video for lazy captioning — but the temp file
    # is already gone.  Instead we persist the content under api_index_cache/
    # using the SHA-256 hash as the filename so the path stays valid for the full
    # lifetime of the cached IRISIndex.
    content = await file.read()
    video_sha = hashlib.sha256(content).hexdigest()

    cache_dir = Path("api_index_cache")
    cache_dir.mkdir(exist_ok=True)
    persistent_video_path = cache_dir / f"{video_sha}{suffix}"

    if not persistent_video_path.exists():
        with open(persistent_video_path, "wb") as fp:
            fp.write(content)

    try:
        # P0-01: validate_video() returns a ValidationResult *dataclass*, not a dict.
        # Calling .get("status") raises AttributeError at runtime.  Use attribute access.
        try:
            from iris import codec_validator
            cv_result = codec_validator.validate_video(str(persistent_video_path))
            if cv_result.status == "reject":
                # Remove the persisted file for rejected content so it doesn't linger.
                try:
                    persistent_video_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"Video rejected by codec validator: "
                        f"{', '.join(cv_result.reasons)}"
                    ),
                )
        except ImportError:
            pass

        # ── 3. Build or reuse the IRISIndex (API-001 cache) ───────────────────
        async with _INDEX_CACHE_LOCK:
            cached_index = _INDEX_CACHE.get(video_sha)

        if cached_index is None:
            # API-003: Run blocking ingest in a thread-pool executor so the async
            # event loop is not blocked while decoding/embedding the video.
            loop = asyncio.get_event_loop()
            new_index = await loop.run_in_executor(
                None, iris_ingest, str(persistent_video_path)
            )
            async with _INDEX_CACHE_LOCK:
                _INDEX_CACHE[video_sha] = new_index
            index = new_index
        else:
            index = cached_index

        # ── 4. Run query against the index ────────────────────────────────────
        # P0-02: The original call was iris_query.query(index, query.strip(), verbose=True).
        # The signature is query(question: str, index: IRISIndex, config=None).
        # Arguments were reversed and `verbose` is not an accepted parameter.
        loop = asyncio.get_event_loop()
        result: dict = await loop.run_in_executor(
            None,
            lambda: iris_query.query(query.strip(), index),
        )


        # ── 5. Build graph data from debug_info ───────────────────────────────
        debug_info = result.get("debug_info", {})
        retrieved_frames = debug_info.get("retrieved_frames", [])
        result["graph_data"] = _build_graph_data(debug_info, retrieved_frames)

        # ── 6. Sanitise the result for JSON serialisation ─────────────────────
        result = _sanitise_for_json(result)

        return JSONResponse(content=result)

    except HTTPException:
        raise  # pass through our own 4xx errors

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[IRIS API] Pipeline error:\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    finally:
        # ── 7. No temp file cleanup needed ───────────────────────────────────
        # P0-06 fix: the video is now written to api_index_cache/ and intentionally
        # kept for the lifetime of the cached IRISIndex (same content-addressed key).
        # There is nothing to delete here.
        pass


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok", "service": "IRIS API", "version": "1.1.0"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise_for_json(obj):
    """
    Recursively convert non-JSON-serialisable objects (numpy arrays, numpy
    scalars, Path objects, etc.) into plain Python types so FastAPI's
    JSONResponse does not throw a serialisation error.
    """
    try:
        import numpy as np
        np_types = (np.integer, np.floating, np.ndarray)
    except ImportError:
        np_types = ()

    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    if np_types and isinstance(obj, np.ndarray):
        return obj.tolist()
    if np_types and isinstance(obj, np.integer):
        return int(obj)
    if np_types and isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    # If still not serialisable, convert to string as a last resort
    try:
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
