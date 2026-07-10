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
import tempfile
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

    Nodes  → one per retrieved frame with action_score, pagerank (if available),
              caption, tier, timestamp.
    Edges  → connect temporally adjacent retrieved frames; weight = motion coherence
              (1 - |action_score_i - action_score_j| / max_range).

    Returns {"nodes": [...], "edges": [...]}
    """
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

    # ── 2. Save the upload to a temp file ─────────────────────────────────────
    tmp_path: Path | None = None
    try:
        # Use NamedTemporaryFile; delete=False so we can pass the path to pipeline
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            tmp_path = Path(tmp.name)
            content = await file.read()
            tmp.write(content)

        # ── 3. Execute the pipeline ───────────────────────────────────────────
        result: dict = run_pipeline(str(tmp_path), query.strip(), verbose=True)

        # ── 4. Build graph data from debug_info ───────────────────────────────
        debug_info = result.get("debug_info", {})
        retrieved_frames = debug_info.get("retrieved_frames", [])
        result["graph_data"] = _build_graph_data(debug_info, retrieved_frames)

        # ── 5. Sanitise the result for JSON serialisation ─────────────────────
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
        # ── 6. Clean up temp file ─────────────────────────────────────────────
        if tmp_path and tmp_path.exists():
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Best-effort cleanup


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
