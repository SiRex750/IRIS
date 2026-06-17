"""
IRIS end-to-end pipeline harness.

Wires: charon_v → l1_elysium → l2_asphodel → aria → cerberus_v

Entry point for integration testing and ablation runs.
Accepts a video path and a natural language query,
returns a verified answer string.

Owner: Track B
"""
from __future__ import annotations
from pathlib import Path


def run(video_path: str | Path, query: str) -> dict:
    """
    Run the full IRIS pipeline on a video and query.

    Returns:
        {
            "answer": str,           # final verified answer
            "verified": bool,        # cerberus_v gate result
            "frames_processed": int, # non-SKIP frames seen
            "peak_count": int,       # PEAK frames found
            "compression_ratio": float  # SKIP% of total frames
        }
    """
    # TODO: implement
    pass
