"""Persistence-parity regression guard: reload-then-query retrieval must equal
build-then-query retrieval on the production ppr path.

REGRESSION GUARD ONLY. If this fails, it means save_index/load_index silently
lost or altered state that the production query path depends on (config
snapshot, schema version, or per-frame data feeding retrieve_ppr). Do NOT
tune ppr_lambda/damping or "fix" rankings to force a pass here — report the
divergence and stop.

Usage: python phase6_persistence_parity.py
       python phase6_persistence_parity.py --graph_mode scene_sparse

NOTE: in scene_sparse mode, retrieve_ppr runs over the block-diagonal graph.
This is NOT real cross-scene descent (that's 2c) — it only exercises the
build == reload round-trip for the scene_sparse structure itself.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

# Lifted verbatim from phase6_verify_62b.py — do not invent new questions.
QUESTIONS = [
    "what is moving in the scene",
    "describe the main action",
    "what happens at the end",
    "is there water or a stream",
    "show a wide landscape shot",
]
TOP_K = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_mode", default="flat", choices=["flat", "scene_sparse"])
    args = parser.parse_args()

    vpath = REPO_ROOT / "videoplayback.mp4"
    if not vpath.exists():
        print(f"ERROR: {vpath} not found")
        sys.exit(1)

    cfg = IRISConfig(
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=0.5,
        ppr_damping=0.5,
        l2_retrieve_top_k=TOP_K,
        graph_mode=args.graph_mode,
    )

    print(f"Ingesting videoplayback.mp4 with production ppr config (graph_mode={args.graph_mode})...")
    sys.stdout.flush()
    built = iris_ingest.ingest(str(vpath), config=cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = Path(tmpdir) / "parity_idx"
        iris_ingest.save_index(built, idx_path)
        reloaded = iris_ingest.load_index(idx_path)

    # ── Cheap snapshot guards (silent-weight-mismatch concern) ──────────────
    assert reloaded.schema_version == built.schema_version, (
        f"schema_version mismatch: built={built.schema_version} "
        f"reloaded={reloaded.schema_version}"
    )
    assert reloaded.config_snapshot == built.config_snapshot, (
        f"config_snapshot mismatch:\n  built={built.config_snapshot}\n"
        f"  reloaded={reloaded.config_snapshot}"
    )

    # ── Max per-frame embedding delta (diagnostic only, not an assertion) ──
    built_emb = {fr.frame_idx: fr.clip_embedding for fr in built.frames}
    reloaded_emb = {fr.frame_idx: fr.clip_embedding for fr in reloaded.frames}
    max_emb_delta = 0.0
    for fi, e_built in built_emb.items():
        e_reloaded = reloaded_emb.get(fi)
        if e_built is None or e_reloaded is None:
            continue
        delta = float(np.max(np.abs(
            np.asarray(e_built, dtype=np.float32) - np.asarray(e_reloaded, dtype=np.float32)
        )))
        max_emb_delta = max(max_emb_delta, delta)

    # ── Exact ranking parity per question ───────────────────────────────────
    for q in QUESTIONS:
        emb = iris_query._embed_query(q, cfg)
        built_top = [f["frame_idx"] for f in iris_query._build_retrieved(built, emb, cfg)]
        reloaded_top = [f["frame_idx"] for f in iris_query._build_retrieved(reloaded, emb, cfg)]

        if built_top != reloaded_top:
            print("===PERSISTENCE_PARITY_FAIL===")
            print(f"Diverging question: {q!r}")
            print(f"  build-then-query : {built_top}")
            print(f"  reload-then-query: {reloaded_top}")
            print(f"max_emb_delta={max_emb_delta}")
            sys.exit(1)

    print(f"PARITY OK  max_emb_delta={max_emb_delta}")
    sys.exit(0)


if __name__ == "__main__":
    main()
