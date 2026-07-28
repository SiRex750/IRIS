"""Cut A identity gate: gate off divergence/curl/jacobian_frobenius computation
in compute_motion_geometry (default IRISConfig.compute_full_geometry=False) and
prove zero behavior change on every currently-exercised path.

Per eval_results/geometry_gate_plan.md ("Cut A"): divergence, curl,
jacobian_frobenius are computed and persisted but read by NO downstream
consumer (graph build: dead by wiring bug; L1 keep_score: never read; L1
dual-vector query: never invoked in production). hessian_max_eigenvalue and
motion_entropy (Cut B, consumed by L1 keep_score) are left fully computed and
untouched by this gate.

Re-ingests the cached VIRAT N=4,892 clip TWICE from the real video (same
config, same frames) -- once with compute_full_geometry=True (today's
behavior), once with compute_full_geometry=False (Cut A, the new default) --
and asserts:
  (4) GRAPH IDENTITY: L2Asphodel.export_graph_data() edges are bit-identical
      (tolerance 0.0) between the two runs.
  (6) L1/ARIA SAFETY: hessian_max_eigenvalue and motion_entropy are
      bit-identical between the two runs (Cut A must not perturb Cut B's
      fields).
  (7) SCHEMA: FrameRecord still carries divergence/curl/jacobian_frobenius
      keys (sentinel 0.0 under Cut A, not missing) -- no KeyError risk
      downstream.
  (8) Extraction wall-time WITH vs WITHOUT Cut A (charon_v.parse_video only,
      isolated from CLIP-encode / graph-build cost).

To keep the two full re-ingests cheap and focused on what this gate is
actually testing (the geometry computation + graph/FrameRecord wiring), CLIP
embedding is monkeypatched to a fast deterministic function of the frame's
pixel bytes (same function both runs, so semantic edge weights are exercised
and comparable, without paying real ViT-B/32 forward-pass cost twice).

Writes eval_results/geometry_cutA_identity_gate.json/.md. Does not touch any
existing index cache.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import numpy as np

import iris.ingest as iris_ingest
import iris.charon_v as charon_v
from iris.iris_config import IRISConfig

CACHE_STEM = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"
OUT_JSON = REPO / "eval_results" / "geometry_cutA_identity_gate.json"
OUT_MD = REPO / "eval_results" / "geometry_cutA_identity_gate.md"

_IRISCONFIG_FIELDS = {f.name for f in dataclasses.fields(IRISConfig)}


def _to_irisconfig(cfg_dict: dict, **overrides) -> IRISConfig:
    base = {k: v for k, v in cfg_dict.items() if k in _IRISCONFIG_FIELDS}
    base.update(overrides)
    return IRISConfig(**base)


def _fake_clip_embedding(pil_image, device):
    """Deterministic, cheap stand-in for the real CLIP forward pass -- a
    488-D+ float32 unit vector seeded from the image's raw pixel bytes, so
    identical decoded frames (same both runs, since decode is deterministic)
    produce identical embeddings without loading any model."""
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    h = hashlib.sha256(arr.tobytes()).digest()
    seed = int.from_bytes(h[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def compare_edges(edges_a: list, edges_b: list) -> dict:
    map_a = {edge_key(e["source"], e["target"]): e for e in edges_a}
    map_b = {edge_key(e["source"], e["target"]): e for e in edges_b}
    only_a = set(map_a) - set(map_b)
    only_b = set(map_b) - set(map_a)
    fields = ["weight", "semantic_weight", "motion_weight", "temporal_weight", "edge_type"]
    mismatches = []
    for k in set(map_a) & set(map_b):
        ea, eb = map_a[k], map_b[k]
        for f in fields:
            va, vb = ea[f], eb[f]
            same = (va == vb) if f == "edge_type" else (float(va) == float(vb))
            if not same:
                mismatches.append({"edge": list(k), "field": f, "a": va, "b": vb})
    return {
        "edge_count_a": len(map_a),
        "edge_count_b": len(map_b),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "field_mismatch_count": len(mismatches),
        "field_mismatches_sample": mismatches[:50],
        "identical": (len(only_a) == 0 and len(only_b) == 0 and len(mismatches) == 0),
    }


def compare_frame_fields(frames_a: list, frames_b: list) -> dict:
    by_a = {f.frame_idx: f for f in frames_a}
    by_b = {f.frame_idx: f for f in frames_b}
    assert set(by_a) == set(by_b), "frame_idx sets differ between runs"

    cutA_gated_fields = ["divergence", "curl", "jacobian_frobenius"]
    cutB_preserved_fields = ["hessian_max_eigenvalue", "motion_entropy"]
    graph_read_fields = [
        "action_score", "persistence_value", "luma_diff_energy", "motion_magnitude",
        "luma_entropy", "packet_size", "pict_type", "codec_conf", "scene_id", "pagerank_score",
    ]

    report = {
        "cutA_gated_fields_sentinel_zero_under_cut": {},
        "cutB_preserved_fields_bit_identical": {},
        "graph_read_fields_bit_identical": {},
    }

    for field in cutA_gated_fields:
        vals_a = [getattr(by_a[fi], field) for fi in by_a]  # full-geometry run
        vals_b = [getattr(by_b[fi], field) for fi in by_a]  # cut-A run
        report["cutA_gated_fields_sentinel_zero_under_cut"][field] = {
            "full_geometry_any_nonzero": any(v != 0.0 for v in vals_a),
            "cutA_all_sentinel_zero": all(v == 0.0 for v in vals_b),
        }

    for field in cutB_preserved_fields:
        mismatches = [
            {"frame_idx": fi, "a": getattr(by_a[fi], field), "b": getattr(by_b[fi], field)}
            for fi in by_a
            if getattr(by_a[fi], field) != getattr(by_b[fi], field)
        ]
        report["cutB_preserved_fields_bit_identical"][field] = {
            "bit_identical": len(mismatches) == 0,
            "mismatch_count": len(mismatches),
            "mismatches_sample": mismatches[:20],
        }

    for field in graph_read_fields:
        mismatches = [
            {"frame_idx": fi, "a": getattr(by_a[fi], field), "b": getattr(by_b[fi], field)}
            for fi in by_a
            if getattr(by_a[fi], field) != getattr(by_b[fi], field)
        ]
        report["graph_read_fields_bit_identical"][field] = {
            "bit_identical": len(mismatches) == 0,
            "mismatch_count": len(mismatches),
        }

    return report


def timed_ingest(video_path, config):
    """Run iris_ingest.ingest(), timing ONLY the charon_v.parse_video call
    (extraction) separately from total wall time (extraction + CLIP-encode +
    graph build), via a thin wrapper around charon_v.parse_video."""
    real_parse_video = charon_v.parse_video
    timing = {"extraction_sec": None}

    def _wrapped(*args, **kwargs):
        t0 = time.time()
        result = real_parse_video(*args, **kwargs)
        timing["extraction_sec"] = time.time() - t0
        return result

    iris_ingest.charon_v.parse_video = _wrapped
    try:
        t0 = time.time()
        idx = iris_ingest.ingest(video_path, config)
        total_sec = time.time() - t0
    finally:
        iris_ingest.charon_v.parse_video = real_parse_video
    return idx, timing["extraction_sec"], total_sec


def main() -> None:
    cache_file = Path(str(CACHE_STEM) + ".npz")
    if not cache_file.exists():
        print(f"FATAL: cache not found at {cache_file}", file=sys.stderr)
        sys.exit(1)

    cached_idx = iris_ingest.load_index(CACHE_STEM)
    video_path = cached_idx.video_path
    cfg_snapshot = cached_idx.config_snapshot
    print(f"video_path={video_path}")
    print(f"cached frame count={len(cached_idx.frames)}")

    cfg_full = _to_irisconfig(cfg_snapshot, compute_full_geometry=True)
    cfg_cut = _to_irisconfig(cfg_snapshot, compute_full_geometry=False)

    real_get_clip = iris_ingest.get_clip_embedding_from_pil
    iris_ingest.get_clip_embedding_from_pil = _fake_clip_embedding
    try:
        print("=== Ingesting WITH full geometry (compute_full_geometry=True, today's behavior) ===")
        idx_full, extraction_full_sec, total_full_sec = timed_ingest(video_path, cfg_full)
        print(f"  extraction={extraction_full_sec:.3f}s total={total_full_sec:.3f}s frames={len(idx_full.frames)}")

        print("=== Ingesting WITH Cut A (compute_full_geometry=False, new default) ===")
        idx_cut, extraction_cut_sec, total_cut_sec = timed_ingest(video_path, cfg_cut)
        print(f"  extraction={extraction_cut_sec:.3f}s total={total_cut_sec:.3f}s frames={len(idx_cut.frames)}")
    finally:
        iris_ingest.get_clip_embedding_from_pil = real_get_clip

    # Gate 4: graph identity
    edges_full = idx_full._graph.export_graph_data()["edges"]
    edges_cut = idx_cut._graph.export_graph_data()["edges"]
    edge_report = compare_edges(edges_full, edges_cut)

    # Gate 4 (secondary): PPR ranking identity for a fixed query embedding
    rng = np.random.default_rng(20260728)
    v = rng.standard_normal(512).astype(np.float32)
    fixed_query_emb = v / np.linalg.norm(v)
    ppr_full = idx_full._graph.retrieve_ppr(fixed_query_emb, top_k=10)
    ppr_cut = idx_cut._graph.retrieve_ppr(fixed_query_emb, top_k=10)
    ppr_order_full = [n.frame_idx for n in ppr_full]
    ppr_order_cut = [n.frame_idx for n in ppr_cut]
    ppr_scores_full = [float(n.last_retrieval_score) for n in ppr_full]
    ppr_scores_cut = [float(n.last_retrieval_score) for n in ppr_cut]
    ppr_report = {
        "order_identical": ppr_order_full == ppr_order_cut,
        "scores_bit_identical": ppr_scores_full == ppr_scores_cut,
        "order_full": ppr_order_full,
        "order_cut": ppr_order_cut,
    }

    # Gate 6 + 7: FrameRecord field comparison
    field_report = compare_frame_fields(idx_full.frames, idx_cut.frames)

    # Gate 7: schema presence check (no KeyError -- attribute access itself proves this,
    # but assert explicitly that the 3 gated fields are still real dataclass fields)
    frame_field_names = {f.name for f in dataclasses.fields(type(idx_cut.frames[0]))}
    schema_ok = {"divergence", "curl", "jacobian_frobenius", "hessian_max_eigenvalue", "motion_entropy"} <= frame_field_names

    cutA_ok = all(
        v["full_geometry_any_nonzero"] and v["cutA_all_sentinel_zero"]
        for v in field_report["cutA_gated_fields_sentinel_zero_under_cut"].values()
    )
    cutB_ok = all(v["bit_identical"] for v in field_report["cutB_preserved_fields_bit_identical"].values())
    graph_fields_ok = all(v["bit_identical"] for v in field_report["graph_read_fields_bit_identical"].values())

    gate4_pass = edge_report["identical"] and ppr_report["order_identical"] and ppr_report["scores_bit_identical"] and graph_fields_ok
    gate6_pass = cutB_ok
    gate7_pass = schema_ok
    overall_pass = gate4_pass and gate6_pass and gate7_pass

    speedup_report = {
        "extraction_full_geometry_sec": extraction_full_sec,
        "extraction_cutA_sec": extraction_cut_sec,
        "delta_sec": extraction_full_sec - extraction_cut_sec,
        "delta_pct_of_full": (extraction_full_sec - extraction_cut_sec) / extraction_full_sec * 100.0 if extraction_full_sec else None,
        "total_full_sec": total_full_sec,
        "total_cutA_sec": total_cut_sec,
    }

    result = {
        "outcome": "GATE_PASS" if overall_pass else "GATE_FAIL",
        "gate4_graph_identity_pass": gate4_pass,
        "gate6_l1_aria_safety_pass": gate6_pass,
        "gate7_schema_pass": gate7_pass,
        "edge_report": edge_report,
        "ppr_report": ppr_report,
        "field_report": field_report,
        "speedup_report": speedup_report,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, default=str))

    md = []
    md.append("# Cut A identity gate: VIRAT N=4,892\n")
    md.append(f"**Outcome: {result['outcome']}**\n")
    md.append(f"- Gate 4 (graph identity, bit-identical edges + PPR): {'PASS' if gate4_pass else 'FAIL'}")
    md.append(f"- Gate 6 (L1/ARIA safety, hessian/entropy bit-identical): {'PASS' if gate6_pass else 'FAIL'}")
    md.append(f"- Gate 7 (schema, gated fields still present): {'PASS' if gate7_pass else 'FAIL'}\n")
    md.append("## Edge comparison\n")
    md.append(f"- edges (full geometry): {edge_report['edge_count_a']}")
    md.append(f"- edges (Cut A): {edge_report['edge_count_b']}")
    md.append(f"- only in full: {edge_report['only_in_a']}, only in Cut A: {edge_report['only_in_b']}")
    md.append(f"- field mismatches (weight/semantic/motion/temporal/edge_type), tolerance 0.0: {edge_report['field_mismatch_count']}\n")
    md.append("## PPR ranking (fixed query embedding, top_k=10)\n")
    md.append(f"- order identical: {ppr_report['order_identical']}")
    md.append(f"- scores bit-identical: {ppr_report['scores_bit_identical']}\n")
    md.append("## Cut A gated fields (divergence/curl/jacobian_frobenius)\n")
    for f, v in field_report["cutA_gated_fields_sentinel_zero_under_cut"].items():
        md.append(f"- {f}: full-geometry run has nonzero values = {v['full_geometry_any_nonzero']}; "
                   f"Cut A run all sentinel 0.0 = {v['cutA_all_sentinel_zero']}")
    md.append("\n## Cut B preserved fields (hessian_max_eigenvalue, motion_entropy)\n")
    for f, v in field_report["cutB_preserved_fields_bit_identical"].items():
        md.append(f"- {f}: bit-identical = {v['bit_identical']} (mismatches: {v['mismatch_count']})")
    md.append("\n## Graph-read fields (action_score, persistence_value, ... pagerank_score)\n")
    for f, v in field_report["graph_read_fields_bit_identical"].items():
        md.append(f"- {f}: bit-identical = {v['bit_identical']} (mismatches: {v['mismatch_count']})")
    md.append("\n## Extraction wall-time (charon_v.parse_video only)\n")
    md.append(f"- full geometry: {speedup_report['extraction_full_geometry_sec']:.3f}s")
    md.append(f"- Cut A: {speedup_report['extraction_cutA_sec']:.3f}s")
    md.append(f"- delta: {speedup_report['delta_sec']:.3f}s ({speedup_report['delta_pct_of_full']:.1f}% of full-geometry extraction time)")
    md.append(f"- total ingest (extraction + CLIP-encode + graph build), full geometry: {speedup_report['total_full_sec']:.3f}s")
    md.append(f"- total ingest (extraction + CLIP-encode + graph build), Cut A: {speedup_report['total_cutA_sec']:.3f}s\n")

    OUT_MD.write_text("\n".join(md) + "\n")

    print(json.dumps(result, indent=2, default=str)[:6000])
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== GATE RESULT: {result['outcome']} ===")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
