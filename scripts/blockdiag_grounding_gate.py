"""Grounding acceptance gate for block_diagonal: end-to-end downstream check.

Context (see eval_results/blockdiag_build_plan.md and
eval_results/blockdiag_identity_gate_result.md): block_diagonal already
passed a graph-construction-level identity gate (bit-identical edges,
PageRank, PPR ranking order vs fully_connected, on VIRAT N=4,892). This
script checks the one level up: does the SAME bit-identical graph produce
bit-identical GROUNDING output once it goes through the real retrieval
pipeline (scene_retrieval's shortcut/descend routing, induced subgraph
copies, add_cross_scene_edges, span construction, IoP/peak_in_gold metrics)?

IMPORTANT SCOPING NOTE (see report emitted at the end): the committed
peak_in_gold==0.3227 VAL cell (eval_results/P_NOWA_grounding_result.md, via
scripts/pnowa_width_topk_sweep.py) uses graph_mode="flat", which never
builds node_groups and therefore can never invoke graph_edge_mode=
"block_diagonal" (that mode raises ValueError without node_groups). Flat's
code path is untouched by this change, so 0.3227 is structurally unaffected
-- not merely "probably fine": the new branch is unreachable from a flat
config. This script:
  (a) reconfirms 0.3227 by rerunning the exact frozen VAL cell (top_k=8,
      half_width=2.2, graph_mode=flat) unchanged, and
  (b) separately proves block_diagonal == fully_connected on a scene_sparse
      grounding cell -- the pairing block_diagonal actually targets. No
      committed grounding number was ever produced with
      graph_mode=scene_sparse + graph_edge_mode=fully_connected (the
      existing scene_sparse committed arms, e.g. scripts/eval_grounding_arms.py,
      use graph_edge_mode's default "hierarchical_sparse", a different
      formula block_diagonal does not touch). So (b) is an equivalence
      proof between the two build paths on a purpose-built scene_sparse+
      fully_connected cell, using the SAME production query pipeline
      (iris.query._build_retrieved, eval.span.predict_span) and the SAME
      question set / config shape as the existing scene_sparse committed
      arms (TAU=0.015, rep_only, top_k=8, half_width=2.2).

Read-only: loads cached frames from eval/data/nextqa/index_cache_ssparse and
eval/data/nextqa/index_cache; NEVER writes to either. sha256 of every file in
both cache dirs is hashed before and after; the run asserts byte-for-byte
unchanged (no re-ingest).
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.scene_retrieval as scene_retrieval
from iris.iris_config import IRISConfig

_IRISCONFIG_FIELDS = {f.name for f in dataclasses.fields(IRISConfig)}


def _to_irisconfig(cfg_dict: dict) -> IRISConfig:
    """Cached config_snapshots can carry stale keys from older IRISConfig
    schemas (e.g. removed fields) -- filter to the current dataclass's
    fields before constructing. _build_graph()/_get() already tolerate
    dict configs with extra keys directly, so this filtering only matters
    for the IRISConfig(**...) calls used by the query-time pipeline."""
    return IRISConfig(**{k: v for k, v in cfg_dict.items() if k in _IRISCONFIG_FIELDS})
from iris.query import _embed_query, _build_retrieved
from eval.grounding_scorer import iop
from eval.span import predict_span

DATA_DIR = REPO / "eval" / "data" / "nextqa"
FLAT_CACHE = DATA_DIR / "index_cache"
SSPARSE_CACHE = DATA_DIR / "index_cache_ssparse"
GQA_JSON = DATA_DIR / "gsub_val.json"
VAL_CSV = DATA_DIR / "val.csv"
VAL_VIDEO_LIST = REPO / "eval_results" / "val_videos.txt"
TEST_VIDEO_LIST = REPO / "eval_results" / "test_videos.txt"

OUT_JSON = REPO / "eval_results" / "blockdiag_grounding_gate_result.json"
OUT_MD = REPO / "eval_results" / "blockdiag_grounding_gate_result.md"

TOP_K = 8
HALF_WIDTH = 2.2  # FROZEN_HALF_WIDTH_SECONDS, matches the committed VAL cell
SPAN_MODE = "ppr_peak"
PEAK_SOURCE = "clip_in_ppr_top8"
TAU = 0.015  # matches scripts/eval_grounding_arms.py's restated tau

FROZEN_PEAK_IN_GOLD_VAL_0_3227 = 0.3227


def dir_sha256(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*.npz")):
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


# ── Part (a): reconfirm the frozen flat VAL cell is untouched ────────────────

def rerun_frozen_flat_cell() -> dict:
    val_videos = {l.strip() for l in open(VAL_VIDEO_LIST, encoding="utf-8") if l.strip()}
    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    assert not (val_videos & test_videos), "val/test video overlap -- split is corrupt"

    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(open(VAL_CSV, encoding="utf-8")))
    grounded_rows = [
        r for r in rows
        if r["video"] in val_videos
        and r["video"] in gsub
        and r["qid"] in gsub[r["video"]]["location"]
    ]
    used_videos = {r["video"] for r in grounded_rows}
    assert not (used_videos & test_videos), "TEST videos leaked into VAL rerun"

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in used_videos}

    cfg = IRISConfig(
        graph_mode="flat",
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=0.5,
        ppr_damping=0.5,
        l2_retrieve_top_k=TOP_K,
    )

    loaded: dict = {}
    for row in grounded_rows:
        vid = row["video"]
        if vid in loaded:
            continue
        npz = FLAT_CACHE / f"{vid}.npz"
        if npz.exists():
            loaded[vid] = iris_ingest.load_index(FLAT_CACHE / vid)

    pig = []
    n = 0
    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        index = loaded.get(vid)
        if index is None:
            continue
        emb = _embed_query(row["question"], cfg)
        retrieved = _build_retrieved(index, emb, cfg)
        gold_spans = gsub[vid]["location"][qid]
        span, t_peak = predict_span(
            retrieved, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb, return_peak=True,
        )
        pig_val = 1 if (t_peak is not None and any(float(s) <= t_peak <= float(e) for s, e in gold_spans)) else 0
        pig.append(pig_val)
        n += 1

    rate = statistics.mean(pig) if pig else None
    return {
        "n": n,
        "n_videos": len(used_videos),
        "peak_in_gold_rate": rate,
        "matches_frozen_0_3227_exactly": (rate is not None and round(rate, 4) == FROZEN_PEAK_IN_GOLD_VAL_0_3227),
    }


# ── Part (b): fully_connected vs block_diagonal on scene_sparse, real pipeline ──

def _load_grounded_ssparse_rows() -> tuple[list[dict], dict]:
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(open(VAL_CSV, encoding="utf-8")))
    ssparse_cached = {p.stem for p in SSPARSE_CACHE.glob("*.npz")}
    grounded = [
        r for r in rows
        if r["video"] in ssparse_cached
        and r["video"] in gsub
        and r["qid"] in gsub[r["video"]]["location"]
    ]
    return grounded, gsub


def run_scene_sparse_equivalence() -> dict:
    grounded_rows, gsub = _load_grounded_ssparse_rows()
    videos = sorted({r["video"] for r in grounded_rows})
    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in videos}

    print(f"[DATA] scene_sparse-cached grounded VAL questions: {len(grounded_rows)} "
          f"across {len(videos)} videos")

    per_video_graphs: dict[str, tuple] = {}
    for vid in videos:
        idx = iris_ingest.load_index(SSPARSE_CACHE / vid)
        base_cfg = dict(idx.config_snapshot)
        base_cfg.update(dict(
            graph_mode="scene_sparse",
            ranking_mode="ppr",
            codec_conf_source="packet_size",
            codec_conf_pictype_norm=True,
            ppr_lambda=0.5,
            ppr_damping=0.5,
            l2_retrieve_top_k=TOP_K,
            scene_shortcut_margin=TAU,
            scene_crossscene_mode="rep_only",
        ))

        cfg_a_dict = dict(base_cfg, graph_edge_mode="fully_connected")
        cfg_b_dict = dict(base_cfg, graph_edge_mode="block_diagonal")

        graph_a = iris_ingest._build_graph(idx.frames, cfg_a_dict)
        graph_b = iris_ingest._build_graph(idx.frames, cfg_b_dict)

        idx_a = dataclasses.replace(idx, config_snapshot=cfg_a_dict, _graph=graph_a)
        idx_b = dataclasses.replace(idx, config_snapshot=cfg_b_dict, _graph=graph_b)

        cfg_a = _to_irisconfig(cfg_a_dict)
        cfg_b = _to_irisconfig(cfg_b_dict)

        per_video_graphs[vid] = (idx_a, cfg_a, idx_b, cfg_b)

    mismatches = []
    pig_a, pig_b = [], []
    iop_a, iop_b = [], []
    n = 0

    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        if vid not in per_video_graphs:
            continue
        idx_a, cfg_a, idx_b, cfg_b = per_video_graphs[vid]
        gold_spans = gsub[vid]["location"][qid]

        scene_retrieval.SCENE_DIAG_RECORDS.clear()
        emb_a = _embed_query(row["question"], cfg_a)
        retrieved_a = _build_retrieved(idx_a, emb_a, cfg_a)
        span_a, peak_a = predict_span(
            retrieved_a, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb_a, return_peak=True,
        )

        scene_retrieval.SCENE_DIAG_RECORDS.clear()
        emb_b = _embed_query(row["question"], cfg_b)
        retrieved_b = _build_retrieved(idx_b, emb_b, cfg_b)
        span_b, peak_b = predict_span(
            retrieved_b, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb_b, return_peak=True,
        )

        iop_val_a = iop(span_a, gold_spans)
        iop_val_b = iop(span_b, gold_spans)
        pig_val_a = 1 if (peak_a is not None and any(float(s) <= peak_a <= float(e) for s, e in gold_spans)) else 0
        pig_val_b = 1 if (peak_b is not None and any(float(s) <= peak_b <= float(e) for s, e in gold_spans)) else 0

        # Bit-identical check on the actual DOWNSTREAM outputs: retrieved
        # frame-idx order, peak timestamp, span, and both scored metrics.
        retrieved_ids_a = [r.get("frame_idx", r.get("timestamp")) for r in retrieved_a]
        retrieved_ids_b = [r.get("frame_idx", r.get("timestamp")) for r in retrieved_b]

        row_match = (
            retrieved_ids_a == retrieved_ids_b
            and peak_a == peak_b
            and span_a == span_b
            and iop_val_a == iop_val_b
            and pig_val_a == pig_val_b
        )
        if not row_match:
            mismatches.append({
                "video": vid, "qid": qid,
                "retrieved_ids_a": retrieved_ids_a, "retrieved_ids_b": retrieved_ids_b,
                "peak_a": peak_a, "peak_b": peak_b,
                "span_a": span_a, "span_b": span_b,
                "iop_a": iop_val_a, "iop_b": iop_val_b,
                "pig_a": pig_val_a, "pig_b": pig_val_b,
            })

        pig_a.append(pig_val_a)
        pig_b.append(pig_val_b)
        iop_a.append(iop_val_a)
        iop_b.append(iop_val_b)
        n += 1

    return {
        "n_questions": n,
        "n_videos": len(videos),
        "peak_in_gold_rate_fully_connected": statistics.mean(pig_a) if pig_a else None,
        "peak_in_gold_rate_block_diagonal": statistics.mean(pig_b) if pig_b else None,
        "mIoP_fully_connected": statistics.mean(iop_a) if iop_a else None,
        "mIoP_block_diagonal": statistics.mean(iop_b) if iop_b else None,
        "n_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:20],
        "bit_identical": (len(mismatches) == 0),
    }


def main() -> None:
    print("Hashing cache dirs (pre-run, read-only)...")
    flat_hash_before = dir_sha256(FLAT_CACHE)
    ssparse_hash_before = dir_sha256(SSPARSE_CACHE)

    print("\n=== Part (a): reconfirm frozen flat VAL cell (top_k=8, half_width=2.2) ===")
    flat_result = rerun_frozen_flat_cell()
    print(json.dumps(flat_result, indent=2))

    print("\n=== Part (b): fully_connected vs block_diagonal, scene_sparse, real pipeline ===")
    ssparse_result = run_scene_sparse_equivalence()
    print(json.dumps({k: v for k, v in ssparse_result.items() if k != "mismatches_sample"}, indent=2))

    print("\nHashing cache dirs (post-run, asserting unchanged)...")
    flat_hash_after = dir_sha256(FLAT_CACHE)
    ssparse_hash_after = dir_sha256(SSPARSE_CACHE)

    flat_cache_untouched = (flat_hash_before == flat_hash_after)
    ssparse_cache_untouched = (ssparse_hash_before == ssparse_hash_after)

    gate_pass = (
        flat_result["matches_frozen_0_3227_exactly"]
        and ssparse_result["bit_identical"]
        and flat_cache_untouched
        and ssparse_cache_untouched
    )

    result = {
        "outcome": "GATE_PASS" if gate_pass else "GATE_FAIL",
        "part_a_frozen_flat_cell": flat_result,
        "part_b_scene_sparse_equivalence": ssparse_result,
        "flat_cache_untouched": flat_cache_untouched,
        "ssparse_cache_untouched": ssparse_cache_untouched,
        "flat_cache_sha256_before": flat_hash_before,
        "flat_cache_sha256_after": flat_hash_after,
        "ssparse_cache_sha256_before": ssparse_hash_before,
        "ssparse_cache_sha256_after": ssparse_hash_after,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Block-diagonal grounding acceptance gate",
        "",
        f"**Outcome: {result['outcome']}**",
        "",
        "## Part (a) -- frozen flat VAL cell (unaffected by this change, reconfirmed)",
        "",
        f"- graph_mode=flat, top_k=8, half_width=2.2, n={flat_result['n']} questions / "
        f"{flat_result['n_videos']} videos",
        f"- Rerun peak_in_gold rate: {flat_result['peak_in_gold_rate']}",
        f"- Matches committed 0.3227 exactly (rounded to 4dp): "
        f"{flat_result['matches_frozen_0_3227_exactly']}",
        "- Note: flat's code path never sets node_groups, so graph_edge_mode="
        "\"block_diagonal\" is UNREACHABLE from this config (the new branch raises "
        "ValueError without node_groups). This cell cannot exercise block_diagonal at "
        "all -- it is reconfirmed here only to prove the fix left it untouched.",
        "",
        "## Part (b) -- fully_connected vs block_diagonal, scene_sparse, real pipeline",
        "",
        f"- N questions: {ssparse_result['n_questions']}, videos: {ssparse_result['n_videos']}",
        f"- peak_in_gold rate -- fully_connected: {ssparse_result['peak_in_gold_rate_fully_connected']}, "
        f"block_diagonal: {ssparse_result['peak_in_gold_rate_block_diagonal']}",
        f"- mIoP -- fully_connected: {ssparse_result['mIoP_fully_connected']}, "
        f"block_diagonal: {ssparse_result['mIoP_block_diagonal']}",
        f"- Per-question bit-identical (retrieved order, peak, span, IoP, peak_in_gold): "
        f"{ssparse_result['bit_identical']} ({ssparse_result['n_mismatches']} mismatches)",
        "- No committed grounding number was ever produced with graph_mode=scene_sparse + "
        "graph_edge_mode=fully_connected (existing scene_sparse committed arms use the "
        "default hierarchical_sparse edge formula, which block_diagonal does not target). "
        "This is therefore an equivalence proof between the two build paths this fix "
        "actually concerns, not a rerun of a pre-existing committed number.",
        "",
        f"## Cache integrity",
        "",
        f"- `eval/data/nextqa/index_cache` sha256 unchanged: {flat_cache_untouched}",
        f"- `eval/data/nextqa/index_cache_ssparse` sha256 unchanged: {ssparse_cache_untouched}",
        "",
    ]
    if ssparse_result["mismatches_sample"]:
        md.append("## Mismatch sample")
        md.append("")
        for m in ssparse_result["mismatches_sample"]:
            md.append(f"- {m['video']}/{m['qid']}: peak_a={m['peak_a']} peak_b={m['peak_b']} "
                       f"span_a={m['span_a']} span_b={m['span_b']}")

    OUT_MD.write_text("\n".join(md) + "\n")

    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== GATE RESULT: {result['outcome']} ===")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
