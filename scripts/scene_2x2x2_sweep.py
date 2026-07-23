"""Scene-sparse 2x2x2 mechanism experiment (P_scene_2x2x2_prereg.md). VAL only.

Reuses existing internals verbatim -- index loading, query embedding,
retrieval, span construction, metrics -- no reimplementation. This script
only adds the 2x2x2 grid loop, branch (shortcut/descend) instrumentation,
and the CLIP-anchor fallback assertion for peak_anchored cells.

Grid (top_k=8 fixed):
  graph_mode            : flat | scene_sparse
  motion_similarity_mode : action_score | geometry_6d
  span                   : minmax | peak_anchored (half_width=2.2, clip_in_ppr_top8)

Question set: VAL half only (eval_results/val_videos.txt, 59 videos / 406
questions). TEST videos (eval_results/test_videos.txt) are asserted absent.

VERIFY: python scripts/scene_2x2x2_sweep.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.scene_retrieval as scene_retrieval
from eval.grounding_scorer import iop, load_indexes
from eval.span import predict_span, _pick_by_clip_similarity
from scripts.pillar2_grounded_qa import iou
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved

DATA_DIR = REPO / "eval" / "data" / "nextqa"
FLAT_CACHE = DATA_DIR / "index_cache"
SSPARSE_CACHE = DATA_DIR / "index_cache_ssparse"
GQA_JSON = DATA_DIR / "gsub_val.json"
VAL_CSV = DATA_DIR / "val.csv"
VAL_VIDEO_LIST = REPO / "eval_results" / "val_videos.txt"
TEST_VIDEO_LIST = REPO / "eval_results" / "test_videos.txt"

TOP_K = 8
HALF_WIDTH = 2.2
PEAK_SOURCE = "clip_in_ppr_top8"

GRAPH_MODES = ["flat", "scene_sparse"]
MOTION_MODES = ["action_score", "geometry_6d"]
SPAN_MODES = ["minmax", "peak_anchored"]

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=TOP_K,
    scene_diag=True,
)


def _load_val_rows() -> list[dict]:
    val_videos = {l.strip() for l in open(VAL_VIDEO_LIST, encoding="utf-8") if l.strip()}
    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    assert not (val_videos & test_videos), "val/test video overlap -- split is corrupt"

    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(open(VAL_CSV, encoding="utf-8")))
    for r in rows:
        r["family"] = r["type"][0]

    grounded = [
        r for r in rows
        if r["video"] in val_videos
        and r["video"] in gsub
        and r["qid"] in gsub[r["video"]]["location"]
    ]
    used_videos = {r["video"] for r in grounded}
    leaked_test_videos = used_videos & test_videos
    assert not leaked_test_videos, f"TEST videos leaked into VAL sweep: {sorted(leaked_test_videos)}"
    return grounded


def main() -> None:
    grounded_rows = _load_val_rows()
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    n_videos = len(set(r["video"] for r in grounded_rows))
    print(f"[DATA] VAL grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in {r["video"] for r in grounded_rows}}

    results = []  # per-question flat dump rows
    cell_agg: dict[tuple[str, str, str], dict] = {}

    n_clip_fallback = 0
    n_clip_total = 0

    # graph_mode x motion_similarity_mode -> ONE retrieval+branch pass; both
    # span modes are then computed from that same retrieved set (span does
    # not affect retrieval, mirroring the width x top_k sweep's efficiency).
    for graph_mode in GRAPH_MODES:
        cache_dir = FLAT_CACHE if graph_mode == "flat" else SSPARSE_CACHE
        for motion_mode in MOTION_MODES:
            cfg = IRISConfig(**BASE, graph_mode=graph_mode, motion_similarity_mode=motion_mode)
            loaded = load_indexes(grounded_rows, cache_dir)

            scene_retrieval.SCENE_DIAG_RECORDS.clear()
            retrieved_cache: dict[tuple[str, str], tuple[list[dict], "object"]] = {}
            branch_by_question: dict[tuple[str, str], str] = {}

            for row in grounded_rows:
                vid, qid = row["video"], str(row["qid"])
                index = loaded.get(vid)
                if index is None:
                    continue
                emb = _embed_query(row["question"], cfg)
                retrieved = _build_retrieved(index, emb, cfg)
                retrieved_cache[(vid, qid)] = (retrieved, emb)

                if graph_mode == "scene_sparse" and scene_retrieval.SCENE_DIAG_RECORDS:
                    branch_by_question[(vid, qid)] = scene_retrieval.SCENE_DIAG_RECORDS[-1]["branch"]

                n_clip_total += 1
                if _pick_by_clip_similarity(retrieved, emb) is None:
                    n_clip_fallback += 1

            n_shortcut = sum(1 for b in branch_by_question.values() if b == "shortcut")
            n_descend = sum(1 for b in branch_by_question.values() if b == "descend")
            n_branch_total = len(branch_by_question)

            for span_mode in SPAN_MODES:
                miop, miou, iop05, iou05, pig = [], [], [], [], []
                n = 0
                for row in grounded_rows:
                    vid, qid = row["video"], str(row["qid"])
                    key = (vid, qid)
                    if key not in retrieved_cache:
                        continue
                    retrieved, emb = retrieved_cache[key]
                    gold_spans = gsub[vid]["location"][qid]

                    if span_mode == "minmax":
                        span = predict_span(retrieved, mode="minmax")
                        t_peak = None
                    else:
                        span, t_peak = predict_span(
                            retrieved, mode="ppr_peak", half_width=HALF_WIDTH,
                            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
                            query_embedding=emb, return_peak=True,
                        )

                    iop_val = iop(span, gold_spans)
                    iou_val = iou(span, gold_spans)
                    pig_val = 1 if (t_peak is not None and any(float(s) <= t_peak <= float(e) for s, e in gold_spans)) else 0

                    miop.append(iop_val)
                    miou.append(iou_val)
                    iop05.append(1.0 if iop_val >= 0.5 else 0.0)
                    iou05.append(1.0 if iou_val >= 0.5 else 0.0)
                    pig.append(pig_val)
                    n += 1

                    results.append({
                        "video": vid, "qid": qid, "graph_mode": graph_mode,
                        "motion_similarity_mode": motion_mode, "span_mode": span_mode,
                        "iop": iop_val, "iou": iou_val, "peak_in_gold": pig_val,
                        "branch": branch_by_question.get(key, "n/a"),
                    })

                cell_key = (graph_mode, motion_mode, span_mode)
                cell_agg[cell_key] = {
                    "peak_in_gold_rate": statistics.mean(pig) if (pig and span_mode == "peak_anchored") else None,
                    "mIoP": statistics.mean(miop) if miop else None,
                    "IoP@0.5": statistics.mean(iop05) if iop05 else None,
                    "mIoU": statistics.mean(miou) if miou else None,
                    "IoU@0.5": statistics.mean(iou05) if iou05 else None,
                    "n": n,
                    "n_shortcut": n_shortcut if graph_mode == "scene_sparse" else None,
                    "n_descend": n_descend if graph_mode == "scene_sparse" else None,
                    "n_branch_total": n_branch_total if graph_mode == "scene_sparse" else None,
                }
                c = cell_agg[cell_key]
                pig_str = f"{c['peak_in_gold_rate']:.4f}" if c["peak_in_gold_rate"] is not None else "  N/A "
                print(f"[CELL] graph_mode={graph_mode:<12} motion={motion_mode:<13} span={span_mode:<14} "
                      f"peak_in_gold={pig_str} mIoP={c['mIoP']:.4f} IoP@0.5={c['IoP@0.5']:.4f} "
                      f"mIoU={c['mIoU']:.4f} IoU@0.5={c['IoU@0.5']:.4f} n={n}", flush=True)

            if graph_mode == "scene_sparse":
                pct_shortcut = n_shortcut / n_branch_total if n_branch_total else float("nan")
                pct_descend = n_descend / n_branch_total if n_branch_total else float("nan")
                print(f"[BRANCH] graph_mode=scene_sparse motion={motion_mode:<13} "
                      f"shortcut={n_shortcut}/{n_branch_total} ({pct_shortcut:.1%})  "
                      f"descend={n_descend}/{n_branch_total} ({pct_descend:.1%})", flush=True)

    fallback_rate = n_clip_fallback / n_clip_total if n_clip_total else 0.0
    print(f"\n[CLIP-ANCHOR] fallback_rate={fallback_rate:.4%} ({n_clip_fallback}/{n_clip_total})")
    if n_clip_fallback != 0:
        print("FATAL: CLIP-anchor fallback rate is nonzero in a peak_anchored-eligible pass.", file=sys.stderr)
        sys.exit(1)

    # ── anchor-cell reproduction check ──────────────────────────────────────
    anchor = cell_agg[("flat", "action_score", "peak_anchored")]
    print(f"\n[ANCHOR CHECK] flat/action_score/peak_anchored: mIoP={anchor['mIoP']:.4f} "
          f"(expect 0.3126)  IoP@0.5={anchor['IoP@0.5']:.4f} (expect 0.3202)")

    # ── full grid report ────────────────────────────────────────────────────
    print("\n=== FULL 2x2x2 GRID ===")
    hdr = (f"{'graph_mode':<12} {'motion':<13} {'span':<14} {'peak_in_gold':>12} "
           f"{'mIoP':>8} {'IoP@0.5':>8} {'mIoU':>8} {'IoU@0.5':>8} {'n':>4}")
    print(hdr)
    print("-" * len(hdr))
    for graph_mode in GRAPH_MODES:
        for motion_mode in MOTION_MODES:
            for span_mode in SPAN_MODES:
                c = cell_agg[(graph_mode, motion_mode, span_mode)]
                pig_str = f"{c['peak_in_gold_rate']:.4f}" if c["peak_in_gold_rate"] is not None else "  N/A "
                print(f"{graph_mode:<12} {motion_mode:<13} {span_mode:<14} {pig_str:>12} "
                      f"{c['mIoP']:>8.4f} {c['IoP@0.5']:>8.4f} {c['mIoU']:>8.4f} {c['IoU@0.5']:>8.4f} {c['n']:>4}")

    print("\n=== SHORTCUT/DESCEND SPLIT (scene_sparse arms) ===")
    for motion_mode in MOTION_MODES:
        c = cell_agg[("scene_sparse", motion_mode, "peak_anchored")]
        print(f"  motion={motion_mode:<13} shortcut={c['n_shortcut']}/{c['n_branch_total']} "
              f"descend={c['n_descend']}/{c['n_branch_total']}")

    # ── per-question dump ────────────────────────────────────────────────────
    logs_dir = REPO / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "scene_2x2x2_perq.json", "w", encoding="utf-8") as fh:
        json.dump({
            "grid": [
                {"graph_mode": gm, "motion_similarity_mode": mm, "span_mode": sm, **agg}
                for (gm, mm, sm), agg in cell_agg.items()
            ],
            "per_question": results,
            "clip_anchor_fallback_rate": fallback_rate,
        }, fh, indent=2)
    print(f"\n[LOG] per-question dump written to {logs_dir / 'scene_2x2x2_perq.json'}")


if __name__ == "__main__":
    main()
