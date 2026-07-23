"""P-NOW-A step 3: top_k x half_width sweep on VAL only.

Reuses eval_grounding_arms.py's internals verbatim (index loading, query
embedding, retrieval, span construction, metrics) -- this script only adds
the grid loop and the retrieve-once-per-top_k efficiency (half_width does
not affect retrieval, so all 5 half_widths are scored from ONE retrieved set
per top_k, not 20 separate retrieval passes).

Question set: VAL half only (eval_results/val_videos.txt, 59 videos / 406
questions). TEST videos (eval_results/test_videos.txt) are asserted absent.
Arm: flat graph_mode only.

VERIFY: python scripts/pnowa_width_topk_sweep.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.grounding_scorer import frames_in_window, iop, load_indexes, uniform_ts
from eval.span import predict_span, _pick_by_clip_similarity
from scripts.pillar2_grounded_qa import iou
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved

DATA_DIR = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
GQA_JSON = DATA_DIR / "gsub_val.json"
VAL_CSV = DATA_DIR / "val.csv"
VAL_VIDEO_LIST = REPO / "eval_results" / "val_videos.txt"
TEST_VIDEO_LIST = REPO / "eval_results" / "test_videos.txt"

TOP_KS = [8, 12, 16, 24]
HALF_WIDTHS = [0.75, 1.0, 1.5, 2.2, 3.0]
SPAN_MODE = "ppr_peak"
PEAK_SOURCE = "clip_in_ppr_top8"

BASE = dict(
    graph_mode="flat",
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
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
    cell_agg: dict[tuple[int, float], dict] = {}

    n_clip_fallback = 0
    n_clip_total = 0

    for top_k in TOP_KS:
        cfg = IRISConfig(**BASE, l2_retrieve_top_k=top_k)
        # Fresh indexes per top_k -- no mutated node state carried over
        # (mirrors eval_grounding_arms.py's per-arm fresh-load discipline).
        loaded = load_indexes(grounded_rows, CACHE_DIR)

        # ── ONE retrieval pass per top_k; cache (retrieved, emb) per question ──
        retrieved_cache: dict[tuple[str, str], tuple[list[dict], "object"]] = {}
        for row in grounded_rows:
            vid, qid = row["video"], str(row["qid"])
            index = loaded.get(vid)
            if index is None:
                continue
            emb = _embed_query(row["question"], cfg)
            retrieved = _build_retrieved(index, emb, cfg)
            retrieved_cache[(vid, qid)] = (retrieved, emb)

            # CLIP-anchor fallback instrumentation (same inputs predict_span
            # would use for peak_source="clip_in_ppr_top8").
            n_clip_total += 1
            if _pick_by_clip_similarity(retrieved, emb) is None:
                n_clip_fallback += 1

        for half_width in HALF_WIDTHS:
            miop, miou, iop05, iou05, pig = [], [], [], [], []
            n = 0
            for row in grounded_rows:
                vid, qid = row["video"], str(row["qid"])
                key = (vid, qid)
                if key not in retrieved_cache:
                    continue
                retrieved, emb = retrieved_cache[key]
                gold_spans = gsub[vid]["location"][qid]
                span, t_peak = predict_span(
                    retrieved, mode=SPAN_MODE, half_width=half_width,
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
                    "video": vid, "qid": qid, "top_k": top_k, "half_width": half_width,
                    "iop": iop_val, "iou": iou_val, "peak_in_gold": pig_val,
                })

            cell_agg[(top_k, half_width)] = {
                "peak_in_gold_rate": statistics.mean(pig) if pig else None,
                "mIoP": statistics.mean(miop) if miop else None,
                "IoP@0.5": statistics.mean(iop05) if iop05 else None,
                "mIoU": statistics.mean(miou) if miou else None,
                "IoU@0.5": statistics.mean(iou05) if iou05 else None,
                "n": n,
            }
            print(f"[CELL] top_k={top_k:>3} half_width={half_width:>4} "
                  f"peak_in_gold={cell_agg[(top_k, half_width)]['peak_in_gold_rate']:.4f} "
                  f"mIoP={cell_agg[(top_k, half_width)]['mIoP']:.4f} "
                  f"IoP@0.5={cell_agg[(top_k, half_width)]['IoP@0.5']:.4f} "
                  f"mIoU={cell_agg[(top_k, half_width)]['mIoU']:.4f} "
                  f"IoU@0.5={cell_agg[(top_k, half_width)]['IoU@0.5']:.4f} "
                  f"n={n}", flush=True)

    fallback_rate = n_clip_fallback / n_clip_total if n_clip_total else 0.0
    print(f"\n[CLIP-ANCHOR] fallback_rate={fallback_rate:.4%} ({n_clip_fallback}/{n_clip_total})")
    if n_clip_fallback != 0:
        print("FATAL: CLIP-anchor fallback rate is nonzero -- peak_source=clip_in_ppr_top8 "
              "is silently degrading to ppr_score for some questions.", file=sys.stderr)
        sys.exit(1)

    # ── full grid report ────────────────────────────────────────────────────
    print("\n=== FULL 4x5 GRID (top_k x half_width) ===")
    hdr = f"{'top_k':>5} {'half_width':>10} {'peak_in_gold':>12} {'mIoP':>8} {'IoP@0.5':>8} {'mIoU':>8} {'IoU@0.5':>8} {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for top_k in TOP_KS:
        for half_width in HALF_WIDTHS:
            c = cell_agg[(top_k, half_width)]
            print(f"{top_k:>5} {half_width:>10} {c['peak_in_gold_rate']:>12.4f} {c['mIoP']:>8.4f} "
                  f"{c['IoP@0.5']:>8.4f} {c['mIoU']:>8.4f} {c['IoU@0.5']:>8.4f} {c['n']:>4}")

    # ── per-question dump ────────────────────────────────────────────────────
    logs_dir = REPO / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / "pnowa_sweep_perq.json", "w", encoding="utf-8") as fh:
        json.dump({
            "grid": [
                {"top_k": tk, "half_width": hw, **agg}
                for (tk, hw), agg in cell_agg.items()
            ],
            "per_question": results,
            "clip_anchor_fallback_rate": fallback_rate,
        }, fh, indent=2)
    print(f"\n[LOG] per-question dump written to {logs_dir / 'pnowa_sweep_perq.json'}")


if __name__ == "__main__":
    main()
