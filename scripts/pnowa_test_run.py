"""P-NOW-A step 4 Part B: the ONE held-out TEST run.

Frozen config selected by the pre-registered selection rule in
eval_results/P_NOWA_sweep_prereg.md, applied to VAL results in
eval_results/P_NOWA_sweep_prereg.md / logs/pnowa_sweep_perq.json:
  top_k=12, half_width=2.2, graph_mode=flat, ranking_mode=ppr,
  ppr_lambda=0.5, ppr_damping=0.5, peak_source=clip_in_ppr_top8.

Runs on the 27 TEST videos ONLY (eval_results/test_videos.txt, 120 questions).
Asserts zero VAL videos leak in and zero CLIP-anchor fallback.

Single-cell mode -- do not add a grid here. This script is meant to run
exactly once against TEST. If TEST is ever run again, it is burned and must
be labelled as such per the pre-registration.

VERIFY: python scripts/pnowa_test_run.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.grounding_scorer import iop, load_indexes
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

# FROZEN — selected on VAL only, per the pre-registered selection rule.
TOP_K = 12
HALF_WIDTH = 2.2
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


def _load_test_rows() -> list[dict]:
    val_videos = {l.strip() for l in open(VAL_VIDEO_LIST, encoding="utf-8") if l.strip()}
    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    assert not (val_videos & test_videos), "val/test video overlap -- split is corrupt"

    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    rows = list(csv.DictReader(open(VAL_CSV, encoding="utf-8")))
    for r in rows:
        r["family"] = r["type"][0]

    grounded = [
        r for r in rows
        if r["video"] in test_videos
        and r["video"] in gsub
        and r["qid"] in gsub[r["video"]]["location"]
    ]
    used_videos = {r["video"] for r in grounded}
    leaked_val_videos = used_videos & val_videos
    assert not leaked_val_videos, f"VAL videos leaked into TEST run: {sorted(leaked_val_videos)}"
    return grounded


def video_clustered_bootstrap(values_by_video: dict[str, list[float]], n_boot: int = 1000, seed: int = 20260722):
    import random
    videos = sorted(values_by_video.keys())
    rng = random.Random(seed)

    def _mean_of_sample(sample_videos):
        vals = []
        for v in sample_videos:
            vals.extend(values_by_video[v])
        return statistics.mean(vals) if vals else float("nan")

    point = _mean_of_sample(videos)
    boots = []
    for _ in range(n_boot):
        sample = [videos[rng.randrange(len(videos))] for _ in range(len(videos))]
        boots.append(_mean_of_sample(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def main() -> None:
    grounded_rows = _load_test_rows()
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    n_videos = len(set(r["video"] for r in grounded_rows))
    print(f"[DATA] TEST grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in {r["video"] for r in grounded_rows}}

    cfg = IRISConfig(**BASE, l2_retrieve_top_k=TOP_K)
    loaded = load_indexes(grounded_rows, CACHE_DIR)

    per_video_iop: dict[str, list[float]] = {}
    per_video_iou: dict[str, list[float]] = {}
    per_video_iop05: dict[str, list[float]] = {}
    per_video_iou05: dict[str, list[float]] = {}
    per_video_pig: dict[str, list[float]] = {}
    per_question_rows = []

    n_clip_fallback = 0
    n_clip_total = 0

    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        index = loaded.get(vid)
        if index is None:
            continue
        emb = _embed_query(row["question"], cfg)
        retrieved = _build_retrieved(index, emb, cfg)
        gold_spans = gsub[vid]["location"][qid]

        n_clip_total += 1
        if _pick_by_clip_similarity(retrieved, emb) is None:
            n_clip_fallback += 1

        span, t_peak = predict_span(
            retrieved, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb, return_peak=True,
        )
        iop_val = iop(span, gold_spans)
        iou_val = iou(span, gold_spans)
        pig_val = 1.0 if (t_peak is not None and any(float(s) <= t_peak <= float(e) for s, e in gold_spans)) else 0.0

        per_video_iop.setdefault(vid, []).append(iop_val)
        per_video_iou.setdefault(vid, []).append(iou_val)
        per_video_iop05.setdefault(vid, []).append(1.0 if iop_val >= 0.5 else 0.0)
        per_video_iou05.setdefault(vid, []).append(1.0 if iou_val >= 0.5 else 0.0)
        per_video_pig.setdefault(vid, []).append(pig_val)

        per_question_rows.append({
            "video": vid, "qid": qid, "top_k": TOP_K, "half_width": HALF_WIDTH,
            "iop": iop_val, "iou": iou_val, "peak_in_gold": pig_val,
        })

    fallback_rate = n_clip_fallback / n_clip_total if n_clip_total else 0.0
    print(f"[CLIP-ANCHOR] fallback_rate={fallback_rate:.4%} ({n_clip_fallback}/{n_clip_total})")
    if n_clip_fallback != 0:
        print("FATAL: CLIP-anchor fallback rate is nonzero on TEST.", file=sys.stderr)
        sys.exit(1)

    print("\n=== TEST RESULT (top_k=12, half_width=2.2, FROZEN) ===")
    metrics = {}
    for name, table in [
        ("peak_in_gold", per_video_pig),
        ("mIoP", per_video_iop),
        ("IoP@0.5", per_video_iop05),
        ("mIoU", per_video_iou),
        ("IoU@0.5", per_video_iou05),
    ]:
        pt, lo, hi = video_clustered_bootstrap(table)
        metrics[name] = {"point": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"  {name:<12} = {pt:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"  n = {len(per_question_rows)} questions / {n_videos} videos")

    out = {
        "config": {**BASE, "l2_retrieve_top_k": TOP_K, "half_width": HALF_WIDTH,
                   "span_mode": SPAN_MODE, "peak_source": PEAK_SOURCE},
        "n_questions": len(per_question_rows),
        "n_videos": n_videos,
        "clip_anchor_fallback_rate": fallback_rate,
        "metrics": metrics,
        "per_question": per_question_rows,
    }
    out_path = REPO / "eval_results" / "P_NOWA_test_raw.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
