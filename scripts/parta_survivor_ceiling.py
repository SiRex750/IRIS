"""PART A: L1 survivor-coverage ceiling -- diagnostic only (metric_registry.json
lists this as "Survivor_coverage_ceiling", is_official=false). Does the
correct frame even survive L1 admission (ingest()'s retrieval_strategy-driven
frame selection) before L2 PPR ranking ever narrows the pool further?

Read-only: does not touch tuning/all_trials.csv, tuning/frozen_state.json,
or anything Family 3 (PART B) writes. Reuses the same cached indexes Family
1/2 already built under retrieval_strategy="hybrid" (config-hash
cab2bac1628012a3) -- no re-ingest, no GPU, no PPR, no captioning/answering.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import part3_tune as pt  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402

SHORT_LONG_THRESHOLD_S = 2.5  # project convention, smoke/smoke_report.md: short <=2.5s, long >2.5s

TYPE_FAMILY = {"CW": "causal", "CH": "causal", "TN": "temporal", "TC": "temporal", "TP": "temporal"}


def load_questions_with_type():
    split = json.loads((REPO / "split_manifest.json").read_text())
    tune_videos = set(split["tune_videos"])
    rows = list(csv.DictReader(open(REPO / "eval" / "data" / "nextqa" / "val.csv", newline="", encoding="utf-8")))
    gsub = json.loads((REPO / "eval" / "data" / "nextqa" / "gsub_val.json").read_text())
    video_dir = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"

    out = []
    for r in rows:
        vid = r["video"]
        if vid not in tune_videos:
            continue
        qid = r["qid"]
        if not (video_dir / f"{vid}.mp4").exists():
            continue
        gold = gsub.get(vid, {}).get("location", {}).get(qid)
        if not gold:
            continue
        span_len = max(g[1] - g[0] for g in gold)  # length of the longest gold span, for short/long bucketing
        out.append({
            "video": vid, "qid": qid, "type": r["type"],
            "family": TYPE_FAMILY.get(r["type"], "other"),
            "gold_spans": gold, "span_len": span_len,
            "frame_count": int(r["frame_count"]),
        })
    return out


def best_overlap(gold_spans, t):
    for g in gold_spans:
        if g[0] <= t <= g[1]:
            return True
    return False


def main():
    questions = load_questions_with_type()
    print(f"[setup] {len(questions)} val_tune questions with type info", flush=True)

    cfg = pt.make_config({"retrieval_strategy": "hybrid"})  # frozen L1 config (Family 1 result)
    h = pt.ingest_config_hash(cfg)
    print(f"[setup] using L1 config-hash {h} (retrieval_strategy=hybrid, same as Family 1/2)", flush=True)

    video_ids = sorted({q["video"] for q in questions})
    index_paths = pt.ensure_indexes(video_ids, cfg, n_workers=8)  # reuses Family 1's cache, no re-ingest expected
    print(f"[setup] {len(index_paths)}/{len(video_ids)} video indexes available", flush=True)

    video_survivor_info = {}
    for vid, path in index_paths.items():
        idx = iris_ingest.load_index(path)
        timestamps = [fr.timestamp for fr in idx.frames]
        video_survivor_info[vid] = {
            "timestamps": timestamps,
            "survivor_count": len(idx.frames),
        }

    rows = []
    n_survived = 0
    for q in questions:
        vid = q["video"]
        if vid not in video_survivor_info:
            continue
        info = video_survivor_info[vid]
        survived = any(best_overlap(q["gold_spans"], t) for t in info["timestamps"])
        total_frames = q["frame_count"]
        survivor_count = info["survivor_count"]
        retention_pct = (survivor_count / total_frames * 100) if total_frames > 0 else 0.0

        rows.append({
            "video_id": vid, "qid": q["qid"], "question_type": q["type"], "question_family": q["family"],
            "gold_span_length_s": round(q["span_len"], 3),
            "span_bucket": "short" if q["span_len"] <= SHORT_LONG_THRESHOLD_S else "long",
            "survived": survived,
            "total_frames": total_frames, "survivor_count": survivor_count,
            "retention_pct": round(retention_pct, 3),
        })
        if survived:
            n_survived += 1

    with open(REPO / "survivor_coverage_ceiling.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_total = len(rows)
    ceiling = n_survived / n_total if n_total else 0.0
    print(f"[result] ceiling = {n_survived}/{n_total} = {ceiling:.4f}", flush=True)

    # Breakdowns
    def bucket_rate(pred):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None, 0
        return sum(1 for r in sub if r["survived"]) / len(sub), len(sub)

    causal_rate, causal_n = bucket_rate(lambda r: r["question_family"] == "causal")
    temporal_rate, temporal_n = bucket_rate(lambda r: r["question_family"] == "temporal")
    other_rate, other_n = bucket_rate(lambda r: r["question_family"] == "other")
    short_rate, short_n = bucket_rate(lambda r: r["span_bucket"] == "short")
    long_rate, long_n = bucket_rate(lambda r: r["span_bucket"] == "long")

    failures = [r for r in rows if not r["survived"]]
    fail_retentions = [r["retention_pct"] for r in failures]
    ok_retentions = [r["retention_pct"] for r in rows if r["survived"]]
    import statistics
    fail_median_retention = statistics.median(fail_retentions) if fail_retentions else None
    ok_median_retention = statistics.median(ok_retentions) if ok_retentions else None

    summary = {
        "n_total_questions": n_total,
        "n_survived": n_survived,
        "ceiling": ceiling,
        "by_family": {
            "causal": {"rate": causal_rate, "n": causal_n},
            "temporal": {"rate": temporal_rate, "n": temporal_n},
            "other": {"rate": other_rate, "n": other_n},
        },
        "by_span_bucket": {
            "short (<=2.5s)": {"rate": short_rate, "n": short_n},
            "long (>2.5s)": {"rate": long_rate, "n": long_n},
        },
        "failure_cluster": {
            "n_failures": len(failures),
            "failure_median_retention_pct": fail_median_retention,
            "success_median_retention_pct": ok_median_retention,
        },
    }
    json.dump(summary, open(REPO / ".parta_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("SURVIVOR_COVERAGE_CEILING_MEASURED", flush=True)


if __name__ == "__main__":
    main()
