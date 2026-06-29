"""Phase 6 — Per-family retrievability ceiling from real ingests.

Read-only over the cache. For each dev question whose video is cached, loads
the index and runs production PPR retrieval at top_k ∈ {5, 8, 12, 20}.

Ceiling proxy: temporal coverage and structural starvation.
Honest framing: this is a COVERAGE/AVAILABILITY ceiling, not an answer-presence
ceiling. We cannot verify answer presence without gold frames. It tells us
whether retrieval is starved (N_nodes < top_k, or covers too little timeline),
which is the failure mode that would bury the lambda delta. Printed as such.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig
from eval.nextqa_loader import load_split

DATA_DIR  = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
VAL_CSV   = DATA_DIR / "val.csv"

TOP_KS = [5, 8, 12, 20]

CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=max(TOP_KS),
)


def temporal_coverage(frame_idxs: list[int], total_frames: int) -> float:
    """Fraction of video timeline spanned by retrieved frames (max - min / total)."""
    if not frame_idxs or total_frames <= 1:
        return 0.0
    return (max(frame_idxs) - min(frame_idxs)) / (total_frames - 1)


def main() -> None:
    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]

    # frame_count not in JSONL (stripped at write); pull from val.csv
    val_rows = load_split(VAL_CSV)
    vid_to_fc: dict[str, int] = {r["video"]: r["frame_count"] for r in val_rows}

    # Load cache manifest
    cached_vids = {p.stem for p in CACHE_DIR.glob("*.npz")}
    print(f"Cached videos available: {len(cached_vids)}")

    # Per-question records: {family: [{n_nodes, frame_count, retrieved_idxs_by_k}]}
    family_records: dict[str, list[dict]] = {}

    # Load each cached video once, run retrieval for all its dev questions
    cached_by_vid: dict = {}
    skipped_vids = []
    for vid in sorted(cached_vids):
        cache_path = CACHE_DIR / vid
        try:
            idx = iris_ingest.load_index(cache_path)
            cached_by_vid[vid] = idx
        except Exception as e:
            skipped_vids.append((vid, str(e)[:80]))

    if skipped_vids:
        print(f"Failed to load {len(skipped_vids)} cached indexes:")
        for v, e in skipped_vids:
            print(f"  {v}: {e}")

    questions_processed = 0
    questions_skipped = 0

    for row in dev_rows:
        vid = row["video"]
        if vid not in cached_by_vid:
            questions_skipped += 1
            continue

        index = cached_by_vid[vid]
        n_nodes = len(index.frames)
        fc = vid_to_fc.get(vid, n_nodes)

        q = row["question"]
        fam = row["family"]

        emb = iris_query._embed_query(q, CFG)

        retrieved_by_k: dict[int, list[int]] = {}
        for k in TOP_KS:
            cfg_k = IRISConfig(
                ranking_mode="ppr",
                codec_conf_source="packet_size",
                codec_conf_pictype_norm=True,
                ppr_lambda=0.5,
                ppr_damping=0.5,
                l2_retrieve_top_k=k,
            )
            ret = iris_query._build_retrieved(index, emb, cfg_k)
            retrieved_by_k[k] = [f["frame_idx"] for f in ret]

        family_records.setdefault(fam, []).append({
            "n_nodes": n_nodes,
            "frame_count": fc,
            "retrieved_by_k": retrieved_by_k,
            "video": vid,
            "qid": row["qid"],
        })
        questions_processed += 1

    print(f"Questions processed: {questions_processed}  skipped (no cache): {questions_skipped}")
    print()

    print("=== RETRIEVABILITY CEILING (coverage/availability proxy — NOT answer-presence ceiling) ===")
    print("Limitation: no gold frames in NExT-QA MC; cannot verify answer presence.")
    print("This measures structural starvation (N_nodes < top_k) and temporal coverage")
    print("of retrieved frames. Low coverage = retrieval clumps; this buries the lambda")
    print("delta by homogenizing the retrieved set across questions.")
    print()

    # Header
    hdr = f"{'family':>8} {'top_k':>6} | {'med_N_nodes':>11} | {'frac_starved':>12} | {'med_coverage':>12}"
    print(hdr)
    print("-" * len(hdr))

    for fam in sorted(family_records.keys()):
        recs = family_records[fam]
        for k in TOP_KS:
            n_nodes_list = [r["n_nodes"] for r in recs]
            starved = [1 if r["n_nodes"] < k else 0 for r in recs]
            coverages = [
                temporal_coverage(r["retrieved_by_k"].get(k, []), r["frame_count"])
                for r in recs
            ]
            med_n = statistics.median(n_nodes_list)
            frac_s = sum(starved) / len(starved) if starved else 0.0
            med_cov = statistics.median(coverages)
            print(f"{fam:>8} {k:>6} | {med_n:>11.0f} | {frac_s:>11.1%}  | {med_cov:>11.1%} ")

        print()

    # ALL families combined
    all_recs = [r for recs in family_records.values() for r in recs]
    if all_recs:
        for k in TOP_KS:
            n_nodes_list = [r["n_nodes"] for r in all_recs]
            starved = [1 if r["n_nodes"] < k else 0 for r in all_recs]
            coverages = [
                temporal_coverage(r["retrieved_by_k"].get(k, []), r["frame_count"])
                for r in all_recs
            ]
            med_n = statistics.median(n_nodes_list)
            frac_s = sum(starved) / len(starved) if starved else 0.0
            med_cov = statistics.median(coverages)
            print(f"{'ALL':>8} {k:>6} | {med_n:>11.0f} | {frac_s:>11.1%}  | {med_cov:>11.1%} ")


if __name__ == "__main__":
    main()
