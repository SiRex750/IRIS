"""Scaling curve v2, query_source=text variant.

Re-runs scripts/scaling_curve_v2.py's flat/scene_sparse curve (same 7 clips,
same watchdog, same N_Q=50) with --query-source text instead of v2's default
--query-source synthetic. Addresses the standing caveat in
eval_results/scaling_curve_v2_NOTES.md: v2's 50 queries/clip were seeded
samples of survivor CLIP *image* embeddings (query_embed_s null throughout),
not real queries. This run encodes a fixed list of surveillance text queries
through the same CLIP ViT-B/32 text encoder used at ingest
(scripts/_scaling_curve_v2_worker.py:TEXT_QUERIES, cycled to 50/clip),
L2-normalized, inside the timed region -- mirroring latency_ab.py's
_timed_query pattern.

query_source=text is opt-in (see scripts/_scaling_curve_v2_worker.py
--query-source); v2's default (query_source=synthetic) is unchanged.

Does NOT re-run the fairness check (independent_ingest vs cached_frames) --
that question is orthogonal to query source and was already settled in v2.

VERIFY: python scripts/scaling_curve_v2_textquery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import scaling_curve_v2 as base

REPO = base.REPO
sys.path.insert(0, str(REPO / "scripts"))
from _scaling_curve_v2_worker import TEXT_QUERIES  # noqa: E402

TMP_DIR = REPO / "eval_results" / "_v2_textquery_tmp"
TMP_DIR.mkdir(exist_ok=True)

OUT_RAW = REPO / "eval_results" / "scaling_curve_v2_textquery_raw.json"

QUERY_SOURCE = "text"


def main():
    clips_out = []

    for label, dataset, cache_path, video_path, basis in base.CLIPS:
        print(f"\n=== {label} (dataset={dataset}) ===", flush=True)
        clip_record = {"label": label, "dataset": dataset, "video": str(video_path.relative_to(REPO)),
                        "target_n_selection_basis": basis}

        print(f"[{label}] scene_sparse (query_source={QUERY_SOURCE}) ...", flush=True)
        ss_raw = base.run_worker("scene_sparse", cache_path, TMP_DIR / f"{label}_scenesparse.json",
                                  query_source=QUERY_SOURCE)
        ss_raw["outcome_normalized"] = base.normalize_outcome(ss_raw)
        clip_record["scene_sparse"] = ss_raw
        n_survivors = ss_raw.get("n_survivors")
        clip_record["n_survivors"] = n_survivors
        print(f"[{label}] scene_sparse -> {clip_record['scene_sparse']['outcome_normalized']} "
              f"(N={n_survivors}, wall_observed={ss_raw.get('wall_sec_observed_by_driver'):.1f}s)", flush=True)

        print(f"[{label}] flat (cached_frames, query_source={QUERY_SOURCE}, "
              f"wall_cap={base.WALL_CAP_SEC}s, mem_cap={base.MEM_CAP_BYTES/1e9:.0f}GB) ...", flush=True)
        flat_raw = base.run_worker("flat", cache_path, TMP_DIR / f"{label}_flat.json",
                                    query_source=QUERY_SOURCE)
        flat_raw["outcome_normalized"] = base.normalize_outcome(flat_raw)
        clip_record["flat"] = flat_raw
        print(f"[{label}] flat -> {flat_raw['outcome_normalized']} "
              f"(wall_observed={flat_raw.get('wall_sec_observed_by_driver'):.1f}s)", flush=True)

        clips_out.append(clip_record)
        _write_checkpoint(clips_out, done=False)

    _write_checkpoint(clips_out, done=True)
    print(f"\nWrote {OUT_RAW}")


def _write_checkpoint(clips_out, done: bool):
    ucf_flat_pts = []
    ucf_ss_pts = []
    for c in clips_out:
        if c["dataset"] != "ucf":
            continue
        n = c["n_survivors"]
        if c["flat"]["outcome_normalized"] == "completed":
            ucf_flat_pts.append((n, c["flat"].get("median_total_retrieval_s")))
        if c["scene_sparse"]["outcome_normalized"] == "completed":
            ucf_ss_pts.append((n, c["scene_sparse"].get("median_total_retrieval_s")))

    fits = {
        "flat_ucf_only": base._fit_exponent(ucf_flat_pts),
        "scene_sparse_ucf_only": base._fit_exponent(ucf_ss_pts),
    }

    provenance = {
        "harness": "scripts/scaling_curve_v2_textquery.py + scripts/scaling_curve_v2.py + "
                    "scripts/_scaling_curve_v2_worker.py",
        "query_source": QUERY_SOURCE,
        "text_queries": TEXT_QUERIES,
        "n_queries_per_clip": base.N_Q,
        "uniform_watchdog": {
            "wall_time_cap_sec": base.WALL_CAP_SEC,
            "mem_cap_bytes": base.MEM_CAP_BYTES,
        },
        "run_complete": done,
        "note": "Re-run of scaling_curve_v2 with query_source=text (real CLIP ViT-B/32 text-encoded "
                "queries, cycling scripts/_scaling_curve_v2_worker.py:TEXT_QUERIES to 50/clip) instead "
                "of v2's default query_source=synthetic (sampled survivor CLIP image embeddings). Same "
                "7 clips, same N, same uniform watchdog as v2. Purpose: confirm the fitted scaling "
                "exponent is invariant to query type. Does NOT re-run v2's fairness check "
                "(independent_ingest vs cached_frames) -- orthogonal to query source, already settled.",
    }

    out = {"provenance": provenance, "clips": clips_out, "fits_ucf_only": fits}
    OUT_RAW.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
