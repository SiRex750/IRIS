"""
IRIS Fine-grained Latency Profiler.

Runs the pipeline with sub-stage timing instrumentation and outputs:
  1. A sorted bottleneck table (stage, latency_ms, % of total)
  2. Search pool reduction at every filtering step

Usage:
    python profiler.py <video_path> [--query "..."] [--runs N]

Example:
    python profiler.py benchmark_results/mov_bbb.mp4
    python profiler.py benchmark_results/mov_bbb.mp4 --runs 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Thin timer context manager
# ---------------------------------------------------------------------------
class Timer:
    """Context manager that records wall-clock time in milliseconds."""
    def __init__(self):
        self.ms: float = 0.0
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.ms = (time.perf_counter() - self._t0) * 1000.0


# ---------------------------------------------------------------------------
# Profiled pipeline run
# ---------------------------------------------------------------------------
def profiled_run(video_path: str, query: str) -> dict:
    """
    Replays the pipeline stage-by-stage with fine-grained timers.
    Returns a dict of sub-stage timings and search pool counts.
    """
    from iris_config import IRISConfig
    import charon_v
    from action_score import ActionScoreConfig, ActionScoreModule
    import aria
    from pipeline import (
        get_clip_model,
        get_clip_embedding_from_pil,
        get_zero_shot_caption,
        wrapper_init_l1_cache,
        wrapper_populate_cache,
        wrapper_cerberus_gate,
    )
    from l2_asphodel import L2Asphodel
    import av
    import re
    import clip
    import torch

    config = IRISConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timings: dict[str, float] = {}

    # ── Stage 1: Charon-V ────────────────────────────────────────────────
    with Timer() as t_charon:
        output_frames, stats, raw_records = charon_v.parse_video(
            str(video_path),
            return_stats=True,
            return_raw=True,
            candidate_thresh=config.candidate_thresh,
            adaptive=config.adaptive,
        )
    timings["charon_v_total"] = t_charon.ms

    total_frames      = stats["total"]
    candidate_frames  = len(output_frames)          # non-SKIP frames
    skipped_frames    = stats["skipped"]

    # ── Stage 2: ActionScore ─────────────────────────────────────────────
    with Timer() as t_norm:
        as_config = ActionScoreConfig(
            residual_weight=config.residual_weight,
            motion_weight=config.motion_weight,
            entropy_weight=config.entropy_weight,
            peak_distance=config.peak_distance,
            peak_prominence=config.peak_prominence,
            persistence_threshold=config.persistence_threshold,
            max_prominence=config.max_prominence,
        )
        score_module = ActionScoreModule(config=as_config)
        score_records = score_module.score_all(raw_records)
    timings["action_score_normalize"] = t_norm.ms

    action_scores = {r["frame_idx"]: r for r in score_records}

    # NMS
    with Timer() as t_nms:
        peak_indices = [idx for idx, s in action_scores.items() if s["is_peak"]]
        peak_indices.sort(key=lambda idx: action_scores[idx]["action_score"], reverse=True)
        accepted_peaks: set[int] = set()
        for idx in peak_indices:
            if any(abs(idx - a) <= 10 for a in accepted_peaks):
                action_scores[idx]["is_peak"] = False
            else:
                accepted_peaks.add(idx)
    timings["action_score_nms"] = t_nms.ms

    # Map scores back
    raw_map = {r["frame_idx"]: r for r in raw_records}
    for frame in output_frames:
        fidx = frame["frame_idx"]
        si = action_scores.get(fidx, {"action_score": 0.0, "is_peak": False, "persistence_value": 0.0})
        frame["action_score"]     = si["action_score"]
        frame["is_peak"]          = si["is_peak"]
        frame["persistence_value"] = si["persistence_value"]
        frame["entropy"]           = raw_map.get(fidx, {}).get("entropy", 0.0)

    peak_frames = len([f for f in output_frames if f.get("is_peak", False)])

    # ── Stage 3: CLIP model load (cached after 1st call) ─────────────────
    with Timer() as t_clip_load:
        clip_model, _ = get_clip_model()
    timings["clip_model_load"] = t_clip_load.ms

    # ── Stage 4: CLIP embedding (from PIL cache — no re-decode) ──────────
    with Timer() as t_clip_emb:
        for f_data in output_frames:
            if f_data.get("pil_image") is not None:
                emb = get_clip_embedding_from_pil(f_data["pil_image"], device)
            else:
                emb = np.zeros(512, dtype=np.float32)
            f_data["clip_embedding"] = emb
            f_data["caption"]        = get_zero_shot_caption(emb, device)
    timings["clip_embedding"] = t_clip_emb.ms

    # ── Stage 5: L2 Asphodel graph build (bulk) ──────────────────────────
    with Timer() as t_graph_build:
        graph = L2Asphodel(config=config)
        feature_records = [{
            "frame_idx":             f["frame_idx"],
            "timestamp":             f["timestamp"],
            "residual_energy":       f["residual_energy"],
            "motion_magnitude":      f.get("motion_magnitude", 0.0),
            "entropy":               f.get("entropy", 0.0),
            "refined_motion_tensor": np.zeros(1, dtype=np.float32),
        } for f in output_frames]
        score_recs = [{
            "action_score":     f["action_score"],
            "persistence_value": f["persistence_value"],
        } for f in output_frames]
        graph.add_frame_nodes_bulk(feature_records, score_recs)
    timings["l2_graph_build"] = t_graph_build.ms

    # ── Stage 6: Enrich + query embedding ────────────────────────────────
    with Timer() as t_enrich:
        enrichment_map = {f["frame_idx"]: f["clip_embedding"] for f in output_frames}
        graph.enrich_nodes_bulk(enrichment_map)

        if clip_model is not None:
            try:
                text_input = clip.tokenize([query]).to(device)
                with torch.no_grad():
                    qf = clip_model.encode_text(text_input)
                    qf /= qf.norm(dim=-1, keepdim=True)
                    query_embedding = qf.cpu().numpy().flatten().astype(np.float32)
            except Exception:
                query_embedding = np.zeros(512, dtype=np.float32)
        else:
            query_embedding = np.zeros(512, dtype=np.float32)
    timings["l2_enrich_and_query_embed"] = t_enrich.ms

    # ── Stage 7: L2 Graph retrieval ──────────────────────────────────────
    with Timer() as t_retrieve:
        qa_score = max(f.get("action_score", 0.0) for f in output_frames) if output_frames else 0.5
        top_k    = config.l2_retrieve_top_k
        retrieved_nodes = graph.retrieve(query_embedding, query_action_score=qa_score, top_k=top_k)
    timings["l2_retrieval"] = t_retrieve.ms

    frame_map = {f["frame_idx"]: f for f in output_frames}
    retrieved_frames = []
    for node in retrieved_nodes:
        orig = frame_map.get(node.frame_idx, {})
        retrieved_frames.append({
            "frame_idx":       node.frame_idx,
            "timestamp":       node.timestamp,
            "residual_energy": node.residual_energy,
            "action_score":    node.action_score,
            "persistence_value": node.persistence_value,
            "is_peak":         orig.get("is_peak", False),
            "clip_embedding":  orig.get("clip_embedding", None),
            "entropy":         orig.get("entropy", 0.0),
            "caption":         orig.get("caption", None),
        })
    if not retrieved_frames:
        retrieved_frames = sorted(output_frames, key=lambda x: x.get("action_score", 0.0), reverse=True)[:top_k]

    retrieved_count = len(retrieved_frames)

    # ── Stage 8: L1 Elysium populate ─────────────────────────────────────
    with Timer() as t_l1:
        cache_obj = wrapper_init_l1_cache(config)
        wrapper_populate_cache(cache_obj, retrieved_frames)
    timings["l1_elysium"] = t_l1.ms

    # ── Stage 9: ARIA prompt build ────────────────────────────────────────
    with Timer() as t_aria_ctx:
        context_text = cache_obj.as_context_text()
        token_count  = len(context_text.split())   # rough word-level proxy
    timings["aria_prompt_build"] = t_aria_ctx.ms

    # ── Stage 10: ARIA inference ──────────────────────────────────────────
    with Timer() as t_aria_inf:
        import aria
        raw_answer = aria.generate(prompt=query, context=context_text)
    timings["aria_inference"] = t_aria_inf.ms

    # ── Stage 11: Cerberus — claim extraction ─────────────────────────────
    with Timer() as t_claim:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', raw_answer)
        claims    = [s.strip() for s in sentences if s.strip()]
    timings["cerberus_claim_extract"] = t_claim.ms

    # ── Stage 12: Cerberus NLI ───────────────────────────────────────────
    with Timer() as t_nli:
        max_score = max(f.get("action_score", 0.0) for f in retrieved_frames) if retrieved_frames else 0.5
        is_verified, verified, rejected, unverifiable, is_mocked = wrapper_cerberus_gate(
            claims, cache_obj, max_score, config
        )
    timings["cerberus_nli"] = t_nli.ms

    total_ms = sum(timings.values())

    return {
        "timings":     timings,
        "total_ms":    total_ms,
        "search_pool": {
            "total_frames":      total_frames,
            "candidate_frames":  candidate_frames,
            "peak_frames":       peak_frames,
            "retrieved_frames":  retrieved_count,
            "frames_sent_to_aria": retrieved_count,
        },
        "token_count":  token_count,
        "verified_count":    len(verified),
        "rejected_count":    len(rejected),
        "unverifiable_count": len(unverifiable),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_bottleneck_table(timings: dict[str, float], total_ms: float) -> None:
    sorted_stages = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 70)
    print("  IRIS LATENCY BREAKDOWN (sorted by time)")
    print("=" * 70)
    print(f"  {'Stage':<35} {'Latency (ms)':>12}  {'% Total':>8}")
    print("  " + "-" * 64)
    for stage, ms in sorted_stages:
        pct = ms / total_ms * 100 if total_ms > 0 else 0.0
        print(f"  {stage:<35} {ms:>12.1f}  {pct:>7.1f}%")
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<35} {total_ms:>12.1f}  {'100.0%':>8}")
    print("=" * 70)

    print("\n  TOP 3 BOTTLENECKS:")
    for i, (stage, ms) in enumerate(sorted_stages[:3], 1):
        pct = ms / total_ms * 100 if total_ms > 0 else 0.0
        print(f"  {i}. {stage}: {ms:.1f} ms ({pct:.1f}%)")


def print_search_pool(pool: dict) -> None:
    total = pool["total_frames"]
    def pct(n):
        return f"{100.0 - n/total*100:.1f}% reduction" if total > 0 else "—"

    print("\n" + "=" * 50)
    print("  SEARCH POOL REDUCTION")
    print("=" * 50)
    print(f"  {total:>6} total frames")
    print(f"  {pool['candidate_frames']:>6} candidate frames  ({pct(pool['candidate_frames'])})")
    print(f"  {pool['peak_frames']:>6} peak frames")
    print(f"  {pool['retrieved_frames']:>6} retrieved frames  ({pct(pool['retrieved_frames'])} from total)")
    print(f"  {pool['frames_sent_to_aria']:>6} frames sent to ARIA")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="IRIS Fine-grained Latency Profiler")
    parser.add_argument("video_path", nargs="?", default="benchmark_results/mov_bbb.mp4")
    parser.add_argument("--query", default="What action events happen in this video?")
    parser.add_argument("--runs",  type=int, default=1, help="Number of runs to average")
    parser.add_argument("--out",   default="benchmark_results/profiler_report.json")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        fallback = Path("benchmark_results/mov_bbb.mp4")
        if fallback.exists():
            video_path = fallback
        else:
            print(f"Error: video not found at {args.video_path}")
            return

    print(f"\nProfiling: {video_path}  (query: '{args.query}')")
    print(f"Runs: {args.runs}\n")

    all_results = []
    for run_idx in range(args.runs):
        print(f"--- Run {run_idx + 1}/{args.runs} ---")
        result = profiled_run(str(video_path), args.query)
        all_results.append(result)
        print(f"  Total: {result['total_ms']:.0f} ms")

    # Average across runs
    avg_timings: dict[str, float] = {}
    for stage in all_results[0]["timings"]:
        avg_timings[stage] = float(np.mean([r["timings"][stage] for r in all_results]))
    avg_total = float(np.mean([r["total_ms"] for r in all_results]))

    # Report
    print_bottleneck_table(avg_timings, avg_total)
    print_search_pool(all_results[-1]["search_pool"])

    # JSON stage list for paper
    stage_list = sorted(
        [{"stage": s, "latency_ms": round(ms, 2), "percentage_of_total": round(ms / avg_total * 100, 2)}
         for s, ms in avg_timings.items()],
        key=lambda x: x["latency_ms"], reverse=True
    )
    report = {
        "video": str(video_path),
        "query": args.query,
        "runs":  args.runs,
        "avg_total_ms": round(avg_total, 2),
        "stages": stage_list,
        "search_pool": all_results[-1]["search_pool"],
        "context_tokens": all_results[-1]["token_count"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved profiler report to: {out_path}")


if __name__ == "__main__":
    main()
