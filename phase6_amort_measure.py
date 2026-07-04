"""Amortization-model unit-cost measurement (measurement only — no curve, no
plot, no edits under iris/). Dumps raw numbers to amort_units.json so a later
script can fit b2's amortization curve without re-running the pipeline.

Captures, on videoplayback.mp4:
  1. GATED build (production path, full_decode=False): survivor/total counts,
     frames actually decoded, nx.pagerank call count during build (the known
     double-build — add_frame_nodes_bulk + enrich_nodes_bulk each trigger one
     _update_pagerank(); recorded as-is, NOT fixed here), wall-clock seconds.
  2. Per-frame unit costs (decode, embed), each measured over a small sample
     of K real frames and divided by K, so b2 can model an ungated all-frames
     build without actually embedding all N_all frames.
  3. Per-query retrieve_ppr cost: nx.pagerank call count (expect 1) and
     wall-clock seconds, on the settled production config.

Usage: python phase6_amort_measure.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
import iris.query as iris_query
import iris.charon_v as charon_v
import iris.aria as aria
from iris.iris_config import IRISConfig
from iris._clip import get_frame_clip_embedding, get_semantic_and_clip_caption

SAMPLE_K = 20  # frames sampled for per-frame decode/embed unit costs
TOP_K = 8


class PagerankCounter:
    """Wraps nx.pagerank in place so any `nx.pagerank(...)` call anywhere in
    iris/ is counted, without editing iris/ source. Restores the original on
    exit."""

    def __init__(self) -> None:
        self._orig = nx.pagerank
        self.count = 0

    def __enter__(self) -> "PagerankCounter":
        def counting_pagerank(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)
        nx.pagerank = counting_pagerank
        return self

    def __exit__(self, *exc) -> None:
        nx.pagerank = self._orig


class CaptionCounter:
    """Wraps the `get_semantic_and_clip_caption` name bound into iris.ingest's
    own namespace (a `from iris._clip import ...`), so it counts calls made by
    ingest()'s per-video enrichment loop without editing iris/ source."""

    def __init__(self) -> None:
        self._orig = iris_ingest.get_semantic_and_clip_caption
        self.count = 0

    def __enter__(self) -> "CaptionCounter":
        def counting_caption(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)
        iris_ingest.get_semantic_and_clip_caption = counting_caption
        return self

    def __exit__(self, *exc) -> None:
        iris_ingest.get_semantic_and_clip_caption = self._orig


class CallTimer:
    """Wraps `getattr(namespace, attr_name)` in place, accumulating total
    wall-time + call count across every invocation, then restores the
    original on exit. Same monkeypatch-in-place trick as PagerankCounter/
    CaptionCounter, generalized to track timing rather than just a count.

    `namespace` must be the object whose attribute is actually looked up at
    call time by the code path being measured (e.g. the defining module for
    a bare-name call, or the importing module for a `from x import y`
    binding) — not necessarily where the function is defined.
    """

    def __init__(self, namespace, attr_name: str) -> None:
        self.namespace = namespace
        self.attr_name = attr_name
        self._orig = getattr(namespace, attr_name)
        self.total_sec = 0.0
        self.count = 0

    def __enter__(self) -> "CallTimer":
        orig = self._orig

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                self.total_sec += time.perf_counter() - t0
                self.count += 1

        setattr(self.namespace, self.attr_name, timed)
        return self

    def __exit__(self, *exc) -> None:
        setattr(self.namespace, self.attr_name, self._orig)


def _measure_gated_build(vpath: Path, cfg: IRISConfig) -> dict:
    with PagerankCounter() as pr_ctr, CaptionCounter() as cap_ctr, \
        CallTimer(charon_v, "parse_video") as t_parse_video, \
        CallTimer(charon_v, "compute_motion_geometry") as t_motion_geom, \
        CallTimer(iris_ingest, "get_frame_clip_embedding") as t_embed_legacy, \
        CallTimer(iris_ingest, "get_clip_embedding_from_pil") as t_embed_pilcache, \
        CallTimer(iris_ingest, "get_semantic_and_clip_caption") as t_caption, \
        CallTimer(iris_ingest, "_build_graph") as t_build_graph, \
        CallTimer(aria, "get_captioner") as t_get_captioner, \
        CallTimer(aria, "get_backend") as t_get_backend:
        t0 = time.perf_counter()
        index = iris_ingest.ingest(str(vpath), config=cfg)
        build_sec = time.perf_counter() - t0
        pagerank_calls_build = pr_ctr.count
        caption_calls_build = cap_ctr.count

    embed_total_sec = t_embed_legacy.total_sec + t_embed_pilcache.total_sec
    embed_total_calls = t_embed_legacy.count + t_embed_pilcache.count
    phase_times = {
        "parse_video_sec": t_parse_video.total_sec,
        "parse_video_calls": t_parse_video.count,
        "compute_motion_geometry_sec": t_motion_geom.total_sec,
        "compute_motion_geometry_calls": t_motion_geom.count,
        "embed_total_sec": embed_total_sec,
        "embed_total_calls": embed_total_calls,
        "embed_legacy_sec": t_embed_legacy.total_sec,
        "embed_legacy_calls": t_embed_legacy.count,
        "embed_pilcache_sec": t_embed_pilcache.total_sec,
        "embed_pilcache_calls": t_embed_pilcache.count,
        "caption_total_sec": t_caption.total_sec,
        "caption_total_calls": t_caption.count,
        "build_graph_sec": t_build_graph.total_sec,
        "build_graph_calls": t_build_graph.count,
        "get_captioner_sec": t_get_captioner.total_sec,
        "get_captioner_calls": t_get_captioner.count,
        "get_backend_sec": t_get_backend.total_sec,
        "get_backend_calls": t_get_backend.count,
    }

    n_surv = len(index.frames)
    assert caption_calls_build == 0, (
        f"caption_calls_build ({caption_calls_build}) != 0 — captioning was "
        f"expected to be lazy (moved to query time), but the build-time "
        f"enrichment loop still called get_semantic_and_clip_caption"
    )

    return {
        "index": index,
        "n_all": int(index.stats.get("total", 0)),
        "n_surv": n_surv,
        "frames_actually_decoded": int(index.stats.get("frames_expensive_processed", 0)),
        "pagerank_calls_build": pagerank_calls_build,
        "caption_calls_build": caption_calls_build,
        "build_wall_sec": build_sec,
        "phase_times": phase_times,
    }


def _measure_per_frame_costs(vpath: Path, k: int) -> dict:
    import av

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    container = av.open(str(vpath))
    frames = []
    t_decode0 = time.perf_counter()
    for i, frame in enumerate(container.decode(video=0)):
        frames.append(frame)
        if len(frames) >= k:
            break
    decode_elapsed = time.perf_counter() - t_decode0
    container.close()

    k_actual = len(frames)
    per_frame_decode_sec = decode_elapsed / k_actual if k_actual else 0.0

    # Embed the same K frames, keeping each clip_emb + pil_img for the
    # caption timing below (mirrors ingest.py's legacy no-pil-cache path:
    # clip_emb = get_frame_clip_embedding(frame, device); pil_img = frame.to_image()).
    clip_embs = []
    pil_imgs = []
    t_embed0 = time.perf_counter()
    for frame in frames:
        clip_embs.append(get_frame_clip_embedding(frame, device))
    embed_elapsed = time.perf_counter() - t_embed0
    per_frame_embed_sec = embed_elapsed / k_actual if k_actual else 0.0

    for frame in frames:
        try:
            pil_imgs.append(frame.to_image())
        except Exception:
            pil_imgs.append(None)

    # Warm the model first (throwaway call, timed separately) — captioning
    # backends (VLM / BLIP fallback) commonly pay a one-time load cost on
    # first invocation that must not be smeared into the per-frame figure.
    caption_warmup_sec = 0.0
    if k_actual:
        t_warm0 = time.perf_counter()
        get_semantic_and_clip_caption(pil_imgs[0], frames[0], clip_embs[0], device)
        caption_warmup_sec = time.perf_counter() - t_warm0

    t_caption0 = time.perf_counter()
    for frame, pil_img, clip_emb in zip(frames, pil_imgs, clip_embs):
        get_semantic_and_clip_caption(pil_img, frame, clip_emb, device)
    caption_elapsed = time.perf_counter() - t_caption0
    per_frame_caption_sec = caption_elapsed / k_actual if k_actual else 0.0

    return {
        "sample_k_requested": k,
        "sample_k_actual": k_actual,
        "per_frame_decode_sec": per_frame_decode_sec,
        "per_frame_embed_sec": per_frame_embed_sec,
        "per_frame_caption_sec": per_frame_caption_sec,
        "caption_warmup_sec": caption_warmup_sec,
        "caption_warmup_is_large_one_time_cost": caption_warmup_sec > 5 * per_frame_caption_sec,
    }


def _measure_query_cost(index, cfg: IRISConfig) -> dict:
    emb = iris_query._embed_query("what is moving in the scene", cfg)
    graph = index._graph
    with PagerankCounter() as ctr:
        t0 = time.perf_counter()
        graph.retrieve_ppr(
            emb,
            top_k=TOP_K,
            damping=cfg.ppr_damping,
            lambda_=cfg.ppr_lambda,
        )
        retrieve_sec = time.perf_counter() - t0
        pagerank_calls_query = ctr.count

    return {
        "pagerank_calls_query": pagerank_calls_query,
        "retrieve_wall_sec": retrieve_sec,
    }


def main() -> None:
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
    )

    print("Measuring GATED build (production path, full_decode=False)...")
    sys.stdout.flush()
    build = _measure_gated_build(vpath, cfg)

    print(f"Measuring per-frame decode/embed unit costs (K={SAMPLE_K})...")
    sys.stdout.flush()
    per_frame = _measure_per_frame_costs(vpath, SAMPLE_K)

    print("Measuring per-query retrieve_ppr cost...")
    sys.stdout.flush()
    query = _measure_query_cost(build["index"], cfg)

    # ── Build reconciliation self-check ─────────────────────────────────────
    # accounted = n_surv * (per-frame decode + embed + caption). If this
    # doesn't cover build_wall_sec, there's a second unmeasured term in the
    # build and we must not proceed to fitting an amortization curve.
    n_surv = build["n_surv"]
    accounted_sec = n_surv * (
        per_frame["per_frame_decode_sec"]
        + per_frame["per_frame_embed_sec"]
        + per_frame["per_frame_caption_sec"]
    )
    residual_sec = build["build_wall_sec"] - accounted_sec
    residual_frac = residual_sec / build["build_wall_sec"] if build["build_wall_sec"] else 0.0

    # ── Phase-timing reconciliation ─────────────────────────────────────────
    # Top-level EXCLUSIVE phases only. parse_video already contains
    # compute_motion_geometry (nested), so the latter is NOT added to the sum
    # (would double-count). get_captioner/get_backend are a one-time model
    # load nested inside caption_total's first call — also NOT added.
    phase_times = build["phase_times"]
    phase_parse_video_sec = phase_times["parse_video_sec"]
    phase_embed_total_sec = phase_times["embed_total_sec"]
    phase_caption_total_sec = phase_times["caption_total_sec"]
    phase_build_graph_sec = phase_times["build_graph_sec"]
    phase_accounted_sec = (
        phase_parse_video_sec
        + phase_embed_total_sec
        + phase_caption_total_sec
        + phase_build_graph_sec
    )
    phase_residual_sec = build["build_wall_sec"] - phase_accounted_sec
    phase_residual_frac = (
        phase_residual_sec / build["build_wall_sec"] if build["build_wall_sec"] else 0.0
    )
    motion_geom_share_of_parse_video = (
        phase_times["compute_motion_geometry_sec"] / phase_parse_video_sec
        if phase_parse_video_sec else 0.0
    )

    result = {
        "video": "videoplayback.mp4",
        "config": {
            "ranking_mode": cfg.ranking_mode,
            "ppr_lambda": cfg.ppr_lambda,
            "ppr_damping": cfg.ppr_damping,
            "codec_conf_source": cfg.codec_conf_source,
            "codec_conf_pictype_norm": cfg.codec_conf_pictype_norm,
            "l2_retrieve_top_k": TOP_K,
        },
        "gated_build": {
            "n_all": build["n_all"],
            "n_surv": build["n_surv"],
            "frames_actually_decoded": build["frames_actually_decoded"],
            "pagerank_calls_build": build["pagerank_calls_build"],
            "caption_calls_build": build["caption_calls_build"],
            "build_wall_sec": build["build_wall_sec"],
        },
        "per_frame_unit_costs": per_frame,
        "per_query_retrieve": query,
        "build_reconciliation": {
            "accounted_sec": accounted_sec,
            "residual_sec": residual_sec,
            "residual_frac": residual_frac,
        },
        "phase_times": {
            "top_level_exclusive": {
                "parse_video_sec": phase_parse_video_sec,
                "parse_video_calls": phase_times["parse_video_calls"],
                "embed_total_sec": phase_embed_total_sec,
                "embed_total_calls": phase_times["embed_total_calls"],
                "embed_legacy_sec": phase_times["embed_legacy_sec"],
                "embed_legacy_calls": phase_times["embed_legacy_calls"],
                "embed_pilcache_sec": phase_times["embed_pilcache_sec"],
                "embed_pilcache_calls": phase_times["embed_pilcache_calls"],
                "caption_total_sec": phase_caption_total_sec,
                "caption_total_calls": phase_times["caption_total_calls"],
                "build_graph_sec": phase_build_graph_sec,
                "build_graph_calls": phase_times["build_graph_calls"],
            },
            "nested_annotations_not_in_sum": {
                "compute_motion_geometry_sec": phase_times["compute_motion_geometry_sec"],
                "compute_motion_geometry_calls": phase_times["compute_motion_geometry_calls"],
                "compute_motion_geometry_share_of_parse_video": motion_geom_share_of_parse_video,
                "get_captioner_sec": phase_times["get_captioner_sec"],
                "get_captioner_calls": phase_times["get_captioner_calls"],
                "get_backend_sec": phase_times["get_backend_sec"],
                "get_backend_calls": phase_times["get_backend_calls"],
            },
            "phase_reconciliation": {
                "build_wall_sec": build["build_wall_sec"],
                "accounted_sec": phase_accounted_sec,
                "residual_sec": phase_residual_sec,
                "residual_frac": phase_residual_frac,
            },
        },
    }

    out_path = REPO_ROOT / "amort_units.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n===AMORT_UNITS===")
    print(
        f"N_all={build['n_all']}  N_surv={build['n_surv']}  "
        f"frames_actually_decoded={build['frames_actually_decoded']}  "
        f"pagerank_calls_build={build['pagerank_calls_build']}  "
        f"caption_calls_build={build['caption_calls_build']}  "
        f"build_wall_sec={build['build_wall_sec']:.4f}"
    )
    print(
        f"per_frame_decode_sec={per_frame['per_frame_decode_sec']:.6f}  "
        f"per_frame_embed_sec={per_frame['per_frame_embed_sec']:.6f}  "
        f"per_frame_caption_sec={per_frame['per_frame_caption_sec']:.6f}  "
        f"(K={per_frame['sample_k_actual']})"
    )
    print(
        f"caption_warmup_sec={per_frame['caption_warmup_sec']:.6f}  "
        f"large_one_time_cost={per_frame['caption_warmup_is_large_one_time_cost']}"
    )
    print(
        f"pagerank_calls_query={query['pagerank_calls_query']}  "
        f"retrieve_wall_sec={query['retrieve_wall_sec']:.6f}"
    )
    print(
        f"\naccounted_sec={accounted_sec:.4f}  residual_sec={residual_sec:.4f}  "
        f"residual_frac={residual_frac:.4f}"
    )
    if abs(residual_frac) > 0.10:
        print(
            "STOP: |residual_frac| > 10% — a second unmeasured term is present "
            "in the build. Do not proceed to fitting an amortization curve."
        )

    print("\n===PHASE_TIMES===")
    print(
        f"parse_video_sec={phase_parse_video_sec:.4f} (calls={phase_times['parse_video_calls']})  "
        f"embed_total_sec={phase_embed_total_sec:.4f} (calls={phase_times['embed_total_calls']}: "
        f"legacy={phase_times['embed_legacy_calls']}, pilcache={phase_times['embed_pilcache_calls']})  "
        f"caption_total_sec={phase_caption_total_sec:.4f} (calls={phase_times['caption_total_calls']})  "
        f"build_graph_sec={phase_build_graph_sec:.4f} (calls={phase_times['build_graph_calls']})"
    )
    print(
        f"[nested] compute_motion_geometry_sec={phase_times['compute_motion_geometry_sec']:.4f} "
        f"(calls={phase_times['compute_motion_geometry_calls']}, "
        f"share_of_parse_video={motion_geom_share_of_parse_video:.4f})"
    )
    print(
        f"[nested] get_captioner_sec={phase_times['get_captioner_sec']:.4f} "
        f"(calls={phase_times['get_captioner_calls']})  "
        f"get_backend_sec={phase_times['get_backend_sec']:.4f} "
        f"(calls={phase_times['get_backend_calls']})"
    )
    print(
        f"\nphase_accounted_sec={phase_accounted_sec:.4f}  "
        f"phase_residual_sec={phase_residual_sec:.4f}  "
        f"phase_residual_frac={phase_residual_frac:.4f}"
    )
    if abs(phase_residual_frac) > 0.10:
        print(
            "STOP: |phase_residual_frac| > 10% — a THIRD unmeasured term exists "
            "in the build beyond parse_video/embed/caption/build_graph. Do not "
            "proceed to fitting an amortization curve."
        )
    else:
        print("OK: |phase_residual_frac| <= 10% — no third term detected.")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
