"""Phase 3 query spine: query(question, index) -> result dict.

Runs PER question against a loaded IRISIndex. No video read, no graph rebuild
(the graph is already on index._graph). Wraps existing iris.* modules and
lifts the L1/Cerberus wrappers + claim-split verbatim from pipeline.py so
behavior is identical. Parity with pipeline.run_pipeline is the exit criterion.

NOTE: the retrieved-frame dict deliberately carries NO motion-geometry keys,
so L1's FrameMotionDescriptor geometry is 0.0 — matching old pipeline.py
exactly (old never propagated geometry into L1 either). Wiring real geometry
into L1 is a deliberate Phase 6 change, not part of this restructure.
"""
from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

import iris.aria as aria
from iris.types import IRISIndex


def wrapper_init_l1_cache(config: object = None) -> object:
    """Isolate L1 Elysium cache instantiation. Fallback to cache.py L1Cache if empty/stub."""
    try:
        from iris.l1_elysium import L1ElysiumCache
        return L1ElysiumCache(config)
    except (ImportError, AttributeError):
        from legacy.cache import L1Cache
        return L1Cache()


def wrapper_populate_cache(cache_obj: object, retrieved_frames: list[dict]) -> None:
    """Populate L1 Cache with knowledge triples or CachedFrames representing retrieved video frames."""
    try:
        from iris.l1_elysium import L1ElysiumCache
        use_elysium = isinstance(cache_obj, L1ElysiumCache)
    except ImportError:
        use_elysium = False

    if use_elysium:
        from iris.cached_frame import CachedFrame
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        for frame in retrieved_frames:
            motion = FrameMotionDescriptor(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                luma_diff_energy=frame.get("luma_diff_energy", 0.0),
                divergence=frame.get("divergence", 0.0),
                curl=frame.get("curl", 0.0),
                jacobian_frobenius=frame.get("jacobian_frobenius", 0.0),
                hessian_max_eigenvalue=frame.get("hessian_max_eigenvalue", 0.0),
                motion_entropy=frame.get("motion_entropy", 0.0)
            )
            cached_frame = CachedFrame(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                action_score=frame.get("action_score", 0.0),
                persistence_value=frame.get("persistence_value", 0.0),
                is_peak=frame.get("is_peak", False),
                motion=motion,
                embedding=frame.get("clip_embedding", None),
                caption=frame.get("caption", None)
            )
            cache_obj.admit(cached_frame)
    else:
        from iris.triple import KnowledgeTriple
        for frame in retrieved_frames:
            frame_idx = frame["frame_idx"]
            timestamp = frame.get("timestamp", 0.0)
            res_energy = frame.get("luma_diff_energy", 0.0)
            action_score = frame.get("action_score", 0.0)
            
            # Generate semantic triple
            triple = KnowledgeTriple(
                subject=f"Frame {frame_idx} at {timestamp:.2f}s",
                verb="depicts",
                object=f"salient visual cues (residual energy {res_energy:.4f}, action score {action_score:.4f})"
            )
            # Use action_score/luma_diff_energy as importance ranking score
            score = action_score or res_energy
            if hasattr(cache_obj, "route_triple"):
                cache_obj.route_triple(triple, pagerank_score=score)
            else:
                cache_obj.add_fact(triple, pagerank_score=score)


def wrapper_cerberus_gate(claims: list[str], cache_obj: object, action_score: float, config: object) -> tuple[bool, list[str], list[str], list[str], bool]:
    """Isolate Cerberus-V NLI truth gate.

    Returns (is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked).

    is_verified is True only when BOTH rejected_claims AND unverifiable_claims are empty.
    A claim with no supporting evidence (unverifiable) is NOT the same as verified.
    """
    from iris.cerberus_v import CerberusV

    gate_obj = CerberusV()
    try:
        res = gate_obj.verify(claims, cache_obj, action_score, config)
        verified_claims = res.get("verified", [])
        rejected_claims = res.get("rejected", [])
        unverifiable_claims = res.get("unverifiable", [])
        # Fix 9b: require zero rejected AND zero unverifiable to count as verified.
        is_verified = len(rejected_claims) == 0 and len(unverifiable_claims) == 0
        is_mocked = False
    except Exception as e:
        error_msg = str(e)
        print(f"Error: CerberusV verification failed — gate closed, all claims unverifiable: {error_msg}")
        verified_claims = []
        rejected_claims = []
        unverifiable_claims = list(claims)
        is_verified = False
        is_mocked = False

    return is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked


def _config_from_index(index: IRISIndex, config: Any) -> Any:
    """Reconstruct the config the index was built with (snapshot is a dict;
    getattr on a dict returns defaults, so we must rebuild the dataclass)."""
    if config is not None:
        return config
    from iris.iris_config import IRISConfig
    try:
        return IRISConfig(**index.config_snapshot)
    except Exception:
        return IRISConfig()


def _embed_query(question: str, config: Any) -> np.ndarray:
    """Query-text CLIP embedding. Lifted from wrapper_l2_retrieve (323-335)."""
    from iris._clip import get_clip_model
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return np.zeros(512, dtype=np.float32)
    model, _ = get_clip_model()
    if model is None:
        return np.zeros(512, dtype=np.float32)
    try:
        text_input = clip.tokenize([question]).to(device)
        with torch.no_grad():
            qf = model.encode_text(text_input)
            qf /= qf.norm(dim=-1, keepdim=True)
            return qf.cpu().numpy().flatten().astype(np.float32)
    except Exception:
        return np.zeros(512, dtype=np.float32)


def _split_claims(raw_answer: str) -> list[str]:
    """Sentence-level claim extraction. Lifted VERBATIM from pipeline.py 632-639.
    Do not 'improve' the regex — parity depends on identical splitting."""
    clean_answer = re.sub(r'\*\*.*?\*\*:?\s*', '', raw_answer)
    clean_answer = re.sub(r'^\s*[-*]\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'^\s*\d+\.\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'\n+', ' ', clean_answer).strip()
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
    return [s.strip() for s in raw_sentences if len(s.strip()) >= 12]


def _build_retrieved(index: IRISIndex, query_embedding: np.ndarray, config: Any) -> list[dict]:
    """Mirror of wrapper_l2_retrieve's node->dict mapping + fallback (419-460),
    using the pre-built index._graph and index.frames. NO geometry keys (L1
    parity). Scores come from the graph node; is_peak/embedding/luma_entropy/
    caption come from the matching FrameRecord."""
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    frame_map = {fr.frame_idx: fr for fr in index.frames}
    graph = index._graph

    retrieved: list[dict] = []
    ranking_mode = getattr(config, "ranking_mode", "legacy")
    if graph is not None:
        if ranking_mode == "ppr":
            lambda_ = getattr(config, "ppr_lambda", 0.5)
            damping  = getattr(config, "ppr_damping", 0.5)
            retrieved_nodes = graph.retrieve_ppr(
                query_embedding,
                top_k=l2_retrieve_top_k,
                damping=damping,
                lambda_=lambda_,
            )
        else:
            retrieved_nodes = graph.retrieve(
                query_embedding,
                query_action_score=index.index_action_score,
                top_k=l2_retrieve_top_k,
            )
    else:
        retrieved_nodes = []

    if retrieved_nodes:
        for node in retrieved_nodes:
            fr = frame_map.get(node.frame_idx)
            retrieved.append({
                "frame_idx": node.frame_idx,
                "timestamp": node.timestamp,
                "luma_diff_energy": node.luma_diff_energy,
                "action_score": node.action_score,
                "persistence_value": node.persistence_value,
                "is_peak": getattr(fr, "is_peak", False),
                "clip_embedding": getattr(fr, "clip_embedding", None),
                "luma_entropy": getattr(fr, "luma_entropy", 0.0),
                "caption": getattr(fr, "caption", None),
                "pagerank_score": node.pagerank_score,
                "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
                "retrieval_contributions": getattr(node, "retrieval_contributions", {}),
            })

    if not retrieved:
        sorted_frames = sorted(
            index.frames,
            key=lambda fr: (fr.action_score, fr.luma_diff_energy),
            reverse=True,
        )
        for fr in sorted_frames[:l2_retrieve_top_k]:
            retrieved.append({
                "frame_idx": fr.frame_idx,
                "timestamp": fr.timestamp,
                "luma_diff_energy": fr.luma_diff_energy,
                "action_score": fr.action_score,
                "persistence_value": fr.persistence_value,
                "is_peak": fr.is_peak,
                "clip_embedding": fr.clip_embedding,
                "luma_entropy": fr.luma_entropy,
                "caption": fr.caption,
                "pagerank_score": 0.0,
                "last_retrieval_score": 0.0,
                "retrieval_contributions": {},
            })
    return retrieved


def query(question: str, index: IRISIndex, config: Any = None) -> dict:
    """Answer one question against a loaded index. No video read, no graph rebuild.
    Returns the same result-dict shape as pipeline.run_pipeline (parity target)."""
    config = _config_from_index(index, config)

    t_l2_start = time.time()
    query_embedding = _embed_query(question, config)
    retrieved_frames = _build_retrieved(index, query_embedding, config)
    t_l2 = time.time() - t_l2_start

    t_elysium_start = time.time()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    t_elysium = time.time() - t_elysium_start

    t_aria_start = time.time()
    context_text = cache_obj.as_context_text()
    raw_answer = aria.generate(prompt=question, context=context_text)
    t_aria = time.time() - t_aria_start

    claims = _split_claims(raw_answer)

    t_cerberus_start = time.time()
    max_score = max((f.get("action_score", 0.0) for f in retrieved_frames), default=0.5)
    is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked = \
        wrapper_cerberus_gate(claims, cache_obj, max_score, config)
    t_cerberus = time.time() - t_cerberus_start

    if verified_claims:
        final_answer = " ".join(verified_claims)
    else:
        final_answer = "Insufficient verified evidence to answer this question."

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "verified": is_verified,
        "nli_mocked": is_mocked,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "unverifiable_claims": unverifiable_claims,
        "frames_processed": index.frames_processed,
        "peak_count": index.peak_count,
        "compression_ratio": index.skipped_frames_ratio,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
        "timings": {
            "l2_retrieval": t_l2,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v": t_cerberus,
            "total": t_l2 + t_elysium + t_aria + t_cerberus,
        },
    }
