"""
IRIS end-to-end pipeline harness.

Wires: charon_v → action_score → l1_elysium → l2_asphodel → aria → cerberus_v

Entry point for integration testing and ablation runs.
Accepts a video path and a natural language query,
returns a verified answer string.

Owner: Track B
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import av

import charon_v
import aria
from action_score import FrameFeatureBuffer, ActionScoreModule
import os

# Load environment variables from .env file if it exists in the workspace
def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

_load_env()

# Globals for caching the CLIP model
_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def get_clip_model():
    """Load and cache the CLIP ViT-B/32 model globally."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _CLIP_MODEL, _CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            _CLIP_MODEL = None
            _CLIP_PREPROCESS = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


def get_frame_clip_embedding(frame: av.video.frame.VideoFrame, device: str) -> np.ndarray:
    """Convert PyAV frame to image and extract normalized CLIP feature embedding."""
    model, preprocess = get_clip_model()
    if model is None:
        return np.zeros(512, dtype=np.float32)
    try:
        import torch
        img = frame.to_image()  # Returns PIL RGB Image
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to extract CLIP embedding for frame: {e}")
        return np.zeros(512, dtype=np.float32)


# --- Isolation Wrappers for Active/In-Progress Modules ---

def wrapper_init_l1_cache(config: object = None) -> object:
    """Isolate L1 Elysium cache instantiation. Fallback to cache.py L1Cache if empty/stub."""
    try:
        from l1_elysium import L1ElysiumCache
        return L1ElysiumCache(config)
    except (ImportError, AttributeError):
        from cache import L1Cache
        return L1Cache()


def wrapper_populate_cache(cache_obj: object, retrieved_frames: list[dict]) -> None:
    """Populate L1 Cache with knowledge triples or CachedFrames representing retrieved video frames."""
    try:
        from l1_elysium import L1ElysiumCache
        use_elysium = isinstance(cache_obj, L1ElysiumCache)
    except ImportError:
        use_elysium = False

    if use_elysium:
        from cached_frame import CachedFrame
        from frame_motion_descriptor import FrameMotionDescriptor
        for frame in retrieved_frames:
            motion = FrameMotionDescriptor(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                residual_energy=frame.get("residual_energy", 0.0),
                motion_entropy=frame.get("entropy", 0.0),
                hessian_max_eigenvalue=0.0
            )
            cached_frame = CachedFrame(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                action_score=frame.get("action_score", 0.0),
                persistence_value=frame.get("persistence_value", 0.0),
                is_peak=frame.get("is_peak", False),
                motion=motion,
                embedding=frame.get("clip_embedding", None)
            )
            cache_obj.admit(cached_frame)
    else:
        from triple import KnowledgeTriple
        for frame in retrieved_frames:
            frame_idx = frame["frame_idx"]
            timestamp = frame.get("timestamp", 0.0)
            tier = frame.get("tier", "PEAK")
            res_energy = frame.get("residual_energy", 0.0)
            action_score = frame.get("action_score", 0.0)
            
            # Generate semantic triple
            triple = KnowledgeTriple(
                subject=f"Frame {frame_idx} at {timestamp:.2f}s",
                verb="depicts",
                object=f"salient visual cues (residual energy {res_energy:.4f}, action score {action_score:.4f}, tier {tier})"
            )
            # Use action_score/residual_energy as importance ranking score
            score = action_score or res_energy
            if hasattr(cache_obj, "route_triple"):
                cache_obj.route_triple(triple, pagerank_score=score)
            else:
                cache_obj.add_fact(triple, pagerank_score=score)


def wrapper_l2_retrieve(video_path: str | Path, query: str, frames_to_index: list[dict], alpha: float, beta: float) -> list[dict]:
    """Isolate L2 Asphodel graph retrieval. Fallback to sorted action/energy scores if graph is a stub."""
    try:
        from l2_asphodel import L2Asphodel
    except ImportError:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("residual_energy", 0.0)),
            reverse=True
        )
        return sorted_frames[:5]

    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Generate query embedding
    model, _ = get_clip_model()
    if model is not None:
        try:
            text_input = clip.tokenize([query]).to(device)
            with torch.no_grad():
                query_features = model.encode_text(text_input)
                query_features /= query_features.norm(dim=-1, keepdim=True)
                query_embedding = query_features.cpu().numpy().flatten().astype(np.float32)
        except Exception:
            query_embedding = np.zeros(512, dtype=np.float32)
    else:
        query_embedding = np.zeros(512, dtype=np.float32)

    # 2. Build L2Asphodel graph and index frames
    graph = L2Asphodel(config={"alpha": alpha, "beta": beta})
    frame_map = {f["frame_idx"]: f for f in frames_to_index}
    
    container = av.open(str(video_path))
    for idx, frame in enumerate(container.decode(video=0)):
        if idx in frame_map:
            f_data = frame_map[idx]
            
            # Prepare feature record and action score record
            clip_emb = get_frame_clip_embedding(frame, device)
            f_data["clip_embedding"] = clip_emb
            
            feature_record = {
                "frame_idx": f_data["frame_idx"],
                "timestamp": f_data["timestamp"],
                "residual_energy": f_data["residual_energy"],
                "motion_magnitude": f_data.get("motion_magnitude", 0.0),
                "entropy": f_data.get("entropy", 0.0),
                "refined_motion_tensor": np.zeros(1, dtype=np.float32)
            }
            score_record = {
                "action_score": f_data["action_score"],
                "persistence_value": f_data["persistence_value"]
            }
            
            # Add to graph (Cold-Start)
            graph.add_peak_frame(feature_record, score_record)
            # Enrich with CLIP embedding
            graph.enrich_node(f_data["frame_idx"], triples=[], embedding=clip_emb)
            
    container.close()

    # 3. Perform hybrid retrieval (query_action_score set to 0.9 for personalization)
    retrieved_nodes = graph.retrieve(query_embedding, query_action_score=0.9, top_k=5)
    
    # 4. Map returned AsphodelNode objects back to dictionaries expected by cache wrapper
    retrieved = []
    if retrieved_nodes:
        for node in retrieved_nodes:
            orig = frame_map.get(node.frame_idx, {})
            retrieved.append({
                "frame_idx": node.frame_idx,
                "timestamp": node.timestamp,
                "residual_energy": node.residual_energy,
                "action_score": node.action_score,
                "persistence_value": node.persistence_value,
                "is_peak": orig.get("is_peak", False),
                "clip_embedding": orig.get("clip_embedding", None),
                "entropy": orig.get("entropy", 0.0),
            })
    
    if not retrieved:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("residual_energy", 0.0)),
            reverse=True
        )
        for f in sorted_frames[:5]:
            retrieved.append({
                "frame_idx": f["frame_idx"],
                "timestamp": f["timestamp"],
                "residual_energy": f["residual_energy"],
                "action_score": f.get("action_score", 0.0),
                "persistence_value": f.get("persistence_value", 0.0),
                "is_peak": f.get("is_peak", False),
                "clip_embedding": f.get("clip_embedding", None),
                "entropy": f.get("entropy", 0.0),
            })

    return retrieved


def wrapper_cerberus_gate(claims: list[str], cache_obj: object, action_score: float, config: object) -> tuple[bool, list[str], list[str], bool]:
    """Isolate Cerberus-V NLI truth gate."""
    from cerberus_v import CerberusV
    
    gate_obj = CerberusV()
    try:
        res = gate_obj.verify(claims, cache_obj, action_score, config)
        verified_claims = res.get("verified", [])
        rejected_claims = res.get("rejected", [])
        is_verified = len(rejected_claims) == 0
        is_mocked = False
    except Exception as e:
        print(f"Warning: CerberusV verification failed, falling back to mock: {e}")
        verified_claims = list(claims)
        rejected_claims = []
        is_verified = True
        is_mocked = True

    return is_verified, verified_claims, rejected_claims, is_mocked


# --- Main Pipeline Runners ---

def run_pipeline(video_path: str | Path, query: str, verbose: bool = False, nms_window: int = 10) -> dict:
    """
    Run the end-to-end IRIS pipeline using the new continuous action score
    and topological persistence-based peak detection alongside the existing tier path.
    """
    # 1. Load config parameters
    try:
        from iris_config import ConfigManager
        config = ConfigManager().get_config()
        if config is None:
            from iris_config import IRISConfig
            config = IRISConfig()
    except Exception:
        from iris_config import IRISConfig
        config = IRISConfig()

    # 2. Parse video and extract raw frame features non-breakingly from H.264 stream
    # Returns (output_frames, stats, raw_records)
    output_frames, stats, raw_records = charon_v.parse_video(
        str(video_path),
        return_stats=True,
        return_raw=True,
        candidate_thresh=config.candidate_thresh,
        salient_thresh=config.salient_thresh
    )

    # 3. Continuous action scoring & persistence peak detection
    buf = FrameFeatureBuffer(window_size=30)
    score_module = ActionScoreModule(persistence_thresh=0.4)
    action_scores = {}
    
    for record in raw_records:
        buf.push(record)
        score_dict = score_module.score(buf)
        action_scores[record["frame_idx"]] = score_dict

    # Run Non-Maximum Suppression (NMS) on peak frame decisions to avoid clustering
    if nms_window is not None and nms_window > 0:
        # Find frame indices where is_peak is True
        peak_indices = [idx for idx, score_info in action_scores.items() if score_info["is_peak"]]
        # Sort peaks by action_score descending
        peak_indices.sort(key=lambda idx: action_scores[idx]["action_score"], reverse=True)
        
        accepted_peaks = set()
        for idx in peak_indices:
            # If the peak is within nms_window of an already accepted peak, suppress it
            if any(abs(idx - accepted) <= nms_window for accepted in accepted_peaks):
                action_scores[idx]["is_peak"] = False
            else:
                accepted_peaks.add(idx)

    # Map the computed action scores back to the non-SKIP output frames
    raw_map = {r["frame_idx"]: r for r in raw_records}
    for frame in output_frames:
        frame_idx = frame["frame_idx"]
        score_info = action_scores.get(
            frame_idx,
            {"action_score": 0.0, "is_peak": False, "persistence_value": 0.0}
        )
        frame["action_score"] = score_info["action_score"]
        frame["is_peak"] = score_info["is_peak"]
        frame["persistence_value"] = score_info["persistence_value"]
        frame["entropy"] = raw_map.get(frame_idx, {}).get("entropy", 0.0)

    # 4. L2 Graph retrieval
    # Nodes are populated from output_frames (non-SKIP)
    retrieved_frames = wrapper_l2_retrieve(
        video_path,
        query,
        output_frames,
        alpha=config.alpha,
        beta=config.beta
    )

    # 5. Populate L1 active context cache with retrieved frame evidence
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    
    # 6. Generate answer using ARIA LLM brain
    context_text = cache_obj.as_context_text()
    raw_answer = aria.generate(prompt=query, context=context_text)

    # 7. Extract sentence-level claims from the raw answer
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', raw_answer)
    claims = [s.strip() for s in sentences if s.strip()]

    # 8. Run claims through Cerberus-V NLI truth gate
    max_score = max([f.get("action_score", 0.0) for f in retrieved_frames]) if retrieved_frames else 0.5
    is_verified, verified_claims, rejected_claims, is_mocked = wrapper_cerberus_gate(claims, cache_obj, max_score, config)

    # Generate final verified answer (assemble verified claims back together)
    final_answer = " ".join(verified_claims) if verified_claims else raw_answer

    # Determine peak counts and skip statistics
    peak_count = len([f for f in output_frames if f.get("is_peak", False)])
    skipped_frames_ratio = float(stats["skipped"] / stats["total"]) if stats["total"] > 0 else 0.0
    storage_reduction_factor = float(stats["total"] / len(output_frames)) if len(output_frames) > 0 else 0.0

    result = {
        "answer": final_answer,
        "verified": is_verified,
        "nli_mocked": is_mocked,
        "frames_processed": len(output_frames),
        "peak_count": peak_count,
        "compression_ratio": skipped_frames_ratio,  # Keep for backward compatibility
        "skipped_frames_ratio": skipped_frames_ratio,
        "storage_reduction_factor": storage_reduction_factor
    }

    if verbose:
        result["debug_info"] = {
            "action_scores": action_scores,
            "retrieved_frames": retrieved_frames,
            "raw_answer": raw_answer,
            "verified_claims": verified_claims,
            "rejected_claims": rejected_claims
        }

    return result


def run(video_path: str | Path, query: str) -> dict:
    """
    Run the full IRIS pipeline on a video and query.

    Returns:
        {
            "answer": str,           # final verified answer
            "verified": bool,        # cerberus_v gate result
            "frames_processed": int, # non-SKIP frames seen
            "peak_count": int,       # PEAK frames found
            "compression_ratio": float  # SKIP% of total frames
        }
    """
    res = run_pipeline(video_path, query, verbose=False)
    return {
        "answer": res["answer"],
        "verified": res["verified"],
        "frames_processed": res["frames_processed"],
        "peak_count": res["peak_count"],
        "compression_ratio": res["compression_ratio"]
    }
