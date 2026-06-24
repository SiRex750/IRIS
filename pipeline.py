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
from action_score import ActionScoreConfig, ActionScoreModule
import os

# ── Environment loading ─────────────────────────────────────────────────────
def _load_env():
    """Load .env from project root if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

_load_env()

# ── CLIP availability flag ─────────────────────────────────────────────────
def _check_clip_available() -> bool:
    try:
        import clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

_CLIP_AVAILABLE = _check_clip_available()

# Globals for caching the CLIP model
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TEXT_FEATURES = None


def get_zero_shot_caption(clip_embedding: np.ndarray, device: str) -> str:
    """Classify clip_embedding against a set of action/scene vocabulary labels."""
    global _CLIP_TEXT_FEATURES

    if not _CLIP_AVAILABLE:
        return "visual cues from the video"

    vocabulary = [
        "a cartoon rabbit standing in a field",
        "a rabbit watching something",
        "a butterfly flying near a rabbit",
        "a rabbit sleeping on the grass",
        "a badger or rodent appearing on screen",
        "a close-up of a bunny's face",
        "animals playing in the forest",
        "a scenic green meadow with trees",
        "a rodent or small animal moving",
        "a cartoon character showing action or movement",
        "a person walking or running",
        "a person talking or speaking",
        "a car or vehicle moving on a street",
        "an indoor room or office setting",
        "a group of people gathering",
        "a close-up of a person's face",
        "a computer screen or technology interface",
        "a person cooking or eating food",
        "a street scene with buildings",
        "a sports game or athletic activity",
    ]

    model, _ = get_clip_model()
    if model is None:
        return "visual cues from the video"

    try:
        import clip
        import torch

        if _CLIP_TEXT_FEATURES is None:
            text_inputs = clip.tokenize(vocabulary).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                _CLIP_TEXT_FEATURES = text_features.cpu().numpy()

        # Compute cosine similarity
        similarities = np.dot(_CLIP_TEXT_FEATURES, clip_embedding)
        top_idx = np.argmax(similarities)
        return vocabulary[top_idx]
    except Exception as e:
        print(f"Warning: Zero-shot classification failed: {e}")
        return "visual cues from the video"


def get_clip_model():
    """Load and cache the CLIP ViT-B/32 model globally."""
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if not _CLIP_AVAILABLE:
        return None, None
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
    if not _CLIP_AVAILABLE:
        return np.zeros(512, dtype=np.float32)
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
                embedding=frame.get("clip_embedding", None),
                caption=frame.get("caption", None),
                reasons=frame.get("reasons", None)
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


def wrapper_l2_retrieve(video_path: str | Path, query: str, frames_to_index: list[dict], config: object = None) -> list[dict]:
    """Isolate L2 Asphodel graph retrieval. Falls back gracefully when CLIP is unavailable."""
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)

    try:
        from l2_asphodel import L2Asphodel
    except ImportError:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("residual_energy", 0.0)),
            reverse=True
        )
        return sorted_frames[:l2_retrieve_top_k]

    # Determine device
    device = "cpu"
    if _CLIP_AVAILABLE:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass

    # Generate query embedding (zero vector if CLIP unavailable)
    query_embedding = np.zeros(512, dtype=np.float32)
    if _CLIP_AVAILABLE:
        try:
            import clip
            import torch
            model, _ = get_clip_model()
            if model is not None:
                text_input = clip.tokenize([query]).to(device)
                with torch.no_grad():
                    query_features = model.encode_text(text_input)
                    query_features /= query_features.norm(dim=-1, keepdim=True)
                    query_embedding = query_features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"Warning: query embedding failed: {e}")

    # Build L2Asphodel graph and index frames
    graph = L2Asphodel(config=config)
    frame_map = {f["frame_idx"]: f for f in frames_to_index}

    container = av.open(str(video_path))
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in frame_map:
                f_data = frame_map[idx]

                # Extract CLIP embedding (zeros if unavailable)
                clip_emb = get_frame_clip_embedding(frame, device)
                f_data["clip_embedding"] = clip_emb
                
                clip_label = get_zero_shot_caption(clip_emb, device)
                caption_res = aria.generate_caption_for_frame(frame)
                f_data["caption"] = {
                    "clip_label": clip_label,
                    "semantic_caption": caption_res.caption
                }

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

                graph.add_frame_node(feature_record, score_record)
                graph.enrich_node(f_data["frame_idx"], triples=[], embedding=clip_emb)
    finally:
        container.close()

    # Perform hybrid retrieval
    if frames_to_index:
        query_action_score = max(f.get("action_score", 0.0) for f in frames_to_index)
    else:
        query_action_score = 0.5

    retrieved_nodes = graph.retrieve(
        query_embedding if _CLIP_AVAILABLE else None,
        query_action_score=query_action_score,
        top_k=l2_retrieve_top_k
    )

    # Map AsphodelNode objects back to dicts
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
                "caption": orig.get("caption", None),
                "pagerank_score": node.pagerank_score,
                "reasons": orig.get("reasons", None),
                "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
                "retrieval_contributions": getattr(node, "retrieval_contributions", {}),
            })

    if not retrieved:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("residual_energy", 0.0)),
            reverse=True
        )
        for f in sorted_frames[:l2_retrieve_top_k]:
            retrieved.append({
                "frame_idx": f["frame_idx"],
                "timestamp": f["timestamp"],
                "residual_energy": f["residual_energy"],
                "action_score": f.get("action_score", 0.0),
                "persistence_value": f.get("persistence_value", 0.0),
                "is_peak": f.get("is_peak", False),
                "clip_embedding": f.get("clip_embedding", None),
                "entropy": f.get("entropy", 0.0),
                "caption": f.get("caption", None),
                "pagerank_score": 0.0,
                "reasons": f.get("reasons", None),
                "last_retrieval_score": 0.0,
                "retrieval_contributions": {},
            })

    return retrieved


def wrapper_cerberus_gate(claims: list[str], cache_obj: object, action_score: float, config: object) -> tuple[bool, list[str], list[str], list[str], bool]:
    """Isolate Cerberus-V NLI truth gate.

    Returns (is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked).

    is_verified is True only when BOTH rejected_claims AND unverifiable_claims are empty.
    A claim with no supporting evidence (unverifiable) is NOT the same as verified.
    """
    from cerberus_v import CerberusV

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
        print(f"Warning: CerberusV verification failed: {e}")
        verified_claims = []
        rejected_claims = []
        unverifiable_claims = list(claims)
        is_verified = False
        is_mocked = False

    return is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked


def annotate_and_save_frame(frame_data: dict, output_dir: str | Path) -> None:
    """
    Diagnostic tool: annotates a frame record with frame index, action score,
    and peak/valley status, draws a border, and saves it to output_dir.
    
    To enable this tool, toggle `visual_debug_mode = True` in iris_config.py.
    """
    import cv2
    
    frame_img = frame_data.get("frame")
    if frame_img is None:
        return
        
    img = frame_img.copy()
    frame_idx = frame_data["frame_idx"]
    action_score = frame_data["action_score"]
    is_peak = frame_data["is_peak"]
    
    h, w = img.shape[:2]
    
    # Border: Bright Green (0, 255, 0) for PEAK, Dim Grey (105, 105, 105) for VALLEY (BGR)
    border_color = (0, 255, 0) if is_peak else (105, 105, 105)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border_color, 10)
    
    # Annotation texts
    status_str = "PEAK" if is_peak else "VALLEY"
    lines = [
        f"FRAME: {frame_idx}",
        f"ACTION SCORE: {action_score:.2f}",
        f"STATUS: {status_str}"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Draw text with shadow for high contrast legibility
    for i, line in enumerate(lines):
        org = (20, 40 + i * 30)
        # Shadow (black, thickness + 2)
        cv2.putText(img, line, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Main text
        cv2.putText(img, line, org, font, scale, text_color, thickness, cv2.LINE_AA)
        
    output_path = Path(output_dir) / f"frame_{frame_idx:04d}.png"
    cv2.imwrite(str(output_path), img)


# --- Main Pipeline Runner ---

def run_pipeline(video_path: str | Path, query: str, verbose: bool = False, nms_window: int = 10) -> dict:
    """
    Run the end-to-end IRIS pipeline using the continuous action score
    and topological persistence-based peak detection alongside the existing tier path.

    Returns a dict with answer, timings, stats, and (if verbose) debug_info.
    """
    import time

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

    # Enforce active backend diagnostics check
    aria.run_diagnostics()

    # 2. Parse video — extract raw frame features from H.264 stream
    t_start = time.time()
    output_frames, stats, raw_records = charon_v.parse_video(
        str(video_path),
        return_stats=True,
        return_raw=True,
        candidate_thresh=config.candidate_thresh,
        salient_thresh=config.salient_thresh,
        visual_debug_mode=getattr(config, "visual_debug_mode", False)
    )
    t_charon = time.time() - t_start

    # 3. Continuous action scoring & persistence peak detection
    t_action_start = time.time()
    action_score_config = ActionScoreConfig(
        residual_weight=getattr(config, "residual_weight", 0.5),
        motion_weight=getattr(config, "motion_weight", 0.3),
        entropy_weight=getattr(config, "entropy_weight", 0.2),
        peak_distance=getattr(config, "peak_distance", 5),
        peak_prominence=getattr(config, "peak_prominence", 0.05),
        persistence_threshold=getattr(config, "persistence_threshold", 0.4),
        max_prominence=getattr(config, "max_prominence", 0.5),
    )
    score_module = ActionScoreModule(config=action_score_config)
    score_records = score_module.score_all(raw_records)
    action_scores = {r["frame_idx"]: r for r in score_records}

    # Non-Maximum Suppression on peaks to avoid clustering
    if nms_window is not None and nms_window > 0:
        peak_indices = [idx for idx, score_info in action_scores.items() if score_info["is_peak"]]
        peak_indices.sort(key=lambda idx: action_scores[idx]["action_score"], reverse=True)
        accepted_peaks = set()
        for idx in peak_indices:
            parent_peak = None
            for accepted in accepted_peaks:
                if abs(idx - accepted) <= nms_window:
                    parent_peak = accepted
                    break
            
            if parent_peak is not None:
                action_scores[idx]["is_peak"] = False
                action_scores[idx]["nms_suppressed"] = True
                action_scores[idx]["nms_parent"] = parent_peak
            else:
                accepted_peaks.add(idx)
                action_scores[idx]["nms_suppressed"] = False
                action_scores[idx]["nms_parent"] = None

    # Map computed action scores back to non-SKIP output frames
    raw_map = {r["frame_idx"]: r for r in raw_records}
    
    # Diagnostic Tool: Set up the output directory for visual debugging if enabled.
    # Enable via iris_config.py by setting visual_debug_mode = True.
    visual_debug = getattr(config, "visual_debug_mode", False)
    if visual_debug:
        output_dir = Path(video_path).parent / "debug_frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        
    for frame in output_frames:
        frame_idx = frame["frame_idx"]
        score_info = action_scores.get(
            frame_idx,
            {"action_score": 0.0, "is_peak": False, "persistence_value": 0.0, "nms_suppressed": False, "nms_parent": None}
        )
        frame["action_score"] = score_info["action_score"]
        frame["is_peak"] = score_info["is_peak"]
        frame["persistence_value"] = score_info["persistence_value"]
        frame["nms_suppressed"] = score_info.get("nms_suppressed", False)
        frame["nms_parent"] = score_info.get("nms_parent", None)
        frame["entropy"] = raw_map.get(frame_idx, {}).get("entropy", 0.0)
        
        # Calculate label
        if frame["is_peak"]:
            label = "PEAK"
        elif frame["action_score"] >= getattr(config, "cerberus_high_thresh", 0.70):
            label = "HIGH_IMPORTANCE"
        elif frame["action_score"] >= getattr(config, "cerberus_low_thresh", 0.35):
            label = "MEDIUM_IMPORTANCE"
        elif frame["action_score"] >= 0.1:
            label = "LOW_IMPORTANCE"
        else:
            label = "BACKGROUND"
            
        frame["label"] = label
        
        # Calculate reasons
        selected = bool(frame["is_peak"])
        reasons = []
        if selected:
            reasons.append("detected local action maximum")
            reasons.append(f"persistence ({frame['persistence_value']:.2f}) >= threshold ({getattr(config, 'persistence_threshold', 0.4):.2f})")
            reasons.append("accepted after NMS")
        else:
            if frame["nms_suppressed"]:
                reasons.append("suppressed by NMS")
                reasons.append(f"parent peak: Frame {frame['nms_parent']}")
            elif score_info.get("persistence_value", 0.0) > 0.0:
                reasons.append("detected local action maximum")
                reasons.append(f"persistence ({score_info['persistence_value']:.2f}) < threshold ({getattr(config, 'persistence_threshold', 0.4):.2f})")
            else:
                reasons.append("not a local action maximum")
                
        frame["selected"] = selected
        frame["reasons"] = reasons
        
        # Diagnostic Tool: Annotate and save the frame if visual_debug_mode is enabled.
        if visual_debug:
            annotate_and_save_frame(frame, output_dir)
    t_action = time.time() - t_action_start

    # 4. L2 Graph retrieval (dynamic indexing strategy selection)
    t_l2_start = time.time()
    strategy = getattr(config, "retrieval_strategy", "hybrid")
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    
    if strategy == "peak_only":
        frames_to_index = [f for f in output_frames if f.get("is_peak", False)]
    elif strategy == "top_k_action":
        top_n = l2_retrieve_top_k * 3
        frames_to_index = sorted(output_frames, key=lambda x: x.get("action_score", 0.0), reverse=True)[:top_n]
    elif strategy == "peak_neighbors":
        peaks = [f for f in output_frames if f.get("is_peak", False)]
        peak_indices = {f["frame_idx"] for f in peaks}
        target_indices = set()
        frame_idx_list = [f["frame_idx"] for f in output_frames]
        for idx in peak_indices:
            try:
                pos = frame_idx_list.index(idx)
                target_indices.add(idx)
                if pos > 0:
                    target_indices.add(frame_idx_list[pos - 1])
                if pos < len(frame_idx_list) - 1:
                    target_indices.add(frame_idx_list[pos + 1])
            except ValueError:
                target_indices.add(idx)
        frames_to_index = [f for f in output_frames if f["frame_idx"] in target_indices]
    else:  # "hybrid"
        frames_to_index = output_frames

    retrieved_frames = wrapper_l2_retrieve(
        video_path, query, frames_to_index, config=config
    )
    t_l2 = time.time() - t_l2_start

    # 5. Populate L1 active context cache with retrieved frame evidence
    t_elysium_start = time.time()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    
    # Ensure retrieved_frames only contains what survived eviction in L1 Cache
    cache_frame_indices = {f.frame_idx for f in cache_obj.frames()}
    retrieved_frames = [f for f in retrieved_frames if f["frame_idx"] in cache_frame_indices]
    
    # Assert ARIA context frames match reported frames exactly
    aria_context_frames = {f.frame_idx for f in cache_obj.frames()}
    reported_frames = {f["frame_idx"] for f in retrieved_frames}
    assert aria_context_frames == reported_frames, (
        f"Mismatch: ARIA context has frames {aria_context_frames}, "
        f"but reported retrieved frames are {reported_frames}"
    )
    t_elysium = time.time() - t_elysium_start

    # 6. Generate answer via ARIA LLM brain
    t_aria_start = time.time()
    context_text = cache_obj.as_context_text()
    try:
        raw_answer = aria.generate(prompt=query, context=context_text)
    except Exception as e:
        print(f"Warning: ARIA generation failed: {e}")
        raw_answer = (
            f"Based on the video analysis, I processed {len(retrieved_frames)} key frames "
            f"with peak action scores. The retrieved context is: {context_text[:500]}. "
            f"(Note: LLM answer generation failed — check OPENAI_API_KEY in .env)"
        )
    t_aria = time.time() - t_aria_start

    # Phase 6: Add Debugging (logs/aria_debug/debug_<timestamp>.json)
    try:
        import datetime
        import json
        log_dir = Path(__file__).parent / "logs" / "aria_debug"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Format filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = log_dir / f"debug_{timestamp_str}.json"
        
        frames_sent = [f.frame_idx for f in cache_obj.frames()]
        
        # Extract semantic captions from cache frames
        semantic_captions = []
        for f in cache_obj.frames():
            caption_val = getattr(f, "caption", None)
            if isinstance(caption_val, dict):
                semantic_captions.append(caption_val.get("semantic_caption", ""))
            else:
                semantic_captions.append(caption_val or "")
                
        log_data = {
            "frames_sent_to_aria": frames_sent,
            "semantic_captions": semantic_captions,
            "prompt": query,
            "response": raw_answer
        }
        
        with open(log_file, "w", encoding="utf-8") as lf:
            json.dump(log_data, lf, indent=2)
    except Exception as le:
        print(f"Warning: Failed to write ARIA debug log: {le}")

    # 7. Extract sentence-level claims from the raw answer
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', raw_answer)
    claims = [s.strip() for s in sentences if s.strip()]

    # 8. Run claims through Cerberus-V NLI truth gate
    t_cerberus_start = time.time()
    max_score = max([f.get("action_score", 0.0) for f in retrieved_frames]) if retrieved_frames else 0.5
    is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked = wrapper_cerberus_gate(claims, cache_obj, max_score, config)
    t_cerberus = time.time() - t_cerberus_start

    # Generate final verified answer (verified claims only; raw_answer and breakdowns are also
    # returned in the result dict so nothing is silently dropped from reporting).
    final_answer = " ".join(verified_claims) if verified_claims else raw_answer

    # Statistics (skipped frames are non-peaks in reconciled architecture)
    peak_count = len([f for f in output_frames if f.get("is_peak", False)])
    skipped_count = stats["total"] - peak_count
    skipped_frames_ratio = float(skipped_count / stats["total"]) if stats["total"] > 0 else 0.0
    storage_reduction_factor = float(stats["total"] / peak_count) if peak_count > 0 else 1.0

    result = {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "verified": is_verified,
        "nli_mocked": is_mocked,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "unverifiable_claims": unverifiable_claims,
        "frames_processed": len(output_frames),
        "peak_count": peak_count,
        "compression_ratio": skipped_frames_ratio,
        "skipped_frames_ratio": skipped_frames_ratio,
        "storage_reduction_factor": storage_reduction_factor,
        "clip_available": _CLIP_AVAILABLE,
        "timings": {
            "charon_v": t_charon,
            "action_score": t_action,
            "l2_retrieval": t_l2,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v": t_cerberus,
            "total": t_charon + t_action + t_l2 + t_elysium + t_aria + t_cerberus
        }
    }

    # Phase 10: Validation Report Stats
    total_captions_attempted = len(frames_to_index)
    successful_captions = sum(
        1 for f in frames_to_index
        if isinstance(f.get("caption"), dict) and f["caption"].get("semantic_caption") != "[CAPTION_FAILED]"
    )
    caption_success_rate = float(successful_captions / total_captions_attempted) if total_captions_attempted > 0 else 0.0
    
    aria_success = 1.0 if not raw_answer.startswith("Based on the video analysis, I processed") or "failed" not in raw_answer.lower() else 0.0
    
    validation_report = {
        "aria": {
            "backend": aria.get_backend().__class__.__name__,
            "model": "gpt-4o-mini",
            "success_rate": aria_success,
            "caption_success_rate": caption_success_rate
        },
        "retrieval": {
            "frames_indexed": len(frames_to_index),
            "frames_retrieved": len(retrieved_frames)
        },
        "elysium": {
            "cache_hits": getattr(cache_obj, "hits", 0),
            "cache_misses": getattr(cache_obj, "misses", 0),
            "context_size": len(context_text)
        },
        "cerberus": {
            "verified": len(verified_claims),
            "rejected": len(rejected_claims),
            "unverifiable": len(unverifiable_claims)
        },
        "failures": {
            "caption_failures": aria.get_caption_failures()
        }
    }
    
    result["validation_report"] = validation_report

    # Phase 9: Observability Logging (logs/pipeline/)
    try:
        import datetime
        import json
        pipeline_log_dir = Path(__file__).parent / "logs" / "pipeline"
        pipeline_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Format filename with timestamp
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pipeline_log_file = pipeline_log_dir / f"run_{timestamp_str}.json"
        
        selected_frames = [f["frame_idx"] for f in output_frames if f.get("is_peak", False)]
        ret_frames = [{
            "frame_idx": f["frame_idx"],
            "timestamp": f.get("timestamp", 0.0),
            "action_score": f.get("action_score", 0.0),
            "persistence_value": f.get("persistence_value", 0.0)
        } for f in retrieved_frames]
        
        captions_map = {}
        for f in output_frames:
            if "caption" in f:
                captions_map[f["frame_idx"]] = f["caption"]
                
        retrieval_scores = {}
        for node in retrieved_frames:
            retrieval_scores[node["frame_idx"]] = node.get("last_retrieval_score", 0.0)
            
        pipeline_log_data = {
            "selected_frames": selected_frames,
            "retrieved_frames": ret_frames,
            "captions": captions_map,
            "aria_prompt": query,
            "aria_response": raw_answer,
            "cerberus_result": {
                "verified": verified_claims,
                "rejected": rejected_claims,
                "unverifiable": unverifiable_claims,
                "is_verified": is_verified
            },
            "retrieval_scores": retrieval_scores
        }
        
        with open(pipeline_log_file, "w", encoding="utf-8") as lf:
            json.dump(pipeline_log_data, lf, indent=2)
    except Exception as ple:
        print(f"Warning: Failed to write pipeline run log: {ple}")

    if verbose:
        result["debug_info"] = {
            "all_frames": output_frames,
            "action_scores": action_scores,
            "retrieved_frames": retrieved_frames,
            "context_text": context_text,
            "raw_answer": raw_answer,
            "verified_claims": verified_claims,
            "rejected_claims": rejected_claims,
            "unverifiable_claims": unverifiable_claims,
            
            # --- ARIA Debug Data ---
            "aria_prompt": query,
            "aria_context": context_text,
            "aria_response": raw_answer,
            "frames_given_to_aria": [f.frame_idx for f in cache_obj.frames()],
            "context_length": len(context_text),
            "retrieval_results": context_text,
            
            # --- Cerberus Debug Data ---
            "cerberus_claims": claims,
            "cerberus_evidence": [entry.text for entry in cache_obj.set_facts.values()],
            "cerberus_result": {
                "verified": verified_claims,
                "rejected": rejected_claims,
                "unverifiable": unverifiable_claims,
                "is_verified": is_verified
            },
            
            # --- Thresholds & Configuration ---
            "thresholds": {
                "persistence_threshold": getattr(config, "persistence_threshold", 0.4),
                "max_prominence": getattr(config, "max_prominence", 0.5),
                "peak_prominence": getattr(config, "peak_prominence", 0.05),
                "peak_distance": getattr(config, "peak_distance", 5)
            }
        }

    return result


def run(video_path: str | Path, query: str) -> dict:
    """
    Simplified entry point for the full IRIS pipeline.
    Returns answer, verified status, frames_processed, peak_count, compression_ratio.
    """
    res = run_pipeline(video_path, query, verbose=False)
    return {
        "answer": res["answer"],
        "verified": res["verified"],
        "frames_processed": res["frames_processed"],
        "peak_count": res["peak_count"],
        "compression_ratio": res["compression_ratio"],
    }
