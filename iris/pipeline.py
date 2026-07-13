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

import iris.charon_v as charon_v
import iris.aria as aria
from iris.action_score import ActionScoreConfig, ActionScoreModule
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

# ── CLIP availability flag ─────────────────────────────────────────────────
def _check_clip_available() -> bool:
    try:
        import clip  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

_CLIP_AVAILABLE = _check_clip_available()

# Globals for caching the CLIP and BLIP models
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TEXT_FEATURES = None

_BLIP_MODEL = None
_BLIP_PROCESSOR = None


def get_blip_model():
    """Load and cache the BLIP image captioning model globally."""
    global _BLIP_MODEL, _BLIP_PROCESSOR
    if _BLIP_MODEL is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            print(f"[INFO] Loading BLIP model on {device} (first run will download ~990MB)...")
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            print("[INFO] BLIP model loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load BLIP model: {e}")
            _BLIP_MODEL = None
            _BLIP_PROCESSOR = None
    return _BLIP_MODEL, _BLIP_PROCESSOR


def get_generative_caption(pil_image, device: str) -> str:
    """Generate a dynamic, natural-language caption for a PIL image using BLIP."""
    if pil_image is None:
        return "visual cues from the video"
    model, processor = get_blip_model()
    if model is None or processor is None:
        return "visual cues from the video"
    try:
        import torch
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Warning: Generative captioning failed: {e}")
        return "visual cues from the video"


def get_zero_shot_caption(clip_embedding: np.ndarray, device: str) -> str:
    """Classify clip_embedding against a set of action/scene vocabulary labels."""
    global _CLIP_TEXT_FEATURES
    
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


def get_semantic_and_clip_caption(pil_img, frame, clip_emb, device) -> dict:
    clip_label = get_zero_shot_caption(clip_emb, device)
    
    # 1. Try VLM OpenAI vision captioning first (ARIA)
    caption_res = aria.generate_caption_for_frame(pil_img if pil_img is not None else frame)
    if caption_res.success:
        semantic_caption = caption_res.caption
    else:
        # 2. Fall back to local generative BLIP model (which is a real generative model!)
        blip_caption = get_generative_caption(pil_img if pil_img is not None else (frame.to_image() if frame else None), device)
        if blip_caption and blip_caption != "visual cues from the video":
            semantic_caption = blip_caption
        else:
            semantic_caption = "[CAPTION_FAILED]"
            
    return {
        "clip_label": clip_label,
        "semantic_caption": semantic_caption
    }

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


def get_clip_embedding_from_pil(pil_image, device: str) -> np.ndarray:
    """Extract normalized CLIP embedding from a PIL image.

    Used by the PIL-cache fast path in wrapper_l2_retrieve to avoid re-opening
    the video file when Charon-V has already captured frame images during its
    2nd decode pass.
    """
    model, preprocess = get_clip_model()
    if model is None or pil_image is None:
        return np.zeros(512, dtype=np.float32)
    try:
        import torch
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten().astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to extract CLIP embedding from PIL image: {e}")
        return np.zeros(512, dtype=np.float32)


# --- Isolation Wrappers for Active/In-Progress Modules ---

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


def wrapper_l2_retrieve(video_path: str | Path, query: str, frames_to_index: list[dict], config: object = None) -> list[dict]:
    """Isolate L2 Asphodel graph retrieval. Fallback to sorted action/energy scores if graph is a stub.

    Optimisations applied:
      1. PIL-cache fast path: if Charon-V stored pil_image for every candidate frame,
         CLIP embeddings are extracted directly from those images — no 3rd video decode.
      2. Bulk graph build: add_frame_nodes_bulk + enrich_nodes_bulk call
         _update_all_edge_weights / _update_pagerank ONCE instead of N times.
    """
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    try:
        from iris.l2_asphodel import L2Asphodel
        if not hasattr(L2Asphodel, "batch_add_frame_nodes"):
            def batch_add_frame_nodes(self, node_records, node_groups=None):
                feature_records = [r[0] for r in node_records]
                action_score_records = [r[1] for r in node_records]
                self.add_frame_nodes_bulk(feature_records, action_score_records, node_groups=node_groups)
            L2Asphodel.batch_add_frame_nodes = batch_add_frame_nodes
        if not hasattr(L2Asphodel, "batch_enrich_nodes"):
            def batch_enrich_nodes(self, enrichment_records, node_groups=None):
                enrichment_map = {r[0]: r[2] for r in enrichment_records}
                self.enrich_nodes_bulk(enrichment_map, node_groups=node_groups)
            L2Asphodel.batch_enrich_nodes = batch_enrich_nodes
    except ImportError:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("luma_diff_energy", 0.0)),
            reverse=True
        )
        return sorted_frames[:l2_retrieve_top_k]

    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Generate query embedding from text
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

    # 2. Build L2Asphodel graph and index frames.
    graph = L2Asphodel(config=config)
    frame_map = {f["frame_idx"]: f for f in frames_to_index}

    # Check whether every candidate frame has a usable PIL image from Charon-V.
    has_pil_cache = bool(frames_to_index) and all(
        f.get("pil_image") is not None for f in frames_to_index
    )

    # Pass 1: collect all node data without touching the graph
    node_records = []       # list of (feature_record, score_record)
    enrichment_records = [] # list of (frame_idx, triples, embedding)

    if has_pil_cache:
        # ── Fast path (no 3rd video decode) ──────────────────────────────
        for f_data in frames_to_index:
            clip_emb = get_clip_embedding_from_pil(f_data["pil_image"], device)
            f_data["clip_embedding"] = clip_emb
            
            caption = get_semantic_and_clip_caption(f_data["pil_image"], None, clip_emb, device)
            f_data["caption"] = caption

            feature_record = {
                "frame_idx":            f_data["frame_idx"],
                "timestamp":            f_data["timestamp"],
                "luma_diff_energy":      f_data["luma_diff_energy"],
                "motion_magnitude":     f_data.get("motion_magnitude", 0.0),
                "luma_entropy":              f_data.get("luma_entropy", 0.0),
                "refined_motion_tensor": np.asarray([
                    float(f_data.get("motion_magnitude", 0.0)),
                    float(f_data.get("divergence", 0.0)),
                    float(f_data.get("curl", 0.0)),
                    float(f_data.get("jacobian_frobenius", 0.0)),
                    float(f_data.get("hessian_max_eigenvalue", 0.0)),
                    float(f_data.get("motion_entropy", 0.0)),
                ], dtype=np.float32),
                "packet_size":          float(f_data.get("packet_size", 0.0)),
                "codec_conf":           float(f_data.get("codec_conf", 0.5)),
                "pict_type":            str(f_data.get("pict_type", "?")),
                "is_peak":              bool(f_data.get("is_peak", False)),
            }
            score_record = {
                "action_score":     f_data["action_score"],
                "persistence_value": f_data["persistence_value"],
            }
            node_records.append((feature_record, score_record))
            enrichment_records.append((f_data["frame_idx"], [], clip_emb))
    else:
        # ── Legacy path (full video decode) ──────────────────────────────
        container = av.open(str(video_path))
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in frame_map:
                f_data   = frame_map[idx]
                clip_emb = get_frame_clip_embedding(frame, device)
                f_data["clip_embedding"] = clip_emb
                
                try:
                    pil_img = frame.to_image()
                except Exception:
                    pil_img = None
                caption = get_semantic_and_clip_caption(pil_img, frame, clip_emb, device)
                f_data["caption"] = caption

                feature_record = {
                    "frame_idx":            f_data["frame_idx"],
                    "timestamp":            f_data["timestamp"],
                    "luma_diff_energy":      f_data["luma_diff_energy"],
                    "motion_magnitude":     f_data.get("motion_magnitude", 0.0),
                    "luma_entropy":              f_data.get("luma_entropy", 0.0),
                    "refined_motion_tensor": np.asarray([
                        float(f_data.get("motion_magnitude", 0.0)),
                        float(f_data.get("divergence", 0.0)),
                        float(f_data.get("curl", 0.0)),
                        float(f_data.get("jacobian_frobenius", 0.0)),
                        float(f_data.get("hessian_max_eigenvalue", 0.0)),
                        float(f_data.get("motion_entropy", 0.0)),
                    ], dtype=np.float32),
                    "packet_size":          float(f_data.get("packet_size", 0.0)),
                    "codec_conf":           float(f_data.get("codec_conf", 0.5)),
                    "pict_type":            str(f_data.get("pict_type", "?")),
                    "is_peak":              bool(f_data.get("is_peak", False)),
                }
                score_record = {
                    "action_score":     f_data["action_score"],
                    "persistence_value": f_data["persistence_value"],
                }
                node_records.append((feature_record, score_record))
                enrichment_records.append((f_data["frame_idx"], [], clip_emb))
        container.close()

    # Pass 2: batch index into graph — single edge+pagerank recompute each
    graph_mode = getattr(config, "graph_mode", "flat")
    node_groups = None
    if graph_mode == "scene_sparse":
        groups = {}
        for f in frames_to_index:
            sid = f.get("scene_id", -1)
            if sid >= 0:
                groups.setdefault(sid, []).append(f["frame_idx"])
        node_groups = list(groups.values())

    graph.batch_add_frame_nodes(node_records, node_groups=node_groups)
    graph.batch_enrich_nodes(enrichment_records, node_groups=node_groups)

    # 3. Perform graph-aware retrieval
    if frames_to_index:
        query_action_score = max(f.get("action_score", 0.0) for f in frames_to_index)
    else:
        query_action_score = 0.5
    ranking_mode = getattr(config, "ranking_mode", "legacy")
    if ranking_mode == "ppr":
        retrieved_nodes = graph.retrieve_ppr(
            query_embedding,
            top_k=l2_retrieve_top_k,
            damping=getattr(config, "ppr_damping", 0.5),
            lambda_=getattr(config, "ppr_lambda", 0.5),
        )
    else:
        retrieved_nodes = graph.retrieve(
            query_embedding,
            query_action_score=query_action_score,
            top_k=l2_retrieve_top_k,
        )
    wrapper_l2_retrieve.last_graph_data = graph.export_graph_data(
        max_edges=getattr(config, "graph_export_max_edges", 5000)
    )
    
    # 4. Map returned AsphodelNode objects back to dictionaries expected by cache wrapper
    retrieved = []
    if retrieved_nodes:
        for node in retrieved_nodes:
            orig = frame_map.get(node.frame_idx, {})
            retrieved.append({
                "frame_idx": node.frame_idx,
                "timestamp": node.timestamp,
                "luma_diff_energy": node.luma_diff_energy,
                "action_score": node.action_score,
                "persistence_value": node.persistence_value,
                "is_peak": orig.get("is_peak", False),
                "clip_embedding": orig.get("clip_embedding", None),
                "luma_entropy": orig.get("luma_entropy", 0.0),
                "caption": orig.get("caption", None),
                "pagerank_score": node.pagerank_score,
                "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
                "retrieval_contributions": getattr(node, "retrieval_contributions", {}),
                "tier": getattr(node, "tier", None),
                "scene_id": getattr(node, "scene_id", None),
                "pict_type": getattr(node, "pict_type", orig.get("pict_type", "?")),
                "codec_conf": getattr(node, "codec_conf", 0.5),
            })
    
    if not retrieved:
        sorted_frames = sorted(
            frames_to_index,
            key=lambda x: (x.get("action_score", 0.0), x.get("luma_diff_energy", 0.0)),
            reverse=True
        )
        for f in sorted_frames[:l2_retrieve_top_k]:
            retrieved.append({
                "frame_idx": f["frame_idx"],
                "timestamp": f["timestamp"],
                "luma_diff_energy": f["luma_diff_energy"],
                "action_score": f.get("action_score", 0.0),
                "persistence_value": f.get("persistence_value", 0.0),
                "is_peak": f.get("is_peak", False),
                "clip_embedding": f.get("clip_embedding", None),
                "luma_entropy": f.get("luma_entropy", 0.0),
                "caption": f.get("caption", None),
                "pagerank_score": 0.0,
                "last_retrieval_score": 0.0,
                "retrieval_contributions": {},
                "tier": "L1_PEAK" if f.get("is_peak", False) else "L3_CANDIDATE",
                "scene_id": None,
                "pict_type": f.get("pict_type", "?"),
                "codec_conf": f.get("codec_conf", 0.5),
            })

    return retrieved


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


# --- Main Pipeline Runners ---

def run_pipeline(video_path: str | Path, query: str, verbose: bool = False, nms_window: int = 10, config: IRISConfig | None = None) -> dict:
    """
    Run the end-to-end IRIS pipeline using the new continuous action score
    and topological persistence-based peak detection alongside the existing tier path.
    """
    import time
    
    # 1. Load config parameters
    if config is None:
        try:
            from iris.iris_config import ConfigManager
            config = ConfigManager().get_config()
            if config is None:
                from iris.iris_config import IRISConfig
                config = IRISConfig()
        except Exception:
            from iris.iris_config import IRISConfig
            config = IRISConfig()

    # Enforce active backend diagnostics check
    aria.run_diagnostics()

    # 2. Parse video and extract raw frame features non-breakingly from H.264 stream
    # Returns (output_frames, stats, raw_records)
    # Uses _CHARON_CACHE to avoid re-decoding the same video when the same
    # decode config is used across multiple queries.
    import copy
    _charon_key = (
        str(video_path),
        float(config.candidate_thresh),
        float(config.salient_thresh),
        bool(getattr(config, "adaptive", True)),
        bool(getattr(config, "visual_debug_mode", False)),
    )
    t_start = time.time()
    if _charon_key not in _CHARON_CACHE:
        _CHARON_CACHE[_charon_key] = charon_v.parse_video(
            str(video_path),
            return_stats=True,
            return_raw=True,
            candidate_thresh=config.candidate_thresh,
            salient_thresh=config.salient_thresh,
            adaptive=getattr(config, "adaptive", True),
            visual_debug_mode=getattr(config, "visual_debug_mode", False)
        )
        print(f"[CACHE] Charon-V decoded {video_path} → cached under key {_charon_key[:2]}")
    else:
        print(f"[CACHE] Charon-V cache HIT for {Path(video_path).name} — skipping re-decode")
    cached_output_frames, stats, raw_records = _CHARON_CACHE[_charon_key]
    output_frames = copy.deepcopy(cached_output_frames)
    t_charon = time.time() - t_start

    # 3. Continuous action scoring & persistence peak detection
    t_action_start = time.time()
    action_score_config = ActionScoreConfig(
        luma_diff_weight=getattr(config, "luma_diff_weight", 0.5),
        motion_weight=getattr(config, "motion_weight", 0.3),
        luma_entropy_weight=getattr(config, "luma_entropy_weight", 0.2),
        peak_distance=getattr(config, "peak_distance", 5),
        peak_prominence=getattr(config, "peak_prominence", 0.05),
        persistence_threshold=getattr(config, "persistence_threshold", 0.4),
        max_prominence=getattr(config, "max_prominence", 0.5),
    )
    score_module = ActionScoreModule(config=action_score_config)
    score_records = score_module.score_all(raw_records)
    action_scores = {r["frame_idx"]: r for r in score_records}

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
        frame["luma_entropy"] = raw_map.get(frame_idx, {}).get("luma_entropy", 0.0)
        frame["pict_type"] = raw_map.get(frame_idx, {}).get("frame_type", frame.get("pict_type", "?"))
        frame["packet_size"] = raw_map.get(frame_idx, {}).get("packet_size", frame.get("packet_size", 0.0))
        frame["divergence"] = raw_map.get(frame_idx, {}).get("divergence", frame.get("divergence", 0.0))
        frame["curl"] = raw_map.get(frame_idx, {}).get("curl", frame.get("curl", 0.0))
        frame["jacobian_frobenius"] = raw_map.get(frame_idx, {}).get("jacobian_frobenius", frame.get("jacobian_frobenius", 0.0))
        frame["hessian_max_eigenvalue"] = raw_map.get(frame_idx, {}).get("hessian_max_eigenvalue", frame.get("hessian_max_eigenvalue", 0.0))
        frame["motion_entropy"] = raw_map.get(frame_idx, {}).get("motion_entropy", frame.get("motion_entropy", 0.0))
    t_action = time.time() - t_action_start

    # If visual_debug_mode is enabled, save annotated candidate frames
    if getattr(config, "visual_debug_mode", False):
        import os
        from PIL import ImageDraw
        debug_frames_dir = os.path.join(os.path.dirname(str(video_path)), "debug_frames")
        os.makedirs(debug_frames_dir, exist_ok=True)
        for f in output_frames:
            pil_img = f.get("pil_image")
            if pil_img is not None:
                annotated = pil_img.copy()
                draw = ImageDraw.Draw(annotated)
                text = f"Frame {f['frame_idx']} | Score: {f.get('action_score', 0.0):.4f}"
                draw.text((10, 10), text, fill=(255, 0, 0))
                out_name = f"frame_{f['frame_idx']:04d}.png"
                annotated.save(os.path.join(debug_frames_dir, out_name))

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
        for p in peak_indices:
            target_indices.update(range(max(0, p - 2), p + 3))
        frames_to_index = [f for f in output_frames if f["frame_idx"] in target_indices]
    else:  # hybrid
        frames_to_index = output_frames

    retrieved_frames = wrapper_l2_retrieve(
        video_path,
        query,
        frames_to_index,
        config=config
    )
    t_l2 = time.time() - t_l2_start

    # 5. Populate L1 active context cache with retrieved frame evidence
    t_elysium_start = time.time()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    t_elysium = time.time() - t_elysium_start
    
    # 6. Generate answer using ARIA LLM brain
    t_aria_start = time.time()
    context_text = cache_obj.as_context_text()
    raw_answer = aria.generate(prompt=query, context=context_text)
    t_aria = time.time() - t_aria_start

    # 7. Extract sentence-level claims from the raw answer
    # Strip markdown before splitting
    clean_answer = re.sub(r'\*\*.*?\*\*:?\s*', '', raw_answer)
    clean_answer = re.sub(r'^\s*[-*]\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'^\s*\d+\.\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'\n+', ' ', clean_answer).strip()
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
    claims = [s.strip() for s in raw_sentences if len(s.strip()) >= 12]

    # 8. Run claims through Cerberus-V NLI truth gate
    t_cerberus_start = time.time()
    max_score = max([f.get("action_score", 0.0) for f in retrieved_frames]) if retrieved_frames else 0.5
    is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked = wrapper_cerberus_gate(claims, cache_obj, max_score, config)
    t_cerberus = time.time() - t_cerberus_start

    # Generate final verified answer (verified claims only; raw_answer and breakdowns are also
    # returned in the result dict so nothing is silently dropped from reporting).
    if verified_claims:
        final_answer = " ".join(verified_claims)
    else:
        final_answer = "Insufficient verified evidence to answer this question."

    # Determine peak counts and skip statistics
    peak_count = len([f for f in output_frames if f.get("is_peak", False)])
    skipped_frames_ratio = float(stats["skipped"] / stats["total"]) if stats["total"] > 0 else 0.0
    storage_reduction_factor = float(stats["total"] / len(output_frames)) if len(output_frames) > 0 else 0.0

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
        "compression_ratio": skipped_frames_ratio,  # Keep for backward compatibility
        "skipped_frames_ratio": skipped_frames_ratio,
        "storage_reduction_factor": storage_reduction_factor,
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

    # Observability Logging: Write ARIA debug logs to logs/aria_debug/
    import json
    import os
    os.makedirs("logs/aria_debug", exist_ok=True)
    os.makedirs("logs/pipeline", exist_ok=True)
    
    timestamp_log = time.strftime("%Y%m%d_%H%M%S")
    import random
    rand_id = random.randint(100000, 999999)
    
    aria_log_file = f"logs/aria_debug/debug_{timestamp_log}_{rand_id}.json"
    aria_log_data = {
        "frames_sent_to_aria": [f.frame_idx for f in cache_obj.frames()],
        "semantic_captions": [
            f.caption.get("semantic_caption") if isinstance(f.caption, dict) else f.caption
            for f in cache_obj.frames()
        ],
        "prompt": query,
        "response": raw_answer
    }
    try:
        with open(aria_log_file, "w", encoding="utf-8") as f:
            json.dump(aria_log_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write ARIA debug log: {e}")

    # Build validation report payload
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

    pipeline_log_file = f"logs/pipeline/run_{timestamp_log}_{rand_id}.json"
    try:
        log_friendly_result = {k: v for k, v in result.items() if k != "validation_report"}
        log_friendly_result["validation_report"] = {
            "aria": {
                "backend": validation_report["aria"]["backend"],
                "model": validation_report["aria"]["model"],
                "success_rate": validation_report["aria"]["success_rate"],
                "caption_success_rate": validation_report["aria"]["caption_success_rate"]
            },
            "retrieval": validation_report["retrieval"],
            "elysium": validation_report["elysium"],
            "cerberus": validation_report["cerberus"]
        }
        with open(pipeline_log_file, "w", encoding="utf-8") as f:
            json.dump(log_friendly_result, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write pipeline run log: {e}")

    if verbose:
        result["debug_info"] = {
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
            "l2_graph": getattr(wrapper_l2_retrieve, "last_graph_data", None),
            
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
