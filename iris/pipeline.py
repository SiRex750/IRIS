"""
IRIS end-to-end pipeline harness.

Wires: charon_v ΓåÆ action_score ΓåÆ l1_elysium ΓåÆ l2_asphodel ΓåÆ aria ΓåÆ cerberus_v

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

_CHARON_CACHE: dict = {}

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

# ΓöÇΓöÇ CLIP availability flag ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
_CLIP_REVISION_LOADED = None

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

    # P2-04: Domain-neutral vocabulary — no dataset-specific terms.
    # The previous list contained "cartoon rabbit", "bunny's face", etc. which
    # were tailored to the Big Buck Bunny test video and produced systematically
    # wrong labels for any other content.
    vocabulary = [
        "a person walking or running outdoors",
        "a person talking or gesturing indoors",
        "a group of people gathered together",
        "a person cooking or preparing food",
        "a close-up of a person's face",
        "a car or vehicle moving on a road",
        "a sports or athletic activity",
        "a street or outdoor urban scene",
        "an indoor room or office environment",
        "a natural landscape with trees or grass",
        "a crowd or public gathering",
        "an animal or wildlife in its environment",
        "a computer screen or technical interface",
        "a scene showing physical action or movement",
        "a calm or static scene with minimal motion",
    ]

    model, _ = get_clip_model()
    if model is None:
        return "visual cues from the video"
        
    try:
        import clip
        import torch
        
        if _CLIP_TEXT_FEATURES is None or _CLIP_TEXT_FEATURES.shape[0] != len(vocabulary):
            # Recompute when cache is empty or vocabulary size has changed.
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


def get_semantic_and_clip_caption(pil_img, frame, clip_emb, device, config: Any = None,
                                   focus_hint: str | None = None) -> dict:
    clip_label = get_zero_shot_caption(clip_emb, device)

    # 1. Try VLM OpenAI vision captioning first (ARIA)
    caption_res = aria.generate_caption_for_frame(
        pil_img if pil_img is not None else frame, config=config, focus_hint=focus_hint,
    )
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


def get_clip_model(config: Any = None):
    """Load and cache the CLIP model globally using the configured revision."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_REVISION_LOADED
    revision = "ViT-B/32"
    if config is not None:
        revision = getattr(config, "clip_revision", "ViT-B/32") or "ViT-B/32"
    
    if _CLIP_MODEL is not None and _CLIP_REVISION_LOADED != revision:
        # Reloading due to change in revision to prevent mixed embeddings
        _CLIP_MODEL = None
        _CLIP_PREPROCESS = None
        
    if _CLIP_MODEL is None:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(revision, device=device)
            _CLIP_REVISION_LOADED = revision
        except Exception as e:
            print(f"Warning: Failed to load CLIP model with revision {revision}: {e}")
            _CLIP_MODEL = None
            _CLIP_PREPROCESS = None
            _CLIP_REVISION_LOADED = None
    return _CLIP_MODEL, _CLIP_PREPROCESS


def get_frame_clip_embedding(frame: av.video.frame.VideoFrame, device: str, config: Any = None) -> np.ndarray:
    """Convert PyAV frame to image and extract normalized CLIP feature embedding."""
    model, preprocess = get_clip_model(config)
    if model is None:
        raise ValueError("CLIP model not loaded; cannot extract frame embedding.")
    try:
        import torch
        img = frame.to_image()  # Returns PIL RGB Image
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            emb = image_features.cpu().numpy().flatten().astype(np.float32)
            if np.linalg.norm(emb) < 1e-6:
                raise ValueError("Generated frame embedding has near-zero norm.")
            return emb
    except Exception as e:
        raise ValueError(f"Failed to extract CLIP embedding for frame: {e}")


def get_clip_embedding_from_pil(pil_image, device: str, config: Any = None) -> np.ndarray:
    """Extract normalized CLIP embedding from a PIL image.

    Used by the PIL-cache fast path in wrapper_l2_retrieve to avoid re-opening
    the video file when Charon-V has already captured frame images during its
    2nd decode pass.
    """
    model, preprocess = get_clip_model(config)
    if model is None:
        raise ValueError("CLIP model not loaded; cannot extract PIL embedding.")
    if pil_image is None:
        raise ValueError("PIL image is None; cannot extract embedding.")
    try:
        import torch
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            emb = image_features.cpu().numpy().flatten().astype(np.float32)
            if np.linalg.norm(emb) < 1e-6:
                raise ValueError("Generated PIL embedding has near-zero norm.")
            return emb
    except Exception as e:
        raise ValueError(f"Failed to extract CLIP embedding from PIL image: {e}")


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
         CLIP embeddings are extracted directly from those images ΓÇö no 3rd video decode.
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
    if model is None:
        raise ValueError("CLIP model could not be loaded; cannot embed query.")
    try:
        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_features = model.encode_text(text_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.cpu().numpy().flatten().astype(np.float32)
            if np.linalg.norm(query_embedding) < 1e-6:
                raise ValueError("Generated query embedding has near-zero norm.")
    except Exception as e:
        raise ValueError(f"Failed to encode query text: {e}")

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
        # ΓöÇΓöÇ Fast path (no 3rd video decode) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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
        # ΓöÇΓöÇ Legacy path (full video decode) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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

    # Pass 2: batch index into graph ΓÇö single edge+pagerank recompute each
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
    Run the end-to-end IRIS pipeline using the canonical ingest -> query flow.
    """
    import os
    import time
    import random
    import json
    from iris.ingest import ingest
    from iris.query import query as query_fn
    
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

    aria.run_diagnostics()

    # 2. Call canonical ingest
    index = ingest(video_path, config, nms_window=nms_window)

    # 3. Call canonical query
    res = query_fn(query, index, config)
    
    # 4. Write run log if needed for backward compatibility
    timestamp_log = time.strftime("%Y%m%d-%H%M%S")
    rand_id = random.randint(1000, 9999)
    os.makedirs("logs/pipeline", exist_ok=True)
    pipeline_log_file = f"logs/pipeline/run_{timestamp_log}_{rand_id}.json"
    try:
        log_friendly_result = {k: v for k, v in res.items() if k != "validation_report"}
        with open(pipeline_log_file, "w", encoding="utf-8") as f:
            json.dump(log_friendly_result, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write pipeline run log: {e}")
        
    if verbose:
        res["debug_info"] = {
            "context_text": res.get("context_text", ""),
            "unverifiable_claims": res.get("unverifiable_claims", []),
            "retrieved_frames": [{"frame_idx": idx} for idx in res.get("retrieved_frame_idxs", [])],
        }

    return res


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
