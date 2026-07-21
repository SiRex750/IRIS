"""CLIP + BLIP captioning/embedding helpers for the Phase 3 ingest spine.

Lifted VERBATIM from pipeline.py so the new ingest path is self-contained
and pipeline.py can be deleted with a clean `rm` after parity. Do not edit
these bodies here — any behavior change happens in a separate, later task.
"""
from __future__ import annotations

import numpy as np
import av
import iris.aria as aria

# Module-global model caches (lifted from pipeline.py)
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

