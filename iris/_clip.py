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

