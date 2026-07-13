"""
ARIA — LLM interface abstraction for IRIS.

Single entry point for all LLM calls in the pipeline.

Captioning:  BLIPCaptioner  — local Salesforce/blip-image-captioning-base
Text gen:    LlamaBackend   — local Ollama at localhost:11434 (default)
             OpenAIBackend  — OpenAI API (opt-in via set_backend)

No other file in IRIS should import openai, transformers, or call
any LLM/VLM API directly — all calls go through here.

Owner: Track B
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from iris.claim_contract import ANSWER_CLAIMS_WIRE_SCHEMA


@dataclass
class CaptionResult:
    success: bool
    caption: str | None
    error: str | None = None


class CaptionGenerationError(Exception):
    """Exception raised when caption generation fails."""
    pass


# ---------------------------------------------------------------------------
# Vision captioning — BLIP
# ---------------------------------------------------------------------------

class BLIPCaptioner:
    """Local image captioner using Salesforce BLIP (via HuggingFace transformers)."""

    DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def caption(self, pil_image) -> str:
        import torch
        self._load()
        inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(out[0], skip_special_tokens=True)


class MoondreamCaptioner:
    def __init__(self) -> None:
        self.model_name = 'vikhyatk/moondream2'
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained('vikhyatk/moondream2', trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda')
        self._device = 'cuda'

    def caption(self, pil_image) -> str:
        import torch
        self._load()
        if str(self._model.device) == 'cpu':
            self._model = self._model.to(self._device)
        enc = self._model.encode_image(pil_image)
        return self._model.answer_question(enc, 'Describe only what is visually present in this single image. State objects, people, colors, and positions. Do not describe motion or changes.', self._tokenizer).strip()
\n\nclass MiniCPMCaptioner:
    """Local image captioner using MiniCPM-V4.6 via Ollama."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "minicpm-v4.6"
        self.endpoint = "http://localhost:11434"
        self.prompt = (
            "List everything visible in this image: every person, object, vehicle, "
            "and action. One short sentence per item. Only what is clearly visible."
        )
        self.num_predict = 250

    def caption(self, pil_image) -> str:
        import io
        import base64
        import requests
        import re

        # 1. Convert image to base64 JPEG
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # 2. Query Ollama vision endpoint
        payload = {
            "model": self.model_name,
            "prompt": self.prompt,
            "images": [image_b64],
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": self.num_predict
            },
            "think": False,
            "stream": False,
        }
        resp = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        raw = data["response"].strip()

        # Check and strip <think>...</think> blocks if present
        think_tag_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        cleaned = think_tag_re.sub("", raw).strip()

        return cleaned


_ACTIVE_CAPTIONER: MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner | None = None

def get_captioner() -> MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner:
    global _ACTIVE_CAPTIONER
    if _ACTIVE_CAPTIONER is None:
        try:
            from iris.iris_config import ConfigManager
            config = ConfigManager().get_config()
            backend_type = getattr(config, 'captioner_backend', 'minicpm')
        except Exception:
            backend_type = 'minicpm'

        if backend_type == 'moondream':
            _ACTIVE_CAPTIONER = MoondreamCaptioner()
        elif backend_type == 'blip':
            _ACTIVE_CAPTIONER = BLIPCaptioner()
        else:
            _ACTIVE_CAPTIONER = MiniCPMCaptioner()
    return _ACTIVE_CAPTIONER

def set_captioner(captioner: MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner) -> None:
    global _ACTIVE_CAPTIONER
    _ACTIVE_CAPTIONER = captioner

def unload_captioner() -> None:
    global _ACTIVE_CAPTIONER
    import torch
    if _ACTIVE_CAPTIONER is not None and hasattr(_ACTIVE_CAPTIONER, '_model') and _ACTIVE_CAPTIONER._model is not None:
        _ACTIVE_CAPTIONER._model.to('cpu')
        torch.cuda.empty_cache()
\n\n
