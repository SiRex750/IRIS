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
        # Ensure model is on GPU before inference (re-to if unloaded)
        self._model = self._model.to(self._device)
        inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(out[0], skip_special_tokens=True)


class MoondreamCaptioner:
    """Local image captioner using Moondream2 (via HuggingFace transformers)."""

    def __init__(self) -> None:
        self.model_name = "vikhyatk/moondream2"
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self._device = "cuda"

    def caption(self, pil_image) -> str:
        import torch
        self._load()
        if str(self._model.device) == "cpu":
            self._model = self._model.to(self._device)
        enc = self._model.encode_image(pil_image)
        return self._model.answer_question(
            enc,
            "Describe only what is visually present in this single image. State objects, people, colors, and positions. Do not describe motion or changes.",
            self._tokenizer
        ).strip()


_ACTIVE_CAPTIONER: BLIPCaptioner | MoondreamCaptioner | None = None


def get_captioner() -> BLIPCaptioner | MoondreamCaptioner:
    """Returns the globally configured captioner (lazy-initialised)."""
    global _ACTIVE_CAPTIONER
    if _ACTIVE_CAPTIONER is None:
        try:
            from iris.iris_config import ConfigManager
            config = ConfigManager().get_config()
            backend_type = getattr(config, "captioner_backend", "blip")
        except Exception:
            backend_type = "blip"

        if backend_type == "moondream":
            _ACTIVE_CAPTIONER = MoondreamCaptioner()
        else:
            _ACTIVE_CAPTIONER = BLIPCaptioner()
    return _ACTIVE_CAPTIONER


def set_captioner(captioner: BLIPCaptioner | MoondreamCaptioner) -> None:
    """Override the active captioner (e.g. for testing)."""
    global _ACTIVE_CAPTIONER
    _ACTIVE_CAPTIONER = captioner

def unload_captioner() -> None:
    """Explicitly move the captioner model off GPU to free VRAM for LLM inference."""
    global _ACTIVE_CAPTIONER
    import torch
    if _ACTIVE_CAPTIONER is not None and hasattr(_ACTIVE_CAPTIONER, "_model") and _ACTIVE_CAPTIONER._model is not None:
        _ACTIVE_CAPTIONER._model.to("cpu")
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Text generation — LLM backends
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are ARIA, a video understanding assistant.\n\n"
    "Use only the provided frame evidence and retrieval context.\n\n"
    "Answer in clear natural language.\n\n"
    "When describing events, reference timestamps and supporting frames.\n\n"
    "If evidence is insufficient, explicitly say so.\n\n"
    "Do not invent events that are not supported by the provided context.\n\n"
    "Prefer concise but human-readable explanations over raw metadata.\n\n"
)


class LLMBackend:
    """Abstract base class for LLM backends."""
    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        raise NotImplementedError("LLM backend must implement generate()")


class LlamaBackend(LLMBackend):
    """Local LLM via Ollama (OpenAI-compatible endpoint at localhost:11434)."""

    DEFAULT_TEXT_MODEL = "llama3.2:3b"

    def __init__(self, endpoint: str = "http://localhost:11434/v1",
                 text_model: str | None = None) -> None:
        self.endpoint = endpoint
        self.text_model = text_model or self.DEFAULT_TEXT_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key="ollama")
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        model_name = model or self.text_model
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        # Added keep_alive=0 to explicitly drop Llama from Ollama VRAM after answering
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            extra_body={"keep_alive": 0},
        )
        return response.choices[0].message.content


class OpenAIBackend(LLMBackend):
    """OpenAI API implementation."""

    DEFAULT_TEXT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.text_model = self.DEFAULT_TEXT_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set.\n"
                    "To run IRIS dynamically, please either:\n"
                    "  1. Set the OPENAI_API_KEY environment variable.\n"
                    "  2. Use the local LlamaBackend via Ollama (default)."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        model_name = model or self.text_model
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content


_ACTIVE_BACKEND: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Returns the globally configured active LLM backend."""
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = LlamaBackend()
    return _ACTIVE_BACKEND


def set_backend(backend: LLMBackend) -> None:
    """Explicitly override the active LLM backend (e.g. for testing)."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def generate(prompt: str, context: str, model: str | None = None) -> str:
    """
    Generate a response from the active LLM backend.

    Args:
        prompt:  the user query or instruction
        context: formatted context string from L1 Elysium (as_context_text())
        model:   model identifier; if None, uses the backend's default

    Returns:
        Raw string response from the model
    """
    return get_backend().generate(prompt, context, model=model)


# ---------------------------------------------------------------------------
# Caption failures log
# ---------------------------------------------------------------------------

_CAPTION_FAILURES: list = []


def get_caption_failures() -> list:
    """Return all stored captioning failures."""
    return _CAPTION_FAILURES


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics() -> dict:
    """Run startup diagnostics; raises RuntimeError if a required backend is misconfigured."""
    import json

    backend = get_backend()
    backend_class = backend.__class__.__name__

    api_key = os.environ.get("OPENAI_API_KEY")
    api_key_present = bool(api_key and api_key.strip())

    if hasattr(backend, "text_model"):
        model = backend.text_model
    elif backend_class in ("MockLLMBackend", "MockBackend"):
        model = "mock-model"
    else:
        model = "unknown"

    captioner = get_captioner()
    captioner_class = captioner.__class__.__name__
    captioner_model = getattr(captioner, "model_name", "unknown")

    diag = {
        "backend": backend_class,
        "model": model,
        "captioner": captioner_class,
        "captioner_model": captioner_model,
        "api_key_present": api_key_present,
    }

    print(f"DIAGNOSTICS: {json.dumps(diag)}")

    if backend_class == "OpenAIBackend" and not api_key_present:
        raise RuntimeError("OpenAIBackend is active but OPENAI_API_KEY is not set.")

    return diag


# ---------------------------------------------------------------------------
# Frame captioning entry point
# ---------------------------------------------------------------------------

def generate_caption_for_frame(frame, frame_idx: int | None = None) -> CaptionResult:
    """Generate a semantic caption for a PyAV VideoFrame or PIL Image using BLIP."""
    import time

    t_start = time.time()

    if frame is None:
        err_msg = "No frame provided for captioning."
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": 0.0, "error": err_msg})
        return result

    # 1. Extract PIL image from PyAV frame if needed
    try:
        img = frame.to_image() if hasattr(frame, "to_image") else frame
    except Exception as e:
        err_msg = f"Failed to convert frame to image for captioning: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.time() - t_start, "error": err_msg})
        return result

    # 2. Caption with active captioner
    try:
        captioner = get_captioner()
        caption = captioner.caption(img)
        return CaptionResult(success=True, caption=caption)
    except Exception as e:
        err_msg = f"Captioning failed: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.time() - t_start, "error": err_msg})
        return result
