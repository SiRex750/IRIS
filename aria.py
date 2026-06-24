"""
ARIA — LLM interface abstraction for IRIS.

Single entry point for all LLM calls in the pipeline.
Backend is swappable: OpenAI during development,
Llama 3.2 3B via Ollama/llama.cpp in production.

No other file in IRIS should import openai or call
any LLM API directly — all calls go through here.

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


class LLMBackend:
    """Abstract base class for LLM backends."""
    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        raise NotImplementedError("LLM backend must implement generate()")


class OpenAIBackend(LLMBackend):
    """OpenAI API implementation."""
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        # Default model if not specified
        model_name = model or "gpt-4o-mini"
        
        system_content = (
            "You are ARIA, a video understanding assistant.\n\n"
            "Use only the provided frame evidence and retrieval context.\n\n"
            "Answer in clear natural language.\n\n"
            "When describing events, reference timestamps and supporting frames.\n\n"
            "If evidence is insufficient, explicitly say so.\n\n"
            "Do not invent events that are not supported by the provided context.\n\n"
            "Prefer concise but human-readable explanations over raw metadata.\n\n"
            f"Provided Frame Evidence and Retrieval Context:\n{context}"
        )
        
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content


class LlamaBackend(LLMBackend):
    """Local Llama 3.2 3B via Ollama/llama.cpp backend (for future swap)."""
    def __init__(self, endpoint: str = "http://localhost:11434/v1") -> None:
        self.endpoint = endpoint

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        # TODO: Implement local Llama 3.2 3B calling Ollama/llama.cpp client here
        raise NotImplementedError("LlamaBackend is not yet implemented.")


# Active backend instance (cached globally)
_ACTIVE_BACKEND: LLMBackend | None = None


def get_backend() -> LLMBackend:
    """Returns the globally configured active LLM backend."""
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        # Default to OpenAI backend
        _ACTIVE_BACKEND = OpenAIBackend()
    return _ACTIVE_BACKEND


def set_backend(backend: LLMBackend) -> None:
    """Explicitly override the active LLM backend (e.g. for testing)."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def generate(prompt: str, context: str, model: str = "gpt-4o-mini") -> str:
    """
    Generate a response from the active LLM backend.

    Args:
        prompt: the user query or instruction
        context: formatted context string from L1 Elysium (as_context_text())
        model: model identifier, ignored when using local backend

    Returns:
        Raw string response from the model
    """
    backend = get_backend()
    return backend.generate(prompt, context, model=model)


_CAPTION_FAILURES = []


def get_caption_failures() -> list:
    """Return all stored captioning failures."""
    return _CAPTION_FAILURES


def run_diagnostics() -> dict:
    """Run startup diagnostics and raise RuntimeError if requested backend is unavailable."""
    import json
    backend = get_backend()
    backend_class = backend.__class__.__name__
    
    api_key = os.environ.get("OPENAI_API_KEY")
    api_key_present = api_key is not None and len(api_key.strip()) > 0
    
    model = "gpt-4o-mini"
    if backend_class in ("MockLLMBackend", "MockBackend"):
        model = "mock-model"
        
    diag = {
        "backend": backend_class,
        "model": model,
        "api_key_present": api_key_present
    }
    
    print(f"DIAGNOSTICS: {json.dumps(diag)}")
    
    if backend_class == "OpenAIBackend" and not api_key_present:
        raise RuntimeError("OpenAIBackend is active but OPENAI_API_KEY is not set.")
        
    return diag


def generate_caption_for_frame(frame, frame_idx: int | None = None) -> CaptionResult:
    """Generate a semantic caption for the given PyAV VideoFrame using the active backend."""
    import base64
    import io
    import time

    t_start = time.time()
    model_name = "gpt-4o-mini"
    
    # 1. Fallback / Mock check
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "mock":
        err_msg = "OPENAI_API_KEY is not set or is 'mock'."
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({
            "frame_idx": frame_idx,
            "latency": time.time() - t_start,
            "model": model_name,
            "error": err_msg
        })
        return result

    # 2. Extract image from PyAV frame and encode as base64 JPEG
    try:
        img = frame.to_image()  # PIL Image
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        err_msg = f"Failed to convert frame to image for captioning: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({
            "frame_idx": frame_idx,
            "latency": time.time() - t_start,
            "model": model_name,
            "error": err_msg
        })
        return result

    # 3. Call OpenAI vision API
    try:
        backend = get_backend()
        if not isinstance(backend, OpenAIBackend):
            err_msg = f"Active backend {backend.__class__.__name__} does not support vision/captioning."
            result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
            _CAPTION_FAILURES.append({
                "frame_idx": frame_idx,
                "latency": time.time() - t_start,
                "model": "mock-model",
                "error": err_msg
            })
            return result
        
        # Access client
        client = backend.client
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this video frame briefly in one clear sentence. Focus on the main subjects and actions. Do not use conversational filler or meta-references."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=60,
            temperature=0.2,
        )
        caption = response.choices[0].message.content.strip()
        return CaptionResult(success=True, caption=caption)
    except Exception as e:
        err_msg = f"OpenAI captioning API call failed: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({
            "frame_idx": frame_idx,
            "latency": time.time() - t_start,
            "model": model_name,
            "error": err_msg
        })
        return result
