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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Intelligent Residual Indexing System (IRIS) LLM Brain.\n"
                    f"Use the following hierarchical video memory context to answer the user's query:\n\n{context}"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
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
