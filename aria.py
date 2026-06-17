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
    # TODO: implement — OpenAI backend first, Llama swap later
    pass
