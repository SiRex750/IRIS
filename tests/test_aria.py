"""
Unit tests for ARIA LLM interface.

Owner: Track B
"""
from __future__ import annotations
import pytest
import aria
from aria import LLMBackend, set_backend, get_backend, generate


class MockBackend(LLMBackend):
    def __init__(self) -> None:
        self.generate_called = False
        self.last_prompt = None
        self.last_context = None
        self.last_model = None
        self.response = "mocked answer"

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        self.generate_called = True
        self.last_prompt = prompt
        self.last_context = context
        self.last_model = model
        return self.response


def test_backend_swap():
    original = get_backend()
    mock_be = MockBackend()
    try:
        set_backend(mock_be)
        assert get_backend() is mock_be
    finally:
        set_backend(original)


def test_generate_delegation():
    original = get_backend()
    mock_be = MockBackend()
    try:
        set_backend(mock_be)
        ans = generate(prompt="test prompt", context="test context", model="test-model")
        assert ans == "mocked answer"
        assert mock_be.generate_called is True
        assert mock_be.last_prompt == "test prompt"
        assert mock_be.last_context == "test context"
        assert mock_be.last_model == "test-model"
    finally:
        set_backend(original)
