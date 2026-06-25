"""
Unit tests for ARIA LLM interface.

Owner: Track B
"""
from __future__ import annotations
import pytest
import iris.aria as aria
from iris.aria import LLMBackend, set_backend, get_backend, generate


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


def test_captioning_result_and_diagnostics():
    import os
    import pytest
    import iris.aria as aria
    from iris.aria import CaptionResult, CaptionGenerationError
    
    # Save original key
    orig_key = os.environ.get("OPENAI_API_KEY")
    try:
        # Check diagnostics with fake/temp environment key
        os.environ["OPENAI_API_KEY"] = "fake-key"
        diag = aria.run_diagnostics()
        assert diag["backend"] in ("MockLLMBackend", "MockBackend", "OpenAIBackend")
        
        # Enforce key missing failure
        os.environ["OPENAI_API_KEY"] = ""
        # Diagnostics should throw RuntimeError if OpenAIBackend is active and key is empty
        backend_class = aria.get_backend().__class__.__name__
        if backend_class == "OpenAIBackend":
            with pytest.raises(RuntimeError):
                aria.run_diagnostics()
                
        # Test generate_caption_for_frame Mock/Fallback behavior
        res = aria.generate_caption_for_frame(None)
        assert isinstance(res, CaptionResult)
        assert res.success is False
        assert res.caption == "[CAPTION_FAILED]"
        assert "OPENAI_API_KEY" in res.error
        
        # Verify failure was logged
        failures = aria.get_caption_failures()
        assert len(failures) > 0
        assert any("OPENAI_API_KEY" in f["error"] for f in failures)
    finally:
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
