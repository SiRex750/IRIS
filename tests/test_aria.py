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
        # Check diagnostics — backend may be LlamaBackend, OpenAIBackend, or a mock
        diag = aria.run_diagnostics()
        assert diag["backend"] in ("MockLLMBackend", "MockBackend", "OpenAIBackend", "LlamaBackend")
        assert diag["captioner"] in ("BLIPCaptioner", "MockCaptioner", "MiniCPMCaptioner")

        # Enforce key missing failure only when OpenAIBackend is active
        os.environ["OPENAI_API_KEY"] = ""
        backend_class = aria.get_backend().__class__.__name__
        if backend_class == "OpenAIBackend":
            with pytest.raises(RuntimeError):
                aria.run_diagnostics()

        # Test generate_caption_for_frame fallback: passing None frame should fail gracefully
        res = aria.generate_caption_for_frame(None)
        assert isinstance(res, CaptionResult)
        assert res.success is False
        assert res.caption == "[CAPTION_FAILED]"
        assert res.error is not None and len(res.error) > 0

        # Verify failure was logged
        failures = aria.get_caption_failures()
        assert len(failures) > 0
        assert all(f["error"] for f in failures)
    finally:
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)


def test_llama_server_backend_outgoing_request():
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaServerBackend
    from iris.claim_contract import ANSWER_CLAIMS_WIRE_SCHEMA

    # Instantiate LlamaServerBackend
    backend = LlamaServerBackend(
        endpoint="http://localhost:8080/v1",
        text_model="granite4:micro",
        timeout=600.0
    )

    # We mock the OpenAI client inside backend
    mock_completions = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked wire response"
    mock_completions.create.return_value = mock_response

    # Directly assign mock client to backend._client to intercept instantiation
    mock_client = MagicMock()
    mock_client.chat.completions = mock_completions
    backend._client = mock_client
    
    # Call generate with schema_format=True
    res = backend.generate(
        prompt="test prompt",
        context="test context",
        schema_format=True
    )
    
    assert res == "mocked wire response"
    mock_completions.create.assert_called_once()
    kwargs = mock_completions.create.call_args[1]
    
    assert kwargs["temperature"] == 0.0
    assert kwargs["extra_body"] == {"cache_prompt": False}
    assert kwargs["model"] == "granite4:micro"
    assert kwargs["timeout"] == 600.0
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_claims",
            "schema": ANSWER_CLAIMS_WIRE_SCHEMA,
            "strict": True
        }
    }


def test_minicpm_captioner_and_mocked_ollama():
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import get_captioner, MiniCPMCaptioner

    # Reset active captioner to ensure clean default initialization
    aria._ACTIVE_CAPTIONER = None

    # Verify that get_captioner() returns MiniCPMCaptioner by default
    captioner = get_captioner()
    assert isinstance(captioner, MiniCPMCaptioner)
    assert captioner.model_name == "minicpm-v4.6"

    # Test custom mock Ollama response
    mock_pil = MagicMock()
    mock_convert = MagicMock()
    mock_pil.convert.return_value = mock_convert

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "mocked caption text",
        "done_reason": "stop"
    }
    mock_response.status_code = 200

    with patch("requests.post", return_value=mock_response) as mock_post:
        cap_text = captioner.caption(mock_pil)
        assert cap_text == "mocked caption text"
        mock_post.assert_called_once()
        
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/generate"
        payload = kwargs["json"]
        assert payload["model"] == "minicpm-v4.6"
        assert payload["prompt"] == (
            "List everything visible in this image: every person, object, vehicle, "
            "and action. One short sentence per item. Only what is clearly visible."
        )
        assert payload["options"] == {"temperature": 0, "seed": 42, "num_predict": 250}
        assert payload["think"] is False
        assert payload["stream"] is False


