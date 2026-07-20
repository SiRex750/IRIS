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

    def generate(self, prompt: str, context: str, model: str | None = None, *args, **kwargs) -> str:
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
        assert diag["backend"] in ("MockLLMBackend", "MockBackend", "OpenAIBackend", "LlamaBackend", "LlamaServerBackend")
        assert diag["captioner"] in ("BLIPCaptioner", "MockCaptioner", "MiniCPMCaptioner", "MoondreamCaptioner")

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


def test_llama_server_http_payload_enforcement():
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaServerBackend
    from iris.claim_contract import ANSWER_CLAIMS_WIRE_SCHEMA

    backend = LlamaServerBackend(
        endpoint="http://127.0.0.1:8091/v1",
        text_model="granite4:micro",
        timeout=600.0
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "mocked http response"
                }
            }
        ]
    }
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = backend.generate(
            prompt="test prompt",
            context="test context",
            schema_format=True
        )

        assert res == "mocked http response"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:8091/v1/chat/completions"
        
        payload = kwargs["json"]
        assert payload["model"] == "granite4:micro"
        assert payload["temperature"] == 0
        assert payload["cache_prompt"] is False
        assert payload["max_tokens"] == 1024
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_claims",
                "schema": ANSWER_CLAIMS_WIRE_SCHEMA,
                "strict": True
            }
        }


def test_llama_backend_forwards_seed_default_chat_path():
    """temperature=0.0 alone is not sufficient for determinism (see P1 smoke
    test finding: 1/12 repeated identical questions diverged with no seed
    forwarded). This asserts the default (non-schema, non-response_format)
    chat.completions.create() path actually carries a seed kwarg."""
    from unittest.mock import MagicMock
    from iris.aria import LlamaBackend

    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c", seed=42)

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["seed"] == 42
    assert kwargs["temperature"] == 0.0


def test_llama_backend_forwards_seed_native_schema_path():
    """schema_format path uses raw requests.post to Ollama's native /api/chat
    -- seed must land in options.seed (Ollama's native seed location), not a
    top-level "seed" key."""
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaBackend

    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "mocked wire response"}}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = backend.generate(prompt="q", context="c", schema_format=True, seed=42)
        assert res == "mocked wire response"
        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["seed"] == 42
        assert payload["options"]["temperature"] == 0.0


def test_llama_backend_forwards_keep_alive_default_chat_path():
    """Item 1 follow-up: seed alone was NOT sufficient for determinism -- a
    real smoke test found a request served against an already-"warm" Ollama
    model handle produced different tokens than one served against a
    freshly-(re)loaded handle, even with seed pinned (and even with
    num_thread=1, ruling out floating-point reduction-order nondeterminism).
    keep_alive=0 forces a fresh reload per request, which was empirically
    confirmed reproducible. Assert it's forwarded via extra_body (the OpenAI
    client has no top-level keep_alive parameter -- Ollama reads it as an
    extension field)."""
    from unittest.mock import MagicMock
    from iris.aria import LlamaBackend

    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c", seed=42, keep_alive=0)

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["extra_body"] == {"keep_alive": 0}


def test_llama_backend_forwards_keep_alive_native_schema_path():
    """Native /api/chat path: keep_alive must be a TOP-LEVEL payload field,
    not nested under options (unlike seed/temperature)."""
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaBackend

    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "mocked wire response"}}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        backend.generate(prompt="q", context="c", schema_format=True, seed=42, keep_alive=0)
        payload = mock_post.call_args[1]["json"]
        assert payload["keep_alive"] == 0
        assert "keep_alive" not in payload["options"]


def test_llama_server_backend_forwards_seed_openai_client_path():
    from unittest.mock import MagicMock
    from iris.aria import LlamaServerBackend

    backend = LlamaServerBackend(endpoint="http://localhost:8080/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c", seed=42)

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["seed"] == 42


def test_llama_server_backend_forwards_seed_native_http_payload():
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaServerBackend

    backend = LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "mocked http response"}}]}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        backend.generate(prompt="q", context="c", schema_format=True, seed=42)
        payload = mock_post.call_args[1]["json"]
        assert payload["seed"] == 42


def test_llama_server_backend_forwards_seed_completion_fallback():
    """Exercise the native /completion fallback (schema_format=False, so the
    OpenAI-compat client raises and the code falls back to raw requests)."""
    from unittest.mock import MagicMock, patch
    from iris.aria import LlamaServerBackend

    backend = LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")
    backend._client = mock_client

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": "mocked completion response"}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = backend.generate(prompt="q", context="c", seed=42)
        assert res == "mocked completion response"
        payload = mock_post.call_args[1]["json"]
        assert payload["seed"] == 42


def test_generate_module_function_defaults_seed_from_config():
    """iris.aria.generate()/generate_v2() must read answerer_seed off the
    config and forward it, even when the caller doesn't pass seed explicitly
    (this is how iris.query.query() invokes it)."""
    from unittest.mock import MagicMock
    import iris.aria as aria
    from iris.aria import LLMBackend

    class RecordingBackend(LLMBackend):
        def __init__(self):
            self.last_seed = "UNSET"
            self.last_keep_alive = "UNSET"

        def generate(self, prompt, context, model=None, system_prompt=None,
                     response_format=None, max_tokens=None, schema_format=False,
                     seed=None, keep_alive=None):
            self.last_seed = seed
            self.last_keep_alive = keep_alive
            return "ok"

    backend = RecordingBackend()
    original = aria.get_backend()
    try:
        aria.set_backend(backend)
        cfg = MagicMock()
        cfg.answerer_seed = 99
        cfg.answerer_keep_alive = 5
        aria.generate(prompt="q", context="c", config=cfg)
        assert backend.last_seed == 99
        assert backend.last_keep_alive == 5

        aria.generate_v2(prompt="q", context="c", config=cfg)
        assert backend.last_seed == 99
        assert backend.last_keep_alive == 5

        # No config at all -> falls back to the documented defaults (seed=42, keep_alive=0)
        aria.generate(prompt="q", context="c", config=None)
        assert backend.last_seed == 42
        assert backend.last_keep_alive == 0
    finally:
        aria.set_backend(original)


def test_minicpm_captioner_and_mocked_ollama():
    """Reverts commit 7b38c30 ("test: mock config in
    test_minicpm_captioner_and_mocked_ollama to reflect moondream default"),
    which mocked captioner_backend="minicpm" here specifically because
    IRISConfig.captioner_backend had drifted to "moondream" at the time.

    A separate infra-seating pass has since confirmed and live-verified that
    the actual seated production captioner is minicpm-v4.6, and
    IRISConfig.captioner_backend's default has been fixed back to "minicpm"
    (iris/iris_config.py) with configs/default_iris_config.json now
    explicitly stating it too. This test is restored to its original intent:
    calling get_captioner() with ZERO config override must resolve to
    MiniCPMCaptioner, because that is what a zero-override production config
    actually resolves to now -- this is the test that would have caught the
    moondream/minicpm mismatch before it needed a separate diagnosis.
    """
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import get_captioner, MiniCPMCaptioner

    # Reset active captioner to ensure clean default initialization
    aria._ACTIVE_CAPTIONER = None

    # Verify that get_captioner() returns MiniCPMCaptioner BY DEFAULT (no
    # config override at all -- this is the real production zero-override path).
    captioner = get_captioner()
    assert isinstance(captioner, MiniCPMCaptioner)
    assert captioner.model_name in ("minicpm-v4.6", "minicpm-v")

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
        assert payload["model"] in ("minicpm-v4.6", "minicpm-v")
        assert payload["prompt"] == (
            "List everything visible in this image: every person, object, vehicle, "
            "and action. One short sentence per item. Only what is clearly visible."
        )
        assert payload["options"] == {"temperature": 0, "seed": 42, "num_predict": 400}
        assert payload["think"] is False
        assert payload["stream"] is False


def test_minicpm_captioner_appends_focus_hint():
    """Item 3: the captioner was previously blind to the question -- assert
    the question text actually reaches the captioner's request payload."""
    from unittest.mock import MagicMock, patch
    from iris.aria import MiniCPMCaptioner

    captioner = MiniCPMCaptioner(model_name="minicpm-v")
    mock_pil = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "a boy in red running", "done_reason": "stop"}
    mock_response.status_code = 200

    with patch("requests.post", return_value=mock_response) as mock_post:
        captioner.caption(mock_pil, focus_hint="Pay attention to: why did the boy in red run?")
        payload = mock_post.call_args[1]["json"]
        assert "Pay attention to: why did the boy in red run?" in payload["prompt"]
        # base instruction is preserved, not replaced
        assert payload["prompt"].startswith("List everything visible in this image")

    # No focus_hint -> exact original generic prompt, unchanged (fallback path)
    with patch("requests.post", return_value=mock_response) as mock_post:
        captioner.caption(mock_pil)
        payload = mock_post.call_args[1]["json"]
        assert payload["prompt"] == captioner.prompt


def test_minicpm_no_truncation_single_call_and_stats():
    """Item 5: the common case (no truncation) must make exactly one HTTP
    call and not increment either truncation counter."""
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import MiniCPMCaptioner

    aria.reset_minicpm_truncation_stats()
    captioner = MiniCPMCaptioner(model_name="minicpm-v")
    mock_pil = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "a complete caption", "done_reason": "stop"}
    mock_response.status_code = 200

    with patch("requests.post", return_value=mock_response) as mock_post:
        cap = captioner.caption(mock_pil)
        assert cap == "a complete caption"
        assert mock_post.call_count == 1
        assert mock_post.call_args[1]["json"]["options"]["num_predict"] == captioner.num_predict

    stats = aria.get_minicpm_truncation_stats()
    assert stats["total_calls"] == 1
    assert stats["truncated_first_attempt"] == 0
    assert stats["truncated_after_retry"] == 0


def test_minicpm_truncation_retries_once_with_higher_budget_then_succeeds():
    """Truncated on the first attempt (done_reason=='length'), succeeds on a
    retry at a higher num_predict -- must return the retry's content, not ""."""
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import MiniCPMCaptioner

    aria.reset_minicpm_truncation_stats()
    captioner = MiniCPMCaptioner(model_name="minicpm-v")
    mock_pil = MagicMock()

    truncated_response = MagicMock()
    truncated_response.json.return_value = {"response": "a partial cap", "done_reason": "length"}
    truncated_response.status_code = 200

    complete_response = MagicMock()
    complete_response.json.return_value = {"response": "a complete caption after retry", "done_reason": "stop"}
    complete_response.status_code = 200

    with patch("requests.post", side_effect=[truncated_response, complete_response]) as mock_post:
        cap = captioner.caption(mock_pil)
        assert cap == "a complete caption after retry"
        assert mock_post.call_count == 2
        first_num_predict = mock_post.call_args_list[0][1]["json"]["options"]["num_predict"]
        second_num_predict = mock_post.call_args_list[1][1]["json"]["options"]["num_predict"]
        assert first_num_predict == captioner.num_predict
        assert second_num_predict == captioner.retry_num_predict
        assert second_num_predict > first_num_predict

    stats = aria.get_minicpm_truncation_stats()
    assert stats["total_calls"] == 1
    assert stats["truncated_first_attempt"] == 1
    assert stats["truncated_after_retry"] == 0


def test_minicpm_truncation_persists_after_retry_returns_empty_and_logs():
    """Truncated on both attempts -- returns "" (unchanged external contract)
    but the truncation is now visible via get_minicpm_truncation_stats(),
    unlike before where it left no trace."""
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import MiniCPMCaptioner

    aria.reset_minicpm_truncation_stats()
    captioner = MiniCPMCaptioner(model_name="minicpm-v")
    mock_pil = MagicMock()

    truncated_response = MagicMock()
    truncated_response.json.return_value = {"response": "still partial", "done_reason": "length"}
    truncated_response.status_code = 200

    with patch("requests.post", return_value=truncated_response) as mock_post:
        cap = captioner.caption(mock_pil)
        assert cap == ""
        assert mock_post.call_count == 2

    stats = aria.get_minicpm_truncation_stats()
    assert stats["total_calls"] == 1
    assert stats["truncated_first_attempt"] == 1
    assert stats["truncated_after_retry"] == 1
    assert stats["truncation_rate_after_retry"] == 1.0


def test_moondream_captioner_appends_focus_hint():
    from unittest.mock import MagicMock
    from iris.aria import MoondreamCaptioner

    captioner = MoondreamCaptioner()
    fake_model = MagicMock()
    fake_model.device = "cpu"
    fake_model.to.return_value = fake_model  # .to(device) must return the same configured mock
    fake_model.encode_image.return_value = "ENC"
    fake_model.answer_question.return_value = "a boy in red running"
    captioner._model = fake_model
    captioner._device = "cpu"
    captioner._tokenizer = MagicMock()

    captioner.caption("PIL_IMG", focus_hint="Pay attention to: why did the boy in red run?")
    args, kwargs = fake_model.answer_question.call_args
    prompt_used = args[1]
    assert "Pay attention to: why did the boy in red run?" in prompt_used
    assert prompt_used.startswith(MoondreamCaptioner.BASE_PROMPT)

    captioner.caption("PIL_IMG")
    args, kwargs = fake_model.answer_question.call_args
    assert args[1] == MoondreamCaptioner.BASE_PROMPT


def test_zero_override_config_resolves_to_seated_production_models():
    """The test that would have caught the moondream/minicpm mismatch before
    it needed a separate infra-seating pass to diagnose: constructing the
    canonical pipeline config with ZERO overrides -- IRISConfig() directly,
    the exact call a caller who never touches captioner_backend/answerer_*
    would make -- must resolve to the seated, live-verified production
    models: captioner minicpm-v4.6 (via Ollama, used at ingest time),
    answerer granite4:micro via llama-server on port 8091 (used at query
    time, NOT Ollama -- llama-server is required for the per-request
    cache_prompt=false determinism guarantee from the answerer-seed fix)."""
    from unittest.mock import MagicMock, patch
    import iris.aria as aria
    from iris.aria import get_backend, get_captioner, LlamaServerBackend, MiniCPMCaptioner
    from iris.iris_config import IRISConfig

    original_captioner = aria._ACTIVE_CAPTIONER
    original_backend = aria._ACTIVE_BACKEND
    original_overridden = aria._BACKEND_OVERRIDDEN
    aria._ACTIVE_CAPTIONER = None
    aria._ACTIVE_BACKEND = None
    aria._BACKEND_OVERRIDDEN = False
    try:
        cfg = IRISConfig()
        assert cfg.captioner_backend == "minicpm"
        assert cfg.answerer_backend == "llama_server"
        assert cfg.answerer_endpoint == "http://127.0.0.1:8091/v1"
        assert cfg.answerer_model == "granite4:micro"

        # Mock the Ollama /api/tags probe (hermetic: doesn't depend on a live
        # server actually being up during test runs) confirming minicpm-v4.6
        # is the tag MiniCPMCaptioner's own auto-detection resolves to.
        mock_tags_response = MagicMock()
        mock_tags_response.status_code = 200
        mock_tags_response.json.return_value = {"models": [{"name": "minicpm-v4.6:latest"}]}
        with patch("requests.get", return_value=mock_tags_response):
            captioner = get_captioner(cfg)
        assert isinstance(captioner, MiniCPMCaptioner)
        assert captioner.model_name == "minicpm-v4.6"

        backend = get_backend(cfg)
        assert isinstance(backend, LlamaServerBackend)
        assert backend.endpoint == "http://127.0.0.1:8091/v1"
        assert backend.text_model == "granite4:micro"
    finally:
        aria._ACTIVE_CAPTIONER = original_captioner
        aria._ACTIVE_BACKEND = original_backend
        aria._BACKEND_OVERRIDDEN = original_overridden