"""A1 regression test: temperature=0.0 alone is not sufficient for strict
greedy decoding on llama.cpp; top_k=1 and top_p=1.0 must be sent on every
request-construction path in LlamaServerBackend, and the same must be applied
to LlamaBackend (Ollama) so the two backends do not silently differ.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from iris.aria import LlamaBackend, LlamaServerBackend


def test_llama_server_openai_client_path_pins_greedy_params():
    backend = LlamaServerBackend(endpoint="http://localhost:8080/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c")

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["extra_body"]["top_k"] == 1


def test_llama_server_schema_http_payload_pins_greedy_params():
    backend = LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "mocked http response"}}]}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        backend.generate(prompt="q", context="c", schema_format=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0
        assert payload["top_k"] == 1
        assert payload["top_p"] == 1.0


def test_llama_server_completion_fallback_payload_pins_greedy_params():
    """Exercise the native /completion fallback (OpenAI-compat client raises,
    code falls back to raw requests.post)."""
    backend = LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")
    backend._client = mock_client

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"content": "mocked completion response"}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        backend.generate(prompt="q", context="c")
        payload = mock_post.call_args[1]["json"]
        assert payload["temperature"] == 0.0
        assert payload["top_k"] == 1
        assert payload["top_p"] == 1.0


def test_llama_backend_default_chat_path_pins_greedy_params():
    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c")

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["extra_body"]["top_k"] == 1


def test_llama_backend_response_format_path_pins_greedy_params():
    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked answer"
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    backend._client = mock_client

    backend.generate(prompt="q", context="c", response_format={"type": "json_object"})

    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["extra_body"]["top_k"] == 1


def test_llama_backend_native_schema_path_pins_greedy_params():
    backend = LlamaBackend(endpoint="http://localhost:11434/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"message": {"content": "mocked wire response"}}
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        backend.generate(prompt="q", context="c", schema_format=True)
        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["top_k"] == 1
        assert payload["options"]["top_p"] == 1.0
