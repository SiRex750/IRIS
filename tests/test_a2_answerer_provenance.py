"""A2 regression test: scripts/answerer_provenance.py must capture serving
process cmdline, server binary hash/version, GGUF path/hash, /v1/models
response, and the sampler params actually sent -- the exact fields the prior
val_confirm_e2e reproduction gap traced back to (different llama-server
binary, same GGUF, nothing in the artifacts flagged it).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root is on sys.path so `scripts` is importable as a namespace package.
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.answerer_provenance import (
    capture_answerer_provenance,
    get_binary_version,
    get_models_endpoint,
    read_proc_cmdline,
    sha256_file,
)


def test_sha256_file_matches_known_hash(tmp_path):
    p = tmp_path / "binary"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_missing_returns_none(tmp_path):
    assert sha256_file(tmp_path / "does_not_exist") is None


def test_read_proc_cmdline_returns_none_when_proc_unavailable():
    # On this (Windows) box /proc never exists -- this must degrade gracefully,
    # never raise.
    assert read_proc_cmdline(999999999) is None


def test_get_binary_version_returns_none_on_missing_binary():
    assert get_binary_version("/definitely/not/a/real/binary/path") is None


def test_get_models_endpoint_returns_none_on_connection_failure():
    assert get_models_endpoint("http://127.0.0.1:1") is None


def test_get_models_endpoint_returns_parsed_json_on_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "granite4:micro"}]}
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp) as mock_get:
        result = get_models_endpoint("http://127.0.0.1:8091/v1")
        mock_get.assert_called_once_with("http://127.0.0.1:8091/v1/models", timeout=10)
        assert result == {"data": [{"id": "granite4:micro"}]}


def test_capture_answerer_provenance_assembles_all_required_fields(tmp_path):
    fake_binary = tmp_path / "llama-server"
    fake_binary.write_bytes(b"fake binary contents")
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"fake gguf contents")

    with patch("scripts.answerer_provenance.find_listening_pid", return_value=None), \
         patch("scripts.answerer_provenance.get_models_endpoint", return_value={"data": []}), \
         patch("scripts.answerer_provenance.get_binary_version", return_value="version 1.2.3"):
        record = capture_answerer_provenance(
            endpoint="http://127.0.0.1:8091/v1",
            port=8091,
            gguf_path=str(fake_gguf),
            gguf_expected_sha256=sha256_file(fake_gguf),
            sampler_params={
                "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                "seed": 42, "cache_prompt": False,
            },
            binary_path=str(fake_binary),
        )

    assert record["binary_sha256"] == sha256_file(fake_binary)
    assert record["binary_version_output"] == "version 1.2.3"
    assert record["gguf_sha256"] == sha256_file(fake_gguf)
    assert record["gguf_sha256_matches_expected"] is True
    assert record["models_endpoint_response"] == {"data": []}
    assert record["sampler_params_sent"] == {
        "temperature": 0.0, "top_k": 1, "top_p": 1.0,
        "seed": 42, "cache_prompt": False,
    }
    assert record["cmdline"] == "UNAVAILABLE (no /proc access or pid not found)"


def test_capture_answerer_provenance_flags_gguf_sha_mismatch(tmp_path):
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"actual contents")

    with patch("scripts.answerer_provenance.find_listening_pid", return_value=None), \
         patch("scripts.answerer_provenance.get_models_endpoint", return_value=None):
        record = capture_answerer_provenance(
            endpoint="http://127.0.0.1:8091/v1",
            port=8091,
            gguf_path=str(fake_gguf),
            gguf_expected_sha256="deadbeef" * 8,
            sampler_params={},
        )

    assert record["gguf_sha256_matches_expected"] is False
