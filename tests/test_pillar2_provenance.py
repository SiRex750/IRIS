"""Dry-path tests for scripts/pillar2_grounded_qa.py's seat guard and
provenance assembly (A3). These never run the benchmark -- half_width stays
unset by design (DECISIONS.md 2026-07-17-later §A6 precondition) -- they only
exercise preflight_backend(), git_provenance(), and config_hash() directly
against mocks.
"""
from unittest.mock import MagicMock, patch

import pytest

import iris.aria as aria
import scripts.pillar2_grounded_qa as pillar2
from iris.iris_config import IRISConfig


def test_preflight_rejects_wrong_backend_class():
    wrong_backend = aria.LlamaBackend(endpoint="http://127.0.0.1:11434/v1", text_model="granite4:micro")
    with pytest.raises(RuntimeError) as exc_info:
        pillar2.preflight_backend(wrong_backend)
    msg = str(exc_info.value)
    assert "LlamaServerBackend" in msg
    assert "LlamaBackend" in msg


def test_preflight_raises_on_dead_endpoint():
    backend = aria.LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    with patch("requests.get", side_effect=ConnectionError("refused")):
        with pytest.raises(RuntimeError) as exc_info:
            pillar2.preflight_backend(backend)
    assert backend.endpoint in str(exc_info.value)


def test_preflight_passes_for_seated_backend_and_live_endpoint():
    backend = aria.LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    with patch("requests.get", return_value=mock_resp) as mock_get:
        pillar2.preflight_backend(backend)  # must not raise
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == f"{backend.endpoint}/models"


def test_backend_exposes_temperature_and_cache_prompt_for_provenance():
    backend = aria.LlamaServerBackend(endpoint="http://127.0.0.1:8091/v1", text_model="granite4:micro")
    assert backend.temperature == 0.0
    assert backend.cache_prompt is False


def test_git_provenance_reads_commit_and_dirty_flag():
    fake_commit = MagicMock(stdout="deadbeef1234\n")
    fake_status_clean = MagicMock(stdout="")
    with patch("subprocess.run", side_effect=[fake_commit, fake_status_clean]):
        commit, dirty = pillar2.git_provenance(pillar2.REPO_ROOT)
    assert commit == "deadbeef1234"
    assert dirty is False

    fake_commit2 = MagicMock(stdout="deadbeef1234\n")
    fake_status_dirty = MagicMock(stdout=" M some/file.py\n")
    with patch("subprocess.run", side_effect=[fake_commit2, fake_status_dirty]):
        commit2, dirty2 = pillar2.git_provenance(pillar2.REPO_ROOT)
    assert commit2 == "deadbeef1234"
    assert dirty2 is True


def test_config_hash_is_stable_and_input_sensitive():
    cfg_a = IRISConfig(ppr_lambda=0.5)
    cfg_b = IRISConfig(ppr_lambda=0.5)
    cfg_c = IRISConfig(ppr_lambda=0.7)

    h_a = pillar2.config_hash(cfg_a)
    h_b = pillar2.config_hash(cfg_b)
    h_c = pillar2.config_hash(cfg_c)

    assert h_a == h_b
    assert h_a != h_c
    assert len(h_a) == 64  # sha256 hex digest length
