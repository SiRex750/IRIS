"""Capture answerer serving provenance for a run's environment.json.

Context: a prior val_confirm_e2e reproduction attempt used
`/usr/local/lib/ollama/llama-server` while the recorded headline run used a
from-source llama.cpp build (tag b10099, commit 1a064ab, CUDA 12.6, sm_89) --
same GGUF, different binary, and nothing in the artifacts flagged it. This
module gives every future run a way to record which binary actually served
the requests, so that class of silent divergence is visible instead of
inferred after the fact.

Not wired into iris/ -- this is run-provenance tooling for scripts/, called
once per run before/after serving starts.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_proc_cmdline(pid: int) -> str | None:
    """Read a process's full command line from /proc/<pid>/cmdline (Linux only).

    Returns None (not an exception) when /proc is unavailable -- e.g. on
    Windows, or when the pid does not exist -- so callers can record
    "unavailable" rather than crash a run over provenance capture.
    """
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return None
    raw = cmdline_path.read_bytes()
    return " ".join(part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part)


def find_listening_pid(port: int) -> int | None:
    """Best-effort PID lookup for whatever process is listening on `port`,
    via `ss -ltnp` (Linux). Returns None if ss is unavailable or nothing
    is found -- this is provenance-capture tooling, not a hard dependency."""
    try:
        out = subprocess.run(
            ["ss", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if f":{port}" in line and "pid=" in line:
            try:
                pid_part = line.split("pid=", 1)[1]
                pid_str = pid_part.split(",", 1)[0]
                return int(pid_str)
            except (IndexError, ValueError):
                continue
    return None


def get_binary_version(binary_path: str | Path) -> str | None:
    try:
        proc = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True, text=True, timeout=15,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out.strip() or None
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def get_models_endpoint(base_url: str) -> dict[str, Any] | None:
    """GET {base_url}/models (OpenAI-compatible). Returns the parsed JSON, or
    None if the request fails (no server running is a normal condition when
    this is called outside a live run, e.g. from a unit test)."""
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 -- provenance capture must never crash a run
        return None


def capture_answerer_provenance(
    *,
    endpoint: str,
    port: int,
    gguf_path: str | None,
    gguf_expected_sha256: str | None,
    sampler_params: dict[str, Any],
    binary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the provenance block to embed under environment.json["server"]
    (or a top-level "provenance" key). `sampler_params` should be exactly the
    kwargs the caller sent on the request (temperature/top_k/top_p/seed/
    cache_prompt), not re-derived here, so this records what was actually
    sent rather than what a config claims should have been sent.
    """
    pid = find_listening_pid(port)
    cmdline = read_proc_cmdline(pid) if pid is not None else None
    resolved_binary_path = binary_path
    if resolved_binary_path is None and cmdline:
        resolved_binary_path = cmdline.split(" ", 1)[0]

    binary_sha256 = sha256_file(resolved_binary_path) if resolved_binary_path else None
    binary_version = get_binary_version(resolved_binary_path) if resolved_binary_path else None
    gguf_sha256 = sha256_file(gguf_path) if gguf_path else None

    return {
        "endpoint": endpoint,
        "listening_port": port,
        "pid": pid,
        "cmdline": cmdline if cmdline is not None else "UNAVAILABLE (no /proc access or pid not found)",
        "binary_path": str(resolved_binary_path) if resolved_binary_path else None,
        "binary_sha256": binary_sha256,
        "binary_version_output": binary_version,
        "gguf_path": gguf_path,
        "gguf_sha256": gguf_sha256,
        "gguf_sha256_matches_expected": (
            gguf_sha256 == gguf_expected_sha256 if gguf_expected_sha256 else None
        ),
        "models_endpoint_response": get_models_endpoint(endpoint),
        "sampler_params_sent": sampler_params,
    }
