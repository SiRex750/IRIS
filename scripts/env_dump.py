"""Capture the runtime environment for Appendix A.2.

Writes eval_results/env_A2.json with hardware, OS, Python, key package
versions, external tool versions, and git state at time of capture.

Uses only stdlib plus whatever is already installed in the active
environment. Never pip installs anything; missing packages/tools are
recorded as "not installed" / "not on PATH" rather than raising.
"""
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def get_cpu_model():
    system = platform.system()
    if system == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"], text=True, stderr=subprocess.DEVNULL
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines[1]
        except Exception:
            pass
        # Fallback: PowerShell CIM (wmic is deprecated/removed on newer Windows)
        try:
            out = subprocess.check_output(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance -ClassName Win32_Processor).Name",
                ],
                text=True, stderr=subprocess.DEVNULL,
            )
            out = out.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
        return platform.processor() or "unknown"
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or "unknown"
    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            )
            return out.strip()
        except Exception:
            pass
        return platform.processor() or "unknown"
    return platform.processor() or "unknown"


def get_core_counts():
    physical = None
    logical = None
    try:
        import psutil  # optional; not required
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        pass
    if logical is None:
        logical = _safe(lambda: __import__("os").cpu_count())
    if physical is None:
        system = platform.system()
        if system == "Windows":
            physical = _safe(lambda: int(subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -ClassName Win32_Processor).NumberOfCores"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip().splitlines()[0]))
        elif system == "Linux":
            def _linux_physical():
                with open("/proc/cpuinfo") as f:
                    text = f.read()
                phys_ids = set()
                core_ids = set()
                cur_phys = None
                for line in text.splitlines():
                    if line.lower().startswith("physical id"):
                        cur_phys = line.split(":", 1)[1].strip()
                        phys_ids.add(cur_phys)
                    elif line.lower().startswith("core id"):
                        core_ids.add((cur_phys, line.split(":", 1)[1].strip()))
                return len(core_ids) if core_ids else None
            physical = _safe(_linux_physical)
        elif system == "Darwin":
            physical = _safe(lambda: int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True
            ).strip()))
    return physical, logical


def get_total_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass
    system = platform.system()
    if system == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip().splitlines()[0]
            return round(int(out) / (1024 ** 3), 2)
        except Exception:
            return None
    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        except Exception:
            return None
    elif system == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(out.strip()) / (1024 ** 3), 2)
        except Exception:
            return None
    return None


def get_package_version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown (no __version__ attribute)")
    except ImportError:
        return "not installed"
    except Exception as e:
        return f"error: {e}"


def get_clip_version():
    """open_clip or clip, whichever is installed."""
    v = get_package_version("open_clip")
    if v != "not installed":
        return {"package": "open_clip", "version": v}
    v = get_package_version("clip")
    if v != "not installed":
        return {"package": "clip", "version": v}
    return {"package": None, "version": "not installed"}


def get_tool_version(tool_name, version_flag="--version"):
    path = shutil.which(tool_name)
    if path is None:
        return "not on PATH"
    try:
        out = subprocess.check_output(
            [tool_name, version_flag], text=True, stderr=subprocess.STDOUT
        )
        return out.strip().splitlines()[0].strip()
    except Exception as e:
        return f"error running {tool_name}: {e}"


def get_git_state():
    head = _safe(lambda: subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip())
    dirty_count = None
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, text=True,
        )
        dirty_count = len([l for l in out.splitlines() if l.strip()])
    except Exception:
        pass
    return {"head": head, "tracked_dirty_file_count": dirty_count}


def get_torch_info():
    try:
        import torch
        return {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": _safe(lambda: torch.cuda.is_available()),
        }
    except ImportError:
        return {"version": "not installed", "cuda_available": None}
    except Exception as e:
        return {"version": f"error: {e}", "cuda_available": None}


def main():
    physical_cores, logical_cores = get_core_counts()
    clip_info = get_clip_version()

    data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": {
            "model": get_cpu_model(),
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
        },
        "ram_total_gb": get_total_ram_gb(),
        "os": {
            "name": platform.system(),
            "version": platform.platform(),
        },
        "python_version": platform.python_version(),
        "torch": get_torch_info(),
        "packages": {
            "numpy": get_package_version("numpy"),
            "networkx": get_package_version("networkx"),
            "av": get_package_version("av"),
            "transformers": get_package_version("transformers"),
            "clip_impl": clip_info,
        },
        "tools": {
            "ffmpeg": get_tool_version("ffmpeg"),
            "ollama": get_tool_version("ollama"),
        },
        "git": get_git_state(),
    }

    out_path = REPO_ROOT / "eval_results" / "env_A2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
