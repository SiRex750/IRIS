"""Runner: execute diag5 and capture all output to diag5_output.log."""
import sys, subprocess, pathlib

REPO = pathlib.Path(r"C:\IRIS")
out_log = REPO / "diag5_output.log"
script = REPO / "scripts" / "phase6_diag5_rankspace.py"

with open(out_log, "w", encoding="utf-8", buffering=1) as f:
    proc = subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(REPO),
        stdout=f,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )

sys.exit(proc.returncode)
