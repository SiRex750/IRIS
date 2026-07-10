import subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
script = REPO / "scripts" / "phase6_verify_62b.py"
log = REPO / "verify62b_output.log"
with open(log, "w", encoding="utf-8") as f:
    subprocess.run(
        [sys.executable, "-u", str(script)],
        cwd=str(REPO),
        stdout=f, stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
print(f"done -> {log}")
