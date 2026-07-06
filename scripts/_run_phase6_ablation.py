import subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
script = REPO / "scripts" / "phase6_run_ablation.py"
log = REPO / "phase6_ablation_output.log"
with open(log, "w", encoding="utf-8") as f:
    subprocess.run([sys.executable, "-u", str(script)],
                   cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT, encoding="utf-8")
print(f"done -> {log}")
