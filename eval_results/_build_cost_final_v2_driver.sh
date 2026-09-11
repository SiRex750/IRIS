#!/bin/bash
set -e
cd "C:/Users/Siddanth Anil/IRIS"
PY=".venv/Scripts/python.exe"
OUT="eval_results/build_cost_final_v2_raw"
for r in 1 2 3; do
  echo "=== round $r: old ==="
  "$PY" scripts/build_cost_final_probe.py --mode old --n 3 --round $r --out-json "$OUT/old_r${r}.json"
  echo "=== round $r: new ==="
  "$PY" scripts/build_cost_final_probe.py --mode new --n 50 --round $r --out-json "$OUT/new_r${r}.json"
done
echo "ALL DONE"
