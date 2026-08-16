#!/bin/bash
# Self-retrying wrapper for scripts/mlvu_ablation.py, mirroring
# scripts/_mlvu_run_with_retry.sh. This environment has been observed to
# kill long-running background processes unpredictably; mlvu_ablation.py
# checkpoints every (arm, task, question_id) triple and resumes from it on
# restart, so this loop just keeps relaunching until it exits 0 (full
# report written + GUARD (b) passed) or a real, non-infra error occurs.
set -u
cd "$(dirname "$0")/.."

MAX_ATTEMPTS=40
LOG="eval_results/_mlvu_ablation_run_stdout.log"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== [retry-wrapper] attempt $attempt/$MAX_ATTEMPTS $(date) ===" >> "$LOG"
  ./.venv/Scripts/python.exe scripts/mlvu_ablation.py \
    --anno-dir ./mlvu/MLVU/json --video-dir ./mlvu/MLVU/video >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [retry-wrapper] SUCCESS on attempt $attempt (exit 0) ===" >> "$LOG"
    exit 0
  fi
  echo "=== [retry-wrapper] attempt $attempt exited $rc, checking whether to retry ===" >> "$LOG"
  # A GUARD (b) failure (HarnessError: codec arm diverges from baseline) is
  # a REAL result, not an infra kill -- stop retrying, it won't self-heal.
  if grep -q "GUARD (b) FAILURE" "$LOG"; then
    echo "=== [retry-wrapper] GUARD (b) failure detected -- NOT retrying, this is a real contamination result ===" >> "$LOG"
    exit 1
  fi
  sleep 5
done
echo "=== [retry-wrapper] exhausted $MAX_ATTEMPTS attempts without success ===" >> "$LOG"
exit 1
