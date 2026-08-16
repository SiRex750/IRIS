#!/bin/bash
# Self-retrying wrapper for scripts/mlvu_eval.py.
#
# This environment has been observed to kill long-running background
# processes unpredictably (twice, both without any explicit stop from the
# controlling session) partway through a ~2-3 hour MLVU eval run. mlvu_eval.py
# checkpoints every question to <out-prefix>_checkpoint.json and resumes from
# it on restart, so this loop just keeps relaunching the eval until it exits
# 0 (full report written + sanity guards passed) or a real, non-infra error
# occurs (in which case it still stops retrying after MAX_ATTEMPTS so a
# genuine bug doesn't spin forever).
set -u
cd "$(dirname "$0")/.."

MAX_ATTEMPTS=15
LOG="eval_results/_mlvu_real_run_stdout.log"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== [retry-wrapper] attempt $attempt/$MAX_ATTEMPTS $(date) ===" >> "$LOG"
  ./.venv/Scripts/python.exe scripts/mlvu_eval.py \
    --anno-dir ./mlvu/MLVU/json --video-dir ./mlvu/MLVU/video >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "=== [retry-wrapper] SUCCESS on attempt $attempt (exit 0) ===" >> "$LOG"
    exit 0
  fi
  echo "=== [retry-wrapper] attempt $attempt exited $rc, checking whether to retry ===" >> "$LOG"
  # A sanity-guard failure (HarnessError) is a REAL result, not an infra
  # kill -- the report was still written, so don't spin retrying it.
  if grep -q "SANITY GUARD FAILURE" "$LOG"; then
    echo "=== [retry-wrapper] sanity guard failed -- this is a real result, not retrying ===" >> "$LOG"
    exit $rc
  fi
  echo "=== [retry-wrapper] looks like an infra kill -- resuming from checkpoint in 5s ===" >> "$LOG"
  sleep 5
done
echo "=== [retry-wrapper] exhausted $MAX_ATTEMPTS attempts without success ===" >> "$LOG"
exit 1
