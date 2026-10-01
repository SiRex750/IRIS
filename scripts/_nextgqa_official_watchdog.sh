#!/bin/bash
# Watchdog for the registered official NExT-GQA test run
# (eval_results/NEXTGQA_official_test_prereg.md, registered 546d392; run script 64a3451).
#
# It launches scripts/nextgqa_official_test_run.py with exactly the session-3 command line and,
# when a session ends, decides whether to start the next one. The run script is resumable
# (completed question-arm rows are never recomputed), so a relaunch only continues the run.
#
#   exit 0 + result .md written      -> DONE, watchdog exits 0
#   STOP: captioner failure          -> relaunch (transient Ollama-side failure; the guard halts
#                                       before any fallback caption reaches an answer)
#   crash with no STOP line          -> relaunch
#   any other STOP                   -> NO relaunch: registration / clean-tree / determinism /
#                                       split / disjointness / answer-format guards are integrity
#                                       failures for a human to look at; watchdog exits 3
#   MAX_NO_PROGRESS sessions in a row that add no checkpoint rows -> give up, exit 4
#
# Nothing registered is touched: same command, same config, same servers, guards left on.
# Environment-only actions it may take, each logged: unload the idle MiniCPM model before a
# relaunch (fresh Ollama runner), and stop SignalRgb if its committed memory passes
# SIGNALRGB_KILL_GB (its leak exhausted the commit limit and caused the session 1-3 stops).
#
# Every session gets its own log (next free nextgqa_registered_run_sessionN.log) and an entry
# appended to eval_results/NEXTGQA_official_test_sessions.md, so no session goes unrecorded.
set -u
cd "$(dirname "$0")/.."

LOGDIR="/c/Users/Siddanth Anil/iris-bin"
WLOG="$LOGDIR/nextgqa_watchdog.log"
SESSIONS="eval_results/NEXTGQA_official_test_sessions.md"
CKPT="eval_results/_nextgqa_official_test_checkpoint.jsonl"
OUT_MD="eval_results/NEXTGQA_official_test_result.md"
OLLAMA="/c/Users/Siddanth Anil/AppData/Local/Programs/Ollama/ollama.exe"
CAPTION_MODEL="minicpm-v4.6:1b"
ANSWERER_BUILD="b10099-1a064ab09"

POLL_S=120
HEARTBEAT_EVERY=5          # polls (~10 min)
PREFLIGHT_RETRY_S=600
MAX_NO_PROGRESS=3
SIGNALRGB_KILL_GB=16
COMMIT_WARN_GB=8
MIN_AVAIL_MB=8192
MIN_GPU_FREE_MB=4096

wlog() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$WLOG"; }

# PowerShell via -EncodedCommand: no .ps1 file, so the machine's execution policy is untouched.
ps_run() { powershell -NoProfile -EncodedCommand "$(printf '%s' "$1" | iconv -f utf-8 -t utf-16le | base64 -w0)" 2>/dev/null | tr -d '\r'; }

PS_PROBE=$(cat <<EOF
\$ErrorActionPreference = 'SilentlyContinue'
\$c = (Get-Counter '\\Memory\\Committed Bytes','\\Memory\\Commit Limit','\\Memory\\Available MBytes').CounterSamples
function PG(\$p) { [math]::Round(((\$p | Measure-Object PrivateMemorySize64 -Sum).Sum) / 1GB, 1) }
\$sr = Get-Process -Name SignalRgb
\$srg = PG \$sr
\$act = ''
if (\$srg -gt $SIGNALRGB_KILL_GB) { \$sr | Stop-Process -Force; \$act = ' SIGNALRGB_KILLED' }
\$ls = Get-Process -Name llama-server
\$olr = PG (\$ls | Where-Object { \$_.Path -like '*Ollama*' })
\$ans = PG (\$ls | Where-Object { \$_.Path -like '*llama-b10099*' })
\$ids = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { \$_.CommandLine -like '*nextgqa_official_test_run.py*' }).ProcessId
\$py = PG (\$ids | ForEach-Object { Get-Process -Id \$_ })
'commit={0:N1}/{1:N1}GB commit_free={2:N1}GB avail={3:N1}GB run_py={4}GB ollama_runner={5}GB answerer={6}GB signalrgb={7}GB{8}' -f (\$c[0].CookedValue/1GB), (\$c[1].CookedValue/1GB), ((\$c[1].CookedValue-\$c[0].CookedValue)/1GB), (\$c[2].CookedValue/1024), \$py, \$olr, \$ans, \$srg, \$act
EOF
)
PS_GAMES='(Get-Process | Where-Object { $_.Name -match "RocketLeague" }).Count'
PS_AVAIL='[math]::Floor((Get-Counter "\Memory\Available MBytes").CounterSamples.CookedValue)'

rows() { if [ -f "$CKPT" ]; then wc -l < "$CKPT" | tr -d ' '; else echo 0; fi; }

next_session() {
  local max=1 n f
  for f in "$LOGDIR"/nextgqa_registered_run_session*.log; do
    [ -e "$f" ] || continue
    n=$(basename "$f" | sed -E 's/^nextgqa_registered_run_session([0-9]+)\.log$/\1/')
    [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$max" ] && max=$n
  done
  echo $((max + 1))
}

# Same checks as the session-2/3 pre-flight. Prints a one-line summary; returns non-zero on any failure.
preflight() {
  local games avail gpu tags build dirty ok=0 why=""
  games=$(ps_run "$PS_GAMES"); games=${games:-0}
  avail=$(ps_run "$PS_AVAIL"); avail=${avail:-0}
  gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'); gpu=${gpu:-0}
  tags=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://127.0.0.1:11434/api/tags || true)
  build=$(curl -s -m 10 http://127.0.0.1:8091/props | grep -o "$ANSWERER_BUILD" | head -1 || true)
  dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
  [ "$games" -eq 0 ] || { ok=1; why="$why game-running"; }
  [ "$avail" -ge "$MIN_AVAIL_MB" ] || { ok=1; why="$why low-RAM"; }
  [ "$gpu" -ge "$MIN_GPU_FREE_MB" ] || { ok=1; why="$why low-GPU"; }
  [ "$tags" = "200" ] || { ok=1; why="$why ollama-down"; }
  [ "$build" = "$ANSWERER_BUILD" ] || { ok=1; why="$why answerer-down-or-wrong-build"; }
  [ "$dirty" -eq 0 ] || { ok=1; why="$why tracked-tree-dirty"; }
  echo "games=$games avail=${avail}MB gpu_free=${gpu}MiB ollama=$tags answerer_build=${build:-none} tracked_dirty=$dirty${why:+ FAIL:$why}"
  return $ok
}

append_session_record() {  # n start end pid slog rc rows0 rows1 pre env outcome action mem
  local n=$1 start=$2 end=$3 pid=$4 slog=$5 rc=$6 r0=$7 r1=$8 pre=$9 env=${10} outcome=${11} action=${12} mem=${13}
  local det ans stopl tb
  det=$(grep -a -m1 '^\[DETERMINISM\]' "$slog" | tr -d '\r')
  ans=$(grep -a '^\[ANSWER ' "$slog" | tail -1 | tr -d '\r')
  stopl=$(grep -a '^\*\*\* STOP:' "$slog" | tail -1 | tr -d '\r')
  tb=$(grep -a -E '^[A-Za-z_.]+(Error|Exception)\b' "$slog" | tail -1 | tr -d '\r')
  {
    echo ""
    echo "## Session $n — $outcome (auto-recorded by scripts/_nextgqa_official_watchdog.sh)"
    echo ""
    echo "- **Run commit:** $(git rev-parse --short HEAD) (tracked-dirty at launch: 0). Launched $start IST (PID $pid);"
    echo "  log \`C:\\Users\\Siddanth Anil\\iris-bin\\$(basename "$slog")\`."
    echo "- **Pre-flight:** $pre"
    echo "- **Environment actions before launch:** $env"
    echo "- **Determinism check:** ${det:-no [DETERMINISM] line reached}"
    echo "- **Progress:** checkpoint rows $r0 -> $r1 (+$((r1 - r0))); last answer line: \`${ans:-none}\`"
    echo "- **End:** $end IST, exit code $rc."
    [ -n "$stopl" ] && echo "  \`$stopl\`"
    [ -z "$stopl" ] && [ -n "$tb" ] && echo "  Exception: \`$tb\`"
    echo "- **Memory at last poll:** $mem"
    echo "- **Watchdog action:** $action"
  } >> "$SESSIONS"
}

if [ "${1:-}" = "--check" ]; then  # dry run: pre-flight + memory probe only, launches nothing
  echo "next session: $(next_session)  checkpoint rows: $(rows)"
  echo "pre-flight: $(preflight)"; echo "pre-flight rc: $?"
  echo "probe: $(ps_run "$PS_PROBE")"
  exit 0
fi

# Single-instance guard (same pidfile pattern as scripts/_mlvu_ablation_long_trimmed_watchdog.sh).
PIDFILE="eval_results/.nextgqa_official_watchdog.pid"
if [ -f "$PIDFILE" ]; then
  old_pid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFUSING TO START: another watchdog instance is already running (pid $old_pid, $PIDFILE)."
    exit 1
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

wlog "WATCHDOG START pid=$$ (commit $(git rev-parse --short HEAD), checkpoint rows $(rows))"
no_progress=0
env_note="${WATCHDOG_ENV_NOTE:-none}"

while true; do
  if [ -f "$OUT_MD" ]; then
    wlog "DONE: $OUT_MD already exists -- nothing to run."
    exit 0
  fi

  # Pre-flight; wait (do not launch) while anything fails.
  while true; do
    pre=$(preflight) && break
    wlog "WAITING: pre-flight failed -- $pre (retry in ${PREFLIGHT_RETRY_S}s)"
    sleep "$PREFLIGHT_RETRY_S"
  done

  n=$(next_session)
  slog="$LOGDIR/nextgqa_registered_run_session$n.log"
  if [ -e "$slog" ]; then wlog "ABORT: $slog already exists"; exit 5; fi
  r0=$(rows)
  start=$(date '+%Y-%m-%d %H:%M:%S')
  ./.venv/Scripts/python.exe scripts/nextgqa_official_test_run.py \
    --test-csv eval/data/nextgqa_official_test/official/test.csv \
    --gsub-test eval/data/nextgqa_official_test/official/gsub_test.json \
    --videos-dir eval/data/nextgqa_official_test/videos \
    --annotation-source "doc-doc/NExT-GQA@63772e0256ad9e34b83d6a108f1ba2041b924607" \
    --server-build b10099 \
    --model-sha256 6c02683809a8dc4eb05c78d44bc63bcd707703b078998fa58829c858ab337bb0 \
    > "$slog" 2>&1 &
  pid=$!
  wlog "SESSION $n LAUNCHED pid=$pid rows=$r0 log=$slog | pre-flight: $pre"

  polls=0; mem="(no poll yet)"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    kill -0 "$pid" 2>/dev/null || break
    polls=$((polls + 1))
    mem=$(ps_run "$PS_PROBE")
    case "$mem" in *SIGNALRGB_KILLED*) wlog "ACTION: SignalRgb passed ${SIGNALRGB_KILL_GB}GB committed and was stopped -- $mem" ;; esac
    cfree=$(echo "$mem" | sed -nE 's/.*commit_free=([0-9.,]+)GB.*/\1/p' | tr -d ',')
    if [ -n "$cfree" ] && awk "BEGIN{exit !($cfree < $COMMIT_WARN_GB)}"; then
      wlog "MEMORY WARNING: commit headroom ${cfree}GB < ${COMMIT_WARN_GB}GB -- $mem"
    fi
    if [ $((polls % HEARTBEAT_EVERY)) -eq 0 ]; then
      wlog "HEARTBEAT session $n: rows=$(rows) last=$(grep -a '^\[ANSWER ' "$slog" | tail -1 | tr -d '\r') | $mem"
    fi
  done
  wait "$pid"; rc=$?
  end=$(date '+%Y-%m-%d %H:%M:%S')
  r1=$(rows)
  stopl=$(grep -a '^\*\*\* STOP:' "$slog" | tail -1 | tr -d '\r')
  [ "$r1" -gt "$r0" ] && no_progress=0 || no_progress=$((no_progress + 1))

  if [ "$rc" -eq 0 ] && [ -f "$OUT_MD" ]; then
    append_session_record "$n" "$start" "$end" "$pid" "$slog" "$rc" "$r0" "$r1" "$pre" "$env_note" \
      "completed; results written" "none -- run complete" "$mem"
    wlog "DONE: session $n completed (rows $r0 -> $r1); wrote $OUT_MD"
    exit 0
  elif [ -n "$stopl" ] && [[ "$stopl" != *"captioner failure"* ]]; then
    append_session_record "$n" "$start" "$end" "$pid" "$slog" "$rc" "$r0" "$r1" "$pre" "$env_note" \
      "stopped by an integrity guard" "none -- watchdog exited; not relaunched (integrity guard)" "$mem"
    wlog "HALT: session $n ended on an integrity guard, not relaunching: $stopl"
    exit 3
  elif [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
    append_session_record "$n" "$start" "$end" "$pid" "$slog" "$rc" "$r0" "$r1" "$pre" "$env_note" \
      "stopped with no progress" "none -- watchdog gave up after $no_progress sessions in a row with no new rows" "$mem"
    wlog "GIVE UP: $no_progress consecutive sessions added no rows. Last: ${stopl:-exit $rc}"
    exit 4
  else
    kind=$([ -n "$stopl" ] && echo "stopped by the captioner guard" || echo "crashed (exit $rc, no STOP line)")
    append_session_record "$n" "$start" "$end" "$pid" "$slog" "$rc" "$r0" "$r1" "$pre" "$env_note" \
      "$kind" "unloaded $CAPTION_MODEL (fresh Ollama runner), then relaunched after pre-flight" "$mem"
    wlog "SESSION $n ENDED ($kind; rows $r0 -> $r1): ${stopl:-$(tail -3 "$slog" | tr -d '\r' | tr '\n' ' ')} -- relaunching"
    "$OLLAMA" stop "$CAPTION_MODEL" >/dev/null 2>&1
    env_note="$CAPTION_MODEL unloaded after the previous session's stop (fresh Ollama runner)"
    sleep 30
  fi
done
