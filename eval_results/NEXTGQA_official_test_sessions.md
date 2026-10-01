# Official NExT-GQA test run — session record

Companion to `NEXTGQA_official_test_prereg.md` (registered 546d392) and the run's own output.
The run script records a resume only when answer rows already exist; this file records the
sessions that end before any answer is written, so none goes unreported.

## Session 1 — stopped by the captioner guard; no answer written, nothing scored

- **Run commit:** 64a3451 (clean tracked tree; registration 546d392 an ancestor). Launched
  2026-09-27 ~23:58 IST in registered mode; stdout/stderr at
  `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run.log` (220,004 bytes).
- **Ingest:** 990/990 videos OK, 0 failed, 23:58 -> 03:22 IST into
  `eval/data/nextgqa_official_test/index_cache/`. These caches are reused by later sessions
  (the script skips any video whose cache exists), so later sessions record no ingest timings.
- **Stop:** 2026-09-28 03:32 IST, during the 20-question determinism check (no `[DETERMINISM]`
  line was reached), at log line 2046:
  `*** STOP: captioner failure (would silently fall back to BLIP): {... 'error': 'Captioning failed: 400 Client Error: Bad Request for url: http://localhost:11434/api/generate'}`
- **Root cause (Ollama log `%TEMP%\ollama_serve.log`):** at 03:32:14, `mtmd_tokenize: error:
  std::bad_alloc` — a host memory-allocation failure while encoding an image for MiniCPM —
  returned as HTTP 400; Ollama's runner process then terminated at 03:37:55 (exit status 1).
  Consistent with memory pressure after a 3.4-hour ingest in the same process, not with a bad
  input: the 3-video smoke run captioned every frame through the same stack without error.
- **What reached the results:** nothing. No checkpoint file was created (it is opened only when
  answering begins); the first answered video's cache holds 0 captions (answering saves captions
  per video). IRIS's caption path loaded BLIP on the failure, but the guard halted before any
  BLIP caption reached an answer. The gold file had been loaded into memory but was never used:
  the determinism check does not read gold, and no question was scored.
- **Answerer:** healthy throughout (llama-server b10099 log shows normal slot processing to 03:31).

## Session 2 — stopped by the captioner guard on the first caption; no answer written, nothing scored

- **Run commit:** 64a3451 (clean tracked tree). Launched 2026-09-28 13:59:18 IST (PIDs
  128208 / 130116); log `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run_session2.log`.
- **Pre-flight:** Ollama and llama-server (build_info b10099-1a064ab09, model sha256-6c02683...)
  both responding; tracked tree clean; system RAM 31.2 GB total, **3.7 GB free at launch**.
- **Ingest:** none (all 990 caches reused from session 1); no `[INGEST]` lines.
- **Stop:** during the determinism check, on the first caption request (no `[DETERMINISM]`
  line reached):
  `*** STOP: captioner failure (would silently fall back to BLIP): {... 'latency': 13.39, 'error': 'Captioning failed: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate'}`
- **Reading so far:** two sessions stopped on Ollama-side captioning errors (400 from
  `std::bad_alloc`, then 500), the second before any long-running work in the process, with
  ~3.7 GB of 31.2 GB free. Consistent with host memory held by something outside the run;
  under investigation before any further session.
- **What reached the results:** nothing (no checkpoint file; guard halted before any BLIP caption
  reached an answer).

## Session 3 — answered 29/990 videos, then stopped by the captioner guard

- **Run commit:** 64a3451 (clean tracked tree). Launched 2026-09-28 14:44:58 IST (PIDs
  137396 / 141512); log `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run_session3.log`.
- **Pre-flight:** no game running; 11.7 GB RAM available; RTX 5070 ~10.8 GB GPU memory free;
  Ollama responding; llama-server build_info b10099-1a064ab09; tracked tree clean.
- **Ingest:** none (all 990 caches reused from session 1).
- **Determinism check:** passed (`[DETERMINISM] 20 Arm-F questions answered twice, byte-identical`).
- **Progress:** `[ANSWER 29/990]` (video 10985344225) was the last completed video: 190 questions x
  3 arms = 570 rows in `_nextgqa_official_test_checkpoint.jsonl` (last write 16:30:27 IST). All 570
  rows: 0 parse failures, 0 caption-failed frames, 0 ingest failures.
- **Stop:** 2026-09-28 ~16:31 IST, on the first caption request for the 30th video:
  `*** STOP: captioner failure (would silently fall back to BLIP): {'frame_idx': None, 'latency': 3.07..., 'error': 'Captioning failed: 400 Client Error: Bad Request for url: http://localhost:11434/api/generate'}`
  Ollama log at 16:31:21: `mtmd_tokenize: error: std::bad_alloc` -> HTTP 400.
- **What reached the results:** the 570 rows of videos 1–29 only. No row for video 30 was
  written; the guard halted before any BLIP caption reached an answer.

### Root cause of the session 1–3 stops (identified 2026-09-28 16:35 IST)

All three stops are `std::bad_alloc` in Ollama's MiniCPM runner (session 2's HTTP 500 is the same
failure: Ollama log 14:00:38, `terminating due to uncaught exception of type std::bad_alloc`).
The failure is Windows commit-charge exhaustion, not physical RAM: 123.6 GB committed of a 127.2 GB
commit limit (32 GB RAM + 98 GB pagefile). **SignalRgb** (RGB-lighting software, running since
2026-09-20) held **66.8 GB** of committed private memory with ~0 GB resident — a leak. Next largest:
Ollama MiniCPM runner 11.4 GB, granite llama-server 10.2 GB.

**Remediation (environment only; no code, config or registered setting changed):** SignalRgb
(PID 33440) was stopped at ~16:39 IST; committed bytes fell to 31.1 GB. Its launcher respawned it
at 16:39:25 (~0.5 GB); at its observed leak rate (~8 GB/day) it cannot exhaust commit within the
run, and it is watched. The idle MiniCPM model was unloaded (`ollama stop`) so the next session
starts with a fresh runner. The answerer llama-server (PID 121108) was not touched.

From session 4 on, sessions are launched by `scripts/_nextgqa_official_watchdog.sh` (same command
line as session 3), which appends one entry per session below. It relaunches only after a
captioner-guard stop or a crash; any other guard stop ends it without a relaunch.

## Session 4 — answered videos 30–161, then killed with its watchdog (recorded by hand)

- **Run commit:** 64a3451 (tracked-dirty 0). Launched 2026-09-28 16:44:26 IST by the watchdog
  (watchdog PID 14931, run PID 14982); log
  `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run_session4.log`.
- **Pre-flight:** games=0 avail=14748MB gpu_free=10950MiB ollama=200 answerer_build=b10099-1a064ab09
  tracked_dirty=0.
- **Environment actions before launch:** SignalRgb stopped and MiniCPM unloaded (see session 3).
- **Determinism check:** passed (`[DETERMINISM] 20 Arm-F questions answered twice, byte-identical`).
- **Progress:** checkpoint rows 570 -> 2844 (+2274); last line `[ANSWER 161/990] 2776910060: 6
  question(s) done`. 161 videos / 948 questions complete in all three arms.
- **End:** 2026-09-28 ~23:47 IST (log last written 23:46:54). No STOP line and no traceback: the
  run and its watchdog were terminated together from outside. No reboot, sleep or shutdown
  event in the System log; the answerer llama-server and Ollama (started outside Claude Code)
  kept running. The watchdog had been started with `nohup` from a Claude Code tool shell, and
  the Claude Code process for that session exited around then, so the likely cause is that its
  exit killed the process tree it had started. The watchdog therefore wrote no entry of its own.
- **State left behind:** the checkpoint ends on a video boundary (last row video 2776910060,
  qid 8, arm U). All 2844 lines are valid JSON with a trailing newline, and no row of video 162
  was written. Nothing was edited.
- **Memory at last poll (23:41:09):** commit=47.6/79.2GB commit_free=31.6GB avail=1.6GB
  run_py=2.2GB ollama_runner=11.4GB answerer=10.2GB signalrgb=0GB. SignalRgb was not running at
  the 23:30 and 23:41 polls. The watchdog did not stop it (no ACTION line), so it exited or was
  closed some other way.
- **Downtime:** about 16 h (23:47 -> relaunch on 2026-09-29).

From session 5 on, the watchdog is started through WMI (`Win32_Process.Create`, parent
WmiPrvSE, same user session) so that it and the run do not depend on any Claude Code process.
Script, command line and config are unchanged. torch in the venv is a CPU-only build
(2.13.0+cpu) whichever way it is launched, so the launch method does not change the compute path.

## Session 5 — stopped by the captioner guard (auto-recorded by scripts/_nextgqa_official_watchdog.sh)

- **Run commit:** 4fe50e2 (tracked-dirty at launch: 0). Launched 2026-09-29 16:04:23 IST (PID 19057);
  log `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run_session5.log`.
- **Pre-flight:** games=0 avail=11109MB gpu_free=10981MiB ollama=200 answerer_build=b10099-1a064ab09 tracked_dirty=0
- **Environment actions before launch:** watchdog restarted via WMI (detached from Claude Code) after session 4 was killed with its launcher; no other change
- **Determinism check:** [DETERMINISM] 20 Arm-F questions answered twice, byte-identical
- **Progress:** checkpoint rows 2844 -> 15831 (+12987); last answer line: `[ANSWER 944/990] 8750015460: 9 question(s) done`
- **End:** 2026-10-01 07:58:39 IST, exit code 2.
  `*** STOP: captioner failure (would silently fall back to BLIP): {'frame_idx': None, 'latency': 12.176809549331665, 'error': 'Captioning failed: 500 Server Error: Internal Server Error for url: http://localhost:11434/api/generate'}`
- **Memory at last poll:** commit=45.9/79.2GB commit_free=33.3GB avail=0.6GB run_py=2.3GB ollama_runner=11.4GB answerer=10.2GB signalrgb=0GB
- **Watchdog action:** unloaded minicpm-v4.6:1b (fresh Ollama runner), then relaunched after pre-flight

## Session 6 — completed; results written (auto-recorded by scripts/_nextgqa_official_watchdog.sh)

- **Run commit:** 4fe50e2 (tracked-dirty at launch: 0). Launched 2026-10-01 08:29:23 IST (PID 41740);
  log `C:\Users\Siddanth Anil\iris-bin\nextgqa_registered_run_session6.log`.
- **Pre-flight:** games=0 avail=18665MB gpu_free=10528MiB ollama=200 answerer_build=b10099-1a064ab09 tracked_dirty=0
- **Environment actions before launch:** minicpm-v4.6:1b unloaded after the previous session's stop (fresh Ollama runner)
- **Determinism check:** [DETERMINISM] 20 Arm-F questions answered twice, byte-identical
- **Progress:** checkpoint rows 15831 -> 16659 (+828); last answer line: `[ANSWER 990/990] 9969305386: 8 question(s) done`
- **End:** 2026-10-01 10:51:38 IST, exit code 0.
- **Memory at last poll:** commit=45.0/79.2GB commit_free=34.2GB avail=0.6GB run_py=2.2GB ollama_runner=11.4GB answerer=10.2GB signalrgb=0GB
- **Watchdog action:** none -- run complete
