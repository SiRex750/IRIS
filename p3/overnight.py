"""Paper 3 overnight driver (one detached process, no input needed). Runs, in order:
  1. sintel_rerun   p3/sweep/run_sweep.py with SWEEP_OUT=p3/sweep_rerun (the pre-registered sweep, single-threaded
                    MV extraction; p3/sweep/DEVIATIONS.md 1)
  2. compare        p3/sweep_rerun/compare.py -> COMPARISON.md                       (needs 1)
  3. virat_collect  p3/virat/run_virat.py -> p3/virat_confirm/ (frames 300-599, 9 arms; PREREGISTRATION d62753f)
  4. virat_analyse  p3/virat_confirm/analyze_confirm.py -> summary.md                (needs 3)
Each stage's output goes to its own log; p3/overnight.log gets timestamped stage events and health lines.
A failed stage is logged and the next stage that does not depend on it runs. Free disk is checked before each stage
and every 30 s during it: under 15 GB the running stage is killed and the job stops (remaining stages not run).
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

P3 = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(P3)
LOG = os.path.join(P3, "overnight.log")
PY = sys.executable
DISK_MIN_GB = 15.0
POLL_S = 30
RERUN = os.path.join(P3, "sweep_rerun")
CONFIRM = os.path.join(P3, "virat_confirm")
CONFIRM_ARMS = ("x264_crf12,x264_crf45,x264_qp24,x264_qp24_bf2_pb1,nvenc_qp18,nvenc_qp23,nvenc_qp45,nvenc_qp28,"
                "nvenc_qp28_bf2_bqeq")


def log(stage, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{stage}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except Exception:
        pass


def free_gb():
    return shutil.disk_usage(REPO).free / 1e9


def kill_tree(pid):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def run_stage(name, cmd, out_log, timeout_h, env=None):
    """Run one stage to completion. Returns ("ok" | "failed" | "disk" | "timeout", detail)."""
    if free_gb() < DISK_MIN_GB:
        return "disk", f"free {free_gb():.1f} GB < {DISK_MIN_GB:.0f} GB before start"
    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    log(name, f"START: {' '.join(cmd)}  (output: {os.path.relpath(out_log, REPO)}; free {free_gb():.1f} GB)")
    t0 = time.time()
    with open(out_log, "ab") as fh:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})
        log(name, f"pid {p.pid}")
        while True:
            try:
                rc = p.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            if free_gb() < DISK_MIN_GB:
                kill_tree(p.pid)
                p.wait()
                return "disk", f"free {free_gb():.1f} GB < {DISK_MIN_GB:.0f} GB after {(time.time() - t0) / 60:.0f} min; killed"
            if time.time() - t0 > timeout_h * 3600:
                kill_tree(p.pid)
                p.wait()
                return "timeout", f"exceeded {timeout_h} h; killed"
    dt = (time.time() - t0) / 60
    return ("ok" if rc == 0 else "failed"), f"exit code {rc} after {dt:.1f} min"


def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [ln.rstrip() for ln in f.readlines()[-n:]]
    except OSError:
        return []


# --------------------------------------------------------------------------- per-stage health
def health_sintel():
    m = json.load(open(os.path.join(RERUN, "manifest.json")))
    ok = os.path.exists(os.path.join(RERUN, "results.csv")) and "end_time" in m
    return ok, (f"runs {m.get('n_runs')}, failures {m.get('n_failures')}, check mismatches {m.get('n_mismatches')}, "
                f"runtime {m.get('runtime_s', 0) / 3600:.2f} h, end_time {m.get('end_time')}")


def health_compare():
    p = os.path.join(RERUN, "COMPARISON.md")
    if not os.path.exists(p):
        return False, "COMPARISON.md missing"
    lines = open(p, encoding="utf-8").read().splitlines()
    keep = [ln for ln in lines if ln.startswith("**Any verdict changed") or ln.startswith("- Encodes byte-identical")
            or ln.startswith("- Rows with any differing value") or ln.startswith("- Block record files compared")]
    return True, " | ".join(keep)


def health_collect():
    m = json.load(open(os.path.join(CONFIRM, "manifest.json")))
    hs = m.get("health_summary", {})
    raft = m.get("raft", {})
    n_raft = sum(r.get("status") == "ok" for r in raft.get("clips", {}).values())
    ok = m.get("status") == "done"
    return ok, (f"status {m.get('status')}; clips {len(m.get('clips', {}))}, excluded {len(m.get('excluded_clips', {}))} "
                f"({', '.join(sorted(m.get('excluded_clips', {})))}); records {hs.get('records')}, ok {hs.get('ok')}, "
                f"failed {hs.get('failed')}, with mismatches {hs.get('with_mismatches')}, skipped {hs.get('skipped')}; "
                f"RAFT {raft.get('status')} ({n_raft} clips ok); free {m.get('disk', {}).get('free_end_gb', 0):.1f} GB")


def health_analysis():
    p = os.path.join(CONFIRM, "verdicts.csv")
    if not os.path.exists(p) or not os.path.exists(os.path.join(CONFIRM, "summary.md")):
        return False, "verdicts.csv / summary.md missing"
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return True, " | ".join(f"{r['key']}: {r['value']} -> {r['verdict']}" for r in rows)


STAGES = [
    {"name": "sintel_rerun", "needs": None, "timeout_h": 5, "out": os.path.join(RERUN, "run.log"),
     "cmd": [PY, "-u", os.path.join(P3, "sweep", "run_sweep.py")], "env": {"SWEEP_OUT": RERUN}, "health": health_sintel},
    {"name": "compare", "needs": "sintel_rerun", "timeout_h": 2, "out": os.path.join(RERUN, "compare.log"),
     "cmd": [PY, "-u", os.path.join(RERUN, "compare.py")], "health": health_compare},
    {"name": "virat_collect", "needs": None, "timeout_h": 10, "out": os.path.join(CONFIRM, "run.log"),
     "cmd": [PY, "-u", os.path.join(P3, "virat", "run_virat.py"), "--out", CONFIRM, "--start", "300", "--frames", "300",
             "--min-frames", "150", "--exclude", "VIRAT_S_000002", "--arms", CONFIRM_ARMS],
     "health": health_collect},
    {"name": "virat_analyse", "needs": "virat_collect", "timeout_h": 3, "out": os.path.join(CONFIRM, "analysis.log"),
     "cmd": [PY, "-u", os.path.join(CONFIRM, "analyze_confirm.py")], "health": health_analysis},
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:  # keep the machine awake while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    T0 = time.time()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "p3"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    log("driver", f"OVERNIGHT START pid {os.getpid()}, HEAD {head}, python {PY}, free {free_gb():.1f} GB"
        + (f"; uncommitted under p3: {dirty!r}" if dirty else ""))
    status = {}
    stopped = False
    for st in STAGES:
        name = st["name"]
        if stopped:
            status[name] = "not run (disk guard)"
            log(name, "NOT RUN: job stopped by the disk guard")
            continue
        if st["needs"] and status.get(st["needs"]) != "ok":
            status[name] = f"skipped ({st['needs']} {status.get(st['needs'])})"
            log(name, f"SKIPPED: depends on {st['needs']}, which is {status.get(st['needs'])}")
            continue
        try:
            res, detail = run_stage(name, st["cmd"], st["out"], st["timeout_h"], st.get("env"))
        except Exception as e:
            res, detail = "failed", f"driver error {e!r}: {traceback.format_exc()[-800:]}"
        if res == "disk":
            stopped = True
        if res == "ok":
            try:
                ok, h = st["health"]()
            except Exception as e:
                ok, h = False, f"health check raised {e!r}"
            log(name, f"HEALTH: {h}")
            if not ok:
                res = "failed"
                detail += "; health check failed"
        if name == "virat_collect" and os.path.exists(os.path.join(CONFIRM, "STOP_DISK")):
            res, stopped = "disk", True
            detail += "; run_virat disk guard tripped (STOP_DISK)"
        status[name] = res
        log(name, f"END {res.upper()}: {detail}")
        if res != "ok":
            for ln in tail(st["out"]):
                log(name, f"  | {ln}")
    log("driver", f"OVERNIGHT {'STOPPED (disk guard)' if stopped else 'DONE'} after {(time.time() - T0) / 3600:.2f} h; "
        + ", ".join(f"{k}: {v}" for k, v in status.items()) + f"; free {free_gb():.1f} GB")


if __name__ == "__main__":
    main()
