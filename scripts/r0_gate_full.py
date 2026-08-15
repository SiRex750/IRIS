"""R0 gate experiment -- full run, n=19 (all annotated anomalous videos).

Validity check for Plan A ("codec-domain triage"): does IRIS's codec-based frame
admission beat a uniform-sampling baseline at matched budget?

Copied from scripts/r0_gate_smoke.py (the n=8 record) and modified in exactly
three ways: Arm B gated off behind --arm-b, a chance-baseline table, and a
paired per-video A-vs-C comparison with bootstrap CIs. Metric definitions,
budgets and selectors are byte-identical to the smoke run.

Four arms, each producing a per-video admitted frame set:
  A  production  -- is_retained_tier == 1 from the committed ground-truth CSV
  B  recomputed  -- fresh charon_v.parse_video + action_score (DISABLED by default)
  C  uniform     -- np.round(np.linspace(0, N-1, int(b*N)))
  D  random      -- 5 seeds, mean +/- sd

Metrics (per video, then averaged across the 8):
  M1 window hit rate  -- fraction of gold windows with >=1 admitted frame
  M2 gold recall      -- fraction of gold-window frames admitted
  M3 precision        -- fraction of admitted frames inside a gold window

Outputs: r0_full/results.csv, r0_full/summary.md
READ-ONLY on iris/*.py. Writes only under r0_full/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKTREE = REPO_ROOT.parent / "Iris-ucfvad"
GT_DIR = WORKTREE / "tuning" / "ucfcrime_vad_exp1" / "per_frame_ground_truth_scores"
FROZEN_STATE_PATH = WORKTREE / "tuning" / "frozen_state.json"
VIDEO_ROOT = Path(r"C:\Users\akash\Downloads\Anomaly-Videos-Part-1")
OUT_DIR = REPO_ROOT / "r0_full"

VIDEOS = [
    "Abuse028", "Abuse030",
    "Arrest001", "Arrest007", "Arrest024", "Arrest030", "Arrest039",
    "Arson007", "Arson009", "Arson010", "Arson011", "Arson016",
    "Arson018", "Arson022", "Arson035", "Arson041",
    "Assault006", "Assault010", "Assault011",
]
BUDGETS_PCT = [5.0, 10.0, 10.5, 20.0, 30.0]
SEEDS = [0, 1, 2, 3, 4]

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260805

CAVEAT = (
    "n=19 is the full set of annotated anomalous videos available. Each video is "
    "5.3pp of any aggregate. Pooled AUC is not computed -- the 150 Normal videos "
    "have no local .mp4, and pooled AUC is confounded regardless (90.7% of "
    "negative pairs come from other videos)."
)

V2_DISCLOSURE = (
    "V2 adds a minimum-margin requirement that was NOT in the original "
    "pre-registration. It was added on 5 Aug 2026 after the n=8 run showed the "
    "original Scenario B condition firing on a 0.05pp margin. Both verdicts are "
    "reported so the change is auditable. V1 is the rule as written."
)


# ---------------------------------------------------------------- forward-pass counter

FWD_COUNT = {"n": 0}
FWD_MODE = "UNSET"


def install_forward_counter() -> str:
    """Monkeypatch torch.nn.Module._call_impl to count forward passes.

    Returns "COUNTED" if torch is importable and the patch took, else
    "ZERO_BY_ABSENCE" (torch not present -> a forward pass is impossible).
    """
    try:
        import torch
    except Exception:
        return "ZERO_BY_ABSENCE"

    orig = torch.nn.Module._call_impl

    def counting_call(self, *a, **kw):
        FWD_COUNT["n"] += 1
        return orig(self, *a, **kw)

    torch.nn.Module._call_impl = counting_call
    return "COUNTED"


# ---------------------------------------------------------------- loader

def mask_to_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a 0/1 mask to a list of inclusive [start, end] spans.

    Multi-span aware: a video with several disjoint anomaly windows yields
    several tuples. Returns [] for an all-zero mask.
    """
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def load_video(vid: str) -> dict:
    path = GT_DIR / f"{vid}_x264.csv"
    frame_idx, gt, score, retained = [], [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_idx.append(int(row["frame_idx"]))
            gt.append(int(float(row["ground_truth"])))
            score.append(float(row["action_score_propagated"]))
            retained.append(int(float(row["is_retained_tier"])))
    fi = np.array(frame_idx, dtype=np.int64)
    assert np.array_equal(fi, np.arange(len(fi))), f"{vid}: frame_idx not 0..N-1 contiguous"
    gt_arr = np.array(gt, dtype=np.int8)
    return {
        "video": vid,
        "n_frames": len(fi),
        "gt": gt_arr,
        "windows": mask_to_windows(gt_arr),
        "score": np.array(score, dtype=np.float64),
        "retained_idx": np.flatnonzero(np.array(retained, dtype=np.int8) == 1),
    }


# ---------------------------------------------------------------- metrics

def metrics(admitted: np.ndarray, gt: np.ndarray, windows: list[tuple[int, int]]) -> dict:
    """M1 window hit rate, M2 gold frame recall, M3 precision."""
    admitted = np.unique(np.asarray(admitted, dtype=np.int64))
    adm_set = set(admitted.tolist())

    if windows:
        hits = sum(1 for s, e in windows if any(i in adm_set for i in range(s, e + 1)))
        m1 = hits / len(windows)
    else:
        m1 = float("nan")

    n_gold = int(gt.sum())
    m2 = float(gt[admitted].sum()) / n_gold if n_gold else float("nan")
    m3 = float(gt[admitted].sum()) / len(admitted) if len(admitted) else float("nan")
    return {"M1": m1, "M2": m2, "M3": m3, "n_admitted": int(len(admitted))}


# ---------------------------------------------------------------- arms

def budget_k(n: int, b_pct: float) -> int:
    """Frame count for a budget, per spec: int(b * N) with b a fraction."""
    return max(1, int((b_pct / 100.0) * n))


def arm_c_uniform(n: int, b_pct: float) -> np.ndarray:
    k = budget_k(n, b_pct)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(np.int64))


def arm_d_random(n: int, b_pct: float, seed: int) -> np.ndarray:
    k = budget_k(n, b_pct)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))


def top_k_by_score(score: np.ndarray, k: int) -> np.ndarray:
    """Top-k frames by score; ties broken by ascending frame_idx (stable)."""
    order = np.lexsort((np.arange(len(score)), -score))
    return np.sort(order[:k])


# ---------------------------------------------------------------- Arm B

def run_arm_b(vid: str, n_frames_expected: int) -> dict:
    """Fresh charon_v + action_score ingest. Returns dict with score array or error."""
    cls = "".join(ch for ch in vid if not ch.isdigit())
    mp4 = VIDEO_ROOT / cls / f"{vid}_x264.mp4"
    if not mp4.exists():
        return {"ok": False, "error": f"video file not found: {mp4}"}

    try:
        import iris.charon_v as charon_v
        from iris.action_score import ActionScoreConfig, ActionScoreModule

        frozen = json.loads(FROZEN_STATE_PATH.read_text())["frozen"]
        # NOTE: main's ActionScoreConfig renamed packet_size_weight -> luma_diff_weight
        # (same slot; still weights the codec packet-size residual per its docstring).
        cfg = ActionScoreConfig(
            luma_diff_weight=frozen["packet_size_weight"],
            motion_weight=frozen["motion_weight"],
            luma_entropy_weight=frozen["luma_entropy_weight"],
            peak_distance=frozen["peak_distance"],
            peak_prominence=frozen["peak_prominence"],
            persistence_threshold=frozen["persistence_threshold"],
            max_prominence=frozen["max_prominence"],
        )
        module = ActionScoreModule(config=cfg)

        t0 = time.time()
        output_frames, stats, raw_records = charon_v.parse_video(
            str(mp4), return_stats=True, return_raw=True, full_decode=False,
        )
        elapsed = time.time() - t0
        n_frames = stats["total"]

        retained_idx_set = {f["frame_idx"] for f in output_frames}
        scored_input = sorted(
            (
                {
                    "frame_idx": r["frame_idx"],
                    "packet_size": r.get("packet_size", 0.0),
                    "motion_magnitude": r.get("motion_magnitude", 0.0),
                    "luma_entropy": r.get("luma_entropy", 0.0),
                }
                for r in raw_records
                if r["frame_idx"] in retained_idx_set
            ),
            key=lambda r: r["frame_idx"],
        )
        scored = module.score_all(scored_input)
        by_idx = {s["frame_idx"]: s["action_score"] for s in scored}

        # Hold-forward fill across all frames (same propagation as stage A).
        full = np.zeros(n_frames, dtype=np.float64)
        keys = sorted(by_idx)
        if keys:
            last = by_idx[keys[0]]
            full[: keys[0]] = last
            ptr = 0
            for fi in range(n_frames):
                if ptr < len(keys) and fi == keys[ptr]:
                    last = by_idx[fi]
                    ptr += 1
                full[fi] = last

        return {
            "ok": True,
            "score": full,
            "n_frames": n_frames,
            "retained_idx": np.array(keys, dtype=np.int64),
            "retention_pct": 100.0 * len(keys) / n_frames if n_frames else 0.0,
            "n_frames_match": n_frames == n_frames_expected,
            "elapsed_s": elapsed,
        }
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


# ---------------------------------------------------------------- main

def main() -> int:
    global FWD_MODE
    ap = argparse.ArgumentParser(description="R0 gate full run (n=19).")
    ap.add_argument("--arm-b", action="store_true", default=False,
                    help="Run Arm B (fresh charon_v/action_score ingest). OFF by default; "
                         "Arm B never contributes to the verdict.")
    args = ap.parse_args()

    FWD_MODE = install_forward_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fwd-pass counting] mode={FWD_MODE}", file=sys.stderr)

    data = {}
    for vid in VIDEOS:
        p = GT_DIR / f"{vid}_x264.csv"
        print(f"[gt] {vid}: {'PRESENT' if p.exists() else 'ABSENT'}", file=sys.stderr)
        data[vid] = load_video(vid)

    rows = []          # results.csv rows
    arm_b_results = {}
    arm_b_errors = {}

    for vid in VIDEOS:
        d = data[vid]
        n, gt, wins = d["n_frames"], d["gt"], d["windows"]

        # ---- Arm A: production, at its own natural retention
        a_idx = d["retained_idx"]
        a_ret = 100.0 * len(a_idx) / n
        m = metrics(a_idx, gt, wins)
        rows.append(dict(video=vid, arm="A_production", budget_pct=round(a_ret, 4),
                         seed="", n_frames=n, n_windows=len(wins), n_gold=int(gt.sum()),
                         **m, note="natural retention"))

        # ---- Arm B: recomputed, at Arm A's natural retention (OFF unless --arm-b)
        b = run_arm_b(vid, n) if args.arm_b else None
        if b is None:
            pass
        elif b["ok"]:
            arm_b_results[vid] = b
            k = len(a_idx) if b["n_frames"] == n else budget_k(b["n_frames"], a_ret)
            b_idx = top_k_by_score(b["score"], k)
            gt_b, wins_b = gt, wins
            if b["n_frames"] != n:
                # length mismatch: clip to the shorter of the two, flag it
                lim = min(b["n_frames"], n)
                b_idx = b_idx[b_idx < lim]
                gt_b = gt[:lim]
                wins_b = mask_to_windows(gt_b)
            mb = metrics(b_idx, gt_b, wins_b)
            note = f"topk@{a_ret:.2f}%; charon_retention={b['retention_pct']:.2f}%"
            if not b["n_frames_match"]:
                note += f"; NFRAME_MISMATCH csv={n} charon={b['n_frames']}"
            rows.append(dict(video=vid, arm="B_recomputed", budget_pct=round(a_ret, 4),
                             seed="", n_frames=b["n_frames"], n_windows=len(wins_b),
                             n_gold=int(gt_b.sum()), **mb, note=note))
        else:
            arm_b_errors[vid] = b["error"]
            print(f"[armB ERROR] {vid}\n{b['error']}", file=sys.stderr)

        # ---- Arms C and D at every budget
        for bpct in BUDGETS_PCT:
            c_idx = arm_c_uniform(n, bpct)
            mc = metrics(c_idx, gt, wins)
            rows.append(dict(video=vid, arm="C_uniform", budget_pct=bpct, seed="",
                             n_frames=n, n_windows=len(wins), n_gold=int(gt.sum()),
                             **mc, note=""))

            for s in SEEDS:
                d_idx = arm_d_random(n, bpct, s)
                md = metrics(d_idx, gt, wins)
                rows.append(dict(video=vid, arm="D_random", budget_pct=bpct, seed=s,
                                 n_frames=n, n_windows=len(wins), n_gold=int(gt.sum()),
                                 **md, note=""))

    # ---- write results.csv
    fields = ["video", "arm", "budget_pct", "seed", "n_frames", "n_windows", "n_gold",
              "n_admitted", "M1", "M2", "M3", "note"]
    with (OUT_DIR / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    write_summary(rows, data, arm_b_results, arm_b_errors, arm_b_enabled=args.arm_b)
    print(f"[done] wrote {OUT_DIR/'results.csv'} and {OUT_DIR/'summary.md'}", file=sys.stderr)
    print(f"[fwd-pass] mode={FWD_MODE} count={FWD_COUNT['n']}", file=sys.stderr)
    return 0


def agg(rows, arm, budget=None):
    """Mean of M1/M2/M3 across videos for one arm (averaging seeds within video first)."""
    sel = [r for r in rows if r["arm"] == arm and (budget is None or abs(r["budget_pct"] - budget) < 1e-9)]
    if not sel:
        return None
    per_video = {}
    for r in sel:
        per_video.setdefault(r["video"], []).append(r)
    out = {}
    for m in ("M1", "M2", "M3"):
        vals = [float(np.mean([x[m] for x in v])) for v in per_video.values()]
        out[m] = float(np.mean(vals))
        out[m + "_sd"] = float(np.std(vals, ddof=0))
    out["n_videos"] = len(per_video)
    out["mean_retention"] = float(np.mean([
        np.mean([100.0 * x["n_admitted"] / x["n_frames"] for x in v]) for v in per_video.values()
    ]))
    return out


def bootstrap_ci(deltas, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    """95% percentile bootstrap CI on the mean per-video delta. Resamples videos."""
    d = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def write_summary(rows, data, arm_b_results, arm_b_errors, arm_b_enabled=False):
    L = []
    L.append("# R0 gate -- full run: codec triage vs uniform sampling (n=19)\n")
    L.append(f"> {CAVEAT}\n")
    L.append(f"**Forward-pass counting mode:** `{FWD_MODE}` "
             f"(nn.Module forward passes observed during ingest: **{FWD_COUNT['n']}**)\n")

    # ---- inputs
    L.append("## Inputs\n")
    L.append("| video | N frames | gold frames | gold % | gold windows | prod retention % |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for vid in VIDEOS:
        d = data[vid]
        n = d["n_frames"]
        L.append(f"| {vid} | {n} | {int(d['gt'].sum())} | {100*d['gt'].sum()/n:.1f} | "
                 f"{len(d['windows'])} | {100*len(d['retained_idx'])/n:.2f} |")
    L.append("")

    # ---- arm A
    a = agg(rows, "A_production")
    L.append("## Arm A -- production (`is_retained_tier`), natural retention\n")
    L.append("| arm | retention % | M1 window hit | M2 gold recall | M3 precision |")
    L.append("|---|---:|---:|---:|---:|")
    L.append(f"| A production | {a['mean_retention']:.2f} | {a['M1']:.4f} | {a['M2']:.4f} | {a['M3']:.4f} |")
    b = agg(rows, "B_recomputed") if arm_b_enabled else None
    if b:
        L.append(f"| B recomputed | {b['mean_retention']:.2f} | {b['M1']:.4f} | {b['M2']:.4f} | {b['M3']:.4f} |")
    L.append("")
    if not arm_b_enabled:
        L.append("Arm B is **disabled** in this run (`--arm-b` off). It contributes to no "
                 "table and no verdict.\n")

    # ---- sweep
    L.append("## Matched-budget sweep -- Arms C (uniform) and D (random, 5 seeds)\n")
    L.append("| budget % | arm | M1 window hit | M2 gold recall | M3 precision |")
    L.append("|---:|---|---:|---:|---:|")
    for bp in BUDGETS_PCT:
        c = agg(rows, "C_uniform", bp)
        dd = agg(rows, "D_random", bp)
        L.append(f"| {bp} | C uniform | {c['M1']:.4f} | {c['M2']:.4f} | {c['M3']:.4f} |")
        L.append(f"| {bp} | D random (mean+/-sd across videos) | {dd['M1']:.4f} +/- {dd['M1_sd']:.4f} | "
                 f"{dd['M2']:.4f} +/- {dd['M2_sd']:.4f} | {dd['M3']:.4f} +/- {dd['M3_sd']:.4f} |")
    L.append("")

    # ---- head-to-head at 10.5
    c105 = agg(rows, "C_uniform", 10.5)
    d105 = agg(rows, "D_random", 10.5)
    L.append("## Head-to-head at matched budget (10.5%, Arm A's natural retention)\n")
    L.append("| arm | M1 | M2 | M3 |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| A production (~{a['mean_retention']:.2f}%) | {a['M1']:.4f} | {a['M2']:.4f} | {a['M3']:.4f} |")
    if b:
        L.append(f"| B recomputed | {b['M1']:.4f} | {b['M2']:.4f} | {b['M3']:.4f} |")
    L.append(f"| C uniform 10.5% | {c105['M1']:.4f} | {c105['M2']:.4f} | {c105['M3']:.4f} |")
    L.append(f"| D random 10.5% | {d105['M1']:.4f} | {d105['M2']:.4f} | {d105['M3']:.4f} |")
    L.append("")
    L.append(f"**Delta A - C (pp) at 10.5%:** M1 {100*(a['M1']-c105['M1']):+.2f}, "
             f"M2 {100*(a['M2']-c105['M2']):+.2f}, M3 {100*(a['M3']-c105['M3']):+.2f}\n")

    # ---- chance baseline (change #2)
    L.append("## Chance baseline -- how far is Arm A from a no-information selector?\n")
    L.append("A selector carrying no label information yields M2 = its budget fraction and "
             "M3 = the gold base rate, by construction. Columns 6 and 9 are the distance "
             "from that null.\n")
    L.append("| video | gold % | A retention % | A M2 | retention/100 | A M2 - retention | "
             "A M3 | gold frac | A M3 - gold frac |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    m2_excess, m3_excess = [], []
    for vid in VIDEOS:
        ra = next(r for r in rows if r["video"] == vid and r["arm"] == "A_production")
        d = data[vid]
        gold_frac = float(d["gt"].sum()) / d["n_frames"]
        ret_frac = ra["n_admitted"] / d["n_frames"]
        e2 = ra["M2"] - ret_frac
        e3 = ra["M3"] - gold_frac
        m2_excess.append(e2)
        m3_excess.append(e3)
        L.append(f"| {vid} | {100*gold_frac:.1f} | {100*ret_frac:.2f} | {ra['M2']:.4f} | "
                 f"{ret_frac:.4f} | {e2:+.4f} | {ra['M3']:.4f} | {gold_frac:.4f} | {e3:+.4f} |")
    L.append("")
    L.append(f"**Mean (A M2 - retention) = {np.mean(m2_excess):+.4f}** "
             f"(sd {np.std(m2_excess, ddof=1):.4f}, "
             f"min {np.min(m2_excess):+.4f}, max {np.max(m2_excess):+.4f})")
    L.append(f"**Mean (A M3 - gold frac) = {np.mean(m3_excess):+.4f}** "
             f"(sd {np.std(m3_excess, ddof=1):.4f}, "
             f"min {np.min(m3_excess):+.4f}, max {np.max(m3_excess):+.4f})\n")

    # ---- paired per-video comparison A vs C @10.5% (change #3)
    L.append("## Paired per-video comparison -- Arm A vs Arm C at 10.5%\n")
    L.append("| video | gold % | A M2 | C M2 | dM2 pp | A M3 | C M3 | dM3 pp |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    dm2, dm3 = [], []
    for vid in VIDEOS:
        ra = next(r for r in rows if r["video"] == vid and r["arm"] == "A_production")
        rc = next(r for r in rows if r["video"] == vid and r["arm"] == "C_uniform"
                  and abs(r["budget_pct"] - 10.5) < 1e-9)
        gold_frac = float(data[vid]["gt"].sum()) / data[vid]["n_frames"]
        p2 = 100 * (ra["M2"] - rc["M2"])
        p3 = 100 * (ra["M3"] - rc["M3"])
        dm2.append(p2)
        dm3.append(p3)
        L.append(f"| {vid} | {100*gold_frac:.1f} | {ra['M2']:.4f} | {rc['M2']:.4f} | {p2:+.3f} | "
                 f"{ra['M3']:.4f} | {rc['M3']:.4f} | {p3:+.3f} |")
    L.append("")

    stats_pair = {}
    for label, arr in (("dM2", dm2), ("dM3", dm3)):
        arr_np = np.asarray(arr)
        pos = int((arr_np > 1e-9).sum())
        neg = int((arr_np < -1e-9).sum())
        tie = int((np.abs(arr_np) < 1e-9).sum())
        lo, hi = bootstrap_ci(arr_np)
        stats_pair[label] = dict(mean=float(arr_np.mean()), sd=float(arr_np.std(ddof=1)),
                                 pos=pos, neg=neg, tie=tie, lo=lo, hi=hi)

    L.append("| metric | mean d (pp) | sd (pp) | A > C | A < C | exact tie | 95% bootstrap CI (pp) |")
    L.append("|---|---:|---:|---:|---:|---:|---|")
    for label in ("dM2", "dM3"):
        s = stats_pair[label]
        L.append(f"| {label} | {s['mean']:+.3f} | {s['sd']:.3f} | {s['pos']} | {s['neg']} | "
                 f"{s['tie']} | [{s['lo']:+.3f}, {s['hi']:+.3f}] |")
    L.append("")
    L.append(f"Bootstrap: {BOOTSTRAP_N:,} percentile resamples over the {len(VIDEOS)} videos, "
             f"seed {BOOTSTRAP_SEED}.\n")
    for label in ("dM2", "dM3"):
        s = stats_pair[label]
        if s["lo"] <= 0.0 <= s["hi"]:
            L.append(f"**The 95% CI for {label} spans zero** "
                     f"([{s['lo']:+.3f}, {s['hi']:+.3f}] pp): this run does not establish "
                     f"that Arm A differs from uniform sampling on {label[1:]}.")
        else:
            L.append(f"The 95% CI for {label} excludes zero "
                     f"([{s['lo']:+.3f}, {s['hi']:+.3f}] pp).")
    L.append("")

    # ---- divergence check
    if not arm_b_enabled:
        L.append("## Divergence check -- production (A) vs recomputed (B)\n")
        L.append("Not applicable: Arm B is disabled in this run.\n")
    elif not arm_b_results:
        L.append("## Divergence check -- production (A) vs recomputed (B)\n")
        L.append("Arm B did not run on any video; no divergence table.\n")
    else:
        L.append("## Divergence check -- production (A) vs recomputed (B)\n")
        L.append("| video | A retention % | B retention % | dRet pp | dM1 pp | dM2 pp | dM3 pp | flag |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for vid in VIDEOS:
            ra = next((r for r in rows if r["video"] == vid and r["arm"] == "A_production"), None)
            rb = next((r for r in rows if r["video"] == vid and r["arm"] == "B_recomputed"), None)
            if not rb:
                L.append(f"| {vid} | {ra['budget_pct']:.2f} | -- | -- | -- | -- | -- | ARM B FAILED |")
                continue
            bret = arm_b_results[vid]["retention_pct"]
            d1 = 100 * (rb["M1"] - ra["M1"]); d2 = 100 * (rb["M2"] - ra["M2"]); d3 = 100 * (rb["M3"] - ra["M3"])
            dret = bret - ra["budget_pct"]
            flag = "DIVERGENT >5pp" if max(abs(d1), abs(d2), abs(d3)) > 5.0 else "ok"
            if "NFRAME_MISMATCH" in rb.get("note", ""):
                flag += " + NFRAME_MISMATCH"
            L.append(f"| {vid} | {ra['budget_pct']:.2f} | {bret:.2f} | {dret:+.2f} | "
                     f"{d1:+.2f} | {d2:+.2f} | {d3:+.2f} | {flag} |")
        L.append("")
    if arm_b_errors:
        L.append("### Arm B errors (verbatim)\n")
        for vid, err in arm_b_errors.items():
            L.append(f"**{vid}**\n\n```\n{err.strip()}\n```\n")

    # ---- verdict (Arm A vs Arm C @10.5% only; Arm B excluded by design)
    L.append("## Verdict\n")

    def c_m1_at(bp):
        r = agg(rows, "C_uniform", bp)
        return r["M1"] if r else 0.0
    uniform_needs = next((bp for bp in BUDGETS_PCT if c_m1_at(bp) >= 0.99), None)

    d1 = 100 * (a["M1"] - c105["M1"])
    d2 = 100 * (a["M2"] - c105["M2"])
    d3 = 100 * (a["M3"] - c105["M3"])
    half_budget_win = (
        a["M1"] >= 0.99 and uniform_needs is not None
        and a["mean_retention"] <= 0.5 * uniform_needs
    )

    L.append("| IRIS arm | dM1 pp | dM2 pp | dM3 pp | half-budget win |")
    L.append("|---|---:|---:|---:|---|")
    L.append(f"| A (production) | {d1:+.2f} | {d2:+.2f} | {d3:+.2f} | {half_budget_win} |")
    L.append("")
    L.append("(All deltas are aggregate Arm A minus Arm C uniform at b = 10.5%. "
             "Arm B is excluded from the verdict by design.)\n")
    L.append(f"- Lowest Arm C budget in the sweep reaching M1 >= 0.99: "
             f"{uniform_needs if uniform_needs is not None else 'none in sweep'}%\n")

    # V1 -- the pre-registered rule, unchanged.
    sa = (d2 > 5.0) or half_budget_win
    tie_m1 = abs(d1) <= 1.0 and a["M1"] >= 0.95 and c105["M1"] >= 0.95
    sb = (not sa) and tie_m1 and (d2 > 0.0 or d3 > 0.0)
    sc = (d1 <= 0.0) and (d2 <= 0.0) and (d3 <= 0.0)

    if sa:
        v1, v1_why = "**Scenario A**", "Arm A beat uniform by >5pp on M2, or won the half-budget test."
    elif sc:
        v1, v1_why = "**Scenario C**", "uniform matched or beat Arm A on all three metrics."
    elif sb:
        v1, v1_why = "**Scenario B**", ("M1 is tied at ceiling and Arm A wins on M2/M3. "
                                        "The margin is not part of this rule.")
    else:
        v1, v1_why = "**No scenario matched cleanly**", "no pre-registered condition was satisfied."

    L.append("### V1 -- as pre-registered (verbatim rule)\n")
    L.append(f"{v1} -- {v1_why}\n")

    # V2 -- margin-qualified Scenario B.
    s2, s3 = stats_pair["dM2"], stats_pair["dM3"]
    margin_ok = (
        (abs(s2["mean"]) > 1.0 and not (s2["lo"] <= 0.0 <= s2["hi"]))
        or (abs(s3["mean"]) > 1.0 and not (s3["lo"] <= 0.0 <= s3["hi"]))
    )
    sb2 = sb and margin_ok
    if sa:
        v2, v2_why = "**Scenario A**", v1_why
    elif sc:
        v2, v2_why = "**Scenario C**", v1_why
    elif sb2:
        v2, v2_why = "**Scenario B**", ("M1 tied at ceiling, Arm A wins on M2/M3, and the "
                                        "per-video margin clears 1.0pp with a CI excluding zero.")
    elif sb:
        v2, v2_why = "**No scenario matched**", (
            "Scenario B fired under V1 but fails the added margin test: per-video mean "
            f"dM2 = {s2['mean']:+.3f} pp (CI [{s2['lo']:+.3f}, {s2['hi']:+.3f}]), "
            f"dM3 = {s3['mean']:+.3f} pp (CI [{s3['lo']:+.3f}, {s3['hi']:+.3f}]). "
            "Neither exceeds 1.0pp with a CI excluding zero.")
    else:
        v2, v2_why = "**No scenario matched cleanly**", "no condition was satisfied."

    L.append("### V2 -- margin-qualified\n")
    L.append(f"{V2_DISCLOSURE}\n")
    L.append(f"{v2} -- {v2_why}\n")

    # ---- kill criterion
    L.append("## Kill criterion\n")
    L.append(f"Lowest swept budget at which Arm C reaches M1 >= 0.99: "
             f"**{uniform_needs if uniform_needs is not None else 'none in sweep'}%** "
             f"(Arm C M1 at 5%/10%/10.5%/20%/30% = "
             + ", ".join(f"{c_m1_at(bp):.4f}" for bp in BUDGETS_PCT) + ").\n")
    if c105["M1"] >= 0.99:
        L.append(f"**TRIGGERED.** Arm C (uniform) reaches M1 = {c105['M1']:.4f} >= 0.99 at b = 10.5%. "
                 "M1 (window hit rate) cannot be a headline metric regardless of which scenario applies -- "
                 "uniform sampling already saturates it.\n")
    else:
        L.append(f"Not triggered: Arm C M1 at 10.5% = {c105['M1']:.4f} (< 0.99).\n")

    L.append("## Caveat\n")
    L.append(f"{CAVEAT}\n")

    (OUT_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
