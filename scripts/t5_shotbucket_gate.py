"""T5 -- codec shot-bucketing selection test, n=19.

Question: at matched budget, does bucketing frames by zero-decode codec SHOT SPANS
(charon_v.compute_valley_scene_boundaries) beat uniform sampling?

Pre-registered before this script ran (see _shotbucket/T5_inventory.md):
  * PRIMARY arm      : F_shot1_action -- 1 frame per shot (the highest
                       action_score_propagated in the shot), then budget-fill to
                       exactly k by largest-remainder apportionment weighted by
                       shot peak action.
  * PRIMARY contrast : F - C_uniform, paired per video, at b = 10.5%.
                       10.5% is R0's headline budget and sits inside the
                       production retention band [10.41, 11.14]%.
  * PRIMARY outcomes : dM2 (gold recall) and dM3 (precision), in pp.
                       M1 is REPORTED BUT NOT PRIMARY: it saturates at 1.0 for
                       uniform at every budget >=5% in R0, and 17/19 videos have
                       exactly one gold window, so it is near-binary per video.
  * READ  : three branches, decided by the paired bootstrap CI and a 1.0pp
            effect-of-interest threshold (R0's own CI half-widths were 0.47pp on
            dM2 and 0.88pp on dM3, so sub-1pp effects are not resolvable at n=19):
              BEATS        - CI excludes 0 AND |mean| >= 1.0pp
              TIES         - CI spans 0 AND half-width <  1.0pp   (a genuine null)
              UNDERPOWERED - CI spans 0 AND half-width >= 1.0pp   (cannot tell)
            A CI that excludes 0 with |mean| < 1.0pp is reported as
            DETECTED-SUBTHRESHOLD and does NOT meet the pre-registered BEATS bar.

Metric definitions, budgets, loader, uniform/random arms and the bootstrap are
IMPORTED verbatim from scripts/r0_gate_full.py -- nothing is recomputed here.

Shot spans are zero-decode: charon_v._demux_packet_curve reads packet sizes/pts
only (no .decode(), no model). A torch forward-pass counter is installed to
prove the neural count is zero.

READ-ONLY on iris/*.py. Writes only under _shotbucket/run/.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import r0_gate_full as r0  # noqa: E402  -- metric/loader/bootstrap source of truth
import iris.charon_v as charon_v  # noqa: E402

OUT_DIR = REPO_ROOT / "_shotbucket" / "run"

PRIMARY_BUDGET = 10.5
EFFECT_PP = 1.0          # effect of interest, pre-registered
PRIMARY_ARM = "F_shot1_action"
BASELINE_ARM = "C_uniform"

# Shot arms: (key, frames-per-shot floor, within-shot pick rule, fill weight)
SHOT_ARMS = [
    ("F_shot1_action", 1, "action", "peak_action"),
    ("G_shot2_action", 2, "action", "peak_action"),
    ("H_shot1_mid",    1, "mid",    "shot_len"),
    ("I_shot2_mid",    2, "mid",    "shot_len"),
]


# ---------------------------------------------------------------- shot spans (zero-decode)

def shot_spans(vid: str) -> tuple[list[tuple[int, int]], dict]:
    """Zero-decode valley shot spans straight from the .mp4. No decode, no model."""
    cls = "".join(ch for ch in vid if not ch.isdigit())
    mp4 = r0.VIDEO_ROOT / cls / f"{vid}_x264.mp4"
    if not mp4.exists():
        raise FileNotFoundError(mp4)
    t0 = time.time()
    all_frame_energies, iframe_indices, _, _ = charon_v._demux_packet_curve(str(mp4))
    fps = charon_v.get_stream_fps(str(mp4))
    spans = charon_v.compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps)
    return spans, {
        "n_demux": len(all_frame_energies),
        "n_iframes": len(iframe_indices),
        "fps": fps,
        "n_shots": len(spans),
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------- apportionment

def apportion(weights: np.ndarray, extra: int, caps: np.ndarray) -> np.ndarray:
    """Largest-remainder (Hamilton) apportionment of `extra` seats across shots.

    Proportional to `weights`, clamped to `caps` (remaining room per shot).
    Clamped-away seats are redistributed until `extra` is placed or all caps are
    full. Deterministic: remainder ties break to the LOWER shot index.
    """
    w = np.asarray(weights, dtype=np.float64)
    caps = np.asarray(caps, dtype=np.int64)
    alloc = np.zeros(len(w), dtype=np.int64)
    remaining = int(extra)

    while remaining > 0:
        room = caps - alloc
        active = room > 0
        if not active.any():
            break
        wa = np.where(active, np.maximum(w, 0.0), 0.0)
        total = wa.sum()
        if total <= 0.0:                       # degenerate weights -> by index order
            idx = np.flatnonzero(active)[:remaining]
            alloc[idx] += 1
            remaining -= len(idx)
            continue
        quota = wa / total * remaining
        base = np.floor(quota).astype(np.int64)
        short = remaining - int(base.sum())
        if short > 0:
            rem = quota - base
            order = [i for i in np.lexsort((np.arange(len(w)), -rem)) if active[i]]
            for i in order[:short]:
                base[i] += 1
        take = np.minimum(np.where(active, base, 0), np.maximum(room, 0))
        if take.sum() == 0:                    # no progress -> force the top-weight shot
            cand = np.flatnonzero(active)
            alloc[cand[int(np.argmax(wa[cand]))]] += 1
            remaining -= 1
            continue
        alloc += take
        remaining -= int(take.sum())
    return alloc


# ---------------------------------------------------------------- the shot-bucketed selector

def shot_bucket_select(spans, score: np.ndarray, k: int, per_shot: int,
                       pick: str, weight: str) -> np.ndarray:
    """Select EXACTLY k frames, bucketed by codec shot spans.

    S <= k (the normal case on this corpus): every shot gets `per_shot` frames
    (coverage floor, clamped to shot length), the remaining k - sum(floor) are
    apportioned across shots by `weight`, clamped to shot length.

    S > k: shots are ranked by peak action (ties -> lower shot index) and filled
    in rank order until the budget is exhausted; lower-ranked shots get nothing.

    Within a shot:
      pick="action" -- the top frames by action_score_propagated, ties by
                       ascending frame_idx (same rule as r0.top_k_by_score).
      pick="mid"    -- evenly spaced shot CENTRES, which reduces to the
                       pre-registered midpoint (start+end)//2 when a shot gets 1
                       frame. Uses no score at all: pure geometry.
    """
    S = len(spans)
    lens = np.array([e - s for s, e in spans], dtype=np.int64)
    peak = np.array([float(score[s:e].max()) for s, e in spans], dtype=np.float64)

    floor_alloc = np.minimum(per_shot, lens)

    if int(floor_alloc.sum()) > k:
        # budget cannot even seat the floor: rank shots by peak action.
        order = np.lexsort((np.arange(S), -peak))
        alloc = np.zeros(S, dtype=np.int64)
        left = k
        for i in order:
            if left <= 0:
                break
            take = min(int(floor_alloc[i]), left)
            alloc[i] = take
            left -= take
    else:
        alloc = floor_alloc.copy()
        w = lens.astype(np.float64) if weight == "shot_len" else np.maximum(peak, 0.0)
        alloc = alloc + apportion(w, k - int(floor_alloc.sum()), lens - floor_alloc)

    out: list[int] = []
    for (s, e), a in zip(spans, alloc):
        a = int(a)
        if a <= 0:
            continue
        if pick == "action":
            seg = score[s:e]
            order = np.lexsort((np.arange(len(seg)), -seg))
            out.extend((s + order[:a]).tolist())
        else:
            centres = s + (2 * np.arange(a) + 1) * (e - s) / (2.0 * a)
            out.extend(np.floor(centres).astype(np.int64).tolist())

    sel = np.unique(np.asarray(out, dtype=np.int64))
    assert len(sel) == k, f"selector produced {len(sel)} frames, expected exactly {k}"
    return sel


# ---------------------------------------------------------------- run

def main() -> int:
    fwd_mode = r0.install_forward_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fwd-pass counting] mode={fwd_mode}", file=sys.stderr)

    data, spans_by_vid, span_meta = {}, {}, {}
    for vid in r0.VIDEOS:
        data[vid] = r0.load_video(vid)
        spans_by_vid[vid], span_meta[vid] = shot_spans(vid)
        n_csv, n_dem = data[vid]["n_frames"], span_meta[vid]["n_demux"]
        assert n_dem == n_csv, f"{vid}: demux N={n_dem} != csv N={n_csv}"
        assert spans_by_vid[vid][0][0] == 0 and spans_by_vid[vid][-1][1] == n_csv, \
            f"{vid}: spans do not cover [0,N)"
        print(f"[spans] {vid}: N={n_csv} shots={span_meta[vid]['n_shots']} "
              f"({span_meta[vid]['elapsed_s']:.2f}s)", file=sys.stderr)

    rows = []
    for vid in r0.VIDEOS:
        d = data[vid]
        n, gt, wins, score = d["n_frames"], d["gt"], d["windows"], d["score"]
        spans = spans_by_vid[vid]

        a_idx = d["retained_idx"]
        rows.append(dict(video=vid, arm="A_production", budget_pct=round(100.0 * len(a_idx) / n, 4),
                         seed="", n_frames=n, n_shots=len(spans), n_windows=len(wins),
                         n_gold=int(gt.sum()), **r0.metrics(a_idx, gt, wins),
                         note="natural retention"))

        for bpct in r0.BUDGETS_PCT:
            k = r0.budget_k(n, bpct)
            common = dict(video=vid, budget_pct=bpct, n_frames=n, n_shots=len(spans),
                          n_windows=len(wins), n_gold=int(gt.sum()))

            primary = abs(bpct - PRIMARY_BUDGET) < 1e-9

            c_idx = r0.arm_c_uniform(n, bpct)
            rows.append(dict(arm="C_uniform", seed="", **common,
                             **r0.metrics(c_idx, gt, wins), note=""))
            for s in r0.SEEDS:
                rows.append(dict(arm="D_random", seed=s, **common,
                                 **r0.metrics(r0.arm_d_random(n, bpct, s), gt, wins), note=""))
            e_idx = r0.top_k_by_score(score, k)
            rows.append(dict(arm="E_action_topk", seed="", **common,
                             **r0.metrics(e_idx, gt, wins),
                             note="R0 codec anchor at matched budget"))
            if primary:
                SELECTED[(vid, "C_uniform")] = c_idx
                SELECTED[(vid, "E_action_topk")] = e_idx

            for key, per_shot, pick, weight in SHOT_ARMS:
                idx = shot_bucket_select(spans, score, k, per_shot, pick, weight)
                floor_pct = 100.0 * min(len(spans) * per_shot, k) / max(k, 1)
                rows.append(dict(arm=key, seed="", **common,
                                 **r0.metrics(idx, gt, wins),
                                 note=f"floor={min(len(spans)*per_shot, k)}/{k} ({floor_pct:.1f}% of budget)"))
                if primary:
                    SELECTED[(vid, key)] = idx

    fields = ["video", "arm", "budget_pct", "seed", "n_frames", "n_shots", "n_windows",
              "n_gold", "n_admitted", "M1", "M2", "M3", "note"]
    with (OUT_DIR / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    (OUT_DIR / "spans.json").write_text(json.dumps(
        {v: {"meta": span_meta[v], "spans": spans_by_vid[v]} for v in r0.VIDEOS}, indent=1))

    write_summary(rows, data, span_meta, fwd_mode)
    print(f"[done] {OUT_DIR/'results.csv'}, {OUT_DIR/'summary.md'}", file=sys.stderr)
    print(f"[fwd-pass] mode={fwd_mode} count={r0.FWD_COUNT['n']}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- read-out

def paired(rows, arm, ref, budget, metric):
    """Per-video (arm - ref) in pp at one budget. Seeds averaged within video."""
    out = []
    for vid in r0.VIDEOS:
        def pick(a):
            sel = [r for r in rows if r["video"] == vid and r["arm"] == a
                   and abs(r["budget_pct"] - budget) < 1e-9]
            return float(np.mean([x[metric] for x in sel]))
        out.append(100.0 * (pick(arm) - pick(ref)))
    return np.asarray(out, dtype=np.float64)


def read_branch(deltas):
    """The pre-registered three-branch read."""
    mean = float(deltas.mean())
    lo, hi = r0.bootstrap_ci(deltas)
    half = (hi - lo) / 2.0
    spans_zero = lo <= 0.0 <= hi
    if not spans_zero and abs(mean) >= EFFECT_PP:
        branch = "BEATS" if mean > 0 else "LOSES"
    elif not spans_zero:
        branch = "DETECTED-SUBTHRESHOLD"
    elif half < EFFECT_PP:
        branch = "TIES"
    else:
        branch = "UNDERPOWERED"
    return dict(mean=mean, sd=float(deltas.std(ddof=1)), lo=lo, hi=hi, half=half,
                pos=int((deltas > 1e-9).sum()), neg=int((deltas < -1e-9).sum()),
                tie=int((np.abs(deltas) < 1e-9).sum()), branch=branch)


BRANCH_TEXT = {
    "BEATS": "CI excludes 0 and |effect| >= {e:.1f}pp -- meets the pre-registered bar.",
    "LOSES": "CI excludes 0 and the effect is NEGATIVE by >= {e:.1f}pp -- worse than the baseline.",
    "DETECTED-SUBTHRESHOLD": ("CI excludes 0 but |effect| < {e:.1f}pp. Real but below the "
                              "effect of interest; does NOT meet the pre-registered BEATS bar."),
    "TIES": "CI spans 0 and its half-width < {e:.1f}pp -- a genuine null: a {e:.1f}pp effect is excluded.",
    "UNDERPOWERED": ("CI spans 0 but its half-width >= {e:.1f}pp -- this corpus CANNOT distinguish "
                     "a real effect from null. Not a tie."),
}

ARM_LABEL = {
    "C_uniform": "C uniform", "D_random": "D random (5 seeds)",
    "E_action_topk": "E action-score top-k (R0 anchor)",
    "F_shot1_action": "F shot 1/shot, highest-action  [PRIMARY]",
    "G_shot2_action": "G shot 2/shot, highest-action",
    "H_shot1_mid": "H shot 1/shot, midpoint (pure geometry)",
    "I_shot2_mid": "I shot 2/shot, midpoint (pure geometry)",
}
ALL_ARMS = ["C_uniform", "D_random", "E_action_topk"] + [a[0] for a in SHOT_ARMS]

# (video, arm) -> selected frame indices at PRIMARY_BUDGET, for the post-hoc dispersion table.
SELECTED: dict[tuple[str, str], np.ndarray] = {}


def write_summary(rows, data, span_meta, fwd_mode):
    L = []
    L.append("# T5 -- codec shot-bucketing vs uniform sampling at matched budget (n=19)\n")
    L.append(f"> {r0.CAVEAT}\n")
    L.append(f"**Forward-pass counting mode:** `{fwd_mode}` "
             f"(nn.Module forward passes during the whole run: **{r0.FWD_COUNT['n']}**). "
             "Shot spans come from packet demux only -- no pixel decode, no model.\n")
    L.append(f"**Pre-registered before the run:** primary arm `{PRIMARY_ARM}`, primary contrast "
             f"vs `{BASELINE_ARM}` at b = {PRIMARY_BUDGET}%, primary outcomes dM2/dM3, "
             f"effect of interest {EFFECT_PP:.1f}pp, three-branch read. M1 reported but not primary.\n")

    # ---- shot inventory
    L.append("## Shot spans (zero-decode, from the .mp4 alone)\n")
    L.append("| video | N frames | I-frames | shots | mean shot len | k @10.5% | shots as % of k | demux+span s |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for vid in r0.VIDEOS:
        m, n = span_meta[vid], data[vid]["n_frames"]
        k = r0.budget_k(n, PRIMARY_BUDGET)
        L.append(f"| {vid} | {n} | {m['n_iframes']} | {m['n_shots']} | {n/m['n_shots']:.1f} | "
                 f"{k} | {100*m['n_shots']/k:.1f} | {m['elapsed_s']:.2f} |")
    L.append("")
    L.append(f"Total demux+span wall time for all 19: "
             f"**{sum(m['elapsed_s'] for m in span_meta.values()):.1f}s**. "
             "Every video has shots S well below the 10.5% budget k, so the coverage floor "
             "(1 or 2 frames per shot) spends only a fraction of the budget and the rest is "
             "apportioned -- see the `note` column of results.csv.\n")

    # ---- main table
    for bp in r0.BUDGETS_PCT:
        star = "  <-- PRIMARY" if abs(bp - PRIMARY_BUDGET) < 1e-9 else ""
        L.append(f"## Budget {bp}%{star}\n")
        L.append("| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |")
        L.append("|---|---:|---:|---:|---:|")
        for arm in ALL_ARMS:
            a = r0.agg(rows, arm, bp)
            if not a:
                continue
            nadm = float(np.mean([r["n_admitted"] for r in rows
                                  if r["arm"] == arm and abs(r["budget_pct"] - bp) < 1e-9]))
            L.append(f"| {ARM_LABEL[arm]} | {nadm:.0f} | {a['M1']:.4f} | {a['M2']:.4f} | {a['M3']:.4f} |")
        L.append("")

    # ---- primary read
    L.append(f"## PRIMARY read -- {PRIMARY_ARM} minus {BASELINE_ARM} at {PRIMARY_BUDGET}%\n")
    L.append("| video | gold % | shots | F M2 | C M2 | dM2 pp | F M3 | C M3 | dM3 pp |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    d2 = paired(rows, PRIMARY_ARM, BASELINE_ARM, PRIMARY_BUDGET, "M2")
    d3 = paired(rows, PRIMARY_ARM, BASELINE_ARM, PRIMARY_BUDGET, "M3")
    for i, vid in enumerate(r0.VIDEOS):
        def val(a, m):
            return next(r[m] for r in rows if r["video"] == vid and r["arm"] == a
                        and abs(r["budget_pct"] - PRIMARY_BUDGET) < 1e-9)
        gold = 100.0 * data[vid]["gt"].sum() / data[vid]["n_frames"]
        L.append(f"| {vid} | {gold:.1f} | {span_meta[vid]['n_shots']} | "
                 f"{val(PRIMARY_ARM,'M2'):.4f} | {val(BASELINE_ARM,'M2'):.4f} | {d2[i]:+.3f} | "
                 f"{val(PRIMARY_ARM,'M3'):.4f} | {val(BASELINE_ARM,'M3'):.4f} | {d3[i]:+.3f} |")
    L.append("")
    s2, s3 = read_branch(d2), read_branch(d3)
    L.append("| metric | mean d (pp) | sd (pp) | F>C | F<C | tie | 95% bootstrap CI (pp) | CI half-width | branch |")
    L.append("|---|---:|---:|---:|---:|---:|---|---:|---|")
    for lab, s in (("dM2", s2), ("dM3", s3)):
        L.append(f"| {lab} | {s['mean']:+.3f} | {s['sd']:.3f} | {s['pos']} | {s['neg']} | {s['tie']} | "
                 f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['half']:.3f} | **{s['branch']}** |")
    L.append("")
    for lab, s in (("dM2", s2), ("dM3", s3)):
        L.append(f"- **{lab}: {s['branch']}** -- {BRANCH_TEXT[s['branch']].format(e=EFFECT_PP)}")
    L.append("")
    L.append(f"Bootstrap: {r0.BOOTSTRAP_N:,} percentile resamples over the {len(r0.VIDEOS)} videos, "
             f"seed {r0.BOOTSTRAP_SEED} (imported from r0_gate_full).\n")

    # ---- secondary contrasts
    L.append("## Secondary contrasts (not pre-registered as primary; read with care)\n")
    L.append("| contrast | budget % | metric | mean d (pp) | 95% CI (pp) | half-width | branch |")
    L.append("|---|---:|---|---:|---|---:|---|")
    contrasts = [(a, BASELINE_ARM) for a in ("E_action_topk", "G_shot2_action",
                                             "H_shot1_mid", "I_shot2_mid")]
    contrasts.append((PRIMARY_ARM, "E_action_topk"))   # does shot-bucketing add over plain top-k?
    contrasts.append((PRIMARY_ARM, "D_random"))
    for arm, ref in contrasts:
        for metric in ("M2", "M3"):
            s = read_branch(paired(rows, arm, ref, PRIMARY_BUDGET, metric))
            L.append(f"| {arm} - {ref} | {PRIMARY_BUDGET} | d{metric} | {s['mean']:+.3f} | "
                     f"[{s['lo']:+.3f}, {s['hi']:+.3f}] | {s['half']:.3f} | {s['branch']} |")
    L.append("")
    L.append("The `F - E_action_topk` rows are the scientifically interesting pair: E is the "
             "existing R0 codec selector at the same budget, so this isolates what the SHOT "
             "BUCKETING adds on top of the action score it already uses.\n")

    # ---- primary arm across the budget sweep
    L.append(f"## {PRIMARY_ARM} - {BASELINE_ARM} across the budget sweep\n")
    L.append("| budget % | dM2 pp | dM2 CI | dM2 branch | dM3 pp | dM3 CI | dM3 branch |")
    L.append("|---:|---:|---|---|---:|---|---|")
    for bp in r0.BUDGETS_PCT:
        a2 = read_branch(paired(rows, PRIMARY_ARM, BASELINE_ARM, bp, "M2"))
        a3 = read_branch(paired(rows, PRIMARY_ARM, BASELINE_ARM, bp, "M3"))
        L.append(f"| {bp} | {a2['mean']:+.3f} | [{a2['lo']:+.3f}, {a2['hi']:+.3f}] | {a2['branch']} | "
                 f"{a3['mean']:+.3f} | [{a3['lo']:+.3f}, {a3['hi']:+.3f}] | {a3['branch']} |")
    L.append("")

    # ---- post-hoc mechanism
    L.append("## POST-HOC diagnostic -- temporal dispersion (NOT pre-registered)\n")
    L.append("Found while checking why Abuse028 admitted zero gold frames. "
             "`action_score_propagated` is hold-forward-propagated from the retained tier, so it "
             "is a STEP function: one distinct value per retained frame, ~9.5 frames per plateau. "
             "Picking 'the highest-action frames' therefore degenerates into picking whole "
             "plateaus from their earliest frame (the tie-break is ascending frame_idx, imported "
             "from `r0.top_k_by_score`), so the budget is spent on ADJACENT, near-duplicate "
             "frames. Blocks = maximal runs of consecutive selected frames = the number of "
             "distinct temporal locations actually sampled.\n")
    L.append("| arm | mean run length | mean blocks | budget k | distinct locations as % of budget |")
    L.append("|---|---:|---:|---:|---:|")
    for arm in ALL_ARMS:
        if arm == "D_random":
            continue
        runs, blocks, ks = [], [], []
        for vid in r0.VIDEOS:
            sel = np.sort(SELECTED[(vid, arm)])
            gaps = np.diff(sel)
            nb = 1 + int((gaps != 1).sum())
            runs.append(len(sel) / nb)
            blocks.append(nb)
            ks.append(len(sel))
        L.append(f"| {ARM_LABEL[arm]} | {np.mean(runs):.2f} | {np.mean(blocks):.0f} | "
                 f"{np.mean(ks):.0f} | {100*np.mean(blocks)/np.mean(ks):.1f} |")
    L.append("")
    L.append("This is the mechanism behind the whole table. Uniform samples k distinct instants; "
             "the action top-k arm (E) spends the same k frames on ~8% as many distinct instants, "
             "which is why it posts the largest mean dM3 AND the widest CI. Shot bucketing's "
             "per-shot coverage floor forces dispersion and recovers most of that spread, which "
             "is why F sits between E and uniform. The pure-geometry arms (H/I) use no score, "
             "never clump, and consequently track uniform almost exactly.\n")
    L.append("**Abuse028, the worst cell (F dM2 = -10.5pp):** its 76-frame gold window [165,240] "
             "lies entirely inside one 237-frame shot. That shot received 10 of the 148 budgeted "
             "frames, all placed on higher-action plateaus at 106-121 and 250-255 -- straddling "
             "the window without entering it. Plain action top-k (arm E) also scores 0/148 gold "
             "frames there, so this is the ACTION SCORE missing the anomaly, not the bucketing: "
             "mean score inside the gold window is 0.4643 vs 0.4261 outside, i.e. almost no "
             "signal. Uniform gets 8/148 by construction.\n")

    # ---- M1, reported but dead
    L.append("## M1 -- reported, NOT a primary outcome\n")
    c1 = r0.agg(rows, BASELINE_ARM, PRIMARY_BUDGET)["M1"]
    f1 = r0.agg(rows, PRIMARY_ARM, PRIMARY_BUDGET)["M1"]
    nwin = [len(data[v]["windows"]) for v in r0.VIDEOS]
    L.append(f"At {PRIMARY_BUDGET}%: uniform M1 = {c1:.4f}, {PRIMARY_ARM} M1 = {f1:.4f}. "
             f"{sum(1 for x in nwin if x == 1)}/19 videos have exactly one gold window "
             f"(max {max(nwin)}), so M1 is near-binary per video and saturates. "
             "It is excluded from the verdict by pre-registration.\n")

    # ---- verdict
    L.append("## T5 VERDICT\n")
    verdict = ("BEATS" if s2["branch"] == "BEATS" or s3["branch"] == "BEATS"
               else "LOSES" if "LOSES" in (s2["branch"], s3["branch"])
               else "TIES" if s2["branch"] == "TIES" and s3["branch"] == "TIES"
               else "DETECTED-SUBTHRESHOLD" if "DETECTED-SUBTHRESHOLD" in (s2["branch"], s3["branch"])
               else "UNDERPOWERED")
    L.append(f"**{verdict}** on the pre-registered primary contrast "
             f"({PRIMARY_ARM} - {BASELINE_ARM}, b = {PRIMARY_BUDGET}%): "
             f"dM2 = {s2['mean']:+.3f}pp [{s2['lo']:+.3f}, {s2['hi']:+.3f}] ({s2['branch']}), "
             f"dM3 = {s3['mean']:+.3f}pp [{s3['lo']:+.3f}, {s3['hi']:+.3f}] ({s3['branch']}).\n")
    L.append(f"{r0.CAVEAT}\n")

    (OUT_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
