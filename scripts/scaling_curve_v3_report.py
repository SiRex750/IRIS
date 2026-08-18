"""Build eval_results/scaling_curve_v3.{json,md} -- ledger C1.5 CI task.

Assembles:
  1. Survivor census over the 350-clip UCF population: container-frame proxy
     for all 350 (eval_results/ucf_inventory_full350.json, read-only probe,
     NOT survivor N) + actual measured survivor N/scene-count/duration for
     every clip with a built index_cache (eval_results/_ci_survivor_census.json,
     read from each .npz's config_snapshot -- no re-ingest).
  2. Binned latency: existing v2/textquery corpus + this task's census/latency
     extension (scripts/scaling_curve_ci.py), real CLIP TEXT queries only.
  3. Fits with clip-level bootstrap CIs (both completions-only and
     censored-as-lower-bound variants), via scaling_curve_ci.fit_report.
  4. Guards: shortcut_pct==0, salience-weight provenance (scaling_curve_ci.run_guards).
  5. PRINT-SAFE interpretation, pre-registered BEFORE this script reads the
     fit numbers (see docstring below the imports) -- flat CI and
     scene_sparse CI must not overlap, AND scene_sparse's upper CI bound
     must sit below 1.0, else the lead falls back to C1.3 tractability
     divergence + C2 block-diagonal.

Read-only against index_cache/*.npz and the two raw JSON corpora; writes
ONLY eval_results/scaling_curve_v3.json and eval_results/scaling_curve_v3.md.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO / "scripts"))

import scaling_curve_ci as ci
import scaling_curve_v2 as base

FULL350_INVENTORY = REPO / "eval_results" / "ucf_inventory_full350.json"
SURVIVOR_CENSUS = REPO / "eval_results" / "_ci_survivor_census.json"

OUT_JSON = REPO / "eval_results" / "scaling_curve_v3.json"
OUT_MD = REPO / "eval_results" / "scaling_curve_v3.md"

# ---------------------------------------------------------------------------
# PRE-REGISTERED interpretation rule (written before reading fit results):
#   PRINT-SAFE iff flat's and scene_sparse's headline (N>=1102) 95% CIs on k
#   do NOT overlap, AND scene_sparse's CI upper bound < 1.0.
#   Otherwise: NOT PRINT-SAFE -> paper leads with C1.3 (tractability
#   divergence) + C2 (block-diagonal), not a clean exponent ratio.
# ---------------------------------------------------------------------------


def decade_histogram(values: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for v in values:
        if v <= 0:
            key = "<=0"
        else:
            d = int(math.floor(math.log10(v)))
            lo, hi = 10 ** d, 10 ** (d + 1) - 1
            key = f"{lo}-{hi}"
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0].split("-")[0]) if "-" in kv[0] else -1))


def _label_to_total_duration_s(pop_records: list[dict]) -> dict[str, float]:
    """Maps clip label (video filename stem, minus _x264) -> total video
    duration (nb_frames_meta / fps from the read-only container probe), so
    survivor-timespan duration (below) can be reported alongside total video
    duration -- they differ because survivors are a sparse subset of frames."""
    out = {}
    for r in pop_records:
        if not r.get("open_ok") or not r.get("nb_frames_meta") or not r.get("fps"):
            continue
        stem = Path(r["path"]).stem
        label = stem[:-5] if stem.endswith("_x264") else stem
        out[label] = r["nb_frames_meta"] / r["fps"]
    return out


def survivor_census_section() -> dict:
    pop = json.loads(FULL350_INVENTORY.read_text(encoding="utf-8"))
    pop_records = pop["records"]
    container_frames = [r["nb_frames_meta"] for r in pop_records if r.get("open_ok") and r.get("nb_frames_meta")]
    total_duration_by_label = _label_to_total_duration_s(pop_records)

    measured = json.loads(SURVIVOR_CENSUS.read_text(encoding="utf-8"))["records"]
    measured_ok = [r for r in measured if "error" not in r and r.get("graph_edge_mode") == "fully_connected"]
    measured_mismatched = [r for r in measured if "error" not in r and r.get("graph_edge_mode") != "fully_connected"]
    ns = sorted(r["n_survivors"] for r in measured_ok)

    def _stat(f, xs):
        return f(xs) if xs else None

    return {
        "population_size": pop["total_files"],
        "population_container_frame_proxy": {
            "note": "Container (encoded) frame count is a WEAK proxy for survivor N -- content/motion "
                    "density, not duration, drives admission (see scripts/scaling_curve_ci.py docstring; "
                    "e.g. Arson019: 126,553 container frames -> 13,506 survivors, ~10.7%, while other "
                    "clips of similar length survive at very different rates). Reported here ONLY to "
                    "describe the population's raw-length spread, NOT as a survivor-N estimate.",
            "n": len(container_frames),
            "min": min(container_frames), "median": sorted(container_frames)[len(container_frames)//2],
            "max": max(container_frames),
        },
        "measured_survivor_n": {
            "note": "Actual post-selection survivor N (+ scene count, duration) for every clip with a "
                    "built index_cache -- i.e. this fit's actual support. NOT all 350 population clips "
                    "have been ingested (ingesting all 350 was out of scope/budget for this task); this "
                    "is the honest subset the fit rests on, not the full population.",
            "n_clips": len(measured_ok),
            "min": _stat(min, ns), "median": _stat(lambda x: x[len(x)//2], ns), "max": _stat(max, ns),
            "histogram_by_decade": decade_histogram(ns),
            "excluded_graph_edge_mode_mismatch": [
                {"label": r["label"], "graph_edge_mode": r["graph_edge_mode"], "n_survivors": r["n_survivors"]}
                for r in measured_mismatched
            ],
            "per_clip": [
                {"label": r["label"], "n_survivors": r["n_survivors"], "n_scenes": r["n_scenes"],
                 "survivor_timespan_s": r["duration_s"],
                 "total_video_duration_s": total_duration_by_label.get(r["label"]),
                 "salience_weights": r["salience_weights"],
                 "weights_match_frozen_default": r["weights_match_frozen_default"]}
                for r in sorted(measured_ok, key=lambda r: r["n_survivors"])
            ],
        },
    }


def bin_occupancy_section(all_clips: list[dict]) -> list[dict]:
    buckets = ci.build_buckets(all_clips)
    out = []
    for t in ci.BUCKET_TARGETS_PRIORITY:
        clips = buckets[t]
        out.append({
            "target_N": t,
            "n_clips": len(clips),
            "meets_target_5": len(clips) >= 5,
            "clips": [{"label": c["label"], "n_survivors": c["n_survivors"], "source": c["source"]} for c in clips],
        })
    return out


def per_clip_raw_latency(all_clips: list[dict]) -> list[dict]:
    out = []
    for c in sorted(all_clips, key=lambda c: c["n_survivors"] or 0):
        row = {"label": c["label"], "dataset": c["dataset"], "n_survivors": c["n_survivors"], "source": c["source"]}
        for arm in ("flat", "scene_sparse"):
            a = c[arm]
            row[arm] = {
                "outcome": a["outcome_normalized"], "median_total_retrieval_s": a["median_total_retrieval_s"],
                "query_embed_s": a["query_embed_s"], "shortcut_pct": a["shortcut_pct"], "censored": a["censored"],
            }
        out.append(row)
    return out


def censoring_table(all_clips: list[dict]) -> list[dict]:
    rows = []
    for c in all_clips:
        for arm in ("flat", "scene_sparse"):
            a = c[arm]
            if a["censored"]:
                rows.append({"label": c["label"], "arm": arm, "n_survivors": c["n_survivors"],
                             "outcome": a["outcome_normalized"],
                             "wall_time_cap_sec": base.WALL_CAP_SEC,
                             "note": "true latency >= cap; excluded from completions-only fit, "
                                     "included as a floor value in the lower-bound fit variant"})
    return rows


def provenance_section() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    dirty = bool(status.strip())
    harness_files = [
        "scripts/scaling_curve_ci.py", "scripts/_scaling_curve_ci_census_worker.py",
        "scripts/scaling_curve_v2.py", "scripts/_scaling_curve_v2_worker.py",
        "scripts/scaling_curve_v3_report.py",
    ]
    hashes = {}
    for f in harness_files:
        r = subprocess.run(["git", "hash-object", f], cwd=REPO, capture_output=True, text=True)
        hashes[f] = r.stdout.strip() if r.returncode == 0 else None
    return {
        "git_head": head,
        "git_dirty": dirty,
        "git_dirty_file_count": len(status.strip().splitlines()) if dirty else 0,
        "harness_config_hash": hashes,
        "query_source": ci.QUERY_SOURCE,
        "n_queries_per_clip": base.N_Q,
        "watchdog": {"wall_time_cap_sec": base.WALL_CAP_SEC, "mem_cap_bytes": base.MEM_CAP_BYTES},
        "bootstrap": {"n_boot": ci.N_BOOT, "seed": ci.BOOT_SEED, "method": "resample WITHIN each N-bucket, refit, "
                      "repeat (clip-level resampling, not bucket-mean resampling)"},
    }


def print_safe_verdict(fit_report_out: dict) -> dict:
    key = f"N_ge_{ci.HEADLINE_MIN_N}_headline"
    flat_fit = fit_report_out["arms"]["flat"]["fits"][key]
    ss_fit = fit_report_out["arms"]["scene_sparse"]["fits"][key]

    def _ci(fit):
        if fit is None or fit.get("ci_low") is None:
            return None
        return (fit["ci_low"], fit["ci_high"])

    flat_ci, ss_ci = _ci(flat_fit), _ci(ss_fit)
    reasons = []
    if flat_ci is None or ss_ci is None:
        overlap = None
        reasons.append("insufficient data for one or both arms' bootstrap CI at the headline range")
    else:
        overlap = not (flat_ci[1] < ss_ci[0] or ss_ci[1] < flat_ci[0])
        if overlap:
            reasons.append(f"flat CI {flat_ci} and scene_sparse CI {ss_ci} OVERLAP")
    ss_below_1 = (ss_ci[1] < 1.0) if ss_ci else None
    if ss_ci is not None and not ss_below_1:
        reasons.append(f"scene_sparse CI upper bound {ss_ci[1]:.3f} does NOT sit below 1.0")

    print_safe = (overlap is False) and (ss_below_1 is True)
    return {
        "flat_headline_ci": flat_ci, "scene_sparse_headline_ci": ss_ci,
        "cis_overlap": overlap, "scene_sparse_ci_upper_below_1": ss_below_1,
        "PRINT_SAFE": print_safe,
        "reasons": reasons if not print_safe else ["CIs are disjoint and scene_sparse's CI sits below 1.0"],
        "fallback": None if print_safe else
            "NOT PRINT-SAFE per pre-registration -- lead with C1.3 (tractability divergence: flat exceeds "
            "the 3600s wall-time budget past N~6,559 while scene_sparse remains tractable through the "
            "measured range) + C2 (block-diagonal), not a clean exponent ratio.",
    }


def main():
    existing = ci._load_existing_corpus()
    census = ci._load_census_corpus()
    all_clips = ci.dedupe_clips(existing + census)

    fits = ci.fit_report(all_clips)
    guards = ci.run_guards(all_clips)
    verdict = print_safe_verdict(fits)

    report = {
        "task": "ledger C1.5 -- confidence intervals on query-latency scaling exponents",
        "provenance": provenance_section(),
        "survivor_census": survivor_census_section(),
        "bin_occupancy": bin_occupancy_section(all_clips),
        "per_clip_raw_latency": per_clip_raw_latency(all_clips),
        "censoring_table": censoring_table(all_clips),
        "fits": fits,
        "guards": guards,
        "interpretation": verdict,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))

    write_markdown(report)
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}", file=sys.stderr)


def write_markdown(r: dict) -> None:
    lines = []
    lines.append("# Query-latency scaling exponents with confidence intervals (ledger C1.5)\n")
    v = r["interpretation"]
    verdict_str = "**PRINT-SAFE**" if v["PRINT_SAFE"] else "**NOT PRINT-SAFE**"
    lines.append(f"## Verdict: {verdict_str}\n")
    lines.append(f"- flat headline (N>={ci.HEADLINE_MIN_N}) 95% CI on k: {v['flat_headline_ci']}")
    lines.append(f"- scene_sparse headline (N>={ci.HEADLINE_MIN_N}) 95% CI on k: {v['scene_sparse_headline_ci']}")
    lines.append(f"- CIs overlap: {v['cis_overlap']}; scene_sparse CI upper bound < 1.0: {v['scene_sparse_ci_upper_below_1']}")
    if not v["PRINT_SAFE"]:
        lines.append(f"\n{v['fallback']}\n")
    lines.append("")

    sc = r["survivor_census"]
    lines.append("## Survivor census\n")
    lines.append(f"- Population (UCF Anomaly-Part-1 + Testing_Normal_Videos_Anomaly): "
                 f"{sc['population_size']} clips (container-frame proxy only, NOT survivor N: "
                 f"min={sc['population_container_frame_proxy']['min']}, "
                 f"median={sc['population_container_frame_proxy']['median']}, "
                 f"max={sc['population_container_frame_proxy']['max']})")
    m = sc["measured_survivor_n"]
    lines.append(f"- Measured survivor N (actual fit support): n={m['n_clips']} clips, "
                 f"min={m['min']}, median={m['median']}, max={m['max']}")
    lines.append("- Histogram by decade of N:")
    for k2, v2 in m["histogram_by_decade"].items():
        lines.append(f"  - {k2}: {v2}")
    if m["excluded_graph_edge_mode_mismatch"]:
        lines.append(f"- Excluded (graph_edge_mode mismatch, not fully_connected): "
                     f"{[e['label'] for e in m['excluded_graph_edge_mode_mismatch']]}")
    lines.append("")

    lines.append("## Bin occupancy (target >=5 clips/bin)\n")
    lines.append("| target N | n clips | meets target | clips |")
    lines.append("|---:|---:|:---:|---|")
    for b in r["bin_occupancy"]:
        lines.append(f"| {b['target_N']} | {b['n_clips']} | {'yes' if b['meets_target_5'] else 'NO -- shortfall'} | "
                     f"{', '.join(c['label'] for c in b['clips'])} |")
    lines.append("")

    lines.append("## Fits (clip-level bootstrap, seed=%d, B=%d)\n" % (ci.BOOT_SEED, ci.N_BOOT))
    for arm in ("flat", "scene_sparse"):
        lines.append(f"### {arm}\n")
        lines.append("| fit range | n_points | k | 95% CI | R^2 |")
        lines.append("|---|---:|---:|---|---:|")
        for name, f in r["fits"]["arms"][arm]["fits"].items():
            if name == "with_censored_as_lower_bound":
                continue
            if f is None or f.get("point_fit") is None:
                lines.append(f"| {name} | - | - | insufficient data | - |")
                continue
            pf = f["point_fit"]
            ci_str = f"[{f['ci_low']:.3f}, {f['ci_high']:.3f}]" if f.get("ci_low") is not None else "n/a"
            lines.append(f"| {name} | {pf['n_points']} | {pf['exponent_k']:.3f} | {ci_str} | {pf.get('r_squared'):.4f} |")
        cb = r["fits"]["arms"][arm]["fits"]["with_censored_as_lower_bound"]
        if cb["point_fit"]:
            pf = cb["point_fit"]
            lines.append(f"\n**Censored-as-lower-bound** (n_completions={cb['n_completions']}, "
                         f"n_censored_substituted_at_cap={cb['n_censored_substituted_at_cap']}): "
                         f"k={pf['exponent_k']:.3f} (R^2={pf.get('r_squared'):.4f}) -- {cb['interpretation']}")
        censored = r["fits"]["arms"][arm]["censored_lower_bounds"]
        if censored:
            lines.append(f"\nCensored (timed_out) points, excluded from completions-only fits: "
                         f"{[f'{c['label']}(N={c['n_survivors']})' for c in censored]}")
        lines.append("")

    lines.append("## Guards\n")
    g = r["guards"]
    lines.append(f"- shortcut_pct guard: {'PASS' if g['shortcut_pct_guard']['PASS'] else 'FAIL -- see violations'} "
                 f"({g['shortcut_pct_guard']['n_scene_sparse_completed_checked']} scene_sparse completions checked)")
    if g["shortcut_pct_guard"]["violations"]:
        lines.append(f"  - VIOLATIONS: {g['shortcut_pct_guard']['violations']}")
    lines.append(f"- salience weights: {g['salience_weights_guard']['note']}")
    lines.append("")

    lines.append("## Censoring table\n")
    if r["censoring_table"]:
        lines.append("| label | arm | N | outcome | cap (s) |")
        lines.append("|---|---|---:|---|---:|")
        for c in r["censoring_table"]:
            lines.append(f"| {c['label']} | {c['arm']} | {c['n_survivors']} | {c['outcome']} | {c['wall_time_cap_sec']} |")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Provenance\n")
    p = r["provenance"]
    lines.append(f"- git HEAD: `{p['git_head']}` (dirty: {p['git_dirty']}, {p['git_dirty_file_count']} changed files)")
    lines.append(f"- query_source: {p['query_source']}, n_queries_per_clip: {p['n_queries_per_clip']}")
    lines.append(f"- watchdog: wall_time_cap_sec={p['watchdog']['wall_time_cap_sec']}, "
                 f"mem_cap_bytes={p['watchdog']['mem_cap_bytes']}")
    lines.append(f"- bootstrap: n_boot={p['bootstrap']['n_boot']}, seed={p['bootstrap']['seed']}, "
                 f"{p['bootstrap']['method']}")
    lines.append("- harness config hashes:")
    for f, h in p["harness_config_hash"].items():
        lines.append(f"  - `{f}`: `{h}`")
    lines.append("")

    lines.append("Full raw data: `eval_results/scaling_curve_v3.json`.\n")
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
