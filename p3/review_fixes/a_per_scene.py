"""Review follow-up A: per-scene tables for both CCTV tests. REPORTING ONLY - no new verdicts.

Reads p3/virat/results.csv (first test) and p3/virat_confirm/results.csv (confirmatory test), tau = 1 px, and
recomputes each pre-registered clip-level comparison exactly as analyze_virat.py / analyze_confirm.py do, then
groups clips by scene (clip-name prefix VIRAT_S_XXXXYY). Writes p3/review_fixes/per_scene.md and per_scene.csv.
No results file is modified.

Scene-level count: a scene "passes" a comparison if strictly more than half of its counted clips pass (ties do not
pass). STALE comparisons count only the scene's eligible clips (as pre-registered); scenes with no eligible clip are
shown as n/a and left out of the scene count. FLIP comparisons count all clips.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
TAU = 1.0


def scene_of(clip):
    return clip[:14]  # VIRAT_S_XXXXYY


def series(r, arm, group, col):
    return r[(r.arm == arm) & (r.group == group)].set_index("clip")[col]


def build(path, comps):
    """comps: (key, label, hi arm, hi group, lo arm, lo group, column, threshold, eligible-only)."""
    res = pd.read_csv(path)
    r = res[res.tau == TAU]
    clips = sorted(r["clip"].unique())
    elig = set(r.loc[r.eligible.astype(bool), "clip"])
    clip_tab = pd.DataFrame(index=clips)
    clip_tab["scene"] = [scene_of(c) for c in clips]
    clip_tab["eligible"] = [c in elig for c in clips]
    for key, _, hi, ghi, lo, glo, col, thr, _e in comps:
        d = (series(r, hi, ghi, col) - series(r, lo, glo, col)).reindex(clips)
        clip_tab[key + " diff"] = d
        clip_tab[key] = d > thr
    rows = []
    for s, d in clip_tab.groupby("scene"):
        row = {"scene": s, "clips": len(d), "eligible": int(d.eligible.sum())}
        for key, _, *_x, e_only in comps:
            dd = d[d.eligible] if e_only else d
            dd = dd[dd[key + " diff"].notna()]
            k, n = int(dd[key].sum()), len(dd)
            row[key] = f"{k}/{n}" if n else "n/a"
            row[key + " scene"] = ("pass" if k > n / 2 else "no") if n else "n/a"
            row[key + " med diff"] = dd[key + " diff"].median() if n else np.nan
        rows.append(row)
    sc = pd.DataFrame(rows)
    totals = {}
    for key, *_x, e_only in comps:
        dd = clip_tab[clip_tab.eligible] if e_only else clip_tab
        dd = dd[dd[key + " diff"].notna()]
        s_ok = sc[sc[key + " scene"] != "n/a"]
        totals[key] = dict(clips_pass=int(dd[key].sum()), clips_n=len(dd),
                           scenes_pass=int((s_ok[key + " scene"] == "pass").sum()), scenes_n=len(s_ok),
                           med=dd[key + " diff"].median())
    return clip_tab, sc, totals


def md(df, fmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("–" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


FIRST = (
    ("V1f x264", "FLIP(x264 CRF 45) > FLIP(CRF 18), all clips", "x264_crf45", "d1", "x264_crf18", "d1", "flip", 0, False),
    ("V1f NV", "FLIP(NVENC QP 45) > FLIP(QP 18), all clips", "nvenc_qp45", "d1", "nvenc_qp18", "d1", "flip", 0, False),
    ("V1s x264", "STALE(x264 CRF 45) - STALE(CRF 12) > 0.0005, eligible", "x264_crf45", "d1", "x264_crf12", "d1",
     "stale", 0.0005, True),
    ("V1s NV", "STALE(NVENC QP 45) - STALE(QP 18) > 0.0005, eligible", "nvenc_qp45", "d1", "nvenc_qp18", "d1",
     "stale", 0.0005, True),
    ("V2 x264", "STALE(x264 QP 24 bf2, B) - STALE(QP 24 bf0, d1) > 0.0005, eligible", "x264_qp24_bf2_pb1", "B",
     "x264_qp24", "d1", "stale", 0.0005, True),
    ("V2 NV", "STALE(NVENC QP 28 bf2 B=P, B) - STALE(QP 28 bf0, d1) > 0.0005, eligible", "nvenc_qp28_bf2_bqeq", "B",
     "nvenc_qp28", "d1", "stale", 0.0005, True),
)
CONFIRM = (
    ("C1", "STALE_MOV(x264 CRF 45) - STALE_MOV(CRF 12) > 0.01, eligible", "x264_crf45", "d1", "x264_crf12", "d1",
     "stale_mov", 0.01, True),
    ("C2", "STALE_MOV(NVENC QP 45) - STALE_MOV(QP 18) > 0.01, eligible", "nvenc_qp45", "d1", "nvenc_qp18", "d1",
     "stale_mov", 0.01, True),
    ("C3", "STALE_MOV(x264 QP 24 B) - STALE_MOV(QP 24 bf0) > 0.01, eligible", "x264_qp24_bf2_pb1", "B", "x264_qp24",
     "d1", "stale_mov", 0.01, True),
    ("C4", "STALE_MOV(NVENC QP 28 B=P) - STALE_MOV(QP 28 bf0) > 0.01, eligible", "nvenc_qp28_bf2_bqeq", "B",
     "nvenc_qp28", "d1", "stale_mov", 0.01, True),
    ("C5", "FLIP_vsNV18(NVENC QP 45) > FLIP_vsNV18(QP 23), all clips", "nvenc_qp45", "d1", "nvenc_qp23", "d1",
     "flip_nv18", 0, False),
)


def section(title, path, comps, L, csv_rows):
    ct, sc, tot = build(path, comps)
    L.append(f"## {title}\n")
    L.append(f"Source: `{os.path.relpath(path, P3)}`, tau = 1 px. {len(ct)} clips in {len(sc)} scenes; "
             f"{int(ct.eligible.sum())} eligible.\n")
    L.append("Comparisons (same definitions and thresholds as the pre-registration):\n")
    for key, lab, *_ in comps:
        L.append(f"- **{key}**: {lab}")
    L.append("\n### Totals: clip counts vs scene counts\n")
    L.append(md(pd.DataFrame([{"comparison": k, "clips passing": f"{v['clips_pass']}/{v['clips_n']}",
                               "scenes passing (majority of clips)": f"{v['scenes_pass']}/{v['scenes_n']}",
                               "median diff over clips": v["med"]} for k, v in tot.items()]), "{:.5f}"))
    L.append("\n### Per scene: clips passing (k/n), scene majority, scene median of the per-clip difference\n")
    cols = ["scene", "clips", "eligible"]
    for key, *_ in comps:
        cols += [key, key + " scene", key + " med diff"]
    L.append(md(sc[cols], "{:.5f}"))
    L.append("")
    for _, row in sc.iterrows():
        csv_rows.append({"test": title.split(":")[0], **row.to_dict()})
    return tot


def main():
    L = ["# Per-scene tables for both CCTV tests (review follow-up A)\n",
         "**Reporting only.** These tables regroup the existing per-clip results by camera scene "
         "(clip-name prefix VIRAT_S_XXXXYY). They are not new verdicts and do not replace the pre-registered "
         "clip-level verdicts. The scene-level count (a scene passes if strictly more than half of its counted clips "
         "pass; ties do not pass) is an exploratory summary that was not pre-registered. STALE comparisons count "
         "eligible clips only, as pre-registered; a scene with no eligible clip is n/a. Generated by "
         "`p3/review_fixes/a_per_scene.py`.\n"]
    rows = []
    t1 = section("First CCTV test: VIRAT frames 0-299", os.path.join(P3, "virat", "results.csv"), FIRST, L, rows)
    t2 = section("Confirmatory CCTV test: VIRAT frames 300-599", os.path.join(P3, "virat_confirm", "results.csv"),
                 CONFIRM, L, rows)
    with open(os.path.join(HERE, "per_scene.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "per_scene.csv"), index=False)
    for k, v in {**t1, **t2}.items():
        print(k, f"clips {v['clips_pass']}/{v['clips_n']}", f"scenes {v['scenes_pass']}/{v['scenes_n']}",
              f"med {v['med']:.5f}")


if __name__ == "__main__":
    main()
