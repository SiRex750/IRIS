"""Insert the segmentation figure (paper Figure 2) into paper/latex/iris_mmsys27.tex.

Builds the pgfplots block from eval_results/fig_valley_Abuse028.json (written by
scripts/fig_valley_dump.py) so the plotted data come straight from the dump, not retyped.
Idempotent: does nothing if the figure is already present.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "paper" / "latex" / "iris_mmsys27.tex"
DATA = REPO / "eval_results" / "fig_valley_Abuse028.json"
ANCHOR = "\nSurvivors are partitioned into scenes. The default rule derives boundaries from valleys in\n"


def build(d: dict) -> str:
    fps = d["fps"]; pk = d["packet_bytes"]; ifr = set(d["iframe_indices"])
    size = {int(i): s for i, s in pk}
    curve = " ".join(f"({i/fps:.2f},{max(s,150)/1000:.2f})" for i, s in pk)
    ifp = " ".join(f"({i/fps:.2f},{size[i]/1000:.2f})" for i in sorted(ifr))
    vp = " ".join(f"({v/fps:.2f},{size[v]/1000:.2f})" for v in d["valleys_before_cap"])
    starts = [a for a, b in d["scene_spans"][1:]]
    cap = [a for a in starts if a not in set(d["valleys_before_cap"])]
    vlines = "".join(f"\\draw[blue!45, line width=0.3pt] (axis cs:{a/fps:.2f},0.12) -- (axis cs:{a/fps:.2f},90);\n" for a in starts if a not in cap)
    caplines = "".join(f"\\draw[blue!45, line width=0.5pt, dashed] (axis cs:{a/fps:.2f},0.12) -- (axis cs:{a/fps:.2f},90);\n" for a in cap)
    rug = " ".join(f"({a['frame_idx']/fps:.2f},0.15)" for a in d["admitted"])
    thr = d["valley_threshold_bytes"] / 1000
    return (r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{semilogyaxis}[width=\columnwidth, height=3.3cm, font=\scriptsize, xmin=0, xmax=47.1, ymin=0.12, ymax=90,
  xlabel={time (s)}, ylabel={packet size (KB)}, ytick={0.1,1,10,100}, axis lines*=left, clip=true]
""" + vlines + caplines + r"""\addplot[gray!75, line width=0.35pt] coordinates {""" + curve + r"""};
\addplot[red!65!black, dashed, line width=0.5pt, domain=0:47.1, samples=2] {""" + f"{thr:.3f}" + r"""};
\addplot[only marks, mark=*, mark size=1.1pt, red!70!black] coordinates {""" + vp + r"""};
\addplot[only marks, mark=triangle*, mark size=1.6pt, black] coordinates {""" + ifp + r"""};
\addplot[only marks, mark=|, mark size=2pt, blue!70!black] coordinates {""" + rug + r"""};
\end{semilogyaxis}
\end{tikzpicture}
\Description{Log-scale plot of per-frame packet size over the 47-second UCF-Crime clip Abuse028. A grey curve shows packet sizes between about 0.2 and 38 kilobytes, with black triangles marking six I-frames of about 54 kilobytes every 250 frames. A dashed red line marks the 25th percentile of non-keyframe packet sizes, 2.4 kilobytes; 18 red dots mark local minima below it, which become scene boundaries, drawn as thin vertical lines; one dashed vertical line marks a split forced by the 300-frame scene cap. Blue ticks along the bottom mark the 148 admitted frames.}
\caption{Segmentation and admission on UCF-Crime Abuse028 (47\,s). Minima below the 25th percentile (red) cut 20 scenes (blue; dashed: 300-frame cap); ticks: 148 admitted frames; triangles: I-frames.}
\label{fig:valleys}
\end{figure}""")


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    if "\\label{fig:valleys}" in tex:
        print("figure already present; nothing to do")
        return
    assert tex.count(ANCHOR) == 1, "anchor paragraph not found exactly once"
    fig = build(json.loads(DATA.read_text(encoding="utf-8")))
    tex = tex.replace(ANCHOR, "\n" + fig + "\n" + ANCHOR, 1)
    TEX.write_text(tex, encoding="utf-8", newline="\n")
    print(f"inserted Figure (fig:valleys), {len(fig)} chars")


if __name__ == "__main__":
    main()
