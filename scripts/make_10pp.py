"""Derive the all-in-10-pages variant of the paper from the MMSys version.

Applies the edits in paper/latex/10pp_edits.txt to paper/latex/iris_mmsys27.tex and writes
paper/latex/iris_mmsys27_10pp.tex (body AND references within 10 pages), and derives
paper/latex/refs_10pp.bib from refs.bib (short venue names, three authors then et al., no
page/volume/doi fields). Each edit must match exactly once, so the 10-page variant cannot
silently drift from the main one: re-run after any change to iris_mmsys27.tex.

Edit file format (raw text, no escaping):  <<<<OLD\\n<old>====NEW\\n<new>>>>>END\\n
"""
import re
from pathlib import Path

LATEX = Path(__file__).resolve().parent.parent / "paper" / "latex"

VENUES = {
    r"Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition \(CVPR\)": "CVPR",
    r"Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition \(CVPR\)": "CVPR",
    r"Advances in Neural Information Processing Systems \(NeurIPS\)": "NeurIPS",
    r"Proceedings of the International Conference on Machine Learning \(ICML\)": "ICML",
    r"Proceedings of the ACM International Conference on Multimedia \(ACM MM\)": "ACM MM",
    r"Proceedings of the International Conference on World Wide Web \(WWW\)": "WWW",
}


def parse_edits(text: str) -> list[tuple[str, str]]:
    edits = []
    for block in text.split("<<<<OLD\n")[1:]:
        old, rest = block.split("====NEW\n", 1)
        new = rest.split(">>>>END\n", 1)[0]
        edits.append((old, new))
    return edits


def compact_bib(b: str) -> str:
    for k, v in VENUES.items():
        b = re.sub(k, v, b)

    def trunc(m):
        names = [n.strip() for n in m.group(2).split(" and ") if n.strip() != "others"]
        if len(names) > 3:
            names = names[:3] + ["others"]
        return m.group(1) + " and ".join(names) + m.group(3)

    b = re.sub(r"(author\s*=\s*\{)(.*?)(\},)", trunc, b, flags=re.S)
    b = re.sub(r"\n  pages\s*=\s*\{[^}]*\},", "", b)
    b = re.sub(r"\n  (volume|doi|primaryClass)\s*=\s*\{[^}]*\},?", "", b)
    return b


def main() -> None:
    tex = (LATEX / "iris_mmsys27.tex").read_text(encoding="utf-8")
    edits = parse_edits((LATEX / "10pp_edits.txt").read_text(encoding="utf-8"))
    for old, new in edits:
        assert tex.count(old) == 1, "anchor not found exactly once: " + old[:80]
        tex = tex.replace(old, new, 1)
    (LATEX / "iris_mmsys27_10pp.tex").write_text(tex, encoding="utf-8", newline="\n")
    bib = (LATEX / "refs.bib").read_text(encoding="utf-8")
    (LATEX / "refs_10pp.bib").write_text(compact_bib(bib), encoding="utf-8", newline="\n")
    print(f"wrote iris_mmsys27_10pp.tex ({len(edits)} edits) and refs_10pp.bib")


if __name__ == "__main__":
    main()
