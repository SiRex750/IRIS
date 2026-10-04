"""Review-2 item 1: configuration, sequence and encode counts of the Sintel sweep. SUPPLEMENTARY - no new verdict.
Reads p3/sweep/results.csv and p3/sweep/encode_log.json (and the rerun's, for agreement). Writes i01_config_count.md.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)


def counts(d):
    r = pd.read_csv(os.path.join(P3, d, "results.csv"), usecols=["pass", "encode_id", "sequence"])
    arms = {p: sorted(g.encode_id.unique()) for p, g in r.groupby("pass")}
    seqs = {p: g.sequence.nunique() for p, g in r.groupby("pass")}
    enc = r.drop_duplicates(["pass", "encode_id", "sequence"])
    lg = json.load(open(os.path.join(P3, d, "encode_log.json")))
    ok = sum(x["status"] == "ok" for x in lg["logs"])
    files = sum(len(os.listdir(os.path.join(P3, d, "encodes", p))) for p in ("final", "clean"))
    return arms, seqs, len(enc), r.sequence.nunique(), ok, len(lg["logs"]), files, enc.groupby("pass").size().to_dict()


def main():
    L = ["# Item 1: configuration count (supplementary; no new verdict)", "",
         "Source: p3/sweep/results.csv (encode_id x sequence x pass), p3/sweep/encode_log.json, p3/sweep/encodes/. "
         "Rerun (p3/sweep_rerun) checked the same way.", ""]
    for d in ("sweep", "sweep_rerun"):
        arms, seqs, n_enc, n_seq, ok, nlog, files, per = counts(d)
        both = sorted(set(arms["final"]) & set(arms["clean"]))
        clean_only = sorted(set(arms["clean"]) - set(arms["final"]))
        L += [f"## p3/{d}", "",
              f"- Final-pass configurations: **{len(arms['final'])}**; clean-pass configurations: **{len(arms['clean'])}**.",
              f"- Distinct configurations overall: {len(set(arms['final']) | set(arms['clean']))}; "
              f"clean-only configurations: {clean_only or 'none'}.",
              f"- Sequences: **{n_seq}** (final {seqs['final']}, clean {seqs['clean']}).",
              f"- Encodes (distinct pass x configuration x sequence in results.csv): **{n_enc}** "
              f"(final {per['final']} = {len(arms['final'])} x 23, clean {per['clean']} = {len(arms['clean'])} x 23); "
              f"encode log: {ok}/{nlog} ok; .mp4 files on disk: {files}.",
              f"- Configurations shared between the final and clean groups ({len(both)}): " + ", ".join(both) + ".",
              "- Final-only configurations (" + str(len(set(arms['final']) - set(arms['clean']))) + "): "
              + ", ".join(sorted(set(arms["final"]) - set(arms["clean"]))) + ".", ""]
    open(os.path.join(HERE, "i01_config_count.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
