"""AUDIT CHECK #1 (MASTER) + #4 residual — run LOCALLY where the cache exists.

For N grounded-AND-cached questions, dump the actual top-k retrieved
(frame_idx, timestamp) sets at lambda = 1.0 / 0.5 / 0.0, cross-tab each
against the gold span, and report pairwise Jaccard overlap of the frame sets.

Decision (per handoff):
  - If J(1.0, 0.5) ~ 1.0  -> the fusion is NOT moving the output; the
    'lam=0.5 ~ uniform' reading needs reinterpretation (it would mean 0.5
    inherits 1.0's frames yet scores at uniform -> something else is off).
  - If sets differ AND lam=1.0 lands in-window more often than 0.0 -> the
    negative result is real (codec seed picks no-better frames than blind).

Also dumps, per video: max(node.timestamp) vs gsub duration, and whether
duration is present/non-zero (closes check #4's uniform-floor source risk).

Verify command:
    python scripts/phase6_grounding_setdump.py
Expect: one block per sampled question + a SCALE table + a one-line VERDICT.
No tuning. This dumps and eyeballs; it does not search for a winner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris.iris_config import IRISConfig
from iris.ingest import load_index
from iris.query import _embed_query
from eval.grounding_scorer import frames_in_window

DATA_DIR  = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
GQA_JSON  = DATA_DIR / "gsub_val.json"

LAMBDAS  = [1.0, 0.5, 0.0]
TOP_K    = 8          # report at the smaller budget; set 12 to cross-check
N_SAMPLE = 5          # first N grounded-AND-cached rows (deterministic order)

BASE_CFG = dict(
    ranking_mode="ppr",
    ppr_damping=0.5,
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    gsub     = json.load(open(GQA_JSON, encoding="utf-8"))
    cached   = {p.stem for p in CACHE_DIR.glob("*.npz")}

    grounded = [
        r for r in dev_rows
        if r["video"] in cached
        and r["video"] in gsub
        and str(r["qid"]) in gsub[r["video"]]["location"]
    ]
    sample = grounded[:N_SAMPLE]
    print(f"grounded-AND-cached: {len(grounded)}  | sampling first {len(sample)} "
          f"at top_k={TOP_K}\n")

    # cache one loaded index per video (fresh; node mutations don't affect ranking)
    idx_cache: dict[str, object] = {}

    def get_index(vid: str):
        if vid not in idx_cache:
            idx_cache[vid] = load_index(CACHE_DIR / vid)
        return idx_cache[vid]

    fiw_accum = {lam: [] for lam in LAMBDAS}
    jac_10_05, jac_10_00 = [], []

    for row in sample:
        vid, qid = row["video"], str(row["qid"])
        question = row["question"]
        gold = [[float(s), float(e)] for s, e in gsub[vid]["location"][qid]]
        index = get_index(vid)
        emb = _embed_query(question, IRISConfig(**BASE_CFG))

        arm_sets: dict[float, set] = {}
        print(f"== {vid} qid={qid} fam={row.get('family')} ==")
        print(f"   Q: {question}")
        print(f"   gold spans (s): {gold}")
        for lam in LAMBDAS:
            nodes = index._graph.retrieve_ppr(
                emb, top_k=TOP_K, damping=0.5, lambda_=lam
            )
            pairs = [(n.frame_idx, round(float(n.timestamp), 2)) for n in nodes]
            ts    = [float(n.timestamp) for n in nodes]
            fiw   = frames_in_window(ts, gold)
            fiw_accum[lam].append(fiw)
            arm_sets[lam] = {fi for fi, _ in pairs}
            inwin = ["*" if any(s <= t <= e for s, e in gold) else "." for t in ts]
            print(f"   lam={lam:<3} FiW={fiw:.3f}  set={pairs}")
            print(f"            in-window: {''.join(inwin)}")
        j1 = jaccard(arm_sets[1.0], arm_sets[0.5])
        j0 = jaccard(arm_sets[1.0], arm_sets[0.0])
        jac_10_05.append(j1)
        jac_10_00.append(j0)
        print(f"   Jaccard(1.0,0.5)={j1:.3f}  Jaccard(1.0,0.0)={j0:.3f}\n")

    # ---- SCALE spot-check (check #4 residual) --------------------------------
    print("=== SCALE: max(node.timestamp) vs gsub duration ===")
    print(f"{'video':<16} | {'max_ts':>8} | {'duration':>8} | {'ratio':>6} | dur_ok")
    print("-" * 56)
    for vid in dict.fromkeys(r["video"] for r in sample):
        index = get_index(vid)
        max_ts = max((float(fr.timestamp) for fr in index.frames), default=0.0)
        dur = float(gsub[vid].get("duration", 0) or 0)
        ratio = (max_ts / dur) if dur > 0 else float("nan")
        dur_ok = "yes" if dur > 0 else "MISSING/0"
        print(f"{vid:<16} | {max_ts:>8.2f} | {dur:>8.2f} | {ratio:>6.2f} | {dur_ok}")
    # ratio should be <=~1.0 and duration must be > 0 for every sampled video.

    # ---- VERDICT -------------------------------------------------------------
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("\n=== VERDICT ===")
    print(f"mean FiW: lam1.0={mean(fiw_accum[1.0]):.3f}  "
          f"lam0.5={mean(fiw_accum[0.5]):.3f}  lam0.0={mean(fiw_accum[0.0]):.3f}")
    print(f"mean Jaccard(1.0,0.5)={mean(jac_10_05):.3f}  "
          f"mean Jaccard(1.0,0.0)={mean(jac_10_00):.3f}")
    if mean(jac_10_05) > 0.9:
        print("FLAG: lam=0.5 set ~= lam=1.0 set -> fusion barely moves output; "
              "REINTERPRET the 0.5~uniform reading before any pivot.")
    else:
        print("OK: lam arms produce materially different sets -> the ablation is "
              "live; if lam1.0 lands in-window more, the negative result is real.")


if __name__ == "__main__":
    main()
