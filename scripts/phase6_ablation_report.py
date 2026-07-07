"""Read ablation_dev_results.jsonl and print the full gate+table+delta report.
No model calls — pure post-processing of the existing JSONL."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "eval" / "data" / "nextqa"
JSONL    = DATA_DIR / "ablation_dev_results.jsonl"

ARMS   = [1.0, 0.5, 0.0]
TOP_KS = [8, 12]
CHANCE = 0.20
GATE_BLIND_HIGH = 0.35
GATE_PARSE_HIGH = 0.15

LETTERS = ["A", "B", "C", "D", "E"]


def _fmt_acc(v) -> str:
    return f"{v:.1%}" if v is not None else " N/A "


def _sign(d: float) -> str:
    return "+" if d >= 0 else ""


def _stats(recs: list[dict]) -> dict:
    n = len(recs)
    if n == 0:
        return {"acc": None, "abstain_pct": None, "parse_fail_pct": None,
                "n": 0, "n_answered": 0, "n_abstain": 0, "n_fail": 0}
    n_fail    = sum(1 for r in recs if r["parse_fail"])
    n_abstain = sum(1 for r in recs if r["abstain"])
    n_answered = n - n_fail - n_abstain
    n_correct  = sum(1 for r in recs if r["correct"])
    acc = n_correct / n_answered if n_answered > 0 else None
    return {
        "acc": acc,
        "abstain_pct": n_abstain / n,
        "parse_fail_pct": n_fail / n,
        "n": n, "n_answered": n_answered,
        "n_abstain": n_abstain, "n_fail": n_fail,
    }


def main() -> None:
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    print(f"Loaded {len(rows)} rows from {JSONL}")

    # Build grid: top_k -> arm -> list[row]
    grid: dict[int, dict[float, list]] = {k: {a: [] for a in ARMS} for k in TOP_KS}
    families: set[str] = set()
    for r in rows:
        tk = r["top_k"]
        arm = r["arm"]
        if tk in grid and arm in grid[tk]:
            grid[tk][arm].append(r)
            families.add(r["family"])

    families = sorted(families)

    # Compute stats
    stats: dict[int, dict[float, dict]] = {}
    for tk in TOP_KS:
        stats[tk] = {}
        for arm in ARMS:
            recs = grid[tk][arm]
            stats[tk][arm] = {
                "overall": _stats(recs),
                "by_family": {fam: _stats([r for r in recs if r["family"] == fam])
                              for fam in families},
            }

    print()
    print("=== DECODE CONFIG ===")
    print("temperature : 0.0  (LlamaBackend, hardcoded at line 131 iris/aria.py)")
    print("decoding    : greedy (temperature=0.0 via OpenAI-compat Ollama endpoint)")
    print("model       : llama3.2:3b  (LlamaBackend.DEFAULT_TEXT_MODEL)")
    print("determinism : re-runs are identical given identical cache + model weights")
    print()

    print("===ABLATION_BEGIN===")
    print()
    print("=== SANITY GATES ===")
    for tk in TOP_KS:
        blind_ov = stats[tk][0.0]["overall"]
        blind_acc = blind_ov["acc"] if blind_ov["acc"] is not None else 0.0
        gate_blind = "SUSPECT" if blind_acc > GATE_BLIND_HIGH else "OK"
        print(f"GATE_BLIND  [top_k={tk}]: lam=0.0 Acc(all)={_fmt_acc(blind_ov['acc'])}"
              f"  (chance~20%, flag>35%)  -> {gate_blind}")
        if gate_blind == "SUSPECT":
            print("  WARNING: Answerer may shortcut from prior. SVO/grounding fallbacks apply.")

        max_fail = max(stats[tk][arm]["overall"]["parse_fail_pct"] for arm in ARMS)
        gate_parse = "SUSPECT" if max_fail > GATE_PARSE_HIGH else "OK"
        print(f"GATE_PARSE  [top_k={tk}]: max parse_fail%={max_fail:.1%}"
              f"  (flag>15%)  -> {gate_parse}")

    print()
    print("GATE_ABSTAIN (lower abstain on lam=0.5 vs 1.0 = corroborating signal):")
    for tk in TOP_KS:
        a10 = stats[tk][1.0]["overall"]["abstain_pct"]
        a05 = stats[tk][0.5]["overall"]["abstain_pct"]
        a00 = stats[tk][0.0]["overall"]["abstain_pct"]
        corr = "YES (corroborating)" if a05 < a10 else "NO"
        print(f"  top_k={tk}: lam=1.0 abstain={a10:.1%}  lam=0.5 abstain={a05:.1%}"
              f"  lam=0.0 abstain={a00:.1%}  -> {corr}")

    fam_hdrs = " | ".join(f"Acc@{f}" for f in families)

    for tk in TOP_KS:
        print()
        print(f"=== TOP_K={tk} TABLE ===")
        hdr = (f"{'arm(lam)':<10} | {fam_hdrs} | {'Acc(all)':<8} | "
               f"{'abstain%':<8} | {'parse_fail%':<11} | {'n':>4}")
        print(hdr)
        print("-" * len(hdr))
        for arm in ARMS:
            s = stats[tk][arm]
            fam_cells = " | ".join(
                f"{_fmt_acc(s['by_family'].get(f, {}).get('acc')):^8}"
                for f in families
            )
            ov = s["overall"]
            print(f"{'lam='+str(arm):<10} | {fam_cells} | "
                  f"{_fmt_acc(ov['acc']):^8} | "
                  f"{ov['abstain_pct']:>7.1%}  | "
                  f"{ov['parse_fail_pct']:>10.1%}  | "
                  f"{ov['n']:>4}")

        print()
        print(f"=== TOP_K={tk} SIGNED DELTAS (lam=0.5 - lam=1.0) ===")
        print("Headline: Acc@C and Acc@T")
        s05 = stats[tk][0.5]
        s10 = stats[tk][1.0]
        for fam in families:
            a05 = s05["by_family"].get(fam, {}).get("acc")
            a10 = s10["by_family"].get(fam, {}).get("acc")
            if a05 is not None and a10 is not None:
                d = a05 - a10
                marker = "  <<HEADLINE>>" if fam in ("C", "T") else ""
                print(f"  Delta Acc@{fam} = {_sign(d)}{d:+.1%}{marker}")
            else:
                print(f"  Delta Acc@{fam} = N/A")
        ov05 = s05["overall"]["acc"]
        ov10 = s10["overall"]["acc"]
        if ov05 is not None and ov10 is not None:
            d = ov05 - ov10
            print(f"  Delta Acc(all) = {_sign(d)}{d:+.1%}")
        else:
            print("  Delta Acc(all) = N/A")

    print()
    print("===END===")


if __name__ == "__main__":
    main()
