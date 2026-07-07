# Measures retrieval quality AS TRANSMITTED THROUGH BLIP captions -> 3B answerer.
# This is a text-bottlenecked, lossy-downstream measurement. Two designated
# fallbacks if the sanity gates fire or the delta is buried in noise:
#   (1) SVO/structured-evidence: extract triples from retrieved captions (HADES
#       REBEL path) and feed those instead of raw captions. Ablated separately.
#   (2) NExT-GQA grounding-overlap: score retrieved-frame overlap with gold
#       answer time-windows -- captioner-free, answerer-free, no hallucination.
# DO NOT add either now; this run must be the clean retrieval A/B on raw captions.
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris.iris_config import IRISConfig
from eval.mc_scorer import score_arm

DATA_DIR  = REPO / "eval" / "data" / "nextqa"
CACHE_DIR = DATA_DIR / "index_cache"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
OUT_JSONL = DATA_DIR / "ablation_dev_results.jsonl"

ARMS    = [1.0, 0.5, 0.0]       # lambda values
TOP_KS  = [8, 12]
BASE_CFG = dict(
    ranking_mode="ppr",
    ppr_damping=0.5,
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
)

CHANCE = 0.20   # 5-way MC chance baseline
GATE_BLIND_HIGH = 0.35
GATE_PARSE_HIGH = 0.15


def _fmt_acc(v) -> str:
    return f"{v:.1%}" if v is not None else " N/A "


def _sign(delta: float) -> str:
    return "+" if delta >= 0 else ""


def main() -> None:
    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    cached_vids = {p.stem for p in CACHE_DIR.glob("*.npz")}

    # Filter to questions whose video is cached
    eval_rows = [r for r in dev_rows if r["video"] in cached_vids]
    print(f"Dev questions: {len(dev_rows)}  cached-video subset: {len(eval_rows)}")
    print()

    # Decode config confirmation
    print("=== DECODE CONFIG ===")
    print("temperature : 0.0  (LlamaBackend, hardcoded at line 131 iris/aria.py)")
    print("decoding    : greedy (temperature=0.0 via OpenAI-compat Ollama endpoint)")
    print("model       : llama3.2:3b  (LlamaBackend.DEFAULT_TEXT_MODEL)")
    print("determinism : re-runs are identical given identical cache + model weights")
    print()

    # Run 3×2 grid, collect per-question rows
    all_results: list[dict] = []
    grid: dict[int, dict[float, dict]] = {}  # top_k -> lambda -> stats dict

    for top_k in TOP_KS:
        grid[top_k] = {}
        for lam in ARMS:
            cfg = IRISConfig(
                l2_retrieve_top_k=top_k,
                ppr_lambda=lam,
                **BASE_CFG,
            )
            print(f"--- Scoring top_k={top_k}  lambda={lam} ---", flush=True)
            stats = score_arm(eval_rows, CACHE_DIR, cfg, results_out=all_results)
            grid[top_k][lam] = stats
            ov = stats["overall"]
            print(f"    n={ov['n']}  acc={_fmt_acc(ov['acc'])}  "
                  f"abstain={ov['abstain_pct']:.1%}  fail={ov['parse_fail_pct']:.1%}",
                  flush=True)

    # Write per-question JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for r in all_results:
            fh.write(json.dumps(r) + "\n")
    print(f"\nPer-question rows written: {len(all_results)} -> {OUT_JSONL}")

    # -------------------------------------------------------------------------
    print()
    print("===ABLATION_BEGIN===")

    # GATE BLOCK — printed first
    print()
    print("=== SANITY GATES ===")
    for top_k in TOP_KS:
        blind_stats = grid[top_k][0.0]["overall"]
        blind_acc = blind_stats["acc"] if blind_stats["acc"] is not None else 0.0
        gate_blind = "SUSPECT" if blind_acc > GATE_BLIND_HIGH else "OK"
        print(f"GATE_BLIND  [top_k={top_k}]: lambda=0.0 Acc(all)={_fmt_acc(blind_stats['acc'])}  "
              f"(chance~20%, flag>{GATE_BLIND_HIGH:.0%})  -> {gate_blind}")
        if gate_blind == "SUSPECT":
            print("  WARNING: Answerer may shortcut from prior. SVO/grounding fallbacks apply.")

        max_fail = max(
            grid[top_k][lam]["overall"]["parse_fail_pct"] for lam in ARMS
        )
        gate_parse = "SUSPECT" if max_fail > GATE_PARSE_HIGH else "OK"
        print(f"GATE_PARSE  [top_k={top_k}]: max parse_fail%={max_fail:.1%}  "
              f"(flag>{GATE_PARSE_HIGH:.0%})  -> {gate_parse}")
        if gate_parse == "SUSPECT":
            print("  WARNING: 3B parser too noisy; accuracy is parser-contaminated.")

    print()
    print("GATE_ABSTAIN (lower abstain on lambda=0.5 vs 1.0 = corroborating signal):")
    for top_k in TOP_KS:
        abs_10 = grid[top_k][1.0]["overall"]["abstain_pct"]
        abs_05 = grid[top_k][0.5]["overall"]["abstain_pct"]
        abs_00 = grid[top_k][0.0]["overall"]["abstain_pct"]
        corr = "YES (corroborating)" if abs_05 < abs_10 else "NO"
        print(f"  top_k={top_k}: lambda=1.0 abstain={abs_10:.1%}  "
              f"lambda=0.5 abstain={abs_05:.1%}  lambda=0.0 abstain={abs_00:.1%}  -> {corr}")

    # TABLE + DELTAS per top_k
    families = sorted({r["family"] for r in eval_rows})
    fam_hdrs = " | ".join(f"Acc@{f}" for f in families)

    for top_k in TOP_KS:
        print()
        print(f"=== TOP_K={top_k} TABLE ===")
        col_w = 8
        hdr = (f"{'arm(lam)':<10} | {fam_hdrs} | {'Acc(all)':<8} | "
               f"{'abstain%':<8} | {'parse_fail%':<11} | {'n':>4}")
        print(hdr)
        print("-" * len(hdr))

        for lam in ARMS:
            stats = grid[top_k][lam]
            fam_cells = " | ".join(
                f"{_fmt_acc(stats['by_family'].get(f, {}).get('acc', None)):^8}"
                for f in families
            )
            ov = stats["overall"]
            print(f"{'lam='+str(lam):<10} | {fam_cells} | "
                  f"{_fmt_acc(ov['acc']):^8} | "
                  f"{ov['abstain_pct']:>7.1%}  | "
                  f"{ov['parse_fail_pct']:>10.1%}  | "
                  f"{ov['n']:>4}")

        # Signed deltas
        print()
        print(f"=== TOP_K={top_k} SIGNED DELTAS (lam=0.5 - lam=1.0) ===")
        print("Headline: Acc@C and Acc@T")

        stats_05 = grid[top_k][0.5]
        stats_10 = grid[top_k][1.0]

        for fam in families:
            acc_05 = stats_05["by_family"].get(fam, {}).get("acc")
            acc_10 = stats_10["by_family"].get(fam, {}).get("acc")
            if acc_05 is not None and acc_10 is not None:
                d = acc_05 - acc_10
                marker = "  <<HEADLINE>>" if fam in ("C", "T") else ""
                print(f"  Δ Acc@{fam} = {_sign(d)}{d:+.1%}{marker}")
            else:
                print(f"  Δ Acc@{fam} = N/A (missing arm data)")

        ov_05 = stats_05["overall"]["acc"]
        ov_10 = stats_10["overall"]["acc"]
        if ov_05 is not None and ov_10 is not None:
            d = ov_05 - ov_10
            print(f"  Δ Acc(all) = {_sign(d)}{d:+.1%}")
        else:
            print("  Δ Acc(all) = N/A")

    print()
    print("===END===")


if __name__ == "__main__":
    main()
