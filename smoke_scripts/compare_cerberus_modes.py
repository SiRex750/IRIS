"""Item 2: compare cerberus_mode="legacy" (default, all-or-nothing gate) vs
cerberus_mode="v2" (typed claim contract + badge system) on the SAME 12 real
smoke-test questions, reusing the already-ingested/cached indexes so this
only re-exercises Layer 3. Does not change the default cerberus_mode -- this
is a measurement, not a silent policy change.

Legacy abstention: answer == "Insufficient verified evidence to answer this question."
v2 "abstention" analog: badge == "unverified" (core claim rejected/unverifiable --
the closest v2 equivalent of "no defensible answer"). v2's "flagged" state (a
non-core claim was rejected but the core claim survived) is reported separately
since it has no legacy equivalent -- legacy's all-or-nothing gate would call
that full abstention, while v2 keeps the core answer with a caveat. That
difference is exactly what this comparison is measuring.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\swara\IRIS")
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest        # noqa: E402
import iris.query as iris_query          # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402

SMOKE_DIR = REPO / "smoke"
CACHE_DIR = SMOKE_DIR / "cache"


def make_config(mode: str) -> IRISConfig:
    cfg = IRISConfig()
    cfg.answerer_backend = "llama"
    cfg.answerer_endpoint = "http://localhost:11434/v1"
    cfg.answerer_model = "granite4:micro"
    cfg.cerberus_mode = mode
    return cfg


def run_mode(mode: str, selected: dict) -> list[dict]:
    cfg = make_config(mode)
    rows = []
    for q in selected["questions"]:
        vid = q["video_id"]
        index = iris_ingest.load_index(str(CACHE_DIR / vid))
        t0 = time.perf_counter()
        try:
            result = iris_query.query(q["question"], index, cfg)
            error = None
        except Exception as exc:  # noqa: BLE001
            result = None
            error = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        if error is not None:
            rows.append({"video_id": vid, "qid": q["qid"], "mode": mode, "error": error})
            continue

        if mode == "legacy":
            abstained = result["answer"] == "Insufficient verified evidence to answer this question."
            row = {
                "video_id": vid, "qid": q["qid"], "mode": mode, "error": None,
                "abstained": abstained, "verified_flag": result["verified"],
                "n_verified_claims": len(result["verified_claims"]),
                "n_rejected_claims": len(result["rejected_claims"]),
                "n_unverifiable_claims": len(result["unverifiable_claims"]),
                "answer": result["answer"], "wall_s": round(dt, 2),
            }
        else:  # v2
            badge = result["badge"]
            abstained = badge == "unverified"
            row = {
                "video_id": vid, "qid": q["qid"], "mode": mode, "error": None,
                "abstained": abstained, "badge": badge,
                "compliance_failed": result["compliance_failed"],
                "n_claim_verdicts": len(result["claim_verdicts"]),
                "answer": result["answer"], "wall_s": round(dt, 2),
            }
        rows.append(row)
        print(f"[{mode}] {vid}/{q['qid']}: abstained={row.get('abstained')} "
              f"({'badge='+str(row.get('badge')) if mode=='v2' else 'verified='+str(row.get('verified_flag'))}) "
              f"[{dt:.1f}s]", flush=True)
    return rows


def main():
    selected = json.loads((SMOKE_DIR / "selected_ids.json").read_text())

    legacy_rows = run_mode("legacy", selected)
    v2_rows = run_mode("v2", selected)

    def rate(rows):
        n = len(rows)
        n_err = sum(1 for r in rows if r.get("error"))
        n_abst = sum(1 for r in rows if r.get("abstained") is True)
        return n, n_err, n_abst

    n_l, err_l, abst_l = rate(legacy_rows)
    n_v, err_v, abst_v = rate(v2_rows)

    summary = {
        "n_questions": n_l,
        "legacy": {
            "n_errors": err_l,
            "n_abstained": abst_l,
            "abstention_rate": round(abst_l / n_l, 3) if n_l else None,
        },
        "v2": {
            "n_errors": err_v,
            "n_abstained": abst_v,
            "abstention_rate": round(abst_v / n_v, 3) if n_v else None,
            "badge_distribution": {
                b: sum(1 for r in v2_rows if r.get("badge") == b)
                for b in ("verified", "partial", "partially_verified", "flagged", "unverified")
            },
        },
        "abstention_rate_delta_v2_minus_legacy": (
            round((abst_v / n_v) - (abst_l / n_l), 3) if n_l and n_v else None
        ),
        "legacy_rows": legacy_rows,
        "v2_rows": v2_rows,
    }
    (SMOKE_DIR / "cerberus_mode_comparison.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(f"legacy abstention rate: {abst_l}/{n_l} = {summary['legacy']['abstention_rate']}")
    print(f"v2     abstention rate: {abst_v}/{n_v} = {summary['v2']['abstention_rate']}")
    print(f"v2 badge distribution: {summary['v2']['badge_distribution']}")


if __name__ == "__main__":
    main()
