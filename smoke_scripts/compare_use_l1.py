"""Item 6: use_l1=True vs False comparison on the same 3 videos / 12 questions.

No blocking reason for use_l1=False was found anywhere in the repo (see
smoke/item6_use_l1_investigation.md: no commit history, no code comment, zero
test coverage, and the "novel contributions" writeup assumes L1 is active).
Per the task instruction, this measures the delta before any default change
is proposed -- it does not change iris_config.py's default.

Reuses the cached indexes from the original smoke run (no re-ingest).
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


def make_config(use_l1: bool) -> IRISConfig:
    cfg = IRISConfig()
    cfg.answerer_backend = "llama"
    cfg.answerer_endpoint = "http://localhost:11434/v1"
    cfg.answerer_model = "granite4:micro"
    cfg.use_l1 = use_l1
    return cfg


def run_mode(use_l1: bool, selected: dict) -> list[dict]:
    cfg = make_config(use_l1)
    rows = []
    for q in selected["questions"]:
        vid = q["video_id"]
        index = iris_ingest.load_index(str(CACHE_DIR / vid))
        t0 = time.perf_counter()
        try:
            result = iris_query.query(q["question"], index, cfg, choices=q.get("choices"))
            error = None
        except Exception as exc:  # noqa: BLE001
            result = None
            error = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        if error is not None:
            rows.append({"video_id": vid, "qid": q["qid"], "use_l1": use_l1, "error": error})
            continue

        abstained = result["answer"] == "Insufficient verified evidence to answer this question."
        row = {
            "video_id": vid, "qid": q["qid"], "use_l1": use_l1, "error": None,
            "abstained": abstained, "verified": result["verified"],
            "retrieved_frame_idxs": result["retrieved_frame_idxs"],
            "l1_consulted": result["query_telemetry"].get("l1_consulted"),
            "l1_hit": result["query_telemetry"].get("l1_hit"),
            "l2_fallback": result["query_telemetry"].get("l2_fallback"),
            "answer": result["answer"],
            "total_latency_s": result["timings"]["total"],
            "wall_s": round(dt, 2),
        }
        rows.append(row)
        print(f"[use_l1={use_l1}] {vid}/{q['qid']}: abstained={abstained} "
              f"l1_consulted={row['l1_consulted']} l1_hit={row['l1_hit']} "
              f"retrieved={row['retrieved_frame_idxs']} [{dt:.1f}s]", flush=True)
    return rows


def main():
    selected = json.loads((SMOKE_DIR / "selected_ids.json").read_text())

    rows_off = run_mode(False, selected)
    rows_on = run_mode(True, selected)

    def summarize(rows):
        n = len(rows)
        n_err = sum(1 for r in rows if r.get("error"))
        n_abst = sum(1 for r in rows if r.get("abstained") is True)
        mean_latency = sum(r.get("total_latency_s", 0) for r in rows if not r.get("error")) / max(1, n - n_err)
        return {"n": n, "n_errors": n_err, "n_abstained": n_abst,
                "abstention_rate": round(n_abst / n, 3) if n else None,
                "mean_total_latency_s": round(mean_latency, 2)}

    retrieval_deltas = []
    by_key_off = {(r["video_id"], r["qid"]): r for r in rows_off if not r.get("error")}
    by_key_on = {(r["video_id"], r["qid"]): r for r in rows_on if not r.get("error")}
    for k in by_key_off:
        if k in by_key_on:
            same_retrieval = by_key_off[k]["retrieved_frame_idxs"] == by_key_on[k]["retrieved_frame_idxs"]
            retrieval_deltas.append({"key": k, "same_retrieved_frames": same_retrieval})

    summary = {
        "use_l1_False": summarize(rows_off),
        "use_l1_True": summarize(rows_on),
        "n_questions_with_different_retrieval": sum(1 for r in retrieval_deltas if not r["same_retrieved_frames"]),
        "n_questions_compared": len(retrieval_deltas),
        "rows_off": rows_off,
        "rows_on": rows_on,
    }
    (SMOKE_DIR / "use_l1_comparison.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print("use_l1=False:", summary["use_l1_False"])
    print("use_l1=True: ", summary["use_l1_True"])
    print("questions with different retrieved frames:",
          summary["n_questions_with_different_retrieval"], "/", summary["n_questions_compared"])


if __name__ == "__main__":
    main()
