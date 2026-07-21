"""Targeted re-run of the determinism gate after item-1's seed fix, reusing the
cached indexes from the original smoke run (smoke/cache/*.npz) so this doesn't
re-decode video -- only re-exercises Layer 3 (captioning is already cached on
the loaded FrameRecords too, so this isolates the answerer's seed behavior).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\swara\IRIS")
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest      # noqa: E402
import iris.query as iris_query        # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402

SMOKE_DIR = REPO / "smoke"
CACHE_DIR = SMOKE_DIR / "cache"


def make_config() -> IRISConfig:
    cfg = IRISConfig()
    cfg.answerer_backend = "llama"
    cfg.answerer_endpoint = "http://localhost:11434/v1"
    cfg.answerer_model = "granite4:micro"
    return cfg  # answerer_seed defaults to 42 now


def main():
    selected = json.loads((SMOKE_DIR / "selected_ids.json").read_text())
    cfg = make_config()

    results = []
    for q in selected["questions"]:
        vid = q["video_id"]
        cache_path = CACHE_DIR / vid
        # Reuse ONE loaded index for both calls (matches the original smoke
        # harness and the correctness gate's "same unmutated index" wording).
        # An earlier version of this script loaded two SEPARATE index copies,
        # which silently reintroduced captioner-side nondeterminism (moondream
        # has no fixed seed) as a confound on top of the answerer-seed question
        # this script is meant to isolate -- caption() is called once per
        # FrameRecord.caption cache slot, so two independently-loaded indexes
        # each pay a fresh (and possibly non-reproducible) captioning cost.
        index = iris_ingest.load_index(str(cache_path))

        t0 = time.perf_counter()
        res_a = iris_query.query(q["question"], index, cfg)
        res_b = iris_query.query(q["question"], index, cfg)
        dt = time.perf_counter() - t0

        same_answer = res_a["answer"] == res_b["answer"]
        same_retrieval = res_a["retrieved_frame_idxs"] == res_b["retrieved_frame_idxs"]
        same_raw = res_a["raw_answer"] == res_b["raw_answer"]
        rec = {
            "video_id": vid, "qid": q["qid"],
            "same_retrieval": same_retrieval,
            "same_raw_answer": same_raw,
            "same_final_answer": same_answer,
            "DETERMINISTIC": same_answer and same_retrieval,
            "answer_a": res_a["answer"], "answer_b": res_b["answer"],
            "wall_s": round(dt, 2),
        }
        results.append(rec)
        print(f"{vid}/{q['qid']}: deterministic={rec['DETERMINISTIC']} "
              f"(retrieval={same_retrieval}, raw_answer={same_raw}) [{dt:.1f}s]", flush=True)

    n = len(results)
    n_det = sum(1 for r in results if r["DETERMINISTIC"])
    summary = {
        "n_questions": n,
        "n_deterministic": n_det,
        "ALL_DETERMINISTIC": n_det == n,
        "answerer_seed_used": cfg.answerer_seed,
        "records": results,
    }
    (SMOKE_DIR / "determinism_post_fix.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{n_det}/{n} deterministic after seed fix (answerer_seed={cfg.answerer_seed})")


if __name__ == "__main__":
    main()
