
"""Part 1 parity proof: cerberus_mode="v2" vs cerberus_mode="none" must produce
IDENTICAL retrieved_frame_idxs, context_text, and raw_answer -- only answer,
verified, and badge may differ. Run against the fixed 12-question validation
set in smoke/selected_ids.json (already recorded before any pipeline
execution this session), using the first N questions across the 3 videos.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

CACHE_DIR = REPO / "smoke" / "cache_part1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_QUESTIONS = 8


def make_config(mode: str) -> IRISConfig:
    cfg = IRISConfig()
    cfg.cerberus_mode = mode
    return cfg


def get_or_build_index(video_id: str, video_path: str):
    idx_path = CACHE_DIR / video_id
    if (idx_path.with_suffix(".idx.json")).exists() or idx_path.exists():
        try:
            return iris_ingest.load_index(str(idx_path))
        except Exception:
            pass
    cfg = make_config("v2")  # ingest config is mode-independent
    index = iris_ingest.ingest(str(REPO / video_path), cfg)
    iris_ingest.save_index(index, str(idx_path))
    return index


def main():
    selected = json.loads((REPO / "smoke" / "selected_ids.json").read_text())
    questions = selected["questions"][:N_QUESTIONS]

    video_ids = sorted({q["video_id"] for q in questions})
    indexes = {}
    for vid in video_ids:
        vpath = next(q["video_path"] for q in questions if q["video_id"] == vid)
        print(f"[ingest] {vid} ...", flush=True)
        t0 = time.perf_counter()
        indexes[vid] = get_or_build_index(vid, vpath)
        print(f"[ingest] {vid} done in {time.perf_counter() - t0:.1f}s", flush=True)

    cfg_v2 = make_config("v2")
    cfg_none = make_config("none")

    results = []
    all_match = True
    for q in questions:
        vid = q["video_id"]
        index = indexes[vid]
        row = {"video_id": vid, "qid": q["qid"], "question": q["question"]}
        errors = {}
        out = {}
        for mode, cfg in (("v2", cfg_v2), ("none", cfg_none)):
            t0 = time.perf_counter()
            try:
                res = iris_query.query(q["question"], index, cfg, choices=q["choices"])
                out[mode] = res
            except Exception as exc:  # noqa: BLE001
                errors[mode] = f"{type(exc).__name__}: {exc}"
            row[f"{mode}_wall_s"] = round(time.perf_counter() - t0, 2)

        if errors:
            row["errors"] = errors
            row["parity_ok"] = False
            all_match = False
            results.append(row)
            print(f"  qid={q['qid']} vid={vid}: ERROR {errors}", flush=True)
            continue

        v2r, nr = out["v2"], out["none"]
        mismatches = []
        for field in ("retrieved_frame_idxs", "context_text", "raw_answer"):
            if v2r.get(field) != nr.get(field):
                mismatches.append(field)

        row["retrieved_frame_idxs_v2"] = v2r["retrieved_frame_idxs"]
        row["retrieved_frame_idxs_none"] = nr["retrieved_frame_idxs"]
        row["raw_answer_v2"] = v2r["raw_answer"]
        row["raw_answer_none"] = nr["raw_answer"]
        row["context_text_match"] = v2r.get("context_text") == nr.get("context_text")
        row["answer_v2"] = v2r["answer"]
        row["answer_none"] = nr["answer"]
        row["verified_v2"] = v2r["verified"]
        row["verified_none"] = nr["verified"]
        row["badge_v2"] = v2r["badge"]
        row["badge_none"] = nr["badge"]
        row["mismatched_fields"] = mismatches
        row["parity_ok"] = len(mismatches) == 0
        if mismatches:
            all_match = False
        print(f"  qid={q['qid']} vid={vid}: parity_ok={row['parity_ok']} mismatches={mismatches}", flush=True)
        results.append(row)

    report = {
        "generated_utc": "2026-07-21T05:00:00Z",
        "n_questions_tested": len(questions),
        "all_parity_ok": all_match,
        "fields_required_identical": ["retrieved_frame_idxs", "context_text", "raw_answer"],
        "fields_allowed_to_differ": ["answer", "verified", "badge"],
        "results": results,
    }
    out_path = REPO / "verification_removal_parity.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWROTE {out_path}")
    print("ALL_PARITY_OK =", all_match)
    if not all_match:
        print("BLOCKED_VERIFICATION_REMOVAL")
        sys.exit(1)
    print("VERIFICATION_REMOVAL_VERIFIED")


if __name__ == "__main__":
    main()
