"""§5.3 third-stage decomposition: measurement-only re-run of the e2e A/B.

Byte-for-byte the same protocol as scripts/e2e_speedup_ab.py (same config,
same N=4,892 VIRAT index, same 5 real CLIP-text queries + discarded warmup,
same per-arm caption-cache reset, same aria.set_backend override to
granite4:micro, same cerberus_mode="legacy"). iris.query.query() already
records lazy_caption / elysium / cerberus_v as separate wall-clock timings
per query (iris/query.py, committed at b6bbd78 -- unmodified here); this
script does NOT add new timing instrumentation to the pipeline, it only:

  (a) writes to a SEPARATE output path so the existing eval_results/
      e2e_speedup.json artifact (and the crossover-projection analysis
      built on top of it by scripts/_e2e_speedup_report.py) is left
      untouched, and
  (b) stamps the run with the actual git commit hash + tracked-file dirty
      count at run time (computed here, not hand-written), because the
      prior e2e_speedup.json run recorded neither correctly (A.5.5: run
      was against 87 dirty files with no commit stamp at all).

No stage boundary, ordering, or pipeline code is touched. Not a re-run of
build_cost_final, the scaling corpus, or the caption diagnosis.

VERIFY: python scripts/e2e_stage3_decomp_ab.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest
import iris.aria as aria
from iris.iris_config import IRISConfig
from iris.query import query as iris_query

from _scaling_curve_v2_worker import TEXT_QUERIES  # noqa: E402 -- reused fixed list, not reconstructed

CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"
NEXTQA_CACHE_DIR = REPO / "eval" / "data" / "nextqa" / "index_cache"
VIRAT_CACHE_DIR = REPO / "eval" / "data" / "virat" / "index_cache"
OUT_JSON = REPO / "eval_results" / "e2e_stage3_decomp_raw.json"
OUT_MD = REPO / "eval_results" / "e2e_stage3_decomp_raw.md"

N_Q = 5  # identical to scripts/e2e_speedup_ab.py -- see that file's comment
ANSWERER_MODEL = "granite4:micro"
CAPTIONER_MODEL = "minicpm-v4.6:1b"  # exact installed Ollama tag -- see main()

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=30,
    scene_diag=True,
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def _git_provenance() -> dict:
    commit = _git(["rev-parse", "HEAD"])
    dirty_tracked = _git(["diff", "--name-only", "HEAD"]).splitlines()
    dirty_tracked = [f for f in dirty_tracked if f]
    staged = _git(["diff", "--cached", "--name-only"]).splitlines()
    staged = [f for f in staged if f]
    return {
        "commit": commit,
        "branch": _git(["branch", "--show-current"]),
        "tracked_dirty_count": len(dirty_tracked),
        "tracked_dirty_files": dirty_tracked,
        "tracked_staged_count": len(staged),
        "clean_checkout": (len(dirty_tracked) == 0 and len(staged) == 0),
    }


def _dir_fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("*.npz")):
        st = f.stat()
        h.update(f.name.encode()); h.update(str(st.st_mtime_ns).encode()); h.update(str(st.st_size).encode())
    return h.hexdigest()


def _reset_captions(index) -> None:
    for fr in index.frames:
        fr.caption = None


def run_arm(mode: str, index, questions: list[str]) -> list[dict]:
    cfg = IRISConfig(**BASE, graph_mode=mode)

    _reset_captions(index)
    print(f"[{mode}] warmup query...", flush=True)
    t0 = time.perf_counter()
    iris_query(questions[0], index, cfg)
    print(f"[{mode}] warmup done in {time.perf_counter()-t0:.2f}s (discarded)", flush=True)
    _reset_captions(index)

    records = []
    for qi, q in enumerate(questions):
        t0 = time.perf_counter()
        result = iris_query(q, index, cfg)
        wall = time.perf_counter() - t0
        timings = result.get("timings", {})
        rec = {
            "qi": qi,
            "graph_mode": mode,
            "question": q,
            "wall_s": wall,
            "timings": timings,
            "frames_decoded_for_captions": result.get("frames_decoded_for_captions"),
            "n_retrieved": len(result.get("retrieved_frame_idxs", []) or []),
            "verified": result.get("verified"),
            "answer_len_chars": len(result.get("answer", "") or ""),
        }
        records.append(rec)
        print(f"[{mode}] q{qi} wall={wall:.2f}s timings={timings}", flush=True)
    return records


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def main():
    provenance = _git_provenance()
    print(f"[provenance] commit={provenance['commit']} tracked_dirty_count={provenance['tracked_dirty_count']}", flush=True)

    aria.set_backend(aria.LlamaBackend(text_model=ANSWERER_MODEL))

    # Same pattern as the answerer override above: committed default config
    # (configs/default_iris_config.json) says captioner_backend="minicpm",
    # and aria.get_captioner()'s auto-detect picks whichever installed Ollama
    # tag contains "minicpm" and truncates it at the first ":" -- e.g. the
    # locally installed "minicpm-v4.6:1b" becomes "minicpm-v4.6", which is
    # NOT a real Ollama tag (no ":latest" for that model here) and 404s on
    # generate. Pin the exact installed tag explicitly via the existing
    # public set_captioner() API rather than touching aria.py's detection
    # logic -- no captioner/pipeline code modified.
    aria.set_captioner(aria.MiniCPMCaptioner(model_name=CAPTIONER_MODEL))

    nextqa_before = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_before = _dir_fingerprint(VIRAT_CACHE_DIR)

    t0 = time.perf_counter()
    idx_scene_sparse = iris_ingest.load_index(CACHE_PATH)
    scene_sparse_setup_s = time.perf_counter() - t0
    assert idx_scene_sparse.config_snapshot.get("graph_mode") == "scene_sparse"
    n_survivors = len(idx_scene_sparse.frames)

    t0 = time.perf_counter()
    flat_cfg_dict = dict(idx_scene_sparse.config_snapshot)
    flat_cfg_dict["graph_mode"] = "flat"
    flat_graph = iris_ingest._build_graph(idx_scene_sparse.frames, flat_cfg_dict)
    flat_setup_s = time.perf_counter() - t0
    idx_flat = dataclasses.replace(idx_scene_sparse, config_snapshot=flat_cfg_dict, _graph=flat_graph)

    nextqa_after_setup = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_after_setup = _dir_fingerprint(VIRAT_CACHE_DIR)

    questions = [TEXT_QUERIES[i % len(TEXT_QUERIES)] for i in range(N_Q)]

    print("=== flat ===", flush=True)
    t_flat0 = time.perf_counter()
    records_flat = run_arm("flat", idx_flat, questions)
    flat_arm_wall_s = time.perf_counter() - t_flat0
    assert len(records_flat) == N_Q, f"flat arm did not complete all {N_Q} queries"

    print("=== scene_sparse ===", flush=True)
    t_ss0 = time.perf_counter()
    records_scene = run_arm("scene_sparse", idx_scene_sparse, questions)
    scene_sparse_arm_wall_s = time.perf_counter() - t_ss0
    assert len(records_scene) == N_Q, f"scene_sparse arm did not complete all {N_Q} queries"

    nextqa_after = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_after = _dir_fingerprint(VIRAT_CACHE_DIR)

    def stage(records, key):
        return [r["timings"].get(key) for r in records]

    def other_stage(records):
        out = []
        for r in records:
            t = r["timings"]
            vals = [t.get("lazy_caption"), t.get("elysium"), t.get("cerberus_v")]
            if any(v is None for v in vals):
                out.append(None)
            else:
                out.append(sum(vals))
        return out

    summary = {
        "task": "§5.3 third-stage (other = caption+elysium+cerberus) component decomposition",
        "provenance": provenance,
        "n_survivors": n_survivors,
        "n_queries_per_arm": N_Q,
        "questions": questions,
        "answerer_model": ANSWERER_MODEL,
        "cerberus_mode": "legacy",
        "top_k": BASE["l2_retrieve_top_k"],
        "setup_costs_one_time_excluded_from_per_query": {
            "scene_sparse_load_s": scene_sparse_setup_s,
            "flat_build_s": flat_setup_s,
        },
        "arm_wall_clock_s_incl_warmup": {
            "flat": flat_arm_wall_s,
            "scene_sparse": scene_sparse_arm_wall_s,
        },
        "per_query_records": {"flat": records_flat, "scene_sparse": records_scene},
        "median_stage_s": {
            "flat": {
                "l2_retrieval_retrieval_mechanics": _median(stage(records_flat, "l2_retrieval")),
                "aria_answerer_llm": _median(stage(records_flat, "aria")),
                "caption_s": _median(stage(records_flat, "lazy_caption")),
                "l1_elysium_s": _median(stage(records_flat, "elysium")),
                "cerberus_verify_s": _median(stage(records_flat, "cerberus_v")),
                "other_caption_elysium_cerberus": _median(other_stage(records_flat)),
                "total": _median(stage(records_flat, "total")),
            },
            "scene_sparse": {
                "l2_retrieval_retrieval_mechanics": _median(stage(records_scene, "l2_retrieval")),
                "aria_answerer_llm": _median(stage(records_scene, "aria")),
                "caption_s": _median(stage(records_scene, "lazy_caption")),
                "l1_elysium_s": _median(stage(records_scene, "elysium")),
                "cerberus_verify_s": _median(stage(records_scene, "cerberus_v")),
                "other_caption_elysium_cerberus": _median(other_stage(records_scene)),
                "total": _median(stage(records_scene, "total")),
            },
        },
        "per_query_total_s": {
            "flat": stage(records_flat, "total"),
            "scene_sparse": stage(records_scene, "total"),
        },
        "per_query_component_s": {
            "flat": {
                "caption_s": stage(records_flat, "lazy_caption"),
                "l1_elysium_s": stage(records_flat, "elysium"),
                "cerberus_verify_s": stage(records_flat, "cerberus_v"),
            },
            "scene_sparse": {
                "caption_s": stage(records_scene, "lazy_caption"),
                "l1_elysium_s": stage(records_scene, "elysium"),
                "cerberus_verify_s": stage(records_scene, "cerberus_v"),
            },
        },
        "assertions": {
            "flat_arm_completed": len(records_flat) == N_Q,
            "scene_sparse_arm_completed": len(records_scene) == N_Q,
            "nextqa_cache_untouched": (nextqa_before == nextqa_after),
            "virat_cache_untouched": (virat_before == virat_after),
        },
        "fingerprints": {
            "nextqa_before": nextqa_before, "nextqa_after_setup": nextqa_after_setup, "nextqa_after": nextqa_after,
            "virat_before": virat_before, "virat_after_setup": virat_after_setup, "virat_after": virat_after,
        },
    }

    med_flat_total = summary["median_stage_s"]["flat"]["total"]
    med_ss_total = summary["median_stage_s"]["scene_sparse"]["total"]
    summary["e2e_speedup_ratio_median_total"] = (
        med_flat_total / med_ss_total if (med_flat_total and med_ss_total) else None
    )

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
