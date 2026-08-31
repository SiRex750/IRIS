"""ledger S1: end-to-end (query text -> final answer) speedup A/B at VIRAT
N=4,892 (the C1.4 point), flat vs scene_sparse, per-stage breakdown.

Measurement only. Calls the existing iris.query.query() full pipeline
unmodified; the only harness-side configuration is an explicit
aria.set_backend() override so the LEGACY cerberus_mode answerer (which
hardcodes LlamaBackend.DEFAULT_TEXT_MODEL="llama3.2:3b" and ignores
config.answerer_model -- verified by reading iris/aria.py, not touched
here) actually talks to granite4:micro via Ollama, matching the task's
stated config. set_backend() is an existing public API documented for
exactly this ("explicitly override the active LLM backend -- e.g. for
testing").

Read-only against the VIRAT/NextQA caches: loads the existing scene_sparse
.npz, builds the flat graph ONCE in memory from the same loaded frames (no
re-ingest, no re-decode to build the graph, nothing written back to disk),
identical setup to scripts/virat_latency_ab.py.

Per-arm protocol:
  1. Reset all frame.caption to None (fair start: no cross-arm caption-cache
     leakage -- idx_flat and idx_scene_sparse alias the same FrameRecord
     objects since idx_flat is a dataclasses.replace() of idx_scene_sparse
     without overriding `frames`).
  2. One warmup query (loads CLIP text-embed model, BLIP captioner, Ollama
     connection, NLI gate) -- discarded, not counted.
  3. Reset frame.caption to None again (undo warmup's own caption-cache
     writes, so the timed loop starts model-warm but content-cache-cold,
     matching a real fresh-session query mix).
  4. N_Q timed queries from the fixed TEXT_QUERIES list (real CLIP-ViT-B/32
     text-encoded queries, not synthetic samples), each a fresh call to
     iris.query.query() -- full path, no internals bypassed.

VERIFY: python scripts/e2e_speedup_ab.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import dataclasses
import hashlib
import json
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
OUT_JSON = REPO / "eval_results" / "e2e_speedup.json"
OUT_MD = REPO / "eval_results" / "e2e_speedup.md"

N_Q = 5  # reduced from the retrieval-only harnesses' 50 -- full pipeline (real
         # captioning + real LLM generation, CPU) costs ~1-2 orders of
         # magnitude more per query than retrieval mechanics alone; 5 real
         # end-to-end queries/arm was the budget that fit this task, with
         # full per-query spread reported (not just a mean) so this is
         # legible as N=5, not silently presented as N=50.
ANSWERER_MODEL = "granite4:micro"

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=30,
    scene_diag=True,
)


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
    aria.set_backend(aria.LlamaBackend(text_model=ANSWERER_MODEL))

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

    # GUARD: flat arm must complete (it did at 7.94s retrieval-only in C1.4).
    # If graph_mode="flat" ever raises/hangs here, that is config drift --
    # abort loudly rather than silently falling back to scene_sparse-only.
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
        # "any other stage" = lazy_caption + elysium + cerberus (everything
        # in the timings dict besides l2_retrieval [retrieval mechanics incl.
        # query embed] and aria [answerer/LLM]).
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
                "other_caption_elysium_cerberus": _median(other_stage(records_flat)),
                "total": _median(stage(records_flat, "total")),
            },
            "scene_sparse": {
                "l2_retrieval_retrieval_mechanics": _median(stage(records_scene, "l2_retrieval")),
                "aria_answerer_llm": _median(stage(records_scene, "aria")),
                "other_caption_elysium_cerberus": _median(other_stage(records_scene)),
                "total": _median(stage(records_scene, "total")),
            },
        },
        "per_query_total_s": {
            "flat": stage(records_flat, "total"),
            "scene_sparse": stage(records_scene, "total"),
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
