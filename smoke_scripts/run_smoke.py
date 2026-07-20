"""Day-2 production smoke test harness.

Calls ONLY the canonical production entry points: iris.ingest.ingest() / save_index() /
load_index(), and iris.query.query() (which internally dispatches to
iris.query._retrieve_with_l1 -> iris.query._build_retrieved -> iris.scene_retrieval.retrieve_scene_sparse
-> iris.l2_asphodel.L2Asphodel.retrieve_ppr, matching canonical_pipeline.md). No alternate/simplified
retrieval logic is implemented in this file. The only "extra" call this script makes beyond query()
is a second, identical call to iris.query._retrieve_with_l1 with the SAME query embedding purely to
read back L2 telemetry that query() computes internally but does not return in its result dict (see
NOTE below) -- this is instrumentation of the real call, not an alternative algorithm.

NOTE on answerer backend: IRISConfig defaults to answerer_backend="llama_server" @
http://127.0.0.1:8091/v1, which is not running in this environment (no llama-server process).
Ollama IS running locally with the exact same model (granite4:micro) already loaded, so this
harness uses answerer_backend="llama" @ http://localhost:11434/v1, model="granite4:micro" --
same model, different (already-running, already-cached) local transport. Recorded as a deviation
in smoke_report.md, not silently substituted.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import psutil

REPO = Path(r"C:\Users\swara\IRIS")
sys.path.insert(0, str(REPO))

import iris.aria as iris_aria              # noqa: E402
import iris.ingest as iris_ingest          # noqa: E402
import iris.query as iris_query            # noqa: E402
from iris.iris_config import IRISConfig    # noqa: E402

SMOKE_DIR = REPO / "smoke"
CACHE_DIR = SMOKE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def make_config() -> IRISConfig:
    cfg = IRISConfig()
    cfg.answerer_backend = "llama"
    cfg.answerer_endpoint = "http://localhost:11434/v1"
    cfg.answerer_model = "granite4:micro"
    return cfg


def graph_structural_fingerprint(index) -> str:
    """Hash of node IDs + edge (u,v,weight,edge_type) -- structural mutation detector.

    Deliberately excludes per-node retrieval_contributions/last_retrieval_score, which are
    documented (canonical_pipeline.md) query-time annotations written onto node objects by
    retrieve_ppr for debug/telemetry purposes -- those are NOT graph structure (no node/edge
    added or removed, no edge weight changed), so they are out of scope for gate 7 ("no
    production graph mutation during query") as specified: structure, not per-query metadata.
    """
    g = index._graph.graph
    node_ids = sorted(g.nodes())
    edges = sorted(
        (u, v, round(float(d.get("weight", 0.0)), 8), d.get("edge_type", ""))
        for u, v, d in g.edges(data=True)
    )
    payload = json.dumps({"nodes": node_ids, "edges": edges}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def l1_report(index, video_id: str, duration_s: float, ingest_wall_s: float, ingest_rss_delta_mb: float) -> dict:
    stats = index.stats or {}
    total = stats.get("total", 0)
    skipped = stats.get("skipped", 0)
    expensive = total - skipped
    survivors = index.frames
    return {
        "video_id": video_id,
        "video_duration_s": duration_s,
        "total_frames": total,
        "frames_decoded_pass2": expensive,  # candidate+salient+peak tiers underwent Pass-2 decode+feature extraction
        "frames_expensive_processing": expensive,
        "survivor_count": len(survivors),
        "retention_pct": round(100.0 * len(survivors) / total, 3) if total else None,
        "l1_runtime_s": round(ingest_wall_s, 4),
        "l1_rss_delta_mb": round(ingest_rss_delta_mb, 2),
        "selected_frames": [
            {
                "frame_idx": fr.frame_idx,
                "timestamp_s": round(fr.timestamp, 3),
                "action_score": round(fr.action_score, 6),
                "persistence_value": round(fr.persistence_value, 6),
                "codec_conf": round(fr.codec_conf, 6),
                "is_peak": fr.is_peak,
                "tier_pict_type": fr.pict_type,
            }
            for fr in survivors
        ],
    }


def l2_report(index, cfg, question: str) -> dict:
    """Instrumentation-only call to the exact canonical L2 functions (see module docstring)."""
    query_embedding, embed_telemetry = iris_query._call_embed_query(question, cfg)
    retrieved_frames, l1_telemetry = iris_query._retrieve_with_l1(index, query_embedding, cfg)

    g = index._graph.graph
    edge_type_counts: dict[str, int] = {}
    for _, _, d in g.edges(data=True):
        et = d.get("edge_type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    density = round(2 * n_edges / (n_nodes * (n_nodes - 1)), 6) if n_nodes > 1 else 0.0

    topk = []
    for f in retrieved_frames:
        contrib = f.get("retrieval_contributions") or {}
        topk.append({
            "frame_idx": f["frame_idx"],
            "timestamp_s": round(f["timestamp"], 3),
            "action_score": round(f["action_score"], 6),
            "sem_rank": contrib.get("sem_rank"),
            "codec_rank": contrib.get("codec_rank"),
            "personalization_seed": contrib.get("seed"),
            "ppr_score": contrib.get("ppr"),
            "lambda_used": contrib.get("lambda"),
            "teleport_fallback": contrib.get("teleport_fallback"),
            "last_retrieval_score": f.get("last_retrieval_score"),
            "tier": f.get("tier"),
            "scene_id": f.get("scene_id"),
        })

    if topk:
        ts_list = [r["timestamp_s"] for r in topk]
        legacy_scattered_span = {
            "start_s": min(ts_list), "end_s": max(ts_list),
            "label": "scattered_minmax_span (LEGACY, NOT an official predicted span -- see configs/peak_span_modes.json)",
        }
        peak = topk[0]
    else:
        legacy_scattered_span = None
        peak = None

    return {
        "query_embedding_identifier": hashlib.sha256(query_embedding.tobytes()).hexdigest()[:16],
        "query_embedding_norm": embed_telemetry.get("norm"),
        "embedding_backend": embed_telemetry.get("embedding_backend"),
        "node_count": n_nodes,
        "edge_count": n_edges,
        "edge_count_by_type": edge_type_counts,
        "graph_density": density,
        "top_k_ordered": topk,
        "selected_peak_NOT_A_REAL_PIPELINE_STAGE": peak,
        "predicted_temporal_span_NOT_A_REAL_PIPELINE_STAGE": legacy_scattered_span,
        "note": "IRIS's canonical production pipeline has NO CLIP-peak-reranking stage and NO "
                "predicted-temporal-span construction (confirmed by direct code trace, see "
                "canonical_pipeline.md). 'selected_peak' above is just the rank-1 retrieved frame; "
                "'predicted_temporal_span' is the min/max envelope of retrieved timestamps -- an "
                "explicitly-labeled LEGACY derived quantity per configs/peak_span_modes.json, never "
                "an actual model output. Peak-in-gold is intentionally NOT computed inline here; see "
                "smoke_report.md for why (gate: peak-in-gold only after prediction is frozen).",
        "l1_telemetry": l1_telemetry,
    }


def run_one_question(index, cfg, q: dict) -> dict:
    video_id = q["video_id"]
    qid = q["qid"]
    question = q["question"]

    rss_before = rss_mb()
    fp_before = graph_structural_fingerprint(index)

    t0 = time.perf_counter()
    l2 = l2_report(index, cfg, question)
    t_l2 = time.perf_counter() - t0

    t1 = time.perf_counter()
    prompt_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    choices = q.get("choices")
    try:
        # choices are the multiple-choice OPTIONS only -- gold_answer_idx is
        # never passed into query(); it is read back out of selected_ids.json
        # only for the post-hoc gold_answer_idx field recorded below.
        result = iris_query.query(question, index, cfg, choices=choices)
        error = None
    except Exception as exc:  # noqa: BLE001
        result = None
        error = {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    t_full = time.perf_counter() - t1

    fp_after = graph_structural_fingerprint(index)
    rss_after = rss_mb()

    record = {
        "video_id": video_id,
        "qid": qid,
        "family": q["family"],
        "question": question,
        "choices": choices,
        "gold_answer_idx": q.get("gold_answer_idx"),  # recorded for post-hoc scoring only, never passed to query()
        "gold_spans": q["gold_spans"],
        "gold_span_total_len_s": q["gold_span_total_len_s"],
        "span_category": q["span_category"],
        "layer2": l2,
        "layer3": None,
        "answerer_prompt_hash": prompt_hash,
        "error": error,
        "graph_nonmutation": {
            "fingerprint_before": fp_before,
            "fingerprint_after": fp_after,
            "structurally_unmutated": fp_before == fp_after,
        },
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "timings_s": {"l2_instrumentation_call": round(t_l2, 4), "full_query_call": round(t_full, 4)},
    }

    if result is not None:
        record["layer3"] = {
            "frames_captioned": result["frames_decoded_for_captions"],
            "captions_in_answerer_order": [
                blk for blk in result["context_text"].split("\n\n---\n\n") if blk.strip()
            ],
            "raw_answer": result["raw_answer"],
            "parsed_final_answer": result["answer"],
            "predicted_choice_idx": result.get("predicted_choice_idx"),
            "verified": result["verified"],
            "verified_claims": result["verified_claims"],
            "rejected_claims": result["rejected_claims"],
            "unverifiable_claims": result["unverifiable_claims"],
            "abstained": result["answer"] == "Insufficient verified evidence to answer this question.",
            "abstention_reason": (
                "no verified claims survived CerberusV verification" if
                result["answer"] == "Insufficient verified evidence to answer this question." else None
            ),
            "retrieved_frame_idxs": result["retrieved_frame_idxs"],
            "timings_s": result["timings"],
        }

    return record


def main():
    selected = json.loads((SMOKE_DIR / "selected_ids.json").read_text())
    cfg = make_config()

    smoke_log = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        smoke_log.append(line)

    per_video_index = {}
    ingest_records = []
    cache_parity = {}
    layer1_rows = []

    videos = selected["videos"]
    log(f"Videos to ingest (cold): {videos}")

    for vid in videos:
        q_for_vid = next(q for q in selected["questions"] if q["video_id"] == vid)
        video_path = REPO / q_for_vid["video_path"]
        duration_s = q_for_vid["video_duration_s"]

        rss_before = rss_mb()
        t0 = time.perf_counter()
        try:
            index = iris_ingest.ingest(str(video_path), cfg)
            ingest_error = None
        except Exception as exc:  # noqa: BLE001
            index = None
            ingest_error = {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        t_ingest = time.perf_counter() - t0
        rss_after = rss_mb()

        if index is None:
            log(f"INGEST FAILED for {vid}: {ingest_error}")
            ingest_records.append({"video_id": vid, "error": ingest_error})
            continue

        per_video_index[vid] = index
        l1 = l1_report(index, vid, duration_s, t_ingest, rss_after - rss_before)
        layer1_rows.append(l1)
        log(f"Ingested {vid}: total={l1['total_frames']} survivors={l1['survivor_count']} "
            f"retention={l1['retention_pct']}% wall={l1['l1_runtime_s']}s rss_delta={l1['l1_rss_delta_mb']}MB")

        # cache parity: save, reload, compare structural fingerprint + frame count + a hash of embeddings
        cache_path = CACHE_DIR / vid
        iris_ingest.save_index(index, str(cache_path))
        reloaded = iris_ingest.load_index(str(cache_path))

        def frame_embedding_hash(idx):
            parts = []
            for fr in sorted(idx.frames, key=lambda f: f.frame_idx):
                emb = fr.clip_embedding
                parts.append(f"{fr.frame_idx}:{emb.tobytes().hex() if emb is not None else 'none'}")
            return hashlib.sha256("|".join(parts).encode()).hexdigest()

        fp_fresh = graph_structural_fingerprint(index)
        fp_reload = graph_structural_fingerprint(reloaded)
        emb_hash_fresh = frame_embedding_hash(index)
        emb_hash_reload = frame_embedding_hash(reloaded)

        cache_parity[vid] = {
            "fresh_frame_count": len(index.frames),
            "reloaded_frame_count": len(reloaded.frames),
            "fresh_graph_fingerprint": fp_fresh,
            "reloaded_graph_fingerprint": fp_reload,
            "graph_fingerprint_matches": fp_fresh == fp_reload,
            "fresh_embedding_hash": emb_hash_fresh,
            "reloaded_embedding_hash": emb_hash_reload,
            "embedding_hash_matches": emb_hash_fresh == emb_hash_reload,
            "PARITY_PASS": (fp_fresh == fp_reload) and (emb_hash_fresh == emb_hash_reload) and (len(index.frames) == len(reloaded.frames)),
        }
        log(f"Cache parity {vid}: {'PASS' if cache_parity[vid]['PARITY_PASS'] else 'FAIL'}")
        # keep using the fresh in-memory index for querying (reloaded index is used only for parity check)

    (SMOKE_DIR / "layer1_outputs.csv").write_text(
        "video_id,duration_s,total_frames,frames_decoded_pass2,survivor_count,retention_pct,l1_runtime_s,l1_rss_delta_mb\n" +
        "\n".join(
            f"{r['video_id']},{r['video_duration_s']},{r['total_frames']},{r['frames_decoded_pass2']},"
            f"{r['survivor_count']},{r['retention_pct']},{r['l1_runtime_s']},{r['l1_rss_delta_mb']}"
            for r in layer1_rows
        ) + "\n"
    )
    json.dump(cache_parity, open(SMOKE_DIR / "cache_parity.json", "w"), indent=2)

    per_question_traces = []
    layer2_rows = []
    layer3_rows = []
    determinism_records = []

    for q in selected["questions"]:
        vid = q["video_id"]
        index = per_video_index.get(vid)
        if index is None:
            log(f"SKIP q{q['qid']} on {vid}: video failed to ingest")
            continue

        log(f"Q {vid}/{q['qid']} ({q['family']}, span={q['span_category']}): {q['question'][:70]}")
        run_a = run_one_question(index, cfg, q)
        log(f"  run A: answer={'<error>' if run_a['error'] else run_a['layer3']['parsed_final_answer'][:80]!r} "
            f"verified={None if run_a['error'] else run_a['layer3']['verified']} "
            f"nonmutated={run_a['graph_nonmutation']['structurally_unmutated']}")

        # determinism: repeat the identical question against the SAME loaded index
        run_b = run_one_question(index, cfg, q)
        log(f"  run B: answer={'<error>' if run_b['error'] else run_b['layer3']['parsed_final_answer'][:80]!r}")

        same_answer = (not run_a["error"] and not run_b["error"] and
                       run_a["layer3"]["parsed_final_answer"] == run_b["layer3"]["parsed_final_answer"] and
                       run_a["layer3"]["retrieved_frame_idxs"] == run_b["layer3"]["retrieved_frame_idxs"])
        determinism_records.append({
            "video_id": vid, "qid": q["qid"],
            "run_a_error": run_a["error"], "run_b_error": run_b["error"],
            "run_a_answer": None if run_a["error"] else run_a["layer3"]["parsed_final_answer"],
            "run_b_answer": None if run_b["error"] else run_b["layer3"]["parsed_final_answer"],
            "run_a_retrieved": None if run_a["error"] else run_a["layer3"]["retrieved_frame_idxs"],
            "run_b_retrieved": None if run_b["error"] else run_b["layer3"]["retrieved_frame_idxs"],
            "DETERMINISTIC": same_answer,
        })

        per_question_traces.append({"video_id": vid, "qid": q["qid"], "run_a": run_a, "run_b": run_b})

        l2 = run_a["layer2"]
        layer2_rows.append({
            "video_id": vid, "qid": q["qid"], "node_count": l2["node_count"], "edge_count": l2["edge_count"],
            "graph_density": l2["graph_density"], "edge_count_by_type": json.dumps(l2["edge_count_by_type"]),
            "query_embedding_id": l2["query_embedding_identifier"], "top1_frame_idx": (l2["top_k_ordered"][0]["frame_idx"] if l2["top_k_ordered"] else None),
            "top1_ppr_score": (l2["top_k_ordered"][0]["ppr_score"] if l2["top_k_ordered"] else None),
            "l2_runtime_s": run_a["timings_s"]["l2_instrumentation_call"],
        })

        if not run_a["error"]:
            l3 = run_a["layer3"]
            layer3_rows.append({
                "video_id": vid, "qid": q["qid"], "frames_captioned": l3["frames_captioned"],
                "verified": l3["verified"], "abstained": l3["abstained"],
                "caption_latency_s": l3["timings_s"]["lazy_caption"],
                "answer_latency_s": l3["timings_s"]["aria"],
                "verification_latency_s": l3["timings_s"]["cerberus_v"],
                "total_latency_s": l3["timings_s"]["total"],
            })
        else:
            layer3_rows.append({
                "video_id": vid, "qid": q["qid"], "frames_captioned": None, "verified": None,
                "abstained": None, "caption_latency_s": None, "answer_latency_s": None,
                "verification_latency_s": None, "total_latency_s": None,
            })

    with open(SMOKE_DIR / "per_question_trace.jsonl", "w") as f:
        for rec in per_question_traces:
            f.write(json.dumps(rec) + "\n")

    def write_csv(path, rows, cols):
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    write_csv(SMOKE_DIR / "layer2_outputs.csv", layer2_rows,
               ["video_id", "qid", "node_count", "edge_count", "graph_density", "edge_count_by_type",
                "query_embedding_id", "top1_frame_idx", "top1_ppr_score", "l2_runtime_s"])
    write_csv(SMOKE_DIR / "layer3_outputs.csv", layer3_rows,
               ["video_id", "qid", "frames_captioned", "verified", "abstained", "caption_latency_s",
                "answer_latency_s", "verification_latency_s", "total_latency_s"])

    json.dump({
        "n_questions": len(determinism_records),
        "n_deterministic": sum(1 for d in determinism_records if d["DETERMINISTIC"]),
        "ALL_DETERMINISTIC": all(d["DETERMINISTIC"] for d in determinism_records) if determinism_records else False,
        "records": determinism_records,
    }, open(SMOKE_DIR / "determinism.json", "w"), indent=2)

    nonmutation_records = [
        {"video_id": t["video_id"], "qid": t["qid"],
         "run_a_nonmutated": t["run_a"]["graph_nonmutation"]["structurally_unmutated"],
         "run_b_nonmutated": t["run_b"]["graph_nonmutation"]["structurally_unmutated"]}
        for t in per_question_traces
    ]
    json.dump({
        "ALL_NONMUTATED": all(r["run_a_nonmutated"] and r["run_b_nonmutated"] for r in nonmutation_records) if nonmutation_records else False,
        "records": nonmutation_records,
        "definition": "structural fingerprint = sha256(sorted node ids + sorted (u,v,weight,edge_type) "
                       "edge tuples) of index._graph.graph, taken immediately before and after each "
                       "query() call. Excludes per-node retrieval_contributions/last_retrieval_score "
                       "annotations (query-time metadata writes, not structural graph mutation).",
    }, open(SMOKE_DIR / "graph_nonmutation.json", "w"), indent=2)

    (SMOKE_DIR / "smoke.log").write_text("\n".join(smoke_log) + "\n")

    json.dump({
        "ingest_errors": [r for r in ingest_records if r.get("error")],
        "question_errors": [
            {"video_id": t["video_id"], "qid": t["qid"], "run_a_error": t["run_a"]["error"], "run_b_error": t["run_b"]["error"]}
            for t in per_question_traces if t["run_a"]["error"] or t["run_b"]["error"]
        ],
    }, open(SMOKE_DIR / "failures.json", "w"), indent=2)

    # Item 5: MiniCPM truncation rate was previously invisible in
    # layer3_outputs.csv -- surface it explicitly even when moondream (the
    # default captioner) was active for this run, so the field is always
    # present and its emptiness is legible rather than the metric just
    # being absent.
    json.dump(iris_aria.get_minicpm_truncation_stats(), open(SMOKE_DIR / "minicpm_truncation_stats.json", "w"), indent=2)

    log("DONE")


if __name__ == "__main__":
    main()
