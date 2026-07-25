"""Canonical retrieval entry point (section 15).

One function every evaluator/production query path should call instead of
each hand-rolling its own reformulation + retrieval + fusion glue:

    retrieve_for_question(question, index, config, type_code=None,
                           family=None, trace=None)
        -> (frames: list[dict], plan: QueryPlan | QueryPlanV2 | None, telemetry: dict)

Mode is selected ONLY by config.query_reformulation_mode -- never silently:
  "none"          -- raw verbatim question, single embedding, single retrieval.
                     This reproduces the frozen baseline exactly (same
                     _call_embed_query + _build_retrieved calls query() uses).
  "legacy"        -- the pre-V2 reformulate_query/fuse_ranked_results/
                     expand_temporal_neighbors path, kept only as an explicit
                     experimental baseline (section 11).
  "structured_v2" -- QueryPlanV2 + batched multi-query scoring (one PPR call)
                     + direction-aware temporal traversal (sections 3-14).

Production callers without a NExT-QA type code simply omit type_code/family;
build_query_plan_v2 falls back to lexical parsing + the safe original-question
fallback in that case, exactly as it does for NExT-QA descriptive questions.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from iris.frame_serialization import node_to_dict, frame_record_to_dict, assign_retrieval_rank
from iris.query_reformulation import (
    QueryPlanV2,
    Relation,
    all_embedding_texts,
    build_query_plan_v2,
    expand_temporal_neighbors,
    fuse_ranked_results,
    reformulate_query,
)
from iris.temporal_traversal import traverse_directional


def _multi_query_retrieve(index: Any, matrix: np.ndarray, weights: list[float], config: Any) -> list[dict]:
    """One PPR (or scene-sparse) call using the batched query matrix.

    Mirrors iris.query._build_retrieved's graph_mode dispatch, but threads
    query_embeddings/query_weights through instead of a single embedding.
    """
    graph_mode = getattr(config, "graph_mode", "flat")
    combine = getattr(config, "multi_query_combine", "weighted_max")
    top_k = getattr(config, "l2_retrieve_top_k", 4)

    if graph_mode == "scene_sparse":
        from iris.scene_retrieval import retrieve_scene_sparse
        return retrieve_scene_sparse(
            index, None, config,
            query_embeddings=matrix, query_weights=weights, multi_query_combine=combine,
        )

    graph = index._graph
    ranking_mode = getattr(config, "ranking_mode", "ppr")
    frame_map = {fr.frame_idx: fr for fr in index.frames}

    if graph is not None and ranking_mode == "ppr":
        nodes = graph.retrieve_ppr(
            None, top_k=top_k,
            damping=getattr(config, "ppr_damping", 0.5),
            lambda_=getattr(config, "ppr_lambda", 0.5),
            query_embeddings=matrix, query_weights=weights, multi_query_combine=combine,
        )
        retrieved = [node_to_dict(node, frame_map.get(node.frame_idx)) for node in nodes]
        return assign_retrieval_rank(retrieved)

    # ranking_mode="legacy" (alpha/beta/gamma/delta blend) has no multi-query
    # form -- fall back to the single highest-weighted query embedding rather
    # than silently averaging vectors. Recorded in telemetry by the caller.
    best_idx = int(np.argmax(weights)) if len(weights) else 0
    from iris.query import _build_retrieved
    return _build_retrieved(index, matrix[best_idx], config)


def _target_similarity_fn(index: Any, target_embedding: "np.ndarray | None"):
    if target_embedding is None:
        return None
    frame_map = {fr.frame_idx: fr for fr in index.frames}
    tnorm = float(np.linalg.norm(target_embedding))

    def _fn(frame_idx: int) -> float:
        fr = frame_map.get(frame_idx)
        if fr is None or fr.clip_embedding is None or tnorm < 1e-8:
            return 0.0
        emb = np.asarray(fr.clip_embedding, dtype=np.float32)
        enorm = float(np.linalg.norm(emb))
        if enorm < 1e-8:
            return 0.0
        return max(0.0, float(np.dot(emb, target_embedding) / (enorm * tnorm)))

    return _fn


def _retrieve_none_mode(question: str, index: Any, config: Any) -> tuple[list[dict], dict]:
    from iris.query import _call_embed_query, _build_retrieved

    embedding, embed_telemetry = _call_embed_query(question, config)
    frames = _build_retrieved(index, embedding, config)
    return frames, {"mode": "none", "embed": embed_telemetry, "num_clip_calls": 1, "num_ppr_calls": 1}


def _retrieve_legacy_mode(
    question: str, index: Any, config: Any, family: str | None,
) -> tuple[list[dict], Any, dict]:
    from iris.query import _call_embed_query, _build_retrieved

    max_queries = getattr(config, "max_retrieval_queries", 5)
    plan = reformulate_query(question, family=family, max_queries=max_queries)

    ranked_lists = []
    num_clip_calls = 0
    for query_text in plan.retrieval_queries:
        embedding, _ = _call_embed_query(query_text, config)
        num_clip_calls += 1
        ranked_lists.append(_build_retrieved(index, embedding, config))

    top_k = getattr(config, "l2_retrieve_top_k", 4)
    fused = fuse_ranked_results(ranked_lists, top_k=top_k, rrf_k=getattr(config, "legacy_rrf_k", 60))

    telemetry = {
        "mode": "legacy",
        "num_clip_calls": num_clip_calls,
        "num_ppr_calls": num_clip_calls,
        "retrieval_queries": list(plan.retrieval_queries),
    }

    if plan.needs_temporal_expansion and getattr(config, "temporal_traversal_mode", "none") == "legacy_symmetric":
        fused = expand_temporal_neighbors(
            index, fused, radius=2, max_frames=getattr(config, "max_context_frames", 8)
        )
        telemetry["legacy_temporal_expansion_applied"] = True

    return fused, plan, telemetry


def _retrieve_structured_v2(
    question: str, index: Any, config: Any, type_code: str | None, family: str | None,
) -> tuple[list[dict], QueryPlanV2, dict]:
    from iris.query import _call_embed_queries

    t_parse0 = time.monotonic()
    plan = build_query_plan_v2(question, type_code=type_code, family=family, config=config)
    t_parse = time.monotonic() - t_parse0

    texts = list(all_embedding_texts(plan))
    max_embeds = getattr(config, "max_retrieval_queries", 3)
    texts = texts[:max_embeds]

    t_embed0 = time.monotonic()
    matrix, embed_telemetry = _call_embed_queries(texts, config)
    t_embed = time.monotonic() - t_embed0

    weights = list(plan.query_weights[: len(texts)]) or [1.0] * len(texts)
    if len(weights) < len(texts):
        weights = weights + [1.0] * (len(texts) - len(weights))

    target_embedding = None
    if plan.target_query and plan.target_query in texts:
        target_embedding = matrix[texts.index(plan.target_query)]

    t_retrieve0 = time.monotonic()
    retrieved = _multi_query_retrieve(index, matrix, weights, config)
    t_retrieve = time.monotonic() - t_retrieve0

    traversal_mode = getattr(config, "temporal_traversal_mode", "none")
    t_traverse0 = time.monotonic()
    scene_boundaries_crossed = 0
    if plan.needs_temporal_traversal and traversal_mode == "directional" and retrieved:
        sim_fn = _target_similarity_fn(index, target_embedding)
        retrieved = traverse_directional(index, retrieved, plan, config, target_similarity_fn=sim_fn)
        scene_boundaries_crossed = sum(
            1 for f in retrieved
            if (f.get("retrieval_contributions") or {}).get("scene_hops_from_anchor", 0) > 0
        )
    elif plan.needs_temporal_traversal and traversal_mode == "legacy_symmetric" and retrieved:
        retrieved = expand_temporal_neighbors(
            index, retrieved, radius=2, max_frames=getattr(config, "max_context_frames", 8)
        )
    t_traverse = time.monotonic() - t_traverse0

    telemetry = {
        "mode": "structured_v2",
        "plan": {
            "relation": plan.relation,
            "relation_source": plan.relation_source,
            "temporal_direction": plan.temporal_direction,
            "type_code": plan.type_code,
            "family": plan.family,
            "parser_confidence": plan.parser_confidence,
            "fallback_reason": plan.fallback_reason,
            "corrections_applied": list(plan.corrections_applied),
            "aliases_applied": list(plan.aliases_applied),
            "notes": list(plan.notes),
            "anchor_queries": list(plan.anchor_queries),
            "target_query": plan.target_query,
            "position_prior": plan.position_prior,
            "occurrence_selector": plan.occurrence_selector,
        },
        "num_clip_texts_embedded": len(texts),
        "num_clip_batches": embed_telemetry.get("num_clip_batches", 0),
        "embed_cache_hits": embed_telemetry.get("cache_hits", 0),
        "embed_cache_misses": embed_telemetry.get("cache_misses", 0),
        "num_ppr_calls": 1,
        "scene_boundaries_crossed": scene_boundaries_crossed,
        "context_frame_count": len(retrieved),
        "timings": {
            "parse_sec": t_parse,
            "embed_sec": t_embed,
            "retrieve_sec": t_retrieve,
            "traverse_sec": t_traverse,
            "total_sec": t_parse + t_embed + t_retrieve + t_traverse,
        },
    }

    return retrieved, plan, telemetry


def retrieve_for_question(
    question: str,
    index: Any,
    config: Any,
    *,
    type_code: str | None = None,
    family: str | None = None,
    trace: dict | None = None,
) -> tuple[list[dict], Any, dict]:
    """Select raw/legacy/structured_v2 mode, build the plan, retrieve, and
    (for structured_v2) run direction-aware traversal + rerank. Returns
    (final_frames, plan_or_None, telemetry). Never mutates config or index.

    trace: optional dict; when config.query_trace_enabled is True, populated
    with the full telemetry this function already computed -- purely
    additive instrumentation, never changes what is retrieved.
    """
    mode = getattr(config, "query_reformulation_mode", "none")

    if mode == "legacy":
        frames, plan, telemetry = _retrieve_legacy_mode(question, index, config, family)
    elif mode == "structured_v2":
        frames, plan, telemetry = _retrieve_structured_v2(question, index, config, type_code, family)
    else:
        frames, telemetry = _retrieve_none_mode(question, index, config)
        plan = None

    if trace is not None and getattr(config, "query_trace_enabled", False):
        trace.update(telemetry)

    return frames, plan, telemetry
