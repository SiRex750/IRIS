"""Phase 3 query spine: query(question, index) -> result dict.

Runs PER question against a loaded IRISIndex. No video read, no graph rebuild
(the graph is already on index._graph). Wraps existing iris.* modules and
lifts the L1/Cerberus wrappers + claim-split verbatim from pipeline.py so
behavior is identical. Parity with pipeline.run_pipeline is the exit criterion.

NOTE: the retrieved-frame dict deliberately carries NO motion-geometry keys,
so L1's FrameMotionDescriptor geometry is 0.0 — matching old pipeline.py
exactly (old never propagated geometry into L1 either). Wiring real geometry
into L1 is a deliberate Phase 6 change, not part of this restructure.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np

import iris.aria as aria
from iris import _perf
from iris.types import IRISIndex


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def wrapper_init_l1_cache(config: object = None) -> object:
    """Isolate L1 Elysium cache instantiation. Fallback to cache.py L1Cache if empty/stub."""
    try:
        from iris.l1_elysium import L1ElysiumCache
        return L1ElysiumCache(config)
    except (ImportError, AttributeError):
        from legacy.cache import L1Cache
        return L1Cache()


def wrapper_populate_cache(cache_obj: object, retrieved_frames: list[dict]) -> None:
    """Populate L1 Cache with knowledge triples or CachedFrames representing retrieved video frames."""
    try:
        from iris.l1_elysium import L1ElysiumCache
        use_elysium = isinstance(cache_obj, L1ElysiumCache)
    except ImportError:
        use_elysium = False

    if use_elysium:
        from iris.cached_frame import CachedFrame
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        for frame in retrieved_frames:
            motion = FrameMotionDescriptor(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                luma_diff_energy=frame.get("luma_diff_energy", 0.0),
                divergence=frame.get("divergence", 0.0),
                curl=frame.get("curl", 0.0),
                jacobian_frobenius=frame.get("jacobian_frobenius", 0.0),
                hessian_max_eigenvalue=frame.get("hessian_max_eigenvalue", 0.0),
                motion_entropy=frame.get("motion_entropy", 0.0)
            )
            cached_frame = CachedFrame(
                frame_idx=frame["frame_idx"],
                timestamp_sec=frame.get("timestamp", 0.0),
                action_score=frame.get("action_score", 0.0),
                persistence_value=frame.get("persistence_value", 0.0),
                is_peak=frame.get("is_peak", False),
                motion=motion,
                embedding=frame.get("clip_embedding", None),
                caption=frame.get("caption", None)
            )
            cache_obj.admit(cached_frame)
    else:
        from iris.triple import KnowledgeTriple
        for frame in retrieved_frames:
            frame_idx = frame["frame_idx"]
            timestamp = frame.get("timestamp", 0.0)
            res_energy = frame.get("luma_diff_energy", 0.0)
            action_score = frame.get("action_score", 0.0)
            
            # Generate semantic triple
            triple = KnowledgeTriple(
                subject=f"Frame {frame_idx} at {timestamp:.2f}s",
                verb="depicts",
                object=f"salient visual cues (residual energy {res_energy:.4f}, action score {action_score:.4f})"
            )
            # Use action_score/luma_diff_energy as importance ranking score
            score = action_score or res_energy
            if hasattr(cache_obj, "route_triple"):
                cache_obj.route_triple(triple, pagerank_score=score)
            else:
                cache_obj.add_fact(triple, pagerank_score=score)


def wrapper_cerberus_gate(claims: list[str], cache_obj: object, action_score: float, config: object) -> tuple[bool, list[str], list[str], list[str], bool]:
    """Isolate Cerberus-V NLI truth gate.

    Returns (is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked).

    is_verified is True only when BOTH rejected_claims AND unverifiable_claims are empty.
    A claim with no supporting evidence (unverifiable) is NOT the same as verified.
    """
    from iris.cerberus_v import CerberusV

    gate_obj = CerberusV()
    try:
        res = gate_obj.verify(claims, cache_obj, action_score, config)
        verified_claims = res.get("verified", [])
        rejected_claims = res.get("rejected", [])
        unverifiable_claims = res.get("unverifiable", [])
        # Fix 9b: require zero rejected AND zero unverifiable to count as verified.
        is_verified = len(rejected_claims) == 0 and len(unverifiable_claims) == 0
        is_mocked = False
    except Exception as e:
        error_msg = str(e)
        print(f"Error: CerberusV verification failed — gate closed, all claims unverifiable: {error_msg}")
        verified_claims = []
        rejected_claims = []
        unverifiable_claims = list(claims)
        is_verified = False
        is_mocked = False

    return is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked


def _config_from_index(index: IRISIndex, config: Any) -> Any:
    """Reconstruct the config the index was built with (snapshot is a dict;
    getattr on a dict returns defaults, so we must rebuild the dataclass)."""
    if config is not None:
        return config
    from iris.iris_config import IRISConfig
    try:
        return IRISConfig(**index.config_snapshot)
    except Exception as e:
        # QUERY-002: Warn when snapshot reconstruction falls back to defaults
        import sys
        print(
            f"[IRISQuery WARNING] Could not reconstruct config from index snapshot "
            f"({e}); falling back to IRISConfig() defaults. "
            "Results may differ from the original build.",
            file=sys.stderr,
        )
        return IRISConfig()


def _embed_query(question: str, config: Any) -> np.ndarray:
    """Query-text CLIP embedding. Lifted from wrapper_l2_retrieve (323-335)."""
    _t0 = time.perf_counter()
    try:
        from iris._clip import get_clip_model
        try:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return np.zeros(512, dtype=np.float32)
        model, _ = get_clip_model()
        if model is None:
            return np.zeros(512, dtype=np.float32)
        try:
            text_input = clip.tokenize([question]).to(device)
            with torch.no_grad():
                qf = model.encode_text(text_input)
                qf /= qf.norm(dim=-1, keepdim=True)
                return qf.cpu().numpy().flatten().astype(np.float32)
        except Exception:
            return np.zeros(512, dtype=np.float32)
    finally:
        _perf.record_time("query_embed_s", time.perf_counter() - _t0)


def _split_claims(raw_answer: str) -> list[str]:
    """Sentence-level claim extraction. Lifted VERBATIM from pipeline.py 632-639.
    Do not 'improve' the regex — parity depends on identical splitting."""
    clean_answer = re.sub(r'\*\*.*?\*\*:?\s*', '', raw_answer)
    clean_answer = re.sub(r'^\s*[-*]\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'^\s*\d+\.\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'\n+', ' ', clean_answer).strip()
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
    return [s.strip() for s in raw_sentences if len(s.strip()) >= 12]


def _build_retrieved(index: IRISIndex, query_embedding: np.ndarray, config: Any) -> list[dict]:
    """Mirror of wrapper_l2_retrieve's node->dict mapping + fallback (419-460),
    using the pre-built index._graph and index.frames. NO geometry keys (L1
    parity). Scores come from the graph node; is_peak/embedding/luma_entropy/
    caption come from the matching FrameRecord."""
    _t_total0 = time.perf_counter()
    try:
        return _build_retrieved_timed(index, query_embedding, config)
    finally:
        _perf.record_time("total_retrieval_s", time.perf_counter() - _t_total0)


def _build_retrieved_timed(index: IRISIndex, query_embedding: np.ndarray, config: Any) -> list[dict]:
    index_graph_mode = index.config_snapshot.get("graph_mode", "flat") if index.config_snapshot else "flat"
    query_graph_mode = getattr(config, "graph_mode", "flat")
    if query_graph_mode != index_graph_mode:
        raise ValueError(
            f"Query graph_mode '{query_graph_mode}' does not match "
            f"index graph_mode '{index_graph_mode}'."
        )

    if query_graph_mode == "scene_sparse":
        from iris.scene_retrieval import retrieve_scene_sparse
        return retrieve_scene_sparse(index, query_embedding, config)

    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    frame_map = {fr.frame_idx: fr for fr in index.frames}
    graph = index._graph

    retrieved: list[dict] = []
    ranking_mode = getattr(config, "ranking_mode", "legacy")
    if graph is not None:
        if ranking_mode == "ppr":
            lambda_ = getattr(config, "ppr_lambda", 0.5)
            damping  = getattr(config, "ppr_damping", 0.5)
            _t_ppr0 = time.perf_counter()
            retrieved_nodes = graph.retrieve_ppr(
                query_embedding,
                top_k=l2_retrieve_top_k,
                damping=damping,
                lambda_=lambda_,
            )
            _perf.record_time("flat_ppr_s", time.perf_counter() - _t_ppr0)
        else:
            retrieved_nodes = graph.retrieve(
                query_embedding,
                query_action_score=index.index_action_score,
                top_k=l2_retrieve_top_k,
            )
    else:
        retrieved_nodes = []

    if retrieved_nodes:
        for node in retrieved_nodes:
            fr = frame_map.get(node.frame_idx)
            retrieved.append({
                "frame_idx": node.frame_idx,
                "timestamp": node.timestamp,
                "luma_diff_energy": node.luma_diff_energy,
                "action_score": node.action_score,
                "persistence_value": node.persistence_value,
                "is_peak": getattr(fr, "is_peak", False),
                "clip_embedding": getattr(fr, "clip_embedding", None),
                "luma_entropy": getattr(fr, "luma_entropy", 0.0),
                "caption": getattr(fr, "caption", None),
                "pagerank_score": node.pagerank_score,
                "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
                "retrieval_contributions": getattr(node, "retrieval_contributions", {}),
                "tier": getattr(node, "tier", None),
                "scene_id": getattr(node, "scene_id", None),
                "pict_type": getattr(fr, "pict_type", "?"),
                "codec_conf": getattr(node, "codec_conf", 0.5),
                "divergence": getattr(fr, "divergence", 0.0),
                "curl": getattr(fr, "curl", 0.0),
                "jacobian_frobenius": getattr(fr, "jacobian_frobenius", 0.0),
                "hessian_max_eigenvalue": getattr(fr, "hessian_max_eigenvalue", 0.0),
                "motion_entropy": getattr(fr, "motion_entropy", 0.0),
            })

    if not retrieved:
        sorted_frames = sorted(
            index.frames,
            key=lambda fr: (fr.action_score, fr.luma_diff_energy),
            reverse=True,
        )
        for fr in sorted_frames[:l2_retrieve_top_k]:
            retrieved.append({
                "frame_idx": fr.frame_idx,
                "timestamp": fr.timestamp,
                "luma_diff_energy": fr.luma_diff_energy,
                "action_score": fr.action_score,
                "persistence_value": fr.persistence_value,
                "is_peak": fr.is_peak,
                "clip_embedding": fr.clip_embedding,
                "luma_entropy": fr.luma_entropy,
                "caption": fr.caption,
                "pagerank_score": 0.0,
                "last_retrieval_score": 0.0,
                "retrieval_contributions": {},
                "tier": "L1_PEAK" if fr.is_peak else "L3_CANDIDATE",
                "scene_id": None,
                "pict_type": getattr(fr, "pict_type", "?"),
                "codec_conf": getattr(fr, "codec_conf", 0.5),
                "divergence": getattr(fr, "divergence", 0.0),
                "curl": getattr(fr, "curl", 0.0),
                "jacobian_frobenius": getattr(fr, "jacobian_frobenius", 0.0),
                "hessian_max_eigenvalue": getattr(fr, "hessian_max_eigenvalue", 0.0),
                "motion_entropy": getattr(fr, "motion_entropy", 0.0),
            })
    return retrieved


def _ensure_captions(index: IRISIndex, retrieved_frames: list[dict]) -> int:
    """Lazily caption retrieved frames lacking a cached caption.

    Captioning moved out of ingest() to keep build cost proportional to
    survivors *embedded*, not survivors *captioned*. The cache lives on each
    FrameRecord.caption (keyed by frame_idx) so a frame is captioned at most
    once per loaded index, regardless of how many queries retrieve it.

    FrameRecords deliberately drop pil_image after build, so a caption-miss
    needs the frame's pixels back. Fetches each miss by SEEK, not a linear
    re-decode of the whole video: container.seek() lands on the keyframe
    at/before the target timestamp, then we decode forward only as far as
    that GOP requires. A full re-scan on the query hot path would reintroduce
    the exact full-decode cost the build-time gate exists to avoid, and at
    CCTV scale (multi-hour footage) that cost is prohibitive.

    Matches on pts/timestamp, not a recounted display index — display index
    is meaningless after a seek since decode() only yields frames from the
    seek point forward, not from frame 0.

    Returns frames_decoded_for_captions: total frames actually decoded across
    all seek-and-scan-forward targets (instrumentation for the CCTV-scale
    cost model — should be O(GOP size * misses), not O(video length)).
    """
    frame_map = {fr.frame_idx: fr for fr in index.frames}
    missing_frs = [
        frame_map[f["frame_idx"]] for f in retrieved_frames
        if frame_map.get(f["frame_idx"]) is not None
        and frame_map[f["frame_idx"]].caption is None
    ]
    if not missing_frs:
        return 0

    import av
    from iris._clip import get_semantic_and_clip_caption

    device = _device()
    frames_decoded_for_captions = 0
    container = av.open(str(index.video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        tolerance = 0.5 / fps  # half a frame-interval

        for fr in missing_frs:
            target_pts = int(round(fr.timestamp / stream.time_base))
            container.seek(target_pts, stream=stream)  # backward=True default -> keyframe at/before

            target_frame = None
            for frame in container.decode(stream):
                frames_decoded_for_captions += 1
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * stream.time_base)
                if frame_time >= fr.timestamp - tolerance:
                    target_frame = frame
                    break

            if target_frame is None:
                continue
            try:
                pil_img = target_frame.to_image()
            except Exception:
                pil_img = None
            fr.caption = get_semantic_and_clip_caption(pil_img, target_frame, fr.clip_embedding, device)
    finally:
        container.close()

    for f in retrieved_frames:
        fr = frame_map.get(f["frame_idx"])
        if fr is not None:
            f["caption"] = fr.caption

    return frames_decoded_for_captions


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> str:
    """Best-effort extraction of a single JSON object from a raw LLM response:
    strips a markdown code fence if present, else finds the first balanced
    {...} span. Returns a substring to hand to AnswerClaims.from_json --
    does not itself parse or validate; a malformed result still raises
    downstream and is handled by _generate_answer_claims_v2's retry."""
    fenced = _JSON_FENCE_RE.search(raw)
    text = fenced.group(1) if fenced else raw

    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _corrective_message(question: str, label: str, e: Exception, wire: bool = False) -> str:
    """Build a corrective retry prompt keyed by taxonomy label -- the
    is_core_invariant case is a semantic fix, not a parse error, and must
    not be told its JSON was invalid.

    wire (task 4, additive, defaults to False -- json-mode callers unchanged
    byte-for-byte): claim_field_shape means something different under the
    sentinel wire schema (iris.claim_contract.ANSWER_CLAIMS_WIRE_SCHEMA) than
    under the old nested json-mode shape -- a variant-required field left at
    its SENTINEL value, not a missing/extra dataclass kwarg -- so it gets its
    own phrasing naming the sentinel convention explicitly."""
    if label == "is_core_invariant":
        return (
            f"{question}\n\n"
            f"Your previous response parsed as valid JSON, but the claims did not satisfy "
            f"the AnswerClaims contract: {e}. Exactly ONE claim among your visual/absence "
            "claims must have is_core=true. Fix the is_core flags and reply again with ONLY "
            "the corrected JSON object -- no prose, no markdown code fences, no other text."
        )
    if label == "claim_field_shape" and wire:
        detail = (
            f"a claim left a field YOUR claim_type actually uses at its SENTINEL value "
            f"instead of a real one: {e}. Sentinels (-1 / \"\" / \"none\") are only correct "
            f"for fields your claim_type does NOT use -- fill the named field(s) with a real value."
        )
    elif label == "json_decode":
        detail = f"parse error: {e}"
    elif label == "schema_shape":
        detail = f"missing required top-level key: {e}"
    elif label == "unknown_claim_type":
        detail = f"unrecognized claim type: {e}"
    elif label == "bad_metadata_field":
        detail = f"invalid metadata field: {e}"
    elif label == "claim_field_shape":
        detail = f"claim object has missing/extra fields: {e}"
    else:
        detail = f"contract violation: {e}"
    return (
        f"{question}\n\n"
        f"Your previous response was not valid AnswerClaims JSON ({detail}). "
        "Reply again with ONLY the JSON object described in the system instructions -- "
        "no prose, no markdown code fences, no other text."
    )


def _generate_answer_claims_v2(question: str, context: str):
    """Cerberus v2 contract generation + strict parse, with ONE corrective
    retry on parse failure. NEVER falls back to legacy verification on
    failure -- that would corrupt the v2 compliance measurement; a second
    failure is reported as compliance_failed instead.

    Returns (answer_claims_or_None, raw_answer_of_last_attempt,
    compliance_failed, n_attempts, failure_labels).
    """
    from iris.claim_contract import AnswerClaims
    import iris.aria as aria

    failure_labels: list[str] = []

    raw = aria.generate_v2(prompt=question, context=context)
    try:
        parsed = AnswerClaims.from_json(_extract_json_object(raw))
        return parsed, raw, False, 1, failure_labels
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        failure_labels.append(label)
        corrective = _corrective_message(question, label, e)
        raw2 = aria.generate_v2(prompt=corrective, context=context)
        try:
            parsed2 = AnswerClaims.from_json(_extract_json_object(raw2))
            return parsed2, raw2, False, 2, failure_labels
        except Exception as e2:
            label2 = getattr(e2, "taxonomy_label", None) or "other"
            failure_labels.append(label2)
            return None, raw2, True, 2, failure_labels


def _generate_answer_claims_v2_wire(question: str, context: str, max_tokens: int | None = None):
    """Schema-constrained (grammar) contract generation via the flat,
    sentinel-valued ANSWER_CLAIMS_WIRE_SCHEMA (iris.claim_contract), with ONE
    corrective retry on a from_wire failure (task 4).

    Task 3 shipped this path with NO retry: grammar makes a PARSE retry moot
    (the response is always syntactically valid JSON matching the schema).
    But claim_field_shape -- a variant-required field left at its sentinel
    value instead of a real one -- is a SEMANTIC failure the grammar cannot
    itself prevent (enforcing "real, not sentinel" would need a per-variant
    anyOf, the exact SIGSEGV class being avoided). A corrective retry is
    therefore meaningful again here, unlike for a parse error: the model CAN
    act on "you left 'event' at its sentinel, give a real value" on a second
    attempt. Max attempts = 2, hard, same as the json-mode function.

    Returns (answer_claims_or_None, raw_answer_of_last_attempt,
    compliance_failed, n_attempts, failure_labels) -- same 5-tuple shape as
    _generate_answer_claims_v2 (json-mode).
    """
    from iris.claim_contract import AnswerClaims
    import iris.aria as aria

    failure_labels: list[str] = []

    raw = aria.generate_v2(prompt=question, context=context, max_tokens=max_tokens, schema_format=True)
    try:
        try:
            raw_obj = json.loads(raw)
        except json.JSONDecodeError as jde:
            from iris.claim_contract import MalformedJSONError
            raise MalformedJSONError(str(jde)) from jde
        parsed = AnswerClaims.from_wire(raw_obj)
        return parsed, raw, False, 1, failure_labels
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        failure_labels.append(label)
        corrective = _corrective_message(question, label, e, wire=True)
        raw2 = aria.generate_v2(prompt=corrective, context=context, max_tokens=max_tokens, schema_format=True)
        try:
            try:
                raw_obj2 = json.loads(raw2)
            except json.JSONDecodeError as jde2:
                from iris.claim_contract import MalformedJSONError
                raise MalformedJSONError(str(jde2)) from jde2
            parsed2 = AnswerClaims.from_wire(raw_obj2)
            return parsed2, raw2, False, 2, failure_labels
        except Exception as e2:
            label2 = getattr(e2, "taxonomy_label", None) or "other"
            failure_labels.append(label2)
            return None, raw2, True, 2, failure_labels


def _query_v2(question: str, index: IRISIndex, config: Any) -> dict:
    """cerberus_mode="v2" path: AnswerClaims JSON contract, layer 1/2/3
    router (iris.cerberus_layers.verify_answer), answer badge. Retrieval
    and lazy-captioning are identical to the legacy path -- only ARIA's
    prompt/parse and the verification step differ."""
    from iris.cerberus_layers import Evidence, get_nli_gate, verify_answer

    t_l2_start = time.time()
    query_embedding = _embed_query(question, config)
    retrieved_frames = _build_retrieved(index, query_embedding, config)
    t_l2 = time.time() - t_l2_start

    t_caption_start = time.time()
    try:
        frames_decoded_for_captions = _ensure_captions(index, retrieved_frames)
    finally:
        aria.unload_captioner()
    t_caption = time.time() - t_caption_start

    t_elysium_start = time.time()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    context_text = cache_obj.as_context_text()
    t_elysium = time.time() - t_elysium_start

    t_aria_start = time.time()
    answer_claims, raw_answer, compliance_failed, n_attempts, failure_labels = _generate_answer_claims_v2_wire(
        question, context_text, max_tokens=getattr(config, "answerer_max_tokens", None)
    )
    t_aria = time.time() - t_aria_start

    t_cerberus_start = time.time()
    if compliance_failed:
        badge = "unverified"
        claim_verdicts: list = []
        core_claim_verdict = None
    else:
        nli = get_nli_gate()
        evidence = Evidence(index=index, retrieved_frames=retrieved_frames, nli=nli)
        verification = verify_answer(answer_claims, evidence)
        badge = verification.badge
        claim_verdicts = verification.claim_verdicts
        core_claim_verdict = verification.core_claim_verdict
    t_cerberus = time.time() - t_cerberus_start

    return {
        "raw_answer": raw_answer,
        "answer_claims": answer_claims,
        "compliance_failed": compliance_failed,
        "n_llm_attempts": n_attempts,
        "compliance_failure_labels": failure_labels,
        "badge": badge,
        "claim_verdicts": claim_verdicts,
        "core_claim_verdict": core_claim_verdict,
        "frames_processed": index.frames_processed,
        "peak_count": index.peak_count,
        "compression_ratio": index.skipped_frames_ratio,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
        "frames_decoded_for_captions": frames_decoded_for_captions,
        "timings": {
            "l2_retrieval": t_l2,
            "lazy_caption": t_caption,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v2": t_cerberus,
            "total": t_l2 + t_caption + t_elysium + t_aria + t_cerberus,
        },
    }


def query(question: str, index: IRISIndex, config: Any = None) -> dict:
    """Answer one question against a loaded index. No video read, no graph rebuild.
    Returns the same result-dict shape as pipeline.run_pipeline (parity target).

    Dispatches to _query_v2 when config.cerberus_mode == "v2"; the legacy
    body below this check is otherwise completely unchanged (default
    cerberus_mode="legacy" -> zero behavior change)."""
    config = _config_from_index(index, config)

    if getattr(config, "cerberus_mode", "legacy") == "v2":
        return _query_v2(question, index, config)

    t_l2_start = time.time()
    query_embedding = _embed_query(question, config)
    retrieved_frames = _build_retrieved(index, query_embedding, config)
    t_l2 = time.time() - t_l2_start

    t_caption_start = time.time()
    try:
        frames_decoded_for_captions = _ensure_captions(index, retrieved_frames)
    finally:
        aria.unload_captioner()
    t_caption = time.time() - t_caption_start

    t_elysium_start = time.time()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    t_elysium = time.time() - t_elysium_start

    t_aria_start = time.time()
    context_text = cache_obj.as_context_text()
    raw_answer = aria.generate(prompt=question, context=context_text)
    t_aria = time.time() - t_aria_start

    claims = _split_claims(raw_answer)

    t_cerberus_start = time.time()
    max_score = max((f.get("action_score", 0.0) for f in retrieved_frames), default=0.5)
    is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked = \
        wrapper_cerberus_gate(claims, cache_obj, max_score, config)
    t_cerberus = time.time() - t_cerberus_start

    if verified_claims:
        final_answer = " ".join(verified_claims)
    else:
        final_answer = "Insufficient verified evidence to answer this question."

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "verified": is_verified,
        "nli_mocked": is_mocked,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "unverifiable_claims": unverifiable_claims,
        "frames_processed": index.frames_processed,
        "peak_count": index.peak_count,
        "compression_ratio": index.skipped_frames_ratio,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
        "frames_decoded_for_captions": frames_decoded_for_captions,
        "timings": {
            "l2_retrieval": t_l2,
            "lazy_caption": t_caption,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v": t_cerberus,
            "total": t_l2 + t_caption + t_elysium + t_aria + t_cerberus,
        },
    }
