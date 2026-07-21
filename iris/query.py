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
    from iris.iris_config import IRISConfig

    # 1. Rebuild config from index snapshot if None
    if config is None:
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

    # 2. If config is custom, check compatibility with index.config_snapshot
    if index.config_snapshot:
        idx_clip = index.config_snapshot.get("clip_revision")
        query_clip = getattr(config, "clip_revision", None)
        if idx_clip and query_clip and idx_clip != query_clip:
            raise ValueError(
                f"Incompatible configurations: Index was built using clip_revision='{idx_clip}', "
                f"but query requested clip_revision='{query_clip}'."
            )

        idx_graph = index.config_snapshot.get("graph_mode")
        query_graph = getattr(config, "graph_mode", None)
        if idx_graph and query_graph and idx_graph != query_graph:
            raise ValueError(
                f"Incompatible configurations: Index was built using graph_mode='{idx_graph}', "
                f"but query requested graph_mode='{query_graph}'."
            )

    return config


def _embed_query(question: str, config: Any) -> tuple[np.ndarray, dict]:
    """Query-text CLIP embedding with fail-fast validation and telemetry."""
    from iris._clip import get_clip_model
    telemetry = {
        "embedding_backend": "unknown",
        "norm": 0.0,
        "fallback_reason": "none",
        "effective_method": "direct"
    }
    try:
        import clip
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        telemetry["embedding_backend"] = f"torch-{device}"
    except Exception as e:
        telemetry["fallback_reason"] = f"Import error: {str(e)}"
        raise ImportError(f"Failed to import torch or clip: {e}")

    model, _ = get_clip_model(config)
    if model is None:
        telemetry["fallback_reason"] = "CLIP model not loaded"
        raise ValueError("CLIP model could not be loaded; cannot embed query.")

    try:
        try:
            text_input = clip.tokenize([question]).to(device)
            with torch.no_grad():
                qf = model.encode_text(text_input)
        except Exception as gpu_err:
            if device == "cuda":
                telemetry["fallback_reason"] = f"CUDA failed ({str(gpu_err)}); falling back to CPU"
                telemetry["effective_method"] = "fallback"
                device = "cpu"
                telemetry["embedding_backend"] = "torch-cpu"
                text_input = clip.tokenize([question]).to(device)
                model = model.to(device)
                with torch.no_grad():
                    qf = model.encode_text(text_input)
            else:
                raise gpu_err

        qf /= qf.norm(dim=-1, keepdim=True)
        emb = qf.cpu().numpy().flatten().astype(np.float32)
        norm_val = float(np.linalg.norm(emb))
        telemetry["norm"] = norm_val
        if norm_val < 1e-6:
            raise ValueError("Generated query embedding has near-zero norm.")
        return emb, telemetry
    except Exception as e:
        telemetry["fallback_reason"] = f"Encoding failed: {str(e)}"
        raise ValueError(f"Failed to encode query text: {e}")


def _split_claims(raw_answer: str) -> list[str]:
    """Sentence-level claim extraction.

    NOTE ON PARITY: this docstring previously claimed the body below was
    "Lifted VERBATIM from pipeline.py 632-639". That is no longer true --
    iris/pipeline.py has been refactored since that note was written and no
    longer contains any claim-splitting logic at all (its `run_pipeline()`
    now just calls iris.ingest.ingest() + iris.query.query()). This function
    is the sole, canonical claim-splitting implementation; there is nothing
    left in pipeline.py to stay in parity with. If a second copy of this
    logic is ever reintroduced elsewhere, apply any future fix here to that
    copy too rather than letting them diverge silently.

    Two bugs fixed here (found via debug_trace.py analysis of real
    granite4:micro outputs, see debug_traces/4279106208__q10 and
    debug_traces/3261079025__q4):

    1. The old bold-strip regex (`r'\\*\\*.*?\\*\\*:?\\s*'`) deleted the ENTIRE
       bolded span, not just the `**` markers -- so "...primarily to
       **ensure the child's safety** (Option B)." lost its actual answer
       content and became "...primarily to (Option B).", which CerberusV
       then had nothing to verify against and rejected. Fixed to strip a
       bold markdown LABEL (word(s) immediately followed by a colon, e.g.
       "**Summary:**") entirely -- these are structural headers, not claim
       content -- but otherwise only remove the `**` markers and KEEP the
       bolded text, since inline emphasis around real answer content (e.g.
       "**ensure the child's safety**") must survive into the claim.

    2. The sentence-boundary regex treats ANY `[.!?]` followed by whitespace
       as a sentence end, including a bare multiple-choice option label like
       "A." / "B." / ... "E." embedded mid-answer (e.g. "...evidence: A.
       support himself - ... B. strong wind - ..."). This orphaned each
       letter onto the END of the PRECEDING clause instead of the start of
       its own explanation, and truncated the resulting claims into
       fragments that didn't correspond to any real sentence. Fixed by
       protecting `<letter>. ` sequences (a single capital A-E followed by a
       period and whitespace, anywhere except literally the first character
       of the string) with a non-terminator placeholder before splitting,
       then restoring the period afterward -- so an option's letter stays
       attached to the start of its own explanation.
    """
    # Fix 1a: strip bold markdown LABELS specifically -- word(s) immediately
    # followed by a colon inside the ** markers (e.g. "**Summary:**") are
    # structural headers, not claim content, and are removed entirely.
    clean_answer = re.sub(r'\*\*([^*]+?):\*\*\s*', '', raw_answer)
    # Fix 1b: any OTHER bold span is inline emphasis around real content --
    # strip only the ** markers, keep the text (previously this deleted it).
    clean_answer = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_answer)
    clean_answer = re.sub(r'^\s*[-*]\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'^\s*\d+\.\s+', '', clean_answer, flags=re.MULTILINE)
    clean_answer = re.sub(r'\n+', ' ', clean_answer).strip()

    # Fix 2: protect mid-answer multiple-choice letter labels ("A. " .. "E. ")
    # from being mistaken for a sentence terminator. Skip index 0 -- if the
    # answer opens with a bare option label there is no preceding clause for
    # it to be wrongly re-attached to, so nothing to protect against there.
    def _protect_option_label(m: re.Match) -> str:
        return m.group(1) + '.\x00'

    if clean_answer:
        protected = clean_answer[0] + re.sub(
            r'\b([A-E])\.\s', _protect_option_label, clean_answer[1:]
        )
    else:
        protected = clean_answer

    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)
    raw_sentences = [s.replace('\x00', ' ') for s in raw_sentences]
    return [s.strip() for s in raw_sentences if len(s.strip()) >= 12]


def _build_retrieved(index: IRISIndex, query_embedding: np.ndarray, config: Any,
                      trace: dict | None = None) -> list[dict]:
    """Mirror of wrapper_l2_retrieve's node->dict mapping + fallback (419-460),
    using the pre-built index._graph and index.frames. NO geometry keys (L1
    parity). Scores come from the graph node; is_peak/embedding/luma_entropy/
    caption come from the matching FrameRecord.

    trace: optional dict, forwarded to retrieve_scene_sparse (scene_sparse
    graph_mode) or populated minimally for the flat-graph path. Read-only
    instrumentation for iris.debug_trace -- see retrieve_scene_sparse's
    docstring; does not change what is retrieved.
    """
    index_graph_mode = index.config_snapshot.get("graph_mode", "flat") if index.config_snapshot else "flat"
    query_graph_mode = getattr(config, "graph_mode", "flat")
    if query_graph_mode != index_graph_mode:
        raise ValueError(
            f"Query graph_mode '{query_graph_mode}' does not match "
            f"index graph_mode '{index_graph_mode}'."
        )

    if query_graph_mode == "scene_sparse":
        from iris.scene_retrieval import retrieve_scene_sparse
        return retrieve_scene_sparse(index, query_embedding, config, trace=trace)

    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    frame_map = {fr.frame_idx: fr for fr in index.frames}
    graph = index._graph

    retrieved: list[dict] = []
    ranking_mode = getattr(config, "ranking_mode", "legacy")
    if graph is not None:
        if ranking_mode == "ppr":
            lambda_ = getattr(config, "ppr_lambda", 0.5)
            damping  = getattr(config, "ppr_damping", 0.5)
            retrieved_nodes = graph.retrieve_ppr(
                query_embedding,
                top_k=l2_retrieve_top_k,
                damping=damping,
                lambda_=lambda_,
            )
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
    if trace is not None:
        trace.update({
            "branch": "flat_" + ranking_mode,
            "shortlisted_scene_ids": None,
            "num_scenes": None,
            "scene_scores": None,
            "margin": None,
            "tau": None,
            "base_pool": len(index.frames),
            "post_pull_pool": len(index.frames),
            "cross_scene_edges_added": None,
            "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved],
            "retrieved_timestamps": [f["timestamp"] for f in retrieved],
            "retrieved_scores": [f["last_retrieval_score"] for f in retrieved],
        })
    return retrieved


def _build_focus_hint(question: str | None, choices: list[str] | None = None) -> str | None:
    """Build the captioner's question-aware prompt suffix.

    Without this, captions are generated with no knowledge of the question or
    multiple-choice options, so a captioner produces generic scene-level text
    ("a woman and two children eating ice cream") with no person/clothing/color
    detail even when the question hinges on exactly that ("what did the boy in
    red do..."). Returns None (no hint) when there is no question -- e.g. a
    cold-cache pre-captioning pass that runs ahead of any specific query --
    so callers fall back to the old generic, question-blind prompt.
    """
    if not question:
        return None
    hint = f"Pay attention to: {question}"
    if choices:
        hint += f" Consider distinguishing details relevant to: {'; '.join(choices)}."
    return hint


def _build_answer_prompt(question: str, choices: list[str] | None = None) -> str:
    """Wire multiple-choice options into the answerer prompt (item 4).

    Previously the pipeline only supported open-ended free-text QA -- choices
    never reached the model even when the dataset (NExT-QA/NExT-GQA) provides
    5 options + a gold index. This still requires the model to justify its
    pick with claim-level grounding (not just parrot option text), so
    CerberusV/cerberus_layers can verify the answer the same way as before --
    only the instruction changes, not the free-text answer format or the
    downstream claim-splitting/verification pipeline.
    """
    if not choices:
        return question
    lettered = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n"
        f"Choose the best answer from the options below, but you must justify it using the "
        f"evidence in the provided frames/captions -- do not just repeat an option verbatim "
        f"without grounding it in visual evidence:\n{lettered}"
    )


def _match_predicted_choice(answer_text: str | None, choices: list[str] | None) -> int | None:
    """Best-effort post-hoc mapping from the model's free-text answer back to
    a choice index, for Acc@QA-style scoring. Returns None (not a guess) when
    the match is ambiguous or absent -- callers must not fabricate an index."""
    if not choices or not answer_text:
        return None
    text_lower = answer_text.lower()
    substring_matches = [i for i, c in enumerate(choices) if c and c.strip().lower() in text_lower]
    if len(substring_matches) == 1:
        return substring_matches[0]
    letter_match = re.search(r"\b([A-E])\b[).:]?", answer_text)
    if letter_match:
        idx = ord(letter_match.group(1)) - ord("A")
        if 0 <= idx < len(choices):
            return idx
    return None


def _ensure_captions(index: IRISIndex, retrieved_frames: list[dict], config: Any = None,
                      question: str | None = None, choices: list[str] | None = None,
                      trace: dict | None = None) -> int:
    """Lazily caption retrieved frames lacking a cached caption.

    trace: optional dict; when not None, trace["captions"] is populated with
    one {frame_idx, timestamp, caption, caption_length, caption_generation_time}
    record per frame actually captioned in this call (read-only instrumentation
    for iris.debug_trace -- never changes which frames get captioned, the
    prompt used, or the caption text produced).

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
    import sys
    import warnings
    from iris._clip import get_semantic_and_clip_caption

    focus_hint = _build_focus_hint(question, choices)
    device = _device()
    frames_decoded_for_captions = 0
    # GOP-aware batching (item 7): process targets in timestamp order with a
    # SINGLE seek + one continuous forward decode pass, instead of one
    # independent seek-and-scan per target. Confirmed via a real smoke-test
    # anomaly that independent per-target seeks redundantly re-decode the
    # same GOP prefix when multiple targets share a GOP: 5 targets at
    # frame_idx [83, 98, 110, 138, 140] (all inside one long GOP) cost 574
    # decoded frames -- matching 83+98+110+138+140=569, i.e. each later
    # target re-paid the decode cost of every earlier target in the same
    # GOP. A single ordered pass covers the same span once.
    missing_frs_sorted = sorted(missing_frs, key=lambda fr: fr.timestamp)
    container = av.open(str(index.video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        tolerance = 0.5 / fps  # half a frame-interval

        first_target_pts = int(round(missing_frs_sorted[0].timestamp / stream.time_base))
        container.seek(first_target_pts, stream=stream)  # backward=True default -> keyframe at/before

        target_iter = iter(missing_frs_sorted)
        current_target = next(target_iter, None)

        for frame in container.decode(stream):
            if current_target is None:
                break
            frames_decoded_for_captions += 1
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            # A single decoded frame may satisfy several pending targets
            # whose timestamps all fall at/before it (e.g. near-duplicate
            # timestamps) -- consume all of them before advancing the decode.
            while current_target is not None and frame_time >= current_target.timestamp - tolerance:
                try:
                    pil_img = frame.to_image()
                except Exception:
                    pil_img = None
                if trace is not None:
                    import time as _time
                    _t0 = _time.perf_counter()
                current_target.caption = get_semantic_and_clip_caption(
                    pil_img, frame, current_target.clip_embedding, device, config=config, focus_hint=focus_hint,
                )
                if trace is not None:
                    _gen_time = _time.perf_counter() - _t0
                    cap = current_target.caption
                    cap_text = cap.get("semantic_caption") if isinstance(cap, dict) else cap
                    trace.setdefault("captions", []).append({
                        "frame_idx": current_target.frame_idx,
                        "timestamp": current_target.timestamp,
                        "caption": cap_text,
                        "caption_length": len(cap_text) if cap_text else 0,
                        "caption_generation_time": _gen_time,
                    })
                current_target = next(target_iter, None)
    finally:
        container.close()

    for f in retrieved_frames:
        fr = frame_map.get(f["frame_idx"])
        if fr is not None:
            f["caption"] = fr.caption

    survivor_count = len(index.frames)
    seeks_requested = len(missing_frs)
    print(
        f"[_ensure_captions] video={index.video_path!r} unique_frame_seeks_requested={seeks_requested} "
        f"frames_decoded={frames_decoded_for_captions} survivor_count={survivor_count} "
        f"decode_overhead_ratio={round(frames_decoded_for_captions / survivor_count, 2) if survivor_count else None}",
        file=sys.stderr,
    )
    if survivor_count > 0 and frames_decoded_for_captions > 3 * survivor_count:
        warnings.warn(
            f"QUERY-CAPTION-001: GOP-seek decode overhead: frames_decoded_for_captions="
            f"{frames_decoded_for_captions} > 3x survivor_count={survivor_count} "
            f"(unique_frame_seeks_requested={seeks_requested}) for video={index.video_path!r}. "
            f"This can indicate very sparse I-frames (long GOPs) forcing a long forward decode "
            f"to reach mid-GOP caption targets.",
            RuntimeWarning,
            stacklevel=2,
        )

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


def _generate_answer_claims_v2(question: str, context: str, max_tokens: int | None = None, model: str | None = None, config: Any = None):
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

    raw = aria.generate_v2(prompt=question, context=context, max_tokens=max_tokens, model=model, config=config)
    try:
        parsed = AnswerClaims.from_json(_extract_json_object(raw))
        return parsed, raw, False, 1, failure_labels
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        failure_labels.append(label)
        corrective = _corrective_message(question, label, e)
        raw2 = aria.generate_v2(prompt=corrective, context=context, max_tokens=max_tokens, model=model, config=config)
        try:
            parsed2 = AnswerClaims.from_json(_extract_json_object(raw2))
            return parsed2, raw2, False, 2, failure_labels
        except Exception as e2:
            label2 = getattr(e2, "taxonomy_label", None) or "other"
            failure_labels.append(label2)
            return None, raw2, True, 2, failure_labels


def _generate_answer_claims_v2_wire(question: str, context: str, max_tokens: int | None = None, model: str | None = None, config: Any = None):
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

    def clean_q(q):
        import re
        return re.sub(r'[^\w\s]', '', q.lower()).strip()

    raw = aria.generate_v2(prompt=question, context=context, max_tokens=max_tokens, model=model, schema_format=True, config=config)
    try:
        try:
            raw_obj = json.loads(raw)
        except json.JSONDecodeError as jde:
            from iris.claim_contract import MalformedJSONError
            raise MalformedJSONError(str(jde)) from jde
        parsed = AnswerClaims.from_wire(raw_obj)
        if clean_q(parsed.query) != clean_q(question):
            import logging
            logging.getLogger(__name__).warning(
                "Wire query %r does not match original question %r",
                parsed.query, question,
            )
        return parsed, raw, False, 1, failure_labels
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        failure_labels.append(label)
        corrective = _corrective_message(question, label, e, wire=True)
        raw2 = aria.generate_v2(prompt=corrective, context=context, max_tokens=max_tokens, model=model, schema_format=True, config=config)
        try:
            try:
                raw_obj2 = json.loads(raw2)
            except json.JSONDecodeError as jde2:
                from iris.claim_contract import MalformedJSONError
                raise MalformedJSONError(str(jde2)) from jde2
            parsed2 = AnswerClaims.from_wire(raw_obj2)
            if clean_q(parsed2.query) != clean_q(question):
                import logging
                logging.getLogger(__name__).warning(
                    "Retry wire query %r does not match original question %r",
                    parsed2.query, question,
                )
            return parsed2, raw2, False, 2, failure_labels
        except Exception as e2:
            label2 = getattr(e2, "taxonomy_label", None) or "other"
            failure_labels.append(label2)
            return None, raw2, True, 2, failure_labels


def _retrieve_with_l1(index: IRISIndex, query_embedding: np.ndarray, config: Any,
                       trace: dict | None = None) -> tuple[list[dict], dict]:
    """Retrieves frames using L1 Elysium Cache if use_l1 is enabled, with L2 fallback.

    trace: optional dict forwarded to _build_retrieved/retrieve_scene_sparse when
    the L2 path is taken. Read-only instrumentation for iris.debug_trace.
    """
    telemetry = {
        "l1_consulted": False,
        "l1_hit": False,
        "l1_candidate_count": 0,
        "l2_fallback": False
    }
    use_l1 = getattr(config, "use_l1", False)
    if not use_l1:
        return _build_retrieved(index, query_embedding, config, trace=trace), telemetry

    # Rebuild L1 Cache on loaded IRISIndex if it is not present
    if getattr(index, "_l1_cache", None) is None:
        from iris.l1_elysium import L1ElysiumCache
        from iris.cached_frame import CachedFrame
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        index._l1_cache = L1ElysiumCache(config)
        for fr in index.frames:
            motion = FrameMotionDescriptor(
                frame_idx=fr.frame_idx,
                timestamp_sec=fr.timestamp,
                luma_diff_energy=fr.luma_diff_energy,
                divergence=fr.divergence,
                curl=fr.curl,
                jacobian_frobenius=fr.jacobian_frobenius,
                hessian_max_eigenvalue=fr.hessian_max_eigenvalue,
                motion_entropy=fr.motion_entropy,
            )
            cf = CachedFrame(
                frame_idx=fr.frame_idx,
                timestamp_sec=fr.timestamp,
                action_score=fr.action_score,
                persistence_value=fr.persistence_value,
                is_peak=fr.is_peak,
                pagerank=fr.pagerank_score,
                motion=motion,
                embedding=fr.clip_embedding,
                caption=fr.caption,
            )
            index._l1_cache.admit(cf)

    telemetry["l1_consulted"] = True
    telemetry["l1_candidate_count"] = len(index._l1_cache)

    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    l1_results = index._l1_cache.query(query_embedding, top_k=l2_retrieve_top_k)

    max_sim = 0.0
    if l1_results:
        max_sim = max(getattr(cf, "query_similarity", 0.0) for cf in l1_results)

    # Fall back explicitly to L2 when L1 cannot satisfy the query (similarity threshold 0.20 or empty)
    satisfies = len(l1_results) >= l2_retrieve_top_k and max_sim >= 0.20

    if satisfies:
        telemetry["l1_hit"] = True
        telemetry["l2_fallback"] = False
        frame_map = {fr.frame_idx: fr for fr in index.frames}
        retrieved = []
        for cf in l1_results:
            fr = frame_map.get(cf.frame_idx)
            retrieved.append({
                "frame_idx": cf.frame_idx,
                "timestamp": cf.timestamp_sec,
                "luma_diff_energy": getattr(fr, "luma_diff_energy", 0.0),
                "action_score": cf.action_score,
                "persistence_value": cf.persistence_value,
                "is_peak": cf.is_peak,
                "clip_embedding": cf.embedding,
                "luma_entropy": getattr(fr, "luma_entropy", 0.0),
                "caption": cf.caption,
                "pagerank_score": getattr(fr, "pagerank_score", 0.0),
                "packet_size": getattr(fr, "packet_size", 0.0),
                "pict_type": getattr(fr, "pict_type", "?"),
                "codec_conf": getattr(fr, "codec_conf", 0.5),
                "scene_id": getattr(fr, "scene_id", -1),
                "last_retrieval_score": max_sim,
            })
        # Re-admit to update access recency / hit counter
        wrapper_populate_cache(index._l1_cache, retrieved)
        if trace is not None:
            trace.update({
                "branch": "l1_hit",
                "shortlisted_scene_ids": None, "num_scenes": None, "scene_scores": None,
                "margin": None, "tau": None,
                "base_pool": telemetry["l1_candidate_count"], "post_pull_pool": telemetry["l1_candidate_count"],
                "cross_scene_edges_added": None,
                "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved],
                "retrieved_timestamps": [f["timestamp"] for f in retrieved],
                "retrieved_scores": [f["last_retrieval_score"] for f in retrieved],
            })
        return retrieved, telemetry
    else:
        telemetry["l1_hit"] = False
        telemetry["l2_fallback"] = True
        retrieved = _build_retrieved(index, query_embedding, config, trace=trace)
        # Update L1 access/PageRank state after retrieval
        wrapper_populate_cache(index._l1_cache, retrieved)
        pagerank_scores = {f["frame_idx"]: f.get("pagerank_score", 0.0) for f in retrieved}
        index._l1_cache.update_pagerank(pagerank_scores)
        return retrieved, telemetry


def _call_embed_query(question: str, config: Any) -> tuple[np.ndarray, dict]:
    res = _embed_query(question, config)
    if isinstance(res, tuple) and len(res) == 2:
        emb, tele = res
        return emb, tele if isinstance(tele, dict) else {}
    return res, {}


def _claim_text(claim: Any) -> str:
    """Extract the human-readable text of any Claim subtype (VisualClaim.assertion,
    MetadataClaim.source_text, AbsenceClaim.event, GlobalClaim.text) for the
    legacy-shaped verified/rejected/unverifiable string lists below."""
    for attr in ("assertion", "source_text", "event", "text"):
        val = getattr(claim, attr, None)
        if val is not None:
            return val
    return str(claim)


def _query_v2(question: str, index: IRISIndex, config: Any, choices: list[str] | None = None) -> dict:
    """cerberus_mode="v2" path: AnswerClaims JSON contract, layer 1/2/3
    router (iris.cerberus_layers.verify_answer), answer badge. Retrieval
    and lazy-captioning are identical to the legacy path -- only ARIA's
    prompt/parse and the verification step differ."""
    from iris.cerberus_layers import Evidence, _label_class, get_nli_gate, verify_answer

    t_l2_start = time.monotonic()
    query_embedding, embed_telemetry = _call_embed_query(question, config)
    retrieved_frames, l1_telemetry = _retrieve_with_l1(index, query_embedding, config)
    t_l2 = time.monotonic() - t_l2_start

    t_caption_start = time.monotonic()
    try:
        try:
            frames_decoded_for_captions = _ensure_captions(index, retrieved_frames, config, question=question, choices=choices)
        except TypeError:
            try:
                frames_decoded_for_captions = _ensure_captions(index, retrieved_frames, config)
            except TypeError:
                frames_decoded_for_captions = _ensure_captions(index, retrieved_frames)
    finally:
        aria.unload_captioner()
    t_caption = time.monotonic() - t_caption_start

    t_elysium_start = time.monotonic()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    context_text = cache_obj.as_context_text()
    t_elysium = time.monotonic() - t_elysium_start

    t_aria_start = time.monotonic()
    answer_prompt = _build_answer_prompt(question, choices)
    use_wire = getattr(config, "answerer_schema_format", True)
    gen_fn = _generate_answer_claims_v2_wire if use_wire else _generate_answer_claims_v2
    answer_claims, raw_answer, compliance_failed, attempts, failure_labels = gen_fn(
        answer_prompt, context_text,
        max_tokens=getattr(config, "answerer_max_tokens", None),
        model=getattr(config, "answerer_model", None),
        config=config,
    )
    t_aria = time.monotonic() - t_aria_start

    cerberus_mode = getattr(config, "cerberus_mode", "v2")

    if cerberus_mode == "none":
        # L3 OFF / "no verifier, raw answerer" arm: verify_answer is never
        # called, so get_nli_gate() (spaCy + DeBERTa NLI load) never runs --
        # the cost of a verification result is not paid when that result is
        # thrown away. verified is "not_applicable" (never True/False --
        # nothing was checked) and answer is always raw_answer (no
        # abstention message), regardless of compliance_failed.
        badge = "skipped"
        claim_verdicts: list[dict] = []
        core_claim_verdict: dict | None = None
        is_verified: bool | str | None = "not_applicable"
        verified_claims: list[str] = []
        rejected_claims: list[str] = []
        unverifiable_claims: list[str] = []
        t_cerberus = 0.0
        final_answer = raw_answer
    elif compliance_failed or answer_claims is None:
        badge = "unverified"
        claim_verdicts = []
        core_claim_verdict = None
        is_verified = False
        verified_claims = []
        rejected_claims = []
        unverifiable_claims = []
        t_cerberus = 0.0
        # Badge-consistent final answer: mirrors legacy's abstention contract
        # (present real content only when is_verified, else the same
        # abstention message) rather than always surfacing the raw,
        # possibly-unverified text.
        final_answer = "Insufficient verified evidence to answer this question."
    else:
        t_cerberus_start = time.monotonic()
        gate = get_nli_gate()
        evidence = Evidence(
            index=index,
            retrieved_frames=retrieved_frames,
            nli=gate,
        )
        verification = verify_answer(answer_claims, evidence)
        t_cerberus = time.monotonic() - t_cerberus_start

        badge = verification.badge
        claim_verdicts = [v.to_dict() for v in verification.claim_verdicts]
        core_claim_verdict = verification.core_claim_verdict.to_dict() if verification.core_claim_verdict else None
        is_verified = bool(badge in ("verified", "partially_verified", "partial"))

        # Legacy-shaped claim-text buckets (schema parity for callers like
        # pipeline.py's debug_info that predate v2 and read these keys).
        verified_claims = [
            _claim_text(v.claim) for v in verification.claim_verdicts if _label_class(v.label) == "pass"
        ]
        rejected_claims = [
            _claim_text(v.claim) for v in verification.claim_verdicts if _label_class(v.label) == "reject"
        ]
        unverifiable_claims = [
            _claim_text(v.claim) for v in verification.claim_verdicts if _label_class(v.label) == "unverifiable"
        ]

        final_answer = raw_answer if is_verified else "Insufficient verified evidence to answer this question."

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "context_text": context_text,
        "verified": is_verified,
        "nli_mocked": False,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "unverifiable_claims": unverifiable_claims,
        "answer_claims": answer_claims.to_dict() if answer_claims is not None else None,
        "compliance_failed": compliance_failed,
        "compliance_attempts": attempts,
        "compliance_failure_labels": failure_labels,
        "badge": badge,
        "claim_verdicts": claim_verdicts,
        "core_claim_verdict": core_claim_verdict,
        "predicted_choice_idx": _match_predicted_choice(final_answer, choices),
        "frames_processed": index.frames_processed,
        "peak_count": index.peak_count,
        "compression_ratio": index.skipped_frames_ratio,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
        "frames_decoded_for_captions": frames_decoded_for_captions,
        "query_telemetry": {
            "embedding_backend": embed_telemetry.get("embedding_backend", "unknown"),
            "norm": embed_telemetry.get("norm", 0.0),
            "fallback_reason": embed_telemetry.get("fallback_reason", "none"),
            "effective_method": embed_telemetry.get("effective_method", "direct"),
            **l1_telemetry
        },
        "timings": {
            "l2_retrieval": t_l2,
            "lazy_caption": t_caption,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v2": t_cerberus,
            "total": t_l2 + t_caption + t_elysium + t_aria + t_cerberus,
        },
    }


def query(question: str, index: IRISIndex, config: Any = None, choices: list[str] | None = None,
          debug_context: dict | None = None) -> dict:
    """Answer one question against a loaded index. No video read, no graph rebuild.
    Returns the same result-dict shape as pipeline.run_pipeline (parity target).

    Dispatches to _query_v2 when config.cerberus_mode is "v2" (the default)
    or "none" (retrieval+answering identical to "v2", verification skipped
    entirely -- see _query_v2's cerberus_mode branch); the legacy body below
    this check is otherwise completely unchanged and only runs when
    cerberus_mode="legacy" is passed explicitly. Note: debug_trace
    instrumentation (iris.debug_trace.DebugTrace) is wired into the legacy
    body only -- _query_v2 has no debug-trace integration yet.

    choices: optional multiple-choice option strings (e.g. NExT-QA/NExT-GQA's
    5 options). Never pass the gold answer index/text here -- only the
    options themselves. Threading choices in lets the captioner focus on
    distinguishing detail (see _build_focus_hint) and the answerer prompt
    ask for a grounded pick among them (see _build_answer_prompt); the
    returned "predicted_choice_idx" is a best-effort post-hoc match of the
    free-text answer back to a choice index (None if ambiguous/unmatched),
    for external Acc@QA-style scoring -- the pipeline itself never compares
    against a gold index.

    debug_context: optional dict of benchmark/manual-query metadata
    ({video_id, question_id, question_type, split, ground_truth_answer,
    ground_truth_options}) attached to the debug trace when
    config.debug_trace is True (see iris.debug_trace). Purely additive
    instrumentation -- has no effect on retrieval, captioning, prompting,
    verification, or the returned answer when debug_trace is False (the
    default), and every field is optional even when it is True.
    """
    config = _config_from_index(index, config)

    if getattr(config, "cerberus_mode", "legacy") in ("v2", "none"):
        return _query_v2(question, index, config, choices=choices)

    trace = None
    if getattr(config, "debug_trace", False):
        from iris.debug_trace import DebugTrace
        dctx = debug_context or {}
        video_id = dctx.get("video_id")
        if video_id is None:
            from pathlib import Path as _Path
            video_id = _Path(str(index.video_path)).stem
        trace = DebugTrace.start(
            out_root=getattr(config, "debug_trace_dir", "debug_traces"),
            video_id=video_id, question=question,
            question_type=dctx.get("question_type"), question_id=dctx.get("question_id"),
            split=dctx.get("split"),
        )
        trace.set_ground_truth(dctx.get("ground_truth_answer"), dctx.get("ground_truth_options") or choices)

    t_l2_start = time.monotonic()
    query_embedding, embed_telemetry = _call_embed_query(question, config)
    retrieval_trace = {} if trace is not None else None
    retrieved_frames, l1_telemetry = _retrieve_with_l1(index, query_embedding, config, trace=retrieval_trace)
    t_l2 = time.monotonic() - t_l2_start
    if trace is not None:
        trace.retrieval = retrieval_trace or {}
        trace.timings["retrieval_time"] = t_l2
        trace.capture_frames(str(index.video_path), retrieved_frames)

    t_caption_start = time.monotonic()
    caption_trace = {} if trace is not None else None
    try:
        try:
            frames_decoded_for_captions = _ensure_captions(index, retrieved_frames, config, question=question, choices=choices, trace=caption_trace)
        except TypeError:
            try:
                frames_decoded_for_captions = _ensure_captions(index, retrieved_frames, config)
            except TypeError:
                frames_decoded_for_captions = _ensure_captions(index, retrieved_frames)
    finally:
        aria.unload_captioner()
    t_caption = time.monotonic() - t_caption_start
    if trace is not None:
        # Full captions list for every RETRIEVED frame (task step 5), not just
        # frames newly captioned in this call -- a frame already captioned by
        # an earlier query against the same loaded index is a cache hit
        # (FrameRecord.caption survives across queries, see _ensure_captions'
        # docstring) and must still appear here with its existing caption
        # text, just with caption_generation_time=None (not measured now).
        newly_timed = {c["frame_idx"]: c for c in (caption_trace or {}).get("captions", [])}
        full_captions = []
        for f in retrieved_frames:
            if f["frame_idx"] in newly_timed:
                full_captions.append(newly_timed[f["frame_idx"]])
                continue
            cap = f.get("caption")
            cap_text = cap.get("semantic_caption") if isinstance(cap, dict) else cap
            full_captions.append({
                "frame_idx": f["frame_idx"],
                "timestamp": f["timestamp"],
                "caption": cap_text,
                "caption_length": len(cap_text) if cap_text else 0,
                "caption_generation_time": None,  # cache hit: already captioned by an earlier query on this index
            })
        trace.captions = full_captions
        trace.timings["caption_time"] = t_caption
        try:
            active_captioner = aria.get_captioner(config)
            trace.set_captioner_info(type(active_captioner).__name__, getattr(active_captioner, "model_name", None))
        except Exception:  # noqa: BLE001
            trace.set_captioner_info(None, None)

    t_elysium_start = time.monotonic()
    cache_obj = wrapper_init_l1_cache(config)
    wrapper_populate_cache(cache_obj, retrieved_frames)
    t_elysium = time.monotonic() - t_elysium_start

    t_aria_start = time.monotonic()
    context_text = cache_obj.as_context_text()
    answer_prompt = _build_answer_prompt(question, choices)
    if trace is not None:
        trace.timings["prompt_construction_time"] = time.monotonic() - t_aria_start
    t_granite_start = time.monotonic()
    granite_debug_capture = {} if trace is not None else None
    raw_answer = aria.generate(
        prompt=answer_prompt,
        context=context_text,
        model=getattr(config, "answerer_model", None),
        max_tokens=getattr(config, "answerer_max_tokens", None),
        config=config,
        debug_capture=granite_debug_capture,
    )
    t_granite = time.monotonic() - t_granite_start
    t_aria = time.monotonic() - t_aria_start
    if trace is not None:
        trace.timings["granite_generation_time"] = t_granite
        trace.set_granite(
            debug_capture=granite_debug_capture or {}, context_text=context_text,
            focus_hint=_build_focus_hint(question, choices),
            verification_instructions=None,  # cerberus_mode="legacy" verifies post-hoc via NLI/NER, not a second generative prompt -- see verification stage
            raw_answer=raw_answer, generation_time=t_granite,
        )

    claims = _split_claims(raw_answer)

    t_cerberus_start = time.monotonic()
    max_score = max((f.get("action_score", 0.0) for f in retrieved_frames), default=0.5)
    is_verified, verified_claims, rejected_claims, unverifiable_claims, is_mocked = \
        wrapper_cerberus_gate(claims, cache_obj, max_score, config)
    t_cerberus = time.monotonic() - t_cerberus_start
    if trace is not None:
        trace.timings["cerberus_verification_time"] = t_cerberus
        trace.set_verification(is_verified, verified_claims, rejected_claims, unverifiable_claims)

    if verified_claims:
        final_answer = " ".join(verified_claims)
    else:
        final_answer = "Insufficient verified evidence to answer this question."

    if trace is not None:
        dctx = debug_context or {}
        trace.set_final_answer(
            raw_answer=raw_answer, verified_answer=final_answer,
            ground_truth_answer=dctx.get("ground_truth_answer"),
            comparison_method="exact_match",
        )
        try:
            trace.finalize()
        except Exception:  # noqa: BLE001
            # Debug-trace finalization must never break the real query response.
            pass

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "context_text": context_text,
        "verified": is_verified,
        "nli_mocked": is_mocked,
        "verified_claims": verified_claims,
        "rejected_claims": rejected_claims,
        "unverifiable_claims": unverifiable_claims,
        "predicted_choice_idx": _match_predicted_choice(final_answer, choices),
        "frames_processed": index.frames_processed,
        "peak_count": index.peak_count,
        "compression_ratio": index.skipped_frames_ratio,
        "skipped_frames_ratio": index.skipped_frames_ratio,
        "storage_reduction_factor": index.storage_reduction_factor,
        "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
        "frames_decoded_for_captions": frames_decoded_for_captions,
        "query_telemetry": {
            "embedding_backend": embed_telemetry.get("embedding_backend", "unknown"),
            "norm": embed_telemetry.get("norm", 0.0),
            "fallback_reason": embed_telemetry.get("fallback_reason", "none"),
            "effective_method": embed_telemetry.get("effective_method", "direct"),
            **l1_telemetry
        },
        "timings": {
            "l2_retrieval": t_l2,
            "lazy_caption": t_caption,
            "elysium": t_elysium,
            "aria": t_aria,
            "cerberus_v": t_cerberus,
            "total": t_l2 + t_caption + t_elysium + t_aria + t_cerberus,
        },
    }
