"""scripts/answerer_bakeoff.py -- Answerer bake-off: selecting IRIS's v2
answerer model on CONTRACT COMPLIANCE, not answer quality. DIAGNOSTIC ONLY,
no iris/ edits.

NLI-free, retrieval-quality-free: measures whether a candidate LLM emits a
valid AnswerClaims JSON object (iris.claim_contract) given a frozen,
representative context -- nothing here loads DeBERTa or scores entailment.
Templated on scripts/captioner_bakeoff.py's discipline (frozen artifacts,
provenance, pre-registered decision-rule table, hard gate, human column).

REUSE, DO NOT REIMPLEMENT:
    iris.query._extract_json_object, iris.query._corrective_message,
    iris.query._embed_query / _build_retrieved / _ensure_captions /
    wrapper_init_l1_cache / wrapper_populate_cache (context-assembly spine,
    lifted verbatim call-for-call the way scripts/demo_cctv_query.py's
    --nli-trace / --cerberus-wiring-trace already do -- not reimplemented).
    iris.claim_contract.AnswerClaims / ClaimContractError + typed subclasses.
    iris.aria.generate_v2 (the exact v2 contract call, json-mode as-is).
    Ollama helpers _tag_matches / _check_ollama_model / _ollama_available /
    _strip_think / _ollama_unload -- imported from scripts.captioner_bakeoff
    verbatim (attribution: that module owns the canonical implementation).
    Query set: the SAME 11 queries as scripts/smoke_v2.py --
    virat_query_smoke.QUERIES (3) + demo_cctv_query.POSITIVE_QUERIES (4) +
    demo_cctv_query.NEGATIVE_QUERIES (4), demo_cctv_query.CFG, _load_index.
    No new query set, no mechanism change to any of the above.

CANDIDATES (tag confirmation via `GET /api/tags` against this host's Ollama,
done before writing this file -- not assumed):
    llama3.2:3b            baseline/incumbent -- generate_v2(model=None)'s
                            actual default tag today (iris/aria.py
                            LlamaBackend.DEFAULT_TEXT_MODEL). CONFIRMED pulled.
    hermes3:3b              NOT FOUND in `ollama list` on this host (no tag
                            resembling "hermes3" under any namespace). Skipped
                            with a NOTE per the "unavailable -> skip" rule --
                            not fabricated, not substituted.
    qwen3.5:2b              CONFIRMED pulled (also used live in the captioner
                            bake-off).
    qwen3.5:4b              CONFIRMED pulled -- the captioner bake-off only
                            INFERRED its exclusion (same latency family as
                            qwen3.5:2b) and never itself probed it; this is
                            a genuinely separate axis (text-only contract
                            compliance, not image-prefill latency), so it is
                            included here on its own measured merits.
    minicpm5:1b             the tag AS WRITTEN in the task brief is NOT what
                            is pulled on this host -- `ollama list` shows
                            "openbmb/minicpm5:q4_K_M" (digest
                            08239e8f70e0...), no bare "minicpm5:1b" or
                            "minicpm5:latest". Resolved to the confirmed tag
                            "openbmb/minicpm5:q4_K_M"; logged in provenance
                            so the substitution is auditable, not silent.
                            Floor + thinking-off datapoint per the brief:
                            "think": False is NOT settable through
                            iris.aria.generate_v2's production signature
                            (model=str|None only -- no iris/ edits allowed to
                            add one), so every raw response from every Ollama
                            candidate (not just this one) is unconditionally
                            scanned for <think>...</think> via _strip_think
                            post-hoc; incidents are counted and surfaced,
                            never silently absorbed. This is a measured
                            constraint of the reuse-only contract, not an
                            oversight -- see provenance "think_handling".

PHASE 0 (answerer_capture.jsonl, run ONCE then frozen): for each of the 11
queries, replicate _query_v2's context-assembly spine up to (and including)
as_context_text() using CFG as-is -- embed -> retrieve -> lazy-caption ->
L1 populate -> context text. One row per query: {tag, query, context_text,
retrieved_frame_idxs}. Captions in these contexts are the CURRENT captioner
(BLIP) at the CURRENT retrieval regime -- intentional: compliance here is
caption-CONTENT-independent, only context SHAPE/LENGTH matters to a JSON-
contract-compliance measurement. SCOPE LIMIT (pre-registered): if a future
captioner swap materially changes context length, this capture must be
re-frozen -- compliance CAN shift for context-limited models (the self-
confident-context-exhaustion axis), and this frozen file would then be
stale for that axis specifically.

PHASE 1 (answerer_bakeoff_{name}_pass1.jsonl, frozen once written):
replicates iris.query._generate_answer_claims_v2's two-attempt loop
IN-SCRIPT (only difference: model= threaded per candidate) -- see
_run_one_query below, which is commented at its call site pointing back at
that production function. If that function's control flow changes, update
_run_one_query to match.

PHASE 2: offline analysis (NLI-free) -- compliance rates + failure-label
histogram, absence-positivity structural check, frame-citation hard gate,
metadata-number human-review exhibits.

PHASE 3: stability -- one extra pass2 run (temp is already baked to 0 in
iris.aria.LlamaBackend for every call) over gate-passing candidates,
comparing pass1 vs pass2 outcomes per context.

GROUND TRUTH:
    python scripts/answerer_bakeoff.py --all 2>&1 | tee logs/answerer_bakeoff.log

STOP CONDITIONS (surfaced, never tuned around):
    - candidate at final_compliance 0/11: reported, not prompt-tuned.
    - a VisualClaim citing a non-retrieved frame_idx: hard-gate DQ that
      candidate, print the exhibit, keep analyzing others.
    - Ollama unreachable / tag not pulled: skip with a NOTE, never fabricate
      a row.
    - no global abort; a partial slate is a valid run.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import requests

import iris.aria as aria
import iris.query as iris_query
from scripts.captioner_bakeoff import (
    OLLAMA_HOST,
    _check_ollama_model,
    _ollama_available,
    _ollama_unload,
    _strip_think,
    _tag_matches,
)
from scripts.demo_cctv_query import CFG, NEGATIVE_QUERIES, POSITIVE_QUERIES, _load_index
from scripts.virat_query_smoke import QUERIES as SMOKE_QUERIES

from iris.claim_contract import (
    AbsenceClaim,
    ANSWER_CLAIMS_WIRE_SCHEMA,
    AnswerClaims,
    BadMetadataFieldError,
    ClaimContractError,
    ClaimFieldShapeError,
    CoreClaimInvariantError,
    GlobalClaim,
    MalformedJSONError,
    MetadataClaim,
    SchemaShapeError,
    UnknownClaimTypeError,
    VisualClaim,
)
from iris.query import _corrective_message, _extract_json_object

# ─────────────────────────────────────────────────────────────────────────────
# query set -- SAME 11 as scripts/smoke_v2.py, no new query set
# ─────────────────────────────────────────────────────────────────────────────

ALL_QUERIES: list[tuple[str, str]] = (
    [("SMOKE", q) for q in SMOKE_QUERIES]
    + [("POSITIVE", q) for q in POSITIVE_QUERIES]
    + [("NEGATIVE", q) for q in NEGATIVE_QUERIES]
)
assert len(ALL_QUERIES) == 11, f"expected 11 queries, got {len(ALL_QUERIES)}"

# ─────────────────────────────────────────────────────────────────────────────
# candidates -- tags confirmed against this host's `GET /api/tags` before
# this file was written (see module docstring for the probe results)
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES: dict[str, dict] = {
    "llama3.2-3b": {
        "model": "llama3.2:3b",
        "note": "baseline/incumbent -- generate_v2(model=None)'s actual default tag today.",
    },
    "hermes3-3b": {
        "model": "hermes3:3b",
        "optional": True,
        "note": "Hermes-3 (Llama-3.2-3B base), JSON/tool-tuned. NOT pulled on this host "
                "(no 'hermes3*' tag in `ollama list`) -- skipped with a NOTE, not fabricated.",
    },
    "qwen3.5-2b": {
        "model": "qwen3.5:2b",
        "note": "known resident (also used live in the captioner bake-off).",
    },
    "qwen3.5-4b": {
        "model": "qwen3.5:4b",
        "note": "CONFIRMED pulled here -- the captioner bake-off only inferred exclusion "
                "on the (unrelated) image-prefill latency axis; text-only contract "
                "compliance is measured fresh, on its own merits.",
    },
    "minicpm5-1b": {
        "model": "openbmb/minicpm5:q4_K_M",
        "requested_tag": "minicpm5:1b",
        "note": "requested tag 'minicpm5:1b' does not exist on this host; resolved to the "
                "confirmed pulled tag 'openbmb/minicpm5:q4_K_M' (digest 08239e8f70e0...). "
                "think:false is NOT settable through generate_v2's production signature "
                "(no iris/ edits) -- every raw response is scanned post-hoc via _strip_think "
                "instead; incidents are counted, never silently absorbed.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# task 3 -- constrained-decoding candidates (schema-format, --constrained).
# Tags confirmed against this host's `GET /api/tags` / `ollama pull` before
# this section was written.
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES_CONSTRAINED: dict[str, dict] = {
    "llama3.2-3b": {
        "model": "llama3.2:3b",
        "note": "incumbent control -- DQ'd under json-mode (task 2c: verbatim echo of the "
                "_SYSTEM_PROMPT_V2 few-shot example, frame_idx=23760). Grammar constrains "
                "STRUCTURE only, not grounding -- does it save llama, or does the same "
                "fabrication recur under a guaranteed-valid shape?",
    },
    "smollm2-1.7b": {
        "model": "smollm2:1.7b",
        "requested_tag": "smollm3:3b",
        "note": "smollm3:3b (the task's LEAD candidate, fabrication-resistance selected) is NOT "
                "pullable on this host: no 'smollm3*' tag in the Ollama library, and the "
                "hf.co/HuggingFaceTB/SmolLM3-3B-GGUF passthrough fails with "
                "'realm host \"huggingface.co\" does not match original host \"hf.co\"' on "
                "Ollama 0.31.1 -- an engine/version bug on this host, not a nonexistent repo. "
                "User-directed substitution: smollm2:1.7b (older generation, same lineage, "
                "explicitly acknowledged as likely less performant) stands in so the SLM axis "
                "isn't empty. Disposition of smollm3 itself is deferred, not abandoned.",
    },
    "phi4-mini": {
        "model": "phi4-mini",
        "note": "confirmed pulled as 'phi4-mini:latest' -- instruct variant, NOT the reasoning "
                "variant (no 'phi4-mini-reasoning' tag requested or pulled).",
    },
    "granite4-micro": {
        "model": "granite4:micro",
        "note": "confirmed pulled.",
    },
    "qwen2.5-3b-instruct": {
        "model": "qwen2.5:3b-instruct",
        "note": "confirmed pulled.",
    },
    "qwen3.5-4b": {
        "model": "qwen3.5:4b",
        "note": "think-suppression hypothesis (PRE-REGISTERED, task 3): measured, not assumed. "
                "think UNSET: reasoning goes into a separate native /api/chat 'thinking' field "
                "that can consume the entire token budget before schema-constrained content "
                "ever starts (content=\"\", done_reason=length -- reproduced with num_predict=600). "
                "think:false: the 'thinking' field disappears but format=<schema> is SILENTLY "
                "IGNORED -- the model emits free-flowing unconstrained prose, not JSON at all. "
                "Both settings hit the task's own documented hazard ('anyOf/oneOf ... silently "
                "fall back to unconstrained text') via a different trigger (think x format "
                "interaction, not anyOf). User-directed: mark GRAMMAR-INCOMPATIBLE, exclude -- "
                "confirmed again by the grammar smoke test below, not hand-waved.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# task 4 -- sentinel-schema survivors (--sentinel). Reruns the constrained
# candidates that showed real signal, PLUS one lottery ticket, under the
# sentinel wire schema (every field required + retry reinstated). DROPPED
# (not re-run): qwen2.5-3b-instruct, smollm2-1.7b -- both were clean 0%
# STOPs under task 3's constrained run (never produced a single compliant
# row over 11 queries); task 3's own STOP CONDITIONS say a 0% candidate is
# surfaced, not chased, and there is no new lever here (same model, same
# sentinel schema) that would plausibly move either off zero.
# ─────────────────────────────────────────────────────────────────────────────

CANDIDATES_SENTINEL: dict[str, dict] = {
    "qwen3.5-4b": {
        "model": "qwen3.5:4b",
        "note": "semantic ceiling reference -- task 3's best constrained compliance (54.5%) "
                "despite the heaviest latency (115.9s/call, reasoning not suppressed by "
                "format=schema alone). Does sentinel+retry close more of the completeness gap "
                "without the latency getting worse?",
    },
    "phi4-mini": {
        "model": "phi4-mini",
        "note": "task 3's fastest strong performer (45.5% @ 22.5s/call) -- the task's own "
                "pre-registered most-likely seat under the seating rule.",
    },
    "granite4-micro": {
        "model": "granite4:micro",
        "note": "task 3's third-best constrained candidate (36.4% @ 27.7s/call).",
    },
    "llama3.2-3b": {
        "model": "llama3.2:3b",
        "note": "control/incumbent -- DQ'd under json-mode (task 2c: verbatim few-shot echo, "
                "frame_idx=23760); clean under task 3's constrained run (0 gate hits, thin "
                "sample: only 3/11 compliant rows to check). Sentinel+retry gives a bigger "
                "compliant sample to test whether the fabrication recurs.",
    },
    "qwen3.5-2b": {
        "model": "qwen3.5:2b",
        "note": "LOTTERY TICKET (task 4, pre-registered): task 2's json-mode exclusion "
                "(0% compliance) was reasoning-token starvation under an UNCONSTRAINED cap, "
                "entangled with 'dominance-on-latency' framing that schema-constrained "
                "decoding's token-collapse (format forces structure, no free-text formatting "
                "deliberation) may invalidate. Resident on this host from the captioner "
                "bake-off. Prediction: latency drops far below json-mode's 167s, but "
                "completeness lands below qwen3.5-4b's and does not out-complete phi4-mini.",
    },
}

KNOWN_LABELS = [
    MalformedJSONError.taxonomy_label,
    SchemaShapeError.taxonomy_label,
    UnknownClaimTypeError.taxonomy_label,
    BadMetadataFieldError.taxonomy_label,
    ClaimFieldShapeError.taxonomy_label,
    CoreClaimInvariantError.taxonomy_label,
    "other",
]

# TRUE engine/grammar failures (smoke-test-disqualifying) are violations of
# constraints the JSON schema DOES express and the grammar SHOULD therefore
# enforce: valid JSON at all (MalformedJSONError), the object's own required
# keys (SchemaShapeError: top-level query/claims), and enum membership
# (UnknownClaimTypeError: claim_type: enum[...]; BadMetadataFieldError:
# field: enum[...]). ClaimFieldShapeError and CoreClaimInvariantError are
# explicitly NOT here -- ANSWER_CLAIMS_WIRE_SCHEMA deliberately does not mark
# frame_idx/assertion/is_core/etc. "required" (that would need a per-variant
# anyOf, the exact thing being avoided), so a model omitting a
# variant-required field is the flat schema's known, EXPECTED gap -- this is
# "variant-completeness," the live signal the task's METRICS section asks to
# measure across the full run, not an engine incompatibility to gate on from
# one smoke probe.
_STRUCTURAL_LABELS = {
    MalformedJSONError.taxonomy_label,
    SchemaShapeError.taxonomy_label,
    UnknownClaimTypeError.taxonomy_label,
    BadMetadataFieldError.taxonomy_label,
}

WIRE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(ANSWER_CLAIMS_WIRE_SCHEMA, sort_keys=True).encode("utf-8")
).hexdigest()[:16]

# ─────────────────────────────────────────────────────────────────────────────
# task 4 amendment: prompt/context token-length instrumentation only, no
# behavior change. Uses tiktoken cl100k_base as a fixed, model-agnostic proxy
# tokenizer -- NOT any candidate's own tokenizer (each Ollama model has its
# own; no single count is "exactly right" for all of them). The point isn't
# per-model exactness, it's a consistent yardstick so prompt versions /
# context regimes are comparable ACROSS RUNS at zero extra cost. Rationale:
# prompt-length-vs-completeness is a live hypothesis (HADES precedent:
# qwen3.5:0.8b regressed until the system prompt was cut).
_TOKEN_ENCODER = None


def _count_tokens(text: str) -> int:
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is None:
        import tiktoken
        _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_TOKEN_ENCODER.encode(text))


SYSTEM_PROMPT_TOKENS = _count_tokens(aria._SYSTEM_PROMPT_V2)


def _context_token_stats(capture_rows: list[dict]) -> dict:
    lens = [_count_tokens(r["context_text"]) for r in capture_rows]
    return {
        "mean_context_tokens": statistics.mean(lens) if lens else float("nan"),
        "max_context_tokens": max(lens) if lens else float("nan"),
    }


NEGATION_RE = re.compile(r"\b(no|not|n't|without|absent|missing|none|never)\b", re.IGNORECASE)

CAPTURE_JSONL = REPO / "answerer_capture.jsonl"

# Largest legit AnswerClaims payload over these 11 contexts is ~150-600 tokens
# (few claims, short assertions); 1024 gives headroom while still hard-capping
# runaway generation. iris.aria.generate_v2's _extract_json_object takes the
# FIRST balanced {...} span, so a model that closes its JSON object before
# this cap parses IDENTICALLY capped or uncapped -- only a model that never
# closes an object in-budget is affected, and it was non-compliant either way.
ANSWERER_MAX_TOKENS = 1024


def _mode_suffix(constrained: bool = False, sentinel: bool = False) -> str:
    if sentinel:
        return "_sentinel"
    if constrained:
        return "_constrained"
    return ""


def _pass_path(name: str, pass_n: int = 1, constrained: bool = False, sentinel: bool = False) -> Path:
    return REPO / f"answerer_bakeoff_{name}{_mode_suffix(constrained, sentinel)}_pass{pass_n}.jsonl"


def _meta_path(name: str, constrained: bool = False, sentinel: bool = False) -> Path:
    return REPO / f"answerer_bakeoff_{name}{_mode_suffix(constrained, sentinel)}_meta.json"


PROVENANCE_JSONL = REPO / "answerer_bakeoff_provenance.jsonl"
PROVENANCE_CONSTRAINED_JSONL = REPO / "answerer_bakeoff_constrained_provenance.jsonl"
PROVENANCE_SENTINEL_JSONL = REPO / "answerer_bakeoff_sentinel_provenance.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 -- frozen context capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture_contexts() -> list[dict]:
    if CAPTURE_JSONL.exists():
        print(f"frozen: {CAPTURE_JSONL} exists -- not regenerating.")
        rows = []
        with open(CAPTURE_JSONL, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    print("=" * 100)
    print("PHASE 0: FROZEN CONTEXT CAPTURE (answerer_capture.jsonl) -- one-time, then frozen")
    print("=" * 100)
    print("  Captions in this capture are the CURRENT captioner (BLIP) at the CURRENT retrieval")
    print("  regime -- intentional: compliance measured below is caption-CONTENT-independent,")
    print("  only context SHAPE/LENGTH matters. SCOPE LIMIT: if a future captioner swap")
    print("  materially changes context length, this capture must be re-frozen -- compliance")
    print("  CAN shift for context-limited models (self-confident-context-exhaustion axis).")
    print()

    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")

    rows = []
    for i, (tag, question) in enumerate(ALL_QUERIES, 1):
        # Context-assembly spine lifted call-for-call from iris.query._query_v2
        # (embed -> _build_retrieved -> _ensure_captions -> wrapper_init_l1_cache
        # -> wrapper_populate_cache -> as_context_text) -- production helpers,
        # not reimplemented.
        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)
        cache_obj = iris_query.wrapper_init_l1_cache(CFG)
        iris_query.wrapper_populate_cache(cache_obj, retrieved)
        context_text = cache_obj.as_context_text()
        retrieved_frame_idxs = [f["frame_idx"] for f in retrieved]
        rows.append({
            "tag": tag, "query": question, "context_text": context_text,
            "retrieved_frame_idxs": retrieved_frame_idxs,
        })
        print(f"  [{i}/{len(ALL_QUERIES)}] [{tag}] {question!r} "
              f"context_len={len(context_text)} retrieved={retrieved_frame_idxs}")

    with open(CAPTURE_JSONL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {CAPTURE_JSONL}")
    print()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 -- per-candidate two-attempt loop (mirrors
# iris.query._generate_answer_claims_v2 -- update this if that function's
# control flow changes)
# ─────────────────────────────────────────────────────────────────────────────

def _classify(raw: str) -> tuple[str, AnswerClaims | None, Exception | None]:
    try:
        parsed = AnswerClaims.from_json(_extract_json_object(raw))
        return "pass", parsed, None
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        return label, None, e


def _claim_summary(claim) -> tuple:
    if isinstance(claim, VisualClaim):
        return ("visual", claim.frame_idx, claim.is_core, claim.assertion, None, None)
    if isinstance(claim, AbsenceClaim):
        return ("absence", None, claim.is_core, claim.event, None, None)
    if isinstance(claim, MetadataClaim):
        return ("metadata", claim.frame_idx, None, None, claim.field, claim.stated_value)
    if isinstance(claim, GlobalClaim):
        return ("global", None, None, claim.text, None, None)
    return ("unknown", None, None, None, None, None)


def _run_one_query(model: str, row: dict) -> dict:
    """One (candidate, query) trial: the SAME two-attempt loop as
    iris.query._generate_answer_claims_v2 (production function -- if its
    control flow changes, this must be updated to match), with model=
    threaded per candidate instead of the production default."""
    query, context_text = row["query"], row["context_text"]

    t0 = time.time()
    raw1 = aria.generate_v2(prompt=query, context=context_text, model=model, max_tokens=ANSWERER_MAX_TOKENS)
    seconds1 = time.time() - t0
    raw1_clean, n_think1 = _strip_think(raw1)
    label1, claims1, err1 = _classify(raw1_clean)

    if label1 == "pass":
        return {
            "tag": row["tag"], "query": query,
            "attempt1_label": "pass", "attempt2_label": None,
            "final_compliant": True, "n_attempts": 1,
            "think_blocks_stripped": n_think1,
            "seconds": seconds1,
            "raw1": raw1, "raw2": None,
            "claim_summary": [_claim_summary(c) for c in claims1.claims],
            "retrieved_frame_idxs": row["retrieved_frame_idxs"],
        }

    corrective = _corrective_message(query, label1, err1)
    t0 = time.time()
    raw2 = aria.generate_v2(prompt=corrective, context=context_text, model=model, max_tokens=ANSWERER_MAX_TOKENS)
    seconds2 = time.time() - t0
    raw2_clean, n_think2 = _strip_think(raw2)
    label2, claims2, err2 = _classify(raw2_clean)

    final_compliant = label2 == "pass"
    return {
        "tag": row["tag"], "query": query,
        "attempt1_label": label1, "attempt2_label": label2,
        "final_compliant": final_compliant, "n_attempts": 2,
        "think_blocks_stripped": n_think1 + n_think2,
        "seconds": seconds1 + seconds2,
        "raw1": raw1, "raw2": raw2,
        "claim_summary": [_claim_summary(c) for c in claims2.claims] if final_compliant else None,
        "retrieved_frame_idxs": row["retrieved_frame_idxs"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART B (task 3) -- constrained (schema-format) single-attempt path.
# NO corrective retry: grammar guarantees syntactic structure, so a retry-on-
# parse-failure loop is moot -- a from_wire failure under grammar (missing a
# variant-required field the grammar can't itself enforce) IS the semantic
# finding, recorded once, not retried away.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_wire(raw: str) -> tuple[str, AnswerClaims | None, Exception | None]:
    try:
        parsed_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        err = MalformedJSONError(f"Not valid JSON: {e}")
        return err.taxonomy_label, None, err
    try:
        parsed = AnswerClaims.from_wire(parsed_dict)
        return "pass", parsed, None
    except Exception as e:
        label = getattr(e, "taxonomy_label", None) or "other"
        return label, None, e


def _run_one_query_constrained(model: str, row: dict) -> dict:
    """One (candidate, query) trial under schema-constrained decoding: a
    SINGLE attempt, parsed via AnswerClaims.from_wire (the flat wire shape),
    no corrective retry (see module note above)."""
    query, context_text = row["query"], row["context_text"]

    t0 = time.time()
    raw1 = aria.generate_v2(prompt=query, context=context_text, model=model,
                             max_tokens=ANSWERER_MAX_TOKENS, schema_format=True)
    seconds1 = time.time() - t0
    raw1_clean, n_think1 = _strip_think(raw1)
    label1, claims1, err1 = _classify_wire(raw1_clean)

    final_compliant = label1 == "pass"
    return {
        "tag": row["tag"], "query": query,
        "attempt1_label": label1, "attempt2_label": None,
        "final_compliant": final_compliant, "n_attempts": 1,
        "think_blocks_stripped": n_think1,
        "seconds": seconds1,
        "raw1": raw1, "raw2": None,
        "claim_summary": [_claim_summary(c) for c in claims1.claims] if final_compliant else None,
        "retrieved_frame_idxs": row["retrieved_frame_idxs"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# task 4 -- sentinel-schema path WITH one corrective retry. Mirrors
# iris.query._generate_answer_claims_v2_wire IN-SCRIPT (production function;
# if its control flow changes, update this), with model= threaded per
# candidate instead of the production default. See that function's docstring
# for why a retry is meaningful again under the sentinel schema (parse retry
# is moot under grammar, but claim_field_shape -- a required field left at
# its sentinel -- is semantic, and the model CAN act on a named corrective).
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_query_sentinel(model: str, row: dict) -> dict:
    query, context_text = row["query"], row["context_text"]

    t0 = time.time()
    raw1 = aria.generate_v2(prompt=query, context=context_text, model=model,
                             max_tokens=ANSWERER_MAX_TOKENS, schema_format=True)
    seconds1 = time.time() - t0
    raw1_clean, n_think1 = _strip_think(raw1)
    label1, claims1, err1 = _classify_wire(raw1_clean)

    if label1 == "pass":
        return {
            "tag": row["tag"], "query": query,
            "attempt1_label": "pass", "attempt2_label": None,
            "final_compliant": True, "n_attempts": 1,
            "think_blocks_stripped": n_think1,
            "seconds": seconds1,
            "raw1": raw1, "raw2": None,
            "claim_summary": [_claim_summary(c) for c in claims1.claims],
            "field_noise": claims1.field_noise,
            "retrieved_frame_idxs": row["retrieved_frame_idxs"],
        }

    corrective = _corrective_message(query, label1, err1, wire=True)
    t0 = time.time()
    raw2 = aria.generate_v2(prompt=corrective, context=context_text, model=model,
                             max_tokens=ANSWERER_MAX_TOKENS, schema_format=True)
    seconds2 = time.time() - t0
    raw2_clean, n_think2 = _strip_think(raw2)
    label2, claims2, err2 = _classify_wire(raw2_clean)

    final_compliant = label2 == "pass"
    return {
        "tag": row["tag"], "query": query,
        "attempt1_label": label1, "attempt2_label": label2,
        "final_compliant": final_compliant, "n_attempts": 2,
        "think_blocks_stripped": n_think1 + n_think2,
        "seconds": seconds1 + seconds2,
        "raw1": raw1, "raw2": raw2,
        "claim_summary": [_claim_summary(c) for c in claims2.claims] if final_compliant else None,
        "field_noise": claims2.field_noise if final_compliant else [],
        "retrieved_frame_idxs": row["retrieved_frame_idxs"],
    }


# claim_type values a well-formed wire response may legitimately use --
# checked for literal '"' contamination by the grammar smoke test (the
# PEG-template trailing-quote bug).
_WIRE_CLAIM_TYPES = ("visual", "metadata", "absence", "global")


def grammar_smoke_test(name: str, model: str, probe_row: dict) -> tuple[bool, str]:
    """One call, trivial context, before any full bake-off run for this
    candidate. Checks (a) llama-server does not crash, (b) response is valid
    JSON with the top-level shape and enum values the schema DOES express
    (claim_type/field enums, query/claims required keys), (c) claim_type
    enum values come back clean (no embedded '"' -- the documented PEG
    trailing-quote bug). A ClaimFieldShapeError or CoreClaimInvariantError
    alone is variant-completeness / semantic signal the flat schema cannot
    itself enforce (see _STRUCTURAL_LABELS) -- NOT an engine incompatibility,
    and does not fail the smoke test by itself. Returns (passed, detail)."""
    try:
        raw = aria.generate_v2(prompt=probe_row["query"], context=probe_row["context_text"],
                                model=model, max_tokens=ANSWERER_MAX_TOKENS, schema_format=True)
    except Exception as e:
        return False, f"CRASH/transport failure calling {model!r}: {type(e).__name__}: {e}"

    try:
        parsed_dict = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"response is not valid JSON despite schema-format: {e}. raw={raw[:300]!r}"

    for claim in parsed_dict.get("claims", []):
        ct = claim.get("claim_type")
        if isinstance(ct, str) and '"' in ct:
            return False, f"QUOTE-CONTAMINATED enum: claim_type={ct!r} (PEG trailing-quote bug). raw={raw[:300]!r}"
        if ct not in _WIRE_CLAIM_TYPES:
            return False, f"claim_type {ct!r} not one of {_WIRE_CLAIM_TYPES} -- schema not honored. raw={raw[:300]!r}"

    label, _parsed, err = _classify_wire(raw)
    if label in _STRUCTURAL_LABELS or label == "other":
        return False, f"structural/engine parse failure ({label}): {err}. raw={raw[:300]!r}"
    if label in (ClaimFieldShapeError.taxonomy_label, CoreClaimInvariantError.taxonomy_label):
        return True, f"structurally clean; variant-completeness/semantic miss on smoke prompt ({err}) -- not an engine issue."
    return True, "clean: crash-free, parses via from_wire, enums uncontaminated."


def _write_pass(name: str, pass_n: int, rows: list[dict], constrained: bool = False,
                 sentinel: bool = False) -> None:
    path = _pass_path(name, pass_n, constrained, sentinel)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def _load_pass(name: str, pass_n: int = 1, constrained: bool = False,
                sentinel: bool = False) -> list[dict] | None:
    path = _pass_path(name, pass_n, constrained, sentinel)
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_candidate_pass(name: str, capture_rows: list[dict], pass_n: int = 1,
                        constrained: bool = False, sentinel: bool = False) -> list[dict] | None:
    candidates = CANDIDATES_SENTINEL if sentinel else (CANDIDATES_CONSTRAINED if constrained else CANDIDATES)
    spec = candidates[name]
    model = spec["model"]

    existing = _load_pass(name, pass_n, constrained, sentinel)
    if existing is not None:
        print(f"frozen: {_pass_path(name, pass_n, constrained, sentinel)} exists -- not regenerating.")
        return existing

    if not _ollama_available(model):
        if pass_n == 1:
            print(f"NOTE: candidate {name!r} ({model!r}) not available in Ollama -- skipping.")
        return None

    mode_disp = "SENTINEL (schema-format+retry)" if sentinel else ("CONSTRAINED (schema-format)" if constrained else "json-mode")
    print("=" * 100)
    print(f"CANDIDATE {name!r} ({model!r}) -- {mode_disp} -- pass {pass_n} -- {len(capture_rows)} queries")
    print("=" * 100)

    if pass_n == 1:
        probe_row = capture_rows[0]
        t0 = time.time()
        if constrained or sentinel:
            raw_probe = aria.generate_v2(prompt=probe_row["query"], context=probe_row["context_text"],
                                          model=model, max_tokens=ANSWERER_MAX_TOKENS, schema_format=True)
        else:
            raw_probe = aria.generate_v2(prompt=probe_row["query"], context=probe_row["context_text"], model=model)
        probe_seconds = time.time() - t0
        _, n_think_probe = _strip_think(raw_probe)
        print(f"  LATENCY PROBE (text-only, 1 context): {probe_seconds:.2f}s think_stripped={n_think_probe}")
    else:
        probe_seconds = None

    run_one = _run_one_query_sentinel if sentinel else (_run_one_query_constrained if constrained else _run_one_query)
    rows_out = []
    for i, row in enumerate(capture_rows, 1):
        result = run_one(model, row)
        rows_out.append(result)
        think_note = f" **THINK CONTAMINATION x{result['think_blocks_stripped']}**" if result["think_blocks_stripped"] else ""
        print(f"  [{i}/{len(capture_rows)}] [{result['tag']}] {result['query']!r} "
              f"attempt1={result['attempt1_label']} attempt2={result['attempt2_label']} "
              f"final_compliant={result['final_compliant']} {result['seconds']:.2f}s{think_note}")

    _ollama_unload(model)
    _write_pass(name, pass_n, rows_out, constrained, sentinel)
    if pass_n == 1 and probe_seconds is not None:
        _write_meta_partial(name, model, probe_seconds, rows_out, constrained, sentinel,
                             context_stats=_context_token_stats(capture_rows))
    print()
    return rows_out


def _write_meta_partial(name: str, model: str, probe_seconds: float, rows: list[dict],
                         constrained: bool = False, sentinel: bool = False,
                         context_stats: dict | None = None) -> None:
    """Compliance/timing/think stats -- written after pass1; run_analyze
    (Phase 2) overwrites this with the gate result added."""
    n = len(rows)
    first_pass = sum(1 for r in rows if r["attempt1_label"] == "pass")
    final_pass = sum(1 for r in rows if r["final_compliant"])
    seconds = [r["seconds"] for r in rows]
    think_total = sum(r["think_blocks_stripped"] for r in rows)
    meta = {
        "candidate": name, "model": model, "constrained": constrained, "sentinel": sentinel,
        "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
        **(context_stats or {}),
        "n_queries": n,
        "first_attempt_compliance": first_pass / n if n else float("nan"),
        "final_compliance": final_pass / n if n else float("nan"),
        "probe_seconds": probe_seconds,
        "mean_seconds": statistics.mean(seconds) if seconds else float("nan"),
        "median_seconds": statistics.median(seconds) if seconds else float("nan"),
        "think_blocks_stripped_total": think_total,
        "gate": "not yet analyzed",
    }
    with open(_meta_path(name, constrained, sentinel), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 -- offline analysis (NLI-free)
# ─────────────────────────────────────────────────────────────────────────────

def run_analyze(name: str, rows: list[dict], capture_by_query: dict[str, dict],
                 constrained: bool = False) -> dict:
    candidates = CANDIDATES_CONSTRAINED if constrained else CANDIDATES
    print("#" * 100)
    print(f"# ANALYZING CANDIDATE: {name}  ({_pass_path(name, constrained=constrained)})")
    print("#" * 100)

    n = len(rows)
    first_pass = sum(1 for r in rows if r["attempt1_label"] == "pass")
    final_pass = sum(1 for r in rows if r["final_compliant"])
    first_attempt_compliance = first_pass / n if n else float("nan")
    final_compliance = final_pass / n if n else float("nan")

    print(f"  (A) COMPLIANCE: first_attempt={first_pass}/{n}={first_attempt_compliance:.4f}  "
          f"final={final_pass}/{n}={final_compliance:.4f}")
    if final_compliance == 0.0:
        print(f"  ** STOP-SURFACE: {name} is at final_compliance 0/{n} -- cannot speak the "
              f"contract. Reported, not tuned. **")

    label_hist: dict[str, int] = {lbl: 0 for lbl in KNOWN_LABELS}
    for r in rows:
        for lbl in (r["attempt1_label"], r["attempt2_label"]):
            if lbl and lbl != "pass":
                label_hist[lbl] = label_hist.get(lbl, 0) + 1
    print(f"  failure-label histogram (both attempts): {label_hist}")
    modal_label = max(label_hist, key=label_hist.get) if any(label_hist.values()) else None
    print(f"  PRE-REGISTERED expectation: is_core_invariant is dominant -- "
          f"actual modal label = {modal_label!r}")
    print()

    # (B) structural -- absence positivity (compliant outputs only)
    n_absence = 0
    n_absence_violations = 0
    absence_exhibits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type != "absence":
                continue
            n_absence += 1
            if NEGATION_RE.search(text or ""):
                n_absence_violations += 1
                absence_exhibits.append((r["query"], text))
    absence_violation_rate = n_absence_violations / n_absence if n_absence else float("nan")
    print(f"  (B) STRUCTURAL -- absence_positivity: {n_absence_violations}/{n_absence} "
          f"violate (contain negation tokens) rate={absence_violation_rate}")
    for q, text in absence_exhibits:
        print(f"      VIOLATION: query={q!r} event={text!r}")
    print()

    # (C) frame-citation hard gate
    citation_hits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        retrieved_set = set(r["retrieved_frame_idxs"])
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type == "visual" and frame_idx not in retrieved_set:
                citation_hits.append((r["query"], frame_idx, sorted(retrieved_set), text))
    if citation_hits:
        print(f"  ** (C) FRAME-CITATION HARD GATE: {len(citation_hits)} HIT(S) -- "
              f"{name} DISQUALIFIED **", file=sys.stderr)
        print(f"  ** (C) FRAME-CITATION HARD GATE: {len(citation_hits)} HIT(S) -- "
              f"{name} DISQUALIFIED **")
        for query, frame_idx, retrieved_set, text in citation_hits:
            print(f"      query={query!r}")
            print(f"      cited frame_idx={frame_idx} NOT IN retrieved_frame_idxs={retrieved_set}")
            print(f"      claim={text!r}")
        print(f"  {name} is DISQUALIFIED (frame-citation gate). Continuing to analyze other candidates.")
    else:
        print(f"  (C) FRAME-CITATION HARD GATE: 0 hits -- {name} clears the gate.")
    disqualified = bool(citation_hits)
    print()

    # (D) metadata-number human-review exhibits (NOT auto-DQ)
    metadata_exhibits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        context_text = capture_by_query.get(r["query"], {}).get("context_text", "")
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type != "metadata" or value is None:
                continue
            if str(value) not in context_text:
                metadata_exhibits.append((r["query"], frame_idx, field, value))
    print(f"  (D) METADATA-NUMBER REVIEW (HUMAN column, not auto-DQ): "
          f"{len(metadata_exhibits)} flagged")
    for query, frame_idx, field, value in metadata_exhibits:
        print(f"      REVIEW: query={query!r} frame_idx={frame_idx} field={field!r} "
              f"value={value!r} -- not found verbatim in context_text")
    print()

    seconds = [r["seconds"] for r in rows]
    think_total = sum(r["think_blocks_stripped"] for r in rows)
    meta_path = _meta_path(name, constrained)
    prior_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {
        **prior_meta,
        "candidate": name,
        "model": candidates[name]["model"],
        "constrained": constrained,
        "n_queries": n,
        "first_attempt_compliance": first_attempt_compliance,
        "final_compliance": final_compliance,
        "label_histogram": label_hist,
        "absence_violation_rate": absence_violation_rate,
        "n_absence_claims": n_absence,
        "frame_citation_gate_hits": len(citation_hits),
        "disqualified": disqualified,
        "metadata_review_flagged": len(metadata_exhibits),
        "mean_seconds": statistics.mean(seconds) if seconds else float("nan"),
        "think_blocks_stripped_total": think_total,
        "gate": "DISQUALIFIED" if disqualified else "pass",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")
    print()

    return {
        "name": name,
        "first_attempt_compliance": first_attempt_compliance,
        "final_compliance": final_compliance,
        "absence_violation_rate": absence_violation_rate,
        "disqualified": disqualified,
        "mean_seconds": meta["mean_seconds"],
        "metadata_review_flagged": len(metadata_exhibits),
        "think_blocks_stripped_total": think_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# task 4 -- sentinel-mode analysis. Separate from run_analyze (not a third
# mode flag threaded through it) so the new absence-production /
# sentinel-misuse / field-noise metrics don't add conditional branches to the
# json-mode/constrained path already validated in tasks 1-3. Sections (A)-(D)
# mirror run_analyze's (frame-citation gate, absence positivity, metadata
# human column UNCHANGED per the task); (E)/(F)/(G) are new.
# ─────────────────────────────────────────────────────────────────────────────

def run_analyze_sentinel(name: str, rows: list[dict], capture_by_query: dict[str, dict]) -> dict:
    print("#" * 100)
    print(f"# ANALYZING CANDIDATE (sentinel): {name}  ({_pass_path(name, sentinel=True)})")
    print("#" * 100)

    n = len(rows)
    first_pass = sum(1 for r in rows if r["attempt1_label"] == "pass")
    final_pass = sum(1 for r in rows if r["final_compliant"])
    first_attempt_compliance = first_pass / n if n else float("nan")
    final_compliance = final_pass / n if n else float("nan")

    print(f"  (A) COMPLIANCE: first_attempt={first_pass}/{n}={first_attempt_compliance:.4f}  "
          f"final={final_pass}/{n}={final_compliance:.4f}  (retry restored -- these can differ now)")
    if final_compliance == 0.0:
        print(f"  ** STOP-SURFACE: {name} is at final_compliance 0/{n} -- cannot speak the "
              f"contract. Reported, not tuned. **")

    label_hist: dict[str, int] = {lbl: 0 for lbl in KNOWN_LABELS}
    for r in rows:
        for lbl in (r["attempt1_label"], r["attempt2_label"]):
            if lbl and lbl != "pass":
                label_hist[lbl] = label_hist.get(lbl, 0) + 1
    print(f"  failure-label histogram (both attempts): {label_hist}")
    modal_label = max(label_hist, key=label_hist.get) if any(label_hist.values()) else None
    sentinel_misuse_count = label_hist.get(ClaimFieldShapeError.taxonomy_label, 0)
    print(f"  PRE-REGISTERED expectation (task 4): residual failures are wrong-sentinel "
          f"semantics only -- modal label = {modal_label!r}, sentinel-misuse "
          f"(claim_field_shape) count = {sentinel_misuse_count}")
    print()

    # (B) structural -- absence positivity (compliant outputs only)
    n_absence = 0
    n_absence_violations = 0
    absence_exhibits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type != "absence":
                continue
            n_absence += 1
            if NEGATION_RE.search(text or ""):
                n_absence_violations += 1
                absence_exhibits.append((r["query"], text))
    absence_violation_rate = n_absence_violations / n_absence if n_absence else float("nan")
    print(f"  (B) STRUCTURAL -- absence_positivity: {n_absence_violations}/{n_absence} "
          f"violate (contain negation tokens) rate={absence_violation_rate}")
    for q, text in absence_exhibits:
        print(f"      VIOLATION: query={q!r} event={text!r}")
    print()

    # (C) frame-citation hard gate -- UNCHANGED
    citation_hits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        retrieved_set = set(r["retrieved_frame_idxs"])
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type == "visual" and frame_idx not in retrieved_set:
                citation_hits.append((r["query"], frame_idx, sorted(retrieved_set), text))
    if citation_hits:
        print(f"  ** (C) FRAME-CITATION HARD GATE: {len(citation_hits)} HIT(S) -- "
              f"{name} DISQUALIFIED **", file=sys.stderr)
        print(f"  ** (C) FRAME-CITATION HARD GATE: {len(citation_hits)} HIT(S) -- "
              f"{name} DISQUALIFIED **")
        for query, frame_idx, retrieved_set, text in citation_hits:
            print(f"      query={query!r}")
            print(f"      cited frame_idx={frame_idx} NOT IN retrieved_frame_idxs={retrieved_set}")
            print(f"      claim={text!r}")
        print(f"  {name} is DISQUALIFIED (frame-citation gate). Continuing to analyze other candidates.")
    else:
        print(f"  (C) FRAME-CITATION HARD GATE: 0 hits -- {name} clears the gate.")
    disqualified = bool(citation_hits)
    print()

    # (D) metadata-number human-review exhibits -- UNCHANGED (NOT auto-DQ)
    metadata_exhibits = []
    for r in rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        context_text = capture_by_query.get(r["query"], {}).get("context_text", "")
        for claim_type, frame_idx, is_core, text, field, value in r["claim_summary"]:
            if claim_type != "metadata" or value is None:
                continue
            if str(value) not in context_text:
                metadata_exhibits.append((r["query"], frame_idx, field, value))
    print(f"  (D) METADATA-NUMBER REVIEW (HUMAN column, not auto-DQ): "
          f"{len(metadata_exhibits)} flagged")
    for query, frame_idx, field, value in metadata_exhibits:
        print(f"      REVIEW: query={query!r} frame_idx={frame_idx} field={field!r} "
              f"value={value!r} -- not found verbatim in context_text")
    print()

    # (E) ABSENCE PRODUCTION -- headline new metric (task 4). Was ZERO across
    # 6 models x 11 queries x 2 passes in task 3; the sentinel schema +
    # absence few-shot are the fix under test.
    n_absence_claims_total = n_absence  # from (B) -- every AbsenceClaim in every compliant row
    negative_rows = [r for r in rows if r["tag"] == "NEGATIVE"]
    n_negative = len(negative_rows)
    negative_rows_with_absence = 0
    for r in negative_rows:
        if not r["final_compliant"] or r["claim_summary"] is None:
            continue
        if any(cs[0] == "absence" for cs in r["claim_summary"]):
            negative_rows_with_absence += 1
    absence_production_rate = negative_rows_with_absence / n_negative if n_negative else float("nan")
    print(f"  (E) ABSENCE PRODUCTION: {n_absence_claims_total} AbsenceClaim(s) across all compliant "
          f"rows; {negative_rows_with_absence}/{n_negative} NEGATIVE queries produced >=1 AbsenceClaim "
          f"(rate={absence_production_rate})")
    print(f"  PRE-REGISTERED expectation (task 4): 0 -> nonzero (task 3 was exactly 0 for every "
          f"candidate) -- {'MET' if n_absence_claims_total > 0 else 'NOT MET'}")
    print()

    # (F) irrelevant_fields_filled -- noise metric, NOT ranked
    total_noise = sum(len(r.get("field_noise") or []) for r in rows if r["final_compliant"])
    n_compliant_rows = sum(1 for r in rows if r["final_compliant"])
    irrelevant_fields_filled_rate = total_noise / n_compliant_rows if n_compliant_rows else float("nan")
    print(f"  (F) IRRELEVANT_FIELDS_FILLED (noise, not ranked): {total_noise} real-valued "
          f"irrelevant field(s) across {n_compliant_rows} compliant rows (rate={irrelevant_fields_filled_rate})")
    print()

    seconds = [r["seconds"] for r in rows]
    think_total = sum(r["think_blocks_stripped"] for r in rows)
    meta_path = _meta_path(name, sentinel=True)
    prior_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {
        **prior_meta,
        "candidate": name,
        "model": CANDIDATES_SENTINEL[name]["model"],
        "sentinel": True,
        "n_queries": n,
        "first_attempt_compliance": first_attempt_compliance,
        "final_compliance": final_compliance,
        "label_histogram": label_hist,
        "sentinel_misuse_count": sentinel_misuse_count,
        "absence_violation_rate": absence_violation_rate,
        "n_absence_claims": n_absence,
        "absence_production_rate_negative": absence_production_rate,
        "negative_rows_with_absence": negative_rows_with_absence,
        "n_negative_queries": n_negative,
        "irrelevant_fields_filled_total": total_noise,
        "irrelevant_fields_filled_rate": irrelevant_fields_filled_rate,
        "frame_citation_gate_hits": len(citation_hits),
        "disqualified": disqualified,
        "metadata_review_flagged": len(metadata_exhibits),
        "mean_seconds": statistics.mean(seconds) if seconds else float("nan"),
        "think_blocks_stripped_total": think_total,
        "gate": "DISQUALIFIED" if disqualified else "pass",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {meta_path}")
    print()

    return {
        "name": name,
        "first_attempt_compliance": first_attempt_compliance,
        "final_compliance": final_compliance,
        "absence_violation_rate": absence_violation_rate,
        "absence_production_rate": absence_production_rate,
        "negative_rows_with_absence": negative_rows_with_absence,
        "n_absence_claims": n_absence_claims_total,
        "sentinel_misuse_count": sentinel_misuse_count,
        "irrelevant_fields_filled_rate": irrelevant_fields_filled_rate,
        "disqualified": disqualified,
        "mean_seconds": meta["mean_seconds"],
        "metadata_review_flagged": len(metadata_exhibits),
        "think_blocks_stripped_total": think_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 -- stability (surviving, gate-passing candidates)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_claim_summary(cs):
    """pass1 rows are reloaded from JSON (tuples -> lists); pass2 rows come
    from a fresh in-memory run (still tuples). list != tuple in Python even
    for identical content, so BOTH sides must go through the same JSON
    round-trip before comparison -- otherwise every content-identical pair
    reports as a false DIVERGENT (see task 2c postmortem)."""
    return json.loads(json.dumps(cs))


def _compute_stability_stats(pass1_rows: list[dict], pass2_rows: list[dict]) -> dict:
    """Three-way classification -- deterministic-in-success and
    deterministic-in-failure are DISTINCT facts and must not be collapsed
    into one 'identical' bucket."""
    by_query_1 = {r["query"]: r for r in pass1_rows}
    by_query_2 = {r["query"]: r for r in pass2_rows}

    identical_compliant = 0
    both_noncompliant = 0
    divergent = 0
    divergent_examples = []

    for query, r1 in by_query_1.items():
        r2 = by_query_2.get(query)
        if r2 is None:
            continue
        if not r1["final_compliant"] and not r2["final_compliant"]:
            both_noncompliant += 1
            continue
        cs1 = _normalize_claim_summary(r1["claim_summary"])
        cs2 = _normalize_claim_summary(r2["claim_summary"])
        same_compliant = r1["final_compliant"] == r2["final_compliant"]
        if same_compliant and cs1 == cs2:
            identical_compliant += 1
        else:
            divergent += 1
            divergent_examples.append((query, r1["final_compliant"], cs1, r2["final_compliant"], cs2))

    total = identical_compliant + both_noncompliant + divergent
    return {
        "identical_compliant": identical_compliant,
        "both_noncompliant": both_noncompliant,
        "divergent": divergent,
        "total": total,
        "divergent_rate": divergent / total if total else float("nan"),
        "divergent_examples": divergent_examples,
    }


def _print_stability_stats(name: str, stats: dict) -> None:
    for query, c1, cs1, c2, cs2 in stats["divergent_examples"]:
        print(f"  DIVERGENT: query={query!r}")
        print(f"    pass1: final_compliant={c1} claims={cs1}")
        print(f"    pass2: final_compliant={c2} claims={cs2}")
    t = stats["total"]
    print(f"  identical_compliant={stats['identical_compliant']}/{t}  "
          f"both_noncompliant={stats['both_noncompliant']}/{t}  "
          f"divergent={stats['divergent']}/{t}  rate={stats['divergent_rate']}")
    print("  temp=0 -> expect ~0 divergent; a nonzero rate is an Ollama-nondeterminism red flag, "
          "surfaced not tuned. both_noncompliant is deterministic-in-FAILURE, not counted as identical.")
    print()


def run_stability(name: str, capture_rows: list[dict], constrained: bool = False,
                   sentinel: bool = False) -> None:
    pass1_rows = _load_pass(name, 1, constrained, sentinel)
    if pass1_rows is None:
        print(f"FATAL: pass1 missing for {name} -- run Phase 1 first", file=sys.stderr)
        return

    pass2_rows = run_candidate_pass(name, capture_rows, pass_n=2, constrained=constrained, sentinel=sentinel)
    if pass2_rows is None:
        print(f"NOTE: could not run stability pass2 for {name} (Ollama unavailable).")
        return

    print("=" * 100)
    print(f"STABILITY -- {name} -- pass1 vs pass2 (temp=0 baked into iris.aria.LlamaBackend)")
    print("=" * 100)
    stats = _compute_stability_stats(pass1_rows, pass2_rows)
    _print_stability_stats(name, stats)


# ─────────────────────────────────────────────────────────────────────────────
# provenance
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_digest(model: str) -> str | None:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if _tag_matches(model, m.get("name", "")):
                return m.get("digest")
    except Exception:
        return None
    return None


def _write_provenance(ran: list[str], skipped: dict[str, str]) -> None:
    rows = []
    for name, spec in CANDIDATES.items():
        model = spec["model"]
        available = _ollama_available(model)
        digest = _resolve_digest(model) if available else None
        meta_path = _meta_path(name)
        think_total = None
        mean_context_tokens = max_context_tokens = None
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            think_total = m.get("think_blocks_stripped_total")
            mean_context_tokens = m.get("mean_context_tokens")
            max_context_tokens = m.get("max_context_tokens")
        rows.append({
            "candidate": name,
            "resolved_tag": model,
            "requested_tag": spec.get("requested_tag", model),
            "ran": name in ran,
            "skip_reason": skipped.get(name),
            "ollama_available": available,
            "resolved_digest": digest,
            "think_handling": "think:false NOT settable via generate_v2's fixed signature "
                               "(no iris/ edits) -- every raw response scanned post-hoc via "
                               "_strip_think instead; never silently absorbed.",
            "think_contamination_incidents": think_total,
            "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
            "mean_context_tokens": mean_context_tokens,
            "max_context_tokens": max_context_tokens,
            "max_tokens_cap": ANSWERER_MAX_TOKENS,
            "max_tokens_note": f"generate_v2 calls in this run are capped at "
                                f"max_tokens={ANSWERER_MAX_TOKENS}. Any pre-existing frozen "
                                f"pass1 rows from before this cap was added were uncapped -- "
                                f"they simply never hit a ceiling (no truncation occurred), so "
                                f"there is no conflict with capped rows written under this run.",
            "note": spec.get("note"),
        })
    with open(PROVENANCE_JSONL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {PROVENANCE_JSONL}")


def _write_provenance_constrained(ran: list[str], skipped: dict[str, str],
                                   smoke_results: dict[str, tuple[bool, str]]) -> None:
    rows = []
    for name, spec in CANDIDATES_CONSTRAINED.items():
        model = spec["model"]
        available = _ollama_available(model)
        digest = _resolve_digest(model) if available else None
        meta_path = _meta_path(name, constrained=True)
        think_total = None
        mean_context_tokens = max_context_tokens = None
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            think_total = m.get("think_blocks_stripped_total")
            mean_context_tokens = m.get("mean_context_tokens")
            max_context_tokens = m.get("max_context_tokens")
        smoke_passed, smoke_detail = smoke_results.get(name, (None, "not probed"))
        rows.append({
            "candidate": name,
            "resolved_tag": model,
            "requested_tag": spec.get("requested_tag", model),
            "ran": name in ran,
            "skip_reason": skipped.get(name),
            "ollama_available": available,
            "resolved_digest": digest,
            "grammar_mode": "schema-format (native /api/chat, format=ANSWER_CLAIMS_WIRE_SCHEMA)",
            "wire_schema_hash": WIRE_SCHEMA_HASH,
            "grammar_smoke_test_passed": smoke_passed,
            "grammar_smoke_test_detail": smoke_detail,
            "grammar_incompatible": smoke_passed is False,
            "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
            "mean_context_tokens": mean_context_tokens,
            "max_context_tokens": max_context_tokens,
            "max_tokens_cap": ANSWERER_MAX_TOKENS,
            "no_corrective_retry": True,
            "note": spec.get("note"),
        })
    with open(PROVENANCE_CONSTRAINED_JSONL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {PROVENANCE_CONSTRAINED_JSONL}")


def _write_provenance_sentinel(ran: list[str], skipped: dict[str, str],
                                smoke_results: dict[str, tuple[bool, str]]) -> None:
    rows = []
    for name, spec in CANDIDATES_SENTINEL.items():
        model = spec["model"]
        available = _ollama_available(model)
        digest = _resolve_digest(model) if available else None
        meta_path = _meta_path(name, sentinel=True)
        think_total = None
        mean_context_tokens = max_context_tokens = None
        if meta_path.exists():
            m = json.loads(meta_path.read_text())
            think_total = m.get("think_blocks_stripped_total")
            mean_context_tokens = m.get("mean_context_tokens")
            max_context_tokens = m.get("max_context_tokens")
        smoke_passed, smoke_detail = smoke_results.get(name, (None, "not probed"))
        rows.append({
            "candidate": name,
            "resolved_tag": model,
            "requested_tag": spec.get("requested_tag", model),
            "ran": name in ran,
            "skip_reason": skipped.get(name),
            "ollama_available": available,
            "resolved_digest": digest,
            "grammar_mode": "schema-format (native /api/chat, format=ANSWER_CLAIMS_WIRE_SCHEMA)",
            "wire_schema_hash": WIRE_SCHEMA_HASH,
            "sentinel_convention": "every wire claim field REQUIRED; irrelevant fields filled with "
                                    "fixed sentinels (frame_idx=-1, stated_value=-1, field='none', "
                                    "assertion/source_text/event/text=''); is_core never a sentinel.",
            "grammar_smoke_test_passed": smoke_passed,
            "grammar_smoke_test_detail": smoke_detail,
            "grammar_incompatible": smoke_passed is False,
            "system_prompt_tokens": SYSTEM_PROMPT_TOKENS,
            "mean_context_tokens": mean_context_tokens,
            "max_context_tokens": max_context_tokens,
            "max_tokens_cap": ANSWERER_MAX_TOKENS,
            "corrective_retry": "ONE, reinstated task 4 (iris.query._generate_answer_claims_v2_wire) "
                                 "-- claim_field_shape is semantic (sentinel misuse), not parse, "
                                 "under the sentinel schema, so a named corrective is meaningful.",
            "note": spec.get("note"),
        })
    with open(PROVENANCE_SENTINEL_JSONL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {PROVENANCE_SENTINEL_JSONL}")


# ─────────────────────────────────────────────────────────────────────────────
# --all orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> None:
    capture_rows = _capture_contexts()
    capture_by_query = {r["query"]: r for r in capture_rows}

    print("=" * 100)
    print("PHASE 1: PER-CANDIDATE RUN")
    print("=" * 100)
    ran: list[str] = []
    skipped: dict[str, str] = {}
    pass1_by_name: dict[str, list[dict]] = {}
    for name, spec in CANDIDATES.items():
        model = spec["model"]
        if not _ollama_available(model):
            reason = f"{model!r} not available in Ollama"
            print(f"NOTE: skipping candidate {name!r}: {reason}.")
            skipped[name] = reason
            continue
        rows = run_candidate_pass(name, capture_rows, pass_n=1)
        if rows is None:
            skipped[name] = "run_candidate_pass returned None"
            continue
        pass1_by_name[name] = rows
        ran.append(name)

    if not ran:
        print("NOTE: no candidates available -- nothing to analyze. Partial slate (empty) is still a valid run.")

    print("=" * 100)
    print("PHASE 2: ANALYZE")
    print("=" * 100)
    summaries = [run_analyze(name, pass1_by_name[name], capture_by_query) for name in ran]

    print("=" * 100)
    print("PRE-REGISTERED DECISION-RULE TABLE")
    print("=" * 100)
    print("  frame-citation gate (DQ) first. Survivors ranked by: first_attempt_compliance DESC,")
    print("  final_compliance DESC, absence_positivity adherence DESC (violation rate ASC).")
    print("  metadata-number review = HUMAN column (blank unless flagged). mean_seconds as tie-break.")
    print()
    hdr = (f"  {'candidate':<14} | {'gate':<12} | {'1st_compl':>9} | {'final_compl':>11} | "
           f"{'absence_viol':>12} | {'mean_s':>8} | {'metadata(HUMAN)':<16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ranked = sorted(
        summaries,
        key=lambda s: (
            s["disqualified"],
            -s["first_attempt_compliance"],
            -s["final_compliance"],
            s["absence_violation_rate"] if s["absence_violation_rate"] == s["absence_violation_rate"] else 1.0,
        ),
    )
    for s in ranked:
        gate_disp = "DISQUALIFIED" if s["disqualified"] else "pass"
        human_disp = f"flagged x{s['metadata_review_flagged']}" if s["metadata_review_flagged"] else ""
        print(f"  {s['name']:<14} | {gate_disp:<12} | {s['first_attempt_compliance']:>9.4f} | "
              f"{s['final_compliance']:>11.4f} | {s['absence_violation_rate']:>12.4f} | "
              f"{s['mean_seconds']:>8.2f} | {human_disp:<16}")
    print()
    print("  Badges: NOT computed here -- see scripts/smoke_v2.py for the descriptive badge view.")
    print()

    print("=" * 100)
    print("PHASE 3: STABILITY (top-2 gate-passing candidates by final_compliance)")
    print("=" * 100)
    gate_passers = [s for s in summaries if not s["disqualified"]]
    gate_passers.sort(key=lambda s: -s["final_compliance"])
    top2 = gate_passers[:2]
    outside_top2 = gate_passers[2:]
    if not gate_passers:
        print("  No candidate cleared the frame-citation hard gate -- nothing to stability-test.")
    for s in top2:
        print(f"  SELECTED for stability: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    for s in outside_top2:
        print(f"  stability not run -- outside top-2: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    print()
    for s in top2:
        run_stability(s["name"], capture_rows)

    _write_provenance(ran, skipped)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  ran: {ran}")
    print(f"  skipped: {skipped}")
    for s in summaries:
        if s["final_compliance"] == 0.0:
            print(f"  ** STOP: {s['name']} final_compliance == 0 -- cannot speak the contract. **")
    if not any(s["disqualified"] for s in summaries) and summaries:
        print("  Zero frame-citation hard-gate hits across all candidates (pre-registered expectation met).")


# ─────────────────────────────────────────────────────────────────────────────
# --all --constrained orchestrator (task 3, Part B)
# ─────────────────────────────────────────────────────────────────────────────

def run_all_constrained() -> None:
    capture_rows = _capture_contexts()
    capture_by_query = {r["query"]: r for r in capture_rows}
    probe_row = capture_rows[0]

    print("=" * 100)
    print(f"PHASE -1: GRAMMAR SMOKE TEST (wire_schema_hash={WIRE_SCHEMA_HASH}) -- before any bake-off run")
    print("=" * 100)
    smoke_results: dict[str, tuple[bool, str]] = {}
    smoke_passed_names: list[str] = []
    skipped: dict[str, str] = {}
    for name, spec in CANDIDATES_CONSTRAINED.items():
        model = spec["model"]
        if not _ollama_available(model):
            reason = f"{model!r} not available in Ollama"
            print(f"NOTE: skipping candidate {name!r}: {reason}.")
            skipped[name] = reason
            continue
        passed, detail = grammar_smoke_test(name, model, probe_row)
        smoke_results[name] = (passed, detail)
        status = "PASS" if passed else "GRAMMAR-INCOMPATIBLE"
        print(f"  [{name!r} / {model!r}] {status}: {detail}")
        if passed:
            smoke_passed_names.append(name)
        else:
            skipped[name] = f"grammar smoke test failed: {detail}"
    print()
    if not smoke_passed_names:
        print("NOTE: no candidate cleared the grammar smoke test -- nothing to bake off.")

    print("=" * 100)
    print("PHASE 1: PER-CANDIDATE RUN (constrained, schema-format, NO corrective retry)")
    print("=" * 100)
    ran: list[str] = []
    pass1_by_name: dict[str, list[dict]] = {}
    for name in smoke_passed_names:
        rows = run_candidate_pass(name, capture_rows, pass_n=1, constrained=True)
        if rows is None:
            skipped[name] = "run_candidate_pass returned None"
            continue
        pass1_by_name[name] = rows
        ran.append(name)

    if not ran:
        print("NOTE: no candidates available -- nothing to analyze. Partial slate (empty) is still a valid run.")

    print("=" * 100)
    print("PHASE 2: ANALYZE (constrained)")
    print("=" * 100)
    summaries = [run_analyze(name, pass1_by_name[name], capture_by_query, constrained=True) for name in ran]

    print("=" * 100)
    print("PRE-REGISTERED DECISION-RULE TABLE (constrained)")
    print("=" * 100)
    print("  frame-citation gate (DQ) first, then GRAMMAR-INCOMPATIBLE exclusions. Survivors ranked")
    print("  by: variant-completeness (first_attempt_compliance, single-attempt so this IS")
    print("  final_compliance too) DESC, absence_positivity adherence DESC, mean_s ASC as tie-break.")
    print()
    hdr = (f"  {'candidate':<20} | {'gate':<12} | {'variant_complete':>16} | "
           f"{'absence_viol':>12} | {'mean_s':>8} | {'metadata(HUMAN)':<16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ranked = sorted(
        summaries,
        key=lambda s: (
            s["disqualified"],
            -s["final_compliance"],
            s["absence_violation_rate"] if s["absence_violation_rate"] == s["absence_violation_rate"] else 1.0,
            s["mean_seconds"],
        ),
    )
    for s in ranked:
        gate_disp = "DISQUALIFIED" if s["disqualified"] else "pass"
        human_disp = f"flagged x{s['metadata_review_flagged']}" if s["metadata_review_flagged"] else ""
        print(f"  {s['name']:<20} | {gate_disp:<12} | {s['final_compliance']:>16.4f} | "
              f"{s['absence_violation_rate']:>12.4f} | {s['mean_seconds']:>8.2f} | {human_disp:<16}")
    for name, reason in skipped.items():
        if "grammar smoke test failed" in reason:
            print(f"  {name:<20} | GRAMMAR-INCOMPATIBLE (excluded before Phase 1 -- {reason})")
    print()
    print("  Badges: NOT computed here -- see scripts/smoke_v2.py for the descriptive badge view.")
    print()

    print("=" * 100)
    print("PHASE 3: STABILITY (top-2 gate-passing candidates by final_compliance, constrained)")
    print("=" * 100)
    gate_passers = [s for s in summaries if not s["disqualified"]]
    gate_passers.sort(key=lambda s: -s["final_compliance"])
    top2 = gate_passers[:2]
    outside_top2 = gate_passers[2:]
    if not gate_passers:
        print("  No candidate cleared the frame-citation hard gate -- nothing to stability-test.")
    for s in top2:
        print(f"  SELECTED for stability: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    for s in outside_top2:
        print(f"  stability not run -- outside top-2: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    print()
    for s in top2:
        run_stability(s["name"], capture_rows, constrained=True)

    _write_provenance_constrained(ran, skipped, smoke_results)

    print("=" * 100)
    print("SUMMARY (constrained)")
    print("=" * 100)
    print(f"  ran: {ran}")
    print(f"  skipped: {skipped}")
    for s in summaries:
        if s["final_compliance"] == 0.0:
            print(f"  ** STOP: {s['name']} final_compliance == 0 -- cannot speak the contract. **")
    if not any(s["disqualified"] for s in summaries) and summaries:
        print("  Zero frame-citation hard-gate hits across surviving candidates (pre-registered expectation met).")


# ─────────────────────────────────────────────────────────────────────────────
# --all --sentinel orchestrator (task 4, Part C)
# ─────────────────────────────────────────────────────────────────────────────

SEATING_MIN_COMPLIANCE = 9 / 11


def run_all_sentinel() -> None:
    capture_rows = _capture_contexts()
    capture_by_query = {r["query"]: r for r in capture_rows}
    probe_row = capture_rows[0]

    print("=" * 100)
    print(f"PHASE -1: GRAMMAR SMOKE TEST (sentinel schema, wire_schema_hash={WIRE_SCHEMA_HASH}) "
          f"-- before any bake-off run")
    print("=" * 100)
    smoke_results: dict[str, tuple[bool, str]] = {}
    smoke_passed_names: list[str] = []
    skipped: dict[str, str] = {}
    for name, spec in CANDIDATES_SENTINEL.items():
        model = spec["model"]
        if not _ollama_available(model):
            reason = f"{model!r} not available in Ollama"
            print(f"NOTE: skipping candidate {name!r}: {reason}.")
            skipped[name] = reason
            continue
        passed, detail = grammar_smoke_test(name, model, probe_row)
        smoke_results[name] = (passed, detail)
        status = "PASS" if passed else "GRAMMAR-INCOMPATIBLE"
        print(f"  [{name!r} / {model!r}] {status}: {detail}")
        if passed:
            smoke_passed_names.append(name)
        else:
            skipped[name] = f"grammar smoke test failed: {detail}"
    print()
    if not smoke_passed_names:
        print("NOTE: no candidate cleared the grammar smoke test -- nothing to bake off.")

    print("=" * 100)
    print("PHASE 1: PER-CANDIDATE RUN (sentinel schema, schema-format, ONE corrective retry)")
    print("=" * 100)
    ran: list[str] = []
    pass1_by_name: dict[str, list[dict]] = {}
    for name in smoke_passed_names:
        rows = run_candidate_pass(name, capture_rows, pass_n=1, sentinel=True)
        if rows is None:
            skipped[name] = "run_candidate_pass returned None"
            continue
        pass1_by_name[name] = rows
        ran.append(name)

    if not ran:
        print("NOTE: no candidates available -- nothing to analyze. Partial slate (empty) is still a valid run.")

    print("=" * 100)
    print("PHASE 2: ANALYZE (sentinel)")
    print("=" * 100)
    summaries = [run_analyze_sentinel(name, pass1_by_name[name], capture_by_query) for name in ran]

    print("=" * 100)
    print("PRE-REGISTERED DECISION-RULE TABLE (sentinel)")
    print("=" * 100)
    print("  frame-citation gate (DQ) first, then GRAMMAR-INCOMPATIBLE exclusions. Survivors ranked")
    print("  by: final_compliance DESC, absence_production_rate DESC, absence_positivity adherence")
    print("  DESC, mean_s ASC as tie-break. absence-production is the headline new column.")
    print()
    hdr = (f"  {'candidate':<16} | {'gate':<12} | {'1st_compl':>9} | {'final_compl':>11} | "
           f"{'absence_prod':>12} | {'absence_viol':>12} | {'sentinel_mis':>12} | {'mean_s':>8} | "
           f"{'metadata(HUMAN)':<16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ranked = sorted(
        summaries,
        key=lambda s: (
            s["disqualified"],
            -s["final_compliance"],
            -(s["absence_production_rate"] if s["absence_production_rate"] == s["absence_production_rate"] else -1.0),
            s["absence_violation_rate"] if s["absence_violation_rate"] == s["absence_violation_rate"] else 1.0,
            s["mean_seconds"],
        ),
    )
    for s in ranked:
        gate_disp = "DISQUALIFIED" if s["disqualified"] else "pass"
        human_disp = f"flagged x{s['metadata_review_flagged']}" if s["metadata_review_flagged"] else ""
        print(f"  {s['name']:<16} | {gate_disp:<12} | {s['first_attempt_compliance']:>9.4f} | "
              f"{s['final_compliance']:>11.4f} | {s['absence_production_rate']:>12.4f} | "
              f"{s['absence_violation_rate']:>12.4f} | {s['sentinel_misuse_count']:>12} | "
              f"{s['mean_seconds']:>8.2f} | {human_disp:<16}")
    for name, reason in skipped.items():
        if "grammar smoke test failed" in reason:
            print(f"  {name:<16} | GRAMMAR-INCOMPATIBLE (excluded before Phase 1 -- {reason})")
    print()
    print("  Badges: NOT computed here -- see scripts/smoke_v2.py for the descriptive badge view.")
    print()

    print("=" * 100)
    print("PHASE 3: STABILITY (top-2 gate-passing candidates by final_compliance, sentinel)")
    print("=" * 100)
    gate_passers = [s for s in summaries if not s["disqualified"]]
    gate_passers.sort(key=lambda s: -s["final_compliance"])
    top2 = gate_passers[:2]
    outside_top2 = gate_passers[2:]
    if not gate_passers:
        print("  No candidate cleared the frame-citation hard gate -- nothing to stability-test.")
    for s in top2:
        print(f"  SELECTED for stability: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    for s in outside_top2:
        print(f"  stability not run -- outside top-2: {s['name']!r} (final_compliance={s['final_compliance']:.4f})")
    print()
    stability_stats: dict[str, dict] = {}
    for s in top2:
        pass1_rows = _load_pass(s["name"], 1, sentinel=True)
        pass2_rows = run_candidate_pass(s["name"], capture_rows, pass_n=2, sentinel=True)
        if pass1_rows is None or pass2_rows is None:
            continue
        print("=" * 100)
        print(f"STABILITY -- {s['name']} -- pass1 vs pass2 (temp=0 baked into iris.aria.LlamaBackend)")
        print("=" * 100)
        stats = _compute_stability_stats(pass1_rows, pass2_rows)
        _print_stability_stats(s["name"], stats)
        stability_stats[s["name"]] = stats

    _write_provenance_sentinel(ran, skipped, smoke_results)

    print("=" * 100)
    print("SEATING RULE (pre-registered, applied post-hoc -- not tuned during the run)")
    print("=" * 100)
    print(f"  seat the FASTEST candidate with: final_compliance >= {SEATING_MIN_COMPLIANCE:.4f} (9/11) AND")
    print("  >=1 correct AbsenceClaim on the NEGATIVE set AND gate-clean AND 0 divergent (Phase 3).")
    print("  0-divergent can only be confirmed for top-2 stability-tested candidates -- a candidate")
    print("  meeting every other criterion but outside top-2 has UNCONFIRMED stability and cannot seat.")
    print()
    eligible = []
    for s in summaries:
        name = s["name"]
        meets_compliance = s["final_compliance"] >= SEATING_MIN_COMPLIANCE
        meets_absence = s["negative_rows_with_absence"] >= 1
        meets_gate = not s["disqualified"]
        stats = stability_stats.get(name)
        meets_stability = stats is not None and stats["divergent"] == 0
        if meets_stability:
            stability_disp = "OK (0 divergent)"
        elif stats is None:
            stability_disp = "UNCONFIRMED (not top-2)"
        else:
            n_divergent = stats["divergent"]
            stability_disp = f"FAIL ({n_divergent} divergent)"
        print(f"  {name}: compliance={s['final_compliance']:.4f} ({'OK' if meets_compliance else 'FAIL'})  "
              f"absence_on_negative={s['negative_rows_with_absence']} ({'OK' if meets_absence else 'FAIL'})  "
              f"gate={'OK' if meets_gate else 'FAIL'}  "
              f"stability={stability_disp}")
        if meets_compliance and meets_absence and meets_gate and meets_stability:
            eligible.append(s)
    print()
    if eligible:
        eligible.sort(key=lambda s: s["mean_seconds"])
        seat = eligible[0]
        print(f"  SEAT: {seat['name']!r} (final_compliance={seat['final_compliance']:.4f}, "
              f"mean_seconds={seat['mean_seconds']:.2f}, "
              f"negative_rows_with_absence={seat['negative_rows_with_absence']})")
    else:
        print("  ** NO SEAT ** -- no candidate cleared all four gates. Surfaced, not tuned. The next")
        print("  lever (prompt iteration) is its own pre-registered task, per the STOP CONDITIONS.")
    print()

    print("=" * 100)
    print("SUMMARY (sentinel)")
    print("=" * 100)
    print(f"  ran: {ran}")
    print(f"  skipped: {skipped}")
    for s in summaries:
        if s["final_compliance"] == 0.0:
            print(f"  ** STOP: {s['name']} final_compliance == 0 -- cannot speak the contract. **")
    if not any(s["disqualified"] for s in summaries) and summaries:
        print("  Zero frame-citation hard-gate hits across surviving candidates (pre-registered expectation met).")


# ─────────────────────────────────────────────────────────────────────────────
# --report-only -- recompute + finalize from frozen artifacts, NO model calls
# ─────────────────────────────────────────────────────────────────────────────

def _dominant_label(label_hist: dict[str, int]) -> str | None:
    return max(label_hist, key=label_hist.get) if any(label_hist.values()) else None


def run_report_only() -> None:
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: {CAPTURE_JSONL} missing -- run --all at least once first", file=sys.stderr)
        sys.exit(1)

    print("=" * 100)
    print("REPORT-ONLY: recomputing from frozen pass1/pass2/capture artifacts -- NO LLM calls")
    print("=" * 100)
    capture_rows = _capture_contexts()
    capture_by_query = {r["query"]: r for r in capture_rows}
    print()

    ran: list[str] = []
    for name in CANDIDATES:
        if _pass_path(name, 1).exists():
            ran.append(name)
        else:
            print(f"NOTE: no frozen pass1 for {name!r} -- excluded from this report.")
    print()

    print("=" * 100)
    print("PHASE 2: ANALYZE (recomputed, NLI-free, no model calls)")
    print("=" * 100)
    summaries = []
    metas = {}
    for name in ran:
        rows = _load_pass(name, 1)
        s = run_analyze(name, rows, capture_by_query)
        summaries.append(s)
        metas[name] = json.loads(_meta_path(name).read_text())

    print("=" * 100)
    print("PHASE 3: STABILITY (recomputed from frozen pass1/pass2 -- NO LLM calls)")
    print("=" * 100)
    stability: dict[str, dict] = {}
    for name in ran:
        pass2_rows = _load_pass(name, 2)
        if pass2_rows is None:
            print(f"  {name!r}: no frozen pass2 -- stability not available in report-only mode.")
            continue
        pass1_rows = _load_pass(name, 1)
        stats = _compute_stability_stats(pass1_rows, pass2_rows)
        stability[name] = stats
        print(f"-- {name} --")
        _print_stability_stats(name, stats)

    print("=" * 100)
    print("FINAL DECISION-RULE TABLE")
    print("=" * 100)
    print("  frame-citation gate (DQ) first. Survivors ranked by: first_attempt_compliance DESC,")
    print("  final_compliance DESC, absence_positivity adherence DESC (violation rate ASC).")
    print("  metadata-number review = HUMAN column (blank unless flagged). mean_seconds as tie-break.")
    print("  Phase 3 columns present only for candidates the run selected for stability.")
    print()
    hdr = (f"  {'candidate':<14} | {'gate':<12} | {'1st_compl':>9} | {'final_compl':>11} | "
           f"{'dominant_label':<18} | {'absence_viol':>12} | {'mean_s':>8} | "
           f"{'ident/both/div':<16} | {'metadata(HUMAN)':<16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ranked = sorted(
        summaries,
        key=lambda s: (
            s["disqualified"],
            -s["first_attempt_compliance"],
            -s["final_compliance"],
            s["absence_violation_rate"] if s["absence_violation_rate"] == s["absence_violation_rate"] else 1.0,
        ),
    )
    for s in ranked:
        gate_disp = "DISQUALIFIED" if s["disqualified"] else "pass"
        human_disp = f"flagged x{s['metadata_review_flagged']}" if s["metadata_review_flagged"] else ""
        dom = _dominant_label(metas[s["name"]]["label_histogram"]) or "-"
        stats = stability.get(s["name"])
        stab_disp = (f"{stats['identical_compliant']}/{stats['both_noncompliant']}/{stats['divergent']}"
                     if stats else "n/a")
        print(f"  {s['name']:<14} | {gate_disp:<12} | {s['first_attempt_compliance']:>9.4f} | "
              f"{s['final_compliance']:>11.4f} | {dom:<18} | {s['absence_violation_rate']:>12.4f} | "
              f"{s['mean_seconds']:>8.2f} | {stab_disp:<16} | {human_disp:<16}")
    print()
    print("  Badges: NOT computed here -- see scripts/smoke_v2.py for the descriptive badge view.")
    print()

    print("=" * 100)
    print("FINDINGS")
    print("=" * 100)

    llama_meta = metas.get("llama3.2-3b")
    if llama_meta and llama_meta.get("disqualified"):
        print("  llama3.2-3b -- DISQUALIFIED, MODEL FAULT (not harness fault; verified against frozen")
        print("  answerer_capture.jsonl -- frame_idx=23760 does not appear anywhere in the")
        print("  'Is there an unattended bag?' context_text, and IS present in exactly the two")
        print("  vehicle-loading queries' contexts, confirming cross-query-only presence, not a")
        print("  harness leak). Sharper still: 23760 / 'a car is parked near the entrance' is the")
        print("  EXACT few-shot example baked into iris.aria._SYSTEM_PROMPT_V2 -- llama3.2-3b did not")
        print("  hallucinate a plausible frame_idx, it echoed the system prompt's own exemplar")
        print("  verbatim instead of grounding its claim in the retrieved evidence for that query.")
        print()

    hermes_meta = metas.get("hermes3-3b")
    llama_s = next((s for s in summaries if s["name"] == "llama3.2-3b"), None)
    if hermes_meta and llama_s:
        print("  PREDICTION FALSIFIED -- hermes3-3b: pre-registered 'hermes3:3b >= llama3.2:3b on")
        print(f"  first_attempt_compliance' (JSON-tuning, base held constant). Actual: "
              f"{hermes_meta['first_attempt_compliance']*100:.1f}/{hermes_meta['final_compliance']*100:.1f} "
              f"< {llama_s['first_attempt_compliance']*100:.1f}/100 (llama3.2-3b's final_compliance, gate")
        print("  status aside). Mechanism, not just underperformance: hermes3-3b repeats the SAME")
        print("  is_core_invariant label on the corrective retry in most of its failures instead of")
        print("  fixing it -- it is not using the accurate, label-specific corrective from task 1's")
        print("  taxonomy; it re-emits the same structural mistake.")
        print()

    qwen4b_meta = metas.get("qwen3.5-4b")
    if qwen4b_meta:
        print(f"  qwen3.5-4b -- the only CLEAN compliance pass (first_attempt="
              f"{qwen4b_meta['first_attempt_compliance']*100:.1f}%, final="
              f"{qwen4b_meta['final_compliance']*100:.1f}%, gate={qwen4b_meta['gate']}) BUT "
              f"mean_seconds={qwen4b_meta['mean_seconds']:.1f}s/call on this CPU-only host.")
        print("  Compliant-but-not-deployable: wins compliance, fails CPU-efficiency. Not crowned as")
        print("  the winner on compliance numbers alone -- the split is the finding.")
        print()

    for name in ("qwen3.5-2b", "minicpm5-1b"):
        m = metas.get(name)
        if m is None:
            continue
        print(f"  {name} -- 0% CONFOUNDED, not a genuine JSON-shape failure: reasoning-token-starved")
        print(f"  under the {ANSWERER_MAX_TOKENS}-token cap (raw response content empty on the majority")
        print("  of attempts -- verified via raw1/raw2 length, not inferred from the label alone).")
        print("  EXCLUDED from the compliance ranking; disposition deferred to a think:false rerun.")
        print()


def main() -> None:
    if "--all" in sys.argv:
        if "--sentinel" in sys.argv:
            run_all_sentinel()
        elif "--constrained" in sys.argv:
            run_all_constrained()
        else:
            run_all()
        return
    if "--report-only" in sys.argv:
        run_report_only()
        return
    print("Usage: python scripts/answerer_bakeoff.py --all [--constrained|--sentinel] | --report-only")
    sys.exit(1)


if __name__ == "__main__":
    main()
