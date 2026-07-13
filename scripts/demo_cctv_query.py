"""CCTV query demo / diagnostic entrypoints against the cached VIRAT_S_000102
scene_sparse index. Diagnostic only, no mechanism change.

--polarity-suite: characterize the Cerberus-V rejection pattern by answer
polarity. Runs a fixed set of POSITIVE-likely ("yes, X is present") and
NEGATIVE/ABSENCE-likely ("no X") CCTV queries through the full query() path
(iris.query.query -- retrieval, lazy captioning, ARIA, Cerberus-V) and tallies
verified vs rejected claims per polarity bucket. Reuses the index cache built
by scripts/virat_query_smoke.py -- no re-ingest.

--nli-trace: per-claim NLI mechanism dump. For each claim ARIA emits, prints
the exact fact strings Cerberus-V compared it against, the per-(claim,fact)
NLI label+score, and the final verdict -- then splits claims into
"cites metadata" (Frame N / timestamp / action_score / persistence) vs "plain
visual assertion" and tallies reject rate for each. Runs CerberusV's REAL
tokenizer/model; the trace hook is a verbatim-logic clone of
iris.cerberus_v.CerberusV._full_nli with capture added, monkeypatched onto
the class only for the duration of each traced call (restored immediately
after) -- no edits to iris/cerberus_v.py, no change to the verdict logic.

--cerberus-wiring-trace: audits whether Cerberus-V's evidence comes from the
NEW scene_sparse retrieval stack or a stale/legacy path. For ~3 queries dumps,
in order: (1) retrieved_frames field presence per frame_idx, (2) the L1
Elysium CachedFrame entries after wrapper_populate_cache, (3) the exact
context_text string ARIA received, (4) the per-claim NLI evidence/label/
verdict trace (reusing the same verbatim-logic _full_nli clone as
--nli-trace). Then answers three explicit yes/no wiring questions from the
dumps. Diagnostic only -- no fix, no code path changes.

VERIFY: python scripts/demo_cctv_query.py --polarity-suite
        python scripts/demo_cctv_query.py --nli-trace
        python scripts/demo_cctv_query.py --cerberus-wiring-trace
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.cerberus_v import CerberusV
from iris.iris_config import IRISConfig

VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_000102.mp4"
CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_000102"

CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
    graph_mode="scene_sparse",
)

POSITIVE_QUERIES = [
    "Is there a car in the parking lot?",
    "Is a person visible in the scene?",
    "Are there vehicles parked?",
    "Is anyone walking?",
]

NEGATIVE_QUERIES = [
    "Is anyone loading a vehicle?",
    "Is there an unattended bag?",
    "Is anyone running or fleeing?",
    "Is there a fire or smoke?",
]


NLI_TRACE_QUERIES = [
    "Is there a car in the parking lot?",
    "Are there vehicles parked?",
    "Is anyone loading a vehicle?",
    "Is there a fire or smoke?",
    "Is anyone walking?",
]

# Group A: claim cites pipeline metadata (frame number, timestamp, action_score,
# persistence) -- the kind of token _frame_to_display_text() puts in ARIA's
# context but _frame_to_nli_fact() deliberately strips from the NLI fact pool.
# Group B: plain visual assertions with no such tokens.
METADATA_RE = re.compile(
    r"\bFrame\s*#?\d+\b|\btimestamps?\b|\baction[\s_]score\b|\bpersistence\b",
    re.IGNORECASE,
)


def _traced_full_nli(self, claims, facts, mode, action_score, sink):
    """Verbatim-logic clone of CerberusV._full_nli (iris/cerberus_v.py) with
    per-(claim,fact) trace capture added. Monkeypatched onto the class for a
    single call only -- does not alter the real module, the verdict logic is
    copied unchanged (same thresholds, same aggregation rule)."""
    if not claims:
        return {"verified": [], "rejected": [], "unverifiable": [], "mode": mode, "action_score": action_score}
    if not facts:
        for c in claims:
            sink.append({"claim": c, "mode": mode, "relevant": [], "total_facts_pool": 0, "verdict": "unverifiable"})
        return {"verified": [], "rejected": [], "unverifiable": claims.copy(), "mode": mode, "action_score": action_score}

    import torch
    import torch.nn.functional as F

    nlp = self._get_spacy()
    tokenizer, model = self._get_nli_model()
    device = model.device

    pairs = [(claim, fact) for claim in claims for fact in facts]

    import psutil
    base_batch_size = 64
    try:
        ram_percent = psutil.virtual_memory().percent
        if ram_percent >= 85:
            base_batch_size = 8
        elif ram_percent >= 75:
            base_batch_size = 16
        elif ram_percent >= 60:
            base_batch_size = 32
    except Exception:
        pass

    pair_results = {}
    for i in range(0, len(pairs), base_batch_size):
        batch = pairs[i:i + base_batch_size]
        batch_claims = [p[0] for p in batch]
        batch_facts = [p[1] for p in batch]

        inputs = tokenizer(batch_facts, batch_claims, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        for j, (claim, fact) in enumerate(batch):
            pred = predictions[j]
            prob_dist = probs[j]

            label = "neutral"
            id2label = getattr(model.config, "id2label", {}) or {}
            raw_label = str(id2label.get(pred, "")).lower()
            if "entail" in raw_label:
                label = "entailment"
            elif "contrad" in raw_label:
                label = "contradiction"
            elif "neutral" in raw_label:
                label = "neutral"
            else:
                if pred == 2:
                    label = "entailment"
                elif pred == 0:
                    label = "contradiction"

            entail_idx = 2
            for idx, lbl in id2label.items():
                if "entail" in str(lbl).lower():
                    entail_idx = idx
                    break
            entailment_score = prob_dist[entail_idx] if entail_idx < len(prob_dist) else prob_dist[-1]

            claim_doc = nlp(claim)
            has_negation_claim = any(t.dep_ == "neg" or t.lower_ == "no" for t in claim_doc)
            fact_doc = nlp(fact)
            has_negation_fact = any(t.dep_ == "neg" or t.lower_ == "no" for t in fact_doc)
            negation_high_risk = has_negation_claim and not has_negation_fact

            threshold = 0.5 if negation_high_risk else 0.85
            if label == "entailment" and negation_high_risk and entailment_score <= threshold:
                label = "neutral"

            if label == "entailment":
                claim_gpes = {ent.text.lower().strip() for ent in claim_doc.ents if ent.label_ in ("GPE", "LOC")}
                if claim_gpes and not any(gpe in fact.lower() for gpe in claim_gpes):
                    label = "neutral"

            claim_lemmas = {t.lemma_.lower() for t in claim_doc if not t.is_stop and t.is_alpha and len(t.text) > 2}
            fact_lemmas = {t.lemma_.lower() for t in fact_doc if not t.is_stop and t.is_alpha and len(t.text) > 2}
            claim_ents_set = {e.text.lower() for e in claim_doc.ents}
            fact_ents_set = {e.text.lower() for e in fact_doc.ents}
            has_topical_overlap = bool((claim_lemmas & fact_lemmas) or (claim_ents_set & fact_ents_set))

            pair_results[(claim, fact)] = (label, entailment_score, has_topical_overlap)

    verified, rejected, unverifiable = [], [], []
    for claim in claims:
        all_results = [pair_results[(claim, fact)] for fact in facts]
        relevant = [(fact, label, score) for fact, (label, score, has_overlap) in zip(facts, all_results) if has_overlap]

        if not relevant:
            unverifiable.append(claim)
            verdict = "unverifiable"
        else:
            entailed_count = sum(1 for _, label, _ in relevant if label == "entailment")
            contradicted_count = sum(1 for _, label, _ in relevant if label == "contradiction")
            if contradicted_count > entailed_count:
                rejected.append(claim)
                verdict = "rejected"
            elif entailed_count > 0:
                verified.append(claim)
                verdict = "verified"
            else:
                unverifiable.append(claim)
                verdict = "unverifiable"

        sink.append({
            "claim": claim, "mode": mode,
            "relevant": [{"fact": f, "label": l, "score": s} for f, l, s in relevant],
            "total_facts_pool": len(facts),
            "verdict": verdict,
        })

    return {"verified": verified, "rejected": rejected, "unverifiable": unverifiable, "mode": mode, "action_score": action_score}


def run_nli_trace() -> None:
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print()

    all_records: list[dict] = []
    original_full_nli = CerberusV._full_nli

    for question in NLI_TRACE_QUERIES:
        print("=" * 100)
        print(f"QUERY: {question!r}")
        print("=" * 100)

        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)
        cache_obj = iris_query.wrapper_init_l1_cache(CFG)
        iris_query.wrapper_populate_cache(cache_obj, retrieved)
        context_text = cache_obj.as_context_text()

        import iris.aria as aria
        raw_answer = aria.generate(prompt=question, context=context_text)
        claims = iris_query._split_claims(raw_answer)

        print(f"-- RAW ANSWER --\n{raw_answer}\n")
        print(f"-- SPLIT CLAIMS ({len(claims)}) --")
        for c in claims:
            print(f"  {c!r}")
        print()

        gate = CerberusV()
        action_score_val = max((f.get("action_score", 0.0) for f in retrieved), default=0.5)
        mode = gate.get_verification_mode(action_score_val, CFG)
        gate._get_spacy()
        if mode in ("full_nli", "filtered_nli"):
            gate._get_nli_model()
        parsed_claims = gate._parse_claims(claims)
        facts = [entry.text for entry in cache_obj.set_facts.values()]

        print(f"-- MODE: {mode}  (action_score={action_score_val:.3f}) --")
        print(f"-- FACT POOL ({len(facts)} entries, semantic-only, no numeric metadata) --")
        for f in facts:
            print(f"  {f!r}")
        print()

        sink: list[dict] = []
        CerberusV._full_nli = lambda self, c, f, m, a, _sink=sink: _traced_full_nli(self, c, f, m, a, _sink)
        try:
            if mode == "full_nli":
                gate._full_nli(parsed_claims, facts, mode, action_score_val)
            elif mode == "filtered_nli":
                high_conf = [c for c in parsed_claims if gate._confidence(c) >= 0.6]
                low_conf = [c for c in parsed_claims if gate._confidence(c) < 0.6]
                gate._full_nli(high_conf, facts, mode, action_score_val)
                for c in low_conf:
                    sink.append({"claim": c, "mode": "filtered_nli(low_conf_bypass)", "relevant": [], "total_facts_pool": len(facts), "verdict": "verified"})
            else:
                gate._ner_overlap(parsed_claims, facts, mode, action_score_val)
                for c in parsed_claims:
                    sink.append({"claim": c, "mode": "ner_only", "relevant": [], "total_facts_pool": len(facts), "verdict": "n/a (ner_only, no NLI pairs)"})
        finally:
            CerberusV._full_nli = original_full_nli

        print("-- PER-CLAIM NLI TRACE --")
        for rec in sink:
            print(f"  CLAIM: {rec['claim']!r}")
            print(f"    mode={rec['mode']}  total_facts_pool={rec['total_facts_pool']}  relevant_facts={len(rec['relevant'])}")
            for r in rec["relevant"]:
                print(f"      evidence={r['fact']!r}  label={r['label']}  entail_score={r['score']:.4f}")
            print(f"    VERDICT: {rec['verdict']}")
            print()

        all_records.extend(sink)

    # ── group A (metadata-citing) vs group B (plain) ────────────────────────
    group_a = [r for r in all_records if METADATA_RE.search(r["claim"])]
    group_b = [r for r in all_records if not METADATA_RE.search(r["claim"])]

    def _reject_rate(group: list[dict]) -> tuple[int, int, float]:
        n = len(group)
        n_rejected = sum(1 for r in group if r["verdict"] == "rejected")
        return n_rejected, n, (n_rejected / n if n else float("nan"))

    a_rej, a_n, a_rate = _reject_rate(group_a)
    b_rej, b_n, b_rate = _reject_rate(group_b)

    print("=" * 100)
    print("=== GROUP A (cites Frame/timestamp/action_score/persistence) vs GROUP B (plain) ===")
    print("=" * 100)
    print(f"  Group A: {a_rej}/{a_n} rejected  (reject rate = {a_rate:.2f})")
    print(f"  Group B: {b_rej}/{b_n} rejected  (reject rate = {b_rate:.2f})")
    print()

    def _print_examples(label: str, group: list[dict]) -> None:
        print(f"-- 3 EXAMPLE TRIPLES FROM {label} --")
        for rec in group[:3]:
            ev = rec["relevant"][0] if rec["relevant"] else None
            evidence_str = repr(ev["fact"]) if ev else "NONE (no topically-relevant fact)"
            nli_str = f"{ev['label']} (score={ev['score']:.4f})" if ev else "n/a"
            print(f"  claim:    {rec['claim']!r}")
            print(f"  evidence: {evidence_str}")
            print(f"  nli:      {nli_str}")
            print(f"  verdict:  {rec['verdict']}")
            print()

    _print_examples("GROUP A", group_a)
    _print_examples("GROUP B", group_b)

    print("=== READ ===")
    if a_n and b_n:
        if a_rate > b_rate + 0.15:
            print("  A rejects far more than B: evidence-decorated claims vs caption-only evidence --")
            print("  interface fix (ARIA's metadata-rich prose vs Cerberus-V's semantic-only fact pool).")
        elif abs(a_rate - b_rate) <= 0.15:
            print("  A ~= B: not a metadata-decoration-specific issue. If plain claims (B) still reject")
            print("  against apparently-matching captions, the evidence join or verdict aggregation is")
            print("  suspect, not the metadata phrasing.")
        else:
            print("  B rejects more than A -- does not match the stated 'evidence-decorated claims' hypothesis.")
    else:
        print("  N/A -- one group empty at this sample size.")


WIRING_TRACE_QUERIES = [
    "Is there a car in the parking lot?",
    "Is anyone loading a vehicle?",
    "Is anyone walking?",
]

# retrieved-frame dict keys worth auditing for presence/None (from
# iris.query._build_retrieved's node->dict mapping).
RETRIEVED_FIELDS = [
    "frame_idx", "timestamp", "luma_diff_energy", "action_score",
    "persistence_value", "is_peak", "clip_embedding", "luma_entropy",
    "caption", "pagerank_score", "last_retrieval_score",
]

FRAME_NUM_RE = re.compile(r"Frame\s+(\d+)")


def run_cerberus_wiring_trace() -> None:
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print()

    import iris.aria as aria

    for question in WIRING_TRACE_QUERIES:
        print("=" * 100)
        print(f"QUERY: {question!r}")
        print("=" * 100)

        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)

        # ── STEP 1: retrieved_frames field presence ─────────────────────────
        print("-- STEP 1: RETRIEVED FRAMES (field presence) --")
        retrieved_frame_idxs = [f["frame_idx"] for f in retrieved]
        for f in retrieved:
            present = {k: (f.get(k) is not None) for k in RETRIEVED_FIELDS}
            print(f"  frame_idx={f['frame_idx']}  timestamp={f.get('timestamp')}")
            for k in RETRIEVED_FIELDS:
                v = f.get(k)
                if k == "clip_embedding" and v is not None:
                    shown = f"<ndarray shape={getattr(v, 'shape', '?')}>"
                elif k == "caption" and v is not None:
                    shown = repr(v)
                else:
                    shown = repr(v)
                print(f"      {k:<20} present={present[k]!s:<5} value={shown}")
        print()

        # ── STEP 2: L1 Elysium cache_obj after wrapper_populate_cache ───────
        cache_obj = iris_query.wrapper_init_l1_cache(CFG)
        iris_query.wrapper_populate_cache(cache_obj, retrieved)

        print("-- STEP 2: L1 ELYSIUM CACHE CONTENTS (after wrapper_populate_cache) --")
        print(f"  cache type: {type(cache_obj).__name__}")
        cached_idxs = []
        use_elysium = hasattr(cache_obj, "frames") and callable(getattr(cache_obj, "frames"))
        if use_elysium:
            for cf in cache_obj.frames():
                cached_idxs.append(cf.frame_idx)
                print(f"  CachedFrame frame_idx={cf.frame_idx}  timestamp_sec={cf.timestamp_sec}")
                print(f"      action_score={cf.action_score}  persistence_value={cf.persistence_value}  is_peak={cf.is_peak}")
                print(f"      embedding is None: {cf.embedding is None}")
                print(f"      caption: {cf.caption!r}")
                print(f"      motion (FrameMotionDescriptor): divergence={cf.motion.divergence} curl={cf.motion.curl} "
                      f"jacobian_frobenius={cf.motion.jacobian_frobenius} hessian_max_eigenvalue={cf.motion.hessian_max_eigenvalue} "
                      f"motion_entropy={cf.motion.motion_entropy}")
                print(f"      pagerank={cf.pagerank}  query_similarity={cf.query_similarity}  admitted_at={cf.admitted_at}")
        else:
            print(f"  (legacy KnowledgeTriple cache -- not L1ElysiumCache; no per-frame field dump available)")
        print()

        missing_from_cache = set(retrieved_frame_idxs) - set(cached_idxs)
        print(f"  retrieved frame_idxs:     {sorted(retrieved_frame_idxs)}")
        print(f"  cached (L1) frame_idxs:   {sorted(cached_idxs)}")
        print(f"  MISSING from cache:       {sorted(missing_from_cache) if missing_from_cache else 'NONE -- all retrieved frames made it in'}")
        print()

        # ── STEP 3: context_text ARIA receives ──────────────────────────────
        context_text = cache_obj.as_context_text() if hasattr(cache_obj, "as_context_text") else "<N/A: legacy cache has no as_context_text()>"
        print("-- STEP 3: context_text (exact string sent to ARIA) --")
        print(context_text)
        print()

        raw_answer = aria.generate(prompt=question, context=context_text)
        claims = iris_query._split_claims(raw_answer)

        # ── STEP 4: per-claim NLI evidence trace ────────────────────────────
        gate = CerberusV()
        action_score_val = max((f.get("action_score", 0.0) for f in retrieved), default=0.5)
        mode = gate.get_verification_mode(action_score_val, CFG)
        gate._get_spacy()
        if mode in ("full_nli", "filtered_nli"):
            gate._get_nli_model()
        parsed_claims = gate._parse_claims(claims)
        facts = [entry.text for entry in cache_obj.set_facts.values()] if hasattr(cache_obj, "set_facts") else []

        print(f"-- STEP 4: MODE={mode}  action_score={action_score_val:.3f} --")
        print(f"-- NLI FACT POOL ({len(facts)} entries) --")
        for f in facts:
            print(f"  {f!r}")
        print()

        fact_frame_nums = set()
        for f in facts:
            m = FRAME_NUM_RE.search(f)
            if m:
                fact_frame_nums.add(int(m.group(1)))

        sink: list[dict] = []
        original_full_nli = CerberusV._full_nli
        CerberusV._full_nli = lambda self, c, f, m, a, _sink=sink: _traced_full_nli(self, c, f, m, a, _sink)
        try:
            if mode == "full_nli":
                gate._full_nli(parsed_claims, facts, mode, action_score_val)
            elif mode == "filtered_nli":
                high_conf = [c for c in parsed_claims if gate._confidence(c) >= 0.6]
                low_conf = [c for c in parsed_claims if gate._confidence(c) < 0.6]
                gate._full_nli(high_conf, facts, mode, action_score_val)
                for c in low_conf:
                    sink.append({"claim": c, "mode": "filtered_nli(low_conf_bypass)", "relevant": [], "total_facts_pool": len(facts), "verdict": "verified"})
            else:
                gate._ner_overlap(parsed_claims, facts, mode, action_score_val)
                for c in parsed_claims:
                    sink.append({"claim": c, "mode": "ner_only", "relevant": [], "total_facts_pool": len(facts), "verdict": "n/a (ner_only, no NLI pairs)"})
        finally:
            CerberusV._full_nli = original_full_nli

        print("-- PER-CLAIM NLI TRACE --")
        for rec in sink:
            print(f"  CLAIM: {rec['claim']!r}")
            print(f"    total_facts_pool={rec['total_facts_pool']}  relevant_facts={len(rec['relevant'])}")
            for r in rec["relevant"]:
                print(f"      evidence={r['fact']!r}  label={r['label']}  entail_score={r['score']:.4f}")
            print(f"    VERDICT: {rec['verdict']}")
            print()

        # ── per-query wiring check: do fact frame numbers match retrieved frames? ──
        retrieved_set = set(retrieved_frame_idxs)
        overlap = retrieved_set & fact_frame_nums
        print(f"  retrieved frame_idxs:        {sorted(retrieved_set)}")
        print(f"  fact-pool frame numbers:     {sorted(fact_frame_nums)}")
        print(f"  overlap (evidence == retrieved frames): {sorted(overlap)}  "
              f"({'MATCH' if overlap == retrieved_set == fact_frame_nums else 'PARTIAL/MISMATCH' if overlap else 'NO OVERLAP'})")
        print()

    print("=" * 100)
    print("=== WIRING AUDIT: THREE QUESTIONS ===")
    print("=" * 100)
    print("Q1. Does the evidence Cerberus verifies against contain the captions of the frames the")
    print("    scene_sparse retrieval actually returned?")
    print("    -> See 'overlap' lines above per query -- if retrieved==fact-pool frame numbers,")
    print("       answer is YES, the evidence pool is built from the same retrieved frames.")
    print()
    print("Q2. Are any fields the L1 cache/Cerberus expects (that the OLD pipeline set) missing or")
    print("    defaulted under the new ingest?")
    print("    -> See STEP 2 'motion (FrameMotionDescriptor)' lines -- if all five geometry fields")
    print("       print 0.0 for every frame, that is because iris.query._build_retrieved's node->dict")
    print("       mapping deliberately omits divergence/curl/jacobian_frobenius/hessian_max_eigenvalue/")
    print("       motion_entropy (documented in query.py as a known Phase-6 gap, not a bug introduced")
    print("       here) -- wrapper_populate_cache's .get(key, 0.0) then silently defaults them.")
    print("       These fields feed L1's keep_score (eviction) and CachedFrame.motion, NOT")
    print("       set_facts()/context_text() (captions), so they do not directly explain the NLI")
    print("       rejection pattern, but they ARE a real, currently-inert gap between the new")
    print("       scene_sparse retrieval path and what L1 Elysium was designed to receive.")
    print()
    print("Q3. Is Cerberus comparing claims against per-frame captions, a summary, or something")
    print("    empty/stale?")
    print("    -> See STEP 4 'NLI FACT POOL' dumps -- each entry is 'Frame {idx} at {ts}s: {caption}.'")
    print("       i.e. PER-FRAME captions (via CachedFrame.set_facts -> _frame_to_nli_fact), not a")
    print("       rolled-up summary and not empty/stale, AS LONG AS the overlap check above shows a")
    print("       frame-number match with the retrieved set for that query.")


def _load_index():
    npz = Path(str(CACHE_PATH) + ".npz")
    if npz.exists():
        print(f"Reusing cached index: {npz}")
        return iris_ingest.load_index(CACHE_PATH)
    if not VIDEO.exists():
        print(f"FATAL: no cache at {npz} and no video at {VIDEO}", file=sys.stderr)
        sys.exit(1)
    print(f"No cache found -- ingesting {VIDEO.name} fresh (long-tail build).")
    sys.stdout.flush()
    idx = iris_ingest.ingest(str(VIDEO), config=CFG)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    iris_ingest.save_index(idx, CACHE_PATH)
    return idx


def run_polarity_suite() -> None:
    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print()

    buckets = [("POSITIVE", POSITIVE_QUERIES), ("NEGATIVE/ABSENCE", NEGATIVE_QUERIES)]
    tally: dict[str, dict[str, int]] = {
        "POSITIVE": {"verified": 0, "rejected": 0, "unverifiable": 0, "n_queries": 0, "n_fully_verified": 0},
        "NEGATIVE/ABSENCE": {"verified": 0, "rejected": 0, "unverifiable": 0, "n_queries": 0, "n_fully_verified": 0},
    }

    for polarity, queries in buckets:
        for question in queries:
            print("=" * 100)
            print(f"[{polarity}] QUERY: {question!r}")
            print("=" * 100)

            result = iris_query.query(question, idx, CFG)

            print("-- RAW LLM ANSWER (pre-claim-split) --")
            print(result["raw_answer"])
            print()

            print("-- CLAIM VERIFICATION (Cerberus-V) --")
            print(f"  nli_mocked:          {result['nli_mocked']}")
            print(f"  verified (overall):  {result['verified']}")
            print(f"  verified_claims:     {result['verified_claims']}")
            print(f"  rejected_claims:     {result['rejected_claims']}")
            print(f"  unverifiable_claims: {result['unverifiable_claims']}")
            print()

            print("-- FINAL ANSWER --")
            print(result["answer"])
            print()

            tally[polarity]["verified"] += len(result["verified_claims"])
            tally[polarity]["rejected"] += len(result["rejected_claims"])
            tally[polarity]["unverifiable"] += len(result["unverifiable_claims"])
            tally[polarity]["n_queries"] += 1
            if result["verified"]:
                tally[polarity]["n_fully_verified"] += 1

    print("=" * 100)
    print("=== POLARITY TALLY (the finding) ===")
    print("=" * 100)
    hdr = f"{'polarity':<18} | {'n_queries':>9} | {'verified':>8} | {'rejected':>8} | {'unverifiable':>12} | {'queries_fully_verified':>22}"
    print(hdr)
    print("-" * len(hdr))
    for polarity, _ in buckets:
        t = tally[polarity]
        print(
            f"{polarity:<18} | {t['n_queries']:>9} | {t['verified']:>8} | {t['rejected']:>8} | "
            f"{t['unverifiable']:>12} | {t['n_fully_verified']:>22}"
        )
    print()

    pos = tally["POSITIVE"]
    neg = tally["NEGATIVE/ABSENCE"]
    print("=== READ ===")
    pos_pass_rate = pos["verified"] / max(1, pos["verified"] + pos["rejected"])
    neg_pass_rate = neg["verified"] / max(1, neg["verified"] + neg["rejected"])
    print(f"  positive verified/(verified+rejected) = {pos_pass_rate:.2f}")
    print(f"  negative verified/(verified+rejected) = {neg_pass_rate:.2f}")
    if pos_pass_rate > 0.5 and neg_pass_rate < 0.5:
        print("  Positives largely pass, negatives largely reject: structural absence-claim-vs-")
        print("  positive-evidence failure. Scoping a routing fix (not a broad Cerberus recalibration).")
    elif pos_pass_rate < 0.5 and neg_pass_rate < 0.5:
        print("  Positives ALSO reject: Cerberus is broadly miscalibrated, not an absence-claim-")
        print("  specific issue. Bigger problem than a routing fix.")
    else:
        print("  Pattern does not cleanly match either hypothesis as stated -- see raw tally above.")


def main() -> None:
    if "--polarity-suite" in sys.argv:
        run_polarity_suite()
    elif "--nli-trace" in sys.argv:
        run_nli_trace()
    elif "--cerberus-wiring-trace" in sys.argv:
        run_cerberus_wiring_trace()
    else:
        print("Usage: python scripts/demo_cctv_query.py --polarity-suite | --nli-trace | --cerberus-wiring-trace")
        sys.exit(1)


if __name__ == "__main__":
    main()
