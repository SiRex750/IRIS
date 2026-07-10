"""scripts/diag_v2_scoping_separation.py — DIAGNOSTIC ONLY, no mechanism change.

Measures whether embedding-similarity scoping (CLIP text-text cosine sim
between a claim and a fact's caption) could replace Cerberus-V's lemma-overlap
topical-relevance filter (iris/cerberus_v.py, "Fix 10" block in _full_nli).

Why this question exists: every NLI fact string is built as
"Frame {idx} at {ts}s: {caption}." (iris/l1_elysium.py:_frame_to_nli_fact).
The lemma "frame" appears in every fact by construction, so the lemma-overlap
predicate passes ~100% of the fact pool for every claim regardless of actual
topical relevance. Unscoped majority-vote NLI then lets off-topic
contradiction votes swamp correct entailments. This script does NOT implement
a fix — it captures real (claim, fact) NLI outcomes from the live pipeline and
replays alternative scoping rules offline against the same labels to see if a
usable separation exists.

Zero edits to iris/. Imports only from scripts/demo_cctv_query.py (VIDEO,
CACHE_PATH, CFG, query lists, _load_index) and iris/* (read-only use).

PHASE 1 (models loaded, live pipeline, writes capture artifacts):
    python scripts/diag_v2_scoping_separation.py --capture

PHASE 2 (pure offline analysis of the two capture artifacts, no model loads):
    python scripts/diag_v2_scoping_separation.py --analyze

VERIFY:
    python scripts/diag_v2_scoping_separation.py --capture 2>&1 | tee diag_v2_capture.log
    python scripts/diag_v2_scoping_separation.py --analyze 2>&1 | tee diag_v2_output.log
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.demo_cctv_query import (
    CACHE_PATH,
    CFG,
    NEGATIVE_QUERIES,
    NLI_TRACE_QUERIES,
    POSITIVE_QUERIES,
    VIDEO,
    _load_index,
)

CAPTURE_JSONL = REPO / "diag_v2_capture.jsonl"
EMBEDS_NPZ = REPO / "diag_v2_embeds.npz"

# Verbatim format from iris/l1_elysium.py:_frame_to_nli_fact:
#   f"Frame {frame.frame_idx} at {frame.timestamp_sec:.2f}s: {semantic_caption}."
FACT_PREFIX_RE = re.compile(r"^Frame \d+ at \d+\.\d+s: ")

FABRICATED_CLAIMS = [
    "A dog runs across the parking lot.",
    "Two people are fighting near the building entrance.",
    "A fire truck is parked in the lot.",
    "A person is holding an umbrella.",
    "A boat is visible in the scene.",
    "A person climbs onto the roof of a car.",
]

TAU_SWEEP = [round(0.50 + 0.05 * i, 2) for i in range(9)]  # 0.50 .. 0.90
K_SWEEP = [1, 3, 5]


def _caption_only(fact_text: str) -> str:
    return FACT_PREFIX_RE.sub("", fact_text)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — capture
# ─────────────────────────────────────────────────────────────────────────────

def _capture_full_nli(self, claims, facts, mode, action_score, records, query, claim_type="real"):
    """Verbatim-logic clone of CerberusV._full_nli (iris/cerberus_v.py) with
    ALL (claim, fact) pairs captured (not just topically-overlapping ones, so
    Phase 2 can measure separation across the full label distribution). Same
    thresholds, same aggregation rule as the real method — copied unchanged."""
    if not claims:
        return {"verified": [], "rejected": [], "unverifiable": [], "mode": mode, "action_score": action_score}
    if not facts:
        for c in claims:
            records.append({
                "query": query, "claim": c, "claim_type": claim_type,
                "fact_text": None, "caption_only": None,
                "label": None, "entailment_score": None,
                "has_topical_overlap": None, "live_verdict_for_claim": "unverifiable",
            })
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
        relevant = [(label, score) for label, score, has_overlap in all_results if has_overlap]

        if not relevant:
            unverifiable.append(claim)
            verdict = "unverifiable"
        else:
            entailed_count = sum(1 for label, _ in relevant if label == "entailment")
            contradicted_count = sum(1 for label, _ in relevant if label == "contradiction")
            if contradicted_count > entailed_count:
                rejected.append(claim)
                verdict = "rejected"
            elif entailed_count > 0:
                verified.append(claim)
                verdict = "verified"
            else:
                unverifiable.append(claim)
                verdict = "unverifiable"

        for fact in facts:
            label, score, has_overlap = pair_results[(claim, fact)]
            records.append({
                "query": query, "claim": claim, "claim_type": claim_type,
                "fact_text": fact, "caption_only": _caption_only(fact),
                "label": label, "entailment_score": score,
                "has_topical_overlap": has_overlap, "live_verdict_for_claim": verdict,
            })

    return {"verified": verified, "rejected": rejected, "unverifiable": unverifiable, "mode": mode, "action_score": action_score}


def run_capture() -> None:
    from iris.cerberus_v import CerberusV

    npz = Path(str(CACHE_PATH) + ".npz")
    if not npz.exists():
        print(f"FATAL: no index cache at {npz}", file=sys.stderr)
        sys.exit(1)
    if not VIDEO.exists():
        print(f"FATAL: no video at {VIDEO} (needed for lazy captioning of any cache miss)", file=sys.stderr)
        sys.exit(1)

    idx = _load_index()
    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")

    queries = []
    for q in NLI_TRACE_QUERIES + POSITIVE_QUERIES + NEGATIVE_QUERIES:
        if q not in queries:
            queries.append(q)
    print(f"queries (deduped): {len(queries)}")

    import iris.aria as aria
    import iris.query as iris_query

    records: list[dict] = []
    original_full_nli = CerberusV._full_nli

    for question in queries:
        print(f"CAPTURE query: {question!r}")
        query_embedding = iris_query._embed_query(question, CFG)
        retrieved = iris_query._build_retrieved(idx, query_embedding, CFG)
        iris_query._ensure_captions(idx, retrieved)
        cache_obj = iris_query.wrapper_init_l1_cache(CFG)
        iris_query.wrapper_populate_cache(cache_obj, retrieved)
        context_text = cache_obj.as_context_text()

        raw_answer = aria.generate(prompt=question, context=context_text)
        claims = iris_query._split_claims(raw_answer)

        gate = CerberusV()
        action_score_val = max((f.get("action_score", 0.0) for f in retrieved), default=0.5)
        mode = gate.get_verification_mode(action_score_val, CFG)
        gate._get_spacy()
        if mode in ("full_nli", "filtered_nli"):
            gate._get_nli_model()
        parsed_claims = gate._parse_claims(claims)
        facts = [entry.text for entry in cache_obj.set_facts.values()]

        CerberusV._full_nli = lambda self, c, f, m, a, _q=question: _capture_full_nli(self, c, f, m, a, records, _q)
        try:
            if mode == "full_nli":
                gate._full_nli(parsed_claims, facts, mode, action_score_val)
            elif mode == "filtered_nli":
                high_conf = [c for c in parsed_claims if gate._confidence(c) >= 0.6]
                low_conf = [c for c in parsed_claims if gate._confidence(c) < 0.6]
                gate._full_nli(high_conf, facts, mode, action_score_val)
                for c in low_conf:
                    records.append({
                        "query": question, "claim": c, "claim_type": "real",
                        "fact_text": None, "caption_only": None,
                        "label": None, "entailment_score": None,
                        "has_topical_overlap": None,
                        "live_verdict_for_claim": "verified",  # filtered_nli low-conf bypass: passes unverified
                    })
            else:  # ner_only -- no NLI pairs, not comparable to full_nli scoping question
                pass
        finally:
            CerberusV._full_nli = original_full_nli

        # ── Fabricated claims: same fact pool, same NLI path (full_nli forced) ──
        gate2 = CerberusV()
        gate2._get_spacy()
        gate2._get_nli_model()
        CerberusV._full_nli = lambda self, c, f, m, a, _q=question: _capture_full_nli(self, c, f, m, a, records, _q, claim_type="fabricated")
        try:
            gate2._full_nli(FABRICATED_CLAIMS, facts, "full_nli", action_score_val)
        finally:
            CerberusV._full_nli = original_full_nli

    with open(CAPTURE_JSONL, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} records to {CAPTURE_JSONL}")

    # ── CLIP text embeddings: every distinct claim, caption_only, fact_text ──
    import clip
    import torch
    from iris._clip import get_clip_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_clip_model()
    if model is None:
        print("FATAL: CLIP model failed to load", file=sys.stderr)
        sys.exit(1)

    distinct_claims = sorted({r["claim"] for r in records if r["claim"] is not None})
    distinct_captions = sorted({r["caption_only"] for r in records if r["caption_only"] is not None})
    distinct_facts = sorted({r["fact_text"] for r in records if r["fact_text"] is not None})

    def _embed_strings(strings: list[str]) -> np.ndarray:
        if not strings:
            return np.zeros((0, 512), dtype=np.float32)
        vecs = []
        batch_size = 32
        for i in range(0, len(strings), batch_size):
            batch = strings[i:i + batch_size]
            # clip.tokenize exactly as iris/query.py:154 -- truncate=True added
            # only because captions can exceed the 77-token context length,
            # which clip.tokenize would otherwise hard-error on; truncation
            # does not change the encode_text computation itself.
            text_input = clip.tokenize(batch, truncate=True).to(device)
            with torch.no_grad():
                qf = model.encode_text(text_input)
                qf /= qf.norm(dim=-1, keepdim=True)
            vecs.append(qf.cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)

    claim_embeds = _embed_strings(distinct_claims)
    caption_embeds = _embed_strings(distinct_captions)
    fact_embeds = _embed_strings(distinct_facts)

    np.savez(
        EMBEDS_NPZ,
        claim_strings=np.array(distinct_claims, dtype=object),
        claim_embeds=claim_embeds,
        caption_strings=np.array(distinct_captions, dtype=object),
        caption_embeds=caption_embeds,
        fact_strings=np.array(distinct_facts, dtype=object),
        fact_embeds=fact_embeds,
    )
    print(f"wrote embeddings ({len(distinct_claims)} claims, {len(distinct_captions)} captions, "
          f"{len(distinct_facts)} facts) to {EMBEDS_NPZ}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — analyze (pure offline, no model loads)
# ─────────────────────────────────────────────────────────────────────────────

def _load_capture() -> list[dict]:
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: capture artifact missing: {CAPTURE_JSONL} (run --capture first)", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(CAPTURE_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_embeds() -> dict:
    if not EMBEDS_NPZ.exists():
        print(f"FATAL: embeds artifact missing: {EMBEDS_NPZ} (run --capture first)", file=sys.stderr)
        sys.exit(1)
    data = np.load(EMBEDS_NPZ, allow_pickle=True)
    return {
        "claim_strings": list(data["claim_strings"]),
        "claim_embeds": data["claim_embeds"],
        "caption_strings": list(data["caption_strings"]),
        "caption_embeds": data["caption_embeds"],
        "fact_strings": list(data["fact_strings"]),
        "fact_embeds": data["fact_embeds"],
    }


def _group_by_claim(records: list[dict]) -> dict[tuple, list[dict]]:
    """Group pair-records by (query, claim, claim_type) -- one claim can
    recur verbatim across queries/claim_types, so key on the triple."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["fact_text"] is None:
            continue  # empty-fact-pool or filtered_nli low-conf bypass rows
        key = (r["query"], r["claim"], r["claim_type"])
        groups.setdefault(key, []).append(r)
    return groups


def section_a_lemma_noop_check(records: list[dict]) -> None:
    print("=" * 100)
    print("(A) LEMMA NO-OP CHECK")
    print("=" * 100)
    print("has_topical_overlap below is read verbatim from the capture -- it is the actual")
    print("boolean the live _full_nli call computed via cerberus_v.py's lemma/entity overlap")
    print("predicate (spaCy lemmas, is_stop, is_alpha, len>2, plus entity sets). Recomputing")
    print("it here would require a spaCy load, which --analyze must not do; the captured value")
    print("already IS that computation, unmodified.")
    print()

    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    total_pass = 0
    total_pairs = 0
    per_claim_rates = []
    for key, rows in real_groups.items():
        n = len(rows)
        n_pass = sum(1 for r in rows if r["has_topical_overlap"])
        total_pass += n_pass
        total_pairs += n
        per_claim_rates.append(n_pass / n if n else float("nan"))

    overall_rate = total_pass / total_pairs if total_pairs else float("nan")
    mean_per_claim_rate = float(np.mean(per_claim_rates)) if per_claim_rates else float("nan")
    print(f"  claims (real) with >=1 fact: {len(real_groups)}")
    print(f"  total (claim,fact) pairs:    {total_pairs}")
    print(f"  overall pool pass rate:      {overall_rate:.4f}  ({total_pass}/{total_pairs})")
    print(f"  mean per-claim pass rate:    {mean_per_claim_rate:.4f}")
    n_full_pass = sum(1 for rate in per_claim_rates if rate == 1.0)
    print(f"  claims with 100% pool pass:  {n_full_pass}/{len(per_claim_rates)}")
    print()


def section_b_parity_self_check(records: list[dict]) -> bool:
    print("=" * 100)
    print("(B) PARITY SELF-CHECK (hard invariant)")
    print("=" * 100)

    groups = _group_by_claim(records)
    diffs = []
    for key, rows in groups.items():
        entailed = sum(1 for r in rows if r["has_topical_overlap"] and r["label"] == "entailment")
        contradicted = sum(1 for r in rows if r["has_topical_overlap"] and r["label"] == "contradiction")
        n_relevant = sum(1 for r in rows if r["has_topical_overlap"])

        if n_relevant == 0:
            replayed = "unverifiable"
        elif contradicted > entailed:
            replayed = "rejected"
        elif entailed > 0:
            replayed = "verified"
        else:
            replayed = "unverifiable"

        captured = rows[0]["live_verdict_for_claim"]
        if replayed != captured:
            diffs.append((key, captured, replayed))

    if diffs:
        print(f"  PARITY FAILED: {len(diffs)} claim(s) disagree between captured and replayed verdict.")
        for key, captured, replayed in diffs[:20]:
            print(f"    query={key[0]!r} claim={key[1]!r} claim_type={key[2]} "
                  f"captured={captured} replayed={replayed}")
        print("  Capture is broken -- stopping before (C)-(E).")
        return False

    print(f"  PARITY OK: {len(groups)} claim groups, replayed verdict == captured live_verdict for all.")
    print()
    return True


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def section_c_separation(records: list[dict], embeds: dict) -> dict:
    print("=" * 100)
    print("(C) SEPARATION: cosine sim (claim x fact) split by NLI label")
    print("=" * 100)

    caption_idx = {s: i for i, s in enumerate(embeds["caption_strings"])}
    fact_idx = {s: i for i, s in enumerate(embeds["fact_strings"])}
    claim_idx = {s: i for i, s in enumerate(embeds["claim_strings"])}

    caption_only_sims: dict[str, list[float]] = {"entailment": [], "contradiction": [], "neutral": []}
    full_string_sims: dict[str, list[float]] = {"entailment": [], "contradiction": [], "neutral": []}
    pair_sims: dict[tuple, float] = {}  # (claim, caption_only) -> sim, for (D)/(E) reuse

    for r in records:
        if r["fact_text"] is None or r["claim"] not in claim_idx:
            continue
        c_i = claim_idx[r["claim"]]
        cap_i = caption_idx.get(r["caption_only"])
        fact_i = fact_idx.get(r["fact_text"])
        if cap_i is not None:
            sim_cap = float(embeds["claim_embeds"][c_i] @ embeds["caption_embeds"][cap_i])
            caption_only_sims[r["label"]].append(sim_cap)
            pair_sims[(r["query"], r["claim"], r["claim_type"], r["fact_text"])] = sim_cap
        if fact_i is not None:
            sim_full = float(embeds["claim_embeds"][c_i] @ embeds["fact_embeds"][fact_i])
            full_string_sims[r["label"]].append(sim_full)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {
            "n": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
            "q25": float(np.percentile(arr, 25)), "median": float(np.percentile(arr, 50)),
            "q75": float(np.percentile(arr, 75)),
        }

    def _print_table(title: str, sims: dict[str, list[float]]) -> None:
        print(f"  -- {title} --")
        hdr = f"    {'label':<14} | {'n':>5} | {'mean':>7} | {'std':>7} | {'q25':>7} | {'median':>7} | {'q75':>7}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for label in ("entailment", "contradiction", "neutral"):
            s = _stats(sims[label])
            if s["n"] == 0:
                print(f"    {label:<14} | {0:>5} | {'n/a':>7} | {'n/a':>7} | {'n/a':>7} | {'n/a':>7} | {'n/a':>7}")
            else:
                print(f"    {label:<14} | {s['n']:>5} | {s['mean']:>7.4f} | {s['std']:>7.4f} | "
                      f"{s['q25']:>7.4f} | {s['median']:>7.4f} | {s['q75']:>7.4f}")
        all_vals = [v for vs in sims.values() for v in vs]
        if all_vals:
            arr = np.array(all_vals)
            print(f"    global: min={arr.min():.4f}  median={np.percentile(arr, 50):.4f}  max={arr.max():.4f}")
        print()

    _print_table("caption_only embeddings (claim vs stripped caption)", caption_only_sims)
    _print_table("full fact_text embeddings (claim vs 'Frame N at Ts: caption.')", full_string_sims)

    ent_mean = _stats(caption_only_sims["entailment"]).get("mean")
    other_vals = caption_only_sims["contradiction"] + caption_only_sims["neutral"]
    other_mean = float(np.mean(other_vals)) if other_vals else float("nan")
    print("  -- READ --")
    if ent_mean is not None and not np.isnan(other_mean):
        gap = ent_mean - other_mean
        print(f"  entailment mean ({ent_mean:.4f}) minus contradiction+neutral mean ({other_mean:.4f}) = {gap:+.4f}")
        if gap > 0.03:
            print("  Entailment pairs sit measurably above the rest in CLIP text-text space.")
        else:
            print("  No clear separation -- entailment does not sit meaningfully above contradiction/neutral.")
    print()

    return {"pair_sims": pair_sims}


def _scoped_verdict(claim_rows: list[dict], pair_sims: dict, key_prefix: tuple, tau: float, k: int) -> str:
    """Scoped rule: pool = facts with sim(claim, caption_only) >= tau, keep top-k by sim.
    Per-pair labels come from capture UNCHANGED. Aggregation rule matches the current one."""
    scored = []
    for r in claim_rows:
        sim = pair_sims.get((r["query"], r["claim"], r["claim_type"], r["fact_text"]))
        if sim is None or sim < tau:
            continue
        scored.append((sim, r["label"]))
    scored.sort(key=lambda x: -x[0])
    pool = scored[:k]

    if not pool:
        return "unverifiable"
    entailed = sum(1 for _, label in pool if label == "entailment")
    contradicted = sum(1 for _, label in pool if label == "contradiction")
    if contradicted > entailed:
        return "rejected"
    elif entailed > 0:
        return "verified"
    return "unverifiable"


def section_d_verdict_replay_sweep(records: list[dict], sep_state: dict) -> None:
    print("=" * 100)
    print("(D) VERDICT REPLAY SWEEP (real claims): scoped rule vs current rule")
    print("=" * 100)

    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}
    pair_sims = sep_state["pair_sims"]

    hdr = f"  {'tau':>5} | {'k':>2} | {'verified':>8} | {'rejected':>8} | {'unverifiable':>12} | {'flips_vs_current':>16}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tau in TAU_SWEEP:
        for k in K_SWEEP:
            counts = {"verified": 0, "rejected": 0, "unverifiable": 0}
            flips = 0
            for key, rows in real_groups.items():
                current = rows[0]["live_verdict_for_claim"]
                scoped = _scoped_verdict(rows, pair_sims, key, tau, k)
                counts[scoped] += 1
                if scoped != current:
                    flips += 1
            print(f"  {tau:>5.2f} | {k:>2} | {counts['verified']:>8} | {counts['rejected']:>8} | "
                  f"{counts['unverifiable']:>12} | {flips:>16}")
    print()


def section_e_fabricated_accept_rate(records: list[dict], sep_state: dict) -> None:
    print("=" * 100)
    print("(E) FABRICATED-CLAIM ACCEPT RATE: scoped rule vs current rule baseline")
    print("=" * 100)

    groups = _group_by_claim(records)
    fab_groups = {k: v for k, v in groups.items() if k[2] == "fabricated"}
    pair_sims = sep_state["pair_sims"]

    if not fab_groups:
        print("  No fabricated-claim records in capture.")
        return

    n_fab = len(fab_groups)
    n_current_accept = sum(1 for rows in fab_groups.values() if rows[0]["live_verdict_for_claim"] == "verified")
    print(f"  n fabricated claim instances: {n_fab}")
    print(f"  CURRENT rule accept (verified) rate: {n_current_accept}/{n_fab} = {n_current_accept / n_fab:.4f}")
    print()

    hdr = f"  {'tau':>5} | {'k':>2} | {'accepted':>8} | {'accept_rate':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tau in TAU_SWEEP:
        for k in K_SWEEP:
            n_accept = 0
            for key, rows in fab_groups.items():
                scoped = _scoped_verdict(rows, pair_sims, key, tau, k)
                if scoped == "verified":
                    n_accept += 1
            print(f"  {tau:>5.2f} | {k:>2} | {n_accept:>8} | {n_accept / n_fab:>11.4f}")
    print()

    # ── per-claim eyeball table for one representative query, tau=median-of-C, k=3 ──
    all_sims = [v for v in pair_sims.values()]
    tau_median = float(np.percentile(np.array(all_sims), 50)) if all_sims else 0.5
    k_fixed = 3

    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}
    rep_query = None
    for key in real_groups:
        rep_query = key[0]
        break

    print(f"  -- EYEBALL TABLE: query={rep_query!r}  tau={tau_median:.4f} (median of C)  k={k_fixed} --")
    hdr2 = f"    {'claim':<45} | {'top1_fact':<45} | {'sim':>6} | {'label':<12} | {'current':<12} | {'scoped':<12}"
    print(hdr2)
    print("    " + "-" * (len(hdr2) - 4))
    for key, rows in real_groups.items():
        if key[0] != rep_query:
            continue
        scored = sorted(
            ((pair_sims.get((r["query"], r["claim"], r["claim_type"], r["fact_text"]), -1.0), r) for r in rows),
            key=lambda x: -x[0],
        )
        top_sim, top_row = scored[0]
        current = rows[0]["live_verdict_for_claim"]
        scoped = _scoped_verdict(rows, pair_sims, key, tau_median, k_fixed)
        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        fact_disp = (top_row["caption_only"][:42] + "...") if len(top_row["caption_only"]) > 45 else top_row["caption_only"]
        print(f"    {claim_disp:<45} | {fact_disp:<45} | {top_sim:>6.4f} | {top_row['label']:<12} | "
              f"{current:<12} | {scoped:<12}")
    print()


def run_analyze() -> None:
    records = _load_capture()
    embeds = _load_embeds()

    section_a_lemma_noop_check(records)
    parity_ok = section_b_parity_self_check(records)
    if not parity_ok:
        sys.exit(1)
    sep_state = section_c_separation(records, embeds)
    section_d_verdict_replay_sweep(records, sep_state)
    section_e_fabricated_accept_rate(records, sep_state)


def main() -> None:
    if "--capture" in sys.argv:
        run_capture()
    elif "--analyze" in sys.argv:
        run_analyze()
    else:
        print("Usage: python scripts/diag_v2_scoping_separation.py --capture | --analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
