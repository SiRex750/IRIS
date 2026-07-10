"""scripts/diag_v3_visual_scoping.py — DIAGNOSTIC ONLY, no mechanism change.

Extends diag_v2 (scripts/diag_v2_scoping_separation.py) with a VISUAL-scoping
arm. diag_v2 established (raw logs are ground truth):
  (A) lemma-overlap filter pool pass rate = 0.9964 -- confirmed no-op.
  (B) replay parity exact.
  (C) CLIP TEXT-TEXT similarity (claim vs caption) does NOT separate
      entailment pairs from contradiction/neutral pairs (means 0.6984 vs
      0.7090) -- text-text scoping is dead.

New hypothesis this script tests: claim-TEXT -> frame-VISUAL CLIP similarity
(CLIP's native trained cross-modal operation, and the same signal the
pipeline's own retrieval already uses) separates where text-text did not.
The evidence objects Cerberus-V is really reasoning about are frames; the
caption is only the NLI premise text, not the ground truth.

Reuses diag_v2's capture artifact (diag_v2_capture.jsonl) as-is. Does NOT
re-run ARIA or NLI -- only adds visual (frame) CLIP embeddings and re-uses
already-embedded claim text (or embeds any that are missing, same
clip.tokenize -> model.encode_text path as iris/query.py:154).

Zero edits to iris/, tests/, demo_cctv_query.py, diag_v2_scoping_separation.py
-- imports only.

PHASE 1 (loads CLIP + cached index, no ARIA/NLI re-run):
    python scripts/diag_v3_visual_scoping.py --capture-addendum

PHASE 2 (pure offline analysis, no model loads):
    python scripts/diag_v3_visual_scoping.py --analyze

--ceiling mode (pure offline over diag_v2_capture.jsonl, no model loads):
tests the hard upper bound on what any per-pair scoping rule (text or
visual) can recover, and classifies the rejection pattern by claim type.
    python scripts/diag_v3_visual_scoping.py --ceiling

--binding-ceiling mode: a THIRD, different hypothesis from scoping (v2/v3)
and the ceiling analysis. Instead of filtering the fact pool by similarity,
it resolves each claim's EXPLICIT frame/timestamp citations (the claim
already tells you which frame it's about), strips the numeric metadata
clause the claim wraps around its visual assertion, and reruns NLI on just
(kernel claim, cited frame's caption) -- i.e. citation-bound verification
rather than similarity-bound. MAY load the DeBERTa NLI model (reusing
CerberusV's loader, same as demo_cctv_query.py) to score these new
(kernel, caption) pairs -- they don't exist in the v2 capture, which only
scored (original claim, full fact_text) pairs.
    python scripts/diag_v3_visual_scoping.py --binding-ceiling

--kernel-normalize mode: same structure as --binding-ceiling (reuses its
citation parse and NLI replay), but after metadata-clause stripping ALSO
normalizes the kernel to plain visual language (drops Frame/timestamp
references, leading hedge phrases, and rewrites a Frame-adjacent "shows"/
"depicts" to "there is") before running NLI. Tests whether --binding-ceiling's
zero-entailment result was a phrasing artifact rather than a genuine NLI
judgment.
    python scripts/diag_v3_visual_scoping.py --kernel-normalize

VERIFY:
    python scripts/diag_v3_visual_scoping.py --capture-addendum 2>&1 | tee diag_v3_capture.log
    python scripts/diag_v3_visual_scoping.py --analyze 2>&1 | tee diag_v3_output.log
    python scripts/diag_v3_visual_scoping.py --ceiling 2>&1 | tee diag_v3_ceiling.log
    python scripts/diag_v3_visual_scoping.py --binding-ceiling 2>&1 | tee diag_v3_binding.log
    python scripts/diag_v3_visual_scoping.py --kernel-normalize 2>&1 | tee diag_v3_normalize.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.demo_cctv_query import CACHE_PATH, CFG, VIDEO, _load_index
from scripts.diag_v2_scoping_separation import (
    CAPTURE_JSONL,
    FACT_PREFIX_RE,
    K_SWEEP,
    _group_by_claim,
    _load_capture,
)

EMBEDS_NPZ = REPO / "diag_v3_embeds.npz"

# Verbatim format from iris/l1_elysium.py:_frame_to_nli_fact:
#   f"Frame {frame.frame_idx} at {frame.timestamp_sec:.2f}s: {semantic_caption}."
FRAME_IDX_RE = re.compile(r"^Frame (\d+) at [\d.]+s:")

QUANTILE_PCTS = list(range(5, 100, 5))  # 5, 10, ..., 95
SWEEP_PCTS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]


def _frame_idx_of(fact_text: str) -> int:
    m = FRAME_IDX_RE.match(fact_text)
    if not m:
        raise ValueError(f"fact_text does not match expected 'Frame N at Ts:' format: {fact_text!r}")
    return int(m.group(1))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — capture-addendum
# ─────────────────────────────────────────────────────────────────────────────

def run_capture_addendum() -> None:
    npz = Path(str(CACHE_PATH) + ".npz")
    if not npz.exists():
        print(f"FATAL: no index cache at {npz}", file=sys.stderr)
        sys.exit(1)
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: diag_v2 capture artifact missing: {CAPTURE_JSONL} (run diag_v2 --capture first)", file=sys.stderr)
        sys.exit(1)

    records = _load_capture()

    distinct_facts = sorted({r["fact_text"] for r in records if r["fact_text"] is not None})
    fact_to_frame_idx: dict[str, int] = {}
    for fact in distinct_facts:
        fact_to_frame_idx[fact] = _frame_idx_of(fact)
    distinct_frame_idxs = sorted(set(fact_to_frame_idx.values()))
    print(f"distinct fact_text strings: {len(distinct_facts)}")
    print(f"distinct frame_idx values:  {len(distinct_frame_idxs)}")

    idx = _load_index()
    frame_map = {fr.frame_idx: fr for fr in idx.frames}

    missing = [fi for fi in distinct_frame_idxs if frame_map.get(fi) is None or frame_map[fi].clip_embedding is None]
    if missing:
        print(f"FATAL: {len(missing)} frame_idx have no stored visual (CLIP) embedding: {missing}", file=sys.stderr)
        sys.exit(1)

    visual_embeds = np.stack([frame_map[fi].clip_embedding.astype(np.float32) for fi in distinct_frame_idxs], axis=0)
    print(f"resolved visual embeddings for all {len(distinct_frame_idxs)} frame_idx values")

    # ── claim text embeddings, including fabricated claims ──────────────────
    distinct_claims = sorted({r["claim"] for r in records if r["claim"] is not None})
    print(f"distinct claims (all claim_types): {len(distinct_claims)}")

    import clip
    import torch
    from iris._clip import get_clip_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_clip_model()
    if model is None:
        print("FATAL: CLIP model failed to load", file=sys.stderr)
        sys.exit(1)

    def _embed_strings(strings: list[str]) -> np.ndarray:
        if not strings:
            return np.zeros((0, 512), dtype=np.float32)
        vecs = []
        batch_size = 32
        for i in range(0, len(strings), batch_size):
            batch = strings[i:i + batch_size]
            # clip.tokenize exactly as iris/query.py:154 -- truncate=True added
            # only because some strings can exceed the 77-token context
            # length, which clip.tokenize would otherwise hard-error on;
            # truncation does not change the encode_text computation itself.
            text_input = clip.tokenize(batch, truncate=True).to(device)
            with torch.no_grad():
                qf = model.encode_text(text_input)
                qf /= qf.norm(dim=-1, keepdim=True)
            vecs.append(qf.cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)

    claim_embeds = _embed_strings(distinct_claims)

    np.savez(
        EMBEDS_NPZ,
        frame_idxs=np.array(distinct_frame_idxs, dtype=np.int64),
        frame_visual_embeds=visual_embeds,
        claim_strings=np.array(distinct_claims, dtype=object),
        claim_embeds=claim_embeds,
        fact_strings=np.array(distinct_facts, dtype=object),
        fact_frame_idxs=np.array([fact_to_frame_idx[f] for f in distinct_facts], dtype=np.int64),
    )
    print(f"wrote {EMBEDS_NPZ}: {len(distinct_frame_idxs)} frame visual embeds, "
          f"{len(distinct_claims)} claim text embeds, {len(distinct_facts)} fact->frame_idx entries")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — analyze (pure offline, no model loads)
# ─────────────────────────────────────────────────────────────────────────────

def _load_v3_embeds() -> dict:
    if not EMBEDS_NPZ.exists():
        print(f"FATAL: embeds artifact missing: {EMBEDS_NPZ} (run --capture-addendum first)", file=sys.stderr)
        sys.exit(1)
    data = np.load(EMBEDS_NPZ, allow_pickle=True)
    return {
        "frame_idxs": list(data["frame_idxs"]),
        "frame_visual_embeds": data["frame_visual_embeds"],
        "claim_strings": list(data["claim_strings"]),
        "claim_embeds": data["claim_embeds"],
        "fact_strings": list(data["fact_strings"]),
        "fact_frame_idxs": list(data["fact_frame_idxs"]),
    }


def section_bprime_parity_self_check(records: list[dict]) -> bool:
    print("=" * 100)
    print("(B') PARITY SELF-CHECK (identical to diag_v2 section B)")
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
        return False

    print(f"  PARITY OK: {len(groups)} claim groups, replayed verdict == captured live_verdict for all.")
    print()
    return True


def _compute_visual_pair_sims(records: list[dict], embeds: dict) -> dict:
    """Cosine sim(claim_text_emb, frame_visual_emb) for every captured
    (claim, fact) pair, plus the label-grouped values and global quantiles.
    Pure computation, no printing -- shared by (C') and --ceiling."""
    claim_idx = {s: i for i, s in enumerate(embeds["claim_strings"])}
    fact_frame_idx_map = {s: fi for s, fi in zip(embeds["fact_strings"], embeds["fact_frame_idxs"])}
    frame_row = {fi: i for i, fi in enumerate(embeds["frame_idxs"])}

    sims_by_label: dict[str, list[float]] = {"entailment": [], "contradiction": [], "neutral": []}
    pair_sims: dict[tuple, float] = {}  # (query, claim, claim_type, fact_text) -> sim

    for r in records:
        if r["fact_text"] is None or r["claim"] not in claim_idx:
            continue
        c_i = claim_idx[r["claim"]]
        frame_idx = fact_frame_idx_map.get(r["fact_text"])
        if frame_idx is None:
            continue
        f_i = frame_row[frame_idx]
        sim = float(embeds["claim_embeds"][c_i] @ embeds["frame_visual_embeds"][f_i])
        pair_sims[(r["query"], r["claim"], r["claim_type"], r["fact_text"])] = sim
        sims_by_label[r["label"]].append(sim)

    all_vals = [v for vs in sims_by_label.values() for v in vs]
    arr_all = np.array(all_vals)
    quantile_taus = {p: float(np.percentile(arr_all, p)) for p in QUANTILE_PCTS}

    return {"pair_sims": pair_sims, "sims_by_label": sims_by_label, "quantile_taus": quantile_taus}


def section_cprime_visual_separation(records: list[dict], embeds: dict) -> dict:
    print("=" * 100)
    print("(C') SEPARATION, VISUAL: cosine sim(claim_text_emb, frame_visual_emb)")
    print("=" * 100)

    computed = _compute_visual_pair_sims(records, embeds)
    pair_sims = computed["pair_sims"]
    sims_by_label = computed["sims_by_label"]
    quantile_taus = computed["quantile_taus"]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        arr = np.array(vals)
        return {
            "n": len(arr), "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "q25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)), "q75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    hdr = (f"    {'label':<14} | {'n':>5} | {'mean':>7} | {'std':>7} | {'min':>7} | "
           f"{'q25':>7} | {'median':>7} | {'q75':>7} | {'max':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for label in ("entailment", "contradiction", "neutral"):
        s = _stats(sims_by_label[label])
        if s["n"] == 0:
            print(f"    {label:<14} | {0:>5} | n/a")
            continue
        print(f"    {label:<14} | {s['n']:>5} | {s['mean']:>7.4f} | {s['std']:>7.4f} | {s['min']:>7.4f} | "
              f"{s['q25']:>7.4f} | {s['median']:>7.4f} | {s['q75']:>7.4f} | {s['max']:>7.4f}")
    print()

    print("  -- GLOBAL OBSERVED DISTRIBUTION QUANTILES (define the (D')/(E') sweep grid) --")
    print("    " + "  ".join(f"p{p}={quantile_taus[p]:.4f}" for p in QUANTILE_PCTS))
    print()

    ent = sims_by_label["entailment"]
    other = sims_by_label["contradiction"] + sims_by_label["neutral"]
    print("  -- READ --")
    if ent and other:
        gap = float(np.mean(ent)) - float(np.mean(other))
        print(f"  entailment mean ({np.mean(ent):.4f}) minus contradiction+neutral mean ({np.mean(other):.4f}) = {gap:+.4f}")
        if gap > 0.03:
            print("  Entailment pairs sit measurably above the rest in claim-text/frame-visual CLIP space.")
        else:
            print("  No clear separation -- entailment does not sit meaningfully above contradiction/neutral.")
    print()

    return {"pair_sims": pair_sims, "quantile_taus": quantile_taus}


def _scoped_verdict(claim_rows: list[dict], pair_sims: dict, tau: float, k: int) -> str:
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


def section_dprime_verdict_replay_sweep(records: list[dict], sep_state: dict) -> None:
    print("=" * 100)
    print("(D') VERDICT REPLAY SWEEP, VISUAL (real claims): scoped rule vs current rule")
    print("=" * 100)

    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}
    pair_sims = sep_state["pair_sims"]
    quantile_taus = sep_state["quantile_taus"]

    hdr = (f"  {'pctile':>6} | {'tau':>7} | {'k':>2} | {'verified':>8} | {'rejected':>8} | "
           f"{'unverifiable':>12} | {'rej_to_ver':>10} | {'other_flips':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pct in SWEEP_PCTS:
        tau = quantile_taus[pct]
        for k in K_SWEEP:
            counts = {"verified": 0, "rejected": 0, "unverifiable": 0}
            rej_to_ver = 0
            other_flips = 0
            for key, rows in real_groups.items():
                current = rows[0]["live_verdict_for_claim"]
                scoped = _scoped_verdict(rows, pair_sims, tau, k)
                counts[scoped] += 1
                if scoped != current:
                    if current == "rejected" and scoped == "verified":
                        rej_to_ver += 1
                    else:
                        other_flips += 1
            print(f"  {pct:>6} | {tau:>7.4f} | {k:>2} | {counts['verified']:>8} | {counts['rejected']:>8} | "
                  f"{counts['unverifiable']:>12} | {rej_to_ver:>10} | {other_flips:>12}")
    print()


def section_eprime_fabricated_accept_rate(records: list[dict], sep_state: dict) -> None:
    print("=" * 100)
    print("(E') FABRICATED-CLAIM ACCEPT RATE, VISUAL: scoped rule vs current rule baseline")
    print("=" * 100)

    groups = _group_by_claim(records)
    fab_groups = {k: v for k, v in groups.items() if k[2] == "fabricated"}
    pair_sims = sep_state["pair_sims"]
    quantile_taus = sep_state["quantile_taus"]

    if not fab_groups:
        print("  No fabricated-claim records in capture.")
        return

    n_fab = len(fab_groups)
    n_current_accept = sum(1 for rows in fab_groups.values() if rows[0]["live_verdict_for_claim"] == "verified")
    print(f"  n fabricated claim instances: {n_fab}")
    print(f"  CURRENT rule accept (verified) rate: {n_current_accept}/{n_fab} = {n_current_accept / n_fab:.4f}")
    print()

    hdr = f"  {'pctile':>6} | {'tau':>7} | {'k':>2} | {'accepted':>8} | {'accept_rate':>11}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for pct in SWEEP_PCTS:
        tau = quantile_taus[pct]
        for k in K_SWEEP:
            n_accept = 0
            for key, rows in fab_groups.items():
                scoped = _scoped_verdict(rows, pair_sims, tau, k)
                if scoped == "verified":
                    n_accept += 1
            print(f"  {pct:>6} | {tau:>7.4f} | {k:>2} | {n_accept:>8} | {n_accept / n_fab:>11.4f}")
    print()

    # ── per-claim eyeball table: one representative query, tau=p75, k=3 ──
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}
    tau_75 = quantile_taus[75]
    k_fixed = 3

    rep_query = None
    for key in real_groups:
        rep_query = key[0]
        break

    fact_to_caption = {r["fact_text"]: r["caption_only"] for r in records if r["fact_text"] is not None}

    print(f"  -- EYEBALL TABLE: query={rep_query!r}  tau=p75={tau_75:.4f}  k={k_fixed} --")
    hdr2 = (f"    {'claim':<45} | {'top1_frame_idx':>14} | {'sim':>6} | {'caption':<40} | "
            f"{'label':<12} | {'current':<12} | {'scoped':<12}")
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
        scoped = _scoped_verdict(rows, pair_sims, tau_75, k_fixed)
        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        top_frame_idx = _frame_idx_of(top_row["fact_text"])
        caption_disp = fact_to_caption.get(top_row["fact_text"], "")
        caption_disp = (caption_disp[:37] + "...") if len(caption_disp) > 40 else caption_disp
        print(f"    {claim_disp:<45} | {top_frame_idx:>14} | {top_sim:>6.4f} | {caption_disp:<40} | "
              f"{top_row['label']:<12} | {current:<12} | {scoped:<12}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# --ceiling: pure offline analysis of diag_v2_capture.jsonl (+ diag_v3_embeds.npz
# for the FLIP AUDIT, which needs the same visual sims as (D')). No model loads.
# ─────────────────────────────────────────────────────────────────────────────

METADATA_RE = re.compile(r"action score|persistence|selection reason", re.IGNORECASE)
META_COMMENTARY_RE = re.compile(r"happy to help|more frames or context|provided (frame )?evidence$", re.IGNORECASE)
ABSENCE_RE = re.compile(r"\bno\b|couldn't find|not possible to confirm|no indication", re.IGNORECASE)

# Domain visual-content nouns (CCTV/VIRAT scene vocabulary). A claim matching
# none of these has no concrete visual referent to check against a frame --
# treated as meta_commentary rather than a checkable visual_assertion.
VISUAL_NOUN_RE = re.compile(
    r"\bcar\b|\bvehicles?\b|\bpeople\b|\bperson\b|\bpedestrians?\b|\bwalk(?:ing|s)?\b|"
    r"\brun(?:ning|s)?\b|\bfle(?:e|eing|es)\b|\bbags?\b|\bfires?\b|\bsmoke\b|\btrucks?\b|"
    r"\bumbrellas?\b|\bboats?\b|\bdogs?\b|\broofs?\b|\bentrances?\b|\bbuildings?\b|"
    r"\bparking\b|\blots?\b|\broads?\b|\bscenes?\b|\bareas?\b|\bitems?\b|\bobjects?\b|"
    r"\bframes?\b|\bactivit(?:y|ies)\b|\bmotion\b",
    re.IGNORECASE,
)


def _classify_claim(claim: str) -> str:
    if METADATA_RE.search(claim):
        return "metadata"
    if META_COMMENTARY_RE.search(claim):
        return "meta_commentary"
    if not VISUAL_NOUN_RE.search(claim):
        return "meta_commentary"
    return "visual_assertion"


def section1_recovery_ceiling(records: list[dict]) -> None:
    print("=" * 100)
    print("(1) RECOVERY CEILING: rejected real claims with >= 1 entailment pair")
    print("=" * 100)

    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    recoverable = []
    for key, rows in real_groups.items():
        if rows[0]["live_verdict_for_claim"] != "rejected":
            continue
        ent_rows = [r for r in rows if r["label"] == "entailment"]
        if ent_rows:
            best = max(ent_rows, key=lambda r: r["entailment_score"])
            recoverable.append((key, best))

    print(f"  ceiling count: {len(recoverable)}")
    print(f"  hypothesis under test (exactly 2): {'CONFIRMED' if len(recoverable) == 2 else 'NOT CONFIRMED'}")
    print()
    for key, best in recoverable:
        print(f"    query={key[0]!r}")
        print(f"      claim={key[1]!r}")
        print(f"      max_entailment_score={best['entailment_score']:.4f}")
        print(f"      fact_text={best['fact_text']!r}")
    print()


def section2_claim_taxonomy_table(records: list[dict]) -> None:
    print("=" * 100)
    print("(2) CLAIM TAXONOMY TABLE")
    print("=" * 100)

    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    classes = ("metadata", "meta_commentary", "visual_assertion")
    verdicts = ("verified", "rejected", "unverifiable")
    crosstab = {c: {v: 0 for v in verdicts} for c in classes}
    has_entailment = {c: 0 for c in classes}
    class_totals = {c: 0 for c in classes}

    for key, rows in real_groups.items():
        cls = _classify_claim(key[1])
        verdict = rows[0]["live_verdict_for_claim"]
        crosstab[cls][verdict] += 1
        class_totals[cls] += 1
        if any(r["label"] == "entailment" for r in rows):
            has_entailment[cls] += 1

    hdr = f"  {'class':<18} | {'verified':>8} | {'rejected':>8} | {'unverifiable':>12} | {'total':>6} | {'has_entailment_pair':>20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cls in classes:
        print(f"  {cls:<18} | {crosstab[cls]['verified']:>8} | {crosstab[cls]['rejected']:>8} | "
              f"{crosstab[cls]['unverifiable']:>12} | {class_totals[cls]:>6} | {has_entailment[cls]:>20}")
    print()


def section3_flip_audit(records: list[dict], embeds: dict) -> None:
    print("=" * 100)
    print("(3) FLIP AUDIT at tau=p75, k=3 (visual scoping)")
    print("=" * 100)

    computed = _compute_visual_pair_sims(records, embeds)
    pair_sims = computed["pair_sims"]
    tau = computed["quantile_taus"][75]
    k = 3
    print(f"  tau (p75) = {tau:.4f}  k = {k}")
    print()

    groups = _group_by_claim(records)
    real_groups = {k2: v for k2, v in groups.items() if k2[2] == "real"}

    hdr = f"  {'claim':<50} | {'class':<18} | {'live -> scoped':<28}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rej_to_unverifiable_special = 0
    total_flips = 0
    for key, rows in real_groups.items():
        live = rows[0]["live_verdict_for_claim"]
        scoped = _scoped_verdict(rows, pair_sims, tau, k)
        if scoped == live:
            continue
        total_flips += 1
        cls = _classify_claim(key[1])
        is_absence = cls == "visual_assertion" and bool(ABSENCE_RE.search(key[1]))
        display_cls = cls + ("+absence" if is_absence else "")
        claim_disp = (key[1][:47] + "...") if len(key[1]) > 50 else key[1]
        print(f"  {claim_disp:<50} | {display_cls:<18} | {live:<12} -> {scoped:<12}")

        if live == "rejected" and scoped == "unverifiable" and (cls in ("metadata", "meta_commentary") or is_absence):
            rej_to_unverifiable_special += 1
    print()

    print(f"  total flips vs live verdict:                                    {total_flips}")
    print(f"  rejected->unverifiable flips on metadata/meta_commentary/absence claims: {rej_to_unverifiable_special}")
    print()


def run_ceiling() -> None:
    records = _load_capture()
    embeds = _load_v3_embeds()

    section1_recovery_ceiling(records)
    section2_claim_taxonomy_table(records)
    section3_flip_audit(records, embeds)


def run_analyze() -> None:
    records = _load_capture()
    embeds = _load_v3_embeds()

    parity_ok = section_bprime_parity_self_check(records)
    if not parity_ok:
        sys.exit(1)
    sep_state = section_cprime_visual_separation(records, embeds)
    section_dprime_verdict_replay_sweep(records, sep_state)
    section_eprime_fabricated_accept_rate(records, sep_state)


# ─────────────────────────────────────────────────────────────────────────────
# --binding-ceiling: offline over diag_v2_capture.jsonl, MAY load the DeBERTa
# NLI model (reusing CerberusV's loader) to score NEW (kernel, cited-caption)
# pairs that don't exist in the v2 capture. No iris/ edits, no changes to
# existing modes.
# ─────────────────────────────────────────────────────────────────────────────

FRAME_CITE_RE = re.compile(r"Frame[s]?\s+(\d+)", re.IGNORECASE)
BARE_TS_RE = re.compile(r"(\d+\.?\d*)s")
FACT_HEADER_RE = re.compile(r"^Frame (\d+) at ([\d.]+)s:")
TS_TOLERANCE = 0.3

METADATA_CLAUSE_RE = re.compile(
    r"with an action score[^,.]*|persistence[^,.]*|action scores? (?:of|ranging)[^,.]*|selection reason[^,.]*",
    re.IGNORECASE,
)
# Words that, left dangling right before a comma/period after clause removal,
# indicate the strip mangled the sentence (orphaned modifier/preposition with
# nothing left to attach to). Deliberately NOT auto-deleted -- conservative
# stripping means surfacing these for human audit, not silently erasing more.
ORPHAN_RE = re.compile(
    r"\b(a|an|the|low|high|very|slight|slightly|in|of|for|to|with|and)\s*[,.]",
    re.IGNORECASE,
)
MANGLE_STOP_THRESHOLD = 2


def _fact_header(fact_text: str) -> tuple[int, float]:
    m = FACT_HEADER_RE.match(fact_text)
    if not m:
        raise ValueError(f"fact_text does not match expected 'Frame N at Ts:' format: {fact_text!r}")
    return int(m.group(1)), float(m.group(2))


def _resolve_citations(claim: str, rows: list[dict]) -> list[dict]:
    """CITATION PARSE: resolve /Frame[s]? N/ and bare /T.TTs/ references in a
    claim to the specific fact rows (from this claim's own retrieved pool)
    they refer to. Bare timestamps match a fact's timestamp within 0.3s."""
    by_frame_idx: dict[int, dict] = {}
    by_ts: list[tuple[float, dict]] = []
    for r in rows:
        idx, ts = _fact_header(r["fact_text"])
        by_frame_idx[idx] = r
        by_ts.append((ts, r))

    cited: list[dict] = []
    seen_facts: set[str] = set()

    for m in FRAME_CITE_RE.finditer(claim):
        fid = int(m.group(1))
        row = by_frame_idx.get(fid)
        if row is not None and row["fact_text"] not in seen_facts:
            cited.append(row)
            seen_facts.add(row["fact_text"])

    for m in BARE_TS_RE.finditer(claim):
        t = float(m.group(1))
        for ts, row in by_ts:
            if abs(ts - t) <= TS_TOLERANCE and row["fact_text"] not in seen_facts:
                cited.append(row)
                seen_facts.add(row["fact_text"])

    return cited


def _decompose_claim(claim: str) -> tuple[str, list[str], bool]:
    """DECOMPOSITION: strip metadata clauses via the conservative regex, tidy
    dangling connector artifacts left by clause removal, and flag orphaned
    modifiers the tidy step deliberately does not auto-fix (surfaced for
    human audit instead). Returns (kernel, metadata_fields, is_mangled)."""
    matches = list(METADATA_CLAUSE_RE.finditer(claim))
    if not matches:
        return claim, [], False

    fields: list[str] = []
    for m in matches:
        text = m.group(0).lower()
        if "action score" in text and "action_score" not in fields:
            fields.append("action_score")
        if "persistence" in text and "persistence_value" not in fields:
            fields.append("persistence_value")
        if "selection reason" in text and "is_peak/selection_reason" not in fields:
            fields.append("is_peak/selection_reason")

    kernel = METADATA_CLAUSE_RE.sub("", claim)
    # tidy: collapse whitespace/punctuation artifacts purely mechanical to
    # clause removal (double spaces, doubled commas, dangling connector words
    # with nothing following).
    kernel = re.sub(r"\b(and|with)\s*([,.])", r"\2", kernel, flags=re.IGNORECASE)
    kernel = re.sub(r",\s*,", ",", kernel)
    kernel = re.sub(r",\s*\.", ".", kernel)
    kernel = re.sub(r"[ \t]{2,}", " ", kernel)
    kernel = kernel.strip()

    is_mangled = bool(ORPHAN_RE.search(kernel)) or len(re.sub(r"[^\w]", "", kernel)) < 10
    return kernel, fields, is_mangled


def _get_nli_gate():
    from iris.cerberus_v import CerberusV
    gate = CerberusV()
    gate._get_spacy()
    gate._get_nli_model()
    return gate


def _score_kernel_pair(gate, claim: str, fact: str) -> tuple[str, float]:
    """Per-pair NLI scoring clone (negation + geo checks) lifted verbatim
    from iris/cerberus_v.py:_full_nli's inner loop -- same thresholds, same
    logic, applied to a single (kernel, caption) pair that doesn't exist in
    the v2 capture."""
    import torch
    import torch.nn.functional as F

    nlp = gate._get_spacy()
    tokenizer, model = gate._get_nli_model()
    device = model.device

    inputs = tokenizer([fact], [claim], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        pred = int(torch.argmax(logits, dim=-1).cpu().item())
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]

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
    entailment_score = probs[entail_idx] if entail_idx < len(probs) else probs[-1]

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

    return label, entailment_score


def run_binding_ceiling() -> None:
    records = _load_capture()
    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    # ── (1) CITATION PARSE ──────────────────────────────────────────────────
    print("=" * 100)
    print("(1) CITATION PARSE")
    print("=" * 100)

    cited_map: dict[tuple, list[dict]] = {}
    for key, rows in real_groups.items():
        cited_map[key] = _resolve_citations(key[1], rows)

    n_cited = sum(1 for v in cited_map.values() if v)
    n_uncited = len(cited_map) - n_cited
    print(f"  claims with >=1 resolvable citation: {n_cited}")
    print(f"  uncited claims:                      {n_uncited}")
    print()
    for key, cited_rows in cited_map.items():
        if cited_rows:
            frame_idxs = [_fact_header(r["fact_text"])[0] for r in cited_rows]
            print(f"    CITED    query={key[0]!r} claim={key[1]!r} -> frame_idxs={frame_idxs}")
    for key, cited_rows in cited_map.items():
        if not cited_rows:
            print(f"    UNCITED  query={key[0]!r} claim={key[1]!r}")
    print()

    # ── (2) DECOMPOSITION ────────────────────────────────────────────────────
    print("=" * 100)
    print("(2) DECOMPOSITION: strip metadata clauses, tidy connectors")
    print("=" * 100)

    decomposed: dict[tuple, tuple[str, list[str], bool]] = {}
    mangled_examples: list[tuple] = []
    for key in real_groups:
        kernel, fields, is_mangled = _decompose_claim(key[1])
        decomposed[key] = (kernel, fields, is_mangled)
        if kernel != key[1]:
            print(f"    ORIGINAL: {key[1]!r}")
            print(f"    KERNEL:   {kernel!r}")
            if fields:
                print(f"    metadata_fields: {fields}")
            if is_mangled:
                print(f"    ** FLAGGED MANGLED **")
            print()
        if is_mangled:
            mangled_examples.append((key, kernel))

    print(f"  claims with metadata clause stripped: {sum(1 for k, (kern, f, m) in decomposed.items() if kern != k[1])}")
    print(f"  claims flagged mangled on audit:      {len(mangled_examples)}")
    print()

    if len(mangled_examples) > MANGLE_STOP_THRESHOLD:
        print(f"STOPPING: decomposition regex mangled {len(mangled_examples)} kernels (> {MANGLE_STOP_THRESHOLD}).")
        print("Examples:")
        for key, kernel in mangled_examples:
            print(f"    query={key[0]!r}")
            print(f"    original={key[1]!r}")
            print(f"    kernel={kernel!r}")
        sys.exit(1)

    # ── (3) BOUND NLI REPLAY ─────────────────────────────────────────────────
    print("=" * 100)
    print("(3) BOUND NLI REPLAY: (kernel, cited frame's caption_only)")
    print("=" * 100)

    gate = _get_nli_gate()
    binding_label: dict[tuple, str] = {}   # key -> top label among cited pairs
    binding_detail: dict[tuple, list] = {}  # key -> [(frame_idx, sim n/a, label, score, caption)]

    for key, cited_rows in cited_map.items():
        if not cited_rows:
            continue
        kernel, fields, _ = decomposed[key]
        scored = []
        for row in cited_rows:
            label, score = _score_kernel_pair(gate, kernel, row["caption_only"])
            frame_idx, _ts = _fact_header(row["fact_text"])
            scored.append((score, label, frame_idx, row["caption_only"]))
        scored.sort(key=lambda x: -x[0])
        top_score, top_label, top_frame_idx, top_caption = scored[0]
        binding_label[key] = top_label
        binding_detail[key] = scored

        print(f"    query={key[0]!r}")
        print(f"    kernel={kernel!r}")
        for score, label, frame_idx, caption in scored:
            print(f"      vs frame_idx={frame_idx} caption={caption!r} -> label={label} score={score:.4f}")
        print(f"    ARGMAX: frame_idx={top_frame_idx} label={top_label} score={top_score:.4f}")
        print()

    metadata_only_note = [k for k in real_groups if decomposed[k][1] and not cited_map[k]]
    if metadata_only_note:
        print("  (metadata clauses on UNCITED claims -- not NLI-scored, listed under checkable_metadata below)")
        print()

    # ── (4) VERDICT TABLE ────────────────────────────────────────────────────
    print("=" * 100)
    print("(4) VERDICT TABLE")
    print("=" * 100)

    hdr = (f"  {'claim':<45} | {'class':<18} | {'live':<13} | {'binding':<15} | {'metadata_disposition':<40}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_rejected = 0
    n_become_verified_kernel = 0
    n_checkable_metadata = 0
    n_overlap = 0

    for key, rows in real_groups.items():
        live = rows[0]["live_verdict_for_claim"]
        cls = _classify_claim(key[1])
        kernel, fields, _ = decomposed[key]

        if key in binding_label:
            top_label = binding_label[key]
            if top_label == "entailment":
                binding_verdict = "verified_kernel"
            elif top_label == "contradiction":
                binding_verdict = "rejected"
            else:
                binding_verdict = "unverifiable"
        else:
            binding_verdict = "no_citation(answer-semantics)"

        if fields:
            disposition = f"checkable_metadata (needs: {', '.join(fields)})"
        else:
            disposition = "no_metadata_clause"

        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        print(f"  {claim_disp:<45} | {cls:<18} | {live:<13} | {binding_verdict:<15} | {disposition:<40}")

        if live == "rejected":
            n_rejected += 1
            becomes_verified = binding_verdict == "verified_kernel"
            is_checkable = bool(fields)
            if becomes_verified:
                n_become_verified_kernel += 1
            if is_checkable:
                n_checkable_metadata += 1
            if becomes_verified and is_checkable:
                n_overlap += 1
    print()

    uncited_by_class: dict[str, int] = {}
    for key, cited_rows in cited_map.items():
        if cited_rows:
            continue
        cls = _classify_claim(key[1])
        uncited_by_class[cls] = uncited_by_class.get(cls, 0) + 1
    print("  -- UNCITED CLAIMS (routed to answer-semantics layer, no NLI attempted) --")
    for cls, n in sorted(uncited_by_class.items()):
        print(f"    {cls:<18} : {n}")
    print()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  currently-rejected real claims:                                    {n_rejected}")
    print(f"  -> become verified_kernel under citation-bound NLI:                {n_become_verified_kernel}")
    print(f"  -> have a deterministically checkable metadata clause:             {n_checkable_metadata}")
    print(f"  -> (overlap: both verified_kernel AND checkable_metadata):         {n_overlap}")
    total_recoveries = n_become_verified_kernel + n_checkable_metadata
    print(f"  total binding recoveries (verified_kernel + checkable_metadata):   {total_recoveries}")
    print(f"  similarity-scoping ceiling (from --ceiling):                       2")
    if total_recoveries > 2:
        print(f"  READ: citation-binding recovers MORE than similarity-scoping's ceiling of 2 ({total_recoveries} > 2).")
    elif total_recoveries == 2:
        print("  READ: citation-binding recovers the SAME as similarity-scoping's ceiling (no improvement).")
    else:
        print(f"  READ: citation-binding recovers FEWER than similarity-scoping's ceiling ({total_recoveries} < 2).")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# --kernel-normalize: same structure as --binding-ceiling (reuses its citation
# parse + NLI replay), but after metadata-clause stripping ALSO normalizes the
# kernel to plain visual language before NLI. Offline over diag_v2_capture.jsonl;
# MAY load the DeBERTa NLI model (same loader as --binding-ceiling) to score
# the new (normalized_kernel, caption) pairs.
# ─────────────────────────────────────────────────────────────────────────────

# Conditional "Frame N (also) shows/depicts" -> "there is": ONLY applied when
# shows/depicts directly follows a Frame citation (checked BEFORE the generic
# Frame-ref strip below runs, since that strip would otherwise erase the
# positional information this substitution depends on).
FRAME_VERB_RE = re.compile(r"Frame[s]?\s+\d+\s+(?:also\s+)?(?:shows?|depicts?)\b", re.IGNORECASE)
FRAME_REF_STRIP_RE = re.compile(r"Frame[s]?\s+\d+\s*", re.IGNORECASE)
TIMESTAMP_PAREN_RE = re.compile(r"\(?Timestamp:?\s*[\d.]+s\)?", re.IGNORECASE)
AT_TS_RE = re.compile(r"at\s+[\d.]+s", re.IGNORECASE)
LEADING_HEDGE_RE = re.compile(
    r"^(based on the provided frame evidence,?\s*|it appears that\s*|specifically[,:]?\s*|"
    r"additionally,?\s*|however,?\s*)",
    re.IGNORECASE,
)


def _normalize_kernel(stripped_kernel: str) -> str:
    """Normalize a metadata-stripped kernel to plain visual language before
    NLI. Order matters: the conditional shows/depicts substitution must run
    BEFORE the generic Frame-ref strip, since it depends on a Frame reference
    still being directly adjacent to the verb."""
    text = stripped_kernel
    text = FRAME_VERB_RE.sub("there is", text)
    text = FRAME_REF_STRIP_RE.sub("", text)
    text = TIMESTAMP_PAREN_RE.sub("", text)
    text = AT_TS_RE.sub("", text)

    prev = None
    while prev != text:
        prev = text
        text = LEADING_HEDGE_RE.sub("", text)

    # tidy: purely mechanical artifacts of the stripping above (emptied
    # parens, dangling connector words, doubled punctuation, leading stray
    # colon/comma where a Frame/Timestamp prefix used to be).
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\b(and|with)\s*([,.])", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"^[,:;]\s*", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def run_kernel_normalize() -> None:
    records = _load_capture()
    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    print("=" * 100)
    print("FULL AUDIT TRAIL: original -> stripped-kernel -> normalized-kernel")
    print("=" * 100)

    cited_map: dict[tuple, list[dict]] = {}
    stripped_map: dict[tuple, tuple[str, list[str]]] = {}
    normalized_map: dict[tuple, tuple[str, bool]] = {}
    mangled_examples: list[tuple] = []

    for key, rows in real_groups.items():
        cited_map[key] = _resolve_citations(key[1], rows)
        stripped, fields, _ = _decompose_claim(key[1])
        normalized = _normalize_kernel(stripped)
        is_mangled = bool(ORPHAN_RE.search(normalized)) or len(re.sub(r"[^\w]", "", normalized)) < 10

        stripped_map[key] = (stripped, fields)
        normalized_map[key] = (normalized, is_mangled)

        print(f"    ORIGINAL:   {key[1]!r}")
        print(f"    STRIPPED:   {stripped!r}")
        print(f"    NORMALIZED: {normalized!r}")
        if fields:
            print(f"    metadata_fields: {fields}")
        if is_mangled:
            print("    ** FLAGGED MANGLED **")
            mangled_examples.append((key, normalized))
        print()

    print(f"  claims flagged mangled (normalized kernel) on audit: {len(mangled_examples)}")
    print()

    if len(mangled_examples) > MANGLE_STOP_THRESHOLD:
        print(f"STOPPING: normalization mangled {len(mangled_examples)} kernels (> {MANGLE_STOP_THRESHOLD}).")
        print("Examples:")
        for key, normalized in mangled_examples:
            print(f"    query={key[0]!r}")
            print(f"    original={key[1]!r}")
            print(f"    normalized={normalized!r}")
        sys.exit(1)

    # ── BOUND NLI REPLAY on normalized kernels ──────────────────────────────
    print("=" * 100)
    print("BOUND NLI REPLAY: (normalized_kernel, cited frame's caption_only)")
    print("=" * 100)

    gate = _get_nli_gate()
    binding_label: dict[tuple, str] = {}

    for key, cited_rows in cited_map.items():
        if not cited_rows:
            continue
        normalized, _ = normalized_map[key]
        scored = []
        for row in cited_rows:
            label, score = _score_kernel_pair(gate, normalized, row["caption_only"])
            frame_idx, _ts = _fact_header(row["fact_text"])
            scored.append((score, label, frame_idx, row["caption_only"]))
        scored.sort(key=lambda x: -x[0])
        top_score, top_label, top_frame_idx, top_caption = scored[0]
        binding_label[key] = top_label

        print(f"    query={key[0]!r}")
        print(f"    normalized_kernel={normalized!r}")
        for score, label, frame_idx, caption in scored:
            print(f"      vs frame_idx={frame_idx} caption={caption!r} -> label={label} score={score:.4f}")
        print(f"    ARGMAX: frame_idx={top_frame_idx} label={top_label} score={top_score:.4f}")
        print()

    # ── VERDICT TABLE ────────────────────────────────────────────────────────
    print("=" * 100)
    print("VERDICT TABLE")
    print("=" * 100)

    hdr = f"  {'claim':<45} | {'class':<18} | {'live':<13} | {'binding_normalized':<20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key, rows in real_groups.items():
        live = rows[0]["live_verdict_for_claim"]
        cls = _classify_claim(key[1])
        if key in binding_label:
            top_label = binding_label[key]
            if top_label == "entailment":
                verdict = "verified_kernel"
            elif top_label == "contradiction":
                verdict = "rejected"
            else:
                verdict = "unverifiable"
        else:
            verdict = "no_citation(answer-semantics)"
        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        print(f"  {claim_disp:<45} | {cls:<18} | {live:<13} | {verdict:<20}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    n_entailments = sum(1 for v in binding_label.values() if v == "entailment")
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  entailments under normalized-kernel NLI replay: {n_entailments}")
    print(f"  entailments under --binding-ceiling (unnormalized kernel):    0")
    if n_entailments > 0:
        print(f"  READ: normalization recovers {n_entailments} entailment(s) that unnormalized "
              f"citation-binding did not.")
    else:
        print("  READ: normalization does not recover any entailments either -- kernel-normalize "
              "is also dead for this sample.")
    print()


def main() -> None:
    if "--capture-addendum" in sys.argv:
        run_capture_addendum()
    elif "--analyze" in sys.argv:
        run_analyze()
    elif "--ceiling" in sys.argv:
        run_ceiling()
    elif "--binding-ceiling" in sys.argv:
        run_binding_ceiling()
    elif "--kernel-normalize" in sys.argv:
        run_kernel_normalize()
    else:
        print("Usage: python scripts/diag_v3_visual_scoping.py --capture-addendum | --analyze | "
              "--ceiling | --binding-ceiling | --kernel-normalize")
        sys.exit(1)


if __name__ == "__main__":
    main()
