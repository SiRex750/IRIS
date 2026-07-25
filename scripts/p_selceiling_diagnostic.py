"""P_SELCEILING diagnostic: caption-signal recovery ceiling on the recoverable set.

Implements eval_results/P_selceiling_prereg.md (a5d1cdd) exactly. Read-only --
selects nothing, changes no default, no frozen config value, no seated
reranker. Reuses pnowa_width_topk_sweep.py::_load_val_rows() verbatim
(including its test-leak assertion) and p_funnel_diagnostic.py's _in_gold /
_cluster_bootstrap verbatim (same point-in-span membership test and same
bootstrap routine the funnel used) rather than reimplementing span math.

TWO fixed arms, both scored over the same top_k=8 pool captions vs the
question text:
  ARM 1 -- LEXICAL (the GATE arm). Token-overlap, fixed inline stopword list.
  ARM 2 -- CLIP-TEXT (context only). Already-loaded CLIP text encoder,
           cosine over caption-text embedding vs question-text embedding.
Ties within an arm are broken by the frame's existing CLIP-in-pool rank
(lower rank wins); tie rate is recorded per arm.

Caption text = the frame's cached semantic_caption field (the natural-
language half of the {"clip_label", "semantic_caption"} dict already sitting
on FrameRecord.caption in index_cache/). A pool frame with no cached caption
is SCORED (lexical score 0.0, CLIP-text score -inf) -- never skipped -- so it
can still be selected via tie-break, but only loses to any frame that has a
real caption.

VERIFY: python scripts/p_selceiling_diagnostic.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.grounding_scorer import load_indexes
from eval.span import predict_span, _frame_embedding
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved
from scripts.pillar2_grounded_qa import git_provenance, config_hash
from scripts.pnowa_width_topk_sweep import (
    _load_val_rows, BASE, GQA_JSON, CACHE_DIR, TEST_VIDEO_LIST,
)
from scripts.p_funnel_diagnostic import _in_gold, _cluster_bootstrap

TOP_K = 8
SPAN_MODE = "ppr_peak"
HALF_WIDTH = 2.2
PEAK_SOURCE = "clip_in_ppr_top8"
SEED = 20260710
NUM_BOOT = 1000

FUNNEL_RAW_JSON = REPO / "eval_results" / "P_funnel_raw.json"
OUT_JSON = REPO / "eval_results" / "P_selceiling_raw.json"

# ARM 1 -- fixed, declared-in-advance stopword list (P_selceiling_prereg.md).
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
    "this", "that", "it", "as", "by", "from",
}
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens = _TOKEN_RE.split(text.lower())
    return {t for t in tokens if t and t not in STOPWORDS}


def _lexical_score(q_tokens: set[str], cap_tokens: set[str]) -> float:
    if not q_tokens:
        return 0.0
    return len(q_tokens & cap_tokens) / len(q_tokens)


def _caption_text(frame: dict) -> str | None:
    cap = frame.get("caption")
    if not isinstance(cap, dict):
        return None
    text = cap.get("semantic_caption")
    if not text or not isinstance(text, str):
        return None
    return text


def _cosine(a, b) -> float:
    av = np.asarray(a, dtype=np.float64).flatten()
    bv = np.asarray(b, dtype=np.float64).flatten()
    a_norm = float(np.linalg.norm(av))
    b_norm = float(np.linalg.norm(bv))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return float("-inf")
    return float(np.dot(av, bv) / (a_norm * b_norm))


def _clip_pool_rank(retrieved: list[dict], emb) -> dict[int, int]:
    """1-based rank of each pool frame by descending raw CLIP cosine
    similarity to the query embedding -- the same signal peak_source=
    clip_in_ppr_top8 uses to pick the anchor. Ties broken by pool order
    (stable). frame_idx -> rank."""
    q = np.asarray(emb, dtype=np.float64).flatten()
    q_norm = float(np.linalg.norm(q))
    scored = []
    for i, f in enumerate(retrieved):
        e = _frame_embedding(f)
        sim = float("-inf")
        if e is not None and q_norm >= 1e-8:
            ev = np.asarray(e, dtype=np.float64).flatten()
            e_norm = float(np.linalg.norm(ev))
            if e_norm >= 1e-8:
                sim = float(np.dot(ev, q) / (e_norm * q_norm))
        scored.append((sim, i, f["frame_idx"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return {frame_idx: rank for rank, (_, _, frame_idx) in enumerate(scored, start=1)}


def _pick_top(scores: list[float], retrieved: list[dict], pool_rank: dict[int, int]) -> tuple[int, bool, int]:
    """Return (picked_frame_idx, was_tied, tie_n). Tie-break: lowest
    CLIP-in-pool rank among the tied top scorers."""
    top = max(scores)
    tied_idxs = [i for i, s in enumerate(scores) if s == top]
    tie_n = len(tied_idxs)
    was_tied = tie_n > 1
    best_i = min(tied_idxs, key=lambda i: pool_rank[retrieved[i]["frame_idx"]])
    return retrieved[best_i]["frame_idx"], was_tied, tie_n


def main() -> None:
    grounded_rows = _load_val_rows()
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    used_videos = {r["video"] for r in grounded_rows}
    n_videos = len(used_videos)
    print(f"[DATA] VAL grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    test_videos = {l.strip() for l in open(TEST_VIDEO_LIST, encoding="utf-8") if l.strip()}
    leaked = used_videos & test_videos
    assert not leaked, f"TEST videos present in P_SELCEILING run: {sorted(leaked)}"
    print(f"[GUARD] zero test videos present: OK ({len(used_videos)} val videos, {len(test_videos)} test videos on file)")

    duration_by_vid = {vid: float(gsub[vid].get("duration", 0)) for vid in used_videos}

    cfg = IRISConfig(**BASE, l2_retrieve_top_k=TOP_K)
    loaded = load_indexes(grounded_rows, CACHE_DIR)
    index_load_failures = sum(1 for vid in used_videos if loaded.get(vid) is None)
    print(f"[GUARD] index load failures: {index_load_failures}")
    assert index_load_failures == 0, f"{index_load_failures} index(es) failed to load"

    # ── funnel cross-check table (assertion 1) ─────────────────────────────
    funnel_raw = json.load(open(FUNNEL_RAW_JSON, encoding="utf-8"))
    funnel_flags = {
        (r["video"], str(r["qid"])): (r["pool_coverage"], r["peak_in_gold"])
        for r in funnel_raw["per_question"] if r["top_k"] == TOP_K
    }

    caption_emb_cache: dict[str, np.ndarray] = {}

    def embed_text_cached(text: str) -> np.ndarray:
        if text not in caption_emb_cache:
            caption_emb_cache[text] = _embed_query(text, cfg)
        return caption_emb_cache[text]

    per_question = []
    n_frames_total = 0
    n_frames_scored = 0
    n_frames_with_caption = 0
    mismatches = []
    len_mismatches = 0

    for row in grounded_rows:
        vid, qid = row["video"], str(row["qid"])
        index = loaded.get(vid)
        gold_spans = gsub[vid]["location"][qid]

        emb = _embed_query(row["question"], cfg)
        retrieved = _build_retrieved(index, emb, cfg)
        if len(retrieved) != TOP_K:
            len_mismatches += 1

        pool_coverage = 1 if any(_in_gold(r["timestamp"], gold_spans) for r in retrieved) else 0
        span, t_peak = predict_span(
            retrieved, mode=SPAN_MODE, half_width=HALF_WIDTH,
            duration=duration_by_vid.get(vid), peak_source=PEAK_SOURCE,
            query_embedding=emb, return_peak=True,
        )
        peak_in_gold = 1 if (t_peak is not None and _in_gold(t_peak, gold_spans)) else 0

        key = (vid, qid)
        if key in funnel_flags:
            f_pool, f_peak = funnel_flags[key]
            if (f_pool, f_peak) != (pool_coverage, peak_in_gold):
                mismatches.append((vid, qid, (f_pool, f_peak), (pool_coverage, peak_in_gold)))
        else:
            mismatches.append((vid, qid, None, (pool_coverage, peak_in_gold)))

        recoverable = (pool_coverage == 1 and peak_in_gold == 0)

        # ── arm scoring over the top_k=8 pool ───────────────────────────────
        q_tokens = _tokenize(row["question"])
        pool_rank = _clip_pool_rank(retrieved, emb)

        lex_scores = []
        clip_scores = []
        for f in retrieved:
            n_frames_total += 1
            n_frames_scored += 1  # never skipped
            cap_text = _caption_text(f)
            if cap_text is not None:
                n_frames_with_caption += 1
                cap_tokens = _tokenize(cap_text)
                lex_scores.append(_lexical_score(q_tokens, cap_tokens))
                clip_scores.append(_cosine(emb, embed_text_cached(cap_text)))
            else:
                lex_scores.append(0.0)
                clip_scores.append(float("-inf"))

        gold_frame_idxs = {r["frame_idx"] for r in retrieved if _in_gold(r["timestamp"], gold_spans)}

        arm1_pick, arm1_tied, arm1_tie_n = _pick_top(lex_scores, retrieved, pool_rank)
        arm2_pick, arm2_tied, arm2_tie_n = _pick_top(clip_scores, retrieved, pool_rank)

        per_question.append({
            "video": vid, "qid": qid,
            "pool_coverage": pool_coverage, "peak_in_gold": peak_in_gold,
            "recoverable": recoverable,
            "arm1_lexical": {
                "picked_frame_idx": arm1_pick,
                "picked_in_gold": 1 if arm1_pick in gold_frame_idxs else 0,
                "tied": arm1_tied, "tie_n": arm1_tie_n,
            },
            "arm2_clip_text": {
                "picked_frame_idx": arm2_pick,
                "picked_in_gold": 1 if arm2_pick in gold_frame_idxs else 0,
                "tied": arm2_tied, "tie_n": arm2_tie_n,
            },
            "agree": arm1_pick == arm2_pick,
        })

    print(f"[GUARD] len(retrieved) != top_k mismatches: {len_mismatches}")
    assert len_mismatches == 0, f"{len_mismatches} questions had len(retrieved) != top_k"

    # ── assertion 1: reproduces the funnel's own per-question flags ────────
    print(f"[GUARD] funnel cross-check mismatches: {len(mismatches)}")
    assert not mismatches, (
        f"{len(mismatches)} questions diverge from P_funnel_raw.json's top_k=8 "
        f"pool_coverage/peak_in_gold flags -- STOP. First few: {mismatches[:5]}"
    )
    n_pool_coverage = sum(r["pool_coverage"] for r in per_question)
    n_peak_in_gold = sum(r["peak_in_gold"] for r in per_question)
    recoverable_rows = [r for r in per_question if r["recoverable"]]
    print(f"[GUARD] assertion 1 PASSED: recoverable_set size = {len(recoverable_rows)} "
          f"== pool_coverage({n_pool_coverage}) - peak_in_gold({n_peak_in_gold}) = "
          f"{n_pool_coverage - n_peak_in_gold}")
    assert len(recoverable_rows) == n_pool_coverage - n_peak_in_gold

    # ── assertion 4: recoverable set is pool_coverage=1 AND peak_in_gold=0 ─
    assert all(r["pool_coverage"] == 1 and r["peak_in_gold"] == 0 for r in recoverable_rows)
    print(f"[GUARD] assertion 4 PASSED: every recoverable-set row has pool_coverage=1, peak_in_gold=0")

    # ── assertion 2: caption availability counted, nothing skipped ─────────
    n_frames_skipped = n_frames_total - n_frames_scored
    print(f"[GUARD] pool frames: {n_frames_total}, scored: {n_frames_scored}, skipped: {n_frames_skipped}, "
          f"with caption: {n_frames_with_caption} ({n_frames_with_caption / n_frames_total:.4%})")
    assert n_frames_skipped == 0, f"{n_frames_skipped} pool frames were skipped instead of scored"

    # ── bootstrap (video-clustered, recoverable set) ────────────────────────
    video_to_arm1: dict[str, list[float]] = {}
    video_to_arm2: dict[str, list[float]] = {}
    for r in recoverable_rows:
        video_to_arm1.setdefault(r["video"], []).append(float(r["arm1_lexical"]["picked_in_gold"]))
        video_to_arm2.setdefault(r["video"], []).append(float(r["arm2_clip_text"]["picked_in_gold"]))

    boot_arm1 = _cluster_bootstrap(video_to_arm1, SEED, NUM_BOOT)
    boot_arm2 = _cluster_bootstrap(video_to_arm2, SEED, NUM_BOOT)

    # ── whole-pool context measure (all 406 questions, not just recoverable) ─
    video_to_arm1_all: dict[str, list[float]] = {}
    video_to_arm2_all: dict[str, list[float]] = {}
    for r in per_question:
        video_to_arm1_all.setdefault(r["video"], []).append(float(r["arm1_lexical"]["picked_in_gold"]))
        video_to_arm2_all.setdefault(r["video"], []).append(float(r["arm2_clip_text"]["picked_in_gold"]))
    boot_arm1_all = _cluster_bootstrap(video_to_arm1_all, SEED, NUM_BOOT)
    boot_arm2_all = _cluster_bootstrap(video_to_arm2_all, SEED, NUM_BOOT)

    tie_rate_arm1 = sum(1 for r in recoverable_rows if r["arm1_lexical"]["tied"]) / len(recoverable_rows) if recoverable_rows else float("nan")
    tie_rate_arm2 = sum(1 for r in recoverable_rows if r["arm2_clip_text"]["tied"]) / len(recoverable_rows) if recoverable_rows else float("nan")
    agreement_rate = sum(1 for r in recoverable_rows if r["agree"]) / len(recoverable_rows) if recoverable_rows else float("nan")

    print(f"\n[RECOVERABLE SET] n={len(recoverable_rows)} ({len(recoverable_rows) / len(per_question):.4%} of {len(per_question)})")
    print(f"[ARM1 LEXICAL] recovered_fraction mean={boot_arm1['mean']:.4f} "
          f"CI=[{boot_arm1['ci_lo']:.4f}, {boot_arm1['ci_hi']:.4f}] tie_rate={tie_rate_arm1:.4f}")
    print(f"[ARM2 CLIP-TEXT] recovered_fraction mean={boot_arm2['mean']:.4f} "
          f"CI=[{boot_arm2['ci_lo']:.4f}, {boot_arm2['ci_hi']:.4f}] tie_rate={tie_rate_arm2:.4f}")
    print(f"[AGREEMENT] arm1 == arm2 pick: {agreement_rate:.4f}")
    print(f"[WHOLE POOL CONTEXT] arm1 mean={boot_arm1_all['mean']:.4f} CI=[{boot_arm1_all['ci_lo']:.4f}, {boot_arm1_all['ci_hi']:.4f}]  "
          f"arm2 mean={boot_arm2_all['mean']:.4f} CI=[{boot_arm2_all['ci_lo']:.4f}, {boot_arm2_all['ci_hi']:.4f}]")

    # ── pre-registered gate, read off ARM 1 (lexical) ONLY ──────────────────
    ci_lo = boot_arm1["ci_lo"]
    if ci_lo <= 0.10:
        branch = "NOT_WORTH_A_FRESH_SPLIT_EXPERIMENT"
    elif ci_lo >= 0.30:
        branch = "WORTH_BUILDING_ON_A_NEW_SPLIT"
    else:
        branch = "INCONCLUSIVE"
    print(f"\n[GATE] ARM 1 (lexical) recovered_fraction CI-lower = {ci_lo:.4f} -> {branch}")

    # ── write outputs ────────────────────────────────────────────────────────
    git_commit, git_dirty = git_provenance(REPO)
    provenance = {
        "span_mode": SPAN_MODE,
        "span_half_width": HALF_WIDTH,
        "span_peak_source": PEAK_SOURCE,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash(cfg),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(grounded_rows),
        "n_videos": n_videos,
        "top_k": TOP_K,
        "num_boot": NUM_BOOT,
        "seed": SEED,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "provenance": provenance,
            "recoverable_set": {
                "n": len(recoverable_rows),
                "fraction_of_all": len(recoverable_rows) / len(per_question),
            },
            "caption_availability": {
                "n_pool_frames": n_frames_total,
                "n_with_caption": n_frames_with_caption,
                "fraction": n_frames_with_caption / n_frames_total,
                "n_skipped": n_frames_skipped,
            },
            "bootstrap": {
                "arm1_lexical_recoverable_set": boot_arm1,
                "arm2_clip_text_recoverable_set": boot_arm2,
                "arm1_lexical_whole_pool": boot_arm1_all,
                "arm2_clip_text_whole_pool": boot_arm2_all,
            },
            "tie_rate": {"arm1_lexical": tie_rate_arm1, "arm2_clip_text": tie_rate_arm2},
            "agreement_rate": agreement_rate,
            "gate": {"ci_lower_arm1": ci_lo, "branch": branch},
            "per_question": per_question,
        }, fh, indent=2)
    print(f"\n[LOG] raw dump written to {OUT_JSON}")


if __name__ == "__main__":
    main()
