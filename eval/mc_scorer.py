"""MC scorer for NExT-QA 5-way multiple-choice evaluation.

build_mc_prompt  -> (prompt, context) for aria.generate
parse_mc_answer  -> (choice|None, reason|None)
score_arm        -> per-family + overall accuracy/abstain/parse_fail stats
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
import iris.aria as aria
from iris.iris_config import IRISConfig

LETTERS = ["A", "B", "C", "D", "E"]
VALID_CHOICES = set(LETTERS) | {"X"}

# Fuzzy-match threshold: fraction of option text that must appear in response
_FUZZY_THRESHOLD = 0.6


def build_mc_prompt(
    question: str,
    opts: dict[str, str],   # {"a0": "...", "a1": "...", ...}
    caption_context: str,
) -> tuple[str, str]:
    """Build (prompt, context) for aria.generate.

    Context = retrieved frames' captions, numbered.
    Prompt = question + lettered options + grounding instruction.
    Demands a single token A/B/C/D/E/X; X = insufficient evidence.
    """
    context = caption_context.strip() if caption_context else "(no captions retrieved)"

    lines = [f"Question: {question}", "", "Options:"]
    for i, letter in enumerate(LETTERS):
        key = f"a{i}"
        lines.append(f"  {letter}. {opts.get(key, '')}")

    lines += [
        "",
        "Instructions: Answer using ONLY the provided frame captions above.",
        "If the captions do not support any option, answer X (insufficient evidence).",
        "Reply with a SINGLE letter: A, B, C, D, E, or X. Nothing else.",
    ]
    prompt = "\n".join(lines)
    return prompt, context


def parse_mc_answer(raw: str, opts: dict[str, str]) -> tuple[str | None, str | None]:
    """Parse model output to a choice in {A,B,C,D,E,X} or None.

    Strategy:
      (a) First standalone token matching a valid choice (case-insensitive).
      (b) Fuzzy-match: if emitted text overlaps clearly with one option string.
      (c) None + reason string — never random-guess.

    X is a valid parsed choice (abstention), distinct from parse-failure (None).
    """
    if not raw:
        return None, "empty response"

    # (a) First standalone letter token
    tokens = re.findall(r'\b([A-Fa-fXx])\b', raw)
    for tok in tokens:
        upper = tok.upper()
        if upper in VALID_CHOICES:
            return upper, None

    # (b) Fuzzy match against option text
    raw_lower = raw.lower()
    best_letter, best_score = None, 0.0
    for i, letter in enumerate(LETTERS):
        opt_text = opts.get(f"a{i}", "").lower().strip()
        if not opt_text:
            continue
        words = opt_text.split()
        if not words:
            continue
        hits = sum(1 for w in words if w in raw_lower)
        score = hits / len(words)
        if score > best_score:
            best_score = score
            best_letter = letter

    if best_score >= _FUZZY_THRESHOLD and best_letter is not None:
        return best_letter, f"fuzzy({best_score:.2f})"

    return None, f"no match in: {raw[:60]!r}"


def score_arm(
    questions: list[dict],
    cache_dir: Path,
    config: IRISConfig,
    results_out: list[dict] | None = None,
) -> dict:
    """Run one ablation arm over cached questions.

    questions: list of dicts from dev_100.jsonl (filtered to cached videos).
    config: IRISConfig with ranking_mode="ppr", ppr_lambda set to arm value.
    results_out: if provided, per-question dicts appended here for JSONL write.

    Returns: {
        "overall": {acc, abstain_pct, parse_fail_pct, n, n_answered, n_abstain, n_fail},
        "by_family": {family: same dict},
    }
    """
    top_k = config.l2_retrieve_top_k
    lambda_ = config.ppr_lambda

    # Load each cached index once per video
    loaded: dict[str, Any] = {}
    for row in questions:
        vid = row["video"]
        if vid in loaded:
            continue
        npz = cache_dir / f"{vid}.npz"
        if npz.exists():
            try:
                loaded[vid] = iris_ingest.load_index(cache_dir / vid)
            except Exception as e:
                loaded[vid] = None
                print(f"  LOAD_ERR {vid}: {e}", flush=True)

    records: list[dict] = []
    for row in questions:
        vid = row["video"]
        index = loaded.get(vid)
        if index is None:
            continue

        q = row["question"]
        opts = {k: row[k] for k in ["a0", "a1", "a2", "a3", "a4"]}
        gold_idx = int(row["answer"])
        gold_letter = LETTERS[gold_idx]
        family = row["family"]

        emb = iris_query._embed_query(q, config)
        retrieved = iris_query._build_retrieved(index, emb, config)

        # Build caption context from retrieved frames
        caption_lines = []
        for i, f in enumerate(retrieved, 1):
            cap = f.get("caption") or ""
            ts = f.get("timestamp", 0.0)
            caption_lines.append(f"[Frame {i} @ {ts:.1f}s] {cap}")
        caption_context = "\n".join(caption_lines)

        prompt, context = build_mc_prompt(q, opts, caption_context)
        raw = aria.generate(prompt=prompt, context=context)
        choice, parse_reason = parse_mc_answer(raw, opts)

        abstain = choice == "X"
        parse_fail = choice is None
        correct = (choice == gold_letter) if (choice is not None and not abstain) else False

        rec = {
            "qid": row["qid"], "video": vid, "family": family,
            "top_k": top_k, "arm": lambda_,
            "pred": choice, "gold": gold_letter,
            "correct": correct, "abstain": abstain, "parse_fail": parse_fail,
            "raw": raw[:120],
        }
        records.append(rec)
        if results_out is not None:
            results_out.append({k: rec[k] for k in
                ["qid", "family", "top_k", "arm", "pred", "gold",
                 "correct", "abstain", "parse_fail"]})

    def _stats(recs: list[dict]) -> dict:
        n = len(recs)
        if n == 0:
            return {"acc": None, "abstain_pct": None, "parse_fail_pct": None,
                    "n": 0, "n_answered": 0, "n_abstain": 0, "n_fail": 0}
        n_fail    = sum(1 for r in recs if r["parse_fail"])
        n_abstain = sum(1 for r in recs if r["abstain"])
        n_answered = n - n_fail - n_abstain
        n_correct  = sum(1 for r in recs if r["correct"])
        acc = n_correct / n_answered if n_answered > 0 else None
        return {
            "acc": acc,
            "abstain_pct": n_abstain / n,
            "parse_fail_pct": n_fail / n,
            "n": n, "n_answered": n_answered,
            "n_abstain": n_abstain, "n_fail": n_fail,
        }

    families = sorted({r["family"] for r in records})
    return {
        "overall": _stats(records),
        "by_family": {fam: _stats([r for r in records if r["family"] == fam])
                      for fam in families},
    }
