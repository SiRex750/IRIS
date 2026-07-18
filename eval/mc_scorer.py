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

# Contraction suffixes stripped before tokenizing so "I'd"/"he'd"/"we'd" can't
# leave a bare 'd' that the letter scanner would read as a standalone D token.
_CONTRACTION_RE = re.compile(r"'(?:d|ll|s|re|ve|m)\b", re.IGNORECASE)

_DELIM_PATTERNS = [
    re.compile(r'\(([A-Fa-fXx])\)'),
    re.compile(r'\[([A-Fa-fXx])\]'),
    re.compile(r'\*\*([A-Fa-fXx])\*\*'),
    re.compile(r'\b(?:Answer|Option)\s*:\s*([A-Fa-fXx])\b', re.IGNORECASE),
]

# Cue words that anchor a standalone letter to "this is the final answer".
# Split by direction: a cue's job is to point AT the answer.
#
# FORWARD_ONLY: copulas/labels that take the answer as predicate/object --
# only "cue -> letter" (letter follows the cue) anchors. A letter that
# PRECEDES a forward-only cue must fall through to clean-leading/single/
# abstain, not anchor -- this is what stops "A is wrong" from reading as A.
# correct/wrong are polarity-bearing; this parser does not model polarity
# (no negation/sentiment detection), so per the documented human decision
# they are classified forward-only too, rather than bidirectional -- letting
# "A is wrong" abstain instead of guessing at the polarity.
# NOTE: "be"/"being" are deliberately excluded. Unlike "is"/"was"/"are",
# bare-infinitive "be" is a modal/hedge complement ("could be A", "might be
# B") that weighs a candidate rather than labeling the answer -- including
# it anchored hedges as if they were final ("Could be A, B, or C" -> A).
_CUE_FORWARD_ONLY = {
    "is", "was", "are", "answer", "option", "correct", "wrong",
}

# BIDIRECTIONAL: answer-delivery verbs/prepositions whose object is
# unambiguous regardless of which side the letter falls on.
_CUE_BIDIRECTIONAL = {
    "choose", "chose", "select", "pick", "towards", "with", "go",
    "say", "says", "said", "think", "thinks", "guess",
    "lean", "leaning", "leans",
}

_CUE_WORDS = _CUE_FORWARD_ONLY | _CUE_BIDIRECTIONAL

# Guard for rule (d) CLEAN-LEADING: a bare leading letter is trustworthy only
# if nothing downstream predicates or reverses it. If the leading letter is
# immediately followed by a copula (it's the SUBJECT of "L is/was/are ..."),
# or a polarity token appears anywhere after it, the letter's role can't be
# told apart from a negated/hedged one without modeling polarity -- so (d)
# must not fire. This is what stops "A is wrong" reading as A while still
# letting "answer is B" anchor B (there B is the copula's predicate, reached
# via rule (c), not the subject reached via (d)).
_PREDICATE_FOLLOWERS = {"is", "was", "are", "isn't", "wasn't", "aren't", "'s"}
_POLARITY_TOKENS = {
    "wrong", "incorrect", "not", "no", "false", "isn't", "doesn't", "wouldn't",
}

_STRIP_CHARS = ".,;:!?)(\"'"


def _clean_word(word: str) -> str:
    return word.strip(_STRIP_CHARS)


def _as_letter(word: str) -> str | None:
    core = _clean_word(word).upper()
    return core if core in VALID_CHOICES else None


class ParseResult(tuple):
    """(letter, reason) for existing callers; structured fields for new ones.

    Compares/unpacks as the original 2-tuple so old call sites and tests
    keep working unchanged. New call sites can use .raw_response,
    .parse_path, .abstained to preserve per-question reasoning.
    """

    def __new__(cls, letter, reason, raw_response, parse_path, abstained):
        self = tuple.__new__(cls, (letter, reason))
        self.parsed_letter = letter
        self.reason = reason
        self.raw_response = raw_response
        self.parse_path = parse_path
        self.abstained = abstained
        return self


def _result(letter, reason, raw_response, parse_path) -> ParseResult:
    return ParseResult(letter, reason, raw_response, parse_path, letter is None)


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


def parse_mc_answer(raw: str, opts: dict[str, str]) -> ParseResult:
    """Parse model output to a choice in {A,B,C,D,E,X} or None.

    Anchored-or-clean-leading policy, first match wins:
      (a) preprocess: strip 'd/'ll/'s/'re/'ve/'m contractions so "I'd"/"we'd"
          can't leave a bare 'd' read as a standalone D.
      (b) delimited: a letter in (X), [X], **X**, or after "Answer:"/"Option:".
      (c) anchored: a standalone letter immediately adjacent to a cue word.
          FORWARD_ONLY cues {is, was, are, answer, option, correct, wrong}
          anchor only "cue -> letter" (letter follows); a letter that
          PRECEDES a forward-only cue does NOT anchor ("A is wrong" must not
          read as A). BIDIRECTIONAL cues {choose, chose, select, pick,
          towards, with, go, say, says, said, think, thinks, guess, lean,
          leaning, leans} anchor on either side. Multiple anchored letters
          -> take the LAST. ("be"/"being" are not cues -- bare-infinitive
          "be" hedges a candidate, e.g. "could be A", it doesn't label one.)
      (d) clean-leading: response is short (<=4 whitespace tokens after
          contraction strip), begins with a standalone letter, no other
          distinct standalone letter appears elsewhere (else ambiguous), AND
          the leading letter is not the SUBJECT of a predication: it does
          not fire if the immediately-following token is a copula
          {is, was, are, isn't, wasn't, aren't, 's}, or if any later token is
          a polarity word {wrong, incorrect, not, no, false, isn't, doesn't,
          wouldn't}. This parser does not model polarity, so it can't tell
          "A is wrong" from "A is correct" -- both must abstain rather than
          guess. (Letters that are the PREDICATE of a copula, e.g. the B in
          "the answer is B", are unaffected -- they're caught by rule (c).)
      (e) single-token: the entire response is one standalone letter.
      (f) else -> (None, reason) — abstain, never first-letter-wins.

    X is a valid parsed choice (abstention), distinct from parse-failure (None).
    opts is accepted for call-site compatibility; no longer used for fuzzy
    matching (the anchored-or-clean-leading policy has no fuzzy fallback).
    """
    raw_response = raw
    if not raw:
        return _result(None, "empty response", raw_response, "empty")

    text0 = _CONTRACTION_RE.sub("", raw.strip())

    # (b) DELIMITED — checked on lightly-trimmed text so brackets/asterisks
    # around the letter are still present.
    for pat in _DELIM_PATTERNS:
        m = pat.search(text0)
        if m:
            letter = m.group(1).upper()
            if letter in VALID_CHOICES:
                return _result(letter, None, raw_response, "delimited")

    text = text0.strip(_STRIP_CHARS)
    words = text.split()

    # (c) ANCHORED — standalone letter adjacent to a cue word. "Letter
    # follows cue" (prev word is a cue) is always forward direction, so it
    # anchors for both cue types. "Letter precedes cue" (next word is a
    # cue) is backward direction, so it only anchors for bidirectional cues
    # -- a forward-only cue on the letter's right must NOT anchor it.
    anchored: list[tuple[int, str]] = []
    for i, w in enumerate(words):
        letter = _as_letter(w)
        if letter is None:
            continue
        prev_word = _clean_word(words[i - 1]).lower() if i > 0 else None
        next_word = _clean_word(words[i + 1]).lower() if i + 1 < len(words) else None
        letter_follows_cue = prev_word in _CUE_WORDS
        letter_precedes_bidi_cue = next_word in _CUE_BIDIRECTIONAL
        if letter_follows_cue or letter_precedes_bidi_cue:
            anchored.append((i, letter))

    if anchored:
        return _result(anchored[-1][1], None, raw_response, "anchored")

    # (d) CLEAN-LEADING — short response, begins with a letter, unambiguous,
    # and not the subject of a predication the parser can't read the
    # polarity of (see docstring / _PREDICATE_FOLLOWERS / _POLARITY_TOKENS).
    if words and len(words) <= 4:
        first = _as_letter(words[0])
        if first is not None:
            rest = words[1:]
            immediate_follower = _clean_word(rest[0]).lower() if rest else None
            is_subject_of_predication = immediate_follower in _PREDICATE_FOLLOWERS
            has_downstream_polarity = any(
                _clean_word(w).lower() in _POLARITY_TOKENS for w in rest
            )
            other_letters = {
                letter for w in rest
                if (letter := _as_letter(w)) is not None and letter != first
            }
            if not other_letters and not is_subject_of_predication and not has_downstream_polarity:
                return _result(first, None, raw_response, "clean_leading")

    # (e) SINGLE-TOKEN — the whole response is one standalone letter.
    if len(words) == 1:
        only = _as_letter(words[0])
        if only is not None:
            return _result(only, None, raw_response, "single_token")

    return _result(None, f"no match in: {text[:60]!r}", raw_response, "no_match")


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
        parsed = parse_mc_answer(raw, opts)
        choice = parsed.parsed_letter

        abstain = choice == "X"
        parse_fail = choice is None
        correct = (choice == gold_letter) if (choice is not None and not abstain) else False

        rec = {
            "qid": row["qid"], "video": vid, "family": family,
            "top_k": top_k, "arm": lambda_,
            "pred": choice, "gold": gold_letter,
            "correct": correct, "abstain": abstain, "parse_fail": parse_fail,
            "raw": raw[:120],
            "raw_response": parsed.raw_response,
            "parse_path": parsed.parse_path,
        }
        records.append(rec)
        if results_out is not None:
            results_out.append({k: rec[k] for k in
                ["qid", "family", "top_k", "arm", "pred", "gold",
                 "correct", "abstain", "parse_fail",
                 "raw_response", "parse_path"]})

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
