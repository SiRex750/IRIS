"""
MLVU multiple-choice evaluation harness for IRIS -- codec-mode baseline.

This is a NEW harness that CALLS the existing pipeline (iris.ingest.ingest,
iris.query.query). It does not modify ingest/graph/retrieval/answerer logic.
MC adaptation happens entirely at the harness layer: the numbered options are
folded into the prompt text handed to iris.query.query(), and the harness
parses IRIS's free-text answer back down to a chosen option index.

Scope (per task spec):
  - MLVU dev set, MC tasks only: AR, AO, AC, NQA, PQA, TR (ER + generation
    tasks VS/SSC excluded).
  - Bounded subset: <=25 questions/task, deterministic sample seed=42,
    video duration cap <=600s (skips logged, not silently dropped).
  - codec mode: scene_segmentation="codec", graph_mode="scene_sparse".
  - Per-video ingest caching.
  - Per-task accuracy + M-Avg, random-chance floor per task.
  - Sanity guards: every task must clear its chance floor; option-parse
    failure rate must be <=10%, else abort loud with examples.

Usage:
  python scripts/mlvu_eval.py --mlvu-root /path/to/MLVU_dev
      (expects an annotations dir with per-task JSON files and a videos dir
       -- see --anno-dir/--video-dir to point at them explicitly if they are
       not simply <root>/json and <root>/video)

  python scripts/mlvu_eval.py --self-test
      (synthetic smoke test against local UCF-Crime fixture videos + a mock
       LLM backend; validates harness mechanics without real MLVU data or a
       running answerer backend)
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import av  # noqa: E402

from iris.iris_config import IRISConfig  # noqa: E402
from iris.ingest import ingest  # noqa: E402
from iris.query import query as iris_query  # noqa: E402
import iris.aria as aria  # noqa: E402


# ── Task definitions ─────────────────────────────────────────────────────
MC_TASKS = ["AR", "AO", "AC", "NQA", "PQA", "TR"]
EXCLUDED_TASKS = {"ER": "egocentric (excluded per spec)",
                   "SSC": "generation task (excluded per spec)",
                   "VS": "generation task (excluded per spec)"}

# Keyword substrings used to identify each task's annotation file among the
# JSON files found under --anno-dir (case-insensitive). Order matters: more
# specific keywords first so e.g. "anomaly" doesn't collide with "count".
TASK_KEYWORDS: dict[str, list[str]] = {
    "PQA": ["plotqa", "plot_qa", "plot-qa", "plot"],
    "NQA": ["needle"],
    "AC":  ["count", "action_count", "actioncount"],
    "AO":  ["order", "action_order", "actionorder"],
    "AR":  ["anomaly", "anomaly_reco", "anomalyreco"],
    "TR":  ["topic", "topic_reasoning", "topicreasoning"],
}

TASK_NAMES = {
    "AR": "Anomaly Recognition",
    "AO": "Action Order",
    "AC": "Action Count",
    "NQA": "Needle QA",
    "PQA": "Plot QA",
    "TR": "Topic Reasoning",
}

# Field-name aliases tolerated in annotation records (schema not verified
# against real MLVU/MLVU HF data in this environment -- fail loud, not silently,
# if none of these match).
QUESTION_KEYS = ["question", "question_text", "Q"]
CANDIDATE_KEYS = ["candidates", "options", "choices", "candidate"]
ANSWER_KEYS = ["answer", "correct_answer", "gt", "gt_answer", "label"]
VIDEO_KEYS = ["video", "video_name", "video_path", "video_id", "videoID"]
ID_KEYS = ["question_id", "qid", "id", "quid"]


class HarnessError(RuntimeError):
    """Fail-loud errors specific to this harness (schema mismatch, missing
    files, sanity-guard violations) -- distinct from bugs, always printed
    with enough context to diagnose without re-running."""


# ── Annotation discovery / loading ───────────────────────────────────────

def find_task_file(anno_dir: Path, task_code: str) -> Path:
    all_json = sorted(anno_dir.rglob("*.json"))
    if not all_json:
        raise HarnessError(f"No .json files found under {anno_dir} -- check --anno-dir.")

    keywords = TASK_KEYWORDS[task_code]
    matches = [
        p for p in all_json
        if any(kw in p.stem.lower().replace(" ", "_") for kw in keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        listing = "\n".join(f"  - {p.relative_to(anno_dir)}" for p in all_json)
        raise HarnessError(
            f"Could not find an annotation file for task {task_code} "
            f"({TASK_NAMES[task_code]}) using keywords {keywords}.\n"
            f"Files found under {anno_dir}:\n{listing}\n"
            f"Pass --task-file-{task_code.lower()} explicitly if the naming differs."
        )
    listing = "\n".join(f"  - {p.relative_to(anno_dir)}" for p in matches)
    raise HarnessError(
        f"Ambiguous annotation file for task {task_code}: {len(matches)} candidates "
        f"matched keywords {keywords}:\n{listing}\n"
        f"Pass --task-file-{task_code.lower()} explicitly to disambiguate."
    )


def _first_present(d: dict, keys: list[str]) -> tuple[Any, str | None]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k], k
    return None, None


def _normalize_candidates(raw) -> list[str]:
    """Candidates may arrive as a list of strings, a list already carrying
    'A. text' prefixes, or a dict {'A': 'text', ...}. Normalize to a plain
    ordered list of option strings with any leading 'A) '/'A. ' stripped."""
    if isinstance(raw, dict):
        items = [raw[k] for k in sorted(raw.keys())]
    else:
        items = list(raw)
    cleaned = []
    for opt in items:
        opt = str(opt).strip()
        opt = re.sub(r"^\(?[A-Da-d1-9]\)?[\.\)]\s*", "", opt)
        cleaned.append(opt)
    return cleaned


def _resolve_gold_idx(answer_raw, options: list[str]) -> int:
    """answer_raw may be an option index (0- or 1-based), a letter (A/B/C..),
    or the literal option text. Try each interpretation and require exactly
    one to resolve unambiguously; fail loud otherwise."""
    if isinstance(answer_raw, int):
        if 0 <= answer_raw < len(options):
            return answer_raw
        if 1 <= answer_raw <= len(options):
            return answer_raw - 1
        raise HarnessError(f"Integer answer {answer_raw} out of range for {len(options)} options.")

    s = str(answer_raw).strip()
    # Letter form: "A", "(B)", "C."
    m = re.fullmatch(r"\(?([A-Da-d])\)?\.?", s)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx
        raise HarnessError(f"Letter answer '{s}' out of range for {len(options)} options.")

    # Exact / stripped-prefix text match against options. Tried BEFORE the
    # numeric-index interpretation below: real MLVU answers are always the
    # literal option text, and for tasks like Action Count the option text
    # is itself a bare digit (e.g. options ["2","1","5","4"], answer "1"
    # meaning the text "1", not index 1) -- an exact text match must win
    # over misreading that digit as a 0/1-based index.
    stripped = re.sub(r"^\(?[A-Da-d1-9]\)?[\.\)]\s*", "", s).strip()
    text_matches = [i for i, opt in enumerate(options) if opt.strip() == s or opt.strip() == stripped]
    if len(text_matches) == 1:
        return text_matches[0]
    if len(text_matches) > 1:
        # Benign source-data quirk (seen in real MLVU AR records): the
        # candidate list itself repeats identical distractor text at more
        # than one index. Since the slots are textually indistinguishable,
        # any one is an equally valid "gold" choice for MC scoring -- pick
        # the first occurrence deterministically rather than fail loud.
        print(f"[warn] gold answer '{answer_raw}' matches options at duplicate indices "
              f"{text_matches} (options={options}); using first occurrence.", file=sys.stderr)
        return text_matches[0]

    # Numeric string form (fallback: only reached if the answer text did not
    # exactly match any option's literal text).
    if s.isdigit():
        n = int(s)
        if 0 <= n < len(options):
            return n
        if 1 <= n <= len(options):
            return n - 1
        raise HarnessError(f"Numeric answer '{s}' out of range for {len(options)} options.")

    raise HarnessError(
        f"Could not resolve gold answer '{answer_raw}' against options {options}."
    )


def load_task_questions(anno_path: Path, task_code: str) -> list[dict]:
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Some MLVU-style exports wrap the list under a top-level key.
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise HarnessError(f"{anno_path}: expected a JSON list of question records, got {type(data)}.")

    out = []
    field_log: dict[str, str] = {}
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise HarnessError(f"{anno_path}[{i}]: expected a dict record, got {type(rec)}.")

        q, qk = _first_present(rec, QUESTION_KEYS)
        cands, ck = _first_present(rec, CANDIDATE_KEYS)
        ans, ak = _first_present(rec, ANSWER_KEYS)
        vid, vk = _first_present(rec, VIDEO_KEYS)
        qid, _ = _first_present(rec, ID_KEYS)

        missing = [name for name, val in
                   [("question", q), ("candidates", cands), ("answer", ans), ("video", vid)]
                   if val is None]
        if missing:
            raise HarnessError(
                f"{anno_path}[{i}]: record missing required field(s) {missing}. "
                f"Record keys present: {sorted(rec.keys())}. "
                f"Expected one of {QUESTION_KEYS}/{CANDIDATE_KEYS}/{ANSWER_KEYS}/{VIDEO_KEYS} "
                f"per field -- update the *_KEYS alias lists in scripts/mlvu_eval.py if the "
                f"real MLVU schema names these differently."
            )
        field_log.update({"question": qk, "candidates": ck, "answer": ak, "video": vk})

        options = _normalize_candidates(cands)
        gold_idx = _resolve_gold_idx(ans, options)

        out.append({
            "task": task_code,
            "question_id": qid if qid is not None else f"{task_code}_{i}",
            "question": str(q).strip(),
            "options": options,
            "gold_idx": gold_idx,
            "video": str(vid),
        })

    print(f"[load] {task_code}: {len(out)} records from {anno_path.name} "
          f"(field mapping: {field_log})")
    return out


# ── Video resolution / duration probing ──────────────────────────────────

def build_video_index(video_dir: Path) -> dict[str, Path]:
    """Map bare filename (and filename w/o extension) -> full path, so
    annotation 'video' fields that are bare names or relative paths both
    resolve regardless of the real on-disk directory layout."""
    index: dict[str, Path] = {}
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    for p in video_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            index[p.name] = p
            index[p.stem] = p
    return index


def resolve_video_path(video_field: str, video_dir: Path, video_index: dict[str, Path]) -> Path | None:
    direct = (video_dir / video_field)
    if direct.exists():
        return direct
    name = Path(video_field).name
    if name in video_index:
        return video_index[name]
    stem = Path(video_field).stem
    if stem in video_index:
        return video_index[stem]
    return None


def probe_duration_seconds(video_path: Path) -> float | None:
    try:
        container = av.open(str(video_path))
        try:
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0  # AV_TIME_BASE
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
        finally:
            container.close()
    except Exception as e:
        print(f"[warn] duration probe failed for {video_path}: {e}", file=sys.stderr)
        return None
    return None


# ── MC prompt construction / answer parsing ───────────────────────────────

def build_mc_prompt(question: str, options: list[str]) -> str:
    lines = [question, "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"{i + 1}) {opt}")
    lines.append("")
    lines.append(f"Respond with ONLY the number (1-{len(options)}) of the correct option.")
    return "\n".join(lines)


_NUM_PATTERNS = [
    (re.compile(r"^\s*\(?([1-9])\)?[\.\):]?\s*$"), "direct_number"),
    (re.compile(r"\banswer is[:\s]*\(?([1-9])\)?", re.IGNORECASE), "answer_is_number"),
    (re.compile(r"\bcorrect option is[:\s]*\(?([1-9])\)?", re.IGNORECASE), "correct_option_is_number"),
    (re.compile(r"\boption\s*\(?([1-9])\)?\b", re.IGNORECASE), "option_number"),
    (re.compile(r"^\s*\(?([1-9])\)?[\.\):]"), "leading_number"),
]
_LETTER_PATTERNS = [
    (re.compile(r"^\s*\(?([A-Da-d])\)?[\.\):]?\s*$"), "direct_letter"),
    (re.compile(r"\banswer is[:\s]*\(?([A-Da-d])\)?", re.IGNORECASE), "answer_is_letter"),
    (re.compile(r"\bcorrect option is[:\s]*\(?([A-Da-d])\)?", re.IGNORECASE), "correct_option_is_letter"),
    (re.compile(r"\boption\s*\(?([A-Da-d])\)?\b", re.IGNORECASE), "option_letter"),
    (re.compile(r"^\s*\(?([A-Da-d])\)?[\.\):]"), "leading_letter"),
]

SIMILARITY_FALLBACK_THRESHOLD = 0.2


def parse_option(raw_answer: str, options: list[str]) -> tuple[int | None, str]:
    """Returns (option_idx_or_None, method). method is logged verbatim into
    the per-question record so the parse-mapping method is auditable."""
    if not raw_answer:
        return None, "empty_answer"

    text = raw_answer.strip()
    n = len(options)

    for pattern, method in _NUM_PATTERNS:
        m = pattern.search(text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < n:
                return idx, method

    for pattern, method in _LETTER_PATTERNS:
        m = pattern.search(text)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < n:
                return idx, method

    # Fallback: nearest-option text similarity (handles IRIS answering in
    # open-ended prose that restates/paraphrases an option rather than
    # citing its number).
    best_idx, best_ratio = None, 0.0
    text_lower = text.lower()
    for i, opt in enumerate(options):
        ratio = difflib.SequenceMatcher(None, text_lower, opt.lower()).ratio()
        # Also reward substring containment (paraphrase / partial echo of option text).
        if opt.lower() in text_lower:
            ratio = max(ratio, 0.5)
        if ratio > best_ratio:
            best_idx, best_ratio = i, ratio
    if best_idx is not None and best_ratio >= SIMILARITY_FALLBACK_THRESHOLD:
        return best_idx, f"similarity_fallback(ratio={best_ratio:.2f})"

    return None, "unparsed"


# ── Provenance ─────────────────────────────────────────────────────────

def git_provenance() -> dict:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception as e:
        head = f"<unavailable: {e}>"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT
        ).decode()
        dirty = bool(status.strip())
    except Exception as e:
        dirty = None
        status = f"<unavailable: {e}>"
    return {"head": head, "dirty": dirty, "status_raw": status if dirty else ""}


def config_hash(config: IRISConfig) -> str:
    blob = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ── Sampling ──────────────────────────────────────────────────────────

def deterministic_sample(pool: list[dict], n: int, seed: int) -> list[dict]:
    # Sort by a stable, filesystem-order-independent key before sampling so
    # the sample is reproducible regardless of JSON record order on disk.
    ordered = sorted(pool, key=lambda r: (r["video"], r["question_id"], r["question"]))
    if len(ordered) <= n:
        return ordered
    rng = random.Random(seed)
    return rng.sample(ordered, n)


# ── Main eval loop ────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path | None) -> tuple[list[dict], list[dict]]:
    """Returns (results, ingest_failures) loaded from a prior interrupted run,
    or ([], []) if no checkpoint exists yet. Used to resume after the process
    is killed mid-run (observed: this environment kills long-lived background
    jobs unpredictably, sometimes 1-2 hours in -- checkpointing after every
    question means a kill loses at most one question's work, not the whole run)."""
    if checkpoint_path is None or not checkpoint_path.exists():
        return [], []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    ingest_failures = data.get("ingest_failures", [])
    print(f"[checkpoint] resuming from {checkpoint_path}: {len(results)} questions already "
          f"answered, {len(ingest_failures)} prior ingest failures")
    return results, ingest_failures


def save_checkpoint(checkpoint_path: Path | None, results: list[dict], ingest_failures: list[dict]) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"results": results, "ingest_failures": ingest_failures}, f, indent=2, default=str)
    tmp.replace(checkpoint_path)  # atomic on both POSIX and Windows


def run_eval(
    anno_dir: Path,
    video_dir: Path,
    n_per_task: int,
    max_duration: float,
    seed: int,
    out_prefix: Path,
    task_files: dict[str, Path] | None = None,
    checkpoint_path: Path | None = None,
) -> dict:
    config = IRISConfig(scene_segmentation="codec", graph_mode="scene_sparse")
    config.validate()

    video_index = build_video_index(video_dir)
    print(f"[setup] indexed {len(video_index)} candidate video filenames under {video_dir}")

    # 1. Load + filter + sample per task
    per_task_pool: dict[str, list[dict]] = {}
    skipped_by_length: dict[str, int] = {}
    skipped_no_video: dict[str, int] = {}
    duration_probe_failed: dict[str, int] = {}

    for task in MC_TASKS:
        anno_path = (task_files or {}).get(task) or find_task_file(anno_dir, task)
        records = load_task_questions(anno_path, task)

        kept = []
        skipped_by_length[task] = 0
        skipped_no_video[task] = 0
        duration_probe_failed[task] = 0
        # cache duration per video within this task loop to avoid re-probing
        dur_cache: dict[str, float | None] = {}
        for r in records:
            vpath = resolve_video_path(r["video"], video_dir, video_index)
            if vpath is None:
                skipped_no_video[task] += 1
                continue
            key = str(vpath)
            if key not in dur_cache:
                dur_cache[key] = probe_duration_seconds(vpath)
            dur = dur_cache[key]
            if dur is None:
                duration_probe_failed[task] += 1
                continue
            if dur > max_duration:
                skipped_by_length[task] += 1
                continue
            r["video_path"] = key
            r["video_duration_s"] = dur
            kept.append(r)

        print(f"[filter] {task}: {len(records)} loaded -> {len(kept)} <= {max_duration}s "
              f"(skipped_by_length={skipped_by_length[task]}, "
              f"skipped_no_video={skipped_no_video[task]}, "
              f"duration_probe_failed={duration_probe_failed[task]})")

        sampled = deterministic_sample(kept, n_per_task, seed)
        print(f"[sample] {task}: sampled {len(sampled)}/{len(kept)} (n_per_task={n_per_task}, seed={seed})")
        per_task_pool[task] = sampled

    # 2. Ingest cache
    ingest_cache: dict[str, Any] = {}
    ingest_failures: list[dict] = []

    def get_index(video_path: str):
        if video_path not in ingest_cache:
            t0 = time.time()
            print(f"[ingest] {video_path} ...")
            try:
                ingest_cache[video_path] = ingest(video_path, config=config)
                print(f"[ingest] {video_path} done in {time.time() - t0:.1f}s")
            except Exception as e:
                print(f"[ingest] FAILED for {video_path}: {e}", file=sys.stderr)
                ingest_cache[video_path] = None
        return ingest_cache[video_path]

    # 3. Run questions
    results, ingest_failures = load_checkpoint(checkpoint_path)
    done_keys = {(x["task"], x["question_id"]) for x in results}
    done_keys |= {(x["task"], x["question_id"]) for x in ingest_failures}
    if done_keys:
        print(f"[checkpoint] {len(done_keys)} (task, question_id) pairs already resolved -- skipping those")

    for task in MC_TASKS:
        for r in per_task_pool[task]:
            if (task, r["question_id"]) in done_keys:
                continue

            idx = get_index(r["video_path"])
            if idx is None:
                ingest_failures.append({"task": task, "video": r["video_path"], "question_id": r["question_id"]})
                save_checkpoint(checkpoint_path, results, ingest_failures)
                continue

            mc_prompt = build_mc_prompt(r["question"], r["options"])
            t0 = time.time()
            try:
                qres = iris_query(mc_prompt, idx, config)
                raw_answer = qres.get("raw_answer", "") or ""
            except Exception as e:
                print(f"[query] FAILED for {task}/{r['question_id']}: {e}", file=sys.stderr)
                raw_answer = ""
                qres = {}
            elapsed = time.time() - t0

            pred_idx, method = parse_option(raw_answer, r["options"])
            correct = (pred_idx is not None and pred_idx == r["gold_idx"])

            results.append({
                "task": task,
                "question_id": r["question_id"],
                "video": r["video_path"],
                "video_duration_s": r["video_duration_s"],
                "question": r["question"],
                "options": r["options"],
                "gold_idx": r["gold_idx"],
                "mc_prompt": mc_prompt,
                "raw_answer": raw_answer,
                "final_answer": qres.get("answer", ""),
                "pred_idx": pred_idx,
                "parse_method": method,
                "correct": correct,
                "elapsed_s": elapsed,
            })
            print(f"[query] {task}/{r['question_id']}: gold={r['gold_idx']} pred={pred_idx} "
                  f"({method}) correct={correct} [{elapsed:.1f}s]")
            save_checkpoint(checkpoint_path, results, ingest_failures)

    # 4. Scoring
    per_task_acc: dict[str, dict] = {}
    for task in MC_TASKS:
        task_results = [x for x in results if x["task"] == task]
        n = len(task_results)
        n_correct = sum(1 for x in task_results if x["correct"])
        acc = (n_correct / n) if n else 0.0
        floor = (sum(1.0 / len(x["options"]) for x in task_results) / n) if n else 0.0
        n_parsed_fail = sum(1 for x in task_results if x["pred_idx"] is None)
        per_task_acc[task] = {
            "task_name": TASK_NAMES[task],
            "n": n,
            "n_correct": n_correct,
            "accuracy": acc,
            "random_floor": floor,
            "above_floor": acc > floor if n else None,
            "n_parse_failed": n_parsed_fail,
            "parse_failure_rate": (n_parsed_fail / n) if n else 0.0,
            "skipped_by_length": skipped_by_length[task],
            "skipped_no_video": skipped_no_video[task],
            "duration_probe_failed": duration_probe_failed[task],
        }

    scored_tasks = [t for t in MC_TASKS if per_task_acc[t]["n"] > 0]
    m_avg = (sum(per_task_acc[t]["accuracy"] for t in scored_tasks) / len(scored_tasks)) if scored_tasks else 0.0

    total_n = len(results)
    total_parse_fail = sum(1 for x in results if x["pred_idx"] is None)
    overall_parse_failure_rate = (total_parse_fail / total_n) if total_n else 0.0

    summary = {
        "m_avg": m_avg,
        "n_total": total_n,
        "overall_parse_failure_rate": overall_parse_failure_rate,
        "n_ingest_failures": len(ingest_failures),
        "ingest_failures": ingest_failures,
        "per_task": per_task_acc,
        "excluded_tasks": EXCLUDED_TASKS,
        "config": asdict(config),
        "config_hash": config_hash(config),
        "provenance": git_provenance(),
        "params": {
            "n_per_task": n_per_task,
            "max_duration_s": max_duration,
            "seed": seed,
        },
    }

    report = {"summary": summary, "results": results}

    # 5. Write outputs
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    write_markdown(md_path, summary, results)

    print("\n" + "=" * 100)
    print("FULL JSON REPORT")
    print("=" * 100)
    print(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 100)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("=" * 100)

    # 6. Sanity guards (assert, fail loud)
    violations = [t for t in scored_tasks if not per_task_acc[t]["above_floor"]]
    if violations:
        print("\n" + "!" * 100)
        print("SANITY GUARD FAILURE: task(s) at/below random-chance floor:")
        for t in violations:
            print(f"\n--- {t} ({TASK_NAMES[t]}): acc={per_task_acc[t]['accuracy']:.3f} "
                  f"floor={per_task_acc[t]['random_floor']:.3f} ---")
            examples = [x for x in results if x["task"] == t][:3]
            for ex in examples:
                print(f"  question_id={ex['question_id']}")
                print(f"  question: {ex['question']}")
                print(f"  prompt sent:\n{ex['mc_prompt']}")
                print(f"  raw_answer: {ex['raw_answer']!r}")
                print(f"  parsed pred_idx={ex['pred_idx']} (method={ex['parse_method']}) gold_idx={ex['gold_idx']}")
                print("  ---")
        print("!" * 100)
        raise HarnessError(
            f"Sanity guard failed: task(s) {violations} did not clear their random-chance floor. "
            "Pipeline or MC-parsing is likely broken -- see examples above."
        )

    if overall_parse_failure_rate > 0.10:
        print("\n" + "!" * 100)
        print(f"SANITY GUARD FAILURE: option-parse failure rate {overall_parse_failure_rate:.1%} > 10%")
        failed_examples = [x for x in results if x["pred_idx"] is None][:5]
        for ex in failed_examples:
            print(f"  task={ex['task']} question_id={ex['question_id']}")
            print(f"  raw_answer: {ex['raw_answer']!r}")
            print("  ---")
        print("!" * 100)
        raise HarnessError(
            f"Sanity guard failed: option-parse failure rate {overall_parse_failure_rate:.1%} exceeds 10% threshold."
        )

    print(f"\n[OK] All {len(scored_tasks)} scored tasks cleared their random-chance floor. "
          f"Parse failure rate {overall_parse_failure_rate:.1%} <= 10%. M-Avg = {m_avg:.3f}")

    return report


def write_markdown(path: Path, summary: dict, results: list[dict]) -> None:
    lines = []
    lines.append("# MLVU codec-mode baseline (IRIS)")
    lines.append("")
    prov = summary["provenance"]
    lines.append(f"- repo HEAD: `{prov['head']}` (dirty={prov['dirty']})")
    lines.append(f"- config hash: `{summary['config_hash']}`")
    p = summary["params"]
    lines.append(f"- n_per_task={p['n_per_task']}, max_duration_s={p['max_duration_s']}, seed={p['seed']}")
    lines.append(f"- scene_segmentation=`{summary['config']['scene_segmentation']}`, "
                 f"graph_mode=`{summary['config']['graph_mode']}`")
    lines.append("")
    lines.append(f"**M-Avg: {summary['m_avg']:.3f}**  (unweighted mean of per-task accuracy, N={summary['n_total']})")
    lines.append("")
    lines.append(f"Overall option-parse failure rate: {summary['overall_parse_failure_rate']:.1%}  "
                 f"(threshold: 10%, abort if exceeded)")
    lines.append(f"Ingest failures: {summary['n_ingest_failures']}")
    lines.append("")
    lines.append("## Per-task results")
    lines.append("")
    lines.append("| Task | Name | N | Correct | Accuracy | Random floor | Above floor | "
                 "Parse-fail rate | Skipped (length) | Skipped (no video) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for t, d in summary["per_task"].items():
        lines.append(
            f"| {t} | {d['task_name']} | {d['n']} | {d['n_correct']} | {d['accuracy']:.3f} | "
            f"{d['random_floor']:.3f} | {d['above_floor']} | {d['parse_failure_rate']:.1%} | "
            f"{d['skipped_by_length']} | {d['skipped_no_video']} |"
        )
    lines.append("")
    lines.append("## Excluded tasks")
    lines.append("")
    for t, reason in summary["excluded_tasks"].items():
        lines.append(f"- {t}: {reason}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Self-test (synthetic, no real MLVU data required) ────────────────────

class _MockMCBackend:
    """Deterministic mock LLM backend for --self-test: always answers with
    the option index that matches a marker embedded in the fixture question,
    so the harness plumbing (ingest -> query -> MC prompt -> parse -> score)
    can be validated end-to-end without a running answerer server."""

    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        m = re.search(r"CORRECT_OPTION=(\d+)", prompt)
        if m:
            return f"The answer is {m.group(1)}."
        return "I am not sure."


def run_self_test(out_prefix: Path) -> None:
    fixture_video = REPO_ROOT / "eval" / "data" / "ucf" / "videos" / "Anomaly-Videos-Part-1" / "Abuse" / "Abuse029_x264.mp4"
    if not fixture_video.exists():
        raise HarnessError(f"Self-test fixture video not found: {fixture_video}")

    tmp = REPO_ROOT / "eval_results" / "_mlvu_selftest_tmp"
    anno_dir = tmp / "json"
    video_dir = tmp / "video"
    anno_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    local_video = video_dir / "Abuse029_x264.mp4"
    if not local_video.exists():
        shutil.copy(fixture_video, local_video)

    task_filenames = {
        "PQA": "1_plotQA.json", "NQA": "2_needle.json", "AC": "4_count.json",
        "AO": "5_order.json", "AR": "6_anomaly_reco.json", "TR": "7_topic_reasoning.json",
    }
    n_q = 3
    for task, fname in task_filenames.items():
        records = []
        for i in range(n_q):
            correct = i % 4
            options = [f"option {j} for {task} q{i}" for j in range(4)]
            question = f"[{task} self-test q{i}] CORRECT_OPTION={correct + 1} What happens in the video?"
            records.append({
                "video": "Abuse029_x264.mp4",
                "question": question,
                "candidates": options,
                "answer": correct,
                "question_id": f"{task}_{i}",
            })
        with open(anno_dir / fname, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    aria.set_backend(_MockMCBackend())
    print("[self-test] installed _MockMCBackend via aria.set_backend()")

    report = run_eval(
        anno_dir=anno_dir,
        video_dir=video_dir,
        n_per_task=n_q,
        max_duration=600.0,
        seed=42,
        out_prefix=out_prefix,
    )
    assert report["summary"]["m_avg"] == 1.0, (
        f"Self-test expected perfect M-Avg with the deterministic mock backend, "
        f"got {report['summary']['m_avg']}"
    )
    print("[self-test] PASSED (harness plumbing verified end-to-end)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlvu-root", type=Path, default=None,
                     help="Root dir containing json/ and video/ subdirs (MLVU dev layout).")
    ap.add_argument("--anno-dir", type=Path, default=None, help="Override annotation dir.")
    ap.add_argument("--video-dir", type=Path, default=None, help="Override video dir.")
    ap.add_argument("--n-per-task", type=int, default=25)
    ap.add_argument("--max-duration", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=Path, default=REPO_ROOT / "eval_results" / "MLVU_codec_baseline")
    ap.add_argument("--self-test", action="store_true",
                     help="Run a synthetic smoke test (no real MLVU data / answerer server required).")
    ap.add_argument("--answerer-endpoint", type=str, default="http://localhost:11434/v1",
                     help="OpenAI-compatible endpoint for the local LLM backend (Ollama by default). "
                          "NOTE: aria.LlamaBackend's built-in default model is a stale 'llama3.2:3b' "
                          "that is not wired to IRISConfig.answerer_model in cerberus_mode='legacy' "
                          "(this harness's mode) -- so this script explicitly constructs and installs "
                          "the backend itself rather than relying on aria.get_backend()'s defaults.")
    ap.add_argument("--answerer-model", type=str, default="granite4:micro",
                     help="Model name to request from --answerer-endpoint.")
    ap.add_argument("--checkpoint-path", type=Path, default=None,
                     help="Path to a checkpoint JSON, written after every question and reloaded "
                          "on startup so an interrupted run resumes instead of restarting "
                          "(defaults to <out-prefix>_checkpoint.json).")
    ap.add_argument("--no-checkpoint", action="store_true",
                     help="Disable checkpointing entirely.")
    args = ap.parse_args()

    if args.self_test:
        run_self_test(args.out_prefix.parent / "_MLVU_selftest")
        return

    aria.set_backend(aria.LlamaBackend(endpoint=args.answerer_endpoint, text_model=args.answerer_model))
    print(f"[setup] answerer backend: LlamaBackend(endpoint={args.answerer_endpoint!r}, "
          f"text_model={args.answerer_model!r})")

    if args.anno_dir is None or args.video_dir is None:
        if args.mlvu_root is None:
            ap.error("Provide --mlvu-root, or both --anno-dir and --video-dir, or --self-test.")
        anno_dir = args.anno_dir or (args.mlvu_root / "json")
        video_dir = args.video_dir or (args.mlvu_root / "video")
    else:
        anno_dir, video_dir = args.anno_dir, args.video_dir

    if not anno_dir.exists():
        raise HarnessError(f"Annotation dir does not exist: {anno_dir}")
    if not video_dir.exists():
        raise HarnessError(f"Video dir does not exist: {video_dir}")

    checkpoint_path = None if args.no_checkpoint else (
        args.checkpoint_path or args.out_prefix.parent / f"{args.out_prefix.name}_checkpoint.json"
    )

    run_eval(
        anno_dir=anno_dir,
        video_dir=video_dir,
        n_per_task=args.n_per_task,
        max_duration=args.max_duration,
        seed=args.seed,
        out_prefix=args.out_prefix,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
