"""QA debug trace mode for the IRIS pipeline.

Instrumentation only: nothing in this module changes what iris.query.query()
retrieves, captions, prompts, or verifies. Every capture point in query.py is
an additive parameter (trace=None by default) that this module's DebugTrace
object fills in; the actual pipeline call sites and their control flow are
byte-for-byte unchanged whether or not a trace is being collected (see
tests/test_debug_trace.py::test_answers_identical_with_trace_enabled).

Enable via IRISConfig.debug_trace = True (default False -> zero overhead:
every call site is a single `if config.debug_trace:` / `trace is not None`
check, no extra work is queued when disabled).

Output layout, one directory per query:
    debug_traces/<query_id>/
        frames/frame_001.jpg, frame_002.jpg, ...
        frames_overview.jpg
        granite_prompt.txt
        trace.json
        summary.md
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def make_query_id(video_id: str, question_id: Any = None, question: str = "") -> str:
    """Deterministic, filesystem-safe query id. Prefers video_id+question_id
    (stable across repeated runs of the same benchmark question, so re-running
    overwrites rather than accumulating duplicate directories); falls back to
    a short hash of the question text plus a timestamp for ad-hoc/manual
    queries that have no question_id."""
    safe_video = re.sub(r"[^A-Za-z0-9_.-]", "_", str(video_id) or "unknown_video")
    if question_id is not None:
        safe_qid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(question_id))
        return f"{safe_video}__q{safe_qid}"
    import hashlib
    qhash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{safe_video}__adhoc_{qhash}_{ts}"


def _extract_frame_images(video_path: str, targets: list[tuple[int, float]]) -> dict[int, Any]:
    """Read-only frame image extraction for visual inspection (task step 4).

    targets: list of (frame_idx, timestamp_seconds). Returns {frame_idx: PIL.Image}.
    Never touches FrameRecord.caption or any pipeline state -- opens its own
    independent container handle and only reads pixels. Uses the same single
    seek + one continuous forward decode pass pattern as
    iris.query._ensure_captions (GOP-aware batching), for the same reason:
    avoids redundantly re-decoding the same GOP prefix once per target frame.
    """
    import av

    images: dict[int, Any] = {}
    if not targets:
        return images

    ordered = sorted(targets, key=lambda t: t[1])
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        tolerance = 0.5 / fps

        first_pts = int(round(ordered[0][1] / stream.time_base))
        container.seek(first_pts, stream=stream)

        target_iter = iter(ordered)
        current = next(target_iter, None)

        for frame in container.decode(stream):
            if current is None:
                break
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            while current is not None and frame_time >= current[1] - tolerance:
                frame_idx, _ts = current
                try:
                    images[frame_idx] = frame.to_image()
                except Exception:
                    pass
                current = next(target_iter, None)
    finally:
        container.close()
    return images


def _build_contact_sheet(frame_entries: list[dict], images: dict[int, Any]):
    """Grid of retrieved frames in timestamp order, each labeled with
    frame_id / timestamp / scene_id underneath (task step 13). Returns a
    PIL.Image, or None if no images could be extracted (never raises --
    this is a nice-to-have visualization, not a required trace field)."""
    from PIL import Image, ImageDraw

    entries = sorted(frame_entries, key=lambda e: e["timestamp"])
    entries = [e for e in entries if e["frame_id"] in images]
    if not entries:
        return None

    thumb_w, thumb_h = 240, 180
    label_h = 54
    cols = min(4, len(entries))
    rows = (len(entries) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for i, entry in enumerate(entries):
        img = images[entry["frame_id"]].convert("RGB").resize((thumb_w, thumb_h))
        col, row = i % cols, i // cols
        x, y = col * thumb_w, row * (thumb_h + label_h)
        sheet.paste(img, (x, y))
        label = (
            f"frame {entry['frame_id']}\n"
            f"t={entry['timestamp']:.2f}s\n"
            f"scene={entry.get('scene_id')}"
        )
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(24, 24, 24))
        draw.multiline_text((x + 6, y + thumb_h + 4), label, fill=(255, 255, 255))
    return sheet


@dataclass
class DebugTrace:
    query_id: str
    out_dir: Path
    video_id: str
    question: str
    question_type: str | None = None
    question_id: Any = None
    split: str | None = None

    ground_truth: dict = field(default_factory=lambda: {"answer": None, "options": None})
    retrieval: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)
    captions: list = field(default_factory=list)
    captioner_backend: str | None = None
    captioner_model: str | None = None
    granite: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    final_answer: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    _stage_starts: dict = field(default_factory=dict, repr=False)

    @classmethod
    def start(cls, out_root: str, video_id: str, question: str,
              question_type: str | None = None, question_id: Any = None,
              split: str | None = None) -> "DebugTrace":
        qid = make_query_id(video_id, question_id, question)
        out_dir = Path(out_root) / qid
        (out_dir / "frames").mkdir(parents=True, exist_ok=True)
        return cls(query_id=qid, out_dir=out_dir, video_id=video_id, question=question,
                    question_type=question_type, question_id=question_id, split=split)

    # ── timing ──────────────────────────────────────────────────────────
    def mark_start(self, stage: str) -> None:
        self._stage_starts[stage] = time.perf_counter()

    def mark_end(self, stage: str) -> None:
        t0 = self._stage_starts.get(stage)
        if t0 is not None:
            self.timings[stage] = time.perf_counter() - t0

    # ── ground truth (2A) ──────────────────────────────────────────────
    def set_ground_truth(self, answer: str | None, options: list[str] | None) -> None:
        self.ground_truth = {"answer": answer, "options": options}

    # ── frames + captions ───────────────────────────────────────────────
    def capture_frames(self, video_path: str, retrieved_frame_dicts: list[dict]) -> None:
        """Populates self.frames and saves frame_NNN.jpg per task step 4.
        Best-effort: an image-extraction failure never raises out of this
        method (this is diagnostic tooling, must not be able to break the
        pipeline it's observing)."""
        try:
            targets = [(f["frame_idx"], f["timestamp"]) for f in retrieved_frame_dicts]
            images = _extract_frame_images(video_path, targets)
        except Exception as e:  # noqa: BLE001
            images = {}
            self.timings["frame_capture_error"] = str(e)

        for i, f in enumerate(sorted(retrieved_frame_dicts, key=lambda d: d["timestamp"]), start=1):
            entry = {
                "frame_id": f["frame_idx"],
                "timestamp": f["timestamp"],
                "scene_id": f.get("scene_id"),
                "image_path": None,
            }
            img = images.get(f["frame_idx"])
            if img is not None:
                fname = f"frame_{i:03d}.jpg"
                try:
                    img.convert("RGB").save(self.out_dir / "frames" / fname, "JPEG", quality=90)
                    entry["image_path"] = f"frames/{fname}"
                except Exception as e:  # noqa: BLE001
                    entry["image_error"] = str(e)
            self.frames.append(entry)
        self._frame_images = images  # kept in-memory for the contact sheet only

    def set_captioner_info(self, backend: str, model: str | None) -> None:
        self.captioner_backend = backend
        self.captioner_model = model

    # ── granite prompt/output ───────────────────────────────────────────
    def set_granite(self, debug_capture: dict, context_text: str, focus_hint: str | None,
                     verification_instructions: str | None, raw_answer: str,
                     generation_time: float) -> None:
        self.granite = {
            "system_prompt": debug_capture.get("system_prompt"),
            "user_prompt": debug_capture.get("user_prompt"),
            "model": debug_capture.get("model"),
            "retrieved_captions_context": context_text,
            "focus_hints": focus_hint,
            "verification_instructions": verification_instructions,
            "raw_generation": raw_answer,
            "generation_time": generation_time,
            "finish_reason": debug_capture.get("finish_reason"),
            "usage": debug_capture.get("usage"),
        }
        self._write_granite_prompt_file()

    def _write_granite_prompt_file(self) -> None:
        g = self.granite
        parts = [
            "=" * 80, "SYSTEM PROMPT", "=" * 80, g.get("system_prompt") or "", "",
            "=" * 80, "USER PROMPT", "=" * 80, g.get("user_prompt") or "", "",
            "=" * 80, "RETRIEVED CAPTIONS / METADATA (context block, verbatim)", "=" * 80,
            g.get("retrieved_captions_context") or "", "",
            "=" * 80, "FOCUS HINTS", "=" * 80, g.get("focus_hints") or "(none)", "",
            "=" * 80, "VERIFICATION INSTRUCTIONS", "=" * 80,
            g.get("verification_instructions") or "(none -- verification is a separate post-hoc stage, not part of the answerer prompt in cerberus_mode=legacy)",
        ]
        (self.out_dir / "granite_prompt.txt").write_text("\n".join(parts), encoding="utf-8")

    # ── verification ────────────────────────────────────────────────────
    def set_verification(self, verified: bool, verified_claims: list[str],
                          rejected_claims: list[str], unverifiable_claims: list[str],
                          mode: str | None = None, confidence: float | None = None) -> None:
        reasons = []
        if rejected_claims:
            reasons.append(f"{len(rejected_claims)} claim(s) contradicted by retrieved captions")
        if unverifiable_claims:
            reasons.append(f"{len(unverifiable_claims)} claim(s) had no supporting evidence in retrieved captions")
        self.verification = {
            "verification_prompt": None,  # legacy CerberusV mode has no separate LLM verification prompt -- NLI/NER over claims vs. cached facts, not a generative call
            "verification_response": {
                "verified_claims": verified_claims,
                "rejected_claims": rejected_claims,
                "unverifiable_claims": unverifiable_claims,
                "mode": mode,
            },
            "verification_decision": "verified" if verified else "not_verified",
            "verified": verified,
            "reason_for_rejection": "; ".join(reasons) if reasons else None,
            "confidence": confidence,
        }

    # ── final answer (task 9 / 2A extension) ────────────────────────────
    def set_final_answer(self, raw_answer: str, verified_answer: str,
                          ground_truth_answer: str | None,
                          comparison_method: str = "exact_match") -> None:
        match = None
        correct = None
        if ground_truth_answer is not None:
            match = _normalize_answer(verified_answer) == _normalize_answer(ground_truth_answer)
            correct = match
        self.final_answer = {
            "raw_answer": raw_answer,
            "verified_answer": verified_answer,
            "ground_truth_answer": ground_truth_answer,
            "match": match,
            "evaluation": {"correct": correct, "comparison_method": comparison_method},
        }

    # ── finalize ─────────────────────────────────────────────────────────
    _CANONICAL_TIMING_STAGES = (
        "retrieval_time", "caption_time", "prompt_construction_time",
        "granite_generation_time", "cerberus_verification_time",
    )

    def finalize(self) -> None:
        # Sum only the canonical top-level stages (task step 10), not every
        # key in self.timings -- some keys (e.g. per-frame decode counters,
        # error strings) aren't additive latency components, and summing
        # everything would double-count sub-stage timings against the
        # coarser ones.
        self.timings["total_latency"] = sum(
            self.timings[k] for k in self._CANONICAL_TIMING_STAGES
            if isinstance(self.timings.get(k), (int, float))
        )
        sheet = None
        try:
            images = getattr(self, "_frame_images", {})
            if images:
                sheet = _build_contact_sheet(self.frames, images)
                if sheet is not None:
                    sheet.save(self.out_dir / "frames_overview.jpg", "JPEG", quality=90)
        except Exception as e:  # noqa: BLE001
            self.timings["contact_sheet_error"] = str(e)

        status, root_cause = diagnose(self)

        trace_dict = {
            "query": {
                "video_id": self.video_id,
                "question_id": self.question_id,
                "question": self.question,
                "question_type": self.question_type,
                "split": self.split,
            },
            "ground_truth": self.ground_truth,
            "retrieval": self.retrieval,
            "frames": self.frames,
            "captions": self.captions,
            "captioner": {"backend": self.captioner_backend, "model": self.captioner_model},
            "granite": self.granite,
            "verification": self.verification,
            "final_answer": self.final_answer,
            "timings": self.timings,
            "pipeline_status": status,
            "root_cause": root_cause,
        }
        (self.out_dir / "trace.json").write_text(json.dumps(trace_dict, indent=2, default=str), encoding="utf-8")
        (self.out_dir / "summary.md").write_text(_render_summary_md(self, status, root_cause), encoding="utf-8")


def _normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    return re.sub(r"\s+", " ", text.strip().lower())


# ── root-cause heuristic (task step 16) ─────────────────────────────────────

_STAGES = [
    "Retrieval", "Frame Selection", "Caption Quality", "Prompt Construction",
    "Granite Reasoning", "Cerberus Verification", "Final Prediction",
]


def diagnose(t: DebugTrace) -> tuple[dict, str]:
    """Best-effort, evidence-only root-cause heuristic. Never claims more than
    the collected trace supports -- any stage lacking evidence is reported as
    "insufficient evidence", not guessed. This is deliberately conservative:
    it flags a stage "Suspect" only when the trace itself contains a concrete
    signal (an error, an empty result, near-zero margin, empty caption text,
    a rejected/unverifiable claim, a wrong final answer), not on stylistic
    grounds no trace field could establish."""
    status: dict[str, str] = {s: "Insufficient evidence" for s in _STAGES}
    notes: list[str] = []

    # Retrieval
    retrieved_idxs = _dget(t.retrieval, "retrieved_frame_idxs")
    if retrieved_idxs is not None:
        if len(retrieved_idxs) == 0:
            status["Retrieval"] = "Suspect"
            notes.append("retrieval returned zero frames")
        else:
            margin = _dget(t.retrieval, "margin")
            tau = _dget(t.retrieval, "tau")
            if isinstance(margin, (int, float)) and isinstance(tau, (int, float)) and margin < tau * 0.1:
                status["Retrieval"] = "Suspect"
                notes.append(f"retrieval margin ({margin:.4f}) was far below tau ({tau}), i.e. the top scene was barely distinguishable from the runner-up")
            else:
                status["Retrieval"] = "Correct"

    # Frame Selection (did the frame extraction/inspection succeed for what was retrieved?)
    if t.frames:
        missing_images = [f for f in t.frames if not f.get("image_path")]
        if len(missing_images) == len(t.frames):
            status["Frame Selection"] = "Suspect"
            notes.append("none of the retrieved frames could be extracted as images (possible seek/decode failure)")
        elif missing_images:
            status["Frame Selection"] = "Suspect"
            notes.append(f"{len(missing_images)}/{len(t.frames)} retrieved frames failed image extraction")
        else:
            status["Frame Selection"] = "Correct"

    # Caption Quality
    if t.captions:
        empty = [c for c in t.captions if not c.get("caption") or not str(c["caption"]).strip()]
        very_short = [c for c in t.captions if c.get("caption") and len(str(c["caption"]).strip()) < 15]
        if empty:
            status["Caption Quality"] = "Suspect"
            notes.append(f"{len(empty)}/{len(t.captions)} retrieved frames produced an empty caption")
        elif len(very_short) == len(t.captions) and len(t.captions) > 0:
            status["Caption Quality"] = "Suspect"
            notes.append("every caption was under 15 characters (likely too generic/uninformative to support reasoning)")
        else:
            status["Caption Quality"] = "Correct"
    elif t.frames:
        status["Caption Quality"] = "Insufficient evidence"

    # Prompt Construction
    if t.granite.get("user_prompt") is not None:
        ctx = t.granite.get("retrieved_captions_context") or ""
        n_captions_in_prompt = sum(1 for c in t.captions if c.get("caption") and str(c["caption"]) in ctx)
        if t.captions and n_captions_in_prompt == 0:
            status["Prompt Construction"] = "Suspect"
            notes.append("none of the generated captions appear verbatim in the context block sent to Granite")
        else:
            status["Prompt Construction"] = "Correct"

    # Granite Reasoning
    if t.granite.get("raw_generation") is not None:
        raw = str(t.granite["raw_generation"]).strip()
        if not raw:
            status["Granite Reasoning"] = "Suspect"
            notes.append("Granite returned an empty raw generation")
        elif t.granite.get("finish_reason") in ("length", "max_tokens"):
            status["Granite Reasoning"] = "Suspect"
            notes.append(f"Granite's generation was truncated (finish_reason={t.granite.get('finish_reason')!r})")
        else:
            status["Granite Reasoning"] = "Correct"

    # Cerberus Verification
    if t.verification:
        v = t.verification.get("verification_response", {})
        if v.get("rejected_claims") or v.get("unverifiable_claims"):
            status["Cerberus Verification"] = "Suspect"
            reason = t.verification.get("reason_for_rejection")
            if reason:
                notes.append(f"Cerberus: {reason}")
        else:
            status["Cerberus Verification"] = "Correct"

    # Final Prediction
    fa = t.final_answer
    if fa.get("match") is not None:
        status["Final Prediction"] = "Correct" if fa["match"] else "Incorrect"
    elif fa.get("verified_answer") is not None:
        status["Final Prediction"] = "Insufficient evidence"
        notes.append("no ground truth available to score the final answer")

    # ── earliest-diverging-stage summary paragraph ──────────────────────
    if fa.get("match") is True:
        paragraph = (
            "The final answer matched ground truth, and no stage in the trace shows a concrete "
            "failure signal. No root-cause investigation needed for this question."
        )
    elif fa.get("match") is False:
        first_suspect = next((s for s in _STAGES if status[s] == "Suspect"), None)
        if first_suspect:
            paragraph = (
                f"The final answer was incorrect. The earliest stage in the trace showing a concrete "
                f"failure signal is **{first_suspect}** ({'; '.join(notes) if notes else 'see stage detail above'}). "
                f"This is the most likely root cause, but stages marked 'Insufficient evidence' were not "
                f"checked and could also be contributing factors."
            )
        elif any(v == "Insufficient evidence" for v in status.values()):
            insufficient = [s for s, v in status.items() if v == "Insufficient evidence"]
            paragraph = (
                f"The final answer was incorrect, but no stage shows a concrete failure signal in the "
                f"collected trace. Additional evidence is required to localize the failure -- in "
                f"particular: {', '.join(insufficient)}. Do not assume any of these stages is at fault "
                f"without checking the corresponding trace fields directly."
            )
        else:
            paragraph = (
                "The final answer was incorrect, but every instrumented stage reports 'Correct' with no "
                "concrete failure signal (empty output, truncation, rejected claim, low retrieval margin, "
                "etc). This suggests either a reasoning error inside Granite that this trace's heuristics "
                "cannot detect from structural signals alone, or a genuine ambiguity in the question/ground "
                "truth. Manual inspection of granite_prompt.txt and the raw generation is required."
            )
    else:
        paragraph = (
            "No ground truth was available for this query, so correctness cannot be scored "
            "automatically. Stage-level status above reflects only structural signals found in the trace."
        )

    return status, paragraph


def _dget(d: dict, key: str):
    return d.get(key) if isinstance(d, dict) else None


def _status_mark(v: str) -> str:
    return {"Correct": "✓", "Suspect": "✗", "Incorrect": "✗"}.get(v, "?")


def _render_summary_md(t: DebugTrace, status: dict, root_cause: str) -> str:
    lines = []
    lines.append("# Query")
    lines.append(f"- Video ID: `{t.video_id}`")
    lines.append(f"- Question ID: `{t.question_id}`")
    lines.append(f"- Dataset Split: `{t.split}`")
    lines.append(f"- Question Type: `{t.question_type}`")
    lines.append(f"- Question: {t.question}")
    lines.append("")

    lines.append("# Ground Truth")
    lines.append(f"- Expected Answer: {t.ground_truth.get('answer')}")
    opts = t.ground_truth.get("options")
    if opts:
        lines.append("- Multiple Choice Options:")
        for i, o in enumerate(opts):
            lines.append(f"  {chr(ord('A') + i)}. {o}")
    lines.append("")

    lines.append("# Retrieved Scenes")
    r = t.retrieval
    lines.append(f"- branch: `{r.get('branch')}`  margin: `{r.get('margin')}`  tau: `{r.get('tau')}`")
    lines.append(f"- base_pool: `{r.get('base_pool')}`  post_pull_pool: `{r.get('post_pull_pool')}`  "
                  f"cross_scene_edges_added: `{r.get('cross_scene_edges_added')}`")
    lines.append(f"- shortlisted_scene_ids: `{r.get('shortlisted_scene_ids')}`")
    lines.append("")

    lines.append("# Retrieved Frames")
    lines.append("| frame_id | timestamp | scene_id | image |")
    lines.append("|---|---|---|---|")
    for f in sorted(t.frames, key=lambda x: x["timestamp"]):
        lines.append(f"| {f['frame_id']} | {f['timestamp']:.2f}s | {f.get('scene_id')} | {f.get('image_path') or '(unavailable)'} |")
    lines.append("")

    lines.append("# Generated Captions")
    lines.append("| frame_id | timestamp | caption | length | gen_time_s |")
    lines.append("|---|---|---|---|---|")
    for c in sorted(t.captions, key=lambda x: x["timestamp"]):
        cap = (c.get("caption") or "").replace("\n", " ").replace("|", "/")
        lines.append(f"| {c['frame_idx']} | {c['timestamp']:.2f}s | {cap[:200]} | {c.get('caption_length')} | {c.get('caption_generation_time'):.3f} |" if c.get('caption_generation_time') is not None else f"| {c['frame_idx']} | {c['timestamp']:.2f}s | {cap[:200]} | {c.get('caption_length')} | n/a |")
    lines.append("")
    lines.append(f"(captioner: {t.captioner_backend} / {t.captioner_model})")
    lines.append("")

    lines.append("# Granite Prompt")
    lines.append("See `granite_prompt.txt` for the exact, untruncated system/user prompt.")
    lines.append("")

    lines.append("# Granite Raw Output")
    lines.append("```")
    lines.append(str(t.granite.get("raw_generation") or ""))
    lines.append("```")
    lines.append(f"- generation_time: `{t.granite.get('generation_time')}`  finish_reason: `{t.granite.get('finish_reason')}`  usage: `{t.granite.get('usage')}`")
    lines.append("")

    lines.append("# Cerberus Verification")
    v = t.verification
    lines.append(f"- decision: `{v.get('verification_decision')}`  verified: `{v.get('verified')}`")
    if v.get("reason_for_rejection"):
        lines.append(f"- reason for rejection: {v['reason_for_rejection']}")
    lines.append(f"- confidence: `{v.get('confidence')}`")
    lines.append("")

    lines.append("# Final Prediction")
    fa = t.final_answer
    lines.append(f"- Final Answer: {fa.get('verified_answer')}")
    lines.append(f"- Ground Truth Answer: {fa.get('ground_truth_answer')}")
    match = fa.get("match")
    lines.append(f"- Correct / Incorrect: {'Correct' if match is True else ('Incorrect' if match is False else 'N/A (no ground truth)')}")
    lines.append("")

    lines.append("# Pipeline Timings")
    for k, v_ in sorted(t.timings.items()):
        lines.append(f"- {k}: `{v_}`")
    lines.append("")

    lines.append("# Root Cause Summary")
    lines.append("")
    lines.append("**Pipeline Status**")
    lines.append("")
    for s in _STAGES:
        mark = _status_mark(status[s])
        lines.append(f"{mark} {s}: {status[s]}")
    lines.append("")
    lines.append("**Root Cause (best effort)**")
    lines.append("")
    lines.append(root_cause)
    lines.append("")

    return "\n".join(lines)
