"""C0 -- persist and replay captions for the e2e eval harness.

iris/query.py::_ensure_captions mutates in-memory FrameRecords and the eval
scripts never re-save the index, so captions generated during a run are
discarded at process exit. That makes any end-to-end run unreproducible by
construction, independent of any LLM-determinism question: even with a
perfectly deterministic answerer, a rerun would still see different captions
if the captioner itself is at all non-deterministic (or simply unavailable).

This module is opt-in scaffolding for scripts/*_eval.py harnesses -- it does
NOT touch iris/ production code and does NOT change default behaviour when
neither --caption-dump nor --caption-load is passed.

Schema (matches tuning/blind_ablation/captions_dump.json, already produced by
scripts/blind_ablation_eval.py): {video: {qid: {str(frame_idx): caption_text}}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_caption_dump(path: str | Path) -> dict[str, dict[str, dict[str, str]]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_caption_dump(path: str | Path, dump: dict[str, dict[str, dict[str, str]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)


def apply_caption_load(
    index: Any, retrieved_frames: list[dict], vid: str, qid: str,
    loaded_dump: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Populate captions on `retrieved_frames` (and the underlying FrameRecords
    so the per-video cache stays consistent with _ensure_captions' own
    caching contract) directly from a frozen dump, skipping captioning
    entirely. Raises KeyError loudly on any (video, qid, frame_idx) miss
    rather than silently falling through to live captioning.
    """
    video_dump = loaded_dump.get(vid)
    if video_dump is None:
        raise KeyError(f"--caption-load: no captions recorded for video={vid!r}")
    qid_dump = video_dump.get(str(qid))
    if qid_dump is None:
        raise KeyError(f"--caption-load: no captions recorded for video={vid!r} qid={qid!r}")

    frame_map = {fr.frame_idx: fr for fr in index.frames}
    for f in retrieved_frames:
        frame_idx = f["frame_idx"]
        key = str(frame_idx)
        if key not in qid_dump:
            raise KeyError(
                f"--caption-load: missing caption for video={vid!r} qid={qid!r} "
                f"frame_idx={frame_idx!r} in {list(qid_dump.keys())!r} -- "
                "refusing to silently caption it live"
            )
        cap_dict = {"semantic_caption": qid_dump[key]}
        f["caption"] = cap_dict
        fr = frame_map.get(frame_idx)
        if fr is not None:
            fr.caption = cap_dict


def record_caption_dump(
    retrieved_frames: list[dict], vid: str, qid: str,
    dump_accumulator: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Extract the semantic-caption text actually used for each retrieved
    frame this question and record it into `dump_accumulator` in-place, in
    the same {video: {qid: {frame_idx: caption}}} schema apply_caption_load
    reads back."""
    frame_captions: dict[str, str] = {}
    for f in retrieved_frames:
        cap = f.get("caption")
        cap_text = cap.get("semantic_caption") if isinstance(cap, dict) else cap
        frame_captions[str(f["frame_idx"])] = cap_text
    dump_accumulator.setdefault(vid, {})[str(qid)] = frame_captions
