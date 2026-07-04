"""One-off smoke test for lazy captioning (query-time, cached by frame_idx).
Not part of the phase6 measurement suite — throwaway verification script.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris._clip as _clip
import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig


class CaptionCallCounter:
    def __init__(self):
        self._orig = _clip.get_semantic_and_clip_caption
        self.count = 0

    def __enter__(self):
        def counting(*a, **kw):
            self.count += 1
            return self._orig(*a, **kw)
        _clip.get_semantic_and_clip_caption = counting
        return self

    def __exit__(self, *exc):
        _clip.get_semantic_and_clip_caption = self._orig


def main():
    vpath = REPO_ROOT / "videoplayback.mp4"
    cfg = IRISConfig(
        ranking_mode="ppr", codec_conf_source="packet_size",
        codec_conf_pictype_norm=True, ppr_lambda=0.5, ppr_damping=0.5,
        l2_retrieve_top_k=8,
    )

    print("Ingesting (build should NOT caption)...")
    with CaptionCallCounter() as ctr:
        index = iris_ingest.ingest(str(vpath), config=cfg)
    print(f"caption calls during ingest = {ctr.count}  (expect 0)")
    n_none = sum(1 for fr in index.frames if fr.caption is None)
    print(f"frames with caption=None after build = {n_none} / {len(index.frames)} (expect all)")

    q = "what is moving in the scene"

    print("\nFirst query (cold — should trigger new captions for retrieved frames)...")
    with CaptionCallCounter() as ctr1:
        result1 = iris_query.query(q, index, config=cfg)
    print(f"answer: {result1['answer'][:200]!r}")
    print(f"retrieved_frame_idxs: {result1['retrieved_frame_idxs']}")
    print(f"new caption calls (1st query) = {ctr1.count}")
    print(f"frames_decoded_for_captions (1st query) = {result1['frames_decoded_for_captions']}")
    print(f"timings: {result1['timings']}")

    print("\nSecond identical query (should hit cache — 0 new caption calls)...")
    with CaptionCallCounter() as ctr2:
        result2 = iris_query.query(q, index, config=cfg)
    print(f"answer: {result2['answer'][:200]!r}")
    print(f"new caption calls (2nd query) = {ctr2.count}  (expect 0)")
    print(f"frames_decoded_for_captions (2nd query) = {result2['frames_decoded_for_captions']}  (expect 0)")
    print(f"timings: {result2['timings']}")

    assert ctr2.count == 0, "second identical query re-captioned a frame — cache is not working"
    print("\nOK: lazy caption cache verified.")


if __name__ == "__main__":
    main()
