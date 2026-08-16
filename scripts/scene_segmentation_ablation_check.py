"""Ablation-enabling change verification for config.scene_segmentation.

Runs on a small sample video (eval/data/ucf/videos/.../Abuse029_x264.mp4,
10s @ 30fps = 300 frames -- fast, deterministic, already in-repo):

  1. PARITY GUARD: ingest with scene_segmentation="codec" (the default) and
     diff every frame's scene_id against a second ingest of the ORIGINAL
     (pre-change) ingest.py, loaded straight from git HEAD (i.e. the last
     commit before this ablation knob existed). Any mismatch aborts loudly
     (AssertionError) and writes nothing.
  2. Ingests fixed_count and fixed_seconds and prints, for all three modes:
     scene count + boundary frame indices (first frame_idx of each new
     scene_id in frame_idx order). Deterministic, single process, no
     retries -- read directly.

Not a pytest file (needs a real video decode + git blob read); the fast
pytest-level checks live in tests/test_ingest.py
(test_scene_segmentation_modes / test_scene_segmentation_codec_unchanged).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace as _replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

VIDEO = REPO_ROOT / "eval" / "data" / "ucf" / "videos" / "Anomaly-Videos-Part-1" / "Abuse" / "Abuse029_x264.mp4"

FIXED_EMB_DIM = 512


def _patch_clip(mod):
    """Swap in cheap deterministic stand-ins for the CLIP/caption calls so
    this stays fast and hermetic w.r.t. model weights -- packet-curve /
    scene_id assignment (what we're checking) does real video decode."""
    import numpy as np
    fixed_emb = np.full(FIXED_EMB_DIM, 1.0 / np.sqrt(FIXED_EMB_DIM), dtype=np.float32)
    mod.get_clip_embedding_from_pil = lambda pil, device: fixed_emb.copy()
    mod.get_semantic_and_clip_caption = (
        lambda pil, frame, emb, device: {"clip_label": "x", "semantic_caption": "a test frame"}
    )


def _boundaries(idx) -> list[int]:
    """First frame_idx (in frame_idx order) at which each new scene_id appears."""
    frames_sorted = sorted(idx.frames, key=lambda fr: fr.frame_idx)
    seen: dict = {}
    boundaries = []
    for fr in frames_sorted:
        if fr.scene_id not in seen:
            seen[fr.scene_id] = fr.frame_idx
            boundaries.append(fr.frame_idx)
    return boundaries


def _load_head_ingest_module():
    """Load the pre-change iris/ingest.py (as committed at HEAD) as an
    isolated module, so the parity check is against the actual git history,
    not a hand-copied snapshot."""
    blob = subprocess.run(
        ["git", "show", "HEAD:iris/ingest.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=True,
    ).stdout
    tmp_path = REPO_ROOT / "scripts" / "_head_ingest_snapshot.py"
    tmp_path.write_text(blob, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("iris._head_ingest_snapshot", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp_path.unlink()
    return mod


def main() -> None:
    if not VIDEO.exists():
        print(f"FATAL: sample video not found: {VIDEO}", file=sys.stderr)
        sys.exit(1)

    import iris.ingest as ingest_mod
    from iris.iris_config import IRISConfig

    _patch_clip(ingest_mod)

    # ── PARITY GUARD ─────────────────────────────────────────────────────
    head_mod = _load_head_ingest_module()
    _patch_clip(head_mod)

    cfg_codec = IRISConfig(scene_segmentation="codec")
    idx_new_codec = ingest_mod.ingest(str(VIDEO), config=cfg_codec)
    idx_head = head_mod.ingest(str(VIDEO), config=IRISConfig())  # HEAD already has codec/fixed_count/fixed_seconds

    new_map = {fr.frame_idx: fr.scene_id for fr in idx_new_codec.frames}
    head_map = {fr.frame_idx: fr.scene_id for fr in idx_head.frames}
    assert new_map == head_map, (
        f"PARITY GUARD FAILED: scene_segmentation='codec' diverges from HEAD "
        f"ingest.py on {VIDEO.name}. new={new_map} head={head_map}"
    )
    print(f"PARITY GUARD OK: codec-mode scene_id bit-identical to HEAD on {VIDEO.name} "
          f"({len(new_map)} frames)")

    # ── Four modes, same sample video ───────────────────────────────────
    idx_fixed_count = ingest_mod.ingest(str(VIDEO), config=_replace(cfg_codec, scene_segmentation="fixed_count"))
    idx_fixed_seconds = ingest_mod.ingest(str(VIDEO), config=_replace(cfg_codec, scene_segmentation="fixed_seconds"))
    idx_fixed_time_matched = ingest_mod.ingest(
        str(VIDEO), config=_replace(cfg_codec, scene_segmentation="fixed_time_matched")
    )

    # PARITY GUARD (STEP 1): adding fixed_time_matched must not perturb the
    # three pre-existing modes -- diff each against the same HEAD ingest.py
    # used above (HEAD already carries codec/fixed_count/fixed_seconds).
    head_fixed_count = head_mod.ingest(str(VIDEO), config=_replace(IRISConfig(), scene_segmentation="fixed_count"))
    head_fixed_seconds = head_mod.ingest(str(VIDEO), config=_replace(IRISConfig(), scene_segmentation="fixed_seconds"))
    for label, new_idx, head_idx in [
        ("fixed_count", idx_fixed_count, head_fixed_count),
        ("fixed_seconds", idx_fixed_seconds, head_fixed_seconds),
    ]:
        nm = {fr.frame_idx: fr.scene_id for fr in new_idx.frames}
        hm = {fr.frame_idx: fr.scene_id for fr in head_idx.frames}
        assert nm == hm, (
            f"PARITY GUARD FAILED: scene_segmentation='{label}' diverges from HEAD "
            f"ingest.py on {VIDEO.name} after adding fixed_time_matched. new={nm} head={hm}"
        )
        print(f"PARITY GUARD OK: {label} scene_id bit-identical to HEAD on {VIDEO.name} ({len(nm)} frames)")

    for label, idx in [
        ("codec", idx_new_codec),
        ("fixed_count", idx_fixed_count),
        ("fixed_seconds", idx_fixed_seconds),
        ("fixed_time_matched", idx_fixed_time_matched),
    ]:
        n_scenes = len(set(fr.scene_id for fr in idx.frames))
        print(f"mode={label}  scene_count={n_scenes}  boundary_frame_idxs={_boundaries(idx)}")

    codec_n = len(set(fr.scene_id for fr in idx_new_codec.frames))
    ftm_n = len(set(fr.scene_id for fr in idx_fixed_time_matched.frames))
    assert ftm_n == codec_n, (
        f"fixed_time_matched scene count ({ftm_n}) != codec scene count ({codec_n})"
    )
    print(f"fixed_time_matched scene count matches codec ({ftm_n}) -- OK")

    # Bonus: fixed_seconds with a period shorter than the 10s clip, so the
    # bucketing is visible (default fixed_scene_seconds=60 collapses a 10s
    # clip to a single scene, which is correct but not illustrative).
    idx_fixed_seconds_2s = ingest_mod.ingest(
        str(VIDEO), config=_replace(cfg_codec, scene_segmentation="fixed_seconds", fixed_scene_seconds=2.0)
    )
    n_scenes_2s = len(set(fr.scene_id for fr in idx_fixed_seconds_2s.frames))
    print(f"mode=fixed_seconds (fixed_scene_seconds=2.0)  scene_count={n_scenes_2s}  "
          f"boundary_frame_idxs={_boundaries(idx_fixed_seconds_2s)}")


if __name__ == "__main__":
    main()
