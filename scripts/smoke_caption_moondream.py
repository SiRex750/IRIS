"""Standalone smoke test: caption the SAME 40 VIRAT frames used by
scripts/smoke_caption.py (SmolVLM2 run), this time with Moondream2, to
compare (a) CPU wall-clock per frame and (b) whether Moondream2 exhibits
the same fabricated before/after "temporal diff" language SmolVLM2 did.

Moondream2 is a pure single-image captioner (no video/temporal context),
so it should NOT produce "X is gone / new Y appeared" language the way
SmolVLM2 did. This script measures that directly rather than assuming it.

This is a diagnostic only. It does not import from any iris/ pipeline
module and does not touch VIRAT annotations. It does not modify
smoke_caption.py or smoke_caption_results.jsonl.

Usage:
    python scripts/smoke_caption_moondream.py

Output:
    smoke_caption_results_moondream.jsonl  (repo root)
"""
import json
import os
import re
import sys
import time

import av

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOLVLM_RESULTS_PATH = os.path.join(REPO_ROOT, "smoke_caption_results.jsonl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "smoke_caption_results_moondream.jsonl")

CLIP_DIR = os.path.join(REPO_ROOT, "eval", "data", "virat", "videos")

# If CPU inference proves impractically slow, cap at the first 15 frames
# (in on-disk order of smoke_caption_results.jsonl) instead of all 40.
MAX_FRAMES_IF_SLOW = 15
SLOW_THRESHOLD_SECONDS_PER_FRAME = 60.0

DIFF_LANGUAGE_PATTERN = re.compile(
    r"gone|before|after|no longer|new |moved|missing", re.IGNORECASE
)


def load_frame_manifest_from_smolvlm_results():
    """Read frame_id/source_clip/peak_or_nonpeak from the SmolVLM2 run's
    output so we caption the EXACT SAME 40 frames (no resampling)."""
    if not os.path.isfile(SMOLVLM_RESULTS_PATH):
        print(f"ERROR: {SMOLVLM_RESULTS_PATH} not found. Run scripts/smoke_caption.py first.")
        sys.exit(1)
    manifest = []
    with open(SMOLVLM_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clip_name, idx_str = row["frame_id"].rsplit(":", 1)
            manifest.append({
                "frame_id": row["frame_id"],
                "source_clip": row["source_clip"],
                "frame_index": int(idx_str),
                "peak_or_nonpeak": row["peak_or_nonpeak"],
            })
    return manifest


def extract_full_frames(video_path, target_indices):
    """Sequential decode pass: grab full-res PIL frames for target indices only."""
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    stream.thread_type = "AUTO"
    out = {}
    target_set = set(target_indices)
    for idx, frame in enumerate(container.decode(stream)):
        if idx in target_set:
            out[idx] = frame.to_image()  # PIL.Image
            if len(out) == len(target_set):
                break
    container.close()
    return out


def attach_images(manifest):
    by_clip = {}
    for entry in manifest:
        by_clip.setdefault(entry["source_clip"], []).append(entry)

    for clip_name, entries in by_clip.items():
        clip_path = os.path.join(CLIP_DIR, clip_name)
        if not os.path.isfile(clip_path):
            print(f"WARNING: clip not found, skipping: {clip_path}")
            continue
        indices = [e["frame_index"] for e in entries]
        print(f"[{clip_name}] extracting {len(indices)} frames (same indices as SmolVLM2 run)...")
        full_frames = extract_full_frames(clip_path, indices)
        for entry in entries:
            idx = entry["frame_index"]
            if idx not in full_frames:
                print(f"WARNING: frame {idx} not recovered for {clip_name}, skipping")
                entry["image"] = None
            else:
                entry["image"] = full_frames[idx]
    return [e for e in manifest if e.get("image") is not None]


def load_model():
    from transformers import AutoModelForCausalLM

    model_id = "vikhyatk/moondream2"
    revision = "2025-06-21"
    print(f"Loading {model_id} (revision={revision}) on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    return model


def caption_frame(model, image):
    result = model.caption(image, length="normal")
    return result["caption"].strip()


def main():
    manifest = load_frame_manifest_from_smolvlm_results()
    print(f"Loaded {len(manifest)} frame_ids from {SMOLVLM_RESULTS_PATH}")

    manifest = attach_images(manifest)
    if not manifest:
        print("ERROR: no frames recovered, aborting.")
        sys.exit(1)
    print(f"Total frames to caption: {len(manifest)}")

    try:
        model = load_model()
    except Exception as e:
        print(f"STOP: Moondream2 failed to load. Raw error:\n{e}")
        sys.exit(1)

    # Truncate/overwrite once so subsequent per-frame appends start clean.
    open(OUTPUT_PATH, "w", encoding="utf-8").close()

    results = []
    errors = []
    capped = False
    frame_limit = len(manifest)
    t0_total = time.time()
    for i, entry in enumerate(manifest):
        if i >= frame_limit:
            break
        t0 = time.time()
        try:
            caption = caption_frame(model, entry["image"])
        except Exception as e:
            errors.append({"frame_id": entry["frame_id"], "error": str(e)})
            print(f"  [{i+1}/{frame_limit}] {entry['frame_id']} ERROR: {e}", flush=True)
            continue
        elapsed = time.time() - t0
        row = {
            "frame_id": entry["frame_id"],
            "source_clip": entry["source_clip"],
            "peak_or_nonpeak": entry["peak_or_nonpeak"],
            "caption": caption,
            "caption_time_seconds": elapsed,
        }
        results.append(row)
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        print(f"[{i+1}/{frame_limit}] running_count={len(results)} {elapsed:.1f}s "
              f"{entry['frame_id']} ({entry['peak_or_nonpeak']}): {caption}", flush=True)

        # After the first frame, decide whether CPU speed is impractical and
        # cap the run rather than grinding through all frames.
        if i == 0 and elapsed > SLOW_THRESHOLD_SECONDS_PER_FRAME and len(manifest) > MAX_FRAMES_IF_SLOW:
            capped = True
            frame_limit = MAX_FRAMES_IF_SLOW
            print(f"NOTE: first frame took {elapsed:.1f}s (> {SLOW_THRESHOLD_SECONDS_PER_FRAME}s "
                  f"threshold). Capping run at first {MAX_FRAMES_IF_SLOW} frames instead of "
                  f"{len(manifest)}.", flush=True)

    total_elapsed = time.time() - t0_total

    error_rate = len(errors) / frame_limit if frame_limit else 1.0
    if error_rate > 0.10:
        print(f"\nSTOP: {len(errors)}/{frame_limit} frames errored ({error_rate:.1%} > 10%). Raw errors:", flush=True)
        for e in errors:
            print(f"  {e['frame_id']}: {e['error']}", flush=True)
        sys.exit(1)

    exact_counts = {}
    for r in results:
        exact_counts[r["caption"]] = exact_counts.get(r["caption"], 0) + 1

    diff_language_count = sum(1 for r in results if DIFF_LANGUAGE_PATTERN.search(r["caption"]))
    avg_sec = sum(r["caption_time_seconds"] for r in results) / len(results) if results else 0.0

    print("\n=== SUMMARY ===")
    print(f"Run capped at {MAX_FRAMES_IF_SLOW} frames due to CPU slowness: {capped}")
    print(f"Total frames captioned: {len(results)} (of {frame_limit} selected)")
    print(f"Load/inference errors: {len(errors)}")
    print(f"Total wall time (captioning loop): {total_elapsed:.1f}s")
    print(f"Avg sec/frame: {avg_sec:.2f}s  (SmolVLM2 comparison point: ~74s/frame measured)")
    print(f"Unique captions: {len(exact_counts)} / {len(results)}")
    print(f"Diff-language captions (gone/before/after/no longer/new /moved/missing): "
          f"{diff_language_count} / {len(results)}")
    print("Top 5 most common exact captions:")
    for cap, count in sorted(exact_counts.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  x{count}: {cap[:150]}")
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
