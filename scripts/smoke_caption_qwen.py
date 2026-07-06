"""Standalone smoke test: caption the SAME VIRAT frames used by
scripts/smoke_caption.py (SmolVLM2) and scripts/smoke_caption_moondream.py
(Moondream2), this time with Qwen2.5-VL-3B-Instruct, as a third anchor point
for CPU speed and confabulation comparison.

Qwen2.5-VL has native transformers support (no trust_remote_code), so it
should not hit the transformers-5.x trust_remote_code regression that broke
Moondream2 (see smoke_caption_moondream.py). Runs in the MAIN env.

This is a diagnostic only. It does not import from any iris/ pipeline
module and does not touch VIRAT annotations. It does not modify the other
two smoke-test scripts or their output files.

Usage:
    python scripts/smoke_caption_qwen.py

Output:
    smoke_caption_results_qwen.jsonl  (repo root)
"""
import json
import os
import re
import sys
import time

import av

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOLVLM_RESULTS_PATH = os.path.join(REPO_ROOT, "smoke_caption_results.jsonl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "smoke_caption_results_qwen.jsonl")

CLIP_DIR = os.path.join(REPO_ROOT, "eval", "data", "virat", "videos")

PROMPT = "Describe what is visible in this single surveillance frame."

# If CPU inference proves impractically slow, cap at the first 15 frames
# (in on-disk order of smoke_caption_results.jsonl) instead of all 40.
MAX_FRAMES_IF_SLOW = 15
SLOW_THRESHOLD_SECONDS_PER_FRAME = 60.0  # decided after timing the first frame

DIFF_LANGUAGE_PATTERN = re.compile(
    r"gone|before|after|no longer|new |moved|missing", re.IGNORECASE
)


def load_frame_manifest_from_smolvlm_results():
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
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    stream.thread_type = "AUTO"
    out = {}
    target_set = set(target_indices)
    for idx, frame in enumerate(container.decode(stream)):
        if idx in target_set:
            out[idx] = frame.to_image()
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
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    # DEVIATION: loaded in bfloat16 rather than float32 -- free RAM on this
    # machine (~5.5GB) is too tight for float32's ~12GB of weights and risks
    # OOM/swap-thrashing. bfloat16 halves that to ~6GB. Flagged per user
    # decision rather than silently downgrading precision.
    print(f"Loading {model_id} on CPU (dtype=bfloat16, see DEVIATION note in load_model) ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    )
    model.eval()
    return processor, model


def caption_frame(processor, model, image):
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return generated_text.strip()


def main():
    manifest = load_frame_manifest_from_smolvlm_results()
    print(f"Loaded {len(manifest)} frame_ids from {SMOLVLM_RESULTS_PATH}")

    manifest = attach_images(manifest)
    if not manifest:
        print("ERROR: no frames recovered, aborting.")
        sys.exit(1)
    print(f"Total frames available: {len(manifest)}")

    try:
        processor, model = load_model()
    except Exception as e:
        print(f"STOP: Qwen2.5-VL-3B failed to load. Raw error:\n{e}")
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
            caption = caption_frame(processor, model, entry["image"])
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
        # cap the run rather than grinding through all 40 frames.
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
    print(f"Total frames captioned: {len(results)}")
    print(f"Load/inference errors: {len(errors)}")
    print(f"Total wall time (captioning loop): {total_elapsed:.1f}s")
    print(f"Avg sec/frame: {avg_sec:.2f}s")
    print(f"Unique captions: {len(exact_counts)} / {len(results)}")
    print(f"Diff-language captions (gone/before/after/no longer/new /moved/missing): "
          f"{diff_language_count} / {len(results)}")
    print("Top 5 most common exact captions:")
    for cap, count in sorted(exact_counts.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  x{count}: {cap[:150]}")
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
