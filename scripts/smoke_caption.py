"""Standalone smoke test: caption real VIRAT frames with SmolVLM2-2.2B and
dump raw output for manual inspection of BLIP-style mode collapse.

This is a diagnostic only. It does not import from any iris/ pipeline
module (ingest.py, query.py, Charon-V/PPR code) and does not touch VIRAT
annotations. Frame "peak"/"nonpeak" labels are a standalone motion-diff
proxy computed in this file, NOT the real Charon-V labels -- see
DEVIATION note below.

Usage:
    python scripts/smoke_caption.py

Output:
    smoke_caption_results.jsonl  (repo root)
"""
import json
import os
import sys
import time

import av
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "smoke_caption_results.jsonl")

CLIPS = [
    os.path.join(REPO_ROOT, "eval", "data", "virat", "videos", "VIRAT_S_000200_00_000100_000171.mp4"),
    os.path.join(REPO_ROOT, "eval", "data", "virat", "videos", "VIRAT_S_000200_03_000657_000899.mp4"),
]

FRAMES_PER_CLIP = 20          # -> 40 total across 2 clips
PEAK_FRACTION = 0.5           # top-50th-percentile motion score = "peak" pool
THUMB_SIZE = (64, 64)         # small grayscale thumbnail for motion scoring only
PROMPT = "Describe what is happening in this surveillance frame."

DEVIATION_NOTE = (
    "DEVIATION: 'peak'/'nonpeak' labels below are NOT real Charon-V L1/L2/L3 "
    "labels. Charon-V logic lives inside iris/ingest.py and this script is "
    "required to stay standalone (no iris/ imports). Instead, peak/nonpeak is "
    "a simple frame-differencing motion-score proxy computed independently in "
    "this script (top 50% mean-abs-diff vs previous frame = 'peak', bottom 50% "
    "= 'nonpeak'), applied per-clip. Treat the peak_or_nonpeak field as a rough "
    "high-motion/low-motion tag, not a pipeline output."
)


def decode_grayscale_thumbnails(video_path):
    """Sequential decode pass: returns list of (frame_index, small_gray_ndarray)."""
    container = av.open(video_path)
    stream = next(s for s in container.streams if s.type == "video")
    stream.thread_type = "AUTO"
    out = []
    for idx, frame in enumerate(container.decode(stream)):
        gray = frame.to_ndarray(format="gray8")
        h, w = gray.shape
        step_y = max(1, h // THUMB_SIZE[1])
        step_x = max(1, w // THUMB_SIZE[0])
        small = gray[::step_y, ::step_x].astype(np.float32)
        out.append((idx, small))
    container.close()
    return out


def compute_motion_scores(thumbnails):
    """Mean abs diff vs previous frame; first frame gets score 0."""
    scores = {}
    prev = None
    for idx, small in thumbnails:
        if prev is None:
            scores[idx] = 0.0
        else:
            min_h = min(prev.shape[0], small.shape[0])
            min_w = min(prev.shape[1], small.shape[1])
            scores[idx] = float(np.abs(small[:min_h, :min_w] - prev[:min_h, :min_w]).mean())
        prev = small
    return scores


def select_frame_indices(scores, n_frames, peak_fraction):
    """Stratified sample: half from top motion-score pool, half from bottom."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    n_peak_pool = max(1, int(len(ranked) * peak_fraction))
    peak_pool = [idx for idx, _ in ranked[:n_peak_pool]]
    nonpeak_pool = [idx for idx, _ in ranked[n_peak_pool:]]

    n_peak = n_frames // 2
    n_nonpeak = n_frames - n_peak

    def spread_sample(pool, k):
        pool_sorted = sorted(pool)
        if len(pool_sorted) <= k:
            return pool_sorted
        step = len(pool_sorted) / k
        return [pool_sorted[int(i * step)] for i in range(k)]

    peak_selected = spread_sample(peak_pool, min(n_peak, len(peak_pool)))
    nonpeak_selected = spread_sample(nonpeak_pool, min(n_nonpeak, len(nonpeak_pool)))

    selected = {idx: "peak" for idx in peak_selected}
    selected.update({idx: "nonpeak" for idx in nonpeak_selected})
    return selected


def extract_full_frames(video_path, target_indices):
    """Second sequential decode pass: grab full-res PIL frames for target indices only."""
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


def build_frame_manifest():
    manifest = []  # list of dicts: frame_id, source_clip, peak_or_nonpeak, image
    for clip_path in CLIPS:
        clip_name = os.path.basename(clip_path)
        if not os.path.isfile(clip_path):
            print(f"WARNING: clip not found, skipping: {clip_path}")
            continue
        print(f"[{clip_name}] pass 1/2: decoding grayscale thumbnails for motion scoring...")
        thumbs = decode_grayscale_thumbnails(clip_path)
        scores = compute_motion_scores(thumbs)
        selected = select_frame_indices(scores, FRAMES_PER_CLIP, PEAK_FRACTION)
        print(f"[{clip_name}] selected {len(selected)} frame indices "
              f"({sum(1 for v in selected.values() if v == 'peak')} peak, "
              f"{sum(1 for v in selected.values() if v == 'nonpeak')} nonpeak) "
              f"out of {len(thumbs)} total frames")

        print(f"[{clip_name}] pass 2/2: extracting full-res frames for selected indices...")
        full_frames = extract_full_frames(clip_path, selected.keys())

        for idx, tag in sorted(selected.items()):
            if idx not in full_frames:
                print(f"WARNING: frame {idx} not recovered on second pass for {clip_name}, skipping")
                continue
            manifest.append({
                "frame_id": f"{clip_name}:{idx}",
                "source_clip": clip_name,
                "frame_index": idx,
                "peak_or_nonpeak": tag,
                "image": full_frames[idx],
            })
    return manifest


def load_model():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on device={device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return processor, model, device


def caption_frame(processor, model, device, image):
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
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]
    return generated_text.strip()


def near_duplicate_key(caption):
    """Cheap normalization for near-duplicate detection: lowercase, strip
    punctuation, collapse whitespace."""
    import re
    norm = caption.lower()
    norm = re.sub(r"[^\w\s]", "", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def main():
    print(DEVIATION_NOTE)
    print()

    manifest = build_frame_manifest()
    if not manifest:
        print("ERROR: no frames selected, aborting.")
        sys.exit(1)
    print(f"\nTotal frames selected for captioning: {len(manifest)}")

    try:
        processor, model, device = load_model()
    except Exception as e:
        print(f"STOP: SmolVLM2-2.2B failed to load. Raw error:\n{e}")
        sys.exit(1)

    results = []
    errors = []
    t0 = time.time()
    for i, entry in enumerate(manifest):
        try:
            caption = caption_frame(processor, model, device, entry["image"])
        except Exception as e:
            errors.append({"frame_id": entry["frame_id"], "error": str(e)})
            print(f"  [{i+1}/{len(manifest)}] {entry['frame_id']} ERROR: {e}")
            continue
        results.append({
            "frame_id": entry["frame_id"],
            "source_clip": entry["source_clip"],
            "peak_or_nonpeak": entry["peak_or_nonpeak"],
            "caption": caption,
        })
        print(f"  [{i+1}/{len(manifest)}] {entry['frame_id']} ({entry['peak_or_nonpeak']}): {caption[:100]}")

    elapsed = time.time() - t0

    error_rate = len(errors) / len(manifest) if manifest else 1.0
    if error_rate > 0.10:
        print(f"\nSTOP: {len(errors)}/{len(manifest)} frames errored ({error_rate:.1%} > 10%). Raw errors:")
        for e in errors:
            print(f"  {e['frame_id']}: {e['error']}")
        # still dump whatever succeeded, then exit non-zero
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    exact_counts = {}
    near_counts = {}
    for r in results:
        exact_counts[r["caption"]] = exact_counts.get(r["caption"], 0) + 1
        key = near_duplicate_key(r["caption"])
        near_counts[key] = near_counts.get(key, 0) + 1

    print("\n=== SUMMARY ===")
    print(f"Total frames captioned: {len(results)} (of {len(manifest)} selected)")
    print(f"Load/inference errors: {len(errors)}")
    print(f"Elapsed captioning time: {elapsed:.1f}s")
    print(f"Unique exact captions: {len(exact_counts)} / {len(results)}")
    print(f"Unique near-duplicate captions (normalized): {len(near_counts)} / {len(results)}")
    print("Top 5 most common exact captions:")
    for cap, count in sorted(exact_counts.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  x{count}: {cap[:150]}")
    print(f"\nResults written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
