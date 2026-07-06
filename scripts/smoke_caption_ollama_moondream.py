"""Standalone smoke test: caption the SAME 40 VIRAT frames used by
scripts/smoke_caption.py (SmolVLM2) via Moondream through Ollama
(llama.cpp runtime), as a CPU-runtime-feasibility check.

The PyTorch/transformers path for Moondream2 is blocked by a transformers
5.x remote-code regression (all_tied_weights_keys, no upstream fix) --
see smoke_test_notes.md. Ollama uses llama.cpp, a different loader where
that bug doesn't exist, and runs a quantized checkpoint (moondream:1.8b,
~1.7GB, an OLDER checkpoint than vikhyatk/moondream2 2025-06-21). This
script answers: does a quantized model on a CPU-optimized runtime break
the CPU speed wall hit in PyTorch (SmolVLM2 ~74s/frame, Qwen2.5-VL-3B
~467s/frame)? Speed here is a valid llama.cpp-vs-PyTorch signal; quality
is comparable in spirit to the other runs but NOT a same-precision,
same-checkpoint comparison against Moondream2's current best.

This is a diagnostic only. It does not import from any iris/ pipeline
module and does not touch VIRAT annotations. It does not modify the
other smoke-test scripts or their output files.

Usage:
    python scripts/smoke_caption_ollama_moondream.py

Output:
    smoke_caption_results_ollama_moondream.jsonl  (repo root)

Requires: Ollama running locally (http://localhost:11434) with
moondream:1.8b already pulled (`ollama pull moondream:1.8b`).
"""
import base64
import io
import json
import os
import re
import sys
import time

import av
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOLVLM_RESULTS_PATH = os.path.join(REPO_ROOT, "smoke_caption_results.jsonl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "smoke_caption_results_ollama_moondream.jsonl")

CLIP_DIR = os.path.join(REPO_ROOT, "eval", "data", "virat", "videos")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream:1.8b"
PROMPT = "Describe what is visible in this single surveillance frame."

# If inference proves impractically slow, cap at the first 15 frames
# instead of all 40. Quantized llama.cpp may clear all 40 fast -- if
# frame 1 is quick, this never engages.
MAX_FRAMES_IF_SLOW = 15
SLOW_THRESHOLD_SECONDS_PER_FRAME = 60.0

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


def image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_ollama_and_model():
    try:
        resp = requests.get("http://localhost:11434/api/version", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable at localhost:11434: {e}")

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        raise RuntimeError(f"Failed to list Ollama models: {e}")

    if not any(t.startswith(MODEL_NAME) for t in tags):
        raise RuntimeError(
            f"Model '{MODEL_NAME}' not found in Ollama (`ollama pull moondream:1.8b`). "
            f"Available: {tags}"
        )


def caption_frame(image_b64):
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data["response"].strip()


def main():
    try:
        check_ollama_and_model()
    except Exception as e:
        print(f"STOP: Ollama/model check failed. Raw error:\n{e}")
        sys.exit(1)

    manifest = load_frame_manifest_from_smolvlm_results()
    print(f"Loaded {len(manifest)} frame_ids from {SMOLVLM_RESULTS_PATH}")

    manifest = attach_images(manifest)
    if not manifest:
        print("ERROR: no frames recovered, aborting.")
        sys.exit(1)
    print(f"Total frames available: {len(manifest)}")

    print(f"Encoding frames to base64 PNG for {MODEL_NAME} via Ollama ({OLLAMA_URL}) ...")
    for entry in manifest:
        entry["image_b64"] = image_to_base64(entry["image"])

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
            caption = caption_frame(entry["image_b64"])
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
    print(f"Run capped at {MAX_FRAMES_IF_SLOW} frames due to slowness: {capped}")
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
