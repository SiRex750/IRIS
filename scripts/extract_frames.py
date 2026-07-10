"""
Extract the 27 kept frames from mov_bbb.mp4 as base64 thumbnails.
Outputs frames_data.json with frame metadata + base64 JPEG images.
"""
import av
import json
import base64
import urllib.request
import io
import numpy as np
from PIL import Image
from iris.charon_v import parse_video

VIDEO_URL  = "https://www.w3schools.com/html/mov_bbb.mp4"
VIDEO_PATH = "mov_bbb.mp4"
OUT_PATH   = "frames_data.json"
THUMB_W    = 192
THUMB_H    = 108   # 16:9

print("Downloading video...")
urllib.request.urlretrieve(VIDEO_URL, VIDEO_PATH)
print("Done. Parsing with Charon-V...")

output_frames, stats = parse_video(VIDEO_PATH, return_stats=True, adaptive=True)
print(f"Charon-V found {len(output_frames)} non-SKIP frames out of {stats['total']} total.")

# Build a lookup: frame_idx → tier/residual/motion
kept = {f["frame_idx"]: f for f in output_frames}

print("Extracting actual image frames...")
container = av.open(VIDEO_PATH)
stream     = container.streams.video[0]
stream.codec_context.options = {"flags2": "+export_mvs"}

results = []
frame_count = 0

for frame in container.decode(video=0):
    idx = frame_count
    frame_count += 1

    if idx not in kept:
        continue

    meta = kept[idx]

    # Convert frame to RGB PIL image and resize to thumbnail
    img_array = frame.to_ndarray(format="rgb24")
    pil_img   = Image.fromarray(img_array)
    pil_img   = pil_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=72)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Motion vector summary: count of non-zero vectors
    mv = meta["motion_vectors"]
    mv_count = len(mv)
    mv_mag = 0.0
    if mv_count > 0:
        mv_mag = float(np.mean([
            (m[4]**2 + m[5]**2)**0.5 for m in mv
        ]))

    results.append({
        "frame_idx":      meta["frame_idx"],
        "timestamp":      round(meta["timestamp"], 3),
        "tier":           meta["tier"],
        "luma_diff_energy": round(meta["luma_diff_energy"], 5),
        "mv_count":       mv_count,
        "mv_magnitude":   round(mv_mag, 3),
        "image_b64":      b64,
    })

container.close()

output = {
    "video":   "mov_bbb.mp4",
    "stats":   stats,
    "frames":  results,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f)

print(f"Saved {len(results)} frames to {OUT_PATH}")
import os
os.remove(VIDEO_PATH)
print("Cleaned up video file.")
