import argparse
import datetime
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import av
import numpy as np
import torch

from iris.aria import BLIPCaptioner

# Pre-defined search paths
SEARCH_PATHS = [
    Path(r"C:\IRIS\test_data"),
    Path(r"C:\IRIS\data"),
    Path(r"C:\Users\akash\Downloads")
]

FLAG_WORDS = {
    "gone", "no longer", "disappeared", "appeared", "before", 
    "after", "missing", "was there", "now", "has left", "used to"
}

def find_virat_dir(args_dir):
    if args_dir:
        paths_to_check = [Path(args_dir)]
    else:
        paths_to_check = SEARCH_PATHS
        
    for p in paths_to_check:
        if not p.exists():
            continue
            
        # For downloads, max depth 2
        # We'll just use a simple walk with depth limit
        max_depth = 2 if "Downloads" in str(p) else 100
        
        for root, dirs, files in os.walk(p):
            depth = Path(root).relative_to(p).parts
            if len(depth) >= max_depth:
                del dirs[:]
                continue
                
            has_video = any(f.lower().endswith(('.mp4', '.avi')) for f in files)
            if has_video:
                return Path(root)
                
    return None

def normalize_caption(caption):
    # Lowercase
    caption = caption.lower()
    # Strip punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption

def check_confabulation(caption):
    lower_cap = caption.lower()
    found = []
    for word in FLAG_WORDS:
        # Check whole word bounds to avoid partial matches
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, lower_cap):
            found.append(word)
    return found

def main():
    print("Started main()", flush=True)
    parser = argparse.ArgumentParser(description="BLIP Caption Benchmark")
    parser.add_argument("--virat-dir", type=str, help="Path to VIRAT directory")
    args = parser.parse_args()
    
    print("Finding VIRAT dir...", flush=True)
    virat_dir = find_virat_dir(args.virat_dir)
    if not virat_dir:
        print("ERROR: No video files found. Pass --virat-dir <path> to specify your VIRAT folder.", flush=True)
        return
        
    print(f"Checking files in {virat_dir}...", flush=True)
    video_files = [f for f in virat_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.mp4', '.avi')]
    if not video_files:
        print("ERROR: No video files found. Pass --virat-dir <path> to specify your VIRAT folder.", flush=True)
        return
        
    print(f"Found {len(video_files)} video(s) in {virat_dir}", flush=True)
    
    # 2. Frame Sampling
    print("Sampling frames...", flush=True)
    sampled_frames = [] # list of (pil_image, frame_idx, timestamp_s, video_file)
    
    for v_path in video_files:
        try:
            container = av.open(str(v_path))
            stream = container.streams.video[0]
            
            target_time_s = 0.0
            actual_frame_idx = 0
            frames_extracted = 0
            
            for frame in container.decode(stream):
                ts = frame.time
                actual_frame_idx += 1
                
                if ts is None:
                    continue
                
                if ts >= target_time_s:
                    pil_img = frame.to_image().convert("RGB")
                    # Downscale huge images to prevent OOM/abort in HuggingFace processor
                    if pil_img.width > 1024 or pil_img.height > 1024:
                        pil_img.thumbnail((1024, 1024))
                        
                    sampled_frames.append((pil_img, actual_frame_idx, ts, v_path.name))
                    target_time_s += 2.0
                    frames_extracted += 1
                    
                    if frames_extracted >= 50:
                        break
                        
            container.close()
        except Exception as e:
            print(f"Error processing {v_path.name}: {e}")
            
    if not sampled_frames:
        print("No frames sampled. Exiting.")
        return
        
    print(f"Sampled {len(sampled_frames)} frames total.")
    
    # 3. BLIP Captioning
    print("Loading BLIP...")
    captioner = BLIPCaptioner()
    # Trigger load lazily
    first_img = sampled_frames[0][0]
    try:
        _ = captioner.caption(first_img)
    except Exception as e:
        print(f"Error during initial BLIP load: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    print("Running captioning...", flush=True)
    results = []
    
    for img, f_idx, ts, v_name in sampled_frames:
        print(f"Captioning frame {f_idx} of video {v_name}...", flush=True)
        t0 = time.perf_counter()
        failed = False
        caption_text = ""
        try:
            caption_text = captioner.caption(img)
        except Exception:
            caption_text = "[CAPTION_FAILED]"
            failed = True
        t1 = time.perf_counter()
        print(f"Done frame {f_idx}. Failed={failed}", flush=True)
        
        latency_ms = (t1 - t0) * 1000.0
        
        flags = check_confabulation(caption_text)
        
        results.append({
            "video_file": v_name,
            "frame_idx": f_idx,
            "timestamp_s": ts,
            "caption": caption_text,
            "latency_ms": latency_ms,
            "is_failed": failed,
            "confabulation_flagged": len(flags) > 0,
            "flag_words_found": flags
        })
        
    # 4. Compute Metrics
    total_frames = len(results)
    succeeded_results = [r for r in results if not r["is_failed"]]
    failed_results = [r for r in results if r["is_failed"]]
    
    total_succeeded = len(succeeded_results)
    total_failed = len(failed_results)
    
    exact_unique_count = 0
    exact_uniqueness_rate = 0.0
    normalized_unique_count = 0
    mean_pairwise_cosine = 0.0
    top_10_repeated = []
    
    confab_count = sum(1 for r in succeeded_results if r["confabulation_flagged"])
    confab_rate = confab_count / total_succeeded if total_succeeded > 0 else 0.0
    
    mean_latency = 0.0
    std_latency = 0.0
    min_latency = 0.0
    max_latency = 0.0
    fps = 0.0
    
    if total_succeeded > 0:
        captions = [r["caption"] for r in succeeded_results]
        exact_unique = set(captions)
        exact_unique_count = len(exact_unique)
        exact_uniqueness_rate = exact_unique_count / total_succeeded
        
        norm_captions = [normalize_caption(c) for c in captions]
        normalized_unique_count = len(set(norm_captions))
        
        counts = Counter(captions)
        top_10 = counts.most_common(10)
        top_10_repeated = [{"caption": k, "count": v} for k, v in top_10 if v > 1]
        
        # Latency stats (across all calls, even failed ones, as per instruction but let's just use all)
        latencies = [r["latency_ms"] for r in results]
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        
        # Semantic diversity
        # Free up memory before loading sentence transformers to avoid silent OOM aborts
        device = getattr(captioner, "_device", "cpu")
        if str(device) != "cpu" and torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram_mb = None
            
        del captioner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Computing semantic diversity...", flush=True)
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        encoded_input = tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        token_embeddings = model_output[0]
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Compute cosine similarity matrix
        cos_sim_matrix = torch.matmul(embeddings, embeddings.T)
        # Compute pairwise distance (1 - cos_sim)
        cos_dist_matrix = 1.0 - cos_sim_matrix
        
        # Extract upper triangle (excluding diagonal)
        n = embeddings.size(0)
        if n > 1:
            upper_tri_indices = torch.triu_indices(n, n, offset=1)
            pairwise_distances = cos_dist_matrix[upper_tri_indices[0], upper_tri_indices[1]]
            mean_pairwise_cosine = float(pairwise_distances.mean().item())
            
    # 5. Output
    print("\n===== BLIP CAPTION BENCHMARK =====")
    print("Model:        Salesforce/blip-image-captioning-base")
    print(f"Device:       {device}")
    print(f"Videos:       {len(video_files)}")
    print(f"Frames:       {total_frames} / {total_succeeded} / {total_failed}")
    print("\n--- UNIQUENESS ---")
    print(f"Exact unique:       {exact_unique_count} / {total_succeeded}  ({exact_uniqueness_rate*100:.1f}%)")
    print(f"Normalized unique:  {normalized_unique_count} / {total_succeeded}  ({(normalized_unique_count/total_succeeded*100) if total_succeeded else 0:.1f}%)")
    print(f"Mean pairwise cosine distance: {mean_pairwise_cosine:.3f}")
    print("\n--- TOP REPEATED CAPTIONS ---")
    for item in top_10_repeated:
        print(f"{item['count']}x: \"{item['caption']}\"")
        
    print("\n--- CONFABULATION ---")
    print(f"Flagged captions: {confab_count} ({confab_rate*100:.1f}%)")
    for r in succeeded_results:
        if r["confabulation_flagged"]:
            print(f"- [\"{','.join(r['flag_words_found'])}\"] {r['caption']}")
            
    print("\n--- SPEED ---")
    print(f"Mean latency:   {mean_latency:.1f}ms  ({fps:.1f} fps)")
    if peak_vram_mb is not None:
        print(f"Peak VRAM:      {peak_vram_mb:.1f}MB")
    else:
        print("Peak VRAM:      N/A")
    print("===================================")
    
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "caption_benchmark_blip.json"
    
    out_data = {
        "model": "Salesforce/blip-image-captioning-base",
        "captioner_mode": "unconditional",
        "device": str(device),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "virat_dir": str(virat_dir),
        "summary": {
            "total_frames": total_frames,
            "total_succeeded": total_succeeded,
            "total_failed": total_failed,
            "exact_unique_count": exact_unique_count,
            "exact_uniqueness_rate": exact_uniqueness_rate,
            "normalized_unique_count": normalized_unique_count,
            "mean_pairwise_cosine_distance": mean_pairwise_cosine,
            "confabulation_flag_count": confab_count,
            "confabulation_rate": confab_rate,
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "frames_per_second": fps,
            "peak_vram_mb": peak_vram_mb
        },
        "top_10_repeated": top_10_repeated,
        "confabulation_flagged_captions": [r for r in succeeded_results if r["confabulation_flagged"]],
        "captions": results
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("eval_results/crash.log", "w") as f:
            traceback.print_exc(file=f)
        raise
