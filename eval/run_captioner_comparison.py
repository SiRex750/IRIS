import argparse
import datetime
import json
import math
import os
import re
import string
import sys
import time
import subprocess
import threading
import io
import gc
from collections import Counter
from pathlib import Path

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import av
import numpy as np
import torch
import PIL.Image

# Target paths
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "captioner_comparison_full.log"
SUMMARY_FILE = RESULTS_DIR / "captioner_comparison_summary.json"

# Flag words for confabulation detection
FLAG_WORDS = {
    "gone", "no longer", "disappeared", "appeared", "before", 
    "after", "missing", "was there", "now", "has left", "used to"
}

def log_message(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def normalize_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption

def check_confabulation(caption):
    lower_cap = caption.lower()
    found = []
    for word in FLAG_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, lower_cap):
            found.append(word)
    return found

class VRAMMonitor:
    def __init__(self):
        self.peak_nvidia_smi = 0.0
        self.peak_ollama_ps = ""
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.peak_nvidia_smi = 0.0
        self.peak_ollama_ps = ""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.thread:
            self.stop_event.set()
            self.thread.join()

    def _monitor(self):
        while not self.stop_event.is_set():
            # Query nvidia-smi
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL
                )
                val = float(out.decode().strip())
                if val > self.peak_nvidia_smi:
                    self.peak_nvidia_smi = val
            except Exception:
                pass

            # Query ollama ps
            try:
                out = subprocess.check_output(
                    ["ollama", "ps"],
                    stderr=subprocess.DEVNULL
                ).decode()
                lines = [line.strip() for line in out.strip().split('\n') if line.strip()]
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = re.split(r'\s{2,}', line)
                        if len(parts) >= 3:
                            size_str = parts[2]
                            self.peak_ollama_ps = size_str
            except Exception:
                pass

            time.sleep(0.1)

class MoondreamCaptioner:
    def __init__(self, device="cuda-fp16"):
        self._model = None
        self._tokenizer = None
        self._device = device

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True
        )
        if self._device == "cuda-fp16":
            import torch
            self._model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True
            )
            self._model = self._model.to("cpu")

    def caption(self, pil_image):
        if self._model is None:
            self._load()
        enc = self._model.encode_image(pil_image)
        return self._model.answer_question(
            enc,
            "Describe only what is visually present in this single image. State objects, people, colors, and positions. Do not describe motion or changes.",
            self._tokenizer
        ).strip()

class OllamaCaptioner:
    def __init__(self, model_name):
        self.model_name = model_name

    def caption(self, pil_image):
        import ollama
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe only what is visually present in this single image. State objects, people, colors, and positions. Do not describe motion or changes.',
                    'images': [img_bytes]
                }
            ],
            options={
                'num_ctx': 512
            }
        )
        return response['message']['content'].strip()

def search_virat_dir():
    search_paths = [
        Path(r"c:\Users\akash\Documents\Iris\test_data\Virat"),
        Path(r"c:\Users\akash\Documents\Iris\test_data"),
        Path(r"C:\IRIS\test_data"),
        Path(r"C:\IRIS\data")
    ]
    for p in search_paths:
        if p.exists() and p.is_dir():
            has_video = any(f.name.lower().endswith(('.mp4', '.avi')) for f in p.iterdir() if f.is_file())
            if has_video:
                return p
    return None

def sample_frames(virat_dir, count_limit=None):
    video_files = [f for f in virat_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.mp4', '.avi')]
    sampled_frames = []
    
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
                    # Original validated benchmark downscale threshold (1024)
                    if pil_img.width > 1024 or pil_img.height > 1024:
                        pil_img.thumbnail((1024, 1024))
                    sampled_frames.append((pil_img, actual_frame_idx, ts, v_path.name))
                    target_time_s += 2.0
                    frames_extracted += 1
                    if frames_extracted >= 50:
                        break
            container.close()
        except Exception as e:
            log_message(f"Error sampling from {v_path.name}: {e}")
            
    if count_limit:
        return sampled_frames[:count_limit]
    return sampled_frames

def check_ollama_processor(model_name):
    try:
        out = subprocess.check_output(["ollama", "ps"], stderr=subprocess.DEVNULL).decode()
        lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            for line in lines[1:]:
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 4:
                    cur_name = parts[0].strip()
                    processor = parts[3].strip()
                    if cur_name == model_name or cur_name.split(':')[0] == model_name.split(':')[0]:
                        return processor
    except Exception:
        pass
    return None

def stop_all_ollama_models(models):
    for m in models:
        if m.startswith("ollama-"):
            name = m.split("ollama-")[1]
            try:
                subprocess.run(["ollama", "stop", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

def wait_for_gpu_clear(timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ollama_clear = True
        try:
            out = subprocess.check_output(["ollama", "ps"], stderr=subprocess.DEVNULL).decode()
            lines = [l.strip() for l in out.strip().split('\n') if l.strip()]
            if len(lines) > 1:
                ollama_clear = False
        except Exception:
            pass
            
        if ollama_clear:
            return True
            
        log_message("GPU is not clear (resident Ollama model active). Waiting 5 seconds...")
        time.sleep(5)
    return False

def send_toast_notification():
    try:
        cmd = [
            "powershell",
            "-Command",
            "(New-Object -ComObject Wscript.Shell).Popup('Captioner benchmark finished — check captioner_comparison_summary.json', 0, 'IRIS Captioner Benchmark', 64)"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    log_message("Starting Captioner Benchmark Comparison Suite")
    
    virat_dir = search_virat_dir()
    if not virat_dir:
        log_message("ERROR: No VIRAT videos directory found.")
        sys.exit(1)
    log_message(f"Using VIRAT directory: {virat_dir}")
    
    models_to_bench = [
        "moondream2-HF-fp16",
        "ollama-qwen3.5:4b",
        "ollama-qwen3.5:2b",
        "ollama-gemma4:e2b",
        "ollama-gemma4:e4b",
        "ollama-moondream2"
    ]
    
    # 1. Clean startup
    log_message("Performing startup cleanup...")
    stop_all_ollama_models(models_to_bench)
    wait_for_gpu_clear()
    
    # Extract identical frames for Phase 1 (10 frames)
    log_message("Extracting 10 frames for Phase 1 Screening...")
    phase1_frames = sample_frames(virat_dir, count_limit=10)
    log_message(f"Extracted {len(phase1_frames)} frames.")
    
    phase1_results = {}
    passed_models = []
    
    # =========================================================================
    # PHASE 1: SCREENING PHASE
    # =========================================================================
    log_message("=========================================================================")
    log_message("PHASE 1: SCREENING START")
    log_message("=========================================================================")
    
    for model_key in models_to_bench:
        log_message(f"Screening model: {model_key}")
        
        # Verify GPU is clear
        stop_all_ollama_models(models_to_bench)
        wait_for_gpu_clear()
        
        captioner = None
        monitor = VRAMMonitor()
        latencies = []
        processor_info = ""
        passed = False
        reason = ""
        
        try:
            monitor.start()
            
            if model_key == "moondream2-HF-fp16":
                captioner = MoondreamCaptioner(device="cuda-fp16")
                # Trigger a load/first query
                _ = captioner.caption(phase1_frames[0][0])
                processor_info = "100% GPU (PyTorch CUDA)"
            else:
                ollama_name = model_key.split("ollama-")[1]
                log_message(f"Ensuring model {ollama_name} is pulled in Ollama...")
                subprocess.run(["ollama", "pull", ollama_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                captioner = OllamaCaptioner(ollama_name)
                # Trigger a load/first query
                _ = captioner.caption(phase1_frames[0][0])
                
                # Check processor via ollama ps
                time.sleep(2)
                processor_info = check_ollama_processor(ollama_name)
                if not processor_info:
                    processor_info = "Unknown"
            
            # Benchmark 10 frames
            for idx, (img, _, _, _) in enumerate(phase1_frames):
                t0 = time.perf_counter()
                _ = captioner.caption(img)
                latencies.append((time.perf_counter() - t0) * 1000.0)
            
            monitor.stop()
            mean_lat = np.mean(latencies)
            
            # Determine threshold check: Must be fully on GPU (contains 100% GPU or is PyTorch CUDA)
            if "100% GPU" in processor_info:
                passed = True
                reason = f"Fits fully on GPU (Processor: {processor_info})"
            else:
                passed = False
                reason = f"Fails GPU verification (Processor: {processor_info})"
                
            log_message(f"  Result: {'PASS' if passed else 'FAIL'} | Reason: {reason} | Mean Latency: {mean_lat:.1f}ms")
            
            phase1_results[model_key] = {
                "passed": passed,
                "reason": reason,
                "mean_latency_ms": mean_lat,
                "peak_vram_nvidia_smi": monitor.peak_nvidia_smi,
                "peak_vram_ollama_ps": monitor.peak_ollama_ps
            }
            
            if passed:
                passed_models.append(model_key)
                
        except Exception as err:
            monitor.stop()
            log_message(f"  [ERROR] Failed during screening of {model_key}: {err}")
            phase1_results[model_key] = {
                "passed": False,
                "reason": f"Exception: {str(err)}",
                "mean_latency_ms": 0.0,
                "peak_vram_nvidia_smi": 0.0,
                "peak_vram_ollama_ps": ""
            }
        finally:
            del captioner
            stop_all_ollama_models(models_to_bench)
            wait_for_gpu_clear()
            
    log_message("PHASE 1 COMPLETE.")
    log_message(f"Models passing screening: {passed_models}")
    
    # =========================================================================
    # PHASE 2: BENCHMARK PHASE (113 frames)
    # =========================================================================
    log_message("=========================================================================")
    log_message("PHASE 2: BENCHMARK START")
    log_message("=========================================================================")
    
    log_message("Extracting full 113 frames for Phase 2 Benchmarking...")
    phase2_frames = sample_frames(virat_dir)
    log_message(f"Extracted {len(phase2_frames)} frames.")
    
    phase2_results = {}
    
    for model_key in passed_models:
        log_message(f"Running full benchmark for: {model_key}")
        
        # Verify GPU is clear
        stop_all_ollama_models(models_to_bench)
        wait_for_gpu_clear()
        
        captioner = None
        monitor = VRAMMonitor()
        latencies = []
        results = []
        
        try:
            monitor.start()
            
            if model_key == "moondream2-HF-fp16":
                captioner = MoondreamCaptioner(device="cuda-fp16")
            else:
                ollama_name = model_key.split("ollama-")[1]
                captioner = OllamaCaptioner(ollama_name)
                
            for idx, (img, f_idx, ts, v_name) in enumerate(phase2_frames):
                t0 = time.perf_counter()
                failed = False
                caption_text = ""
                try:
                    caption_text = captioner.caption(img)
                except Exception as exc:
                    caption_text = "[CAPTION_FAILED]"
                    failed = True
                    log_message(f"  [ERROR] Failed on frame {f_idx} (video {v_name}): {exc}")
                    
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)
                
                flags = check_confabulation(caption_text)
                results.append({
                    "caption": caption_text,
                    "latency_ms": lat_ms,
                    "is_failed": failed,
                    "confabulation_flagged": len(flags) > 0,
                    "flag_words_found": flags
                })
                
            monitor.stop()
            
            succeeded_results = [r for r in results if not r["is_failed"]]
            total_succeeded = len(succeeded_results)
            
            exact_uniq = 0.0
            norm_uniq = 0.0
            mean_pairwise_cosine = 0.0
            confab_count = 0
            confab_rate = 0.0
            
            if total_succeeded > 0:
                captions = [r["caption"] for r in succeeded_results]
                exact_uniq = (len(set(captions)) / total_succeeded) * 100.0
                
                norm_captions = [normalize_caption(c) for c in captions]
                norm_uniq = (len(set(norm_captions)) / total_succeeded) * 100.0
                
                confab_count = sum(1 for r in succeeded_results if r["confabulation_flagged"])
                confab_rate = (confab_count / total_succeeded) * 100.0
                
                # Compute semantic diversity
                try:
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
                    cos_sim_matrix = torch.matmul(embeddings, embeddings.T)
                    cos_dist_matrix = 1.0 - cos_sim_matrix
                    
                    n = embeddings.size(0)
                    if n > 1:
                        upper_tri_indices = torch.triu_indices(n, n, offset=1)
                        pairwise_distances = cos_dist_matrix[upper_tri_indices[0], upper_tri_indices[1]]
                        mean_pairwise_cosine = float(pairwise_distances.mean().item())
                except Exception as e:
                    log_message(f"  [WARNING] Failed to compute semantic similarity: {e}")
                    
            mean_lat = np.mean(latencies) if latencies else 0.0
            peak_pytorch = 0.0
            if torch.cuda.is_available():
                peak_pytorch = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            log_message(f"  Finished: Uniqueness (Exact/Norm): {exact_uniq:.1f}%/{norm_uniq:.1f}% | Cos Dist: {mean_pairwise_cosine:.3f} | Latency: {mean_lat:.1f}ms")
            
            phase2_results[model_key] = {
                "exact_uniqueness": exact_uniq,
                "normalized_uniqueness": norm_uniq,
                "mean_cosine_dist": mean_pairwise_cosine,
                "confab_count": confab_count,
                "confab_rate": confab_rate,
                "mean_latency_ms": mean_lat,
                "peak_vram_nvidia_smi": monitor.peak_nvidia_smi,
                "peak_vram_ollama_ps": monitor.peak_ollama_ps,
                "peak_vram_pytorch": peak_pytorch
            }
            
        except Exception as err:
            monitor.stop()
            log_message(f"  [ERROR] Benchmark run crashed for {model_key}: {err}")
        finally:
            del captioner
            stop_all_ollama_models(models_to_bench)
            wait_for_gpu_clear()
            
    # Combine results
    final_output = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase1_screen": phase1_results,
        "phase2_benchmark": phase2_results
    }
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as sf:
        json.dump(final_output, sf, indent=2)
        
    log_message("PHASE 2 COMPLETE.")
    log_message(f"Final structured summary saved to {SUMMARY_FILE}")
    log_message("BENCHMARK_COMPLETE")
    
    # Trigger Toast Notification
    send_toast_notification()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_message(f"Coordinator crashed globally: {e}")
        raise
