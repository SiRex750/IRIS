import sys
import os
import torch
import gc
import time

# Direct console logging
pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris.pipeline import run_pipeline
from iris.iris_config import IRISConfig, ConfigManager
import iris.aria
import iris.pipeline

def MB(bytes_val):
    return f"{bytes_val / 1024**2:.2f} MB"

original_unload = iris.aria.unload_captioner
original_generate = iris.aria.generate
original_l2_retrieve = iris.pipeline.wrapper_l2_retrieve

def wrapped_l2_retrieve(video_path, query, frames, config=None):
    # Log VRAM immediately before Moondream loads
    print(f"  [VRAM] Immediately before Moondream loads: {MB(torch.cuda.memory_allocated())}")
    torch.cuda.reset_peak_memory_stats()
    # Limit the indexed batch to exactly 15 frames (meeting the 10-30 user requirement while keeping runs fast)
    limited_frames = frames[:15]
    return original_l2_retrieve(video_path, query, limited_frames, config)

def wrapped_unload():
    # Peak during the real captioning batch
    print(f"  [VRAM] Peak VRAM during the real captioning batch: {MB(torch.cuda.max_memory_allocated())}")
    original_unload()
    # VRAM immediately after unload_captioner() runs
    print(f"  [VRAM] VRAM immediately after unload_captioner() runs: {MB(torch.cuda.memory_allocated())}")
    torch.cuda.reset_peak_memory_stats()

def wrapped_generate(prompt, context, model=None):
    # Reset peak before LLM generation phase
    torch.cuda.reset_peak_memory_stats()
    ans = original_generate(prompt, context, model)
    # VRAM while Llama is generating/after generation
    print(f"  [VRAM] Peak VRAM while Llama is generating (via Ollama): {MB(torch.cuda.max_memory_allocated())}")
    print(f"  [VRAM] VRAM after Llama generation: {MB(torch.cuda.memory_allocated())}")
    return ans

def main():
    # Set the config manager's default config to use Moondream with peak_only strategy to speed up testing
    custom_config = IRISConfig(captioner_backend="moondream", retrieval_strategy="peak_only")
    
    def mock_get_config(self):
        return custom_config
    ConfigManager.get_config = mock_get_config

    # Intercept functions for lifecycle VRAM logging
    iris.pipeline.wrapper_l2_retrieve = wrapped_l2_retrieve
    iris.aria.unload_captioner = wrapped_unload
    iris.aria.generate = wrapped_generate
    
    video_path = "test_data/Virat/VIRAT_S_000206_09_001714_001851.mp4"
    queries = [
        "What color is the car?",
        "Is there a person walking?",
        "How many people are in the frame?",
        "Is there a bicycle parked?",
        "Where is the building located?"
    ]
    
    results = []
    
    for i, q in enumerate(queries):
        print(f"\n============================")
        print(f"--- Query {i+1}: {q} ---")
        
        # Capture pre-query allocations
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        print(f"  [VRAM] Start of query: {MB(start_allocated)}")
        
        import traceback
        t_start = time.time()
        try:
            run_pipeline(video_path, q)
        except BaseException as e:
            print(f"\n[CRASH] BaseException/SystemExit occurred during run_pipeline: {type(e)} - {str(e)}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            raise e
        t_end = time.time()
        
        duration = t_end - t_start
        end_allocated = torch.cuda.memory_allocated()
        print(f"  [VRAM] End of query: {MB(end_allocated)}")
        print(f"  [Time] Query {i+1} duration: {duration:.2f} seconds")
        
        results.append({
            "query_idx": i + 1,
            "query": q,
            "start_vram": start_allocated,
            "end_vram": end_allocated,
            "duration": duration
        })
        
    print(f"\n\n==================================================")
    print(f"LIFECYCLE SUMMARY TABLE")
    print(f"==================================================")
    print(f"{'Query #':<8}{'Duration (s)':<15}{'Start VRAM':<15}{'End VRAM':<15}")
    for res in results:
        print(f"{res['query_idx']:<8}{res['duration']:<15.2f}{MB(res['start_vram']):<15}{MB(res['end_vram']):<15}")
    print(f"==================================================")

if __name__ == "__main__":
    main()

