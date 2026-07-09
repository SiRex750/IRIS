# IRIS VLM Captioning Experiment Log

This document log outlines the full progression of experiments, runtime investigations, and findings regarding the visual language models (VLMs) evaluated for the IRIS project captioning pipeline.

## 1. Initial Feasibility Smoke Tests (CPU Execution)
During the initial development phase, several VLM architectures were tested on CPU-only execution paths. The key metrics gathered were:

| Model | Framework | Precision | Avg Latency | Confabulation Rate | Observations / Quality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolVLM2-2.2B** | PyTorch | fp32 | ~74.0s / frame | Moderate | Confabulates before/after temporal-diff language. |
| **Qwen2.5-VL-3B** | PyTorch | bf16 | ~467.0s / frame | Low (0/6) | Boilerplate-heavy, generic templates; poor discrimination. |
| **Moondream2-2B** | PyTorch | fp16 | N/A | N/A | Blocked by HuggingFace `all_tied_weights_keys` regression. |
| **Moondream:1.8b** | llama.cpp (Ollama) | Q4_0 | ~13.1s / frame | Moderate (Object) | No temporal confabulation; high object hallucination. |

### CPU Feasibility Conclusions
1. **PyTorch on CPU is Impractical**: Running standard PyTorch VLM inference on CPU takes between 74s to 467s per frame, rendering it completely unusable for processing video files.
2. **Quantized llama.cpp Breaks the Wall**: Running GGUF quantized models via Ollama achieves a **5.6x to 35x speedup** on CPU (~13s/frame).
3. **Quality & Hallucination Bottleneck**: Smaller 2B-class models exhibit significant object hallucinations (inventing stop signs, benches, etc. that do not exist). Captions must remain a lazy, secondary weak evidence signal; the primary video retrieval must rely on CLIP embeddings.

---

## 2. GPU VRAM Collision & Execution Bottlenecks
When migrating the benchmark to a 6.1 GB VRAM GPU environment, a critical bottleneck occurred:
* **The Collision**: `ollama-qwen3.5:4b` requires ~6.0 GB of memory when loaded with Ollama's default `num_ctx = 4096` context window. Due to host OS and IDE VRAM overhead (~1.5 GB), the available VRAM is insufficient.
* **The Symptom**: Ollama loaded the model in **hybrid CPU/GPU mode** (31% CPU layers, 69% GPU layers).
* **The Performance Impact**: During execution, activation tensors are continuously transferred across the PCIe bus between the system RAM and GPU memory. This caused the query latency to spike to **~85 seconds per frame**.
* **Mitigation**:
  1. **Context Constraining**: Constrained context size to `512` (`options={"num_ctx": 512}`), freeing up KV cache memory.
  2. **Resolution Constraint**: Rescaled images to `224x224`.
  3. **Outcome**: Achieved a **5.2x speedup** (~12-18s/frame).

---

## 3. Captioner Benchmark Comparison Suite
To perform an automated, unattended evaluation of VLM options, we implemented a two-phase benchmark suite (`eval/run_captioner_comparison.py`):

### Phase 1: Screening Phase (10 frames)
Checks if the candidate model can fit 100% on the GPU:
* **Moondream2-HF-fp16**: Automatically passes (PyTorch CUDA execution).
* **ollama-qwen3.5:4b**: Expected to fail due to hybrid VRAM spillover.
* **ollama-qwen3.5:2b**: Expected to pass (fully fits on GPU VRAM).
* **ollama-gemma4:e2b**: Expected to pass (fully fits on GPU VRAM).
* **ollama-gemma4:e4b**: Expected to pass/fail depending on model memory footprint.
* **ollama-moondream2**: Expected to pass (fully fits on GPU VRAM).

### Phase 2: Quality Benchmarking (113 frames)
Runs the full 113-frame quality suite (1024x1024 resolution) ONLY for models that passed the Phase 1 GPU screening. This prevents slow hybrid-CPU models from blocking the runner resources.

---

## 4. Run Status & Log Output
The automated benchmark runs in the background. The full log files are written directly to the repository:
* **Log Location**: `eval/results/captioner_comparison_full.log`
* **JSON Summary Table**: `eval/results/captioner_comparison_summary.json`
* **Archival Log Export**: A 2,169-page complete chronological session log has been exported and saved to `docs/antigravity_session_log_2026-07.pdf`.
