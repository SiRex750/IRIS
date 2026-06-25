# IRIS — Intelligent Residual Indexing System

Codec-native hierarchical memory system for efficient video-language understanding.

**Target venues:** CVPR / MLSys  
**Supervisor:** Dr. Uma D, CDSAML, PES University

## Core novelty

IRIS uses H.264 codec residual energy as a unified cognitive controller that simultaneously gates (1) spatiotemporal knowledge graph indexing and (2) NLI-based hallucination verification. Local maxima of the residual energy curve (PEAK frames) are identified via persistent homology (via ripser or equivalent) — a more precise keyframe signal than the absolute thresholds used in prior work (CodecSight 2026, OneVision-Encoder).

## Architecture

```
.mp4 → charon_v (Codec Oracle)
         ↓ PEAK / SALIENT / CANDIDATE / SKIP tiers
       l1_elysium (Active Context Cache)
         ↓ residual-pressure-scaled budget
       l2_asphodel (Motion-Aware Video RAG Graph)
         ↓ hybrid retrieval: (semantic × α) + (motion × β)
       aria (LLM Brain — ARIA model)
         ↓ generated claims → SCRATCH
       cerberus_v (NLI Truth Gate)
         ↓ DeBERTa-v3 verification
       Final Verified Answer
```

## Ablation table

| Condition   | Codec gating | NLI verification |
|-------------|--------------|------------------|
| Baseline    | None         | None             |
| Ablation 1  | KG only      | None             |
| Ablation 2  | None         | Uniform NLI      |
| Full IRIS   | Both jointly | Risk-proportional|

## Team tracks

| Track | Owner | Files |
|-------|-------|-------|
| A — Codec + Cache | Siddanth | `charon_v.py`, `l1_elysium.py`, `iris_config.py` |
| B — LLM + Pipeline | Teammate 2 | `aria.py`, `pipeline.py` |
| C — Video RAG | Teammate 3 | `l2_asphodel.py` |
| D — NLI + Eval | Teammate 4 | `cerberus_v.py`, `eval_suite.py` |

## Branch strategy

- `main` — verified, tested code only
- `track-a/l1-elysium`
- `track-b/aria-pipeline`
- `track-c/l2-asphodel`
- `track-d/cerberus-eval`

Nothing merges to `main` without tests passing.

## Setup

```bash
pip install -r requirements.txt
```

Run the API from the same Python environment that installed the requirements, for example:

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

If you want the frontend to target a different backend host or port, set `VITE_IRIS_API_URL` before starting Vite, for example:

```bash
VITE_IRIS_API_URL=http://localhost:8000
```

## Hardware target

Ryzen 9 9800X3D · RTX 5070 12GB · 32GB DDR5  
RAM budget ceiling: 8GB for full pipeline
