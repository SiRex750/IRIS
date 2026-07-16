# Architectural and Seating Decisions

## 2026-07-13: Cerberus V2 Answerer Stack & Diagnostics

### 1. Answerer Seat
**Decision:** `granite4:micro` on `llama-server` (build `b9976`) is seated as the answerer, with `cache_prompt=false` pinned. This is provisional-on-runtime.
**Metrics:** Seated based on bakeoff metrics.
*Note on Amended Latency-Gate Seating Rule:* "Models failing the latency gate cannot be seated for production query paths, regardless of ceiling metric performance."

### 2. Vacated Answerer Seat
**Decision:** `qwen3.5:4b` is vacated from the answerer seat due to failing the latency gate. It is kept as a semantic-ceiling reference only.

### 3. Captioner Seat
**Decision:** `minicpm-v4.6-inventory` is seated as the default captioner.
**Note:** The frame-3334 "devoid-of-people" label in the analyze-log was confirmed as a mislabel (people were actually present in the frame; the minicpm caption was correct).

### 4. Critical Bug Fixes
* **P0-03:** Fixed silent structural corruption of `AnswerClaims` by switching `cerberus_mode="v2"` to use the schema-constrained wire generator instead of the legacy nested JSON parser. (Commit: `b88f9b4`)
* **P0-07:** Fixed a layer-3 absence path bug where an empty caption list resulted in a zero-evidence "verified_absent" verdict. Added a guard to return "unverifiable" instead. (Commit: `0ee8775`)

### 5. Terminology and Claim-Wording Corrections
* Use **"selective processing"** instead of "selective decode".
* Use **"consistent with recorded evidence"** instead of "visually verified".

### 6. Artifact-Commit Discipline (P0-11)
**Decision:** Strict artifact-commit discipline must be maintained. All evaluation outputs, diagnostic capture logs, and bakeoff results must be committed alongside the code changes that produced them to maintain unbroken provenance.

## 2026-07-17: Metric Integrity & Seat-Violation Findings

### 1. Codec-at-Query-Time Negative (paper-grade)
**Decision:** Codec structures ingest; it does NOT belong in query-time ranking.
**Evidence:** Deconfounded ablation + T0 rerun replication (89 videos / 255 events, CPU),
monotonic in λ — codec-only 0.4941 < 50/50 hybrid 0.5451 < semantic-only 0.5882 (Hit@5).
**Consequence:** The production `ppr_lambda=0.5` is a KNOWN-WRONG operating point, pending the
λ sweep. Pre-registered prediction: λ*=0.

### 2. Cache-Determinism (paper-grade serving finding)
**Decision:** temp-0 output is a deterministic function of prompt-cache state (fresh prefill vs
cached), not model instability. Eliminated by pinning `cache_prompt=false` per request.
**Consequence:** The team's earlier rejection of granite4:micro as "unstable" was this artifact.
Runtimes without per-request cache control (Ollama) cannot serve the answerer seat.

### 3. Span-Construction Bug (metric corruption)
**Decision:** The predicted grounding span is built as min(ts)→max(ts) over the top-K retrieved
frames — the enclose-all-top-K anti-pattern. IoP divides by prediction length, so scattered
top-K frames yield a near-video-length span and IoP collapses even when retrieval is correct.
**Sites:** `scripts/pillar2_grounded_qa.py::iop()` and `::iou()`; `scripts/tune_l1_weights.py::iop()`
(that file has no iou()). Three independent copies of the same construction.
**Consequence:** ALL grounding metrics reported to date (mIoP 0.3091, IoP@0.5 25.00%, mIoU 0.2479,
IoU@0.5 17.86%) are corrupted, and every Optuna study to date optimized a corrupted objective.
`optuna_best_weights.json` is PROVISIONAL.
**Fix (not yet applied):** span-from-PPR-peak — top-scoring frame t* → tightest contiguous window
over the top-scoring cluster → single half-width w, tuned on val only, then frozen. This is a bug
fix, not tuning. One shared constructor imported by all call sites; no per-script copies.

### 4. Seat Violation in the V2 Calibration Run
**Decision:** The V2 calibration run (N=56) did NOT run on the seated answerer configuration.
`scripts/pillar2_grounded_qa.py` constructs `aria.LlamaBackend(endpoint="http://127.0.0.1:11434/v1")`
— Ollama, the rejected runtime — with no `cache_prompt` control, at temp=0.1 (hardcoded in
`iris/aria.py`; six sites across both backends).
**Consequence:** All V2 numbers are QUARANTINED as **UNPINNED**. This includes
P(correct|grounded)=35.7% and therefore the "the answerer is the entire Acc@GQA deficit" diagnosis,
which stands as a HYPOTHESIS, not a measurement. The +3.57pp Acc@QA "improvement" is confounded by
prompt-cache state per finding (2) and is not attributable to `_QA_MCQ_PROMPT`.
**Bar for lifting the quarantine:** rerun on llama-server (b9976) per the seat contract —
`cache_prompt=false`, `--parallel 1`, fixed threads, temp=0 — with the span fix applied.
**Note:** `LlamaServerBackend` in `iris/aria.py` DOES correctly pin `cache_prompt=false` on all
three of its paths. The seat contract exists in code; the benchmark bypassed it by constructing
the wrong backend class.
