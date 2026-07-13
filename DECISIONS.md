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
