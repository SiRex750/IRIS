# NExT-QA Single-Video Evaluation and Accuracy Fix Plan

This document explains:

1. How to run NExT-QA testing on questions from **one video only**.
2. What is likely going wrong in the current multiple-choice evaluator.
3. How to fix the evaluation flow while keeping the current answer model:

```text
llama3.2:3b
```

4. How to add query reformulation as part of the retrieval fix.

---

## 1. Why single-video testing is useful

Running all dev questions makes debugging slow and noisy.

For IRIS, single-video testing is better because you can inspect:

```text
same video
same cached index
multiple questions
same retrieved frame pool
same caption cache behavior
```

This helps isolate whether the problem is:

```text
retrieval
captioning
answer generation
MC option parsing
Cerberus verification
```

---

## 2. Pick one video that has cached index and multiple questions

Run this from the project root:

```bash
cd /Users/apotdar/Downloads/IRIS
```

List cached videos and how many dev questions each one has:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

DATA = Path("eval/data/nextqa")
CACHE = DATA / "index_cache"
DEV = DATA / "dev_100.jsonl"

rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
cached = {p.stem for p in CACHE.glob("*.npz")}

counts = Counter(r["video"] for r in rows if r["video"] in cached)

print("video_id question_count")
for vid, n in counts.most_common():
    print(vid, n)
PY
```

Pick a video ID from the output.

Example:

```text
4430422083
```

In later commands, replace:

```text
VIDEO_ID = "4430422083"
```

with your chosen video ID.

---

## 3. If the video is downloaded but not cached, build cache for only that video

Use this if:

```text
eval/data/nextqa/NExTVideo_flat/<video_id>.mp4 exists
```

but:

```text
eval/data/nextqa/index_cache/<video_id>.npz does not exist
```

Build just one video:

```bash
python - <<'PY'
from pathlib import Path

import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig

VIDEO_ID = "4430422083"  # change this

DATA = Path("eval/data/nextqa")
VIDEO_PATH = DATA / "NExTVideo_flat" / f"{VIDEO_ID}.mp4"
CACHE_PATH = DATA / "index_cache" / VIDEO_ID

cfg = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
)

if not VIDEO_PATH.exists():
    raise SystemExit(f"missing video: {VIDEO_PATH}")

index = iris_ingest.ingest(str(VIDEO_PATH), config=cfg)
iris_ingest.save_index(index, CACHE_PATH)

print("saved", str(CACHE_PATH) + ".npz")
print("survivor frames:", len(index.frames))
print("total frames:", index.stats.get("total"))
PY
```

---

## 4. If the video is missing, download only that video

Use Hugging Face raw-video mirror:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

VIDEO_ID = "4430422083"  # change this

DEST = Path("eval/data/nextqa/NExTVideo_flat")
DEST.mkdir(parents=True, exist_ok=True)

p = hf_hub_download(
    repo_id="VLM2Vec/nextqa-rawvideo",
    repo_type="dataset",
    filename=f"{VIDEO_ID}.mp4",
    local_dir=str(DEST),
)

print("downloaded:", p)
PY
```

Then run the one-video cache build from Section 3.

---

## 5. Current likely bug: option parser predicts `A` too often

If the evaluator uses a parser like:

```python
re.search(r"\b([A-E])\b", text.upper())
```

then it can incorrectly parse the English article `"A"` as option `A`.

Example:

```text
A boy is stretching his arm because...
```

The parser sees the first word:

```text
A
```

and records:

```text
pred = A
```

This makes accuracy look much worse and biases predictions toward `A`.

### Fix

Force the model to return a strict answer marker:

```text
ANSWER: A
```

Then parse only:

```python
re.search(r"ANSWER:\s*([A-E])\b", text.upper())
```

---

## 6. Current likely design bug: retrieval query includes answer options

Do **not** use the full multiple-choice prompt for frame retrieval.

Bad retrieval query:

```text
Question:
why did the boy stretch his arm?

Options:
A. ...
B. ...
C. ...
D. ...
E. ...
```

This pollutes CLIP/PPR retrieval because the retrieval embedding now includes every answer option.

Correct separation:

```text
retrieval query = question only
answer prompt   = question + options + retrieved frame evidence
```

So:

```python
query_embedding = _embed_query(row["question"], cfg)
retrieved_frames = _build_retrieved(index, query_embedding, cfg)
```

Then use options only when asking ARIA to select A/B/C/D/E.

---

## 7. Single-video evaluator with fixed retrieval and fixed parser

Implemented script:

```bash
python scripts/nextqa_single_video_eval.py
```

Without `--video-id`, it lists cached videos found in the split:

```bash
python scripts/nextqa_single_video_eval.py \
  --split eval/data/nextqa/dev_100.jsonl \
  --cache-dir eval/data/nextqa/index_cache
```

Run one cached video:

```bash
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --split eval/data/nextqa/dev_100.jsonl \
  --cache-dir eval/data/nextqa/index_cache \
  --output-jsonl eval/results/nextqa_4430422083_fixed.jsonl
```

Useful ablations:

```bash
# strict parser + question-only retrieval, but no reformulation/temporal expansion
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --no-reformulation \
  --no-temporal-expansion

# add reformulation only
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --no-temporal-expansion

# full fixed path: reformulation + C/T temporal expansion
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083
```

The helper logic lives in:

```text
iris/query_reformulation.py
```

This evaluator:

- keeps the current model,
- retrieves using the question only,
- captions retrieved frames lazily,
- asks MC answer using retrieved context,
- parses only `ANSWER: <letter>`,
- reports per-question accuracy for one video.

Run:

```bash
python - <<'PY'
import json
import re
from pathlib import Path

import iris.aria as aria
import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

VIDEO_ID = "4430422083"  # change this

DATA = Path("eval/data/nextqa")
DEV = DATA / "dev_100.jsonl"
CACHE = DATA / "index_cache"

cfg = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
)

def mc_prompt(row: dict) -> str:
    return f"""You are answering a NExT-QA multiple-choice video question.

Use only the provided retrieved frame evidence.

Question:
{row["question"]}

Options:
A. {row["a0"]}
B. {row["a1"]}
C. {row["a2"]}
D. {row["a3"]}
E. {row["a4"]}

Return exactly this format:
ANSWER: <A|B|C|D|E>
REASON: <one short sentence grounded in the frame evidence>
"""

def parse_answer(text: str):
    m = re.search(r"ANSWER:\s*([A-E])\b", text.upper())
    if not m:
        return None
    return "ABCDE".index(m.group(1))

rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
rows = [r for r in rows if r["video"] == VIDEO_ID]

if not rows:
    raise SystemExit(f"no dev rows for video {VIDEO_ID}")

cache_path = CACHE / VIDEO_ID
npz = Path(str(cache_path) + ".npz")
if not npz.exists():
    raise SystemExit(f"missing cache: {npz}")

index = iris_ingest.load_index(cache_path)

total = 0
correct = 0
parse_fail = 0

print(f"VIDEO_ID={VIDEO_ID}")
print(f"questions={len(rows)}")
print(f"cached frames={len(index.frames)}")
print()

for row in rows:
    # Retrieval uses question only.
    query_embedding = iris_query._embed_query(row["question"], cfg)
    retrieved_frames = iris_query._build_retrieved(index, query_embedding, cfg)

    # Lazy caption only retrieved frames.
    decoded_for_captions = iris_query._ensure_captions(index, retrieved_frames)

    # Populate L1 context from retrieved/captioned frames.
    cache_obj = iris_query.wrapper_init_l1_cache(cfg)
    iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
    context_text = cache_obj.as_context_text()

    # Answer prompt includes options, but retrieval did not.
    prompt = mc_prompt(row)
    raw_answer = aria.generate(prompt=prompt, context=context_text)
    pred = parse_answer(raw_answer)
    gold = int(row["answer"])

    total += 1
    correct += int(pred == gold)
    parse_fail += int(pred is None)

    print(f"qid={row['qid']} type={row['type']} family={row['family']}")
    print("question:", row["question"])
    print("gold:", gold, "pred:", pred, "ok:", pred == gold)
    print("retrieved:", [f["frame_idx"] for f in retrieved_frames])
    print("decoded_for_captions:", decoded_for_captions)
    print("raw:", raw_answer.replace("\\n", " ")[:300])
    print("-" * 80)

print("SUMMARY")
print("total:", total)
print("correct:", correct)
print("accuracy:", correct / total if total else 0.0)
print("parse_fail:", parse_fail)
PY
```

This is the first evaluator you should trust more than the previous background scratch run.

---

## 8. Keep the same model for now

Do not change the model while debugging the evaluator.

Current default in `iris/aria.py`:

```text
llama3.2:3b
```

Keep it fixed until these are corrected:

```text
1. strict MC output parsing
2. question-only retrieval
3. temporal neighbor/context expansion
4. query reformulation
5. raw-answer vs verified-answer comparison
```

Otherwise, model changes and pipeline changes get mixed together and the metric becomes hard to interpret.

---

## 9. Add query reformulation as a retrieval-only fix

Query reformulation should improve retrieval, not answer generation.

The goal is to rewrite:

```text
why did the boy stretch his arm in the middle of the video
```

into retrieval-friendly visual search strings:

```text
boy stretching arm
child reaches arm outward
boy extends hand toward something
middle of video arm movement
```

Important rule:

```text
Do not let reformulation invent the answer.
```

Good:

```text
person carrying object
vehicle stopping near entrance
child stretching arm
```

Bad:

```text
the boy stretched his arm to catch a ball
```

The bad version invents cause before evidence is retrieved.

---

## 10. Deterministic query reformulation baseline

Start with a simple, non-LLM reformulator. This keeps behavior stable and avoids adding another model variable.

Add this logic inside the single-video evaluator before embedding:

```python
def reformulate_query(question: str, family: str | None = None) -> list[str]:
    q = question.strip()
    q_lower = q.lower()

    queries = [q]

    # Convert question phrasing into visual-description phrasing.
    replacements = [
        ("why did ", ""),
        ("what did ", ""),
        ("what does ", ""),
        ("what is ", ""),
        ("what are ", ""),
        ("how did ", ""),
        ("where did ", ""),
        ("when did ", ""),
        ("in the video", ""),
        ("middle of the video", ""),
        ("at the end of the video", ""),
        ("at the beginning of the video", ""),
    ]

    visual = q_lower
    for src, dst in replacements:
        visual = visual.replace(src, dst)
    visual = " ".join(visual.split())

    if visual and visual != q_lower:
        queries.append(visual)

    # Add temporal intent hints without answering.
    if "after" in q_lower:
        queries.append(q_lower.replace("after", "").strip())
        queries.append("event after action")
    if "before" in q_lower:
        queries.append(q_lower.replace("before", "").strip())
        queries.append("event before action")
    if "why" in q_lower:
        queries.append("cause of action")
        queries.append("person action reason")

    # Deduplicate while preserving order.
    out = []
    seen = set()
    for item in queries:
        item = item.strip(" ?.")
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out[:5]
```

Then instead of one embedding:

```python
query_embedding = iris_query._embed_query(row["question"], cfg)
retrieved_frames = iris_query._build_retrieved(index, query_embedding, cfg)
```

use multiple reformulated queries:

```python
query_texts = reformulate_query(row["question"], row.get("family"))

merged = {}
for qt in query_texts:
    emb = iris_query._embed_query(qt, cfg)
    frames = iris_query._build_retrieved(index, emb, cfg)
    for rank, f in enumerate(frames):
        fi = f["frame_idx"]
        score = 1.0 / (rank + 1)
        if fi not in merged or score > merged[fi][0]:
            merged[fi] = (score, f)

retrieved_frames = [
    f for _, f in sorted(merged.values(), key=lambda x: x[0], reverse=True)
][: cfg.l2_retrieve_top_k]
```

This is a simple reciprocal-rank fusion strategy.

---

## 11. Query reformulation with temporal neighbor expansion

For temporal and causal questions, retrieved isolated frames are often insufficient.

Families:

```text
C = causal/common
T = temporal
D = descriptive
```

For `C` and `T`, expand around retrieved frames.

Concept:

```text
retrieved frame = 219
also include nearby indexed frames before and after 219
```

Add:

```python
def expand_temporal_neighbors(index, retrieved_frames, radius: int = 2):
    by_idx = {fr.frame_idx: fr for fr in index.frames}
    ordered = sorted(fr.frame_idx for fr in index.frames)
    pos = {fi: i for i, fi in enumerate(ordered)}

    selected = {}
    for f in retrieved_frames:
        fi = f["frame_idx"]
        if fi not in pos:
            continue
        p = pos[fi]
        for j in range(max(0, p - radius), min(len(ordered), p + radius + 1)):
            nfi = ordered[j]
            fr = by_idx[nfi]
            selected[nfi] = {
                "frame_idx": fr.frame_idx,
                "timestamp": fr.timestamp,
                "luma_diff_energy": fr.luma_diff_energy,
                "action_score": fr.action_score,
                "persistence_value": fr.persistence_value,
                "is_peak": fr.is_peak,
                "clip_embedding": fr.clip_embedding,
                "luma_entropy": fr.luma_entropy,
                "caption": fr.caption,
                "pagerank_score": fr.pagerank_score,
                "last_retrieval_score": 0.0,
                "retrieval_contributions": {"temporal_expansion": True},
            }

    return [selected[k] for k in sorted(selected)]
```

Use it only for temporal/causal families:

```python
if row.get("family") in {"C", "T"}:
    retrieved_frames = expand_temporal_neighbors(index, retrieved_frames, radius=2)
```

Then call:

```python
iris_query._ensure_captions(index, retrieved_frames)
```

This gives ARIA before/after context without changing the base model.

---

## 12. Recommended fixed evaluation order

Run these in order.

### A. Baseline single-video fixed parser

Use Section 7 exactly.

Record:

```text
accuracy
parse_fail
retrieved frames
decoded_for_captions
raw answers
```

### B. Add question-only retrieval if not already present

Section 7 already does this.

Expected effect:

```text
retrieval becomes less polluted by answer choices
```

### C. Add query reformulation

Use Sections 10 and 11.

Expected effect:

```text
retrieval should improve for causal/temporal questions
```

### D. Compare raw answer vs Cerberus verified answer

For MC accuracy, first score the raw answer. Cerberus can reject answers because captions are incomplete.

Recommended reporting:

```text
raw_mc_accuracy
verified_mc_accuracy
parse_fail_rate
retrieval_frame_overlap
caption_decode_cost
```

### E. Only after this, try a stronger model

Do not change the model until the above is stable.

---

## 13. What to log per question

For each question, log:

```text
video
qid
type
family
question
gold option
pred option
raw answer
retrieved frame indices
retrieval query texts
decoded frames for lazy captions
caption snippets
```

This is the minimum needed to debug wrong predictions.

---

## 14. Summary of fixes while keeping `llama3.2:3b`

Required fixes:

```text
1. Run one video at a time for debugging.
2. Parse only ANSWER: A/B/C/D/E, not any A-E letter.
3. Retrieve using question only, not the full MC prompt.
4. Use options only in the final answer prompt.
5. Add deterministic query reformulation for retrieval.
6. Add temporal neighbor expansion for C/T questions.
7. Score raw MC answer separately from Cerberus-verified answer.
8. Log retrieved frames and captions for every miss.
```

Expected result:

```text
The metric becomes trustworthy.
The A-bias should drop.
Retrieval should become more relevant.
Temporal/causal questions should get better context.
```

Only after this should you compare:

```text
llama3.2:3b vs qwen3:8b vs qwen3:14b
```
