# Test IRIS From Scratch on One NExT-GQA / NExT-QA Video

This is a clean other-laptop runbook for testing the updated IRIS pipeline on **one video only**.

Important naming note:

- The code currently uses folder/script names like `nextqa`.
- Your research target may be called **NExT-GQA**, but the current evaluator is wired for **NExT-QA-style multiple-choice rows**:

```text
video, question, answer, qid, type, a0, a1, a2, a3, a4
```

So in commands below, keep using:

```text
eval/data/nextqa/
```

even if you are thinking of the broader NExT-GQA task.

---

## 0. What this test will verify

This run checks whether the updated IRIS code can:

1. Load one NExT-QA/NExT-GQA video.
2. Build one persistent IRIS index cache.
3. Build the updated sparse/hierarchical L2 graph.
4. Use PPR retrieval.
5. Use question-only retrieval, not answer-option retrieval.
6. Use deterministic query reformulation.
7. Use temporal neighbor expansion for causal/temporal questions.
8. Ask the current ARIA model to answer MC questions.
9. Write a per-question JSONL result log.

It does **not** require running the whole dataset.

---

## 1. Start with a clean repo

Clone or copy the updated IRIS repo onto the other laptop.

```bash
cd path/to/IRIS
```

Confirm these files exist:

```bash
ls iris/query_reformulation.py
ls scripts/nextqa_single_video_eval.py
ls docs/nextgqa_single_video_from_scratch.md
```

If any of these are missing, the laptop does not have the latest code changes.

---

## 2. Create and activate Python environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface_hub datasets pandas
```

If PyAV fails to install, install FFmpeg system packages first, then retry.

---

## 3. Confirm Ollama model is available

This keeps the same model as the current project setup.

```bash
ollama pull llama3.2:3b
```

In another terminal, make sure Ollama is running:

```bash
ollama serve
```

If `ollama serve` says it is already running, that is fine.

---

## 4. Create dataset folders

From the IRIS repo root:

```bash
mkdir -p eval/data/nextqa
mkdir -p eval/data/nextqa/NExTVideo_flat
mkdir -p eval/data/nextqa/index_cache
mkdir -p eval/results
```

Expected layout:

```text
eval/data/nextqa/
  val.csv
  dev_100.jsonl
  NExTVideo_flat/
    <video_id>.mp4
  index_cache/
    <video_id>.npz
eval/results/
```

---

## 5. Download or place metadata

If you already have `val.csv`, put it here:

```text
eval/data/nextqa/val.csv
```

It must contain:

```text
video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
```

If you do not have it, create it from Hugging Face:

```bash
python - <<'PY'
from datasets import load_dataset
from pathlib import Path

out = Path("eval/data/nextqa")
out.mkdir(parents=True, exist_ok=True)

ds = load_dataset("lmms-lab/NExTQA", "MC", split="test")
df = ds.to_pandas()

cols = [
    "video", "frame_count", "width", "height", "question",
    "answer", "qid", "type", "a0", "a1", "a2", "a3", "a4",
]

df[cols].to_csv(out / "val.csv", index=False)
print("wrote", out / "val.csv", "rows=", len(df))
PY
```

Verify:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("eval/data/nextqa/val.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(2)[["video", "qid", "type", "question", "answer"]])
PY
```

---

## 6. Create a one-video split

Instead of running all 100 dev questions, make a split containing questions for one video.

First list videos with the most questions:

```bash
python - <<'PY'
import pandas as pd
from collections import Counter

df = pd.read_csv("eval/data/nextqa/val.csv")
counts = Counter(df["video"].astype(str))

print("video_id question_count")
for vid, n in counts.most_common(20):
    print(vid, n)
PY
```

Pick one `VIDEO_ID` from the output.

Example:

```text
4430422083
```

Now create `one_video.jsonl`.

Replace `4430422083` with your chosen video ID:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

VIDEO_ID = "4430422083"  # change this

DATA = Path("eval/data/nextqa")
df = pd.read_csv(DATA / "val.csv")
df["video"] = df["video"].astype(str)
rows = df[df["video"] == VIDEO_ID].copy()

if rows.empty:
    raise SystemExit(f"no rows found for video {VIDEO_ID}")

def family(type_code: str) -> str:
    first = str(type_code)[0].upper() if str(type_code) else "?"
    return first if first in {"C", "T", "D"} else "?"

out = DATA / "one_video.jsonl"
keep = ["qid", "video", "question", "a0", "a1", "a2", "a3", "a4", "answer", "type"]

with open(out, "w", encoding="utf-8") as f:
    for _, r in rows.iterrows():
        item = {k: r[k].item() if hasattr(r[k], "item") else r[k] for k in keep}
        item["video"] = str(item["video"])
        item["answer"] = int(item["answer"])
        item["qid"] = int(item["qid"])
        item["family"] = family(item["type"])
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("wrote", out)
print("video", VIDEO_ID)
print("questions", len(rows))
PY
```

Check:

```bash
head -2 eval/data/nextqa/one_video.jsonl
wc -l eval/data/nextqa/one_video.jsonl
```

---

## 7. Put the video file in place

The evaluator expects:

```text
eval/data/nextqa/NExTVideo_flat/<VIDEO_ID>.mp4
```

Example:

```text
eval/data/nextqa/NExTVideo_flat/4430422083.mp4
```

If you already downloaded the dataset, copy only the chosen video there.

If you need to download only that one video from Hugging Face, run:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

VIDEO_ID = "4430422083"  # change this

dest = Path("eval/data/nextqa/NExTVideo_flat")
dest.mkdir(parents=True, exist_ok=True)

p = hf_hub_download(
    repo_id="VLM2Vec/nextqa-rawvideo",
    repo_type="dataset",
    filename=f"{VIDEO_ID}.mp4",
    local_dir=str(dest),
)

print("downloaded:", p)
PY
```

Verify:

```bash
ls -lh eval/data/nextqa/NExTVideo_flat/4430422083.mp4
```

Change the filename if you picked a different video ID.

---

## 8. Build the IRIS index cache for only this video

Replace `4430422083` with your chosen video ID:

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
    graph_edge_mode="hierarchical_sparse",
)

if not VIDEO_PATH.exists():
    raise SystemExit(f"missing video: {VIDEO_PATH}")

index = iris_ingest.ingest(str(VIDEO_PATH), config=cfg)
iris_ingest.save_index(index, CACHE_PATH)

print("saved", str(CACHE_PATH) + ".npz")
print("indexed survivor frames:", len(index.frames))
print("graph nodes:", index._graph.graph.number_of_nodes())
print("graph edges:", index._graph.graph.number_of_edges())
print("video path:", index.video_path)
PY
```

Expected output:

```text
saved eval/data/nextqa/index_cache/<VIDEO_ID>.npz
indexed survivor frames: ...
graph nodes: ...
graph edges: ...
```

If `graph edges` is close to `N*(N-1)/2`, the code is still using fully connected graph mode.

If `graph edges` is much closer to `N*k`, the sparse hierarchy is active.

---

## 9. List cached videos available for evaluation

```bash
python scripts/nextqa_single_video_eval.py \
  --split eval/data/nextqa/one_video.jsonl \
  --cache-dir eval/data/nextqa/index_cache
```

Expected:

```text
Cached videos present in split:
video_id question_count
4430422083 ...
```

---

## 10. Run baseline evaluator on the one video

This disables the new query reformulation and temporal expansion.

Use this to get a clean baseline.

```bash
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --split eval/data/nextqa/one_video.jsonl \
  --cache-dir eval/data/nextqa/index_cache \
  --output-jsonl eval/results/nextgqa_4430422083_baseline.jsonl \
  --no-reformulation \
  --no-temporal-expansion
```

Expected summary:

```text
SUMMARY
video: 4430422083
total: ...
correct: ...
accuracy: ...
parse_fail: ...
decoded_for_captions_total: ...
```

---

## 11. Run query reformulation only

```bash
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --split eval/data/nextqa/one_video.jsonl \
  --cache-dir eval/data/nextqa/index_cache \
  --output-jsonl eval/results/nextgqa_4430422083_reformulation.jsonl \
  --no-temporal-expansion
```

Compare with baseline:

```text
baseline accuracy
vs
reformulation accuracy
```

Also check retrieved frame spread:

```bash
python - <<'PY'
import json
from pathlib import Path

for p in [
    Path("eval/results/nextgqa_4430422083_baseline.jsonl"),
    Path("eval/results/nextgqa_4430422083_reformulation.jsonl"),
]:
    if not p.exists():
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    acc = sum(r["correct"] for r in rows) / len(rows)
    parse_fail = sum(not r["parse_ok"] for r in rows)
    avg_frames = sum(len(r["retrieved_frame_idxs"]) for r in rows) / len(rows)
    print(p.name)
    print("  accuracy:", round(acc, 4))
    print("  parse_fail:", parse_fail)
    print("  avg_context_frames:", round(avg_frames, 2))
PY
```

---

## 12. Run full fixed evaluator

This enables:

```text
question-only retrieval
strict MC parser
query reformulation
PPR retrieval
temporal neighbor expansion
```

```bash
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --split eval/data/nextqa/one_video.jsonl \
  --cache-dir eval/data/nextqa/index_cache \
  --output-jsonl eval/results/nextgqa_4430422083_full_fixed.jsonl \
  --top-k 8 \
  --temporal-radius 2 \
  --max-context-frames 24
```

Expected:

```text
query_reformulation=True
temporal_expansion=True radius=2
```

For `C` and `T` family questions, `retrieved_frame_idxs` may contain more than 8 frames because temporal neighbors are added.

---

## 13. Compare all three runs

```bash
python - <<'PY'
import json
from pathlib import Path

files = [
    "eval/results/nextgqa_4430422083_baseline.jsonl",
    "eval/results/nextgqa_4430422083_reformulation.jsonl",
    "eval/results/nextgqa_4430422083_full_fixed.jsonl",
]

for fp in files:
    p = Path(fp)
    if not p.exists():
        print("missing", fp)
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    total = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    parse_fail = sum(not bool(r["parse_ok"]) for r in rows)
    decoded = sum(int(r["decoded_for_captions"]) for r in rows)
    avg_ctx = sum(len(r["retrieved_frame_idxs"]) for r in rows) / total

    print(p.name)
    print("  total:", total)
    print("  correct:", correct)
    print("  accuracy:", round(correct / total, 4) if total else 0.0)
    print("  parse_fail:", parse_fail)
    print("  decoded_for_captions:", decoded)
    print("  avg_context_frames:", round(avg_ctx, 2))
PY
```

Interpretation:

- `parse_fail` should ideally be `0`.
- If baseline predicts `A` too often, the strict parser fixed the old false-A parser issue.
- If reformulation improves accuracy or retrieves better frames, keep it.
- If full fixed version improves `C` and `T` questions, temporal expansion is helping.
- If full fixed version gets slower, reduce `--max-context-frames` or `--temporal-radius`.

---

## 14. Inspect wrong answers

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("eval/results/nextgqa_4430422083_full_fixed.jsonl")
rows = [json.loads(l) for l in open(p, encoding="utf-8")]

for r in rows:
    if r["correct"]:
        continue
    print("=" * 100)
    print("qid:", r["qid"], "type:", r["type"], "family:", r["family"])
    print("question:", r["question"])
    print("gold:", r["gold_label"], "pred:", r["pred_label"], "parse_ok:", r["parse_ok"])
    print("retrieval_queries:", r["retrieval_queries"])
    print("retrieved_frame_idxs:", r["retrieved_frame_idxs"])
    print("raw_answer:", r["raw_answer"])
PY
```

Use this to identify whether the failure is:

```text
retrieval failure
caption failure
reasoning/model failure
parser failure
```

---

## 15. Optional: compare PPR vs legacy retrieval

Run the same full evaluator but with legacy ranking:

```bash
python scripts/nextqa_single_video_eval.py \
  --video-id 4430422083 \
  --split eval/data/nextqa/one_video.jsonl \
  --cache-dir eval/data/nextqa/index_cache \
  --output-jsonl eval/results/nextgqa_4430422083_legacy.jsonl \
  --ranking-mode legacy
```

Compare:

```text
full_fixed PPR
vs
legacy
```

PPR should be the main graph-aware path.

---

## 16. Common errors and fixes

### Error: missing cache

```text
Missing cache for video ...
```

Fix:

Run Section 8 for the same `VIDEO_ID`.

### Error: no rows for video

```text
No rows for video ...
```

Fix:

Your `one_video.jsonl` does not contain that video ID. Recreate Section 6.

### Error: video file missing

```text
missing video: eval/data/nextqa/NExTVideo_flat/<VIDEO_ID>.mp4
```

Fix:

Copy or download the `.mp4` into `NExTVideo_flat`.

### Error: Ollama connection failed

Fix:

Start Ollama:

```bash
ollama serve
```

Then verify:

```bash
ollama list
```

### Error: PyAV / `av` import fails

Fix:

Install FFmpeg, then reinstall PyAV:

```bash
pip install av
```

### Accuracy is low

Do not immediately change the model. First inspect:

```text
parse_fail
retrieved_frame_idxs
retrieval_queries
raw_answer
decoded_for_captions
```

If retrieved frames are wrong, fix retrieval/graph/query reformulation.

If retrieved frames look correct but the answer is wrong, then model quality is the issue.

---

## 17. What files to send back after testing

Send these files/logs for debugging:

```text
eval/results/nextgqa_<VIDEO_ID>_baseline.jsonl
eval/results/nextgqa_<VIDEO_ID>_reformulation.jsonl
eval/results/nextgqa_<VIDEO_ID>_full_fixed.jsonl
```

Also send terminal summary:

```text
total
correct
accuracy
parse_fail
decoded_for_captions_total
wall_sec
```

That is enough to tell whether the issue is retrieval, captioning, parsing, or the answer model.

