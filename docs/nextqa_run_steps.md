# Running IRIS on NExT-QA / NExT-GQA Videos

This runbook explains how to download the NExT-QA-style dataset files and run the updated IRIS ingest/query pipeline on it.

Important distinction:

- The current IRIS code is wired for **NExT-QA multiple-choice rows**.
- NExT-GQA is the grounded extension, but this repo does **not yet** load or evaluate the NExT-GQA temporal grounding labels.
- The raw videos are shared conceptually, so this is the correct first step before adding grounding evaluation.

The local code path used here is:

```text
eval/nextqa_loader.py
scripts/phase6_nextqa_devset_report.py
scripts/_run_download_dev.py
scripts/phase6_build_dev_cache.py
scripts/phase6_cache_fidelity.py
scripts/phase6_ceiling.py
iris.ingest
iris.query
```

---

## 0. Start from the project root

```bash
cd /Users/apotdar/Downloads/IRIS
```

Confirm you are on the updated branch:

```bash
git branch --show-current
```

Expected:

```text
feat/novel-contributions
```

---

## 1. Install Python dependencies

Use your normal IRIS Python environment.

```bash
pip install -r requirements.txt
pip install huggingface_hub datasets pandas
```

If you want a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub datasets pandas
```

Notes:

- `datasets` is needed to fetch NExT-QA metadata from Hugging Face.
- `huggingface_hub` is needed for raw video downloads.
- First run may download large model weights for CLIP / BLIP / transformers.

---

## 2. Create dataset folders

```bash
mkdir -p eval/data/nextqa
mkdir -p eval/data/nextqa/NExTVideo_flat
mkdir -p eval/data/nextqa/index_cache
```

Target folder structure:

```text
eval/data/nextqa/
  val.csv
  dev_100.jsonl
  report_500.jsonl
  NExTVideo_flat/
    <video_id>.mp4
  index_cache/
    <video_id>.npz
```

---

## 3. Download NExT-QA metadata into `val.csv`

The repo loader expects a CSV with these columns:

```text
video, frame_count, width, height, question, answer, qid, type, a0, a1, a2, a3, a4
```

Create it from Hugging Face:

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
head -1 eval/data/nextqa/val.csv
```

Expected header:

```text
video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4
```

---

## 4. Build the frozen dev/report split

Run:

```bash
python scripts/phase6_nextqa_devset_report.py
```

Expected outputs:

```text
eval/data/nextqa/dev_100.jsonl
eval/data/nextqa/report_500.jsonl
```

Verify:

```bash
wc -l eval/data/nextqa/dev_100.jsonl
wc -l eval/data/nextqa/report_500.jsonl
```

Expected:

```text
100 eval/data/nextqa/dev_100.jsonl
500 eval/data/nextqa/report_500.jsonl
```

If the script prints missing `map_vid_vidorID.json` or `NExTVideo`, that is acceptable for the flat Hugging Face video layout. The key output is `dev_100.jsonl`.

---

## 5. Download raw dev videos

The existing helper downloads the distinct video IDs from:

```text
eval/data/nextqa/dev_100.jsonl
```

into:

```text
eval/data/nextqa/NExTVideo_flat/
```

Run:

```bash
python scripts/_run_download_dev.py
```

Watch the log:

```bash
tail -40 download_dev_output.log
```

Count downloaded videos:

```bash
find eval/data/nextqa/NExTVideo_flat -name "*.mp4" | wc -l
```

Expected files look like:

```text
eval/data/nextqa/NExTVideo_flat/3462517143.mp4
eval/data/nextqa/NExTVideo_flat/<video_id>.mp4
```

---

## 6. Build IRIS index cache for dev videos

This uses the updated persistent memory path:

```text
iris.ingest.ingest(video)
iris.ingest.save_index(index)
```

Run:

```bash
python scripts/phase6_build_dev_cache.py
```

Expected output:

```text
[  1/N] OK  <video_id>  N=<survivor_count>  t=<seconds>s
```

Indexes are saved here:

```text
eval/data/nextqa/index_cache/<video_id>.npz
```

Verify:

```bash
find eval/data/nextqa/index_cache -name "*.npz" | wc -l
```

Notes:

- This step can take time.
- It should embed survivor frames.
- It should **not** eagerly caption every survivor frame.
- Captions are lazy and generated during query only.

---

## 7. Run cache fidelity check

This verifies:

- codec sanity,
- save/load stability,
- PPR retrieval ordering after reload.

Run directly:

```bash
python scripts/phase6_cache_fidelity.py
```

Or run the wrapper:

```bash
python scripts/_run_phase6_fidelity.py
tail -100 phase6_fidelity_output.log
```

Expected final result:

```text
CACHE_FIDELITY: PASS
```

If it fails, do not tune thresholds blindly. First inspect:

```text
phase6_fidelity_output.log
```

---

## 8. Run retrieval coverage / ceiling diagnostic

This checks whether retrieval is structurally starved. It is **not** answer accuracy.

Run:

```bash
python scripts/phase6_ceiling.py
```

Or:

```bash
python scripts/_run_phase6_ceiling.py
tail -140 phase6_ceiling_output.log
```

The report groups questions by family:

```text
C = causal/common
T = temporal
D = descriptive
```

and checks:

```text
top_k = 5, 8, 12, 20
```

Use this to see whether retrieved frames cover enough of the video timeline before you trust accuracy numbers.

---

## 9. Run one actual question manually

This tests:

- loading a cached video index,
- PPR retrieval,
- lazy captioning,
- ARIA answer generation,
- Cerberus verification.

```bash
python - <<'PY'
import json
from pathlib import Path

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

DATA = Path("eval/data/nextqa")
CACHE = DATA / "index_cache"
DEV = DATA / "dev_100.jsonl"

cfg = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
)

rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]

for row in rows:
    cache_path = CACHE / row["video"]
    if Path(str(cache_path) + ".npz").exists():
        print("video:", row["video"])
        print("question:", row["question"])
        print("choices:")
        for i in range(5):
            print(f"  {i}: {row[f'a{i}']}")
        print("gold:", row["answer"], row[f"a{row['answer']}"])

        index = iris_ingest.load_index(cache_path)
        result = iris_query.query(row["question"], index, config=cfg)

        print("\nIRIS answer:")
        print(result["answer"])
        print("\nraw answer:")
        print(result["raw_answer"])
        print("\nretrieved frames:", result["retrieved_frame_idxs"])
        print("frames decoded for lazy captions:", result["frames_decoded_for_captions"])
        break
PY
```

Expected behavior:

- First query on a video may decode frames for lazy captions.
- Later queries on the same loaded index should reuse cached captions in memory.

---

## 10. Run a simple multiple-choice evaluator

The repo does not currently contain a finished NExT-QA MC accuracy runner. Use this lightweight harness for a first accuracy number.

```bash
python - <<'PY'
import json
import re
from pathlib import Path

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

DATA = Path("eval/data/nextqa")
CACHE = DATA / "index_cache"
DEV = DATA / "dev_100.jsonl"

cfg = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
)

def option_prompt(row):
    letters = ["A", "B", "C", "D", "E"]
    opts = "\n".join(f"{letters[i]}. {row[f'a{i}']}" for i in range(5))
    return (
        f"{row['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Choose the best option based only on the video evidence. "
        "Start your answer with exactly one letter: A, B, C, D, or E."
    )

def parse_letter(text):
    m = re.search(r"\b([A-E])\b", text.upper())
    return "ABCDE".index(m.group(1)) if m else None

rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]

total = 0
correct = 0
skipped = 0
fail_parse = 0

for row in rows:
    cache_path = CACHE / row["video"]
    if not Path(str(cache_path) + ".npz").exists():
        skipped += 1
        continue

    index = iris_ingest.load_index(cache_path)
    prompt = option_prompt(row)
    result = iris_query.query(prompt, index, config=cfg)

    text = result.get("raw_answer") or result.get("answer") or ""
    pred = parse_letter(text)
    gold = int(row["answer"])

    if pred is None:
        fail_parse += 1

    total += 1
    correct += int(pred == gold)

    print(f"[{total}] video={row['video']} qid={row['qid']} pred={pred} gold={gold} ok={pred == gold}")
    print("  q:", row["question"])
    print("  raw:", text[:220].replace("\n", " "))
    print()

print("=== SUMMARY ===")
print("answered:", total)
print("skipped:", skipped)
print("parse_failures:", fail_parse)
print("correct:", correct)
print("accuracy:", correct / total if total else 0.0)
PY
```

Limitations of this evaluator:

- It uses answer-text prompting, not a calibrated MC head.
- Parsing may fail if ARIA does not start with A/B/C/D/E.
- Cerberus may reject claims and return insufficient evidence even if retrieval found useful frames.
- This gives a first smoke-test number, not a paper-ready metric.

---

## 11. Run report set after dev works

After the dev run is stable, expand to:

```text
eval/data/nextqa/report_500.jsonl
```

The existing download helper reads only `dev_100.jsonl`.

To download report videos, make a copy:

```bash
cp scripts/_run_download_dev.py scripts/_run_download_report.py
```

Edit `scripts/_run_download_report.py` and change:

```python
open("eval/data/nextqa/dev_100.jsonl")
```

to:

```python
open("eval/data/nextqa/report_500.jsonl")
```

Then run:

```bash
python scripts/_run_download_report.py
python scripts/phase6_build_dev_cache.py
```

Note: `phase6_build_dev_cache.py` currently reads `dev_100.jsonl`, so for report-set cache building you should either:

1. make a report-specific copy of the script, or
2. modify `DEV_JSONL` inside it to point to `report_500.jsonl`.

Do not overwrite the dev script unless you are intentionally switching evaluation modes.

---

## 12. Optional: test the old API path

Current FastAPI still uses:

```text
api.py → iris.pipeline.run_pipeline()
```

not the new:

```text
iris.ingest → iris.query
```

So API tests are not the same as the updated persistent-memory evaluation.

If you still want to run the app:

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd iris-frontend
npm install
npm run dev
```

But for NExT-QA evaluation, prefer the scripts above.

---

## 13. Troubleshooting

### `ModuleNotFoundError: datasets`

Install:

```bash
pip install datasets
```

### `ModuleNotFoundError: huggingface_hub`

Install:

```bash
pip install huggingface_hub
```

### Video download fails

Check:

```bash
tail -80 download_dev_output.log
```

Common causes:

- network issue,
- Hugging Face rate limit,
- missing file in mirror,
- local disk space.

### `val.csv` missing

Re-run Step 3.

### `dev_100.jsonl` missing

Re-run Step 4.

### No cached indexes

Check that videos exist:

```bash
find eval/data/nextqa/NExTVideo_flat -name "*.mp4" | head
```

Then rebuild:

```bash
python scripts/phase6_build_dev_cache.py
```

### ARIA / Ollama failure

By default, `iris.aria` uses local Ollama:

```text
http://localhost:11434/v1
model: llama3.2:3b
```

If Ollama is not running, answer generation can fail.

Start Ollama separately if needed:

```bash
ollama serve
ollama pull llama3.2:3b
```

Alternatively configure `OpenAIBackend` in code, but do not mix backend changes into metric runs unless you record them.

---

## 14. What this run proves and does not prove

This run can test:

```text
codec ingest gate
persistent index build/load
PPR retrieval
lazy captioning
seek-based caption fetch
basic NExT-QA MC answer behavior
```

This run does not yet test:

```text
NExT-GQA temporal grounding labels
scene-sparse hierarchical graph
grounded visual evidence accuracy
full paper-ready benchmark
```

For NExT-GQA specifically, the next engineering step is to add a loader for the grounding annotations and compare retrieved frame timestamps against the gold temporal evidence windows.

