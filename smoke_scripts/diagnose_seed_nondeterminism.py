"""Diagnose why seed=42 + temperature=0.0 didn't fully fix determinism.
Hypothesis: CPU multi-threaded matmul reduction order in llama.cpp/Ollama's
backend is non-associative under floating point, so fixed seed + temp=0
alone isn't sufficient -- pinning num_thread (and num_ctx, to rule out
context-window effects) may also be required.
Uses the REAL context text and question from the smoke trace that failed
determinism (6936757706/qid 7), at real pipeline scale (not a toy prompt).
"""
import json
import requests

with open(r"C:\Users\swara\IRIS\smoke\per_question_trace.jsonl") as f:
    rec = None
    for line in f:
        r = json.loads(line)
        if r["video_id"] == "6936757706" and r["qid"] == "7":
            rec = r
            break

captions = rec["run_a"]["layer3"]["captions_in_answerer_order"]
context_text = "\n\n---\n\n".join(captions)
question = rec["run_a"]["question"]
print("context chars:", len(context_text))
print("question:", question)

SYSTEM = (
    "You are ARIA, a visual reasoning assistant. Answer the user's question using ONLY the "
    "provided frame evidence. Be concise and factual.\n\nProvided Frame Evidence and Retrieval Context:\n"
    + context_text
)


def call(options: dict, n: int = 3) -> list[str]:
    payload = {
        "model": "granite4:micro",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 1024,
    }
    payload.update(options)
    outs = []
    for _ in range(n):
        r = requests.post("http://localhost:11434/v1/chat/completions", json=payload, timeout=120).json()
        outs.append(r["choices"][0]["message"]["content"])
    return outs


print("\n=== default options (no thread pin) ===")
outs_default = call({})
for i, o in enumerate(outs_default):
    print(f"--- run {i} ({len(o)} chars) ---")
    print(o)
print("ALL IDENTICAL:", len(set(outs_default)) == 1)

print("\n=== options={'seed':42} nested under top-level 'options' (native-style) + num_thread=1 ===")
outs_pinned = call({"options": {"seed": 42, "temperature": 0.0, "num_thread": 1}})
for i, o in enumerate(outs_pinned):
    print(f"--- run {i} ({len(o)} chars) ---")
    print(o)
print("ALL IDENTICAL:", len(set(outs_pinned)) == 1)
