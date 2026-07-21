"""Zero-confound diagnostic: call the REAL iris.aria.generate() module
function (real _SYSTEM_PROMPT, real LlamaBackend.generate(), real openai
client) three times with the exact same inputs, and print the literal kwargs
sent to chat.completions.create() by monkeypatching the client to capture
them before calling through -- to confirm seed is actually being forwarded
end-to-end from IRISConfig.answerer_seed, not silently dropped somewhere.
"""
import sys
sys.path.insert(0, r"C:\Users\swara\IRIS")

import json
import iris.aria as aria
from iris.iris_config import IRISConfig

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

cfg = IRISConfig()
cfg.answerer_backend = "llama"
cfg.answerer_endpoint = "http://localhost:11434/v1"
cfg.answerer_model = "granite4:micro"
print("answerer_seed:", cfg.answerer_seed)

# force a fresh LlamaBackend so kwargs capture isn't polluted by a cached client
aria._ACTIVE_BACKEND = None
aria._BACKEND_OVERRIDDEN = False
backend = aria.get_backend(cfg)
print("backend class:", backend.__class__.__name__, "endpoint:", backend.endpoint, "model:", backend.text_model)

real_create = backend.client.chat.completions.create
captured_kwargs = []

def spying_create(**kwargs):
    captured_kwargs.append({k: v for k, v in kwargs.items() if k != "messages"})
    return real_create(**kwargs)

backend.client.chat.completions.create = spying_create

outs = []
for i in range(3):
    out = aria.generate(prompt=question, context=context_text, model=cfg.answerer_model,
                         max_tokens=cfg.answerer_max_tokens, config=cfg)
    outs.append(out)
    print(f"--- run {i} kwargs sent: {captured_kwargs[-1]} ---")
    print(f"--- run {i} output ({len(out)} chars) ---")
    print(out)

print("\nALL IDENTICAL:", len(set(outs)) == 1)
