import sys
sys.path.insert(0, r"C:\Users\swara\IRIS")

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

cfg = IRISConfig()
cfg.answerer_backend = "llama"
cfg.answerer_endpoint = "http://localhost:11434/v1"
cfg.answerer_model = "granite4:micro"

index = iris_ingest.load_index(r"C:\Users\swara\IRIS\smoke\cache\6936757706")
question = "why did the lady held the spoonful of ice cream and looked at the girl in the middle of the video"

res_a = iris_query.query(question, index, cfg)
res_b = iris_query.query(question, index, cfg)

print("context_text identical:", res_a["context_text"] == res_b["context_text"])
print("retrieved_frame_idxs identical:", res_a["retrieved_frame_idxs"] == res_b["retrieved_frame_idxs"])
print("raw_answer identical:", res_a["raw_answer"] == res_b["raw_answer"])

if res_a["context_text"] != res_b["context_text"]:
    ca, cb = res_a["context_text"], res_b["context_text"]
    print("len a:", len(ca), "len b:", len(cb))
    for i, (x, y) in enumerate(zip(ca, cb)):
        if x != y:
            print("first diff at char", i, repr(ca[max(0,i-40):i+40]), "VS", repr(cb[max(0,i-40):i+40]))
            break
else:
    print("--- context_text (shared) ---")
    print(res_a["context_text"])
