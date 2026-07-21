import sys
sys.path.insert(0, r"C:\Users\swara\IRIS")

import iris.aria as aria
import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

cfg = IRISConfig()
cfg.answerer_backend = "llama"
cfg.answerer_endpoint = "http://localhost:11434/v1"
cfg.answerer_model = "granite4:micro"

index = iris_ingest.load_index(r"C:\Users\swara\IRIS\smoke\cache\6936757706")
question = "why did the lady held the spoonful of ice cream and looked at the girl in the middle of the video"

captured = []
real_generate = aria.generate

def spying_generate(*args, **kwargs):
    result = real_generate(*args, **kwargs)
    captured.append({"kwargs": {k: v for k, v in kwargs.items() if k != "context"}, "result": result})
    return result

aria.generate = spying_generate
try:
    res_a = iris_query.query(question, index, cfg)
    res_b = iris_query.query(question, index, cfg)
finally:
    aria.generate = real_generate

print("call 1 kwargs:", captured[0]["kwargs"])
print("call 2 kwargs:", captured[1]["kwargs"])
print("raw_answer identical:", captured[0]["result"] == captured[1]["result"])
print("backend used call1:", aria.get_backend(cfg).__class__.__name__, id(aria.get_backend(cfg)))
