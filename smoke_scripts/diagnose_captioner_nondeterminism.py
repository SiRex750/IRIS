"""Isolate whether MoondreamCaptioner itself is non-deterministic (no fixed
seed/greedy-decoding guarantee), which would explain why two independently
loaded indexes (each captioning fresh) produced different answerer prompts
even after the answerer's seed was pinned."""
import sys
sys.path.insert(0, r"C:\Users\swara\IRIS")

import av
from iris.aria import MoondreamCaptioner

VIDEO = r"C:\Users\swara\IRIS\eval\data\nextqa\NExTVideo_flat\6936757706.mp4"

container = av.open(VIDEO)
stream = container.streams.video[0]
frame = next(container.decode(stream))
pil_img = frame.to_image()
container.close()

cap = MoondreamCaptioner()
outs = [cap.caption(pil_img) for _ in range(4)]
for i, o in enumerate(outs):
    print(f"run {i}: {o!r}")
print("ALL IDENTICAL:", len(set(outs)) == 1)
