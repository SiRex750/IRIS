"""Pilot: exercises gates.py (V0, drift gate) on ONE collected clip and prints only whether the code ran and its
inputs/outputs are finite. It prints no V0 or drift value and writes nothing."""
import sys

import numpy as np

import collect
import gates

clip, root = sys.argv[1], sys.argv[2]
w = collect.windows()[clip]
T, ok, Hc, Wc = gates.load_truth(clip, root)
assert T.shape[0] == w["L"] - 1, (T.shape, w["L"])
q, f = gates.drift_false_moves(T, ok, w["k"], w["keyint"])
print(f"drift gate code ran: counts are ints, 0 <= false <= quiet: {isinstance(q, int) and 0 <= f <= q}")
v = gates.v0_clip(clip, root, w["start"], w["L"])
q75 = gates.tex_q75([v["tex_hist"]])  # placeholder threshold: this clip alone (the real one pools all 27 clips)
m = gates.v0_median(v["v0_tex"], v["v0_err"], q75)
print(f"V0 code ran: vector/q repaint mismatches {v['q_mismatch']}, NaN errors {int(np.isnan(v['v0_err']).sum())}, "
      f"median is a number: {bool(np.isfinite(m))}")
