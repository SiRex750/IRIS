# Latency A/B: flat vs scene_sparse — RESULT (VAL 406 Qs / 59 videos, CPU, 8 threads, idle machine)

POOLED: flat 0.001959s vs scene_sparse 0.001909s median per query — a 2.5% tie.
Descend-only 0.001955s (even with flat). Shortcut 0.000331s but fires on only 5.9% of queries.

MECHANISM CONFIRMED: subgraph is 38/120 nodes (32%) and 67/395 edges (17%); PPR time drops
0.001912 -> 0.001010 (-47%). The predicted PPR saving is REAL.

OVERHEAD CANCELS IT: subgraph induction 0.000413 + cross_scene_edges 0.000134 + centroid ranking
0.000129 = 0.000676 against a 0.000902 PPR saving. Subgraph induction (.subgraph().copy(), Python
dict copying that scales with the SOURCE graph) is the dominant overhead — larger than the other
two combined.

SCALING — THE RESULT THAT MATTERS: median latency by frame count.
  <100 frames  (n=167): flat 0.001497 | scene 0.001653  -> scene 10% SLOWER
  100-299      (n=173): flat 0.002464 | scene 0.002032  -> scene 18% faster
  300-599      (n=66):  flat 0.004260 | scene 0.002500  -> scene 41% faster
Crossover ~100 frames. Across the range flat grows 2.85x, scene_sparse 1.51x — flat scales
roughly twice as steeply with video length.

LIMITATION: longest videos are 599 frames; top bucket n=66. CCTV footage is ~100x longer. The
trend is directionally clear and mechanistically explained but NOT verified at CCTV scale. State
as a hypothesis with supporting evidence, not a claim.

PERSPECTIVE: query embedding is 0.016s (8x all of retrieval) and the answerer is ~60s, so
retrieval latency is ~0.003% of end-to-end cost. This matters ONLY for the long-video scaling
argument, never for throughput.

CONTROL: query embedding time matched across arms (0.016046 vs 0.016091) as predicted.
Index load (excluded from per-query): flat 3.29s, scene_sparse 3.59s for 59 videos.

DECISION: flat stays default — no accuracy gain (measured), no latency gain at these lengths
(measured). scene_sparse's case is the scaling trend, to be tested properly in P4/VIRAT where
videos are minutes to hours.
