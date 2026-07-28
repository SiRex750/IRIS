# Build-Cost 3-Way Baseline — Notes (defensible claim + required caveat)

Source: `scripts/build_cost_3way.py`, results in `build_cost_3way.{json,md}`,
raw run log `build_cost_3way_run.log`. Read-only against
`eval/data/{ucf,virat}/**` and `eval/data/nextqa/**` — no committed index
cache was touched or re-derived by this script.

## Defensible claim

Given the frame-similarity signal is already needed, codec edge-**weighting**
is 1-2 orders of magnitude cheaper than pixel-diff or semantic construction,
on the identical scene-sparse `block_diagonal` graph topology (same node
partition, same edge count per clip, asserted equal across all arms).

On the VIRAT N=4,892 clip (528 scenes, 23,571 edges — the largest clip
measured):

| arm | weighting (s) |
|---|---|
| codec | 0.017 |
| pixel_diff | 0.119 |
| semantic_similarity_only | 0.122 |
| semantic_with_encode | 0.102 |

Both semantic sub-arms are reported explicitly:
- `semantic_similarity_only` — CLIP cosine similarity, embeddings assumed
  already computed (amortized ingest cost paid once, not charged here). The
  "given embeddings" number.
- `semantic_with_encode` — same signal, but a full sequential decode + CLIP
  ViT-B/32 forward pass per target frame is charged to extraction. The true
  "from raw frames" number.

This holds across all four clips measured (Normal_Videos_289, Arrest016,
Abuse042, VIRAT_S_040001_01_000448_001101) — see `build_cost_3way.md` for the
full per-clip table.

## Required caveat (do not drop when citing this result)

**Codec EXTRACTION is NOT cheaper than decode at scale.** On the VIRAT clip,
codec's re-run demux pass (323.236s) is the *largest* extraction cost of all
four arms — larger than pixel-diff's full frame-by-frame decode (72.357s) and
even semantic's decode+CLIP-forward-pass (246.839s). This is dominated by
`compute_motion_geometry`'s per-frame gradient math inside
`charon_v.parse_video`, not the demux read itself (see
`eval_results/geometry_gate_plan.md`).

Codec's extraction cost is not marginal to the graph: it is the same
`charon_v.parse_video()` demux pass that ingest already runs once for frame
selection, regardless of which downstream graph is built. It is **amortized
against that ingest demux**, not paid specifically for the graph the way
pixel-diff's and semantic's decode-for-the-graph cost is.

**Do not claim codec extraction is cheap.** The defensible claim above is
scoped to edge-*weighting* only — the one step that is actually specific to
graph construction once each signal already exists.
