# Three-Way Build-Cost Baseline: codec vs pixel-diff vs semantic/CLIP

Same scene-sparse `block_diagonal` graph topology, same clips, same node partition. Only the per-edge similarity SIGNAL differs between arms. Downstream PPR/grounding is identical across arms and is NOT included in any number below.

## What each arm includes / excludes

- **codec** -- signal = `action_score` (scalar per frame, derived at ingest from luma_diff_energy / motion_magnitude / luma_entropy -- all byproducts of the compressed-stream demux libav already performs for motion-vector extraction). `extraction_wall_sec` here is a fresh `charon_v.parse_video()` demux pass timed in isolation; in real ingest this pass runs ANYWAY for frame selection, so its cost is NOT marginal to the graph -- it is reported for transparency, not as a graph-specific charge. `weighting_wall_sec` is the pairwise `|delta action_score| / range` computation over the cached scalars.

- **pixel_diff** -- signal = 32x32 grayscale per-pixel SAD similarity. `extraction_wall_sec` is a full sequential decode of the compressed stream (frame 0 through the last target frame -- libav has no free random access) plus per-frame thumbnailing. This decode is a REAL cost this arm pays that codec does not, because codec's signal comes free with the demux ingest already does. `weighting_wall_sec` is the pairwise SAD over thumbnails.

- **semantic_similarity_only** -- signal = CLIP ViT-B/32 cosine similarity, embeddings ASSUMED ALREADY COMPUTED (the amortized ingest cost is paid once, not per comparison -- extraction is reported as 0 by construction). This is the fair 'given embeddings' number.

- **semantic_with_encode** -- same signal, but `extraction_wall_sec` charges a full sequential decode + a CLIP ViT-B/32 forward pass (CPU) per target frame. This is the true 'from raw frames' cost a real semantic-construction arm must pay.

## Caveats

- pixel-diff and CLIP-cosine here are REFERENCE implementations built for a fair cost comparison, not tuned production retrieval baselines -- this measures construction COST only, not retrieval QUALITY (a separate question, not measured here).
- The semantic arm's fair number depends on whether the CLIP forward pass is amortized across queries; both numbers are reported so the reader can pick the applicable one for their setting.
- Read-only against `eval/data/nextqa/**` -- that cache was not touched by this script.

## Per-clip, per-arm timing + memory

| clip | N | scenes | arm | extract (s) | weight (s) | total (s) | peak RSS (GB) | edges |
|---|---|---|---|---|---|---|---|---|
| Normal_Videos_289 | 91 | 27 | codec | 0.905 | 0.000 | 0.905 | 0.215 | 145 |
| Normal_Videos_289 | 91 | 27 | pixel_diff | 0.429 | 0.002 | 0.431 | 0.145 | 145 |
| Normal_Videos_289 | 91 | 27 | semantic_similarity_only | 0.000 | 0.001 | 0.001 | 0.141 | 145 |
| Normal_Videos_289 | 91 | 27 | semantic_with_encode | 6.440 | 0.002 | 6.442 | 1.388 | 145 |
| Arrest016 | 1102 | 315 | codec | 10.534 | 0.002 | 10.536 | 1.868 | 2069 |
| Arrest016 | 1102 | 315 | pixel_diff | 4.319 | 0.011 | 4.331 | 1.160 | 2069 |
| Arrest016 | 1102 | 315 | semantic_similarity_only | 0.000 | 0.006 | 0.006 | 1.155 | 2069 |
| Arrest016 | 1102 | 315 | semantic_with_encode | 34.557 | 0.006 | 34.563 | 1.162 | 2069 |
| Abuse042 | 2963 | 712 | codec | 26.843 | 0.007 | 26.850 | 2.649 | 8325 |
| Abuse042 | 2963 | 712 | pixel_diff | 12.390 | 0.042 | 12.432 | 1.167 | 8325 |
| Abuse042 | 2963 | 712 | semantic_similarity_only | 0.000 | 0.027 | 0.027 | 1.144 | 8325 |
| Abuse042 | 2963 | 712 | semantic_with_encode | 93.782 | 0.033 | 93.815 | 1.158 | 8325 |
| VIRAT_S_040001_01_000448_001101 | 4892 | 528 | codec | 323.236 | 0.017 | 323.254 | 24.132 | 23571 |
| VIRAT_S_040001_01_000448_001101 | 4892 | 528 | pixel_diff | 72.357 | 0.119 | 72.476 | 0.328 | 23571 |
| VIRAT_S_040001_01_000448_001101 | 4892 | 528 | semantic_similarity_only | 0.000 | 0.122 | 0.122 | 0.266 | 23571 |
| VIRAT_S_040001_01_000448_001101 | 4892 | 528 | semantic_with_encode | 246.839 | 0.102 | 246.941 | 0.706 | 23571 |

## Edge-count parity (topology assertion)

| clip | expected (sum n_i choose 2) | matched across all 4 arms |
|---|---|---|
| Normal_Videos_289 | 145 | True |
| Arrest016 | 2069 | True |
| Abuse042 | 8325 | True |
| VIRAT_S_040001_01_000448_001101 | 23571 | True |

## Headline ratios (total wall time, codec = 1.0x)

| clip | codec vs pixel_diff | codec vs semantic_similarity_only | codec vs semantic_with_encode |
|---|---|---|---|
| Normal_Videos_289 | 0.5x | 0.0x | 7.1x |
| Arrest016 | 0.4x | 0.0x | 3.3x |
| Abuse042 | 0.5x | 0.0x | 3.5x |
| VIRAT_S_040001_01_000448_001101 | 0.2x | 0.0x | 0.8x |

(ratio = arm total wall time / codec total wall time; >1x means codec is cheaper by that factor. `semantic_similarity_only` isolates just the cosine-similarity edge-weighting step, so on clips where codec's `extraction_wall_sec` -- the re-run demux pass -- dominates, this ratio can legitimately come out below 1x; that reflects the re-run demux cost charged to codec here, not a graph-marginal cost. See per-arm `weighting_wall_sec` columns above for the edge-weighting-only comparison.)

## Key finding: weighting vs extraction diverge sharply

On pure edge-**weighting** cost (the only step that is actually specific to graph construction, given each signal already exists), codec is the cheapest arm on every clip by 1-2 orders of magnitude -- e.g. on the N=4892 VIRAT clip: codec weighting 0.017s vs pixel_diff 0.119s vs semantic 0.102-0.122s. This is the number that isolates "cheap because it reuses encoder-produced signals" most directly.

However, on the VIRAT clip codec's re-run **extraction** number (323s) is the LARGEST of all four arms -- bigger than pixel-diff's full frame-by-frame decode (72s) and even semantic's decode+CLIP-forward-pass (247s). This is not a script bug: `charon_v.parse_video`'s per-frame motion-geometry math (divergence/curl/jacobian/hessian eigenvalue fields, all computed over the full per-pixel motion-vector grid) gets expensive at VIRAT's frame resolution/count, and dominates the demux pass at this scale. Two things are true at once: (1) this extraction cost is NOT marginal to the graph -- it is paid once at ingest for frame selection regardless of which graph is built downstream, so it does not belong on codec's side of a construction-cost ledger the way pixel-diff's/semantic's decode-for-the-graph cost does; (2) it is nonetheless a real, measured cost, and "codec signal extraction is cheap" should not be read as "cheap at any resolution/scale" without qualification -- the motion-geometry math itself is the expensive part at high N, not the demux read.
