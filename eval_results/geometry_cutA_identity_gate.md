# Cut A identity gate: VIRAT N=4,892

**Outcome: GATE_PASS**

- Gate 4 (graph identity, bit-identical edges + PPR): PASS
- Gate 6 (L1/ARIA safety, hessian/entropy bit-identical): PASS
- Gate 7 (schema, gated fields still present): PASS

## Edge comparison

- edges (full geometry): 23571
- edges (Cut A): 23571
- only in full: 0, only in Cut A: 0
- field mismatches (weight/semantic/motion/temporal/edge_type), tolerance 0.0: 0

## PPR ranking (fixed query embedding, top_k=10)

- order identical: True
- scores bit-identical: True

## Cut A gated fields (divergence/curl/jacobian_frobenius)

- divergence: full-geometry run has nonzero values = True; Cut A run all sentinel 0.0 = True
- curl: full-geometry run has nonzero values = True; Cut A run all sentinel 0.0 = True
- jacobian_frobenius: full-geometry run has nonzero values = True; Cut A run all sentinel 0.0 = True

## Cut B preserved fields (hessian_max_eigenvalue, motion_entropy)

- hessian_max_eigenvalue: bit-identical = True (mismatches: 0)
- motion_entropy: bit-identical = True (mismatches: 0)

## Graph-read fields (action_score, persistence_value, ... pagerank_score)

- action_score: bit-identical = True (mismatches: 0)
- persistence_value: bit-identical = True (mismatches: 0)
- luma_diff_energy: bit-identical = True (mismatches: 0)
- motion_magnitude: bit-identical = True (mismatches: 0)
- luma_entropy: bit-identical = True (mismatches: 0)
- packet_size: bit-identical = True (mismatches: 0)
- pict_type: bit-identical = True (mismatches: 0)
- codec_conf: bit-identical = True (mismatches: 0)
- scene_id: bit-identical = True (mismatches: 0)
- pagerank_score: bit-identical = True (mismatches: 0)

## Extraction wall-time (charon_v.parse_video only)

- full geometry: 362.685s
- Cut A: 358.706s
- delta: 3.979s (1.1% of full-geometry extraction time)
- total ingest (extraction + CLIP-encode + graph build), full geometry: 662.558s
- total ingest (extraction + CLIP-encode + graph build), Cut A: 662.968s

