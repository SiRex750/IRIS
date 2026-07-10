"""End-to-end CCTV query smoke -- does the full chain produce an answer?

NOT a quality eval. A DOES-IT-RUN check, no mechanism changes.

Ingests (or reuses a cached) scene_sparse index on VIRAT_S_000102.mp4 under
production defaults (rep_only, tau=0.015), then runs the FULL top-level
question->answer entry point (iris.query.query -- retrieval, lazy captioning,
L1 Elysium population, ARIA generation, claim split, Cerberus-V gate) on 3
natural CCTV queries. Prints everything raw: retrieved frames+timestamps,
their lazily-generated captions, and the final answer. No scoring.

VERIFY: python scripts/virat_query_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_000102.mp4"
CACHE_DIR = REPO / "eval" / "data" / "virat" / "index_cache"
CACHE_PATH = CACHE_DIR / "VIRAT_S_000102"

# Production defaults: only graph_mode is set explicitly; scene_crossscene_mode
# (rep_only) and scene_shortcut_margin (0.015, tau) come from IRISConfig defaults.
CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
    graph_mode="scene_sparse",
)

QUERIES = [
    "Is anyone loading or unloading a vehicle?",
    "Did a person enter the building?",
    "Is there an unattended bag or object left behind?",
]


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    if not VIDEO.exists():
        print(f"FATAL: {VIDEO} not found.", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz = Path(str(CACHE_PATH) + ".npz")

    if npz.exists():
        print(f"Reusing cached index: {npz}")
        idx = iris_ingest.load_index(CACHE_PATH)
    else:
        print(f"No cache found -- ingesting {VIDEO.name} fresh "
              f"(production defaults: scene_crossscene_mode={CFG.scene_crossscene_mode}, "
              f"scene_shortcut_margin={CFG.scene_shortcut_margin}). This is the long-tail build.")
        sys.stdout.flush()
        idx = iris_ingest.ingest(str(VIDEO), config=CFG)
        iris_ingest.save_index(idx, CACHE_PATH)
        print(f"Saved index cache to {CACHE_PATH}")

    print(f"num_survivors={len(idx.frames)}  num_scenes={len(idx._scene_centroids)}")
    print(f"torch device available: {_device()}")
    print()

    for qi, question in enumerate(QUERIES, 1):
        print("=" * 100)
        print(f"QUERY {qi}: {question!r}")
        print("=" * 100)

        result = iris_query.query(question, idx, CFG)

        frame_map = {fr.frame_idx: fr for fr in idx.frames}
        retrieved_idxs = result["retrieved_frame_idxs"]

        print(f"-- RETRIEVED TOP-{len(retrieved_idxs)} FRAMES --")
        for fi in retrieved_idxs:
            fr = frame_map.get(fi)
            ts = fr.timestamp if fr is not None else None
            cap = fr.caption if fr is not None else None
            print(f"  frame_idx={fi}  timestamp={ts}")
            print(f"    caption: {cap!r}")

        print()
        print(f"-- FRAMES DECODED FOR LAZY CAPTIONING: {result['frames_decoded_for_captions']} --")
        print()

        print("-- RAW LLM ANSWER (pre-claim-split) --")
        print(result["raw_answer"])
        print()

        print("-- CLAIM VERIFICATION (Cerberus-V) --")
        print(f"  nli_mocked (mocked/fallback path used): {result['nli_mocked']}")
        print(f"  verified (all claims passed):           {result['verified']}")
        print(f"  verified_claims:   {result['verified_claims']}")
        print(f"  rejected_claims:   {result['rejected_claims']}")
        print(f"  unverifiable_claims: {result['unverifiable_claims']}")
        print()

        print("-- FINAL ANSWER --")
        print(result["answer"])
        print()

        print("-- TIMINGS (sec) --")
        for k, v in result["timings"].items():
            print(f"  {k:<12}: {v:.3f}")
        print()


if __name__ == "__main__":
    main()
