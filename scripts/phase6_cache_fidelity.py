"""Phase 6 — Cache-fidelity + codec-sanity gate on ONE video.

(a) CODEC_SANITY: ingest the video, check pict_type mix and packet_size variance.
    Real inter-coded H.264/VidOR content must have I/P/B mix (not all-I) and
    wide packet_size spread. SUSPECT → stop (codec signal untrustworthy).

(b) CACHE_FIDELITY: save_index → load_index, assert identical frame_idx ordering
    and identical codec_conf per node for PPR retrieval on dev questions.
    FAIL → stop and report raw mismatch; do not loosen the assert.

Never mocks or synthesizes video/frame data.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

DATA_DIR    = REPO / "eval" / "data" / "nextqa"
FLAT_DIR    = DATA_DIR / "NExTVideo_flat"
CACHE_DIR   = DATA_DIR / "index_cache"
DEV_JSONL   = DATA_DIR / "dev_100.jsonl"

# Video with 3 dev questions — more thorough fidelity check
TEST_VID = "3462517143"

CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
)


def main() -> None:
    video_path = FLAT_DIR / f"{TEST_VID}.mp4"
    if not video_path.exists():
        print(f"ERROR: {video_path} not found — aborting")
        sys.exit(1)

    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    vid_questions = [r for r in dev_rows if r["video"] == TEST_VID]
    print(f"Test video : {TEST_VID}")
    print(f"Dev questions for this video: {len(vid_questions)}")

    # ── (a) CODEC_SANITY ─────────────────────────────────────────────────────

    print("\n=== PHASE 1a: CODEC_SANITY ===")
    t0 = time.time()
    index = iris_ingest.ingest(str(video_path), config=CFG)
    elapsed = time.time() - t0
    print(f"Ingest elapsed: {elapsed:.1f}s")

    n_nodes = len(index.frames)
    print(f"N_nodes: {n_nodes}")

    pict_counts: dict = {}
    for fr in index.frames:
        pict_counts[fr.pict_type] = pict_counts.get(fr.pict_type, 0) + 1
    print("pict_type counts:", pict_counts)

    packet_sizes = [fr.packet_size for fr in index.frames if fr.packet_size > 0]
    if packet_sizes:
        ps_min = min(packet_sizes)
        ps_max = max(packet_sizes)
        ps_mean = statistics.mean(packet_sizes)
        ps_std = statistics.stdev(packet_sizes) if len(packet_sizes) > 1 else 0.0
        print(f"packet_size: min={ps_min:.0f} max={ps_max:.0f} mean={ps_mean:.0f} std={ps_std:.0f}")
    else:
        ps_min = ps_max = ps_mean = ps_std = 0.0
        print("packet_size: no non-zero values found")

    # Sanity checks
    n_i = pict_counts.get("I", 0)
    n_p = pict_counts.get("P", 0)
    n_b = pict_counts.get("B", 0)
    n_total = n_i + n_p + n_b

    reasons = []
    suspect = False

    if n_total == 0:
        reasons.append("NO frames with pict_type")
        suspect = True
    elif n_p == 0 and n_b == 0:
        reasons.append(f"all-I encode (I={n_i}, P={n_p}, B={n_b}) — likely re-encoded to keyframes-only")
        suspect = True
    else:
        reasons.append(f"I/P/B mix confirmed (I={n_i}, P={n_p}, B={n_b})")

    if ps_std < 10.0:
        reasons.append(f"packet_size std={ps_std:.1f} < 10 — near-uniform bytes, codec signal has no variance")
        suspect = True
    elif ps_max / max(ps_min, 1) < 5.0:
        reasons.append(f"packet_size spread max/min={ps_max/max(ps_min,1):.1f}x < 5x — low dynamic range")
        suspect = True
    else:
        reasons.append(f"packet_size spread ok (max/min={ps_max/max(ps_min,1):.0f}x, std={ps_std:.0f})")

    status = "SUSPECT" if suspect else "PASS"
    print(f"CODEC_SANITY: {status} — {'; '.join(reasons)}")

    if suspect:
        print("STOPPING: codec sanity failed — codec_conf signal cannot be trusted on this mirror.")
        sys.exit(1)

    # ── (b) CACHE_FIDELITY ───────────────────────────────────────────────────

    print("\n=== PHASE 1b: CACHE_FIDELITY ===")

    cache_path = CACHE_DIR / TEST_VID
    print(f"Saving index to {cache_path}.npz ...")
    iris_ingest.save_index(index, cache_path)

    print("Loading index back ...")
    loaded_index = iris_ingest.load_index(cache_path)

    # Assert codec_conf per node is identical
    fresh_cc  = {fr.frame_idx: fr.codec_conf for fr in index.frames}
    loaded_cc = {fr.frame_idx: fr.codec_conf for fr in loaded_index.frames}
    cc_mismatch = []
    for fi, cc in fresh_cc.items():
        lcc = loaded_cc.get(fi)
        if lcc is None or abs(lcc - cc) > 1e-6:
            cc_mismatch.append((fi, cc, lcc))

    if cc_mismatch:
        print(f"CACHE_FIDELITY: FAIL — codec_conf mismatch on {len(cc_mismatch)} frames, first: {cc_mismatch[0]}")
        sys.exit(1)
    print(f"codec_conf round-trip: OK ({len(fresh_cc)} nodes, all within 1e-6)")

    # Assert identical PPR retrieval ordering on dev questions
    questions = [r["question"] for r in vid_questions[:3]]
    if not questions:
        questions = ["what is happening in the video", "describe the scene"]
    print(f"Running PPR retrieval on {len(questions)} question(s):")

    fidelity_ok = True
    for qi, q in enumerate(questions):
        emb_fresh  = iris_query._embed_query(q, CFG)
        emb_loaded = iris_query._embed_query(q, CFG)  # same model, same input

        ret_fresh  = iris_query._build_retrieved(index,        emb_fresh,  CFG)
        ret_loaded = iris_query._build_retrieved(loaded_index, emb_loaded, CFG)

        idxs_fresh  = [f["frame_idx"] for f in ret_fresh]
        idxs_loaded = [f["frame_idx"] for f in ret_loaded]

        match = idxs_fresh == idxs_loaded
        print(f"  Q{qi+1}: {repr(q[:60])}")
        print(f"    fresh  top-8: {idxs_fresh}")
        print(f"    loaded top-8: {idxs_loaded}")
        print(f"    ordering match: {match}")
        if not match:
            fidelity_ok = False
            print(f"    FIRST MISMATCH: fresh={idxs_fresh}, loaded={idxs_loaded}")

    if not fidelity_ok:
        print("CACHE_FIDELITY: FAIL — retrieval ordering differs between fresh and loaded index")
        sys.exit(1)

    print("CACHE_FIDELITY: PASS")


if __name__ == "__main__":
    main()
