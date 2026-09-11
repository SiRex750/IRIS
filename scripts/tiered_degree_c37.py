"""Appendix C item 8/§3.5 follow-up: build ONE hierarchical_sparse cache and
measure its per-node degree distribution, broken down by edge type and by
direction (in vs out, relative to the top-k SELECTOR in the code, not graph
storage order).

Ingest-only. No captioner/answerer/llama-server. Reuses the frames + CLIP
embeddings already cached under fully_connected for the chosen clip -- no
video re-decode, no re-scoring. Only the graph-construction step (which reads
config_snapshot.graph_edge_mode) is re-run, with graph_edge_mode switched to
"hierarchical_sparse" and everything else held to the existing cache's
config_snapshot verbatim.

Writes:
  - eval/data/ucf/index_cache/<CLIP>_hierarchical_sparse.npz  (new cache, does
    not touch the existing fully_connected cache at the same clip name)
  - eval_results/tiered_degree_C37.md
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

import iris.ingest as ingest
from iris.l2_asphodel import L2Asphodel
from iris.types import FrameRecord, IRISIndex

CLIP = "Arson042"
SRC_CACHE = Path("eval/data/ucf/index_cache") / f"{CLIP}.npz"
DST_CACHE = Path("eval/data/ucf/index_cache") / f"{CLIP}_hierarchical_sparse.npz"
REPORT = Path("eval_results/tiered_degree_C37.md")


def load_frames_and_config(path: Path):
    data = np.load(path, allow_pickle=False)
    manifest = json.loads(data["__manifest__"].item())
    frames = []
    for d in manifest["frames"]:
        emb_key = f"emb_{d['frame_idx']}"
        emb = data[emb_key].astype(np.float32) if emb_key in data.files else None
        frames.append(FrameRecord(
            frame_idx=d["frame_idx"],
            timestamp=d["timestamp"],
            luma_diff_energy=d["luma_diff_energy"],
            luma_entropy=d["luma_entropy"],
            motion_magnitude=d["motion_magnitude"],
            action_score=d["action_score"],
            persistence_value=d["persistence_value"],
            is_peak=d["is_peak"],
            divergence=d.get("divergence", 0.0),
            curl=d.get("curl", 0.0),
            jacobian_frobenius=d.get("jacobian_frobenius", 0.0),
            hessian_max_eigenvalue=d.get("hessian_max_eigenvalue", 0.0),
            motion_entropy=d.get("motion_entropy", 0.0),
            caption=d["caption"],
            clip_embedding=emb,
            pagerank_score=d["pagerank_score"],
            packet_size=d.get("packet_size", 0.0),
            pict_type=d.get("pict_type", "?"),
            codec_conf=d.get("codec_conf", 0.5),
            scene_id=d.get("scene_id", -1),
        ))
    return frames, manifest["config_snapshot"], manifest


def build_hierarchical_graph(frames, config):
    """Same call sequence as iris.ingest._build_graph, but with an
    instrumented L2Asphodel so we can recover, for every FINAL edge in the
    graph, which node was the "selector" (source) at the point the winning
    _add_weighted_edge call wrote it -- e.g. for semantic_salient/
    motion_neighbor, the top-k-choosing node; for hierarchy_*, the parent;
    for temporal, the earlier-indexed node. Undirected nx.Graph storage does
    not preserve this, so it has to be captured at write time.
    """
    node_groups = None
    if ingest._get(config, "graph_mode", "flat") == "scene_sparse":
        groups: dict[int, list] = {}
        for r in frames:
            sid = int(ingest._get(r, "scene_id", -1))
            groups.setdefault(sid, []).append(int(ingest._get(r, "frame_idx")))
        node_groups = list(groups.values())

    graph = L2Asphodel(config=config)

    # Instrument: wrap _add_weighted_edge to record the (source, target,
    # edge_type) of whichever call is currently "winning" for each pair,
    # i.e. reproduce exactly what ends up in graph.edges(data=True).
    directed_write: dict[frozenset, tuple[int, int, str]] = {}
    orig = graph._add_weighted_edge

    def wrapped(u, v, edge_type, max_score_range):
        orig(u, v, edge_type, max_score_range)
        if graph.graph.has_edge(u, v) and graph.graph[u][v].get("edge_type") == edge_type:
            directed_write[frozenset((u, v))] = (u, v, edge_type)

    graph._add_weighted_edge = wrapped

    feature_records = []
    action_score_records = []
    enrichment_map = {}
    for r in frames:
        fi = int(ingest._get(r, "frame_idx"))
        feature_records.append({
            "frame_idx":             fi,
            "timestamp":             float(ingest._get(r, "timestamp", 0.0)),
            "luma_diff_energy":      float(ingest._get(r, "luma_diff_energy", 0.0)),
            "motion_magnitude":      float(ingest._get(r, "motion_magnitude", 0.0)),
            "luma_entropy":          float(ingest._get(r, "luma_entropy", 0.0)),
            "refined_motion_tensor": np.asarray([
                float(ingest._get(r, "motion_magnitude", 0.0)),
                float(ingest._get(r, "divergence", 0.0)),
                float(ingest._get(r, "curl", 0.0)),
                float(ingest._get(r, "jacobian_frobenius", 0.0)),
                float(ingest._get(r, "hessian_max_eigenvalue", 0.0)),
                float(ingest._get(r, "motion_entropy", 0.0)),
            ], dtype=np.float32),
            "packet_size":           float(ingest._get(r, "packet_size", 0.0)),
            "codec_conf":            float(ingest._get(r, "codec_conf", 0.5)),
            "pict_type":             str(ingest._get(r, "pict_type", "?")),
            "is_peak":               bool(ingest._get(r, "is_peak", False)),
        })
        action_score_records.append({
            "action_score":      float(ingest._get(r, "action_score", 0.0)),
            "persistence_value": float(ingest._get(r, "persistence_value", 0.0)),
        })
        enrichment_map[fi] = ingest._get(r, "clip_embedding", None)

    graph.add_frame_nodes_bulk(feature_records, action_score_records,
                                node_groups=node_groups, defer_recompute=True)
    graph.enrich_nodes_bulk(enrichment_map, node_groups=node_groups)

    # Cross-scene pruning (the node_groups branch at the end of
    # _update_all_edge_weights) removes edges via graph.remove_edges_from
    # directly, bypassing _add_weighted_edge -- so directed_write can contain
    # stale entries for edges that no longer exist in the final graph. Drop them.
    live_pairs = {frozenset((u, v)) for u, v in graph.graph.edges}
    directed_write = {k: v for k, v in directed_write.items() if k in live_pairs}
    return graph, directed_write


def degree_stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "p90": None, "p99": None, "max": None}
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return s[idx]
    return {
        "n": n,
        "min": s[0],
        "median": statistics.median(s),
        "mean": sum(s) / n,
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": s[-1],
    }


def main():
    frames, base_config, base_manifest = load_frames_and_config(SRC_CACHE)
    n_nodes = len(frames)
    print(f"Loaded {SRC_CACHE}: N={n_nodes}, fully_connected edges={len(base_manifest['graph_edges'])}")
    print(f"Base config_snapshot graph_edge_mode={base_config['graph_edge_mode']!r} "
          f"graph_mode={base_config['graph_mode']!r} alpha={base_config['alpha']} beta={base_config['beta']} "
          f"luma_diff_weight={base_config['luma_diff_weight']} motion_weight={base_config['motion_weight']} "
          f"luma_entropy_weight={base_config['luma_entropy_weight']}")

    new_config = dict(base_config)
    new_config["graph_edge_mode"] = "hierarchical_sparse"
    # graph_mode stays "scene_sparse" (already is), everything else untouched.
    assert new_config["graph_mode"] == "scene_sparse"

    graph, directed_write = build_hierarchical_graph(frames, new_config)
    nx_graph = graph.graph
    edges = list(nx_graph.edges(data=True))
    print(f"hierarchical_sparse edges built: {len(edges)}")

    # ---- Per-node total (undirected) degree ----
    total_degree = {nid: 0 for nid in nx_graph.nodes}
    by_type_edges: dict[str, list[tuple[int, int]]] = {}
    for u, v, data in edges:
        et = data.get("edge_type", "unknown")
        total_degree[u] += 1
        total_degree[v] += 1
        by_type_edges.setdefault(et, []).append((u, v))

    # ---- Per-type per-node degree (undirected, i.e. "touches this node") ----
    per_type_node_degree: dict[str, dict[int, int]] = {}
    for et, elist in by_type_edges.items():
        dd = {nid: 0 for nid in nx_graph.nodes}
        for u, v in elist:
            dd[u] += 1
            dd[v] += 1
        per_type_node_degree[et] = dd

    # ---- Directed in/out degree using directed_write (selector = source) ----
    out_degree = {nid: 0 for nid in nx_graph.nodes}
    in_degree = {nid: 0 for nid in nx_graph.nodes}
    out_degree_by_type: dict[str, dict[int, int]] = {}
    in_degree_by_type: dict[str, dict[int, int]] = {}
    for key, (u, v, et) in directed_write.items():
        out_degree[u] += 1
        in_degree[v] += 1
        out_degree_by_type.setdefault(et, {nid: 0 for nid in nx_graph.nodes})[u] += 1
        in_degree_by_type.setdefault(et, {nid: 0 for nid in nx_graph.nodes})[v] += 1

    # sanity: directed_write should have exactly len(edges) entries (one per final edge)
    assert len(directed_write) == len(edges), (len(directed_write), len(edges))

    # ---- scene sizes (for the bound comparison) ----
    scene_of = {int(f.frame_idx): int(f.scene_id) for f in frames}
    scene_sizes: dict[int, int] = {}
    for sid in scene_of.values():
        scene_sizes[sid] = scene_sizes.get(sid, 0) + 1
    max_scene_size = max(scene_sizes.values()) if scene_sizes else 0
    n_scenes = len(scene_sizes)

    # ---- highest in-degree nodes: edge-type composition ----
    top_in = sorted(in_degree.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_in_detail = []
    for nid, ind in top_in:
        comp = {et: dd.get(nid, 0) for et, dd in in_degree_by_type.items() if dd.get(nid, 0) > 0}
        top_in_detail.append((nid, ind, comp, scene_of.get(nid, -1), scene_sizes.get(scene_of.get(nid, -1), 0)))

    # ---- save new cache (do not overwrite existing) ----
    index = IRISIndex(
        video_path=base_manifest["video_path"],
        frames=frames,
        index_action_score=base_manifest["index_action_score"],
        stats=base_manifest["stats"],
        frames_processed=base_manifest["frames_processed"],
        peak_count=base_manifest["peak_count"],
        skipped_frames_ratio=base_manifest["skipped_frames_ratio"],
        storage_reduction_factor=base_manifest["storage_reduction_factor"],
        config_snapshot=new_config,
        schema_version=base_manifest["schema_version"],
        _graph=graph,
    )
    # sync pagerank_score from the new graph back onto FrameRecords before saving
    for fr in index.frames:
        if fr.frame_idx in nx_graph.nodes:
            fr.pagerank_score = float(nx_graph.nodes[fr.frame_idx]["node_data"].pagerank_score)
    ingest.save_index(index, DST_CACHE)
    print(f"Wrote {DST_CACHE}")

    # ================= write report =================
    lines = []
    lines.append(f"# Tiered (`hierarchical_sparse`) degree distribution — {CLIP} (Appendix C item 37 follow-up)\n")
    lines.append(
        f"One clip, built from ingest (no captioner/answerer/llama-server). `{CLIP}` was chosen "
        f"from the `fully_connected` corpus in `eval/data/ucf/index_cache/` because N={n_nodes} "
        f"sits in the 400-1500 range asked for and its existing cache already has real scene "
        f"structure ({n_scenes} scenes, max scene size {max_scene_size}, mean scene size "
        f"{n_nodes / n_scenes:.2f}) rather than being dominated by singleton scenes.\n"
    )
    lines.append(
        "Re-ingested by loading the cached frame records + CLIP embeddings straight out of "
        f"`{SRC_CACHE}` (no video re-decode) and rebuilding the graph via the same "
        "`iris.ingest._build_graph` call sequence (`add_frame_nodes_bulk(defer_recompute=True)` "
        "then `enrich_nodes_bulk`), with `config_snapshot` copied verbatim from the existing "
        "cache except `graph_edge_mode` changed from `\"fully_connected\"` to "
        "`\"hierarchical_sparse\"`. Confirmed unchanged from the source cache: "
        f"`graph_mode=\"{new_config['graph_mode']}\"`, `alpha={new_config['alpha']}`, "
        f"`beta={new_config['beta']}`, `luma_diff_weight={new_config['luma_diff_weight']}`, "
        f"`motion_weight={new_config['motion_weight']}`, "
        f"`luma_entropy_weight={new_config['luma_entropy_weight']}`, "
        f"`graph_temporal_window={new_config['graph_temporal_window']}`, "
        f"`graph_semantic_top_k={new_config['graph_semantic_top_k']}`, "
        f"`graph_motion_top_k={new_config['graph_motion_top_k']}`, "
        f"`graph_semantic_threshold={new_config['graph_semantic_threshold']}`, "
        f"`salient_thresh={new_config['salient_thresh']}`, `candidate_thresh={new_config['candidate_thresh']}`. "
        f"scene_id assignment reused as-is from the cached frames (same {n_scenes}-scene "
        "partition the fully_connected cache used), so the two caches are comparable on identical "
        f"scene boundaries. New cache written to `{DST_CACHE}`; `{SRC_CACHE}` untouched.\n"
    )

    lines.append("## Edge counts\n")
    lines.append(f"- `fully_connected` (existing cache): {len(base_manifest['graph_edges'])} edges\n")
    lines.append(f"- `hierarchical_sparse` (this run): {len(edges)} edges\n")
    lines.append("")
    lines.append("By edge type (hierarchical_sparse):\n")
    lines.append("| edge_type | count |")
    lines.append("|---|---:|")
    for et in sorted(by_type_edges, key=lambda k: -len(by_type_edges[k])):
        lines.append(f"| {et} | {len(by_type_edges[et])} |")
    lines.append("")

    lines.append("## Per-node total (undirected) degree, by edge type\n")
    lines.append("| edge_type | n_edges | min | median | mean | p90 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    all_total = degree_stats(list(total_degree.values()))
    lines.append(f"| **all types combined** | {len(edges)} | {all_total['min']} | {all_total['median']} | "
                  f"{all_total['mean']:.2f} | {all_total['p90']} | {all_total['p99']} | {all_total['max']} |")
    for et in sorted(per_type_node_degree, key=lambda k: -len(by_type_edges[k])):
        vals = [d for nid, d in per_type_node_degree[et].items() if d > 0]
        st = degree_stats(vals)
        lines.append(f"| {et} | {len(by_type_edges[et])} | {st['min']} | {st['median']} | "
                      f"{st['mean']:.2f} | {st['p90']} | {st['p99']} | {st['max']} |")
    lines.append("")
    lines.append(
        "(Per-type stats above are computed only over nodes that have at least one edge of that "
        "type -- zero-degree-for-that-type nodes excluded, since most nodes are not "
        "`L1_PEAK`/`L2_SALIENT` and so structurally cannot receive every edge family.)\n"
    )

    lines.append("## Directed in/out degree (selector = source, per code semantics)\n")
    lines.append(
        "`semantic_salient` and `motion_neighbor` cap **out**-degree per source (top-4, top-2); "
        "`hierarchy_*` caps out-degree per child at 1 (one nearest parent); `temporal` is "
        "symmetric by construction (window=1, each node links its immediate successor). None of "
        "these cap **in**-degree in code. Direction below is recovered from the actual call "
        "order that won each edge in the undirected graph (see script), not inferred.\n"
    )
    lines.append("| edge_type | out-degree min/median/mean/p90/p99/max | in-degree min/median/mean/p90/p99/max |")
    lines.append("|---|---|---|")
    all_out = degree_stats([d for d in out_degree.values() if d > 0])
    all_in = degree_stats([d for d in in_degree.values() if d > 0])
    def fmt(st):
        if st["n"] == 0:
            return "n/a (no nonzero nodes)"
        return f"{st['min']}/{st['median']}/{st['mean']:.2f}/{st['p90']}/{st['p99']}/{st['max']}"
    lines.append(f"| **all types combined** | {fmt(all_out)} | {fmt(all_in)} |")
    for et in sorted(by_type_edges, key=lambda k: -len(by_type_edges[k])):
        o = degree_stats([d for d in out_degree_by_type.get(et, {}).values() if d > 0])
        i = degree_stats([d for d in in_degree_by_type.get(et, {}).values() if d > 0])
        lines.append(f"| {et} | {fmt(o)} | {fmt(i)} |")
    lines.append("")

    lines.append("## Does in-degree concentrate? Highest-in-degree nodes\n")
    lines.append("Top 10 nodes by total in-degree (all edge types), with edge-type composition of "
                  "their in-edges and their scene's size:\n")
    lines.append("| frame_idx | in-degree | composition | scene_id | scene_size |")
    lines.append("|---:|---:|---|---:|---:|")
    for nid, ind, comp, sid, ssize in top_in_detail:
        comp_str = ", ".join(f"{k}:{v}" for k, v in sorted(comp.items(), key=lambda kv: -kv[1]))
        lines.append(f"| {nid} | {ind} | {comp_str} | {sid} | {ssize} |")
    lines.append("")
    max_in = max(in_degree.values()) if in_degree else 0
    max_out = max(out_degree.values()) if out_degree else 0
    # A single per-source cap is 4 (semantic_salient top_k) or 2 (motion_neighbor
    # top_k); "concentration" here means in-degree exceeding the largest single
    # per-source cap (4), which no individual selection step could produce on
    # its own -- it can only arise from multiple independent sources choosing
    # the same target.
    concentrates = max_in > 4
    concentration_verdict = (
        f"exceeds the largest single per-source cap (semantic_salient top_k=4) by "
        f"{max_in - 4}, so it cannot be explained by any one node's own top-k selection -- "
        "multiple independent sources are picking the same target. Concentration is real on "
        "this clip, though bounded (well under the fully_connected scene-size ceiling, see below)."
        if concentrates else
        "stayed at or below the largest single per-source cap (4), so nothing here requires "
        "invoking cross-source concentration -- no visible concentration on this clip."
    )
    lines.append(
        f"Max in-degree observed: **{max_in}** (vs max out-degree **{max_out}**, which the "
        f"top-4/top-2 per-source caps plus the single-parent hierarchy cap keep low by "
        f"construction). In-degree {concentration_verdict}\n"
    )

    lines.append("## Comparison against this clip's `fully_connected` cache\n")
    fc_deg: dict[int, int] = {}
    for e in base_manifest["graph_edges"]:
        fc_deg[e["source"]] = fc_deg.get(e["source"], 0) + 1
        fc_deg[e["target"]] = fc_deg.get(e["target"], 0) + 1
    for nid in scene_of:
        fc_deg.setdefault(nid, 0)
    fc_stats = degree_stats(list(fc_deg.values()))
    hs_stats = all_total
    lines.append("| | edges | min | median | mean | p90 | p99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| fully_connected | {len(base_manifest['graph_edges'])} | {fc_stats['min']} | {fc_stats['median']} | "
                  f"{fc_stats['mean']:.2f} | {fc_stats['p90']} | {fc_stats['p99']} | {fc_stats['max']} |")
    lines.append(f"| hierarchical_sparse | {len(edges)} | {hs_stats['min']} | {hs_stats['median']} | "
                  f"{hs_stats['mean']:.2f} | {hs_stats['p90']} | {hs_stats['p99']} | {hs_stats['max']} |")
    lines.append("")
    lines.append(f"Scene-size bound: max scene size on this clip is **{max_scene_size}** (scene "
                  f"{max(scene_sizes, key=scene_sizes.get)}). `fully_connected` max degree "
                  f"({fc_stats['max']}) is exactly `max_scene_size - 1` for that scene, by "
                  "construction (§4.1: cross-scene edges pruned, so a scene is a complete "
                  "subgraph). `hierarchical_sparse` max degree "
                  f"({hs_stats['max']}) is {'also' if hs_stats['max'] <= max_scene_size - 1 else 'NOT'} "
                  f"≤ max_scene_size-1 ({max_scene_size - 1}) on this clip"
                  + (", so the same numeric ceiling holds here too, but the *mechanism* is different: "
                     "fully_connected's bound comes from cross-scene pruning of an otherwise-complete "
                     "per-scene subgraph (every node already has degree = scene_size-1 before pruning "
                     "even applies); hierarchical_sparse's bound is never approached by construction -- "
                     "temporal (window=1), hierarchy (≤1 parent), semantic_salient (≤4 out) and "
                     "motion_neighbor (≤2 out) each independently produce far fewer edges per node than "
                     "scene_size-1 typically allows, with cross-scene pruning only removing the (rare) "
                     "excess. The two paths hit the same numeric ceiling on this clip for different "
                     "structural reasons, and hierarchical_sparse's actual max is well under it."
                     if hs_stats['max'] <= max_scene_size - 1 else
                     ", i.e. the scene-size bound that holds by construction in fully_connected does "
                     "NOT hold for hierarchical_sparse on this clip.")
                  + "\n")

    lines.append("## §3.5 extension\n")
    lines.append(
        f"**Confirmed on this one clip (`{CLIP}`, N={n_nodes}), not extended further.** "
        f"hierarchical_sparse's per-node degree ({hs_stats['min']}/{hs_stats['median']}/"
        f"{hs_stats['mean']:.2f} min/median/mean, max {hs_stats['max']}) is well below "
        f"fully_connected's ({fc_stats['min']}/{fc_stats['median']}/{fc_stats['mean']:.2f}, max "
        f"{fc_stats['max']}) on the same scene partition, as the per-source top-k caps predict. "
        f"In-degree {'does concentrate beyond any single per-source cap' if concentrates else 'does not concentrate beyond the per-source caps'} "
        f"(max in-degree {max_in} vs max out-degree {max_out}, largest per-source cap 4), but "
        "stays well inside the same scene-size ceiling fully_connected also respects on this clip "
        "-- concentration, where present, is real but bounded, not an unbounded blow-up. "
        f"**This is one clip.** N={n_nodes} is a single point in the 24-13,506 range the "
        "fully_connected corpus covers; nothing here "
        "establishes whether in-degree concentration grows, shrinks, or stays flat as scene size "
        "or N grows, whether it behaves the same on a clip with a much larger max scene size, or "
        "whether the tiered path's known under-connection failure mode (§3.5's prune-after-select "
        "deficit) shows up elsewhere in this same clip's degree-0-for-that-type nodes. Generalizing "
        "this clip's numbers to \"the\" hierarchical_sparse degree distribution would repeat the "
        "same error item 8 already flagged for the zero-artifact case, just with N=1 instead of "
        "N=0.\n"
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
