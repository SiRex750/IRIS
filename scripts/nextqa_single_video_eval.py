"""Run fixed NExT-QA MC evaluation for questions from one video.

Key fixes vs the scratch all-video evaluator:
  * retrieval uses question text only, never answer options;
  * MC parsing accepts only an explicit "ANSWER: <A-E>" marker;
  * retrieval goes through iris.retrieval_entry.retrieve_for_question, the
    single canonical entry point shared with eval/grounding_scorer.py and
    iris.query.query() -- no more per-script reformulation glue;
  * --query-mode selects none/legacy/structured_v2 explicitly; the frozen
    raw-question baseline (--query-mode none) is never silently overridden;
  * --top-k defaults to 4 and --max-context-frames to 8, matching the frozen
    val_confirm configuration (tuning/frozen_state.json), not the old debug
    defaults of 8/24.

Previously this evaluator called `iris_query_module._embed_query(query_text,
config)` and used the result directly as an ndarray, but _embed_query returns
(embedding, telemetry) -- passing the raw tuple into _build_retrieved would
have broken any run that actually exercised this path. Fixed by routing
through retrieve_for_question, which always uses the tuple-safe
_call_embed_query/_call_embed_queries wrappers.

The answer model is unchanged and still flows through iris.aria.generate().
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris.query_reformulation import format_mc_label, parse_mc_answer
from iris.retrieval_entry import retrieve_for_question

DATA_DIR = REPO / "eval" / "data" / "nextqa"
DEFAULT_SPLIT = DATA_DIR / "dev_100.jsonl"
DEFAULT_CACHE_DIR = DATA_DIR / "index_cache"


def build_mc_prompt(row: dict[str, Any]) -> str:
    return f"""You are answering a NExT-QA multiple-choice video question.

Use only the provided retrieved frame evidence. Do not use outside knowledge if
the evidence is insufficient.

Question:
{row["question"]}

Options:
A. {row["a0"]}
B. {row["a1"]}
C. {row["a2"]}
D. {row["a3"]}
E. {row["a4"]}

Return exactly this format:
ANSWER: <A|B|C|D|E>
REASON: <one short sentence grounded in the frame evidence>
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing split file: {path}")
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_npz_path(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / f"{video_id}.npz"


def list_cached_video_counts(rows: list[dict[str, Any]], cache_dir: Path) -> None:
    cached = {path.stem for path in cache_dir.glob("*.npz")}
    counts = Counter(row["video"] for row in rows if row["video"] in cached)

    print("Cached videos present in split:")
    print("video_id question_count")
    for video_id, count in counts.most_common():
        print(f"{video_id} {count}")
    if not counts:
        print(f"<none found under {cache_dir}>")


def retrieve_frames_for_question(
    *,
    row: dict[str, Any],
    index: Any,
    config: Any,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """Thin wrapper over the canonical retrieval entry point.

    NExT-QA's `type` column (TN/TP/TC/CW/CH/...) is passed through as
    type_code so structured_v2 gets the type-code-primary relation (section
    4); `family` (C/T/D) is passed through for the legacy reformulator's
    family-gated temporal expansion. Both are None-safe for rows without
    them.
    """
    frames, plan, telemetry = retrieve_for_question(
        row["question"],
        index,
        config,
        type_code=row.get("type"),
        family=row.get("family"),
    )
    return frames, plan, telemetry


def evaluate_one_video(args: argparse.Namespace) -> int:
    split_path = Path(args.split)
    cache_dir = Path(args.cache_dir)
    rows = load_jsonl(split_path)

    if not args.video_id:
        list_cached_video_counts(rows, cache_dir)
        print()
        print("Run again with --video-id <id> to evaluate one video.")
        return 0

    video_id = str(args.video_id)
    rows = [row for row in rows if str(row["video"]) == video_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print(f"No rows for video {video_id} in {split_path}", file=sys.stderr)
        return 2

    npz_path = cache_npz_path(cache_dir, video_id)
    if not npz_path.exists():
        print(f"Missing cache for video {video_id}: {npz_path}", file=sys.stderr)
        print("Build it first with scripts/phase6_build_dev_cache.py or the one-video build command in docs.", file=sys.stderr)
        return 2

    import iris.aria as aria
    import iris.ingest as iris_ingest
    import iris.query as iris_query
    from iris.iris_config import IRISConfig

    query_mode, traversal_mode = _resolve_query_and_traversal_mode(args)
    max_retrieval_queries = min(args.max_queries, 3) if query_mode == "structured_v2" else args.max_queries

    config = IRISConfig(
        ranking_mode=args.ranking_mode,
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=args.ppr_lambda,
        ppr_damping=args.ppr_damping,
        l2_retrieve_top_k=args.top_k,
        query_reformulation_mode=query_mode,
        max_retrieval_queries=max_retrieval_queries,
        temporal_traversal_mode=traversal_mode,
        temporal_context_seconds=args.temporal_context_seconds,
        temporal_scene_hops=args.temporal_scene_hops,
        max_context_frames=args.max_context_frames,
    )

    index = iris_ingest.load_index(npz_path)
    output_handle = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = open(output_path, "w", encoding="utf-8")

    total = 0
    correct = 0
    parse_fail = 0
    decoded_total = 0
    start = time.time()

    print(f"VIDEO_ID={video_id}")
    print(f"questions={len(rows)}")
    print(f"cached_index={npz_path}")
    print(f"indexed_frames={len(index.frames)}")
    print(f"model_backend=iris.aria default")
    print(f"top_k={args.top_k}")
    print(f"query_mode={query_mode}")
    print(f"temporal_traversal_mode={traversal_mode}")
    print()

    try:
        for row_num, row in enumerate(rows, 1):
            q_start = time.time()
            retrieved_frames, plan, retrieval_telemetry = retrieve_frames_for_question(
                row=row,
                index=index,
                config=config,
            )
            query_texts = (
                list(getattr(plan, "retrieval_queries", None) or getattr(plan, "anchor_queries", None) or [])
                if plan is not None
                else [row["question"]]
            )
            temporal_applied = bool(
                retrieval_telemetry.get("legacy_temporal_expansion_applied")
                or (
                    retrieval_telemetry.get("mode") == "structured_v2"
                    and traversal_mode == "directional"
                    and getattr(plan, "needs_temporal_traversal", False)
                )
            )

            decoded_for_captions = iris_query._ensure_captions(index, retrieved_frames)
            decoded_total += decoded_for_captions

            cache_obj = iris_query.wrapper_init_l1_cache(config)
            iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
            context_text = cache_obj.as_context_text()

            prompt = build_mc_prompt(row)
            raw_answer = aria.generate(prompt=prompt, context=context_text)
            pred = parse_mc_answer(raw_answer)
            gold = int(row["answer"])
            ok = pred == gold

            total += 1
            correct += int(ok)
            parse_fail += int(pred is None)
            elapsed = time.time() - q_start

            result = {
                "video": video_id,
                "row_num": row_num,
                "qid": row.get("qid"),
                "type": row.get("type"),
                "family": row.get("family"),
                "question": row["question"],
                "gold": gold,
                "gold_label": format_mc_label(gold),
                "pred": pred,
                "pred_label": format_mc_label(pred),
                "correct": ok,
                "parse_ok": pred is not None,
                "query_mode": query_mode,
                "retrieval_queries": query_texts,
                "retrieved_frame_idxs": [f["frame_idx"] for f in retrieved_frames],
                "temporal_expansion_applied": temporal_applied,
                "plan_relation": getattr(plan, "relation", None),
                "plan_relation_source": getattr(plan, "relation_source", None),
                "retrieval_telemetry": retrieval_telemetry,
                "decoded_for_captions": decoded_for_captions,
                "raw_answer": raw_answer,
                "elapsed_sec": elapsed,
            }

            print(
                f"[{row_num:03d}/{len(rows):03d}] "
                f"qid={row.get('qid')} type={row.get('type')} family={row.get('family')} "
                f"gold={format_mc_label(gold)} pred={format_mc_label(pred)} "
                f"ok={ok} parse_ok={pred is not None} t={elapsed:.1f}s"
            )
            print("question:", row["question"])
            print("retrieved:", result["retrieved_frame_idxs"])
            print("queries:", " | ".join(query_texts))
            print("raw:", raw_answer.replace("\n", " ")[:300])
            print("-" * 80)
            sys.stdout.flush()

            if output_handle is not None:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()

    wall = time.time() - start
    accuracy = correct / total if total else 0.0

    print("SUMMARY")
    print(f"video: {video_id}")
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"parse_fail: {parse_fail}")
    print(f"decoded_for_captions_total: {decoded_total}")
    print(f"wall_sec: {wall:.1f}")
    if args.output_jsonl:
        print(f"wrote: {args.output_jsonl}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed single-video NExT-QA evaluator for IRIS indexes.",
    )
    parser.add_argument("--video-id", help="NExT-QA video id to evaluate.")
    parser.add_argument("--split", default=str(DEFAULT_SPLIT), help="JSONL split path.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Index cache directory.")
    parser.add_argument("--limit", type=int, help="Optional number of rows from that video.")
    parser.add_argument("--output-jsonl", help="Optional detailed result log path.")
    parser.add_argument("--top-k", type=int, default=4,
                        help="Retrieved anchor frames (frozen val_confirm K=4; was 8).")
    parser.add_argument("--ranking-mode", default="ppr", choices=["ppr", "legacy"], help="L2 retrieval mode.")
    parser.add_argument("--ppr-lambda", type=float, default=0.5)
    parser.add_argument("--ppr-damping", type=float, default=0.5)
    parser.add_argument(
        "--query-mode", dest="query_mode", default="none",
        choices=["none", "legacy", "structured_v2"],
        help="Retrieval query mode via iris.retrieval_entry.retrieve_for_question. "
             "'none' (default) is the frozen raw-question baseline -- never enabled "
             "implicitly. 'legacy' is the pre-V2 reformulator (experimental baseline). "
             "'structured_v2' is the type-code-aware structured planner.",
    )
    parser.add_argument("--max-queries", type=int, default=5,
                        help="Max retrieval query variants (legacy mode); structured_v2 is hard-capped at 3.")
    parser.add_argument(
        "--temporal-traversal-mode", dest="temporal_traversal_mode", default=None,
        choices=["none", "legacy_symmetric", "directional"],
        help="Temporal context expansion. Defaults to 'none' for --query-mode none, "
             "'legacy_symmetric' for legacy, 'directional' for structured_v2.",
    )
    parser.add_argument("--temporal-context-seconds", type=float, default=4.0,
                        help="Directional traversal time window in seconds (structured_v2).")
    parser.add_argument("--temporal-scene-hops", type=int, default=1,
                        help="Directional traversal max adjacent-scene hops (structured_v2).")
    parser.add_argument("--max-context-frames", type=int, default=8,
                        help="Cap frames captioned/inserted into context (edge default; was 24).")
    # Deprecated aliases kept for existing callers -- map onto --query-mode/--temporal-traversal-mode.
    parser.add_argument("--no-reformulation", action="store_true",
                        help="Deprecated alias for --query-mode none.")
    parser.add_argument("--no-temporal-expansion", action="store_true",
                        help="Deprecated alias forcing --temporal-traversal-mode none.")
    parser.add_argument("--temporal-radius", type=int, default=2,
                        help="Deprecated, currently inert: legacy_symmetric traversal now always uses "
                             "expand_temporal_neighbors(radius=2) inside iris.retrieval_entry. Kept only "
                             "so old invocations of this script do not fail on an unknown flag.")
    return parser


def _resolve_query_and_traversal_mode(args: argparse.Namespace) -> tuple[str, str]:
    """Apply the deprecated --no-reformulation/--no-temporal-expansion aliases
    on top of --query-mode/--temporal-traversal-mode, then fill in each mode's
    natural traversal default when --temporal-traversal-mode was not given."""
    query_mode = "none" if args.no_reformulation else args.query_mode

    traversal_mode = args.temporal_traversal_mode
    if traversal_mode is None:
        traversal_mode = {"none": "none", "legacy": "legacy_symmetric", "structured_v2": "directional"}[query_mode]
    if args.no_temporal_expansion:
        traversal_mode = "none"

    return query_mode, traversal_mode


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return evaluate_one_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
