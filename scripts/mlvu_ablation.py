"""
Four-arm MLVU segmentation ablation on the IDENTICAL subset as the codec
baseline (eval_results/MLVU_codec_baseline.{json,md}).

Reuses scripts/mlvu_eval.py's task loading / sampling / MC-prompt /
answer-parsing / provenance machinery verbatim (imported, not copied).
Does not change ingest/graph/retrieval/answerer logic beyond the
scene_segmentation="fixed_time_matched" mode added to iris/ingest.py.

Arms: codec, fixed_count, fixed_time_matched, fixed_seconds (60s).

Reuse structure (video-outer, arm-inner): per video, iris.charon_v.parse_video
(decode + codec parse) and _demux_packet_curve are each called ONCE and
shared across all 4 arms; only scene_id assignment / graph build / retrieval
/ answer are re-run per arm, via iris.ingest._build_index_from_records
(the existing pure-compute core, called once per arm on a shallow per-frame
copy of the shared output_frames so per-arm mutation -- action_score,
is_peak, scene_id, clip_embedding, caption -- doesn't cross-contaminate).

GUARDS:
  (a) the sampled question set is identical across all four arms (built
      once, shared) -- asserted via a content hash recorded in provenance.
  (b) the codec arm's per-task accuracy AND M-Avg must reproduce
      eval_results/MLVU_codec_baseline.json EXACTLY. Mismatch aborts loudly
      with the diff -- the ablation is contaminated otherwise.

Checkpointed after every (arm, task, question_id) so a killed process
resumes instead of restarting (mirrors mlvu_eval.py's checkpoint/resume;
this environment has been observed to kill long background jobs
unpredictably).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlvu_eval as base  # noqa: E402  (task loading / sampling / prompt / parse / provenance)

from iris.iris_config import IRISConfig  # noqa: E402
from iris import charon_v  # noqa: E402
from iris import aria  # noqa: E402
from iris.ingest import _build_index_from_records  # noqa: E402
from iris.query import query as iris_query  # noqa: E402

ARMS = ["codec", "fixed_count", "fixed_time_matched", "fixed_seconds"]
BASELINE_PATH = REPO_ROOT / "eval_results" / "MLVU_codec_baseline.json"


def arm_config(arm: str) -> IRISConfig:
    if arm == "fixed_seconds":
        return IRISConfig(scene_segmentation="fixed_seconds", fixed_scene_seconds=60.0, graph_mode="scene_sparse")
    return IRISConfig(scene_segmentation=arm, graph_mode="scene_sparse")


def parse_once(video_path: str, parse_cfg: IRISConfig):
    """Decode + codec parse ONCE per video -- shared across all 4 arms.
    Mirrors iris.ingest.ingest()'s prelude exactly (same codec_validator
    gate, same parse_video/_demux_packet_curve calls, same kwargs)."""
    try:
        from iris import codec_validator
        result = codec_validator.validate_video(str(video_path))
        if result.status == "reject":
            raise ValueError(
                f"Video rejected by codec validator: {', '.join(result.reasons) if result.reasons else 'unknown'}"
            )
        elif result.status == "warn":
            print(f"[codec warn] {video_path}: {', '.join(result.reasons) if result.reasons else ''}",
                  file=sys.stderr)
    except ImportError:
        pass

    aria.run_diagnostics()
    output_frames, stats, raw_records = charon_v.parse_video(
        str(video_path),
        return_stats=True,
        return_raw=True,
        candidate_thresh=parse_cfg.candidate_thresh,
        salient_thresh=parse_cfg.salient_thresh,
        adaptive=getattr(parse_cfg, "adaptive", True),
        visual_debug_mode=getattr(parse_cfg, "visual_debug_mode", False),
        compute_full_geometry=getattr(parse_cfg, "compute_full_geometry", False),
    )
    all_frame_energies, iframe_indices, _, _ = charon_v._demux_packet_curve(str(video_path))
    fps = charon_v.get_stream_fps(str(video_path))
    packet_curve = (all_frame_energies, iframe_indices, fps)
    return output_frames, stats, raw_records, packet_curve


def build_index_for_arm(output_frames, raw_records, stats, video_path, config, packet_curve):
    frames_copy = [dict(f) for f in output_frames]  # per-arm mutation isolation
    return _build_index_from_records(
        frames_copy, raw_records, stats, video_path, config, nms_window=10, packet_curve=packet_curve,
    )


# ── Checkpointing (per arm/task/question_id) ─────────────────────────────

def load_checkpoint(path: Path | None) -> tuple[list[dict], list[dict]]:
    if path is None or not path.exists():
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    ingest_failures = data.get("ingest_failures", [])
    print(f"[checkpoint] resuming from {path}: {len(results)} question-results, "
          f"{len(ingest_failures)} prior ingest failures")
    return results, ingest_failures


def save_checkpoint(path: Path | None, results: list[dict], ingest_failures: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"results": results, "ingest_failures": ingest_failures}, f, indent=2, default=str)
    tmp.replace(path)


def question_set_hash(pool: dict[str, list[dict]]) -> str:
    flat = []
    for task in base.MC_TASKS:
        for r in pool[task]:
            flat.append((r["task"], r["question_id"], r["question"], tuple(r["options"]), r["gold_idx"], r["video_path"]))
    blob = json.dumps(flat, sort_keys=False, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_ablation(
    anno_dir: Path,
    video_dir: Path,
    n_per_task: int,
    max_duration: float,
    seed: int,
    out_prefix: Path,
    checkpoint_path: Path | None,
) -> dict:
    parse_cfg = IRISConfig(scene_segmentation="codec", graph_mode="scene_sparse")

    video_index = base.build_video_index(video_dir)
    print(f"[setup] indexed {len(video_index)} candidate video filenames under {video_dir}")

    # 1. Load + filter + sample per task -- ONCE, shared identically by all 4 arms.
    per_task_pool: dict[str, list[dict]] = {}
    skipped_by_length: dict[str, int] = {}
    skipped_no_video: dict[str, int] = {}
    for task in base.MC_TASKS:
        anno_path = base.find_task_file(anno_dir, task)
        records = base.load_task_questions(anno_path, task)
        kept = []
        skipped_by_length[task] = 0
        skipped_no_video[task] = 0
        dur_cache: dict[str, float | None] = {}
        for r in records:
            vpath = base.resolve_video_path(r["video"], video_dir, video_index)
            if vpath is None:
                skipped_no_video[task] += 1
                continue
            key = str(vpath)
            if key not in dur_cache:
                dur_cache[key] = base.probe_duration_seconds(vpath)
            dur = dur_cache[key]
            if dur is None or dur > max_duration:
                skipped_by_length[task] += 1
                continue
            r["video_path"] = key
            r["video_duration_s"] = dur
            kept.append(r)
        sampled = base.deterministic_sample(kept, n_per_task, seed)
        print(f"[sample] {task}: sampled {len(sampled)}/{len(kept)} (n_per_task={n_per_task}, seed={seed})")
        per_task_pool[task] = sampled

    q_hash = question_set_hash(per_task_pool)
    print(f"[guard-a] question set hash (shared across all 4 arms by construction): {q_hash}")

    # 2. Group sampled questions by video so we can parse each video ONCE
    #    and reuse it across all 4 arms (video-outer, arm-inner).
    by_video: dict[str, list[dict]] = {}
    for task in base.MC_TASKS:
        for r in per_task_pool[task]:
            by_video.setdefault(r["video_path"], []).append(r)
    videos_sorted = sorted(by_video.keys())
    print(f"[setup] {len(videos_sorted)} unique videos across {sum(len(v) for v in by_video.values())} sampled questions")

    # 3. Run: video-outer, arm-inner, question-innermost.
    results, ingest_failures = load_checkpoint(checkpoint_path)
    done_keys = {(x["arm"], x["task"], x["question_id"]) for x in results}
    done_keys |= {(x["arm"], x["task"], x["question_id"]) for x in ingest_failures}
    if done_keys:
        print(f"[checkpoint] {len(done_keys)} (arm, task, question_id) triples already resolved")

    all_needed = {(arm, r["task"], r["question_id"]) for arm in ARMS for qs in by_video.values() for r in qs}
    total_needed = len(all_needed)
    print(f"[setup] {total_needed} total (arm,task,question_id) work items")

    for video_path in videos_sorted:
        qrecords = by_video[video_path]
        remaining_arms = [
            arm for arm in ARMS
            if any((arm, r["task"], r["question_id"]) not in done_keys for r in qrecords)
        ]
        if not remaining_arms:
            continue

        t0 = time.time()
        print(f"[parse] {video_path} ...")
        try:
            output_frames, stats, raw_records, packet_curve = parse_once(video_path, parse_cfg)
            print(f"[parse] {video_path} done in {time.time() - t0:.1f}s "
                  f"({len(output_frames)} survivor frames)")
        except Exception as e:
            print(f"[parse] FAILED for {video_path}: {e}", file=sys.stderr)
            for arm in ARMS:
                for r in qrecords:
                    key = (arm, r["task"], r["question_id"])
                    if key in done_keys:
                        continue
                    ingest_failures.append({"arm": arm, "task": r["task"], "video": video_path,
                                             "question_id": r["question_id"]})
                    done_keys.add(key)
            save_checkpoint(checkpoint_path, results, ingest_failures)
            continue

        for arm in remaining_arms:
            cfg = arm_config(arm)
            t1 = time.time()
            try:
                idx = build_index_for_arm(output_frames, raw_records, stats, video_path, cfg, packet_curve)
                print(f"[build] {video_path} arm={arm} done in {time.time() - t1:.1f}s")
            except Exception as e:
                print(f"[build] FAILED for {video_path} arm={arm}: {e}", file=sys.stderr)
                for r in qrecords:
                    key = (arm, r["task"], r["question_id"])
                    if key in done_keys:
                        continue
                    ingest_failures.append({"arm": arm, "task": r["task"], "video": video_path,
                                             "question_id": r["question_id"]})
                    done_keys.add(key)
                save_checkpoint(checkpoint_path, results, ingest_failures)
                continue

            for r in qrecords:
                key = (arm, r["task"], r["question_id"])
                if key in done_keys:
                    continue
                mc_prompt = base.build_mc_prompt(r["question"], r["options"])
                t2 = time.time()
                try:
                    qres = iris_query(mc_prompt, idx, cfg)
                    raw_answer = qres.get("raw_answer", "") or ""
                except Exception as e:
                    print(f"[query] FAILED for {arm}/{r['task']}/{r['question_id']}: {e}", file=sys.stderr)
                    raw_answer = ""
                    qres = {}
                elapsed = time.time() - t2

                pred_idx, method = base.parse_option(raw_answer, r["options"])
                correct = (pred_idx is not None and pred_idx == r["gold_idx"])

                results.append({
                    "arm": arm,
                    "task": r["task"],
                    "question_id": r["question_id"],
                    "video": video_path,
                    "video_duration_s": r["video_duration_s"],
                    "question": r["question"],
                    "options": r["options"],
                    "gold_idx": r["gold_idx"],
                    "raw_answer": raw_answer,
                    "final_answer": qres.get("answer", ""),
                    "pred_idx": pred_idx,
                    "parse_method": method,
                    "correct": correct,
                    "elapsed_s": elapsed,
                })
                done_keys.add(key)
                print(f"[query] arm={arm} {r['task']}/{r['question_id']}: gold={r['gold_idx']} "
                      f"pred={pred_idx} ({method}) correct={correct} [{elapsed:.1f}s] "
                      f"[{len(done_keys)}/{total_needed}]")
                save_checkpoint(checkpoint_path, results, ingest_failures)

    # 4. Score per arm / per task.
    per_arm_summary: dict[str, dict] = {}
    for arm in ARMS:
        per_task_acc: dict[str, dict] = {}
        for task in base.MC_TASKS:
            task_results = [x for x in results if x["arm"] == arm and x["task"] == task]
            n = len(task_results)
            n_correct = sum(1 for x in task_results if x["correct"])
            acc = (n_correct / n) if n else 0.0
            floor = (sum(1.0 / len(x["options"]) for x in task_results) / n) if n else 0.0
            n_parsed_fail = sum(1 for x in task_results if x["pred_idx"] is None)
            per_task_acc[task] = {
                "task_name": base.TASK_NAMES[task],
                "n": n, "n_correct": n_correct, "accuracy": acc, "random_floor": floor,
                "above_floor": (acc > floor) if n else None,
                "n_parse_failed": n_parsed_fail,
                "parse_failure_rate": (n_parsed_fail / n) if n else 0.0,
            }
        scored = [t for t in base.MC_TASKS if per_task_acc[t]["n"] > 0]
        m_avg = (sum(per_task_acc[t]["accuracy"] for t in scored) / len(scored)) if scored else 0.0
        arm_results = [x for x in results if x["arm"] == arm]
        total_parse_fail = sum(1 for x in arm_results if x["pred_idx"] is None)
        per_arm_summary[arm] = {
            "m_avg": m_avg,
            "n_total": len(arm_results),
            "overall_parse_failure_rate": (total_parse_fail / len(arm_results)) if arm_results else 0.0,
            "per_task": per_task_acc,
            "config": asdict(arm_config(arm)),
        }

    # 5. GUARD (b): codec arm must reproduce the committed baseline EXACTLY.
    baseline_ok = True
    baseline_diff = {}
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        b_summary = baseline["summary"]
        codec_summary = per_arm_summary["codec"]
        if abs(codec_summary["m_avg"] - b_summary["m_avg"]) > 1e-12:
            baseline_ok = False
            baseline_diff["m_avg"] = {"baseline": b_summary["m_avg"], "codec_arm": codec_summary["m_avg"]}
        for task in base.MC_TASKS:
            b_acc = b_summary["per_task"][task]["accuracy"]
            c_acc = codec_summary["per_task"][task]["accuracy"]
            b_n = b_summary["per_task"][task]["n"]
            c_n = codec_summary["per_task"][task]["n"]
            if abs(b_acc - c_acc) > 1e-12 or b_n != c_n:
                baseline_ok = False
                baseline_diff[task] = {"baseline_acc": b_acc, "codec_arm_acc": c_acc,
                                        "baseline_n": b_n, "codec_arm_n": c_n}
    else:
        print(f"[warn] baseline file not found at {BASELINE_PATH} -- cannot run GUARD (b)", file=sys.stderr)

    report = {
        "arms": ARMS,
        "per_arm_summary": per_arm_summary,
        "results": results,
        "ingest_failures": ingest_failures,
        "guards": {
            "question_set_hash": q_hash,
            "question_set_identical_across_arms": True,  # by construction: single shared pool
            "codec_reproduces_baseline": baseline_ok,
            "codec_vs_baseline_diff": baseline_diff,
        },
        "params": {"n_per_task": n_per_task, "max_duration_s": max_duration, "seed": seed},
        "provenance": base.git_provenance(),
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    print("\n" + "=" * 100)
    print("FULL JSON REPORT (ablation)")
    print("=" * 100)
    print(json.dumps(report, indent=2, default=str))

    if not baseline_ok:
        print("\n" + "!" * 100)
        print("GUARD (b) FAILURE: codec arm does NOT reproduce the committed baseline EXACTLY.")
        print(json.dumps(baseline_diff, indent=2))
        print("!" * 100)
        raise base.HarnessError(
            "Codec arm diverges from eval_results/MLVU_codec_baseline.json -- ablation is contaminated. "
            f"Diff: {json.dumps(baseline_diff)}"
        )

    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno-dir", type=Path, default=REPO_ROOT / "mlvu" / "MLVU" / "json")
    ap.add_argument("--video-dir", type=Path, default=REPO_ROOT / "mlvu" / "MLVU" / "video")
    ap.add_argument("--n-per-task", type=int, default=25)
    ap.add_argument("--max-duration", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=Path, default=REPO_ROOT / "eval_results" / "MLVU_ablation")
    ap.add_argument("--answerer-endpoint", type=str, default="http://localhost:11434/v1")
    ap.add_argument("--answerer-model", type=str, default="granite4:micro")
    ap.add_argument("--checkpoint-path", type=Path, default=None)
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()

    aria.set_backend(aria.LlamaBackend(endpoint=args.answerer_endpoint, text_model=args.answerer_model))
    print(f"[setup] answerer backend: LlamaBackend(endpoint={args.answerer_endpoint!r}, "
          f"text_model={args.answerer_model!r})")

    if not args.anno_dir.exists():
        raise base.HarnessError(f"Annotation dir does not exist: {args.anno_dir}")
    if not args.video_dir.exists():
        raise base.HarnessError(f"Video dir does not exist: {args.video_dir}")

    checkpoint_path = None if args.no_checkpoint else (
        args.checkpoint_path or args.out_prefix.parent / f"{args.out_prefix.name}_checkpoint.json"
    )

    run_ablation(
        anno_dir=args.anno_dir, video_dir=args.video_dir, n_per_task=args.n_per_task,
        max_duration=args.max_duration, seed=args.seed, out_prefix=args.out_prefix,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
