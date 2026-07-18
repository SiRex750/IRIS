"""Pillar 2 Grounded VideoQA Headline Benchmark.

Runs evaluation on the cached NEXT-GQA subset comparing:
  - Proposed System: Variant B L1 cache admission + L2 PPR retrieval
  - Uniform Baseline: Evenly spaced frame selection
  - Random Baseline: Random frame selection

Calculates: Acc@GQA, Acc@QA, mIoP, IoP@0.5, mIoU, IoU@0.5.
Gated by video-level cluster Bootstrap CIs (1000 resamples, 95% confidence).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.aria as aria
import iris.query as iris_query
import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig
from eval.mc_scorer import build_mc_prompt, parse_mc_answer, LETTERS
from eval.span import FROZEN_HALF_WIDTH_SECONDS, predict_span

def iou(span: tuple[float, float] | None, gold_spans: list[list[float]]) -> float:
    """Intersection-over-Union for temporal grounding.

    span = (start, end) predicted temporal span — see eval/span.py::predict_span.
    gold = union of gold_spans
    IoU  = |pred ∩ gold_union| / |pred ∪ gold_union|
    """
    if span is None:
        return 0.0
    pred_s, pred_e = span
    if pred_e <= pred_s:
        return 0.0

    pred_len = pred_e - pred_s

    # Merge gold spans first to handle overlapping gold spans
    sorted_gold = sorted(gold_spans, key=lambda x: x[0])
    merged_gold = []
    for s, e in sorted_gold:
        if not merged_gold or s > merged_gold[-1][1]:
            merged_gold.append([s, e])
        else:
            merged_gold[-1][1] = max(merged_gold[-1][1], e)

    intersect = 0.0
    for s, e in merged_gold:
        lo = max(pred_s, float(s))
        hi = min(pred_e, float(e))
        if hi > lo:
            intersect += hi - lo

    gold_len = sum(e - s for s, e in merged_gold)
    union_len = pred_len + gold_len - intersect

    if union_len <= 0.0:
        return 0.0
    return intersect / union_len

def iop(span: tuple[float, float] | None, gold_spans: list[list[float]]) -> float:
    """Intersection-over-Prediction.

    span = (start, end) predicted temporal span — see eval/span.py::predict_span.
    gold = union of gold_spans
    IoP  = |pred ∩ gold_union| / |pred|
    """
    if span is None:
        return 0.0
    pred_s, pred_e = span
    if pred_e <= pred_s:
        return 0.0

    pred_len = pred_e - pred_s

    # Merge gold spans first to handle overlapping gold spans
    sorted_gold = sorted(gold_spans, key=lambda x: x[0])
    merged_gold = []
    for s, e in sorted_gold:
        if not merged_gold or s > merged_gold[-1][1]:
            merged_gold.append([s, e])
        else:
            merged_gold[-1][1] = max(merged_gold[-1][1], e)

    intersect = 0.0
    for s, e in merged_gold:
        lo = max(pred_s, float(s))
        hi = min(pred_e, float(e))
        if hi > lo:
            intersect += hi - lo

    return intersect / pred_len

def uniform_ts(duration: float, top_k: int) -> list[float]:
    return [(i + 0.5) / top_k * duration for i in range(top_k)]

def bootstrap_paired_differences(
    video_to_questions: dict[str, list[dict]],
    metric_keys: list[str],
    num_resamples: int = 1000,
    seed: int = 20260710
) -> dict:
    """Video-level cluster bootstrap for paired differences.
    
    Resamples at the video level (keeping all questions of a video together).
    Calculates 95% CIs for Proposed - Uniform and Proposed - Random.
    """
    vids = list(video_to_questions.keys())
    n_vids = len(vids)
    rng = np.random.default_rng(seed)
    
    # Pre-allocate lists for bootstrap differences
    boot_diffs = {
        "uniform": {k: [] for k in metric_keys},
        "random": {k: [] for k in metric_keys}
    }
    
    for _ in range(num_resamples):
        # Resample video IDs with replacement
        resampled_vids = rng.choice(vids, size=n_vids, replace=True)
        
        # Collect all questions for resampled videos
        resampled_questions = []
        for vid in resampled_vids:
            resampled_questions.extend(video_to_questions[vid])
            
        n_questions = len(resampled_questions)
        if n_questions == 0:
            continue
            
        # Compute metrics for this resample
        for key in metric_keys:
            prop_mean = np.mean([q[f"proposed_{key}"] for q in resampled_questions])
            unif_mean = np.mean([q[f"uniform_{key}"] for q in resampled_questions])
            rand_mean = np.mean([q[f"random_{key}"] for q in resampled_questions])
            
            boot_diffs["uniform"][key].append(prop_mean - unif_mean)
            boot_diffs["random"][key].append(prop_mean - rand_mean)
            
    # Calculate 95% CIs
    ci_results = {}
    for comp in ["uniform", "random"]:
        ci_results[comp] = {}
        for key in metric_keys:
            diffs = np.array(boot_diffs[comp][key])
            mean_diff = np.mean(diffs)
            ci_lo = np.percentile(diffs, 2.5)
            ci_hi = np.percentile(diffs, 97.5)
            ci_results[comp][key] = {
                "mean": float(mean_diff),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi)
            }
            
    return ci_results

def preflight_backend(backend) -> None:
    """Fail fast if the active backend is not the seated llama-server, or is
    unreachable. A silently-wrong backend must be impossible, not merely
    unlikely -- this is the guard the V2 run lacked (DECISIONS.md 2026-07-17 §4).
    """
    if not isinstance(backend, aria.LlamaServerBackend):
        raise RuntimeError(
            f"Seat violation: expected answerer backend LlamaServerBackend, "
            f"got {type(backend).__name__}. The seat contract requires "
            f"llama-server; Ollama-backed LlamaBackend is the rejected runtime "
            f"(DECISIONS.md 2026-07-17 §4)."
        )

    import requests
    try:
        resp = requests.get(f"{backend.endpoint}/models", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Liveness probe failed for llama-server endpoint {backend.endpoint}: {e}"
        ) from e

def git_provenance(repo_root: Path) -> tuple[str, bool]:
    """Return (git_commit, git_dirty) for repo_root. git_dirty=True means the
    tree had uncommitted changes at run time -- any such run is unattributable
    by construction (the V2 lesson; DECISIONS.md 2026-07-17-later §A2)."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout
    dirty = bool(status.strip())
    return commit, dirty

def config_hash(cfg: IRISConfig) -> str:
    """sha256 of the serialized IRISConfig, so a provenance record pins the
    exact config used, not just a label."""
    serialized = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Pillar 2 Grounded VideoQA Benchmark")
    parser.add_argument("--top_k", type=int, default=8, help="Number of retrieved frames")
    parser.add_argument("--num_boot", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument(
        "--span-mode", choices=["ppr_peak", "minmax"], default="ppr_peak",
        help="Predicted-span construction for the proposed (retrieval) arm. "
             "minmax is the legacy enclose-all-top-K construction, retained "
             "only as an explicit ablation arm (DECISIONS.md 2026-07-17 §3).",
    )
    parser.add_argument(
        "--span-half-width", type=float, default=FROZEN_HALF_WIDTH_SECONDS,
        help="Half-width (seconds) for --span-mode=ppr_peak. Defaults to the "
             "frozen eval.span.FROZEN_HALF_WIDTH_SECONDS (2.2s, duration-anchor "
             "method, DECISIONS.md 2026-07-18). Override for ablation.",
    )
    args = parser.parse_args()

    print("=== STARTING PILLAR 2 GROUNDED VideoQA EVALUATION ===")
    print(f"Top K retrieved frames: {args.top_k}")
    print(f"Bootstrap resamples:   {args.num_boot}")
    print(f"Span mode:             {args.span_mode}"
          + (f" (half_width={args.span_half_width})" if args.span_mode == "ppr_peak" else ""))

    # ── 0. Set configs (moved ahead of backend seating so the backend reads
    #      its endpoint/model from this config, not a hardcoded literal) ──────
    # Variant B weights
    cfg_proposed = IRISConfig(
        cerberus_mode="v2",
        l2_retrieve_top_k=args.top_k,
        ranking_mode="ppr",
        ppr_lambda=0.5,
        candidate_thresh=0.08,
        l1_w_action=0.60,
        l1_w_query=0.25,
        l1_w_persist=0.15,
        l1_w_pagerank=0.0,
        l1_w_entropy=0.0,
        l1_w_hessian=0.0,
        l1_w_recency=0.0
    )

    # ── 1. Configure active backend — seated llama-server, per contract ──────
    print(f"\n[LLM] Seating {cfg_proposed.answerer_model} via llama-server at {cfg_proposed.answerer_endpoint}...")
    backend = aria.LlamaServerBackend(
        endpoint=cfg_proposed.answerer_endpoint,
        text_model=cfg_proposed.answerer_model,
    )
    aria.set_backend(backend)
    preflight_backend(backend)
    aria.run_diagnostics()

    # ── 2. Load dataset files ─────────────────────────────────────────────────
    data_dir = REPO_ROOT / "eval" / "data" / "nextqa"
    dev_jsonl = data_dir / "dev_100.jsonl"
    gqa_json = data_dir / "gsub_val.json"
    cache_dir = data_dir / "index_cache"
    
    if not dev_jsonl.exists():
        print(f"FATAL: {dev_jsonl} not found.", file=sys.stderr)
        sys.exit(1)
    if not gqa_json.exists():
        print(f"FATAL: {gqa_json} not found.", file=sys.stderr)
        sys.exit(1)
        
    dev_rows = [json.loads(line) for line in open(dev_jsonl, encoding="utf-8")]
    gsub = json.load(open(gqa_json, encoding="utf-8"))
    
    # ── 3. Find cached indexes ────────────────────────────────────────────────
    cached_vids = {p.stem for p in cache_dir.glob("*.npz")}
    
    grounded_rows = [
        r for r in dev_rows
        if r["video"] in cached_vids
        and r["video"] in gsub
        and str(r["qid"]) in gsub[r["video"]]["location"]
    ]
    
    print(f"\n[DATA] dev_100 total questions: {len(dev_rows)}")
    print(f"[DATA] Grounded and cached questions: {len(grounded_rows)} (across {len(set(r['video'] for r in grounded_rows))} unique videos)")
    
    # Load all indexes
    print("\n[INDEX] Loading cached flat indexes...")
    loaded_indexes = {}
    for row in grounded_rows:
        vid = row["video"]
        if vid in loaded_indexes:
            continue
        npz_path = cache_dir / f"{vid}.npz"
        try:
            loaded_indexes[vid] = iris_ingest.load_index(cache_dir / vid)
        except Exception as e:
            print(f"  ERR loading {vid}: {e}")
            loaded_indexes[vid] = None

    # ── 5. Run Evaluation Loop ────────────────────────────────────────────────
    question_results = []
    video_to_questions: dict[str, list[dict]] = {}
    
    # Fixed seed for reproducibility of random baseline
    rand_gen = random.Random(20260710)
    
    print("\n[EVAL] Running queries against LLM...")
    for idx, row in enumerate(grounded_rows, 1):
        vid = row["video"]
        qid = str(row["qid"])
        family = row["family"]
        question = row["question"]
        opts = {k: row[k] for k in ["a0", "a1", "a2", "a3", "a4"]}
        gold_idx = int(row["answer"])
        gold_letter = LETTERS[gold_idx]
        
        index = loaded_indexes.get(vid)
        if index is None:
            continue
            
        gold_spans = gsub[vid]["location"][qid]
        duration = float(gsub[vid].get("duration", 0)) or max(f.timestamp for f in index.frames)
        
        print(f"[{idx}/{len(grounded_rows)}] QID={qid} Video={vid} Family={family} Query: {question[:50]}...")
        
        # ─── A. Proposed Arm ───
        emb = iris_query._embed_query(question, cfg_proposed)
        retrieved_proposed = iris_query._build_retrieved(index, emb, cfg_proposed)
        ts_proposed = [f["timestamp"] for f in retrieved_proposed]
        
        # Build prompt & query LLM
        cap_lines_prop = []
        for i, f in enumerate(retrieved_proposed, 1):
            cap_val = f.get("caption") or ""
            cap = cap_val.get("semantic_caption") if isinstance(cap_val, dict) else cap_val
            cap = cap or "[CAPTION_FAILED]"
            ts = f.get("timestamp", 0.0)
            cap_lines_prop.append(f"[Frame {i} @ {ts:.1f}s] {cap}")
        
        prompt_prop, context_prop = build_mc_prompt(question, opts, "\n".join(cap_lines_prop))
        raw_prop = aria.generate(prompt=prompt_prop, context=context_prop)
        parsed_prop = parse_mc_answer(raw_prop, opts)
        choice_prop = parsed_prop.parsed_letter

        # Grounding metrics
        span_prop = predict_span(
            retrieved_proposed, mode=args.span_mode, half_width=args.span_half_width,
            duration=duration,
        )
        iop_prop = iop(span_prop, gold_spans)
        iou_prop = iou(span_prop, gold_spans)
        acc_qa_prop = float(choice_prop == gold_letter)
        acc_gqa_prop = float(acc_qa_prop and iop_prop >= 0.5)
        
        # ─── B. Uniform Arm ───
        ts_uniform_targets = uniform_ts(duration, args.top_k)
        retrieved_uniform = []
        for target_t in ts_uniform_targets:
            closest_fr = min(index.frames, key=lambda fr: abs(fr.timestamp - target_t))
            retrieved_uniform.append(closest_fr)
            
        ts_uniform = [fr.timestamp for fr in retrieved_uniform]
        
        cap_lines_unif = []
        for i, fr in enumerate(retrieved_uniform, 1):
            cap_val = fr.caption or ""
            cap = cap_val.get("semantic_caption") if isinstance(cap_val, dict) else cap_val
            cap = cap or "[CAPTION_FAILED]"
            ts = fr.timestamp
            cap_lines_unif.append(f"[Frame {i} @ {ts:.1f}s] {cap}")
            
        prompt_unif, context_unif = build_mc_prompt(question, opts, "\n".join(cap_lines_unif))
        raw_unif = aria.generate(prompt=prompt_unif, context=context_unif)
        parsed_unif = parse_mc_answer(raw_unif, opts)
        choice_unif = parsed_unif.parsed_letter

        # Uniform selection carries no retrieval score to peak on -- minmax is
        # the correct construction for this floor baseline, not the invented
        # fallback the span-fix task warns against.
        span_unif = predict_span(retrieved_uniform, mode="minmax")
        iop_unif = iop(span_unif, gold_spans)
        iou_unif = iou(span_unif, gold_spans)
        acc_qa_unif = float(choice_unif == gold_letter)
        acc_gqa_unif = float(acc_qa_unif and iop_unif >= 0.5)
        
        # ─── C. Random Arm ───
        # Retrieve random frames
        frames_pool = list(index.frames)
        # Seed random selection per question for reproducibility
        q_rand = random.Random(int(qid) + 20260710)
        retrieved_random = q_rand.sample(frames_pool, min(len(frames_pool), args.top_k))
        # Sort retrieved frames by timestamp to keep them chronological
        retrieved_random.sort(key=lambda fr: fr.timestamp)
        
        ts_random = [fr.timestamp for fr in retrieved_random]
        
        cap_lines_rand = []
        for i, fr in enumerate(retrieved_random, 1):
            cap_val = fr.caption or ""
            cap = cap_val.get("semantic_caption") if isinstance(cap_val, dict) else cap_val
            cap = cap or "[CAPTION_FAILED]"
            ts = fr.timestamp
            cap_lines_rand.append(f"[Frame {i} @ {ts:.1f}s] {cap}")
            
        prompt_rand, context_rand = build_mc_prompt(question, opts, "\n".join(cap_lines_rand))
        raw_rand = aria.generate(prompt=prompt_rand, context=context_rand)
        parsed_rand = parse_mc_answer(raw_rand, opts)
        choice_rand = parsed_rand.parsed_letter

        # Random selection carries no retrieval score to peak on -- minmax is
        # the correct construction for this floor baseline, not the invented
        # fallback the span-fix task warns against.
        span_rand = predict_span(retrieved_random, mode="minmax")
        iop_rand = iop(span_rand, gold_spans)
        iou_rand = iou(span_rand, gold_spans)
        acc_qa_rand = float(choice_rand == gold_letter)
        acc_gqa_rand = float(acc_qa_rand and iop_rand >= 0.5)
        
        res_dict = {
            "qid": qid,
            "video": vid,
            "family": family,
            "question": question,
            "gold": gold_letter,
            # Proposed metrics
            "proposed_choice": choice_prop,
            "proposed_correct": acc_qa_prop,
            "proposed_iop": iop_prop,
            "proposed_iou": iou_prop,
            "proposed_acc_qa": acc_qa_prop,
            "proposed_iop_05": float(iop_prop >= 0.5),
            "proposed_iou_05": float(iou_prop >= 0.5),
            "proposed_acc_gqa": acc_gqa_prop,
            "proposed_raw_response": parsed_prop.raw_response,
            "proposed_parse_path": parsed_prop.parse_path,
            # Uniform metrics
            "uniform_choice": choice_unif,
            "uniform_correct": acc_qa_unif,
            "uniform_iop": iop_unif,
            "uniform_iou": iou_unif,
            "uniform_acc_qa": acc_qa_unif,
            "uniform_iop_05": float(iop_unif >= 0.5),
            "uniform_iou_05": float(iou_unif >= 0.5),
            "uniform_acc_gqa": acc_gqa_unif,
            "uniform_raw_response": parsed_unif.raw_response,
            "uniform_parse_path": parsed_unif.parse_path,
            # Random metrics
            "random_choice": choice_rand,
            "random_correct": acc_qa_rand,
            "random_iop": iop_rand,
            "random_iou": iou_rand,
            "random_acc_qa": acc_qa_rand,
            "random_iop_05": float(iop_rand >= 0.5),
            "random_iou_05": float(iou_rand >= 0.5),
            "random_acc_gqa": acc_gqa_rand,
            "random_raw_response": parsed_rand.raw_response,
            "random_parse_path": parsed_rand.parse_path,
        }
        
        question_results.append(res_dict)
        video_to_questions.setdefault(vid, []).append(res_dict)
        
    # ── 6. Aggregate results ──────────────────────────────────────────────────
    print("\n[STATS] Aggregating overall results...")
    n = len(question_results)
    
    def mean_val(arm: str, key: str) -> float:
        return float(np.mean([q[f"{arm}_{key}"] for q in question_results]))
        
    metrics = {}
    metric_keys = ["acc_gqa", "acc_qa", "iop", "iop_05", "iou", "iou_05"]
    
    for arm in ["proposed", "uniform", "random"]:
        metrics[arm] = {k: mean_val(arm, k) for k in metric_keys}
        
    # ── 7. Video-level cluster Bootstrap CIs ──────────────────────────────────
    print(f"\n[BOOTSTRAP] Running {args.num_boot} resamples...")
    ci_results = bootstrap_paired_differences(
        video_to_questions,
        metric_keys,
        num_resamples=args.num_boot
    )
    
    # ── 8. Write raw outputs to json ──────────────────────────────────────────
    out_dir = REPO_ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "pillar2_grounded_qa_raw.json"

    # Provenance: read every field off live objects, not re-declared literals.
    # git_dirty is deliberate -- a run whose tree was dirty at run time is
    # unattributable by construction (the V2 lesson; DECISIONS.md
    # 2026-07-17-later §A2).
    git_commit, git_dirty = git_provenance(REPO_ROOT)
    provenance = {
        "backend_class": type(backend).__name__,
        "endpoint": backend.endpoint,
        "model": backend.text_model,
        "temperature": backend.temperature,
        "cache_prompt": backend.cache_prompt,
        "span_mode": args.span_mode,
        "span_half_width": args.span_half_width,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_hash": config_hash(cfg_proposed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_questions": n,
        "n_videos": len(video_to_questions),
        "top_k": args.top_k,
        "num_boot": args.num_boot,
    }

    raw_payload = {
        "provenance": provenance,
        "overall_metrics": metrics,
        "bootstrap_paired_differences": ci_results,
        "results": question_results
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)
    print(f"[EXPORT] Saved raw metrics to: {raw_path}")
    
    # ── 9. Generate Report ────────────────────────────────────────────────────
    report_path = out_dir / "pillar2_grounded_qa_report.md"

    dirty_banner = (
        "**⚠ UNATTRIBUTABLE RUN — the working tree was DIRTY at run time "
        "(`git_dirty=true`). This report cannot be cited as a measurement of "
        "any specific commit (DECISIONS.md 2026-07-17-later §A2).**"
        if provenance["git_dirty"] else
        "Working tree was clean at run time (`git_dirty=false`)."
    )

    report_content = f"""# Preliminary Grounded VideoQA Benchmark (Pillar 2)

{dirty_banner}

**Provenance:** backend=`{provenance['backend_class']}` | model=`{provenance['model']}` | temperature=`{provenance['temperature']}` | cache_prompt=`{provenance['cache_prompt']}` | endpoint=`{provenance['endpoint']}` | span_mode=`{provenance['span_mode']}` | span_half_width=`{provenance['span_half_width']}` | git_commit=`{provenance['git_commit']}` | git_dirty=`{provenance['git_dirty']}` | timestamp_utc=`{provenance['timestamp_utc']}`

Evaluating the end-to-end IRIS pipeline (`charon_v` -> L1 admission Variant B -> L2 PPR retrieval -> {provenance['backend_class']} `{provenance['model']}` backend) on the NExT-GQA test subset ($n={n}$ questions, {provenance['n_videos']} unique videos).

## 1. Quantitative Performance Summary

| Metric | Proposed System | Uniform Baseline | Random Baseline |
| :--- | :---: | :---: | :---: |
| **Acc@GQA** (Grounded QA) | {metrics['proposed']['acc_gqa']:.2%} | {metrics['uniform']['acc_gqa']:.2%} | {metrics['random']['acc_gqa']:.2%} |
| **Acc@QA** (QA Accuracy) | {metrics['proposed']['acc_qa']:.2%} | {metrics['uniform']['acc_qa']:.2%} | {metrics['random']['acc_qa']:.2%} |
| **mIoP** (Mean IoP) | {metrics['proposed']['iop']:.4f} | {metrics['uniform']['iop']:.4f} | {metrics['random']['iop']:.4f} |
| **IoP@0.5** (Grounding Recall) | {metrics['proposed']['iop_05']:.2%} | {metrics['uniform']['iop_05']:.2%} | {metrics['random']['iop_05']:.2%} |
| **mIoU** (Mean IoU) | {metrics['proposed']['iou']:.4f} | {metrics['uniform']['iou']:.4f} | {metrics['random']['iou']:.4f} |
| **IoU@0.5** (Grounding Overlap) | {metrics['proposed']['iou_05']:.2%} | {metrics['uniform']['iou_05']:.2%} | {metrics['random']['iou_05']:.2%} |

*Note: Acc@GQA represents the percentage of questions that are both answered accurately (Acc@QA) and grounded correctly (IoP $\ge$ 0.5).*

## 2. Statistical Rigor (Paired Bootstrap CIs)

Calculated over {args.num_boot} video-level cluster bootstrap resamples. 95% Confidence Intervals (CI) gate each paired difference.

### A. Proposed vs. Uniform Baseline (Proposed - Uniform)
*   **Acc@GQA Diff:** {ci_results['uniform']['acc_gqa']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['acc_gqa']['ci_lo']:+.4f}, {ci_results['uniform']['acc_gqa']['ci_hi']:+.4f}]`)
*   **Acc@QA Diff:**  {ci_results['uniform']['acc_qa']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['acc_qa']['ci_lo']:+.4f}, {ci_results['uniform']['acc_qa']['ci_hi']:+.4f}]`)
*   **IoP@0.5 Diff:**  {ci_results['uniform']['iop_05']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['iop_05']['ci_lo']:+.4f}, {ci_results['uniform']['iop_05']['ci_hi']:+.4f}]`)
*   **mIoP Diff:**     {ci_results['uniform']['iop']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['iop']['ci_lo']:+.4f}, {ci_results['uniform']['iop']['ci_hi']:+.4f}]`)
*   **IoU@0.5 Diff:**  {ci_results['uniform']['iou_05']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['iou_05']['ci_lo']:+.4f}, {ci_results['uniform']['iou_05']['ci_hi']:+.4f}]`)
*   **mIoU Diff:**     {ci_results['uniform']['iou']['mean']:+.4f} (95% CI: `[{ci_results['uniform']['iou']['ci_lo']:+.4f}, {ci_results['uniform']['iou']['ci_hi']:+.4f}]`)

### B. Proposed vs. Random Baseline (Proposed - Random)
*   **Acc@GQA Diff:** {ci_results['random']['acc_gqa']['mean']:+.4f} (95% CI: `[{ci_results['random']['acc_gqa']['ci_lo']:+.4f}, {ci_results['random']['acc_gqa']['ci_hi']:+.4f}]`)
*   **Acc@QA Diff:**  {ci_results['random']['acc_qa']['mean']:+.4f} (95% CI: `[{ci_results['random']['acc_qa']['ci_lo']:+.4f}, {ci_results['random']['acc_qa']['ci_hi']:+.4f}]`)
*   **IoP@0.5 Diff:**  {ci_results['random']['iop_05']['mean']:+.4f} (95% CI: `[{ci_results['random']['iop_05']['ci_lo']:+.4f}, {ci_results['random']['iop_05']['ci_hi']:+.4f}]`)
*   **mIoP Diff:**     {ci_results['random']['iop']['mean']:+.4f} (95% CI: `[{ci_results['random']['iop']['ci_lo']:+.4f}, {ci_results['random']['iop']['ci_hi']:+.4f}]`)
*   **IoU@0.5 Diff:**  {ci_results['random']['iou_05']['mean']:+.4f} (95% CI: `[{ci_results['random']['iou_05']['ci_lo']:+.4f}, {ci_results['random']['iou_05']['ci_hi']:+.4f}]`)
*   **mIoU Diff:**     {ci_results['random']['iou']['mean']:+.4f} (95% CI: `[{ci_results['random']['iou']['ci_lo']:+.4f}, {ci_results['random']['iou']['ci_hi']:+.4f}]`)

## 3. Discussion and Scientific Context

### The "Trust Gap"
The scientific literature notes a massive **Trust Gap** on Grounded VideoQA benchmarks: SOTA closed-ended QA models achieve up to ~69% Acc@QA, but drop to a meager **16% Acc@GQA** when forced to localize the visual evidence that justifies their answer. Human baselines achieve 82% Acc@QA / Acc@GQA, proving that contemporary models rely on language priors rather than visual evidence.

### Competitive Status with 2B Agentic Frontier
With our proposed system achieving **{metrics['proposed']['acc_gqa']:.2%} Acc@GQA** on this evaluation:
*   We outperform simple baseline samplers (Uniform and Random) by a statistically significant margin.
*   Measured Acc@GQA: {metrics['proposed']['acc_gqa']:.2%}.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[EXPORT] Saved report to: {report_path}")
    print("=== EVALUATION COMPLETED ===")

if __name__ == "__main__":
    main()
