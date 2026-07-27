"""END-TO-END VAL_CONFIRM RUN -- full pipeline, verification layer off.

First run in the tuning phase producing the project's actual target
metric (Acc@GQA), not a retrieval-only proxy. Every prior number
(mIoP/mIoU/IoP@0.5) came from scripts/part3_tune.py, which never calls
the captioner, answerer, or Cerberus. This script uses the real pipeline
(captioner + answerer via iris.aria.generate(), the same call path
scripts/nextqa_single_video_eval.py exercises for one video), extended
to the full val_confirm split, all 12 currently-frozen hyperparameters
applied live from tuning/frozen_state.json, and Cerberus verification
explicitly disabled (cerberus_mode="none") -- an explicit choice, not an
oversight.

val_confirm is the held-out split (per split_manifest.json) never used
by any hyperparameter family above -- specifically meant to catch
overfitting to val_tune.

Deliberately NOT run through part3_tune.py's run_family()/FAMILIES
machinery (this isn't a grid sweep, no family selection) -- but reuses
its ingest/index-cache/config machinery pattern and eval/metrics.py's
predicted_span_from_frames_peak (Method D) and the canonical
nextgqa_metrics.py the same way part3_tune.py does, so results are
directly comparable to every prior family's numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import iris.aria as aria  # noqa: E402
import iris.query as iris_query  # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402
from iris.query_reformulation import parse_mc_answer, format_mc_label  # noqa: E402
from iris.retrieval_entry import retrieve_for_question  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402

from answerer_provenance import capture_answerer_provenance  # noqa: E402
from caption_dump_io import (  # noqa: E402
    apply_caption_load, load_caption_dump, record_caption_dump, write_caption_dump,
)
from part3_tune import (  # noqa: E402
    INGEST_RELEVANT_KEYS, load_frozen_state, TUNING_DIR,
)

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)

VIDEO_DIR = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"
# Dedicated fresh cache dir -- NOT tuning/index_cache/ or index_cache_scenespans/
# or any other cache used by any hyperparameter family. Must not exist before
# this run starts (checked in main()).
INDEX_CACHE_DIR = TUNING_DIR / "index_cache_val_confirm_e2e"

PER_QUESTION_CSV = TUNING_DIR / "val_confirm_e2e_per_question.csv"
REPORT_PATH = TUNING_DIR / "val_confirm_e2e_report.md"

# --- Split parameterisation -------------------------------------------------
# A split controls EXACTLY three inputs: which video ids are in scope, which
# questions CSV supplies the rows, and which JSON supplies the gold spans.
# Nothing else -- retrieval config, span method, scorer, and parser are read
# live from tuning/frozen_state.json and are identical across splits. The
# output/cache paths below are bookkeeping, not behaviour: they exist so a
# test-split run cannot overwrite a recorded val_confirm artifact.
#
# official_test video ids come from the benchmark_runs manifest, NOT the repo
# root one. The root split_manifest.json has no "partitions" key at all -- it
# only carries tune_videos/confirm_videos -- and split_guard.py resolves its
# manifest relative to its own directory, so the benchmark_runs copy is the
# only file that has (or can have) an official_test partition. See
# tuning/prerun_fixes/test_split_preflight.md section 5.
BENCHMARK_MANIFEST = (
    REPO / "benchmark_runs" / "paper_setup_20260720T074844Z_1e431b7" / "split_manifest.json"
)
OFFICIAL_TEST_DIR = TUNING_DIR / "official_test"

SPLIT_SPECS: dict[str, dict] = {
    "val_confirm": {
        "questions_csv": REPO / "eval" / "data" / "nextqa" / "val.csv",
        "gold_json": REPO / "eval" / "data" / "nextqa" / "gsub_val.json",
        "stem": "val_confirm_e2e",
        # Historical nominal count, printed in the setup banner and recorded in
        # the metrics block. 113 manifest videos -> 112 usable (one has no mp4).
        "nominal_videos": 113,
    },
    "official_test": {
        "questions_csv": REPO / "eval" / "data" / "nextqa" / "test.csv",
        "gold_json": REPO / "eval" / "data" / "nextqa" / "gsub_test.json",
        "stem": "official_test_e2e",
        "nominal_videos": 990,
    },
}


# Directory resolution is deliberately done through these two functions rather
# than baked into SPLIT_SPECS at import time. The existing test suite
# monkeypatches the module globals TUNING_DIR / INDEX_CACHE_DIR /
# PER_QUESTION_CSV to redirect a run into tmp_path; a frozen dict captured at
# import would silently ignore those patches and write into the real repo.
def out_dir_for(split: str) -> Path:
    return TUNING_DIR if split == "val_confirm" else TUNING_DIR / "official_test"


def index_cache_dir_for(split: str) -> Path:
    return INDEX_CACHE_DIR if split == "val_confirm" else TUNING_DIR / "index_cache_official_test"


def split_video_ids(split: str) -> set[str]:
    """The in-scope video id list for a split. This is one of the three things
    --split controls."""
    if split == "val_confirm":
        manifest = json.loads((REPO / "split_manifest.json").read_text())
        return set(manifest["confirm_videos"])
    if split == "official_test":
        manifest = json.loads(BENCHMARK_MANIFEST.read_text())
        return set(manifest["partitions"]["official_test"]["video_ids"])
    raise ValueError(f"unknown split {split!r}")


def output_paths_for_mode(query_mode: str, traversal_mode: str,
                          split: str = "val_confirm") -> tuple[Path, Path]:
    """The frozen baseline run (query_mode="none") keeps its historical
    filenames; every reformulation arm writes to its own suffixed pair so a
    non-baseline run can never silently overwrite the recorded held-out
    baseline artifacts.

    Signature is backward-compatible: `split` defaults to val_confirm, so the
    two-argument call used everywhere before --split existed returns exactly
    the historical (tuning/val_confirm_e2e_per_question.csv, ..._report.md).
    """
    stem = SPLIT_SPECS[split]["stem"]
    out_dir = out_dir_for(split)
    if query_mode == "none" and traversal_mode == "none":
        # val_confirm's baseline pair is returned from the module globals so a
        # test that patches PER_QUESTION_CSV/REPORT_PATH still takes effect.
        if split == "val_confirm":
            return PER_QUESTION_CSV, REPORT_PATH
        return out_dir / f"{stem}_per_question.csv", out_dir / f"{stem}_report.md"
    suffix = f"_{query_mode}_{traversal_mode}"
    return (
        out_dir / f"{stem}_per_question{suffix}.csv",
        out_dir / f"{stem}_report{suffix}.md",
    )


PER_Q_FIELDNAMES = [
    "video", "qid", "type", "question",
    "pred_answer_idx", "pred_answer_label", "gold_answer_idx", "gold_answer_label",
    "acc_qa", "pred_span_start", "pred_span_end", "gold_spans", "iop", "iou",
    "acc_gqa_unverified", "used_clip_anchor", "raw_answer_nonempty",
    "retrieval_span_ms", "caption_answer_ms",
    # Query-reformulation telemetry (blank/none for the frozen baseline arm).
    "query_mode", "traversal_mode", "relation", "relation_source",
    "fallback_reason", "num_ppr_calls", "context_frame_count",
]


def build_mc_prompt(row: dict) -> str:
    return f"""You are answering a NExT-QA multiple-choice video question.

Use only the provided retrieved frame evidence. Do not use outside knowledge if
the evidence is insufficient.

Question:
{row["question"]}

Options:
A. {row["choices"][0]}
B. {row["choices"][1]}
C. {row["choices"][2]}
D. {row["choices"][3]}
E. {row["choices"][4]}

Return exactly this format:
ANSWER: <A|B|C|D|E>
REASON: <one short sentence grounded in the frame evidence>
"""


def make_e2e_config(frozen: dict, args: argparse.Namespace | None = None) -> IRISConfig:
    """Same construction as part3_tune.py's make_config, but explicit
    about every field the task spec calls out (not silently relying on
    IRISConfig() defaults for anything the task named).

    Query reformulation is opt-in and explicit: with no --query-mode the
    config is byte-identical to the pre-existing frozen baseline
    (query_reformulation_mode="none", temporal_traversal_mode="none"), which
    routes through retrieve_for_question's "none" branch -- the same
    _call_embed_query + _build_retrieved pair this script called directly
    before the flag existed.
    """
    cfg = IRISConfig()
    cfg.cerberus_mode = "none"
    cfg.ranking_mode = frozen.get("ranking_mode", "ppr")
    cfg.codec_conf_source = frozen.get("codec_conf_source", "packet_size")
    cfg.codec_conf_pictype_norm = True
    for key in ("retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
                "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
                "luma_entropy_weight", "persistence_threshold", "max_prominence"):
        setattr(cfg, key, frozen[key])

    if args is not None:
        cfg.query_reformulation_mode = args.query_mode
        cfg.temporal_traversal_mode = args.temporal_traversal_mode
        cfg.temporal_context_seconds = args.temporal_context_seconds
        cfg.temporal_scene_hops = args.temporal_scene_hops
        cfg.max_context_frames = args.max_context_frames
        cfg.multi_query_combine = args.multi_query_combine
        cfg.max_retrieval_queries = (
            min(args.max_queries, 3) if args.query_mode == "structured_v2" else args.max_queries
        )
    return cfg


def ingest_config_hash(cfg: IRISConfig) -> str:
    payload = {k: getattr(cfg, k) for k in INGEST_RELEVANT_KEYS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def load_split_questions(split: str = "val_confirm") -> list[dict]:
    """Load the question list for a split.

    The filter chain, the row->dict mapping, and the iteration order are
    identical for every split -- only the id set, the CSV, and the gold JSON
    differ. Row order is CSV order, exactly as before.
    """
    spec = SPLIT_SPECS[split]
    in_scope = split_video_ids(split)
    rows = list(csv.DictReader(open(spec["questions_csv"], newline="", encoding="utf-8")))
    gsub = json.loads(Path(spec["gold_json"]).read_text())
    out = []
    for r in rows:
        vid = r["video"]
        if vid not in in_scope:
            continue
        qid = r["qid"]
        vpath = VIDEO_DIR / f"{vid}.mp4"
        if not vpath.exists():
            continue
        gold = gsub.get(vid, {}).get("location", {}).get(qid)
        if not gold:
            continue
        out.append({
            "video": vid, "qid": qid, "question": r["question"], "type": r.get("type"),
            "choices": [r["a0"], r["a1"], r["a2"], r["a3"], r["a4"]],
            "gold_answer_idx": int(r["answer"]),
            "gold_spans": gold, "duration": gsub[vid]["duration"],
        })
    return out


def load_val_confirm_questions() -> list[dict]:
    """Backward-compatible alias kept so every pre---split import site and the
    reproduction test can call the historical name and get byte-identical
    output."""
    return load_split_questions("val_confirm")


DISK_ABORT_PREFLIGHT_GB = 5.0
DISK_ABORT_MIDRUN_GB = 2.0
DISK_RECHECK_EVERY = 50


class DiskSpaceAbort(RuntimeError):
    """Raised to unwind cleanly (not crash) when free space drops below the
    mid-run floor. Indexes already written stay on disk, so --resume picks up
    exactly where this left off."""


def free_gb(path: Path) -> float:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / 1e9


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def ensure_indexes_e2e(video_ids: list[str], cfg: IRISConfig, n_workers: int = 8,
                       cache_dir: Path | None = None) -> tuple[dict[str, str], int, int]:
    """Same shape as part3_tune.ensure_indexes but pointed at a per-split cache
    dir. Returns (paths, n_fresh_ingests, n_cache_hits).

    `cache_dir` defaults to the historical val_confirm_e2e dir so every
    pre---split call site behaves identically.

    Disk is checked before the first ingest (abort under 5 GB) and every 50
    completions (clean abort under 2 GB). The root filesystem on worker-1 runs
    at 98% -- this is a live risk, not a formality.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else INDEX_CACHE_DIR
    h = ingest_config_hash(cfg)
    paths = {}
    todo = []
    n_cache_hits = 0
    for vid in video_ids:
        p = cache_dir / f"{vid}__{h}"
        if p.with_suffix(p.suffix + ".npz").exists():
            paths[vid] = str(p)
            n_cache_hits += 1
        else:
            todo.append(vid)

    if todo:
        avail = free_gb(cache_dir)
        print(f"[ingest] {len(todo)}/{len(video_ids)} videos need ingest under config-hash {h}", flush=True)
        print(f"[disk] {avail:.1f} GB free at {cache_dir}", flush=True)
        if avail < DISK_ABORT_PREFLIGHT_GB:
            raise SystemExit(
                f"[disk] ABORT before ingest: {avail:.1f} GB free at {cache_dir}, "
                f"below the {DISK_ABORT_PREFLIGHT_GB:.0f} GB pre-flight floor. "
                "Free space and rerun with --resume."
            )

        def _do(vid: str) -> tuple[str, float]:
            t0 = time.perf_counter()
            vpath = VIDEO_DIR / f"{vid}.mp4"
            idx = iris_ingest.ingest(str(vpath), cfg)
            out_path = cache_dir / f"{vid}__{h}"
            iris_ingest.save_index(idx, str(out_path))
            return vid, time.perf_counter() - t0

        done = 0
        t_ingest0 = time.perf_counter()
        aborted = False
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_do, vid): vid for vid in todo}
            try:
                for fut in as_completed(futs):
                    vid = futs[fut]
                    try:
                        _, secs = fut.result()
                        paths[vid] = str(cache_dir / f"{vid}__{h}")
                    except Exception as exc:  # noqa: BLE001
                        secs = float("nan")
                        print(f"[ingest FAIL] {vid}: {type(exc).__name__}: {exc}", flush=True)
                    done += 1
                    elapsed = time.perf_counter() - t_ingest0
                    rate = elapsed / done
                    remaining = rate * (len(todo) - done)
                    print(
                        f"[ingest {done}/{len(todo)}] video={vid} ingest_s={secs:.2f} "
                        f"elapsed={_fmt_hms(elapsed)} projected_remaining={_fmt_hms(remaining)}",
                        flush=True,
                    )
                    if done % DISK_RECHECK_EVERY == 0:
                        avail = free_gb(cache_dir)
                        print(f"[disk] {avail:.1f} GB free after {done} ingests", flush=True)
                        if avail < DISK_ABORT_MIDRUN_GB:
                            print(
                                f"[disk] ABORT: {avail:.1f} GB free, below the "
                                f"{DISK_ABORT_MIDRUN_GB:.0f} GB mid-run floor. Cancelling "
                                "queued ingests; completed indexes are on disk, rerun "
                                "with --resume after freeing space.",
                                flush=True,
                            )
                            aborted = True
                            break
            finally:
                if aborted:
                    for f in futs:
                        f.cancel()
        if aborted:
            raise DiskSpaceAbort(
                f"free space fell below {DISK_ABORT_MIDRUN_GB:.0f} GB after {done} ingests"
            )
    return paths, len(todo), n_cache_hits


def smoke_test_backend(cfg: IRISConfig) -> str:
    """Confirm the answerer backend is reachable before starting the full
    loop, per the task's explicit requirement not to discover this
    partway through."""
    raw = aria.generate(
        prompt="Return exactly this format:\nANSWER: <A|B|C|D|E>\nREASON: one sentence.\n\n"
               "Question: What color is the sky on a clear day?\nOptions:\nA. red\nB. blue\nC. green\nD. purple\nE. black",
        context="No frame evidence provided for this smoke test.",
        config=cfg,
    )
    if not raw or not raw.strip():
        raise RuntimeError("Smoke test: backend returned empty response")
    return raw


REQUIRED_SAMPLER = {
    "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": False,
}
REQUIRED_ALIAS = "granite4:micro"


def assert_models_endpoint(cfg: IRISConfig) -> dict:
    """GET /v1/models and assert the served alias is granite4:micro.

    Also refuses Ollama's port outright: llama-server silently accepts and
    ignores cache_prompt when the request lands on Ollama's OpenAI-compat
    endpoint, which would break determinism with no visible failure. The
    official run serves only via serve-granite-pinned.sh on 8091.
    """
    import requests

    endpoint = cfg.answerer_endpoint
    if "11434" in endpoint:
        raise SystemExit(
            f"[serving] ABORT: answerer_endpoint={endpoint!r} is Ollama's port. "
            "The official run must be served by /home/ccbd/.local/iris/bin/"
            "serve-granite-pinned.sh on 8091. Never fall back to Ollama."
        )
    resp = requests.get(f"{endpoint}/models", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("data") or data.get("models") or []
    names = [e.get("id") or e.get("name") or e.get("model") for e in entries]
    if REQUIRED_ALIAS not in names:
        raise SystemExit(
            f"[serving] ABORT: /v1/models does not advertise {REQUIRED_ALIAS!r}. "
            f"Got {names!r}. Refusing to score against an unidentified model."
        )
    print(f"[serving] /v1/models OK -- alias {REQUIRED_ALIAS!r} present (advertised: {names!r})",
          flush=True)
    return data


def capture_outgoing_sampler_payload(cfg: IRISConfig) -> dict:
    """Send ONE real answerer request and read the sampler fields off the wire.

    This deliberately does not inspect iris/aria.py's source or the config --
    it hooks httpx.Client.send (the transport the OpenAI SDK actually uses) and
    reads the serialised request body, so what is asserted is what the server
    genuinely received. A source-level check would pass even if a wrapper,
    default, or SDK version quietly dropped a field.
    """
    import httpx

    captured: dict = {}
    original_send = httpx.Client.send

    def _spy(self, request, *a, **kw):
        try:
            if b"chat/completions" in request.url.raw_path or b"completion" in request.url.raw_path:
                captured.setdefault("url", str(request.url))
                captured.setdefault("body", json.loads(request.content.decode("utf-8")))
        except Exception:  # noqa: BLE001 -- never let instrumentation break the call
            pass
        return original_send(self, request, *a, **kw)

    httpx.Client.send = _spy
    try:
        raw = smoke_test_backend(cfg)
    finally:
        httpx.Client.send = original_send

    if not captured.get("body"):
        raise SystemExit(
            "[serving] ABORT: could not capture an outgoing request body. The "
            "sampler contract cannot be verified from source alone, so this run "
            "must not proceed."
        )

    body = captured["body"]
    # llama-server accepts the llama.cpp-specific fields either at top level or
    # nested under extra_body depending on how the SDK serialises them; check
    # both so a passing result means the field really is on the wire.
    def _wire_get(key):
        if key in body:
            return body[key]
        eb = body.get("extra_body") or {}
        return eb.get(key, "<ABSENT>")

    actual = {k: _wire_get(k) for k in REQUIRED_SAMPLER}
    mismatches = {
        k: {"expected": v, "actual": actual[k]}
        for k, v in REQUIRED_SAMPLER.items()
        if actual[k] != v
    }
    if mismatches:
        raise SystemExit(
            "[serving] ABORT: outgoing sampler payload does not match the "
            f"determinism contract. Mismatches: {json.dumps(mismatches, indent=2)}\n"
            f"Full captured body keys: {sorted(body.keys())}"
        )
    print(f"[serving] outgoing sampler payload verified on the wire: {actual}", flush=True)
    return {"url": captured["url"], "sampler_on_wire": actual,
            "smoke_response": raw[:200], "captured_body_keys": sorted(body.keys())}


def assert_split_guard(split: str, video_ids: list[str]) -> dict:
    """Wire the previously-dead split_guard.py.

    Two assertions, both required by the task spec:
      1. the official_test partition is populated (guard_official_test_command);
      2. no val_tune or val_confirm video id appears in the loaded test id list.

    split_guard resolves its manifest relative to its own directory and
    therefore reads the benchmark_runs copy. That is correct and intended --
    see tuning/prerun_fixes/test_split_preflight.md section 5 -- so its path
    resolution is left untouched.
    """
    if split != "official_test":
        return {"guard": "not applicable for split=val_confirm"}

    guard_dir = BENCHMARK_MANIFEST.parent / "scripts"
    sys.path.insert(0, str(guard_dir))
    import split_guard  # noqa: E402

    print(f"[guard] split_guard manifest: {Path(split_guard.MANIFEST_PATH).resolve()}", flush=True)
    split_guard.guard_official_test_command("official_test")
    print("[guard] guard_official_test_command('official_test') returned cleanly", flush=True)

    root = json.loads((REPO / "split_manifest.json").read_text())
    tuning_ids = set(root["tune_videos"]) | set(root["confirm_videos"])
    leaked = sorted(set(video_ids) & tuning_ids)
    if leaked:
        raise SystemExit(
            f"[guard] ABORT: {len(leaked)} tuning video id(s) present in the "
            f"official_test id list: {leaked[:10]}{'...' if len(leaked) > 10 else ''}. "
            "This would contaminate the held-out test split."
        )
    print(f"[guard] leakage check OK -- 0 of {len(video_ids)} test videos appear in "
          f"val_tune or val_confirm ({len(tuning_ids)} tuning ids checked)", flush=True)
    return {
        "manifest_read": str(Path(split_guard.MANIFEST_PATH).resolve()),
        "guard_official_test_command": "returned cleanly",
        "tuning_ids_checked": len(tuning_ids),
        "leaked_ids": leaked,
    }


def load_completed_qids(csv_path: Path) -> set[tuple[str, str]]:
    """(video, qid) pairs already present in a partially-written output CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["video"], r["qid"]) for r in csv.DictReader(f)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--query-mode", dest="query_mode", default="none",
        choices=["none", "legacy", "structured_v2"],
        help="Retrieval query mode, routed through "
             "iris.retrieval_entry.retrieve_for_question. 'none' (default) is "
             "the frozen verbatim-question baseline.",
    )
    p.add_argument(
        "--temporal-traversal-mode", dest="temporal_traversal_mode", default="none",
        choices=["none", "legacy_symmetric", "directional"],
        help="Temporal context expansion mode (default: none).",
    )
    p.add_argument("--temporal-context-seconds", dest="temporal_context_seconds", type=float, default=4.0)
    p.add_argument("--temporal-scene-hops", dest="temporal_scene_hops", type=int, default=1)
    p.add_argument("--max-context-frames", dest="max_context_frames", type=int, default=8)
    p.add_argument("--multi-query-combine", dest="multi_query_combine", default="weighted_max",
                   choices=["weighted_max", "logsumexp"])
    p.add_argument("--max-queries", dest="max_queries", type=int, default=3,
                   help="Retrieval query budget (hard-capped to 3 under structured_v2).")
    p.add_argument(
        "--caption-dump", dest="caption_dump", default=None,
        help="Write every generated caption to this path as "
             "{video: {qid: {frame_idx: caption}}} (same schema as "
             "tuning/blind_ablation/captions_dump.json). Opt-in; omitting "
             "this flag leaves behaviour unchanged.",
    )
    p.add_argument(
        "--caption-load", dest="caption_load", default=None,
        help="Load captions from a dump written by --caption-dump and skip "
             "captioning entirely. Errors loudly on any (video, qid, "
             "frame_idx) miss rather than silently captioning it live. "
             "Opt-in; omitting this flag leaves behaviour unchanged.",
    )
    p.add_argument(
        "--split", dest="split", default="val_confirm",
        choices=["val_confirm", "official_test"],
        help="Which split to evaluate. Controls exactly three things: the video "
             "id list, the questions CSV, and the gold-span JSON. Everything "
             "else -- retrieval config, span method, scorer, parser -- is "
             "identical. Default val_confirm reproduces the historical run.",
    )
    p.add_argument(
        "--resume", dest="resume", action="store_true",
        help="Skip any question already present in the output CSV and any video "
             "already ingested in the split's index cache. Without this flag an "
             "existing output CSV is refused rather than overwritten.",
    )
    p.add_argument(
        "--limit-videos", dest="limit_videos", type=int, default=None,
        help="Evaluate only the first N video ids by sort order. Plumbing/smoke "
             "validation only -- accuracy from a truncated run is NOT a result.",
    )
    p.add_argument(
        "--repeat", dest="repeat", type=int, default=1,
        help="Run the answer stage this many times over identical retrieval "
             "and (with --caption-load) identical captions, reporting "
             "mean/min/max/flip-count Acc@QA and Acc@GQA instead of a single "
             "point estimate. Default 1 preserves current single-run "
             "behaviour exactly.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    split = args.split
    spec = SPLIT_SPECS[split]
    index_cache_dir = index_cache_dir_for(split)
    out_dir = out_dir_for(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_question_csv, _report_path = output_paths_for_mode(
        args.query_mode, args.temporal_traversal_mode, split
    )
    if per_question_csv.exists() and not args.resume:
        raise SystemExit(
            f"[setup] {per_question_csv} already exists -- refusing to overwrite a "
            "recorded held-out artifact. Move or delete it first if this rerun is "
            "intended, or pass --resume to continue it."
        )
    completed = load_completed_qids(per_question_csv) if args.resume else set()
    if args.resume:
        print(f"[resume] {len(completed)} question(s) already present in "
              f"{per_question_csv.name} -- these will be skipped", flush=True)
    print(f"[setup] split={split} query_mode={args.query_mode} "
          f"temporal_traversal_mode={args.temporal_traversal_mode} -> {per_question_csv.name}", flush=True)

    state = load_frozen_state()
    frozen = state["frozen"]
    print(f"[setup] frozen hyperparameters read live from tuning/frozen_state.json: {frozen}", flush=True)

    required_keys = ["retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
                      "span_method", "span_method_half_width_s", "peak_distance", "peak_prominence",
                      "packet_size_weight", "motion_weight", "luma_entropy_weight",
                      "persistence_threshold", "max_prominence"]
    missing = [k for k in required_keys if k not in frozen]
    if missing:
        raise SystemExit(f"[setup] frozen_state.json missing expected keys: {missing}")

    # A fresh (non-resume) run must not inherit a prior cache. Under --resume
    # reusing the split's own cache is the entire point, so the emptiness guard
    # is lifted -- but only for that split's dedicated dir.
    if index_cache_dir.exists() and any(index_cache_dir.iterdir()) and not args.resume:
        raise SystemExit(f"[setup] {index_cache_dir} already exists and is non-empty -- "
                          "this run must not reuse any prior cache. Aborting "
                          "(pass --resume if continuing an interrupted run).")
    index_cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_e2e_config(frozen, args)
    print(f"[setup] config: cerberus_mode={cfg.cerberus_mode} ranking_mode={cfg.ranking_mode} "
          f"codec_conf_source={cfg.codec_conf_source} codec_conf_pictype_norm={cfg.codec_conf_pictype_norm} "
          f"answerer_backend={cfg.answerer_backend} answerer_endpoint={cfg.answerer_endpoint} "
          f"answerer_model={cfg.answerer_model}", flush=True)

    # Serving contract. The hard /v1/models + on-the-wire sampler assertions are
    # part of the OFFICIAL-RUN protocol specifically; val_confirm keeps its
    # historical single smoke call so that --split=val_confirm stays
    # behaviourally identical to every invocation that predates --split (it must
    # not acquire a new hard network precondition).
    if split == "official_test":
        models_response = assert_models_endpoint(cfg)
        wire_check = capture_outgoing_sampler_payload(cfg)
        smoke_raw = wire_check["smoke_response"]
        sampler_recorded = wire_check["sampler_on_wire"]
    else:
        print("[setup] smoke-testing answerer backend before the full loop...", flush=True)
        models_response = None
        wire_check = None
        smoke_raw = smoke_test_backend(cfg)
        sampler_recorded = {
            "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "seed": cfg.answerer_seed, "cache_prompt": False,
        }
    print(f"[setup] smoke test OK, backend reachable. Raw response: {smoke_raw[:200]!r}", flush=True)

    from urllib.parse import urlparse
    parsed_endpoint = urlparse(cfg.answerer_endpoint)
    provenance = capture_answerer_provenance(
        endpoint=cfg.answerer_endpoint,
        port=parsed_endpoint.port or 8091,
        gguf_path=None,
        gguf_expected_sha256=None,
        # For official_test this is recorded from the wire, not re-derived from
        # config -- what the server actually received on a real request.
        sampler_params=sampler_recorded,
    )
    env_suffix = (
        "" if (args.query_mode == "none" and args.temporal_traversal_mode == "none")
        else f"_{args.query_mode}_{args.temporal_traversal_mode}"
    )
    env_path = out_dir / f"{spec['stem']}_environment{env_suffix}.json"

    half_width_s = float(frozen["span_method_half_width_s"])
    assert frozen["span_method"] == "D", f"expected span_method=D, got {frozen['span_method']!r}"

    # Routed through the historical alias for val_confirm so the existing test
    # suite's monkeypatch of load_val_confirm_questions still intercepts it.
    # The alias delegates to load_split_questions("val_confirm"), so the two
    # paths are the same code.
    questions = (
        load_val_confirm_questions() if split == "val_confirm"
        else load_split_questions(split)
    )
    video_ids = sorted({q["video"] for q in questions})
    guard_result = assert_split_guard(split, video_ids)

    if args.limit_videos is not None:
        kept = set(video_ids[: args.limit_videos])
        questions = [q for q in questions if q["video"] in kept]
        video_ids = sorted(kept)
        print(f"[setup] --limit-videos={args.limit_videos} -> {len(video_ids)} videos, "
              f"{len(questions)} questions. PLUMBING CHECK ONLY, not a result.", flush=True)

    print(f"[setup] {split} questions loaded: {len(questions)} usable "
          f"(nominal {spec['nominal_videos']} videos) across {len(video_ids)} usable videos", flush=True)

    env_path.write_text(json.dumps({
        "split": split,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "frozen": frozen,
        "answerer_provenance": provenance,
        "models_endpoint_response": models_response,
        "sampler_wire_check": wire_check,
        "split_guard": guard_result,
        "index_cache_dir": str(index_cache_dir),
        "limit_videos": args.limit_videos,
        "resume": args.resume,
        "n_questions": len(questions),
        "n_videos": len(video_ids),
    }, indent=2))
    print(f"[setup] wrote environment/provenance to {env_path}", flush=True)

    # Called with the historical two-argument signature for val_confirm; the
    # existing test suite patches this symbol with a 2-arg lambda, and the
    # default cache_dir is INDEX_CACHE_DIR anyway, so this is the same call.
    if split == "val_confirm":
        index_paths, n_fresh, n_hits = ensure_indexes_e2e(video_ids, cfg)
    else:
        index_paths, n_fresh, n_hits = ensure_indexes_e2e(video_ids, cfg, cache_dir=index_cache_dir)
    fresh_note = (
        "ALL FRESH -- OK" if n_hits == 0
        else ("resumed from cache" if args.resume
              else "WARNING: cache hits found in a supposedly-fresh dir")
    )
    print(f"[ingest] fresh_ingests={n_fresh} cache_hits={n_hits} ({fresh_note})", flush=True)

    if args.repeat > 1 and not args.caption_load:
        raise SystemExit(
            "[setup] --repeat > 1 requires --caption-load so the caption stage is "
            "frozen and only the answerer varies across repeats -- otherwise a "
            "flip could come from either the captioner or the answerer and "
            "would be unattributable."
        )
    caption_load_dump = load_caption_dump(args.caption_load) if args.caption_load else None
    caption_dump_accumulator: dict = {} if args.caption_dump else None

    index_cache: dict = {}
    retrieval_ms_list = []
    n_answer_nonempty_sample = 0
    records: list[dict] = []  # one entry per successfully-retrieved-and-captioned question

    n_repeats = max(args.repeat, 1)
    # Streaming mode (the single-pass run mode, including the official test
    # run): each question is retrieved, captioned, answered, scored, and
    # flushed to disk before the next one starts. This is what makes --resume
    # meaningful over a 4-6 hour run -- a kill at hour 3 costs one question,
    # not three hours of captioning. --repeat > 1 keeps the batched two-phase
    # shape, which it needs in order to re-answer an identical record set.
    #
    # Interleaving does not change any value: context_text is already
    # finalised per question at caption time in the loop below, and the
    # answerer is stateless per request (temperature=0, seed=42,
    # cache_prompt=false), so moving its call earlier cannot alter its input
    # or its output.
    streaming = n_repeats == 1

    def _answer_and_score_one(rec: dict) -> dict:
        prompt = build_mc_prompt(rec)
        t_ans = time.perf_counter()
        raw_answer = aria.generate(prompt=prompt, context=rec["context_text"], config=cfg)
        answer_ms = (time.perf_counter() - t_ans) * 1000
        pred_idx = parse_mc_answer(raw_answer)
        acc_qa = nextgqa_metrics.acc_qa(pred_idx, rec["gold_answer_idx"])
        acc_gqa = bool(acc_qa and rec["iop"] >= 0.5)
        return {
            "video": rec["video"], "qid": rec["qid"], "type": rec["type"], "question": rec["question"],
            "pred_answer_idx": pred_idx, "pred_answer_label": format_mc_label(pred_idx),
            "gold_answer_idx": rec["gold_answer_idx"], "gold_answer_label": format_mc_label(rec["gold_answer_idx"]),
            "acc_qa": acc_qa, "pred_span_start": round(rec["pred_span"][0], 3), "pred_span_end": round(rec["pred_span"][1], 3),
            "gold_spans": json.dumps(rec["gold_spans"]), "iop": round(rec["iop"], 5), "iou": round(rec["iou"], 5),
            "acc_gqa_unverified": acc_gqa, "used_clip_anchor": rec["used_clip_anchor"],
            "raw_answer_nonempty": bool(raw_answer and raw_answer.strip()),
            "retrieval_span_ms": round(rec["retrieval_span_ms"], 2),
            "caption_answer_ms": round(rec["caption_ms"] + answer_ms, 2),
            "query_mode": rec["query_mode"], "traversal_mode": rec["traversal_mode"],
            "relation": rec["relation"], "relation_source": rec["relation_source"],
            "fallback_reason": rec["fallback_reason"], "num_ppr_calls": rec["num_ppr_calls"],
            "context_frame_count": rec["context_frame_count"],
            "_answer_ms": answer_ms, "_raw_answer_nonempty": bool(raw_answer and raw_answer.strip()),
        }

    streamed_rows: list[dict] = []
    append_mode = per_question_csv.exists() and args.resume
    stream_fh = open(per_question_csv, "a" if append_mode else "w", newline="") if streaming else None
    stream_writer = None
    if stream_fh is not None:
        stream_writer = csv.DictWriter(stream_fh, fieldnames=PER_Q_FIELDNAMES)
        if not append_mode:
            stream_writer.writeheader()
            stream_fh.flush()

    t_run_start = time.perf_counter()
    n_resume_skipped = 0
    for i, q in enumerate(questions, 1):
        vid = q["video"]
        if (vid, q["qid"]) in completed:
            n_resume_skipped += 1
            continue
        if vid not in index_paths:
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(index_paths[vid])
        index = index_cache[vid]

        t0 = time.perf_counter()
        try:
            retrieved_frames, plan, telemetry = retrieve_for_question(
                q["question"], index, cfg,
                type_code=q.get("type"), family=q.get("family"),
            )
            # Span construction (frozen Method D) is deliberately anchored on
            # the VERBATIM question embedding in every query mode. Method D is
            # a frozen span parameter, not part of the variable under test, so
            # letting the anchoring embedding change with query_mode would vary
            # two things at once and make an mIoP delta unattributable.
            query_embedding, _ = iris_query._call_embed_query(q["question"], cfg)
            pred_span, used_clip_anchor = predicted_span_from_frames_peak(
                retrieved_frames, query_embedding, half_width_s=half_width_s,
                duration_s=q["duration"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL retrieval] video={vid} qid={q['qid']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        t_retrieval_span = (time.perf_counter() - t0) * 1000
        retrieval_ms_list.append(t_retrieval_span)

        t1 = time.perf_counter()
        if caption_load_dump is not None:
            apply_caption_load(index, retrieved_frames, vid, q["qid"], caption_load_dump)
        else:
            try:
                iris_query._ensure_captions(index, retrieved_frames, cfg)
            except TypeError:
                iris_query._ensure_captions(index, retrieved_frames)
        if caption_dump_accumulator is not None:
            record_caption_dump(retrieved_frames, vid, q["qid"], caption_dump_accumulator)
        cache_obj = iris_query.wrapper_init_l1_cache(cfg)
        iris_query.wrapper_populate_cache(cache_obj, retrieved_frames)
        context_text = cache_obj.as_context_text()
        t_caption_ms = (time.perf_counter() - t1) * 1000

        gold_idx = q["gold_answer_idx"]
        gold_tuples = [(g[0], g[1]) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)

        rec = {
            "video": vid, "qid": q["qid"], "type": q.get("type"), "question": q["question"],
            "choices": q["choices"],
            "gold_answer_idx": gold_idx, "gold_spans": q["gold_spans"],
            "pred_span": pred_span, "used_clip_anchor": used_clip_anchor,
            "iop": iop, "iou": iou, "context_text": context_text,
            "retrieval_span_ms": t_retrieval_span, "caption_ms": t_caption_ms,
            "query_mode": cfg.query_reformulation_mode, "traversal_mode": cfg.temporal_traversal_mode,
            "relation": getattr(plan, "relation", None), "relation_source": getattr(plan, "relation_source", None),
            "fallback_reason": getattr(plan, "fallback_reason", None),
            "num_ppr_calls": telemetry.get("num_ppr_calls"), "context_frame_count": len(retrieved_frames),
        }
        records.append(rec)

        if streaming:
            row = _answer_and_score_one(rec)
            streamed_rows.append(row)
            stream_writer.writerow({k: row[k] for k in PER_Q_FIELDNAMES})
            stream_fh.flush()
            os.fsync(stream_fh.fileno())
            # context_text is the only large field and it is no longer needed
            # once the row is on disk; drop it so a 5553-question run does not
            # hold every caption block in memory for hours.
            rec["context_text"] = None

        if i % 25 == 0 or i == len(questions):
            done_total = len(completed) + len(streamed_rows) if streaming else len(records)
            elapsed = time.perf_counter() - t_run_start
            rate = elapsed / max(1, len(records))
            remaining = rate * (len(questions) - n_resume_skipped - len(records))
            print(f"[{i}/{len(questions)}] done={done_total} elapsed={_fmt_hms(elapsed)} "
                  f"projected_remaining={_fmt_hms(remaining)}", flush=True)

    if stream_fh is not None:
        stream_fh.close()

    if caption_dump_accumulator is not None:
        write_caption_dump(args.caption_dump, caption_dump_accumulator)
        print(f"[setup] wrote caption dump to {args.caption_dump}", flush=True)

    def _write_pass_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PER_Q_FIELDNAMES)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in PER_Q_FIELDNAMES})

    all_passes: list[list[dict]] = []

    if streaming:
        # Rows were already answered, scored, and flushed inside the loop above.
        all_passes.append(streamed_rows)
        n_scored = len(streamed_rows)
        correct_qa = sum(r["acc_qa"] for r in streamed_rows)
        correct_gqa = sum(r["acc_gqa_unverified"] for r in streamed_rows)
        n_answer_nonempty_sample += sum(1 for r in streamed_rows if r["_raw_answer_nonempty"])
        print(f"[repeat 1/1] scored={n_scored} "
              f"Acc@QA={correct_qa / n_scored if n_scored else 0.0:.4f} "
              f"Acc@GQA={correct_gqa / n_scored if n_scored else 0.0:.4f}", flush=True)
    else:
        for rep in range(n_repeats):
            pass_rows = [_answer_and_score_one(rec) for rec in records]
            all_passes.append(pass_rows)
            n_scored = len(pass_rows)
            correct_qa = sum(r["acc_qa"] for r in pass_rows)
            correct_gqa = sum(r["acc_gqa_unverified"] for r in pass_rows)
            n_answer_nonempty_sample += sum(1 for r in pass_rows if r["_raw_answer_nonempty"])
            print(f"[repeat {rep + 1}/{n_repeats}] scored={n_scored} "
                  f"Acc@QA={correct_qa / n_scored if n_scored else 0.0:.4f} "
                  f"Acc@GQA={correct_gqa / n_scored if n_scored else 0.0:.4f}", flush=True)
            pass_csv_path = per_question_csv.with_name(
                per_question_csv.stem + f"_repeat{rep + 1}" + per_question_csv.suffix
            )
            _write_pass_csv(pass_csv_path, pass_rows)

    total_wall_s = time.perf_counter() - t_run_start

    # Under --resume, `records` holds only the questions scored in THIS
    # invocation, so aggregates must come from the full on-disk CSV or a
    # resumed run would report metrics over a fraction of the split.
    resumed_rows: list[dict] | None = None
    if args.resume and completed:
        with open(per_question_csv, newline="") as f:
            resumed_rows = list(csv.DictReader(f))
        n = len(resumed_rows)
        iops = [float(r["iop"]) for r in resumed_rows]
        ious = [float(r["iou"]) for r in resumed_rows]
        caption_answer_ms_list = [float(r["caption_answer_ms"]) for r in resumed_rows]
        print(f"[resume] aggregates recomputed over the full CSV: {n} questions "
              f"({len(completed)} resumed + {len(records)} scored now)", flush=True)
    else:
        n = len(records)
        iops = [r["iop"] for r in records]
        ious = [r["iou"] for r in records]
        caption_answer_ms_list = [row["caption_answer_ms"] for row in all_passes[0]] if all_passes else []

    # Grounding metrics are deterministic (retrieval + span construction do
    # not vary across repeats) -- reported once, not once per repeat.
    metrics = {
        "split": split,
        "n_scored": n,
        "n_repeats": n_repeats,
        "n_nominal_videos": spec["nominal_videos"],
        "n_usable_videos": len(video_ids),
        "n_resume_skipped": n_resume_skipped,
        "n_fresh_ingests": n_fresh,
        "n_cache_hits": n_hits,
        "mIoP": sum(iops) / n if n else 0.0,
        "mIoU": sum(ious) / n if n else 0.0,
        "IoP@0.3": sum(1 for x in iops if x >= 0.3) / n if n else 0.0,
        "IoP@0.5": sum(1 for x in iops if x >= 0.5) / n if n else 0.0,
        "IoU@0.3": sum(1 for x in ious if x >= 0.3) / n if n else 0.0,
        "IoU@0.5": sum(1 for x in ious if x >= 0.5) / n if n else 0.0,
        "median_retrieval_span_ms": statistics.median(retrieval_ms_list) if retrieval_ms_list else 0.0,
        "p95_retrieval_span_ms": (statistics.quantiles(retrieval_ms_list, n=20)[18] if len(retrieval_ms_list) >= 20 else max(retrieval_ms_list, default=0.0)),
        "median_caption_answer_ms": statistics.median(caption_answer_ms_list) if caption_answer_ms_list else 0.0,
        "p95_caption_answer_ms": (statistics.quantiles(caption_answer_ms_list, n=20)[18] if len(caption_answer_ms_list) >= 20 else max(caption_answer_ms_list, default=0.0)),
        "total_wall_s": total_wall_s,
        "minicpm_truncation_stats": aria.get_minicpm_truncation_stats(),
    }

    if n_repeats == 1:
        if resumed_rows is not None:
            correct_qa = sum(r["acc_qa"] == "True" for r in resumed_rows)
            correct_gqa = sum(r["acc_gqa_unverified"] == "True" for r in resumed_rows)
        else:
            pass_rows = all_passes[0] if all_passes else []
            correct_qa = sum(r["acc_qa"] for r in pass_rows)
            correct_gqa = sum(r["acc_gqa_unverified"] for r in pass_rows)
        metrics["Acc@QA"] = correct_qa / n if n else 0.0
        metrics["Acc@GQA_unverified"] = correct_gqa / n if n else 0.0
    else:
        acc_qa_per_rep = [sum(r["acc_qa"] for r in p) / n if n else 0.0 for p in all_passes]
        acc_gqa_per_rep = [sum(r["acc_gqa_unverified"] for r in p) / n if n else 0.0 for p in all_passes]

        def _flip_count(key: str) -> int:
            flips = 0
            for qidx in range(n):
                values = {all_passes[rep][qidx][key] for rep in range(n_repeats)}
                if len(values) > 1:
                    flips += 1
            return flips

        repeat_summary = {
            "n_repeats": n_repeats,
            "Acc@QA": {
                "mean": statistics.mean(acc_qa_per_rep), "min": min(acc_qa_per_rep), "max": max(acc_qa_per_rep),
                "per_repeat": acc_qa_per_rep, "n_flipped_questions": _flip_count("acc_qa"),
            },
            "Acc@GQA_unverified": {
                "mean": statistics.mean(acc_gqa_per_rep), "min": min(acc_gqa_per_rep), "max": max(acc_gqa_per_rep),
                "per_repeat": acc_gqa_per_rep, "n_flipped_questions": _flip_count("acc_gqa_unverified"),
            },
            "grounding_metrics_reported_once": {
                "mIoP": metrics["mIoP"], "mIoU": metrics["mIoU"],
                "IoP@0.5": metrics["IoP@0.5"], "IoU@0.5": metrics["IoU@0.5"],
            },
        }
        env_suffix_for_summary = (
            "" if (args.query_mode == "none" and args.temporal_traversal_mode == "none")
            else f"_{args.query_mode}_{args.temporal_traversal_mode}"
        )
        summary_path = out_dir / f"{spec['stem']}_repeat_summary{env_suffix_for_summary}.json"
        summary_path.write_text(json.dumps(repeat_summary, indent=2))
        print(f"[repeat] wrote variance summary to {summary_path}", flush=True)
        metrics["repeat_summary"] = repeat_summary

    metrics_path = out_dir / f"{spec['stem']}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Historical stdout contract preserved verbatim for val_confirm so any
    # existing log scraper keeps working; the test split gets its own marker.
    marker = "VAL_CONFIRM_E2E" if split == "val_confirm" else "OFFICIAL_TEST_E2E"
    print(f"{marker}_METRICS_JSON=" + json.dumps(metrics), flush=True)
    print(f"{marker}_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except DiskSpaceAbort as exc:
        # Clean, non-crashing exit: everything scored so far is already flushed
        # to the output CSV and every completed index is on disk.
        print(f"[abort] {exc}", flush=True)
        print("[abort] Rerun with --resume after freeing space.", flush=True)
        raise SystemExit(4)
