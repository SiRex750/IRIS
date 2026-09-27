"""Official NExT-GQA test-set Acc@GQA: scene-sparse complete-block (S) vs flat (F) vs uniform (U).

Implements eval_results/NEXTGQA_official_test_prereg.md (registered at commit 546d392). Every guard
in that document is enforced here; deviations found while writing this script are recorded in the
output's "deviations" block rather than by editing the registration.

Modes
  --smoke N   Pipeline check on the first N test videos. Reads NO gold: never opens gsub_test.json,
              never reads the answer column, scores nothing. Uses a scratch cache and scratch output.
              Registration/clean-tree guards are skipped. Not a touch of the split.
  (default)   The registered run. All guards enforced. Resumable: re-running continues from the
              checkpoint; completed question rows are never recomputed.

Environment the run requires
  - llama-server (determinism-gated build, --parallel 1) serving granite4:micro at the
    IRISConfig answerer_endpoint.
  - An Ollama server with a minicpm-v model: frame captioning is routed through
    aria.get_captioner(), which resolves from the shipped config (ConfigManager), NOT from the
    captioner_backend field of the IRISConfig passed to ingest.

Example
  python scripts/nextgqa_official_test_run.py --test-csv <official test.csv> \
      --gsub-test <official gsub_test.json> --videos-dir <dir of test videos> \
      --annotation-source "doc-doc/NExT-GQA@<commit>" --server-build b10099 --model-sha256 <gguf sha256>
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.aria as aria  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.iris_config import IRISConfig  # noqa: E402
from iris.query import _embed_query, _build_retrieved, _ensure_captions  # noqa: E402
from eval.mc_scorer import build_mc_prompt, parse_mc_answer, LETTERS  # noqa: E402
from eval.span import predict_span  # noqa: E402
from eval.metrics_official import best_over_gold_spans  # noqa: E402  (official max-over-gold-spans)
from scripts.pillar2_grounded_qa import (  # noqa: E402
    iop as iop_union, uniform_ts, preflight_backend, config_hash,
)

PREREG = REPO / "eval_results" / "NEXTGQA_official_test_prereg.md"
PREREG_COMMIT_PREFIX = "546d392"
OUT_RAW = REPO / "eval_results" / "NEXTGQA_official_test_raw.json"
OUT_MD = REPO / "eval_results" / "NEXTGQA_official_test_result.md"
CHECKPOINT = REPO / "eval_results" / "_nextgqa_official_test_checkpoint.jsonl"
CACHE_DIR = REPO / "eval" / "data" / "nextgqa_official_test" / "index_cache"
SMOKE_DIR = REPO / "eval_results" / "_smoke_nextgqa_official"
VAL_CSV = REPO / "eval" / "data" / "nextqa" / "val.csv"
GSUB_VAL = REPO / "eval" / "data" / "nextqa" / "gsub_val.json"
OUR_VAL_TUNING = REPO / "eval_results" / "val_videos.txt"
OUR_VAL_HELDOUT = REPO / "eval_results" / "test_videos.txt"  # held-out VALIDATION carve -- not the official test split

EXPECTED_N_QUESTIONS, EXPECTED_N_VIDEOS = 5553, 990
TOP_K, HALF_WIDTH, SPAN_MODE, PEAK_SOURCE = 12, 2.2, "ppr_peak", "clip_in_ppr_top8"
N_BOOT, BOOT_SEED = 10_000, 20260927
N_DETERMINISM = 20
ARMS = ("S", "F", "U")


# ---------------------------------------------------------------- helpers

def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def stop(msg: str) -> None:
    log(f"\n*** STOP: {msg}")
    sys.exit(2)


def read_lines(p: Path) -> set[str]:
    return {l.strip() for l in open(p, encoding="utf-8") if l.strip()}


def row_video(r: dict) -> str:
    return str(r.get("video") or r.get("video_id")).strip()


def caption_lines(frames) -> list[str]:
    """Identical formatting to scripts/pnowa_test_accgqa_run.py::_caption_lines."""
    lines = []
    for i, f in enumerate(frames, 1):
        if isinstance(f, dict):
            cap_val, ts = f.get("caption") or "", f.get("timestamp", 0.0)
        else:
            cap_val, ts = f.caption or "", f.timestamp
        cap = cap_val.get("semantic_caption") if isinstance(cap_val, dict) else cap_val
        lines.append(f"[Frame {i} @ {ts:.1f}s] {cap or '[CAPTION_FAILED]'}")
    return lines


def video_clustered_bootstrap(values_by_video: dict[str, list[float]]) -> tuple[float, float, float]:
    """Same algorithm as scripts/pnowa_test_accgqa_run.py::video_clustered_bootstrap."""
    videos = sorted(values_by_video)
    rng = random.Random(BOOT_SEED)

    def mean_of(sample):
        vals = [v for vid in sample for v in values_by_video[vid]]
        return sum(vals) / len(vals) if vals else float("nan")

    point = mean_of(videos)
    boots = sorted(mean_of([videos[rng.randrange(len(videos))] for _ in videos]) for _ in range(N_BOOT))
    return point, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]


# ---------------------------------------------------------------- configuration (prereg section 3)

def ingest_and_flat_config() -> IRISConfig:
    """Arm F / ingest config: identical to the section 6.1 harness (pnowa_test_accgqa_run.py)."""
    return IRISConfig(
        cerberus_mode="v2", l2_retrieve_top_k=TOP_K, ranking_mode="ppr",
        ppr_lambda=0.5, ppr_damping=0.5, candidate_thresh=0.08,
        l1_w_action=0.60, l1_w_query=0.25, l1_w_persist=0.15, l1_w_pagerank=0.0,
        l1_w_entropy=0.0, l1_w_hessian=0.0, l1_w_recency=0.0,
        captioner_backend="minicpm",  # recorded; not read by the caption path (see deviations)
    )


_IRIS_FIELDS = {f.name for f in dataclasses.fields(IRISConfig)}


def scene_sparse_arm(idx):
    """Arm S graph from the same frames, exactly as scripts/blockdiag_grounding_gate.py builds it."""
    cfg = dict(idx.config_snapshot)
    cfg.update(dict(graph_mode="scene_sparse", graph_edge_mode="block_diagonal",
                    scene_shortlist_width=0, scene_shortcut_margin=0.015,
                    scene_neighbor_window=30, scene_crossscene_mode="rep_only"))
    graph = iris_ingest._build_graph(idx.frames, cfg)
    idx_s = dataclasses.replace(idx, config_snapshot=cfg, _graph=graph)  # shares idx.frames -> shared caption cache
    return idx_s, IRISConfig(**{k: v for k, v in cfg.items() if k in _IRIS_FIELDS})


# ---------------------------------------------------------------- guards (prereg sections 2, 8)

def guard_registration_and_clean_tree(deviations: list) -> dict:
    added = git("log", "--diff-filter=A", "--format=%H", "--", str(PREREG.relative_to(REPO))).stdout.split()
    if not added:
        stop("pre-registration file has no adding commit in history")
    reg_commit = added[-1]
    if not reg_commit.startswith(PREREG_COMMIT_PREFIX):
        stop(f"pre-registration was added at {reg_commit[:12]}, expected {PREREG_COMMIT_PREFIX}")
    if git("merge-base", "--is-ancestor", reg_commit, "HEAD").returncode != 0:
        stop("registration commit is not an ancestor of HEAD")
    if git("diff", "--quiet", reg_commit, "HEAD", "--", str(PREREG.relative_to(REPO))).returncode != 0:
        stop("pre-registration file changed after registration")
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines() if l.strip()]
    if dirty:
        stop(f"working tree has {len(dirty)} tracked-dirty file(s): {dirty[:5]}")
    return {"registration_commit": reg_commit, "head": git("rev-parse", "HEAD").stdout.strip(),
            "tracked_dirty": 0, "prereg_sha256": sha256_file(PREREG)}


def guard_captioner(deviations: list) -> dict:
    from iris.iris_config import ConfigManager
    resolved_backend = getattr(ConfigManager().get_config(), "captioner_backend", None)
    if resolved_backend != "minicpm":
        stop(f"shipped config resolves captioner_backend={resolved_backend!r}; registration requires minicpm")
    cap = aria.get_captioner()
    if type(cap).__name__ != "MiniCPMCaptioner":
        stop(f"active captioner is {type(cap).__name__}, not MiniCPMCaptioner")
    deviations.append(
        "captioner_backend in the IRISConfig passed to ingest is not read by the caption path "
        "(iris._clip -> aria.get_captioner resolves from ConfigManager). MiniCPM was enforced by "
        "asserting the resolved captioner class instead; the registered intent (minicpm) is met.")
    return {"resolved_captioner_class": type(cap).__name__, "provenance": aria.get_captioner_provenance()}


def ensure_captions_or_stop(idx, retrieved_dicts: list[dict], counters: dict) -> None:
    """Lazy captioning with a hard stop on any VLM caption failure: iris._clip silently falls back
    to BLIP when the active captioner fails, which would switch captioners mid-run."""
    n_fail_before = len(aria._CAPTION_FAILURES)
    t0 = time.perf_counter()
    counters["frames_decoded"] += _ensure_captions(idx, retrieved_dicts)
    counters["caption_seconds"] += time.perf_counter() - t0
    if len(aria._CAPTION_FAILURES) > n_fail_before:
        stop(f"captioner failure (would silently fall back to BLIP): {aria._CAPTION_FAILURES[-1]}")


def server_reported_build(endpoint: str) -> dict:
    """Best-effort: whatever build/model metadata the server itself reports (prereg 8.5)."""
    import requests
    found = {}
    base = endpoint[:-3] if endpoint.rstrip("/").endswith("/v1") else endpoint
    for url in (f"{base.rstrip('/')}/props", f"{endpoint.rstrip('/')}/models"):
        try:
            r = requests.get(url, timeout=5)
            if r.ok:
                found[url] = r.json()
        except Exception:
            pass
    return found


# ---------------------------------------------------------------- data

OPTION_KEYS = ("a0", "a1", "a2", "a3", "a4")


def gold_letters(r: dict) -> set[str]:
    """Official test.csv stores the answer as option TEXT, not an index. Gold = every option whose
    text matches it exactly (one row has two identical options; either letter is correct)."""
    ans = r["answer"].strip()
    return {LETTERS[i] for i, k in enumerate(OPTION_KEYS) if r[k].strip() == ans}


def guard_answer_format(rows: list[dict], deviations: list) -> None:
    unmatched = [(r["video"], r["qid"]) for r in rows if not gold_letters(r)]
    if unmatched:
        stop(f"{len(unmatched)} question(s) whose answer text matches no option: {unmatched[:5]}")
    multi = sum(len(gold_letters(r)) > 1 for r in rows)
    deviations.append(
        "official test.csv stores the answer as option text, not an index (unlike the validation CSV the "
        "section 6.1 harness reads with int()); the gold letter is derived by exact text match to a0-a4. "
        f"{multi} question(s) have duplicate option text matching the answer; any matching letter is scored correct.")

def load_questions(test_csv: Path, smoke: int | None) -> tuple[list[dict], list[str]]:
    rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
    for r in rows:
        r["video"] = row_video(r)
        r["qid"] = str(r["qid"]).strip()
        r["family"] = str(r.get("type", "?"))[:1]
    videos = sorted({r["video"] for r in rows})
    if smoke:
        keep = set(videos[:smoke])
        rows = [r for r in rows if r["video"] in keep]
        videos = sorted(keep)
    return rows, videos


def guard_disjoint(test_videos: list[str]) -> None:
    ours = read_lines(OUR_VAL_TUNING) | read_lines(OUR_VAL_HELDOUT)
    ours |= {row_video(r) for r in csv.DictReader(open(VAL_CSV, encoding="utf-8"))}
    ours |= set(json.load(open(GSUB_VAL, encoding="utf-8")))
    leaked = sorted(set(test_videos) & ours)
    if leaked:
        stop(f"{len(leaked)} official-test video(s) overlap our validation data: {leaked[:10]}")


def resolve_video_paths(videos: list[str], videos_dir: Path, vid_map: Path | None) -> dict[str, Path]:
    mapping = json.load(open(vid_map, encoding="utf-8")) if vid_map else {}
    paths, missing = {}, []
    for v in videos:
        cands = [videos_dir / f"{v}.mp4"]
        if v in mapping:
            cands += [videos_dir / f"{mapping[v]}.mp4", videos_dir / Path(str(mapping[v])).name]
        hit = next((c for c in cands if c.exists()), None)
        (paths.__setitem__(v, hit) if hit else missing.append(v))
    if missing:
        stop(f"{len(missing)} test video(s) have no file under {videos_dir}: {missing[:10]}")
    return paths


# ---------------------------------------------------------------- ingest (one ingest, shared survivors)

def ingest_all(videos, paths, cache_dir: Path, cfg: IRISConfig) -> tuple[dict, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed, secs = {}, {}
    for i, v in enumerate(videos, 1):
        target = cache_dir / v
        if Path(str(target) + ".npz").exists():
            continue
        t0 = time.perf_counter()
        try:
            iris_ingest.save_index(iris_ingest.ingest(paths[v], cfg), target)
            secs[v] = time.perf_counter() - t0
            log(f"[INGEST {i}/{len(videos)}] {v} ok {secs[v]:.1f}s")
        except Exception as e:  # counted, listed; questions scored incorrect (prereg section 7)
            failed[v] = f"{type(e).__name__}: {e}"
            log(f"[INGEST {i}/{len(videos)}] {v} FAILED {failed[v]}")
    return failed, secs


# ---------------------------------------------------------------- answering

def answer_arms(row, idx_f, cfg_f, idx_s, cfg_s, gold, duration, counters) -> dict[str, dict]:
    """Returns {arm: result}. gold is None in smoke mode (nothing is scored)."""
    q = row["question"]
    opts = {k: row[k] for k in OPTION_KEYS}
    emb = _embed_query(q, cfg_f)  # CLIP text embedding; identical for S and F
    out = {}
    for arm, idx, cfg in (("S", idx_s, cfg_s), ("F", idx_f, cfg_f), ("U", idx_f, cfg_f)):
        t0 = time.perf_counter()
        if arm == "U":
            frs = [min(idx.frames, key=lambda fr, t=t: abs(fr.timestamp - t)) for t in uniform_ts(duration, TOP_K)]
            ensure_captions_or_stop(idx, [{"frame_idx": fr.frame_idx} for fr in frs], counters)
            context_frames, span = frs, predict_span(frs, mode="minmax")
        else:
            retrieved = _build_retrieved(idx, emb, cfg)
            ensure_captions_or_stop(idx, retrieved, counters)
            context_frames = retrieved
            span = predict_span(retrieved, mode=SPAN_MODE, half_width=HALF_WIDTH, duration=duration,
                                peak_source=PEAK_SOURCE, query_embedding=emb)
        lines = caption_lines(context_frames)
        prompt, context = build_mc_prompt(q, opts, "\n".join(lines))
        raw = aria.generate(prompt=prompt, context=context)
        parsed = parse_mc_answer(raw, opts)
        res = {"raw": raw, "parsed_letter": parsed.parsed_letter,
               "parse_failed": parsed.parsed_letter is None,
               "caption_failed_frames": sum("[CAPTION_FAILED]" in l for l in lines),
               "span": list(span) if span else None, "seconds": time.perf_counter() - t0}
        if gold is not None:
            gold_set, gold_spans = gold
            iou_off, iop_off = best_over_gold_spans(gold_spans, tuple(span)) if span else (0.0, 0.0)
            acc_qa = float(parsed.parsed_letter in gold_set)
            res.update(iop_official=iop_off, iou_official=iou_off,
                       iop_union=iop_union(tuple(span), gold_spans) if span else 0.0,
                       acc_qa=acc_qa, acc_gqa=float(acc_qa and iop_off >= 0.5))
        out[arm] = res
    return out


def determinism_check(rows, loaded, cfg_f, counters) -> None:
    """Prereg 8.3: first N questions of Arm F answered twice -> raw outputs byte-identical."""
    checked = 0
    for r in rows:
        idx = loaded.get(r["video"])
        if idx is None:
            continue
        emb = _embed_query(r["question"], cfg_f)
        retrieved = _build_retrieved(idx, emb, cfg_f)
        ensure_captions_or_stop(idx, retrieved, counters)
        opts = {k: r[k] for k in ("a0", "a1", "a2", "a3", "a4")}
        prompt, context = build_mc_prompt(r["question"], opts, "\n".join(caption_lines(retrieved)))
        a, b = aria.generate(prompt=prompt, context=context), aria.generate(prompt=prompt, context=context)
        if a != b:
            stop(f"determinism spot-check failed on {r['video']}/{r['qid']}")
        checked += 1
        if checked == N_DETERMINISM:
            break
    log(f"[DETERMINISM] {checked} Arm-F questions answered twice, byte-identical")


# ---------------------------------------------------------------- aggregation

def aggregate(rows_out, failed_videos) -> dict:
    by_arm = {a: [r for r in rows_out if r["arm"] == a] for a in ARMS}
    summary = {}
    for a, rs in by_arm.items():
        n = len(rs)
        m = lambda k: sum(r.get(k, 0.0) for r in rs) / n if n else None  # noqa: E731
        excl = [r for r in rs if not r["ingest_failed"]]
        summary[a] = {"n": n, "acc_gqa": m("acc_gqa"), "acc_qa": m("acc_qa"),
                      "mIoP_official": m("iop_official"), "iop05_official": (sum(r.get("iop_official", 0) >= 0.5 for r in rs) / n) if n else None,
                      "mIoU_official": m("iou_official"), "mIoP_union": m("iop_union"),
                      "parse_fail_rate": (sum(r.get("parse_failed", False) for r in rs) / n) if n else None,
                      "n_excluding_ingest_failures": len(excl),
                      "acc_gqa_excluding_ingest_failures": (sum(r.get("acc_gqa", 0) for r in excl) / len(excl)) if excl else None}
    def paired(k, a1, a2):
        key = lambda r: (r["video"], r["qid"])  # noqa: E731
        m2 = {key(r): r for r in by_arm[a2]}
        by_vid = {}
        for r in by_arm[a1]:
            o = m2.get(key(r))
            if o is not None:
                by_vid.setdefault(r["video"], []).append(r.get(k, 0.0) - o.get(k, 0.0))
        p, lo, hi = video_clustered_bootstrap(by_vid)
        return {"point": p, "ci95": [lo, hi], "excludes_zero": (lo > 0 or hi < 0)}
    summary["paired"] = {"acc_gqa_S_minus_F": paired("acc_gqa", "S", "F"),
                         "acc_qa_S_minus_F": paired("acc_qa", "S", "F"),
                         "acc_qa_S_minus_U": paired("acc_qa", "S", "U")}
    fam = {}
    for a in ("S", "F"):
        for r in by_arm[a]:
            fam.setdefault(a, {}).setdefault(r["family"], []).append(r.get("acc_gqa", 0.0))
    summary["by_family_acc_gqa"] = {a: {f: sum(v) / len(v) for f, v in d.items()} for a, d in fam.items()}
    summary["ingest_failures"] = failed_videos
    return summary


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--gsub-test", type=Path, required=True)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--vid-map", type=Path, default=None)
    ap.add_argument("--annotation-source", required=True, help='e.g. "doc-doc/NExT-GQA@<commit>"')
    ap.add_argument("--compare-test-csv", type=Path, default=None, help="mirror copy to sha256-compare")
    ap.add_argument("--compare-gsub-test", type=Path, default=None, help="mirror copy to sha256-compare")
    ap.add_argument("--server-build", required=True, help="llama-server build as launched, e.g. b10099")
    ap.add_argument("--model-sha256", required=True, help="sha256 of the GGUF file llama-server was launched with")
    ap.add_argument("--smoke", type=int, default=None)
    args = ap.parse_args()
    smoke = args.smoke
    deviations: list[str] = []
    started = datetime.now(timezone.utc).isoformat()

    prov = {"mode": "smoke" if smoke else "registered_run", "started_utc": started,
            "annotation_source": args.annotation_source,
            "test_csv_sha256": sha256_file(args.test_csv)}
    if not smoke:
        prov.update(guard_registration_and_clean_tree(deviations))
        prov["gsub_test_sha256"] = sha256_file(args.gsub_test)
        for ours, theirs, name in ((args.test_csv, args.compare_test_csv, "test.csv"),
                                   (args.gsub_test, args.compare_gsub_test, "gsub_test.json")):
            if theirs is not None and sha256_file(ours) != sha256_file(theirs):
                stop(f"{name}: official and mirror copies differ")
    rows, videos = load_questions(args.test_csv, smoke)
    if not smoke and (len(rows), len(videos)) != (EXPECTED_N_QUESTIONS, EXPECTED_N_VIDEOS):
        stop(f"split has {len(rows)} questions / {len(videos)} videos; registered {EXPECTED_N_QUESTIONS} / {EXPECTED_N_VIDEOS}")
    guard_disjoint(videos)
    if not smoke:
        guard_answer_format(rows, deviations)
    paths = resolve_video_paths(videos, args.videos_dir, args.vid_map)
    log(f"[DATA] {len(rows)} questions over {len(videos)} videos; disjointness and video-file checks passed")

    cfg_f = ingest_and_flat_config()
    backend = aria.LlamaServerBackend(endpoint=cfg_f.answerer_endpoint, text_model=cfg_f.answerer_model)
    aria.set_backend(backend)
    preflight_backend(backend)
    prov["answerer"] = {"model": cfg_f.answerer_model, "endpoint": cfg_f.answerer_endpoint,
                        "temperature": getattr(backend, "temperature", None),
                        "cache_prompt": getattr(backend, "cache_prompt", None)}
    prov["answerer"]["server_build_attested"] = args.server_build
    prov["answerer"]["model_gguf_sha256_attested"] = args.model_sha256
    prov["answerer"]["server_reported"] = server_reported_build(backend.endpoint)
    if not prov["answerer"]["server_reported"]:
        deviations.append("llama-server exposed no build information via /props or /models; "
                          "the build is recorded as operator-attested (--server-build) only")
    prov["captioner"] = guard_captioner(deviations)

    cache_dir = (SMOKE_DIR / "index_cache") if smoke else CACHE_DIR
    failed, ingest_secs = ingest_all(videos, paths, cache_dir, cfg_f)
    prov["ingest_seconds_this_session"] = ingest_secs

    gsub = None if smoke else json.load(open(args.gsub_test, encoding="utf-8"))
    counters = {"frames_decoded": 0, "caption_seconds": 0.0}

    def load(v):
        return None if v in failed else iris_ingest.load_index(cache_dir / v)

    if not smoke:
        det_loaded = {}
        for r in rows[:200]:
            if r["video"] not in det_loaded:
                det_loaded[r["video"]] = load(r["video"])
        determinism_check(rows, det_loaded, cfg_f, counters)
        del det_loaded

    ckpt = (SMOKE_DIR / "smoke_rows.jsonl") if smoke else CHECKPOINT
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if ckpt.exists():
        for line in open(ckpt, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["video"], r["qid"], r["arm"]))
        if done:
            deviations.append(f"resumed from checkpoint with {len(done)} completed question-arm rows")

    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    with open(ckpt, "a", encoding="utf-8") as out:
        for vi, v in enumerate(videos, 1):
            pending = [r for r in by_video[v] if any((v, r["qid"], a) not in done for a in ARMS)]
            if not pending:
                continue
            if v in failed:
                for r in pending:
                    for a in ARMS:
                        out.write(json.dumps({"video": v, "qid": r["qid"], "arm": a, "family": r["family"],
                                              "ingest_failed": True, "acc_qa": 0.0, "acc_gqa": 0.0,
                                              "iop_official": 0.0, "iou_official": 0.0, "iop_union": 0.0}) + "\n")
                out.flush()
                continue
            idx_f = load(v)
            idx_s, cfg_s = scene_sparse_arm(idx_f)
            duration = (float(gsub[v].get("duration", 0)) if gsub else 0.0) or max(fr.timestamp for fr in idx_f.frames)
            for r in pending:
                gold = None
                if gsub is not None:
                    gold = (gold_letters(r), gsub[v]["location"][r["qid"]])
                res = answer_arms(r, idx_f, cfg_f, idx_s, cfg_s, gold, duration, counters)
                for a, rr in res.items():
                    if (v, r["qid"], a) in done:
                        continue
                    out.write(json.dumps({"video": v, "qid": r["qid"], "arm": a, "family": r["family"],
                                          "ingest_failed": False, **rr}) + "\n")
                out.flush()
            iris_ingest.save_index(idx_f, cache_dir / v)  # persist captions so a resume reuses them
            log(f"[ANSWER {vi}/{len(videos)}] {v}: {len(pending)} question(s) done")

    rows_out = [json.loads(l) for l in open(ckpt, encoding="utf-8")]
    prov.update(finished_utc=datetime.now(timezone.utc).isoformat(), counters=counters,
                config_hash_F=config_hash(cfg_f),
                config_hash_S=config_hash(scene_sparse_arm(load(videos[0]))[1]) if videos[0] not in failed else None,
                deviations=deviations)
    if smoke:
        (SMOKE_DIR / "smoke_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        log(f"\n[SMOKE] {len(rows_out)} question-arm rows answered, nothing scored. Outputs in {SMOKE_DIR}")
        return

    summary = aggregate(rows_out, failed)
    OUT_RAW.write_text(json.dumps({"provenance": prov, "summary": summary, "rows": rows_out}, indent=2, default=str))
    s, pr = summary, summary["paired"]
    md = ["# Official NExT-GQA test set -- registered run (prereg 546d392)", "",
          f"Questions: {s['S']['n']} per arm; ingest failures: {len(failed)} videos (their questions scored incorrect).", "",
          "| arm | Acc@GQA | Acc@QA | mIoP (official) | IoP@0.5 (official) | mIoU | parse-fail |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, name in (("S", "S scene-sparse, complete-block"), ("F", "F flat (section 6.1 config)"), ("U", "U uniform")):
        x = s[a]
        md.append(f"| {name} | {x['acc_gqa']:.4f} | {x['acc_qa']:.4f} | {x['mIoP_official']:.4f} | "
                  f"{x['iop05_official']:.4f} | {x['mIoU_official']:.4f} | {x['parse_fail_rate']:.3f} |")
    md += ["", "Paired, video-clustered bootstrap (B=10,000, seed 20260927):", ""]
    for k, v in pr.items():
        md.append(f"- {k}: {v['point']:+.4f}, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                  f"{' -- excludes zero' if v['excludes_zero'] else ' -- includes zero'}")
    md += ["", "Deviations from the registration:", ""] + [f"- {d}" for d in deviations or ["none"]]
    OUT_MD.write_text("\n".join(md) + "\n")
    log(f"\nWrote {OUT_RAW}\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
