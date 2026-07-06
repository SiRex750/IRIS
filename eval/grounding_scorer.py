"""Caption-free grounding metrics for NExT-GQA.

No LLM, no captions, no aria.generate.  Pure geometry on retrieved timestamps
vs. gold temporal spans from gsub_val.json.

Public API
----------
frames_in_window(ts_list, gold_spans) -> float
iop(ts_list, gold_spans)              -> float
uniform_ts(duration, top_k)           -> list[float]
score_grounding_arm(grounded_rows, cache_dir, cfg, gsub) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ── Pure metric functions ─────────────────────────────────────────────────────

def frames_in_window(ts_list: list[float], gold_spans: list[list[float]]) -> float:
    """Fraction of retrieved timestamps that fall inside ANY gold span.

    in_count = sum(any(s <= t <= e for s, e in gold_spans) for t in ts_list)
    return in_count / len(ts_list)
    """
    if not ts_list:
        return 0.0
    in_count = sum(
        any(s <= t <= e for s, e in gold_spans)
        for t in ts_list
    )
    return in_count / len(ts_list)


def iop(ts_list: list[float], gold_spans: list[list[float]]) -> float:
    """Intersection-over-Prediction.

    predicted span = [min(ts_list), max(ts_list)]
    gold           = union of gold_spans
    IoP            = |pred ∩ gold_union| / |pred|
    Returns 0.0 when pred has zero width (all timestamps identical).
    """
    if not ts_list:
        return 0.0
    pred_s = min(ts_list)
    pred_e = max(ts_list)
    if pred_e <= pred_s:
        return 0.0
    pred_len = pred_e - pred_s
    intersect = 0.0
    for s, e in gold_spans:
        lo = max(pred_s, float(s))
        hi = min(pred_e, float(e))
        if hi > lo:
            intersect += hi - lo
    return intersect / pred_len


def uniform_ts(duration: float, top_k: int) -> list[float]:
    """Evenly-spaced timestamps across [0, duration] — the floor baseline."""
    return [(i + 0.5) / top_k * duration for i in range(top_k)]


# ── Arm scorer ────────────────────────────────────────────────────────────────

def load_indexes(grounded_rows: list[dict], cache_dir: Path) -> dict[str, Any]:
    """Load one IRISIndex per video; return {video_str: IRISIndex | None}.

    Called once per top_k iteration by the ablation runner so that fresh index
    objects (no mutated node state from a prior top_k run) are always used.
    """
    import iris.ingest as iris_ingest
    loaded: dict[str, Any] = {}
    for row in grounded_rows:
        vid = row["video"]
        if vid in loaded:
            continue
        npz = cache_dir / f"{vid}.npz"
        if npz.exists():
            try:
                loaded[vid] = iris_ingest.load_index(cache_dir / vid)
            except Exception as exc:
                loaded[vid] = None
                print(f"  LOAD_ERR {vid}: {exc}", flush=True)
    return loaded


def score_grounding_arm(
    grounded_rows: list[dict],
    cache_dir: Path,
    cfg: Any,
    gsub: dict,
    *,
    arm_name: str = "",
    loaded: dict[str, Any] | None = None,
) -> dict:
    """Score one retrieval arm against NExT-GQA gold spans.

    grounded_rows: rows already filtered to grounded-AND-cached.
    cache_dir:     Path to index_cache/ directory (used only when loaded=None).
    cfg:           IRISConfig instance (ranking_mode, ppr_lambda, top_k, …).
    gsub:          parsed gsub_val.json dict.
    arm_name:      label for "uniform" arm; if non-empty, bypasses retrieval.
    loaded:        pre-loaded {video: IRISIndex} dict from load_indexes().
                   When provided, no disk I/O is performed here; the caller is
                   responsible for loading fresh indexes per top_k so that node
                   mutations from one arm do not carry over to the next.

    Returns {
        "overall":      {"fiw": float, "iop": float, "n": int},
        "by_family":    {"C": {"fiw":…, "iop":…, "n":…}, "T": {…}},
        "per_question": {(video_str, qid_str): fiw_val, …},
    }
    """
    from iris.query import _embed_query, _build_retrieved

    top_k = getattr(cfg, "l2_retrieve_top_k", 8)

    # Use caller-supplied indexes if available; otherwise load from disk.
    if loaded is None:
        loaded = load_indexes(grounded_rows, cache_dir)

    fiw_by_fam: dict[str, list[float]] = {}
    iop_by_fam: dict[str, list[float]] = {}
    fiw_all: list[float] = []
    iop_all: list[float] = []
    per_question: dict[tuple[str, str], float] = {}

    for row in grounded_rows:
        vid   = row["video"]
        qid   = str(row["qid"])
        fam   = row["family"]
        index = loaded.get(vid)
        if index is None:
            continue

        gold_spans: list[list[float]] = gsub[vid]["location"][qid]

        if arm_name == "uniform":
            duration = float(gsub[vid].get("duration", 0))
            ts = uniform_ts(duration, top_k)
        else:
            emb       = _embed_query(row["question"], cfg)
            retrieved = _build_retrieved(index, emb, cfg)
            ts        = [f["timestamp"] for f in retrieved]

        fiw_val = frames_in_window(ts, gold_spans)
        iop_val = iop(ts, gold_spans)

        fiw_all.append(fiw_val)
        iop_all.append(iop_val)
        fiw_by_fam.setdefault(fam, []).append(fiw_val)
        iop_by_fam.setdefault(fam, []).append(iop_val)
        per_question[(vid, qid)] = fiw_val

    def _agg(fiws: list[float], iops: list[float]) -> dict:
        n = len(fiws)
        if n == 0:
            return {"fiw": None, "iop": None, "n": 0}
        return {"fiw": sum(fiws) / n, "iop": sum(iops) / n, "n": n}

    families = sorted(set(fiw_by_fam) | set(iop_by_fam))
    return {
        "overall":      _agg(fiw_all, iop_all),
        "by_family":    {
            f: _agg(fiw_by_fam.get(f, []), iop_by_fam.get(f, []))
            for f in families
        },
        "per_question": per_question,
    }
