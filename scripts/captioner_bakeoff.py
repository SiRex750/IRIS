"""scripts/captioner_bakeoff.py — Captioner bake-off: selecting IRIS's witness
model under the RAM budget. DIAGNOSTIC ONLY, no iris/ edits.

Generalizes scripts/diag_v4_moondream_captions.py's --caption/--stability/
--analyze machinery behind a --captioner {name} switch, over the SAME 55
frozen frame_idx values already captured in diag_v2_capture.jsonl (identical
protocol: fixed query-blind prompt "Describe this image.", temperature 0,
fixed seed, one request at a time, keep_alive=0 after each candidate).

CANDIDATES:
  blip         incumbent, existing transformers path (iris.aria.BLIPCaptioner,
               unconditional greedy generation -- this codebase's BLIPCaptioner
               takes no text prompt, so "unconditional caption" already IS its
               fixed protocol; nothing to change there).
  moondream    moondream:1.8b -- REUSES the frozen diag_v4 pass1/pass2 jsonl
               verbatim (frame_idx, moondream_caption, seconds); never
               re-queried. bakeoff_moondream_pass{1,2}.jsonl are a one-time,
               deterministic field-renaming transform of those frozen files,
               not a regeneration.
  qwen3.5-2b   qwen3.5:2b via Ollama.
  qwen3.5-4b   qwen3.5:4b via Ollama.
  qwen3-vl-2b  qwen3-vl:2b via Ollama (optional -- skipped with a note if not
               pulled).
  minicpm-v4.6 minicpm-v4.6 via Ollama (amendment). Thinking OFF: "think":
               False is sent on every request to every Ollama candidate here
               (harmless no-op for non-reasoning models); additionally every
               raw response is scanned for <think>...</think> contamination
               and, if found, stripped before use with the incident counted
               and surfaced -- never silently absorbed. Uses the plain
               "minicpm-v4.6" tag, not any "-thinking" variant.

PHASE 1 (per candidate; live Ollama/transformers calls; writes
bakeoff_{name}_pass1.jsonl -- frozen, never regenerated once written):
    python scripts/captioner_bakeoff.py --caption --captioner {name}

PHASE 2 (offline; may load DeBERTa NLI + CLIP; no live captioning calls):
    python scripts/captioner_bakeoff.py --analyze --captioner {name}

--stability (2 more passes, verbatim/paraphrase/divergent classification --
reserved for the top-2 non-BLIP candidates by (B) coverage, per the
pre-registered decision rule):
    python scripts/captioner_bakeoff.py --stability --captioner {name}

--all: orchestrates every step above across every available candidate, then
prints the pre-registered decision-rule table (fabricated=0 hard gate; audit
column left BLANK -- that's the flagged human step) and a reminder banner.

VERIFY:
    bash -c 'set -o pipefail; python scripts/captioner_bakeoff.py --all 2>&1 | tee bakeoff.log'

STOP conditions: none abort the whole run. A fabricated-claim entailment
(section C) is a HARD GATE on that ONE candidate (printed loudly, candidate
marked DISQUALIFIED) -- analysis of every other candidate continues, per
task spec ("print the pair, keep analyzing others").

HUMAN STEP (not automated, flagged every run): audit each candidate's
captions for the 7 ground-truth frames (258, 873, 1263, 3243, 3334, 23748,
27184) -- this script cannot know ground truth, only print the candidate
text so the human column can be filled in.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.demo_cctv_query import CACHE_PATH, VIDEO, _load_index
from scripts.diag_v2_scoping_separation import CAPTURE_JSONL, FABRICATED_CLAIMS, _load_capture
from scripts.diag_v3_visual_scoping import EMBEDS_NPZ as V3_EMBEDS_NPZ
from scripts.diag_v3_visual_scoping import _fact_header, _load_v3_embeds
from scripts.diag_v4_moondream_captions import (
    PASS1_JSONL as MOONDREAM_DIAG_V4_PASS1,
    _fetch_frames_pil,
    _image_to_b64_jpeg,
)
from scripts.verify_layer2 import _regenerate_normalized_kernels
from scripts.verify_stack_e2e import NEGATIVE_SPECS

from iris.cerberus_layers import (
    get_nli_gate,
    score_nli_pair,
    verify_absence_claim,
    verify_visual_claim,
    _sentencize,
)
from iris.claim_contract import AbsenceClaim, VisualClaim

MOONDREAM_DIAG_V4_PASS2 = REPO / "diag_v4_moondream_captions_pass2.jsonl"

OLLAMA_HOST = "http://localhost:11434"
FIXED_PROMPT = "Describe this image."
# qwen3.5:2b-inventory arm (amended registration): a list-format prompt
# instead of free description, with a num_predict overflow guard. Compared
# on CONTENT in section (A), not sentence-count/length style metrics --
# inventory format inflates sentence count and deflates per-sentence length
# by construction.
INVENTORY_PROMPT = (
    "List everything visible in this image: every person, object, vehicle, "
    "and action. One short sentence per item. Only what is clearly visible."
)
INVENTORY_NUM_PREDICT = 200
DETERMINISM_SEED = 42
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Baselines under the CURRENT judge (entailment-floor fix active --
# scripts/diag_l3_judge.py A2 / iris.cerberus_layers.score_nli_pair), from
# scripts/verify_layer2.py's own runs (layer2_verify_v3.log): section (B)'s
# sentence-scoped bound-NLI replay of normalized kernels.
COVERAGE_BASELINE_BLIP = 3
COVERAGE_BASELINE_MOONDREAM = 8

# scripts/diag_l3_judge.py / human review exhibits: known-confabulated
# captions, checked in section (F) regardless of which candidate produced
# a caption for that frame_idx (same 55-frame set for every candidate).
KNOWN_CONFABULATED = {
    258: "BLIP frame 258 (child)",
    27184: "Moondream frame 27184 (University sign)",
    23748: "Moondream frame 23748 (door)",
    3334: "Moondream frame 3334 (devoid-of-people)",
}

# Human audit ground-truth frames (scripts/verify_layer2.py's canonical
# frame + diag_l3_judge/diag_v4 confabulation exhibits) -- NOT auto-scored,
# printed every run as a reminder of what the human column covers.
HUMAN_AUDIT_FRAMES = [258, 873, 1263, 3243, 3334, 23748, 24420, 27184]

LEN_BUCKETS = [("<=30", 0, 30), ("31-60", 31, 60), (">60", 61, None)]


LATENCY_BUDGET_SEC_PER_FRAME = 35  # cold-query budget: sec/frame x l2_retrieve_top_k(=8) must stay tractable

# EXCLUSION RECORD (measured, not assumed) -- scripts/captioner_bakeoff.py
# qwen probe autopsy, run on this CPU-only host (torch.cuda.is_available()
# == False; Ollama has no GPU offload here either):
#
#   Config verified clean before excluding on latency (no bug to fix):
#     prompt sent:  "Describe this image." (FIXED_PROMPT, verbatim)
#     think:        False sent on every request; raw response had ZERO
#                   <think>...</think> contamination (regex-checked)
#     options:      {"temperature": 0, "seed": 42} (fixed, per protocol)
#
#   qwen3.5:2b, 3 frames, cold-ish (model freshly loaded / evicted between
#   calls): 215.57s, 208.12s, 208.06s -- mean ~210s/frame.
#   qwen3.5:2b, 1 frame, warm re-probe (same frame_idx=123, model already
#   resident): total_duration=106.25s load_duration=11.32s
#   prompt_eval_count=2058 prompt_eval_duration=74.96s (THE bottleneck --
#   CPU prefill of ~2058 tokens, almost all of it the image's vision-token
#   encoding, not text) eval_count=299 eval_duration=19.65s (generation
#   itself is fine, ~15 tok/s).
#
#   READ: the cost is CPU-bound prefill of the image's vision tokens, not a
#   template/thinking misconfiguration -- there is nothing to fix by
#   reprobing. Even the WARM best case (106s/frame) is 3x over the 35s/frame
#   budget. qwen3.5:4b (larger) and qwen3-vl:2b (same size class, same
#   vision-token prefill cost) are excluded on the same measured basis
#   without individually reprobing each -- they cannot beat a smaller
#   sibling's prefill-bound floor.
EXCLUDED_ON_LATENCY = {
    "qwen3.5-2b": "measured ~210s/frame cold, 106-108s/frame warm (CPU-bound image-token prefill, "
                  "74.96s of it) -- 3-6x over the 35s/frame budget. Config verified clean (fixed prompt, "
                  "think:false honored, zero <think> contamination) before exclusion.",
    "qwen3.5-4b": "same family, strictly larger than qwen3.5:2b -- excluded on qwen3.5:2b's measured "
                  "prefill-bound floor without reprobing (cannot be faster).",
    "qwen3-vl-2b": "same size class as qwen3.5:2b, same vision-token prefill cost -- excluded on "
                   "qwen3.5:2b's measured floor without reprobing.",
}

CANDIDATES: dict[str, dict] = {
    "blip": {"kind": "transformers"},
    "moondream": {"kind": "ollama", "model": "moondream:1.8b", "frozen_reuse": True},
    "qwen3.5-2b": {"kind": "ollama", "model": "qwen3.5:2b", "excluded": EXCLUDED_ON_LATENCY["qwen3.5-2b"]},
    "qwen3.5-4b": {"kind": "ollama", "model": "qwen3.5:4b", "excluded": EXCLUDED_ON_LATENCY["qwen3.5-4b"]},
    "qwen3-vl-2b": {"kind": "ollama", "model": "qwen3-vl:2b", "optional": True, "excluded": EXCLUDED_ON_LATENCY["qwen3-vl-2b"]},
    "minicpm-v4.6": {"kind": "ollama", "model": "minicpm-v4.6", "optional": True},
    # Amended registration: same model as qwen3.5-2b (excluded above), but a
    # DIFFERENT prompt (list/inventory format, num_predict=200 overflow
    # guard) -- the original exclusion was measured against the free-
    # description prompt's prefill cost, which num_predict cannot touch
    # (num_predict caps GENERATION length only; the ~75s bottleneck was
    # PROMPT prefill of the image's ~2000 vision tokens, paid before the
    # first output token). Registered as a genuinely separate arm, gated by
    # its own 1-frame latency probe (see run_qwen_inventory_probe) before
    # being allowed into --all.
    "qwen3.5-2b-inventory": {"kind": "ollama", "model": "qwen3.5:2b",
                             "prompt": INVENTORY_PROMPT, "num_predict": INVENTORY_NUM_PREDICT,
                             "optional": True, "latency_gated": True,
                             # MEASURED (run_latency_gate_probe, frame_idx=123): 99.89s/frame, think_stripped=0,
                             # truncated=False. Confirms num_predict cannot rescue this arm: the ~75-99s cost is
                             # PROMPT PREFILL of the image's vision tokens, paid before generation starts and
                             # independent of prompt text or output-length cap. Exclusion stands, no reprobe.
                             "excluded": "measured 99.89s/frame (1-frame probe) > 35s budget -- same "
                                         "prefill-bound floor as qwen3.5-2b; num_predict cannot fix a "
                                         "prefill-phase cost."},
    # Registered config (one shot, no iteration): same model as minicpm-v4.6,
    # inventory-format prompt + num_predict=250 overflow guard. minicpm-v4.6
    # was within budget (33.9s/frame free-description) so this arm is NOT
    # latency-gated -- goes straight to full protocol.
    "minicpm-v4.6-inventory": {"kind": "ollama", "model": "minicpm-v4.6",
                                "prompt": INVENTORY_PROMPT, "num_predict": 250},
}


def _pass1_path(name: str) -> Path:
    return REPO / f"bakeoff_{name}_pass1.jsonl"


def _pass_path(name: str, pass_n: int) -> Path:
    if pass_n == 1:
        return _pass1_path(name)
    return REPO / f"bakeoff_{name}_pass{pass_n}.jsonl"


def _meta_path(name: str) -> Path:
    return REPO / f"bakeoff_{name}_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# shared frame set (55 distinct frame_idx from diag_v2_capture.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

def _frame_idxs_and_pil():
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: {CAPTURE_JSONL} missing (run diag_v2_scoping_separation.py --capture first)", file=sys.stderr)
        sys.exit(1)
    records = _load_capture()
    frame_idxs = sorted({_fact_header(r["fact_text"])[0] for r in records if r["fact_text"] is not None})
    idx = _load_index()
    pil_by_frame = _fetch_frames_pil(idx, frame_idxs)
    return frame_idxs, pil_by_frame


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers (generic across candidates -- no per-model special-casing
# beyond the model tag itself)
# ─────────────────────────────────────────────────────────────────────────────

def _tag_matches(model: str, tag: str) -> bool:
    """Ollama tags list carries the explicit tag (e.g. 'minicpm-v4.6:latest')
    even when a model is referenced without one (implying ':latest') -- match
    either the exact string or the ':latest'-qualified form."""
    return tag == model or tag == f"{model}:latest" or tag.split(":")[0] == model.split(":")[0] and model.count(":") == 0


def _check_ollama_model(model: str) -> None:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"FATAL: Ollama not reachable at {OLLAMA_HOST}: {e}", file=sys.stderr)
        sys.exit(1)
    if not any(_tag_matches(model, t) for t in tags):
        print(f"FATAL: model {model!r} not found in Ollama. Available: {tags}", file=sys.stderr)
        sys.exit(1)


def _ollama_available(model: str) -> bool:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = {m["name"] for m in resp.json().get("models", [])}
    except Exception:
        return False
    return any(_tag_matches(model, t) for t in tags)


def _strip_think(raw: str) -> tuple[str, int]:
    """Strip <think>...</think> contamination (thinking-capable models that
    ignored think:false). Returns (cleaned, n_removed_blocks) -- never
    silently absorbed, callers surface n_removed_blocks."""
    n = len(THINK_TAG_RE.findall(raw))
    cleaned = THINK_TAG_RE.sub("", raw).strip()
    return cleaned, n


def _ollama_caption(model: str, image_b64: str, prompt: str = FIXED_PROMPT,
                     num_predict: int | None = None) -> tuple[str, int, bool]:
    """Returns (caption, n_think_blocks_stripped, truncated). truncated is
    True iff a num_predict cap was hit (done_reason == 'length') -- an
    overflow event to log and treat as a malformed caption, not data."""
    options = {"temperature": 0, "seed": DETERMINISM_SEED}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "options": options,
        "think": False,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    raw = data["response"].strip()
    truncated = num_predict is not None and data.get("done_reason") == "length"
    cleaned, n_think = _strip_think(raw)
    return cleaned, n_think, truncated


def _ollama_ram(model: str) -> dict | None:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=10)
        resp.raise_for_status()
    except Exception:
        return None
    for m in resp.json().get("models", []):
        if _tag_matches(model, m.get("model", "")) or _tag_matches(model, m.get("name", "")):
            return {"size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram")}
    return None


def _ollama_unload(model: str) -> None:
    try:
        requests.post(f"{OLLAMA_HOST}/api/generate",
                      json={"model": model, "prompt": "", "keep_alive": 0}, timeout=60)
    except Exception as e:
        print(f"  WARNING: unload request for {model!r} failed: {e}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — capture
# ─────────────────────────────────────────────────────────────────────────────

def _run_blip_capture(frame_idxs: list[int], pil_by_frame: dict) -> tuple[dict, dict]:
    import psutil
    from iris.aria import BLIPCaptioner

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    captioner = BLIPCaptioner()
    captioner._load()
    rss_after_load = proc.memory_info().rss

    results: dict[int, dict] = {}
    for i, frame_idx in enumerate(frame_idxs):
        t0 = time.time()
        caption = captioner.caption(pil_by_frame[frame_idx])
        elapsed = time.time() - t0
        results[frame_idx] = {"frame_idx": frame_idx, "caption": caption, "seconds": elapsed, "think_blocks_stripped": 0}
        print(f"  [{i + 1}/{len(frame_idxs)}] frame_idx={frame_idx} {elapsed:.2f}s blip={caption!r}")

    ram = {"kind": "process_rss_delta", "rss_delta_bytes": rss_after_load - rss_before,
           "rss_before_bytes": rss_before, "rss_after_load_bytes": rss_after_load}
    return results, ram


def _run_ollama_capture(model: str, frame_idxs: list[int], pil_by_frame: dict,
                         prompt: str = FIXED_PROMPT, num_predict: int | None = None) -> tuple[dict, dict]:
    _check_ollama_model(model)
    results: dict[int, dict] = {}
    total_think_blocks = 0
    total_truncated = 0
    ram = None
    for i, frame_idx in enumerate(frame_idxs):
        b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
        t0 = time.time()
        caption, n_think, truncated = _ollama_caption(model, b64, prompt=prompt, num_predict=num_predict)
        elapsed = time.time() - t0
        total_think_blocks += n_think
        total_truncated += int(truncated)
        if n_think:
            print(f"  ** think-tag contamination: {n_think} block(s) stripped for frame_idx={frame_idx} **")
        if truncated:
            print(f"  ** OVERFLOW: num_predict={num_predict} hit for frame_idx={frame_idx} -- malformed, logged as truncated, not data **")
        results[frame_idx] = {"frame_idx": frame_idx, "caption": caption, "seconds": elapsed,
                               "think_blocks_stripped": n_think, "truncated": truncated}
        print(f"  [{i + 1}/{len(frame_idxs)}] frame_idx={frame_idx} {elapsed:.2f}s {model}={caption!r}")
        if ram is None:
            ram = _ollama_ram(model)  # sample once the model is resident

    _ollama_unload(model)
    ram_info = {"kind": "ollama_ps", **(ram or {})}
    if total_think_blocks:
        ram_info["think_blocks_stripped_total"] = total_think_blocks
    if total_truncated:
        ram_info["truncated_total"] = total_truncated
    return results, ram_info


def _write_pass(name: str, pass_n: int, results: dict[int, dict], frame_idxs: list[int]) -> None:
    path = _pass_path(name, pass_n)
    with open(path, "w", encoding="utf-8") as fh:
        for frame_idx in frame_idxs:
            fh.write(json.dumps(results[frame_idx]) + "\n")
    print(f"wrote {len(results)} rows to {path}")


def _write_meta(name: str, ram_info: dict, results: dict[int, dict]) -> None:
    seconds = [r["seconds"] for r in results.values()]
    meta = {
        "candidate": name,
        "ram": ram_info,
        "mean_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "max_seconds": max(seconds),
        "n_frames": len(seconds),
    }
    with open(_meta_path(name), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {_meta_path(name)}: {json.dumps(meta, indent=2)}")


def _ensure_moondream_frozen_reuse() -> None:
    """Transform (never re-query) the frozen diag_v4 pass1/pass2 jsonl into
    the common bakeoff schema (frame_idx, caption, seconds). One-time,
    deterministic field rename -- not a caption regeneration."""
    if not MOONDREAM_DIAG_V4_PASS1.exists():
        print(f"FATAL: frozen {MOONDREAM_DIAG_V4_PASS1} missing -- cannot reuse", file=sys.stderr)
        sys.exit(1)

    for pass_n, src in ((1, MOONDREAM_DIAG_V4_PASS1), (2, MOONDREAM_DIAG_V4_PASS2)):
        dst = _pass_path("moondream", pass_n)
        if dst.exists():
            print(f"{dst} already present (frozen) -- not re-deriving")
            continue
        if not src.exists():
            print(f"NOTE: {src} missing -- skipping bakeoff pass{pass_n} for moondream")
            continue
        rows = []
        with open(src, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "frame_idx": r["frame_idx"], "caption": r["moondream_caption"],
                    "seconds": r["seconds"], "think_blocks_stripped": 0,
                }) + "\n")
        print(f"transformed {src} -> {dst} ({len(rows)} rows, reused verbatim, not regenerated)")

    if not _meta_path("moondream").exists():
        # RAM probe only (no caption regeneration): a single empty-prompt,
        # no-image generate call to load weights, read /api/ps, unload.
        model = CANDIDATES["moondream"]["model"]
        if _ollama_available(model):
            try:
                requests.post(f"{OLLAMA_HOST}/api/generate",
                               json={"model": model, "prompt": "", "keep_alive": "1m"}, timeout=120)
                ram = _ollama_ram(model)
                _ollama_unload(model)
            except Exception as e:
                print(f"  WARNING: RAM probe for {model!r} failed: {e}", file=sys.stderr)
                ram = None
        else:
            ram = None
        rows = [json.loads(l) for l in open(_pass_path("moondream", 1), "r", encoding="utf-8") if l.strip()]
        results = {r["frame_idx"]: r for r in rows}
        _write_meta("moondream", {"kind": "ollama_ps_ram_probe_not_caption_regen", **(ram or {})}, results)


def run_latency_gate_probe(name: str) -> bool:
    """One-frame latency probe for a 'latency_gated' candidate (amended
    registration: >35s/frame = exclusion stands, no iteration). Mutates
    CANDIDATES[name] in place -- sets 'excluded' with the measured number if
    the budget is blown, so run_caption/run_all skip it consistently."""
    spec = CANDIDATES[name]
    if not _ollama_available(spec["model"]):
        print(f"NOTE: latency-gated candidate {name!r} ({spec['model']!r}) not available in Ollama -- skipping probe.")
        spec["excluded"] = "model not available in Ollama"
        return False

    frame_idxs, pil_by_frame = _frame_idxs_and_pil()
    fi = frame_idxs[0]
    b64 = _image_to_b64_jpeg(pil_by_frame[fi])
    print(f"LATENCY PROBE: {name!r} ({spec['model']!r}) frame_idx={fi} prompt={spec.get('prompt', FIXED_PROMPT)!r} "
          f"num_predict={spec.get('num_predict')}")
    t0 = time.time()
    caption, n_think, truncated = _ollama_caption(
        spec["model"], b64, prompt=spec.get("prompt", FIXED_PROMPT), num_predict=spec.get("num_predict"),
    )
    elapsed = time.time() - t0
    _ollama_unload(spec["model"])
    print(f"  {elapsed:.2f}s think_stripped={n_think} truncated={truncated} caption={caption!r}")

    if elapsed > LATENCY_BUDGET_SEC_PER_FRAME:
        reason = (f"latency probe measured {elapsed:.2f}s/frame > {LATENCY_BUDGET_SEC_PER_FRAME}s budget "
                  f"(1-frame probe, frame_idx={fi}) -- exclusion stands, no reprobe.")
        print(f"  EXCLUDED: {reason}")
        spec["excluded"] = reason
        return False

    print(f"  WITHIN BUDGET ({elapsed:.2f}s <= {LATENCY_BUDGET_SEC_PER_FRAME}s) -- {name!r} proceeds to full protocol.")
    return True


def run_caption(name: str) -> None:
    spec = CANDIDATES[name]

    if spec.get("frozen_reuse"):
        _ensure_moondream_frozen_reuse()
        return

    if spec.get("excluded"):
        print(f"EXCLUDED (latency): {name!r} -- {spec['excluded']}")
        return

    path = _pass1_path(name)
    if path.exists():
        print(f"{path} already exists -- FROZEN, not regenerating. Delete manually to redo this candidate.")
        return

    if spec["kind"] == "ollama" and not _ollama_available(spec["model"]):
        if spec.get("optional"):
            print(f"NOTE: optional candidate {name!r} ({spec['model']!r}) not available in Ollama -- skipping.")
            return
        print(f"FATAL: required candidate {name!r} ({spec['model']!r}) not available in Ollama.", file=sys.stderr)
        sys.exit(1)

    frame_idxs, pil_by_frame = _frame_idxs_and_pil()
    print(f"CAPTURE candidate={name!r}: {len(frame_idxs)} frames")

    if spec["kind"] == "transformers":
        results, ram_info = _run_blip_capture(frame_idxs, pil_by_frame)
    else:
        results, ram_info = _run_ollama_capture(
            spec["model"], frame_idxs, pil_by_frame,
            prompt=spec.get("prompt", FIXED_PROMPT), num_predict=spec.get("num_predict"),
        )

    _write_pass(name, 1, results, frame_idxs)
    _write_meta(name, ram_info, results)


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers for analysis
# ─────────────────────────────────────────────────────────────────────────────

def _load_pass(name: str, pass_n: int = 1) -> list[dict]:
    path = _pass_path(name, pass_n)
    if not path.exists():
        print(f"FATAL: {path} missing (run --caption --captioner {name} first)", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _embed_texts(strings: list[str]) -> np.ndarray:
    import clip
    import torch
    from iris._clip import get_clip_model

    if not strings:
        return np.zeros((0, 512), dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_clip_model()
    if model is None:
        print("FATAL: CLIP model failed to load", file=sys.stderr)
        sys.exit(1)
    vecs = []
    batch_size = 32
    for i in range(0, len(strings), batch_size):
        batch = strings[i:i + batch_size]
        text_input = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            qf = model.encode_text(text_input)
            qf /= qf.norm(dim=-1, keepdim=True)
        vecs.append(qf.cpu().numpy().astype(np.float32))
    return np.concatenate(vecs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# (A) DIVERSITY
# ─────────────────────────────────────────────────────────────────────────────

def section_a_diversity(name: str, rows: list[dict]) -> dict:
    print("=" * 100)
    print(f"(A) DIVERSITY -- {name}")
    print("=" * 100)
    if CANDIDATES.get(name, {}).get("prompt") == INVENTORY_PROMPT:
        print("  NOTE: inventory-format prompt inflates sentence count and deflates per-sentence length")
        print("  BY CONSTRUCTION (one short sentence per item) -- compare this candidate on the cosine-")
        print("  distance/content columns below, not on raw length/sentence-count style metrics.")

    captions = [r["caption"] for r in rows]
    n = len(captions)
    counts = Counter(captions)
    distinct = len(counts)
    ratio = distinct / n if n else float("nan")
    mean_len = sum(len(c) for c in captions) / n if n else float("nan")
    top5 = counts.most_common(5)

    embeds = _embed_texts(captions)
    sims = embeds @ embeds.T
    iu = np.triu_indices(n, k=1)
    mean_cos_sim = float(sims[iu].mean()) if len(iu[0]) else float("nan")
    mean_cos_dist = 1.0 - mean_cos_sim

    print(f"  n captions: {n}")
    print(f"  exact-unique ratio: {distinct}/{n} = {ratio:.4f}")
    print(f"  mean caption length (chars): {mean_len:.1f}")
    print(f"  top-5 most repeated:")
    for cap, count in top5:
        print(f"    x{count}: {cap!r}")
    print(f"  mean pairwise CLIP text cosine SIMILARITY: {mean_cos_sim:.4f}")
    print(f"  mean pairwise CLIP text cosine DISTANCE:   {mean_cos_dist:.4f}  <- the honest column")
    print("  READ: exact-unique overstates information when a template varies only its tail")
    print("  (e.g. \"The image shows X.\" with X swapped) -- the cosine distance column is not")
    print("  fooled by that, since near-duplicate templates cluster near cos_dist=0 regardless")
    print("  of whether every string is literally distinct.")
    print()

    return {"n": n, "distinct_ratio": ratio, "mean_len": mean_len,
            "mean_cos_sim": mean_cos_sim, "mean_cos_dist": mean_cos_dist}


# ─────────────────────────────────────────────────────────────────────────────
# (B) COVERAGE -- sentence-scoped bound-NLI replay of normalized kernels
# ─────────────────────────────────────────────────────────────────────────────

def section_b_coverage(name: str, gate, kernel_fixtures: list[tuple[str, int, str]],
                        caption_by_frame: dict[int, str]) -> int:
    print("=" * 100)
    print(f"(B) COVERAGE -- sentence-scoped bound-NLI replay (layer-2 path) -- {name}")
    print("=" * 100)

    n_entailments = 0
    rows_out = []
    for normalized, frame_idx, _blip_caption in kernel_fixtures:
        caption = caption_by_frame.get(frame_idx)
        if caption is None:
            continue
        claim = VisualClaim(frame_idx=frame_idx, assertion=normalized)
        result = verify_visual_claim(claim, caption, gate)
        rows_out.append((normalized, frame_idx, result.verdict, result.best_sentence, result.best_score))
        if result.verdict == "verified":
            n_entailments += 1

    hdr = f"  {'normalized_kernel':<45} | {'frame':>7} | {'verdict':<12} | {'best_sentence':<40} | {'score':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for normalized, frame_idx, verdict, best_sentence, score in rows_out:
        k_disp = (normalized[:42] + "...") if len(normalized) > 45 else normalized
        s_disp = best_sentence or ""
        s_disp = (s_disp[:37] + "...") if len(s_disp) > 40 else s_disp
        print(f"  {k_disp:<45} | {frame_idx:>7} | {verdict:<12} | {s_disp:<40} | {score:>6.4f}")
    print()

    print(f"  entailments (coverage) for {name}: {n_entailments}/{len(rows_out)}")
    print(f"  BLIP baseline (current judge):      {COVERAGE_BASELINE_BLIP}")
    print(f"  Moondream baseline (current judge): {COVERAGE_BASELINE_MOONDREAM}")
    print()
    return n_entailments


# ─────────────────────────────────────────────────────────────────────────────
# (C) FABRICATED PROBE -- hard gate
# ─────────────────────────────────────────────────────────────────────────────

def section_c_fabricated_probe(name: str, gate, rows: list[dict]) -> list[tuple]:
    print("=" * 100)
    print(f"(C) FABRICATED PROBE -- 6 fabricated claims x every caption (sentence-scoped) -- {name}")
    print("=" * 100)

    entailed_pairs = []
    n_pairs = 0
    for claim_text in FABRICATED_CLAIMS:
        for r in rows:
            n_pairs += 1
            claim = VisualClaim(frame_idx=r["frame_idx"], assertion=claim_text)
            result = verify_visual_claim(claim, r["caption"], gate)
            if result.verdict == "verified":
                entailed_pairs.append((claim_text, r["frame_idx"], r["caption"], result.best_sentence, result.best_score))

    print(f"  n (claim, caption) pairs scored: {n_pairs}")
    print(f"  entailed pairs found: {len(entailed_pairs)}")
    print()

    if entailed_pairs:
        print(f"  ** HARD GATE: fabricated claim(s) entailed by a {name} caption -- DISQUALIFIED **", file=sys.stderr)
        print(f"  ** HARD GATE: fabricated claim(s) entailed by a {name} caption -- DISQUALIFIED **")
        for claim_text, frame_idx, caption, best_sentence, score in entailed_pairs:
            print(f"    claim={claim_text!r}")
            print(f"    frame_idx={frame_idx} caption={caption!r}")
            print(f"    entailing_sentence={best_sentence!r} score={score:.4f}")
            print()
        print(f"  {name} is DISQUALIFIED (fabricated=0 is a hard gate). Continuing to analyze other candidates.")
    else:
        print(f"  No fabricated claim entailed -- {name} clears the hard gate.")
    print()
    return entailed_pairs


# ─────────────────────────────────────────────────────────────────────────────
# (D) LEN -- premise-token distribution + per-bucket fabricated rates
# ─────────────────────────────────────────────────────────────────────────────

def section_d_length(name: str, gate, rows: list[dict]) -> None:
    print("=" * 100)
    print(f"(D) LEN -- premise-token distribution + per-bucket fabricated rates -- {name}")
    print("=" * 100)

    tokenizer, _model = gate._get_nli_model()

    pair_data = []
    for r in rows:
        tokens = len(tokenizer.encode(r["caption"], add_special_tokens=False))
        entailed_any = False
        for claim_text in FABRICATED_CLAIMS:
            claim = VisualClaim(frame_idx=r["frame_idx"], assertion=claim_text)
            result = verify_visual_claim(claim, r["caption"], gate)
            if result.verdict == "verified":
                entailed_any = True
        pair_data.append({"frame_idx": r["frame_idx"], "tokens": tokens, "entailed_any_fabricated": entailed_any})

    hdr = f"  {'bucket':<8} | {'n_captions':>10} | {'n_fabricated_entailed':>22} | {'rate':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, lo, hi in LEN_BUCKETS:
        if hi is None:
            bucket = [p for p in pair_data if p["tokens"] >= lo]
        else:
            bucket = [p for p in pair_data if lo <= p["tokens"] <= hi]
        n = len(bucket)
        n_ent = sum(1 for p in bucket if p["entailed_any_fabricated"])
        rate = n_ent / n if n else float("nan")
        rate_disp = f"{rate:>8.4f}" if n else f"{'n/a':>8}"
        print(f"  {label:<8} | {n:>10} | {n_ent:>22} | {rate_disp}")
    print()

    all_tokens = [p["tokens"] for p in pair_data]
    print(f"  premise tokens: mean={statistics.mean(all_tokens):.1f} median={statistics.median(all_tokens):.1f} "
          f"max={max(all_tokens)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# (E) ABSENCE REPLAY -- 7 e2e absence fixtures against this candidate's captions
# ─────────────────────────────────────────────────────────────────────────────

def section_e_absence_replay(name: str, gate, rows: list[dict]) -> None:
    print("=" * 100)
    print(f"(E) ABSENCE REPLAY -- 7 e2e absence fixtures (layer-3, current rules) -- {name}")
    print("=" * 100)
    print("  NOTE: layer 3 checks the bounded set of frames actually retrieved for a query; the")
    print("  bake-off has no per-query retrieval, so this candidate's full 55-frame caption set")
    print("  stands in as the bounded set -- a superset of any real query's retrieval, i.e. a")
    print("  STRICTER replay (more captions for a false-positive trigger to hide in) than any of")
    print("  the 7 fixtures saw in scripts/verify_stack_e2e.py.")
    print()

    all_captions = [(r["frame_idx"], r["caption"]) for r in rows]

    for question, event in NEGATIVE_SPECS:
        result = verify_absence_claim(AbsenceClaim(event=event, is_core=True), all_captions, gate)
        print(f"  question={question!r}")
        print(f"    event={event!r}")
        print(f"    verdict={result.verdict}")
        if result.verdict == "rejected":
            print(f"    ** REJECTION (possible confabulation-laundering exhibit): "
                  f"frame_idx={result.rejecting_frame_idx} sentence={result.rejecting_sentence!r} "
                  f"score={result.rejecting_score:.4f} **")
        print(f"    {result.phrasing}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# (F) FIDELITY MARGIN -- CLIP text-vs-visual alignment margin
# ─────────────────────────────────────────────────────────────────────────────

def section_f_fidelity_margin(name: str, rows: list[dict]) -> None:
    print("=" * 100)
    print(f"(F) FIDELITY MARGIN -- CLIP caption-vs-own-frame minus caption-vs-other-frames -- {name}")
    print("=" * 100)

    embeds3 = _load_v3_embeds()
    frame_idx_list = list(embeds3["frame_idxs"])
    frame_row = {fi: i for i, fi in enumerate(frame_idx_list)}
    visual = embeds3["frame_visual_embeds"]

    captions = [r["caption"] for r in rows]
    cap_embeds = _embed_texts(captions)

    margins: dict[int, float] = {}
    caption_by_frame = {r["frame_idx"]: r["caption"] for r in rows}
    for r, cap_emb in zip(rows, cap_embeds):
        fi = r["frame_idx"]
        vi = frame_row.get(fi)
        if vi is None:
            continue
        own_sim = float(cap_emb @ visual[vi])
        other_idx = [j for j in range(len(frame_idx_list)) if j != vi]
        other_mean = float(np.mean(visual[other_idx] @ cap_emb))
        margins[fi] = own_sim - other_mean

    vals = list(margins.values())
    if vals:
        arr = np.array(vals)
        print(f"  n captions with a visual embedding: {len(vals)}")
        print(f"  margin distribution: mean={arr.mean():.4f} median={np.percentile(arr,50):.4f} "
              f"std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  quantiles: p10={np.percentile(arr,10):.4f} p25={np.percentile(arr,25):.4f} "
              f"p75={np.percentile(arr,75):.4f} p90={np.percentile(arr,90):.4f}")
    print()

    print("  -- KNOWN-CONFABULATED CAPTIONS (do they sit low? print, don't conclude) --")
    for fi, label in KNOWN_CONFABULATED.items():
        if fi in margins:
            cap = caption_by_frame.get(fi, "")
            cap_disp = (cap[:70] + "...") if len(cap) > 70 else cap
            print(f"    {label}: margin={margins[fi]:.4f} caption={cap_disp!r}")
        else:
            print(f"    {label}: frame_idx={fi} not in this candidate's captured set")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# analyze orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_analyze(name: str) -> dict:
    print("#" * 100)
    print(f"# ANALYZING CANDIDATE: {name}  ({_pass1_path(name)})")
    print("#" * 100)
    print()

    rows = _load_pass(name, 1)
    n_truncated = sum(1 for r in rows if r.get("truncated"))
    if n_truncated:
        print(f"  ** {n_truncated}/{len(rows)} captions hit the num_predict overflow guard -- "
              f"malformed, EXCLUDED from analysis as logged, not data. **")
        rows = [r for r in rows if not r.get("truncated")]
    caption_by_frame = {r["frame_idx"]: r["caption"] for r in rows}

    gate = get_nli_gate()
    records = _load_capture()
    kernel_fixtures = _regenerate_normalized_kernels(gate, records)

    div = section_a_diversity(name, rows)
    coverage = section_b_coverage(name, gate, kernel_fixtures, caption_by_frame)
    fabricated_hits = section_c_fabricated_probe(name, gate, rows)
    section_d_length(name, gate, rows)
    section_e_absence_replay(name, gate, rows)
    section_f_fidelity_margin(name, rows)

    meta_path = _meta_path(name)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "name": name, "coverage": coverage, "fabricated_n": len(fabricated_hits),
        "disqualified": bool(fabricated_hits), "diversity": div, "meta": meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# --stability (diag_v4 protocol, generalized)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_caption_pair(gate, cap_i: str, cap_j: str) -> str:
    if cap_i == cap_j:
        return "verbatim"
    label_i_premise, _ = score_nli_pair(gate, cap_j, cap_i)
    label_j_premise, _ = score_nli_pair(gate, cap_i, cap_j)
    if label_i_premise == "contradiction" or label_j_premise == "contradiction":
        return "divergent"
    return "paraphrase"


def run_stability(name: str) -> None:
    spec = CANDIDATES[name]
    pass1_path = _pass1_path(name)
    if not pass1_path.exists():
        print(f"FATAL: {pass1_path} missing -- run --caption first", file=sys.stderr)
        sys.exit(1)
    pass1_rows = _load_pass(name, 1)
    frame_idxs = sorted(r["frame_idx"] for r in pass1_rows)
    pass_captions: dict[int, dict[int, str]] = {1: {r["frame_idx"]: r["caption"] for r in pass1_rows}}

    if spec.get("frozen_reuse"):
        pass2_rows = _load_pass(name, 2)
        pass_captions[2] = {r["frame_idx"]: r["caption"] for r in pass2_rows}
        print(f"NOTE: {name} stability reuses frozen pass1/pass2 -- only 1 live pass (pass3) run below.")
        idx = _load_index()
        pil_by_frame = _fetch_frames_pil(idx, frame_idxs)
        extra_passes = [3]
    else:
        idx = _load_index()
        pil_by_frame = _fetch_frames_pil(idx, frame_idxs)
        extra_passes = [2, 3]

    for pass_n in extra_passes:
        path = _pass_path(name, pass_n)
        if path.exists():
            rows = _load_pass(name, pass_n)
            pass_captions[pass_n] = {r["frame_idx"]: r["caption"] for r in rows}
            print(f"{path} already exists (frozen) -- reusing.")
            continue
        print(f"=== {name}: CAPTIONING PASS {pass_n} ===")
        results: dict[int, dict] = {}
        for i, frame_idx in enumerate(frame_idxs):
            b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
            t0 = time.time()
            if spec["kind"] == "transformers":
                from iris.aria import BLIPCaptioner
                captioner = BLIPCaptioner()
                caption = captioner.caption(pil_by_frame[frame_idx])
                n_think = 0
            else:
                caption, n_think, _truncated = _ollama_caption(
                    spec["model"], b64, prompt=spec.get("prompt", FIXED_PROMPT), num_predict=spec.get("num_predict"),
                )
            elapsed = time.time() - t0
            results[frame_idx] = {"frame_idx": frame_idx, "caption": caption, "seconds": elapsed,
                                   "think_blocks_stripped": n_think}
            print(f"  [{i + 1}/{len(frame_idxs)}] frame_idx={frame_idx} {elapsed:.2f}s {name}={caption!r}")
        _write_pass(name, pass_n, results, frame_idxs)
        pass_captions[pass_n] = {fi: results[fi]["caption"] for fi in frame_idxs}
        if spec["kind"] == "ollama":
            _ollama_unload(spec["model"])

    gate = get_nli_gate()
    pair_keys = [(1, 2), (1, 3), (2, 3)]
    counts = {pk: {"verbatim": 0, "paraphrase": 0, "divergent": 0} for pk in pair_keys}
    divergent_examples = []

    print("=" * 100)
    print(f"STABILITY CLASSIFICATION -- {name}")
    print("=" * 100)
    for frame_idx in frame_idxs:
        for (pi, pj) in pair_keys:
            cap_i = pass_captions[pi][frame_idx]
            cap_j = pass_captions[pj][frame_idx]
            cls = _classify_caption_pair(gate, cap_i, cap_j)
            counts[(pi, pj)][cls] += 1
            if cls == "divergent":
                divergent_examples.append((frame_idx, pi, pj, cap_i, cap_j))
    print()

    total_divergent = total_all = 0
    for pk in pair_keys:
        c = counts[pk]
        n = c["verbatim"] + c["paraphrase"] + c["divergent"]
        print(f"  pass{pk[0]}_vs_pass{pk[1]}: verbatim={c['verbatim']}/{n} paraphrase={c['paraphrase']}/{n} "
              f"divergent={c['divergent']}/{n} (rate={c['divergent']/n:.4f})" if n else f"  pass{pk[0]}_vs_pass{pk[1]}: n=0")
        total_divergent += c["divergent"]
        total_all += n
    print()
    if total_all:
        print(f"  HEADLINE DIVERGENT RATE ({total_all} pair-instances): {total_divergent}/{total_all} = "
              f"{total_divergent/total_all:.4f}")
    print()

    if divergent_examples:
        print("  -- DIVERGENT EXAMPLES --")
        for frame_idx, pi, pj, cap_i, cap_j in divergent_examples:
            print(f"    frame_idx={frame_idx} pass{pi}_vs_pass{pj}")
            print(f"      pass{pi}: {cap_i!r}")
            print(f"      pass{pj}: {cap_j!r}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# --all orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def _print_human_audit_banner() -> None:
    print("#" * 100)
    print("# HUMAN STEP (flagged for Siddanth, NOT automated)")
    print("#" * 100)
    print("  Audit each candidate's captions for the 7 ground-truth frames below against your")
    print("  existing notes -- count true/false assertions per candidate. This column decides")
    print("  ties and can OVERRIDE coverage: a high-coverage confabulator loses to a lower-")
    print("  coverage honest witness.")
    print(f"  ground-truth frames: {HUMAN_AUDIT_FRAMES}")
    print()


def run_all() -> None:
    names = list(CANDIDATES.keys())

    print("=" * 100)
    print("PHASE 1: CAPTURE")
    print("=" * 100)
    available = []
    for name in names:
        spec = CANDIDATES[name]
        if spec["kind"] == "ollama" and not spec.get("frozen_reuse") and not _ollama_available(spec["model"]):
            if spec.get("optional"):
                print(f"SKIP optional candidate {name!r}: {spec['model']!r} not pulled in Ollama.")
                continue
        if spec.get("latency_gated") and not spec.get("excluded") and not _pass1_path(name).exists():
            if not run_latency_gate_probe(name):
                continue  # excluded on measured latency, no reprobe
        run_caption(name)
        available.append(name)

    print("=" * 100)
    print("PHASE 2: ANALYZE")
    print("=" * 100)
    summaries = []
    for name in available:
        if not _pass1_path(name).exists():
            continue
        summaries.append(run_analyze(name))

    print("=" * 100)
    print("PRE-REGISTERED DECISION-RULE TABLE")
    print("=" * 100)
    print("  fabricated=0 is a HARD GATE. Then rank by: audit confabulation rate ASC (HUMAN,")
    print("  blank below), coverage DESC; RAM/latency breaks ties. qwen3.5-2b gets the")
    print("  one-model-two-seats note ONLY if its audit column is competitive.")
    print()
    hdr = f"  {'candidate':<14} | {'gate':<12} | {'coverage':>8} | {'mean_s':>8} | {'ram':>14} | {'audit(HUMAN)':<14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    ranked = sorted(summaries, key=lambda s: (s["disqualified"], -s["coverage"]))
    for s in ranked:
        gate_disp = "DISQUALIFIED" if s["disqualified"] else "pass"
        meta = s["meta"]
        mean_s = meta.get("mean_seconds")
        mean_s_disp = f"{mean_s:.2f}" if isinstance(mean_s, (int, float)) else "n/a"
        ram = meta.get("ram", {})
        ram_bytes = ram.get("rss_delta_bytes") or ram.get("size_bytes")
        ram_disp = f"{ram_bytes/1e9:.2f} GB" if isinstance(ram_bytes, (int, float)) else "n/a"
        print(f"  {s['name']:<14} | {gate_disp:<12} | {s['coverage']:>8} | {mean_s_disp:>8} | {ram_disp:>14} | {'____':<14}")
    print()

    print("=" * 100)
    print("PHASE 3: STABILITY (top-2 non-BLIP candidates by coverage, gate-passing only)")
    print("=" * 100)
    eligible = [s for s in summaries if s["name"] != "blip" and not s["disqualified"]]
    eligible.sort(key=lambda s: -s["coverage"])
    top2 = eligible[:2]
    if not top2:
        print("  No non-BLIP candidate cleared the fabricated hard gate -- nothing to stability-test.")
    for s in top2:
        print(f"  running --stability for {s['name']!r} (coverage={s['coverage']})")
        run_stability(s["name"])

    _print_human_audit_banner()


def main() -> None:
    if "--all" in sys.argv:
        run_all()
        return

    captioner = None
    if "--captioner" in sys.argv:
        idx = sys.argv.index("--captioner")
        if idx + 1 >= len(sys.argv):
            print("FATAL: --captioner requires a name", file=sys.stderr)
            sys.exit(1)
        captioner = sys.argv[idx + 1]
        if captioner not in CANDIDATES:
            print(f"FATAL: unknown --captioner {captioner!r}. Known: {list(CANDIDATES)}", file=sys.stderr)
            sys.exit(1)

    if "--probe" in sys.argv:
        if captioner is None:
            print("FATAL: --probe requires --captioner {name}", file=sys.stderr)
            sys.exit(1)
        run_latency_gate_probe(captioner)
    elif "--caption" in sys.argv:
        if captioner is None:
            print("FATAL: --caption requires --captioner {name}", file=sys.stderr)
            sys.exit(1)
        spec = CANDIDATES[captioner]
        if spec.get("latency_gated") and not spec.get("excluded"):
            if not run_latency_gate_probe(captioner):
                return
        run_caption(captioner)
    elif "--analyze" in sys.argv:
        if captioner is None:
            print("FATAL: --analyze requires --captioner {name}", file=sys.stderr)
            sys.exit(1)
        run_analyze(captioner)
    elif "--stability" in sys.argv:
        if captioner is None:
            print("FATAL: --stability requires --captioner {name}", file=sys.stderr)
            sys.exit(1)
        run_stability(captioner)
    else:
        print("Usage: python scripts/captioner_bakeoff.py --all | "
              "(--caption | --analyze | --stability) --captioner {name}")
        print(f"Known captioners: {list(CANDIDATES)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
