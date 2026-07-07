"""scripts/diag_v4_moondream_captions.py — DIAGNOSTIC ONLY, no mechanism change.

Evaluates moondream:1.8b (via Ollama) as a caption source, substituted in
place of BLIP, against the same 55 distinct frames already captured in
diag_v2_capture.jsonl. --kernel-normalize established that normalized-kernel
NLI replay against BLIP captions recovers 4 entailments (out of 0 under raw
citation-binding). This script asks: is 4 a ceiling imposed by BLIP's caption
quality, or by the claim/citation-binding approach itself? Same
citation-resolution and kernel-normalization machinery, captions swapped.

Zero edits to iris/, tests/, demo_cctv_query.py, diag_v2_scoping_separation.py,
diag_v3_visual_scoping.py -- imports only. Does not re-run ARIA, does not
re-run BLIP, does not touch the v2/v3 capture artifacts.

PHASE 1 (live Ollama calls, seek-based frame fetch, writes capture artifact):
    python scripts/diag_v4_moondream_captions.py --caption

diag_v4_moondream_captions_pass1.jsonl is the FROZEN, canonical first-pass
caption artifact (originally written by --caption, then renamed) -- never
regenerated. --caption still writes diag_v4_moondream_captions.jsonl if
re-run, but that is a distinct file; pass1 is untouched by any later run.

--stability mode: re-captions all 55 frames twice more (pass2, pass3; same
fixed prompt, temp=0, seed=42, one request at a time, PLUS num_thread=1 to
test whether pinning threads removes the nondeterminism --caption's
determinism check already found -- 3/5 sampled frames disagreed with
themselves under default threading). Classifies every frame's (pass_i,
pass_j) caption pair as verbatim / paraphrase / divergent (NLI contradiction
in either direction) and reports the divergent rate as the headline number.
    python scripts/diag_v4_moondream_captions.py --stability

PHASE 2 (offline from the pass jsonl files; MAY load the DeBERTa NLI model
for the bound-NLI replay and fabricated-claim probe; no Ollama calls;
deterministic). Takes --pass {1,2,3} to select which caption set sections
(A)-(D) run against; defaults to pass 1 if omitted:
    python scripts/diag_v4_moondream_captions.py --analyze --pass 1

--split-premise mode: offline (pass1 + pass2 jsonl + capture files; loads
the DeBERTa NLI model; no Ollama calls; deterministic per pass). Tests the
length-degradation hypothesis behind the fabricated-claim false entailments
found in --analyze: splits each Moondream caption into sentences (spaCy
sentencizer) and scores a claim against each sentence individually instead
of the whole multi-sentence premise at once. Reports split-premise vs
whole-premise entailment counts for both the fabricated probe and the
bound-NLI replay, plus an explicit premise-token-count-bucketed entailment
rate. No STOP conditions here -- every outcome (confirms, refutes, or is
inconclusive about the length hypothesis) is a finding to surface.
    python scripts/diag_v4_moondream_captions.py --split-premise

VERIFY:
    python scripts/diag_v4_moondream_captions.py --caption 2>&1 | tee diag_v4_caption.log
    python scripts/diag_v4_moondream_captions.py --stability 2>&1 | tee diag_v4_stability.log
    python scripts/diag_v4_moondream_captions.py --analyze --pass 1 2>&1 | tee diag_v4_output_pass1.log
    python scripts/diag_v4_moondream_captions.py --analyze --pass 2 2>&1 | tee diag_v4_output_pass2.log
    python scripts/diag_v4_moondream_captions.py --analyze --pass 3 2>&1 | tee diag_v4_output_pass3.log
    python scripts/diag_v4_moondream_captions.py --split-premise 2>&1 | tee diag_v4_split.log

STOP conditions: Ollama unreachable or model absent; a --caption
determinism-check repeat that doesn't match its first pass; any fabricated
claim entailed by any Moondream caption, on ANY pass (surface, never tune
around). --stability's divergent rate is a finding at any value, not a stop.
--split-premise has no STOP conditions -- every outcome is a finding.
"""
from __future__ import annotations

import base64
import io
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.demo_cctv_query import CACHE_PATH, VIDEO, _load_index
from scripts.diag_v2_scoping_separation import CAPTURE_JSONL, FABRICATED_CLAIMS, _group_by_claim, _load_capture
from scripts.diag_v3_visual_scoping import (
    _decompose_claim,
    _fact_header,
    _get_nli_gate,
    _normalize_kernel,
    _resolve_citations,
    _score_kernel_pair,
    _classify_claim,
)

CAPTIONS_JSONL = REPO / "diag_v4_moondream_captions.jsonl"
PASS1_JSONL = REPO / "diag_v4_moondream_captions_pass1.jsonl"  # frozen, canonical -- never regenerated


def _pass_jsonl_path(pass_n: int) -> Path:
    if pass_n == 1:
        return PASS1_JSONL
    return REPO / f"diag_v4_moondream_captions_pass{pass_n}.jsonl"


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "moondream:1.8b"
FIXED_PROMPT = "Describe this image."
N_DETERMINISM_SAMPLE = 5
DETERMINISM_SEED = 42

# Established in --kernel-normalize: 4 entailments recovered via normalized
# kernel + BLIP caption replay (vs 0 under raw citation-binding). This
# script's baseline for comparison.
BLIP_BASELINE_ENTAILMENTS = 4
TOP_K_FOR_LATENCY_PROJECTION = 8


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — --caption
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama() -> list[str]:
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"FATAL: Ollama not reachable at {OLLAMA_HOST}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"FATAL: could not parse Ollama /api/tags response: {e}", file=sys.stderr)
        sys.exit(1)

    if OLLAMA_MODEL not in tags:
        print(f"FATAL: model {OLLAMA_MODEL!r} not found in Ollama. Available: {tags}", file=sys.stderr)
        sys.exit(1)

    print(f"Ollama reachable at {OLLAMA_HOST}. {OLLAMA_MODEL!r} present. Available models: {tags}")
    return tags


def _image_to_b64_jpeg(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _caption_with_moondream(image_b64: str, extra_options: dict | None = None) -> str:
    options = {"temperature": 0, "seed": 42}
    if extra_options:
        options.update(extra_options)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": FIXED_PROMPT,
        "images": [image_b64],
        "options": options,
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _fetch_frames_pil(index, frame_idxs: list[int]) -> dict[int, object]:
    """Seek-based frame fetch -- same algorithm as iris/query.py:_ensure_captions's
    per-miss seek loop (container.seek to the keyframe at/before the target
    timestamp, then decode forward only as far as that GOP requires). Not a
    new decoder: isolated here to return a PIL image directly instead of
    mutating a FrameRecord.caption, since this diagnostic never touches BLIP
    or iris/ captioning."""
    import av

    frame_map = {fr.frame_idx: fr for fr in index.frames}
    result: dict[int, object] = {}
    container = av.open(str(index.video_path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        tolerance = 0.5 / fps
        for frame_idx in frame_idxs:
            fr = frame_map.get(frame_idx)
            if fr is None:
                raise ValueError(f"frame_idx {frame_idx} not present in index")
            target_pts = int(round(fr.timestamp / stream.time_base))
            container.seek(target_pts, stream=stream)

            target_frame = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * stream.time_base)
                if frame_time >= fr.timestamp - tolerance:
                    target_frame = frame
                    break
            if target_frame is None:
                raise RuntimeError(f"could not decode a frame for frame_idx={frame_idx}")
            result[frame_idx] = target_frame.to_image()
    finally:
        container.close()
    return result


def run_caption() -> None:
    _check_ollama()

    npz = Path(str(CACHE_PATH) + ".npz")
    if not npz.exists():
        print(f"FATAL: no index cache at {npz}", file=sys.stderr)
        sys.exit(1)
    if not VIDEO.exists():
        print(f"FATAL: no video at {VIDEO} (needed for seek-based frame fetch)", file=sys.stderr)
        sys.exit(1)
    if not CAPTURE_JSONL.exists():
        print(f"FATAL: diag_v2 capture artifact missing: {CAPTURE_JSONL} (run diag_v2 --capture first)", file=sys.stderr)
        sys.exit(1)

    records = _load_capture()
    blip_caption_by_frame: dict[int, str] = {}
    for r in records:
        if r["fact_text"] is None:
            continue
        frame_idx, _ts = _fact_header(r["fact_text"])
        blip_caption_by_frame[frame_idx] = r["caption_only"]

    frame_idxs = sorted(blip_caption_by_frame.keys())
    print(f"distinct frame_idx values: {len(frame_idxs)}")

    idx = _load_index()
    print(f"fetching {len(frame_idxs)} frames via seek...")
    pil_by_frame = _fetch_frames_pil(idx, frame_idxs)

    results: dict[int, dict] = {}
    for i, frame_idx in enumerate(frame_idxs):
        b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
        t0 = time.time()
        caption = _caption_with_moondream(b64)
        elapsed = time.time() - t0
        results[frame_idx] = {
            "frame_idx": frame_idx,
            "blip_caption": blip_caption_by_frame[frame_idx],
            "moondream_caption": caption,
            "seconds": elapsed,
        }
        print(f"  [{i + 1}/{len(frame_idxs)}] frame_idx={frame_idx} {elapsed:.2f}s moondream={caption!r}")

    with open(CAPTIONS_JSONL, "w", encoding="utf-8") as fh:
        for frame_idx in frame_idxs:
            fh.write(json.dumps(results[frame_idx]) + "\n")
    print(f"wrote {len(results)} rows to {CAPTIONS_JSONL}")

    # ── determinism check ────────────────────────────────────────────────────
    random.seed(DETERMINISM_SEED)
    sample = random.sample(frame_idxs, min(N_DETERMINISM_SAMPLE, len(frame_idxs)))
    print(f"determinism check: re-captioning frame_idxs={sample}")

    mismatches = []
    for frame_idx in sample:
        b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
        caption_second = _caption_with_moondream(b64)
        caption_first = results[frame_idx]["moondream_caption"]
        if caption_second != caption_first:
            mismatches.append((frame_idx, caption_first, caption_second))

    if mismatches:
        print(f"FATAL: determinism check failed for {len(mismatches)}/{len(sample)} frame(s):", file=sys.stderr)
        for frame_idx, c1, c2 in mismatches:
            print(f"  frame_idx={frame_idx}", file=sys.stderr)
            print(f"    first pass:  {c1!r}", file=sys.stderr)
            print(f"    second pass: {c2!r}", file=sys.stderr)
        sys.exit(1)

    print(f"determinism check OK: {len(sample)}/{len(sample)} repeats matched exactly")


# ─────────────────────────────────────────────────────────────────────────────
# --stability: two more captioning passes (num_thread=1) over the same 55
# frames as the frozen pass1, then classify every (pass_i, pass_j) caption
# pair per frame as verbatim / paraphrase / divergent. No stop on divergence
# -- the divergent rate is the finding, at any value.
# ─────────────────────────────────────────────────────────────────────────────

def _classify_caption_pair(gate, cap_i: str, cap_j: str) -> tuple[str, str, str]:
    """verbatim (identical strings); paraphrase (different strings, no NLI
    contradiction in either direction); divergent (NLI contradiction in
    either direction -- the model disagrees with itself about content)."""
    if cap_i == cap_j:
        return "verbatim", "n/a", "n/a"
    label_i_premise, _ = _score_kernel_pair(gate, claim=cap_j, fact=cap_i)  # cap_i as premise
    label_j_premise, _ = _score_kernel_pair(gate, claim=cap_i, fact=cap_j)  # cap_j as premise
    if label_i_premise == "contradiction" or label_j_premise == "contradiction":
        return "divergent", label_i_premise, label_j_premise
    return "paraphrase", label_i_premise, label_j_premise


def run_stability() -> None:
    _check_ollama()

    if not PASS1_JSONL.exists():
        print(f"FATAL: frozen pass1 artifact missing: {PASS1_JSONL} (run --caption first, "
              f"then rename its output to this filename)", file=sys.stderr)
        sys.exit(1)

    npz = Path(str(CACHE_PATH) + ".npz")
    if not npz.exists():
        print(f"FATAL: no index cache at {npz}", file=sys.stderr)
        sys.exit(1)
    if not VIDEO.exists():
        print(f"FATAL: no video at {VIDEO} (needed for seek-based frame fetch)", file=sys.stderr)
        sys.exit(1)

    pass1_rows = []
    with open(PASS1_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pass1_rows.append(json.loads(line))

    frame_idxs = sorted(r["frame_idx"] for r in pass1_rows)
    blip_by_frame = {r["frame_idx"]: r["blip_caption"] for r in pass1_rows}
    pass_captions: dict[int, dict[int, str]] = {1: {r["frame_idx"]: r["moondream_caption"] for r in pass1_rows}}
    print(f"loaded frozen pass1: {len(frame_idxs)} frames from {PASS1_JSONL}")

    idx = _load_index()
    pil_by_frame = _fetch_frames_pil(idx, frame_idxs)

    for pass_n in (2, 3):
        print(f"=== CAPTIONING PASS {pass_n} (num_thread=1) ===")
        results: dict[int, dict] = {}
        for i, frame_idx in enumerate(frame_idxs):
            b64 = _image_to_b64_jpeg(pil_by_frame[frame_idx])
            t0 = time.time()
            caption = _caption_with_moondream(b64, extra_options={"num_thread": 1})
            elapsed = time.time() - t0
            results[frame_idx] = {
                "frame_idx": frame_idx,
                "blip_caption": blip_by_frame[frame_idx],
                "moondream_caption": caption,
                "seconds": elapsed,
            }
            print(f"  [{i + 1}/{len(frame_idxs)}] frame_idx={frame_idx} {elapsed:.2f}s moondream={caption!r}")

        path = _pass_jsonl_path(pass_n)
        with open(path, "w", encoding="utf-8") as fh:
            for frame_idx in frame_idxs:
                fh.write(json.dumps(results[frame_idx]) + "\n")
        print(f"wrote {len(results)} rows to {path}")
        pass_captions[pass_n] = {fi: results[fi]["moondream_caption"] for fi in frame_idxs}

    # ── classification ──────────────────────────────────────────────────────
    print("=" * 100)
    print("PER-FRAME CLASSIFICATION: (pass_i, pass_j) caption pairs")
    print("=" * 100)

    gate = _get_nli_gate()
    pair_keys = [(1, 2), (1, 3), (2, 3)]
    counts = {pk: {"verbatim": 0, "paraphrase": 0, "divergent": 0} for pk in pair_keys}
    divergent_examples = []

    for frame_idx in frame_idxs:
        for (pi, pj) in pair_keys:
            cap_i = pass_captions[pi][frame_idx]
            cap_j = pass_captions[pj][frame_idx]
            cls, label_i_premise, label_j_premise = _classify_caption_pair(gate, cap_i, cap_j)
            counts[(pi, pj)][cls] += 1
            print(f"  frame_idx={frame_idx} pass{pi}_vs_pass{pj}: {cls}")
            if cls == "divergent":
                divergent_examples.append((frame_idx, pi, pj, cap_i, cap_j, label_i_premise, label_j_premise))
    print()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    total_verbatim = total_paraphrase = total_divergent = 0
    for pk in pair_keys:
        c = counts[pk]
        n = c["verbatim"] + c["paraphrase"] + c["divergent"]
        print(f"  pass{pk[0]}_vs_pass{pk[1]}: verbatim={c['verbatim']}/{n} "
              f"paraphrase={c['paraphrase']}/{n} divergent={c['divergent']}/{n} "
              f"(divergent rate={c['divergent'] / n:.4f})")
        total_verbatim += c["verbatim"]
        total_paraphrase += c["paraphrase"]
        total_divergent += c["divergent"]
    total = total_verbatim + total_paraphrase + total_divergent
    print()
    print(f"  DIVERGENT RATE (all {len(pair_keys)} pair-types x {len(frame_idxs)} frames = {total} pair-instances): "
          f"{total_divergent}/{total} = {total_divergent / total:.4f}")
    print()

    if divergent_examples:
        print("  -- DIVERGENT EXAMPLES --")
        for frame_idx, pi, pj, cap_i, cap_j, label_i_premise, label_j_premise in divergent_examples:
            print(f"    frame_idx={frame_idx} pass{pi}_vs_pass{pj}")
            print(f"      pass{pi}: {cap_i!r}")
            print(f"      pass{pj}: {cap_j!r}")
            print(f"      NLI(premise=pass{pi}, hyp=pass{pj})={label_i_premise}  "
                  f"NLI(premise=pass{pj}, hyp=pass{pi})={label_j_premise}")
        print()

    # pass1 used default threading (and --caption's determinism check already
    # found 3/5 mismatches under it); pass2/pass3 both pin num_thread=1, so
    # the (2,3) rate isolates whether that alone removes the drift.
    rate_2_3 = counts[(2, 3)]["divergent"] / (counts[(2, 3)]["verbatim"] + counts[(2, 3)]["paraphrase"] + counts[(2, 3)]["divergent"])
    print("  -- DOES num_thread=1 REMOVE THE DRIFT? --")
    print(f"  pass2_vs_pass3 (both num_thread=1) divergent rate: {rate_2_3:.4f}")
    if rate_2_3 == 0.0:
        print("  READ: no divergence between two num_thread=1 runs -- pinning threads may explain")
        print("  (or at least correlate with) the earlier drift under default threading.")
    else:
        print("  READ: divergence PERSISTS even with num_thread=1 on both passes -- thread count is")
        print("  not the (sole) source of nondeterminism; something else in the runtime/sampler is.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — --analyze (offline; may load DeBERTa NLI model; no Ollama calls)
# ─────────────────────────────────────────────────────────────────────────────

def _load_v4_captions(pass_n: int = 1) -> list[dict]:
    path = _pass_jsonl_path(pass_n)
    if not path.exists():
        print(f"FATAL: {path} missing (run --caption for pass1, --stability for pass2/pass3)", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def section_a_diversity(rows: list[dict]) -> None:
    print("=" * 100)
    print("(A) DIVERSITY: BLIP vs Moondream over the same 55 frames")
    print("=" * 100)

    n = len(rows)
    blip_caps = [r["blip_caption"] for r in rows]
    moon_caps = [r["moondream_caption"] for r in rows]

    def _report(label: str, caps: list[str]) -> None:
        counts: dict[str, int] = {}
        for c in caps:
            counts[c] = counts.get(c, 0) + 1
        distinct = len(counts)
        ratio = distinct / len(caps) if caps else float("nan")
        mean_len = sum(len(c) for c in caps) / len(caps) if caps else float("nan")
        top5 = sorted(counts.items(), key=lambda kv: -kv[1])[:5]

        print(f"  -- {label} --")
        print(f"    distinct-caption ratio: {distinct}/{len(caps)} = {ratio:.4f}")
        print(f"    mean caption length (chars): {mean_len:.1f}")
        print(f"    top-5 most repeated:")
        for cap, count in top5:
            print(f"      x{count}: {cap!r}")
        print()

    _report("BLIP", blip_caps)
    _report("Moondream", moon_caps)
    print(f"  n frames compared: {n}")
    print()


def section_b_bound_nli_replay(v4_rows: list[dict]) -> int:
    print("=" * 100)
    print("(B) BOUND-NLI REPLAY: normalized kernel vs Moondream caption (substituted premise)")
    print("=" * 100)

    moondream_by_frame = {r["frame_idx"]: r["moondream_caption"] for r in v4_rows}

    records = _load_capture()
    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    gate = _get_nli_gate()
    binding_label: dict[tuple, str] = {}

    for key, rows in real_groups.items():
        cited_rows = _resolve_citations(key[1], rows)
        if not cited_rows:
            continue
        stripped, _fields, _mangled = _decompose_claim(key[1])
        normalized = _normalize_kernel(stripped)

        scored = []
        for row in cited_rows:
            frame_idx, _ts = _fact_header(row["fact_text"])
            moondream_caption = moondream_by_frame.get(frame_idx)
            if moondream_caption is None:
                print(f"FATAL: no Moondream caption for cited frame_idx={frame_idx} "
                      f"(claim={key[1]!r})", file=sys.stderr)
                sys.exit(1)
            label, score = _score_kernel_pair(gate, normalized, moondream_caption)
            scored.append((score, label, frame_idx, moondream_caption))
        scored.sort(key=lambda x: -x[0])
        top_score, top_label, top_frame_idx, top_caption = scored[0]
        binding_label[key] = top_label

        print(f"    query={key[0]!r}")
        print(f"    normalized_kernel={normalized!r}")
        for score, label, frame_idx, caption in scored:
            print(f"      vs frame_idx={frame_idx} moondream_caption={caption!r} -> label={label} score={score:.4f}")
        print(f"    ARGMAX: frame_idx={top_frame_idx} label={top_label} score={top_score:.4f}")
        print()

    print("=" * 100)
    print("VERDICT TABLE (Moondream-substituted)")
    print("=" * 100)
    hdr = f"  {'claim':<45} | {'class':<18} | {'live':<13} | {'binding_moondream':<20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, rows in real_groups.items():
        live = rows[0]["live_verdict_for_claim"]
        cls = _classify_claim(key[1])
        if key in binding_label:
            top_label = binding_label[key]
            if top_label == "entailment":
                verdict = "verified_kernel"
            elif top_label == "contradiction":
                verdict = "rejected"
            else:
                verdict = "unverifiable"
        else:
            verdict = "no_citation(answer-semantics)"
        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        print(f"  {claim_disp:<45} | {cls:<18} | {live:<13} | {verdict:<20}")
    print()

    n_entailments = sum(1 for v in binding_label.values() if v == "entailment")
    print("  -- SUMMARY --")
    print(f"  entailments under Moondream-substituted bound-NLI replay: {n_entailments}")
    print(f"  BLIP baseline (--kernel-normalize):                       {BLIP_BASELINE_ENTAILMENTS}")
    if n_entailments > BLIP_BASELINE_ENTAILMENTS:
        print(f"  READ: Moondream recovers MORE entailments than BLIP ({n_entailments} > {BLIP_BASELINE_ENTAILMENTS}).")
    elif n_entailments == BLIP_BASELINE_ENTAILMENTS:
        print("  READ: Moondream ties the BLIP baseline -- no improvement from the caption swap alone.")
    else:
        print(f"  READ: Moondream recovers FEWER entailments than BLIP ({n_entailments} < {BLIP_BASELINE_ENTAILMENTS}).")
    print()
    return n_entailments


def section_c_fabricated_probe(v4_rows: list[dict]) -> None:
    print("=" * 100)
    print("(C) FABRICATED PROBE: 6 fabricated claims x every Moondream caption")
    print("=" * 100)

    gate = _get_nli_gate()
    entailed_pairs = []
    n_pairs = 0
    for claim in FABRICATED_CLAIMS:
        for row in v4_rows:
            n_pairs += 1
            label, score = _score_kernel_pair(gate, claim, row["moondream_caption"])
            if label == "entailment":
                entailed_pairs.append((claim, row["frame_idx"], row["moondream_caption"], score))

    print(f"  n (claim, caption) pairs scored: {n_pairs}")
    print(f"  BLIP baseline fabricated accept rate (diag_v2 section E): 0.0000")
    print(f"  Moondream entailed pairs found: {len(entailed_pairs)}")
    print()

    if entailed_pairs:
        print("  ** RED FLAG: fabricated claim(s) entailed by a Moondream caption **", file=sys.stderr)
        print("  ** RED FLAG: fabricated claim(s) entailed by a Moondream caption **")
        for claim, frame_idx, caption, score in entailed_pairs:
            print(f"    claim={claim!r}")
            print(f"    frame_idx={frame_idx} moondream_caption={caption!r} entailment_score={score:.4f}")
            print()
        print("STOP: fabricated-claim entailment found -- surfacing, not tuning around.", file=sys.stderr)
        sys.exit(1)
    else:
        print("  No fabricated claim entailed by any Moondream caption -- consistent with BLIP baseline.")
    print()


def section_d_cost(v4_rows: list[dict]) -> None:
    print("=" * 100)
    print("(D) COST")
    print("=" * 100)

    seconds = [r["seconds"] for r in v4_rows]
    mean_s = statistics.mean(seconds)
    median_s = statistics.median(seconds)
    max_s = max(seconds)

    print(f"  n captions: {len(seconds)}")
    print(f"  mean seconds/caption:   {mean_s:.3f}")
    print(f"  median seconds/caption: {median_s:.3f}")
    print(f"  max seconds/caption:    {max_s:.3f}")
    print()

    cold_latency = TOP_K_FOR_LATENCY_PROJECTION * mean_s
    cached_latency = 0.0
    print(f"  projected per-query latency at top_k={TOP_K_FOR_LATENCY_PROJECTION}:")
    print(f"    cold (all {TOP_K_FOR_LATENCY_PROJECTION} frames uncached): {cold_latency:.2f}s")
    print(f"    cached (all {TOP_K_FOR_LATENCY_PROJECTION} frames already captioned this session): {cached_latency:.2f}s")
    print(f"    (matches iris/query.py:_ensure_captions -- a frame is captioned at most once per")
    print(f"     loaded index, regardless of how many queries retrieve it)")
    print()


def run_analyze(pass_n: int = 1) -> None:
    print("#" * 100)
    print(f"# ANALYZING PASS {pass_n}  ({_pass_jsonl_path(pass_n)})")
    print("#" * 100)
    print()

    v4_rows = _load_v4_captions(pass_n)

    section_a_diversity(v4_rows)
    section_b_bound_nli_replay(v4_rows)
    section_c_fabricated_probe(v4_rows)
    section_d_cost(v4_rows)


# ─────────────────────────────────────────────────────────────────────────────
# --split-premise: offline (pass1/pass2 jsonl + capture files; loads DeBERTa
# NLI model; no Ollama calls; deterministic per pass). No STOP conditions --
# every outcome is a finding.
# ─────────────────────────────────────────────────────────────────────────────

LEN_BUCKETS = [("<=30", 0, 30), ("31-60", 31, 60), (">60", 61, None)]


def _split_sentences(nlp, text: str) -> list[str]:
    """Sentence split via spaCy's sentencizer (parser-derived sentence
    boundaries -- already loaded as part of en_core_web_sm). Collapses \\n\\n
    paragraph breaks (and any other whitespace run) to a single space before
    splitting, then drops empty/whitespace-only segments."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    doc = nlp(normalized)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _score_claim_vs_sentences(gate, claim: str, premise_text: str) -> dict:
    """Split-premise scoring: score claim against EACH sentence of premise_text
    with the existing per-pair NLI logic. Pair verdict = entailment if ANY
    sentence entails; else contradiction if ANY sentence contradicts (an
    extension of the spec's explicit entailment rule, needed so split-premise
    verdicts remain comparable to the existing verified/rejected/unverifiable
    mapping); else neutral. Reports the max-entailment-score sentence as the
    representative one, regardless of which rule decided the overall label."""
    nlp = gate._get_spacy()
    tokenizer, _model = gate._get_nli_model()
    sentences = _split_sentences(nlp, premise_text)

    per_sentence = []
    for s in sentences:
        label, score = _score_kernel_pair(gate, claim, s)
        tok = _token_count(tokenizer, s)
        per_sentence.append((s, label, score, tok))

    if not per_sentence:
        return {"label": "neutral", "best_score": 0.0, "best_sentence": None,
                "best_sentence_tokens": 0, "per_sentence": []}

    entailed = [p for p in per_sentence if p[1] == "entailment"]
    contradicted = [p for p in per_sentence if p[1] == "contradiction"]

    if entailed:
        best = max(entailed, key=lambda p: p[2])
        overall_label = "entailment"
    elif contradicted:
        best = max(contradicted, key=lambda p: p[2])
        overall_label = "contradiction"
    else:
        best = max(per_sentence, key=lambda p: p[2])
        overall_label = "neutral"

    return {
        "label": overall_label,
        "best_score": best[2],
        "best_sentence": best[0],
        "best_sentence_tokens": best[3],
        "per_sentence": per_sentence,
    }


def _compute_fabricated_pair_data(gate, tokenizer, v4_rows: list[dict]) -> list[dict]:
    """Whole-premise + split-premise scoring for every (fabricated claim,
    Moondream caption) pair, computed once and shared by (C-split) and (LEN)."""
    pair_data = []
    for claim in FABRICATED_CLAIMS:
        for row in v4_rows:
            caption = row["moondream_caption"]
            whole_label, whole_score = _score_kernel_pair(gate, claim, caption)
            tokens = _token_count(tokenizer, caption)
            split_result = _score_claim_vs_sentences(gate, claim, caption)
            pair_data.append({
                "claim": claim,
                "frame_idx": row["frame_idx"],
                "caption": caption,
                "tokens": tokens,
                "whole_label": whole_label,
                "whole_score": whole_score,
                "split_result": split_result,
            })
    return pair_data


def section_c_split_fabricated_probe(pair_data: list[dict], pass_n: int) -> None:
    print("=" * 100)
    print(f"(C-split) FABRICATED PROBE, SPLIT PREMISES -- pass {pass_n}")
    print("=" * 100)

    whole_accepted = [p for p in pair_data if p["whole_label"] == "entailment"]
    split_accepted = [p for p in pair_data if p["split_result"]["label"] == "entailment"]

    print(f"  n pairs: {len(pair_data)}")
    print(f"  whole-premise accepted (entailment): {len(whole_accepted)}")
    print()

    for p in whole_accepted:
        print(f"    CLAIM: {p['claim']!r}")
        print(f"    frame_idx={p['frame_idx']}")
        print(f"    FULL PREMISE (verbatim, as fed to the model): {p['caption']!r}")
        print(f"    premise token count: {p['tokens']}")
        print(f"    whole-premise: label=entailment score={p['whole_score']:.4f}")
        print(f"    -- per-sentence table --")
        for s, label, score, tok in p["split_result"]["per_sentence"]:
            print(f"      sentence={s!r} tokens={tok} label={label} score={score:.4f}")
        print(f"    split-premise verdict: {p['split_result']['label']} "
              f"(best_score={p['split_result']['best_score']:.4f}, "
              f"best_sentence={p['split_result']['best_sentence']!r})")
        print()

    print(f"  whole-premise accept count: {len(whole_accepted)}/{len(pair_data)}")
    print(f"  split-premise accept count: {len(split_accepted)}/{len(pair_data)}")
    if split_accepted:
        print("  -- split-premise accepted pairs not caught by whole-premise (if any) --")
        whole_keys = {(p["claim"], p["frame_idx"]) for p in whole_accepted}
        new_ones = [p for p in split_accepted if (p["claim"], p["frame_idx"]) not in whole_keys]
        if new_ones:
            for p in new_ones:
                print(f"    NEW under split: claim={p['claim']!r} frame_idx={p['frame_idx']} "
                      f"best_sentence={p['split_result']['best_sentence']!r} "
                      f"score={p['split_result']['best_score']:.4f}")
        else:
            print("    (none -- every split-premise accept was already a whole-premise accept)")
    print()


def section_len_premise_length(pair_data: list[dict], pass_n: int) -> None:
    print("=" * 100)
    print(f"(LEN) PREMISE-LENGTH vs JUDGMENT -- pass {pass_n} (length-degradation hypothesis test)")
    print("=" * 100)

    hdr = f"  {'bucket':<8} | {'n_pairs':>7} | {'n_entailed':>10} | {'entailment_rate':>15}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, lo, hi in LEN_BUCKETS:
        if hi is None:
            bucket_pairs = [p for p in pair_data if p["tokens"] >= lo]
        else:
            bucket_pairs = [p for p in pair_data if lo <= p["tokens"] <= hi]
        n = len(bucket_pairs)
        n_entailed = sum(1 for p in bucket_pairs if p["whole_label"] == "entailment")
        rate = n_entailed / n if n else float("nan")
        print(f"  {label:<8} | {n:>7} | {n_entailed:>10} | {rate:>15.4f}" if n else
              f"  {label:<8} | {n:>7} | {n_entailed:>10} | {'n/a':>15}")
    print()

    rates = []
    for label, lo, hi in LEN_BUCKETS:
        bucket_pairs = [p for p in pair_data if (p["tokens"] >= lo and (hi is None or p["tokens"] <= hi))]
        if bucket_pairs:
            rates.append((label, sum(1 for p in bucket_pairs if p["whole_label"] == "entailment") / len(bucket_pairs)))
    print("  -- READ --")
    if len(rates) >= 2 and rates[-1][1] > rates[0][1]:
        print(f"  entailment rate rises from {rates[0][0]} ({rates[0][1]:.4f}) to {rates[-1][0]} ({rates[-1][1]:.4f}) "
              f"-- consistent with the length-degradation hypothesis.")
    elif len(rates) >= 2 and rates[-1][1] < rates[0][1]:
        print(f"  entailment rate FALLS from {rates[0][0]} ({rates[0][1]:.4f}) to {rates[-1][0]} ({rates[-1][1]:.4f}) "
              f"-- contradicts the length-degradation hypothesis as stated.")
    else:
        print("  no clear monotonic trend across buckets at this sample size.")
    print()


def section_b_split_bound_nli_replay(v4_rows: list[dict], pass_n: int) -> None:
    print("=" * 100)
    print(f"(B-split whole-premise baseline) -- pass {pass_n}")
    print("=" * 100)
    whole_n_entailments = section_b_bound_nli_replay(v4_rows)

    print("=" * 100)
    print(f"(B-split) BOUND-NLI REPLAY, SPLIT PREMISES -- pass {pass_n}")
    print("=" * 100)

    moondream_by_frame = {r["frame_idx"]: r["moondream_caption"] for r in v4_rows}
    records = _load_capture()
    groups = _group_by_claim(records)
    real_groups = {k: v for k, v in groups.items() if k[2] == "real"}

    gate = _get_nli_gate()
    binding_label: dict[tuple, str] = {}

    for key, rows in real_groups.items():
        cited_rows = _resolve_citations(key[1], rows)
        if not cited_rows:
            continue
        stripped, _fields, _mangled = _decompose_claim(key[1])
        normalized = _normalize_kernel(stripped)

        scored = []
        for row in cited_rows:
            frame_idx, _ts = _fact_header(row["fact_text"])
            caption = moondream_by_frame.get(frame_idx)
            if caption is None:
                print(f"FATAL: no Moondream caption for cited frame_idx={frame_idx} "
                      f"(claim={key[1]!r})", file=sys.stderr)
                sys.exit(1)
            res = _score_claim_vs_sentences(gate, normalized, caption)
            scored.append((res["best_score"], res["label"], frame_idx, res["best_sentence"]))
        scored.sort(key=lambda x: -x[0])
        top_score, top_label, top_frame_idx, top_sentence = scored[0]
        binding_label[key] = top_label

        print(f"    query={key[0]!r}")
        print(f"    normalized_kernel={normalized!r}")
        for score, label, frame_idx, sentence in scored:
            print(f"      vs frame_idx={frame_idx} best_sentence={sentence!r} -> label={label} score={score:.4f}")
        print(f"    ARGMAX: frame_idx={top_frame_idx} label={top_label} score={top_score:.4f} "
              f"sentence={top_sentence!r}")
        print()

    print("=" * 100)
    print("VERDICT TABLE (split-premise)")
    print("=" * 100)
    hdr = f"  {'claim':<45} | {'class':<18} | {'live':<13} | {'binding_split':<20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for key, rows in real_groups.items():
        live = rows[0]["live_verdict_for_claim"]
        cls = _classify_claim(key[1])
        if key in binding_label:
            top_label = binding_label[key]
            if top_label == "entailment":
                verdict = "verified_kernel"
            elif top_label == "contradiction":
                verdict = "rejected"
            else:
                verdict = "unverifiable"
        else:
            verdict = "no_citation(answer-semantics)"
        claim_disp = (key[1][:42] + "...") if len(key[1]) > 45 else key[1]
        print(f"  {claim_disp:<45} | {cls:<18} | {live:<13} | {verdict:<20}")
    print()

    n_entailments = sum(1 for v in binding_label.values() if v == "entailment")
    print("  -- SUMMARY --")
    print(f"  entailments under split-premise bound-NLI replay: {n_entailments}")
    print(f"  entailments under whole-premise bound-NLI replay (this pass): {whole_n_entailments}")
    print(f"  BLIP baseline (--kernel-normalize):                           {BLIP_BASELINE_ENTAILMENTS}")
    print()


def run_split_premise() -> None:
    for pass_n in (1, 2):
        print("#" * 100)
        print(f"# --split-premise: PASS {pass_n}  ({_pass_jsonl_path(pass_n)})")
        print("#" * 100)
        print()

        v4_rows = _load_v4_captions(pass_n)
        gate = _get_nli_gate()
        tokenizer, _model = gate._get_nli_model()

        pair_data = _compute_fabricated_pair_data(gate, tokenizer, v4_rows)

        section_c_split_fabricated_probe(pair_data, pass_n)
        section_b_split_bound_nli_replay(v4_rows, pass_n)
        section_len_premise_length(pair_data, pass_n)


def main() -> None:
    if "--caption" in sys.argv:
        run_caption()
    elif "--stability" in sys.argv:
        run_stability()
    elif "--split-premise" in sys.argv:
        run_split_premise()
    elif "--analyze" in sys.argv:
        pass_n = 1
        if "--pass" in sys.argv:
            idx = sys.argv.index("--pass")
            if idx + 1 >= len(sys.argv):
                print("FATAL: --pass requires a value in {1, 2, 3}", file=sys.stderr)
                sys.exit(1)
            try:
                pass_n = int(sys.argv[idx + 1])
            except ValueError:
                print(f"FATAL: --pass value must be an integer in {{1, 2, 3}}, got {sys.argv[idx + 1]!r}", file=sys.stderr)
                sys.exit(1)
            if pass_n not in (1, 2, 3):
                print(f"FATAL: --pass must be 1, 2, or 3, got {pass_n}", file=sys.stderr)
                sys.exit(1)
        run_analyze(pass_n)
    else:
        print("Usage: python scripts/diag_v4_moondream_captions.py --caption | --stability | "
              "--analyze [--pass {1,2,3}] | --split-premise")
        sys.exit(1)


if __name__ == "__main__":
    main()
