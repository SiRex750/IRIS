# Machine setup notes (new-machine bring-up, 2026-07-24)

Recorded while reproducing the P-NOW-A VAL grounding cell on a fresh clone at `b446a7a`.

## val.csv source
`eval/data/nextqa/val.csv` is copied from `doc-doc/NExT-QA`'s `dataset/nextqa/val.csv`,
**NOT** from `doc-doc/NExT-GQA`'s `datasets/nextgqa/val.csv`. The NExT-GQA copy uses header
`video_id` and stores the answer as the literal answer string (e.g. `"unwrap it"`);
`eval/nextqa_loader.py::load_split` hard-reads `raw["video"]` and does `int(raw["answer"])`,
which only matches the original NExT-QA file (header `video`, 0-indexed int answer). Using
the NExT-GQA copy verbatim would KeyError / ValueError on load.

## gsub_val.json — two copies required
`gsub_val.json` must exist at BOTH:
  - `NExT-GQA/gsub_val.json` (first entry in `_GQA_CANDIDATES`, e.g. in
    `scripts/phase6_gqa_coverage_probe.py`)
  - `eval/data/nextqa/gsub_val.json` (hardcoded directly as `GQA_JSON` in
    `scripts/pnowa_width_topk_sweep.py`, no fallback search)
Both are gitignored and must be copied/fetched on every new machine. Same source file
(`doc-doc/NExT-GQA`'s `datasets/nextgqa/gsub_val.json`), placed at both paths.

## Grounding acceptance test — version sensitivity
Re-ran `scripts/pnowa_width_topk_sweep.py` (VAL half, 406 questions / 59 videos, decode-free
`index_cache_ssparse` rebuilt from `index_cache` via `scripts/_rebuild_ssparse_from_flat.py`).
`top_k=8, half_width=2.2` reproduced the committed numbers EXACTLY:
  peak_in_gold=0.3227  mIoP=0.3126  IoP@0.5=0.3202  mIoU=0.1486
CLIP-anchor fallback rate: 0/1624 (0%).
Installed on this run: `torch==2.13.0+cpu`, `numpy==2.5.1`, `scipy==1.18.0`,
`transformers==4.57.6`, `clip==1.0` — all newer than the `>=` floors in `requirements.txt`
(no lock file exists in the repo). Exact match despite the drift means this metric is
NOT sensitive to these library versions across this range.

## Open item — val.csv answer column unverified
Only the grounding path (CLIP retrieval + span prediction) has been exercised so far, which
never reads the `answer`/`a0..a4` columns. The MC-answering path (Acc@QA) that consumes those
columns is still unverified on this machine. Confirm by running the A6 answerer harness once
`llama-server` is copied over, and checking Acc@QA == 0.53125.
