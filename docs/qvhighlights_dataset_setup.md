# QVHighlights Dataset Setup

## 1. Why QVHighlights is being added

QVHighlights (Lei et al., NeurIPS 2021) is an external temporal-grounding
benchmark: natural-language queries paired with relevant temporal windows
in YouTube videos. It is being acquired here as an *external* dataset so
IRIS's temporal-grounding components can eventually be developed/validated
against data that is not NExT-GQA. This step covers acquisition and
structural validation **only** -- no reranker, adapter, video encoder, or
adaptive-span work happens here.

## 2. Official sources

- Repository: https://github.com/jayleicn/moment_detr
- Annotation README: https://github.com/jayleicn/moment_detr/blob/main/data/README.md
- Pinned commit used by `scripts/setup_qvhighlights.py`:
  `b7e553ac3b0c898ee6b85e03ee507c064eab89ca` (verified current against
  `main` on 2026-07-25; annotations are fetched from this exact commit via
  `raw.githubusercontent.com`, not a moving branch, for reproducibility).
- Annotation files fetched: `data/highlight_train_release.jsonl`,
  `data/highlight_val_release.jsonl`, `data/LICENSE`.
- Raw video archive (official, per `data/README.md`):
  `https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz`

## 3. Licence

The QVHighlights annotations are released under **CC BY-NC-SA 4.0**
(Attribution-NonCommercial-ShareAlike 4.0 International); full text at
`https://raw.githubusercontent.com/jayleicn/moment_detr/<commit>/data/LICENSE`
(mirrored locally at `external_data/qvhighlights/annotations/LICENSE` once
downloaded, not committed to Git -- see §6). **The dataset is for
non-commercial research use only, unless separately authorised.**

## 4. Exact setup commands

See `external_data/qvhighlights/manifests/setup_commands.txt` for the exact
commands run and `external_data/qvhighlights/logs/setup_output.txt` for the
captured terminal output of the annotation download.

```
# Annotations only (safe default; no raw videos)
python3 scripts/setup_qvhighlights.py --annotations-only \
    --data-root external_data/qvhighlights

# Re-verify existing downloads / regenerate manifests without downloading
python3 scripts/setup_qvhighlights.py --verify-only \
    --data-root external_data/qvhighlights

# Raw videos (NOT run in this task -- see §6 storage blocker)
python3 scripts/setup_qvhighlights.py --download-videos \
    --data-root external_data/qvhighlights
```

`--resume` may be added to any invocation to resume a partially downloaded
`.part` file. The script is idempotent: rerunning it never re-downloads a
valid, already-complete file and never duplicates manifest rows.

## 5. Expected directory structure

```
external_data/qvhighlights/
  annotations/   # raw train/val JSONL + LICENSE (downloaded, gitignored)
  videos/        # extracted .mp4 files, if --download-videos used (gitignored)
  archives/      # qvhilights_videos.tar.gz, preserved after extraction (gitignored)
  manifests/     # small JSONL/JSON manifests -- COMMITTED
  logs/          # setup_output.txt -- COMMITTED
```

## 6. Download and storage requirements

- Annotation files: ~4.6 MB total (train 3.96 MB, val 821 KB, LICENSE 21 KB).
- Raw video archive: server-reported `Content-Length` = **143,734,787,897
  bytes (~133.8 GiB)**, verified via `HEAD` request without downloading
  the body.
- At acquisition time, this machine had **~30 GiB free**. This is far
  short of the ~134 GiB archive alone (before extraction, which roughly
  doubles peak usage while both the archive and extracted files exist).
- **Blocker: raw video download was not performed.** Per the task's
  safety rules, video download only proceeds if there is verified
  sufficient storage; there was not. `--download-videos` was never
  invoked. This is a storage blocker, not a data-integrity or access
  problem -- the official archive URL responds normally.
- Only annotations (train + val) were downloaded and validated.

## 7. How validation works

`scripts/setup_qvhighlights.py` parses each JSONL row and checks: `qid`
uniqueness within its split, non-empty `query`, presence of `vid`, a
finite positive `duration`, well-formed `relevant_windows` (each a
`[start, end]` pair, `start >= 0`, `end > start`, `end <= duration + 0.5s`
tolerance, all values finite). Rows that fail are never dropped silently
-- they are counted by reason in `invalid_row_counts` in the validation
manifest. On the current train/val download, **zero rows failed
validation** in either split.

## 8. Which splits are permitted

- **train** -> used for training (later steps only; not used here).
- **validation** -> used for external development (`external_dev.jsonl`).
- **test** -> left completely untouched. This script's `ANNOTATION_FILES`
  mapping only contains `train` and `val`; `highlight_test_release.jsonl`
  and `highlight_test_with_gt.jsonl` are never requested. The validation
  manifest records `"test_split_used": false`.

## 9. NExT-GQA splits are out of scope here

`val_confirm`, `tuning/frozen_state.json`, and NExT-GQA `test` data were
not read, run, or modified by this task. This task only touches the new
`external_data/qvhighlights/` tree, `scripts/setup_qvhighlights.py`,
`tests/test_qvhighlights_setup.py`, and this documentation.

## 10. Known missing/corrupt-video limitations

Because raw videos were not downloaded, every row in `external_train.jsonl`
and `external_dev.jsonl` currently has `video_available: false`, and
`external_data/qvhighlights/manifests/missing_videos_report.json` lists
all 8,768 expected-but-absent videos. No corrupt-video check has run (that
requires the actual video files); `qvhighlights_validation.json` records
`n_videos_corrupt` as an explanatory string rather than a fabricated
count. `external_data/qvhighlights/manifests/video_validation.csv` will
only be produced once videos are actually present, per §7 of the task
spec (optional video validation) -- it does not exist yet.

## 11. How to rerun verification

```
python3 scripts/setup_qvhighlights.py --verify-only --data-root external_data/qvhighlights
python3 -m pytest tests/test_qvhighlights_setup.py -q
```

## 12. No model training or IRIS evaluation was performed

This task acquired and structurally validated QVHighlights train/validation
annotations only. No reranker, adapter, video encoder, or adaptive-span
method was implemented, trained, or tested; no IRIS retrieval metric was
computed; the existing retrieval pipeline, query reformulation, and
traversal modules were not modified. Those are separate, later steps.
