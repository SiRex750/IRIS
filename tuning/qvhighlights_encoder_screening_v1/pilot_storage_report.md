# QVHighlights Pilot Video Acquisition — Storage/Feasibility Report

## Status: BLOCKED on individual acquisition; bulk archive is the only official path

## 1. Feasibility check (required before any download attempt)

Inspected the full file tree of the official repository at the pinned
commit (`b7e553ac3b0c898ee6b85e03ee507c064eab89ca`,
https://github.com/jayleicn/moment_detr). There is:

- no per-video download script or tool anywhere in the repo;
- no published index of per-clip byte offsets or individual URLs;
- only one official raw-video source: the single archive
  `https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz`.

`data/README.md` documents a *YouTube-embed viewing URL* reconstructed from
`vid` (`{youtube_id}_{start}_{end}` → `youtube.com/embed/{id}?start=...&end=...`).
This is not an official redistribution path for the dataset: it points at
the live YouTube original (which may since have been edited, re-encoded,
deleted, or region-restricted), and player-embed start/end parameters do
not guarantee frame-accurate reproduction of the exact trimmed clip
shipped in the official archive. Downloading via that route (e.g. with
`yt-dlp` + `ffmpeg` trim) would risk silently substituting
differently-trimmed footage for the official one, which the task
explicitly forbids.

**Conclusion: selective/individual video acquisition is not officially
supported. The complete archive is required.** No download was attempted
for any of the 200 pilot videos; `pilot_selection.json` /
`pilot_selection.csv` record `download_status: "not_attempted"` and
`video_available: false` for every row (unchanged from the existing
`external_dev` manifest — nothing was fabricated).

## 2. Archive facts

| Field | Value |
|---|---|
| Archive URL | `https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz` |
| Reported size (`Content-Length`, verified via `HEAD`) | 143,734,787,897 bytes (≈133.8 GiB) |
| Expected extracted size | Not officially published. Estimate: **≈130–150 GiB** — MP4 video is already compressed, so gzip wrapping typically shrinks it very little; extracted size should be close to the archive size, not several times larger. Treat as an estimate, not a measured value. |
| Contains | Video clips for train + val + test `vid`s combined (single flat archive; splits are not separated at the file level) |

## 3. Recommended external storage capacity

To hold the archive **and** its extraction simultaneously (the task
requires preserving the archive after extraction unless the user removes
it later):

- Archive: ~134 GiB
- Extraction: ~130–150 GiB (estimate, see above)
- Working headroom (temp files, `.part` staging, filesystem overhead): ~20 GiB
- **Recommended minimum free space: 300 GiB** on whatever external volume is used.

Current local machine: only ~30 GiB free. Do not attempt this internally.

## 4. Exact command for downloading directly to an external drive

Assume the external drive is mounted at `/mnt/qvh_external` (adjust to the
actual mount point):

```bash
# 1. Confirm the mount and free space
df -h /mnt/qvh_external

# 2. Create the external data root (mirrors the existing repo convention)
mkdir -p /mnt/qvh_external/qvhighlights/{annotations,videos,archives,manifests,logs}

# 3. Run the existing setup script pointed at the external root.
#    This reuses the same resumable/idempotent/safe-extraction logic
#    already implemented and tested in scripts/setup_qvhighlights.py --
#    no new download code is needed.
python3 scripts/setup_qvhighlights.py --download-videos --resume \
    --data-root /mnt/qvh_external/qvhighlights
```

This downloads annotations (if not already present at that root),
downloads the video archive with resumable `.part` staging, verifies size,
computes SHA-256, and safely extracts it (rejecting absolute paths and
`..` traversal, confined to `/mnt/qvh_external/qvhighlights/videos/`).

## 5. Required configuration/path changes

- Point `--data-root` at the external mount for every subsequent command
  (`setup_qvhighlights.py`, and eventually the pilot decode-validation step)
  instead of the repo-local `external_data/qvhighlights`.
- `tuning/qvhighlights_encoder_screening_v1/pilot_selection.json` already
  stores `local_path` as an absolute path under the repo's
  `external_data/qvhighlights/videos/` root. After extracting to the
  external drive, either:
  - symlink `external_data/qvhighlights/videos` → the external drive's
    `videos/` directory (keeps existing manifests/paths valid, and the
    external target stays outside Git since it's a symlink to
    already-gitignored content), or
  - regenerate the dev/pilot manifests with `--data-root` pointed at the
    external location so `local_path` reflects the real mount.
- No repo code changes are required — the script already accepts
  `--data-root` as a parameter for exactly this purpose.

## 6. How to validate the completed extraction

```bash
# Re-run in --verify-only mode against the external root to recompute
# hashes/stats without re-downloading anything.
python3 scripts/setup_qvhighlights.py --verify-only \
    --data-root /mnt/qvh_external/qvhighlights

# Confirm the archive SHA-256 recorded in the validation manifest is stable
# across the download (script computes and stores this automatically).
python3 -c "import json; d=json.load(open('/mnt/qvh_external/qvhighlights/manifests/qvhighlights_validation.json')); print(d['video_archive'])"

# Spot-check a handful of extracted files with ffprobe before trusting the
# whole set (full per-file validation happens in the pilot decode step,
# limited to the 200 selected videos -- see below).
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 \
    /mnt/qvh_external/qvhighlights/videos/<some_vid>.mp4
```

## 7. How to resume the pilot acquisition step afterward

1. Re-run `scripts/select_qvhighlights_pilot.py` if desired — it is
   deterministic (fixed seed 42) and will reproduce the exact same 200
   `video_id`s already recorded in `pilot_selection.json`, so this step
   does not need to be repeated; the existing selection file remains
   valid.
2. Once the external drive has the extracted videos, for each of the 200
   selected `video_id`s: locate `{video_id}.mp4` under the external
   `videos/` directory, run `ffprobe` to check container readability,
   duration (within documented tolerance of the annotated `duration`),
   presence of a video stream, and successful decode of frames near
   start/middle/end.
3. Update `pilot_selection.json` / `.csv` fields (`download_status`,
   `local_path`, `file_size_bytes`, `sha256`, `decoding_validation_status`,
   `video_available`) only from what is actually observed on disk —
   never set `video_available: true` without a validated file present.
4. If ≥100 of the 200 pass validation, proceed to the pilot smoke tests
   and the `PILOT_ONLY`-labeled zero-shot encoder screening described in
   the task. If fewer than 100 pass, report the shortfall and storage
   requirements instead of running any accuracy comparison.

## 8. What was NOT done in this step

- No video bytes were downloaded, internally or externally.
- No `video_available` flag was changed from its existing (`false`) value.
- No encoder-screening experiment, smoke test, or accuracy comparison was run.
- No temporal-adapter training was started.
