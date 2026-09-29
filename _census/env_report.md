# Environment + data inventory for Paper 3

Generated 2026-09-29. Read-only survey. The only file written is this one; temp encodes went to the session scratchpad and were deleted.

## 1. Tools

### ffmpeg / ffprobe
**FAIL: not on PATH.** Neither `which` (Git Bash) nor `Get-Command` (PowerShell) finds `ffmpeg` or `ffprobe`. `imageio_ffmpeg` is not installed in `.venv` either. So there is no ffmpeg CLI to query for `--enable-libx264` and similar flags; everything below comes from PyAV's bundled FFmpeg.

### PyAV (`.venv`, Python 3.13.3)
- PyAV **17.1.0**, bundled FFmpeg libs: libavcodec 62.28.101, libavformat 62.12.101, libavutil 60.26.101 (FFmpeg 8.x series)

| Encoder | In `av.codecs_available` | Opens as encoder |
|---|---|---|
| libx264 | yes | yes |
| libx265 | yes | yes |
| mpeg4 (MPEG-4 Part 2) | yes | yes |
| libaom-av1 | **no** | no (`UnknownCodecError`) |
| libsvtav1 | yes | yes |
| librav1e | no | no |
| h264_nvenc / hevc_nvenc / av1_nvenc | yes | yes (not exercised) |
| libdav1d | yes (decoder only) | n/a |

## 2. Motion-vector export test

Source: first 60 frames of `test_data/clips/people_detection.mp4` (768x432, 12 fps). Each frame was re-encoded at 30 fps with default encoder settings (libx264 `preset=medium`), then decoded with `codec_context.options = {"flags2": "+export_mvs"}`. Every temp file was confirmed deleted.

| Codec | Encoder → decoder | Frame types (60) | Frames with `MOTION_VECTORS` | MVs on frame 10 |
|---|---|---|---|---|
| H.264 | libx264 → h264 | I 1 / P 21 / B 38 | **59 / 60** (all but the I-frame) | **1817** |
| MPEG-4 Part 2 | mpeg4 → mpeg4 | I 5 / P 55 | **55 / 60** (all P-frames) | **1292** |
| HEVC | libx265 → hevc | I 1 / P 16 / B 43 | **0 / 60** | 0 |
| AV1 (libaom) | not available | — | **FAIL: encoder not built** | — |
| AV1 (SVT-AV1, substitute) | libsvtav1 → libdav1d | I 1 / P 59 | **0 / 60** | 0 |

**Takeaway:** motion-vector export works only for H.264 and MPEG-4 Part 2. FFmpeg's HEVC decoder and the dav1d AV1 decoder decode normally but attach no motion-vector side data. HEVC and AV1 would need a different MV source, such as a patched decoder or an encoder-side dump.

## 3. Test-data files (first 300 decoded frames for frame types)

| File | Codec / profile | Resolution | fps | Duration (s) | Bitrate (kbps) | Frame types in 300 | B-frames? |
|---|---|---|---|---|---|---|---|
| Virat/VIRAT_S_000205_03_000860_000922.mp4 | mpeg4 / Simple | 1280x720 | 30 | 61.2 | 8363 | I 11 / P 289 | no |
| Virat/VIRAT_S_000205_05_001092_001124.mp4 | mpeg4 / Simple | 1280x720 | 30 | 31.9 | 8400 | I 10 / P 290 | no |
| Virat/VIRAT_S_000206_09_001714_001851.mp4 | mpeg4 / Simple | 1280x720 | 30 | 127.0 | 8404 | I 10 / P 290 | no |
| Virat/VIRAT_S_000207_02_000498_000530.mp4 | mpeg4 / Simple | 1280x720 | 30 | 31.4 | 8375 | I 11 / P 289 | no |
| clips/bear_clip.mp4 | h264 / Constrained Baseline | 320x240 | 29.65 | 12.6 | 202 | I 1 / P 299 | no |
| clips/big_buck_bunny.mp4 | h264 / Main | 320x176 | 25 | 10.0 | 629 | I 2 / P 104 / B 144 (250 total) | **yes** |
| clips/bottle_detection.mp4 | h264 / High | 640x360 | 29.83 | 39.9 | 101 | I 2 / P 75 / B 223 | **yes** |
| clips/car_detection.mp4 | h264 / Baseline | 768x432 | 12.5 | 30.2 | 746 | I 5 / P 295 | no |
| clips/classroom.mp4 | h264 / High | 1920x1080 | 30 | 32.8 | 3304 | I 2 / P 78 / B 220 | **yes** |
| clips/driver_action_recognition.mp4 | **hevc** / Rext | 1920x1080 | 30 | 418.4 | 1029 | I 2 / P 75 / B 223 | **yes** |
| clips/one_by_one_person_detection.mp4 | h264 / Main | 768x432 | 10 | 139.4 | 189 | I 30 / P 92 / B 178 | **yes** |
| clips/people_detection.mp4 | h264 / Baseline | 768x432 | 12 | 49.7 | 883 | I 5 / P 295 | no |
| clips/person_bicycle_car_detection.mp4 | h264 / Baseline | 768x432 | 12 | 53.9 | 895 | I 5 / P 295 | no |
| clips/store_aisle_detection.mp4 | h264 / Main | 720x404 | 59.94 | 65.4 | 1127 | I 6 / P 76 / B 218 | **yes** |

All 14 files decoded without errors. The VIRAT clips are all MPEG-4 Part 2 at 720p and ~8.4 Mbps with no B-frames, so their MVs can be exported. `driver_action_recognition.mp4` is HEVC, so per section 2 it yields no MVs.

## 4. Folders in Downloads and Documents

### `C:\Users\akash\Downloads`
| Folder | Size |
|---|---|
| Anomaly-Videos-Part-1 | 6.0 GB |

It is the only folder. There are also 765 loose top-level files (22.0 GB total). The largest are `Anomaly-Videos-Part-1.zip` (5.8 GB), `DaVinci_Resolve_20.0.1_Windows.zip` (2.7 GB), two granite GGUF models (2.0 GB each) and **`archive (1).zip` (1.5 GB), which on inspection holds 54 VIRAT clips (`CCTV 01/VIRAT_S_*.mp4`)**, a possible larger VIRAT source.

### `C:\Users\akash\Documents` (junctions `My Music/Pictures/Videos` not followed)
| Folder | Size | | Folder | Size |
|---|---|---|---|---|
| Amazon ml | 18.66 GB | | Iris-ucfvad | 0.14 GB |
| Hackoween | 9.81 GB | | Agentic AI | 0.14 GB |
| Iris | 6.78 GB | | ML | 0.08 GB |
| HADES | 3.06 GB | | Iris-scaling | 0.06 GB |
| S3 | 2.48 GB | | FCC | 0.06 GB |
| Hackoweenidk | 2.08 GB | | CCBD | 0.05 GB |
| Exoplanet detection | 1.38 GB | | Ideathon | 0.02 GB |
| Tensor 5 | 1.09 GB | | Nexus, SE_LAB | 0.01 GB |
| Trying papers | 0.67 GB | | 17 others | < 0.01 GB each |
| Usb | 0.29 GB | | | |
| Structify | 0.15 GB | | | |

### Optical-flow datasets
**None found.** I searched every file and folder name under both roots, at any depth, for `sintel|spring|kitti|flyingchairs|flyingthings|middlebury|hd1k|tartanair|davis|optical_flow|*.flo|*.pfm`. Every hit was a false positive: library source in virtualenvs (`torchvision/datasets/kitti.py`, `torchvision/models/optical_flow`, mediapipe `optical_flow_field_data_pb2.py`), tz files (`Antarctica/Davis`), `framer-motion` spring animations, and `Usb/Print/lost spring.pdf`. No zip in Downloads is named like a flow dataset. MPI-Sintel, Spring and KITTI-flow would have to be downloaded.

## 5. Hardware

| | |
|---|---|
| Disk C: | **78.6 GB free** of 474.7 GB |
| CPU | AMD Ryzen 7 7435HS, 8 cores / 16 threads |
| RAM | 23.7 GB |
| GPU | NVIDIA GeForce RTX 4050 Laptop, **6.0 GB VRAM**; `torch 2.6.0+cu124`, `torch.cuda.is_available() = True` |

## Failures (reported, not worked around)
1. `ffmpeg` / `ffprobe` are not on PATH, so the CLI build flags couldn't be checked.
2. `libaom-av1` isn't in PyAV's build. As a substitute I ran AV1 through `libsvtav1`, which is available.
3. HEVC (`hevc` decoder) and AV1 (`libdav1d` decoder) produced no `MOTION_VECTORS` side data under `+export_mvs`.
