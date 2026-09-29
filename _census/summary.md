# UCF-Crime encoder-settings census (Anomaly-Videos-Part-1)

- Root: `C:\Users\akash\Downloads\Anomaly-Videos-Part-1`
- Files: **200** .mp4 (200 probed OK, 0 PyAV errors)
- Frame-type analysis: first 600 decoded frames per file; x264 SEI searched in first 4 MB

## Codec

### Codec

| value | files | % |
|---|---:|---:|
| h264 | 198 | 99.0 |
| mjpeg | 2 | 1.0 |

### Profile

| value | files | % |
|---|---:|---:|
| Constrained Baseline | 198 | 99.0 |
| Baseline | 2 | 1.0 |

### Pixel format

| value | files | % |
|---|---:|---:|
| yuv420p | 198 | 99.0 |
| yuvj420p | 2 | 1.0 |

**Filename/codec mismatches** (`_x264` in name but codec != h264): 2

- `Arrest/Arrest050_x264.mp4` → mjpeg
- `Assault/Assault017_x264.mp4` → mjpeg

## Resolution / frame rate / bitrate

### Resolution

| value | files | % |
|---|---:|---:|
| 320x240 | 200 | 100.0 |

### Average fps

| value | files | % |
|---|---:|---:|
| 30.0 | 199 | 99.5 |
| 29.97 | 1 | 0.5 |

### Container bitrate (kbps)

| min | p25 | median | p75 | max | mean |
|---:|---:|---:|---:|---:|---:|
| 362 | 1424 | 1722 | 2013 | 3276 | 1693 |

### Container bitrate bins (kbps)

| value | files | % |
|---|---:|---:|
| 300–400 | 1 | 0.5 |
| 600–1000 | 12 | 6.0 |
| 1000–2000 | 129 | 64.5 |
| >=2000 | 58 | 29.0 |

Duration (s): min 4.6, median 83.1, max 4218.5, total 8.27 h

## x264 settings (from SEI user-data string)

### x264 SEI presence

| value | files | % |
|---|---:|---:|
| found | 198 | 99.0 |
| no_x264_sei | 2 | 1.0 |

### x264 core version

| value | files | % |
|---|---:|---:|
| 148 | 198 | 100.0 |

### rc mode

| value | files | % |
|---|---:|---:|
| abr | 198 | 100.0 |

### crf

| value | files | % |
|---|---:|---:|
| (blank) | 198 | 100.0 |

### qp

| value | files | % |
|---|---:|---:|
| (blank) | 198 | 100.0 |

### bitrate (x264 option)

| value | files | % |
|---|---:|---:|
| 2000 | 198 | 100.0 |

### keyint

| value | files | % |
|---|---:|---:|
| 250 | 198 | 100.0 |

### keyint_min

| value | files | % |
|---|---:|---:|
| 25 | 198 | 100.0 |

### scenecut

| value | files | % |
|---|---:|---:|
| 40 | 198 | 100.0 |

### bframes

| value | files | % |
|---|---:|---:|
| 0 | 198 | 100.0 |

### ref

| value | files | % |
|---|---:|---:|
| 2 | 198 | 100.0 |

### me

| value | files | % |
|---|---:|---:|
| umh | 198 | 100.0 |

### subme

| value | files | % |
|---|---:|---:|
| 6 | 198 | 100.0 |

### me_range

| value | files | % |
|---|---:|---:|
| 16 | 198 | 100.0 |

### trellis

| value | files | % |
|---|---:|---:|
| 0 | 198 | 100.0 |

### rc_lookahead

| value | files | % |
|---|---:|---:|
| 40 | 198 | 100.0 |

### b_adapt

| value | files | % |
|---|---:|---:|
| (blank) | 198 | 100.0 |

### weightp

| value | files | % |
|---|---:|---:|
| 0 | 198 | 100.0 |

### direct

| value | files | % |
|---|---:|---:|
| (blank) | 198 | 100.0 |

### Identical settings strings

- Distinct full `options:` strings: **1** across 198 files with SEI
- Largest group sharing one identical string: **198** files (100.0% of SEI files)

| group size | files |
|---:|---:|
| 198 | 1 |

Most common settings string:

```
cabac=0 ref=2 deblock=1:0:0 analyse=0x1:0x111 me=umh subme=6 psy=1 psy_rd=1.00:0.00 mixed_ref=1 me_range=16 chroma_me=1 trellis=0 8x8dct=0 cqm=0 deadzone=21,11 fast_pskip=1 chroma_qp_offset=-2 threads=6 lookahead_threads=1 sliced_threads=0 nr=0 decimate=1 interlaced=0 bluray_compat=0 constrained_intra=0 bframes=0 weightp=0 keyint=250 keyint_min=25 scenecut=40 intra_refresh=0 rc_lookahead=40 rc=abr mbtree=1 bitrate=2000 ratetol=1.0 qcomp=0.60 qpmin=0 qpmax=69 qpstep=4 ip_ratio=1.40 aq=1:1.00
```

Option keys that vary across the distinct strings: none

## Frame types (first 600 frames)

### B-frames present

| value | files | % |
|---|---:|---:|
| False | 200 | 100.0 |

I-frames per file in first 600: min 1, median 3.0, max 600

Files with ≤1 I-frame in the window (spacing undefined, i.e. GOP ≥ window): 1

Files shorter than the 600-frame window: 9

Files where packet keyframe count ≠ decoded I count: 26 (in all of them decoded I > keyframe packets, i.e. non-IDR I-frames — x264 emits a plain I-frame instead of an IDR when a scenecut falls within keyint_min of the last IDR)

### I-frame spacing, H.264 files only (all consecutive I-pairs)

| pairs | mean | median | max | min |
|---:|---:|---:|---:|---:|
| 478 | 200.69 | 250.0 | 250 | 1 |

Exactly 250 frames (= keyint): 317 of 478 gaps (66.3%); gaps < 25 (scenecut non-IDR I near an IDR): 43

### I-frames per H.264 file in window

| value | files | % |
|---|---:|---:|
| 1 | 1 | 0.5 |
| 2 | 5 | 2.5 |
| 3 | 145 | 73.2 |
| 4 | 28 | 14.1 |
| 5 | 9 | 4.5 |
| 6 | 6 | 3.0 |
| 8 | 2 | 1.0 |
| 10 | 1 | 0.5 |
| 11 | 1 | 0.5 |

### I-frame spacing, all files pooled (MJPEG files contribute 1-frame gaps)

| pairs | mean | median | max | min |
|---:|---:|---:|---:|---:|
| 1676 | 57.95 | 1.0 | 250 | 1 |

Per-file median spacing: min 1, median 250.0, max 250.0 (199 files with ≥2 I-frames)

## Verdict

**Effectively a single encoder setting:** 198/200 files carry the identical x264 core 148 options string (1-pass ABR 2000 kbps, keyint=250, scenecut=40, bframes=0, 320x240); the only exceptions are 2 MJPEG files mislabelled `_x264`, and the per-file variation that remains (realised bitrate, scenecut I-frame positions) comes from content, not settings.
