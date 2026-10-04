# Item 3: missed reuse / valid, tau = 1 (descriptive; no prediction)

## Sintel, CLEAN pass, d1 P-frame vectors (23 sequences)

| arm | median | min | max | median, FINAL pass (for reference) |
|---|---|---|---|---|
| x264 CRF 12 | 0.0008 | 0.0000 | 0.0970 | 0.0008 |
| x264 CRF 18 | 0.0008 | 0.0000 | 0.0786 | 0.0008 |
| x264 CRF 23 | 0.0008 | 0.0000 | 0.0770 | 0.0009 |
| x264 CRF 28 | 0.0008 | 0.0000 | 0.0876 | 0.0010 |
| x264 CRF 33 | 0.0009 | 0.0000 | 0.1231 | 0.0012 |
| x264 CRF 38 | 0.0016 | 0.0000 | 0.2324 | 0.0018 |
| x264 CRF 45 | 0.0036 | 0.0000 | 0.1939 | 0.0032 |
| NVENC QP 18 | 0.0010 | 0.0000 | 0.1730 | 0.0009 |
| NVENC QP 23 | 0.0009 | 0.0000 | 0.1652 | 0.0009 |
| NVENC QP 28 | 0.0009 | 0.0000 | 0.1614 | 0.0009 |
| NVENC QP 33 | 0.0009 | 0.0000 | 0.1610 | 0.0009 |
| NVENC QP 38 | 0.0010 | 0.0000 | 0.1442 | 0.0011 |
| NVENC QP 45 | 0.0012 | 0.0000 | 0.0781 | 0.0016 |

## CCTV confirmatory test, d1 P-frame vectors, RAFT-static (|RAFT| < 0.5 px) cells recomputed / valid

| arm | median (50 clips) | min | max | median (27 eligible) |
|---|---|---|---|---|
| x264 CRF 12 | 0.0061 | 0.0020 | 0.0743 | 0.0072 |
| x264 CRF 45 | 0.0018 | 0.0002 | 0.0125 | 0.0034 |
| x264 QP 24 | 0.0064 | 0.0018 | 0.0230 | 0.0074 |
| NVENC QP 18 | 0.0141 | 0.0040 | 0.0504 | 0.0171 |
| NVENC QP 23 | 0.0072 | 0.0020 | 0.0231 | 0.0079 |
| NVENC QP 28 | 0.0046 | 0.0015 | 0.0210 | 0.0054 |
| NVENC QP 45 | 0.0015 | 0.0001 | 0.0059 | 0.0025 |

CCTV static/valid uses RAFT, not ground truth; values are descriptive.
