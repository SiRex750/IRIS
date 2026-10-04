# Item 8(a): C1-C4 on high-texture cells only (robustness; not a verdict)

High texture: cell texture >= 9.829 grey levels/px (75th percentile over all cells of the 50 clips; V0's threshold, summary.md: 9.829). 27 eligible clips, tau = 1, STALE_MOV.

| prediction | cells | clips with rise > 0.01 | pre-reg count | count met | median low / bf 0 | median high / B | median diff | median moving cells (low arm) | min moving cells |
|---|---|---|---|---|---|---|---|---|---|
| C1 x264 CRF 45 - CRF 12 | all valid (original) | 27/27 | >= 22 | yes | 0.0547 | 0.2487 | 0.1504 | 7793 | 2063 |
| C1 x264 CRF 45 - CRF 12 | high texture | 27/27 | >= 22 | yes | 0.0668 | 0.2210 | 0.1387 | 4937 | 725 |
| C2 NVENC QP 45 - QP 18 | all valid (original) | 27/27 | >= 21 | yes | 0.0487 | 0.1005 | 0.0441 | 7793 | 2063 |
| C2 NVENC QP 45 - QP 18 | high texture | 17/27 | >= 21 | no | 0.0599 | 0.0798 | 0.0177 | 4937 | 725 |
| C3 x264 QP 24 B - bf 0 | all valid (original) | 27/27 | >= 22 | yes | 0.0535 | 0.1924 | 0.1196 | 7793 | 2063 |
| C3 x264 QP 24 B - bf 0 | high texture | 27/27 | >= 22 | yes | 0.0574 | 0.1914 | 0.1284 | 4937 | 725 |
| C4 NVENC QP 28 B=P - bf 0 | all valid (original) | 27/27 | >= 21 | yes | 0.0429 | 0.1701 | 0.1281 | 7793 | 2063 |
| C4 NVENC QP 28 B=P - bf 0 | high texture | 27/27 | >= 21 | yes | 0.0441 | 0.1566 | 0.1302 | 4937 | 725 |

- Original rows reproduce p3/virat_confirm/results.csv stale_mov: yes.
- High-texture moving cells as a share of all moving cells (pooled over clips, per arm): nvenc_qp18/d1 0.485, nvenc_qp28/d1 0.485, nvenc_qp28_bf2_bqeq/B 0.487, nvenc_qp45/d1 0.485, x264_crf12/d1 0.485, x264_crf45/d1 0.485, x264_qp24/d1 0.485, x264_qp24_bf2_pb1/B 0.487.
- High texture, clips with diff > 0 / diff < 0: C1 27 / 0 (min 0.0153); C2 24 / 3 (min -0.0101); C3 27 / 0 (min 0.0780); C4 27 / 0 (min 0.0717).
- 'count met' applies the pre-registered count rule to this subset for orientation only.
