# Item 1: configuration count (supplementary; no new verdict)

Source: p3/sweep/results.csv (encode_id x sequence x pass), p3/sweep/encode_log.json, p3/sweep/encodes/. Rerun (p3/sweep_rerun) checked the same way.

## p3/sweep

- Final-pass configurations: **31**; clean-pass configurations: **16**.
- Distinct configurations overall: 31; clean-only configurations: none.
- Sequences: **23** (final 23, clean 23).
- Encodes (distinct pass x configuration x sequence in results.csv): **1081** (final 713 = 31 x 23, clean 368 = 16 x 23); encode log: 1081/1081 ok; .mp4 files on disk: 1081.
- Configurations shared between the final and clean groups (16): nvenc_qp18, nvenc_qp23, nvenc_qp28, nvenc_qp28_bf2_bqeq, nvenc_qp33, nvenc_qp38, nvenc_qp45, x264_crf12, x264_crf18, x264_crf23, x264_crf28, x264_crf33, x264_crf38, x264_crf45, x264_qp24, x264_qp24_bf2_pb1.
- Final-only configurations (15): mpeg4_q16, mpeg4_q2, mpeg4_q31, mpeg4_q4, mpeg4_q8, nvenc_qp28_bf2, x264_crf23_bf2, x264_crf23_keyint30, x264_crf23_ref2, x264_crf23_ref3, x264_crf23_ultrafast, x264_crf23_umh16, x264_crf23_umh32, x264_crf23_umh64, x264_crf23_veryslow.

## p3/sweep_rerun

- Final-pass configurations: **31**; clean-pass configurations: **16**.
- Distinct configurations overall: 31; clean-only configurations: none.
- Sequences: **23** (final 23, clean 23).
- Encodes (distinct pass x configuration x sequence in results.csv): **1081** (final 713 = 31 x 23, clean 368 = 16 x 23); encode log: 1081/1081 ok; .mp4 files on disk: 1081.
- Configurations shared between the final and clean groups (16): nvenc_qp18, nvenc_qp23, nvenc_qp28, nvenc_qp28_bf2_bqeq, nvenc_qp33, nvenc_qp38, nvenc_qp45, x264_crf12, x264_crf18, x264_crf23, x264_crf28, x264_crf33, x264_crf38, x264_crf45, x264_qp24, x264_qp24_bf2_pb1.
- Final-only configurations (15): mpeg4_q16, mpeg4_q2, mpeg4_q31, mpeg4_q4, mpeg4_q8, nvenc_qp28_bf2, x264_crf23_bf2, x264_crf23_keyint30, x264_crf23_ref2, x264_crf23_ref3, x264_crf23_ultrafast, x264_crf23_umh16, x264_crf23_umh32, x264_crf23_umh64, x264_crf23_veryslow.
