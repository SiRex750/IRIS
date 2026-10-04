# Item 7: CCTV encode frame rate (supplementary)

Each clip is encoded at its own source average frame rate (p3/virat/run_virat.py: `encode_yuv(src, path, codec, fps, ...)` with `fps = stream.average_rate` of the source, `add_stream(codec, rate=fps)`). Per clip: source avg_fps (manifest), fps_used (manifest), the `loaded ... @ fps` run.log line, the fps in every arm's npz meta, and the average_rate of every encoded .mp4 (PyAV).

## p3/virat

- Clips: 54. fps used: 2397/100 in 45 clips, 30 in 8 clips, 30000/1001 in 1 clips.
- Source avg_fps = fps_used = run.log = arm npz fps in 54/54 clips; encoded .mp4 average_rate = fps_used in 54/54 clips (1188 files).
- Clips at 30 fps: VIRAT_S_000205_02_000409_000566, VIRAT_S_000206_07_001501_001600, VIRAT_S_000206_08_001618_001712, VIRAT_S_000207_00_000000_000045, VIRAT_S_000207_01_000094_000156, VIRAT_S_000207_03_000556_000590, VIRAT_S_000207_04_000902_000934, VIRAT_S_000207_05_001125_001193; every other clip is listed with its rate in i07_cctv_fps.csv.

## p3/virat_confirm

- Clips: 50. fps used: 2397/100 in 42 clips, 30 in 8 clips.
- Source avg_fps = fps_used = run.log = arm npz fps in 50/50 clips; encoded .mp4 average_rate = fps_used in 50/50 clips (450 files).
- Clips at 30 fps: VIRAT_S_000205_02_000409_000566, VIRAT_S_000206_07_001501_001600, VIRAT_S_000206_08_001618_001712, VIRAT_S_000207_00_000000_000045, VIRAT_S_000207_01_000094_000156, VIRAT_S_000207_03_000556_000590, VIRAT_S_000207_04_000902_000934, VIRAT_S_000207_05_001125_001193; every other clip is listed with its rate in i07_cctv_fps.csv.
- Eligible clips (27): 2397/100 in 20, 30 in 7.
