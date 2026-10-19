# Truth pilot: gates per clip per candidate

Plan e7ac3b0. This pilot chooses a reference after the RAFT-based truth failed (reading B drift gate 0/25, `p3/codecsight_native2fps/gates.md`). Only truth was computed on these clips.

Gate 1 (drift): quiet cell-frames whose candidate displacement is > 2 px; pass if <= 0.01 on all 3 clips. Gate 2 (agreement): median |candidate - native RAFT chain| over spans <= 16 frames, points moving > 2 px; pass if < 1.0 px on all 3 clips. A candidate passes only if both pass.

- **dot_360x640**: gate 1 FAIL, gate 2 FAIL -> **FAIL**. Wall time 2.3 min.
- **cotracker3**: gate 1 FAIL, gate 2 FAIL -> **FAIL**. Wall time 159.8 min.
- DOT at 480x856 ran out of GPU memory (VIRAT_S_000206_07_001501_001600: FAILED; VIRAT_S_000206_08_001618_001712: FAILED; VIRAT_S_000207_00_000000_000045: FAILED); per PLAN §1 DOT is evaluated at 360x640 on every clip (first attempt: `out/dot_manifest.json`).

| clip | candidate | g1 quiet | g1 false | g1 rate | g1 | g2 points | g2 median px | g2 | infer s | peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| VIRAT_S_000206_07_001501_001600 | dot_360x640 | nan | nan | nan | fail | nan | nan | fail | nan | 16.20 |
| VIRAT_S_000206_07_001501_001600 | cotracker3 | 258837.0 | 11005.0 | 0.04252 | fail | 3879.0 | 1.597 | fail | 3511 | 5.83 |
| VIRAT_S_000206_08_001618_001712 | dot_360x640 | nan | nan | nan | fail | nan | nan | fail | nan | 13.39 |
| VIRAT_S_000206_08_001618_001712 | cotracker3 | 258484.0 | 9882.0 | 0.03823 | fail | 4122.0 | 1.488 | fail | 3871 | 5.83 |
| VIRAT_S_000207_00_000000_000045 | dot_360x640 | nan | nan | nan | fail | nan | nan | fail | nan | 13.39 |
| VIRAT_S_000207_00_000000_000045 | cotracker3 | 153833.0 | 5592.0 | 0.03635 | fail | 1725.0 | 2.652 | fail | 2149 | 5.83 |
