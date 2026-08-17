# T5 Inventory — Codec Shot-Bucketing Selection Test (PRE-REGISTRATION, READ-ONLY)

**Scope:** inventory + pre-register only. The selector comparison was **NOT** run (that is the next prompt).
**Date:** 2026-08-17 (Asia/Calcutta).
**Audited by:** Claude (Cowork), read-only over the device bridge to machine `sonu` (win32).

## 0. Provenance / environment

| Item | Value |
|---|---|
| Iris main worktree HEAD | `312de77a3913c355df0e4b81bfffa76c996540b1` — branch `sonu/claims-ledger-and-t1-audit` |
| ⚠ HEAD status | This is the **unpushed** ledger/T1-audit HEAD (`git branch -r --contains` = none). The run does not depend on it, but the R0 gate script lives here. |
| `iris/charon_v.py` | last commit `a8059fc` (2026-07-13); sha256 `bb640da98b4857fcd3ca518b2d3ba1ab34793884782d2abfa1ed8d644f007016`; working tree clean |
| `iris/ingest.py` | last commit `a8059fc` (2026-07-13); sha256 `20a2eab337056ba85c927493df24b94dc352abb87324e0229ad3ae32129d493d`; clean |
| `scripts/r0_gate_full.py` | last commit `312de77` (2026-08-16); sha256 `c09aacd0cd902c3ef1be8301ccd8217a927900654bd7bf6e2f6db3c9b2e4cce0`; clean |
| Iris-ucfvad worktree | detached **linked** worktree; `.git` → `gitdir: C:/Users/akash/Documents/Iris/.git/worktrees/Iris-ucfvad`. The gitdir is a **Windows-absolute** path, so HEAD **cannot be resolved** through the Linux mount. GT data pinned by content hash instead. |
| GT CSV dir | `Iris-ucfvad\tuning\ucfcrime_vad_exp1\per_frame_ground_truth_scores\` |
| GT CSV sample hash | `Arson011_x264.csv` sha256 `0df4621133133b95029f1cf56a95a3e73cee33c51fea560ff16361c2f51fc6b5` |
| Videos | `C:\Users\akash\Downloads\Anomaly-Videos-Part-1\{class}\{vid}_x264.mp4` |

**Scratch I created (safe to delete):** `Iris\_shotbucket\_scratch\{Arson011,Arson041,Assault010}.pkts.csv` — ffprobe packet dumps used for cross-validation below.

---

## 1. Scene-span reproduction, zero re-ingest — **CONFIRMED, zero-decode, no neural**

**Call chain (all in `iris/charon_v.py`):**
1. `get_stream_fps(path)` (line 10) — reads `stream.average_rate` from container metadata. **No demux, no decode.**
2. `_demux_packet_curve(path)` (line 469) — `av.open` + `container.demux(stream)`, reads `pkt.size` / `pkt.pts` / `pkt.is_keyframe` only, re-sorts to display order by pts. **No `.decode()` call anywhere; no pixel decode; no neural pass.**
3. `compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps)` (line 534) — pure NumPy/SciPy (`argrelextrema` valleys below `valley_percentile=25`), `PEAK_WINDOW_SECONDS=0.5`. Enforces `MAX_SCENE_LEN=300` (long codec spans get chopped into ≤300-frame chunks).

**Dependencies:** `av` (PyAV), `numpy`, `scipy` only. Input = the `.mp4` alone. The module is self-contained — nothing outside `charon_v.py` is needed for shot spans. **No file outside `Anomaly-Videos-Part-1` is required.**

**Spot-check (ran the ACTUAL functions):**

| video | N frames | I-frames | shots | wall (real av-demux) | local demux wall¹ |
|---|---|---|---|---|---|
| Arson011 | 1266 | 11 | **31** | 0.029 s | 0.16 s |
| Arson041 | 3754 | 18 | **54** | 0.033 s | 0.20 s |
| Assault010 | 16177 | 65 | **430** | (via validated ffprobe curve²) | 1.70 s |

¹ On-device `ffprobe -show_packets` wall time — the real local demux cost (dominant term; the valley compute is ~1.5 ms).
² The 110 MB Assault010 mp4 would not transfer over the bridge (two staging timeouts). Instead its packet curve was dumped on-device via `ffprobe` and fed to the **same** `compute_valley_scene_boundaries`. This method is validated: the ffprobe-derived curve reproduces the real `av`-demux spans **bit-identically** on Arson011 and Arson041 (`MATCHES_REAL_AV=True` for both), so the Assault010 spans are trustworthy.

**Spans (first video, for the record):** Arson011 → `[(0,4),(4,40),(40,58),(58,94),(94,136),(136,172),(172,196),(196,232),(232,250),(250,274),(274,304),(304,352),(352,394),(394,592),(592,686),(686,718),(718,755),(755,772),(772,809),(809,850),(850,886),(886,916),(916,940),(940,970),(970,994),(994,1031),(1031,1084),(1084,1126),(1126,1217),(1217,1264),(1264,1266)]`. Full spans for all three saved with the run.

**Full-19 feasibility:** even the largest video demuxes in 1.7 s locally; all 19 ≪ 1 minute total. **Not run (per instruction).**

---

## 2. Gold + budget join — **CONFIRMED**

- **Columns present, exactly:** `frame_idx, ground_truth, action_score_propagated, is_retained_tier` (verified on the header + `load_video()`'s contract, which also asserts `frame_idx` is contiguous `0..N-1`).
- **Corpus:** 169 CSVs = **19 anomalous** (Abuse×2, Arrest×5, Arson×9, Assault×3) + 150 `Normal_Videos`.
- **mp4 join:** all 19 anomalous CSVs have a matching `.mp4` in Downloads — **missing = 0**.
- **All 19 are h264 / 30 fps** (see §Guards).

**Production retention % per video (from `is_retained_tier`):**

| video | frames | ret % | gold % | | video | frames | ret % | gold % |
|---|---|---|---|---|---|---|---|---|
| Abuse028 | 1412 | 10.48 | 5.38 | | Arson016 | 1795 | 10.47 | 44.29 |
| Abuse030 | 1544 | 10.49 | 5.57 | | Arson018 | 842 | 10.57 | 39.31 |
| Arrest001 | 2374 | 10.45 | 12.68 | | Arson022 | 8640 | 10.42 | 5.80 |
| Arrest007 | 3144 | 10.59 | 20.07 | | Arson035 | 1437 | 10.58 | 20.95 |
| Arrest024 | 3629 | 10.42 | 57.89 | | Arson041 | 3754 | 10.50 | 39.58 |
| Arrest030 | 8642 | 10.45 | 19.28 | | Assault006 | 8096 | 10.42 | 85.36 |
| Arrest039 | 15835 | 10.41 | 19.71 | | Assault010 | 16177 | 10.41 | 6.32 |
| Arson007 | 6252 | 10.43 | 55.20 | | Assault011 | 2288 | 10.45 | 25.61 |
| Arson009 | 743 | 11.04 | 12.92 | | | | | |
| Arson010 | 3159 | 10.64 | 10.95 | | **range** | | **10.41–11.14** | 5.38–85.36 |
| Arson011 | 1266 | 11.14 | 67.69 | | | | | |

Retention band **[10.41, 11.14]%** matches R0 exactly. Match the test budget to each video's own retention (or use a fixed 10.5% — R0's headline point).

---

## 3. Existing code — do not reinvent

**No existing shot-bucketed / per-scene / per-shot *selector* exists.** Grep for `scene_bucket|per_scene_sample|shot_sample|one_per_scene|per_shot|shot_bucket` across `iris/` + `scripts/` returned **nothing**. The guard "you find an existing shot-bucketed selector → stop" does **not** trigger. This test writes the first one.

**Reusable scene assignment (use as-is):** `iris/ingest.py` §4.5 (lines ~238–256) already does the zero-decode assignment:
```
scene_spans = charon_v.compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps)
for f in frames_to_index:
    fi = f["frame_idx"]
    for scene_idx,(start,end) in enumerate(scene_spans):
        if start <= fi < end: f["scene_id"] = scene_idx; break
    else: f["scene_id"] = -1
```
The **span computation is reusable verbatim.** One adaptation for selection: ingest assigns `scene_id` only to `frames_to_index` (survivors); the selection test must bucket **all** `0..N-1` frames, i.e. run the same membership loop over the full range. Trivial, same logic.

**⚠ Do not confuse two scene_id schemes.** `iris/l2_asphodel.py::_refresh_scene_ids` (lines 211–227) is a **different**, cruder scheme (monotonic counter that ticks on each I-frame). The shot-bucketing test must use the **valley-boundary spans** from `charon_v`, *not* the asphodel I-frame counter.

**R0 metric functions — reuse verbatim from `scripts/r0_gate_full.py`:**

| fn | line | role |
|---|---|---|
| `metrics(admitted, gt, windows)` | 143 | M1/M2/M3 |
| `mask_to_windows(mask)` | 104 | gold windows from 0/1 mask |
| `load_video(vid)` | 119 | reads the 4 GT columns, asserts contiguity |
| `budget_k(n, b_pct)` | 162 | `max(1, int((b_pct/100)*n))` |
| `arm_c_uniform(n, b_pct)` | 167 | `np.unique(round(linspace(0,n-1,k)))` |
| `arm_d_random(n, b_pct, seed)` | 172 | `rng.choice(n, k, replace=False)` |
| `top_k_by_score(score, k)` | 178 | top-k, ties by ascending frame_idx |
| `bootstrap_ci(...)` | 374 | paired bootstrap CI (n=19) |

Metric definition (verbatim): **M1** = fraction of gold *windows* with ≥1 admitted frame; **M2** = admitted gold frames / all gold frames; **M3** = admitted gold frames / admitted frames. The new test **imports** these — it does not recompute metrics differently.

---

## 4. Proposed selector set (PROPOSAL ONLY — not implemented)

Matched budget: `k = budget_k(N, 10) = max(1, int(0.10·N))` per video (or per-video retention).

| arm | definition | family |
|---|---|---|
| **Uniform** | `arm_c_uniform` (R0's tie baseline) | cheap / zero-decode |
| **Random ×5** | `arm_d_random`, seeds 0–4, averaged within video | cheap |
| **Codec action-score (R0 anchor)** | `top_k_by_score(action_score_propagated, k)` — the existing R0 method | cheap / zero-decode |
| **Codec shot-bucketed 1/shot** | valley shots → 1 frame/shot + budget-fill to `k` | cheap / zero-decode (NEW) |
| **Codec shot-bucketed 2/shot** | valley shots → 2 frames/shot + budget-fill to `k` | cheap / zero-decode (NEW) |
| *Optional:* CLIP-similarity | embed a reduced candidate set, pick representatives | **expensive** (see below) |

**Per-shot frame pick — two pre-declared variants:**
- **(a) Highest-action-in-shot** *(recommended primary shot arm)* — argmax `action_score_propagated` within each shot. Reuses the CSV column, no recompute; it is the shot-aware analog of the R0 action-score arm.
- **(b) Shot midpoint** *(pure-geometry ablation)* — `(start+end)//2`. Survivor-independent, uses **no** score; the cleanest "codec-only" arm.

**Exact-budget rule (this is the crux — flagged as a design finding).**
In every spot-check the shot count S is **far below** the 10% frame budget k, so a literal "1–2 frames/shot" **under-spends** the budget and would produce an unmatched (confounded) comparison:

| video | N | k @10% | shots S | 1/shot | 2/shot |
|---|---|---|---|---|---|
| Arson011 | 1266 | 126 | 31 | 31 (2.4%) | 62 (4.9%) |
| Arson041 | 3754 | 375 | 54 | 54 (1.4%) | 108 (2.9%) |
| Assault010 | 16177 | 1617 | 430 | 430 (2.7%) | 860 (5.3%) |

So the arm must **fill to exactly k**:
- **S ≤ k (the normal case here):** give every shot 1 frame (coverage floor), then distribute the remaining `k−S` frames across shots by **largest-remainder (Hamilton) apportionment**, weight = shot length (variant b/geometry) or shot peak-action (variant a). Within a shot, place extra frames either evenly spaced (geometry) or by descending `action_score_propagated` (action). Clamp each shot's allocation to its length.
- **S > k (short high-cut video; did not occur in the 3 spot-checks but is possible):** keep the `k` shots with the highest peak `action_score_propagated`, 1 frame each; tie-break lower shot index.
- Round to **exactly k** every time via largest-remainder; unit-test that `len(selected)==k` and `selected⊆[0,N)`.

**Pre-commit ONE shot arm as primary** (recommend **1/shot, highest-action, action-weighted fill**) to avoid a garden-of-forking-paths; report the rest as secondary.

**CLIP-similarity (expensive family) — feasible on CPU but NOT zero-decode.** A CLIP selector needs **pixel decode + a neural forward pass** (ViT), so it breaks the codec-only property and belongs to a separate "expensive-family" contrast. Embedding every frame of ~100k frames on CPU is impractical; to stay tractable, restrict the candidate set (e.g. shot midpoints, or uniform 1 fps), embed those, then pick diverse/representative frames to fill k. Mark **OPTIONAL**; if included, label it clearly as the expensive anchor, not a peer of the codec arms.

---

## Guards

| guard | result |
|---|---|
| scene-span needs neural fwd / full decode | **PASS** — packet demux only, no `.decode()`, no model. Proceed. |
| any of 19 mp4 missing / mislabeled mjpeg | **PASS** — all 19 are `codec_name=h264`, 30 fps; missing=0. The 2 mjpeg files R0 noted are **not** among the annotated 19. |
| existing shot-bucketed selector | **none found** — write the first one. |

---

## Pre-registration / statistical-power note (REQUIRED before running — per Sonu)

- **n = 19.** R0's own paired bootstrap gave CI half-widths ≈ **0.47 pp (dM2)** and **0.88 pp (dM3)**. Any shot-bucketing effect **< ~1 pp is undetectable** on this corpus — it returns a CI spanning zero *regardless of whether the effect is real*.
- **M1 is dead as an outcome.** It saturates at 1.0000 for uniform at every budget ≥5% in R0, and **17/19 videos have exactly one gold window** (only Arson011 and Assault010 have 2). M1 is near-binary per video → do **not** use it as a primary outcome.
- **Primary outcomes:** paired **dM2** (gold recall) and **dM3** (precision), shot-bucketed − uniform, matched budget, via `bootstrap_ci` (reuse `BOOTSTRAP_N`/`BOOTSTRAP_SEED`).
- **Pre-committed three-branch read** (the third branch is mandatory to prevent claim inflation, audit §3.3):
  1. **BEATS uniform** — CI excludes 0 **and** |effect| ≥ ~1 pp.
  2. **TIES uniform** — tight CI centered near 0 (half-width < ~1 pp): a genuine null.
  3. **UNDERPOWERED** — CI spans 0 **with** half-width ≥ the effect of interest: cannot distinguish a real 0.5 pp effect from null. **Report branch 3, not branch 2, whenever the CI is wide.** "Ties uniform" may only be claimed when the CI is tight enough to exclude a ~1 pp effect.

---

## T5 VERDICT: READY-TO-RUN-LOCALLY  — all 19 inputs are present and verified, the codec shot spans reproduce zero-decode from the .mp4 alone, the R0 M1/M2/M3 metrics and scene-span code are reusable verbatim, and no competing selector exists; the only caveat is statistical, not setup — the n=19 corpus can only resolve effects ≳1 pp, so the pre-committed read must include the "underpowered" branch above.
