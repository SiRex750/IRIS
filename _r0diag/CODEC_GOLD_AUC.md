# Codec admission score vs gold frames — Phase 1 (schema discovery)

**Status: STOPPED AT PHASE 1, STEP 3. Guard tripped. Phase 2 was not run.**

Reason in one line: `is_retained_tier` is **not** derived by thresholding any column in
the CSV. It is driven by the per-frame normalized packet size `ps` inside
`charon_v.parse_video`, and `ps` is not present in the CSV. Per the stated guard —
"if admission is driven by a score not present in the CSV, say so and STOP" — this run
stops here.

Every number below was computed in this run from the files named. Nothing is recalled or
reconstructed.

---

## 1. The CSVs Arm A reads

Arm A's loader is [`scripts/r0_gate_full.py:119-138`](scripts/r0_gate_full.py:119) (`load_video`),
which reads:

```
C:\Users\akash\Documents\Iris-ucfvad\tuning\ucfcrime_vad_exp1\per_frame_ground_truth_scores\<VIDEO>_x264.csv
```

built from `GT_DIR` at [`scripts/r0_gate_full.py:41-42`](scripts/r0_gate_full.py:41):
`WORKTREE = REPO_ROOT.parent / "Iris-ucfvad"`, then
`tuning/ucfcrime_vad_exp1/per_frame_ground_truth_scores`.

These files live **outside this repo**, in a sibling git worktree of the same repository:

| | |
|---|---|
| Worktree path | `C:/Users/akash/Documents/Iris-ucfvad` |
| Worktree HEAD | `a5cea2502e287c23c30cc2444e564e61e30501a3` (detached), *"UCF-Crime VAD Experiment 1: Stage B (scene-sparse graph + PageRank) -- negative result"*, Wed Jul 29 15:38:23 2026 +0530 |
| Analysis repo HEAD (`C:/Users/akash/Documents/Iris`) | `16831838277c0e5fe8806cdbfb5fbf386d1ccb8e` |

Committed status: **tracked and clean.** `git ls-files` reports 169 CSVs under that
directory, `git status --porcelain` on the directory is empty (no modifications, no
untracked files). All 19 videos in `VIDEOS` are present.

Content hash of one file as a fixity anchor:
`Abuse028_x264.csv` sha256 = `82c94e74c0d3b521ba99e4fa45250caa1ef00859d128f43dfaf34d8e9af1695a`

## 2. Columns and raw sample rows

Full column list (verified identical across all 19 CSVs in this run):

```
frame_idx, ground_truth, action_score_propagated, is_retained_tier
```

First 5 raw data rows of `Abuse028_x264.csv`, verbatim including the header:

```
frame_idx,ground_truth,action_score_propagated,is_retained_tier
0,0,0.17531026899814606,1
1,0,0.17531026899814606,0
2,0,0.4493710398674011,1
3,0,0.16315588355064392,1
4,0,0.16315588355064392,0
```

## 3. What `is_retained_tier` is actually derived from — GUARD TRIPPED

**The admission score is `ps` (normalized codec packet size). It is not a column in the CSV.**

Trace:

**(a) The CSV writer.** `C:/Users/akash/Documents/Iris-ucfvad/scripts/ucfcrime_vad_exp1_stageA.py:231-237`:

```python
# Write per-frame CSV (frame_idx, ground_truth, action_score_propagated, is_retained_tier).
csv_path = GT_SCORE_DIR / f"{Path(name).stem}.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame_idx", "ground_truth", "action_score_propagated", "is_retained_tier"])
    for fi in range(n_frames):
        w.writerow([fi, int(gt[fi]), full_scores[fi], int(fi in retained_idx_set)])
```

So `is_retained_tier == 1` iff `fi in retained_idx_set`, and (same file, line 188):

```python
retained_idx_set = {f["frame_idx"] for f in output_frames}
```

`output_frames` is the first return value of `charon_v.parse_video(...)`.

**(b) The threshold, inside charon_v.** `C:/Users/akash/Documents/Iris-ucfvad/iris/charon_v.py:322-331` —
this is the line where the continuous score meets the cutoff:

```python
if frame.key_frame:
    tier = "I_FRAME"
elif total_frames in peak_frame_ids:
    tier = "PEAK"
elif ps > curr_salient:          # <-- continuous score `ps` vs per-scene threshold
    tier = "SALIENT"
elif ps >= curr_candidate:       # <-- same score, lower threshold
    tier = "CANDIDATE"
else:
    tier = "SKIP"
```

and `charon_v.py:419` gates membership of `output_frames`:

```python
if tier != "SKIP":
    ...
    output_frames.append({... "packet_size": ps, ...})
```

`ps` is the per-frame normalized packet size read from the container packet
(`charon_v.py:300-308`); `curr_salient` / `curr_candidate` are per-scene adaptive
thresholds (`charon_v.py:311-318`). Admission is therefore `ps` thresholded per scene,
OR'd with two label-free structural overrides (`frame.key_frame`, membership in
`peak_frame_ids`). **None of `ps`, `curr_salient`, `curr_candidate`, `key_frame`, or
`peak_frame_ids` appears in the CSV.**

**(c) `action_score_propagated` is downstream of admission, not upstream of it.**
`stageA.py:188-215`: the action score is computed **only on frames already retained**, then
hold-forward filled across all frames. Retention determines where the score is sampled;
the score does not determine retention. The arrow runs the opposite way from what Phase 2
would need.

**Empirical confirmation from this run** (all 19 videos): for each video I computed the
best agreement with `is_retained_tier` achievable by *any* threshold on
`action_score_propagated` (sweep all N+1 cut points, score TP+TN). Mean best agreement =
**0.8951**, against retention rates of 10.41%–11.14% — i.e. the best threshold does no
better than the trivial "nothing is retained" classifier (1 − 0.1049 ≈ 0.8951). Per video
the number of distinct values of `action_score_propagated` also equals the number of
retained frames almost exactly (e.g. Abuse028: 148 retained, 148 distinct values), which is
the signature of the hold-forward step function described in (c). No threshold on the CSV's
continuous column reconstructs admission.

## 4. Gold frame definition and its source

Gold frames come from the column `ground_truth` in the same CSV, written at
`stageA.py:237` from `gt = build_ground_truth(n_frames, row["spans"])`
(`stageA.py:183`), where `build_ground_truth` (`stageA.py:83-89`) sets `gt[s:e+1] = 1` for
each annotated span, clamped to `[0, n_frames-1]`. The spans are parsed from the official
UCF-Crime temporal annotation file at
`eval/data/ucf/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt`
(path recorded in the committed
`Iris-ucfvad/tuning/ucfcrime_vad_exp1/annotation_validation.json`, commit `c59967e`).

Note for the auditor: **that .txt file is no longer on disk** — the `eval/data/ucf/`
tree does not exist in the worktree and the file is not tracked by git in either repo. Its
content is attested only by the committed `annotation_validation.json`, which records
290 rows / 290 unique names / 140 anomalous / 150 normal, 169 matched to a local .mp4,
0 decode failures, and 3 spans whose end index exceeded the decoded frame count
(Arson011 span [680,1267] vs 1266 frames; Arson016 [1000,1796] vs 1795; Assault006
[1185,8096] vs 8096) — these were clamped by `build_ground_truth`, not dropped.

## 5. Population guard (step 5) — PASSES

Each CSV contains the **full** frame population, not only admitted frames. Verified in this
run: for all 19 videos `frame_idx` is exactly `0..N-1` contiguous, and both retained and
non-retained rows are present.

| video | n_rows (=N) | frame_idx 0..N-1 | n_gold | n_retained | retention % | distinct score values |
|---|---:|---|---:|---:|---:|---:|
| Abuse028 | 1412 | yes | 76 | 148 | 10.48 | 148 |
| Abuse030 | 1544 | yes | 86 | 162 | 10.49 | 162 |
| Arrest001 | 2374 | yes | 301 | 248 | 10.45 | 248 |
| Arrest007 | 3144 | yes | 631 | 333 | 10.59 | 332 |
| Arrest024 | 3629 | yes | 2101 | 378 | 10.42 | 378 |
| Arrest030 | 8642 | yes | 1666 | 903 | 10.45 | 903 |
| Arrest039 | 15835 | yes | 3121 | 1648 | 10.41 | 1648 |
| Arson007 | 6252 | yes | 3451 | 652 | 10.43 | 652 |
| Arson009 | 743 | yes | 96 | 82 | 11.04 | 82 |
| Arson010 | 3159 | yes | 346 | 336 | 10.64 | 336 |
| Arson011 | 1266 | yes | 857 | 141 | 11.14 | 139 |
| Arson016 | 1795 | yes | 795 | 188 | 10.47 | 188 |
| Arson018 | 842 | yes | 331 | 89 | 10.57 | 89 |
| Arson022 | 8640 | yes | 501 | 900 | 10.42 | 898 |
| Arson035 | 1437 | yes | 301 | 152 | 10.58 | 151 |
| Arson041 | 3754 | yes | 1486 | 394 | 10.50 | 394 |
| Assault006 | 8096 | yes | 6911 | 844 | 10.42 | 843 |
| Assault010 | 16177 | yes | 1022 | 1684 | 10.41 | 1677 |
| Assault011 | 2288 | yes | 586 | 239 | 10.45 | 239 |

The population guard is not the blocker. The blocker is step 3: the population is there, the
gold labels are there, but the admission score is not.

---

## Verdict on Phase 1

**BLOCKED.** The pre-registered Phase 2 question — "does the codec admission score rank
gold frames above non-gold frames within a video" — **cannot be answered from these CSVs**,
because the codec admission score (`ps`, normalized packet size, thresholded per scene at
`iris/charon_v.py:326-331`) was never written to them. The only continuous column present,
`action_score_propagated`, is computed *after* admission on the retained subset and
hold-forward filled; it is a different quantity, and empirically no threshold on it
reconstructs `is_retained_tier` better than chance-at-base-rate (mean best agreement 0.8951
vs 0.8951 trivial).

No AUC is reported. No per-video table of AUCs exists in this run. Nothing here should be
read as evidence for or against signal in codec admission — the measurement was not made.

For the strategist, the two ways forward (**not taken in this run**, both require a decision):

1. Answer the *different* question that these CSVs can answer: within-video AUC of
   `action_score_propagated` vs `ground_truth`. That is threshold-free and computable now,
   but it tests the action-score ranker, not codec admission. It must not be relabeled as
   the admission question.
2. Answer the *actual* question, which requires `ps` per frame — meaning a re-ingest, which
   the hard constraints of this task forbid.

---

**Provenance sentence (gold-frame definition):** Gold frames are the `ground_truth` column
of `Iris-ucfvad/tuning/ucfcrime_vad_exp1/per_frame_ground_truth_scores/<VIDEO>_x264.csv`
(worktree commit `a5cea25`, tracked and clean), written by
`scripts/ucfcrime_vad_exp1_stageA.py:237` from spans parsed out of the official UCF-Crime
`Temporal_Anomaly_Annotation.txt` and expanded to a per-frame 0/1 mask by
`build_ground_truth` at `stageA.py:83-89` — the annotation .txt itself is no longer on disk
and is attested only by the committed `annotation_validation.json` (commit `c59967e`).
