# T1 — End-to-end / Amdahl artifact reconciliation

**Run date:** 2026-08-15 · **Mode:** READ-ONLY (no model was run; no measurement was produced)
**Scope:** `C:\Users\akash\Documents\Iris`, `...\Iris-scaling`, `...\Iris-ucfvad`

---

## 0. Search perimeter (what "no artifact" is a statement about)

The three working trees are **three worktrees of one repository**, sharing a single object
store at `C:/Users/akash/Documents/Iris/.git`:

```
C:/Users/akash/Documents/Iris          1683183 [main]
C:/Users/akash/Documents/Iris-scaling  378daa7 (detached HEAD)
C:/Users/akash/Documents/Iris-ucfvad   a5cea25 (detached HEAD)
```

| tree | HEAD sha | contained in |
|---|---|---|
| Iris | `16831838277c0e5fe8806cdbfb5fbf386d1ccb8e` | `refs/heads/main` |
| Iris-scaling | `378daa7be2c4412bcd7cd29dc32d48af2d083ba7` | `origin/siddanth/peak-source-a6-p1` only |
| Iris-ucfvad | `a5cea2502e287c23c30cc2444e564e61e30501a3` | `origin/siddanth/ucf-vad-exp1` only |

Searched, exhaustively:

1. **All 29 refs** — 12 local branches (incl. `wip/laptop-preserve-2026-08-04` @ `5d0969d4`,
   2026-08-04), 16 remote-tracking branches, `refs/stash`.
2. **Both stash entries** explicitly — `stash@{0}` (WIP on `pillar4-diagnostics`),
   `stash@{1}` (WIP on `main`). Zero hits in either.
3. **Every blob in history, not only ref tips.** `git rev-list --objects --all` → 2,464
   objects → 1,502 text-extension blobs → **960 prose/code blobs** (`.md .py .tex .yaml
   .yml .ipynb`), each `git cat-file`'d and grepped individually. This catches artifacts
   that were committed and later deleted, which a tip-only `git grep` would miss.
4. **Working-tree files including untracked ones** in all three trees (`CLAIMS.md`,
   `CLAUDE_CODE_TASKS.md`, `IRIS_Repo_Audit_2026-08-15.md`, `_n6audit/`, `_r0check/`,
   `_r0diag/`, `r0_full/`, `r0_smoke/`).

---

## STEP 1 — Locate

### 1.1 Summary of every target string

| target | committed artifacts, any ref, any point in history | verdict |
|---|---|---|
| `amdahl` | **0** | absent from the repository entirely |
| `e2e_speedup` | **0** | absent from the repository entirely |
| `end_to_end` | **1**, and it is a Python test-function name | not a measurement |
| `crossover` | **2**, both `~100 frames`, a different quantity | not the claimed crossover |
| `2644` / `N≈2,644` | **0** in prose; binary noise only | absent |
| `4.6` as an e2e speedup | **0** | **ABSENT** |
| `2.2` as an LLM/answerer stage cost | **0**; 19 blobs use 2.2s as a *grounding half-width* | **ABSENT — and the number means something else** |

The only places `4.6×` / `N≈2,644` appear on this machine are three **untracked documents
dated 2026-08-15** — the audit deliverables themselves, which are downstream restatements,
not measurements, and one of which already labels the figures `UNVERIFIED-LOCALLY`.

### 1.2 Every hit, with surrounding context

---

**HIT 1 — `crossover`, but it is a frame count, not N≈2,644**

`C:\Users\akash\Documents\Iris-scaling\eval_results\P_latency_result.md`
· commit `c5141ce9e622028a6c9e4c1e835c2d97f8ded5a5` · blob `3bee36c6`
· sha256 `8ac39060949a66c509dff72ae088bba076483f1e96c44d0d767a43e09a3ecf87`

```
15	  <100 frames  (n=167): flat 0.001497 | scene 0.001653  -> scene 10% SLOWER
16	  100-299      (n=173): flat 0.002464 | scene 0.002032  -> scene 18% faster
17	  300-599      (n=66):  flat 0.004260 | scene 0.002500  -> scene 41% faster
18	Crossover ~100 frames. Across the range flat grows 2.85x, scene_sparse 1.51x — flat scales
19	roughly twice as steeply with video length.
```

This is the **flat-vs-scene_sparse retrieval crossover in frames** (~100), i.e. the video
length above which scene_sparse retrieval beats flat retrieval. It is not, and cannot be
rewritten as, "the N at which retrieval cost equals the answerer cost."

---

**HIT 2 — same `crossover`, restated in the decision log**

`C:\Users\akash\Documents\Iris-scaling\DECISIONS.md`
· commit `c5141ce9e622028a6c9e4c1e835c2d97f8ded5a5` · blob `a48b5db2`
· sha256 `7b38c08ea1cd9b90200c47cbc0443897be4c16cd365fcc32088b2694990beee1`

```
329	Mechanism confirmed: PPR time drops 47% on the induced subgraph, but subgraph induction overhead
330	(.subgraph().copy() cost, scales with SOURCE graph) cancels the saving at these video lengths.
331	SCALING trend: crossover ~100 frames; below it scene_sparse is 10% slower, above 300 frames it is
332	41% faster; flat's latency grows ~2x steeper than scene_sparse across the tested range. Longest
333	videos tested are 599 frames (n=66) — CCTV footage is ~100x longer, so this is a hypothesis with
334	supporting evidence, not a verified claim at CCTV scale. Retrieval is ~0.003% of end-to-end cost
335	(query embedding 0.016s, answerer ~60s), so this only matters for the long-video scaling argument.
```

---

**HIT 3 — `end_to_end`, a test-function name**

`C:\Users\akash\Documents\Iris-scaling\tests\test_answerer_contract.py:509`
(identical file present on `main`)

```
507	
508	
509	def test_production_query_v2_end_to_end(monkeypatch):
510	    import json
511	    from iris import aria
512	    from iris.query import query
513	    from iris.iris_config import IRISConfig
```

A monkeypatched contract test. Carries no timing.

---

**HIT 4 — every `2.2 second` in the entire repository history is a grounding half-width**

Nineteen blobs across history match `2.2\s*(s|sec|second)`. Every one is
`FROZEN_HALF_WIDTH_SECONDS`. Representative:

`C:\Users\akash\Documents\Iris-scaling\eval\span.py:26`
```
26	FROZEN_HALF_WIDTH_SECONDS: float = 2.2
```

`C:\Users\akash\Documents\Iris-scaling\eval_results\half_width_confirmation_report.md`
· commit `83215c00bcc9642cf35e7ef032ef77df8e70b579` (2026-07-19) · blob `405d2e0b`
· sha256 `2d4dc08dca8ce2dfebe6e9d0069a737316de5db7f30d291346537604be5448ea`

```
14	Bootstrap: video-level cluster resampling, 1000 resamples, seed=20260710 (matches
15	`bootstrap_paired_differences` convention already in the codebase).
16	
17	Anchor **w\*=2.2s** = median gold half-span on the 64-question set (decided in chat prior to
18	this run — this is a confirmation sweep, not an argmax search).
```

`C:\Users\akash\Documents\Iris-scaling\DECISIONS.md:173`
```
171	them exactly 0.0 (no predicted/gold overlap at all). Retrieval is upstream of and dominates the
172	answerer term. Verified NOT a mechanical artifact: units consistent (seconds throughout, no fps
173	distance min 2.55s, median 8.37s — all beyond the 2.2s window). Retrieved clusters are
174	internally coherent but land in the wrong video neighborhood.
```

Also in `scripts/eval_grounding_arms.py:48`, `scripts/pillar2_grounded_qa.py:226`,
`eval/grounding_scorer.py:125`, `tuning/val_confirm_gap_diagnostic_report.md:50,84,85`,
and CLAIMS.md C4.5 (`half_width 2.2`).

> **2.2 s is the median gold half-span of a NExT-GQA grounding window.** It is a metric
> parameter measured in video-timeline seconds. It has never been a wall-clock latency of
> anything, and no artifact on any ref associates it with an LLM stage.

---

**HIT 5–7 — `4.6×` / `N≈2,644`: only in the 2026-08-15 audit prose (all UNTRACKED)**

`C:\Users\akash\Documents\Iris\CLAIMS.md` · **UNTRACKED** (no commit sha)
· sha256 `5e66285a5a96a94100d1f3fe59c6215c9e76b45c53f2181e04920091b01e064c`

```
107	## Scoping rules — violating one of these is how a reviewer catches you
108	
109	- **S1 — Amdahl.** 1,104× is *retrieval mechanics only*. Never print it without the end-to-end
110	  figure in the same sentence: ~4.6× at N=4,892 against a ~2.2 s LLM stage, crossover N≈2,644.
111	  *(The e2e figures are `UNVERIFIED-LOCALLY` — confirm against an artifact before printing.)*
112	- **S2 — codec cost.** Codec edge *weighting* is cheap. Codec *extraction* is not free
```

`C:\Users\akash\Documents\Iris\IRIS_Repo_Audit_2026-08-15.md` · **UNTRACKED**
· sha256 `b287633e57a29a41e42ff60c63ecd36e4df12793f3dbb36d0d936d89b9566f20`

```
161	the 8/05 status review, and the headline doc all describe work whose artifacts you can't open.
162	
163	**3. Claim inflation between measurement and prose.** 1,104× (retrieval mechanics only; e2e is
164	4.6× at N=4,892, crossover N≈2,644). 99.85% coverage (which R0 showed is a property of UCF
165	window length, not of your selector). The question-echo result (never existed).
```

`C:\Users\akash\Documents\Iris\CLAUDE_CODE_TASKS.md` · **UNTRACKED**
· sha256 `2d1a92d3068e6ed03a09eb98488e420427ee07b89189d673c191adb3f8c50f42`
— contains the text of this task (lines 13–21, 35–45). Not evidence.

**No fourth source exists.** All three are dated 2026-08-15, all three are untracked, and
`CLAIMS.md:111` already flags the figures as unconfirmed. They are one assertion restated
three times, not three independent artifacts.

---

**NON-HITS — binary noise, ruled out explicitly**

- `video_paper/git_state.txt` (on `main`, `qv-highlights-progress`,
  `wip/laptop-preserve-2026-08-04`): matches `2644` at lines 12628, 33156, 34719, 46412,
  48272, 49162, 54134, … — these are PDF cross-reference table offsets and
  `/Length 2644` stream lengths inside a committed PDF diff dump.
- `Iris-scaling/docs/antigravity_session_log_2026-07.pdf` (6.6 MB, commit `40804c3a`):
  6 byte-level matches for `2644`, **0** for `amdahl`, `crossover`, `end_to_end`,
  `e2e_speedup`, `4.6x`, `4.6×`. Compressed-stream offsets, same PDF.
- `Iris-ucfvad/tuning/ucfcrime_vad_exp1/per_frame_*/*.csv` (~200 files): numeric matches
  for the digit string `2644` inside per-frame score columns.
- `tuning/query_reformulation_v2_report.md`: `2644` matched inside NExT-QA question IDs.

None of these is an artifact of the claimed quantity.

---

## STEP 2 — Reconcile (no averaging, no new number computed)

### 2.1 Figure (a) — the on-disk figure

| field | value |
|---|---|
| **Statement** | "query embedding is 0.016s (8x all of retrieval) and the answerer is ~60s, so retrieval latency is ~0.003% of end-to-end cost" |
| **Exact location** | `C:\Users\akash\Documents\Iris-scaling\eval_results\P_latency_result.md` **lines 25–27**; restated at `...\Iris-scaling\DECISIONS.md` **lines 334–335** |
| **Commit sha** | `c5141ce9e622028a6c9e4c1e835c2d97f8ded5a5` (both files) |
| **Blob sha** | `3bee36c624aa4a288c7df2916abba68c78fd2423` / `a48b5db27655b73c48d714d1a2ec3a76aa49240e` |
| **sha256** | `8ac3906094…` / `7b38c08ea1…` (full values in §4) |
| **N regime** | NExT-GQA **VAL**, 406 questions / 59 videos, **videos ≤ 599 frames**; buckets <100 / 100–299 / 300–599. Median retrieval 0.001959 s (flat) pooled; 0.004260 s in the top bucket |
| **Answerer model + config** | **NOT RECORDED IN THIS ARTIFACT.** The `~60s` is an unsourced prose assertion in a PERSPECTIVE paragraph. The document's own measurement harness times *retrieval only* |
| **Ingest** | **OUTSIDE** the boundary, and stated so: line 30 — "Index load (excluded from per-query): flat 3.29s, scene_sparse 3.59s for 59 videos" |
| **What was actually measured** | median per-query retrieval latency, flat vs scene_sparse, CPU / 8 threads / idle machine. Nothing else |
| **Run date** | 2026-07-24 (`DECISIONS.md` §heading line 326) |

**The `~60s` term has no timing artifact behind it in this file.** The nearest corroborating
timing on disk is:

`C:\Users\akash\Documents\Iris\eval_results\grounding_report.json` (byte-identical copy at
`...\Iris-scaling\eval_results\grounding_report.json`)
· commit `40804c3a940d413bfac7b52a9de2b1a7f5befe69` (2026-07-10)
· sha256 `3ca97bb5d035ff0e2966bc728f8a013164812ea3655bb537d272fcfd1e28c1cf`

```json
"ppr": { "avg_latency_ms": 63081.80832862854, "abstention_rate": 1.0 }
```

63.1 s — consistent with "~60s". But this artifact is **n = 1 query** on one clip
(`car_detection`), the answer was `"Insufficient verified evidence to answer this
question."` (i.e. an abstention, not a full generation), and **it records no model
identity**. Its sibling `eval_results/ablation_report.json`
(sha256 `173e72c1499e3c5d2f4de0b08574f3b66cf17768da800ec0eb85c28d0de41a97`, same commit,
timestamp 2026-06-24) records `full_iris.avg_latency_ms = 222323` (222 s) over 4 successful
clips, also with no model recorded.

### 2.2 Figure (b) — the 4.6× / 2.2 s figure

| field | value |
|---|---|
| **Statement** | "~4.6× end-to-end at N=4,892 against a ~2.2 s LLM stage, crossover N≈2,644" |
| **Exact location** | `C:\Users\akash\Documents\Iris\CLAIMS.md:110` (untracked); `IRIS_Repo_Audit_2026-08-15.md:163–164` (untracked) |
| **Commit sha** | **NONE — both files are untracked and appear on no ref** |
| **N regime** | asserted as N=4,892; **unverifiable — no run record** |
| **Answerer model + config** | **UNKNOWN. No artifact, therefore no seat, no date, no runtime, no thread count, no `cache_prompt` state** |
| **Ingest** | **UNDETERMINABLE** — no boundary is declared anywhere for this figure |

**GUARD FIRED: no artifact for the 4.6× figure exists on any ref.** Per the task
instruction, this is reported as ABSENT and no substitute figure is derived.

### 2.3 On the answerer-seat hypothesis

The task named a different answerer seat as the most likely explanation and asked whether
`DECISIONS.md` settles it. It does not, and here is precisely why.

`C:\Users\akash\Documents\Iris-scaling\DECISIONS.md` lines 3–11 (commit `c5141ce9`):

```
3	## 2026-07-13: Cerberus V2 Answerer Stack & Diagnostics
4	
5	### 1. Answerer Seat
6	**Decision:** `granite4:micro` on `llama-server` (build `b9976`) is seated as the answerer, with `cache_prompt=false` pinned. This is provisional-on-runtime.
7	**Metrics:** Seated based on bakeoff metrics.
8	*Note on Amended Latency-Gate Seating Rule:* "Models failing the latency gate cannot be seated for production query paths, regardless of ceiling metric performance."
9	
10	### 2. Vacated Answerer Seat
11	**Decision:** `qwen3.5:4b` is vacated from the answerer seat due to failing the latency gate. It is kept as a semantic-ceiling reference only.
```

Seat timeline:

| date | event |
|---|---|
| 2026-06-24 | `ablation_report.json` — 222 s/query, **model not recorded** |
| 2026-07-10 | `grounding_report.json` — 63.1 s (n=1, abstained), **model not recorded** |
| 2026-07-13 | `granite4:micro` seated; `qwen3.5:4b` vacated **for failing the latency gate** |
| 2026-07-24 | `P_latency_result.md` written, asserting "answerer ~60s" with no timing run |

Three things follow:

1. **The seat hypothesis cannot be confirmed, because figure (b) has no artifact and
   therefore no seat.** The guard "if the two figures use different answerer seats, report
   both" cannot be satisfied — there is nothing on the other side to attribute. Both figures
   are reported above; neither is picked.
2. **Figure (a)'s `~60s` is also seat-unattributed.** The only timing artifacts that
   corroborate it (2026-06-24 and 2026-07-10) *predate the granite4:micro seating* by
   3–19 days and name no model. So `~60s` is most plausibly a pre-seat number too. Neither
   published framing is currently attached to the seated answerer.
3. **`DECISIONS.md` records no latency-gate threshold number.** No bakeoff artifact exists
   in either `eval_results/` directory (`ls | grep -i bakeoff` → empty). So the gate that
   vacated `qwen3.5:4b` and seated `granite4:micro` cannot be reconstructed, and neither can
   granite4:micro's per-question latency.

### 2.4 Why the two framings look contradictory — stated as mechanism, not as a new number

They differ on **two independent axes at once**, and the audit prose collapses both:

**Axis 1 — N regime.** Figure (a) is NExT-GQA VAL at ≤599 frames, where measured median
retrieval is `0.004260 s` in the largest bucket (`P_latency_result.md:17`). Figure (b)
asserts N=4,892, where the measured flat retrieval median is `7.944805 s`
(`virat_latency_N4892_result.md:15–17`). That is a difference of roughly three orders of
magnitude in the retrieval term alone, from the same codebase, under the same graph mode.

**Axis 2 — the answerer constant.** 60 s vs 2.2 s, a ~27× difference, with no artifact
supporting either as a measurement of the seated model.

Per the task instruction, I am **not** composing an end-to-end ratio from these, and I am
not averaging them. Both terms above are quoted from their own artifacts, in their own
regimes, and left there.

### 2.5 Provenance reconstruction of figure (b) — forensic only, NOT a derivation

This subsection identifies *where the absent number probably came from*. It does not make
4.6× citable and produces no defensible e2e figure.

Taking `virat_latency_N4892_result.md`'s two measured values and the asserted 2.2 s constant:

- flat `7.944804999999178` + 2.2, over scene_sparse `0.0071983500092756` + 2.2 → **4.596**,
  i.e. "~4.6×".
- Solving for the N at which the flat retrieval curve equals 2.2 s, anchored at
  (N=4,892 → 7.9448 s) with the flat exponent **k = 2.086** fitted in
  `scaling_curve_v2.md:99–100` → **N = 2,643.3**, i.e. "N≈2,644". (The same solve with
  k = 2.0 gives 2,574 — it does not reproduce 2,644. Only the fitted 2.086 does.)

Both quoted values reproduce to within rounding. The conclusion is that **4.6× and N≈2,644
were almost certainly composed on paper** from the C1.4 retrieval measurement plus an
assumed 2.2 s answerer constant, never measured end to end. That is consistent with every
search result above, and with `CLAIMS.md:111` already marking them `UNVERIFIED-LOCALLY`.

The 2.2 s constant itself has no latency provenance anywhere in the repository. The only
2.2 s on disk is `FROZEN_HALF_WIDTH_SECONDS` — the median gold half-span of a NExT-GQA
grounding window (`eval/span.py:26`, `half_width_confirmation_report.md:17`). **A
number-collision between a grounding-window half-width and an assumed LLM stage cost is a
live possibility and should be checked with whoever wrote the team report** before 4.6× is
used again in any form.

---

## STEP 3 — What a defensible e2e number would require

### 3.1 Required measurements

| # | measurement | inputs present locally? | blocker |
|---|---|---|---|
| M1 | Answerer wall-time distribution (median + IQR, n ≥ 100) under the **seated** config: `granite4:micro` Q4_K_M ~3.4B, `llama-server` b9976, `temp 0`, `cache_prompt=false`, CPU | Model seat and harness documented; **no timing artifact exists at any seat** | **Requires running a model — out of scope for T1.** This is the single load-bearing missing number |
| M2 | Query-embedding cost at the target N | Yes for VAL: `0.016046 / 0.016091 s` (`P_latency_result.md:29`). **No** for VIRAT — `query_embed_s` is `null` by design in both arms (`virat_latency_N4892_result.md:29, 64–69`) | Needs a text-query run at N=4,892 |
| M3 | Retrieval latency at the target N **under real CLIP text queries** | **NO.** The N=4,892 numbers are seeded synthetic CLIP-embedding samples, explicitly caveated as measuring "retrieval **graph mechanics** … not real end-to-end text-query latency" (`virat_latency_N4892_result.md:64–69`). The real-text-query curve is commit `15a7d79`, which `CLAIMS.md:26` records as `git cat-file -t 15a7d79` → *"Not a valid object name"* | **BLOCKED** — artifact on a teammate's machine |
| M4 | Ingest wall time at the target N | Yes: `ingest_wall_sec: 773.8998` at N=4,892 (`virat_latency_N4892_result.md:53`); `flat_build_wall_sec: 78.82` (:60); 32-video CPU sample wall mean 2.945 s (CLAIMS C5.2) | — |
| M5 | **Queries per video** — the amortization denominator if ingest is inside the boundary | **NOT MEASURED ANYWHERE** | This is a *policy declaration*, not a measurement. Nothing in the repo declares it |
| M6 | A written boundary definition: ingest in or out; index load in or out; captioning in ingest or at query time | **NOT DECLARED ANYWHERE.** `P_latency_result.md:30` excludes index load; `virat_latency_N4892_result.md:22` excludes setup; neither states an e2e convention | Must be decided before any number means anything |
| M7 | Hardware/thread/config parity statement across M1–M4 | Partial. `P_latency` says CPU/8 threads/idle. `amort_units.json` records **no** device or thread count. C5.2 records `device_used=cpu`, `gpu_available=False` | — |

Nothing above was run. M1 and M3 are the two that actually gate the claim, and both are
currently unavailable — M1 because running it is out of T1's scope, M3 because the artifact
is on an unpushed ref.

### 3.2 Can an e2e figure be composed from `amort_units.json`?

**Artifact:** `C:\Users\akash\Documents\Iris-scaling\amort_units.json`
· commit `40804c3a940d413bfac7b52a9de2b1a7f5befe69` (2026-07-10) · blob `c163f695`
· sha256 `10fda67e338d74105bb0b8ded0c80eccd59fd694f0dcde09b6f083ceb2e27787`

Contents (verbatim values):

```
video: videoplayback.mp4   n_all: 1440   n_surv: 220   build_wall_sec: 28.9241
per_frame_decode_sec:   0.00116605      per_frame_embed_sec: 0.070859
per_frame_caption_sec:  0.81703747      caption_warmup_sec:  53.9558  (flagged one-time)
per_query_retrieve.retrieve_wall_sec: 0.0448802
phase_times: parse_video 6.0022 | embed_total 19.6172 (220 calls) | caption_total 0.0 (0 calls)
             build_graph 0.3067
build_reconciliation: accounted 195.5938 | residual -166.6697 | residual_frac -5.7623
```

**Answer: no.** An end-to-end figure cannot be composed from it, for one disqualifying
reason and eight compounding ones.

**Disqualifying:** the file contains **no answerer term at all**. Every stage it measures is
upstream of the LLM. Any e2e figure built on it must import the answerer cost from a
different run, on a different corpus, at an unknown seat — which is exactly the defect that
produced the absent 4.6× figure in the first place.

**If someone attempted it anyway, every one of these assumptions would be required:**

1. **That the file's own unit costs reconcile — they do not.** `220 × (0.00116605 +
   0.070859 + 0.81703747) = 195.594 s`, which is the recorded `accounted_sec`, against a
   measured `build_wall_sec` of **28.924 s**. The file records this itself:
   `residual_frac: -5.7623`. Multiplying these unit costs out **over-counts ingest by
   ~5.8×**. Composition from `per_frame_unit_costs` is arithmetically unsound as recorded.
2. **That captioning is in the ingest path — it was not.** `caption_calls_build: 0`,
   `caption_total_sec: 0.0`. The 0.817 s/frame caption cost is a 20-frame side probe for a
   stage that **never ran** in the build being measured. It is also the term responsible for
   ~92% of the bogus 195.6 s. Worse, the sibling commit `60bd4c7` is
   *"feat(6.3-c): lazy captioning at query time with seek-based frame fetch"* — so in the
   seated architecture caption cost belongs to the **query** side, multiplied by frames
   captioned per query, not by N.
3. **That the probe unit costs match the in-situ path — they do not.** In-situ embed is
   `19.6172 / 220 = 0.08917 s/frame` vs the probe's `0.070859` (**+26%**). In-situ
   `parse_video` is `6.0022 / 220 = 0.02728 s/frame` vs the probe's `per_frame_decode_sec`
   `0.00116605` (**23×**), because `parse_video` demuxes all 1,440 frames and runs gating and
   motion geometry, while the probe measures an isolated seek-decode of a survivor.
4. **That per-frame costs are N-independent.** They are not for the term that matters:
   `build_graph_sec = 0.3067` at N=220 is the superquadratic stage, is **absent** from
   `per_frame_unit_costs`, and cannot be scaled linearly to N=4,892.
5. **That a single 220-survivor clip transfers to the target corpus.** These units come from
   one video (`videoplayback.mp4`). The target regimes are VIRAT N=4,892 and NExT-GQA
   ≤599 frames.
6. **That the survivor ratio transfers.** This clip retains `220/1440 = 15.3%`. UCF measured
   `retention mean 10.494%` (CLAIMS C5.2); VIRAT recorded
   `skipped_frames_ratio: 0.74999` (25% retention). Three different regimes; decode is
   charged on survivors, so the choice moves the ingest term directly.
7. **That `caption_warmup_sec = 53.96` amortizes.** The file flags it
   `caption_warmup_is_large_one_time_cost: true` and provides no denominator. It exceeds the
   entire measured build wall (28.92 s) by 1.9×.
8. **That the retrieval term is the right one.** Three mutually incompatible retrieval costs
   exist on disk: `0.0448802 s` here at N=220; `0.001959 s` (flat) / `0.001909 s`
   (scene_sparse) at VAL scale; `7.944805 s` (flat) / `0.0071984 s` (scene_sparse) at
   N=4,892. They differ by graph mode, query construction (text vs seeded embedding), and N.
   A composition must declare which, and the two most-quoted ones are seeded-synthetic.
9. **That the config is the frozen one.** `amort_units.json` records
   `ppr_lambda: 0.5`, which `DECISIONS.md:34–35` calls *"a KNOWN-WRONG operating point,
   pending the λ sweep"*; and `CLAIMS.md:121–123` (S6) warns the scaling caches were built
   under weights 0.5/0.3/0.2, not the frozen 0.8/0.1/0.1.

Plus M5 and M6 from §3.1: without a declared queries-per-video denominator and a written
boundary, "end-to-end" is not yet a defined quantity for this system, let alone a measured one.

---

## 4. Artifact register — every file cited

| absolute path | commit sha | sha256 |
|---|---|---|
| `C:\Users\akash\Documents\Iris-scaling\eval_results\P_latency_result.md` | `c5141ce9e622028a6c9e4c1e835c2d97f8ded5a5` | `8ac39060949a66c509dff72ae088bba076483f1e96c44d0d767a43e09a3ecf87` |
| `C:\Users\akash\Documents\Iris-scaling\DECISIONS.md` | `c5141ce9e622028a6c9e4c1e835c2d97f8ded5a5` | `7b38c08ea1cd9b90200c47cbc0443897be4c16cd365fcc32088b2694990beee1` |
| `C:\Users\akash\Documents\Iris-scaling\eval_results\virat_latency_N4892_result.md` | `ccf764f9e2857997af34ff48351a1675cf0793f2` | `5eca39bbd53b3ff9eb3b45bf2f5fd4504744dba0af534063d4b7c99e24de3461` |
| `C:\Users\akash\Documents\Iris-scaling\eval_results\scaling_curve_v2.md` | `601df7614f0ffa1c9ecf3e4491fa5488def112ad` | `cdf96c8fe50e3078711bd2a346c32ab4e70d089888dd9f3f4a0aa9cf392748d1` |
| `C:\Users\akash\Documents\Iris-scaling\amort_units.json` | `40804c3a940d413bfac7b52a9de2b1a7f5befe69` | `10fda67e338d74105bb0b8ded0c80eccd59fd694f0dcde09b6f083ceb2e27787` |
| `C:\Users\akash\Documents\Iris-scaling\eval_results\half_width_confirmation_report.md` | `83215c00bcc9642cf35e7ef032ef77df8e70b579` | `2d4dc08dca8ce2dfebe6e9d0069a737316de5db7f30d291346537604be5448ea` |
| `C:\Users\akash\Documents\Iris\eval_results\grounding_report.json` | `40804c3a940d413bfac7b52a9de2b1a7f5befe69` | `3ca97bb5d035ff0e2966bc728f8a013164812ea3655bb537d272fcfd1e28c1cf` |
| `C:\Users\akash\Documents\Iris\eval_results\ablation_report.json` | `40804c3a940d413bfac7b52a9de2b1a7f5befe69` | `173e72c1499e3c5d2f4de0b08574f3b66cf17768da800ec0eb85c28d0de41a97` |
| `C:\Users\akash\Documents\Iris\CLAIMS.md` | **UNTRACKED — no commit** | `5e66285a5a96a94100d1f3fe59c6215c9e76b45c53f2181e04920091b01e064c` |
| `C:\Users\akash\Documents\Iris\IRIS_Repo_Audit_2026-08-15.md` | **UNTRACKED — no commit** | `b287633e57a29a41e42ff60c63ecd36e4df12793f3dbb36d0d936d89b9566f20` |
| `C:\Users\akash\Documents\Iris\CLAUDE_CODE_TASKS.md` | **UNTRACKED — no commit** | `2d1a92d3068e6ed03a09eb98488e420427ee07b89189d673c191adb3f8c50f42` |

Worktree state at time of audit: all tracked files above `clean` (no local modifications).

**Note on artifact location:** `CLAIMS.md` and `CLAUDE_CODE_TASKS.md` refer to
`eval_results/P_latency_result.md` as though it were in the main tree. It is not.
`C:\Users\akash\Documents\Iris\eval_results\` contains 9 files and does **not** include it.
The file exists only in the `Iris-scaling` worktree at detached HEAD `378daa7`, reachable
solely via `origin/siddanth/peak-source-a6-p1`. The same applies to
`virat_latency_N4892_result.md`, `scaling_curve_v2.md`, `amort_units.json`, and
`DECISIONS.md`. Citations should carry the worktree name.

---

## 5. Guard status

| guard | status |
|---|---|
| no artifact for the 4.6× figure on any ref → report ABSENT, do not derive one | **FIRED.** Reported ABSENT. No e2e figure derived or proposed. §2.5 reconstructs the *provenance* of the absent claim and explicitly does not license its use |
| the two figures use different answerer seats → report both, do not pick one | **CANNOT BE EVALUATED.** Figure (b) has no artifact, hence no seat. Both figures reported in full; neither picked. Separately: figure (a)'s `~60s` is *also* seat-unattributed and its corroborating timings predate the current seat |
| any step would require running a model → stop | **RESPECTED.** No model, harness, eval, or ingest was executed. M1 (answerer wall time) is named as required and left unrun |

---

## T1 VERDICT: ABSENT

The 4.6× end-to-end speedup, the ~2.2 s LLM stage cost, and the N≈2,644 crossover have no
supporting artifact on any of 29 refs, either stash, or any of 960 prose/code blobs in the
repository's entire history — they exist only in three untracked 2026-08-15 audit documents,
one of which already labels them `UNVERIFIED-LOCALLY`, and both quoted values reproduce
exactly as a paper composition of the measured N=4,892 retrieval figures with an assumed
2.2 s constant whose sole on-disk referent is `FROZEN_HALF_WIDTH_SECONDS`, a NExT-GQA
grounding half-width rather than any latency.
