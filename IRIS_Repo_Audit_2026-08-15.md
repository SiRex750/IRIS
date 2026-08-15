# IRIS — repo audit and route to a submittable paper

**Date:** 2026-08-15 · **Audited by:** Claude (strategist/auditor role), read-only pass over
`C:\Users\akash\Documents\Iris`, `Iris-scaling`, `Iris-ucfvad`, `C:\Users\akash\Downloads`.

Every number below was read out of a file on your laptop during this audit, or is explicitly
marked as **not present**. Nothing is recalled from chat history.

---

## 0. The short version

Your friend's efficiency arm is good — you're right about that. But **the single number it
leads with is not on your machine**, and neither is the 140-video VAD run. Your last fetched
commit from any teammate is `a5cea25`, dated **2026-07-29**. The report you're working from is
dated **2026-07-31** and describes work done after that.

So the honest state is not "we have a good paper, make it publishable." It is:

- Two contributions are **verified on your disk right now** (block-diagonal identity; NExT-GQA
  correctness floor).
- One contribution is **verified on your disk and is yours** (the R0 budget-matched control).
- The **lead** contribution — the corrected scaling law — exists only as an unpushed commit on
  a teammate's machine.

The first move is not writing and it is not an experiment. It's a `git push` you don't control.

---

## 1. What is verified on your disk

### C2 — Block-diagonal construction (STRONGEST, and it holds up)

`Iris-scaling/eval_results/blockdiag_identity_gate_result.{json,md}`,
`blockdiag_gate_and_savings_summary.md`. Commit `3d83b2c` **is** in your object DB
(on `origin/siddanth/peak-source-a6-p1`).

| | value |
|---|---|
| outcome | `GATE_PASS` |
| N survivors / scenes | 4,892 / 528 |
| edges, theoretical vs OLD vs NEW | 23,571 / 23,571 / 23,571 |
| edge-field mismatches | **0**, exact float equality, tolerance 0.0 |
| PageRank | bit-identical on all 4,892 nodes |
| PPR top-20 order | identical across 5 seeded personalization vectors |
| build wall | 69.47 s → 0.316 s (**~220×**) |
| build peak RSS | 6.36 GB → 0.95 GB (**~6.7×**) |

This is a proof-grade equivalence check with a measured speedup. Almost nothing in this
literature ships that. It is indifferent to whether codec carries signal, so R0 did not touch it.

Framing discipline: the claim is **"provably identical, dramatically cheaper."** Not "sparse is
a better graph." Do not let that drift.

### C4 — NExT-GQA correctness floor

`Iris-scaling/eval_results/P_NOWA_accgqa_result.md`, `P_NOWA_grounding_result.md`.

Acc@GQA **0.1667** [0.088, 0.243], n=120 / 27 videos · Acc@QA 0.375 · mIoP 0.3349 ·
IoP@0.5 0.3500 · mIoU 0.1855 · peak_in_gold 0.3667 · P(correct|grounded) 0.476 vs
P(correct|ungrounded) 0.321 · parser 120/120 clean.

Two things already recorded in that file that you must carry into the paper rather than
rediscover under review:

- **The test half is burned.** No further test-half measurement without a new split.
- The proposed-vs-uniform Acc@QA advantage is **+0.120 [0.000, 0.248]** held out — the CI
  touches zero. It survives against *random* (+0.127 [+0.036, +0.229]). Write it that way.

### C3 — R0, the budget-matched control (this one is yours)

`Iris/r0_full/summary.md`, produced by `Iris/scripts/r0_gate_full.py`. n=19, forward-pass
counter in `COUNTED` mode, **0** nn.Module forward passes during ingest.

- **M1 = 1.0000 for every arm at every budget down to 5%.** Kill criterion TRIGGERED: window
  hit-rate cannot headline anything, uniform saturates it at half your budget.
- Arm A − Arm C at 10.5%: **dM2 +0.256 pp, CI [-0.236, +0.710]** · **dM3 +0.780 pp,
  CI [-0.005, +1.753]**. Both span zero.
- Distance from the label-blind identity: mean (M2 − retention) = **+0.0017**,
  mean (M3 − gold fraction) = **+0.0073**. The admitted set sits at the base rate.
- Retention 10.41–11.14% across all 19 videos — the "residual-pressure-scaled budget" is a
  fixed quota, not adaptation.

Supporting mechanism doc: `Iris/_r0diag/CODEC_GOLD_AUC.md`. Admission is thresholded on `ps`
(normalized packet size, `charon_v.parse_video`), which was **never written to the CSV**. The
diagnostic stopped at the guard rather than substituting a proxy. That is the correct call and
it is the reason the null is a *mechanism* finding, not just a statistical one.

### N6 — settled, and it's a zero

`Iris/_n6audit/N6_EVIDENCE.md`. All 206 commits across 31 refs searched, including
`refs/stash` and `wip/laptop-preserve-2026-08-04`. All four caption numbers **ABSENT**; the
scorer definition **ABSENT**; `wrong_question` and `donor_swap` return 0 hits on every ref. A
prior audit already committed at `457355da4` says
*"(no question-aware caption arm exists in this dump to compare against)"*.

Stop counting N6 as an asset. It is not a backup paper.

---

## 2. What is NOT on your disk — and this is the actual blocker

### 2.1 The lead number cannot be opened by you

The 7/31 report and the headline doc quote **flat ≈ 2.0 / scene_sparse ≈ 0.82 (R² .9957)**
from real CLIP *text* queries, committed as `15a7d79`, "not yet pushed."

```
$ git cat-file -t 15a7d79
fatal: Not a valid object name 15a7d79
```

What you actually have is **v2**, in `Iris-scaling/eval_results/scaling_curve_v2*`:

- flat k = **2.086** (4 UCF points, N ≤ 2,963)
- scene_sparse k = **0.663** (all 6 UCF points)
- seeded CLIP **image**-embedding queries, `shortcut% 4–24%`

v2 is the run the 7/31 report itself calls *"measuring a code path real queries don't take."*
So the exponent you would put in the abstract is one you cannot reproduce, cite a commit for,
or defend to a reviewer today.

### 2.2 The 140-video VAD diagnostic is not here either

`grep -rl motion_entropy` across your `main` checkout: **zero hits**. The eight-feature
per-frame CSVs, the Spearman r ≥ 0.956 redundancy result, and the motion_magnitude macro AUC
0.653 [0.626, 0.679] all live somewhere you can't reach.

What you do have is the 7/29 lineage in `Iris-ucfvad`: 169 ground-truth CSVs
(19 anomalous + 150 Normal), 32 action-score CSVs, `stageA/stageB/efficiency` JSONs, at
`a5cea25`. That is why R0 ran at n=19 — **not** an error on your part given what was fetched,
but the re-run at n=140 is blocked on the same push.

### 2.3 Housekeeping

- **No `CLAIMS.md`.** `DECISIONS.md` is 26 lines, last touched 2026-07-13, and is about
  answerer seating — not claims.
- **No paper draft anywhere.** No `.tex`, no draft doc, in any of the three trees.
- **README is still on the dead thesis:** *"Target venues: CVPR / MLSys"*, *"codec residual
  energy as a unified cognitive controller"*, the Cerberus-V NLI gate, and a team table listing
  "Teammate 2 / 3 / 4". Every one of those is now false.
- Your working tree shows 70 modified files, but `git diff --ignore-all-space` is **empty** —
  it's pure CRLF churn from the Windows round-trip. Don't commit it. Set `core.autocrlf` before
  it pollutes a real diff.
- Video corpus: `Downloads/Anomaly-Videos-Part-1` = 200 mp4 (Abuse/Arrest/Arson/Assault only).
  Parts 2–4 not downloaded. Irrelevant if the CSVs arrive; relevant if you have to re-ingest.

---

## 3. What went wrong — the pattern, not the incidents

**1. The harness keeps throwing away the raw continuous score.** Three times now: the codec
admission score `ps`, the raw action score, the caption metrics. Each time, a derived decision
was persisted and the number underneath it wasn't — and each time it cost a re-run or killed a
question outright. Adopt the rule and put it in the repo: *whenever you write a decision, write
the score it was derived from, in the same row.*

**2. Results live on personal laptops and in chat exports, not in `origin`.** Your `Downloads`
folder has ~40 IRIS documents including **seven roadmap versions (v2 → v8)** and six overlapping
handoff files. That volume of restatement is the symptom, not the archive. The 7/31 report,
the 8/05 status review, and the headline doc all describe work whose artifacts you can't open.

**3. Claim inflation between measurement and prose.** 1,104× (retrieval mechanics only; e2e is
4.6× at N=4,892, crossover N≈2,644). 99.85% coverage (which R0 showed is a property of UCF
window length, not of your selector). The question-echo result (never existed). None of these
were dishonest — they were a number measured in one scope and repeated in a wider one. A
journal reviewer catches this in the abstract-vs-table comparison.

**4. Pivot-as-avoidance.** Accuracy → VAD → efficiency reframe → codec-null → "scrap it and
restart from the original repo." Four pivots, and not one of them was ever the blocker. The
blocker has been the same thing since July: **nothing is written down as a paper.** You are
about to hit this urge again — the codec result feels bad, so a new domain or a new detector
feels productive. It isn't. The paper survives the codec collapse *by design*; that was the
whole point of building the control.

**One thing you should stop apologising for:** the codec code is not bad. It computes a
motion/residual signal correctly. On UCF-Crime, "visually busy" is orthogonal to "anomalous" —
that's a hypothesis mismatch, not a bug. And R0 is not a consolation prize. A budget-matched
control that kills your own headline is exactly the section that buys reviewer trust at a
systems venue, and it's the part of this paper that is unambiguously yours.

---

## 4. What to do, in order

### Step 0 — today, and it is not optional

**Get Siddanth to push `15a7d79` and the 140-video VAD branch to `origin`.** Then
`git fetch --all` and re-verify. Ask for exactly this:

> Push the v3 scaling-curve commit (`15a7d79`, real CLIP text queries) and the branch holding
> the 140-video UCF per-frame CSVs. I can't open either one and the abstract depends on the
> first.

Until that lands, every hour spent on prose is spent around a number nobody in this repo can
verify. Also recover, from Siddanth or Swara, **what Dr. Uma actually said** — the exact
journal, scope, and any deadline. "Some journal" is not a target and it changes the work list
below materially.

### Step 1 — freeze `CLAIMS.md` (do this before any prose)

One row per claim: statement · exact number · the file it came from · the commit · the control
that was run · print-safe yes/no. The framing has moved three times and the README has drifted
back to dead claims three times. A journal reviewer punishes abstract-vs-table inconsistency
harder than a modest result. This is a two-hour job and it is the highest-leverage two hours
available to you.

### Step 2 — rewrite `README.md` to match reality

Kill "CVPR / MLSys", "unified cognitive controller", and the Cerberus-V block. State the actual
contribution stack. Ten minutes, and it stops the drift permanently.

### Step 3 — the four not-print-safe numbers (journal review demands all four)

1. **R0 at n=140** — same script, same pre-registration, only needs the CSVs from Step 0.
   Turns a suggestive n=19 null into a tight one. This is what makes C3 publishable.
2. **Exponent CIs** — bin the 350-clip corpus by survivor N. Your headline currently rests on
   an n=4 fit with no error bar. Highest-leverage upgrade to the lead contribution.
3. **mIoP / IoP@0.5 rescore** on the official max-per-span convention. Yours is union, which is
   biased upward relative to the leaderboard. Needs a re-run; the raw dump has no predicted spans.
4. **Verify every competitor cell against the source PDFs.** Two of three already have known
   internal errors (MUPA §4.1 says 29.0/39.7 where Tables 1 and 3 say 28.7/39.1; EgoSG Table 3
   is column-permuted vs Table 2). A wrong competitor number is worse than a low own-number.

### Step 4 — draft

Contribution order: scaling law (lead) → block-diagonal identity (bulletproof) →
budget-matched control (rigour) → correctness floor. Codec gets **one sentence and one honest
negative**. The property matrix — training-free end-to-end, no frontier LLM, CPU-only, fitted
latency-vs-N law, grounding eval — is your opening move against MUPA / EgoSG / GOPAgen.

### Explicitly not now

No new domain. No swapping the detector. No sports corpus. No rebuild from the original repo.
No N4, no N6 reconstruction, no chasing Acc@GQA. A different-motion domain cannot revive a
signal that is absent at the exact operating point of the method — that question is answered,
and answering it again costs weeks you don't have against a journal cycle.

---

## Appendix — provenance of every claim in this document

| Claim | Source file on your disk |
|---|---|
| blockdiag gate, 220×/6.7×, 23,571 edges | `Iris-scaling/eval_results/blockdiag_identity_gate_result.json`, `blockdiag_gate_and_savings_summary.md` |
| Acc@GQA 0.1667, grounding metrics, burned test half | `Iris-scaling/eval_results/P_NOWA_accgqa_result.md`, `P_NOWA_grounding_result.md` |
| R0 n=19 full result, CIs, kill criterion | `Iris/r0_full/summary.md` |
| `ps` never persisted, guard tripped | `Iris/_r0diag/CODEC_GOLD_AUC.md` |
| N6 absent across 206 commits / 31 refs | `Iris/_n6audit/N6_EVIDENCE.md` |
| scaling v2 exponents 2.086 / 0.663, shortcut 4–24% | `Iris-scaling/eval_results/scaling_curve_v2.md`, `scaling_curve_v2_NOTES.md` |
| `15a7d79` absent | `git cat-file -t 15a7d79` → "Not a valid object name" |
| 140-video / 8-feature run absent | `grep -rl motion_entropy` over `Iris/` → 0 hits |
| newest fetched teammate ref = 2026-07-29 | `git for-each-ref --sort=-committerdate` |
| 169 + 32 CSVs, 19 anomalous | `Iris-ucfvad/tuning/ucfcrime_vad_exp1/` |
| README on dead claims | `Iris/README.md` lines 1–20 |
| dirty tree is CRLF only | `git diff --stat --ignore-all-space` → empty |
| 200 mp4 available locally | `Downloads/Anomaly-Videos-Part-1` |
