# CLAIMS.md — IRIS claim ledger

**Frozen:** 2026-08-15 · **Owner:** Sonu · **Rule:** nothing enters a draft, an abstract, a
README, a slide, or a supervisor email unless it has a row here with status `PRINT-SAFE`.

Every row was populated by reading the named artifact on `C:\Users\akash\Documents\{Iris,
Iris-scaling, Iris-ucfvad}` on 2026-08-15. Rows marked `UNVERIFIED-LOCALLY` are numbers that
appear in team reports or chat exports but whose artifact could not be opened on this machine.

**Status vocabulary**

| status | meaning |
|---|---|
| `PRINT-SAFE` | artifact on disk, number reproduced from it, control run, scope stated |
| `PRINT-SAFE-CAVEATED` | as above, but must ship with the stated caveat in the same sentence |
| `BLOCKED` | artifact exists somewhere but not reachable here — cannot be checked or cited |
| `NOT-PRINT-SAFE` | known methodological defect; needs a re-run before it can be printed |
| `DEAD` | investigated and closed; must not appear as a positive claim anywhere |

---

## C1 — Query-latency scaling law (intended LEAD contribution)

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C1.1 | flat graph query latency is ~quadratic in N; scene-sparse is sub-linear, under **real CLIP text queries** | flat ≈ 2.0 · scene_sparse ≈ 0.82 (R² .9957, N≥1102, n=4) · `shortcut_pct` 0% | commit `15a7d79` — **`git cat-file -t 15a7d79` → "Not a valid object name"** | **BLOCKED** |
| C1.2 | same law under **seeded CLIP image-embedding** queries (superseded methodology) | flat k = 2.086 (4 UCF pts, N≤2,963) · scene_sparse k = 0.663 (6 pts, N 91→13,506) · `shortcut%` 4–24% | `Iris-scaling/eval_results/scaling_curve_v2.{md,json}`, `scaling_curve_v2_NOTES.md` | **NOT-PRINT-SAFE** — the 7/31 report classes this as measuring a code path real queries don't take |
| C1.3 | tractability divergence: flat censors, scene-sparse completes | flat `timed_out` @3600s at N=6,559 (17.0 GB RSS, climbing) and N=13,506 (26.9 GB, climbing); scene_sparse completes both at 0.0104 s / 0.0210 s | `scaling_curve_v2.md` per-clip table | **PRINT-SAFE-CAVEATED** — frame as *tractability divergence*, never as a clean exponent ratio; uniform 3600 s / 29 GB watchdog on every flat arm incl. VIRAT |
| C1.4 | point comparison at N=4,892 (VIRAT) | flat 7.9448 s vs scene_sparse 0.0072 s = **1,104×** retrieval-mechanics ratio; flat PPR graph 4,892 nodes / 11,963,386 edges vs 214.5 / 1,264 | `Iris-scaling/eval_results/virat_latency_N4892_result.md` | **NOT-PRINT-SAFE** *(downgraded 2026-08-15 by T1)* — the ratio itself is sound, but S1 requires an end-to-end correction beside it and **no valid end-to-end figure exists** (see DEAD row D8). Also: this clip is **mpeg4, not H.264**, and its own ingest log warns MV export may be unavailable — no codec-feature claim may lean on it |
| C1.5 | exponents carry confidence intervals | — | **no artifact** | **NOT-PRINT-SAFE** — fit is n=4 with no error bar; needs survivor census + binned latency over the 350-clip corpus |

## C2 — Block-diagonal construction (STRONGEST; fully verified here)

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C2.1 | block-diagonal build is **bit-identical** to dense-then-prune | `GATE_PASS`; N=4,892, 528 scenes; edges 23,571 = 23,571 = 23,571 (theoretical/OLD/NEW); node set identical; 0 edges only-in-A, 0 only-in-B; **0** field mismatches on `weight`/`semantic_weight`/`motion_weight`/`temporal_weight`/`edge_type` at tolerance **0.0**; PageRank bit-identical on all 4,892 nodes; PPR top-20 order + exact scores identical across 5 seeds | `Iris-scaling/eval_results/blockdiag_identity_gate_result.{json,md}` · commit `3d83b2c` present in local object DB | **PRINT-SAFE** |
| C2.2 | measured build savings | wall 69.47 s → 0.316 s (**~220×**); peak RSS 6.36 GB → 0.95 GB (**~6.7×**) | `blockdiag_gate_and_savings_summary.md`, `blockdiag_probe_{old,new}.json` | **PRINT-SAFE-CAVEATED** — isolates `_build_graph` from cached frames/embeddings only; not a full-ingest number |
| C2.3 | grounding gate unaffected | 526 NExT-GQA VAL questions, 0 mismatches | `blockdiag_grounding_gate_result.{json,md}` | **PRINT-SAFE** |
| C2.4 | codec edge-**weighting** is the cheapest signal, given the demux | at N=2,963: weight-only 0.007 s (codec) vs 0.042 s (pixel-diff) vs 0.027 s (semantic, embeddings assumed) | `Iris-scaling/eval_results/build_cost_3way.{md,json}` | **PRINT-SAFE-CAVEATED** — see scoping rule S2; extraction is **not** free (codec 26.843 s vs pixel-diff 12.390 s at the same N) |

## C3 — The budget-matched control (RIGOUR section; this is Sonu's)

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C3.1 | coverage metrics are uninformative without a budget-matched control — uniform saturates M1 | M1 = **1.0000** for arms A/C/D at **every** budget swept, down to 5% | `Iris/r0_full/summary.md` (`scripts/r0_gate_full.py`) | **PRINT-SAFE** — kill criterion pre-registered and TRIGGERED |
| C3.2 | codec admission is statistically indistinguishable from uniform at matched budget | A−C at 10.5%: dM2 **+0.256 pp** CI [-0.236, +0.710]; dM3 **+0.780 pp** CI [-0.005, +1.753] (10,000 percentile bootstrap, seed 20260805). Both span zero | same | **PRINT-SAFE-CAVEATED** — n=19; each video is 5.3 pp of any aggregate |
| C3.3 | the admitted set sits at the label-blind identity | mean(M2 − retention) = **+0.0017** (sd 0.0099); mean(M3 − gold fraction) = **+0.0073** (sd 0.0212) | same | **PRINT-SAFE** — this is the mechanism-level version of C3.2 and the stronger statement |
| C3.4 | the "residual-pressure-scaled budget" is a fixed quota | retention 10.41–11.14% across all 19 videos | same | **PRINT-SAFE** |
| C3.5 | admission score was never persisted | `is_retained_tier` thresholds `ps` (normalized packet size, `charon_v.parse_video`); `ps` is absent from every CSV. Phase-1 diagnostic stopped at the guard rather than substituting `action_score_propagated` | `Iris/_r0diag/CODEC_GOLD_AUC.md` | **PRINT-SAFE** as a reproducibility/limitations note |
| C3.6 | R0 at journal scale | n=140 | **no artifact** — needs the 140-video CSVs (see BLOCKED) | **NOT-PRINT-SAFE** |

## C4 — Correctness floor, NExT-GQA (held out)

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C4.1 | Acc@GQA on a held-out split | **0.1667** [0.088, 0.243], n=120 / 27 videos; Acc@QA 0.375 [0.275, 0.477]; parser 120/120 clean; answerer `granite4:micro` ~3.4B Q4_K_M, temp 0, `cache_prompt=false`, CPU | `Iris-scaling/eval_results/P_NOWA_accgqa_result.md` | **PRINT-SAFE-CAVEATED** — n=120 from a val-derived split with ~8 pt CI; published figures are full-test. Placement is *indicative*, not a leaderboard entry |
| C4.2 | grounded answers are more often correct | P(correct\|grounded) 0.476 vs P(correct\|ungrounded) 0.321; gap replicated in-sample (65 vs 46) and held out (47.6 vs 32.1) | same | **PRINT-SAFE** |
| C4.3 | grounding metrics | peak_in_gold 0.3667 [0.287, 0.445] · mIoP 0.3349 [0.249, 0.424] · IoP@0.5 0.3500 [0.263, 0.444] · mIoU 0.1855 [0.140, 0.228] · IoU@0.5 0.1333 | `P_NOWA_grounding_result.md` | **NOT-PRINT-SAFE** — mIoP/IoP use the **union** convention, biased upward vs the leaderboard's max-per-span. Needs a re-run; the raw dump has no predicted spans |
| C4.4 | retrieval beats uniform frame sampling on answer accuracy | held out **+0.120 [0.000, 0.248]** — CI touches zero. vs **random**: +0.127 [+0.036, +0.229] | `P_NOWA_accgqa_result.md` (CORRECTION 2) | **PRINT-SAFE-CAVEATED** — only the *vs random* form is separated; state both |
| C4.5 | the tuned config is effectively the default | top_k 12 / half_width 2.2 vs top_k 8: mIoP +0.0068 [-0.0132, +0.0264], IoP@0.5 +0.0000 | `P_NOWA_grounding_result.md` | **PRINT-SAFE** |
| C4.6 | test half status | **BURNED.** No further test-half measurement without a new split | same | **PRINT-SAFE** as a constraint on future work |

## C5 — Ingest cost

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C5.1 | ingest performs **zero** neural forward passes | `total_nn_forward_pass_calls_across_all_sampled_videos = 0` over 32 videos, verified by monkeypatching `torch.nn.Module.__call__` for the duration of every `parse_video()` and counting per-video, **not asserted from source** | `Iris-ucfvad/tuning/ucfcrime_vad_exp1/efficiency_measurements.json`; independently reproduced in `COUNTED` mode in `Iris/r0_full/summary.md` | **PRINT-SAFE** |
| C5.2 | ingest wall time and memory, CPU-only | 32-video deterministic stratified sample: wall mean 2.945 s (min 0.337, max 15.474); peak RSS mean 530 MB, max 1.291 GB; retention mean 10.494% | same | **PRINT-SAFE-CAVEATED** — `device_used = cpu`, `gpu_available_on_this_box = False`. State that the box had no GPU; a CPU-only run on a GPU-equipped box is the stronger demonstration and has not been done |

## C6 — The honest codec negative (one sentence + one number in the paper)

| # | Claim | Number | Artifact | Status |
|---|---|---|---|---|
| C6.1 | codec admission carries no information about gold-frame location on UCF-Crime | C3.2 + C3.3 | `Iris/r0_full/summary.md` | **PRINT-SAFE** |
| C6.2 | within-video localization AUC, partial corpus | macro AUC **0.5650** over 19 anomalous videos; pooled 0.7232 (all frames, propagated) / 0.7086 (retained only) over 169 videos | `Iris-ucfvad/tuning/ucfcrime_vad_exp1/stageA_results.json` | **PRINT-SAFE-CAVEATED** — the file's own `IMPORTANT_CAVEAT`: 169/290 videos, 9 categories entirely absent, **not** comparable to full-290 LAVAD / EventVAD figures |
| C6.3 | full-corpus VAD diagnostic (8 features, 140 videos, flow-structure + kinematic/semantic hypotheses both falsified; motion_magnitude macro AUC 0.653 [0.626, 0.679]) | — | **no artifact reachable** — `grep -rl motion_entropy` over `Iris/` returns 0 hits | **BLOCKED** / UNVERIFIED-LOCALLY |

---

## BLOCKED — artifacts that exist on a teammate's machine and nowhere reachable

Newest fetched ref from any teammate: `origin/siddanth/ucf-vad-exp1` @ `a5cea25`, **2026-07-29**.
Everything below postdates it.

**Citation-path rule (added 2026-08-15).** Every artifact path in this ledger must carry its
working-tree prefix. Five cited artifacts — `P_latency_result.md`, `virat_latency_N4892_result.md`,
`scaling_curve_v2*`, `build_cost_3way*`, `blockdiag_*`, `P_NOWA_*` — exist **only** in
`Iris-scaling` at detached HEAD `378daa7`, reachable solely via `origin/siddanth/peak-source-a6-p1`.
A bare `eval_results/...` path does not resolve from the `Iris` main checkout.

**Untracked-artifact alert (added 2026-08-15).** `r0_full/`, `r0_smoke/`, `_r0diag/`, `_n6audit/`
and this file are **UNTRACKED** in `Iris`. C3 — the one contribution that is unambiguously
Sonu's — exists on exactly one laptop, in no commit, on no remote. Commit and push before
anything else in this section.

| item | needed for | ask |
|---|---|---|
| commit `15a7d79` — scaling curve v3, real CLIP text queries | **C1.1, the paper's lead number** | push to `origin` |
| 140-video UCF per-frame CSVs (8 features + ground truth) | C3.6 (R0 at n=140), C6.3 | push the branch |
| exact venue / scope / deadline from the Dr. Uma meeting | all sequencing | recover from Siddanth or Swara |

---

## DEAD — closed with evidence; must not appear as a positive claim

| claim | why it's dead | evidence |
|---|---|---|
| **N6 question-echo caption finding** (0.704 / 0.148 / "within 0.4pt" / 0.699 / 0.325) | no artifact on any of 206 commits across 31 refs; scorer definition absent; `wrong_question` and `donor_swap` return 0 hits on every ref | `Iris/_n6audit/N6_EVIDENCE.md`; prior audit at `457355da4` states *"no question-aware caption arm exists in this dump"* |
| **VAD accuracy line** (codec beats VAD SOTA) | 0.653 vs EventVAD 0.82; both pre-registered hypotheses falsified; codec-MV VAD is 2014 prior art (Biswas & Babu) | 7/31 team report §2–3 |
| **Cerberus-V / NLI verification as a measured contribution** | never measured, zero results anywhere in the project | strategic reset 2026-08-02 |
| **"codec residual as a unified cognitive controller"** | only the retrieval half was ever built; the gating half is C6.1 null | README (still asserts it — must be rewritten) |
| **99.85% gold coverage at 13.4% retention, framed as selector quality** | R0 showed uniform reaches M1 = 1.0000 at 5%; the number measures UCF window length, not selection | `Iris/r0_full/summary.md` |
| **"sparse graph is a better graph"** | wrong framing — the result is *provably identical, dramatically cheaper* | C2.1 |
| **"long-form video ⇒ our selection beats uniform"** | UCF-Crime **is** the long-form corpus and uniform tied there. The long-form win is **query-cost scaling**, not selection | C1.3, C3.2 |
| **D8 — the Amdahl correction: "~4.6× end-to-end at N=4,892, ~2.2 s LLM stage, crossover N≈2,644"** | **Never measured. Composed on paper from a unit error.** Zero artifacts across 29 refs, 2 stashes and all 2,464 history objects. The "2.2 s LLM stage" is `FROZEN_HALF_WIDTH_SECONDS` — the NExT-GQA grounding half-span in seconds, never a wall-clock latency. Both figures reproduce exactly from it: (7.9448+2.2)/(0.0072+2.2) = **4.596**, and solving flat = 2.2 s at the fitted k=2.086 gives N = **2,643.3** (k=2.0 gives 2,574 — only the fitted exponent reproduces 2,644) | `eval_results/T1_e2e_reconciliation.md`; arithmetic independently reproduced 2026-08-15 |
| **the ~60 s answerer figure as an end-to-end anchor** | unsourced prose in a PERSPECTIVE paragraph. Corroborating timings (`grounding_report.json` 63.1 s, n=1, abstained; `ablation_report.json` 222 s) date to 2026-07-10 / 06-24, both **predating** the `granite4:micro` seating on 2026-07-13, and neither records a model. No bakeoff artifact exists, so the latency gate that vacated `qwen3.5:4b` cannot be reconstructed | `Iris-scaling/eval_results/P_latency_result.md`; T1 Step 2 |

---

## Scoping rules — violating one of these is how a reviewer catches you

- **S1 — Amdahl. REWRITTEN 2026-08-15 after T1.** 1,104× is *retrieval mechanics only*, and
  **IRIS currently has no defensible end-to-end cost claim of any kind.** The correction that
  circulated for weeks (~4.6×, ~2.2 s LLM stage, crossover N≈2,644) is `DEAD` — see D8. Until an
  end-to-end measurement exists at the seated answerer config, 1,104× may be printed **only** as
  an explicitly-scoped retrieval-mechanics ratio, in a sentence that names the exclusion:
  ingest, query embedding, and the answerer are all outside the boundary. Do not offer a
  self-audit correction you cannot source — an unsourced correction is worse than a scoped claim.
- **S2 — codec cost.** Codec edge *weighting* is cheap. Codec *extraction* is not free
  (26.843 s vs pixel-diff 12.390 s at N=2,963). The defence is that the demux runs anyway for
  frame selection — say that explicitly rather than quoting the weighting number bare.
- **S3 — wall time, not memory.** Flat's frontier is `timed_out`, never `oom`, at every N
  tested. Say "tractability divergence", not "flat runs out of memory".
- **S4 — retrieval speedup ≠ end-to-end speedup.** Applies everywhere, not just S1.
- **S5 — persist the raw score.** Standing engineering rule after losing `ps`, the raw action
  score, and the caption metrics: whenever a derived decision is written to disk, write the
  continuous score it was derived from in the same row.
- **S6 — config provenance.** The scaling-curve `.npz` caches were built under weights
  0.5/0.3/0.2, not the frozen 0.8/0.1/0.1. Latency claims survive this; any **retention** or
  **survivor-count** number quoted from those caches does not. Reconcile or state it.
- **S7 — never tune on UCF test.** Frame-level labels exist only in the 290 test videos.
  Any reweighting needs a split declared before looking.

---

## Change log

| date | change |
|---|---|
| 2026-08-15 | Ledger created and frozen from a read-only audit of all three working trees. |
| 2026-08-15 | **T1 result applied.** C1.4 downgraded `PRINT-SAFE-CAVEATED` → `NOT-PRINT-SAFE`. S1 rewritten. New DEAD rows D8 (the 4.6× / N≈2,644 Amdahl correction — never measured, composed from a unit error) and the ~60 s answerer anchor. Added citation-path rule and untracked-artifact alert. Arithmetic reconstruction independently reproduced before acceptance. |
