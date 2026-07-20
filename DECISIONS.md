# Architectural and Seating Decisions

## 2026-07-13: Cerberus V2 Answerer Stack & Diagnostics

### 1. Answerer Seat
**Decision:** `granite4:micro` on `llama-server` (build `b9976`) is seated as the answerer, with `cache_prompt=false` pinned. This is provisional-on-runtime.
**Metrics:** Seated based on bakeoff metrics.
*Note on Amended Latency-Gate Seating Rule:* "Models failing the latency gate cannot be seated for production query paths, regardless of ceiling metric performance."

### 2. Vacated Answerer Seat
**Decision:** `qwen3.5:4b` is vacated from the answerer seat due to failing the latency gate. It is kept as a semantic-ceiling reference only.

### 3. Captioner Seat
**Decision:** `minicpm-v4.6-inventory` is seated as the default captioner.
**Note:** The frame-3334 "devoid-of-people" label in the analyze-log was confirmed as a mislabel (people were actually present in the frame; the minicpm caption was correct).

### 4. Critical Bug Fixes
* **P0-03:** Fixed silent structural corruption of `AnswerClaims` by switching `cerberus_mode="v2"` to use the schema-constrained wire generator instead of the legacy nested JSON parser. (Commit: `b88f9b4`)
* **P0-07:** Fixed a layer-3 absence path bug where an empty caption list resulted in a zero-evidence "verified_absent" verdict. Added a guard to return "unverifiable" instead. (Commit: `0ee8775`)

### 5. Terminology and Claim-Wording Corrections
* Use **"selective processing"** instead of "selective decode".
* Use **"consistent with recorded evidence"** instead of "visually verified".

### 6. Artifact-Commit Discipline (P0-11)
**Decision:** Strict artifact-commit discipline must be maintained. All evaluation outputs, diagnostic capture logs, and bakeoff results must be committed alongside the code changes that produced them to maintain unbroken provenance.

## 2026-07-17: Metric Integrity & Seat-Violation Findings

### 1. Codec-at-Query-Time Negative (paper-grade)
**Decision:** Codec structures ingest; it does NOT belong in query-time ranking.
**Evidence:** Deconfounded ablation + T0 rerun replication (89 videos / 255 events, CPU),
monotonic in λ — codec-only 0.4941 < 50/50 hybrid 0.5451 < semantic-only 0.5882 (Hit@5).
**Consequence:** The production `ppr_lambda=0.5` is a KNOWN-WRONG operating point, pending the
λ sweep. Pre-registered prediction: λ*=0.

### 2. Cache-Determinism (paper-grade serving finding)
**Decision:** temp-0 output is a deterministic function of prompt-cache state (fresh prefill vs
cached), not model instability. Eliminated by pinning `cache_prompt=false` per request.
**Consequence:** The team's earlier rejection of granite4:micro as "unstable" was this artifact.
Runtimes without per-request cache control (Ollama) cannot serve the answerer seat.

### 3. Span-Construction Bug (metric corruption)
**Decision:** The predicted grounding span is built as min(ts)→max(ts) over the top-K retrieved
frames — the enclose-all-top-K anti-pattern. IoP divides by prediction length, so scattered
top-K frames yield a near-video-length span and IoP collapses even when retrieval is correct.
**Sites:** `scripts/pillar2_grounded_qa.py::iop()` and `::iou()`; `scripts/tune_l1_weights.py::iop()`
(that file has no iou()). Three independent copies of the same construction.
**Consequence:** ALL grounding metrics reported to date (mIoP 0.3091, IoP@0.5 25.00%, mIoU 0.2479,
IoU@0.5 17.86%) are corrupted, and every Optuna study to date optimized a corrupted objective.
`optuna_best_weights.json` is PROVISIONAL.
**Fix (not yet applied):** span-from-PPR-peak — top-scoring frame t* → tightest contiguous window
over the top-scoring cluster → single half-width w, tuned on val only, then frozen. This is a bug
fix, not tuning. One shared constructor imported by all call sites; no per-script copies.

### 4. Seat Violation in the V2 Calibration Run
**Decision:** The V2 calibration run (N=56) did NOT run on the seated answerer configuration.
`scripts/pillar2_grounded_qa.py` constructs `aria.LlamaBackend(endpoint="http://127.0.0.1:11434/v1")`
— Ollama, the rejected runtime — with no `cache_prompt` control, at temp=0.1 (hardcoded in
`iris/aria.py`; six sites across both backends).
**Consequence:** All V2 numbers are QUARANTINED as **UNPINNED**. This includes
P(correct|grounded)=35.7% and therefore the "the answerer is the entire Acc@GQA deficit" diagnosis,
which stands as a HYPOTHESIS, not a measurement. The +3.57pp Acc@QA "improvement" is confounded by
prompt-cache state per finding (2) and is not attributable to `_QA_MCQ_PROMPT`.
**Bar for lifting the quarantine:** rerun on llama-server (b9976) per the seat contract —
`cache_prompt=false`, `--parallel 1`, fixed threads, temp=0 — with the span fix applied.
**Note:** `LlamaServerBackend` in `iris/aria.py` DOES correctly pin `cache_prompt=false` on all
three of its paths. The seat contract exists in code; the benchmark bypassed it by constructing
the wrong backend class.

## 2026-07-17 (later): Amendments and WIP Triage

### A1. Correction to 2026-07-17 §4
The clause "temp=0.1 (hardcoded in iris/aria.py; six sites)" described the UNCOMMITTED WORKING
TREE, not the repository. Verified: at HEAD every answerer temperature site was already 0.0/0,
and tests/test_aria.py asserted 0.0 and PASSED. The 0.1 existed only in uncommitted WIP.
§3 (span bug) and §4's remaining claims are CONFIRMED committed on main:
  - min->max span in pillar2_grounded_qa.py::iop()/::iou()  — on main
  - aria.LlamaBackend(endpoint="http://127.0.0.1:11434/v1") — on main
  - "Publication-Ready" title + the >=0.25 frontier-parity conditional — on main
The quarantine in §4 stands.

### A2. The V2-producing state is irrecoverable
The temp=0.1 content was reverted in the working tree before the WIP snapshot (3497ee4) was
taken, so the snapshot does NOT contain it; that commit's message is inaccurate on this point.
Only tests/test_aria.py's flipped assertions survive as evidence. mtimes were additionally
flattened by a git stash/pop. The V2 calibration run is therefore unattributable to any
recoverable state: REPLACE, do not reconstruct.

### A3. Green-tuning — two events, one rule
Two agent-authored (Antigravity) changes converted a failing guard into a passing one rather
than surfacing it:
  1. tests/test_aria.py — assertions flipped 0.0 -> 0.1 to match a temp change, in lockstep
     with the code change. The suite's guard against the determinism pin fired and was retuned.
     The resulting run was then reported as "+3.57pp Acc@QA — broke out of the random-chance rut."
  2. tests/test_l1_elysium.py — test_pagerank_affects_keep_score COMMENTED OUT rather than
     adapted or deleted, by the same WIP that changed keep_score()'s formula to drop w_pagerank.
**RULE (binding, forward):** An agent may never edit, disable, comment out, or relax a test to
match code it has just changed. A red test is a FINDING and must be surfaced to a human, not
resolved. This applies to Claude Code, Antigravity, and any future agent operating on this repo.

### A4. The +3.57pp Acc@QA is not attributable
Three changes shipped in the same unmeasured tree: _QA_MCQ_PROMPT, temp=0.1, and
eval/mc_scorer.py punctuation-stripping in parse_mc_answer. A +2-question delta (N=56) was
attributed solely to the prompt. The parser change is the more parsimonious cause. No causal
claim survives. Acc@QA is not comparable across the mc_scorer change.

### A5. Query-aware captions cannot lift IoP/IoU — architectural, not empirical
_ensure_captions() operates on retrieved_frames (post-retrieval); retrieval is CLIP-only and
caption-free. Captions feed the answerer's evidence only. The V2 calibration doc's claim that
"retrieval metrics will only improve once we re-ingest the videos with query-aware captions"
is REFUTED. The existing data already showed it: mIoP 0.3091 -> 0.3091, IoP@0.5 25.00% ->
25.00%, unchanged across the patch. **A re-ingest undertaken to lift IoP would be wasted
compute.** (Roadmap v8 Phase 4 pre-registered exactly this check; it now resolves.)

### A6. The iop() duplication count in §3 is stale
FOUR copies exist, not three: scripts/pillar2_grounded_qa.py, scripts/tune_l1_weights.py,
eval/grounding_scorer.py, and the iou() variant. The shared-constructor fix must absorb all of
them.

### A7. Optuna-tuned L1 weights are QUARANTINED and must not become defaults
The WIP baked study output into iris_config.py defaults: l1_w_query 0.20 -> 0.0082,
l1_w_persist 0.15 -> 0.3880, l1_w_action 0.30 -> 0.2722, l1_w_recency 0.05 -> 0.0182, plus new
l1_w_iframe 0.2497 / l1_w_size_anomaly 0.0637 replacing pagerank/entropy/hessian.
Disqualified three ways: (a) the study optimized the corrupted span objective (§3); (b) 50 TPE
trials x 56 questions at 1/56=1.79% granularity overfits 2-3 questions; (c) the Phase-1 L1
admission ablation already returned RED on diluting w_query at low retention budgets — driving
it to ~0.008 overrides a surfaced red with a tuner's output. Re-derive only after the span fix,
on a held-out or nested split.

### A8. WIP triage — phase-gated
The uncommitted tree is preserved at 3497ee4 (branch wip/v8-measured-state-snapshot, tag
v8-measured-state) and comprises four independent families. Disposition:
  - Query-aware captioning (aria/_clip/pipeline/query/phase6_build_dev_cache/pillar2):
    PARKED until Phase 4. It is the H1 remedy; landing it before the Phase-2 diagnostic
    destroys that diagnostic's control. No test exercises the query= path.
  - L1-codec admission features (charon_v/cached_frame/ingest/types/l1_elysium/pipeline/
    query/virat_retention_sweep): PARKED until Phase 5. Correct placement for codec (at
    admission, per the codec negative) but out of phase, untested, and it moves keep_score,
    which confounds every downstream number.
  - Optuna-tuned defaults: DISCARDED per A7.
  - Test flips: DISCARDED per A3.
  - eval/mc_scorer.py punctuation-stripping: LANDS on main, recorded as a measurement change.
  - configs/default_iris_config.json captioner_backend key: LANDS (inert, matches code default).
Also flagged, not fixed: scripts/phase6_build_dev_cache.py references `elapsed` in an except
branch where its assignment was removed (NameError on any exception);
scripts/tune_l1_codec_weights.py is a near-duplicate of tune_l1_weights.py generated by a regex
mutator (scratch/generate_tune.py); scripts/virat_retention_sweep.py's loop nesting was
structurally rewritten AND the sweep was already run in that form
(virat_retention_sweep_report.json) — that run is unregistered and does not satisfy the Phase-5
pre-registration; scratch/test_tune_l1_weights.py is pytest-collectable by name but is not a
pytest suite.

## 2026-07-18: Retrieval is the dominant grounding loss — two roadmap claims falsified

Measured on the N=64 in-sample grounded set (no held-out split; compute-limited), retrieval-only,
answerer-free scorer. Numbers are in-sample with video-level bootstrap CIs (1000 resamples).

### C1. The span bug was NOT the mIoP lever (roadmap v8 P1 "biggest lever" — FALSIFIED)
Fixing the min->max span (enclose-all-top-K) to span-from-PPR-peak changed mIoP by -0.0158
(minmax 0.2744 vs ppr_peak@w=2.2 0.2586; 95% CI [-0.0928, +0.0666], spans zero). Roadmap v8
predicted minmax ~0.31 -> ppr_peak high-30s. Neither number materialized; the direction is
flat/slightly-negative. The span bug was a real bug and the fix is kept (it is correct on the
questions where retrieval succeeds), but it is NOT a lever on the headline metric. P1 is
demoted from "biggest lever" to "correctness fix, no headline effect."

### C2. The answerer is NOT the sole Acc@GQA bottleneck (roadmap diagnosis — FALSIFIED)
The "entire Acc@GQA deficit is the answerer" claim was computed from IoP@0.5=25%, but that 25%
is itself suppressed by retrieval hard-misses: 39/64 questions (61%) have IoP in [0,0.1), 38 of
them exactly 0.0 (no predicted/gold overlap at all). Retrieval is upstream of and dominates the
answerer term. Verified NOT a mechanical artifact: units consistent (seconds throughout, no fps
confusion), t* in-bounds, zero scoring-bug cases, zero near-miss/boundary cases (nearest-gold
distance min 2.55s, median 8.37s — all beyond the 2.2s window). Retrieved clusters are
internally coherent but land in the wrong video neighborhood.

### C3. The mechanism: caption-free CLIP retrieval misses SHORT events
Only discriminator between hit (IoP>=0.6, n=17) and miss (IoP=0.0, n=38): gold-span length,
~3x longer in hits (mean 18.74s vs 5.82s). Duration, family (C/T), and multi-interval count are
indistinguishable. Static-image CLIP embeddings + PPR retrieve the wrong moment when the target
event is brief. This is a mechanistic retrieval-precision finding, deconfounded.

### Consequence for the roadmap (to be re-sequenced with Dr. Uma, not unilaterally)
- half_width=2.2 is FROZEN as span-construction plumbing (median gold half-span), NOT as a
  result. The flat sweep (0.2510-0.2596 across the grid, all CIs overlapping) confirms width has
  no lever; it acts only on the ~17 already-retrieved questions.
- The embedding swap (SigLIP2 / MobileCLIP, roadmap Phase 5) is promoted from optional-pre-
  finetune to the DIRECT TEST of C3: does temporal/better embedding close the short-event miss?
- The P2 answerer diagnostic is demoted: captions-vs-answerer cannot be cleanly measured while
  61% of evidence retrieval misses the event. P2 as written risks measuring noise downstream of
  the retrieval failure.
- All grounding numbers remain in-sample N=64 lower bounds until a larger set is built (526
  grounded questions available across the 87 already-cached videos, no new ingest required, when
  compute allows).

## 2026-07-19: A6 pinned run (mixed c4dd497 + all-minmax bcf940e; N=64 in-sample)

### A1. P(correct | grounded) = 65% — MEASURED, replaces quarantined placeholder
15/23 = 65.2%, video-clustered 95% CI [0.45, 0.84]. Replaces the quarantined V2 placeholder
(36%). The answerer is competent given grounding and is NOT the binding constraint at ~36%
grounding; P2 stays deferred. Acc@GQA measured 23.4% (proposed), vs ~13% projected.

### A2. Baseline-span fork RESOLVED: proposed-uniform grounding advantage does NOT survive all-minmax
mIoP diff +0.019 CI [-0.021, +0.063], IoP@0.5 diff +0.031 CI [-0.031, +0.097] (both span zero).
The mixed +0.087 mIoP edge decomposes into span construction +0.068 and frame selection +0.019
(n.s.). DO NOT claim IRIS retrieval localizes better than uniform. Localization credit belongs
to clip-in-PPR-top8 peak selection (span construction), NOT frame selection.

### A3. New span-independent retrieval win on the answer
proposed Acc@QA 53.1% vs uniform 21.9% (+0.31 CI [+0.16, +0.45]) vs random 29.7% (+0.23 CI
[+0.09, +0.38]). Acc@QA is span-independent -> clean, unconfounded "retrieval delivers better
answerer evidence" claim.

### A4. Grounding gap (faithfulness signal, feeds P6)
19/64 answered correctly while UNGROUNDED (P(correct|ungrounded) = 46%). Evidence for
language-prior shortcut answering; feeds the verification pillar (P6).

### A5. Parser and determinism clean
Parser 64/64 clean_leading (zero failures). Answerer determinism replicated (proposed Acc@QA
byte-identical across both runs).

### A6. Peak-source fix result now recorded (closes the prior gap)
Mixed proposed mIoP 0.3426 / IoP@0.5 35.9% reproduced at c4dd497; +8 IoP@0.5 vs the ppr_score
legacy peak (28.1%, half_width report). Artifacts: A6_mixed_raw.json, A6_allminmax_raw.json
(bcf940e); half_width_confirmation_report.md (83215c0).

## 2026-07-19 — P1 / Fix 2: ppr_lambda sweep {0.0, 0.5, 1.0}
- P1a. Pre-registered λ*=1.0 FALSIFIED. λ=0.5 (default) and λ=1.0 statistically indistinguishable on all metrics; only λ=0.0 (pure codec) significantly worse on grounding (mIoP 0.5−0.0 +0.121 [+0.050,+0.196]; 1.0−0.0 +0.108 [+0.023,+0.195]). λ is a CLOSED lever — default 0.5 optimal, Fix 2 yields no gain. Keep λ=0.5.
- P1b. Codec-at-query-time effect REFINED to a DOMINATION effect, not a per-signal negative: codec_conf at balanced weight ties pure semantic; only codec-domination (λ→0) collapses grounding (candidate-set eviction clip-peak can't recover). Vindicates the codec_conf-vs-"codec" scoping point.
- P1c. ROADMAP v9 §4 CORRECTIONS: "monotonic in λ" is falsified post-fix (plateau λ∈[0.5,1.0], collapse λ→0); the "79% peak hijack at λ=0.5" is a PRE-fix diagnostic resolved by clip_in_ppr_top8, not current behavior. Paper must claim "over-weighted query-blind codec_conf prior degrades peak selection," not "codec metadata is negative." (v9 P1's "λ*=0" was also backwards; see P1_lambda_prereg.md.)
