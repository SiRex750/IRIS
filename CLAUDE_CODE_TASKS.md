# Claude Code task queue — 2026-08-15

Three read-only verification tasks, in priority order. **All three are report-only.** None of
them computes a new headline number, and none of them is a new experiment. Each one resolves a
row in `CLAIMS.md` that is currently un-printable.

Run them one at a time. Paste the report back into the audit chat before running the next one —
the point of the two-layer split is that nothing here gets believed until it's independently
checked against the artifact it names.

---

## T1 — Reconcile the end-to-end / Amdahl claim (HIGHEST PRIORITY)

**Why:** `CLAIMS.md` scoping rule S1 says 1,104× must never be printed without the end-to-end
correction beside it. The correction — "~4.6× end-to-end at N=4,892 against a ~2.2 s LLM stage,
crossover N≈2,644" — has **no artifact anywhere on this machine**. Meanwhile
`eval_results/P_latency_result.md` states the opposite-direction figure: *"query embedding is
0.016s (8x all of retrieval) and the answerer is ~60s, so retrieval latency is ~0.003% of
end-to-end cost."* Those two statements cannot both be the headline framing. Until this is
resolved, the paper's most quotable number has no defensible caveat sentence.

```
TASK T1 — end-to-end / Amdahl artifact reconciliation. READ-ONLY. Report only.

Working trees: C:\Users\akash\Documents\Iris, Iris-scaling, Iris-ucfvad.

GOAL
Establish, from artifacts only, what IRIS's end-to-end (ingest + retrieval + answerer)
cost breakdown actually is, and in which regime each published framing applies.

STEP 1 — locate
Search all three trees, all committed refs (git log --all, git stash list, and the
wip/laptop-preserve-2026-08-04 branch), for any artifact containing:
  - the value 4.6 used as an end-to-end speedup factor
  - a crossover N near 2644
  - a "2.2" second LLM/answerer stage cost
  - the strings: amdahl, crossover, end_to_end, e2e_speedup
Report every hit with file path, commit sha, and the surrounding 5 lines.

STEP 2 — reconcile, do not average
Two figures are already on disk and appear to conflict:
  (a) eval_results/P_latency_result.md — "retrieval latency is ~0.003% of end-to-end
      cost", answerer ~60s, measured on NExT-GQA VAL, videos <=599 frames.
  (b) the 4.6x / 2.2s figure quoted in team reports, no artifact found so far.
For each, report: the N regime it was measured in, the answerer model and config, whether
ingest is inside or outside the boundary, and the exact file+line. Do NOT reconcile them
by computing a new number. Do NOT average them. If they are measured under different
answerer seats (granite4:micro vs an earlier seat), say so explicitly — that is the most
likely explanation and it is checkable via DECISIONS.md.

STEP 3 — state what a defensible e2e number would require
List the specific measurements needed and whether the inputs exist locally. Do not run them.
Note that amort_units.json exists at the root of Iris-scaling with per-frame decode/embed/
caption unit costs and a per-query retrieve cost — assess whether an e2e figure could be
composed from it, and name every assumption that composition would require.

GUARDS — stop and report instead of proceeding if:
  - no artifact for the 4.6x figure exists on any ref  -> report ABSENT, do not derive one
  - the two figures use different answerer seats       -> report both, do not pick one
  - any step would require running a model             -> stop, this task is read-only

OUTPUT
Write eval_results/T1_e2e_reconciliation.md. Include for every artifact cited: absolute
path, git commit sha, sha256 of the file. End with an explicit verdict line:
  T1 VERDICT: <RESOLVED | ABSENT | CONFLICTING>  and one sentence of justification.
```

---

## T2 — IoP/mIoP rescore feasibility (unblocks `CLAIMS.md` row C4.3)

**Why:** C4.3 (mIoP 0.3349 / IoP@0.5 0.3500) uses the **union** span convention, which is
biased upward against the leaderboard's official max-per-span. It cannot be printed as-is.
I have already confirmed that `eval_results/P_NOWA_test_raw.json` has 120 per-question records
with keys `[video, qid, top_k, half_width, iop, iou, peak_in_gold]` and **no predicted spans and
no gold spans** — so a pure rescore from that file is impossible. The open question is whether
spans are recoverable from a cached artifact or whether the scoring pass must be re-run, and if
re-run, whether it needs a re-ingest.

```
TASK T2 — IoP rescore feasibility. READ-ONLY. Report only. Do not re-run the eval.

GOAL
Determine the cheapest path from the union-convention IoP/mIoP numbers in
eval_results/P_NOWA_grounding_result.md to numbers on the official max-per-span convention.

ESTABLISHED ALREADY (do not re-derive):
  eval_results/P_NOWA_test_raw.json -> per_question records carry only
  [video, qid, top_k, half_width, iop, iou, peak_in_gold]. No spans of any kind.

STEP 1 — hunt for spans
Find any artifact on any ref that contains, for the P-NOW-A TEST split (27 videos / 120 Qs):
predicted temporal spans, retrieved frame indices/timestamps per question, or gold spans.
Check eval_results/, tuning/, benchmark_runs/, frame_dumps/, logs/, and any .npz index cache.
Report path + commit + record count + whether the 120 TEST qids are covered.

STEP 2 — identify the scorer
Locate the code that computes iop/iou/peak_in_gold and the exact line where the union
convention is applied. Quote it. Confirm whether a max-per-span variant already exists in
the codebase (search for: max_per_span, per_span, union, span_merge).

STEP 3 — cost the three paths, do not execute any
  (a) rescore from a cached spans artifact, if STEP 1 found one
  (b) re-run the scoring pass only, from cached indices/embeddings
  (c) full re-ingest + re-run
For each: is it possible on this machine today, what inputs are missing, rough wall time,
and does it touch the TEST split (which DECISIONS/CLAIMS records as BURNED — flag loudly
if any path would consume it further).

GUARDS
  - do NOT compute any new IoP number, not even as a demonstration
  - do NOT modify the scorer
  - if the only viable path re-consumes the burned TEST split, STOP and say so first

OUTPUT
eval_results/T2_iop_rescore_feasibility.md, with path + commit + sha256 for every artifact,
ending in:  T2 VERDICT: <RESCORE-ABLE | RESCORE-NEEDS-RERUN | NEEDS-REINGEST | BLOCKED>
```

---

## T3 — Exponent-CI data inventory (unblocks `CLAIMS.md` row C1.5)

**Why:** the lead contribution's exponent is an n=4 fit with no error bar. The fix is to bin the
350-clip corpus by survivor N. Before writing any harness, we need to know whether the inputs
exist on **this** machine or only on the remote box — because if it's the latter, this queues
behind the same push that C1.1 is waiting on.

```
TASK T3 — exponent-CI data inventory. READ-ONLY. Report only. Write no harness yet.

GOAL
Determine whether a binned latency-vs-N fit with confidence intervals can be produced from
data present on this machine.

STEP 1 — what latency points exist
Enumerate every (clip, N_survivors, arm, median_total_retrieval_s) point on disk. Start with
eval_results/scaling_curve_raw.json, scaling_curve_v2_raw.json, virat_latency_N4892_raw.json.
Report the full table and the total distinct N values per arm. State plainly how many points
the current published fit rests on.

STEP 2 — what index caches exist
Locate every index cache (.npz or equivalent) under eval/data/ and anywhere else. For each,
report: clip id, survivor count N, the config weights it was built under, and its mtime.
CLAIMS.md scoping rule S6 records that some caches were built under weights 0.5/0.3/0.2
rather than the frozen 0.8/0.1/0.1 — verify this per-cache and report which are affected.
This matters because latency claims survive the drift but retention/survivor-count claims
do not.

STEP 3 — what is missing
State exactly what would be needed to get, say, 20-30 latency points spanning N=100..15,000:
how many clips are locally available with usable caches, how many would need a fresh ingest,
and estimated wall time per ingest from the measured numbers in
Iris-ucfvad/tuning/ucfcrime_vad_exp1/efficiency_measurements.json.

GUARDS
  - do NOT run any ingest
  - do NOT fit anything
  - if fewer than 8 usable local points exist, say so directly and stop

OUTPUT
eval_results/T3_exponent_ci_inventory.md, path + commit + sha256 throughout, ending in:
  T3 VERDICT: <FEASIBLE-LOCALLY | NEEDS-INGEST | BLOCKED-ON-REMOTE>
```

---

## Human tasks — these are not Claude Code's and nothing above substitutes for them

### H1 — the message to Siddanth (send today)

> Two asks. First, can you push `15a7d79` — the v3 scaling curve with real CLIP text queries?
> I only have v2 on my laptop (scene_sparse k = 0.663, image-embedding queries), and the report
> says v2 measures a code path real queries don't take. The abstract's lead number depends on
> v3 and I can't open it. Second, push whatever branch holds the 140-video UCF per-frame CSVs
> (the 8-feature ones with ground truth) — I need them to re-run R0 at n=140, which is what
> makes the control section publishable for a journal. My newest fetched commit from anyone is
> `a5cea25` from 29 July, so anything after that I simply don't have.

### H2 — recover the Uma meeting details (from Siddanth or Swara, today)

Exact journal, exact scope she asked for, whether she named a deadline, and what the "next ask"
was. "Some journal" changes the work list materially — a journal cycle makes all four
not-print-safe rows blocking rather than nice-to-have.

### H3 — decide the contribution order with a fact you now have

C2 (block-diagonal) is the only contribution that is `PRINT-SAFE` outright today. C1 (scaling
law) has one row blocked, one superseded, one with no CIs. If Uma is expecting the scaling law
to lead, she should know that the identity result is currently the more defensible one.

---
---

# Queued 2026-08-17 — filed out of the T5 shot-bucketing run (`01c159d`)

## T5 is CLOSED as a negative — do not reopen it

The T5 verdict is **UNDERPOWERED** on the pre-registered primary and a **POWERED null** on the
pure-geometry secondary (`_shotbucket/run/summary.md`). The one-line takeaway to preserve
everywhere this is cited: **the powered result is the geometry null — codec shot geometry ties
uniform, properly — not the underpowered primary.** The `+1.667pp` dM3 point estimate is a
precision *lean* inside a CI of `[-0.941, +4.646]`; it must never be quoted as a positive.

Standing guards on this result:

- **Do not re-run T5 to chase significance.** No new shot-bucketing arms, no new seeds or
  budgets hunting for a positive — that is forking-paths. The corpus has exactly 19
  gold-annotated videos (the 150 Normal videos have no gold windows, so M2/M3 are undefined
  on them), so the primary is *structurally* underpowered at ~1pp. No amount of re-running
  fixes it; only annotating more anomaly videos would, and that is out of scope. State this
  if anyone proposes another T5 pass.
- **No `iris/` change is warranted from T5.** A null does not justify touching production.

## T4-followup / candidate T6 — Is R0's action-score "tie" a coverage artifact of hold-forward propagation?

**Why:** post-hoc from T5. `action_score_propagated` is hold-forward-propagated from the
retained tier — it is a step function, ~9.5 frames per plateau. Top-k by this score (T5 arm
`E_action_topk`, which *is* the R0 method) therefore selects the earliest frame of each plateau
and spends the rest of the budget on adjacent near-duplicates: **42 distinct temporal instants
out of k≈503 (8.4%), mean run length 13.19 vs uniform's 1.00.**

This matters because R0's headline — "codec motion-scoring admission ties uniform at matched
budget" — may be partly a **coverage artifact of propagation** (the score never resolves
distinct gold frames, so it cannot get credit for them), not evidence that motion scoring is
uninformative. If so, R0's negative is real but its *mechanism* is mislabeled.

This is a **new pre-registered study**, not a T5 continuation. Step 1 may moot the whole thread.

```
TASK — R0 propagation-coverage check. STEP 1 IS READ-ONLY. Report before building anything.

STEP 1 — verify before building (do this first, on its own)
Confirm what R0's admission actually ranks on: action_score_propagated, a raw/peak
(un-propagated) score, or is_retained_tier. Quote the exact file+line where the ranked
signal is chosen. Then measure that selection's distinct-instant coverage (maximal runs of
consecutive selected frames / budget k) over the same 19 videos.
  -> If R0 already selects on a NON-PLATEAUED signal, the thread is moot: CLOSE the ticket
     and report that. Do not proceed to Step 2.

STEP 2 — only if Step 1 shows the thread is live: pre-register separately as T6
Contrast action top-k vs action top-k with PLATEAU-DEDUP (one representative per plateau,
then fill from distinct plateaus / min-gap NMS), at matched budget, on M2/M3 plus a
coverage metric, paired bootstrap over the same 19 videos.
Pre-commit the read BEFORE running:
  coverage-corrected score lifts M2/M3 past the powered threshold
      -> R0's tie is partly a propagation artifact
  flat
      -> the score genuinely ties uniform
Reuse r0_gate_full.py's metrics/loader/bootstrap verbatim, as T5 did.

GUARDS
  - the same n=19 power ceiling applies (~1pp). A null is only informative if the CI
    half-widths clear 1pp — otherwise the honest verdict is UNDERPOWERED, not TIES.
  - write the pre-registration down before the run, not after
  - READ-ONLY on iris/*.py
```
