# Pre-registration dates — Appendix C items 30–33

Read-only audit. Covers every standalone pre-registration document in
`eval_results/` (ten files) plus B.6 (R0 codec-admission gate), which the
paper lists as standalone but which has no file matching that description in
`eval_results/`. B.3/B.3a/B.4 (NExT-GQA split, already dated 2026-07-22 in
the paper and corroborated by `P_NOWA_split_declaration.md`) and the in-spec
entries B.7–B.11 are out of scope for this note — B.1 does not claim
standalone documents exist for them, so there is nothing here to recover.

**Method.** For each entry: (1) the in-document date, verbatim; (2) the git
commit(s) that touched the file, with author date, commit date, and branch;
(3) the run's own timestamp, taken from `timestamp_utc` in that run's raw
JSON where one exists; (4) whether (2) precedes (3), and the gap. Filesystem
mtimes were not used anywhere below — every file in this working tree carries
the same bulk-checkout timestamp and dates the clone, not any run.

All ten prereg files and B.6's supporting documents were committed by
`Siddanth Anil` (author date == committer date in every case — no rebase or
cherry-pick divergence to flag) except the R0 claim-ledger commit, authored by
`Sonu Akash`. All ten prereg files exist **only** on
`siddanth/peak-source-a6-p1` (and its `origin` mirror); B.6's documents exist
only on `origin/sonu/audit-docs`.

---

## Registrations that can be shown to precede their run

For these four, an independent run-artifact timestamp (`timestamp_utc` in the
raw JSON, not a commit date) exists, and the pre-registration's git commit
precedes it.

### P1 / λ sweep (`P1_lambda_prereg.md`)

- In-document date: **2026-07-19** ("P1 / Fix 2 — ppr_lambda sweep
  pre-registration (2026-07-19)").
- Commit: `4fc5148`, author/committer 2026-07-20T04:53:54+05:30 (=
  2026-07-19T23:23:54 UTC — same calendar day as the in-document date in UTC).
  Branch: `siddanth/peak-source-a6-p1`.
- Run timestamps: `P1_lambda00_raw.json` 2026-07-20T00:09:46 UTC;
  `P1_lambda10_raw.json` 2026-07-20T07:40:25 UTC.
- Precedes: **yes**, both runs. Gap to λ=0.0 run: **46 min**. Gap to λ=1.0
  run: **8h 16m**.

### P_FUNNEL diagnostic (`P_funnel_prereg.md`)

- In-document date: **none**.
- Commit: `4591c72`, 2026-07-24T21:04:44+05:30 (= 2026-07-24T15:34:44 UTC).
  Branch: `siddanth/peak-source-a6-p1`.
- Run timestamp: `P_funnel_raw.json` → 2026-07-24T19:13:41 UTC.
- Precedes: **yes**. Gap: **3h 39m**.
- This is the B.5 entry (item 30/32, below).

### Held-out Acc@GQA (`P_NOWA_accgqa_prereg.md`)

- In-document date: **2026-07-22** ("Held-out Acc@GQA — pre-registration
  (2026-07-22)"). Note this is a day *before* the commit that introduced the
  file — see flag below.
- Commit: `30d557b`, 2026-07-23T21:06:45+05:30 (= 2026-07-23T15:36:45 UTC).
  Branch: `siddanth/peak-source-a6-p1`.
- Run timestamp: `P_NOWA_accgqa_raw.json` → 2026-07-23T16:36:01 UTC.
- Precedes: **yes**. Gap: **59 min**.
- **Flag**: the in-document date (07-22) is a full calendar day earlier than
  the commit that introduces the file (07-23). This does not threaten the
  ordering conclusion — even using the later, verifiable commit timestamp,
  registration still precedes the run by close to an hour — but it means the
  in-document date is not itself trustworthy as a standalone fact; only the
  commit timestamp is. Plausibly explained by B.3/B.3a/B.4 in the same
  cluster of experiments all being dated 2026-07-22 in the paper's table,
  suggesting the date was fixed at drafting time and the file committed the
  next day.

### P_SELCEILING (`P_selceiling_prereg.md`)

- In-document date: **none**.
- Two commits, not one:
  - `f3308aa`, 2026-07-25T01:40:47+05:30 — original registration (single
    undecided text-similarity signal, gate unspecified between lexical and
    CLIP-text).
  - `a5d1cdd`, 2026-07-25T14:34:41+05:30 (= 2026-07-25T09:04:41 UTC) — a
    same-day rewrite that fixes the actual protocol used: two arms (lexical
    = gate, CLIP-text = context-only), added specifically to close the
    shared-encoder confound the original version had only flagged as an open
    limitation.
- Run timestamp: `P_selceiling_raw.json` → 2026-07-25T09:29:19 UTC.
- Precedes: **yes**, but only against the *second* (rewritten) commit. Gap:
  **25 min**.
- **Flag**: the criterion actually exercised by the run (lexical-only gate)
  did not exist in the original registration; it was written in 25 minutes
  before the run that used it. The original registration's looser design
  ("specify exactly which \[signal\]") would not have been falsifiable as a
  gate. This is a legitimate pre-registration of the run that was actually
  performed, but the effective registration window is thin, and "the
  pre-registration predates the run" is true only because the document was
  edited same-day, close to run time, to match the run.

---

## Registrations with no run-artifact timestamp — ordering rests on git commit dates only, or on nothing at all

### P_latency (`P_latency_prereg.md`)

- In-document date: **none**.
- Prereg commit: `1c82c0b`, 2026-07-24T15:14:04+05:30 (= 09:44:04 UTC).
- Result commit: `c5141ce`, 2026-07-24T16:33:16+05:30 (= 11:03:16 UTC), 79 min
  later. Neither `P_latency_result.md` nor the later-promoted
  `virat_latency_N4892_raw.json` (commit `ccf764f`, 2026-07-27) carries a
  `timestamp_utc` or equivalent field.
- Ordering: **consistent but unverified** — the only evidence the run
  happened after registration is that the commit recording results was made
  79 minutes after the commit recording the pre-registration. There is no
  artifact-level timestamp independent of the commit process itself.

### P_scene_2x2x2 (`P_scene_2x2x2_prereg.md`)

- In-document date: **none**.
- Prereg commit: `2cf0423`, 2026-07-23T11:56:30+05:30 (= 06:26:30 UTC).
- Findings commit (`P_scene_2x2x2_result.md`): `9a2ac06`,
  2026-07-23T12:22:40+05:30 (= 06:52:40 UTC) — **26 minutes** after the
  prereg commit.
- Harness commit ("commit scene-sparse 2x2x2 sweep harness"): `30df4c4`,
  2026-07-23T21:03:00+05:30 (= 15:33:00 UTC) — **9 hours after the findings
  were already committed**.
- Ordering: **unknown / anomalous**. No run has an independent timestamp.
  Worse, the commit that adds the sweep *harness script* to the repository
  postdates the commit that records the sweep's *findings* by 9 hours — the
  code used to produce the result was not yet in version control when the
  result was written up. This does not by itself mean the registered
  protocol wasn't followed (the harness could have existed uncommitted,
  locally, before either commit), but git history alone cannot establish
  that the registered config is what ran, nor when the run itself occurred.
  This is the clearest case in the ledger of "ordering rests only on git
  commit dates" being too weak a foundation to call something registered.

### P_scene_sparse — standalone A/B (`P_scene_sparse_prereg.md`)

- In-document date: originally **2026-07-19** in the first committed
  version, changed to **2026-07-22** in a substantial rewrite.
- Two commits:
  - `abebbe6`, 2026-07-21T00:31:46+05:30 — first version, text says
    "(2026-07-19)" (already 2 days stale relative to its own commit).
  - `efcc183`, 2026-07-22T15:56:24+05:30 — rewrite. Changes N (from "largest
    reachable" to a fixed N=64), changes the arm/harness reference (pins to
    commit `76660ca`), adds a new "KNOWN DEGENERACY" section describing a
    confound in `crossscene_mode`/`pctile` discovered after the fact, and
    updates the in-document date to match this commit's day.
- Run artifact: **none found**. No `P_scene_sparse_result.md`, no raw JSON,
  anywhere in git history on any branch. The only result touching
  "scene_sparse vs flat" that exists is `P_scene_2x2x2_result.md`, a
  *different*, later-registered experiment (`P_scene_2x2x2_prereg.md`,
  2026-07-23) which explicitly frames itself as a fresh 2×2×2 mechanism test,
  not as the reporting artifact for this prereg.
- Ordering: **cannot be established — there is nothing to order against.**
  Either this run was never executed as specified, or its output was folded
  into P_scene_2x2x2 without being labelled as such. Either way, this
  document registers a run this ledger cannot show ever happened.

### P_dtedge, P_width_topk, P_NOWA_sweep

- In-document dates: **none**, for all three.
- Commits: `98a5f18` (2026-07-25T15:26:14+05:30), `48732bb`
  (2026-07-22T12:59:40+05:30) and `79d0c30` (2026-07-23T11:15:13+05:30)
  respectively. All on `siddanth/peak-source-a6-p1` only.
- Run artifacts: **none found**, on any branch, in any commit, or in the
  current working tree. Searched by filename glob and by commit history;
  nothing matches `*dtedge*`, `*width_topk*` (beyond the prereg itself), or
  `*NOWA_sweep*` outside the three prereg commits above.
- Ordering: **N/A — no run exists to order against.** These are
  three of the ten entries in the ledger's own prereg-document count that,
  as far as this repository's history shows, were registered and never
  executed (or executed and never committed anywhere).

---

## B.6 — R0 codec-admission gate (item 31/33)

B.2 lists B.6 as **standalone** — "a dated pre-registration document written
before the run." No such document exists.

- Searched all branches for any file introducing an R0/codec-admission kill
  criterion before the run: nothing. `git log --all -S "budget-matched
  control"` returns exactly one commit in the entire history.
- That one commit, `a3d3e95` (`sonu/audit-docs`, author Sonu Akash,
  2026-08-16T01:44:43+05:30), simultaneously adds:
  - `CLAIMS.md` — the only place the phrase "kill criterion pre-registered
    and TRIGGERED" appears anywhere in the repository;
  - `r0_full/summary.md` and `r0_full/results.csv` — the run's own output;
  - `_r0check/R0_READINESS.md` — a readiness check whose *prose* is dated
    "2026-08-04";
  - the audit documents that assert the ledger status.
- There is no `timestamp_utc` (or equivalent) in `r0_full/summary.md` or
  `results.csv`. The only date-shaped value anywhere in the run's own
  artifacts is `BOOTSTRAP_SEED = 20260805`, a hardcoded integer literal in
  `scripts/r0_gate_full.py` used to seed `numpy.random.default_rng` for the
  percentile bootstrap — it is not a recorded execution time, just a
  seed chosen (evidently) to look like one.
- **This is not a registration that can be shown to precede its run — it is
  a claim of registration that cannot be independently checked at all.**
  Every piece of evidence for "the kill criterion was fixed before the run"
  — the criterion's wording, the run's results, and the readiness check that
  purportedly preceded the run by 12 days — landed in version control in the
  same commit, authored 12 days after the readiness check's own claimed date
  and with no artifact placing the actual gate run (as opposed to the
  readiness check) at any specific time. At best this is simultaneous
  registration-and-reporting; nothing in the repository rules out that the
  criterion was written to match a result already in hand.
- This should not be reported as "standalone" alongside B.3–B.5 in B.2. It
  belongs in its own category — see recommendation below.

---

## Direct answers

**For how many registrations can the prereg be shown to precede the run?**
**Four**: P1_lambda, P_funnel (B.5), P_NOWA_accgqa, P_selceiling. In every
case this rests on comparing a git commit timestamp against an independent
`timestamp_utc` field in the run's own raw JSON — not on the in-document
date, which exists for only one of these four (P1_lambda) and is
self-consistent, and is present-but-wrong-by-a-day for a second
(P_NOWA_accgqa).

**For how many is the ordering unknown or reversed?**
**Seven.** Unknown because no run-artifact timestamp exists to check
against, and (for four of them) no run artifact of any kind exists:
P_latency and P_scene_2x2x2 (git-commit-only ordering, no run timestamp);
P_scene_sparse, P_dtedge, P_width_topk, P_NOWA_sweep (no run artifact at
all — registered, apparently never executed or never committed). One case,
B.6, is worse than "unknown": the sole document asserting pre-registration
was committed in the same commit as the run's results, 12 days after the
run's own claimed (but uncorroborated) date — this is a registration that
cannot be shown to precede its run and should be disclosed as such rather
than carried as "standalone."

**Does B.1's two-tier distinction still capture the real variation?**
No. "Standalone" currently means "a dated pre-registration document exists,"
but existence and dating are not the same as verified precedence, and this
audit turned up three different failure modes that the two-tier system
collapses into "standalone":

1. A standalone document that precedes its run, verified against an
   independent run timestamp (P1_lambda, P_funnel, P_NOWA_accgqa,
   P_selceiling — noting P_NOWA_accgqa's in-document date and
   P_selceiling's effective 25-minute margin are themselves weaker
   sub-cases worth a footnote).
2. A standalone document whose ordering rests **only on git commit dates**,
   with no independent run timestamp to check them against (P_latency,
   P_scene_2x2x2) — and for P_scene_2x2x2 the commit order is itself
   internally inconsistent (harness postdates its own findings write-up).
3. A standalone document with **no discoverable run at all** (P_scene_sparse,
   P_dtedge, P_width_topk, P_NOWA_sweep) — registered, but nothing in the
   repository shows it was ever executed.
4. A document asserted to be standalone whose criterion-bearing text was
   **committed alongside the run's own results**, not before them (B.6) —
   this is a fourth, distinct failure mode, worse than "unknown": it is a
   registration claim with no artifact capable of falsifying or
   corroborating it as prior.

A third tier is needed at minimum (git-commit-only ordering, case 2), and
arguably a fourth (no run found, case 3) and a fifth (registration and
result co-committed, case 4) if the appendix wants to be precise rather than
just less wrong.

## Recommendation for B.1 / B.5 / B.6

- **B.1**: Replace the two-tier "standalone / in-spec" framing with an
  explicit statement that "standalone" as currently used only means a dated
  file exists, not that its precedence has been checked against an
  independent run timestamp. State the four categories above (or collapse
  2+3 into "unverified" and keep 4 separate, if space is tight), and say
  plainly that of the ten standalone documents in `eval_results/`, only four
  have been checked against an independent run-artifact timestamp and shown
  to precede their run.
- **B.5**: Replace `<!-- open item 30 -->` with **no date** (none exists in
  the document) and cite the commit instead: "registered `4591c72`,
  2026-07-24T15:34:44 UTC; run recorded `timestamp_utc`
  2026-07-24T19:13:41 UTC — registration precedes run by 3h39m." For item
  32's outcome: at top_k=8, selection headroom = 0.3363, 95% CI
  [0.2778, 0.3939] — CI lower ≥ 0.15, so rule A resolves to **material
  headroom; in-pool caption reranking is the highest-value cheap
  experiment** (this is exactly what B.5's prose already says, so B.5's text
  is consistent with the artifact — only the date placeholder needs
  filling). Rule B: index_coverage 0.9925 ≥ 0.95, so **do not cite this run
  as support for densification** — worth stating explicitly since the
  appendix currently omits it. Rule C: best_gold_rank at top_k=8 is
  concentrated at ranks 1–4 (131/35/27/23 of 267), consistent with "a
  different query-conditional signal plausibly recovers those questions."
- **B.6**: Do **not** present this as standalone with a recoverable date.
  State plainly that no pre-registration document independent of the run's
  own results has been found; the kill criterion first appears in the same
  commit (`a3d3e95`, 2026-08-16) that adds the run's output, 12 days after
  the readiness check's own claimed (but likewise only-committed-on-08-16)
  date of 2026-08-04. The honest framing is: the criterion is *reported* as
  having been fixed in advance, and the outcome (uniform admission
  saturating M1 at every budget) is real and reproducible from
  `r0_full/results.csv`, but the "registered before the run" claim itself is
  not independently verifiable and should be disclosed as such rather than
  filled in with a git date dressed up as a registration date. An undated,
  disclosed registration is worth more to a reader than a git date presented
  as though it were one.
