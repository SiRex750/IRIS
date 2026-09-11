# Missing pre-registration artifacts — Appendix C item 36

Read-only. No paper edits. Searched: all local branches on both remotes
(`origin`, `swara`, 27 branches total) plus every dangling/unreachable object
(`git fsck --unreachable --lost-found`) and the full reflog (149 entries); the
`ucf-vad-exp1` worktree; four other local clones found alongside this repo
(`Desktop/IRIS-feat-prerun-fixes`, `IRIS2`, `IRIS_SEQUEL`,
`IRIS-prerewrite-backup.git`) — none is a git repo except `IRIS2`/`IRIS_SEQUEL`,
and neither contains any of the four filenames or terms below; every tracked
file across every commit on every branch, by content, for each prereg's
distinctive parameters/predictions, not just by filename; every currently
untracked file under `eval_results/` and `tuning/` in the working tree, by
content.

This partially overlaps and partially corrects an earlier audit,
`eval_results/prereg_dates_C30_33.md` (items 30–33), which already searched
three of these four by filename and concluded "no run artifact." That
conclusion turns out to be right for two of the three and **wrong for the
fourth** — see P_NOWA_sweep below, found this pass by content rather than
filename.

## P_scene_sparse

**Nothing found.** No `P_scene_sparse_result.md`, no raw JSON, no log, no
checkpoint, on any branch, in any dangling commit, or in the working tree.

The prereg (`eval_results/P_scene_sparse_prereg.md`) is distinctive enough to
search by content, not just name: it specifies N=64 (`dev_100` grounded ∩
both caches), a 36/28 clinical/temporal (C/T) split, arms
`flat | scene_sparse/rep | /t75 | /t90`, and a `crossscene_mode`/`pctile`
degeneracy specific to its own design. None of `dev_100`, `crossscene_mode`,
`rep_only` (as a crossscene mode, distinct from its appearance as a
`config_snapshot` field elsewhere), or the C:36/T:28 split appears in any
result file, log, or commit message anywhere in this repository's history.

**Partial evidence, but not of this run**: the prereg's commit
(`efcc183`, 2026-07-22) added a "per-question IoP dump" feature to
`scripts/eval_grounding_arms.py` — instrumentation, not a result. The next
day (2026-07-23, `2cf0423`) a *separate*, explicitly distinct prereg,
`P_scene_2x2x2_prereg.md`, was registered — it states its own purpose in its
first line as explaining "the teammate's reported scene_sparse > flat
result," runs a 2×2×2 grid (`graph_mode × motion_similarity_mode × span`) on
406 VAL questions, and explicitly frames itself as "a fresh 2×2×2 mechanism
test, not... the reporting artifact for" the standalone A/B. `DECISIONS.md`'s
2026-07-22 entry ("scene-sparse: no effect...") documents *that* experiment's
finding, not P_scene_sparse's. The two share a hypothesis (does scene-sparse
retrieval help?) but not a design, split, or config — P_scene_2x2x2 cannot be
read as P_scene_sparse's result under a different name.

**Verdict: the run never happened, or was abandoned same-week in favor of the
differently-designed P_scene_2x2x2.** Not "artifact lost" — there is no trace
of an artifact ever existing (no orphaned raw JSON, no log referencing
`dev_100` or the C:36/T:28 split), and not "elsewhere" — the one candidate
(`P_scene_2x2x2`) is independently pre-registered as a different experiment.
This is the strongest of the four for "never ran," short of a definitive
negative (git history cannot prove a negative; it can only fail to show a
positive after this thorough a search).

## P_dtedge

**Nothing found — the cleanest case of the four.** The prereg
(`eval_results/P_dtedge_prereg.md`, commit `98a5f18`, 2026-07-25) is the
*only* commit in this repository's entire history, on any branch, that
touches that file or mentions `dtedge`/`P_DTEDGE`/"directed temporal edges"
in a commit message. No later commit adds a result file, wires the
`DIRECTED` arm, or reports a fire rate. `DECISIONS.md` never mentions
"directed" or "dtedge." The prereg's own §"Wiring constraints (for the
implementation step, NOT this one)" frames the directional-edge wiring as
future work, separate from the registration — consistent with registration
having been the only step ever taken.

**Verdict: the run never happened.** No partial evidence of any kind (no
log, no checkpoint, no wiring commit) — this is not "no evidence found" in
the weak sense of "we didn't look hard enough"; a full-history content and
filename search across every branch and every dangling object turned up
exactly one commit total for this prereg's entire existence.

## P_width_topk

**No dedicated result found**, but with a real, identified precursor and
successor, which is why this is reported as weaker evidence than P_dtedge's
absolute zero.

- **Precursor** (2026-07-19, `83215c00`, `eval_results/half_width_confirmation_report.md`,
  explicitly labeled `UNRATIFIED`/"untracked, not committed" in its own
  header despite being a real commit): a width-only sweep at the same N=64
  (`dev_100` ∩ index_cache ∩ `gsub_val.json` grounded), no `top_k` axis, no
  `mIoU` floor rule. This predates P_width_topk by 3 days and shares its
  data source but not its design (P_width_topk adds the `top_k` grid and the
  "narrowest half_width with mIoU ≥ 0.179" selection rule this file lacks).
- P_width_topk_prereg.md itself was committed 2026-07-22 (`48732bb`,
  "peak-in-gold instrumentation + width/top_k pre-registration"). Its cited
  "current measured value" of mIoU=0.179 traces to `A6_analysis.md` (a
  different, already-run experiment), not to a run of this prereg's own
  grid — confirming the 0.179 floor was borrowed as an anchor, not produced
  by this sweep.
- **Successor**: one day later (2026-07-23, `79d0c30`), a differently-scoped
  but design-identical sweep, `P_NOWA_sweep_prereg.md`, was registered and
  *did* run (see below) — same `top_k ∈ {8,12,16,24}` grid, nearly the same
  `half_width` grid, same reference cell `(8, 2.2)`, same selection rule
  shape — but at N=406 (VAL) with a held-out TEST, not N=64 in-sample.
  `scripts/pnowa_width_topk_sweep.py` (the only sweep script matching this
  design in the repo) is hard-coded to the 406-question VAL/TEST split
  (`eval_results/val_videos.txt`), not to the N=64 grounding-only set
  P_width_topk specifies.

No file, log, or raw JSON anywhere in history reports a `top_k × half_width`
grid at N=64 with an `mIoU ≥ 0.179` selection rule applied.

**Verdict: the run never happened as its own registered design** — most
likely superseded within 24 hours by the larger, split-declared P_NOWA_sweep
before it was ever executed at its original N=64 scope. This is not
"indeterminate": there is a specific, dated, named successor experiment that
plausibly absorbed this one's purpose, and no artifact anywhere matches
P_width_topk's own N=64/mIoU-floor design. Flagged slightly softer than
P_dtedge because a same-family run (P_NOWA_sweep) genuinely happened days
later and could be mistaken for this one at a skim — worth stating explicitly
in the appendix so a reader doesn't conflate the two.

## P_NOWA_sweep — FOUND (corrects the prior C30–33 audit)

**The run happened, and its artifact exists under a different filename:
`eval_results/P_NOWA_grounding_result.md`.** `prereg_dates_C30_33.md`
searched for this by filename glob (`*NOWA_sweep*`) and concluded no run
artifact exists; that search missed the result because it isn't named to
match. Content match, not name match:

- `P_NOWA_sweep_prereg.md` (commit `79d0c30`, 2026-07-23T11:15, following the
  split declaration `50dc236` at 2026-07-23T11:08) registers a VAL-only
  (406Q/59 videos) `top_k × half_width` sweep, reference cell `(8, 2.2)`,
  selection rule "maximize mIoP subject to mIoU ≥ the reference cell's
  mIoU," and a frozen-config TEST run afterward.
- `P_NOWA_grounding_result.md` (commit `5e418dc`, 2026-07-24T09:26 — ~22
  hours after the prereg commit, consistent ordering) reports exactly that
  protocol: "VAL SWEEP (top_k x half_width): top_k is FLAT — peak_in_gold
  0.3227/0.3276/0.3251/0.3251 across top_k 8/12/16/24," the width
  trade-off at top_k=8, the pre-registered selection landing on
  `(top_k=12, half_width=2.2)` and being reported as noise rather than a
  real win (exactly per the prereg's own stated honesty norm), and a
  held-out TEST run (commit `300d857`) with `peak_in_gold 0.3667
  [0.2871,0.4454]`.
- `scripts/pnowa_width_topk_sweep.py` (added in the same commit as the
  prereg) is the actual grid-sweep harness: `TOP_KS = [8, 12, 16, 24]`,
  `HALF_WIDTHS = [0.75, 1.0, 1.5, 2.2, 3.0]`, VAL-only, asserts TEST videos
  absent — matching the prereg's design exactly.
- `DECISIONS.md`'s "2026-07-22 — P-NOW-A held-out grounding" entry restates
  the same numbers independently, and `P_NOWA_split_declaration.md`
  (committed same day as the sweep prereg) matches the split the sweep and
  result both assume.

**No separate raw per-cell JSON was found** (unlike, e.g., `P_funnel_raw.json`)
— the full grid's numbers live only in `P_NOWA_grounding_result.md`'s prose
and the two flat/width summary lines, not in a machine-readable artifact.
That is a real, disclosable gap (re-deriving CIs cell-by-cell from this
document alone isn't possible), but it is a different problem from "the run
never happened," and item 36 should not describe it that way.

**Verdict: the run happened; its artifact is present, just filed under
`P_NOWA_grounding_result.md` rather than a name matching the prereg.**

## Recommendation for Appendix B

Do not collapse these four into one sentence — the evidence differs
materially:

1. **P_NOWA_sweep**: change from "no artifact found" to "artifact found at
   `eval_results/P_NOWA_grounding_result.md` (commit `5e418dc`), corroborated
   by `DECISIONS.md` and the sweep harness `scripts/pnowa_width_topk_sweep.py`;
   note that no per-cell raw JSON survives, only the summary report, so exact
   CIs for cells other than top_k=8 and the frozen test point cannot be
   independently re-derived from this repository."
2. **P_dtedge**: state plainly that the pre-registration was never followed
   by any wiring, run, or result commit — a single commit is the entirety of
   this experiment's footprint in the repository. Do not describe this as
   "artifact lost"; there is no evidence an artifact ever existed.
3. **P_width_topk**: state that no result matching this design's specific
   scope (N=64 in-sample, mIoU≥0.179 floor) was found, that a same-family
   sweep at a larger, held-out scale (P_NOWA_sweep) was registered one day
   later and did run, and that the most likely reading is supersession before
   execution rather than loss — but flag this explicitly as inference, not a
   located artifact.
4. **P_scene_sparse**: state that no result matching its specific design
   (dev_100, N=64, C:36/T:28, rep/t75/t90 crossscene arms) was found; that
   the harness was instrumented for it three days before a differently-scoped
   experiment (`P_scene_2x2x2`) tested a related hypothesis at VAL scale and
   is the one entry in `DECISIONS.md` addressing scene-sparse's effect; and
   that P_scene_2x2x2 should not be cited as this prereg's result, since it
   pre-registers itself as a distinct mechanism test.

General framing for B: three of four pre-registered experiments in this
cluster (P_scene_sparse, P_dtedge, P_width_topk) show no evidence of
execution anywhere this search can reach — narrowly stated as "not found,"
not "did not happen," since absence of an artifact is not proof of absence
of a run, only the strongest available evidence in its favor. The fourth
(P_NOWA_sweep) did run and was mis-flagged as missing by the prior audit
because that audit searched by filename only; Appendix B should cite
`P_NOWA_grounding_result.md` by name going forward so this doesn't recur.
