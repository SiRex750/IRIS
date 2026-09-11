# Commit provenance check: P-NOW-A Acc@GQA (`dc59de3` vs `b170007b`) — C22d

Read-only, paper not edited. Question: `P_NOWA_accgqa_result.md` is titled
"(dc59de3, second and final test-half touch)" and Appendix A cites commit
`dc59de3` for the Acc@GQA row, but `P_NOWA_accgqa_raw.json`'s own
machine-written provenance block records `"git_commit":
"b170007b329dcd203851527f708fd49db602a9f9"`, `"git_dirty": false`. Which
commit did the measurement actually run at?

All checks below were run with both remotes fetched (`git fetch --all`,
`origin` = `SiRex750/IRIS`, `swara` = `swarapotd-rgb/IRIS`) and `--all`
branch scope throughout.

## 1. Does `b170007b` exist?

Yes.

```
commit b170007b329dcd203851527f708fd49db602a9f9
author Antigravity AI <antigravity@gemini.google>
date   2026-07-23T21:09:59+05:30
subject feat(eval): TEST-only Acc@GQA harness (3-arm MC-answering + seam check)
```

`git branch -a --contains b170007b329dcd203851527f708fd49db602a9f9`:

```
* siddanth/peak-source-a6-p1
  remotes/origin/siddanth/peak-source-a6-p1
```

Present only on `siddanth/peak-source-a6-p1` (local + `origin`); not on
`swara`, not on any other branch.

## 2. Does `dc59de3` exist?

Yes.

```
commit dc59de38c85bade59778e9071e5689f15f26e91e
author Antigravity AI <antigravity@gemini.google>
date   2026-07-23T22:09:32+05:30
subject measure(P-NOW-A): held-out Acc@GQA, frozen config, second and final test-half touch
```

Same branch scope as `b170007b`: `siddanth/peak-source-a6-p1` (local +
`origin`) only.

## 3. Relationship between them

`dc59de3` is `b170007b`'s direct (sole) child — one commit apart, ~60
minutes later:

```
$ git merge-base --is-ancestor b170007b329dcd203851527f708fd49db602a9f9 dc59de38c85bade59778e9071e5689f15f26e91e && echo ancestor
ancestor

$ git log --oneline b170007b329dcd203851527f708fd49db602a9f9..dc59de38c85bade59778e9071e5689f15f26e91e
dc59de3 measure(P-NOW-A): held-out Acc@GQA, frozen config, second and final test-half touch

$ git show -s --format='%P' dc59de38c85bade59778e9071e5689f15f26e91e
b170007b329dcd203851527f708fd49db602a9f9
```

**`dc59de3` is exactly the commit that added the result file, not a commit
about running the harness.** Its full diffstat:

```
$ git show --stat dc59de38c85bade59778e9071e5689f15f26e91e
commit dc59de38c85bade59778e9071e5689f15f26e91e
Author: Antigravity AI <antigravity@gemini.google>
Date:   Thu Jul 23 22:09:32 2026 +0530

    measure(P-NOW-A): held-out Acc@GQA, frozen config, second and final test-half touch

 eval_results/P_NOWA_accgqa_raw.json | 4233 +++++++++++++++++++++++++++++++++++
 1 file changed, 4233 insertions(+)
```

One file, pure addition, no code touched. The timeline is: pre-registration
commit `30d557b` (21:06:45) → harness commit `b170007b` (21:09:59) → [harness
runs against a clean `b170007b` tree, recording `git_commit: b170007b`,
`git_dirty: false` into its own output] → that output is committed an hour
later as `dc59de3` (22:09:32). `git_dirty: false` is internally consistent
with this: at `b170007b` the only pending change was the not-yet-written
output file itself, which the harness doesn't track as tree state.

**So yes — `dc59de3` is plausibly, and in fact demonstrably, the commit that
recorded the result into git, not the commit the measurement executed
against.** The commit the measurement ran at is `b170007b`.

## 4. Introducing commit of each P-NOW-A file

```
--- eval_results/P_NOWA_accgqa_raw.json ---
dc59de38c85bade59778e9071e5689f15f26e91e 2026-07-23T22:09:32+05:30 measure(P-NOW-A): held-out Acc@GQA, frozen config, second and final test-half touch

--- eval_results/P_NOWA_accgqa_result.md ---
b446a7afe6b69ca9416728030940b82a6679f064 2026-07-24T09:26:43+05:30 docs(P-NOW-A): held-out Acc@GQA + answerer and uniform-baseline corrections

--- eval_results/P_NOWA_accgqa_prereg.md ---
30d557bdeb30e695f8105ded8a03bf7e5c6bddf2 2026-07-23T21:06:45+05:30 chore: pre-register held-out Acc@GQA run (second and final test-half touch)
```

Note the `.md` result file's own title ("dc59de3...") was itself written the
*next morning* in `b446a7a`, a full 11 hours after `dc59de3` — i.e. whoever
wrote the human-readable result doc looked up "which commit holds this
number" (the data-add commit) rather than "which commit did the harness run
at" (the field already sitting in the same JSON they were summarizing).

## 5. Do other Appendix A rows show the same pattern?

Every artifact in Appendix A that cites a commit, checked against that
artifact's own self-recorded git field (where one exists):

| Artifact | Cited in paper | Self-recorded field | Match? |
|---|---|---|---|
| `blockdiag_identity_gate_result.{json,md}` | commit `3d83b2c` | — no `git_commit`/`commit` field in the JSON at all | N/A — nothing to compare against. `3d83b2c` is confirmed (via `git log --diff-filter=A`) to be the commit that introduced both files, bundled into a feature commit ("Add block_diagonal graph_edge_mode..."), not a pure data-add. Can't independently verify what commit the gate script itself ran at. |
| `blockdiag_grounding_gate_result.{json,md}` | commit `3d83b2c` | — same, no field | N/A — same as above; same introducing commit. |
| `e2e_stage3_decomp_raw.{json,md}` | commit `deeeba8` (`tracked_dirty_count` 0) | `"commit": "deeeba80c464ef87a86948df34db9feb746b0945"` | **Matches exactly.** These files are untracked (never committed at all — `git log --diff-filter=A` returns nothing), so there is no separate "commit that recorded it" to conflate with "commit that ran it"; the cited hash is only ever the self-recorded one. |
| `scaling_curve_v3.{json,md}` | "git HEAD `9c66393d…` (dirty, 94 changed files); committed at `90ef59b`" | `"git_head": "9c66393d1624ebae5443f62ddbe8f5fb8dc12b7f"`, `"git_dirty": true`, `"git_dirty_file_count": 94` | **Matches, and this row already does correctly what the P-NOW-A row fails to do** — it names the run-time HEAD and the commit-that-recorded-it as two separate values, explicitly labeled. Verified: `9c66393d` is a real, different commit ("Run four-arm MLVU segmentation ablation..."); `90ef59b` is confirmed (via `git log --diff-filter=A`) to be the introducing commit of `scaling_curve_v3.json`. |
| `efficiency_measurements.json` (`origin/siddanth/ucf-vad-exp1`) | "added in commit `41d7205`" | no `git_commit` field in the JSON (only `generated_utc`) | **Already correctly phrased** — "added in commit" is honest about what `41d7205` is (confirmed via `git log --diff-filter=A` to be the introducing commit), and does not claim it is the commit the measurement ran at, since no such field exists in the artifact to check against. |
| `env_A2.json` | "captured 2026-09-08T18:43:07Z at commit `48c082a`" | `"git": {"head": "48c082a64465bc363510c4bcb31bbe90513ebcc7", "tracked_dirty_file_count": 2}` | **Matches.** `48c082a` is the file's own recorded HEAD, and the prose already says "captured at" rather than implying a code-authorship commit. |
| `P_NOWA_accgqa_result.md` / `P_NOWA_accgqa_raw.json` | commit `dc59de3` | `"git_commit": "b170007b329dcd203851527f708fd49db602a9f9"`, `"git_dirty": false` | **Mismatch.** This is the row under investigation, and the only row where the cited commit is neither the self-recorded run commit nor honestly labeled as "the commit that recorded this" — it's presented as if it were the run commit. |
| `MLVU_codec_baseline.{json,md}` | "commit not recorded — see A.5.3" | git HEAD field present but garbage/unusable (per A.5.3, already an open item) | Not this pattern — already flagged as its own, different problem (no usable commit at all, not a wrong one). |

**Conclusion: this is an isolated case, not a systematic convention.** The
paper's own `scaling_curve_v3` row demonstrates the authors already know how
to report both values separately and label them correctly when the
distinction matters (run-time HEAD vs. commit-that-recorded-it). The
`efficiency_measurements.json` and `env_A2.json` rows are self-consistent.
The two blockdiag-gate rows and `e2e_stage3_decomp_raw` have no opportunity
to exhibit this specific error (no separate recorded-vs-running commit to
conflate). Only the P-NOW-A row takes the commit that added the output file
to git and presents it, unlabeled, as if it were the commit the harness ran
at — while the correct value sits in the very JSON the row cites, one field
away.

## What Appendix A should carry for this row

**`b170007b`** (or `b170007`, matching the paper's existing 7-char
convention), not `dc59de3` — because `b170007b329dcd203851527f708fd49db602a9f9`
is the value `P_NOWA_accgqa_raw.json` itself records as the commit the
measurement ran against (`git_dirty: false`, so no working-tree drift to
further qualify). If the authors want to keep `dc59de3` visible as the commit
that recorded the result into git (the same style `scaling_curve_v3` uses
for `90ef59b`), the row should read: "git commit `b170007b` (clean tree);
committed at `dc59de3`" — mirroring the phrasing already used one row up in
the same appendix, rather than inventing a new convention.

This does not change the reported number (Acc@GQA 0.1667 [0.088, 0.243]) or
its uncertainty — the raw JSON's own field already says which commit it ran
at; only the citation in the prose and the `.md` file's title are wrong.
