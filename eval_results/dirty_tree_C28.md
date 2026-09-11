# Appendix C item 28 — dirty working trees behind §5.1 and §5.3 (verification)

Question: A.5.5 claims the 94 (§5.1 scaling) and 87 (§5.3 end-to-end) dirty
files were "scratch and evaluation outputs rather than pipeline code." Item
23 proved the identical assumption false for `build_cost_final` — the dirty
file there was `iris/ingest.py`'s `defer_recompute` change, sitting on the
timed path. Does A.5.5's claim survive for these two runs?

**Bottom line up front:**

- **§5.1 (`scaling_curve_v3`, 94 files): unverifiable as stated, but
  well-supported by circumstantial evidence.** The artifact captured a
  live-computed *count* only — the file list itself was never retained. A
  same-day, same-commit sibling run (`MLVU_ablation_long_trimmed`, ~2h15m
  earlier) did capture its full list: 79 files, all untracked, zero
  `iris/*.py` entries. Combined with a byte-identical `iris/` tree between
  the run-time commit and the commit that later recorded the artifact, this
  makes A.5.5's characterization *probably true* but **not directly
  checkable** — A.5.5 currently states it as a checked fact, which it is not.
- **§5.3 (`e2e_speedup`, 87 files): unverifiable, and worse than
  uncaptured.** The "87" is a **hardcoded literal** in the report script,
  not a git-status count computed anywhere in the harness or measurement
  code. No raw `git status` output, log, or cross-checkable sibling artifact
  exists for this run. There is no basis — positive or negative — for the
  "scratch and evaluation outputs" claim; it is asserted, not measured.
- **defer_recompute (item 23) is not in play for either run.** Both ran in
  August 2026; `defer_recompute` was introduced in a single commit
  (`3a2c1db`, 2026-09-10, this session) and appears nowhere else in
  `iris/ingest.py`'s history. Neither run's window overlaps it.

---

## 1. What actually exists per run

### §5.1 — `scaling_curve_v3.{json,md}`

- Run-time HEAD: `9c66393d1624ebae5443f62ddbe8f5fb8dc12b7f` (commit "Run
  four-arm MLVU segmentation ablation...", 2026-08-16 23:21:56).
- Artifact committed later, at `90ef59bbcccc9f644509f50d259580ca40f46e2d`
  (2026-08-18 17:10:18) — a separate commit, two days after the run-time
  HEAD (already noted correctly in `commit_provenance_C22d.md` as two
  distinct values).
- `provenance.git_dirty_file_count = 94`, computed in
  `scripts/scaling_curve_v3_report.py::provenance_section()`:
  ```python
  status = subprocess.run(["git", "status", "--porcelain"], ...).stdout
  ...
  "git_dirty_file_count": len(status.strip().splitlines()) if dirty else 0,
  ```
  This is a real, live `git status --porcelain` call — **note it counts
  every porcelain line, tracked and untracked alike**, not "tracked files"
  as A.5.5/item 28's phrasing implies. But only `len(...)` is kept; the
  `status` string itself (the actual 94 filenames) is discarded and never
  written to the JSON. **The file list was never captured, only the count.**

### §5.3 — `e2e_speedup.{json,md}`

- HEAD: `90ef59bbcccc9f644509f50d259580ca40f46e2d` (the same commit that
  later recorded `scaling_curve_v3` — this run happened on top of it).
- `provenance.git_dirty_file_count = 87`. Grepping both
  `scripts/e2e_speedup_ab.py` (the measurement harness) and
  `scripts/_e2e_speedup_report.py` (the report generator) for any of
  `git|dirty|porcelain|subprocess` finds **no git invocation at all**.
  Line 174 of `_e2e_speedup_report.py` is:
  ```python
  d["provenance"] = {
      "git_head": "90ef59bbcccc9f644509f50d259580ca40f46e2d",
      "git_dirty_file_count": 87,
      ...
  ```
  Both values are **string/int literals typed into the script**, not
  computed. There is no `git status --porcelain` call anywhere in the two
  files that produced this artifact.
- No log file, checkpoint, or sibling artifact from 2026-08-18 exists to
  cross-check against (searched `eval_results/*.log` for that date: none).
  The file's own mtime window is 17:25:59–17:43:26 that day; nothing else
  in the repo was touched in that window.

**Conclusion for part 1:** in neither case was the tree state captured
beyond a count, and in the §5.3 case not even the count was actually
computed — it is an unsourced number. Per the task's own framing, this
alone is the finding: A.5.5's claim was never checkable from either
artifact as it stands.

## 2. Classification of the dirty files

Not directly recoverable for either run (no list was retained). One
indirect data point exists:

`MLVU_ablation_long_trimmed.json` records the **same run-time HEAD**
(`9c66393d`) as `scaling_curve_v3`, with a full `status_raw` (not just a
count) captured by a different harness (`scripts/mlvu_eval.py::git_provenance()`,
which does keep the raw porcelain text). Its run finished at 21:16:22 on
2026-08-17 (file mtime); `scaling_curve_v3.json` was written at 23:31:04 the
same day — 2h15m later, same commit, same working tree, no intervening
commit. That status_raw lists **79 lines, all `??` (untracked), zero
`iris/*.py` entries** — all `eval_results/` scratch and `scripts/` harness
files (full breakdown: checkpoints, shard logs, watchdog scripts, inventory
JSONs, `_backup_singleproc/`, tmp dirs, etc. — see the file for the full
79-line list).

This is not proof for `scaling_curve_v3`'s own 94-file state (15 more files
accumulated in the intervening 2h15m, and the list wasn't re-captured), but
it is a genuine same-commit, same-session data point, not mere assertion —
strengthened by:

```
$ git diff --stat 9c66393d..90ef59b -- iris/
(empty)
$ git log --oneline 9c66393d..90ef59b -- iris/
(empty)
```

`iris/` is byte-identical between the run-time HEAD and the commit that
later recorded `scaling_curve_v3` — no `iris/` change was ever committed in
that window. That doesn't rule out an edit-then-revert inside the window,
but combined with the 79-file sibling snapshot showing zero `iris/*.py`
entries, it makes "no pipeline code was dirty" the well-supported reading
for §5.1, short of being a directly verified one.

For §5.3, no comparable cross-check exists. The commit immediately after
`90ef59b` to touch `iris/` is `48c082a` (2026-09-09) — three weeks later, a
different editing session, uninformative either way about the 87-file state
on 2026-08-18.

## 3. Measured-path relevance

Moot given part 2's finding: no pipeline file is confirmed dirty in either
run, so there is nothing to place on or off the measured path
(`_build_graph`/graph construction for §5.1; retrieval/answerer for §5.3).
If a pipeline file *had* turned up dirty, the check would be the same one
item 23 already ran: is it in the call path the harness times. It doesn't
arise here.

## 4. `defer_recompute` window

`defer_recompute=True` was introduced in exactly one commit:

```
$ git log --oneline --all -- iris/ingest.py | grep -i defer
3a2c1db perf(ingest): defer_recompute=True in _build_graph, behaviour-preserving
$ git log -1 3a2c1db --format="%ci"
2026-09-10 19:50:48 +0530
```

It appears in no other commit and, per the task's own account, was
uncommitted-but-present only from earlier in *this* session until
`3a2c1db` landed today (2026-09-11 session, committed 2026-09-10 evening).

Both runs under review predate that entirely:

- §5.1 run-time HEAD `9c66393d`: 2026-08-16 23:21:56; artifact written
  2026-08-17 23:31.
- §5.3 run HEAD `90ef59b`: 2026-08-18 17:10:18; artifact written
  2026-08-18 17:43.

Both are **~3–4 weeks before `defer_recompute` existed anywhere in the
repository's history**, committed or not. Neither run's dirty tree could
possibly have carried that change — the question of whether it "matters"
(it wouldn't, per item 23's zero-tolerance identity result) doesn't arise
because the premise doesn't hold.

## 5. Direct answers

**§5.1 (`scaling_curve_v3`, 94 files): A.5.5's claim is UNVERIFIABLE, not
true.** The artifact retained a count, not a list, so nothing about content
is directly checkable from it. It is, however, well-supported by
circumstantial evidence external to the artifact (same-commit sibling
capture at 79 untracked/zero-`iris/` files 2h15m earlier; `iris/`
byte-identical across the run-time-HEAD-to-recording-commit span). A.5.5
should not state this as an established fact. It should say: *the dirty
count (94) was captured but the file list was not; a same-commit sibling
artifact (`MLVU_ablation_long_trimmed`, `mlvu_provenance_C26.md`) captured
its own list at 79 files, all untracked scratch/harness files with no
`iris/*.py` entries, consistent with but not proof of this run's state.*

**§5.3 (`e2e_speedup`, 87 files): A.5.5's claim is UNVERIFIABLE, and the
number itself is unsourced.** `git_dirty_file_count: 87` is a hardcoded
literal in `scripts/_e2e_speedup_report.py`, not a computed git-status
result — the harness that produced this artifact never called `git status`
at all. There is no cross-checkable sibling for this commit/date. A.5.5
should say: *the reported dirty-file count for this run was not computed by
the harness; it was typed into the report script with no retained or
reproducible source. Nothing about the content of that dirty state — count
or composition — is verifiable from any artifact in this repository.*

Neither case is a repeat of item 23 (no pipeline file is shown dirty, and
`defer_recompute` postdates both runs by weeks), but neither is A.5.5's
"scratch and evaluation outputs" a checked fact either. The corrected
framing for A.5.5: one run's claim rests on indirect, same-commit
corroborating evidence; the other run's claim rests on nothing at all.

## Artifacts referenced

- `eval_results/scaling_curve_v3.json` (`provenance`), `scripts/scaling_curve_v3_report.py:172-195`
- `eval_results/e2e_speedup.json` (`provenance`), `scripts/_e2e_speedup_report.py:172-190`, `scripts/e2e_speedup_ab.py` (no git code)
- `eval_results/MLVU_ablation_long_trimmed.json` (`provenance.status_raw`), `eval_results/mlvu_provenance_C26.md`
- `scripts/mlvu_eval.py:380-395` (`git_provenance()`, retains raw status text)
- git commits `9c66393d`, `90ef59b`, `48c082a`, `3a2c1db`
