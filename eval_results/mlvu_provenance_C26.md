# MLVU provenance audit — item 26 ("re-stamp or re-run before submission")

Read-only. Paper not edited. Question: is a commit identifiable for the MLVU
artifacts at all, and if not, what does the narrowest defensible date bracket
look like, what would a re-run cost, and does it settle item 37 too.

## 0. Affected-artifact list — confirmed, with one addition

The prompt's list (`MLVU_codec_baseline.*`, `MLVU_ablation.json`,
`MLVU_ablation_long*`, and their stdout logs) is **complete for what feeds
§6.2/§7.1**, with one clarification: `MLVU_ablation.md` and
`MLVU_ablation_long_trimmed.md` (the human-readable companions) also carry
provenance lines and should travel with their `.json` siblings. Full set,
each checked directly:

| artifact | feeds | own commit field |
|---|---|---|
| `MLVU_codec_baseline.{json,md}` | §6 M-Avg 0.340 (headline) | **records no commit** — `git rev-parse HEAD` raised (exit `3221225794` = Windows `STATUS_INTEGER_DIVIDE_BY_ZERO`), `dirty: null`. This is A.5.3's subject. |
| `MLVU_codec_baseline_checkpoint.json` | (intermediate, same run) | no provenance block at all (checkpoints are bare `{results, ingest_failures}`) |
| `MLVU_ablation.{json,md}` | §7 four-arm segmentation table | **records a commit**: `77f625badae67ad24fc57a9e986ef24e0ea6fe4b`, `dirty: true` |
| `MLVU_ablation_long_checkpoint.json`, `.shard0.json`, `.shard1.json` | (intermediate; final unsharded `MLVU_ablation_long.json` was never written — the full 175-video/6-task run was abandoned, see §3) | no provenance block |
| `MLVU_ablation_long_trimmed.{json,md}` | §7 long-video delta (+0.088 etc.) | **records a commit**: `9c66393d1624ebae5443f62ddbe8f5fb8dc12b7f`, `dirty: true` — but see §1 caveat, this pins `iris/` only, not the harness script |
| `MLVU_ablation_long_trimmed_checkpoint.shard0/1.json` | (intermediate) | no provenance block |
| stdout/stderr logs (`_mlvu_real_run_stdout.log`, `_mlvu_ablation_run_stdout.log`, `MLVU_ablation_long*_stdout.log`, `*_stderr.log`, `*_watchdog_stdout.log`, `*_merge_stdout.log`) | run narrative, not a separate provenance source | inherit whatever the run that wrote them recorded |

So **only `MLVU_codec_baseline.{json,md}`** — the item A.5.3 actually names —
has a *missing* commit field. The other two named artifacts have a commit
field present, but §1 below shows why "has a field" and "is trustworthy"
diverge for one of them, and §0 surfaces a fourth file
(`MLVU_ablation_long_trimmed.md`) not separately named in the prompt but
carrying the same field.

## 1. Can the producing commit be identified at all?

**No discriminating field exists for `MLVU_codec_baseline.json` the way
`captioner_backend` did for item 9.** The artifact's `config` block records
`captioner_backend: "moondream"` and `config_hash: e67562fb00850bd1`, but
A.5.7 (already in the paper) establishes `captioner_backend` is a dead field
— `aria.get_captioner()` ignores the passed `IRISConfig` entirely and always
resolves MiniCPM via its own `ConfigManager()` load of
`configs/default_iris_config.json`, which has specified `minicpm` unchanged
across every commit examined. A field that never varies across the
repository's history cannot bracket a commit; item 9's technique worked
because `beta` genuinely differed between trees. `config_hash` is the same
problem restated — it incorporates the dead field, so two runs at different
hashes can share an identical actual code path, and no *other* field in the
`config` block (all `IRISConfig` dataclass values) varies with commit either
— they are options selected on the CLI, not code-state fingerprints.

**So the direct-identification path item 9 used is closed here.** The
artifact carries nothing that pins code state.

**But the working-tree evidence closes the gap anyway, for two of the three
named artifacts, through a different mechanism: matching the harness
script's own mtime against its commit time.**

- `scripts/mlvu_eval.py` (produces `MLVU_codec_baseline.json`): working-tree
  mtime `2026-08-16 02:08:37 +0530`; committed as `6baf0f9` at
  `2026-08-16 11:27:29 +0530`, ~9h19m later; `git diff HEAD -- scripts/mlvu_eval.py`
  is empty right now. The file has not been touched since 02:08, and what's
  on disk now is byte-identical to `6baf0f9`.
- `scripts/mlvu_ablation.py` (produces `MLVU_ablation.json`): working-tree
  mtime `2026-08-16 12:08:31 +0530`; committed as `77f625b` at
  `2026-08-16 12:16:16 +0530`, ~8 minutes later; also clean against HEAD now.

Neither of these is a formal proof — a file's mtime records its last write,
not a hash comparison against history, and nothing rules out an edit between
last-write and commit that happened not to change the file's content-visible
diff footprint (impossible for a plain edit, but worth stating as the
residual gap). Taken with the "no diff today" fact, though, this is about as
strong as circumstantial evidence gets short of the commit itself: the
script was last saved once, some minutes-to-hours before its own commit,
and was never touched again.

**`MLVU_ablation_long_trimmed.{json,md}` is a different case, and its commit
field is misleading if read as pinning "the code that ran."** Its
`provenance.status_raw` at run time lists `scripts/mlvu_ablation_long_trimmed.py`
itself as `??` (untracked) — and `git log --all` for that file, and for its
three siblings (`mlvu_ablation_long.py`, `mlvu_ablation_long_merge.py`,
`mlvu_ablation_long_trimmed_merge.py`), returns **nothing, on any ref,
ever**. These four scripts have never been committed anywhere. So
`9c66393d` genuinely pins `iris/` (the dirty list is all `eval_results/`
scratch and `scripts/` harness files — no `iris/*.py` entries — consistent
with A.5.5's characterization), but it says nothing about the harness script
that actually drove retrieval, answering, and scoring for the long-video
delta. That script's current working-tree mtime is `2026-08-17 17:04:44`,
about 20h after the run that used `9c66393d` as HEAD — so even the
"last-saved-shortly-before-run" argument that closes the gap for the other
two scripts doesn't apply cleanly here; the file kept moving after the run
started (shard0/shard1 span `2026-08-17` daytime into evening). Practically:
trust the `iris/` pin, not the illusion that the whole harness is pinned by
it.

**Bottom line on Q1:** no field-based discriminator exists for any MLVU
artifact (item 9's technique doesn't transfer). Circumstantial mtime/diff
evidence is strong for `MLVU_codec_baseline.json` (→ effectively `6baf0f9`)
and `MLVU_ablation.json` (→ effectively `77f625b`), weak-to-absent for the
harness half of `MLVU_ablation_long_trimmed.json` (only its `iris/` half is
pinned, at `9c66393d`).

## 2. Date bracket (not relying on filesystem clone mtimes)

Per C22c's own finding, tracked files share a bulk-checkout mtime and are
not usable for dating; these `eval_results/`/`scripts/` files are
**untracked**, so git never touched them and their mtimes are genuine
per-file write times, not clone artifacts — confirmed by their mutual
consistency below (each pair of "run finished" / "commit landed" timestamps
lines up to single-digit minutes, which a bulk stamp could not produce).

**`MLVU_codec_baseline.json` — narrowest bracket: `[2026-08-16 02:16:44,
2026-08-16 05:50:49] +0530`, i.e. under 3h34m.**
- Lower bound: `_mlvu_real_run_stdout.log` line 1 — retry-wrapper
  `attempt 1/15`, `Sun, Aug 16, 2026 2:16:44 AM`. This log **is** the
  baseline run — confirmed by grepping it: it contains the identical
  `"m_avg": 0.34` and the identical `<unavailable: ...>` HEAD string at the
  point the full JSON report is printed (line ~4186 prints
  `Wrote ...MLVU_codec_baseline.json` / `.md`), followed immediately by the
  sanity-guard failure (AO/AC below floor) that made the retry-wrapper stop
  ("this is a real result, not retrying") rather than discard the output.
  **`iris/ingest.py` writes its output files (step 5) before the sanity
  guard runs (step 6)** — so a guard-*failed* run still leaves a completed,
  valid 150-question artifact on disk, which is exactly what happened here.
- Upper bound: `MLVU_codec_baseline.json`/`.md`/`_checkpoint.json` mtimes,
  all `2026-08-16 05:50:44–49 +0530`, matching the stdout log's own last
  write (`05:50:49`).
- What distinguishes the endpoints: nothing in `iris/` — no `iris/*.py`
  commit falls inside this window (the prior `iris/` commit is `3d83b2c`,
  2026-07-27; the next is `b6bbd78`, 2026-08-16 11:27:13, hours after this
  run finished). **The run predates every `iris/` commit made that
  morning**, including `b6bbd78` ("fix(query): truncate long CLIP text
  queries instead of silently zero-embedding") and `bffa92e`
  ("scene_segmentation ablation knob"). `b6bbd78`'s bug — long
  `clip.tokenize()` inputs raising past the 77-token cap, silently falling
  back to an all-zero embedding, which then fails retrieval — is a plausible
  (not confirmed by direct token-count measurement here) match for exactly
  the failure this run hit: `mlvu_eval.py` embeds the **full `mc_prompt`**
  (question + enumerated options + instruction line, not the bare question)
  for retrieval, and it is the two tasks with the most verbose option sets
  (AO, AC) that failed the floor, not all six. This is circumstantial, not
  proven — no query token count was measured in this pass — but it is a
  concrete, checkable reason the pre-fix window matters, not merely a
  date technicality.

**`MLVU_ablation.json` — narrowest bracket: essentially the commit itself.**
Ran `2026-08-16 12:18 PM → ~9:39 PM` per its own `.md` (wall clock ~9h15m,
600/600 work items); file mtime `23:20:42` (about 1h40m past the `.md`'s
stated end time — likely guard-check + markdown-write + retry-wrapper
overhead after the last query, not a discrepancy worth chasing further).
This entirely postdates `b6bbd78`/`bffa92e`/`465800b` (all landed by
`12:05:27` that day) and its own script commit `77f625b` (`12:16:16`,
2 minutes before the stated start). **This run's codec arm is the strongest
evidence in the whole audit**: `MLVU_ablation.md`'s GUARD (b) reports
`codec_reproduces_baseline: true`, `codec_vs_baseline_diff: {}` — a
from-scratch re-run of the identical 150 questions, **after** the
CLIP-truncation fix landed, reproduced the pre-fix baseline's M-Avg and all
six per-task accuracies **exactly**. Whatever the pre-fix query-embedding
bug did or didn't do to `MLVU_codec_baseline.json`, it left no visible mark
— the number is independently reproduced under known-good code. That doesn't
retroactively give the baseline a commit, but it is direct empirical
evidence that the commit uncertainty doesn't move the number, which is the
thing item 26 is actually protecting against.

**`MLVU_ablation_long_trimmed.json` — `iris/` pinned at `9c66393d`
(2026-08-16 23:21:56, i.e. the commit landed ~1 minute after
`MLVU_ablation.json` finished writing — this is the commit that records the
ablation *results*, not new pipeline code); harness script unpinned, last
written `2026-08-17 17:04:44`, mid-run** (shard0/shard1 finish at
`20:53:52`/`21:15:36` that evening, merge at `21:16:22`). No `iris/` commit
falls between `9c66393d` (08-16 23:21:56) and the next one, `48c082a`
(2026-09-09) — so for over three weeks `iris/` sat at one commit, and this
run is deep inside that window with nothing else changing under it. If the
question is "did pipeline code move under this run," the answer is no; if
the question is "is the harness script pinned," the answer is still no.

**One more thing this bracket incidentally surfaces**, not asked for but
material to the same family of artifacts: `MLVU_ablation_long_trimmed.py`'s
`--answerer-endpoint` CLI default is `http://localhost:11434/v1` (Ollama's
own OpenAI-compatible port) and the shard logs confirm that default was used
unmodified — whereas `MLVU_codec_baseline.json` and `MLVU_ablation.json`
both recorded `answerer_endpoint: http://127.0.0.1:8091/v1` (a standalone
llama-server process). **The long-video answerer never touched llama-server
at all**, which sidesteps item 22's build-identity question for this one
artifact specifically (there is no llama-server build to identify — it ran
through Ollama), but it means the MLVU family answers its questions through
two different serving paths depending on which script produced them, an
inconsistency not previously disclosed.

## 3. Re-run cost

- **Corpus**: `./mlvu` (12 GB) is present locally and is the full MLVU
  corpus the harness samples from. `./mlvu_long_staging` (471 MB) and
  `./mlvu_long_staging_trimmed` (62 KB, i.e. emptied after its run) are
  scratch download/delete staging dirs for the long-video arms — the videos
  themselves are downloaded per-video and deleted after ingest, not kept.
  Re-running the long-video arms means re-downloading those videos.
- **Caches**: no IRIS `index_cache/*.npz` entries exist for any MLVU video —
  `mlvu/.cache` is only a HuggingFace model-download cache (157 files, 940
  KB, all `.metadata`/`huggingface` housekeeping), not a per-video ingest
  cache. A re-run rebuilds every scene graph from scratch; there is no
  cached shortcut the way `index_cache/*.npz` provided for the scaling
  corpus.
- **Captioner and answerer availability**: `ollama list` currently shows
  `granite4:micro` (2.1 GB, the recorded answerer model) and
  `minicpm-v4.6:1b` (1.6 GB, the captioner A.5.7 establishes is what
  actually ran regardless of the recorded `captioner_backend` field) both
  installed locally. Both are runnable today.
- **llama-server**: port 8091 (the baseline/ablation's recorded answerer
  endpoint) is not currently listening (`curl` connection refused). Per
  item 22 (already resolved in the paper), no llama-server build string can
  be verified for *any* reported run — `b9976` is prose-only, the one
  independently-verified build (`b10099`) ran on a different machine for
  unrelated work and was confirmed gone as of 2026-07-27. A re-run through
  the `8091` path would need a llama-server stood up again, with no
  verified-build precedent to match against — it would produce a *new*,
  independently verifiable build identity rather than resolve the old one.
  A re-run through Ollama's `11434` path (as the long-video trimmed run
  already did) sidesteps this entirely, at the cost of matching a different
  serving stack than the baseline/short-ablation used.
- **Wall clock, from the runs themselves**: baseline (150 Q, single arm)
  ~3h34m; four-arm ablation (600 work items, shared ingest across arms)
  ~9h15m; the *trimmed* long-video ablation (45 videos × 2 arms × 2 tasks,
  sharded across 2 workers) ran from some point on 2026-08-17 into the
  evening (shard0/1 finish ~20:54/21:16). **The full-spec long-video run
  (175 videos, 4 arms, 6 tasks) was attempted and abandoned** — its watchdog
  log (`MLVU_ablation_long_trimmed_watchdog_stdout.log`... actually the
  un-trimmed watchdog, `_mlvu_ablation_long_watchdog.sh`'s log) shows a
  hard abort on free RAM staying below a 1024 MB floor for two consecutive
  polls, and no final `MLVU_ablation_long.json` was ever written — only
  checkpoints and shards survive. The *trimmed* spec exists specifically
  because the full one was "too slow for the question at hand" (the
  script's own docstring) on this machine's memory budget. A full re-run at
  the original 175-video/4-arm/6-task scope should be assumed to hit the
  same memory ceiling again unless run with fewer parallel shards or more
  RAM headroom.

## 4. Does the MLVU path touch `hierarchical_sparse`?

**Yes, confirmed directly, not assumed.** `graph_edge_mode` is not
overridden by any of the four MLVU harness scripts
(`mlvu_eval.py`, `mlvu_ablation.py`, `mlvu_ablation_long.py`,
`mlvu_ablation_long_trimmed.py` — checked each `IRISConfig(...)`
construction site) so all four take `IRISConfig`'s dataclass default,
which is `graph_edge_mode: str = "hierarchical_sparse"`
(`iris/iris_config.py:66`). `MLVU_codec_baseline.json`'s own recorded
`config.graph_edge_mode` confirms `"hierarchical_sparse"` directly (not
inferred). This matches §3.5's table (`§6.2 MLVU, §7.1 segmentation
ablation → hierarchical_sparse`).

**This means item 37 is correct that no cached `hierarchical_sparse` index
exists anywhere in the repository, and a re-run of any MLVU artifact would
produce one as a side effect.** A re-run to close item 26 and a re-run to
close item 37 are the same build: any MLVU harness run ingests every sampled
video under the tiered path and would leave `index_cache/*.npz` entries (or
whatever ephemeral in-memory structure the harness uses — worth checking
whether the harness persists index caches to disk at all, since the current
absence of any MLVU `index_cache` entries despite these runs having already
happened suggests the harness may not cache to disk by default) with real
`temporal`/`hierarchy_*`/`semantic_salient`/`motion_neighbor` edges,
answering item 37's open measurement (per-node degree distribution,
edge-type breakdown, in-degree concentration) directly from the re-run.
**If the team re-runs for item 26, instrument that run (or a small
single-video companion run against the same corpus) to dump its
`index_cache` so item 37's degree-distribution analysis
(`eval_results/degree_distribution_C8.md`'s method) can be pointed at a real
`hierarchical_sparse` cache instead of closing with "still unmeasured."**

## Recommendation

**Disclose a bracketed range for `MLVU_codec_baseline.json`; do not
re-stamp it with a commit it cannot support.** No commit can be honestly
assigned — `6baf0f9` is a strong circumstantial match (harness script
unchanged from 9 hours before its own commit) but the artifact's own
`git rev-parse` failure means there is no proof, and no discriminating field
closes the gap the way item 9's did. Write A.5.3 as: *produced between
2026-08-16 02:16:44 and 05:50:49 (+0530), before every `iris/` commit made
that day, including the CLIP-query-truncation fix (`b6bbd78`) — and that its
number is independently reproduced by a same-day, post-fix re-run
(`MLVU_ablation.json`'s codec arm, exact match) is the strongest evidence
available that the missing commit does not put the headline M-Avg 0.340 in
doubt.* That is a real, checkable claim — stronger than a ledger quote,
honest about the gap, and it doesn't require touching the harness again.

**Re-running is the better option only if the team wants item 37 closed at
the same time** — the MLVU path is the one place in the whole paper that
already runs `hierarchical_sparse`, so a re-run instrumented to persist its
index cache pays for itself twice: it gives `MLVU_codec_baseline.json` (or
a deliberate replacement of it) a real commit going forward, and it gives
item 37 the first-ever measured `hierarchical_sparse` degree distribution.
Cost: corpus and both models are present and runnable today; no llama-server
build exists to match against 8091 so budget for either standing one up
fresh (new, verifiable build — doesn't resolve item 22's old citations, but
stops adding to the pile) or switching to the Ollama `11434` path the
long-video scripts already default to (simpler, but widens the
already-undisclosed inconsistency in §2 unless the baseline is re-run
through the same path too, for consistency across the MLVU family). Time
budget: ~3.5h for a `mlvu_eval.py`-only re-run at the original scope; the
175-video full long-video ablation previously failed on a 1 GB RAM floor and
should be run with fewer parallel shards or explicit headroom if attempted
at full scope again.
