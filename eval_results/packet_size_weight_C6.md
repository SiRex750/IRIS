# Appendix C item 6 — is `packet_size_weight` a dead key, and is a re-run actually required?

Read-only investigation. No ingest was re-run; no paper edits were made.

## Bottom line

**The premise in item 6 is wrong.** `packet_size_weight` was a real, defined field —
just not on the branch/commit currently checked out. It was defined on both
`ActionScoreConfig` and `IRISConfig`, was read by the scoring stage, and the
recorded triple `(0.8, 0.1, 0.1)` is the actual output of a real hyperparameter
sweep on `val_tune`, not a hand-typed or stale key. Git history plus the
existing tuning artifacts (`frozen_state.json`, `all_trials.csv`,
`selection_decisions.md` in the `ucf-vad-exp1` worktree) already establish
which config produced the 10.49% figure. **No re-run is needed to attribute
it.** The real caveat is narrower and different from what item 6 states (see
"What should replace item 6" below).

---

## 1. Was `packet_size_weight` ever a defined field on `IRISConfig`?

Yes. It was added in commit `5beab10` ("fix: production pipeline determinism,
defaults, and robustness fixes", 2026-07-20 20:47 +0530), which renamed the
existing field `luma_diff_weight` → `packet_size_weight` on **both**
`iris/action_score.py:ActionScoreConfig` and `iris/iris_config.py:IRISConfig`,
with a deprecation bridge:

```python
packet_size_weight: float = 0.5
luma_diff_weight: float | None = None  # deprecated
# __post_init__: if luma_diff_weight is set, it must agree with
# packet_size_weight (or raise), and packet_size_weight <- luma_diff_weight.
```

All consumers were updated in the same commit — `ActionScoreModule.score_all`
reads `self.config.packet_size_weight`, and `pipeline.py` forwards
`packet_size_weight=getattr(config, "packet_size_weight", 0.5)` when building
the action-score config from `IRISConfig`.

This landed on `main` via PR #8 (`fix/charon-full-decode-geometry`,
commit `87caa24`). **It was reverted the next day**: `091347d`
("Revert 'Merge pull request #8...'", 2026-07-21 08:30 +0530), which restored
`luma_diff_weight` as the live field name on `main`. That revert is why the
field does not exist on `main` or on the current branch
(`siddanth/peak-source-a6-p1`) today — this branch's root (`6f252ad`) predates
the rename entirely, so it never had `packet_size_weight` to lose.

But the merge base of `main` and the rename is `a752954` (2026-07-21 08:24
+0530), six minutes *before* the revert. The branch that produced the ingest
artifact in question, `siddanth/ucf-vad-exp1`, branches from `a752954` —
i.e. from `main` **after** the rename merged and **before** it was reverted.
On that branch `packet_size_weight` is the canonical, live field on both
`IRISConfig` and `ActionScoreConfig`, confirmed directly in the branch's own
`iris/iris_config.py` and `iris/action_score.py` (and still present in the
`.claude/worktrees/ucf-vad-exp1` checkout today).

So: `IRISConfig` **has** defined `packet_size_weight` — on the branch this
artifact came from, for the ~13 hours between the rename merging and the
revert landing on `main`. The item's claim that "`IRISConfig` has never
defined" it is false as a categorical statement; it is true only of the
branch currently checked out.

## 2. Renamed, not removed

There is no separate `packet_weight` / `packet_size_salience` term to chase —
`packet_size_weight` *is* the rename target. The shape match is exact: on
`ucf-vad-exp1`, `IRISConfig`/`ActionScoreConfig` carry the same three-weight
triple as the production branch, just under the new name for the first slot:
`(packet_size_weight, motion_weight, luma_entropy_weight)` replacing
`(luma_diff_weight, motion_weight, luma_entropy_weight)`, both validated to
be non-negative and sum to a positive value, both consumed identically in
the weighted residual sum.

## 3. What wrote `frozen_config_used` — object serialization, not a hand dict

The writer is `tuning/ucfcrime_vad_exp1_efficiency.py` (added in `41d7205`,
2026-07-29 13:48 +0530, on `ucf-vad-exp1`). It does **not** hand-assemble a
dict with typed-out keys. It:

1. Loads `tuning/frozen_state.json` (the Part-3 tuning harness's running
   state — see below) and pulls `frozen = state["frozen"]`.
2. Builds a real `ActionScoreConfig` from it by keyword:
   ```python
   action_cfg = ActionScoreConfig(
       packet_size_weight=frozen["packet_size_weight"],
       motion_weight=frozen["motion_weight"],
       luma_entropy_weight=frozen["luma_entropy_weight"],
       ...
   )
   action_module = ActionScoreModule(config=action_cfg)
   ```
   This only works because `packet_size_weight` was a real constructor
   parameter on that branch's `ActionScoreConfig` — a stale/unknown kwarg
   would raise `TypeError`, not silently succeed.
3. Writes the summary dict with `"frozen_config_used": frozen` — a direct
   copy of the same frozen-state dict used to build the live config object,
   not a re-typed literal.

`frozen_state.json` itself is written by the earlier Part-3 tuning harness
(`scripts/part3_tune.py`, introduced in `325775e`). That harness builds a
real `IRISConfig()` and does `setattr(cfg, k, v)` for each swept parameter,
selects a winner per family by `val_tune` mIoP (with IoP@0.5 tie-break), and
persists the winning values into `frozen["<field_name>"]` via
`save_frozen_state`. The `action_score_weights` family (added in `5058e56`,
"action_score_weights grid -- codec-dominant (0.8, 0.1, 0.1) beats default")
swept:
```
[(0.5, 0.3, 0.2), (0.8, 0.1, 0.1), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (0.34, 0.33, 0.33)]
```
against `IRISConfig.packet_size_weight/motion_weight/luma_entropy_weight`,
measured real mIoP/IoP@0.5 on `val_tune`, and selected `(0.8, 0.1, 0.1)` —
recorded in `tuning/selection_decisions.md`:

> Selected **action_score_weights = packet_size_weight=0.8,motion_weight=0.1,luma_entropy_weight=0.1**
> (mIoP primary; IoP@0.5 tie-break if within 0.005; then lower median retrieval
> latency; then default).

`tuning/frozen_state.json` in the `ucf-vad-exp1` worktree today still shows
`"packet_size_weight": 0.8, "motion_weight": 0.1, "luma_entropy_weight": 0.1`
under `"completed_families"` including `"action_score_weights"` — an exact
match to `frozen_config_used` in the efficiency artifact. This is a config
class that had a field and lost it (between branches), not a hand-assembled
dict with a typo.

## 4. Salience weights recorded alongside `packet_size_weight`

Yes — all three are present, and only three (no separate `luma_diff_weight`
key exists in this schema, since on this branch `packet_size_weight` occupies
that slot):

```json
"packet_size_weight": 0.8,
"motion_weight": 0.1,
"luma_entropy_weight": 0.1
```

This does **not** disagree with item 9's `(0.5, 0.3, 0.2)` — that figure
belongs to a different corpus entirely (the §5.1 scaling corpus, built on
production `main` under the shipped `luma_diff_weight` defaults). The
`ucf-vad-exp1` artifact is a separate experiment, on a separate branch, that
deliberately swept and selected a different, codec-dominant weighting. Both
numbers are real; they just describe different runs.

## 5. Does `packet_size_weight` appear in any other artifact?

In the tracked repo (outside the `ucf-vad-exp1` worktree), it appears in
exactly three files, and two of those are downstream quotes of the same
number:

- `paper/IRIS_paper_draft.md` — discusses the figure (§5.1 note 3, and
  Appendix C item 6 itself).
- `eval_results/clip_enrichment_cost_32sample.json` — embeds the *same*
  `reference_c5_measurements` block (including `frozen_config_used`) verbatim
  as a citation of the earlier artifact, not an independent second
  measurement.
- `eval_results/clip_enrichment_cost_stdout.log` — log output of the script
  that produced the above, printing the same embedded block.

Inside the (untracked, gitignored) `.claude/worktrees/ucf-vad-exp1/` checkout,
`packet_size_weight` is pervasive and consistent: `iris/action_score.py`,
`iris/iris_config.py`, `iris/ingest.py`, `tuning/frozen_state.json`,
`tuning/all_trials.csv`, `tuning/selection_decisions.md`,
`tuning/ucfcrime_vad_exp1/efficiency_measurements.json` (the source of the
figure copied into `clip_enrichment_cost_32sample.json`), several `tuning/`
report `.md` files, and three test files. All of it is internally consistent
with the branch's post-rename schema. There is no second, independent
artifact anywhere that used the name and disagreed with this one.

## Direct answers

**Is the recorded value meaningful under a different name, or a key no code
ever read?** Meaningful. On the branch that produced it,
`packet_size_weight` was the live, canonical name for what production `main`
calls `luma_diff_weight`. It was a real dataclass field, validated,
constructed via keyword argument (which would have raised `TypeError` if the
field didn't exist), and read directly in the weighted action-score sum. The
value `0.8` is the empirically selected winner of a five-way `val_tune`
sweep against the default `(0.5, 0.3, 0.2)`, recorded independently in
`frozen_state.json`, `all_trials.csv`, and `selection_decisions.md` — three
mutually consistent artifacts, not one.

**Is a re-run required to attribute 10.49% to a real config?** No. Git
history (the exact rename commit, its merge, and its revert, all dated) plus
the existing tuning artifacts already establish, without ambiguity, that
`retention_pct_mean = 10.49...` in `efficiency_measurements.json` was
produced with `ActionScoreConfig(packet_size_weight=0.8, motion_weight=0.1,
luma_entropy_weight=0.1, ...)` on commit `41d7205` of `siddanth/ucf-vad-exp1`
(2026-07-29 13:48 +0530, `generated_utc` in the artifact: `2026-07-29T08:17:11Z`
— the same instant in UTC). This is a fully attributed number.

## What should replace item 6

Item 6 currently frames this as "the weights are unrecoverable, re-run to
find out." That's not the finding. The real, narrower caveat — already
half-stated correctly in §5.1 note 3, but overcorrected there — is:

- `packet_size_weight=0.8` is **not** dead or unread; it was read, on the
  branch and commit that produced this artifact.
- It **is** true that on `main`/the current branch today, a field literally
  named `packet_size_weight` does not exist (§5.1 note 3's "consumed nowhere
  in the pipeline" is correct *only* as a statement about current
  production code, not as a statement about whether the artifact's own
  pipeline read it).
- The real gap is provenance-labeling, not data loss: the 10.49% figure
  should be attributed to the `ucf-vad-exp1` branch's selected
  `action_score_weights` config (0.8/0.1/0.1), explicitly distinguished from
  the production default (0.5/0.3/0.2, under `luma_diff_weight`) used
  elsewhere in the paper (e.g. the §5.1 scaling corpus, item 9). It is a
  same-project, different-branch/different-experiment number, not a
  production figure and not a corrupted one.

---

## Follow-up: was 5beab10 a pure rename, or did the computation change?

### 1. Full diff of `iris/action_score.py` in `5beab10`

Quoted in full above the config dataclass and `__post_init__` addition. The
arithmetic-relevant hunks:

```diff
 residual_n = self._normalize(residual)          # unchanged, both sides
 ...
 weight_sum = (
-    self.config.luma_diff_weight
+    self.config.packet_size_weight
     + self.config.motion_weight
     + self.config.luma_entropy_weight
 )
 ...
 action_score = (
-    self.config.luma_diff_weight * residual_n
+    self.config.packet_size_weight * residual_n
     + self.config.motion_weight * motion_n
     + self.config.luma_entropy_weight * entropy_n
 ) / weight_sum
```

`residual` and `residual_n` are **untouched** by this commit — same variable
name, same upstream computation, same normalization call, on both sides of
the diff. The only thing that changed is which config attribute is read to
supply the multiplier. **This is an identifier rename, not a computation
change.** (The same commit also fixes two unrelated bugs in the same
function — a `weights must be non-negative` check, and switching the
persistence-normalization denominator from the local per-video max to the
configured `max_prominence` — but neither touches `residual`/`residual_n` or
what feeds them.)

What does `residual` actually hold, before and after? Both before and after
`5beab10` (checked directly — before: `git show 5beab10^:iris/action_score.py`;
after: the current file, unchanged since):

```python
residual = np.array(
    [float(f.get("packet_size", 0.0)) for f in frame_features],
    dtype=np.float32,
)
...
residual_n = self._normalize(residual)
```

`residual` is read from `frame_features["packet_size"]` — the demuxed codec
packet byte size — **identically before and after the rename**, and
identically on `main` today (the field is back to being called
`luma_diff_weight` there, but it still multiplies this same `packet_size`-derived
`residual_n`). The class docstring, unchanged across the rename and present
on `main` right now, confirms this directly:

```
Input per frame:
    frame_idx
    packet_size        (codec coded packet size — residual channel)
    motion_magnitude
    luma_entropy
    luma_diff_energy   (diagnostic only; not consumed by scorer)
```

There is a genuine `luma_diff_energy` field on frame records — but it is
explicitly marked "diagnostic only; not consumed by scorer." The weight in
question, under either name, has never multiplied it.

The pre-rename field comment makes this explicit even before `5beab10`
existed: `iris/action_score.py:17` on `main` today (and on the pre-rename
code all the way back to `01f6eb2`, 2026-06-29) reads:

```python
luma_diff_weight: float = 0.5  # weights the codec packet-size residual; field rename deferred to Phase 7 (pipeline.py retirement)
```

The developers knew, from the field's introduction, that "`luma_diff_weight`"
was the wrong name for what it does ("weights the codec packet-size
residual") and had already flagged the rename as deferred work. `5beab10`
executed that deferred rename — it is documented in the codebase itself as a
name correction, three weeks before it happened.

### 2. What does the 0.8-weighted term compute on `ucf-vad-exp1` today?

Read directly from `.claude/worktrees/ucf-vad-exp1/iris/action_score.py`
(not `main`):

```python
residual = np.array(
    [float(f.get("packet_size", 0.0)) for f in frame_features],
    dtype=np.float32,
)
...
residual_n = self._normalize(residual)
...
action_score = (
    self.config.packet_size_weight * residual_n
    + self.config.motion_weight * motion_n
    + self.config.luma_entropy_weight * entropy_n
) / weight_sum
```

Tracing `frame_features[...]["packet_size"]` upstream to
`.claude/worktrees/ucf-vad-exp1/iris/charon_v.py`: it is populated as `"packet_size": ps`
at three sites in the demux loop — `ps` being the coded byte size of each
decoded packet read directly off the container (not a luma/pixel quantity at
all). So on `ucf-vad-exp1`, the term weighted at 0.8 is: *the per-frame codec
packet byte size, rank-normalized to [0,1] by `_normalize` (min-max over the
sampled frames), multiplied by 0.8, contributing to `action_score` alongside
`motion_weight`-weighted motion magnitude and `luma_entropy_weight`-weighted
luma entropy.* This is a genuine packet-size (codec residual) signal, not a
luma-difference signal, under either field name.

### 3. Why was it reverted? No stated reason found — and no reason citing naming

- `091347d`'s commit message is a bare git-generated revert: *"Revert 'Merge
  pull request #8 from swarapotd-rgb/fix/charon-full-decode-geometry' — This
  reverts commit 2c76ea0..., reversing changes made to 7b38c30f..."* No body,
  no rationale.
- `gh pr view 8` (repo `SiRex750/IRIS`) returns an empty PR body and zero
  comments. No review discussion is recorded anywhere reachable.
- `DECISIONS.md` (both on `main` and in the `ucf-vad-exp1` worktree) has no
  entry on 2026-07-20 or 2026-07-21 mentioning PR #8, the revert, or
  `packet_size_weight`/`luma_diff_weight`.
- The revert removed the *entire* PR #8 — 60+ files including unrelated
  determinism fixes (answerer seeding, `captioner_backend` default,
  multiple-choice wiring, GOP-seek decode overhead, malformed-PTS handling)
  and process docs (`FINAL_PRE_BENCHMARK_VERIFICATION_...md`,
  `benchmark_runs/...`) — not a targeted revert of the action-score rename
  alone. `main` never re-applied any part of PR #8 afterward (091347d is
  still `main`'s tip today); nothing on `main` suggests the rename
  specifically was rejected as wrong.
- Independent corroboration that the rename was *not* considered a
  correctness problem: `scripts/r0_gate_full.py` and `r0_gate_smoke.py` on
  the unrelated `origin/sonu/audit-docs` branch (a different contributor,
  much later) contain the comment:
  ```python
  # NOTE: main's ActionScoreConfig renamed packet_size_weight -> luma_diff_weight
  # (same slot; still weights the codec packet-size residual per its docstring).
  cfg = ActionScoreConfig(
      luma_diff_weight=frozen["packet_size_weight"],
      ...
  ```
  This is an independent contributor, working against `main`'s reverted
  schema, explicitly documenting the two names as the same slot and the same
  signal — direct third-party agreement with the "pure rename" conclusion,
  not a naming complaint.

So: there is no evidence anywhere in the repo that the revert happened
because the name "misdescribed the signal" — no such claim is recorded. The
revert reads as a broad rollback of PR #8 (likely for one or more of its
*other* changes, or a process/merge reason), not a rejection of the
`packet_size_weight` rename on its merits.

### 4. Does the tuning sweep documentation call this packet size or luma difference?

Packet size, consistently and explicitly. From
`tuning/selection_decisions.md` in the `ucf-vad-exp1` worktree:

- `` `packet_size_weight` (codec-residual) shifts the exact local maximum ``
- `` unlike `packet_size`, per-frame diagnostics don't move as directly... ``
- `` a direct codec-residual proxy for `packet_size` ``
- `` `packet_size_weight`'s dominance costs a little, consistent with codec... ``
- `` `action_score_weights` (a real winner, `packet_size_weight`-dominant) ``
- family header: `## Family: action_score_weights`, all rows and the
  selection line spelled `packet_size_weight=0.8,motion_weight=0.1,luma_entropy_weight=0.1`.

The sweep documentation never describes this term as luma difference at any
point — it consistently frames it as the codec packet-size / codec-residual
channel. The naming did not inherit any luma-difference framing to mislead
readers; if anything it is more accurately labeled here than the field name
that ended up shipping on `main`.

### 5. Is `packet_size_weight`/`luma_diff_weight` the same channel as `codec_conf`, or different?

Same underlying raw channel (`packet_size`), consumed in two different
places with two different transforms — confirmed by reading
`.claude/worktrees/ucf-vad-exp1/iris/ingest.py:340-376`:

```python
codec_conf_source = getattr(config, "codec_conf_source", "packet_size")
...
if codec_conf_source == "action_score":
    raw_signal[fi] = float(f.get("action_score", 0.0))
else:
    raw_signal[fi] = float(f.get("packet_size", 0.0))   # default
...
codec_conf_map = {fi: 0.1 + 0.9 * rp_map.get(fi, 0.5) for fi in raw_signal}
```

By default, `codec_conf` is `frame_features["packet_size"]` rank-percentile
normalized per pict-type and rescaled to `[0.1, 1.0]` — the **same raw
`packet_size` field** that `action_score.py`'s `residual`/`residual_n`
reads, just normalized differently (rank-percentile vs. min-max) and used
downstream as a graph-node confidence annotation rather than as an
action-score component. There is no second, independently-computed
packet-size salience term on this branch; `packet_size_weight` (a.k.a.
`luma_diff_weight` on `main`) and `codec_conf`'s default source are two
consumers of one channel. Item 9's `(luma_diff, motion, luma_entropy) =
(0.5, 0.3, 0.2)` corpus reads that identical `packet_size` field too (under
the `luma_diff_weight` name, on `main`'s code) — so the "luma_diff" in that
item's own label is, by the same evidence, itself a naming artifact rather
than a description of a real luma-difference computation.

## Direct answer

**What signal was weighted at 0.8 in the run that produced 10.49%
retention?** The per-frame codec packet byte size (`packet_size`, read
directly from the demuxed video stream in `charon_v.py`), min-max normalized
across the sampled frames — i.e. the codec packet-size residual channel.
Confirmed identical before and after the rename (`5beab10` changed only
which config attribute supplies the multiplier, not `residual`/`residual_n`
or their inputs), confirmed on the `ucf-vad-exp1` worktree's live code today,
and confirmed by the tuning sweep's own documentation, which never describes
the term as anything other than packet size / codec-residual.

**What may the paper truthfully call it?** "Packet-size weight" or "codec
packet-size residual weight" — not "luma-difference weight." This holds
regardless of which commit's field name is used to look it up, because the
arithmetic behind the name never changed: `main`'s current
`luma_diff_weight` and `ucf-vad-exp1`'s `packet_size_weight` are the same
slot multiplying the same `packet_size`-derived quantity. If the paper
describes item 9's `(0.5, 0.3, 0.2)` corpus using the term "luma_diff," that
label should be corrected too, on the same evidence — it names the field on
`main`, not the signal the field actually carries.
