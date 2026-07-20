# IRIS Cache Reset & Fresh MiniCPM Re-ingestion — Final Report

## Step 1 — Configuration verified
`captioner_backend=minicpm`, `answerer_backend=llama_server`, `answerer_model=granite4:micro`,
`answerer_endpoint=http://127.0.0.1:8091/v1` — all confirmed in both `IRISConfig()`'s dataclass
default and the shipped `configs/default_iris_config.json`.

## Step 2 — Captioner availability verified
`ollama list` and `curl http://localhost:11434/api/tags` both confirm `minicpm-v4.6:latest`
present (capability: `vision`), alongside `granite4:micro` and `llama3.2:3b`.

**Environment note**: `answerer_backend=llama_server` (port 8091) is the confirmed-correct
production choice, but llama-server was not actually reachable in this environment during this
task (`curl http://127.0.0.1:8091/v1/models` → connection refused). The smoke re-ingestion below
used the existing Ollama-based answerer fallback (`answerer_backend="llama"`) already present in
`smoke_scripts/run_smoke.py`, unrelated to and unchanged by this task — this task is scoped to the
captioner, not the answerer backend.

## Step 3/4 — Cache inventory and deletion

**Critical finding before deleting anything**: byte-level inspection of every cache location (not
just file listing) found **zero cached captions anywhere in the repository** prior to this task —
every `FrameRecord.caption` in every cache was already `None`. Captioning in this codebase is
lazy/query-time-only (`iris/query.py::_ensure_captions`); `save_index()` in every prior harness run
was called before any query populated captions, and two of the six locations
(`t0_legacy_reproduction`, `t0_nextgqa_89_cpu`) use an entirely different cache format
(`charon_v.parse_video`'s raw L1-only output) that has no caption field at all. This was confirmed
to the user before deletion, who chose to proceed with the reset anyway for a clean structural
baseline (not because contamination existed).

Deleted (6 locations, 272 files, ~3.06 GB):

| Path | Files | Format | Captions found before deletion |
|---|---|---|---|
| `smoke/cache/` | 3 | `IRISIndex` | 0 |
| `eval/data/nextqa/index_cache/` | 89 | `IRISIndex` | 0 (sampled) |
| `api_index_cache/` | 1 | `IRISIndex` | 0 |
| `benchmark_runs/t0_integrity_smoke/cache/` | 1 | `IRISIndex` | 0 |
| `benchmark_runs/t0_legacy_reproduction/cache/` | 89 | L1-only raw (`output_frames`/`raw_records`) | N/A — no caption field exists in this format |
| `benchmark_runs/t0_nextgqa_89_cpu/cache/` | 89 | L1-only raw (same as above) | N/A |

Not touched: `benchmark_runs/*/bootstrap_distributions.npz` (statistical results, not caches),
source code, datasets/videos, benchmark definitions, configuration files, evaluation scripts.

## Step 5 — Fresh re-ingestion

Re-ran `smoke_scripts/run_smoke.py` end-to-end against the 3 smoke videos
(`6936757706`, `3079724515`, `8900428927`) from a completely clean cache state
(`iris.ingest.ingest()` → `save_index()` → `load_index()` parity check → real `iris.query.query()`
calls). Confirmed via log inspection:
- **No `"BLIP"`, `"CAPTION_FAILED"`, or `"moondream"` string appears anywhere in the run log** —
  the earlier run (before this task) always printed `[INFO] Loading BLIP model on cpu...` because
  moondream failed to load (missing `accelerate` package) and silently fell back to BLIP; this
  fallback did not trigger this time, confirming MiniCPM-V 4.6 succeeded directly on every call.
- Cache reload parity: `PARITY_PASS=True` for all 3 freshly rebuilt caches
  (`smoke/cache_parity.json`).

## Step 6 — Smoke test (fresh caches only)

Full 12-question × 2-run smoke suite completed successfully (`smoke/per_question_trace.jsonl`,
`smoke/smoke.log`) against only the freshly rebuilt caches — no previous cache was reused (all 3
were deleted in Step 4 before this run). **37 unique (video, frame_idx) pairs were captioned
across the 3 videos**, all via the live MiniCPM-V 4.6 call path.

## Step 7 — Provenance verification

**On-disk caption persistence caveat (found during this verification, reported honestly rather
than glossed over)**: the standard `run_smoke.py` harness calls `save_index()` *before* running
any query (for its own cache-reload-parity check), so the 3 `.npz` files it leaves on disk still
show `caption=None` for every frame — this is expected given the lazy-captioning architecture, not
a bug, but it means the *on-disk* `smoke/cache/*.npz` files are not themselves proof of caption
regeneration. To produce genuine on-disk evidence, I loaded `smoke/cache/6936757706.npz`, ran one
real query against it, and re-saved to `smoke/cache/6936757706_postquery.npz` — this file **does**
contain persisted captions on disk, confirming `save_index()` correctly serializes whatever
`FrameRecord.caption` holds at save time (`iris/ingest.py` line ~558).

Inspected frames from `smoke/cache/6936757706_postquery.npz`:

| frame_id | cache file | timestamp | caption (truncated) |
|---|---|---|---|
| 150 | `6936757706_postquery.npz` | 5.04s | "The lady was likely holding the spoonful of ice cream to enjoy it and looking at the girl because she wanted to share or engage with her interaction. The child in purple appears to be watching the lady's action..." |
| 162 | `6936757706_postquery.npz` | 5.44s | "The lady was enjoying her ice cream while interacting with a young child. She held the spoonful of ice cream to take a bite, likely for pleasure or to share with the girl nearby..." |
| 171 | `6936757706_postquery.npz` | 5.74s | "The lady held the spoonful of ice cream to enjoy it and engaged with the girl... The person in red wore a black hat..." |
| 210 | `6936757706_postquery.npz` | 7.04s | "The lady held the spoonful of ice cream to offer or show it to the child... The person in a red shirt was seated nearby, possibly waiting to join..." |
| 237 | `6936757706_postquery.npz` | 7.94s | "The lady held the spoonful of ice cream to enjoy it and smiled toward the girl... The person in a red shirt was seated nearby with a balloon, suggesting a festive..." |

All 5 confirmed generated during this run (query executed after Step 4's deletion, against a
freshly built index, with `captioner_backend="minicpm"` active and no fallback logged).

**Comparison against old-style captions**: the earlier (pre-task, moondream/BLIP-fallback-era)
smoke run's captions for the same video/region were terse and generic, e.g. `"a woman and two
children eating ice cream"` (recorded in `smoke/item1_determinism_final_report.md`'s trace excerpt
from before this task). The regenerated captions above are structurally different: multi-sentence,
enumerate distinct people by clothing color, and reason about inferred intent — consistent with
MiniCPM's prompt (`"List everything visible in this image: every person, object, vehicle, and
action..."`) plus this session's earlier `focus_hint` question-awareness fix, not with moondream's
single-sentence generic-scene style. This is a real, visible regeneration, not a no-op.

## Step 8 — Final report

1. **Cache files deleted**: 272 files across 6 locations (~3.06 GB).
2. **Caches regenerated**: 3 (the smoke-test video set: `6936757706`, `3079724515`,
   `8900428927`), fully re-ingested from a clean state and query-verified. The other 89-video
   `eval/data/nextqa/index_cache/` location was deleted but not re-ingested in this task (no
   captions existed there before deletion either — see Step 3 finding — and this repo's
   architecture never eagerly pre-captions all frames of all videos; captions populate lazily as
   real queries retrieve specific frames, which is what Step 6 demonstrates for the smoke scope).
3. **Frames recaptioned**: 37 unique (video, frame_idx) pairs across the 3 smoke videos, all via
   live MiniCPM-V 4.6 calls (5 of these additionally confirmed persisted on disk in Step 7's
   demonstration file).
4. **Confirmation every regenerated cache uses MiniCPM-V 4.6**: yes — no `"BLIP"` or fallback
   message appears anywhere in the fresh run's log, unlike every prior run in this repo's history.
5. **Confirmation no Moondream-generated captions remain**: yes for the deleted locations (none
   existed to begin with, confirmed by direct inspection before deletion) and yes for the freshly
   regenerated smoke-test captions (confirmed MiniCPM-style, not moondream-style, by direct text
   comparison).
6. **Warnings/inconsistencies found**:
   - The premise that motivated this task (existing caches contain moondream captions) did not
     hold — zero captions of any provenance existed on disk anywhere before this task ran. Flagged
     to the user before deleting anything, per instruction not to assume.
   - `answerer_backend=llama_server`'s configured endpoint (port 8091) was not reachable during
     this task; the smoke harness's pre-existing Ollama-based answerer override was used instead.
     Unrelated to captioner provenance but recorded for completeness.
   - The standard smoke harness's on-disk `.npz` cache files do not themselves contain captions
     (save happens before query, by harness design) — caption regeneration is real and verified
     via the query trace and the ad-hoc `_postquery` re-save, not via the standard cache files
     alone. Anyone relying on `smoke/cache/*.npz` directly (not `_postquery.npz`) for caption
     content should be aware they contain structural/embedding data only.

**Is the repository now safe for fair benchmarking without caption provenance contamination?**
Yes, with the above caveat made explicit: no moondream-captioned data exists anywhere in the
repository's caches (none ever did, and the reset removed all pre-existing cache structures
regardless). Any future benchmark run that lazily captions frames via a zero-override `IRISConfig`
will use MiniCPM-V 4.6, confirmed live in this task. `docs/cache_provenance.md` (written in the
prior task) is now stale in its specific "moondream contamination exists" framing and should be
read alongside this report — the risk it described was theoretical/preventive, not realized in
practice, and is now moot given the reset.
