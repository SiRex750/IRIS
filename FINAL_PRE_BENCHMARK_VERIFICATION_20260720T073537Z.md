# IRIS Final Pre-Benchmark Verification Pass

**Verification timestamp (UTC):** 2026-07-20T07:35:37Z
**Branch:** `fix/charon-full-decode-geometry`
**Method:** Independent, evidence-based source verification. Every status below was
assigned by opening the current source at the cited `file:line`, comparing it to the
original defect definition in `MAIN_BRANCH_REAUDIT_2026-07-15.md`, and against the
prior status in `ZIP_SCOPED_FIX_VERIFICATION_2026-07-19.md`. `walkthrough.md` claims
were **not** trusted; each was re-derived from code.

---

## GO / NO-GO DECISION

**GO (conditional) for the paper-grade NExT-GQA benchmark.**

All prior NO-GO blockers from 2026-07-19 that were correctness bugs are now fixed:
scene_id and full motion geometry now propagate into the canonical graph (P1-09,
P1-12), graph build is now sub-quadratic via ANN for large N (P1-10), and the V2
contradiction-probability bug (P1-21) is fixed. There are **zero P0-open, zero
NOT-FIXED** items and the full suite passes (338 passed, 1 xfailed, 0 failed).

GO is **conditioned** on scoping paper claims to match 8 PARTIAL items — chiefly:
1. **L1 is OFF by default (`use_l1=False`).** L1 selection is now wired and functional,
   but under production defaults L1 still does not control retrieval. Do **not** claim
   L1 improves end-to-end retrieval/latency unless you run with `use_l1=True` and show
   real cache pressure/eviction (P1-01, C1).
2. **No production-FAISS acceleration claim.** `L2TieredIndex` internal defects are
   fixed but it is still not wired into `ingest()->query()` (P1-15).
3. **Visual verification still lets a weaker entailment override a stronger
   contradiction** in `verify_visual_claim` (P1-22). Acknowledge this limitation for
   any V2 visual-grounding verification claim.
4. Report "selective **pixel** processing," not zero-decode frame skipping (P1-03), and
   note adaptive thresholds replace configured `salient/candidate_thresh` (P1-05).

> **Note on `walkthrough.md`'s "20/20 fully resolved" claim: this is an OVERCLAIM.**
> Of the 20 findings it lists as fully resolved, this pass finds 12 FIXED and 8 PARTIAL.
> None are regressions or crashes; the PARTIALs are claim-scoping / inherent-design
> items, which is why the overall decision is still GO(conditional).

---

## Provenance / document-availability note

None of the governing documents exist as loose files in the repo working tree. They
were recovered from the untracked **`CCBD.zip`** (`CCBD/` prefix) at repo root:

- `MAIN_BRANCH_REAUDIT_2026-07-15.md` — recovered from `CCBD.zip`, used as defect checklist.
- `ZIP_SCOPED_FIX_VERIFICATION_2026-07-19.md` — recovered from `CCBD.zip`, used as prior status.
- `walkthrough.md` — recovered from `CCBD.zip` (dated 2026-07-20), treated as an untrusted self-report.
- `FINAL_PRE_BENCHMARK_VERIFICATION_PROMPT.md` — **not present anywhere** (not in repo, not in `CCBD.zip`,
  not supplied as an inline document). Parts A–F and the output structure below were reconstructed
  from the task instructions. Part D (environment blockers) and Part E (test-integrity files) had no
  authoritative enumerations available, so they are answered best-effort from the reaudit's own
  "environment limitations" / "test-suite gaps" sections; this is flagged inline.

---

## Part A — Fresh test-suite run

Command: `python -m pytest -v tests/` (full run, log at `full_test_output.log`).

```
=========== 338 passed, 1 xfailed, 10 warnings in 276.24s (0:04:36) ===========
```

| Result | Count |
|---|---:|
| passed | 338 |
| failed | 0 |
| errors | 0 |
| xfailed | 1 |
| skipped | 0 |
| **collected** | **339** |

No failures, no errors, no unexpected skips. (For comparison, `walkthrough.md` reported
335 passed / 1 xfailed; the count grew because new tests were added, e.g.
`test_e2e_integration.py`.)

---

## Part B — Full 33-ID defect verification

Legend: **FIXED** = original defect condition no longer present in cited current code.
**PARTIAL** = behavior changed / diagnostics added, but at least one condition of the
original finding still exists (usually inherent design or a claim-scoping caveat).

### Summary table

| Severity | In scope | Fixed | Partial | Not fixed |
|---|---:|---:|---:|---:|
| P0 | 1 | 1 | 0 | 0 |
| P1 | 24 | 19 | 5 | 0 |
| P2 | 8 | 5 | 3 | 0 |
| **Total** | **33** | **25** | **8** | **0** |

Movement vs 2026-07-19 (13 fixed / 16 partial / 4 not-fixed): **+12 fixed, all 4
previously NOT-FIXED items resolved or upgraded to PARTIAL, zero regressions.**

### Full defect table

| ID | Sev | 07-19 status | Current status | Current file:line evidence |
|---|---|---|---|---|
| P0-03 | P0 | FIXED | **FIXED** | `iris/l2_asphodel.py:609` `add_cross_scene_edges()` present; `graph_override` accepted; `tests/test_scene_sparse_descend.py` all pass. No regression. |
| P1-01 | P1 | NOT FIXED | **PARTIAL** | New `_retrieve_with_l1()` (`iris/query.py:584`) calls `index._l1_cache.query()` (`:630`) and is wired into both canonical paths (`:694`, `:796`). BUT gated on `use_l1` which defaults **False** (`iris/iris_config.py:106`). Under defaults L1 still does not select. Improved from NOT FIXED. |
| P1-02 | P1 | FIXED | **FIXED** | `geom` computed unconditionally for every processed frame incl. `SKIP` under `full_decode` (`iris/charon_v.py:406-408`). No regression. |
| P1-03 | P1 | PARTIAL | **PARTIAL** | Instrumentation added (`iris/charon_v.py:512-517`), claim corrected to "selective pixel processing". FFmpeg still `container.decode()`s every dependent frame (inherent to H.264). Correctly reported, not eliminated. |
| P1-04 | P1 | FIXED | **FIXED** | `parse_video()` returns packet curve + fps in stats (`iris/charon_v.py:519-522`); `ingest()` reuses them (`iris/ingest.py:428+`). No regression. |
| P1-05 | P1 | PARTIAL | **PARTIAL** | Configured vs effective thresholds separately reported (`iris/charon_v.py:524-529`). With `adaptive=True` configured values are still replaced by scene percentiles — inherent, now honestly reported. |
| P1-06 | P1 | PARTIAL | **FIXED** | `packet_size_weight` w/ deprecation bridge (`iris/action_score.py:17-37`); constant channel → 0 not 0.5; persistence uses configured `max_prominence` (`:119`,`:181`); docstring **corrected** to "does not replace tier decisions" (`:57-59`) — the last 07-19 residual. |
| P1-07 | P1 | FIXED | **FIXED** | Signed divergence (`iris/charon_v.py:62-67`); L1 normalizes entropy/saturates Hessian (`iris/l1_elysium.py:184`). No regression. |
| P1-08 | P1 | FIXED | **FIXED** | `IRISConfig.graph_mode` defaults `scene_sparse` (`iris/iris_config.py:45`). (See C4 for the JSON-config caveat.) |
| P1-09 | P1 | PARTIAL | **FIXED** | `_motion_similarity()` is 6-D cosine (`iris/l2_asphodel.py:271-303`) AND the canonical `_build_graph()` feature record now carries divergence/curl/jacobian/hessian/motion_entropy (`iris/ingest.py:138-142`) — the exact propagation gap 07-19 flagged is closed. |
| P1-10 | P1 | NOT FIXED | **FIXED** | ANN neighbor search now wired: FAISS `IndexFlatIP` for semantic edges at N≥200 (`iris/l2_asphodel.py:517-528`) and `scipy.spatial.cKDTree` for motion edges (`:595-605`), with chunked bounded-memory fallback. Small-N (<200) uses exact matrix — bounded, acceptable. |
| P1-11 | P1 | FIXED | **FIXED** | Edges retain sorted set of all relation labels (`iris/l2_asphodel.py:372-401`). No regression. |
| P1-12 | P1 | PARTIAL | **FIXED** | `_refresh_scene_ids()` preserves authoritative IDs (`iris/l2_asphodel.py:239-247`) AND ingest now passes `scene_id` in the feature record (`iris/ingest.py:137`). Production graph nodes now receive the packet-valley scene IDs. |
| P1-13 | P1 | PARTIAL | **FIXED** | Same-scene-first parent search (`iris/l2_asphodel.py:424-436`) plus explicit self-exclusion of the candidate from its own parent pool (`:456`) prevents the all-candidate orphan case. |
| P1-14 | P1 | PARTIAL | **FIXED** | Near-zero embeddings now raise (`iris/_clip.py:172-176`,`:195-197`) AND the previously-missing per-query telemetry (`embedding_backend`, `fallback_reason`, `effective_method`) is now emitted (`iris/query.py:172-202`). |
| P1-15 | P1 | PARTIAL | **PARTIAL** | HNSW inner-product + post-train PQ insertion fixed (`iris/l2_index.py:85`,`:163`), 17 tiered-index tests pass. Still referenced only by tests — **not** wired into canonical `ingest()->query()`. No production FAISS claim permitted. |
| P1-16 | P1 | PARTIAL | **FIXED** | `get_captioner(config=...)` now accepts config (`iris/aria.py:175`) and both caption-path callers pass it (`iris/aria.py:693`,`:745`). Residual: `_ACTIVE_CAPTIONER` is a process singleton, so a later differing config is ignored within a run — minor. |
| P1-17 | P1 | PARTIAL | **FIXED** | Hard-coded `"model": "loaded"` removed; direct/native payloads now use `model_name = model or self.text_model` (`iris/aria.py:354`,`:376`,`:434`,`:443`,`:470`). Passed `max_tokens` respected. No test asserts `"loaded"` anymore. |
| P1-18 | P1 | FIXED | **FIXED** | `unload_captioner()` clears `_ACTIVE_CAPTIONER` forcing correct-device reinit (`iris/aria.py:198`). No regression. |
| P1-19 | P1 | FIXED | **FIXED** | Moondream picks CUDA only when available, else float32 CPU (`iris/aria.py:77-100`). No regression. |
| P1-20 | P1 | PARTIAL | **FIXED** | `_ner_overlap()` now marks **all** claims `unverifiable` — entity-overlap-as-verification removed (`iris/cerberus_v.py:442-445`). filtered_nli low-confidence → neutral/unverifiable (`:343-346`); shared aggregation rejects on stronger contradiction (`:405-407`). |
| P1-21 | P1 | NOT FIXED | **FIXED** | `score_nli_pair()` returns **label-specific** probability (`iris/cerberus_layers.py:149-154`) with symmetric 0.85 floor (0.5 negation-risk) for both entailment and contradiction (`:162-167`). Best-contradiction now selected on contradiction prob (`:247`). |
| P1-22 | P1 | NOT FIXED | **PARTIAL** | Oversize-sentence truncation risk removed — oversize sentences scored as neutral 0.0 rather than fed to a truncating tokenizer (`iris/cerberus_layers.py:228-234`). BUT `verify_visual_claim()` still sets `verified` whenever any sentence entails, before contradictions (`:250-252`) — a stronger contradiction cannot defeat a weaker entailment on the V2 visual route. |
| P1-23 | P1 | FIXED | **FIXED** | `run_pipeline()` delegates to canonical `ingest()`/`query()` (`iris/pipeline.py:585`,`:597`,`:600`). One active end-to-end route. No regression. |
| P1-24 | P1 | PARTIAL | **FIXED** | V2 output is JSON-safe: `to_dict()` on claim verdicts / core verdict / answer_claims (`iris/query.py:742-743`,`:750`) and a top-level `"answer"`+`"verified"` field (`:747`; `to_dict` defs `iris/claim_contract.py:385`, `iris/cerberus_layers.py:489`). `test_e2e_integration.py:135` proves `json.dumps()`-safety. Residual: flat vs scene-shortcut dicts may still differ in some optional fields. |
| P2-01 | P2 | FIXED | **FIXED** | `ingest()` passes `config.peak_order`; both Charon paths use one fps-aware `effective_order` (`iris/charon_v.py:259-269`, `peak_order_used` `:503`). No regression. |
| P2-02 | P2 | PARTIAL | **FIXED** | Canonical `ingest()` now calls `codec_validator.validate_video(..., level=lvl)` with `lvl` defaulting to **strict** (`iris/ingest.py:480-481`; `codec_validation_level="strict"` `iris/iris_config.py:109`). The 07-19 defect (ingest used fast prefix mode) is closed. Strict whole-stream mode exists in the validator. |
| P2-03 | P2 | FIXED | **FIXED** | `add_frame_nodes_bulk()` raises `ValueError` on mismatched parallel list lengths before `zip()`. No regression. |
| P2-04 | P2 | FIXED | **FIXED** | Domain-neutral zero-shot vocabulary; rabbit/cartoon/BBB labels gone (`iris/_clip.py:64`, `iris/pipeline.py:101`). No regression. |
| P2-05 | P2 | PARTIAL | **PARTIAL** | `ppr_damping` comment corrected ("teleport probability is 1 - d", `iris/iris_config.py:42`); `clip_revision` configurable (`:101`). Remaining: `candidate_thresh` still not used by tier assignment; sparse edge-family constants not governed by `alpha/beta`; no legacy-retrieval weight-sum validation. |
| P2-06 | P2 | PARTIAL | **PARTIAL** | Empty caption now recorded as failure (`iris/aria.py:716-757`); V2 now has a simple `answer` field (P1-24). Remaining: timings still `time.time()` not `perf_counter()`; global failure/diagnostic lists still unbounded. |
| P2-07 | P2 | FIXED | **FIXED** | Absence verdict hard-codes honest bounded phrasing + frame counts (`iris/cerberus_layers.py:383+`). No regression. |
| P2-08 | P2 | PARTIAL | **PARTIAL** | spaCy no longer downloads at runtime — raises a clear `ImportError` with install instructions (`iris/cerberus_v.py:39-43`); NLI cache now keyed by model (`:48`); response query checked against input (`iris/query.py:552`, warns). Remaining: JSON extraction still custom brace scanning; some nested Absence/Global text fields lack runtime non-empty guards; query mismatch warns rather than rejects. |

---

## Part C — Gap questions

### C1 — `use_l1` default
**Answer: `use_l1` defaults to `False`** (`iris/iris_config.py:106`). L1 selection is fully
implemented (`_retrieve_with_l1`, `iris/query.py:584-669`) and wired into both canonical
query paths (`:694`, `:796`) and into `ingest()` cache population (`iris/ingest.py:423`),
but it is **inert under production defaults**. Consequence: the reaudit's P1-01 core
concern (L1 is a post-retrieval context tray, not a selector) is resolved *mechanically*
but not *by default*. **No new test required** — the wiring is exercised by existing
query tests with `use_l1` toggled. Action for the paper: either ship benchmarks with
`use_l1=True` (and demonstrate eviction/cache pressure), or drop any claim that L1
improves retrieval.

### C2 — Production-assert test coverage
**Answer: Present and real.** `_build_index_from_records()` contains always-on `assert`
statements in the production path (`iris/ingest.py:398-410`) verifying, for every frame,
that the `AsphodelNode` equals the `FrameRecord` on timestamp, luma_diff_energy,
luma_entropy, motion_magnitude, action_score, persistence_value, all five geometry
fields, and `scene_id`. This is exactly the "production-path assertion comparing
FrameRecord and AsphodelNode" the 07-19 doc required for P1-09/P1-12. Test coverage:
`tests/test_geom_fields.py` and `tests/test_e2e_integration.py::test_ingest_produces_valid_index`
drive this path and pass. **Caveat:** these are bare `assert`s and would be stripped
under `python -O` — recommend they not be relied on as the sole guard in an optimized
production run, but for benchmark runs (no `-O`) they are active. No new test required.

### C3 — Quadratic-complexity scope
**Answer: Substantially resolved for large graphs; exact matrix retained only for small
N.** Both edge builders now branch on size: semantic edges use FAISS `IndexFlatIP` ANN
for K≥200 (`iris/l2_asphodel.py:501/516-528`) and motion edges use `cKDTree` for N≥200
(`:583/595-605`), with a chunked O(N)-memory fallback when FAISS is absent. For N<200 an
exact `N×N`/`K×K` matrix is still built (`:501-514`, `:583-593`) — bounded and cheap.
**Scope statement for the paper:** build time is sub-quadratic *only on the large-N ANN
path*; for short videos it remains exact-quadratic-but-bounded. Claim "sparse storage +
ANN neighbor construction for large graphs," not universal sub-quadratic build.

### C4 — `graph_mode` default inconsistency
**Answer: Real inconsistency confirmed.** The dataclass defaults `graph_mode="scene_sparse"`
(`iris/iris_config.py:45`), but `configs/default_iris_config.json` contains only 5 keys
(`salient_thresh`, `candidate_thresh`, `alpha`, `beta`, `peak_order`) and **omits
`graph_mode` entirely**. Two downstream consequences:
1. `scene_sparse` requires `scene_id`-assigned records or it raises
   (`iris/ingest.py:105-109`); the synthetic/loaded-index path has no scene_id and
   silently needs `graph_mode="flat"` — which is why `walkthrough.md` had to patch
   `test_ingest.py`/`test_geom_fields.py`/`test_ppr_retrieve.py` to `flat`.
2. An index loaded with an empty/legacy `config_snapshot` effectively defaults to `flat`,
   diverging from the live dataclass default of `scene_sparse`.
**Recommendation (trivial, not applied — verification-only):** add `"graph_mode":
"scene_sparse"` to `default_iris_config.json` and document that `scene_sparse` requires a
scene-id-assigned ingest. No behavior code change needed.

---

## Part D — Environment blockers
*(No authoritative Part-D enumeration was available — the prompt file is absent. The
following are reconstructed from `MAIN_BRANCH_REAUDIT_2026-07-15.md` §10 "Missing/unlocked
dependencies" and the live config, and should be treated as best-effort.)*

1. **Answerer LLM server.** Default answerer is `llama_server` at
   `http://127.0.0.1:8091/v1` with model `granite4:micro` (`iris/iris_config.py:69-71`).
   A live llama-server with that model must be running for V2/answer generation; otherwise
   answer generation errors out. Model-backed answerer tests are skipped without it.
2. **spaCy `en_core_web_sm`.** Now a hard prerequisite — Cerberus raises `ImportError`
   with install instructions if the model is missing (`iris/cerberus_v.py:39-43`). Must be
   pre-installed (`python -m spacy download en_core_web_sm`); no runtime download.
3. **DeBERTa NLI + torch/transformers.** `_get_nli_model()` loads a DeBERTa NLI model
   (`iris/cerberus_v.py:46-57`), forced to CPU on Windows to avoid a CUDA access-violation
   segfault (`:55-56`). The model weights + transformers/torch must be present; this also
   makes NLI CPU-bound on Windows (latency, not correctness).
4. **Dataset provisioning.** NExT-GQA / VIRAT are referenced as git Gitlinks with **no
   `.gitmodules`** (reaudit P1-29), so a fresh clone cannot auto-fetch them; the dataset
   must be manually placed (see `eval/data/...`). PyAV/ffmpeg with H.264 MV export is also
   required for real-video ingest (the one real-video codec test skips without a fixture).

**Environment status:** the pure-Python + numpy/scipy/networkx test surface runs clean
(339 collected, 0 failed). The four items above are external provisioning steps, not code
defects; all are satisfiable on the target GPU benchmark box.

---

## Part E — Test-integrity check
*(No authoritative list of "the 4 test files" was available — prompt file absent. Checked
the 4 production-facing test files `walkthrough.md` reported modifying, plus the new e2e
test, for the 07-19 concern that tests might lock broken behavior.)*

| Test file | Integrity finding |
|---|---|
| `tests/test_e2e_integration.py` (new) | **Genuine.** Asserts required output keys incl. `answer`/`verified`, performs a real `json.dumps()` roundtrip and compares values (`:135-141`) — a true P1-24 guard, not a rubber stamp. |
| `tests/test_answerer_contract.py` | **Genuine.** Compares actual `core_verdict.to_dict()` output (`:340-341`); added `config_snapshot={"graph_mode":"scene_sparse"}` fixture. Does not lock old dataclass-return behavior. |
| `tests/test_l2_asphodel.py` | **Genuine.** Edge-weight assertion updated 0.0→0.6 to reflect the P1-09 6-D-cosine motion similarity — tracks the fix, not the bug. |
| `tests/test_ingest.py`, `tests/test_geom_fields.py`, `tests/test_ppr_retrieve.py` | **Acceptable.** Pinned to `graph_mode="flat"` because synthetic records carry no `scene_id` (the C4 inconsistency). This is a legitimate hermetic-fixture choice, not concealment — the scene_sparse path is separately exercised by `test_scene_sparse_descend.py` (all pass). |
| ARIA HTTP / `"model":"loaded"` lock (07-19 concern) | **Resolved.** No test asserts `"loaded"` anymore (repo-wide grep empty); the P1-17 fix removed the hard-code and the test was updated accordingly. |
| filtered_nli / ner_only behavior locks (07-19 concern) | **Resolved.** ner_only now returns unverifiable for all claims (`cerberus_v.py:442-445`) and tests pass against that, not the old entity-overlap acceptance. |

No test was found to lock a known-broken behavior. The suite reflects the fixes.

---

## Open items (carry into the benchmark run)

Ordered by benchmark impact. None are P0/crash; all are claim-scoping or inherent-design.

1. **P1-01 / C1 — L1 default off.** Decide L1's role. If claiming L1 benefits, run
   `use_l1=True` with real eviction and paired CIs; else remove the L1-improves-retrieval claim.
2. **P1-22 — visual verification override.** A weaker entailment still beats a stronger
   contradiction in `verify_visual_claim` (`cerberus_layers.py:250-252`). Either add a
   contradiction-defeats-entailment rule (as already done in `cerberus_v.py:405`) or scope
   the V2 visual-grounding verification claim.
3. **P1-15 — no production FAISS.** `L2TieredIndex` remains unwired; do not claim
   three-tier FAISS acceleration.
4. **P1-03 / P1-05 — wording.** "selective pixel processing" (not zero-decode); adaptive
   thresholds override configured ones.
5. **P2-05 / P2-06 / P2-08 — residuals.** Inert `candidate_thresh`; `alpha/beta` not
   governing sparse families; wall-clock timings; unbounded diagnostic lists; custom JSON
   brace scanning; query-mismatch warns not rejects. Low benchmark risk.
6. **C4 (trivial) — add `graph_mode` to `default_iris_config.json`** and document the
   scene_id prerequisite.
7. **C2 caveat — do not run the benchmark with `python -O`** (would strip the
   FrameRecord↔AsphodelNode production asserts).

**Bottom line:** the correctness blockers that drove the 2026-07-19 NO-GO are fixed and
regression-free (339 tests, 0 failures). Proceed to the benchmark with the claim-scoping
conditions above.
