# Appendix C item 39 resolution — backend/endpoint provenance across all artifacts

Read-only. Paper not edited.

## 0. Two mechanisms found, plus two edge cases — and a third affected artifact family beyond what item 39 named

Every artifact recording a backend, endpoint, port, model, or server field falls
into one of four buckets by **writer mechanism**, not by field name (per
instruction 2, field names were not trusted — each was traced to the code that
populates it):

| mechanism | reliability | artifacts |
|---|---|---|
| **(A) `asdict(config)` / config-snapshot dump** — serializes an `IRISConfig` instance never wired to the live answerer backend | **UNRELIABLE — proven false where checked** | `MLVU_codec_baseline.json`, `MLVU_ablation.json`, `MLVU_ablation_long.json`, `MLVU_ablation_long_trimmed.json`, `_mlvu_long_mini_smoketest.json`, `_MLVU_selftest.json` |
| **(B) harness-written provenance dict that introspects the live backend object + a liveness preflight** | **RELIABLE (mechanism verified)** | `pillar2_grounded_qa_raw.json`, `A6_allminmax_raw.json`, `A6_mixed_raw.json`, `P1_lambda00_raw.json`, `P1_lambda10_raw.json`, `P_NOWA_accgqa_raw.json` |
| **(C) OS-level process/binary introspection** (`scripts/answerer_provenance.py`: real PID lookup, `/proc/<pid>/cmdline`, SHA-256 of the actual binary file, live `/models` GET) | **RELIABLE (gold standard)** | `tuning/determinism_gate/environment.json`, `tuning/official_test/smoke_10/environment.json` + `resume_test_environment.json`, `tuning/blind_ablation/environment.json` (all `worker-1`/`ucf-vad-exp1`, none is a paper-reported number except via the A1 gate, already covered by item 22) |
| **(D) hand-written prose note, or absent field with a structurally-unambiguous hardcoded call site** | **RELIABLE where the call site has no override path at all; otherwise only as reliable as the prose is accurate** | `e2e_speedup.json` (prose note, verified accurate against source); `e2e_stage3_decomp_raw.json`, `caption_stage_diagnosis.json`/`.log` (no endpoint field recorded at all — traced to source instead, see §6) |

Beyond the four MLVU artifacts item 39 named, tracing mechanism (D) surfaced a
**third affected family not previously flagged**: the §5.3 end-to-end runs and
the §5.4 caption-stage diagnosis also ran through Ollama, not llama-server —
see §6. This bears directly on A.2's blanket "for all reported results other
than the MLVU family" answerer sentence.

## 1–2. Full inventory, fields verbatim, writer mechanism

### (A) `asdict(config)` dead-field family — confirmed false (per C38)

| artifact | fields (verbatim) | writer |
|---|---|---|
| `MLVU_codec_baseline.json` | `"answerer_backend": "llama_server"`, `"answerer_endpoint": "http://127.0.0.1:8091/v1"`, `"captioner_backend": "moondream"` | `asdict(config)` in `scripts/mlvu_eval.py`, where `config` is the `IRISConfig` built for the ingest pipeline |
| `MLVU_ablation.json` (×4 arm blocks) | same two fields, same values, at lines 111-112, 251-252, 391-392, 531-532 | `asdict(arm_config(arm))` in `scripts/mlvu_ablation.py` |
| `MLVU_ablation_long.json` | not written (run abandoned, per `mlvu_provenance_C26.md`) | — |
| `MLVU_ablation_long_trimmed.json` (×2 arm blocks) | same two fields, same values, at lines 73-74, 180-181 | `asdict(arm_config(arm))` in `scripts/mlvu_ablation_long_trimmed.py` |
| `_mlvu_long_mini_smoketest.json` (×4 blocks) | same two fields | `scripts/mlvu_eval.py` self-test/smoketest path — same dead-field mechanism, not a reported result |
| `_MLVU_selftest.json` | same two fields | `scripts/mlvu_eval.py` self-test path — installs `_MockMCBackend()` for the actual self-test query, so this field was never live-relevant even in principle |

**Writer trace (confirmed in the prior C38 audit, re-verified here):** all
four production MLVU scripts (`mlvu_eval.py`, `mlvu_ablation.py`,
`mlvu_ablation_long.py`, `mlvu_ablation_long_trimmed.py`) construct the real
answerer backend independently, via `aria.set_backend(aria.LlamaBackend(
endpoint=args.answerer_endpoint, ...))`, where `args.answerer_endpoint`
defaults to `http://localhost:11434/v1` and is never overridden at any
call site in the repository. Once `aria.set_backend()` runs,
`_BACKEND_OVERRIDDEN = True` (`iris/aria.py:628`) and `get_backend()` never
again consults `IRISConfig.answerer_backend`/`answerer_endpoint` — the fields
recorded in the `config` block come from a *different* `IRISConfig` instance,
built for the retrieval/ingest arm and passed through `asdict()`, structurally
incapable of reflecting the answerer that was actually wired. This is the same
class of failure the paper's own A.5.7 already documents for
`captioner_backend` in the identical `config` block (`eval_results/
captioner_provenance.json` independently confirms `captioner_backend` is dead
by the same `get_captioner()`-ignores-the-caller's-config mechanism, for a
different field on the same dataclass).

### (B) Live-backend-introspection family — mechanism verified reliable

`A6_allminmax_raw.json`, `A6_mixed_raw.json`, `P1_lambda00_raw.json`,
`P1_lambda10_raw.json`, and `pillar2_grounded_qa_raw.json` are, by git history,
the same output file (`pillar2_grounded_qa_raw.json`, always written under
that literal name by `scripts/pillar2_grounded_qa.py`) renamed after each run
(`git show bcf940e`: `pillar2_grounded_qa_raw.json => A6_mixed_raw.json`,
etc.) — one script, five labeled runs. `P_NOWA_accgqa_raw.json` is written by
`scripts/pnowa_test_accgqa_run.py`, which imports `preflight_backend`,
`git_provenance`, and `config_hash` directly from `pillar2_grounded_qa.py`
(`from scripts.pillar2_grounded_qa import (..., preflight_backend, ...)`) —
same mechanism, not merely similar code.

Verbatim fields (all six artifacts share this shape; values shown for
`A6_allminmax_raw.json`):

```json
"backend_class": "LlamaServerBackend",
"endpoint": "http://127.0.0.1:8091/v1",
"model": "granite4:micro",
"temperature": 0.0,
"cache_prompt": false
```

**Writer trace.** Both scripts:

1. Build `backend = aria.LlamaServerBackend(endpoint=cfg_proposed.answerer_endpoint, text_model=cfg_proposed.answerer_model)` — `cfg_proposed.answerer_endpoint` is not overridden in the `IRISConfig(...)` constructor call, so it resolves to the dataclass default `http://127.0.0.1:8091/v1`, but critically **this value is actually passed into the object that becomes the live backend**, not discarded.
2. `aria.set_backend(backend)` installs that exact object as the process's active backend (`_BACKEND_OVERRIDDEN = True`).
3. `preflight_backend(backend)` (`scripts/pillar2_grounded_qa.py:170-190`) runs immediately after, and **raises before any query is issued** if either check fails: `isinstance(backend, aria.LlamaServerBackend)` (rejects a silently-substituted Ollama `LlamaBackend`, with the rejection message naming that exact failure mode — "Ollama-backed LlamaBackend is the rejected runtime"), and a real `requests.get(f"{backend.endpoint}/models", timeout=5)` liveness probe against the literal endpoint that will be used.
4. The provenance block written to JSON reads `type(backend).__name__`, `backend.endpoint`, `backend.text_model`, `backend.temperature`, `backend.cache_prompt` — attributes read directly off the same live object that passed the preflight and served every subsequent `generate()` call, not re-declared literals and not a re-read of `cfg_proposed`.

This is categorically different from the MLVU (A) family: there, the recorded
field and the live backend are two unrelated objects; here, the recorded
field *is* an attribute of the live backend object, checked for liveness
before the run proceeded. A run that completed and wrote this JSON could only
have done so if something was actually listening at `127.0.0.1:8091` and
answered `/models` successfully — the alternative (nothing listening, or the
wrong backend class) is a `RuntimeError` before any question is asked, not a
silently-wrong artifact.

**What this does and does not establish.** It establishes that a real,
OpenAI-compatible server was live at 8091 and that `granite4:micro`,
temperature 0, and `cache_prompt=False` were genuinely included in the
requests `LlamaServerBackend.generate()` sends (confirmed from
`iris/aria.py`: `self.cache_prompt` is sent as a literal request field / via
`extra_body` in all three of that class's request branches, unlike
`LlamaBackend`, which never sends it — see `long_arm_determinism_C38.md`
§2). It does **not** establish which llama-server *build* was listening
(that is item 22's separate, already-disclosed question — no SHA-256 or
`--version` capture exists for any of these six runs) and does not itself
confirm `--parallel 1` was set at launch (a launch-time flag, invisible to a
per-request liveness probe).

### (C) OS-level introspection family — gold standard, already reliable, not newly affected

`tuning/determinism_gate/environment.json`, `tuning/official_test/smoke_10/
environment.json` and its `resume_test_environment.json` sibling, and
`tuning/blind_ablation/environment.json` (all on `worker-1` /
`origin/siddanth/ucf-vad-exp1`) are populated by `scripts/answerer_provenance.py`
(`capture_answerer_provenance()`), which:

- finds the actual PID listening on the target port via `ss -ltnp`,
- reads that PID's real `/proc/<pid>/cmdline`,
- computes SHA-256 over the **actual binary file on disk** at the resolved path,
- runs `<binary> --version` and captures the output,
- performs a live `GET {endpoint}/models`,
- and records `sampler_params_sent` as "exactly the kwargs the caller sent on the
  request... not re-derived here" (module docstring/comment, `scripts/
  answerer_provenance.py:104-107`).

Every field here traces to the operating system's own view of the running
process and the file on disk, not to any `IRISConfig` object at all. This
mechanism is unrelated to, and unaffected by, the MLVU dead-field defect.
None of these four files backs a number reported in the paper directly — they
back the A1 determinism gate and adjacent infrastructure runs on `worker-1`,
already covered by A.2/item 22 — but they are the reference point against
which (A) and (B) above are being compared, and they confirm (C) is a
structurally distinct, higher-assurance mechanism.

### (D) Prose note / no field — traced to source instead

| artifact | field or its absence | resolution |
|---|---|---|
| `e2e_speedup.json` | `"answerer_backend_override"`: a full prose sentence stating `iris.aria.set_backend(iris.aria.LlamaBackend(text_model='granite4:micro'))` was called, with no endpoint argument | Verified accurate against `scripts/e2e_speedup_ab.py` directly (see §6) — this is the one artifact in the whole inventory that **correctly self-reports** using Ollama, in prose, rather than asserting llama-server. |
| `e2e_stage3_decomp_raw.json` | records `"answerer_model": "granite4:micro"` only — **no backend/endpoint field of any kind** | Traced to `scripts/e2e_stage3_decomp_ab.py:150`: `aria.set_backend(aria.LlamaBackend(text_model=ANSWERER_MODEL))`. No CLI argument for endpoint exists anywhere in this script — `LlamaBackend`'s constructor default (`http://localhost:11434/v1`) is the *only* value that could ever have been used; there is no override path even in principle. |
| `caption_stage_diagnosis.json` / `caption_stage_diagnosis_stdout.log` | no backend/endpoint field in the JSON; the log prints only `"[checkpoint] set_backend done"` (no endpoint named) | Traced to `scripts/caption_stage_diagnosis.py:316`: `aria.set_backend(aria.LlamaBackend(text_model=ANSWERER_MODEL))`. Same as above — no argparse, no endpoint override possible. |
| `caption_benchmark_blip.json`, `caption_benchmark_moondream.json` | `"model"` field only, naming the captioner under test | Single-purpose captioner-timing benchmarks; no answerer backend/endpoint ambiguity, out of scope for this item |

## 3. `P_NOWA_accgqa_raw.json` specifically

**Confirmed: mechanism (B), not a config dump.** Traced directly:
`scripts/pnowa_test_accgqa_run.py:148-153` builds `backend =
aria.LlamaServerBackend(endpoint=cfg_proposed.answerer_endpoint,
text_model=cfg_proposed.answerer_model)`, calls `aria.set_backend(backend)`,
then `preflight_backend(backend)` (imported from `pillar2_grounded_qa.py`,
same function described in §2 above — `isinstance` check + live `GET
{endpoint}/models`). The `provenance` dict at write time
(`pnowa_test_accgqa_run.py:326-331`) reads `type(backend).__name__`,
`backend.endpoint`, `backend.text_model`, `backend.temperature`,
`backend.cache_prompt` off that same object.

**This does query the live backend, and it does so before the held-out
question set is ever answered** — the isinstance+liveness preflight raises
before any query if the wrong backend class is active or nothing answers at
`/models`. A.2's §6.1 claim (held-out P-NOW-A Acc@GQA served via
llama-server) is **not** the same defect as the MLVU family and does **not**
need revisiting on these grounds. It remains subject to the separately
disclosed item 22 gap (no SHA-256/`--version` capture for this specific run,
so the *build* behind `b9976` is still unverified) — that gap is orthogonal
to, and predates, this item.

## 4. Cross-checks against independent evidence

| artifact family | independent evidence checked | result |
|---|---|---|
| MLVU ×4 (mechanism A) | each script's own printed `[setup] answerer backend: LlamaBackend(endpoint=...)` line, present in every surviving stdout log | **Contradicts** the recorded `config` block. Confirmed FALSE (`long_arm_determinism_C38.md`). |
| `pillar2_grounded_qa_raw.json`/`A6_*`/`P1_*`/`P_NOWA_accgqa_raw.json` (mechanism B) | `preflight_backend`'s live `isinstance` + `GET /models` check, which must have passed for the run to have produced output at all; no separate stdout `[setup]` line was found for these specific runs beyond the code's own `print(f"[LLM] Seating ...")` line (present in source, not confirmed in a surviving log for these particular runs) | **Agrees** with the recorded field, on the strength of the preflight gate rather than an independently-recovered log. Treated as corroborated, not gold-standard-verified (no binary hash). |
| `tuning/determinism_gate/environment.json` + siblings (mechanism C) | the file's own `cmdline`/`binary_sha256`/`binary_version_output` fields, cross-referenced in `llama_build_C22.md` against a second, independently-computed SHA-256 of the same destination path | **Agrees**, independently double-checked in a prior audit (`llama_build_C22.md`: "Destination SHA-256... match"). |
| `e2e_speedup.json` (mechanism D, prose) | source of `scripts/e2e_speedup_ab.py` | **Agrees** — the prose note accurately describes the source. |
| `e2e_stage3_decomp_raw.json`, `caption_stage_diagnosis.json` (mechanism D, absent field) | source of `scripts/e2e_stage3_decomp_ab.py` and `scripts/caption_stage_diagnosis.py` | No field to check *against* — but the source has no override path at all, so **the answer is unambiguous from the source alone**: both used Ollama at 11434. Reported as such, not treated as "confirmed by the artifact" (the artifact says nothing). |

Where no independent evidence exists at all — none was found for this
inventory; every field traced either agreed, disagreed, or (for the two
no-field cases) was settled unambiguously by the absence of any override
path in the calling code.

## 5. Does this affect the determinism-gate `environment.json`?

**No.** Its mechanism (C) — OS-level PID lookup, `/proc/<pid>/cmdline`,
SHA-256 of the actual binary file, live `/models` GET — shares no code path
with `asdict(IRISConfig)` and does not depend on any script correctly wiring
a config field to a backend constructor. It queries the operating system and
the filesystem directly, after the server is already running, rather than
asking a Python config object what it thinks should be running. A.2 can cite
it with confidence for what it claims (the specific `b10099`/`1a064ab`
binary, hash-verified, served the A1 gate on `worker-1`) — the open
question about that file, per the paper's own A.2, is only that its
demonstration doesn't transfer to a different machine or a different backend
class, not that the file's own contents might be wrong.

## 6. New finding: the §5.3 and §5.4 Windows-machine runs also used Ollama

Not asked for by name in item 39, but surfaced by tracing mechanism (D)
artifacts per instruction 1's "every artifact" scope, and material to
instruction 4's cross-check:

- **§5.3, both end-to-end runs.** `e2e_speedup.json` (the retained 0.778×
  prior measurement) explicitly and correctly self-reports
  `aria.set_backend(aria.LlamaBackend(text_model='granite4:micro'))` — Ollama,
  port 11434 by the class default, no override. `e2e_stage3_decomp_raw.json`
  (the promoted 0.802× primary measurement, commit `deeeba8`) records no
  backend/endpoint field at all, but its harness
  (`scripts/e2e_stage3_decomp_ab.py:150`) contains the identical call with no
  CLI override path — structurally, it cannot have used anything but Ollama
  either.
- **§5.4, caption-stage diagnosis.** `scripts/caption_stage_diagnosis.py:316`
  is the same call, same absence of an override path. `caption_stage_diagnosis.json`
  records no backend/endpoint field; its stdout log confirms `set_backend` ran
  but does not name the endpoint.

None of these three runs' recorded artifacts *asserts* llama-server the way
the MLVU family's dead `config` block does — two say nothing, one correctly
says Ollama — so this is not a second instance of false provenance. But it
does mean A.2's answerer sentence, as it currently reads ("served via
llama-server, temperature 0, `cache_prompt=false`, `--parallel 1`, for all
reported results other than the MLVU family"), is not accurate: §5.3 and §5.4
are also outside that description. This is a scoping gap for the team to
close in A.2, not resolved here (this task is read-only).

## Answer

**VERIFIED backend provenance** (mechanism (B) or (C), independently
corroborated):

- `A6_allminmax_raw.json`, `A6_mixed_raw.json`, `P1_lambda00_raw.json`,
  `P1_lambda10_raw.json`, `pillar2_grounded_qa_raw.json`, `P_NOWA_accgqa_raw.json`
  — `LlamaServerBackend` at `127.0.0.1:8091`, temperature 0, `cache_prompt=false`
  sent, corroborated by a live preflight gate that would have aborted the run
  otherwise. (The llama-server *build* behind these remains item 22's separate,
  already-disclosed open question.)
- `tuning/determinism_gate/environment.json` and its `worker-1` siblings —
  hash-verified binary, real cmdline, live `/models` response. Gold standard.

**UNCORROBORATED but not contradicted** (mechanism (D), no independent
artifact-level check possible, settled instead by an unambiguous source
trace):

- `e2e_speedup.json` — self-reports Ollama accurately; this is corroborated,
  not merely uncorroborated, but note it separately because it is the one
  self-report that turned out to be honest.
- `e2e_stage3_decomp_raw.json`, `caption_stage_diagnosis.json` — no field
  recorded at all; determined from source to be Ollama, with no override path
  possible, so this is as settled as (B)-level corroboration would be, just
  via source instead of a runtime log.

**Provenance now known to be FALSE:**

- `MLVU_codec_baseline.json`, `MLVU_ablation.json`, `MLVU_ablation_long.json`
  (never completed), `MLVU_ablation_long_trimmed.json` — all record
  `"llama_server"`/8091 via a dead `asdict(config)` field, all actually served
  by Ollama at 11434, confirmed by each run's own contradicting stdout line.

**Which of A.2's current claims survive:**

- The determinism demonstration's citation of `worker-1`/`b10099`/
  `environment.json` — **survives**, mechanism (C) is unaffected.
- "The answerer's determinism assurance... does not extend to §6.2 or §7.1"
  — **survives**, and is now additionally supported by this audit rather than
  only the prior one.
- §6.1's held-out P-NOW-A result being served by `LlamaServerBackend` at
  8091 — **survives**, mechanism (B) with a live preflight, not the MLVU
  defect; item 22's build-identity gap is unaffected either way.
- "Answerer: `granite4:micro`... served via llama-server... for all reported
  results other than the MLVU family" — **does not survive as written**. §5.3
  (both runs) and §5.4 also used Ollama's `LlamaBackend`, not llama-server,
  with no override possible in either script. This sentence needs to name
  §5.3 and §5.4 alongside §6.2/§7.1, or be reworded to state which sections
  *are* confirmed llama-server (§6.1's A6/P-NOW-A/P1 family) rather than
  which are the exception.
