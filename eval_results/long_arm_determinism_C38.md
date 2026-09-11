# Appendix C item 38 resolution — long-video arm serving path and determinism

Read-only. Paper not edited.

## 0. Headline correction to item 38's own framing

Item 38 states that `MLVU_codec_baseline.json` and `MLVU_ablation.json`
"recorded `http://127.0.0.1:8091/v1`," implying that field reflects the
server that actually answered those runs' questions. **It does not, for any
of the four MLVU artifacts — including `MLVU_ablation_long_trimmed.json`,
which the item's own source (`mlvu_provenance_C26.md`) already correctly
identified as Ollama-served.** `MLVU_ablation_long_trimmed.json` itself
carries `"answerer_backend": "llama_server"` / `"answerer_endpoint":
"http://127.0.0.1:8091/v1"` in its own `config` block (lines 73-74, 180-181)
— the exact same field the item treats as proof of 8091 for the other two
artifacts, on an artifact everyone agrees ran through Ollama on 11434
instead. That field is not run telemetry; see §1.

## 1. Which server actually served §7.1's long-video run — and the other three MLVU artifacts

All four MLVU harness scripts (`mlvu_eval.py` → `MLVU_codec_baseline.json`,
`mlvu_ablation.py` → `MLVU_ablation.json`, `mlvu_ablation_long.py` →
`MLVU_ablation_long*`, `mlvu_ablation_long_trimmed.py` →
`MLVU_ablation_long_trimmed.json`) share identical structure at the call
site:

```python
ap.add_argument("--answerer-endpoint", type=str, default="http://localhost:11434/v1")
...
aria.set_backend(aria.LlamaBackend(endpoint=args.answerer_endpoint, text_model=args.answerer_model))
print(f"[setup] answerer backend: LlamaBackend(endpoint={args.answerer_endpoint!r}, ...")
```

`aria.LlamaBackend` (`iris/aria.py:348`) is documented in its own docstring
as *"Local LLM via Ollama (OpenAI-compatible endpoint at localhost:11434)"*
— it is not a generic OpenAI-compatible client that happens to default to
that port; it is the Ollama-specific backend class in this codebase, as
distinct from `LlamaServerBackend` (`iris/aria.py:428`, *"Local LLM via
llama-server OpenAI-compatible endpoint"*, default port 8080, the class that
carries `cache_prompt`/`temperature` attributes and is used elsewhere in the
pipeline).

**No wrapper script overrides `--answerer-endpoint` anywhere in the repo.**
Checked every invocation site: `scripts/_mlvu_ablation_run_with_retry.sh`
(mlvu_ablation.py), `scripts/_mlvu_ablation_long_run_with_retry.sh`
(mlvu_ablation_long.py), `scripts/_mlvu_ablation_long_watchdog.sh` and
`scripts/_mlvu_ablation_long_trimmed_watchdog.sh` (both relaunch paths for
the long/trimmed scripts) — none passes `--answerer-endpoint`. A default in
source is not proof of what ran, so the stdout logs were checked directly
for the printed `[setup]` line, which every script prints unconditionally
before any query is issued:

| artifact | source log(s) | printed endpoint |
|---|---|---|
| `MLVU_codec_baseline.json` | `_mlvu_real_run_stdout.log:8` (identified as this run by `mlvu_provenance_C26.md` via matching `m_avg: 0.34` and HEAD string) | `http://localhost:11434/v1` |
| `MLVU_ablation.json` | `_mlvu_ablation_run_stdout.log:8` | `http://localhost:11434/v1` |
| `MLVU_ablation_long.json` (abandoned/incomplete — see C26) | `MLVU_ablation_long_stdout.log:1`, `MLVU_ablation_long_shard0_stdout.log:1,315`, `MLVU_ablation_long_shard1_stdout.log:1,36,62` | `http://localhost:11434/v1` |
| `MLVU_ablation_long_trimmed.json` | `MLVU_ablation_long_trimmed_shard0_stdout.log:1,65,364`, `MLVU_ablation_long_trimmed_shard1_stdout.log:1,21,56,75,345` | `http://localhost:11434/v1` |

**All four ran through Ollama's daemon on port 11434, not through a
standalone llama-server on 8091.** §7.1's long-video arm is not a special
case — it is one of four MLVU artifacts that all took the same serving
path, unmodified from source default.

The `"answerer_backend": "llama_server"` / `"answerer_endpoint":
"http://127.0.0.1:8091/v1"` fields recorded inside these JSON files'
`config` blocks come from a different, unrelated source: `asdict(config)`
where `config` is an `IRISConfig` dataclass instance (`iris/iris_config.py:81-82`)
constructed for the retrieval/ingest arm pipeline, carrying its own
class-level defaults (`answerer_backend: str = "llama_server"`,
`answerer_endpoint: str = "http://127.0.0.1:8091/v1"`). This is a genuinely
separate code path from the answerer LLM backend, which is wired
independently via `aria.set_backend(...)` in each script's `main()`. Once
`set_backend()` is called, `aria.py`'s global dispatcher
(`_BACKEND_OVERRIDDEN = True`, `get_backend()` at `iris/aria.py:592-628`)
never consults `IRISConfig.answerer_backend`/`answerer_endpoint` again for
the remainder of the process — the `IRISConfig` fields are inert for these
four scripts regardless of their value. This is the same class of dead-field
problem the paper's own A.5.7 already documents for `captioner_backend` in
this identical `config` block (`mlvu_provenance_C26.md` §1 notes it
explicitly for that field, but did not extend the same skepticism to
`answerer_endpoint` sitting right next to it) — corroborating evidence:
`ollama list` (per `mlvu_provenance_C26.md` §3) shows `granite4:micro`
actually installed under Ollama's own registry naming, and all four runs
produced complete, parseable, non-error result sets (matching expected
question counts), consistent with a live, correctly-answering Ollama daemon
at 11434 rather than a misconfigured or silently-failing endpoint.

**Conclusion: the recorded 8091/`llama_server` fields in `MLVU_codec_baseline.json`
and `MLVU_ablation.json` do not establish, and are contradicted by, the
actual serving path for those two runs.** Both ran through Ollama at 11434,
exactly like the long-video arms. There is no MLVU artifact in this family
that is actually confirmed to have been served by a standalone llama-server
on 8091.

## 2. Did `cache_prompt=false` hold?

**No artifact's request path ever sends a `cache_prompt` parameter at all**
for any of the four MLVU runs. `aria.LlamaBackend.generate()`
(`iris/aria.py:366-425`) has three request branches (native `/api/chat` for
`schema_format=True`, `client.chat.completions.create` with
`response_format`, and the plain default path) — none of the three
constructs a payload containing `cache_prompt` in any form (not as a
top-level key, not in `extra_body`, not in `options`). This is unlike
`LlamaServerBackend`, which has `self.cache_prompt = False` as an instance
attribute and sends it explicitly via `extra_body={"cache_prompt":
self.cache_prompt}` or as a top-level payload key in its native fallback —
that class is not what any MLVU script uses.

This is a stronger and more specific finding than the A1 gate's warning that
"Ollama's bundled llama-server... drops `cache_prompt=false`," which
describes a scenario where the parameter *is sent* to Ollama's internally
vendored llama-server binary (`/usr/local/lib/ollama/llama-server`) and
silently ignored server-side. That scenario does not describe what happened
here: the MLVU scripts talk to Ollama's own daemon API (port 11434) using a
client class that never attempts to send `cache_prompt` in the first place.
Whichever internal binary Ollama's daemon shells out to, and whether that
binary would honor or drop the parameter if it somehow arrived, is moot —
it never arrived. **The condition was not exercised, for any of the four
MLVU artifacts, not held or violated but simply absent from the wire
contract.**

## 3. Were temperature 0 and `--parallel 1` in force?

**Temperature 0: sent, for all four artifacts.** `aria.LlamaBackend.generate()`
hardcodes `temperature=0.0` in both the `chat.completions.create` branches
(response_format and default) and `"options": {"temperature": 0.0}` in the
native `schema_format` branch — there is no code path in this class that
sends any other value or omits it. This holds uniformly across
`MLVU_codec_baseline.json`, `MLVU_ablation.json`, and both long-video
scripts, since all four use the identical `LlamaBackend` class.

**`--parallel 1`: cannot be determined, for any of the four artifacts —
not only the long-video arm.** `--parallel` is a `llama-server`
command-line launch flag controlling concurrent inference slots; it has no
equivalent in Ollama's client-facing request contract, and Ollama's own
concurrency is instead governed by daemon-level environment variables
(`OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`) that are set when the
Ollama service itself is started, not per-request and not from any of these
harness scripts. No artifact in this repository records how the Ollama
daemon serving any MLVU run was launched or what those environment
variables were set to. This was already undeterminable for the two
"8091" artifacts under the old (incorrect) assumption that they used
llama-server with an explicit `--parallel 1`; under the corrected finding
that all four used Ollama, it is undeterminable for all four uniformly, and
there is no reason to assume single-slot serving was in effect — an
undetermined daemon config, not a verified `--parallel 1`.

## 4. Blast radius

The serving-path question is **not scoped to §7.1's long-video arm alone.**
It applies identically to all four MLVU artifacts: `MLVU_codec_baseline.json`
(§6 headline M-Avg 0.340), `MLVU_ablation.json` (§7 four-arm segmentation
table), `MLVU_ablation_long.json` (abandoned, no final report written — see
`mlvu_provenance_C26.md` §0/§3), and `MLVU_ablation_long_trimmed.json` (§7
long-video delta). Item 38's premise that the "other two" used 8091 does not
survive checking the actual stdout logs — it survived only as long as
nobody read past the `config` block's `IRISConfig`-default fields to the
`[setup]` line each script actually prints.

## 5. Evidence bearing on whether the long-video run was actually deterministic

None exists, in either direction. Checked every long-video checkpoint file
directly for repeated `(arm, task, question_id)` triples that would
constitute a natural repeat (e.g. from a crash-and-resume re-answering a
question already checkpointed) — there are none:

| checkpoint | n_results | unique (arm,task,qid) keys | duplicate keys |
|---|---|---|---|
| `MLVU_ablation_long_checkpoint.json` | 108 | 108 | 0 |
| `MLVU_ablation_long_checkpoint.shard0.json` | 64 | 64 | 0 |
| `MLVU_ablation_long_checkpoint.shard1.json` | 0 | 0 | 0 |
| `MLVU_ablation_long_trimmed_checkpoint.shard0.json` | 104 | 104 | 0 |
| `MLVU_ablation_long_trimmed_checkpoint.shard1.json` | 112 | 112 | 0 |

Every question was answered exactly once by design (the checkpoint-and-skip
resume logic means a crash never re-asks a question already saved). There is
no overlapping question set between arms/tasks that should have reproduced
another and can be checked, and the full 175-video run was abandoned before
completion (`mlvu_provenance_C26.md` §0/§3), so no independent full-vs-trimmed
comparison exists either. The A1 determinism gate's 639/639 byte-identical
result provides no transferable evidence here: it ran on a different
machine (`worker-1`), a different dataset (not MLVU), a different backend
class (`LlamaServerBackend`, direct pinned `b10099` binary), and explicitly
did not exercise the Ollama path at all — its own recommendation names
Ollama's route as the one to avoid, not one it validated.

## Answer

**Yes — A.2's determinism claim must be scoped to exclude the entire MLVU
family (§6.2, §7.1), not only the long-video arm.** All four MLVU artifacts
were served by Ollama at `http://localhost:11434/v1` via `aria.LlamaBackend`,
confirmed directly from each run's own printed `[setup]` line, with no
call-site override anywhere in the repository. None of the three conditions
A.2's demonstrated determinism rests on can be carried over to this serving
path: `cache_prompt` is never sent by the client code used here (not
"dropped," simply absent from the request); `--parallel 1` has no Ollama
equivalent and was never recorded; and the one existing byte-identity
demonstration (the A1 gate) ran a different backend class, on a different
machine, on a different dataset, and explicitly frames the Ollama route as
the thing to avoid rather than something it verified. Temperature 0 is the
one condition that *is* confirmed sent, uniformly, to all four MLVU runs.

**Proposed scoping wording for A.2:**

> "The determinism demonstration in this section (639/639 byte-identical
> answers under `cache_prompt=false`, `--parallel 1`, temperature 0) applies
> only to the run and machine it was measured on (`worker-1`, `b10099`,
> `LlamaServerBackend`). It does not extend to the MLVU results in §6.2 and
> §7.1 (`MLVU_codec_baseline.json`, `MLVU_ablation.json`,
> `MLVU_ablation_long_trimmed.json`), which were served by Ollama
> (`aria.LlamaBackend`, `http://localhost:11434/v1`) rather than
> llama-server. That serving path sends temperature 0 but never sends a
> `cache_prompt` parameter at all, and its concurrency configuration
> (Ollama's `OLLAMA_NUM_PARALLEL`) was not recorded for any of these runs.
> No repeat-run evidence exists for any MLVU artifact, so whether these
> specific results are reproducible under re-query is undetermined, not
> confirmed either way."

This is deliberately broader than item 38's own framing (which scoped the
open question to "the long-video arm"): the artifact evidence does not
support drawing the line there. If the team wants a narrower carve-out
instead of excluding all of §6.2/§7.1, that would require re-running at
least one of the four MLVU arms against a pinned, verified llama-server
build with `cache_prompt`/`--parallel` actually exercised and captured
(e.g. via `scripts/answerer_provenance.py`, the tool already built for this
purpose) — nothing in the current repository substitutes for that.
