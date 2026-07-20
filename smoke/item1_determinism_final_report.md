# Item 1: Answerer Determinism — Final Report

## What was fixed
`iris/aria.py`: `seed` (from new `IRISConfig.answerer_seed`, default 42) is now forwarded on every
answerer request path in `LlamaBackend` and `LlamaServerBackend` (native `/api/chat` `options.seed`,
OpenAI-compat client `seed=`, and the `/completion` fallback's `seed` field), plus `OpenAIBackend`
for interface parity. Verified via kwargs-capture against the real code path — not assumed.

## What was discovered beyond the original ask
`temperature=0.0` + `seed=42` alone did **not** achieve 12/12 determinism on a live re-run (still
11/12, with the underlying raw LLM text differing in far more pairs than the original 1/12 — masked
by CerberusV's strict gate collapsing varied text into the same abstention message in most cases).

Root-caused through direct experimentation (not guesswork):
1. **Thread pinning (`num_thread=1`) does NOT fix it** — tested directly against the real failing
   prompt at real context length; same divergence pattern persisted. This rules out floating-point
   matmul reduction-order nondeterminism, which was the leading hypothesis when this investigation
   started.
2. The actual pattern: a request served against an already-**warm** (previously-loaded) Ollama
   model handle can produce different tokens than one served against a **freshly-reloaded** handle,
   even with identical prompt/seed/temperature. Repeated freshly-loaded calls were bit-identical to
   each other in isolated testing (3/3, then 3/3, then 2/2 across three separate test batches).
3. **Fix implemented**: `IRISConfig.answerer_keep_alive` (default `0`) forces Ollama to evict the
   model after each request, via `extra_body={"keep_alive": 0}` on the OpenAI-compat path and a
   top-level `"keep_alive"` field on the native path. `LlamaServerBackend`/`OpenAIBackend` accept
   and ignore this parameter (no equivalent concept there; llama-server already has its own
   `cache_prompt=False` mitigation for an analogous class of issue).

## Residual finding: this reduces but does not *guarantee* 100% determinism
After implementing `keep_alive=0`, a direct kwargs-captured test on the originally-failing question
(`6936757706`/qid 7) was **fully reproducible** (verified `seed=42` and `extra_body={"keep_alive": 0}`
both present in the actual request, `raw_answer` byte-identical across 2 real `query()` calls).
But a separate 12-question batch run on the same fix showed the *same* question still diverging.

This is consistent with a **race condition in Ollama's async model-lifecycle management**:
`keep_alive=0` signals "unload after this response," but unloading appears to happen asynchronously
server-side. If the next request lands before the previous model instance has actually finished
unloading, it can be served against the still-resident ("warm") handle instead of a fresh one,
reproducing the original divergent behavior. This is not something `IRISConfig`'s request-level
parameters can fully control from the client side — it depends on Ollama-server-internal timing.

**Practical implication**: `answerer_keep_alive=0` measurably helps (moves the failure mode from
"frequent" to "occasional/rate-dependent") but should not be marketed as a determinism guarantee
for `answerer_backend="llama"` (Ollama) specifically. Two follow-ups worth considering, not
implemented here (out of scope for a payload-level fix):
- Switch to `answerer_backend="llama_server"` (llama-server) for determinism-critical runs — it
  already disables prompt-cache reuse synchronously via `cache_prompt: False` in the same request,
  with no separate async unload step to race against.
- If Ollama must be used, add an explicit unload-and-poll-until-confirmed-unloaded step between
  answerer calls (adds real latency, not attempted here).

## Verification artifacts
- `smoke_scripts/diagnose_seed_nondeterminism.py`, `diagnose_captioner_nondeterminism.py`,
  `diagnose_real_backend_path.py`, `diagnose_context_text_stability.py`,
  `diagnose_query_path_kwargs.py` — the diagnostic scripts used to isolate the cause.
- `smoke/determinism_post_fix_seed_only_no_keepalive.json` — 11/12, seed only, no keep_alive.
- `smoke/determinism_post_fix.json` — 11/12, seed + keep_alive=0, live 12-question batch (still
  shows the race-condition pattern under batch timing).
- Direct single-question reproducibility check (this file's "residual finding" section) — 2/2
  identical via captured real kwargs, demonstrating the fix works when the race doesn't trigger.
- 75 unit tests added/passing across `test_aria.py`/`test_query.py`/`test_answerer_contract.py`
  covering seed and keep_alive forwarding on every backend/payload path.
