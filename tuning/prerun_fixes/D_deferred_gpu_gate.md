# Deferred: A1 determinism verification (GPU box only)

**This check was explicitly NOT run as part of this fix pack.** This task ran
CPU-only, on a Windows box that has no compiled llama-server binary and no
CUDA GPU (see `A2_binary_inventory.json`). CPU llama.cpp is substantially more
deterministic than CUDA (no cross-thread floating-point reduction-order
nondeterminism from GPU kernels), so a CPU-side pass through this check would
be **falsely reassuring** — it could pass on CPU while the real GPU serving
path (which is what the official run actually uses) still diverges. Per the
task instructions for Part D, do not attempt it here.

## What must be run instead, on the GPU box, before the official test run

1. **Serving command.** Start llama-server with:
   - The pinned binary whose provenance is recorded via
     `scripts/answerer_provenance.py` (added in this fix pack) — i.e. capture
     and check in its SHA-256 and `--version` output *before* trusting a run
     against it.
   - `--parallel 1` (no concurrent request slots).
   - No continuous batching (do not pass `--cont-batching`, or pass the
     equivalent disable flag for whatever llama-server version is in use — the
     goal is one request fully served before the next begins, eliminating
     cross-request batching as a nondeterminism source).
   - See `exact_commands.txt` for the literal command template.

2. **Sampler parameters**, sent on every request (now pinned in `iris/aria.py`
   by A1 in this fix pack — verify the served requests actually carry them,
   don't just trust the code):
   ```
   temperature = 0
   top_k = 1
   top_p = 1.0
   seed = 42
   cache_prompt = false
   ```

3. **Frozen captions.** Run the answer stage using `--caption-load` against a
   single frozen `captions_dump.json` (from this fix pack's C0 work) so the
   captioner is not a second source of variance — this check is specifically
   about the *answerer's* determinism, not the captioner's.

4. **The check itself.** Run the answer stage **twice**, back-to-back, over
   the same frozen captions_dump.json, same retrieval, same 639 val_confirm
   questions (or the official 5553 at official scale, but 639 is sufficient
   to gate on before committing to the full run). Use
   `scripts/val_confirm_e2e_eval.py --caption-load <dump> --repeat 2` (H2, this
   fix pack) to get both passes plus a flip count in one invocation.

5. **Pass condition.** Raw answer strings byte-identical on **639/639**
   questions (0 flips reported by `--repeat`'s summary). Anything less is a
   **fail**, and the official run **does not proceed** until the divergence
   source is found and fixed — do not average over it, do not treat a small
   flip count as "close enough." The whole point of this gate is that the
   prior reproduction attempt found 118/639 (18.5%) flips and the two
   uncontrolled variables behind it (different binary, unpersisted captions)
   are exactly what this fix pack's A2/C0 changes are meant to close off. If
   flips remain after both are controlled for, that points to a third,
   still-unidentified source (batching, kernel nondeterminism even at
   `--parallel 1`, etc.) that must be root-caused before spending official-run
   budget on 990 videos / 5553 questions.

## Why this can't be simulated here

There is no CUDA device, no compiled llama-server binary, and no GGUF model
file on this box (confirmed in `A2_binary_inventory.json`) — there is
nothing to serve requests against. This is a specification for the GPU box
to execute, not a result.
