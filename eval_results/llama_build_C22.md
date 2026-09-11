# llama-server build-identifier audit — C22

Search scope: full text search (`git grep`) across the tip of every ref in
`refs/heads` and `refs/remotes` (32 refs, both remotes `origin` and `swara`),
the working tree, and `.claude/worktrees/ucf-vad-exp1` (which is just a
checkout of `siddanth/ucf-vad-exp1`, already covered by the ref search); plus
`git log --all --grep` over commit messages. Pattern: `\bb[0-9]{3,6}\b`
filtered to lines mentioning `llama`/`answerer`/`server`, and a plain search
for the known `b9976`. No paper or artifact files were edited.

## Distinct build strings found

Three, not one:

| build string | first seen | status |
|---|---|---|
| **b9976** | `DECISIONS.md` (2026-07-13 seating decision) | Self-reported in run-log prose only. **No binary hash, `--version` output, or environment-capture artifact anywhere in the repo ties any actual llama-server process to this build.** |
| **b10099** (commit `1a064ab`) | `tuning/prerun_fixes/` / `tuning/determinism_gate/` (siddanth/ucf-vad-exp1 branch, 2026-07-24 – 2026-07-27) | **Independently verified**: SHA-256 of the binary, `--version` output, full `cmdline`, and GGUF path captured in `tuning/determinism_gate/environment.json` / `pinned_binary.json`. Confirmed by the repo's own later audit to be **gone** from every reachable box as of 2026-07-27. |
| **b9985** (`b9985-1-g91c631b21`) | `tuning/prerun_fixes/A2_binary_inventory.json`, `final_report.md` (siddanth/ucf-vad-exp1, 2026-07-27 Windows-box audit) | The `llama.cpp` **source** checkout's current HEAD at audit time — never compiled (`is_compiled: false`), not a served binary. Listed only to show it is *not* b9976 either, and doesn't even contain the commit (`1a064ab`) that b10099 was built from. |

## Files and the runs they describe

| file | branch(es) | run date | build cited | what the run measured |
|---|---|---|---|---|
| `DECISIONS.md` (lines 6, 65) | main, all branches (universal, 25+ refs) | decision recorded 2026-07-13 | b9976 | Answerer-seat decision doc, not a run artifact. Explicitly labeled "provisional-on-runtime" — i.e. never itself verified. |
| `eval_results/A6_analysis.md` | siddanth/peak-source-a6-p1 (+ origin mirror) | 2026-07-19 | b9976 | A6 pinned grounded-VideoQA measurement, N=64 in-sample, 59 videos (mixed + all-minmax arms). |
| `eval_results/P_NOWA_accgqa_prereg.md` | siddanth/peak-source-a6-p1 (+ origin mirror) | pre-registered 2026-07-22, run 2026-07-23 (`300d857`/`dc59de3`) | b9976 | Held-out Acc@GQA pre-registration — the frozen config for the paper's headline held-out P-NOW-A result (`eval_results/P_NOWA_accgqa_result.md`, Acc@GQA = 0.1667, n=120). The *result* file itself names no build. |
| `eval_results/T1_e2e_reconciliation.md` (lines 274, 357) | origin/sonu/audit-docs, origin/sonu/t5-shotbucket | READ-ONLY audit, run date 2026-08-15/16 | b9976 (quoted from `DECISIONS.md`) | Not a run — an audit noting that **M1 (answerer wall-time distribution / any timing artifact under the seated build) does not exist at any seat.** Confirms no independent verification exists. |
| `IRIS_paper_bundle_explained.md` (line 1520) | origin/swara/feat/prerun-fixes | doc dated 2026-09-02 | b9976 | Contains the same open TODO as the paper: "confirm the llama-server build `b9976` was used for every run, not just the one prereg that records it." |
| `paper/IRIS_paper_draft.md` (line 1988, item 22) | siddanth/peak-source-a6-p1 | — | b9976 | The paper's own open-items ledger already flags this as unresolved. |
| `tuning/prerun_fixes/final_report.md`, `tuning/determinism_gate/*` | siddanth/ucf-vad-exp1 (+ origin mirror), origin/swara/feat/prerun-fixes | 2026-07-24 – 2026-07-27 | **b10099** / commit `1a064ab` | Git commit `6f36f08` (2026-07-24, "val_confirm end-to-end run — Acc@GQA=0.1894 (unverified)"): built llama.cpp **from source at b10099** for CUDA/sm_89 because no llama-server was running and the b9976 seat could not simply be resumed. The same b10099 binary was then pinned and used for the A1 determinism gate PASS on 2026-07-27 (`tuning/determinism_gate/final_report.md`), with SHA-256 `250d2eb3…` and `--version` output `1 (1a064ab)` captured. |
| `tuning/prerun_fixes/A2_binary_inventory.json` | siddanth/ucf-vad-exp1 (+ origin mirror), origin/swara/feat/prerun-fixes | audit 2026-07-27 | b9985 (current unbuilt source HEAD) vs. b10099 (searched for, not found) | States plainly: the b10099/`1a064ab` build that produced the val_confirm_e2e headline numbers "does NOT exist on this box," which "is not even the box it would have existed on" (that box was a remote Linux GPU box, gone/unsearched). |

## Timeline (why this matters)

- 2026-07-13 — `DECISIONS.md` records b9976 as the seated answerer build. No binary artifact.
- 2026-07-19 — `A6_analysis.md` run, cites b9976 (prose only).
- 2026-07-22/23 — `P_NOWA_accgqa_prereg.md` pre-registers b9976 as the frozen answerer for the paper's headline held-out Acc@GQA result; the actual run (`300d857`/`dc59de3`) produces `P_NOWA_accgqa_result.md` — which itself records **no build identifier at all**.
- 2026-07-24 — One day later, the val_confirm_e2e commit message states outright that **no llama-server was running** and the seat had to be rebuilt from source — at **b10099**, not b9976 — because no Linux CUDA prebuilt of b9976(-equivalent) existed and falling back to Ollama would have silently broken the `cache_prompt=false` determinism requirement.
- 2026-07-27 — b10099/`1a064ab` is hash-pinned and used for the A1 determinism gate (the only run in this whole search with an independently captured binary SHA-256 + `--version`). The same day, an inventory audit finds that build gone from every reachable box, and finds the local `llama.cpp` source checkout sitting at yet a third, uncompiled version (b9985).

No environment-capture artifact (`environment.json`, `pinned_binary.json`, or equivalent) exists anywhere in the repo, on any branch, for the A6 run or the P-NOW-A held-out run. `scripts/answerer_provenance.py`, the only tool in the repo that captures binary SHA-256/`--version` for a running llama-server, was added in the 2026-07-27 pre-run fix pack — i.e. **after** both b9976-citing runs had already completed.

## Answer

**No — b9976 cannot be confirmed as the build used for every reported run in the paper.** It can be confirmed only as the build *named* in the P-NOW-A prereg (and, separately, in A6_analysis.md) — and even there "confirmed" overstates it: those are self-reported strings in run-log prose, with no binary hash or `--version` capture behind them anywhere in the repository. The one build in this repo that *does* have that kind of independent verification (SHA-256 + `--version`, in `tuning/determinism_gate/environment.json`) is **b10099** (commit `1a064ab`) — used for the 2026-07-24 val_confirm_e2e headline run and the 2026-07-27 determinism gate, and confirmed by the repo's own 2026-07-27 audit to no longer exist on any reachable box. A third, different, never-compiled version (b9985) is what the local `llama.cpp` checkout currently sits at. So not only is b9976 unconfirmed beyond its own citing documents — the one build that *is* independently verified in this repo is a different build number entirely, used for a different (later, GPU-box) run.

**Strongest defensible A.2 sentence:**

> "The answerer was served via llama-server; the A6 (2026-07-19) and held-out P-NOW-A (2026-07-23) runs cite build `b9976` in their own logs, but no binary hash or `--version` capture exists for either run to verify this independently. The only llama-server build independently verified in this repository (SHA-256 + `--version`, `tuning/determinism_gate/environment.json`) is a different build, `b10099`/commit `1a064ab`, used for the later (2026-07-24 val_confirm and 2026-07-27 determinism-gate) runs on a separate GPU machine and confirmed gone from that machine as of 2026-07-27; b9976 itself was never binary-verified on any box."
