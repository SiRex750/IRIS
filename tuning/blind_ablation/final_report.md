# Blind ablation (val_confirm) — final report

## 1. Headline result

**The A−C ablation (video contribution to answer accuracy) was never measured.**
The reproduction gate on arm A (real captions, default system prompt — the
exact recorded configuration) **failed**: 345/639 (53.99%) vs the recorded
349/639 (54.62%). Per the task's explicit instructions, a failed reproduction
gate is a stop condition more important than the ablation itself, so arms
B/C/D were **not** run. This report documents the gate failure, not the
video-contribution question the task set out to answer.

## 2. Reproduction gate result

**FAILED.**

| | Acc@QA |
|---|---|
| Recorded (`tuning/val_confirm_e2e_per_question.csv`) | 349/639 = 0.5462 |
| This run (arm A, identical config) | 345/639 = 0.5399 |
| Net delta | −4 |

The net delta of −4 questions is misleadingly small. Diffing per-question
against the recorded run shows **118 of 639 questions (18.5%) flipped their
answer** relative to the recorded run — 61 that were correct are now wrong,
and 57 that were wrong are now correct. These almost cancel in aggregate but
represent substantially different per-question model behavior, not a stable
rerun of the same computation.

Same config was used in both runs: all 12 frozen hyperparameters read live
from `tuning/frozen_state.json`, `cerberus_mode="none"`, `ranking_mode="ppr"`,
`codec_conf_source="packet_size"`, K=4, Method D `half_width_s=2.2`,
question-blind captioning via `_ensure_captions` (no `question=`/`choices=`),
same 639 questions in the same order, same `granite4:micro` GGUF blob
(SHA-256 confirmed matching `97c417d…` before the run), same `llama-server`
endpoint with `cache_prompt=False`, `temperature=0`, `seed=42`.

Five example flips (full raw model output, not truncated):

- **video 9873067604, qid 4** — correct→wrong (recorded pred=2, new pred=0):
  `"ANSWER: A\nREASON: The caption mentions the boy's visible face and the car's side panel with a sound system, suggesting he is looking through the window."`
- **video 7508439506, qid 1** — correct→wrong (recorded pred=0, new pred=3):
  `"ANSWER: D\nREASON: The caption describes the baby being supported by a hand, indicating the boy placed his hands on the baby to provide support."`
- **video 7499763064, qid 5** — wrong→correct (recorded pred=2, new pred=4):
  `"ANSWER: E\nREASON: The caption mentions the man is wearing a harness and carrying a backpack, indicating he is likely adjusting his own harness for safety."`
- **video 4087654898, qid 0** — correct→wrong (recorded pred=0, new pred=1):
  `"ANSWER: B\nREASON: The caption mentions the child in a floral dress is carrying a pink watering can, indicating they used a straw to fill it."`
- **video 3471421544, qid 4** — wrong→correct (recorded pred=1, new pred=0):
  `"ANSWER: A\nREASON: The captions describe the baby as seated in a colorful jumper with toys attached, indicating the baby is jumping."`

Full detail for all 118 flips (video, qid, both predictions, both acc_qa
values, and the new run's full raw answer) is in
`tuning/blind_ablation/reproduction_gate_FAILED.json`.

## 3. Four-arm table

Only arm A ran.

| Arm | Acc@QA (all, n=639) | Acc@QA (parsed-only) | Parse-fail rate | Refusal rate | n |
|---|---|---|---|---|---|
| A_FULL | 345/639 = 0.5399 | 345/639 = 0.5399 (n_parsed=639) | 0.0% | 2/639 = 0.31% | 639 |
| B_BLIND_STRICT | not run | — | — | — | 0 |
| C_BLIND_FAIR | not run | — | — | — | 0 |
| D_SHUFFLE | not run | — | — | — | 0 |

Refusal regex: `insufficient|cannot determine|not enough|unable to|no evidence|don't have access|do not have access` (case-insensitive).

Per-question-type breakdown, arm A only:

| Type | n | Acc@QA |
|---|---|---|
| CW | 284 | 0.5528 |
| CH | 86 | 0.5349 |
| TN | 167 | 0.5449 |
| TC | 87 | 0.4943 |
| TP | 15 | 0.5333 |

## 4. Subset deltas (A−C on grounded / ungrounded / zero-overlap)

Not computed — arm C never ran.

## 5. Answer-letter distribution

Arm A only (index 0=A .. 4=E), 639 total:

| Letter | n |
|---|---|
| A (0) | 144 |
| B (1) | 153 |
| C (2) | 129 |
| D (3) | 122 |
| E (4) | 91 |
| Parse failures | 0 |

No collapse onto a single letter; distribution is close to but not perfectly
uniform (E under-selected relative to A/B/C/D).

## 6. Honest read

The task set out to measure whether IRIS's visual pipeline contributes
anything to answer accuracy beyond language priors. That question could not
be answered on this run, because a more basic assumption failed first: **the
recorded val_confirm_e2e run is not exactly reproducible under nominally
identical conditions.** Rerunning the identical configuration (same model
blob, same hyperparameters, same prompts, same captioning code path, same
seed=42/temperature=0 settings on both the answerer and captioner) produced a
different per-question answer on 18.5% of questions. The aggregate accuracy
happened to land close to the recorded number (53.99% vs 54.62%), which could
easily give false confidence that the pipeline is deterministic if only the
aggregate were checked — the per-question diff is what exposes the real
instability.

This is a more fundamental finding than the blind-ablation question itself:
if the answer to a fixed question against fixed evidence can change from run
to run, then the 349/639 headline number is a single noisy draw, not a fixed
ground truth. Any future comparison against it (this ablation, a tuning
decision, a regression check) needs to account for that noise floor, or it
risks attributing normal run-to-run variance to whatever variable is actually
being tested. The most likely source, given seed/temperature are pinned on
both the llama-server answerer and the MiniCPM captioner, is GPU floating-
point non-determinism in reduction order under full GPU offload
(`-ngl 999`) — a known category of issue with llama.cpp / cuBLAS, not a bug
in the harness (which reuses the production `iris/` code paths unmodified
and reproduces the exact same call sequence as the recorded run).

**Negative reading, stated plainly:** this run cannot support any claim about
whether the video contributes to accuracy, because it never got there. It
*can* support a claim that the currently-recorded 54.62% figure should not be
treated as an exact, reproducible constant without first characterizing the
run-to-run noise band (e.g. by running arm A alone multiple times and looking
at the spread).

## 7. What did not run, and why

- **Arms B, C, D**: not run. The task instructs stopping immediately on a
  failed reproduction gate rather than adjusting anything to force a pass;
  the script itself enforces this (`raise SystemExit(1)` after writing
  `reproduction_gate_FAILED.json`).
- **Acc@GQA / mIoP / mIoU for arm A**: computed and present in
  `arm_A_full_per_question.csv` (columns `iop`, `iou`, `acc_gqa_unverified`)
  but not reported above since the task scopes GQA metrics as meaningful for
  arm A only, and arm A itself failed its own reproduction check, so these
  numbers carry the same caveat as Acc@QA above.
- **Bootstrap / paired-delta analysis (§7 of the task)**: not run — requires
  arms A and C both, and C never ran.

## Appendix: run reliability notes (not part of the scientific result)

This diagnostic required three separate process launches, none of which were
caused by a bug in the measurement itself:

1. First launch: manually stopped by request at ~500/639 through arm A. At
   that point the script only wrote output CSVs at the end of a full arm, so
   this progress was not recoverable — the script was subsequently patched to
   checkpoint every question incrementally (`arm_A_checkpoint.jsonl`,
   `arm_BCD_checkpoint.jsonl`), fixing that gap.
2. Second launch: reached 141/639 checkpointed, then was killed when a login
   session ended — this account had `Linger=no`, so systemd tore down the
   user's process slice on logout despite the process being `nohup`'d,
   `disown`'d, and reparented to `init`. Diagnosed via `loginctl show-user`
   and fixed with `loginctl enable-linger ccbd`, so this cannot recur.
3. Third launch: resumed from the 141/639 checkpoint (skipped, not
   redone), ran arm A to completion (639/639), hit the reproduction gate
   failure described above.

Protected files (`tuning/frozen_state.json`, `split_manifest.json`,
`dataset_manifest.json`, `tuning/val_confirm_e2e_per_question.csv`,
`tuning/val_confirm_e2e_report.md`, `eval/data/nextqa/val.csv`,
`eval/data/nextqa/gsub_val.json`) were confirmed byte-identical (SHA-256)
before and after all three launches — see `protected_hashes_before.txt` /
`protected_hashes_after.txt`. The NExT-GQA test split was never opened (no
`gsub_test.json` or `test.csv` reference anywhere in
`scripts/blind_ablation_eval.py`, checked both by static grep and by test
`test_no_test_split_file_referenced_or_opened` in
`tests/test_blind_ablation_fairness.py`).
