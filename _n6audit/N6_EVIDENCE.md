# N6 evidence audit — caption-study numbers

Read-only audit. No script was run, no number recomputed, no value reconstructed.
Scope searched: `C:\Users\akash\Documents\Iris` working tree (incl. untracked),
`..\Iris-ucfvad`, `..\Iris-scaling`, and all 206 commits reachable from
`git rev-list --all` (31 refs, incl. `refs/stash` and
`wip/laptop-preserve-2026-08-04`).

## Headline

**The caption study left no artifacts on disk.** Not in the working tree, not in
any commit on any ref, not in the preserved chat archives. All four target
numbers, the scorer definition, and both control procedures are ABSENT.

The strongest positive evidence is a prior audit committed at `457355da4`,
[tuning/prerun_fixes/C2_caption_audit.md:92](tuning/prerun_fixes/C2_caption_audit.md):

> `(no question-aware caption arm exists in this dump to compare against)`

## N5. Classification of the four target numbers

| # | Number | Claimed meaning | Classification | Basis |
|---|---|---|---|---|
| 1 | **0.704** | caption recall of question content words, question-aware | **ABSENT** | Zero standalone occurrences across all 206 refs. Every `0.704` hit is a substring of a longer, unrelated float (e.g. `0.7040816326530612` = an IoU column in `video_paper/runs/nextqa/per_event_results.csv`; `0.704119410001763` in `metric_parity_diff.csv`). No caption artifact anywhere contains a `recall` key. |
| 2 | **0.148** | same metric, question-agnostic | **ABSENT** | Same. Every hit is a substring of a longer float (e.g. `0.14893617021276595`, an IoU value repeated across NExT-QA per-event rows). No standalone `0.148` on any ref. |
| 3 | **"within 0.4pt"** | wrong-question control reproducing the treatment effect | **ABSENT** | The wrong-question control does not exist: `wrong_question` returns **0 hits** across all 206 refs, as do all naming variants tried. There is no artifact for the claim to be within 0.4pt *of*. |
| 4 | **0.699 / 0.325** | donor-swap control | **ABSENT** | `0.699`: only substring hits (`3260.6991` = a `latency_ms` value in `caption_benchmark_moondream.json`; PDF text-matrix coordinates). `0.325`: two standalone occurrences, **neither** caption-related — see below. `donor_swap` returns **0 hits** across all refs. |

### Why the two standalone `0.325` hits are not the donor-swap number

Reported for completeness; neither is substituted for the target value.

1. `tests/test_ppr_production.py:133-134` (on refs `5d0969d`, `a5cea25`) —
   ```
   # flag OFF: node 1 driven by P-frames → cc = 0.1 + 0.9*(1/4) = 0.325
   assert abs(cc_off[1] - 0.325) < 1e-9, f"Expected 0.325, got {cc_off[1]}"
   ```
   A hand-derived `codec_conf` constant in a PPR unit test. Not a caption metric.

2. `a5cea25:tuning/prerun_fixes/item_D_audit.md:29,43` —
   ```
   `~0.53 / 0.658 / 0.798 / 0.857` at `K = 4 / 8 / 16 / 24` and `peak_in_gold ~0.325`
   ...
   The `K = 8/16/24` columns and the `0.325` peak figure appear in no committed artifact.
   ```
   A retrieval metric (`peak_in_gold`), and a *prior audit already classified it
   as unsourced*. Not a caption metric.

## Supporting definitions — also ABSENT

| Item | Classification | Basis |
|---|---|---|
| Scorer: "caption recall of question content words" | **ABSENT** | `git grep -E '"[a-z_]*recall[a-z_]*"' <all refs> -- '*caption*'` → **0 hits**. `content_word`, `recall_question`, `question_recall` → **0 hits** each across all refs. No definition exists anywhere. |
| Wrong-question generation procedure | **ABSENT** | `wrong_question` → 0 hits, all refs. |
| Donor-swap generation procedure | **ABSENT** | `donor_swap` → 0 hits, all refs. |

## N2. Term search results (all 206 refs)

| Term | Hits |
|---|---:|
| `recall_question` | 0 |
| `question_recall` | 0 |
| `content_word` | 0 |
| `wrong_question` | 0 |
| `donor_swap` | 0 |
| `question_aware` (underscore) | 0 |
| `question_agnostic` | 0 |
| `query_aware` | 0 |
| `question.conditioned` | 0 |
| `conditional_caption` | 0 |
| `question.echo` / `echo (effect\|control\|study\|finding)` / `caption.recall` | 0 |
| `question[- ]aware` (hyphen/space) | 141 (9 distinct files — all prose, see below) |

Every `question-aware` occurrence is **prose in a report or a code comment**, never
a result. The two substantive ones:

- `457355da4:tuning/prerun_fixes/C2_caption_audit.md:67` — a forward-looking cost
  estimate: *"Cost implication: question-aware captioning (captioning per-question
  instead of per-frame) would multiply captioning calls from 1665 (today, one per
  distinct frame) up toward 2556…"*. A projection, not a run.
- `457355da4:tuning/prerun_fixes/C2_caption_audit.md:92` — the explicit null quoted
  above.
- `tuning/prerun_fixes/final_report.md:88` — *"sampled captions are generic scene
  descriptions, not question-aware"* — a qualitative observation on the existing
  question-blind dump.

## N3. Caption-related scripts ever committed (all history)

```
eval/caption_benchmark.py            eval/run_captioner_comparison.py
scripts/caption_dump_io.py           scripts/captioner_bakeoff.py
scripts/diag_v4_moondream_captions.py
scripts/smoke_caption{,_moondream,_ollama_moondream,_qwen}.py
scripts/smoke_lazy_caption.py        smoke_scripts/diagnose_captioner_nondeterminism.py
tests/test_c0_h2_caption_replay_and_repeat.py
```

None references question-aware captioning, caption recall, wrong-question, or
donor-swap. The bake-off docstring states the protocol is **query-blind**:

> `scripts/captioner_bakeoff.py` — *"identical protocol: fixed query-blind prompt
> "Describe this image.", temperature 0, fixed seed, one request at a time"*

`captioner_mode` takes exactly two values across all history: `"unconditional"`
and `"prompted"`. The three `"prompted"` artifacts
(`caption_benchmark_moondream{,_cuda-fp16,_cuda-int8}.json`) carry the same metric
schema as the unconditional ones — uniqueness, cosine distance, confabulation,
latency. **No recall metric, and no question field.**

## N1. Key inventory of the named artifacts

| File | Top-level keys | Metrics held |
|---|---|---|
| `eval_results/caption_benchmark_blip.json` | `model, captioner_mode, device, timestamp, virat_dir, summary, top_10_repeated, confabulation_flagged_captions, captions` | `summary/`: total_frames, total_succeeded, total_failed, exact_unique_count, exact_uniqueness_rate, normalized_unique_count, mean_pairwise_cosine_distance, confabulation_flag_count, confabulation_rate, mean/std/min/max_latency_ms, frames_per_second |
| `eval_results/caption_benchmark_moondream.json` | identical schema | identical (`captioner_mode: "prompted"`) |
| `eval_results/caption_benchmark_moondream_cuda-fp16.json` | identical schema | identical |
| `eval_results/caption_benchmark_moondream_cuda-int8.json` | identical schema | identical |
| `eval_results/grounding_report.json` | `semantic_only, hybrid_legacy, ppr` | each: avg_precision, avg_recall, abstention_rate, avg_latency_ms, runs — **retrieval** recall over grounding arms, not caption recall |
| `eval_results/ablation_report.json` | `metadata, aggregated_metrics, failures, detailed_runs` | aggregated_metrics: baseline, ablation_1, full_iris |
| `eval_results/result_moondream2_HF_fp16.json` | `model, exact_uniqueness, normalized_uniqueness, mean_cosine_dist, confab_count, confab_rate, mean_latency_ms, peak_vram_nvidia_smi, peak_vram_ollama_ps, peak_vram_pytorch` | uniqueness / confabulation / latency / VRAM |

**No artifact in this set contains a recall-of-question-content-words metric, a
question field, or any question-conditioned arm.** `grounding_report.json` has an
`avg_recall`, but it is retrieval recall across grounding strategies — a different
quantity, not substituted here.

## N4. Untracked files and chat archives

- Working tree (incl. untracked, all of `.py/.md/.json/.csv/.txt/.yaml`): **0 hits**
  for every caption-study term.
- No archive files exist in the working tree.
- On `wip/laptop-preserve-2026-08-04` (`5d0969d`), four archives exist:
  `full_chat_history.md`, `full_chat_history.pdf`, `full_session_raw_archive.md`,
  `session_archive_20260716_0132.md`.

Grepping all four for `question.aware|question.agnostic|wrong.question|donor.swap|
content word|caption recall|question echo` returns **0 hits**.

So the numbers are not even CHAT-ONLY *on disk* — the committed transcripts do not
contain them. They exist only in the prior conversation, which is not an on-disk
artifact. **No number qualifies as SOURCED or CHAT-ONLY; all four are ABSENT.**

## What would have to be RE-RUN to make each number citable

Nothing on disk can be cited. Every item below requires new work — none of it is a
re-derivation from existing data, because the underlying arm was never run.

1. **A scorer definition, written and committed first.** "Caption recall of question
   content words" has no definition anywhere. Required: the content-word extraction
   rule (stopword list, lemmatization/stemming, whether question words like
   who/what/why count), the matching rule (exact / lemma / substring), and the
   denominator (per-question content words, or per-caption). Without this, 0.704 and
   0.148 are not reproducible even in principle.

2. **A question-agnostic caption arm → 0.148.** The existing dumps are query-blind,
   so caption *text* may be reusable, but no recall was ever scored over them. Needs
   the scorer from (1) applied to a committed caption dump with question pairings.

3. **A question-aware caption arm → 0.704.** Does not exist in any form.
   `C2_caption_audit.md:92` states this explicitly. Requires captioning per-question
   rather than per-frame — the same audit estimates this raises captioning calls from
   1665 to ~2556 on the 112-video / 639-question ablation set.

4. **A wrong-question control → "within 0.4pt".** Needs a committed procedure for
   generating a deliberately mismatched question per frame (sampling pool,
   same-video vs cross-video, seed), then arm (3) re-run against it.

5. **A donor-swap control → 0.699 / 0.325.** Needs a committed donor-selection
   procedure (what is swapped, from which donor, under what pairing constraint),
   then scoring under (1). Note the claim is two numbers; nothing on disk indicates
   which condition each belongs to.

6. **Committed artifacts with recall keys.** All five of the above must write JSON/CSV
   carrying the recall metric, seeds, and input manifest, so the numbers have a
   citable source rather than a transcript.

## Method notes / limits

- Literal-value search was run twice: bare (`0\.704`) and standalone
  (`0\.704([^0-9]|$)`). The bare form returned 2,749 / 18,124 / 1,923 / 11,450 hits
  for the four values respectively — all inspected samples were substrings of longer
  unrelated floats. The standalone form is the one reported above.
- One correction during the audit: an initial N3 path search passed all 206 refs to
  `git ls-tree`, which takes a single tree-ish and parsed the rest as pathspecs,
  returning nothing. Re-run via `git log --all --name-only`; the corrected result is
  what appears in N3.
- `refs/stash` (`e03898a`) is included in `git rev-list --all` and was searched.
- Binary/derived files (`full_chat_history.pdf`, `repomix-output.xml`,
  `video_paper/git_state.txt`) produced only coordinate/diff noise and were excluded
  from the reported hit lists, never from the searches themselves.
