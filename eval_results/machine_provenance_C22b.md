# Machine / hardware provenance audit — C22b

Read-only. Paper not edited. Question: did any measurement reported in
`paper/IRIS_paper_draft.md` run on a machine other than the Windows 11 /
Ryzen 7 9800X3D box described in Appendix A.2 — specifically the Linux GPU
box referred to elsewhere in the repo as `worker-1` (`/home/ccbd/IRIS-1`)?

## 1. `tuning/ucfcrime_vad_exp1/efficiency_measurements.json` at `origin/siddanth/ucf-vad-exp1`

Source of the paper's §4 ingest claim (0 neural forward passes; 2.945 s /
530 MB over 32 videos). This artifact had never been read directly before
this task. Full content:

```json
{
  "generated_utc": "2026-07-29T08:17:11Z",
  "frozen_config_used": { ... },
  "sample_size": 32,
  "sample_selection": "deterministic, stratified by nb_frames_meta across the full duration range of the 350 locally-available videos ...",
  "n_ingest_ok": 32,
  "n_ingest_failed": 0,
  "failed_videos": [],
  "wall_time_s_mean": 2.9447330906259594,
  "wall_time_s_min": 0.3366962000000058,
  "wall_time_s_max": 15.473844399995869,
  "peak_rss_bytes_mean": 530395520.0,
  "peak_rss_bytes_max": 1291190272,
  "retention_pct_mean": 10.49402558254949,
  "retention_pct_min": 10.403863037752414,
  "retention_pct_max": 10.94890510948905,
  "device_used": "cpu",
  "gpu_available_on_this_box": false,
  "gpu_required_at_any_point": false,
  "total_nn_forward_pass_calls_across_all_sampled_videos": 0,
  "no_nn_forward_pass_verified": true,
  "note_on_verification_method": "torch.nn.Module.__call__ was monkeypatched for the duration of each parse_video() call and every invocation counted; the counter is reported per-video and summed here, not just asserted from source reading."
}
```

**Every hostname, path prefix, OS, device, CUDA-availability, torch-device
string this file records — verbatim, in full:**

- `"device_used": "cpu"` — a software-level flag, computed as
  `"cuda" if torch.cuda.is_available() else "cpu"` (see below).
- `"gpu_available_on_this_box": false` — likewise derived from
  `torch.cuda.is_available()`.
- `"gpu_required_at_any_point": false`.

That is the entire set. **No hostname, no OS field, no CPU model, no path
prefix, no torch-device string beyond the bare `"cpu"` label, and no
`torch.cuda.is_available()` raw value are recorded anywhere in this JSON.**

The generating script, `scripts/ucfcrime_vad_exp1_efficiency.py`
(`origin/siddanth/ucf-vad-exp1`), confirms why — it never calls
`socket.gethostname()`, `platform.*`, or `os.uname()` anywhere; the only
device-related code is:

```python
device_used = "cuda" if torch.cuda.is_available() else "cpu"
gpu_available = torch.cuda.is_available()
...
device_used = "cpu (torch not importable)"
...
"device_used": device_used,
```

So `gpu_available_on_this_box` is strictly a **torch-level** determination
(`torch.cuda.is_available()`), not a hardware inventory — it cannot
distinguish "this box has no physical GPU" from "this box has a GPU but
torch was not built with CUDA support" (the A.2-machine's own torch is
`2.13.0+cpu`, i.e. built without CUDA regardless of hardware).

The one piece of circumstantial evidence is the companion
`tuning/ucfcrime_vad_exp1/efficiency_per_video.csv`, whose `path` column uses
Windows-style backslash paths (e.g.
`eval\data\ucf\videos\Anomaly-Videos-Part-1\Assault\Assault038_x264.mp4`),
consistent with — but not proof of identity with — the Windows 9800X3D box.
No hostname anywhere ties it to that specific machine.

**Finding: the artifact records NO hardware provenance beyond a
torch-derived cpu/no-cuda flag. Absence of hostname/OS/path-prefix fields is
itself the finding, not a gap to fill by inference.**

## 2. The P_NOWA_accgqa run (commit `dc59de3`)

`eval_results/P_NOWA_accgqa_raw.json`, commit `dc59de38c85bade59778e9071e5689f15f26e91e`
(2026-07-23 22:09:32 +0530, author `Antigravity AI <antigravity@gemini.google>`).
Full `provenance` block, verbatim:

```json
{
  "backend_class": "LlamaServerBackend",
  "endpoint": "http://127.0.0.1:8091/v1",
  "model": "granite4:micro",
  "temperature": 0.0,
  "cache_prompt": false,
  "span_mode": "ppr_peak",
  "span_half_width": 2.2,
  "span_peak_source": "clip_in_ppr_top8",
  "ppr_lambda": 0.5,
  "ppr_damping": 0.5,
  "graph_mode": "flat",
  "motion_similarity_mode": "action_score",
  "git_commit": "b170007b329dcd203851527f708fd49db602a9f9",
  "git_dirty": false,
  "config_hash": "439be4e3a414197d248cf56df9460765acd895fa15174dd50390f7d82f24342a",
  "timestamp_utc": "2026-07-23T16:36:01.534593+00:00",
  "n_questions": 120,
  "n_videos": 27,
  "top_k": 12,
  "num_boot": 1000,
  "clip_anchor_fallback_rate": 0.0
}
```

**Finding: no hostname, OS, CPU/GPU model, or path prefix anywhere in this
record.** `endpoint: http://127.0.0.1:8091/v1` only says the answerer was
reached over localhost from whichever machine ran the eval script — it says
nothing about which physical box that was. The prereg (already audited in
`llama_build_C22.md`) names `llama-server b9976` but that string, too, is
unverified prose with no binary/host evidence attached. **This run records
NO hardware provenance at all.**

## 3. The 2026-07-24 `val_confirm_e2e` run

Commit `6f36f08adc70bfb74d79c6512801b85e8f6a376b`
("feat: val_confirm end-to-end run — Acc@GQA=0.1894 (unverified)..."),
full body:

> Blocker resolved: iris_config.py's default answerer_backend="llama_server"
> expects a real llama-server on 127.0.0.1:8091, which was not running. Built
> llama-server from source (llama.cpp b10099, CUDA sm_89 for the RTX 4090,
> since no Linux CUDA prebuilt exists in llama.cpp's release assets and the
> box's Vulkan ICD list has no NVIDIA entry) rather than silently falling
> back to Ollama's LlamaBackend...

And from `tuning/val_confirm_e2e_report.md` (`origin/siddanth/ucf-vad-exp1`),
verbatim:

> **Binary**: built from source, `llama.cpp` release tag `b10099` (commit
> `1a064ab`), with `GGML_CUDA=ON` targeting `sm_89` (RTX 4090) via the
> CUDA 12.6 toolkit at `/usr/local/cuda-12.6` (the system default `nvcc` at
> `/usr/bin/nvcc` was a stale CUDA 11.5 install that doesn't support Ada and
> had to be bypassed explicitly via `-DCMAKE_CUDA_COMPILER`). No matching
> Linux CUDA prebuilt exists in llama.cpp's release assets (only
> Vulkan/ROCm/SYCL/CPU for Ubuntu x64), and the machine's Vulkan ICD list has
> no NVIDIA entry, so a prebuilt binary would not have reached the GPU —
> building from source was necessary, not a fallback of convenience.

**Finding: this run explicitly and verifiably ran on a Linux machine with an
NVIDIA RTX 4090 GPU (CUDA 12.6, `sm_89`)** — categorically not the Windows
11 / Ryzen 7 9800X3D CPU-only box in A.2. `tuning/val_confirm_e2e_report.md`
does not use the literal string `worker-1` itself, but the same b10099
binary this run produced is later identified, three days on
(`tuning/determinism_gate/final_report.md`, 2026-07-27), as living at
`/home/ccbd/.claude/jobs/de60ff71/tmp/build/llama.cpp/build/bin/llama-server`
— a `/home/ccbd/...` path — and that later document is explicitly
"**Generated:** 2026-07-27, on `worker-1` (`/home/ccbd/IRIS-1`)". The
`/home/ccbd/` path prefix ties the val_confirm_e2e binary to the same
`ccbd`-home-directory Linux box subsequently named `worker-1`.

## 4. `eval_results/env_A2.json`

Full content:

```json
{
  "timestamp_utc": "2026-09-08T18:43:07.315716+00:00",
  "cpu": {
    "model": "AMD Ryzen 7 9800X3D 8-Core Processor",
    "physical_cores": 8,
    "logical_cores": 16
  },
  "ram_total_gb": 31.16,
  "os": {
    "name": "Windows",
    "version": "Windows-11-10.0.26200-SP0"
  },
  "python_version": "3.12.10",
  "torch": { "version": "2.13.0+cpu", "cuda_available": false },
  ...
  "git": {
    "head": "48c082a64465bc363510c4bcb31bbe90513ebcc7",
    "tracked_dirty_file_count": 2
  }
}
```

This is exactly the source the paper's A.2 already names (`env_A2.json`,
"captured 2026-09-08T18:43:07Z at commit `48c082a`"). **What it does and does
not claim:**

- It **does** record a single snapshot of one machine's state — CPU, OS,
  Python/torch/library versions, git HEAD — captured on 2026-09-08.
- It **does not** claim to have been captured at the time of any specific
  run; the paper's own A.2 prose already concedes this explicitly: "This
  environment was captured on the same machine after the reported runs
  rather than at run time... the full record is in `eval_results/env_A2.json`."
  That "after the reported runs" caveat is doing real work — the file has no
  field naming which runs it does or does not describe.
- It **makes no assertion, anywhere in the JSON, that this was the only
  machine used** for any measurement in the paper. There is no "machines
  used" list, no per-artifact machine attribution, nothing that generalizes
  this one snapshot to every reported number. The paper's A.2 prose implies
  single-machine framing by omission (describing "the" environment
  singular), but `env_A2.json` itself is scoped to nothing more than: *this
  is what one machine looked like on 2026-09-08*.

## Answer

**Yes — measurements reported in the paper ran on a machine other than the
Windows 11 / Ryzen 7 9800X3D box in A.2.** The 2026-07-24 `val_confirm_e2e`
run and the answerer-provenance chain around it (`b10099`/`1a064ab`,
`tuning/determinism_gate/`) are independently and explicitly verified to have
run on a Linux, NVIDIA RTX 4090 / CUDA 12.6 machine — the same
`/home/ccbd/...`-rooted box later named `worker-1`. The P-NOW-A held-out
Acc@GQA run (`dc59de3`) and the §4 ingest-efficiency artifact
(`efficiency_measurements.json`) both carry **zero** hardware provenance —
no hostname, OS, or path prefix in either — so neither can be placed on any
specific box; they can only be said to have used `llama-server` over
localhost (P-NOW-A) or to have measured `torch.cuda.is_available() == False`
(ingest efficiency), which is a software-level flag, not a hardware
guarantee. `env_A2.json` documents only its own one-machine, post-hoc
snapshot and makes no exclusivity claim.

**Is §8.9's sentence "Our ingest measurements were taken on a machine
without a GPU" true as written?** Narrowly, yes as far as it can be
checked: the only artifact behind the §4 ingest claim
(`efficiency_measurements.json`) records `"device_used": "cpu"` and
`"gpu_available_on_this_box": false`, and there is no evidence anywhere in
this audit that ingest (as opposed to answerer) measurements ran on
`worker-1` or any other GPU box — the GPU box is tied, by explicit and
verified text, only to the *answerer* runs (`val_confirm_e2e`,
`P_NOWA_accgqa`'s presumed llama-server host, the A1 determinism gate), never
to ingest. But the sentence should not be read as stronger than its evidence:
`gpu_available_on_this_box` is a `torch.cuda.is_available()` result, not a
hardware inventory, so it does not independently establish the ingest box
had no physical GPU (only that no CUDA device was visible to torch there) —
and because the artifact records no hostname at all, "a machine without a
GPU" cannot be tied to *the* A.2 machine specifically, only to *some*
unidentified CPU-only-to-torch machine. The sentence is defensible as
written but rests on a thinner evidentiary base than its confident phrasing
suggests.
