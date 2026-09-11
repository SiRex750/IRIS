# Incidental machine fingerprints — C22c

Read-only pass. Not using filesystem mtimes (every file in this checkout
carries the same bulk-checkout timestamp — verified, and irrelevant to
content anyway). This is *not* a repeat of the provenance-block audit in
`machine_provenance_C22b.md` (already checked, mostly empty of hardware
fields) — this pass looks only at what leaked into the artifacts
*incidentally*: stray absolute paths, path separators inside string values,
line endings, and similar.

One methodological note that changes how "line endings" gets read below:
`core.autocrlf=true` is set in this checkout (`git config core.autocrlf` →
`true`), and the git-stored blobs for every *tracked* target artifact are
confirmed LF-only (checked via `git show HEAD:<path> | grep -c $'\r'` → 0 for
all of them). That means CRLF observed in the *working tree* of a tracked
file is a property of this checkout, not of the machine that produced the
artifact — it is not evidence. Only **untracked** files (git never touched
them) carry line-ending evidence of their origin. This is called out per row.

## Fingerprint table

| Artifact | Fingerprint found | Verbatim value | Implies |
|---|---|---|---|
| `build_cost_final.md` | OS name string | `` on Windows — reported as `null`/`[0.0,0.0,0.0]` from psutil's Windows `` (line 21); `Machine state before old-loop: cpu_percent=10.1%, available RAM=19.96 GB / 33.46 GB total, 16 logical / 8 physical CPUs, getloadavg=[0,0,0] (Windows — not meaningful).` (line 28) | Windows/Ryzen box |
| `build_cost_final.md` | line endings | untracked file, genuine CRLF (110/110 lines) | Windows/Ryzen box (consistent) |
| `build_cost_final.json` | CPU inventory (×6 embedded snapshots) | `"logical_cpu_count": 16, "physical_cpu_count": 8` | Windows/Ryzen box — matches `env_A2.json`'s `physical_cores: 8, logical_cores: 16` for the AMD Ryzen 7 9800X3D |
| `build_cost_final.json` | line endings | untracked, genuine CRLF (193/192) | Windows/Ryzen box |
| `build_cost_final_raw/new_r1.json`, `new_r2.json`, `new_r3.json`, `old_r1.json`, `old_r2.json`, `old_r3.json` | CPU inventory | `"logical_cpu_count": 16, "physical_cpu_count": 8"` in all six | Windows/Ryzen box |
| `build_cost_final_raw/*.json` | line endings | untracked, genuine CRLF in all six | Windows/Ryzen box |
| `MLVU_codec_baseline.json` | backslash path separators inside JSON string values (150 occurrences) | `"video": "mlvu\\MLVU\\video\\6_anomaly_reco\\surveil_106.mp4"` | Windows box |
| `MLVU_codec_baseline_checkpoint.json` | same field, same pattern | `"video": "mlvu\\MLVU\\video\\6_anomaly_reco\\surveil_106.mp4"` | Windows box |
| `MLVU_ablation.json` | backslash path separators inside JSON string values (600 occurrences) | `"video": "mlvu\\MLVU\\video\\1_plotQA\\en_tv_1.mp4"` | Windows box |
| `MLVU_ablation_long_*.json` (checkpoint, shard0/1, trimmed, trimmed_checkpoint shard0/1) | *forward*-slash paths in the same field | `"video": "MLVU/video/1_plotQA/4.mp4"` | Not diagnostic on its own (forward slashes are OS-ambiguous) — see stdout logs below for the same run, which settle it |
| `MLVU_ablation_long_stdout.log`, `MLVU_ablation_long_shard0_stdout.log`, `MLVU_ablation_long_shard1_stdout.log`, `MLVU_ablation_long_trimmed_shard0_stdout.log` | absolute Windows paths + username, staging-dir download/delete log lines | `` [download] MLVU/video/1_plotQA/4.mp4 -> C:\Users\Siddanth Anil\IRIS\mlvu_long_staging\MLVU\video\1_plotQA\4.mp4 (23.8s, 68.6 MB) `` and `` [disk] deleted C:\Users\Siddanth Anil\IRIS\mlvu_long_staging\...`` (repeated throughout) | Windows box, username "Siddanth Anil" — same box producing the forward-slash JSON above |
| `MLVU_ablation_long_trimmed_merge_stdout.log` | absolute Windows path, username | `Wrote C:\Users\Siddanth Anil\IRIS\eval_results\MLVU_ablation_long_trimmed.json` / `...trimmed.md` | Windows box, username "Siddanth Anil" |
| `MLVU_ablation_long_*` (all 15 files) | line endings | all untracked, genuine CRLF throughout | Windows box |
| `e2e_stage3_decomp_raw_stdout.log` | absolute Windows path, username | `Wrote C:\Users\Siddanth Anil\IRIS\eval_results\e2e_stage3_decomp_raw.json` | Windows box, username "Siddanth Anil" |
| `e2e_stage3_decomp_raw_stdout.log` | line endings | untracked, genuine CRLF | Windows box (consistent) |
| `e2e_speedup_stdout.log` | absolute Windows path, username | `Wrote C:\Users\Siddanth Anil\IRIS\eval_results\e2e_speedup.json` | Windows box, username "Siddanth Anil" |
| `e2e_speedup_stdout.log` | line endings | untracked, genuine CRLF | Windows box (consistent) |
| `caption_stage_diagnosis_stdout.log` | absolute Windows path, username | `Wrote C:\Users\Siddanth Anil\IRIS\eval_results\caption_stage_diagnosis.json` / `...diagnosis.md` | Windows box, username "Siddanth Anil" |
| `caption_stage_diagnosis_stdout.log` | line endings | untracked, genuine CRLF | Windows box (consistent) |
| `virat_smoke_N4892_scenesparse.json` | absolute Windows path, username, ×2 fields | `"video": "C:\\Users\\Siddanth Anil\\IRIS\\eval\\data\\virat\\videos\\VIRAT_S_040001_01_000448_001101.mp4"` and `"cache_path": "C:\\Users\\Siddanth Anil\\IRIS\\eval\\data\\virat\\index_cache\\VIRAT_S_040001_01_000448_001101.npz"` | Windows box, username "Siddanth Anil" |
| `virat_smoke_N4892_flat.json` | none found | — | indeterminate (no path/hostname/OS fields; only edge/node counts) |
| `scaling_curve_v3.json` / `.md` | none beyond `git_dirty_file_count: 94` (not machine-diagnostic) | — | indeterminate |
| `scaling_curve_v3_stdout.log` | line endings only; no path/hostname/OS string found (checked for "Wrote", absolute paths, CUDA/torch/device strings — none) | genuine CRLF (untracked) | Windows-*consistent* (CRLF is the one signal), but weak on its own since CRLF alone doesn't prove origin the way an explicit path does — treat as corroborating, not conclusive |
| `virat_latency_N4892_raw.json` / `_result.md` | none found beyond `virat_fingerprint_*` hashes (content hashes, not host) | — | indeterminate |
| `virat_latency_N4892_stdout.log` | none beyond scene_diag debug dicts (no paths/hostnames); line endings genuine CRLF (untracked) | — | CRLF-consistent with Windows, otherwise indeterminate |
| `P_NOWA_accgqa_raw.json` | none found | `"endpoint": "http://127.0.0.1:8091/v1"` is present but localhost — says nothing about which physical box reached it | indeterminate (confirms `machine_provenance_C22b.md`'s finding — zero hardware provenance) |
| `blockdiag_identity_gate_result.json` / `.md` | none found | only edge/mismatch counts (`common_edge_count`, `field_mismatch_count`, `pagerank_mismatch_count`) | indeterminate |
| `blockdiag_grounding_gate_result.json` / `.md` | none found | — | indeterminate |
| `build_dedup_repeats.json` / `.md` | none found beyond `expected_edge_count: 23571` (content, not host) | — | indeterminate |
| `efficiency_measurements.json` (`origin/siddanth/ucf-vad-exp1`) | none directly in this file; LF line endings (0 CR found) | `"device_used": "cpu"`, `"gpu_available_on_this_box": false` — torch-derived flags, not hardware fields (matches `machine_provenance_C22b.md` §1) | indeterminate on its own |
| `efficiency_per_video.csv` (companion file, same branch/dir) | backslash path separators inside a CSV field value | `eval\data\ucf\videos\Anomaly-Videos-Part-1\Assault\Assault038_x264.mp4` (repeated in every row's `path` column) | Windows box — the *only* incidental leak tying the §4 ingest-efficiency measurement to a machine at all; already flagged in `machine_provenance_C22b.md` as "consistent with, not proof of identity with" the Windows box. No hostname anywhere, so it corroborates but doesn't uniquely identify. |

## What did *not* leak, anywhere

Checked across every artifact listed above (plus their `_raw`, `_checkpoint`,
`.shard*`, and `_stdout`/`_stderr` companions): no `/home/`, `/mnt/`,
`/tmp/`-prefixed paths; no `ccbd`; no literal hostname strings
(`worker-1`, `IRIS-1`, or any other); no `torch`/`CUDA`/`+cu*`/`+cpu`
version strings, device strings, or GPU model names (`nvidia`, `GeForce`,
`RTX`); no `os.name`/`platform.*` dumps; no timezone offset other than
`+00:00` (the one hit, `P_NOWA_accgqa_raw.json:18`, `16:36:01.534593+00:00`,
is UTC and excluded per the brief). No CPU-model string anywhere outside
`build_cost_final.md`/`.json`/`_raw/*`.

## Per-reported-number answer

Going through the artifacts in the order they back numbers in
`paper/IRIS_paper_draft.md`:

- **Build/graph-construction cost (`build_cost_final.*`, `build_cost_final_raw/*`)** — **Windows/Ryzen box.** Explicit `Windows` string in the prose, psutil `getloadavg` note, and a CPU inventory (`16 logical / 8 physical`) repeated identically across all 6 raw run files that matches `env_A2.json`'s AMD Ryzen 7 9800X3D exactly. This is the one artifact where the incidental leak is unambiguous and internally corroborated (prose + CPU count + CRLF all agree).
- **`build_dedup_repeats`** — indeterminate. No incidental leak beyond content counts.
- **Identity/grounding block-diagram gates (`blockdiag_identity_gate_result.*`, `blockdiag_grounding_gate_result.*`)** — indeterminate. Nothing beyond edge/mismatch counts leaked in either file.
- **Scaling curve (`scaling_curve_v3.*`)** — indeterminate on the `.json`/`.md`; the untracked `_stdout.log` is genuinely CRLF (Windows-consistent) but carries no path, hostname, or username — treat as weak corroboration only, not identification.
- **End-to-end decomposition / speedup / caption-stage diagnosis (`e2e_stage3_decomp_raw.*`, `e2e_speedup.*`, `caption_stage_diagnosis.*`)** — **Windows box, username "Siddanth Anil".** Each has a `Wrote C:\Users\Siddanth Anil\IRIS\eval_results\...` line in its untracked `_stdout.log`, an absolute path that survives independent of git's line-ending normalization.
- **VIRAT latency (`virat_latency_N4892_*`)** — indeterminate. No path/hostname leaked in the raw json, result md, or stdout log (only scene-diagnostic dicts and content hashes); CRLF in the stdout log is Windows-consistent but not conclusive alone.
- **VIRAT smoke test (`virat_smoke_N4892_*`)** — **`_scenesparse.json`: Windows box, username "Siddanth Anil"** (two absolute `C:\Users\Siddanth Anil\IRIS\...` fields). **`_flat.json`: indeterminate** (no leak).
- **MLVU codec baseline (`MLVU_codec_baseline.*`)** — **Windows box** (150 backslash-separated `mlvu\MLVU\video\...` path strings in both the `.json` and its `_checkpoint.json`).
- **MLVU ablation, short (`MLVU_ablation.json`/`.md`)** — **Windows box** (600 backslash-separated `mlvu\MLVU\video\...` path strings).
- **MLVU ablation, long (`MLVU_ablation_long*`)** — **Windows box, username "Siddanth Anil".** The JSON/checkpoint files alone use forward slashes and would be indeterminate on their own, but the companion download/delete `_stdout.log` files (shard0, shard1, unsharded, and the trimmed-merge log) carry explicit `C:\Users\Siddanth Anil\IRIS\mlvu_long_staging\...` absolute paths for the same run, settling it.
- **P-NOW-A Acc@GQA (`P_NOWA_accgqa_raw.json`)** — indeterminate. Confirms `machine_provenance_C22b.md`'s finding of zero hardware provenance; no incidental leak either (only a localhost endpoint, which says nothing about the physical host).
- **Ingest efficiency (`efficiency_measurements.json` + `efficiency_per_video.csv`, `origin/siddanth/ucf-vad-exp1`)** — **Windows box, but only via the companion CSV, not the JSON itself.** The JSON that the paper actually cites carries zero hardware fields (torch-derived `device_used`/`gpu_available_on_this_box` only). The sibling `efficiency_per_video.csv` in the same directory leaks Windows-style backslash paths in its `path` column for every row. This is circumstantial — same directory, same run's per-video detail — but there is no hostname anywhere tying it to the specific Ryzen box in A.2 or any other named machine, so read it as "consistent with a Windows box," not "identified."

**Machines that do NOT show up incidentally anywhere in this artifact set:** no artifact behind any reported number in this list carries a `/home/ccbd/`, `worker-1`, CUDA, or torch-GPU fingerprint. Per `machine_provenance_C22b.md`, the Linux/RTX-4090 `worker-1` box is tied — by explicit, non-incidental text — only to *answerer* infrastructure (llama-server build/run), never to any of the artifacts audited here. Nothing in this incidental-fingerprint pass changes that: every incidental leak found points to the Windows/Ryzen box (or is silent).
