# Antigravity prompt: remote-GPU, paper-grade video benchmark

Copy everything below this line into Antigravity.

---

You are the lead research engineer responsible for turning the current repository into a reproducible, paper-grade benchmark and then running it on a remotely accessed GPU server. Complete the implementation, validation, execution, statistical analysis, and paper-ready artifact generation. Do not stop after writing a plan or after a smoke test.

The publication-facing name is **the proposed model** or **the proposed system**. Do not use the repository's legacy project name in plots, tables, result prose, captions, or manuscript-facing artifacts. Internal Python imports and existing file names may remain unchanged to avoid a risky repository-wide rename.

## Scientific objective and non-negotiable rule

Evaluate whether the proposed codec-guided frame selector improves temporally relevant frame selection under an exactly matched output-frame budget, and separately evaluate whether its graph-based query ranking adds value. The experiment must include working optical-flow and CLIP baselines. VIRAT Ground Release 2.0 is part of the requested core evaluation as a surveillance-domain event-selection and codec-efficiency stress test.

A faster GPU makes a large evaluation feasible; it does **not** make a superiority claim valid. Do not manufacture, optimize toward, or assume a positive outcome. The permitted claim is determined only by the predeclared metrics, complete raw results, paired uncertainty estimates, and claim gate below. A negative or mixed result must be reported honestly.

Do not use the phrase "state of the art" merely for beating sampling baselines. That phrase requires comparisons against published task systems under their official protocols. The narrower phrase "outperforms matched-budget frame-selection baselines" is allowed only if the claim gate passes.

## Inputs

Use these values if already available; otherwise ask once for only the missing values:

```text
REPO_URL=https://github.com/SiRex750/IRIS.git
SOURCE_BRANCH=pillar4-diagnostics
SSH_HOST=<remote GPU hostname or IP>
SSH_USER=<remote username>
SSH_PORT=<usually 22>
SSH_KEY=<private key path on laptop; never copy it to the server>
REMOTE_ROOT=<large writable server filesystem>
SCHEDULER_ACCOUNT=<only if required by Slurm/PBS>
ANSWER_MODEL_ID=<only needed for the optional end-to-end QA track>
```

Treat `SOURCE_BRANCH` as the requested upstream source, but first inspect the current working tree. Record `git status --short --branch`, `git rev-parse HEAD`, remotes, and the difference from `origin/pillar4-diagnostics`. Preserve every existing user change. Do not reset, clean, discard, overwrite, force-push, or silently merge unrelated work. Put benchmark code in additive modules/configs where possible. If essential work exists only in an uncommitted laptop tree, report that fact and use a non-destructive `rsync` or a user-approved commit; do not pretend the server clone contains it.

## Phase 1 — audit and freeze the scientific contract

Before changing code:

1. Read the current selector, ingest/index/query paths, benchmark scripts, reports, tests, and configuration.
2. Identify the one canonical production path to benchmark. The live API path and persistent `ingest -> index -> query` path must not be mixed in one result table if their behavior differs.
3. Write `eval/paper_benchmark/PREREGISTRATION.md` and a machine-readable `preregistration.yaml` before the complete evaluation. Include datasets/splits, valid-sample policy, methods, budgets, primary endpoints, secondary metrics, random seeds, model/checkpoint identities, hyperparameters, timing scopes, statistical tests, exclusions, and the claim gate.
4. Freeze all selector thresholds, action-score weights, peak prominence, temporal NMS, CLIP checkpoint, optical-flow parameters, graph edge weights, PPR damping, query-seed weight, prompt, answer model, and decoding settings using training/development data only.
5. Record the Git commit and config hashes before the first held-out run. Never retune after inspecting held-out results.

Correct the scientific descriptions before benchmarking:

- Describe the implemented low-level signal as **coded packet size / codec metadata**, unless direct transform residuals are actually extracted and verified. Do not call packet size DCT or transform-residual energy.
- Do not call local peak prominence "persistent homology" unless the production path actually computes and consumes a topological persistence algorithm.
- Do not claim selective decoding if the current second pass still decodes every frame. In that case, claim only reduced expensive pixel/model processing and report full decode cost.
- Remove any dataset-specific caption vocabulary or hard-coded concepts, including the existing rabbit/meadow-style label filter, from evaluation paths.
- Do not attribute gains to cache eviction, tiered indexing, verification, or another component unless that exact path is exercised in the measured run.

## Phase 2 — connect to and prepare the remote server

The laptop is the SSH controller. All large downloads, extraction, preprocessing, inference, and full experiments run on the server.

Recommend this laptop-side SSH configuration, substituting real values without printing secrets:

```sshconfig
Host paper-gpu
    HostName <SSH_HOST>
    User <SSH_USER>
    Port <SSH_PORT>
    IdentityFile <SSH_KEY>
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

Connect with `ssh paper-gpu`. Never commit credentials, print them in logs, copy the private key to the server, or put access tokens in command-line arguments when a protected credential store is supported.

On the server, record this preflight before installing or downloading anything:

```bash
hostname
whoami
uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
df -h "${REMOTE_ROOT}"
df -i "${REMOTE_ROOT}"
quota -s || true
free -h
lscpu
ffmpeg -version
ffprobe -version
git --version
python3 --version
```

Estimate archive, extracted-data, cache, feature, and result sizes with at least 20% free headroom. As a planning guide, the three core raw-video datasets can require several hundred GB together; adding ActivityNet, Ego4D, or LongVideoBench can push the requirement well beyond 1 TB. Use actual provider/download estimates, not the guide, for the final decision. Stop before a download only if storage, license access, or server policy is genuinely blocking.

Detect Slurm, PBS, or another scheduler. If present, use proper GPU jobs/job arrays and never run heavy work on a login node. Discover valid account, partition, wall time, CPU, RAM, and GPU settings instead of guessing. If there is no scheduler, use a persistent session:

```bash
tmux new-session -A -s paper-benchmark
```

Every long operation must run inside the scheduler or `tmux`. Verify one short job survives detach/reconnect before starting full runs.

Create isolated storage, adapting only the root path:

```bash
mkdir -p "${REMOTE_ROOT}/video_paper"/{src,data/archives,data/raw,data/processed,cache,env,runs,logs}
```

Raw validated videos are immutable. Never put data, archives, checkpoints, caches, credentials, or run bundles inside Git. Never transcode an official video before the primary codec-metadata experiment; the official encoded bitstream is part of the input signal.

Clone the requested branch on the server and record its exact commit:

```bash
git clone --branch pillar4-diagnostics --single-branch https://github.com/SiRex750/IRIS.git "${REMOTE_ROOT}/video_paper/src/model"
cd "${REMOTE_ROOT}/video_paper/src/model"
git rev-parse HEAD
git status --short --branch
```

If the approved source must instead be transferred from the laptop, use resumable `rsync` without `--delete` and exclude datasets, caches, results, `.git`, and bytecode.

## Phase 3 — reproducible environment and mandatory hardware smoke tests

Use an isolated conda/micromamba environment if available, otherwise `venv`; never modify system Python or use `sudo` without explicit permission. Respect repository pins. Install the official PyTorch build compatible with the installed NVIDIA driver; do not infer the wheel solely from the "CUDA Version" shown by `nvidia-smi`.

The environment must include the repository requirements plus PyAV/FFmpeg bindings, `opencv-python-headless`, scikit-learn, SciPy, pandas, PyArrow, psutil, pynvml, and the exact CLIP dependency already used by the project. Pin OpenAI CLIP's Git commit instead of installing an unversioned repository head. Keep OpenAI CLIP `ViT-B/32` as the mandatory common encoder unless the preregistration explicitly freezes another encoder for every method. Save `pip freeze`, `pip check`, Python version, CUDA/PyTorch details, and a lock/environment file in every run.

Set server-local caches outside Git:

```bash
export HF_HOME="${REMOTE_ROOT}/video_paper/cache/huggingface"
export TORCH_HOME="${REMOTE_ROOT}/video_paper/cache/torch"
export XDG_CACHE_HOME="${REMOTE_ROOT}/video_paper/cache/xdg"
```

Before any benchmark, save a diagnostic proving all of the following:

- `torch.cuda.is_available()` is true, expected GPU name/count/VRAM are visible, and a CUDA matrix multiplication succeeds;
- the pinned CLIP checkpoint loads on CUDA and encodes a non-empty image batch and text batch;
- FFmpeg, ffprobe, PyAV, and OpenCV decode representative files from every dataset;
- mandatory OpenCV Farneback flow processes a short video and returns finite scores;
- if the optional learned-flow baseline is enabled, its pinned RAFT checkpoint processes a batch on CUDA;
- peak allocated GPU memory is captured;
- `cv2.getBuildInformation()` is saved. Do not label pip OpenCV/Farneback as GPU accelerated when it is CPU-only.

Missing CLIP, OpenCV, or a required checkpoint is a failed setup—not an unavailable leaderboard row. Repair the environment before proceeding.

## Phase 4 — acquire and extract official datasets on the server

Download directly on the server in `tmux` or a scheduler job. Follow every dataset license and gated-access workflow. Use only official releases/links for paper results. Use `wget -c`, `curl --continue-at -`, `gdown`, Hugging Face CLI, or the official dataset CLI so downloads can resume.

If Google Drive quota or authentication prevents a server-side `gdown`, use the official link in an authenticated laptop browser and transfer that unchanged archive with `rsync -avhP --append-verify`. Never replace it with an unverified mirror.

After every dataset:

1. hash the archive or provider parts with SHA-256;
2. test the archive before extraction;
3. extract into a new versioned staging directory, never over a validated copy;
4. count files and annotations;
5. run `ffprobe` plus a complete decode validation;
6. create a manifest with video ID, split, path, bytes, SHA-256, codec, pixel format, resolution, duration, average and real frame rate, frame count, time base, and decode status;
7. use decoded presentation timestamps (PTS), not `frame_index / FPS`, for all ground-truth comparisons;
8. freeze one method-independent valid-video manifest and disclose requested, found, valid, corrupt, unavailable, and evaluated counts.

### Core dataset A: NExT-GQA (mandatory)

This is the primary grounded VideoQA/evidence benchmark. Preserve the original MP4 files; the official 6-fps feature extraction is not a replacement for the encoded-video input needed here.

Official source: `https://github.com/doc-doc/NExT-GQA`

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw"
git clone https://github.com/doc-doc/NExT-GQA.git nextgqa_annotations
mkdir -p nextgqa_videos
cd nextgqa_videos
gdown --fuzzy 'https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view' -O NExTVideo.zip
unzip -t NExTVideo.zip
mkdir -p extracted
unzip -q NExTVideo.zip -d extracted
```

Use the official train/validation/test CSVs, grounding-subset JSON, frame/time mapping, and QA-video-to-source mapping from the cloned repository. Do not tune on the held-out split.

### Core dataset B: QVHighlights (mandatory)

This supplies over ten thousand query/video pairs with temporal moment and 2-second-clip saliency supervision.

Official source: `https://github.com/jayleicn/moment_detr`

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw"
git clone https://github.com/jayleicn/moment_detr.git qvhighlights_annotations
mkdir -p qvhighlights_videos
cd qvhighlights_videos
wget -c https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz
tar -tzf qvhilights_videos.tar.gz >/dev/null
mkdir -p extracted
tar -xzf qvhilights_videos.tar.gz -C extracted
```

Use the JSONL annotations in the official repository. Test labels are withheld; tune on train/development data and evaluate the frozen configuration on validation, or use the official evaluation server where appropriate. Record the CC BY-NC-SA 4.0 annotation terms.

### Core dataset C: Charades-STA (mandatory cross-domain validation)

Official sources: `https://prior.allenai.org/projects/charades` and `https://github.com/jiyanggao/TALL`

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw"
mkdir -p charades_sta
cd charades_sta
wget -c https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip
unzip -t Charades_v1_480.zip
mkdir -p videos
unzip -q Charades_v1_480.zip -d videos
gdown --fuzzy 'https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view?usp=sharing' -O charades_sta_train.txt
gdown --fuzzy 'https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view?usp=sharing' -O charades_sta_test.txt
```

Parse each STA line as `[video_id] [start_seconds] [end_seconds]##[sentence]`. Follow the official non-commercial research/evaluation license and do not redistribute videos or modified data.

### Core dataset D: VIRAT Ground Release 2.0 (mandatory surveillance-domain validation)

Official access page: `https://viratdata.org/`

Official Release 2.0 Kitware collection: `https://data.kitware.com/#collection/56f56db28d777f753209ba9f`

Use the **Ground Camera Release 2.0** videos and matching event annotations, not the unannotated aerial release. The ground videos are protected by the VIRAT Video Dataset Usage Agreement. Every person with access must accept the agreement; protect personally identifiable information, do not attempt identification, and do not redistribute the videos to unauthorized people. Download only through the official Kitware collection after acceptance.

After accepting the agreement in the authenticated browser, create a restricted Kitware/Girder API key and download the official collection directly on the GPU server. Never paste the key into the prompt, repository, `commands.sh`, or logs; ensure shell tracing is disabled before reading it:

```bash
mkdir -p "${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2"/{download,archives,extracted,annotations,tmp}
python -m pip install -U girder-client
set +x
read -rsp "Kitware API key: " GIRDER_API_KEY
export GIRDER_API_KEY
echo
export TMPDIR="${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2/tmp"
girder-client \
  --api-url https://data.kitware.com/api/v1 \
  download \
  --parent-type collection \
  56f56db28d777f753209ba9f \
  "${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2/download"
unset GIRDER_API_KEY
```

Girder normally downloads the provider hierarchy recursively, so extra extraction may not be necessary. List file types, hash every received provider file, and use the downloaded hierarchy directly when it already contains videos and annotations:

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2"
find download -type f -exec file {} \; > file_types.log
find download -type f -print0 | sort -z | xargs -0 sha256sum > official_download.sha256
```

If API-key download is not permitted by the account, use the authenticated browser to download the same official collection/provider files and transfer them unchanged with `rsync -avhP --append-verify` to `archives/`. Do not use Kaggle or another mirror. If any provider files are archives, test the actual archive type before extracting into a distinct empty directory:

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2"

# For each provider ZIP file:
unzip -t "archives/<exact-provider-file>.zip"
unzip -q "archives/<exact-provider-file>.zip" -d extracted

# For each provider tar.gz file instead:
tar -tzf "archives/<exact-provider-file>.tar.gz" >/dev/null
tar -xzf "archives/<exact-provider-file>.tar.gz" -C extracted

# For each provider 7z file instead:
7z t "archives/<exact-provider-file>.7z"
7z x "archives/<exact-provider-file>.7z" -o"extracted/<unique-provider-file-name>"
```

Do not execute an extraction example for an archive type that was not downloaded. Record the exact provider file names and hashes. Clone the official DIVA-era annotations, or download their official GitLab archive in the browser if GitLab blocks automated clients:

```bash
git clone https://gitlab.kitware.com/viratdata/viratannotations.git \
  "${REMOTE_ROOT}/video_paper/data/raw/virat_ground_r2/annotations/diva"
```

VIRAT has multiple annotation releases. Do **not** mix Release 2.0 event files with DIVA annotations without an explicit, tested mapping. Select one matching video/annotation release in the preregistration, record its commit/archive SHA, and validate every annotated video ID. Prefer the official Release 2.0 train/test or scene-independent folds for the primary analysis; if the DIVA train/validate annotations are used, state that precisely. Parse temporal event intervals from the official event localization and count every event once.

Use VIRAT in Track A for query-independent event-window coverage, Frames-in-Window, nearest-event distance, retention, timing, and per-activity results. It is not a VideoQA dataset. For the query-aware CLIP control in Track B, convert only the official activity definition into one frozen text template per class and use it unchanged for every instance of that class; the proposed selector remains query-independent. Do not create favorable prompts after viewing results. Its 2--30 Hz videos require decoded PTS rather than nominal-FPS timestamp conversion.

### External dataset E: ActivityNet Captions (run if official videos are available)

Official access page: `https://activity-net.org/download.html`

```bash
cd "${REMOTE_ROOT}/video_paper/data/raw"
mkdir -p activitynet_captions
cd activitynet_captions
wget -c https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip -t captions.zip
mkdir -p annotations
unzip -q captions.zip -d annotations
```

The annotation JSONs contain train/val splits, duration, timestamp intervals, and sentences. Obtain raw videos through the official ActivityNet download page/request form. The access link is time-limited and some original YouTube videos are unavailable. If server-side authorization cannot be completed, download the authorized archive once on the laptop and transfer it with resumable `rsync -avhP --append-verify`; do not use unofficial mirrors. As a subset alternative, use FiftyOne's official integration, freezing the exact downloaded ID manifest:

```python
import fiftyone.zoo as foz
dataset = foz.load_zoo_dataset(
    "activitynet-200",
    split="validation",
    dataset_dir="<REMOTE_ROOT>/video_paper/data/raw/activitynet_200",
)
print(dataset)
```

Never silently discard unavailable videos, and never use withheld test ground truth.

### External dataset F: Ego4D NLQ v2 (strong gated stretch test)

First have the user accept the official Ego4D agreement. Put AWS credentials in a named server-side profile without logging them. Download the filtered NLQ clips, not the multi-terabyte full dataset:

Official access and CLI documentation: `https://ego4d-data.org/docs/start-here/` and `https://ego4d-data.org/docs/CLI/`

```bash
python -m pip install -U ego4d
ego4d \
  --output_directory "${REMOTE_ROOT}/video_paper/data/raw/ego4d" \
  --datasets clips annotations \
  --benchmarks NLQ \
  --version v2 \
  --aws_profile_name ego4d \
  -y
```

If the installed CLI differs, inspect `ego4d --help` and use the current official equivalent while preserving `clips + annotations + NLQ + v2`. Save the command and CLI version.

### Optional end-to-end dataset G: LongVideoBench

Use this only for long-video QA accuracy and compute efficiency, not as evidence for selector superiority because public temporal evidence windows are not provided.

Official source: `https://github.com/longvideobench/LongVideoBench`

After accepting the Hugging Face dataset terms on the laptop and authenticating securely on the server:

```bash
python -m pip install -U huggingface_hub
huggingface-cli download longvideobench/LongVideoBench \
  --repo-type dataset \
  --local-dir "${REMOTE_ROOT}/video_paper/data/raw/longvideobench" \
  --local-dir-use-symlinks False
cd "${REMOTE_ROOT}/video_paper/data/raw/longvideobench"
cat videos.tar.part.* > videos.tar
tar -tf videos.tar >/dev/null
mkdir -p videos subtitles
tar -xf videos.tar -C videos
tar -tf subtitles.tar >/dev/null
tar -xf subtitles.tar -C subtitles
```

If disk cannot hold both the parts and merged tar, hash every part and use `cat videos.tar.part.* | tar -xf - -C videos`, recording that extraction path. Use `lvb_val.json` for public-label evaluation; do not infer test answers.

Do not make ActivityNet, Ego4D, or LongVideoBench a hidden prerequisite for completing the four-dataset core study. VIRAT is part of the requested core suite; if its protection-agreement access is still pending, mark VIRAT `BLOCKED_EXTERNAL`, continue the other datasets sequentially, and report that the complete requested suite is not yet finished.

### Mandatory one-dataset-at-a-time execution rule

Process and test datasets strictly one at a time in this order unless the preregistration documents a justified change:

```text
NExT-GQA -> QVHighlights -> Charades-STA -> VIRAT Ground R2
          -> ActivityNet Captions -> Ego4D NLQ -> LongVideoBench
```

Only one dataset may be in an active acquisition, validation, preprocessing, smoke, pilot, full-run, aggregation, or statistics stage. Do not benchmark two datasets concurrently and do not use spare GPUs to start the next dataset. Multiple GPUs or CPU workers may shard videos **within the currently active dataset**, but latency measurements must still run without competing jobs.

For each dataset, complete this entire state machine before advancing:

```text
ACQUIRE -> VALIDATE -> SMOKE_ALL_METHODS -> PILOT_ALL_METHODS
        -> FULL_RUN -> AGGREGATE -> STATISTICS -> FIGURES_TABLES
        -> VERIFY_ARTIFACTS -> SYNC_TO_LAPTOP -> DATASET_COMPLETE
```

Maintain `runs/<run_id>/CURRENT_DATASET.json` with dataset, stage, start time, last update, job IDs, completed/total units, and status. Use a run lock so a second dataset runner exits rather than overlapping. A dataset may advance only after every mandatory baseline has completed on its frozen common set, its checksums pass, its human-readable summary and results index are updated, and its result bundle has been synchronized to the laptop. If gated access is the only blocker, write a `BLOCKED_EXTERNAL` record with the exact required user action before moving to the next dataset.

## Phase 5 — implement one auditable benchmark harness

Add a dedicated benchmark package, configs, and tests without duplicating the production algorithm. It must provide:

- dataset adapters with one canonical schema;
- manifest creation/validation;
- selectors behind one interface;
- shared retrieval and graph-ablation runners;
- end-to-end QA adapter if `ANSWER_MODEL_ID` is supplied;
- structured per-item output, resume, aggregation, statistics, tables, and plots;
- one CLI/config entry point for smoke, pilot, validation, and held-out runs.

Canonical per-query fields must include dataset, split, video ID, question/query ID, query text, answer choices/label where applicable, all ground-truth intervals, video PTS table, method, seed, requested budget, actual `K`, ordered unique selected timestamps/frame IDs/scores, retrieval ranking, metrics, component timings, device, and failure state.

Write results incrementally per video/question with atomic temporary-to-final renames. A restart must validate and skip completed items. Cache keys must include video SHA-256, code/config hash, model/checkpoint, preprocessing, resolution, and method. Keep accuracy caches separate from cold/warm timing measurements.

Required assertions and tests:

- exactly `K` unique valid frames for every matched-budget method;
- timestamp monotonicity and PTS/frame mapping correctness, including variable-frame-rate cases;
- deterministic reruns for deterministic methods;
- random sampling without replacement and seed reproducibility;
- no NaN/Inf/zero-length score arrays;
- `K=1`, `K=N`, `K>N`, one-frame video, short video, empty/corrupt decode, no I-frame metadata, and no scene boundary;
- CLIP K-means handles `K > number_of_valid_embeddings`, duplicate nearest-centroid representatives, and empty embeddings by a defined retry/fail path rather than `argmin` on an empty sequence;
- graphless and graph variants receive byte-for-byte identical candidate pools;
- metric unit tests with hand-computed examples;
- aggregation counts every event/question once rather than multiplying bucket totals by method or seed.

Never put `-1`, NaN, or a fabricated zero into a leaderboard for a missing method. A mandatory method failure makes the run incomplete and must be repaired and rerun.

## Phase 6 — selectors and baselines that must actually run

All methods operate on the same official valid videos and must return exactly the same output budget `K_v` for video `v`. Tie-break by lower PTS and then lower decoded frame index. Apply the same post-selection temporal-spacing/NMS rule to every score-ranking method, or provide both native and common-NMS results. Do not give only the proposed method a diversity advantage.

Mandatory selectors:

1. `uniform`: evenly spaced over the complete valid PTS/frame universe.
2. `random`: sample without replacement for seeds `0..29`.
3. `iframe_prior`: if there are at least `K` I-frames, choose `K` uniformly across that ordered list; otherwise take all I-frames and deterministically fill from uniformly spaced non-I frames. Report native all-I-frame behavior separately, never mixed into matched-budget results.
4. `scene_change`: pinned PySceneDetect content score; rank boundaries/local maxima and deterministically fill shortages.
5. `luma_difference`: mean absolute consecutive-frame Y-channel difference.
6. `optical_flow_farneback`: the mandatory classical baseline matching the earlier report. Install OpenCV and run it on CPU. Unless an existing frozen implementation already defines the parameters, resize while preserving aspect ratio so the long side is 320 pixels, convert to grayscale, and use `cv2.calcOpticalFlowFarneback` with `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`. Score each destination frame by the spatial mean flow magnitude; frame zero receives the minimum score. Include decoding and all frame-pair computation in cold end-to-end time.
7. `clip_kmeans_diversity`: use the pinned OpenAI CLIP `ViT-B/32` image encoder on the complete frame universe, not a candidate list created by the proposed selector. L2-normalize finite embeddings. For the full-scale run, use deterministic scikit-learn `MiniBatchKMeans` with frozen `random_state`, `n_init=10`, `max_iter=100`, `batch_size=min(4096, N)`, `reassignment_ratio=0`, and Euclidean distance on normalized embeddings; describe it accurately as mini-batch K-means. On the stratified 1% audit subset, also run exact Lloyd K-means with `n_init=20` and quantify selection/metric agreement so the scalable approximation is not assumed harmless. If exact full-scale K-means is demonstrably feasible, it may be added as a separately named method. Assign centroids to unique real frames using a deterministic unique assignment (for example Hungarian assignment on centroid-to-frame distance), not independent nearest-neighbor choices that can duplicate a frame. Batch CLIP encoding on GPU, adapt batch size only for OOM, and never reduce resolution/model to rescue a sample without declaring a new method.
8. `clip_query_topk`: encode the query with the same CLIP checkpoint and rank the complete frame universe by cosine similarity. This is a strong query-aware control; do not conflate it with query-independent K-means diversity.
9. `packet_size_only`: rank by the actual coded-packet signal used by the implementation.
10. `proposed_action_selector`: production gate + action score + peak/NMS, with the frozen configuration.

Recommended strong additional motion control:

11. `optical_flow_raft`: a pinned RAFT-Small or RAFT-Large pretrained checkpoint on CUDA, with fixed weights, resize, precision, preprocessing, and score reducer. Label it as learned GPU optical flow and keep it separate from Farneback. Pin the implementation commit/checkpoint checksum.

For all full-frame baselines, the expensive input computation is allowed because output budget—not hidden compute—is matched. Report the additional input/decode/embedding/flow cost honestly. Also report an optional compute-matched control if it is rigorously defined, but never substitute it for the output-budget comparison.

## Phase 7 — two budget protocols

For video `v` with `N_v` valid decoded frames:

1. **Native matched budget:** run the proposed selector to obtain `K_v` unique frames. Every baseline returns exactly that `K_v` on the same video.
2. **Fixed budget sweep:** run every selector at retention budgets `B in {2%, 5%, 10%, 15%, 20%}` with `K_v(B) = min(N_v, max(5, round(B*N_v)))`. Predeclare 10% as the primary fixed budget and report actual retention and selected frames/minute. The `max(5, ...)` floor exists only for metrics at rank 5 and must be disclosed.

Never compare quality at different output budgets. Do not report Recall@10 when only 8 results were requested. A rank metric is valid only when the requested/retrieved depth supports it; record the eligible denominator and never duplicate-pad results.

## Phase 8 — remove selection/retrieval/answering confounds

Run and label four separate tracks:

### Track A: selector-only

Evaluate the selected timestamps directly against all ground-truth evidence/moment intervals. No PPR, query ranker, captioner, or answer model is allowed in this track. This supports only a frame-selection claim.

### Track B: shared retrieval

Pass each selector's `K` output frames to the identical pinned CLIP text/image encoder, cosine scorer, temporal post-processing, and top-r retrieval code. The proposed selector does not use PPR in this track. This isolates the effect of the selected pool.

### Track C: graph ablation

Hold the proposed selected frames, embeddings, query, budget, and top-r fixed. Compare:

- graphless CLIP cosine ranking;
- semantic-only PPR;
- codec-only PPR;
- hybrid semantic+codec PPR;
- the full graph with each edge family removed one at a time;
- flat versus hierarchical graph if both are genuine production paths.

Tune graph/PPR settings only on development data. Do not assume codec weighting at query time helps; prior diagnostics suggest it can hurt grounding.

### Track D: end-to-end QA (secondary)

Only if a fixed answer model is supplied, pass each method's evidence through the exact same captioner/VLM, prompt, answer options, precision, batch size, decoding parameters, and verifier. Cache prompts, retrieved evidence, model outputs, and answers. NExT-GQA is suitable for grounded QA. LongVideoBench is an optional long-context QA/efficiency test but not selector-superiority evidence.

Do not combine Track A, B, C, and D into one method name or one causal claim.

## Phase 9 — metrics

Predeclare this primary selector endpoint:

```text
WindowCoverage@K at 10% retention = fraction of queries for which at least one
selected timestamp lies inside at least one ground-truth evidence interval.
```

Report it on NExT-GQA, QVHighlights, Charades-STA, and VIRAT. Secondary selector metrics:

- Frames-in-Window at K: fraction of selected frames lying inside any valid evidence interval;
- evidence/moment coverage when multiple intervals exist;
- minimum temporal distance to evidence;
- selected-frame temporal diversity;
- results by short/medium/long event duration, video duration, question/query type, codec, resolution, bitrate, GOP length, and frame rate.

Shared ranked-retrieval metrics:

- Recall@1 and Recall@5;
- FiW@5, defined as `1/5 * sum_{r=1..5} 1[t_r in any ground-truth interval]`;
- MRR, mAP, and nDCG@5 where labels support them;
- QVHighlights saliency metrics using the official 2-second clip mapping.

Grounded-QA metrics where supported:

- official QA accuracy;
- mIoP, IoP@0.3/0.5, mIoU, IoU@0.3/0.5, and official Acc@GQA;
- exact multiple-choice accuracy for end-to-end QA.

Use official moment-retrieval metrics such as R@1/R@5 at IoU 0.3/0.5/0.7 only if the method genuinely emits intervals through one common, frozen interval-construction rule. Do not label a point-frame heuristic as temporal IoU and do not repeat the earlier "temporal IoU proxy" as if it were an official metric.

Efficiency metrics:

- codec demux, full decode, selector feature computation, CLIP embedding, optical flow, graph construction, index construction, query retrieval, caption/answer, and total wall time;
- cold-cache and warm-cache timings separately;
- offline build cost and online per-query latency separately, plus amortized cost for 1, 5, and 10 queries/video;
- frames decoded, frames receiving expensive pixel/model processing, and frames returned;
- videos/hour, frames/second, peak GPU VRAM, peak CPU RAM, GPU-hours, and index/cache size;
- GPU power/energy if telemetry is reliable.

Use `time.perf_counter()` and call `torch.cuda.synchronize()` immediately before and after timed CUDA regions. Do not compare a cached proposed run with an uncached baseline unless both cold and warm results are reported. Do not run competing jobs during latency trials. For multi-GPU throughput, shard deterministically by video, record the GPU for each shard, and measure single-GPU latency separately.

## Phase 10 — statistics

- Use all eligible queries but resample at the **video level**, keeping questions from one video together.
- Generate 95% confidence intervals with 10,000 paired video-cluster bootstrap replicates and fixed bootstrap seed `20260710`.
- For paired method differences, use paired cluster bootstrap or a video-level paired permutation test. Use McNemar's test for paired binary QA correctness where appropriate.
- Average the 30 random-selector seeds per video before comparison with deterministic methods; also report between-seed standard deviation and confidence interval.
- Correct the family of comparisons against baselines with Holm-Bonferroni.
- Report absolute difference, relative difference, 95% CI, adjusted p-value, effect size, video count, and query count.
- Predeclare a practically meaningful improvement of at least 3 absolute percentage points on the primary metric.
- Include a worst-case sensitivity analysis for any irrecoverable common-set exclusions.

Do not write "significant" without the named test, corrected p-value, and confidence interval. Do not use an unclustered question-level t-test when multiple questions share a video.

## Phase 11 — required ablations and codec robustness

Run at least:

- packet-size only, motion only, entropy only;
- each action-score component removed in turn;
- gate removed while retaining action scoring;
- peak prominence removed and NMS removed;
- proposed selector versus graphless retrieval versus graph variants;
- every graph edge family removed in turn;
- fixed-budget sweep;
- common NMS versus native selection behavior;
- captioner/answerer and verifier ablations only if those components are part of the claimed end-to-end system.

Because the proposed input is codec-derived, add a controlled encoding-robustness study on a fixed, predeclared subset. Keep semantic video content identical and re-encode copies at several H.264 CRF values and GOP lengths; optionally add H.265/AV1 if the production parser truly supports them. The primary benchmark remains the untouched official bitstream. Report how selection overlap and grounding quality change across encodes, and stratify the main results by codec/bitrate/GOP. Do not hide encoding sensitivity.

## Phase 12 — execution ladder and failure policy

Execute the following ladder for the currently active dataset only; finish, verify, index, and sync that dataset before activating the next one:

1. unit/integration tests;
2. one-video smoke test for every dataset and every mandatory method;
3. stratified 1% pilot, including short/medium/long events and multiple codecs;
4. complete official validation/development run;
5. freeze commit/config and create the preregistration snapshot;
6. complete held-out/test run where labels or an official server permit it;
7. aggregate/statistics/tables/figures;
8. reproduce a small run from a fresh environment or clean server directory.

Do not start the full benchmark until Farneback and both CLIP baselines pass the same smoke and pilot samples as the proposed method.

For OOM, retry with a smaller batch while keeping model, weights, resolution, precision policy, and scoring unchanged. Do not silently fall back to CPU or a weaker model. A required method must not drop a difficult video. Repair and rerun it. If a video is irrecoverably corrupt for every method, remove it only through the method-independent valid-manifest rule and report it. If failure is method-specific, the comparison remains incomplete until fixed, or it must be explicitly reported without a superiority claim.

Every failure record must include dataset, split, video/query ID, method, seed, exception, retry count, host, device, config hash, and final state.

## Phase 13 — artifacts and paper-ready output

No result may exist only in terminal output. Every metric value, selected-frame list, timing, statistical test, warning, failure, table, and graph must be written to an easy-to-access file. Each immutable run directory must contain a top-level human-readable index and a self-contained per-dataset bundle:

```text
runs/<run_id>/
  RESULTS_INDEX.md
  RESULTS_INDEX.html
  LATEST_STATUS.json
  CURRENT_DATASET.json
  progress.log
  artifact_manifest.csv
  all_datasets_summary.csv
  all_datasets_summary.json
  preregistration.yaml
  manifest.json
  config.yaml
  git_state.txt
  environment.txt
  requirements-lock.txt
  hardware.json
  dataset_manifest.parquet
  gpu_telemetry.csv
  claims.json
  logs/
  SHA256SUMS
  <dataset_name>/
    README.md
    STATUS.json
    commands.sh
    summary.md
    summary.json
    metrics.csv
    metrics.json
    timings.csv
    timings.parquet
    statistical_tests.csv
    statistical_tests.json
    failures.csv
    failures.jsonl
    selections/<method>.parquet
    retrievals/<method>.parquet
    answers.jsonl
    tables/*.tex
    tables/*.csv
    figures/FIGURE_INDEX.md
    figures/figure_manifest.csv
    figures/*.pdf
    figures/*.png
    figure_data/*.csv
    logs/run.log
    logs/events.jsonl
    logs/gpu_telemetry.csv
    SHA256SUMS
```

`RESULTS_INDEX.md` is the main entry point a team member should open. It must show the active/completed/blocked state of every dataset, exact sample counts, links to each dataset's summary, metrics, timings, failures, tables, figures, raw per-item files, logs, Git SHA, config, and checksums. Generate `RESULTS_INDEX.html` from the same data so it can be opened locally without a server. `LATEST_STATUS.json` must be machine-readable.

For every graph, save both PDF and PNG plus the exact source values in `figure_data/<figure_name>.csv`. `FIGURE_INDEX.md` must display/link each PNG, link the matching PDF and CSV, and state the metric, dataset, methods, uncertainty interval, and generation command. `figure_manifest.csv` and the top-level `artifact_manifest.csv` must contain relative path, artifact type, dataset, method, metric, description, generation time, file size, and SHA-256. Never make a graph whose plotted values cannot be recovered from a nearby CSV/Parquet file.

Capture complete human-readable stdout/stderr in each dataset's `logs/run.log`, and structured progress/metric/failure events in `logs/events.jsonl`. Use `set -o pipefail` whenever a command is logged through `tee`, so a failed benchmark is not mistaken for success:

```bash
set -o pipefail
python -m <benchmark_entrypoint> --dataset <active_dataset> 2>&1 \
  | tee -a "runs/<run_id>/<dataset_name>/logs/run.log"
```

Start periodic GPU telemetry, for example with `nvidia-smi dmon`, and retain scheduler job IDs or the tmux session name. Generate every result table and plot directly from per-item artifacts—never by manually copying numbers. Update the indexes atomically after every completed stage and at least every 60 seconds during long stages.

Produce:

1. a concise reproducibility README with exact commands;
2. a dataset availability/failure table;
3. primary matched-budget table with 95% CIs;
4. fixed-budget curves with confidence bands;
5. selector-versus-shared-retrieval-versus-graph ablation tables;
6. quality/retention/latency/VRAM trade-off plots;
7. short/medium/long event analysis;
8. codec/bitrate/GOP robustness analysis;
9. a claim matrix marking each proposed paper claim as supported, unsupported, or not tested;
10. publication-ready LaTeX tables and vector PDF figures using **Proposed Model** labels only;
11. an honest `RESULTS_AND_CONCLUSION.md` that states the strongest defensible conclusion and limitations.

After **each dataset**, sync only code changes, manifests, logs, configs, per-sample outputs, aggregate files, tables, plots, indexes, and reports back to the laptop with resumable `rsync`. Do not wait until all datasets finish, and do not copy raw datasets or model caches. Mirror the bundle under a stable laptop directory such as `paper_results/<run_id>/`, update `paper_results/LATEST_RUN.txt`, and verify `SHA256SUMS` before and after transfer.

## Claim gate

The narrow claim **"the proposed model provides superior matched-budget frame selection"** is permitted only if all of the following are true:

1. all four mandatory datasets (NExT-GQA, QVHighlights, Charades-STA, and VIRAT Ground R2) and every applicable mandatory baseline finish on the same frozen per-dataset valid-video sets; any scientifically inapplicable method is labeled `NOT_APPLICABLE` with a written reason and is never encoded as a numeric score;
2. output budgets match exactly for every video and no invalid/missing method values enter aggregation;
3. the proposed selector beats the strongest predeclared baseline on `WindowCoverage@K` at 10% retention;
4. the paired, video-clustered, Holm-corrected 95% CI for that improvement excludes zero;
5. the absolute improvement is at least the predeclared 3 percentage points;
6. the improvement direction replicates on at least two independent mandatory datasets;
7. no major short-event, category, codec, or encoding-condition regression is hidden;
8. the efficiency measurements support every speed/compute claim; and
9. all raw artifacts, failures, exclusions, configs, checkpoints, hashes, and environment details are archived and reproducible.

Graph superiority is a separate claim and requires the frozen-candidate Track C paired comparison to pass the same uncertainty/effect-size discipline. End-to-end QA superiority is also separate and requires a fixed common answer model plus official QA metrics.

If the gate fails, do not search for a more favorable subset, seed, budget, or metric. Report the result accurately, identify the limiting cases, and recommend the next scientifically valid experiment.

## Definition of done

Do not declare completion after creating scripts. Completion requires:

- a clean mandatory-baseline smoke test including CPU Farneback, GPU CLIP encoding plus deterministic mini-batch K-means, and GPU query-CLIP;
- validated official dataset manifests;
- full four-dataset core evaluation; if VIRAT access is pending, an explicit `BLOCKED_EXTERNAL` record and an honest statement that the requested core suite remains incomplete;
- exact budget and common-sample checks passing;
- complete paired statistical analysis;
- paper-ready tables/figures and claim matrix;
- reproducible run commands and environment locks;
- per-dataset `README.md`, metrics files, graph source files, complete logs, and checksums linked from `RESULTS_INDEX.md` and `RESULTS_INDEX.html`;
- each completed dataset safely synchronized back to the laptop before the next dataset starts.

At the end, report the run IDs, Git SHA, exact datasets/splits/counts, completed and failed methods, result-bundle path, primary estimates with 95% CIs, adjusted p-values, practical effect sizes, efficiency results, and which claims—if any—passed the gate.
