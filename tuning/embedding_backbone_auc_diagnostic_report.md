# Embedding-backbone AUC diagnostic — does a stronger image-text model raise the gold-frame separability ceiling?

Pure measurement, no production change. Follow-on to the motion diagnostic
(commit `3943181`), which found CLIP ViT-B/32's own gold-vs-non-gold AUC is
only 0.580 — the ceiling bucket (c) (64% of the Acc@GQA failure group,
c24f8f8) is capped by. This asks the one question that decides whether a
full backbone swap is worth the re-ingest + re-tune cost: does a *stronger*
image-text model actually separate gold frames better, on the same
frames/questions?

Script (not committed — ephemeral, same convention as prior diagnostics):
`embedding_backbone_diagnostic.py`. Raw outputs committed:
`tuning/embedding_backbone_step0_availability.json`,
`tuning/embedding_backbone_step2_results.json`,
`tuning/embedding_backbone_bootstrap_ci.json`,
`tuning/embedding_backbone_per_question.csv`. `frozen_state.json` and all
production code are untouched; no full re-ingest was run.

## Step 0 — offline weight availability

The box runs `HF_HUB_OFFLINE=1`; only weights already resident locally
are usable without a download.

| backbone | available? | model id / revision | notes |
|---|---|---|---|
| **control: CLIP ViT-B/32** | **yes** | `openai/CLIP@d05afc4`, weight sha256 prefix `40d365715913c9da` | production backbone (matches `environment.json`'s recorded revision and the known-good OpenAI checksum) |
| SigLIP2-base-patch16-224 | yes | `google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | full weights + tokenizer present, loads cleanly offline |
| SigLIP2-so400m-patch14-384 | **no** | `google/siglip2-so400m-patch14-384` @ `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | vision tower + config present (4.3GB) but the text-tower tokenizer's `spiece.model` vocab file is absent from the local HF cache — `AutoProcessor.from_pretrained` fails (`TypeError` in `tokenization_siglip.py`'s `get_spm_processor`, `self.vocab_file is None`). Not re-attempted, not downloaded — this was the intended "strong" candidate (400M-param SoViT backbone) and its absence is the main limitation of this diagnostic (see Step 3). |
| OpenCLIP / EVA-CLIP (any) | **no** | — | `open_clip` package not installed; no OpenCLIP/EVA-CLIP checkpoint found anywhere on disk outside the HF hub cache |

One dependency note for transparency: loading SigLIP2-base initially failed
with `ModuleNotFoundError: sentencepiece` (a small pure tokenizer library,
not a model weight) — `pip install sentencepiece` succeeded (network access
for package installs is not blocked the same way HF weight downloads are;
`HF_HUB_OFFLINE=1` only gates the HF hub client). No model weights were
downloaded at any point; the so400m gap is a genuinely missing weight file,
not a missing package, and was left unresolved per the "don't stall trying
to fetch it" instruction.

**Two backbones available: control (CLIP ViT-B/32) + one candidate
(SigLIP2-base-patch16-224). Proceeding to Steps 1–2.** SigLIP2-base is
roughly the same parameter scale as CLIP ViT-B/32 (~86–93M vs ~88M params)
— it is a newer/better-trained model at comparable capacity, not a bigger
one. The originally-intended "clearly stronger, bigger" candidate
(so400m, ~400M params) could not be tested; this caveat matters for how
much weight the verdict below can carry (see Step 3).

## Step 1 — subset

27 val_confirm videos (the highest-question-count videos in the split,
deterministic tie-break by video id), covering **236 questions** (≥150
floor cleared). Frame population = each video's already-selected codec
survivor frames, loaded from `tuning/index_cache_val_confirm_e2e/`
(config-hash `4edae64ed40256e3`) — so codec frame selection is held fixed
and only the embedding model varies. Pixels re-decoded from source `.mp4`
via a single seek + forward-scan pass per video (same GOP-seek pattern as
`iris/query.py::_ensure_captions`), decoded once and reused across both
backbones. **5,638 frames decoded across the 27 videos** (100% of
requested survivor-frame targets landed — no decode misses).

## Step 2 — embed and score

Both backbones embedded the identical 5,638 frames (image tower) and 236
question texts (text tower, question only — never the answer options).
Cosine similarity computed pool-wise (every question × every frame in its
own video, 49,291 pairs both backbones).

| backbone | AUC (gold vs non-gold) | Cohen's d | approx retrieved-in-gold (top-4 raw cosine)† | top-1-in-gold (argmax raw cosine) |
|---|---|---|---|---|
| **control: CLIP ViT-B/32** | **0.5907** | 0.3156 | 55.51% | 33.47% |
| SigLIP2-base-patch16-224 | 0.5963 | 0.3037 | 58.05% | 35.17% |

† **Explicitly an approximation**, not a production retrieval-in-gold
prediction — this is flat top-4-by-raw-cosine over the subset's survivor
frames, with no PPR/graph, no `ppr_lambda` sem/codec blend, and no
`l2_retrieve_top_k` behavior. It measures raw embedding discriminative
power in isolation, not what production retrieval would actually return.

**Consistency check on the control**: 0.5907 on this 236-question subset
vs 0.5798 on the full 639-question population (motion diagnostic, commit
`3943181`) — close enough (Δ=0.011, within normal subset-sampling noise
for this sample size) to trust the subset is representative and the
measurement pipeline reproduces the known number. **Passes; no
inconsistency to report.**

## Bootstrap CI on the AUC delta (1000 draws, question-level resampling)

Resampled the 236 questions with replacement (paired across both
backbones on each draw, same resampled question set), recomputed pooled
AUC for each backbone per draw, and took the delta distribution.

| candidate | point Δ AUC (candidate − control) | bootstrap mean Δ | 95% CI | excludes zero? |
|---|---|---|---|---|
| SigLIP2-base-patch16-224 | **+0.0055** | +0.0054 | **[−0.0260, +0.0379]** | **No** |

The 95% CI comfortably contains zero. A 0.0055 point delta is not
distinguishable from noise at this sample size — this is exactly the
"0.58→0.59 wiggle" the task's decision gate was written to catch, not a
real lift.

## Step 3 — verdict and decision gate

Gate: candidate AUC needs a clear, meaningful lift over control (rough
bar ≥0.63) with a bootstrap CI that excludes zero, to justify a full
backbone swap.

**SigLIP2-base-patch16-224 does not clear the gate.** AUC 0.596 vs
control 0.591 — both essentially at the same ~0.58–0.60 ceiling, delta
statistically indistinguishable from zero (95% CI crosses zero by a wide
margin relative to the point estimate). Top-1-in-gold and the approximate
retrieved-in-gold rate both move a couple of points in SigLIP2's favor,
consistent with a mild, real-but-small edge, not the kind of jump that
would change the retrieval-failure story.

**Important scope limitation, stated plainly**: this result rules out
"swap to a same-capacity SigLIP2 model," not "no embedding model could
ever do better." The candidate this diagnostic was actually built to test
— SigLIP2-so400m-patch14-384, a ~400M-parameter SoViT backbone trained at
384px resolution, a substantially bigger and differently-trained model
than ViT-B/32 — could not be loaded offline (Step 0: missing tokenizer
vocab file) and was not tested. No OpenCLIP/EVA-CLIP ViT-L/14 candidate
was available either. **The honest conclusion is narrower than "backbone
swaps don't help":** the one candidate this environment could actually
test, at roughly matched capacity to the control, produced no significant
lift. That is meaningfully different from having tested and ruled out a
clearly-stronger model.

**Recommendation**: do not proceed with a full-pipeline backbone swap on
the strength of this result alone — the one clean same-capacity
comparison available offline came back negative (CI includes zero), so
there's no positive signal to act on right now. But this is not the
decisive negative the task's "no candidate clears the gate → stop chasing
retrieval, compete on efficiency" framing describes, because the
strongest available candidate was never actually reachable in this
environment. If the embedding-quality question is still worth settling
definitively, the concrete next step is narrow and cheap relative to a
full swap: get `spiece.model` for `google/siglip2-so400m-patch14-384`
(or an equivalent large/so400m-scale SigLIP2/EVA-CLIP checkpoint) onto
the box and rerun this exact script — Steps 1–2 need no changes, only
Step 0's availability check picks up the new candidate. Absent that,
treat the embedding-quality question as **open, not closed** — worth a
one-file weight-transfer, not worth a full re-ingest commitment yet.
