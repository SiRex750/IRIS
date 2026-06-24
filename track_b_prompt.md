# Antigravity prompt — Track B (aria.py + pipeline.py + action_score.py)

You are implementing **Track B** of IRIS (github.com/SiRex750/IRIS), a codec-native hierarchical memory system for video-language understanding. Track B owns the LLM abstraction and the end-to-end pipeline glue. I am extending Track B's scope to also include `action_score.py`, a new module that didn't exist in the original repo layout — it sits between `charon_v.py` and everything downstream, and `pipeline.py` is what will wire it in.

## Repo context (read this before writing anything)

Current repo state, for reference:
- `charon_v.py` — DONE. Outputs PEAK / SALIENT / CANDIDATE / SKIP tiers via two-pass adaptive threshold + `argrelextrema` local maxima on the residual energy curve.
- `l1_elysium.py` — ACTIVE (Track A, owned by Siddanth). Currently being ported from `cache.py`, with budget multipliers keyed off the PEAK/SALIENT/CANDIDATE/SKIP tiers from Charon-V.
- `l2_asphodel.py` — ACTIVE (Track C). NetworkX graph, node schema includes a `tier` field, hybrid retrieval = (semantic × α) + (motion × β).
- `cerberus_v.py` — ACTIVE (Track D). Ported from HADES, DeBERTa-v3 NLI backend.
- `iris_config.py` — NEXT (Track A). `IRISConfig` dataclass + `ConfigManager`, will load GEPA-tunable JSON params.
- `aria.py` — exists as a stub/early version. This is mine to build out.
- `pipeline.py` — not yet built. This is mine to build.

**Important tension you need to know about:** the current repo's design has Charon-V emitting discrete tiers, and downstream modules (l1_elysium, l2_asphodel) consuming those tiers directly. A separate architecture review concluded this should move toward Charon-V emitting only raw per-frame signal (no tiering), with a new `action_score.py` module owning all thresholding logic via persistence-filtered continuous scoring instead of discrete tiers. **This has not yet been adopted by the whole team** — Track A and Track C are actively building against the tiered design right now. So: build `action_score.py` as a new, additive module that can run *alongside* the existing tiered output for now (don't break Track A/C's in-flight work), but write it so the continuous score is the "real" output Track B's own `pipeline.py` uses internally. Flag in your output/comments anywhere a future migration would require Track A or C to change their tier-consuming code, so I can take that back to the team instead of silently deciding it for them.

## Part 1 — `action_score.py` (new module)

This module sits between `charon_v.py`'s raw signal and all downstream components. Build it as additive/parallel to the existing tier output, not a replacement (per the note above).

**Input contract** — a rolling buffer of per-frame records (these fields should already exist or be easy to expose from `charon_v.py`'s internals, even though Charon-V's *public* output today is just the tier):

```python
{
    "frame_idx":        int,
    "frame_type":       str,    # "I", "P", "B"
    "residual_energy":  float,
    "motion_magnitude": float,
    "entropy":          float
}
```

**Output contract** — emitted per frame:

```python
{
    "frame_idx":         int,
    "action_score":      float,  # persistence-filtered composite, normalised [0,1]
    "is_peak":           bool,   # continuous-score-based peak decision
    "persistence_value": float   # raw lifetime of the topological feature
}
```

**What to implement:**

1. `FrameFeatureBuffer` class — maintains a rolling window of Charon-V per-frame records. Window size is a tunable parameter (default 30 frames), loaded from `IRISConfig` if it exists yet, otherwise a plain constructor default (Track A's `iris_config.py` is still "Next," not done — don't hard-block on it).
2. `compute_action_score()` — runs single-parameter persistence on the scalar residual energy time-series using `scipy.signal.find_peaks` with prominence as the persistence proxy. This is Phase 1 (multi-parameter persistence across residual energy + motion + entropy jointly is Phase 2 — stub it as a TODO, don't build it now).
3. Composite score: `action_score = w1 * norm(residual_energy) + w2 * norm(motion_magnitude) + w3 * entropy`, with tunable defaults `w1=0.5, w2=0.3, w3=0.2`.
4. `is_peak` is `True` when `persistence_value >= persistence_thresh` (tunable default `0.4`).
5. All tunable parameters (window size, w1/w2/w3, persistence_thresh) should be structured so they can be swapped in from `IRISConfig`/`ConfigManager` later without changing this module's internals — accept them as constructor args with sane defaults now, don't hardcode them inline.
6. **Do not** implement any PEAK/SALIENT/CANDIDATE/SKIP tier logic here. This module's only output is `action_score` (continuous) + `is_peak` (boolean) + `persistence_value`. No re-classification into buckets.

**Sanity check to run after implementation:**

```python
from action_score import FrameFeatureBuffer, ActionScoreModule

buf = FrameFeatureBuffer(window_size=10)
module = ActionScoreModule()

# Simulate a pan shot — uniform motion, low entropy
for i in range(10):
    buf.push({"frame_idx": i, "frame_type": "P",
              "residual_energy": 0.4, "motion_magnitude": 0.4, "entropy": 0.05})
result = module.score(buf)
assert result["is_peak"] == False, "Pan shot should not be a peak"

# Simulate a complex action scene — high entropy spike
buf.push({"frame_idx": 10, "frame_type": "P",
          "residual_energy": 0.8, "motion_magnitude": 0.7, "entropy": 0.9})
result = module.score(buf)
assert result["action_score"] > 0.6, "Action scene should score high"
print("All assertions passed.")
```

## Part 2 — `aria.py` (LLM interface)

Per the team roadmap, this is the LLM abstraction layer. Build:

- A single `generate(prompt: str, context: str) -> str` function as the entire public interface. Nothing outside `aria.py` should ever call an LLM API directly.
- Backend: OpenAI API for now (read key from env var, don't hardcode). Structure the internals so swapping to a local Llama 3.2 3B backend later is a backend-class swap, not a rewrite of callers.
- No OpenAI-specific imports or calls anywhere else in the codebase — if `pipeline.py` or anything else needs LLM generation, it goes through `aria.generate()`.

## Part 3 — `pipeline.py` (end-to-end harness)

This is the integration glue. Build:

- Accepts a video file path + a user query string, returns a verified answer string (plus, ideally, intermediate debug info — action scores, retrieved frames, raw vs. verified answer — behind a `verbose=True` flag).
- Wires together, in order: `charon_v.py` (decode + signal extraction) → `action_score.py` (peak gating, run alongside whatever tier output Charon-V currently exposes — don't remove the tier path) → `l1_elysium.py` (cache) → `l2_asphodel.py` (retrieval) → `aria.py` (generation) → `cerberus_v.py` (verification).
- Since Track A (`l1_elysium.py`), Track C (`l2_asphodel.py`), and Track D (`cerberus_v.py`) are all still "Active"/in-progress per the roadmap, write `pipeline.py` against their *current* interfaces as they exist in the repo right now, and isolate each call behind a thin wrapper function so that when those modules' interfaces change (e.g. if the team later adopts continuous action_score everywhere instead of tiers), you only need to update the wrapper, not the whole pipeline flow.
- This becomes the harness `eval_suite.py` (Track D) will call for ablation testing, so keep the function signature stable and simple: something like `run_pipeline(video_path: str, query: str, verbose: bool = False) -> dict`.

## What not to do

- Don't modify `charon_v.py`, `l1_elysium.py`, `l2_asphodel.py`, or `cerberus_v.py` directly — those are other tracks' active work. If `action_score.py` needs something exposed from `charon_v.py` that isn't currently public, add it as a non-breaking addition (e.g. a new method that returns the raw per-frame record) and flag it clearly as a cross-track dependency rather than restructuring their files.
- Don't silently replace the tier-based design — this needs a team decision, not a unilateral one buried in a PR.

## Branch

Work on `track-b/aria-pipeline` per the repo's branch strategy. Nothing merges to `main` without tests passing.
