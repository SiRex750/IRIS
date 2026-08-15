# IRIS, explained from zero — what it is, where it stands, and where the idea itself is wrong

Written 2026-08-15, for Sonu, assuming you know nothing. Read it slowly. The last section is
the one that matters; everything before it is so that section makes sense.

A note on the GitHub link you sent (`SiRex750/IRIS`): that repo is **stale**. Its `main` has
one commit and the old README — "codec-native hierarchical memory," "CVPR/MLSys," the NLI truth
gate. None of that is where you actually are. Everything below is from the real code and the
real result files on your laptop, not from GitHub.

---

## Part 1 — What problem is this project even trying to solve?

Imagine you have an hour-long CCTV video and you want to ask a computer: *"when did someone
break in?"* You'd like to hand the whole video to an AI (a language model) and let it answer.

You can't. Here's the wall, in dumb-dumb terms:

- An hour of video is ~100,000+ frames (still pictures).
- A language model that can "look" at pictures can only look at a few dozen at a time before it
  gets impossibly slow and expensive. (You saw exactly why in that "first API call" PDF — every
  frame you show it, you pay for, every single time you ask a question.)
- So you **cannot show it all the frames.** You have to pick a small handful — say, the 100 that
  matter — and throw away the other 99,900.

That picking step is the whole ballgame. If you pick the wrong 100 frames, the AI never sees the
break-in and gives a confidently wrong answer. If you pick well, a cheap AI looking at 100
frames can answer as well as an expensive one looking at thousands.

So every system in this area does two jobs:

1. **Selection** — which frames do we keep? (the "what to look at" problem)
2. **Indexing/retrieval** — once we've kept some frames, how do we organize them so that when a
   question comes in, we can quickly fetch the relevant ones? (the "how to look it up" problem)

**IRIS's original big bet was about job #1.** The clever idea was: *the video file already did
the hard work of finding where stuff happens — for free — and we can read it off the compressed
file without doing any AI at all.*

That "for free, from the compressed file" idea is what the whole project is named after:
**I**ntelligent **R**esidual **I**ndexing **S**ystem. "Residual" is a compression word. Hold
that thought — it's going to be the thing that's broken.

---

## Part 2 — The one piece of background you need: what "codec residual" means

When a video is compressed (H.264, the normal `.mp4` codec), the encoder is lazy in a smart way.
Instead of storing every frame as a full picture, it stores:

- Occasional **full pictures** (called **I-frames** or **keyframes**), and
- In between, just **the changes** — "this frame is the last one, but this patch moved right by
  3 pixels and this corner got brighter." That "what changed" data is the **residual**, and
  crucially it comes with **motion vectors** (arrows saying which way each patch moved).

Here's the seductive part. When a lot is happening in the video — someone runs, a car crashes,
a crowd surges — the encoder has to store **a lot** of change data. The residual is **big**.
When nothing happens — an empty hallway — the residual is **tiny**.

So: **big residual ≈ lots of motion ≈ (we hoped) something important is happening.**

And you get this for *free*, because the encoder already computed it. You just read the file
sizes of each chunk. No AI, no GPU, runs on a laptop CPU in about a second per thousand frames.

That is a genuinely beautiful idea. The problem, which took a year to prove, is in the word
"hoped." **Big residual means big motion. It does not mean important.** More on that in Part 5 —
it's the heart of everything.

---

## Part 3 — The pipeline, layer by layer (what actually happens to a video)

Here is the real flow in your code (`iris/` folder), stage by stage. I'll give each one a plain
name, say what it does, and — importantly — where it stands and what's wrong with it. Verdicts
come from your own result files (see `CLAIMS.md`).

```
  video.mp4
     │
     ▼
 [0] charon_v      "the codec reader"      ── reads residual + motion, no decode
     │
     ▼
 [1] action_score  "the importance scorer" ── turns codec numbers into one score per frame
     │
     ▼
 [2] l1_elysium    "the keep-list"         ── keeps the top ~10% of frames, drops the rest
     │
     ▼
 [3] l2_asphodel   "the map"               ── builds a searchable graph of the kept frames
     │
     ▼
 [4] aria          "the answerer (LLM)"    ── reads retrieved frames, writes the answer
     │
     ▼
 [5] cerberus_v    "the fact-checker"      ── was meant to catch made-up answers
     │
     ▼
  answer  (+ when it happened, for grounding questions)
```

### Layer 0 — `charon_v`, the codec reader

**What it does (dumb-dumb):** opens the `.mp4`, reads the compressed chunks *without turning them
back into pictures* (that's the "zero-decode" trick — decoding is the slow part, and it skips
it). For each frame it pulls out: how big the residual chunk is (`packet_size`), the motion
vectors, and where the keyframes are. It also uses the keyframes to chop the video into
**scenes** (`compute_valley_scene_boundaries`).

**Where it stands:** works, fast, exactly as advertised. ~1 s per 1,000 frames, CPU only,
**zero** neural network calls (you verified this by literally counting — 0).

**What's wrong:** nothing *mechanically*. The problem isn't here; it's what the next layer does
with this data. **But note one thing for later:** this layer does two different jobs at once —
it measures *how big* each residual is (importance signal) **and** it finds *where the scene
cuts are* (boundary signal). Those two jobs have completely different fates. Remember this.

### Layer 1 — `action_score`, the importance scorer

**What it does:** takes three numbers per frame — residual/packet size (weight 0.5), motion
magnitude (0.3), luma entropy i.e. "busyness of brightness" (0.2) — and blends them into a
single **action score**. Then it finds the "peaks" of that score over time and calls those the
important moments.

**Where it stands:** this is the layer the whole selection thesis lives in. And this is the
layer **R0 killed.**

**What's wrong — two things, and they're both fatal to the *idea*, not the code:**

1. **The three ingredients are basically the same ingredient three times.** Residual size,
   motion magnitude, and brightness-busyness all spike together — they all measure "the picture
   is changing a lot." Blending three copies of the same signal with fixed weights doesn't make
   it smarter. (Your team's larger run reportedly found these correlate at r ≥ 0.95 — that
   number is on a machine you can't currently open, so treat it as strongly-suspected, not
   proven. But you can see it's *plausible* just from what the three things measure.)

2. **The score doesn't actually know what's important — it knows what's *moving*.** This is the
   whole ballgame and it gets its own section (Part 5). The "peaks" it finds are motion peaks.

3. **A smaller, practical sin:** the "adaptive budget" that's supposed to keep more frames when
   more is happening actually lands at ~10.5% *no matter what the video is* (10.41%–11.14%
   across all 19 test videos). It's not adapting. It's a fixed quota wearing an adaptive costume.

### Layer 2 — `l1_elysium`, the keep-list

**What it does:** a cache that holds the best frames and, when full, evicts the lowest-scoring
one. "Best" is a blend of action score, query similarity, PageRank, recency, etc.

**Where it stands:** fine as engineering. Clean code.

**What's wrong:** it's carefully, cleverly managing a shortlist that was chosen by a signal
carrying no useful information (Layer 1). It's a well-built filing system for the wrong files.
There's nothing to *fix* here — it inherits Layer 1's problem and can't fix it from downstream.

### Layer 3 — `l2_asphodel`, the map (THE GOOD ONE)

**What it does:** builds a **graph** — dots (frames) connected by lines (how related two frames
are, by motion and by meaning). When a question comes in, it walks this graph to find the
relevant cluster of frames. The clever bit: instead of connecting *every* frame to *every* other
frame (which explodes as the video gets long), it only connects frames **within the same scene**,
using the scene cuts that Layer 0 found from the codec. This is the **"scene-sparse /
block-diagonal"** graph.

**Where it stands:** **this is the best result in the entire project, and it's real.** Two hard
facts from your files:
- The scene-sparse graph is **provably, bit-for-bit identical** to the expensive full graph
  (0 differences, exact) — but builds **~220× faster** and uses **~6.7× less memory.**
- As video gets longer, the full graph's query time blows up (roughly with the square of length,
  and it literally times out past ~6,500 frames), while the scene-sparse graph stays cheap and
  keeps working out to 13,500 frames.

**What's wrong:** almost nothing, and that's the point — but notice *why* it works. It works
because it uses the codec for the job the codec is **actually good at**: finding scene
boundaries. (Keyframes literally *are* scene boundaries — that's how encoders decide where to
put them.) Keep holding that thought; Part 6 pays it off.

Two honest caveats live here: the headline scaling *exponent* was measured with fake stand-in
queries in the version you have (the real-query version is a commit you can't open yet), and the
one big test clip is `mpeg4`, not H.264. Both are in `CLAIMS.md`.

### Layer 4 — `aria`, the answerer

**What it does:** a small language model (~3.4B parameters, runs on CPU) reads the handful of
retrieved frames and writes the answer.

**Where it stands:** it's a **floor, not a trophy.** On the standard NExT-GQA test it scores
Acc@GQA **0.1667** — which sits in the band of respectable 2023-era systems, on a CPU, which is
the point (cheap, not best). It is not going to beat the big GPU models and was never going to.

**What's wrong:** nothing to fix — it's doing its job. The mistake would be *trying* to make it
win the accuracy race. That race is lost and recedes every month (the current best training-free
video-anomaly systems are ~30 points ahead and climbing). Don't enter it.

### Layer 5 — `cerberus_v`, the fact-checker

**What it does (was supposed to):** after the LLM writes an answer, a second model (an NLI model,
DeBERTa) checks whether the answer is actually supported by the frames — catching hallucinations.
The residual was *also* supposed to decide **how hard** to fact-check (busy scene = check more).

**Where it stands:** **never measured. Zero results anywhere, ever.** This was pitched as the
genuinely novel half of the original project and it was never built or evaluated. The code exists;
no experiment does.

**What's wrong:** it's vapor. Treat it as not-in-the-project until something measures it. (This
is the same lesson as the "N6 caption finding" you went looking for and found didn't exist.)

---

## Part 4 — So what stage are we actually at? What have we achieved?

Plain scorecard, from your result files:

| Layer / claim | Status | One-line truth |
|---|---|---|
| Codec reading (L0) | ✅ works | fast, CPU, zero AI — as promised |
| **Frame selection (L1)** | ❌ **disproven** | no better than keeping every 10th frame |
| Keep-list (L2) | ⚙️ fine, moot | good code, inherits L1's dead signal |
| **Scene-sparse graph (L3)** | ✅ **strong, real** | provably identical, ~220× cheaper, scales sub-linearly |
| Answerer (L4) | ✅ floor | ~0.167, respectable-for-CPU, not a win |
| Fact-checker (L5) | 👻 never built | zero results, treat as absent |

**What you've genuinely achieved, stated honestly:**

1. A **scene-sparse video index** that is proven identical to the expensive version but
   dramatically cheaper to build and that stays fast as video gets long. *(This is Siddanth's,
   and it's the paper's spine.)*
2. A **clean, pre-registered negative result** — you *proved*, with a control you wrote and
   locked in advance, that the codec-selection idea doesn't beat random. *(This is yours, and at
   a systems venue a rigorous self-kill like this is worth more than a mediocre positive.)*
3. A **cheap CPU correctness floor** showing the whole thing produces sane answers without a
   giant GPU model.

**The tests you ran to get here:** R0 (the budget-matched selection control, n=19), the
block-diagonal identity gate (4,892 frames, 0 differences), the scaling-curve measurements
(N up to 13,506), the NExT-GQA grounding eval (120 questions), and a couple of audits that found
two "results" (the caption finding, the 4.6× correction) were never real. That last part isn't
failure — catching a made-up number *before* a reviewer does is exactly what this stage is for.

---

## Part 5 — Where the *idea itself* is wrong (this is what you actually asked)

You said: maybe it's not that we're doing it badly, maybe the idea is wrong — "codec might not
be good for what we want it to be good for." **You're right, and here is precisely why.**

The codec residual measures **motion / change**. The selection layer *assumed* motion = importance.
Those are different things, and on your task they come apart badly:

- A person standing still, holding a gun, at the moment of the crime — **visually almost static,
  tiny residual** — is the single most important frame, and the codec scores it low.
- Wind in trees, a passing bus, camera shake, a crowd milling — **huge residual** — and totally
  irrelevant to "when did the arson happen."

So the score fires on the wrong frames. Not sometimes — *systematically*, because "busy" and
"important" are genuinely unrelated in surveillance footage. That's why R0 found the selected
frames sit exactly at the base rate: the signal is orthogonal to the target. **It's not a tuning
problem. No weighting of a motion signal recovers a non-motion target.**

And here's the trap you must not fall into (you've circled it three times already, per your own
handoffs): *"maybe codec beats uniform in a different domain — sports! fast motion!"* **That's
the same mistake wearing a jersey.** In sports there's motion *everywhere*, all the time — so
"where's the motion" tells you even less about "where's the goal." Faster motion doesn't fix a
signal that measures the wrong quantity. A different-motion domain cannot revive a signal that's
absent at the operating point of the method. That question is closed. Don't reopen it.

**The deeper ideation error, stated bluntly:** IRIS was named and sold around the residual being
an *intelligence* signal — a thing that knows what matters. It isn't. It's a *motion* signal.
The project's identity was built on a capability the physics of compression doesn't provide.

---

## Part 6 — The reframe that actually saves the good work

Now the payoff of everything I told you to "keep holding."

Go back to Layer 0. The codec gives you **two** different things:

- **(a) how big each residual is** → an *importance* guess. **This is the part that died.**
- **(b) where the keyframes / scene cuts are** → a *boundary* guess. **This is the part that
  works.**

And look at which layers use which:

- Layer 1 (action_score, **dead**) uses **(a)** — codec as an importance ranker.
- Layer 3 (scene-sparse graph, **strong**) uses **(b)** — codec as a scene segmenter, via
  `compute_valley_scene_boundaries(... iframe_indices ...)`. The whole 220×-cheaper result
  **depends on codec-derived scene cuts.**

**That is not a coincidence. That is the entire diagnosis in one line:**

> The part of IRIS that uses the codec for what the codec is *genuinely* good at — finding where
> one scene ends and the next begins — is the part that works. The part that uses the codec for
> what we *wished* it were good at — judging which frame matters — is the part that died.

Codec is excellent at boundaries because keyframes literally *are* boundaries; the encoder puts
them where the picture changes enough that continuing to store "just the differences" stops
paying off. That's a real, physical, reliable signal. Codec is bad at importance because
importance is semantic — it's about meaning, and compression knows nothing about meaning.

**So the honest, defensible shape of this project is:**

1. **Stop selling codec as a frame *selector*.** Demote it to one honest sentence + the R0 null:
   "residual energy is the cheapest motion prior available since the demux runs anyway, but at
   matched budget it does not beat uniform sampling for evidence localization — we tested it and
   report the null."
2. **Lead with the indexing result** (the scene-sparse graph), which stands on codec's *real*
   strength (boundaries) and doesn't need codec to be smart about importance at all.
3. **Frame the paper's spine as:** *"which frames a training-free index admits doesn't matter
   much — uniform is as good as anything cheap. What matters is that once you've admitted them,
   codec-derived scene structure lets you build and query the index far more cheaply, and it
   scales to long video where the dense approach falls over."*

That reframe costs you nothing you actually have and keeps everything that actually works. It
also converts the codec negative from an embarrassment into a *feature* of the paper — a
rigorous "we tried the obvious thing, here's proof it doesn't work," which is the kind of thing
that makes reviewers trust the parts that *do* work.

---

## Part 7 — "Is even doing what we're doing good for what we want?" — the blunt answers

**If what you want is to beat accuracy benchmarks:** No. Not with this, not ever. That race is
lost and accelerating away. Stop looking at it. (Decision rule D1/D3 in your own handoffs already
says this.)

**If what you want is a real, publishable contribution:** Yes — but it's the *indexing* one, not
the *selection* one, and you have to say so out loud instead of letting the old codec story ride.

**If you're attached to codec being the star:** the only *honest* research direction where codec
is genuinely strong is **temporal segmentation** — scene/shot boundaries, "where does this event
start and end." You're already using that quietly inside Layer 3. If you wanted a *second* paper,
"free codec-derived temporal structure for long-video retrieval" is a real, unoccupied angle —
but it's a different project and it's future work, not this deadline.

**The one thing to stop doing:** looking for a new domain, a new detector, a new dataset where
the *importance* idea suddenly works. It won't. That instinct — "let's try X instead" — has shown
up as sports, as swapping the detector, as restarting from the old repo. Every time, it's the
same move: reaching for a new experiment instead of writing down the result you already have. The
result you already have is a good paper. It's just not the paper you set out to write.

---

## The whole thing in five sentences

You tried to use video compression data to pick the important frames, because compression is free
and computed already. It turns out compression measures *motion*, not *importance*, and on your
task those are unrelated — so the frame-picking idea doesn't work, and you proved that cleanly.
But the *same* compression data is genuinely good at finding *scene boundaries*, and that quietly
powers the one part that's excellent: a scene-partitioned index that's provably identical to the
expensive version, ~220× cheaper to build, and stays fast on long video. So the paper isn't
"codec picks smart frames" (dead) — it's "codec-derived scene structure makes a training-free CPU
index cheap and scalable, and here's a rigorous proof that the frame-picking everyone assumes you
need doesn't actually beat uniform." Lead with what works, report the rest as an honest null, and
stop hunting for a domain that rescues the wrong half.
