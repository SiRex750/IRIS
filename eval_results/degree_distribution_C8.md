# Per-node edge degree under scene_sparse — Appendix C item 8

Read-only. Computed from the 33 cached `.npz` manifests in
`eval/data/ucf/index_cache/` (32 clips) and `eval/data/virat/index_cache/`
(1 clip) — no re-ingest, no rebuild. `graph_edges` was read directly out of
each manifest's `__manifest__` JSON blob (the same list `load_index`
restores onto the rebuilt graph verbatim, per `iris/ingest.py:661–678`);
degree was computed by counting each node's appearances as `source` or
`target` across that list. Node identity is `frame_idx`. No filesystem
mtimes were used anywhere in this note.

## Headline finding: the four requested edge types do not exist in any cached artifact

Every one of the 33 cached indices was built with `config_snapshot.graph_mode
== "scene_sparse"`, and every single edge in every one of them carries
`edge_type == "fully_connected"` — 30 UCF clips plus the VIRAT clip under
`graph_edge_mode="fully_connected"`, and two UCF clips
(`Normal_Videos_924`, `Normal_Videos_935`) under `graph_edge_mode
="block_diagonal"` (which also labels its edges `"fully_connected"` —
`iris/l2_asphodel.py:221`, same string, different construction path). **Zero
of the 33 caches contain a `temporal`, `hierarchy_peak_salient`,
`hierarchy_salient_candidate`, `semantic_salient`, or `motion_neighbor`
edge.** The tiered path (`graph_edge_mode="hierarchical_sparse"`) — the
configuration §3.5's table says actually produced every accuracy number in
this paper (§5.3, §5.4, §6.2, §7.1) — has never been cached to disk anywhere
this audit can find. `git log --all` and a filename/content search for
`hierarchical_sparse` turn up the code path and the §3.5 prose describing it,
never a `.npz` built under it.

This is the central fact this note has to report honestly: **the per-node
degree breakdown by `temporal` / `hierarchy_*` / `semantic_salient` /
`motion_neighbor` that item 8 and this task ask for cannot be measured from
any existing artifact.** What follows is (a) a full empirical characterization
of the degree distribution that *is* recoverable — the `fully_connected` /
`block_diagonal` corpus, which is also the exact corpus §5.1's scaling
exponents are fit on — and (b) a source-level argument, explicitly flagged as
inference rather than measurement, about what the tiered path's degree
distribution can and cannot do, given the one hard constraint the code
enforces regardless of edge family.

## Per-clip degree table

`gem` = `config_snapshot.graph_edge_mode`. Degree = source+target appearances
in `graph_edges`, one row per clip, per-node summary in edges (not weight).

| corpus | clip | N | edges | gem | min | median | mean | p90 | p99 | max |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ucf | Normal_Videos_881 | 24 | 46 | fully_connected | 0 | 3.5 | 3.83 | 6 | 6 | 6 |
| ucf | Normal_Videos_289 | 91 | 145 | fully_connected | 0 | 3 | 3.19 | 5 | 7 | 7 |
| ucf | Normal_Videos_908 | 93 | 229 | fully_connected | 0 | 5 | 4.93 | 12 | 12 | 12 |
| ucf | Assault036 | 97 | 258 | fully_connected | 0 | 3 | 5.32 | 13 | 13 | 13 |
| ucf | Abuse023 | 106 | 444 | fully_connected | 0 | 9 | 8.38 | 16 | 16 | 16 |
| ucf | Assault045 | 131 | 480 | fully_connected | 0 | 3 | 7.33 | 24 | 24 | 24 |
| ucf | Arson050 | 166 | 294 | fully_connected | 0 | 3 | 3.54 | 7 | 9 | 9 |
| ucf | Normal_Videos_758 | 166 | 484 | fully_connected | 0 | 4 | 5.83 | 21 | 21 | 21 |
| ucf | Arrest022 | 167 | 328 | fully_connected | 0 | 3 | 3.93 | 7 | 13 | 13 |
| ucf | Normal_Videos_168 | 181 | 360 | fully_connected | 0 | 3 | 3.98 | 8 | 14 | 14 |
| ucf | Abuse037 | 188 | 851 | fully_connected | 0 | 6 | 9.05 | 22 | 22 | 22 |
| ucf | Normal_Videos_891 | 188 | 450 | fully_connected | 0 | 3 | 4.79 | 13 | 15 | 15 |
| ucf | Arrest031 | 194 | 503 | fully_connected | 0 | 3.5 | 5.19 | 13 | 13 | 13 |
| ucf | Abuse025 | 212 | 995 | fully_connected | 0 | 6.5 | 9.39 | 23 | 23 | 23 |
| ucf | Arson040 | 297 | 940 | fully_connected | 0 | 4 | 6.33 | 17 | 24 | 24 |
| ucf | Arrest007 | 333 | 1448 | fully_connected | 0 | 5 | 8.70 | 22 | 28 | 28 |
| ucf | Arrest036 | 399 | 1104 | fully_connected | 0 | 4 | 5.53 | 13 | 18 | 18 |
| ucf | Abuse043 | 405 | 1179 | fully_connected | 0 | 4 | 5.82 | 16 | 19 | 19 |
| ucf | Arson002 | 463 | 613 | fully_connected | 0 | 2 | 2.65 | 5 | 11 | 11 |
| ucf | Abuse036 | 496 | 1159 | fully_connected | 0 | 5 | 4.67 | 8 | 12 | 12 |
| ucf | Arson042 | 613 | 3855 | fully_connected | 0 | 8 | 12.58 | 31 | 33 | 33 |
| ucf | Normal_Videos_925 | 806 | 3919 | fully_connected | 0 | 7 | 9.73 | 23 | 40 | 40 |
| ucf | Arrest030 | 903 | 1713 | fully_connected | 0 | 4 | 3.79 | 7 | 9 | 9 |
| ucf | Assault042 | 936 | 2082 | fully_connected | 0 | 5 | 4.45 | 8 | 11 | 11 |
| ucf | Arrest016 | 1102 | 2069 | fully_connected | 0 | 3 | 3.76 | 7 | 13 | 13 |
| ucf | Arrest039 | 1648 | 3207 | fully_connected | 0 | 4 | 3.89 | 7 | 10 | 10 |
| ucf | Abuse042 | 2963 | 8325 | fully_connected | 0 | 5 | 5.62 | 12 | 19 | 20 |
| ucf | Normal_Videos_940 | 3747 | 7688 | fully_connected | 0 | 4 | 4.10 | 7 | 11 | 12 |
| virat | VIRAT_S_040001_01_000448_001101 | 4892 | 23571 | fully_connected | 0 | 8 | 9.64 | 14 | 23 | 23 |
| ucf | Arrest047 | 6559 | 12144 | fully_connected | 0 | 3 | 3.70 | 8 | 17 | 21 |
| ucf | **Normal_Videos_935** | 11232 | 15638 | **block_diagonal** | 0 | 2 | 2.79 | 5 | 8 | 11 |
| ucf | **Normal_Videos_924** | 11233 | 17347 | **block_diagonal** | 0 | 3 | 3.09 | 6 | 10 | 13 |
| ucf | Arson019 | 13506 | 23821 | fully_connected | 0 | 3 | 3.53 | 6 | 9 | 13 |

(31 `fully_connected` clips — 30 UCF + 1 VIRAT — plus the 2 `block_diagonal`
clips, listed last, in N order.)

## Pooled distribution shape

Pooling every node across the 31 `fully_connected` clips (42,072 nodes):
min 0, median 4, mean 4.98, p90 11, p99 22, max 40.

| degree bucket | nodes | share |
|---|---:|---:|
| 0 | 1,333 | 3.2% |
| 1–2 | 10,670 | 25.4% |
| 3–5 | 17,851 | 42.4% |
| 6–10 | 7,890 | 18.8% |
| 11–20 | 3,797 | 9.0% |
| 21–45 | 531 | 1.3% |
| 46+ | 0 | 0.0% |

Right-skewed with a short tail: the mode sits at 3–5, the distribution is
already past its 99th percentile by degree 22, and **no node in any of the 31
`fully_connected` clips — including the two largest at N=6,559 and N=13,506 —
ever exceeds degree 40.** There is no long tail here in the sense a
scale-free or power-law degree distribution would produce; it looks like a
distribution with a hard, small ceiling.

That ceiling has an exact, structural explanation, not an empirical one. All
three edge modes remove cross-scene edges (`iris/l2_asphodel.py:233–242`, and
`block_diagonal`'s pairwise loop never visits a cross-scene pair to begin
with), so **within any of these three edge modes, no node's degree can exceed
(its scene's size − 1).** This is why min = 0 in every single clip: a scene
with exactly one frame contributes an isolated node — 3.2% of all nodes,
pooled, sit in a scene of size 1.

## Question 1 — does per-node degree drift upward with N?

**No — not for the corpus and edge modes actually cached.** Across the 31
`fully_connected` clips (N = 24 to 13,506, a ~560× range), degree statistics
show no growth trend against N:

| N range | clips | mean(deg_mean) | mean(deg_max) | max(deg_max) |
|---|---:|---:|---:|---:|
| [0, 300) | 15 | 5.67 | 15.47 | 24 |
| [300, 1000) | 9 | 6.44 | 20.11 | 40 |
| [1000, 3000) | 3 | 4.42 | 14.33 | 20 |
| [3000, 7000) | 3 | 5.81 | 18.67 | 23 |
| [7000, 15000) | 1 | 3.53 | 13.00 | 13 |

Pearson correlation of clip-level degree statistics against N:

| statistic | r(N, ·) | r(log N, log ·) |
|---|---:|---:|
| deg_mean | −0.155 | −0.070 |
| deg_median | −0.067 | — |
| deg_p90 | −0.265 | — |
| deg_p99 | −0.125 | — |
| deg_max | −0.010 | 0.224 |

All correlations are small and several are negative; none support degree
growing with N. **The mechanism is scene count, not scene size.** Recovering
scene membership from connected components of `graph_edges` (each scene is a
disjoint clique under these edge modes, so components == scenes) gives:

| N range (example clips) | n_scenes | scene_mean | scene_max |
|---|---:|---:|---:|
| N=24 (Normal_Videos_881) | 6 | 4.0 | 7 |
| N=613 (Arson042) | 110 | 5.57 | 34 |
| N=4,892 (VIRAT) | 528 | 9.27 | 24 |
| N=6,559 (Arrest047) | 2,178 | 3.01 | 22 |
| N=13,506 (Arson019) | 3,592 | 3.76 | 14 |

Pearson correlations across all 31 `fully_connected` clips:

- r(N, n_scenes) = **0.975**; r(log N, log n_scenes) = **0.988** — number of
  scenes grows almost perfectly linearly with N (in log-log terms, close to
  slope 1).
- r(N, scene_mean) = 0.135; r(N, scene_max) = −0.010 — scene *size* does not
  grow with N at all.

**This is exactly the property the sub-quadratic scaling claim needs, and it
holds empirically for the configuration §5.1 actually fits.** As N grows, the
graph adds more scenes rather than bigger ones, so per-node degree — bounded
by scene size under every edge mode this corpus contains — stays flat instead
of drifting upward. If scene size had grown with N instead, a query touching
even a fixed number of scenes would see per-scene PPR cost grow with it, and
§5.1's exponent would not mean what §5.1 says it means.

**What this does not cover.** This measurement is over `fully_connected` /
`block_diagonal` graphs — exactly what §5.1 fits, but *not* the
`hierarchical_sparse` graphs that produced §5.3, §5.4, §6.2, and §7.1's
numbers. Scene-count-vs-N is a property of the scene partition, which is
identical across all three edge modes (§3.5: "the block structure is
therefore common to all three"), so the scene-growth mechanism above applies
unchanged to `hierarchical_sparse`. But whether `hierarchical_sparse`'s
*within-scene* degree also stays flat as scenes grow larger at the tail is a
separate question this corpus cannot answer, because no `hierarchical_sparse`
cache exists to check it against (see Question 2).

## Question 2 — which edge type dominates the high-degree nodes?

**Cannot be answered from any cached artifact — 100% of edges in all 33
caches are labelled `fully_connected`.** Nothing here supports assigning any
share of the tail to `temporal`, `hierarchy_*`, `semantic_salient`, or
`motion_neighbor`, because none of those labels appear even once.

The following is source-code analysis, not measurement, and is flagged as
such throughout.

Reading `iris/l2_asphodel.py`'s tiered-path construction
(`_add_temporal_edges`, `_add_hierarchy_edges`, `_add_salient_semantic_edges`,
`_add_motion_neighbor_edges`, lines 437–510):

- **`temporal`**: window=1, so each node gets at most 2 temporal edges (one
  each side) before scene pruning. Structurally bounded, small, independent
  of scene size.
- **`hierarchy_peak_salient` / `hierarchy_salient_candidate`**: each
  `L2_SALIENT` or `L3_CANDIDATE` node gets **at most one** parent edge
  (`_nearest_node` returns a single node). Bounded per-source at 1; a hub
  node could still accumulate many *incoming* parent edges if many children
  independently pick it as nearest, again capped only by scene size.
- **`semantic_salient`**: `top_k=4` **out**-edges per `L2_SALIENT` source,
  selected only among other `L2_SALIENT` nodes above a 0.5 similarity
  threshold (lines 474–494). This is a per-source cap, not a per-node
  (in-degree) cap — the selection is not required to be mutual, so nothing in
  the code prevents many different sources from independently placing one
  popular node inside their own top-4. **A per-node in-degree cap for this
  family exists in intent (top_k=4 reads as "each node gets ≤4 semantic
  neighbours") but not in code — only source out-degree is capped.**
- **`motion_neighbor`**: same shape, `top_k=2` **out**-edges per source, but
  selected from *all* nodes, not only salient ones (lines 496–510) — a wider
  candidate pool than `semantic_salient`'s. Same asymmetry: out-degree ≤ 2,
  in-degree uncapped in code.

**However, this asymmetry cannot produce unbounded degree, because of the one
constraint every edge mode enforces after selection: cross-scene edges are
removed regardless of edge family (§3.5; `l2_asphodel.py:233–242`).** So the
correct statement is narrower than "semantic_salient is unbounded": no edge
family can push a node's degree past **(its scene's size − 1)**, the same
hard ceiling this note measured empirically for `fully_connected` — and that
ceiling does not grow with N in this corpus (Question 1). What *is* true, and
is worth stating as the finding this question is actually looking for: within
that ceiling, a node's in-degree from `semantic_salient` or `motion_neighbor`
specifically has no per-node cap in the code, only a per-source out-degree
cap — a real gap between the top_k parameters' apparent per-node intent and
what the code enforces — but the consequence of that gap is capped by scene
size, not unbounded, given the scene sizes this corpus actually produces (max
41 frames, pooled). Whether that stays true at a scene size this corpus never
exercises is untested. We did not verify this by running `hierarchical_sparse`
and measuring it directly — no such artifact exists (see the headline
finding above) — and the reading here should not be taken as equivalent to a
direct measurement.

Separately, §3.5's own text already names a **degree-deficit** risk in the
opposite direction from "unbounded": under prune-after-select, "a node whose
two nearest motion neighbours both lie in other scenes therefore ends with
zero motion edges rather than its two best in-scene ones." So the tiered
path's known, stated failure mode is nodes ending up *under*-connected
relative to their per-source top_k, not a single node accumulating unbounded
in-degree. Both effects (deficit and unbounded in-degree) are logically
possible in the same graph — deficit for some nodes, mild concentration for
others — but neither has been measured, only reasoned about from source.

## Question 3 — do the two `block_diagonal` clips differ from the `scene_sparse`/`fully_connected` ones?

Reported separately, not folded into the fully_connected corpus:

| clip | N | edges | min | median | mean | p90 | p99 | max | n_scenes | scene_mean | scene_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal_Videos_935 | 11,232 | 15,638 | 0 | 2 | 2.79 | 5 | 8 | 11 | 3,517 | 3.19 | 12 |
| Normal_Videos_924 | 11,233 | 17,347 | 0 | 3 | 3.09 | 6 | 10 | 13 | 3,476 | 3.23 | 14 |

Pooled (22,465 nodes): min 0, median 3, mean 2.94, p90 5, p99 8, max 13; 3.2%
of nodes at degree 0 — the same isolated-singleton-scene fraction as the
`fully_connected` corpus.

**They are not meaningfully different in shape, only slightly lower and
tighter.** Both are the two largest UCF clips (N≈11,232–11,233, the two
biggest after Arson019's 13,506), and their degree ceiling (max 13) is lower
than several much smaller `fully_connected` clips (e.g. Arson042 at N=613
reaches max 33, Normal_Videos_925 at N=806 reaches max 40). This is
consistent with `block_diagonal` and `fully_connected` producing the
identical intra-scene-complete-graph structure (§4.1's byte-identity gate;
`block_diagonal` differs only in *how* the same edge set is constructed, not
*what* it is) — the small remaining difference in medians/maxima reflects
these two clips' own scene-size distribution (scene_mean 3.19–3.23, in line
with the rest of the corpus) rather than anything specific to the
construction path. **Excluding them from the §5.1 fit for edge-mode mismatch
does not exclude them for having a different degree profile — on this
evidence they would have fit the same flat-vs-N pattern as every other clip.**

## Question 4 — does §5.1 or §3 state or imply a per-node degree bound?

**No explicit bound is stated anywhere in §3 or §5.1.** §3.5 states the
per-source *selection* parameters for the tiered path (`graph_temporal_window
=1`, salient-semantic `top_k=4`, motion-neighbour `top_k=2`, one hierarchy
parent) but never converts these into a claimed per-node degree bound, and
the paragraph immediately following that description is itself the trigger
for this item: "The production graph's edge count consequently depends on
scene boundaries in a way we have not characterised." That sentence is an
explicit admission of *not* having a characterization, not a claim that one
exists — so there is no claim here for the measured distribution to
contradict.

**§5.1 implies the property this note confirms, without naming it as a
degree bound.** The sub-quadratic exponent claim ("scene-sparse query cost
grows sublinearly," §5.1) is a claim about *query* latency, but it rests on
retrieval touching a shortlist of scenes rather than the whole graph (§3.4);
that only stays sublinear if the per-scene subgraphs a query actually
traverses don't grow with N. This note's Question 1 finding — scene count
grows near-linearly with N while scene size and per-node degree do not — is
the structural precondition §5.1's exponent needs and does not itself state.
Nothing in §5.1's text makes this claim explicitly, so there is no
overstatement to correct there; but there was, before this note, no
artifact-level confirmation that the precondition holds. There now is, for
the `fully_connected`/`block_diagonal` configuration §5.1 fits.

## Does the scaling story hold as written?

**Yes, for the configuration §5.1 actually measures — confirmed, not merely
assumed.** Per-node degree under `fully_connected` and `block_diagonal` does
not drift upward across the full measured range (N=24 to N=13,506); the
mechanism is that scene count scales with N (r=0.975) while scene size does
not (r=0.135 for mean, −0.010 for max). No reported exponent needed
adjustment and none was adjusted.

**No, the same confirmation cannot be extended to the configuration that
produced every accuracy number in the paper.** `hierarchical_sparse` — used
in §5.3, §5.4, §6.2, and §7.1 — has no cached artifact anywhere in this
repository. Its per-node degree distribution, its edge-type breakdown, and
whether its known under-connection risk (§3.5's prune-after-select deficit)
or its unmeasured over-concentration risk (Question 2, code-level only) show
up in practice remain open. This is not a new limitation invented here — §3.5
already flags it as uncharacterised (item 8) — but this audit's contribution
is to confirm that the gap is total: not one of the 33 available caches can
close it, and closing it requires generating a `hierarchical_sparse` index
cache and re-running this same analysis, which was out of scope for a
read-only, no-reingest pass.

## Recommendation for Appendix C item 8

State plainly: per-node degree under the `fully_connected`/`block_diagonal`
configuration is empirically flat against N (this note's Question 1 table)
and hard-bounded by scene size at every N measured (max 40 pooled across
31 clips) — the property §5.1's sublinear claim needs, now confirmed rather
than assumed. State separately, and do not conflate the two: the
`hierarchical_sparse` configuration that produced every accuracy number in
the paper has never been cached, so its degree distribution — and in
particular whether `semantic_salient` or `motion_neighbor`'s asymmetric
per-source top-k selection produces meaningfully concentrated in-degree
within a scene — remains open, capped only by the same scene-size ceiling
in principle, unmeasured in practice.
