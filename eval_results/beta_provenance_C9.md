# beta / config provenance audit — C9

Source: `config_snapshot` read from each index_cache `.npz`'s `__manifest__`
(`np.load(path, allow_pickle=False)`, then
`json.loads(d['__manifest__'].item())['config_snapshot']`), for every clip in
`eval/data/ucf/index_cache/*.npz` plus
`eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz` — 33 files
total (the 31-clip fit support of `eval_results/scaling_curve_v3.json` plus
the two clips it excludes for `graph_edge_mode` mismatch, `Normal_Videos_924`
and `Normal_Videos_935`, included here for comparison and flagged below).

`n_frames` = manifest `frames_processed` (matches `n_survivors` in
`scaling_curve_v3.json`'s `survivor_census`).

| clip | n_frames | alpha | beta | ranking_mode | graph_mode | graph_edge_mode | captioner_backend | (luma_diff_w, motion_w, luma_entropy_w) | scene_segmentation present? |
|---|---:|---:|---:|---|---|---|---|---|:-:|
| Abuse023 | 106 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Abuse025 | 212 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Abuse036 | 496 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Abuse037 | 188 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Abuse042 | 2963 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Abuse043 | 405 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest007 | 333 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest016 | 1102 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest022 | 167 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest030 | 903 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest031 | 194 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest036 | 399 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest039 | 1648 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arrest047 | 6559 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arson002 | 463 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arson019 | 13506 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arson040 | 297 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arson042 | 613 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Arson050 | 166 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Assault036 | 97 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Assault042 | 936 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Assault045 | 131 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_168 | 181 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_289 | 91 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_758 | 166 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_881 | 24 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_891 | 188 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_908 | 93 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_925 | 806 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| Normal_Videos_940 | 3747 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| VIRAT_S_040001_01_000448_001101 | 4892 | 0.4 | 0.3 | ppr | scene_sparse | fully_connected | moondream | (0.5, 0.3, 0.2) | n |
| **Normal_Videos_924** *(excluded from fit)* | 11233 | 0.4 | 0.3 | ppr | scene_sparse | **block_diagonal** | moondream | (0.5, 0.3, 0.2) | n |
| **Normal_Videos_935** *(excluded from fit)* | 11232 | 0.4 | 0.3 | ppr | scene_sparse | **block_diagonal** | moondream | (0.5, 0.3, 0.2) | n |

31 rows above the divider = the actual 31-clip fit support of
`scaling_curve_v3.json`; the last two rows are the excluded pair, shown for
comparison.

## (a) Is beta identical across all clips?

Yes. All 33 index caches (the 31-clip fit support plus the two excluded
clips) carry `beta = 0.3`. No variation anywhere in the support.

## (b) Does any clip's config_snapshot match `configs/default_iris_config.json`?

No. `configs/default_iris_config.json` specifies `beta = 0.6` and
`captioner_backend = "minicpm"`. Every one of the 33 caches instead shows
`beta = 0.3` and `captioner_backend = "moondream"` — i.e. every clip matches
the `IRISConfig` dataclass defaults (`iris/iris_config.py`: `alpha=0.4`,
`beta=0.3`, `captioner_backend="moondream"`), not `default_iris_config.json`.
(`alpha=0.4` also matches the dataclass default and differs from
`default_iris_config.json`'s implicit absence of an alpha override — that
file only sets `salient_thresh`, `candidate_thresh`, `alpha`, `beta`,
`peak_order`, `captioner_backend`, and its `alpha=0.4` happens to agree, but
`beta` and `captioner_backend` do not.)

Note also: `graph_mode` (`scene_sparse`) and `graph_edge_mode`
(`fully_connected` / `block_diagonal`) are both explicit non-default
settings — the `IRISConfig` dataclass defaults are `graph_mode="flat"` and
`graph_edge_mode="hierarchical_sparse"`, neither of which appears in any
cache.

## (c) The two clips `scaling_curve_v3.md` excludes for graph_edge_mode mismatch

`Normal_Videos_924` and `Normal_Videos_935` both have
`graph_edge_mode = "block_diagonal"`, versus `fully_connected` for every
other clip in the 31-clip fit support — confirmed both from the npz
manifests above and from `scaling_curve_v3.json`'s
`survivor_census.measured_survivor_n.excluded_graph_edge_mode_mismatch`
block, which lists the same two labels with the same `graph_edge_mode` and
matching survivor counts (11233 / 11232).

## `scene_segmentation` presence

The key `scene_segmentation` is not present in `config_snapshot` for any of
the 33 caches checked. `iris/ingest.py` reads it via
`getattr(config, "scene_segmentation", "codec")`, so these caches were built
before/without that field being serialized into the snapshot and silently
fall back to the dataclass default (`"codec"`) rather than recording an
explicit value.
