# Erratum: `scaling_curve_v3.json`

The result file is left exactly as produced. This note records where its own annotations are
wrong or easy to misread, so that nobody cites them as written. Every correction below was
checked against source code or the file's own fields (2026-09-27 to 2026-10-01); the paper
(`paper/latex/iris_mmsys27.tex`, §5) already reflects all of them.

## 1. `guards.salience_weights_guard.note` is wrong on two counts

The note says the corpus was built at (0.5, 0.3, 0.2), "NOT the frozen production default
(0.8, 0.1, 0.1)", and that survivor-N values are therefore "NOT comparable" to retention figures
computed elsewhere.

- **(0.5, 0.3, 0.2) is the production default.** These are the `IRISConfig` dataclass defaults
  (`iris/iris_config.py`: `luma_diff_weight` 0.5, `motion_weight` 0.3, `luma_entropy_weight` 0.2),
  and `configs/default_iris_config.json` does not override them. The (0.8, 0.1, 0.1) triple is
  the setting selected by the `ucf-vad-exp1` tuning sweep (`tuning/frozen_state.json` on that
  branch), not a production default.
- **Salience weights cannot change survivor counts.** Frame admission happens in
  `iris/charon_v.py::parse_video` from the packet-size curve alone (I-frames, packet-curve peaks,
  and frames whose packet size clears per-scene adaptive thresholds). The weighted action score
  is computed afterwards on the survivors (`iris/ingest.py`), and under the default
  `retrieval_strategy="hybrid"` every survivor is indexed. The weights shape each survivor's
  action score and hence edge weights; they do not decide which frames survive.

The census field `weights_match_frozen_default` rests on the same wrong reference point.

## 2. The census is not the fit's support

`survivor_census.measured_survivor_n.note` describes the 31-clip census as "this fit's actual
support". The fits use fewer clips:

| fit | clips used |
|---|---|
| `full_range`, scene_sparse | 19 |
| `full_range`, flat | 17 (Arrest047 and Arson019 censored) |
| `N_ge_1102_headline`, scene_sparse | 6 |
| `N_ge_1102_headline`, flat | 4 |

The census is 30 UCF-Crime clips plus one VIRAT clip. VIRAT has a completed latency row but is
excluded from every fit as cross-dataset. Of the other 11 census clips, 10 have
`outcome: "not_measured"` in both arms and one (Normal_Videos_881) has no latency row at all.
**No clip was measured and then dropped**: every UCF-Crime clip with a completed latency
measurement is in the fit.

## 3. `with_censored_as_lower_bound` is not a valid bound

This variant substitutes the 3600 s wall cap as a floor on the censored dense clips' per-query
latency, giving k >= 2.597. But the watchdog bounded the **whole measurement protocol**
(index load, flat graph build, then 3 warm-up + 250 timed queries = 253 executions), not a
single query. The cap is therefore not a latency floor for any one query, and the bound should
not be cited.

What the censored runs do show (paper §5.2):
- **N = 6,559 (Arrest047):** peak RSS 18.1 GB, consistent with a fully built dense graph of
  21.5 M edges at the ~690 bytes per edge measured at N = 4,892. Queries were running when the
  cap hit. The fit extrapolates to roughly 16 s per query; that is an extrapolation, not a
  measurement.
- **N = 13,506 (Arson019):** peak RSS 26.9 GB and still rising. The full graph of 91 M edges would
  need roughly 64 GB at the same rate, so the build had most likely not finished.

## 4. Two rows are carried over from an earlier run

The Arrest047 and Arson019 rows have `source: existing_v2_textquery_corpus`: they come from the
v2 text-query run (`scaling_curve_v2_textquery_raw.json`), not from a fresh v3 measurement.
Their scene-sparse medians are **0.0174 s and 0.0290 s**. The figures 0.0104 s and 0.0210 s
quoted in earlier drafts come from the superseded v2 *synthetic-query* run
(`scaling_curve_v2_raw.json`), whose queries were sampled as survivor embeddings.

## 5. Provenance

`provenance.git_head` is 9c66393 with `git_dirty: true` (94 tracked-dirty files), so these
exponents are not tied to a clean commit; the paper discloses this in §8. The harness is
`scripts/scaling_curve_ci.py` driving `scripts/scaling_curve_v2.py` and its worker (hashes in
`provenance.harness_config_hash`), not `scaling_curve_v2_textquery.py`.
