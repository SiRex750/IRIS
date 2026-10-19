# Truth pilot: deviations from PLAN.md (e7ac3b0)

1. **gates.md table, clip column (report formatting only).** `truth_gates.py` wrote `r.clip`, which on a pandas row
   is the `Series.clip` method, not the clip name (same issue as `p3/codecsight_native2fps` d9251bf). Every gate value
   and pass/fail was computed correctly; only the first column of the gates.md table was garbled. Fixed to
   `r["clip"]`; the gate step was rerun from the same candidate outputs (no candidate rerun needed: the fix touches
   no candidate code or gate computation). Run 1 values are identical to run 2 (gates.csv unchanged).

Not a deviation (pre-registered fallback, PLAN §1): DOT at 480×856 ran out of GPU memory on all 3 clips, so DOT was
run at 360×640 on every clip. It ran out of memory there too (all 3 clips). Both attempts are in
`out/dot_manifest.json` and `out/dot_360x640_manifest.json`. PLAN has no further fallback, so DOT has no gate value
and does not pass.
