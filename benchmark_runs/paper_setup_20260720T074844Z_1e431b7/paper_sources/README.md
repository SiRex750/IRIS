# Paper Sources

`paper_comparison_registry.csv` in the run root is the registry (7 rows, tiers A/B/C, many fields
marked `UNVERIFIED` where a single web-search pass did not confirm exact table numbers). This
directory records where primary-source PDFs can be found, not copies of them (repo already has a
local archive at `../../../CCBD.zip`, unzipped path `CCBD/`, which is NOT copied here to avoid
duplicating ~50MB+ of third-party PDFs into an "immutable" setup snapshot).

Confirmed local primary-source matches (by arXiv id embedded in filename):
- MUPA (arXiv 2506.18071) -> `CCBD/2506.18071v2.pdf`
- ReMoRa (arXiv 2602.16412) -> `CCBD/2602.16412v2.pdf`
- Two other PDFs in the archive, `CCBD/2602.08683v3.pdf` and `CCBD/2606.06532v1.pdf`, were not
  identified against any of the 7 registry entries in this pass — check manually before assuming
  they're unrelated.

No local PDF was located for: NExT-GQA (2309.01327), VideoChat-TPO (unconfirmed identity), TOGA
(2506.09445), VideoMind (2503.13444), Chain-of-Glimpse (2604.14692). These would need to be fetched
from arXiv directly (not done here — no benefit to a SETUP-ONLY task, and avoids an unreviewed bulk
download).

**Action item before paper writing begins:** open each confirmed/likely-matching local PDF and the
remaining arXiv links, extract the exact NExT-GQA table rows (Acc@QA, Acc@GQA, IoP, IoU and their
sub-metrics) referenced in `paper_comparison_registry.csv`, and replace every `UNVERIFIED` cell with
a page/table-number citation. Row 2 (VideoChat-TPO) additionally needs its identity confirmed before
any number is quoted at all.
