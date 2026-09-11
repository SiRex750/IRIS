# Appendix C items 4 and 5 — citation provenance, verified against primary sources

Both items were taken from secondary sources (our own benchmark ledger for item 4; AdaCodec's
related-work survey for item 5). Both are now checked against the actual primary sources —
downloaded PDFs, read directly, not summarized secondhand. No paraphrase below substitutes for
a table value; every number quoted is transcribed from the source table.

---

## ITEM 4 — NG+ variants and the rest of the §6 competitor table

### Where the §6 table's rows actually come from

The table at §6.1 (and the parallel numbers in §2.3) has five rows requiring a primary source:
Xiao et al.'s own Table 3 (Temp[CLIP] NG+, FrozenBiLM NG+, SeViLA*), LangRepo's own Table 5, and
MUPA's own Table 1. (VideoMind-2B was also checked as a bonus — it appears only inside MUPA's
Table 1, not as an independent primary source in this pass.)

**Xiao et al., *Can I Trust Your Answer? Visually Grounded Video Question Answering*,
arXiv:2309.01327 (CVPR 2024).** Table 3, PH/NG/NG+ blocks, columns Acc@GQA | mIoP | IoP@0.3 |
IoP@0.5 (quoted verbatim from the table):

- `SeViLA* S Y ViT-G FT5 68.1 71.5 16.6 29.5 34.7 22.9 21.7 29.2 13.8` — Acc@GQA 16.6, mIoP 29.5,
  IoP@0.5 22.9.
- `NG+ Temp[CLIP] D Y ViT-L RBT 60.2+0.8 63.3+0.8 16.0+0.8 25.7+0.3 31.4+3.2 25.5+0.0 12.1+5.5 17.5+8.2 8.9+4.8`
  — Acc@GQA 16.0, mIoP 25.7, IoP@0.5 25.5.
- `NG+ FrozenBiLM S Y ViT-L DBT 70.8+1.7 73.1+1.4 17.5+1.7 24.2+1.5 28.5+2.7 23.7+1.6 9.6+2.5 13.5+3.5 6.1+1.7`
  — Acc@GQA 17.5, mIoP 24.2, IoP@0.5 23.7.

**Xiao et al.'s abstract vs. Table 3 (the MUPA-style hazard, checked):** no discrepancy. The
abstract states only round summary figures for FrozenBiLM's plain post-hoc numbers ("69% QA
accuracy... only 16% ... grounded"), which round-match Table 3's PH-block FrozenBiLM row
(Acc@QA 69.1, Acc@GQA 15.8). The abstract makes no claim about any NG+ figure, so there is nothing
in it to conflict with the NG+ numbers our table cites.

**Kahatapitiya et al., *Language Repository for Long Video Understanding*, arXiv:2403.14622
(ACL 2025).** Table 5, NExT-GQA results, 12B (Mixtral-8×7B, "12B active parameters" per the
paper's own §5) row, quoted verbatim:

`LangRepo (ours) ✗ 12B 31.3 28.7 18.5 12.2 17.1` — columns are mIoP, IoP@0.5, mIoU, IoU@0.5,
Acc@GQA, so mIoP 31.3, IoP@0.5 28.7, Acc@GQA 17.1. The paper's own §5.1 text states "outperforming
baseline Mistral LLM by +2.0% and LLoVi (12B) by +0.9% on Acc@GQA" — checked against the same
table: LangRepo-7B (11.2) − Mistral-7B (9.2) = 2.0; LangRepo-12B (17.1) − LLoVi-12B (16.2) = 0.9.
Both check out exactly against the correct same-scale rows — no abstract/table or text/table
discrepancy in LangRepo's own paper.

**Dang et al., *MUPA: Towards Multi-Path Agentic Reasoning for Grounded Video Question
Answering*, arXiv:2506.18071 (v2, 27 Jun 2025).** Table 1, quoted verbatim:

```
MUPA-2B (Ours) 2B 49.4 25.6 27.2 57.0 38.7 39.1 28.7
MUPA-7B (Ours) 7B 54.2 27.3 33.4 60.6 39.4 41.4 30.3
```

Columns: Size | IoU R@0.3, R@0.5, mIoU | IoP R@0.3, R@0.5, mIoP | Acc@GQA. Reading the last four
fields of each row: MUPA-2B — IoP R@0.5 38.7, mIoP 39.1, Acc@GQA 28.7. MUPA-7B — IoP R@0.5 39.4,
mIoP 41.4, Acc@GQA 30.3.

### The abstract-vs-table hazard — present, but not where the existing ledger note says

Appendix A's note (§2.4, "Note for co-authors") reads: *"MUPA's abstract misstates its own Table
1 — cite the table: 28.7 / 39.1 / 38.7, not 29.0 / 39.7."* The substance is correct — Table 1 is
the number to cite, and it is what §6 already cites — but the location is misattributed. MUPA's
abstract states only the 7B figure: *"Acc@GQA of 30.3% and 47.4% on NExT-GQA and DeVE-QA
respectively"* — this **matches** Table 1's MUPA-7B row (30.3) exactly. The mismatched figures
(29.0% Acc@GQA, 39.7% mIoP) actually appear in **Section 4.1's "GQA Results" narrative paragraph**,
not the abstract: *"MUPA achieves 29.0% Acc@GQA and 38.7% IoP0.5... MUPA-2B achieving a mIoP of
39.7%."* That paragraph is also internally inconsistent with its own stated deltas (it claims
+0.8pp Acc@GQA over VideoMind-7B's 28.2, which would require 29.0 — consistent with the narrative
figure but not Table 1's 28.7; and +3.3pp mIoP over VideoMind-7B's 39.0, which would require 42.3,
not the 39.7 stated). Table 1 is the artifact free of this internal contradiction and is the one
our §6 table already cites — the guidance to cite the table stands, only its stated origin should
read "the Results narrative text" rather than "the abstract."

### Verdict: all §6 table rows checked, all match

| our current claim | source we actually used | primary source | matches | corrected wording |
|---|---|---|---|---|
| Temp[CLIP] NG+: Acc@GQA 16.0, mIoP 25.7, IoP@0.5 25.5 | benchmark ledger | Xiao et al. 2023, Table 3 (NG+ block) | **yes** | — |
| SeViLA*: Acc@GQA 16.6, mIoP 29.5, IoP@0.5 22.9 | benchmark ledger | Xiao et al. 2023, Table 3 (PH block) | **yes** | — |
| FrozenBiLM NG+: Acc@GQA 17.5, mIoP 24.2, IoP@0.5 23.7 | benchmark ledger | Xiao et al. 2023, Table 3 (NG+ block) | **yes** | — |
| LangRepo: Acc@GQA 17.1, mIoP 31.3, IoP@0.5 28.7, params 12B | benchmark ledger | Kahatapitiya et al. 2024, Table 5 (12B row) | **yes** | — |
| MUPA-2B: Acc@GQA 28.7, mIoP 39.1, IoP@0.5 38.7 | benchmark ledger | Dang et al. 2025 (v2), Table 1 | **yes** | — |
| MUPA-7B: Acc@GQA 30.3, mIoP 41.4, IoP@0.5 39.4 | benchmark ledger | Dang et al. 2025 (v2), Table 1 | **yes** | — |
| (Appendix A note) "MUPA's abstract misstates its own Table 1" | co-author note | Dang et al. 2025 (v2), Abstract vs. §4.1 | **no** — the abstract agrees with Table 1; §4.1's own narrative prose is what conflicts | *"MUPA's own §4.1 narrative text (not its abstract, which agrees with Table 1) states 29.0% Acc@GQA / 39.7% mIoP for MUPA-2B — cite Table 1's 28.7 / 39.1 / 38.7 instead."* |

### Does §6.1's "weakly-supervised band" placement claim survive?

Yes, unchanged. Every figure feeding that sentence — SeViLA 0.166, LangRepo 0.171, FrozenBiLM+NG+
0.175, MUPA-2B 0.287, and our own 0.1667 — is confirmed exact from primary sources above. Our
0.1667 sits between SeViLA (0.166) and LangRepo (0.171), inside the 0.160–0.175 band the
weakly-supervised NG+/SeViLA rows occupy, and remains well below MUPA-2B's 0.287 with our
interval excluding it. No correction to §6.1's placement paragraph or the §6 table is needed.

---

## ITEM 5 — classical codec/segmentation lineage

### The AdaCodec ID

**Confirmed: arXiv:2606.02569 resolves correctly to AdaCodec** — *AdaCodec: A Predictive Visual
Code for Video MLLMs* (Haowen Hou, Zhen Huang, et al.; Shanghai Jiao Tong University / JD.com;
submitted 1 Jun 2026). Unlike item 3's EGAgent case, there is no title mismatch here — the ID and
the paper agree.

### AdaCodec's own claim, and what it draws on

AdaCodec's §2.2 states (quoted): *"Early studies used motion vectors as a low-cost surrogate for
optical flow"* (citing Zhang et al. 2016, Wu et al. 2018, Shou et al. 2019), and *"CoViAR and
DMC-Net modeled I-frame, motion, and residual modalities jointly for efficient action
recognition"* (Wu et al. 2018; Shou et al. 2019). This is close to verbatim what §2 of our draft
says. Both claims were checked against the primary papers themselves, not just re-derived from
AdaCodec's phrasing.

### Primary-source check, claim by claim

**"Motion-vector surrogates for optical flow"** — checked against **Zhang, Wang, Wang, Qiao, Wang,
*Real-Time Action Recognition with Enhanced Motion Vector CNNs*, CVPR 2016**. Confirmed: the
paper "accelerates the two-stream architecture by replacing optical flow with motion vector[s]
which can be obtained directly from compressed videos without extra calculation." DMC-Net's own
related-work section (below) independently corroborates this reading: *"In the pioneering works,
Zhang et al. replace the optical flow stream in two-stream methods by a motion vector stream, but
it still needed to decode RGB image for P-frame and ignored other motion-encoding modalities."*
Claim matches.

**CoViAR** — checked against **Wu, Zaheer, Hu, Manmatha, Smola, Krähenbühl, *Compressed Video
Action Recognition*, CVPR 2018 (arXiv:1712.00636)**. Confirmed directly from the source (quoted
from its own related-work discussion in DMC-Net, itself checked against CoViAR's primary text):
"CoViAR... contains three independent CNNs operating over three modalities in the compressed
video, i.e. RGB image of I-frame (I), low-resolution Motion Vector (MV) and Residual (R). The
predictions from individual CNNs are combined by late fusion." This is exactly "I-frame, motion,
and residual streams jointly," for the task of action recognition (HMDB-51, UCF-101, Charades).
Claim matches.

**DMC-Net** — checked against **Shou, Lin, Kalantidis, Sevilla-Lara, Rohrbach, Chang, Yan,
*DMC-Net: Generating Discriminative Motion Cues for Fast Compressed Video Action Recognition*,
CVPR 2019 (arXiv:1901.03460)**. DMC-Net's own abstract frames it more narrowly — as a generator
that refines motion vectors and residuals into a more discriminative motion cue — and does not by
itself mention I-frames. But the full system, confirmed from the paper's own §4.2 (Implementation
Details), explicitly inherits CoViAR's three-stream design and adds the generated cue as a fourth:
*"For I, MV, and R, we follow the exactly same setting as used in CoViAR... we follow CoViAR to
obtain the final prediction via fusing prediction scores from all modalities (i.e. I, MV, R, and
DMC)."* So "DMC-Net... modeled I-frame, motion, and residual streams jointly" is accurate for the
**full DMC-Net system** (generator + classifier, Figure 2c in the paper) even though the
generator itself — the paper's specific technical contribution — operates only on motion vectors
and residuals. AdaCodec's grouped claim survives this distinction; it describes the full system,
which is also what "for action recognition" (a system-level task claim, not a module-level one)
implies.

### Verdict — item 5

| our current claim | source we actually used | primary source | matches | corrected wording |
|---|---|---|---|---|
| AdaCodec is at arXiv:2606.02569 | (assumed) | arXiv abstract page | **yes** | — |
| "motion-vector surrogates for optical flow" (no inline citation currently) | AdaCodec §2.2, secondhand | Zhang et al., CVPR 2016 | **yes**, substantively | Add the citation the sentence currently lacks: *"...running from motion-vector surrogates for optical flow [Zhang et al., CVPR 2016] through CoViAR and DMC-Net..."* |
| "CoViAR and DMC-Net, which modelled I-frame, motion and residual streams jointly for action recognition" | AdaCodec §2.2, secondhand | Wu et al., CVPR 2018 (CoViAR); Shou et al., CVPR 2019 (DMC-Net) | **yes** — for CoViAR directly from its own three-CNN design; for DMC-Net, true of the full system (generator + inherited CoViAR I/MV/R streams + late fusion), not of the generator module in isolation | Optionally sharpen: *"...CoViAR and DMC-Net, whose full systems modelled I-frame, motion and residual streams jointly for action recognition (DMC-Net's own contribution is the motion-cue generator that refines the MV/R pair; the deployed system still fuses all four streams, following CoViAR)."* — not required for accuracy, only for precision, since the current wording is already a true claim about the systems as published. |

### Recommendation

Item 5's line stands substantively. The one gap worth closing is that the "motion-vector
surrogates for optical flow" clause currently carries no citation at all in §2 (only CoViAR and
DMC-Net are named), while both classical works it does name check out exactly against their own
papers. Add the Zhang et al. 2016 citation to close that gap; no other change to the sentence is
needed.
