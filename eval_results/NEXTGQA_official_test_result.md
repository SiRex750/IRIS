# Official NExT-GQA test set -- registered run (prereg 546d392)

Questions: 5553 per arm; ingest failures: 0 videos (their questions scored incorrect).

| arm | Acc@GQA | Acc@QA | mIoP (official) | IoP@0.5 (official) | mIoU | parse-fail |
|---|---:|---:|---:|---:|---:|---:|
| S scene-sparse, complete-block | 0.1385 | 0.4021 | 0.2992 | 0.3074 | 0.1724 | 0.000 |
| F flat (section 6.1 config) | 0.1349 | 0.4018 | 0.2890 | 0.2971 | 0.1653 | 0.000 |
| U uniform | 0.0522 | 0.4322 | 0.2168 | 0.0927 | 0.2128 | 0.000 |

Paired, video-clustered bootstrap (B=10,000, seed 20260927):

- acc_gqa_S_minus_F: +0.0036, 95% CI [-0.0043, +0.0117] -- includes zero
- acc_qa_S_minus_F: +0.0004, 95% CI [-0.0096, +0.0101] -- includes zero
- acc_qa_S_minus_U: -0.0301, 95% CI [-0.0409, -0.0192] -- excludes zero

Deviations from the registration:

- official test.csv stores the answer as option text, not an index (unlike the validation CSV the section 6.1 harness reads with int()); the gold letter is derived by exact text match to a0-a4. 1 question(s) have duplicate option text matching the answer; any matching letter is scored correct.
- captioner_backend in the IRISConfig passed to ingest is not read by the caption path (iris._clip -> aria.get_captioner resolves from ConfigManager). MiniCPM was enforced by asserting the resolved captioner class instead; the registered intent (minicpm) is met.
- resumed from checkpoint with 15831 completed question-arm rows
