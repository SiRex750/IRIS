# CodecSight-matched 2 fps arm: deviations and observations
Pre-registration: `PREREGISTRATION.md` (cc2eaf9).

## 1. x264 CRF arms at 2 fps are not QP-equivalent to the native-rate CRF arms (observation, 2026-10-14)
Written before any metric of this arm. This is an observation; no definition changes.
- In the pilot (`pilot/manifest.json`, clip VIRAT_S_000207_01_000094_000156, 80 frames at 2 fps), x264 CRF gave a
  mean P-frame QP of 4.8 at CRF 12 (PSNR-Y 56.1 dB, 5,111 kbps) and 37.9 at CRF 45.
- At native rate the same CRF settings give P-frame QPs near the CRF value. The difference arises because x264's CRF
  rate control depends on the frame rate.
- The arm definitions are unchanged, and F1 still compares x264 CRF 45 with CRF 12. But the 2 fps CRF arms are not
  QP-equivalent to the native-rate CRF arms of the confirmatory test and the CodecSight simulation.
- The constant-QP arms (x264 QP 24 bf 0 and bf 2; every NVENC arm) are unaffected: their decoded QPs match the set
  values (checked per encode).
