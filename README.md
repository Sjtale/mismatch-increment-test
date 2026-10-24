# mismatch-increment-test

**Does your harmonization help a follow-up? Test it against a matched rescan.**

Tumor response is read from volume change. When a follow-up CT is reconstructed with a different slice thickness or kernel than its baseline, the measured change carries an extra error that reads as tumor change. Distribution checks (as used for ComBat) and same-scan checks (as used for kernel conversion) cannot see this error.

`mismatch-increment-test` measures it on same-day repeat scans, and tells you in one line whether a correction removes it — for any segmentation model and any correction.

## Install

```bash
pip install git+https://github.com/Sjtale/mismatch-increment-test
```

Python 3.9+, NumPy only.

## Try it

```bash
mismatch-increment-test examples/volumes.csv --corrected examples/calibrated.csv --average 2
```

```
Mismatch-increment test · examples/volumes.csv · 30 patients, 830 directions
  Mismatch increment θ      5.7 [2.6, 9.4]
  Matched follow-ups        13.5   (mean 100·|ln V2/V1|)
  Mismatched follow-ups     19.2   (+42% vs. matched)

Correction · examples/calibrated.csv  (30 patients, 830 directions in common)
  Δθ                        -1.7 [-3.5, +0.2]    lowers the increment: no
  Matched follow-ups        13.5 → 13.5  (change 0.0 [0.0, 0.0])    less repeatable: no
  θ remaining               4.1 [1.6, 6.8]
  Verdict                   FAIL  (the increment is not lower)

Averaging 2 reconstructions per time point  (30 patients, 2286 set pairs)
  Δθ                        -1.9 [-3.1, -0.8]    lowers the increment: yes
  Matched follow-ups        13.5 → 9.8  (change -3.7 [-4.6, -2.9])    less repeatable: no
  θ remaining               3.8 [1.6, 6.4]
  Verdict                   PASS
```

The example volumes are simulated. Per-setting calibration aligns the settings yet fails the test; averaging two reconstructions per time point passes.

## Your data

One row per image, comma or tab separated. Leave `volume` empty when a measurement failed or was rejected.

| patient | scan | setting | volume |
|---|---|---|---|
| P01 | 1 | 1.25L | 13552.1 |
| P01 | 2 | 1.25L | 16381.9 |
| P01 | 2 | 5.0S | 17401.7 |

`scan` is 1 for the baseline and 2 for the same-day rescan, which plays the follow-up. `setting` is any label for a reconstruction.

Some corrections depend on both settings of a pair, such as converting both scans to the coarser setting. Give those one row per direction instead:
`patient, baseline_setting, followup_setting, baseline_volume, matched_volume, mismatched_volume`.

## What it reports

* **θ, the mismatch increment.** For every patient and ordered pair of settings (a, b), the change of a follow-up reconstructed with b minus that of the matched follow-up reconstructed with a, both against the same baseline:
  `c = 100|ln(V2b/V1a)| − 100|ln(V2a/V1a)|`. Each patient counts once. Units are close to percent volume change.
* **Δθ, the effect of a correction.** θ after the correction minus θ before, on the directions available in both.
* **PASS** when the 95% CI of Δθ lies below 0 and matched follow-ups do not become less repeatable (the CI of their change in mean `100|ln V2/V1|` does not lie above 0).
* **`--average K`** applies the same test to averaging K reconstructions per time point (geometric mean), with no new segmentation.

CIs are percentile intervals from 10 000 whole-patient bootstrap replicates (`--boot`, `--seed`). Add `--json` for machine-readable output.

## Python

```python
from mismatch_increment_test import mismatch_increment, judge, judge_averaging

mismatch_increment("volumes.csv").theta            # 5.7 [2.6, 9.4]
judge("volumes.csv", "calibrated.csv").passed      # False
judge_averaging("volumes.csv", k=2).passed         # True
```

## Citation

Jintao Song, Ying Gong. *Quantifying Harmonization-Resistant Volumetric Errors from Reconstruction Mismatch in Promptable Lung Tumor Segmentation.* Under review, 2026.

## License

MIT
