"""Write the synthetic example tables (simulated volumes, not patient data).

30 lesions, same-day scans 1 and 2, six reconstructions (slice thickness 1.25 / 2.5 / 5.0 mm x kernel L / S).
Each reconstruction shifts the measured volume by a setting effect that differs between lesions (partial volume
grows as lesions shrink and depends on lesion shape), plus independent measurement noise.

    volumes.csv      native volumes (a few failed measurements are left empty)
    calibrated.csv   the same volumes after per-setting calibration to 1.25L (one factor per setting)
"""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = ["1.25L", "1.25S", "2.5L", "2.5S", "5.0L", "5.0S"]
THICK = {"1.25": 1.25, "2.5": 2.5, "5.0": 5.0}


def simulate(seed=7, n=30):
    rng = np.random.default_rng(seed)
    rows = {}
    for i in range(n):
        p = f"P{i + 1:02d}"
        d = float(np.exp(rng.uniform(np.log(10), np.log(60))))             # diameter, mm
        v0 = math.pi / 6 * d ** 3
        shape = rng.normal(0, 1)                                             # lesion-specific response
        kern = rng.normal(0, 0.05)
        for scan in (1, 2):
            for s in SETTINGS:
                t, k = THICK[s[:-1]], s[-1]
                bias = (0.9 * (t - 1.25) / d) * (1 + 1.2 * shape)            # partial volume, lesion-specific
                bias += (0.03 + kern) * (k == "S")                           # kernel, lesion-specific
                noise = rng.normal(0, 0.10 + 0.6 / d)
                v = v0 * math.exp(bias + noise)
                fail = rng.random() < 0.03
                rows[(p, scan, s)] = None if fail else v
    return rows


def calibrate(rows):
    """Divide each volume by the median same-scan ratio of its setting to 1.25L."""
    fac = {}
    for s in SETTINGS:
        r = [rows[(p, sc, s)] / rows[(p, sc, "1.25L")] for (p, sc, ss) in rows
             if ss == s and rows[(p, sc, s)] and rows[(p, sc, "1.25L")]]
        fac[s] = float(np.median(r))
    return {k: (v / fac[k[2]] if v else None) for k, v in rows.items()}


def write(rows, name):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "scan", "setting", "volume"])
        for (p, sc, s), v in sorted(rows.items()):
            w.writerow([p, sc, s, "" if v is None else f"{v:.1f}"])


if __name__ == "__main__":
    native = simulate()
    write(native, "volumes.csv")
    write(calibrate(native), "calibrated.csv")
    print("wrote examples/volumes.csv and examples/calibrated.csv")
