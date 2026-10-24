"""Synthetic checks with known answers (run: python -m pytest tests)."""
import math
import os
import subprocess
import sys

import numpy as np

from mismatch_increment_test import from_images, judge, judge_averaging, mismatch_increment
from mismatch_increment_test.core import Volumes

SETTINGS = ["1.25L", "1.25S", "2.5L", "2.5S", "5.0L", "5.0S"]
HERE = os.path.dirname(os.path.abspath(__file__))


def images(fn, n=12):
    return {(f"P{i:02d}", sc, s): fn(i, sc, s) for i in range(n) for sc in (1, 2) for s in SETTINGS}


def test_identical_volumes_give_zero():
    v = from_images(images(lambda i, sc, s: 1000.0))
    inc = mismatch_increment(v, n_boot=200)
    assert abs(inc.theta.value) < 1e-12 and inc.n_directions == 12 * 30


def test_fixed_setting_bias_gives_analytic_theta():
    bias = dict(zip(SETTINGS, [0.0, -0.05, 0.03, -0.02, 0.14, 0.08]))
    v = from_images(images(lambda i, sc, s: 1000.0 * math.exp(bias[s])))
    expected = np.mean([100 * abs(bias[b] - bias[a]) for a in SETTINGS for b in SETTINGS if a != b])
    assert abs(mismatch_increment(v, n_boot=200).theta.value - expected) < 1e-9


def test_correction_that_removes_a_shared_bias_passes():
    rng = np.random.default_rng(0)
    bias = dict(zip(SETTINGS, [0.0, -0.05, 0.03, -0.02, 0.14, 0.08]))
    noise = {k: rng.normal(0, 0.05) for k in images(lambda *a: 0, n=40)}
    native = from_images({k: 1000 * math.exp(bias[k[2]] + e) for k, e in noise.items()})
    fixed = from_images({k: 1000 * math.exp(e) for k, e in noise.items()})
    v = judge(native, fixed, n_boot=1000)
    assert v.lowers_increment and not v.less_repeatable and v.passed
    assert abs(v.matched_change.value) < 1e-9          # matched pairs share their setting, so they are unchanged


def test_no_op_correction_fails():
    v = from_images(images(lambda i, sc, s: 1000.0 * (1 + 0.1 * ((i * 7 + sc * 3 + len(s)) % 5))))
    r = judge(v, v, n_boot=200)
    assert abs(r.delta_theta.value) < 1e-12 and not r.passed


def test_direction_table_equals_image_table():
    rng = np.random.default_rng(1)
    im = images(lambda i, sc, s: 1000 * math.exp(rng.normal(0, 0.2)))
    a = from_images(im)
    b = Volumes({p: dict(d) for p, d in a.directions.items()})
    assert abs(mismatch_increment(a, n_boot=100).theta.value - mismatch_increment(b, n_boot=100).theta.value) < 1e-12


def test_averaging_reduces_matched_change_under_noise():
    rng = np.random.default_rng(2)
    v = from_images(images(lambda i, sc, s: 1000 * math.exp(rng.normal(0, 0.1)), n=300))
    r = judge_averaging(v, k=2, n_boot=200)
    assert 0.6 < r.matched_after / r.matched_before < 0.8      # about 1/sqrt(2)


def test_cli_on_examples():
    ex = os.path.join(HERE, "..", "examples")
    out = subprocess.run([sys.executable, "-m", "mismatch_increment_test", os.path.join(ex, "volumes.csv"),
                          "--corrected", os.path.join(ex, "calibrated.csv"), "--average", "2", "--boot", "500"],
                         capture_output=True, text=True, check=True).stdout
    assert "Mismatch increment" in out and "FAIL" in out and "PASS" in out
