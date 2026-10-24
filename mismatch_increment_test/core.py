"""The mismatch-increment test: does a correction lower the error that a mismatched follow-up adds?

Volumes come from same-day repeat scans (scan 1 = baseline, scan 2 = follow-up), each reconstructed with several
settings.  For a patient and an ordered pair of settings (a, b), a != b:

    L(a -> b) = 100 |ln(V2b / V1a)|          change of a follow-up reconstructed with b
    c(a, b)   = L(a -> b) - L(a -> a)        excess change of the mismatched over the matched follow-up

The mismatch increment theta is the mean of c over a patient's directions, averaged over patients (each patient
counts once).  A correction passes the test when the 95% CI of delta-theta = theta_corrected - theta_native lies
below 0 and matched follow-ups do not become less repeatable (the CI of the change in mean L(a -> a) does not lie
above 0).  CIs are percentile intervals of a whole-patient bootstrap.
"""
from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

N_BOOT = 10_000
SEED = 20260911

IMAGE_COLUMNS = ("patient", "scan", "setting", "volume")
DIRECTION_COLUMNS = ("patient", "baseline_setting", "followup_setting",
                     "baseline_volume", "matched_volume", "mismatched_volume")


# ------------------------------------------------------------------------------------------------ data
@dataclass
class Volumes:
    """Volume triplets per direction, and per-image volumes when the input was an image table.

    directions: {patient: {(a, b): (V1a, V2a, V2b)}}
    images:     {(patient, scan, setting): V} with scan 1 or 2, or None for a direction table
    """
    directions: dict
    images: dict | None = None
    name: str = ""

    @property
    def patients(self):
        return sorted(p for p, d in self.directions.items() if d)

    @property
    def n_directions(self):
        return sum(len(d) for d in self.directions.values())


def _num(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in ("na", "nan", "none"):
        return None
    v = float(x)
    return v if math.isfinite(v) and v > 0 else None


def _scan(x):
    x = str(x).strip().lower()
    if x in ("1", "scan1", "baseline", "first"):
        return 1
    if x in ("2", "scan2", "followup", "follow-up", "rescan", "second"):
        return 2
    raise ValueError(f"scan must be 1 (baseline) or 2 (follow-up), got {x!r}")


def from_images(images, name=""):
    """Volumes from {(patient, scan, setting): V}.  Every ordered pair of settings with all three volumes is used."""
    by_patient = {}
    for (p, sc, s), v in images.items():
        if v is not None:
            by_patient.setdefault(p, {})[(sc, s)] = v
    directions = {}
    for p, cells in by_patient.items():
        settings = sorted({s for _, s in cells})
        d = {}
        for a, b in itertools.permutations(settings, 2):
            trip = (cells.get((1, a)), cells.get((2, a)), cells.get((2, b)))
            if None not in trip:
                d[(a, b)] = trip
        directions[p] = d
    return Volumes(directions, {k: v for k, v in images.items() if v is not None}, name)


def read_table(path):
    """Read an image table (patient, scan, setting, volume) or a direction table
    (patient, baseline_setting, followup_setting, baseline_volume, matched_volume, mismatched_volume).
    Comma or tab separated; empty volumes mean a failed or rejected measurement."""
    with open(path, newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rows = list(csv.DictReader(fh, dialect=dialect))
    if not rows:
        raise ValueError(f"{path}: empty table")
    cols = {c.strip().lower() for c in rows[0].keys()}
    rows = [{k.strip().lower(): v for k, v in r.items()} for r in rows]
    if set(IMAGE_COLUMNS) <= cols:
        images = {}
        for r in rows:
            key = (r["patient"].strip(), _scan(r["scan"]), r["setting"].strip())
            if key in images:
                raise ValueError(f"{path}: duplicate row {key}")
            images[key] = _num(r["volume"])
        return from_images(images, str(path))
    if set(DIRECTION_COLUMNS) <= cols:
        directions = {}
        for r in rows:
            p, a, b = r["patient"].strip(), r["baseline_setting"].strip(), r["followup_setting"].strip()
            if a == b:
                raise ValueError(f"{path}: baseline and follow-up setting are equal for {p}")
            trip = tuple(_num(r[c]) for c in DIRECTION_COLUMNS[3:])
            d = directions.setdefault(p, {})
            if (a, b) in d:
                raise ValueError(f"{path}: duplicate direction {p} {a}->{b}")
            if None not in trip:
                d[(a, b)] = trip
        return Volumes(directions, None, str(path))
    raise ValueError(f"{path}: need columns {', '.join(IMAGE_COLUMNS)} or {', '.join(DIRECTION_COLUMNS)}")


def _as_volumes(x):
    return x if isinstance(x, Volumes) else read_table(x)


# ------------------------------------------------------------------------------------------------ statistics
def _L(v2, v1):
    return 100.0 * abs(math.log(v2 / v1))


def _per_patient(dirs, keys=None):
    """(c, L_matched, L_mismatched) per direction -> per-patient means, NaN without directions."""
    out = np.full((3,), np.nan)
    rows = [(_L(v2b, v1a) - _L(v2a, v1a), _L(v2a, v1a), _L(v2b, v1a))
            for k, (v1a, v2a, v2b) in dirs.items() if keys is None or k in keys]
    if rows:
        out = np.mean(np.array(rows), axis=0)
    return out


def _wmean(q, w):
    """Patient-equal mean of per-patient values q (NaN = no data) under bootstrap weights w (rows of W)."""
    has = ~np.isnan(q)
    qz = np.where(has, q, 0.0)
    den = w @ has.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (w @ qz) / den, np.nan)


def _weights(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)


@dataclass
class Estimate:
    value: float
    ci: tuple

    def __format__(self, spec):
        f = spec or ".1f"

        def num(x):
            s = format(x, f)
            return format(0.0, f.replace("+", "")) if float(s) == 0 else s
        return f"{num(self.value)} [{num(self.ci[0])}, {num(self.ci[1])}]"

    @property
    def below_zero(self):
        return self.ci[1] < 0

    @property
    def above_zero(self):
        return self.ci[0] > 0


def _estimate(q, W):
    point = float(_wmean(q, np.ones((1, len(q))))[0])
    reps = _wmean(q, W)
    reps = reps[~np.isnan(reps)]
    return Estimate(point, (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))))


@dataclass
class Increment:
    """Mismatch increment of one set of volumes."""
    theta: Estimate
    matched: float
    mismatched: float
    n_patients: int
    n_directions: int
    name: str = ""

    def as_dict(self):
        return dict(theta=self.theta.value, theta_ci=list(self.theta.ci), matched_change=self.matched,
                    mismatched_change=self.mismatched, n_patients=self.n_patients, n_directions=self.n_directions)


def mismatch_increment(volumes, n_boot=N_BOOT, seed=SEED):
    """theta with its bootstrap CI, and the mean change of matched and mismatched follow-ups."""
    v = _as_volumes(volumes)
    pats = v.patients
    if not pats:
        raise ValueError(f"{v.name}: no patient has a complete direction (V1a, V2a, V2b)")
    Q = np.array([_per_patient(v.directions[p]) for p in pats])
    W = _weights(len(pats), n_boot, seed)
    one = np.ones((1, len(pats)))
    return Increment(_estimate(Q[:, 0], W), float(_wmean(Q[:, 1], one)[0]), float(_wmean(Q[:, 2], one)[0]),
                     len(pats), v.n_directions, v.name)


@dataclass
class Verdict:
    """Result of the mismatch-increment test for one correction."""
    delta_theta: Estimate
    matched_before: float
    matched_after: float
    matched_change: Estimate
    theta_after: Estimate | None
    n_patients: int
    n_directions: int
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def lowers_increment(self):
        return self.delta_theta.below_zero

    @property
    def less_repeatable(self):
        return self.matched_change.above_zero

    @property
    def passed(self):
        return self.lowers_increment and not self.less_repeatable

    def as_dict(self):
        d = dict(delta_theta=self.delta_theta.value, delta_theta_ci=list(self.delta_theta.ci),
                 matched_change_before=self.matched_before, matched_change_after=self.matched_after,
                 matched_change_difference=self.matched_change.value,
                 matched_change_difference_ci=list(self.matched_change.ci),
                 lowers_increment=self.lowers_increment, less_repeatable=self.less_repeatable,
                 passed=self.passed, n_patients=self.n_patients, n_directions=self.n_directions)
        if self.theta_after is not None:
            d.update(theta_after=self.theta_after.value, theta_after_ci=list(self.theta_after.ci))
        d.update(self.extra)
        return d


def judge(native, corrected, n_boot=N_BOOT, seed=SEED):
    """Mismatch-increment test of a correction: delta-theta and the change in matched-follow-up change, on the directions
    complete under both, with one shared whole-patient bootstrap."""
    nat, cor = _as_volumes(native), _as_volumes(corrected)
    pats = sorted(set(nat.patients) | set(cor.patients))
    rows = []
    for p in pats:
        dn, dc = nat.directions.get(p, {}), cor.directions.get(p, {})
        joint = set(dn) & set(dc)
        qn, qc = _per_patient(dn, joint), _per_patient(dc, joint)
        rows.append(np.concatenate([qc - qn, qn, qc, _per_patient(dc)]))
    Q = np.array(rows)                      # [dc, dLm, dLx, nc, nLm, nLx, cc, cLm, cLx, own_cc, own_cLm, own_cLx]
    W = _weights(len(pats), n_boot, seed)
    one = np.ones((1, len(pats)))
    n_joint = sum(len(set(nat.directions.get(p, {})) & set(cor.directions.get(p, {}))) for p in pats)
    if n_joint == 0:
        raise ValueError("native and corrected volumes share no complete direction")
    return Verdict(_estimate(Q[:, 0], W), float(_wmean(Q[:, 4], one)[0]), float(_wmean(Q[:, 7], one)[0]),
                   _estimate(Q[:, 1], W), _estimate(Q[:, 9], W), int((~np.isnan(Q[:, 0])).sum()), n_joint,
                   cor.name)


def judge_averaging(native, k=2, n_boot=N_BOOT, seed=SEED):
    """Mismatch-increment test of averaging k reconstructions per time point (geometric mean of the volumes).

    For every baseline set Sb of k settings and disjoint follow-up set Sf of k settings with all volumes present,
    the averaged mismatched change is 100|ln(Vbar2(Sf) / Vbar1(Sb))| and the averaged matched change
    100|ln(Vbar2(Sb) / Vbar1(Sb))|.  They are compared with the single-reconstruction changes of the same
    directions (a in Sb, b in Sf).  Needs an image table."""
    nat = _as_volumes(native)
    if nat.images is None:
        raise ValueError("averaging needs an image table (patient, scan, setting, volume)")
    if k < 2:
        raise ValueError("k must be at least 2")
    cells = {}
    for (p, sc, s), v in nat.images.items():
        cells.setdefault(p, {})[(sc, s)] = math.log(v)
    pats = sorted(cells)
    rows, n_sets = [], 0
    for p in pats:
        lv = cells[p]
        settings = sorted({s for _, s in lv})
        vals = []
        for Sb in itertools.combinations(settings, k):
            rest = [s for s in settings if s not in Sb]
            for Sf in itertools.combinations(rest, k):
                need = [(1, a) for a in Sb] + [(2, a) for a in Sb] + [(2, b) for b in Sf]
                if not all(x in lv for x in need):
                    continue
                m1b = np.mean([lv[(1, a)] for a in Sb])
                m2b = np.mean([lv[(2, a)] for a in Sb])
                m2f = np.mean([lv[(2, b)] for b in Sf])
                Lbar_X, Lbar_M = 100 * abs(m2f - m1b), 100 * abs(m2b - m1b)
                L_X = np.mean([100 * abs(lv[(2, b)] - lv[(1, a)]) for a in Sb for b in Sf])
                L_M = np.mean([100 * abs(lv[(2, a)] - lv[(1, a)]) for a in Sb])
                vals.append((Lbar_X, Lbar_M, L_X, L_M))
        n_sets += len(vals)
        if vals:
            Lbar_X, Lbar_M, L_X, L_M = np.mean(np.array(vals), axis=0)
            rows.append(((Lbar_X - Lbar_M) - (L_X - L_M), Lbar_M - L_M, L_M, Lbar_M, Lbar_X - Lbar_M))
        else:
            rows.append((np.nan,) * 5)
    Q = np.array(rows)
    if np.isnan(Q[:, 0]).all():
        raise ValueError(f"no patient has two disjoint sets of {k} settings with all volumes")
    W = _weights(len(pats), n_boot, seed)
    one = np.ones((1, len(pats)))
    return Verdict(_estimate(Q[:, 0], W), float(_wmean(Q[:, 2], one)[0]), float(_wmean(Q[:, 3], one)[0]),
                   _estimate(Q[:, 1], W), _estimate(Q[:, 4], W), int((~np.isnan(Q[:, 0])).sum()), n_sets,
                   f"averaging {k} reconstructions per time point", extra=dict(k=k))
