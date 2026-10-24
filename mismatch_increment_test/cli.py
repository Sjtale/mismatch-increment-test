"""Command line: mismatch-increment-test VOLUMES [--corrected CORRECTED ...] [--average K] [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .core import N_BOOT, SEED, judge, judge_averaging, mismatch_increment, read_table


def _verdict_lines(v, label):
    yes = lambda b: "yes" if b else "no"
    out = [f"\n{label}  ({v.n_patients} patients, {v.n_directions} "
           f"{'set pairs' if 'k' in v.extra else 'directions in common'})",
           f"  Δθ                        {v.delta_theta:+.1f}    lowers the increment: {yes(v.lowers_increment)}",
           f"  Matched follow-ups        {v.matched_before:.1f} → {v.matched_after:.1f}  "
           f"(change {v.matched_change:+.1f})    less repeatable: {yes(v.less_repeatable)}"]
    if v.theta_after is not None:
        out.append(f"  θ remaining               {v.theta_after:.1f}")
    if v.passed:
        out.append("  Verdict                   PASS")
    elif not v.lowers_increment:
        out.append("  Verdict                   FAIL  (the increment is not lower)")
    else:
        out.append("  Verdict                   FAIL  (matched follow-ups become less repeatable)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mismatch-increment-test",
        description="Test whether a correction lowers the volume error that a differently reconstructed "
                    "follow-up adds, using same-day repeat scans.")
    ap.add_argument("volumes", help="native volumes: CSV/TSV with patient, scan (1 or 2), setting, volume")
    ap.add_argument("--corrected", nargs="+", default=[], metavar="TABLE",
                    help="volumes after a correction (image table, or direction table for corrections that depend "
                         "on both settings of a pair)")
    ap.add_argument("--average", type=int, default=0, metavar="K",
                    help="also test averaging K reconstructions per time point (e.g. 2)")
    ap.add_argument("--boot", type=int, default=N_BOOT, help=f"bootstrap replicates (default {N_BOOT})")
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap seed")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    ap.add_argument("--version", action="version", version=f"mismatch-increment-test {__version__}")
    a = ap.parse_args(argv)

    try:
        native = read_table(a.volumes)
        inc = mismatch_increment(native, a.boot, a.seed)
        verdicts = [(path, judge(native, read_table(path), a.boot, a.seed)) for path in a.corrected]
        if a.average:
            verdicts.append((f"average {a.average}", judge_averaging(native, a.average, a.boot, a.seed)))
    except (OSError, ValueError) as exc:
        ap.exit(2, f"mismatch-increment-test: error: {exc}\n")

    if a.json:
        print(json.dumps(dict(volumes=a.volumes, increment=inc.as_dict(), boot=a.boot, seed=a.seed,
                              corrections={k: v.as_dict() for k, v in verdicts}), indent=1))
        return 0

    excess = 100 * (inc.mismatched - inc.matched) / inc.matched if inc.matched else float("nan")
    lines = [f"Mismatch-increment test · {a.volumes} · {inc.n_patients} patients, {inc.n_directions} directions",
             f"  Mismatch increment θ      {inc.theta:.1f}",
             f"  Matched follow-ups        {inc.matched:.1f}   (mean 100·|ln V2/V1|)",
             f"  Mismatched follow-ups     {inc.mismatched:.1f}   ({excess:+.0f}% vs. matched)"]
    for path, v in verdicts:
        label = f"Averaging {v.extra['k']} reconstructions per time point" if "k" in v.extra else f"Correction · {path}"
        lines += _verdict_lines(v, label)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
