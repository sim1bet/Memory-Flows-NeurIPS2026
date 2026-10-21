#!/usr/bin/env python3
"""Fail when a required reproducibility artifact is missing or malformed."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED = {
    "model_comparison": ["model_comparison_timeseries.csv", "model_comparison_summary.csv"],
    "limit_cycles": ["limit_cycles_timeseries.csv", "limit_cycles_summary.csv"],
    "escape_time": ["escape_time_events.csv", "escape_time_summary.csv"],
    "return_map": ["return_map.csv", "return_map_fixed_points.csv"],
    "kappa_sweep": ["kappa_sweep_events.csv", "kappa_sweep_summary.csv", "kappa_sweep_theory.csv"],
    "noise_sweep": ["noise_sweep_trials.csv", "noise_sweep_summary.csv"],
    "capacity": ["capacity_trials.csv", "capacity_pattern_sets.csv", "capacity_probability.csv"],
    "correlation": ["correlation_trials.csv", "correlation_summary.csv"],
    "correlation_dynamics": ["correlation_dynamics_trials.csv", "correlation_dynamics_events.csv", "correlation_dynamics_summary.csv"],
    "correlation_phase_plane": ["correlation_phase_plane_trials.csv", "correlation_phase_plane_summary.csv"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    failures: list[str] = []
    for experiment, names in EXPECTED.items():
        for name in names:
            path = root / experiment / name
            if not path.exists():
                failures.append(f"missing {path}")
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                failures.append(f"empty {path}")
            if any(column.startswith("Unnamed:") for column in frame.columns):
                failures.append(f"unnamed index column in {path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {sum(map(len, EXPECTED.values()))} CSV contracts under {root}")


if __name__ == "__main__":
    main()
