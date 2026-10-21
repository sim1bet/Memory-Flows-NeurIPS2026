#!/usr/bin/env python3
"""Run raw correlated-pattern critical-gain, escape-time, and fixed-point sweeps."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from idp_hopfield.config import load_config
from idp_hopfield.correlation_dynamics import Config, run
from idp_hopfield.io import ensure_dir, write_json


def resolved_config(raw: dict) -> Config:
    return Config(
        n_neurons=int(raw["N"]),
        n_memories=int(raw["P"]),
        cycle=tuple(int(v) for v in raw["cycle"]),
        epsilons=tuple(float(v) for v in raw["epsilons"]),
        pattern_sets=int(raw["pattern_sets"]),
        activation=str(raw["activation"]),
        reference_kappa=float(raw["reference_kappa"]),
        supercritical_factor=(None if raw.get("supercritical_factor") is None else float(raw["supercritical_factor"])),
        evaluation_kappa_max=float(raw["evaluation_kappa_max"]),
        kappa_min=float(raw["kappa_min"]),
        kappa_max=float(raw["kappa_max"]),
        kappa_scan_points=int(raw["kappa_scan_points"]),
        kappa_bisection_steps=int(raw["kappa_bisection_steps"]),
        epsilon_max_success_probability=float(raw["epsilon_max_success_probability"]),
        z_initial=float(raw["z_initial"]),
        tau_fast=float(raw["tau_fast"]),
        tau_slow=float(raw["tau_slow"]),
        dt_fast=float(raw["dt_fast"]),
        max_slow_time_per_transition=float(raw["max_slow_time_per_transition"]),
        check_interval=float(raw["check_interval"]),
        hold_time=float(raw["hold_time"]),
        min_alignment=float(raw["min_alignment"]),
        min_margin=float(raw["min_margin"]),
        fast_noise_per_neuron=float(raw["fast_noise_per_neuron"]),
        asymptotic_cycles=int(raw["asymptotic_cycles"]),
        terminal_cycles=int(raw["terminal_cycles"]),
        plateau_relative_tolerance=float(raw["plateau_relative_tolerance"]),
        seed=int(raw["seed"]),
        bootstrap_reps=int(raw["bootstrap_reps"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raw = load_config(args.config)
    cfg = resolved_config(raw)
    output = ensure_dir(args.output)
    trials, events, summary, theory = run(cfg, workers=args.workers)
    trials.to_csv(output / "correlation_dynamics_trials.csv", index=False)
    events.to_csv(output / "correlation_dynamics_events.csv", index=False)
    summary.to_csv(output / "correlation_dynamics_summary.csv", index=False)
    write_json(asdict(cfg), output / "correlation_dynamics_config.json")
    write_json(theory, output / "correlation_dynamics_theory.json")
    print(json.dumps(theory, indent=2))


if __name__ == "__main__":
    main()
