#!/usr/bin/env python3
"""Validate reduced escape-time theory against the full finite-epsilon model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.config import load_config
from idp_hopfield.events import measure_escape_events
from idp_hopfield.graphs import cycle_adjacency
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import orthogonal_patterns
from idp_hopfield.simulation import DynamicsConfig, simulate_trajectory
from idp_hopfield.theory import fixed_points, theoretical_escape_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)

    n, p = int(cfg["N"]), int(cfg["P"])
    patterns = orthogonal_patterns(n, p)
    order = np.asarray(cfg["cycle"], dtype=int)
    adjacency = cycle_adjacency(p, order)
    dynamics = DynamicsConfig(
        activation=cfg["activation"],
        kappa=float(cfg["kappa"]),
        tau_fast=float(cfg["tau_fast"]),
        tau_slow=float(cfg["tau_slow"]),
        dt_fast=float(cfg["dt_fast"]),
        slow_time_horizon=float(cfg["slow_time_horizon"]),
        sample_dt_slow=float(cfg["sample_dt_slow"]),
        fast_noise=float(cfg["fast_noise"]),
        slow_noise=0.0,
        seed=int(cfg["seed"]),
    )
    history = simulate_trajectory(
        patterns,
        adjacency,
        dynamics,
        initial_source=int(cfg["initial_source"]),
        initial_saliency=float(cfg["initial_saliency"]),
        initial_fast_perturbation=float(cfg["initial_perturbation"]),
    )
    events = measure_escape_events(
        np.asarray(history["saliency"]),
        np.asarray(history["activated_overlap"]),
        adjacency,
        float(history["sample_dt_slow"]),
        dynamics.tau_slow,
        dynamics.kappa,
        dynamics.activation,
        hold_time=float(cfg["switch_hold_time"]),
    )
    valid = events.dropna(subset=["T_esc_empirical", "T_esc_theory"])
    steady = valid.iloc[1:] if len(valid) > 1 else valid
    z_infty = float(steady["Z_entry"].mean()) if len(steady) else np.nan
    theory_at_z, _, _ = theoretical_escape_time(
        z_infty,
        dynamics.kappa,
        dynamics.activation,
    ) if np.isfinite(z_infty) else (np.nan, np.nan, np.nan)
    roots = fixed_points(dynamics.kappa, dynamics.activation)
    stable = max(roots) if roots else np.nan
    theory_stable, _, _ = theoretical_escape_time(
        stable, dynamics.kappa, dynamics.activation
    ) if np.isfinite(stable) else (np.nan, np.nan, np.nan)
    summary = pd.DataFrame(
        [
            {
                "activation": dynamics.activation,
                "kappa": dynamics.kappa,
                "N": n,
                "P": p,
                "seed": dynamics.seed,
                "tau_fast": dynamics.tau_fast,
                "tau_slow": dynamics.tau_slow,
                "epsilon": dynamics.epsilon,
                "dt_fast": dynamics.dt_fast,
                "sample_dt_slow": float(history["sample_dt_slow"]),
                "valid_events": len(valid),
                "steady_events": len(steady),
                "Z_infty_empirical": z_infty,
                "T_esc_empirical_mean": float(steady["T_esc_empirical"].mean()) if len(steady) else np.nan,
                "T_esc_theory_eventwise_mean": float(steady["T_esc_theory"].mean()) if len(steady) else np.nan,
                "T_esc_theory_at_Z_infty": theory_at_z,
                "fixed_points": ";".join(f"{root:.12g}" for root in roots),
                "stable_fixed_point": stable,
                "T_esc_theory_at_stable_fixed_point": theory_stable,
                "mean_abs_relative_error_percent": float(valid["T_esc_relative_error_percent"].abs().mean()) if len(valid) else np.nan,
            }
        ]
    )
    time = np.asarray(history["time_slow"])
    series = []
    for memory in range(p):
        series.append(pd.DataFrame({
            "time_slow": time,
            "memory": memory,
            "saliency": np.abs(np.asarray(history["saliency"])[memory]),
            "activated_overlap": np.asarray(history["activated_overlap"])[memory],
            "cosine_overlap": np.asarray(history["cosine"])[memory],
        }))
    write_csv(events, output / "escape_time_events.csv")
    write_csv(pd.concat(series, ignore_index=True), output / "escape_time_timeseries.csv")
    write_csv(summary, output / "escape_time_summary.csv")
    write_json(cfg, output / "escape_time_config.json")


if __name__ == "__main__":
    main()
