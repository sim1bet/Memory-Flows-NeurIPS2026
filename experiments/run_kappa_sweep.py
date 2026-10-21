#!/usr/bin/env python3
"""Asymptotic escape time versus slow-feedback gain kappa."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.activations import activation, branch_amplitude
from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.events import measure_escape_events, terminal_plateau
from idp_hopfield.graphs import cycle_adjacency
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import orthogonal_patterns
from idp_hopfield.theory import asymptotic_escape, critical_point


def simulate_batched(
    patterns: np.ndarray,
    adjacency: np.ndarray,
    kappas: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    n, p = patterns.shape
    k_count = len(kappas)
    epsilon = float(cfg["tau_fast"]) / float(cfg["tau_slow"])
    dt_fast = float(cfg["dt_fast"])
    dt_slow = dt_fast * epsilon
    stride = max(1, int(round(float(cfg["sample_dt_slow"]) / dt_slow)))
    sample_dt = stride * dt_slow
    n_steps = int(np.ceil(float(cfg["slow_time_horizon"]) / dt_slow))
    n_steps = int(np.ceil(n_steps / stride) * stride)
    n_samples = n_steps // stride + 1

    coefficients = np.zeros((p, k_count))
    coefficients[int(cfg["initial_source"]), :] = branch_amplitude(
        float(cfg["initial_saliency"]), cfg["activation"]
    )
    saliency = np.zeros((p, k_count))
    saliency[int(cfg["initial_source"]), :] = float(cfg["initial_saliency"])

    rng = np.random.default_rng(int(cfg["seed"]))
    period = int(cfg.get("noise_period", 1024))
    coefficient_noise = float(cfg["fast_noise_neuron_equivalent"]) / np.sqrt(n)
    noise = rng.standard_normal((period, p, 1))
    noise -= noise.mean(axis=0, keepdims=True)
    noise *= coefficient_noise

    z_history = np.empty((k_count, p, n_samples), dtype=np.float32)
    overlap_history = np.empty_like(z_history)

    def projection_from(coeff: np.ndarray) -> np.ndarray:
        state = patterns @ coeff
        phi = activation(state, cfg["activation"])
        return patterns.T @ phi / float(n)

    projection = projection_from(coefficients)
    z_history[:, :, 0] = saliency.T
    overlap_history[:, :, 0] = np.abs(projection).T
    sample = 1
    for step in tqdm(range(1, n_steps + 1), desc="kappa sweep", leave=False):
        projection = projection_from(coefficients)
        fast_rhs = -coefficients + np.abs(saliency) * projection
        slow_rhs = -saliency + (adjacency @ projection) * kappas[None, :]
        next_coefficients = coefficients + dt_fast * fast_rhs + noise[(step - 1) % period]
        next_saliency = saliency + dt_slow * slow_rhs
        coefficients, saliency = next_coefficients, next_saliency
        if step % stride == 0:
            projection = projection_from(coefficients)
            z_history[:, :, sample] = saliency.T
            overlap_history[:, :, sample] = np.abs(projection).T
            sample += 1
    return z_history, overlap_history, sample_dt


def run_kappa_batch(task):
    return simulate_batched(*task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)

    critical = critical_point(cfg["activation"])
    q = np.linspace(0.0, 1.0, int(cfg["kappa_points"]))
    kappa_min = critical.kappa + float(cfg["minimum_offset"])
    kappas = kappa_min + (float(cfg["kappa_max"]) - kappa_min) * q ** float(cfg["grid_power"])
    patterns = orthogonal_patterns(int(cfg["N"]), int(cfg["P"]))
    adjacency = cycle_adjacency(int(cfg["P"]), np.asarray(cfg["cycle"], dtype=int))
    batch_size = int(cfg.get("parallel_kappa_batch_size", 2))
    if batch_size < 1:
        raise ValueError("parallel_kappa_batch_size must be positive")
    # Fixed widths make BLAS rounding and checkpoint keys independent of workers.
    chunks = [kappas[i:i+batch_size] for i in range(0, len(kappas), batch_size)]
    batches = list(ordered_map(run_kappa_batch, [(patterns, adjacency, chunk, cfg) for chunk in chunks], args.workers, "kappa_sweep"))
    z_history = np.concatenate([batch[0] for batch in batches])
    overlap_history = np.concatenate([batch[1] for batch in batches])
    sample_dt = batches[0][2]

    event_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for index, kappa in enumerate(kappas):
        events = measure_escape_events(
            z_history[index],
            overlap_history[index],
            adjacency,
            sample_dt,
            float(cfg["tau_slow"]),
            float(kappa),
            cfg["activation"],
            hold_time=float(cfg["switch_hold_time"]),
        )
        events.insert(0, "kappa", float(kappa))
        event_frames.append(events)
        plateau = terminal_plateau(
            events,
            float(cfg["terminal_window_time"]),
            int(cfg["plateau_min_events"]),
            float(cfg["plateau_relative_tolerance"]),
        )
        theory = asymptotic_escape(float(kappa), cfg["activation"], critical)
        summaries.append(
            {
                **theory,
                "activation": cfg["activation"],
                "T_esc_empirical_infty": plateau.mean,
                "T_esc_empirical_window_std": plateau.std,
                "T_esc_empirical_window_sem": plateau.sem,
                "plateau_event_count": plateau.n_events,
                "plateau_window_start": plateau.window_start,
                "plateau_window_end": plateau.window_end,
                "plateau_relative_range": plateau.relative_range,
                "plateau_relative_drift": plateau.relative_drift,
                "plateau_slope": plateau.slope,
                "plateau_ok": plateau.ok,
                "relative_error_percent": (
                    100.0 * (plateau.mean - theory["T_esc_theory_infty"]) / theory["T_esc_theory_infty"]
                    if np.isfinite(plateau.mean) and np.isfinite(theory["T_esc_theory_infty"])
                    else np.nan
                ),
            }
        )

    dense_kappa = np.linspace(critical.kappa, float(cfg["kappa_max"]), int(cfg["theory_points"]))
    theory_curve = pd.DataFrame(
        [
            {**asymptotic_escape(float(kappa), cfg["activation"], critical), "activation": cfg["activation"]}
            for kappa in dense_kappa
        ]
    )
    write_csv(pd.concat(event_frames, ignore_index=True), output / "kappa_sweep_events.csv")
    write_csv(pd.DataFrame(summaries), output / "kappa_sweep_summary.csv")
    write_csv(theory_curve, output / "kappa_sweep_theory.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "activation": critical.activation,
                    "kappa_critical": critical.kappa,
                    "source_at_escape_critical": critical.source_at_escape,
                    "fixed_amplitude_critical": critical.fixed_amplitude,
                    "T_esc_critical": critical.escape_time,
                }
            ]
        ),
        output / "kappa_critical_point.csv",
    )
    write_json(cfg, output / "kappa_sweep_config.json")


if __name__ == "__main__":
    main()
