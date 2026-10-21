#!/usr/bin/env python3
"""Measure the gap T_switch-T_esc over a fast-state noise sweep."""

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

from idp_hopfield.activations import branch_amplitude, stability_threshold
from idp_hopfield.config import load_config
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.stats import bootstrap_mean_interval, wilson_interval
from idp_hopfield.theory import asymptotic_escape, critical_point


def projections(source: np.ndarray, successor: np.ndarray, gain_phi: float) -> tuple[np.ndarray, np.ndarray]:
    plus = np.tanh(gain_phi * (source + successor))
    minus = np.tanh(gain_phi * (source - successor))
    return 0.5 * (plus + minus), 0.5 * (plus - minus)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["activation"] != "tanh":
        raise ValueError("The exact two-memory noise module currently supports tanh.")
    output = ensure_dir(args.output)

    critical = critical_point("tanh")
    theory = asymptotic_escape(float(cfg["kappa"]), "tanh", critical)
    z_fixed = float(theory["Z_theory_infty"])
    gamma = branch_amplitude(z_fixed, "tanh", float(cfg["gain_phi"]), float(cfg["gain_w"]))
    noise_levels = np.geomspace(float(cfg["noise_min"]), float(cfg["noise_max"]), int(cfg["noise_points"]))
    n_trials = int(cfg["trials"])
    shape = (noise_levels.size, n_trials)

    dt_fast = float(cfg["dt_fast"])
    dt_slow = dt_fast * float(cfg["tau_fast"]) / float(cfg["tau_slow"])
    n_steps = int(np.ceil(float(cfg["slow_time_horizon"]) / dt_slow))
    hold_steps = max(1, int(round(float(cfg["switch_hold_time"]) / dt_slow)))

    a_source = np.full(shape, gamma)
    a_successor = np.zeros(shape)
    z_source = np.full(shape, z_fixed)
    z_successor = np.zeros(shape)
    t_escape = np.full(shape, np.nan)
    t_switch = np.full(shape, np.nan)
    hold = np.zeros(shape, dtype=int)
    p_source, p_successor = projections(a_source, a_successor, float(cfg["gain_phi"]))
    previous_margin = np.abs(z_successor) - stability_threshold(np.abs(z_source), "tanh")

    scale = noise_levels[:, None] / np.sqrt(float(cfg["N"]))
    if cfg["noise_convention"] == "diffusion":
        scale *= np.sqrt(dt_fast)
    elif cfg["noise_convention"] != "per_step":
        raise ValueError("noise_convention must be per_step or diffusion.")
    rng = np.random.default_rng(int(cfg["seed"]))

    for step in tqdm(range(1, n_steps + 1), desc="noise sweep", leave=False):
        fast_source_rhs = -a_source + float(cfg["gain_w"]) * np.abs(z_source) * p_source
        fast_successor_rhs = -a_successor + float(cfg["gain_w"]) * np.abs(z_successor) * p_successor
        slow_source_rhs = -z_source
        slow_successor_rhs = -z_successor + float(cfg["kappa"]) * p_source
        next_source = a_source + dt_fast * fast_source_rhs + scale * rng.standard_normal(n_trials)[None, :]
        next_successor = a_successor + dt_fast * fast_successor_rhs + scale * rng.standard_normal(n_trials)[None, :]
        next_z_source = z_source + dt_slow * slow_source_rhs
        next_z_successor = z_successor + dt_slow * slow_successor_rhs
        a_source, a_successor = next_source, next_successor
        z_source, z_successor = next_z_source, next_z_successor
        p_source, p_successor = projections(a_source, a_successor, float(cfg["gain_phi"]))

        margin = np.abs(z_successor) - stability_threshold(np.abs(z_source), "tanh")
        new_escape = np.isnan(t_escape) & (previous_margin < 0.0) & (margin >= 0.0)
        denominator = margin - previous_margin
        fraction = np.zeros(shape)
        mask = new_escape & (denominator != 0.0)
        fraction[mask] = -previous_margin[mask] / denominator[mask]
        t_escape[new_escape] = ((step - 1) + fraction[new_escape]) * dt_slow

        clean = (
            np.abs(p_successor) >= float(cfg["switch_min_overlap"])
        ) & (
            np.abs(p_successor) >= np.abs(p_source) + float(cfg["switch_margin"])
        )
        hold = np.where(clean, hold + 1, 0)
        new_switch = np.isnan(t_switch) & (hold >= hold_steps)
        t_switch[new_switch] = (step - hold_steps + 1) * dt_slow
        previous_margin = margin
        if np.all(np.isfinite(t_escape) & np.isfinite(t_switch)):
            break

    rows: list[dict[str, object]] = []
    for level_index, sigma in enumerate(noise_levels):
        for trial in range(n_trials):
            escape = float(t_escape[level_index, trial])
            switch = float(t_switch[level_index, trial])
            gap = switch - escape if np.isfinite(escape) and np.isfinite(switch) else np.nan
            if np.isfinite(gap):
                outcome = "post_escape_switch" if gap >= 0.0 else "premature_switch"
            elif np.isfinite(escape):
                outcome = "no_switch_after_escape"
            elif np.isfinite(switch):
                outcome = "switch_without_detected_escape"
            else:
                outcome = "unresolved"
            rows.append(
                {
                    "noise_amplitude": float(sigma),
                    "noise_level_index": level_index,
                    "trial": trial,
                    "seed": int(cfg["seed"]),
                    "noise_convention": cfg["noise_convention"],
                    "kappa": float(cfg["kappa"]),
                    "N": int(cfg["N"]),
                    "T_esc_theory_infty": float(theory["T_esc_theory_infty"]),
                    "Z_theory_infty": z_fixed,
                    "T_esc_empirical": escape,
                    "T_switch_empirical": switch,
                    "switch_delay_after_stability_loss": gap,
                    "relative_switch_delay": gap / escape if np.isfinite(gap) and escape else np.nan,
                    "outcome": outcome,
                }
            )
    trials = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for index, group in trials.groupby("noise_level_index", sort=True):
        valid = group["switch_delay_after_stability_loss"].dropna().to_numpy(float)
        bootstrap_rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), int(index), 991]))
        low, high = bootstrap_mean_interval(valid, bootstrap_rng, int(cfg["bootstrap_reps"]))
        premature = int((group["outcome"] == "premature_switch").sum())
        p_low, p_high = wilson_interval(premature, len(group))
        summary_rows.append(
            {
                "noise_amplitude": float(group["noise_amplitude"].iloc[0]),
                "trials": len(group),
                "resolved_gaps": len(valid),
                "mean_T_esc": float(group["T_esc_empirical"].mean()),
                "mean_T_switch": float(group["T_switch_empirical"].mean()),
                "mean_gap": float(np.mean(valid)) if len(valid) else np.nan,
                "median_gap": float(np.median(valid)) if len(valid) else np.nan,
                "gap_std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "mean_gap_ci_low": low,
                "mean_gap_ci_high": high,
                "premature_switch_probability": premature / len(group),
                "premature_switch_ci_low": p_low,
                "premature_switch_ci_high": p_high,
                "post_escape_switch_probability": float((group["outcome"] == "post_escape_switch").mean()),
            }
        )
    write_csv(trials, output / "noise_sweep_trials.csv")
    write_csv(pd.DataFrame(summary_rows), output / "noise_sweep_summary.csv")
    write_json(cfg, output / "noise_sweep_config.json")


if __name__ == "__main__":
    main()
