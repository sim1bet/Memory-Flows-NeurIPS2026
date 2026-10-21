#!/usr/bin/env python3
"""Phase plane of escape time versus correlation and slow gain for raw patterns."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.correlation_dynamics import Config, orthogonal_fixed_point, simulate_asymptotic_events
from idp_hopfield.graphs import cycle_adjacency, successor_array
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import correlated_patterns
from idp_hopfield.theory import theoretical_escape_time


def resolved_base(raw: dict[str, Any]) -> Config:
    return Config(
        n_neurons=int(raw["N"]),
        n_memories=int(raw["P"]),
        cycle=tuple(int(v) for v in raw["cycle"]),
        epsilons=tuple(float(v) for v in raw["epsilons"]),
        pattern_sets=int(raw["pattern_sets"]),
        activation=str(raw["activation"]),
        reference_kappa=float(raw["kappa_min"]),
        supercritical_factor=None,
        evaluation_kappa_max=float(raw["kappa_max"]),
        kappa_min=float(raw["kappa_min"]),
        kappa_max=float(raw["kappa_max"]),
        kappa_scan_points=int(raw["kappa_points"]),
        kappa_bisection_steps=1,
        epsilon_max_success_probability=float(raw.get("transition_probability_threshold", 0.5)),
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
        plateau_relative_tolerance=float(raw.get("plateau_relative_tolerance", 1.0)),
        seed=int(raw["seed"]),
        bootstrap_reps=int(raw["bootstrap_reps"]),
    )


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)



def escape_horizon(raw: dict[str, Any], reference_times, observed_switches=()) -> float:
    """Per-edge allowance in slow time, including post-escape ejection/holding."""
    factor = float(raw.get("escape_horizon_factor", 3.0))
    padding = float(raw.get("escape_horizon_padding", 1.0))
    if not np.isfinite(factor) or factor <= 1 or not np.isfinite(padding) or padding < 0:
        raise ValueError("escape_horizon_factor must exceed 1 and padding must be nonnegative")
    times = np.asarray(list(reference_times) + list(observed_switches), dtype=float)
    times = times[np.isfinite(times) & (times >= 0)]
    return max(float(raw["max_slow_time_per_transition"]),
               factor * float(times.max(initial=0)) + padding + float(raw["hold_time"]))


def simulate_with_escape_horizon(patterns, successor, cfg, seed, raw, reference_time):
    """Replay a censored trajectory with its same seed and a longer allowance.

    Identical seeds preserve the stochastic path up to the earlier stopping
    time. Wrong-memory/collapse outcomes are not retried or relabeled success.
    """
    initial_time, _, _ = theoretical_escape_time(cfg.z_initial, cfg.reference_kappa, cfg.activation)
    horizon = escape_horizon(raw, [reference_time, initial_time])
    cap = float(raw.get("horizon_max_slow_time", 60.0))
    retries = int(raw.get("horizon_max_retries", 3))
    if not np.isfinite(cap) or cap < horizon or retries < 0:
        raise ValueError("horizon_max_slow_time must cover the escape-based initial allowance; retries >= 0")
    history = []
    for attempt in range(retries + 1):
        local = replace(cfg, max_slow_time_per_transition=horizon)
        result, events = simulate_asymptotic_events(patterns, successor, local, seed)
        history.append({"attempt": attempt, "edge_time_budget": horizon,
                        "outcome": result["reference_outcome"],
                        "completed_transitions": result["reference_completed_transitions"],
                        "pending_T_esc": result.get("pending_T_esc", np.nan)})
        if result["reference_outcome"] != "timeout" or attempt == retries or horizon >= cap:
            break
        observed_escapes = [e["T_esc"] for e in events] + [result.get("pending_T_esc", np.nan)]
        observed_switches = [e["T_switch"] for e in events]
        horizon = min(cap, max(2 * horizon, escape_horizon(raw,
                         [reference_time, initial_time] + observed_escapes, observed_switches)))
    result.update(edge_time_budget=horizon,
                  total_time_budget=horizon * cfg.asymptotic_cycles * len(cfg.cycle),
                  horizon_retries=len(history)-1,
                  timing_right_censored=int(result["reference_outcome"] == "timeout"),
                  reference_T_esc=float(reference_time), initial_T_esc=float(initial_time))
    return result, events, history

def run_phase_trial(task):
    patterns, successor, cfg, trial_seed, raw, reference_time, epsilon, pattern_set, kappa_index, kappa = task
    rows, event_rows, attempt_rows = [], [], []
    local = replace(cfg, reference_kappa=float(kappa))
    result, events, attempts = simulate_with_escape_horizon(
        patterns, successor, local, trial_seed, raw, reference_time)
    identifiers = {"epsilon": float(epsilon), "kappa": float(kappa),
                   "pattern_set": pattern_set, "trial_seed": trial_seed}
    event_rows.extend({**identifiers, **event} for event in events)
    attempt_rows.extend({**identifiers, **attempt} for attempt in attempts)
    complete = result["reference_outcome"] == "success"
    finite_tesc = np.isfinite(float(result["asymptotic_T_esc"]))
    correct_transition = bool(complete and finite_tesc)
    rows.append({
        "epsilon": float(epsilon),
        "kappa": float(kappa),
        "kappa_index": int(kappa_index),
        "pattern_set": int(pattern_set),
        "correct_transition": int(correct_transition),
        "outcome": str(result["reference_outcome"]),
        "completed_transitions": int(result["reference_completed_transitions"]),
        "T_esc": float(result["asymptotic_T_esc"]) if correct_transition else np.nan,
        "Z_at_escape": float(result["asymptotic_Z"]) if correct_transition else np.nan,
        "observed_escape_events": int(result["terminal_T_esc_events"]),
        "trial_seed": int(trial_seed),
        **{key: result[key] for key in ("edge_time_budget", "total_time_budget", "horizon_retries",
           "timing_right_censored", "reference_T_esc", "initial_T_esc", "elapsed_slow_time", "pending_T_esc")},
    })

    return rows[0], event_rows, attempt_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raw = load_config(args.config)
    cfg = resolved_base(raw)
    output = ensure_dir(args.output)

    theory = orthogonal_fixed_point(cfg.activation, float(raw["kappa_min"]))
    kcrit = float(theory["orthogonal_kappa_crit"])
    kappa_values = np.linspace(float(raw["kappa_min"]), float(raw["kappa_max"]), int(raw["kappa_points"]))
    if np.any(kappa_values <= kcrit):
        raise ValueError(
            f"Correlation phase-plane gains must satisfy kappa > orthogonal kappa_crit={kcrit:.6g}; "
            f"received minimum {kappa_values.min():.6g}."
        )

    successor = successor_array(cycle_adjacency(cfg.n_memories, cfg.cycle))
    epsilon_children = np.random.SeedSequence(cfg.seed).spawn(len(cfg.epsilons))
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    references = {float(k): orthogonal_fixed_point(cfg.activation, float(k))["orthogonal_T_esc_reference"]
                  for k in kappa_values}

    jobs = []
    for epsilon_index, (epsilon, epsilon_child) in enumerate(zip(cfg.epsilons, epsilon_children)):
        pattern_children = epsilon_child.spawn(cfg.pattern_sets)
        for pattern_set, child in enumerate(pattern_children):
            rng = np.random.default_rng(child)
            if float(epsilon) >= 1.0:
                patterns = np.ones((cfg.n_neurons, cfg.n_memories), dtype=np.float64)
            else:
                patterns = correlated_patterns(
                    rng, cfg.n_neurons, cfg.n_memories, float(epsilon), "biased"
                ).astype(np.float64)
            trial_seed = int(rng.integers(1, 2**31 - 1))
            for kappa_index, kappa in enumerate(kappa_values):
                jobs.append((patterns, successor, cfg, trial_seed, raw, references[float(kappa)],
                             epsilon, pattern_set, kappa_index, float(kappa)))

    for row, events, attempts in ordered_map(run_phase_trial, jobs, args.workers, "correlation_phase_plane"):
        rows.append(row)
        event_rows.extend(events)
        attempt_rows.extend(attempts)

    trials = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for index, ((epsilon, kappa), group) in enumerate(trials.groupby(["epsilon", "kappa"], sort=True)):
        success = group["correct_transition"].to_numpy(float)
        tesc = group.loc[group["correct_transition"].astype(bool), "T_esc"].to_numpy(float)
        lo, hi = bootstrap_mean(tesc, int(raw["bootstrap_reps"]), int(cfg.seed + index + 4401))
        summary_rows.append({
            "epsilon": float(epsilon),
            "kappa": float(kappa),
            "pattern_sets": int(len(group)),
            "correct_transition_count": int(success.sum()),
            "transition_probability": float(success.mean()),
            "timeout_fraction": float(group["timing_right_censored"].mean()),
            "min_edge_time_budget": float(group["edge_time_budget"].min()),
            "max_edge_time_budget": float(group["edge_time_budget"].max()),
            "max_total_time_budget": float(group["total_time_budget"].max()),
            "mean_T_esc": float(np.nanmean(tesc)) if len(tesc) else np.nan,
            "T_esc_ci_low": lo,
            "T_esc_ci_high": hi,
            "mean_Z_at_escape": float(np.nanmean(group.loc[group["correct_transition"].astype(bool), "Z_at_escape"])) if len(tesc) else np.nan,
        })
    summary = pd.DataFrame(summary_rows)

    write_csv(pd.DataFrame(event_rows, columns=list(dict.fromkeys(
        ["epsilon", "kappa", "pattern_set", "trial_seed", "event_index", "T_esc", "T_switch"] +
        [key for row in event_rows for key in row]))), output / "correlation_phase_plane_events.csv")
    write_csv(pd.DataFrame(attempt_rows), output / "correlation_phase_plane_attempts.csv")
    write_csv(trials, output / "correlation_phase_plane_trials.csv")
    write_csv(summary, output / "correlation_phase_plane_summary.csv")
    write_json(raw, output / "correlation_phase_plane_config.json")
    write_json(
        {
            "orthogonal_kappa_crit": kcrit,
            "activation": cfg.activation,
            "horizon_policy": "max(configured minimum, factor*max(initial and asymptotic orthogonal escape)+padding+hold); same-seed timeout retries incorporate observed escape and switch times",
            "timeout_interpretation": "Right-censored at the recorded horizon, not proof of dynamical impossibility.",
            "no_transition_definition": "No pattern realization completed the configured cycles with a finite T_esc.",
        },
        output / "correlation_phase_plane_theory.json",
    )


if __name__ == "__main__":
    main()
