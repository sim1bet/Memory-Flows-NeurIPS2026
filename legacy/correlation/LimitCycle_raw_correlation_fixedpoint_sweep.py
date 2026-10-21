#!/usr/bin/env python3
"""Raw correlated-pattern sweep for kappa_crit, epsilon_max, T_esc, and Z_infty.

No generalized pseudoinverse is used. Patterns satisfy
P(xi_i^mu=+1)=(1+epsilon)/2, hence E[xi^mu xi^nu]=epsilon**2.

Definitions
-----------
* kappa_crit: first gain yielding one complete graph-consistent cycle.
* epsilon_max: correlation where the monotone full-cycle success curve at the
  reference gain crosses a configurable probability threshold.
* T_esc: first re-armed crossing of the leading full fast-Jacobian eigenvalue.
* Z_infty: intended-successor saliency at T_esc, averaged over terminal cycles.

The orthogonal scalar map is used only for the reference values of T_esc and Z.
All correlated-pattern quantities are estimated from the raw finite-N system.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar


def _load_sibling(names: tuple[str, ...], module_name: str):
    for name in names:
        path = Path(__file__).with_name(name)
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Missing sibling module; tried {names}")


base = _load_sibling(
    ("LimitCycle_correlated_gpi_sweep.py", "LimitCycle_correlated_gpi_sweep(2).py"),
    "corr_base",
)
tanh_theory = _load_sibling(
    ("tesc_kappa_sweep.py", "tesc_kappa_sweep(1).py"),
    "tesc_theory",
)


@dataclass(frozen=True)
class Config:
    n_neurons: int = 256
    n_memories: int = 15
    cycle: tuple[int, ...] = (0, 1, 2, 3, 4)
    epsilons: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
    pattern_sets: int = 24
    activation: str = "tanh"
    reference_kappa: float = 15.0
    kappa_min: float = 3.0
    kappa_max: float = 60.0
    kappa_scan_points: int = 24
    kappa_bisection_steps: int = 8
    epsilon_max_success_probability: float = 0.50
    z_initial: float = 8.0
    tau_fast: float = 0.01
    tau_slow: float = 1.0
    dt_fast: float = 0.05
    max_slow_time_per_transition: float = 4.0
    check_interval: float = 0.01
    hold_time: float = 0.02
    min_alignment: float = 0.90
    min_margin: float = 0.10
    fast_noise_per_neuron: float = 1e-6
    asymptotic_cycles: int = 4
    terminal_cycles: int = 2
    plateau_relative_tolerance: float = 0.10
    seed: int = 20260803
    bootstrap_reps: int = 4000

    @property
    def dt_slow(self) -> float:
        return self.dt_fast * self.tau_fast / self.tau_slow


# ---------------------------------------------------------------------------
# Model and theory helpers
# ---------------------------------------------------------------------------

def activation(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(x)
    if name == "smoothf":
        return x / (1.0 + np.abs(x))
    raise ValueError(name)


def activation_derivative(x: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        y = np.tanh(x)
        return 1.0 - y * y
    if name == "smoothf":
        return 1.0 / (1.0 + np.abs(x)) ** 2
    raise ValueError(name)


def empirical_overlap(patterns: np.ndarray) -> tuple[float, float]:
    gram = patterns.T @ patterns / patterns.shape[0]
    mask = ~np.eye(gram.shape[0], dtype=bool)
    return float(np.mean(gram[mask])), float(np.mean(np.abs(gram[mask])))


def orthogonal_fixed_point(activation_name: str, kappa: float) -> dict[str, float]:
    """Return scalar-map critical point and attracting fixed point."""
    if activation_name == "tanh":
        critical = tanh_theory.theoretical_tanh_critical_point()
        stable = tanh_theory.theoretical_tanh_asymptotic_escape(kappa, critical)
        return {
            "orthogonal_kappa_crit": float(critical.kappa_crit),
            "orthogonal_T_esc_critical": float(critical.T_esc),
            "orthogonal_Z_critical": float(critical.Z_fixed),
            "orthogonal_T_esc_reference": float(stable["T_esc_theory_infty"]),
            "orthogonal_Z_reference": float(stable["Z_theory_infty"]),
        }

    if activation_name != "smoothf":
        raise ValueError(activation_name)

    def kappa_of_x(x: float) -> float:
        denominator = 2.0 * x**3 - 3.0 * x**2 + 1.0
        return np.inf if x <= 1.0 or denominator <= 0.0 else 2.0 * x**5 / denominator

    result = minimize_scalar(kappa_of_x, bounds=(1.0 + 1e-7, 20.0), method="bounded")
    xcrit = float(result.x)
    kcrit = float(kappa_of_x(xcrit))
    if kappa < kcrit:
        zref = tref = np.nan
    else:
        residual = lambda x: kappa_of_x(x) - kappa
        left = xcrit * (1.0 + 1e-8)
        right = max(2.0 * left, left + 1.0)
        while residual(right) < 0.0:
            right *= 1.5
        xref = float(brentq(residual, left, right))
        zref, tref = xref**2, math.log(xref)
    return {
        "orthogonal_kappa_crit": kcrit,
        "orthogonal_T_esc_critical": math.log(xcrit),
        "orthogonal_Z_critical": xcrit**2,
        "orthogonal_T_esc_reference": float(tref),
        "orthogonal_Z_reference": float(zref),
    }


def leading_fast_eigenvalue(
    patterns: np.ndarray,
    state: np.ndarray,
    saliency: np.ndarray,
    activation_name: str,
) -> float:
    """Largest real fast eigenvalue via the exact P x P nonzero spectrum."""
    derivative = activation_derivative(state, activation_name)
    weighted = patterns * derivative[:, None]
    reduced = np.diag(np.abs(saliency)) @ (patterns.T @ weighted / patterns.shape[0])
    return float(np.max(np.real(np.linalg.eigvals(reduced))) - 1.0)


# ---------------------------------------------------------------------------
# Critical-gain and long-run event simulation
# ---------------------------------------------------------------------------

def simulate_cycle_success(
    patterns: np.ndarray,
    successor: np.ndarray,
    cfg: Config,
    kappa: float,
    trial_seed: int,
) -> bool:
    gamma = base.branch_amplitude(cfg.z_initial, cfg.activation)
    activation_code = base.ACTIVATION_NAMES.index(cfg.activation)
    check_stride = max(1, int(round(cfg.check_interval / cfg.dt_slow)))
    check_dt = check_stride * cfg.dt_slow
    hold_checks = max(1, int(math.ceil(cfg.hold_time / check_dt)))
    coefficient_noise = cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)
    result = base._simulate_trial(
        patterns,
        np.eye(cfg.n_memories),
        successor,
        int(cfg.cycle[-1]),
        len(cfg.cycle),
        0,
        activation_code,
        gamma,
        cfg.z_initial,
        float(kappa),
        cfg.dt_fast,
        cfg.dt_slow,
        cfg.max_slow_time_per_transition,
        cfg.min_alignment,
        cfg.min_margin,
        hold_checks,
        hold_checks,
        hold_checks,
        check_stride,
        coefficient_noise,
        int(trial_seed),
    )
    return int(result[0]) == 0


def estimate_kappa_crit(
    patterns: np.ndarray,
    successor: np.ndarray,
    cfg: Config,
    trial_seed: int,
) -> tuple[float, bool, float, float]:
    grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.kappa_scan_points)
    outcomes = [simulate_cycle_success(patterns, successor, cfg, k, trial_seed) for k in grid]
    success_indices = np.flatnonzero(outcomes)
    if success_indices.size == 0:
        return np.nan, True, float(grid[-1]), np.nan
    first = int(success_indices[0])
    if first == 0:
        return float(grid[0]), False, np.nan, float(grid[0])
    lo, hi = float(grid[first - 1]), float(grid[first])
    for _ in range(cfg.kappa_bisection_steps):
        mid = 0.5 * (lo + hi)
        if simulate_cycle_success(patterns, successor, cfg, mid, trial_seed):
            hi = mid
        else:
            lo = mid
    return hi, False, lo, hi


def _interpolate_crossing(
    previous_value: float,
    current_value: float,
    previous_time: float,
    current_time: float,
) -> float:
    denominator = current_value - previous_value
    fraction = 0.0 if abs(denominator) < 1e-15 else -previous_value / denominator
    return previous_time + float(np.clip(fraction, 0.0, 1.0)) * (current_time - previous_time)


def simulate_asymptotic_events(
    patterns: np.ndarray,
    successor: np.ndarray,
    cfg: Config,
    trial_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run several cycles and measure edgewise T_esc and Z at stability loss."""
    rng = np.random.default_rng(trial_seed)
    p = cfg.n_memories
    cycle_length = len(cfg.cycle)
    required_transitions = cfg.asymptotic_cycles * cycle_length
    source = int(cfg.cycle[-1])
    gamma = base.branch_amplitude(cfg.z_initial, cfg.activation)

    coefficients = np.zeros(p)
    coefficients[source] = gamma
    coefficients += (cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)) * rng.standard_normal(p)
    saliency = np.zeros(p)
    saliency[source] = cfg.z_initial

    check_stride = max(1, int(round(cfg.check_interval / cfg.dt_slow)))
    check_dt = check_stride * cfg.dt_slow
    hold_checks = max(1, int(math.ceil(cfg.hold_time / check_dt)))
    max_steps = int(math.ceil(
        required_transitions * cfg.max_slow_time_per_transition / cfg.dt_slow
    )) + check_stride

    completed = 0
    elapsed_edge = 0.0
    edge_start_time = 0.0
    target_hold = wrong_hold = collapse_hold = 0
    events: list[dict[str, Any]] = []

    state = patterns @ coefficients
    previous_lambda = leading_fast_eigenvalue(patterns, state, saliency, cfg.activation)
    previous_check_time = 0.0
    previous_target_saliency = float(abs(saliency[int(successor[source])]))
    armed = previous_lambda < 0.0
    edge_tesc = np.nan
    edge_z_escape = np.nan

    outcome = "timeout"
    for step in range(max_steps):
        state = patterns @ coefficients
        raw = patterns.T @ activation(state, cfg.activation) / cfg.n_neurons
        drive = np.zeros(p)
        valid = np.flatnonzero(successor >= 0)
        np.add.at(drive, successor[valid], raw[valid])

        coefficients += cfg.dt_fast * (-coefficients + np.abs(saliency) * raw)
        coefficients += (
            cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)
        ) * rng.standard_normal(p)
        saliency += cfg.dt_slow * (-saliency + cfg.reference_kappa * drive)
        elapsed_edge += cfg.dt_slow

        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(saliency)):
            outcome = "nonfinite"
            break
        if (step + 1) % check_stride != 0:
            continue

        current_time = (step + 1) * cfg.dt_slow
        state = patterns @ coefficients
        state_norm = np.linalg.norm(state)
        if state_norm < 1e-12:
            collapse_hold += 1
            if collapse_hold >= hold_checks:
                outcome = "collapse"
                break
            continue
        collapse_hold = 0

        target = int(successor[source])
        current_lambda = leading_fast_eigenvalue(patterns, state, saliency, cfg.activation)
        current_target_saliency = float(abs(saliency[target]))

        # Re-arm only after the current branch has entered a stable interval.
        if not armed and current_lambda < 0.0:
            armed = True
            previous_lambda = current_lambda
            previous_check_time = current_time
            previous_target_saliency = current_target_saliency
        elif armed and not np.isfinite(edge_tesc) and previous_lambda < 0.0 <= current_lambda:
            crossing_time = _interpolate_crossing(
                previous_lambda, current_lambda, previous_check_time, current_time
            )
            edge_tesc = crossing_time - edge_start_time
            fraction = (crossing_time - previous_check_time) / max(
                current_time - previous_check_time, 1e-15
            )
            edge_z_escape = previous_target_saliency + fraction * (
                current_target_saliency - previous_target_saliency
            )

        scores = np.abs(patterns.T @ state) / (math.sqrt(cfg.n_neurons) * state_norm)
        order = np.argsort(scores)
        winner = int(order[-1])
        margin = float(scores[order[-1]] - scores[order[-2]])
        pure = bool(scores[winner] >= cfg.min_alignment and margin >= cfg.min_margin)

        target_hold = target_hold + 1 if pure and winner == target else 0
        wrong_hold = wrong_hold + 1 if pure and winner not in (source, target) else 0
        if wrong_hold >= hold_checks:
            outcome = "wrong_memory"
            break

        if target_hold >= hold_checks:
            switch_global_time = current_time - (hold_checks - 1) * check_dt
            switch_time = switch_global_time - edge_start_time
            events.append({
                "event_index": completed,
                "cycle_index": completed // cycle_length,
                "edge_index": completed % cycle_length,
                "source": source,
                "target": target,
                "T_esc": float(edge_tesc),
                "Z_at_escape": float(edge_z_escape),
                "T_switch": float(switch_time),
                "switch_delay": (
                    float(switch_time - edge_tesc) if np.isfinite(edge_tesc) else np.nan
                ),
                "source_saliency_at_switch": float(abs(saliency[source])),
                "target_saliency_at_switch": float(abs(saliency[target])),
            })
            completed += 1
            source = target
            edge_start_time = switch_global_time
            elapsed_edge = current_time - edge_start_time
            target_hold = wrong_hold = 0
            edge_tesc = edge_z_escape = np.nan
            armed = current_lambda < 0.0
            previous_lambda = current_lambda
            previous_check_time = current_time
            next_target = int(successor[source])
            previous_target_saliency = float(abs(saliency[next_target]))
            if completed >= required_transitions:
                outcome = "success"
                break
        elif elapsed_edge >= cfg.max_slow_time_per_transition:
            outcome = "timeout"
            break

        if armed:
            previous_lambda = current_lambda
            previous_check_time = current_time
            previous_target_saliency = current_target_saliency

    event_frame = pd.DataFrame(events)
    complete_cycle = completed >= cycle_length
    asymptotic_complete = completed >= required_transitions
    tail_events = cfg.terminal_cycles * cycle_length
    tail = event_frame.tail(tail_events) if len(event_frame) >= tail_events else pd.DataFrame()

    mean_tesc = float(tail["T_esc"].mean()) if not tail.empty else np.nan
    mean_z = float(tail["Z_at_escape"].mean()) if not tail.empty else np.nan
    tesc_observed = int(tail["T_esc"].notna().sum()) if not tail.empty else 0
    z_observed = int(tail["Z_at_escape"].notna().sum()) if not tail.empty else 0

    plateau_ok = False
    relative_cycle_change_tesc = relative_cycle_change_z = np.nan
    if asymptotic_complete and cfg.terminal_cycles >= 2:
        cycle_means = event_frame.groupby("cycle_index")[["T_esc", "Z_at_escape"]].mean()
        last = cycle_means.tail(cfg.terminal_cycles)
        if len(last) >= 2:
            tvals = last["T_esc"].to_numpy(float)
            zvals = last["Z_at_escape"].to_numpy(float)
            if np.all(np.isfinite(tvals[-2:])) and np.all(np.isfinite(zvals[-2:])):
                relative_cycle_change_tesc = abs(tvals[-1] - tvals[-2]) / max(abs(np.mean(tvals[-2:])), 1e-12)
                relative_cycle_change_z = abs(zvals[-1] - zvals[-2]) / max(abs(np.mean(zvals[-2:])), 1e-12)
                plateau_ok = bool(
                    relative_cycle_change_tesc <= cfg.plateau_relative_tolerance
                    and relative_cycle_change_z <= cfg.plateau_relative_tolerance
                )

    summary = {
        "reference_outcome": outcome,
        "reference_completed_transitions": completed,
        "reference_complete_cycle": int(complete_cycle),
        "reference_asymptotic_complete": int(asymptotic_complete),
        "asymptotic_plateau_ok": int(plateau_ok),
        "asymptotic_T_esc": mean_tesc,
        "asymptotic_Z": mean_z,
        "terminal_T_esc_events": tesc_observed,
        "terminal_Z_events": z_observed,
        "relative_cycle_change_T_esc": float(relative_cycle_change_tesc),
        "relative_cycle_change_Z": float(relative_cycle_change_z),
    }
    return summary, events


# ---------------------------------------------------------------------------
# Aggregation and epsilon_max
# ---------------------------------------------------------------------------

def bootstrap_interval(values: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = np.mean(values[indices], axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def isotonic_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pooled-adjacent-violators fit constrained to be decreasing."""
    y = -np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    blocks: list[dict[str, Any]] = []
    for index, (value, weight) in enumerate(zip(y, w)):
        blocks.append({"start": index, "end": index, "weight": weight, "mean": value})
        while len(blocks) >= 2 and blocks[-2]["mean"] > blocks[-1]["mean"]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left["weight"] + right["weight"]
            blocks.append({
                "start": left["start"],
                "end": right["end"],
                "weight": weight_sum,
                "mean": (
                    left["mean"] * left["weight"] + right["mean"] * right["weight"]
                ) / weight_sum,
            })
    fitted = np.empty_like(y)
    for block in blocks:
        fitted[block["start"]:block["end"] + 1] = block["mean"]
    return -fitted


def estimate_epsilon_max(
    epsilons: np.ndarray,
    fitted_success: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    above = np.flatnonzero(fitted_success >= threshold)
    if above.size == 0:
        return np.nan, np.nan
    last = int(above[-1])
    grid_value = float(epsilons[last])
    if last == len(epsilons) - 1:
        return grid_value, grid_value
    x0, x1 = float(epsilons[last]), float(epsilons[last + 1])
    y0, y1 = float(fitted_success[last]), float(fitted_success[last + 1])
    if abs(y1 - y0) < 1e-15:
        return grid_value, grid_value
    interpolated = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
    return grid_value, float(np.clip(interpolated, x0, x1))


def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    successor = base.make_successor_array(cfg.n_memories, cfg.cycle)
    theory = orthogonal_fixed_point(cfg.activation, cfg.reference_kappa)
    children = np.random.SeedSequence(cfg.seed).spawn(len(cfg.epsilons))
    trial_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for epsilon_index, (epsilon, child) in enumerate(zip(cfg.epsilons, children)):
        for pattern_set, pchild in enumerate(child.spawn(cfg.pattern_sets)):
            rng = np.random.default_rng(pchild)
            if float(epsilon) >= 1.0:
                patterns = np.ones((cfg.n_neurons, cfg.n_memories), dtype=np.float64)
            else:
                patterns = base.generate_correlated_patterns(
                    rng, cfg.n_neurons, cfg.n_memories, float(epsilon), "biased"
                ).astype(np.float64)
            signed_overlap, abs_overlap = empirical_overlap(patterns)
            rank = int(np.linalg.matrix_rank(patterns))
            seed = int(rng.integers(1, 2**31 - 1))

            kcrit, censored, lower, upper = estimate_kappa_crit(
                patterns, successor, cfg, seed
            )
            reference, events = simulate_asymptotic_events(
                patterns, successor, cfg, seed + 17
            )
            trial_rows.append({
                "epsilon": float(epsilon),
                "expected_pairwise_overlap": float(epsilon**2),
                "epsilon_index": epsilon_index,
                "pattern_set": pattern_set,
                "N": cfg.n_neurons,
                "P": cfg.n_memories,
                "rank": rank,
                "empirical_pairwise_overlap": signed_overlap,
                "empirical_abs_pairwise_overlap": abs_overlap,
                "kappa_crit": kcrit,
                "kappa_crit_right_censored": int(censored),
                "kappa_failure_lower": lower,
                "kappa_success_upper": upper,
                "reference_kappa": cfg.reference_kappa,
                "seed": seed,
                **reference,
            })
            for event in events:
                event_rows.append({
                    "epsilon": float(epsilon),
                    "pattern_set": pattern_set,
                    "seed": seed + 17,
                    **event,
                })

    trials = pd.DataFrame(trial_rows)
    events = pd.DataFrame(event_rows)
    summary_rows: list[dict[str, Any]] = []
    for epsilon, group in trials.groupby("epsilon", sort=True):
        k = group["kappa_crit"].to_numpy(float)
        t = group.loc[group["asymptotic_plateau_ok"] == 1, "asymptotic_T_esc"].to_numpy(float)
        z = group.loc[group["asymptotic_plateau_ok"] == 1, "asymptotic_Z"].to_numpy(float)
        success = group["reference_complete_cycle"].to_numpy(float)
        klo, khi = bootstrap_interval(k, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 1)
        tlo, thi = bootstrap_interval(t, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 2)
        zlo, zhi = bootstrap_interval(z, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 3)
        slo, shi = bootstrap_interval(success, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 4)
        summary_rows.append({
            "epsilon": float(epsilon),
            "expected_pairwise_overlap": float(epsilon**2),
            "pattern_sets": len(group),
            "mean_empirical_pairwise_overlap": float(group["empirical_pairwise_overlap"].mean()),
            "mean_kappa_crit": float(np.nanmean(k)) if np.any(np.isfinite(k)) else np.nan,
            "kappa_crit_ci_low": klo,
            "kappa_crit_ci_high": khi,
            "kappa_crit_observed_fraction": float(np.mean(np.isfinite(k))),
            "kappa_crit_censored_fraction": float(group["kappa_crit_right_censored"].mean()),
            "reference_cycle_success_fraction": float(np.mean(success)),
            "reference_cycle_success_ci_low": slo,
            "reference_cycle_success_ci_high": shi,
            "reference_asymptotic_success_fraction": float(group["reference_asymptotic_complete"].mean()),
            "plateau_fraction": float(group["asymptotic_plateau_ok"].mean()),
            "mean_asymptotic_T_esc": float(np.nanmean(t)) if np.any(np.isfinite(t)) else np.nan,
            "asymptotic_T_esc_ci_low": tlo,
            "asymptotic_T_esc_ci_high": thi,
            "asymptotic_T_esc_observed_fraction": float(np.sum(np.isfinite(t)) / len(group)),
            "mean_asymptotic_Z": float(np.nanmean(z)) if np.any(np.isfinite(z)) else np.nan,
            "asymptotic_Z_ci_low": zlo,
            "asymptotic_Z_ci_high": zhi,
            "asymptotic_Z_observed_fraction": float(np.sum(np.isfinite(z)) / len(group)),
        })

    summary = pd.DataFrame(summary_rows).sort_values("epsilon").reset_index(drop=True)
    success_fit = isotonic_decreasing(
        summary["plateau_fraction"].to_numpy(float),
        summary["pattern_sets"].to_numpy(float),
    )
    summary["isotonic_sustained_success_fraction"] = success_fit
    eps_grid, eps_interp = estimate_epsilon_max(
        summary["epsilon"].to_numpy(float),
        success_fit,
        cfg.epsilon_max_success_probability,
    )
    summary["within_epsilon_max"] = (
        summary["epsilon"] <= eps_interp + 1e-12
    ).astype(int) if np.isfinite(eps_interp) else 0

    base_row = summary.iloc[0]
    summary["delta_kappa_crit"] = summary["mean_kappa_crit"] - base_row["mean_kappa_crit"]
    summary["delta_asymptotic_T_esc_from_epsilon0"] = (
        summary["mean_asymptotic_T_esc"] - base_row["mean_asymptotic_T_esc"]
    )
    summary["delta_asymptotic_Z_from_orthogonal"] = (
        summary["mean_asymptotic_Z"] - theory["orthogonal_Z_reference"]
    )
    summary["orthogonal_T_esc_reference"] = theory["orthogonal_T_esc_reference"]
    summary["orthogonal_Z_reference"] = theory["orthogonal_Z_reference"]
    summary["epsilon_max_grid"] = eps_grid
    summary["epsilon_max_interpolated"] = eps_interp
    summary["epsilon_max_success_probability"] = cfg.epsilon_max_success_probability
    summary["epsilon_max_definition"] = "plateau-qualified sustained cycles at reference_kappa"

    theory.update({
        "epsilon_max_grid": float(eps_grid),
        "epsilon_max_interpolated": float(eps_interp),
        "epsilon_max_success_probability": float(cfg.epsilon_max_success_probability),
        "epsilon_max_definition": "plateau-qualified sustained cycles at reference_kappa",
    })
    return trials, events, summary, theory


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(summary: pd.DataFrame, theory: dict[str, float], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), sharex=True)
    x = summary["epsilon"].to_numpy(float)
    epsmax = float(theory["epsilon_max_interpolated"])
    threshold = float(theory["epsilon_max_success_probability"])

    ax = axes[0, 0]
    y = summary["plateau_fraction"].to_numpy(float)
    ax.plot(x, y, "o", label="Plateau-qualified sustained probability")
    ax.plot(x, summary["isotonic_sustained_success_fraction"], linewidth=2, label="Monotone fit")
    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold = {threshold:g}")
    if np.isfinite(epsmax):
        ax.axvline(epsmax, linestyle=":", linewidth=2, label=rf"$\epsilon_{{\max}}={epsmax:.3f}$")
    ax.set_ylabel("Sustained-transition probability")
    ax.set_title(r"Operational correlation limit at fixed $\kappa$")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    y = summary["mean_kappa_crit"].to_numpy(float)
    low = summary["kappa_crit_ci_low"].to_numpy(float)
    high = summary["kappa_crit_ci_high"].to_numpy(float)
    ax.plot(x, y, marker="o", linewidth=2)
    ax.fill_between(x, low, high, alpha=0.2)
    ax.axhline(float(summary["mean_kappa_crit"].iloc[0]), linestyle="--", linewidth=1,
               label=r"Empirical $\epsilon=0$ baseline")
    ax.axhline(float(theory["orthogonal_kappa_crit"]), linestyle=":", linewidth=1.5,
               label="Orthogonal scalar-map value")
    if np.isfinite(epsmax):
        ax.axvline(epsmax, linestyle=":", linewidth=2)
    censored = summary["kappa_crit_censored_fraction"].to_numpy(float) > 0
    ax.scatter(x[censored], y[censored], facecolors="none", edgecolors="black", s=70)
    ax.set_ylabel(r"$\kappa_{\rm crit}$")
    ax.set_title("Raw-pattern critical gain")
    ax.legend(frameon=False, fontsize=9)

    mask = summary["within_epsilon_max"].to_numpy(bool)
    ax = axes[1, 0]
    tx, ty = x[mask], summary.loc[mask, "mean_asymptotic_T_esc"].to_numpy(float)
    tlow = summary.loc[mask, "asymptotic_T_esc_ci_low"].to_numpy(float)
    thigh = summary.loc[mask, "asymptotic_T_esc_ci_high"].to_numpy(float)
    ax.plot(tx, ty, marker="o", linewidth=2, label="Raw correlated patterns")
    ax.fill_between(tx, tlow, thigh, alpha=0.2)
    ax.axhline(float(theory["orthogonal_T_esc_reference"]), linestyle="--", linewidth=1.5,
               label="Orthogonal fixed-point map")
    if np.isfinite(epsmax):
        ax.axvline(epsmax, linestyle=":", linewidth=2)
    ax.set_ylabel(r"Asymptotic $T_{\rm esc}$")
    ax.set_title(r"Escape time on $0\leq\epsilon\leq\epsilon_{\max}$")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    zy = summary.loc[mask, "mean_asymptotic_Z"].to_numpy(float)
    zlow = summary.loc[mask, "asymptotic_Z_ci_low"].to_numpy(float)
    zhigh = summary.loc[mask, "asymptotic_Z_ci_high"].to_numpy(float)
    ax.plot(tx, zy, marker="o", linewidth=2, label="Raw correlated patterns")
    ax.fill_between(tx, zlow, zhigh, alpha=0.2)
    ax.axhline(float(theory["orthogonal_Z_reference"]), linestyle="--", linewidth=1.5,
               label=rf"Orthogonal $\bar Z={theory['orthogonal_Z_reference']:.3f}$")
    if np.isfinite(epsmax):
        ax.axvline(epsmax, linestyle=":", linewidth=2)
    ax.set_ylabel(r"Asymptotic $\bar Z$")
    ax.set_title("Terminal saliency-map amplitude")
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.ravel():
        ax.set_xlabel(r"Correlation parameter $\epsilon$")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("raw_correlation_fixedpoint"))
    p.add_argument("--pattern-sets", type=int, default=24)
    p.add_argument("--epsilon-points", type=int, default=21)
    p.add_argument("--reference-kappa", type=float, default=15.0)
    p.add_argument("--kappa-max", type=float, default=60.0)
    p.add_argument("--kappa-scan-points", type=int, default=24)
    p.add_argument("--kappa-bisection-steps", type=int, default=8)
    p.add_argument("--epsilon-max-success-probability", type=float, default=0.50)
    p.add_argument("--asymptotic-cycles", type=int, default=4)
    p.add_argument("--terminal-cycles", type=int, default=2)
    p.add_argument("--dt-fast", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        epsilons=tuple(np.linspace(0.0, 1.0, args.epsilon_points)),
        pattern_sets=args.pattern_sets,
        reference_kappa=args.reference_kappa,
        kappa_max=args.kappa_max,
        kappa_scan_points=args.kappa_scan_points,
        kappa_bisection_steps=args.kappa_bisection_steps,
        epsilon_max_success_probability=args.epsilon_max_success_probability,
        asymptotic_cycles=args.asymptotic_cycles,
        terminal_cycles=args.terminal_cycles,
        dt_fast=args.dt_fast,
    )
    if args.smoke:
        cfg = Config(
            epsilons=(0.0, 0.3, 0.5, 0.7, 1.0),
            pattern_sets=2,
            reference_kappa=args.reference_kappa,
            kappa_max=min(args.kappa_max, 40.0),
            kappa_scan_points=8,
            kappa_bisection_steps=3,
            epsilon_max_success_probability=args.epsilon_max_success_probability,
            asymptotic_cycles=3,
            terminal_cycles=2,
            dt_fast=0.1,
            check_interval=0.02,
            bootstrap_reps=300,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    trials, events, summary, theory = run(cfg)
    trials.to_csv(args.output / "raw_correlation_fixedpoint_trials.csv", index=False)
    events.to_csv(args.output / "raw_correlation_fixedpoint_events.csv", index=False)
    summary.to_csv(args.output / "raw_correlation_fixedpoint_summary.csv", index=False)
    with open(args.output / "raw_correlation_fixedpoint_config.json", "w") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    with open(args.output / "raw_correlation_fixedpoint_theory.json", "w") as handle:
        json.dump(theory, handle, indent=2)
    plot(summary, theory, args.output / "raw_correlation_kcrit_Tesc_Z")

    print(json.dumps(theory, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
