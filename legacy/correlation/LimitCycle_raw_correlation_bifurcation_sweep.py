#!/usr/bin/env python3
"""Raw correlated-pattern sweep for empirical kappa_crit and T_esc.

No generalized pseudoinverse is used. Patterns follow
P(xi_i^mu=+1)=(1+epsilon)/2, so expected pairwise overlap is epsilon^2.

Definitions
-----------
* kappa_crit(epsilon): smallest gain yielding one complete graph-consistent
  cycle, estimated per pattern set by a coarse scan followed by bisection.
* T_esc(epsilon): first zero crossing of the leading fast-subsystem Jacobian
  eigenvalue during the first transition at a fixed reference gain.

Both are finite-N empirical quantities. The orthogonal scalar return-map
formula is not applied to correlated raw overlaps.
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

BASE = Path(__file__).with_name("LimitCycle_correlated_gpi_sweep.py")
if not BASE.exists():
    BASE = Path(__file__).with_name("LimitCycle_correlated_gpi_sweep(2).py")
spec = importlib.util.spec_from_file_location("corr_base", BASE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
assert spec.loader is not None
spec.loader.exec_module(base)


@dataclass(frozen=True)
class Config:
    n_neurons: int = 256
    n_memories: int = 15
    cycle: tuple[int, ...] = (0, 1, 2, 3, 4)
    epsilons: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 11))
    pattern_sets: int = 12
    activation: str = "tanh"
    reference_kappa: float = 15.0
    kappa_min: float = 3.0
    kappa_max: float = 40.0
    kappa_scan_points: int = 16
    kappa_bisection_steps: int = 7
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
    seed: int = 20260803
    bootstrap_reps: int = 4000

    @property
    def dt_slow(self) -> float:
        return self.dt_fast * self.tau_fast / self.tau_slow


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


def simulate_cycle_success(
    patterns: np.ndarray,
    successor: np.ndarray,
    cfg: Config,
    kappa: float,
    trial_seed: int,
) -> bool:
    """Use the existing raw-rule simulator for one complete cycle."""
    gamma = base.branch_amplitude(cfg.z_initial, cfg.activation)
    activation_code = base.ACTIVATION_NAMES.index(cfg.activation)
    check_stride = max(1, int(round(cfg.check_interval / cfg.dt_slow)))
    check_dt = check_stride * cfg.dt_slow
    hold_checks = max(1, int(math.ceil(cfg.hold_time / check_dt)))
    wrong_hold_checks = hold_checks
    collapse_hold_checks = hold_checks
    coefficient_noise = cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)
    dummy_pinv = np.eye(cfg.n_memories)
    result = base._simulate_trial(
        patterns,
        dummy_pinv,
        successor,
        int(cfg.cycle[-1]),
        len(cfg.cycle),
        0,  # raw rule
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
        wrong_hold_checks,
        collapse_hold_checks,
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
    """Return threshold, right-censor flag, lower failure, upper success."""
    grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.kappa_scan_points)
    outcomes = [simulate_cycle_success(patterns, successor, cfg, k, trial_seed) for k in grid]
    success_indices = np.flatnonzero(outcomes)
    if success_indices.size == 0:
        return np.nan, True, float(grid[-1]), np.nan
    first = int(success_indices[0])
    if first == 0:
        return float(grid[0]), False, np.nan, float(grid[0])
    lo = float(grid[first - 1])
    hi = float(grid[first])
    # Same perturbation seed at each gain isolates the gain effect.
    for _ in range(cfg.kappa_bisection_steps):
        mid = 0.5 * (lo + hi)
        if simulate_cycle_success(patterns, successor, cfg, mid, trial_seed):
            hi = mid
        else:
            lo = mid
    return hi, False, lo, hi


def leading_fast_eigenvalue(
    patterns: np.ndarray,
    state: np.ndarray,
    saliency: np.ndarray,
    activation_name: str,
) -> float:
    """Largest real fast eigenvalue using the exact P x P reduction."""
    derivative = activation_derivative(state, activation_name)
    weighted = patterns * derivative[:, None]
    reduced = np.diag(np.abs(saliency)) @ (patterns.T @ weighted / patterns.shape[0])
    eig = np.linalg.eigvals(reduced)
    return float(np.max(np.real(eig)) - 1.0)


def estimate_first_escape(
    patterns: np.ndarray,
    successor: np.ndarray,
    cfg: Config,
    trial_seed: int,
) -> dict[str, float | int | str]:
    """Measure the first full-Jacobian stability crossing at reference kappa."""
    rng = np.random.default_rng(trial_seed)
    P = cfg.n_memories
    source = int(cfg.cycle[-1])
    target = int(successor[source])
    gamma = base.branch_amplitude(cfg.z_initial, cfg.activation)
    coefficients = np.zeros(P)
    coefficients[source] = gamma
    coefficients += (cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)) * rng.standard_normal(P)
    saliency = np.zeros(P)
    saliency[source] = cfg.z_initial
    check_stride = max(1, int(round(cfg.check_interval / cfg.dt_slow)))
    max_steps = int(math.ceil(cfg.max_slow_time_per_transition / cfg.dt_slow))

    previous_time = 0.0
    state = patterns @ coefficients
    previous_lambda = leading_fast_eigenvalue(patterns, state, saliency, cfg.activation)
    t_escape = np.nan
    t_switch = np.nan
    switch_hold = 0
    hold_checks = max(1, int(math.ceil(cfg.hold_time / (check_stride * cfg.dt_slow))))

    for step in range(max_steps):
        state = patterns @ coefficients
        act = activation(state, cfg.activation)
        raw = patterns.T @ act / cfg.n_neurons
        drive = np.zeros(P)
        valid = np.flatnonzero(successor >= 0)
        drive[successor[valid]] += raw[valid]
        coefficients += cfg.dt_fast * (-coefficients + np.abs(saliency) * raw)
        coefficients += (cfg.fast_noise_per_neuron / math.sqrt(cfg.n_neurons)) * rng.standard_normal(P)
        saliency += cfg.dt_slow * (-saliency + cfg.reference_kappa * drive)

        if (step + 1) % check_stride != 0:
            continue
        current_time = (step + 1) * cfg.dt_slow
        state = patterns @ coefficients
        current_lambda = leading_fast_eigenvalue(patterns, state, saliency, cfg.activation)
        if np.isnan(t_escape) and previous_lambda < 0.0 <= current_lambda:
            fraction = -previous_lambda / max(current_lambda - previous_lambda, 1e-15)
            t_escape = previous_time + fraction * (current_time - previous_time)

        denominator = math.sqrt(cfg.n_neurons) * np.linalg.norm(state)
        if denominator > 0:
            scores = np.abs(patterns.T @ state) / denominator
            order = np.argsort(scores)
            winner = int(order[-1])
            margin = float(scores[order[-1]] - scores[order[-2]])
            pure_target = winner == target and scores[winner] >= cfg.min_alignment and margin >= cfg.min_margin
        else:
            pure_target = False
        switch_hold = switch_hold + 1 if pure_target else 0
        if np.isnan(t_switch) and switch_hold >= hold_checks:
            t_switch = current_time - (hold_checks - 1) * check_stride * cfg.dt_slow

        if np.isfinite(t_escape) and np.isfinite(t_switch):
            break
        previous_lambda = current_lambda
        previous_time = current_time

    outcome = "complete" if np.isfinite(t_escape) else "no_stability_crossing"
    return {
        "T_esc": float(t_escape),
        "T_switch": float(t_switch),
        "switch_delay": float(t_switch - t_escape) if np.isfinite(t_escape) and np.isfinite(t_switch) else np.nan,
        "escape_outcome": outcome,
    }


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


def run(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    successor = base.make_successor_array(cfg.n_memories, cfg.cycle)
    children = np.random.SeedSequence(cfg.seed).spawn(len(cfg.epsilons))
    rows: list[dict[str, Any]] = []
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
            trial_seed = int(rng.integers(1, 2**31 - 1))
            kcrit, censored, lower, upper = estimate_kappa_crit(patterns, successor, cfg, trial_seed)
            escape = estimate_first_escape(patterns, successor, cfg, trial_seed + 17)
            rows.append({
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
                "seed": trial_seed,
                **escape,
            })
    trials = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    for epsilon, group in trials.groupby("epsilon", sort=True):
        k = group["kappa_crit"].to_numpy(float)
        t = group["T_esc"].to_numpy(float)
        klo, khi = bootstrap_interval(k, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 1)
        tlo, thi = bootstrap_interval(t, cfg.bootstrap_reps, cfg.seed + int(1000 * epsilon) + 2)
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
            "mean_T_esc": float(np.nanmean(t)) if np.any(np.isfinite(t)) else np.nan,
            "T_esc_ci_low": tlo,
            "T_esc_ci_high": thi,
            "T_esc_observed_fraction": float(np.mean(np.isfinite(t))),
        })
    summary = pd.DataFrame(summary_rows).sort_values("epsilon")
    base_row = summary.iloc[0]
    summary["delta_kappa_crit"] = summary["mean_kappa_crit"] - base_row["mean_kappa_crit"]
    summary["relative_kappa_crit_change"] = summary["mean_kappa_crit"] / base_row["mean_kappa_crit"] - 1.0
    summary["delta_T_esc"] = summary["mean_T_esc"] - base_row["mean_T_esc"]
    summary["relative_T_esc_change"] = summary["mean_T_esc"] / base_row["mean_T_esc"] - 1.0
    return trials, summary


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    x = summary["epsilon"].to_numpy()

    y = summary["delta_kappa_crit"].to_numpy()
    low = summary["kappa_crit_ci_low"].to_numpy() - summary["mean_kappa_crit"].iloc[0]
    high = summary["kappa_crit_ci_high"].to_numpy() - summary["mean_kappa_crit"].iloc[0]
    axes[0].plot(x, y, marker="o", linewidth=2)
    axes[0].fill_between(x, low, high, alpha=0.2)
    censored = summary["kappa_crit_censored_fraction"].to_numpy() > 0
    axes[0].scatter(x[censored], y[censored], facecolors="none", edgecolors="black", s=75, label="partly censored")
    axes[0].axhline(0.0, linewidth=1, linestyle="--")
    axes[0].set_ylabel(r"$\kappa_{\rm crit}(\epsilon)-\kappa_{\rm crit}(0)$")
    axes[0].set_title("Raw-rule critical-gain shift")
    if np.any(censored):
        axes[0].legend(frameon=False)

    y = summary["delta_T_esc"].to_numpy()
    low = summary["T_esc_ci_low"].to_numpy() - summary["mean_T_esc"].iloc[0]
    high = summary["T_esc_ci_high"].to_numpy() - summary["mean_T_esc"].iloc[0]
    axes[1].plot(x, y, marker="o", linewidth=2)
    axes[1].fill_between(x, low, high, alpha=0.2)
    missing = summary["T_esc_observed_fraction"].to_numpy() < 1.0
    axes[1].scatter(x[missing], y[missing], facecolors="none", edgecolors="black", s=75, label="missing crossings")
    axes[1].axhline(0.0, linewidth=1, linestyle="--")
    axes[1].set_ylabel(r"$T_{\rm esc}(\epsilon)-T_{\rm esc}(0)$")
    axes[1].set_title(r"Escape-time shift at fixed $\kappa$")
    if np.any(missing):
        axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xlabel(r"Correlation parameter $\epsilon$")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("raw_correlation_bifurcation"))
    p.add_argument("--pattern-sets", type=int, default=12)
    p.add_argument("--epsilon-points", type=int, default=11)
    p.add_argument("--reference-kappa", type=float, default=15.0)
    p.add_argument("--kappa-max", type=float, default=40.0)
    p.add_argument("--kappa-scan-points", type=int, default=16)
    p.add_argument("--kappa-bisection-steps", type=int, default=7)
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
        dt_fast=args.dt_fast,
    )
    if args.smoke:
        cfg = Config(
            epsilons=(0.0, 0.4, 0.8, 1.0), pattern_sets=2,
            reference_kappa=args.reference_kappa, kappa_max=args.kappa_max,
            kappa_scan_points=8, kappa_bisection_steps=3,
            dt_fast=0.1, bootstrap_reps=300,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    trials, summary = run(cfg)
    trials.to_csv(args.output / "raw_correlation_bifurcation_trials.csv", index=False)
    summary.to_csv(args.output / "raw_correlation_bifurcation_summary.csv", index=False)
    with open(args.output / "raw_correlation_bifurcation_config.json", "w") as handle:
        json.dump(asdict(cfg), handle, indent=2)
    plot(summary, args.output / "raw_correlation_kcrit_Tesc_shift")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
