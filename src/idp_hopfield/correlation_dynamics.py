"""Raw correlated-pattern bifurcation and fixed-point analysis.

Canonical implementation of the adaptive-supercritical protocol.  For each
correlation and pattern realization it first estimates kappa_crit, then
evaluates sustained cycles at a matched gain kappa_eval=r*kappa_crit with r>1.
No pseudoinverse is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .parallel import ordered_map
from .activations import branch_amplitude
from .graphs import cycle_adjacency, successor_array
from .patterns import correlated_patterns
from .theory import asymptotic_escape as theory_asymptotic_escape
from .theory import critical_point as theory_critical_point

@dataclass(frozen=True)
class Config:
    n_neurons: int = 256
    n_memories: int = 15
    cycle: tuple[int, ...] = (0, 1, 2, 3, 4)
    epsilons: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
    pattern_sets: int = 24
    activation: str = "tanh"
    reference_kappa: float = 15.0
    supercritical_factor: float | None = None
    evaluation_kappa_max: float = 150.0
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
    """Return orthogonal scalar-map critical point and stable fixed point."""
    critical = theory_critical_point(activation_name)
    stable = theory_asymptotic_escape(kappa, activation_name, critical)
    return {
        "orthogonal_kappa_crit": float(critical.kappa),
        "orthogonal_T_esc_critical": float(critical.escape_time),
        "orthogonal_Z_critical": float(critical.fixed_amplitude),
        "orthogonal_T_esc_reference": float(stable["T_esc_theory_infty"]),
        "orthogonal_Z_reference": float(stable["Z_theory_infty"]),
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
    """Return whether one complete directed cycle is retrieved at ``kappa``.

    The same raw-overlap integrator used for the asymptotic event measurement
    is used here, avoiding a second numerical implementation of the dynamics.
    """
    one_cycle_cfg = replace(
        cfg,
        reference_kappa=float(kappa),
        asymptotic_cycles=1,
        terminal_cycles=1,
    )
    result, _ = simulate_asymptotic_events(
        patterns, successor, one_cycle_cfg, int(trial_seed)
    )
    return result["reference_outcome"] == "success"

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
    gamma = branch_amplitude(cfg.z_initial, cfg.activation)

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
        "pending_T_esc": float(edge_tesc),
        "pending_edge_elapsed": float(elapsed_edge),
        "elapsed_slow_time": float((step + 1) * cfg.dt_slow),
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


def prepare_realization(task):
    cfg, epsilon_index, epsilon, pattern_set, pchild = task
    successor = successor_array(cycle_adjacency(cfg.n_memories, cfg.cycle))
    prepared = []
    rng = np.random.default_rng(pchild)
    if float(epsilon) >= 1.0:
        patterns = np.ones((cfg.n_neurons, cfg.n_memories), dtype=np.float64)
    else:
        patterns = correlated_patterns(
            rng, cfg.n_neurons, cfg.n_memories, float(epsilon), "biased"
        ).astype(np.float64)
    signed_overlap, abs_overlap = empirical_overlap(patterns)
    rank = int(np.linalg.matrix_rank(patterns))
    seed = int(rng.integers(1, 2**31 - 1))
    kcrit, censored, lower, upper = estimate_kappa_crit(
        patterns, successor, cfg, seed
    )
    prepared.append({
        "epsilon": float(epsilon),
        "expected_pairwise_overlap": float(epsilon**2),
        "epsilon_index": epsilon_index,
        "pattern_set": pattern_set,
        "patterns": patterns,
        "rank": rank,
        "empirical_pairwise_overlap": signed_overlap,
        "empirical_abs_pairwise_overlap": abs_overlap,
        "kappa_crit": kcrit,
        "kappa_crit_right_censored": int(censored),
        "kappa_failure_lower": lower,
        "kappa_success_upper": upper,
        "seed": seed,
    })

    return prepared[0]


def evaluate_realization(task):
    cfg, row, factor, baseline_mean_kcrit = task
    successor = successor_array(cycle_adjacency(cfg.n_memories, cfg.cycle))
    trial_rows, event_rows = [], []
    kcrit = float(row["kappa_crit"])
    evaluation_kappa = factor * kcrit if np.isfinite(kcrit) else np.nan
    gain_censored = bool(
        not np.isfinite(evaluation_kappa)
        or evaluation_kappa > cfg.evaluation_kappa_max
    )

    if gain_censored:
        evaluation = {
            "evaluation_outcome": "gain_unavailable",
            "evaluation_completed_transitions": 0,
            "evaluation_complete_cycle": 0,
            "evaluation_asymptotic_complete": 0,
            "asymptotic_plateau_ok": 0,
            "asymptotic_T_esc": np.nan,
            "asymptotic_Z": np.nan,
            "terminal_T_esc_events": 0,
            "terminal_Z_events": 0,
            "relative_cycle_change_T_esc": np.nan,
            "relative_cycle_change_Z": np.nan,
        }
        orthogonal = {
            "orthogonal_T_esc_matched": np.nan,
            "orthogonal_Z_matched": np.nan,
        }
        events: list[dict[str, Any]] = []
    else:
        evaluation_cfg = replace(cfg, reference_kappa=float(evaluation_kappa))
        raw_evaluation, events = simulate_asymptotic_events(
            row["patterns"], successor, evaluation_cfg, int(row["seed"]) + 17
        )
        evaluation = {
            key.replace("reference_", "evaluation_", 1): value
            for key, value in raw_evaluation.items()
        }
        matched = orthogonal_fixed_point(cfg.activation, float(evaluation_kappa))
        orthogonal = {
            "orthogonal_T_esc_matched": matched["orthogonal_T_esc_reference"],
            "orthogonal_Z_matched": matched["orthogonal_Z_reference"],
        }

    trial_rows.append({
        key: value for key, value in row.items() if key != "patterns"
    } | {
        "N": cfg.n_neurons,
        "P": cfg.n_memories,
        "baseline_reference_kappa": cfg.reference_kappa,
        "baseline_mean_kappa_crit": baseline_mean_kcrit,
        "supercritical_factor": factor,
        "evaluation_kappa": float(evaluation_kappa),
        "evaluation_kappa_max": cfg.evaluation_kappa_max,
        "evaluation_gain_censored": int(gain_censored),
        "evaluation_minus_kappa_crit": (
            float(evaluation_kappa - kcrit)
            if np.isfinite(evaluation_kappa) and np.isfinite(kcrit)
            else np.nan
        ),
        "evaluation_over_kappa_crit": (
            float(evaluation_kappa / kcrit)
            if np.isfinite(evaluation_kappa) and np.isfinite(kcrit) and kcrit > 0
            else np.nan
        ),
        **orthogonal,
        **evaluation,
    })
    for event in events:
        event_rows.append({
            "epsilon": row["epsilon"],
            "pattern_set": row["pattern_set"],
            "seed": int(row["seed"]) + 17,
            "kappa_crit": kcrit,
            "evaluation_kappa": float(evaluation_kappa),
            "supercritical_factor": factor,
            **event,
        })

    return trial_rows[0], event_rows


def run(cfg: Config, workers: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Estimate kappa_crit first, then evaluate dynamics above each threshold.

    The matched gain is

        kappa_eval(epsilon, realization) = r * kappa_crit(epsilon, realization),

    where r>1 is either supplied explicitly or calibrated so that the mean
    evaluation gain at epsilon=0 equals ``reference_kappa``. This keeps the
    relative supercriticality fixed while guaranteeing kappa_eval>kappa_crit
    for every uncensored realization.
    """
    successor = successor_array(cycle_adjacency(cfg.n_memories, cfg.cycle))
    critical_theory = orthogonal_fixed_point(cfg.activation, cfg.reference_kappa)
    children = np.random.SeedSequence(cfg.seed).spawn(len(cfg.epsilons))

    jobs = [(cfg, index, float(epsilon), pattern_set, pchild)
            for index, (epsilon, child) in enumerate(zip(cfg.epsilons, children))
            for pattern_set, pchild in enumerate(child.spawn(cfg.pattern_sets))]
    prepared = list(ordered_map(prepare_realization, jobs, workers, "correlation_dynamics_critical"))

    baseline = np.array([
        row["kappa_crit"]
        for row in prepared
        if row["epsilon_index"] == 0 and np.isfinite(row["kappa_crit"])
    ], dtype=float)
    if baseline.size == 0:
        raise RuntimeError("No finite epsilon=0 critical-gain estimates")
    baseline_mean_kcrit = float(np.mean(baseline))
    factor = (
        float(cfg.supercritical_factor)
        if cfg.supercritical_factor is not None
        else float(cfg.reference_kappa / baseline_mean_kcrit)
    )
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError(
            "The matched supercritical factor must exceed one; "
            f"received {factor:.6g}. Increase --reference-kappa or set "
            "--supercritical-factor explicitly."
        )

    trial_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    jobs = [(cfg, row, factor, baseline_mean_kcrit) for row in prepared]
    for row, batch in ordered_map(evaluate_realization, jobs, workers, "correlation_dynamics_evaluation"):
        trial_rows.append(row)
        event_rows.extend(batch)

    trials = pd.DataFrame(trial_rows)
    events = pd.DataFrame(event_rows)
    summary_rows: list[dict[str, Any]] = []
    for epsilon, group in trials.groupby("epsilon", sort=True):
        k = group["kappa_crit"].to_numpy(float)
        eval_k = group["evaluation_kappa"].to_numpy(float)
        plateau = group["asymptotic_plateau_ok"] == 1
        t = group.loc[plateau, "asymptotic_T_esc"].to_numpy(float)
        z = group.loc[plateau, "asymptotic_Z"].to_numpy(float)
        orth_t = group.loc[plateau, "orthogonal_T_esc_matched"].to_numpy(float)
        orth_z = group.loc[plateau, "orthogonal_Z_matched"].to_numpy(float)
        success = group["evaluation_complete_cycle"].to_numpy(float)

        seed_offset = cfg.seed + int(1000 * epsilon)
        klo, khi = bootstrap_interval(k, cfg.bootstrap_reps, seed_offset + 1)
        tlo, thi = bootstrap_interval(t, cfg.bootstrap_reps, seed_offset + 2)
        zlo, zhi = bootstrap_interval(z, cfg.bootstrap_reps, seed_offset + 3)
        slo, shi = bootstrap_interval(success, cfg.bootstrap_reps, seed_offset + 4)
        elo, ehi = bootstrap_interval(eval_k, cfg.bootstrap_reps, seed_offset + 5)
        otlo, othi = bootstrap_interval(orth_t, cfg.bootstrap_reps, seed_offset + 6)
        ozlo, ozhi = bootstrap_interval(orth_z, cfg.bootstrap_reps, seed_offset + 7)

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
            "mean_evaluation_kappa": float(np.nanmean(eval_k)) if np.any(np.isfinite(eval_k)) else np.nan,
            "evaluation_kappa_ci_low": elo,
            "evaluation_kappa_ci_high": ehi,
            "evaluation_gain_censored_fraction": float(group["evaluation_gain_censored"].mean()),
            "mean_evaluation_over_kappa_crit": (
                float(np.nanmean(group["evaluation_over_kappa_crit"]))
                if np.any(np.isfinite(group["evaluation_over_kappa_crit"]))
                else np.nan
            ),
            "evaluation_cycle_success_fraction": float(np.mean(success)),
            "evaluation_cycle_success_ci_low": slo,
            "evaluation_cycle_success_ci_high": shi,
            "evaluation_asymptotic_success_fraction": float(group["evaluation_asymptotic_complete"].mean()),
            "plateau_fraction": float(group["asymptotic_plateau_ok"].mean()),
            "mean_asymptotic_T_esc": float(np.nanmean(t)) if np.any(np.isfinite(t)) else np.nan,
            "asymptotic_T_esc_ci_low": tlo,
            "asymptotic_T_esc_ci_high": thi,
            "asymptotic_T_esc_observed_fraction": float(np.sum(np.isfinite(t)) / len(group)),
            "mean_orthogonal_T_esc_matched": float(np.nanmean(orth_t)) if np.any(np.isfinite(orth_t)) else np.nan,
            "orthogonal_T_esc_matched_ci_low": otlo,
            "orthogonal_T_esc_matched_ci_high": othi,
            "mean_asymptotic_Z": float(np.nanmean(z)) if np.any(np.isfinite(z)) else np.nan,
            "asymptotic_Z_ci_low": zlo,
            "asymptotic_Z_ci_high": zhi,
            "asymptotic_Z_observed_fraction": float(np.sum(np.isfinite(z)) / len(group)),
            "mean_orthogonal_Z_matched": float(np.nanmean(orth_z)) if np.any(np.isfinite(orth_z)) else np.nan,
            "orthogonal_Z_matched_ci_low": ozlo,
            "orthogonal_Z_matched_ci_high": ozhi,
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
    summary["delta_asymptotic_T_esc_from_matched_orthogonal"] = (
        summary["mean_asymptotic_T_esc"] - summary["mean_orthogonal_T_esc_matched"]
    )
    summary["delta_asymptotic_Z_from_matched_orthogonal"] = (
        summary["mean_asymptotic_Z"] - summary["mean_orthogonal_Z_matched"]
    )
    summary["epsilon_max_grid"] = eps_grid
    summary["epsilon_max_interpolated"] = eps_interp
    summary["epsilon_max_success_probability"] = cfg.epsilon_max_success_probability
    summary["epsilon_max_definition"] = (
        "plateau-qualified sustained cycles at matched relative supercriticality"
    )

    theory: dict[str, Any] = {
        "orthogonal_kappa_crit": critical_theory["orthogonal_kappa_crit"],
        "orthogonal_T_esc_critical": critical_theory["orthogonal_T_esc_critical"],
        "orthogonal_Z_critical": critical_theory["orthogonal_Z_critical"],
        "baseline_reference_kappa": cfg.reference_kappa,
        "baseline_mean_empirical_kappa_crit": baseline_mean_kcrit,
        "supercritical_factor": factor,
        "gain_rule": "kappa_eval = supercritical_factor * kappa_crit for each realization",
        "evaluation_kappa_max": cfg.evaluation_kappa_max,
        "epsilon_max_grid": float(eps_grid),
        "epsilon_max_interpolated": float(eps_interp),
        "epsilon_max_success_probability": float(cfg.epsilon_max_success_probability),
        "epsilon_max_definition": (
            "plateau-qualified sustained cycles at matched relative supercriticality"
        ),
    }
    return trials, events, summary, theory

