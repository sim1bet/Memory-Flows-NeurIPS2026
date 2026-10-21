#!/usr/bin/env python3
"""Capacity sweeps for scaled cycles, fixed cycles, and fanout branching."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:  # Optional progress display; numerical behavior is unchanged.
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.activations import activation, branch_amplitude
from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.graphs import nested_fanout_successors, scaled_cycle_length
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import random_patterns
from idp_hopfield.stats import bootstrap_mean_interval, wilson_interval


EXPERIMENT_IDS = {
    "scaled_cycle": 11,
    "fixed_cycle": 12,
    "fanout": 21,
    "fanout_phase": 22,
}


def p_grid(
    n: int,
    *,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    count: int | None = None,
    alpha_values: list[float] | None = None,
    min_p: int = 2,
) -> list[int]:
    """Return distinct integer P values without imposing P <= N.

    Overcomplete fixed-length experiments intentionally use P/N >> 1.
    """
    if alpha_values is not None:
        values = [max(min_p, int(round(float(alpha) * n))) for alpha in alpha_values]
        unique = list(dict.fromkeys(values))
        if len(unique) != len(values):
            raise ValueError(f"Alpha values do not map to distinct P values for N={n}.")
        return unique
    if alpha_min is None or alpha_max is None or count is None:
        raise ValueError("Provide alpha_values or alpha_min/alpha_max/count.")
    if count < 1 or alpha_max < alpha_min:
        raise ValueError("Invalid alpha grid.")
    targets = np.linspace(float(alpha_min) * n, float(alpha_max) * n, int(count))
    values = [max(min_p, int(round(value))) for value in targets]
    if len(set(values)) != len(values):
        # Move repeated rounded values to the nearest unused integer.
        chosen: list[int] = []
        for target in targets:
            candidate = max(min_p, int(round(target)))
            if candidate in chosen:
                radius = 1
                while candidate in chosen:
                    options = [max(min_p, int(round(target)) - radius), int(round(target)) + radius]
                    candidate = min((v for v in options if v not in chosen), key=lambda v: abs(v - target))
                    radius += 1
            chosen.append(candidate)
        values = chosen
    return values


def scores(patterns: np.ndarray, state: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(state)
    if norm == 0.0:
        zero = np.zeros(patterns.shape[1])
        return zero, zero
    signed = patterns.T @ state / (np.sqrt(patterns.shape[0]) * norm)
    return (np.abs(signed) if mode == "absolute" else signed), signed


def top_two(values: np.ndarray) -> tuple[int, int, float, float]:
    indices = np.argpartition(values, -2)[-2:]
    indices = indices[np.argsort(values[indices])[::-1]]
    return int(indices[0]), int(indices[1]), float(values[indices[0]]), float(values[indices[1]])


def cycle_trial(
    patterns: np.ndarray,
    order: np.ndarray,
    source: int,
    activation_name: str,
    dynamics: dict[str, Any],
    rng: np.random.Generator,
    sparse_decoy_scoring: bool = False,
) -> dict[str, Any]:
    n, p = patterns.shape
    successor = np.full(p, -1, dtype=int)
    successor[order] = np.roll(order, -1)
    gamma = branch_amplitude(float(dynamics["z_init"]), activation_name)
    state = gamma * patterns[:, source] + float(dynamics["initial_noise"]) * rng.standard_normal(n)
    saliency = np.zeros(p)
    saliency[source] = float(dynamics["z_init"])
    dt_fast = float(dynamics["dt_fast"])
    dt_slow = dt_fast * float(dynamics["tau_fast"]) / float(dynamics["tau_slow"])
    check_steps = max(1, int(round(float(dynamics["check_interval"]) / dt_slow)))
    hold_checks = max(1, int(math.ceil(float(dynamics["hold_time"]) / (check_steps * dt_slow))))
    max_steps = int(math.ceil(len(order) * float(dynamics["max_time_per_transition"]) / dt_slow))
    current = int(source)
    completed = 0
    hold = wrong_hold = collapse_hold = 0
    hold_start = np.nan
    last_transition = 0.0
    transition_times: list[float] = []
    final_best = current
    final_best_value = 1.0
    final_second_value = 0.0
    outcome = "timeout"

    active_patterns = patterns[:, order]
    for step in range(max_steps):
        phi = activation(state, activation_name)
        # Only cycle memories have nonzero saliency, so this is algebraically
        # identical to the full P-dimensional synaptic update.
        active_projection = active_patterns.T @ phi / float(n)
        drive = np.zeros(p)
        drive[np.roll(order, -1)] = active_projection
        next_state = state + dt_fast * (
            -state + active_patterns @ (np.abs(saliency[order]) * active_projection)
        )
        next_saliency = saliency + dt_slow * (-saliency + float(dynamics["kappa"]) * drive)
        state, saliency = next_state, next_saliency
        if (step + 1) % check_steps:
            continue
        time = (step + 1) * dt_slow
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(saliency)):
            outcome = "numerical_failure"
            break
        rms = np.linalg.norm(state) / np.sqrt(n)
        collapse_hold = collapse_hold + 1 if rms < float(dynamics["collapse_rms"]) else 0
        if collapse_hold >= hold_checks:
            outcome = "collapse"
            break
        target = int(successor[current])
        if sparse_decoy_scoring:
            active_value, _ = scores(active_patterns, state, dynamics["score_mode"])
            best_local, _, active_best, active_second = top_two(active_value)
            best = int(order[best_local])
            best_value, second_value = active_best, active_second
            pure = best_value >= float(dynamics["min_cosine"]) and best_value - second_value >= float(dynamics["min_margin"])
            # A success candidate is accepted only after checking every stored
            # pattern, so overcomplete decoys remain part of the retrieval test.
            if pure and best == target:
                value, _ = scores(patterns, state, dynamics["score_mode"])
                best, _, best_value, second_value = top_two(value)
                pure = best_value >= float(dynamics["min_cosine"]) and best_value - second_value >= float(dynamics["min_margin"])
                final_best, final_best_value, final_second_value = best, best_value, second_value
        else:
            value, _ = scores(patterns, state, dynamics["score_mode"])
            best, _, best_value, second_value = top_two(value)
            pure = best_value >= float(dynamics["min_cosine"]) and best_value - second_value >= float(dynamics["min_margin"])
            final_best, final_best_value, final_second_value = best, best_value, second_value
        if pure and best == target:
            if hold == 0:
                hold_start = time
            hold += 1
        else:
            hold = 0
            hold_start = np.nan
        if sparse_decoy_scoring:
            # Decoys have zero saliency and cannot become dynamical attractors;
            # they are nevertheless checked before every accepted transition.
            wrong_hold = 0
        else:
            wrong_hold = wrong_hold + 1 if pure and best not in (current, target) else 0
            if wrong_hold >= hold_checks:
                outcome = "wrong_memory"
                break
        if hold >= hold_checks:
            event_time = float(hold_start)
            transition_times.append(event_time - last_transition)
            last_transition = event_time
            completed += 1
            current = target
            hold = wrong_hold = 0
            if completed == len(order):
                outcome = "success" if current == source else "wrong_memory"
                break
        if time - last_transition > float(dynamics["max_time_per_transition"]):
            outcome = "timeout"
            break

    return {
        "success": outcome == "success",
        "outcome": outcome,
        "transitions_completed": completed,
        "required_transitions": len(order),
        "sequence_fraction_completed": completed / len(order),
        "first_transition_time": transition_times[0] if transition_times else np.nan,
        "mean_transition_time": float(np.mean(transition_times)) if transition_times else np.nan,
        "completion_time": last_transition if outcome == "success" else np.nan,
        "final_best_memory": final_best,
        "final_best_cosine": final_best_value,
        "final_second_cosine": final_second_value,
    }


def fanout_trial(
    patterns: np.ndarray,
    source: int,
    targets: np.ndarray,
    activation_name: str,
    dynamics: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    n, p = patterns.shape
    gamma = branch_amplitude(float(dynamics["z_init"]), activation_name)
    state = gamma * patterns[:, source] + float(dynamics["initial_noise"]) * rng.standard_normal(n)
    saliency = np.zeros(p)
    saliency[source] = float(dynamics["z_init"])
    dt_fast = float(dynamics["dt_fast"])
    dt_slow = dt_fast * float(dynamics["tau_fast"]) / float(dynamics["tau_slow"])
    check_steps = max(1, int(round(float(dynamics["check_interval"]) / dt_slow)))
    hold_checks = max(1, int(math.ceil(float(dynamics["hold_time"]) / (check_steps * dt_slow))))
    max_steps = int(math.ceil(float(dynamics["max_time_per_transition"]) / dt_slow))
    candidate = -1
    hold = wrong_hold = collapse_hold = 0
    hold_start = np.nan
    final_best = source
    final_best_value = 1.0
    final_second_value = 0.0
    selected = -1
    selection_time = np.nan
    outcome = "timeout_mixed"
    target_set = set(int(value) for value in targets)

    for step in range(max_steps):
        phi = activation(state, activation_name)
        projection = patterns.T @ phi / float(n)
        drive = np.zeros(p)
        drive[targets] = projection[source]
        next_state = state + dt_fast * (-state + patterns @ (np.abs(saliency) * projection))
        next_saliency = saliency + dt_slow * (-saliency + float(dynamics["kappa"]) * drive)
        state, saliency = next_state, next_saliency
        if (step + 1) % check_steps:
            continue
        time = (step + 1) * dt_slow
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(saliency)):
            outcome = "numerical_failure"
            break
        rms = np.linalg.norm(state) / np.sqrt(n)
        collapse_hold = collapse_hold + 1 if rms < float(dynamics["collapse_rms"]) else 0
        if collapse_hold >= hold_checks:
            outcome = "collapse"
            break
        value, _ = scores(patterns, state, dynamics["score_mode"])
        best, _, best_value, second_value = top_two(value)
        final_best, final_best_value, final_second_value = best, best_value, second_value
        pure = best_value >= float(dynamics["min_cosine"]) and best_value - second_value >= float(dynamics["min_margin"])
        valid = pure and best in target_set
        if valid:
            if candidate == best:
                hold += 1
            else:
                candidate, hold, hold_start = best, 1, time
        else:
            candidate, hold, hold_start = -1, 0, np.nan
        wrong_hold = wrong_hold + 1 if pure and best != source and best not in target_set else 0
        if wrong_hold >= hold_checks:
            outcome = "wrong_memory"
            break
        if hold >= hold_checks:
            selected = candidate
            selection_time = float(hold_start)
            outcome = "success"
            break
    if outcome.startswith("timeout"):
        pure = final_best_value >= float(dynamics["min_cosine"]) and final_best_value - final_second_value >= float(dynamics["min_margin"])
        outcome = "timeout_source" if pure and final_best == source else "timeout_mixed"
    z_abs = np.abs(saliency)
    total_sq = float(z_abs @ z_abs)
    selected_z = float(z_abs[selected]) if selected >= 0 else np.nan
    return {
        "success": outcome == "success",
        "outcome": outcome,
        "transitions_completed": int(outcome == "success"),
        "required_transitions": 1,
        "sequence_fraction_completed": float(outcome == "success"),
        "first_transition_time": selection_time,
        "mean_transition_time": selection_time,
        "completion_time": selection_time,
        "selected_successor": selected,
        "final_best_memory": final_best,
        "final_best_cosine": final_best_value,
        "final_second_cosine": final_second_value,
        "active_saliency_count_at_selection": int(np.sum(z_abs > 1.0)) if selected >= 0 else np.nan,
        "effective_load_at_selected_memory": (
            (total_sq - selected_z ** 2) / (n * selected_z ** 2)
            if selected >= 0 and selected_z > 0.0
            else np.nan
        ),
    }


def _seed_rngs(seed: int, n: int, p: int, pattern_set: int, experiment: str):
    family_id = 20 if experiment.startswith("fanout") else EXPERIMENT_IDS[experiment]
    sequence = np.random.SeedSequence([seed, family_id, n, p, pattern_set])
    return [np.random.default_rng(child) for child in sequence.spawn(4)]


def run_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    n, p, set_index = task["N"], task["P"], task["pattern_set"]
    cfg = task["config"]
    experiment = str(task["experiment"])
    memory_rng, graph_rng, source_rng, noise_seed_rng = _seed_rngs(int(cfg["seed"]), n, p, set_index, experiment)
    patterns = random_patterns(memory_rng, n, p, dtype=np.float32)
    rows: list[dict[str, Any]] = []

    if experiment in ("scaled_cycle", "fixed_cycle"):
        if experiment == "scaled_cycle":
            cycle_length = scaled_cycle_length(p, float(task["cycle_fraction"]))
        else:
            cycle_length = int(task["cycle_length"])
            if p < cycle_length:
                raise ValueError(f"Fixed cycle L={cycle_length} requires P >= L, got P={p}.")
        cycle_order = graph_rng.choice(p, size=cycle_length, replace=False)
        cycle_sources = source_rng.choice(
            cycle_order,
            size=min(int(task["cycle_sources_per_set"]), cycle_length),
            replace=False,
        )
        for activation_index, activation_name in enumerate(cfg["activations"]):
            for source_index, source in enumerate(cycle_sources):
                trial_seed = int(noise_seed_rng.integers(1, 2**31 - 1)) + 1000 * activation_index + source_index
                result = cycle_trial(
                    patterns, cycle_order, int(source), activation_name, cfg["dynamics"],
                    np.random.default_rng(trial_seed), sparse_decoy_scoring=(experiment == "fixed_cycle")
                )
                rows.append({
                    "experiment": experiment,
                    "activation": activation_name,
                    "N": n,
                    "P": p,
                    "alpha_actual": p / n,
                    "pattern_set": set_index,
                    "source": int(source),
                    "cycle_length": cycle_length,
                    "cycle_length_over_P": cycle_length / p,
                    "outdegree": 1,
                    **result,
                })
        return rows

    if experiment in ("fanout", "fanout_phase"):
        degrees = [int(v) for v in task["outdegrees"]]
        maximum = min(max(degrees), p - 1)
        fanout_sources = source_rng.choice(
            p,
            size=min(int(task["fanout_sources_per_set"]), p),
            replace=False,
        )
        for source_index, source in enumerate(fanout_sources):
            ordering = nested_fanout_successors(graph_rng, p, int(source), maximum)
            for activation_index, activation_name in enumerate(cfg["activations"]):
                for outdegree in degrees:
                    if outdegree >= p:
                        continue
                    targets = ordering[:outdegree]
                    # Deterministic paired noise across fanout outputs for the same cell.
                    trial_seed = int(np.random.SeedSequence([
                        int(cfg["seed"]), 9901, n, p, set_index, int(source), activation_index, outdegree
                    ]).generate_state(1)[0])
                    result = fanout_trial(
                        patterns, int(source), targets, activation_name, cfg["dynamics"], np.random.default_rng(trial_seed)
                    )
                    rows.append({
                        "experiment": experiment,
                        "activation": activation_name,
                        "N": n,
                        "P": p,
                        "alpha_actual": p / n,
                        "pattern_set": set_index,
                        "source": int(source),
                        "cycle_length": 0,
                        "cycle_length_over_P": 0.0,
                        "outdegree": outdegree,
                        "successor_set": ";".join(str(int(value)) for value in targets),
                        **result,
                    })
        return rows

    raise ValueError(f"Unknown experiment {experiment!r}")


def summarize(trials: pd.DataFrame, seed: int, repetitions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = ["experiment", "activation", "N", "P", "alpha_actual", "cycle_length", "outdegree"]
    set_rows: list[dict[str, Any]] = []
    for keys, group in trials.groupby(group_keys + ["pattern_set"], dropna=False):
        row = dict(zip(group_keys + ["pattern_set"], keys))
        row.update({
            "sources": len(group),
            "success_probability_within_set": float(group["success"].mean()),
            "mean_sequence_fraction": float(group["sequence_fraction_completed"].mean()),
        })
        set_rows.append(row)
    pattern_sets = pd.DataFrame(set_rows)

    probability_rows: list[dict[str, Any]] = []
    for index, (keys, group) in enumerate(pattern_sets.groupby(group_keys, dropna=False)):
        values = group["success_probability_within_set"].to_numpy(float)
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1771]))
        if np.ptp(values) == 0.0:
            effective_successes = int(round(float(values.mean()) * len(values)))
            low, high = wilson_interval(effective_successes, len(values))
        else:
            low, high = bootstrap_mean_interval(values, rng, repetitions)
        trial_group = trials
        for key, value in zip(group_keys, keys):
            trial_group = trial_group[trial_group[key] == value]
        row = dict(zip(group_keys, keys))
        row.update({
            "pattern_sets": len(group),
            "source_trials": len(trial_group),
            "success_probability": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "mean_sequence_fraction": float(group["mean_sequence_fraction"].mean()),
            "mean_transition_time_successes": float(trial_group.loc[trial_group["success"], "mean_transition_time"].mean()),
        })
        probability_rows.append(row)
    return pattern_sets, pd.DataFrame(probability_rows)


def _grid_spec(section: dict[str, Any], n: int) -> list[int]:
    return p_grid(
        n,
        alpha_min=section.get("alpha_min"),
        alpha_max=section.get("alpha_max"),
        count=section.get("p_count"),
        alpha_values=section.get("alpha_values"),
    )


def build_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for experiment in ("scaled_cycle", "fixed_cycle", "fanout", "fanout_phase"):
        section = cfg.get(experiment)
        if not section or not bool(section.get("enabled", True)):
            continue
        for n in section["N_values"]:
            for p in _grid_spec(section, int(n)):
                if experiment == "fixed_cycle" and p < int(section["cycle_length"]):
                    continue
                for pattern_set in range(int(cfg["pattern_sets"])):
                    task = {
                        "N": int(n),
                        "P": int(p),
                        "pattern_set": pattern_set,
                        "experiment": experiment,
                        "config": cfg,
                    }
                    task.update(section)
                    tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)
    tasks = build_tasks(cfg)
    if not tasks:
        raise ValueError("Capacity configuration produced no tasks.")

    all_rows: list[dict[str, Any]] = []
    for rows in ordered_map(run_task, tasks, args.workers, "capacity"):
        all_rows.extend(rows)

    trials = pd.DataFrame(all_rows)
    pattern_sets, probability = summarize(trials, int(cfg["seed"]), int(cfg["bootstrap_reps"]))
    grid = probability[["experiment", "N", "P", "alpha_actual"]].drop_duplicates().sort_values(["experiment", "N", "P"])
    write_csv(trials, output / "capacity_trials.csv")
    write_csv(pattern_sets, output / "capacity_pattern_sets.csv")
    write_csv(probability, output / "capacity_probability.csv")
    write_csv(grid, output / "capacity_grid.csv")
    write_json(cfg, output / "capacity_config.json")


if __name__ == "__main__":
    main()
