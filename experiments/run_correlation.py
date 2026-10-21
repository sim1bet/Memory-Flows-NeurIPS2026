#!/usr/bin/env python3
"""Controlled-correlation retrieval with raw and pseudoinverse overlaps."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.graphs import successor_array, cycle_adjacency
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import correlated_patterns, gram_pseudoinverse
from idp_hopfield.stats import wilson_interval

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args: Any, **kwargs: Any):
        def decorate(function):
            return function
        return decorate

OUTCOMES = {0: "success", 1: "timeout", 2: "wrong_memory", 3: "collapse", 4: "nonfinite"}


def branch_gamma(z: float, activation_name: str) -> float:
    if activation_name == "smoothf":
        return max(z - 1.0, 0.0)
    return float(brentq(lambda value: value - z * np.tanh(value), 1e-12, z + 1.0))


@njit(cache=True)
def activation_scalar(value: float, code: int) -> float:
    return math.tanh(value) if code == 0 else value / (1.0 + abs(value))


@njit(cache=True)
def simulate_trial(
    patterns: np.ndarray,
    pinv: np.ndarray,
    successor: np.ndarray,
    source_initial: int,
    required_transitions: int,
    rule_code: int,
    activation_code: int,
    gamma_initial: float,
    z_initial: float,
    kappa: float,
    gain_w: float,
    dt_fast: float,
    dt_slow: float,
    max_time_per_transition: float,
    min_alignment: float,
    min_margin: float,
    hold_checks: int,
    wrong_hold_checks: int,
    collapse_hold_checks: int,
    check_stride: int,
    coefficient_noise: float,
    seed: int,
) -> tuple:
    np.random.seed(seed)
    n, p = patterns.shape
    coefficients = np.zeros(p)
    saliency = np.zeros(p)
    coefficients[source_initial] = gamma_initial
    saliency[source_initial] = z_initial
    for memory in range(p):
        coefficients[memory] += coefficient_noise * np.random.randn()

    state = np.zeros(n)
    phi = np.zeros(n)
    raw = np.zeros(p)
    decoded = np.zeros(p)
    drive = np.zeros(p)
    current = source_initial
    completed = target_hold = wrong_hold = collapse_hold = 0
    edge_time = total_time = 0.0
    switch_sum = switch_square_sum = 0.0
    min_alignment_seen = 1e100
    min_margin_seen = 1e100
    max_active = 1
    max_distractor = 0.0
    max_steps = int(math.ceil(required_transitions * max_time_per_transition / dt_slow)) + check_stride + 1

    for step in range(max_steps):
        for neuron in range(n):
            value = 0.0
            for memory in range(p):
                value += patterns[neuron, memory] * coefficients[memory]
            state[neuron] = value
            phi[neuron] = activation_scalar(value, activation_code)
        for memory in range(p):
            value = 0.0
            for neuron in range(n):
                value += patterns[neuron, memory] * phi[neuron]
            raw[memory] = value / n
        if rule_code == 1:
            for row in range(p):
                value = 0.0
                for column in range(p):
                    value += pinv[row, column] * raw[column]
                decoded[row] = value
        else:
            for memory in range(p):
                decoded[memory] = raw[memory]

        intended = successor[current]
        for memory in range(p):
            if memory != current and memory != intended:
                max_distractor = max(max_distractor, abs(decoded[memory]))
        for memory in range(p):
            drive[memory] = 0.0
        for source in range(p):
            target = successor[source]
            if target >= 0:
                drive[target] += decoded[source]

        next_coefficients = np.empty(p)
        next_saliency = np.empty(p)
        for memory in range(p):
            next_coefficients[memory] = coefficients[memory] + dt_fast * (
                -coefficients[memory] + gain_w * abs(saliency[memory]) * decoded[memory]
            ) + coefficient_noise * np.random.randn()
            next_saliency[memory] = saliency[memory] + dt_slow * (
                -saliency[memory] + kappa * drive[memory]
            )
            if not math.isfinite(next_coefficients[memory]) or not math.isfinite(next_saliency[memory]):
                return 4, completed, total_time, math.nan, math.nan, math.nan, math.nan, max_active, max_distractor
        coefficients, saliency = next_coefficients, next_saliency
        edge_time += dt_slow
        total_time += dt_slow
        if (step + 1) % check_stride != 0:
            continue

        # Classify the updated state.
        state_norm_sq = 0.0
        for neuron in range(n):
            value = 0.0
            for memory in range(p):
                value += patterns[neuron, memory] * coefficients[memory]
            state[neuron] = value
            state_norm_sq += value * value
        state_norm = math.sqrt(state_norm_sq)
        if state_norm < 1e-12:
            collapse_hold += 1
            if collapse_hold >= collapse_hold_checks:
                return 3, completed, total_time, math.nan, math.nan, math.nan, math.nan, max_active, max_distractor
            continue
        collapse_hold = 0

        best_index = -1
        best = -1e100
        second = -1e100
        normalization = math.sqrt(n) * state_norm
        for memory in range(p):
            dot = 0.0
            for neuron in range(n):
                dot += patterns[neuron, memory] * state[neuron]
            value = abs(dot) / normalization
            if value > best:
                second = best
                best = value
                best_index = memory
            elif value > second:
                second = value
        margin = best - second
        target = successor[current]
        pure = best >= min_alignment and margin >= min_margin
        target_hold = target_hold + 1 if pure and best_index == target else 0
        wrong_hold = wrong_hold + 1 if pure and best_index != current and best_index != target else 0
        active = 0
        for memory in range(p):
            if abs(saliency[memory]) > 1.0:
                active += 1
        max_active = max(max_active, active)

        if wrong_hold >= wrong_hold_checks:
            mean = switch_sum / completed if completed else math.nan
            return 2, completed, total_time, mean, math.nan, min_alignment_seen if completed else math.nan, min_margin_seen if completed else math.nan, max_active, max_distractor
        if target_hold >= hold_checks:
            switch_time = edge_time - (hold_checks - 1) * check_stride * dt_slow
            completed += 1
            switch_sum += switch_time
            switch_square_sum += switch_time * switch_time
            min_alignment_seen = min(min_alignment_seen, best)
            min_margin_seen = min(min_margin_seen, margin)
            current = target
            edge_time = 0.0
            target_hold = wrong_hold = 0
            if completed >= required_transitions:
                mean = switch_sum / completed
                variance = switch_square_sum / completed - mean * mean
                return 0, completed, total_time, mean, math.sqrt(max(variance, 0.0)), min_alignment_seen, min_margin_seen, max_active, max_distractor
        if edge_time >= max_time_per_transition:
            mean = switch_sum / completed if completed else math.nan
            return 1, completed, total_time, mean, math.nan, min_alignment_seen if completed else math.nan, min_margin_seen if completed else math.nan, max_active, max_distractor

    return 1, completed, total_time, math.nan, math.nan, math.nan, math.nan, max_active, max_distractor


def run_correlation_set(task):
    cfg, correlation, set_index = task
    n, p = int(cfg["N"]), int(cfg["P"])
    cycle = np.asarray(cfg["cycle"], dtype=int)
    successor = successor_array(cycle_adjacency(p, cycle))
    dt_slow = float(cfg["dt_fast"]) * float(cfg["tau_fast"]) / float(cfg["tau_slow"])
    check_stride = max(1, int(round(float(cfg["check_interval"]) / dt_slow)))
    check_dt = check_stride * dt_slow
    hold_checks = max(1, int(math.ceil(float(cfg["hold_time"]) / check_dt)))
    wrong_checks = max(1, int(math.ceil(float(cfg["wrong_hold_time"]) / check_dt)))
    collapse_checks = max(1, int(math.ceil(float(cfg["collapse_hold_time"]) / check_dt)))

    rows = []
    seed_sequence = np.random.SeedSequence([int(cfg["seed"]), int(round(correlation * 1e6)), set_index])
    memory_rng, source_rng, noise_rng = [np.random.default_rng(seed) for seed in seed_sequence.spawn(3)]
    patterns = correlated_patterns(memory_rng, n, p, correlation, cfg["correlation_model"])
    gram, pinv, diagnostics = gram_pseudoinverse(patterns, float(cfg["pinv_rcond"]))
    source = int(source_rng.choice(cycle))
    for activation_name in cfg["activations"]:
        gamma = branch_gamma(float(cfg["z_initial"]), activation_name)
        for rule in cfg["rules"]:
            result = simulate_trial(
                patterns,
                pinv,
                successor,
                source,
                len(cycle),
                1 if rule == "gpi" else 0,
                0 if activation_name == "tanh" else 1,
                gamma,
                float(cfg["z_initial"]),
                float(cfg["kappa"]),
                1.0,
                float(cfg["dt_fast"]),
                dt_slow,
                float(cfg["max_time_per_transition"]),
                float(cfg["min_alignment"]),
                float(cfg["min_margin"]),
                hold_checks,
                wrong_checks,
                collapse_checks,
                check_stride,
                float(cfg["fast_noise_per_neuron"]) / np.sqrt(n),
                int(noise_rng.integers(0, 2**31 - 1)),
            )
            outcome_code, completed, total_time, mean_switch, switch_std, min_align, min_margin, max_active, max_distractor = result
            rows.append(
                {
                    "correlation_parameter": correlation,
                    "expected_pairwise_overlap": correlation ** 2 if cfg["correlation_model"] == "biased" else correlation,
                    "empirical_pairwise_overlap": diagnostics["gram_offdiag_mean"],
                    "pattern_set": set_index,
                    "activation": activation_name,
                    "rule": rule,
                    "source": source,
                    "required_transitions": len(cycle),
                    "transitions_completed": int(completed),
                    "sequence_fraction_completed": completed / len(cycle),
                    "success": outcome_code == 0,
                    "outcome": OUTCOMES[int(outcome_code)],
                    "total_slow_time": total_time,
                    "mean_switch_time": mean_switch,
                    "switch_time_std": switch_std,
                    "minimum_winner_alignment": min_align,
                    "minimum_winner_margin": min_margin,
                    "max_active_saliencies": int(max_active),
                    "max_abs_decoded_distractor": max_distractor,
                    **diagnostics,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)

    jobs = [(cfg, float(corr), index) for corr in cfg["correlations"]
            for index in range(int(cfg["pattern_sets"]))]
    rows = [row for batch in ordered_map(run_correlation_set, jobs, args.workers, "correlation") for row in batch]
    trials = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for keys, group in trials.groupby(["activation", "rule", "correlation_parameter"], sort=True):
        successes = int(group["success"].sum())
        low, high = wilson_interval(successes, len(group))
        summaries.append(
            {
                "activation": keys[0],
                "rule": keys[1],
                "correlation_parameter": keys[2],
                "expected_pairwise_overlap": float(group["expected_pairwise_overlap"].mean()),
                "empirical_pairwise_overlap": float(group["empirical_pairwise_overlap"].mean()),
                "pattern_sets": len(group),
                "success_probability": successes / len(group),
                "ci_low": low,
                "ci_high": high,
                "mean_sequence_fraction": float(group["sequence_fraction_completed"].mean()),
                "mean_switch_time_successes": float(group.loc[group["success"], "mean_switch_time"].mean()),
                "mean_gram_condition": float(group["gram_condition"].mean()),
                "mean_pinv_residual": float(group["pinv_reconstruction_residual"].mean()),
                "mean_decoded_distractor": float(group["max_abs_decoded_distractor"].mean()),
            }
        )
    write_csv(trials, output / "correlation_trials.csv")
    write_csv(pd.DataFrame(summaries), output / "correlation_summary.csv")
    write_json(cfg, output / "correlation_config.json")


if __name__ == "__main__":
    main()
