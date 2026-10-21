#!/usr/bin/env python3
"""Correlation robustness sweep for the two-timescale sequential IDP network.

This script extends ``LimitCycle_bipartite(1).py`` to non-orthogonal patterns.
It implements the generalized pseudoinverse (GPI) overlap rule

    q = O^+ m,     O = Xi.T @ Xi / N,
    m = Xi.T @ psi(y) / N,

and uses q in both the fast saliency-gated reconstruction and the slow graph
feedback.  The original (raw-overlap) rule is retained as a paired baseline.

The default correlation generator matches Chaudhry et al.'s public notebook:
P(xi_i^mu = +/-1) = (1 +/- epsilon) / 2.  Its expected pairwise overlap is
approximately epsilon**2.  A zero-mean prototype-flip generator with direct
expected pairwise overlap rho is also available.

The script also evaluates the critical slow gain and asymptotic escape time.
For a full-column-rank pattern matrix, the GPI decoder exactly removes the Gram
matrix from the source-memory branch, reduced slow flow, and fast Jacobian
spectrum.  Consequently kappa_crit and T_esc are correlation-invariant in the
ideal GPI model.  The sweep verifies the required identities for every sampled
pattern set and reports where rank loss or pseudoinverse truncation would make
that conclusion invalid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize_scalar

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    def njit(*args: Any, **kwargs: Any):
        def decorator(function):
            return function
        return decorator


RULE_NAMES = ("raw", "gpi")
ACTIVATION_NAMES = ("tanh", "smoothf")
OUTCOME_NAMES = {
    0: "success",
    1: "timeout",
    2: "wrong_memory",
    3: "collapse",
    4: "nonfinite",
}


@dataclass(frozen=True)
class SweepConfig:
    n_neurons: int = 256
    n_memories: int = 15
    cycle: tuple[int, ...] = (0, 1, 2, 3, 4)
    correlation_model: str = "biased"
    correlation_values: tuple[float, ...] = tuple(np.linspace(0.0, 0.7, 8))
    pattern_sets: int = 24
    rules: tuple[str, ...] = RULE_NAMES
    activation: str = "tanh"
    kappa: float = 15.0
    z_initial: float = 8.0
    tau_fast: float = 0.01
    tau_slow: float = 1.0
    dt_fast: float = 0.05
    max_slow_time_per_transition: float = 4.0
    min_alignment: float = 0.90
    min_margin: float = 0.10
    hold_time: float = 0.02
    wrong_hold_time: float = 0.05
    collapse_hold_time: float = 0.05
    check_interval: float = 0.005
    fast_noise_per_neuron: float = 1e-6
    pinv_rcond: float = 1e-10
    seed: int = 20260727
    bootstrap_reps: int = 4000

    @property
    def epsilon_timescale(self) -> float:
        return self.tau_fast / self.tau_slow

    @property
    def dt_slow(self) -> float:
        return self.dt_fast * self.epsilon_timescale


# ---------------------------------------------------------------------------
# Pattern generation and GPI construction
# ---------------------------------------------------------------------------

def generate_correlated_patterns(
    rng: np.random.Generator,
    n_neurons: int,
    n_memories: int,
    parameter: float,
    model: str,
) -> np.ndarray:
    """Return an N x P Rademacher pattern matrix with controlled correlation.

    ``biased`` reproduces the public LongSequenceHopfieldMemory notebook:

        P(xi=+1) = (1+epsilon)/2,

    so E[xi^mu_i xi^nu_i] = epsilon^2 for mu != nu.

    ``prototype`` uses a latent zero-mean prototype and independent bit flips;
    the parameter is then the direct expected pairwise overlap rho.
    """
    parameter = float(parameter)
    if not 0.0 <= parameter < 1.0:
        raise ValueError("Correlation parameter must lie in [0, 1).")

    if model == "biased":
        probability_plus = 0.5 * (1.0 + parameter)
        draws = rng.random((n_neurons, n_memories)) < probability_plus
        return np.where(draws, 1.0, -1.0)

    if model == "prototype":
        prototype = rng.choice(np.array([-1.0, 1.0]), size=n_neurons)
        mean_flip = math.sqrt(parameter)
        probability_same = 0.5 * (1.0 + mean_flip)
        same = rng.random((n_neurons, n_memories)) < probability_same
        flips = np.where(same, 1.0, -1.0)
        return prototype[:, None] * flips

    raise ValueError(f"Unknown correlation model: {model!r}")


def gram_and_pseudoinverse(
    patterns: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compute O=Xi^T Xi/N, O^+, and conditioning diagnostics."""
    n_neurons, n_memories = patterns.shape
    gram = patterns.T @ patterns / float(n_neurons)
    gram = 0.5 * (gram + gram.T)
    pseudoinverse = np.linalg.pinv(gram, rcond=rcond, hermitian=True)

    eigenvalues = np.linalg.eigvalsh(gram)
    cutoff = max(float(np.max(eigenvalues)) * rcond, np.finfo(float).eps)
    positive = eigenvalues[eigenvalues > cutoff]
    rank = int(positive.size)
    condition_number = (
        float(np.max(positive) / np.min(positive)) if positive.size else np.inf
    )

    off_diagonal = gram[~np.eye(n_memories, dtype=bool)]
    projector = pseudoinverse @ gram
    diagnostics = {
        "gram_rank": rank,
        "gram_min_eigenvalue": float(np.min(eigenvalues)),
        "gram_max_eigenvalue": float(np.max(eigenvalues)),
        "gram_condition_number": condition_number,
        "pairwise_overlap_mean": float(np.mean(off_diagonal)),
        "pairwise_overlap_abs_mean": float(np.mean(np.abs(off_diagonal))),
        "pairwise_overlap_std": float(np.std(off_diagonal, ddof=1)),
        "pairwise_overlap_max": float(np.max(off_diagonal)),
        "gpi_identity_error": float(
            np.linalg.norm(projector - np.eye(n_memories), ord="fro")
            / math.sqrt(n_memories)
        ),
    }
    return gram, pseudoinverse, diagnostics


def make_successor_array(n_memories: int, cycle: Iterable[int]) -> np.ndarray:
    """Return successor[j]=i for the directed edge j -> i, else -1."""
    cycle = tuple(int(value) for value in cycle)
    if len(cycle) < 2:
        raise ValueError("The cycle must contain at least two memories.")
    if len(set(cycle)) != len(cycle):
        raise ValueError("Cycle entries must be distinct; closure is implicit.")
    if min(cycle) < 0 or max(cycle) >= n_memories:
        raise ValueError("Cycle contains an invalid memory index.")

    successor = np.full(n_memories, -1, dtype=np.int64)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        successor[source] = target
    return successor


# ---------------------------------------------------------------------------
# Activation-specific initialization
# ---------------------------------------------------------------------------

def tanh_branch_amplitude(z_value: float, gain_phi: float = 1.0) -> float:
    """Positive solution of gamma = z*tanh(gain_phi*gamma)."""
    z_value = float(z_value)
    if z_value * gain_phi <= 1.0:
        return 0.0
    function = lambda gamma: gamma - z_value * np.tanh(gain_phi * gamma)
    return float(brentq(function, 1e-12, z_value * (1.0 + 1e-10)))


def branch_amplitude(z_value: float, activation: str) -> float:
    if activation == "tanh":
        return tanh_branch_amplitude(z_value)
    if activation == "smoothf":
        if z_value <= 1.0:
            return 0.0
        return float(z_value - 1.0)
    raise ValueError(f"Unknown activation: {activation!r}")



# ---------------------------------------------------------------------------
# Activation-specific critical gain and asymptotic escape time
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CriticalPoint:
    """Saddle-node of the scalar saliency return map."""

    activation: str
    kappa_crit: float
    z_source_at_escape: float
    z_fixed: float
    T_esc_critical: float


def tanh_branch_u(
    z_value: np.ndarray | float,
) -> np.ndarray | float:
    """Return the positive solution of u=z*tanh(u), or zero below z=1."""
    z = np.asarray(np.abs(z_value), dtype=float)
    u = np.zeros_like(z)
    active = z > 1.0 + 1e-12

    if np.any(active):
        z_active = z[active]
        current = np.where(
            z_active < 1.5,
            np.sqrt(np.maximum(3.0 * (z_active - 1.0) / z_active, 1e-18)),
            z_active,
        )
        for _ in range(50):
            tanh_current = np.tanh(current)
            residual = current - z_active * tanh_current
            derivative = 1.0 - z_active * (1.0 - tanh_current**2)
            candidate = current - residual / derivative
            invalid = (~np.isfinite(candidate)) | (candidate <= 0.0)
            candidate[invalid] = 0.5 * current[invalid]
            if np.max(np.abs(candidate - current)) < 1e-13:
                current = candidate
                break
            current = candidate
        u[active] = current

    return float(u) if u.ndim == 0 else u


def tanh_stability_threshold(z_source: np.ndarray | float) -> np.ndarray | float:
    """Return h(z)=1/psi'(gamma(z)) for psi=tanh."""
    u = tanh_branch_u(z_source)
    return np.cosh(u) ** 2


def _tanh_fixed_point_gain(
    z_source_at_escape: float,
    integration_points: int = 4001,
) -> float:
    """Return kappa on the tanh fixed-point branch, parameterized by z_s."""
    source = float(z_source_at_escape)
    if source <= 1.0:
        return np.inf

    z_fixed = float(tanh_stability_threshold(source))
    X = z_fixed / source
    if not np.isfinite(X) or X <= 1.0:
        return np.inf

    r = np.linspace(1.0, X, int(integration_points))
    g_values = np.tanh(tanh_branch_u(z_fixed / r))
    convolution = trapezoid(g_values, r) / X
    if not np.isfinite(convolution) or convolution <= 0.0:
        return np.inf
    return float(z_fixed / convolution)


def tanh_critical_point(integration_points: int = 8001) -> CriticalPoint:
    """Compute the tanh saddle-node from the paper's implicit escape map."""
    lower = 1.0 + 1e-5
    upper = 5.0
    for _ in range(8):
        result = minimize_scalar(
            lambda source: _tanh_fixed_point_gain(
                source, integration_points=integration_points
            ),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success:
            raise RuntimeError("Could not determine the tanh critical gain.")
        if result.x < 0.85 * upper:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Tanh critical-gain search did not converge.")

    source = float(result.x)
    kappa_crit = float(
        _tanh_fixed_point_gain(
            source,
            integration_points=max(int(integration_points), 16001),
        )
    )
    z_fixed = float(tanh_stability_threshold(source))
    return CriticalPoint(
        activation="tanh",
        kappa_crit=kappa_crit,
        z_source_at_escape=source,
        z_fixed=z_fixed,
        T_esc_critical=float(np.log(z_fixed / source)),
    )


def _smoothf_fixed_point_G(X: float) -> float:
    """Return G(X^2,X) from the paper's SmoothF fixed-point equation."""
    X = float(X)
    if X <= 1.0:
        return 0.0
    return float((X - 1.0) ** 2 * (2.0 * X + 1.0) / (2.0 * X**5))


def smoothf_critical_point() -> CriticalPoint:
    """Return the explicit SmoothF saddle-node derived in the appendix."""
    X = float((5.0 + np.sqrt(105.0)) / 8.0)
    z_fixed = X**2
    return CriticalPoint(
        activation="smoothf",
        kappa_crit=float(1.0 / _smoothf_fixed_point_G(X)),
        z_source_at_escape=X,
        z_fixed=z_fixed,
        T_esc_critical=float(np.log(X)),
    )


def critical_point(
    activation: str,
    integration_points: int = 8001,
) -> CriticalPoint:
    if activation == "tanh":
        return tanh_critical_point(integration_points)
    if activation == "smoothf":
        return smoothf_critical_point()
    raise ValueError(f"Unknown activation: {activation!r}")


def asymptotic_escape_at_kappa(
    kappa: float,
    point: CriticalPoint,
    integration_points: int = 4001,
) -> dict[str, float]:
    """Return the attracting fixed point and T_esc for a supercritical kappa."""
    kappa = float(kappa)
    if kappa < point.kappa_crit * (1.0 - 1e-10):
        return {
            "kappa": kappa,
            "z_source_at_escape": np.nan,
            "z_fixed": np.nan,
            "T_esc": np.nan,
        }
    if np.isclose(kappa, point.kappa_crit, rtol=1e-10, atol=1e-12):
        return {
            "kappa": kappa,
            "z_source_at_escape": point.z_source_at_escape,
            "z_fixed": point.z_fixed,
            "T_esc": point.T_esc_critical,
        }

    if point.activation == "tanh":
        left = point.z_source_at_escape * (1.0 + 1e-9)
        right = max(left * 1.25, left + 0.1)

        def residual(source: float) -> float:
            return _tanh_fixed_point_gain(
                source, integration_points=integration_points
            ) - kappa

        for _ in range(60):
            value = residual(right)
            if np.isfinite(value) and value > 0.0:
                break
            right *= 1.35
        else:
            raise RuntimeError(
                f"Could not bracket the stable tanh branch for kappa={kappa}."
            )
        source = float(
            brentq(residual, left, right, xtol=1e-10, rtol=1e-10)
        )
        z_fixed = float(tanh_stability_threshold(source))
        return {
            "kappa": kappa,
            "z_source_at_escape": source,
            "z_fixed": z_fixed,
            "T_esc": float(np.log(z_fixed / source)),
        }

    X_left = point.z_source_at_escape * (1.0 + 1e-10)
    target = 1.0 / kappa
    X_right = max(2.0 * X_left, X_left + 1.0)
    while _smoothf_fixed_point_G(X_right) > target:
        X_right *= 1.5
        if X_right > 1e6:
            raise RuntimeError(
                f"Could not bracket the stable SmoothF branch for kappa={kappa}."
            )
    X = float(
        brentq(
            lambda value: _smoothf_fixed_point_G(value) - target,
            X_left,
            X_right,
            xtol=1e-12,
            rtol=1e-12,
        )
    )
    return {
        "kappa": kappa,
        "z_source_at_escape": X,
        "z_fixed": X**2,
        "T_esc": float(np.log(X)),
    }


def activation_value(value: float, activation: str) -> float:
    if activation == "tanh":
        return float(np.tanh(value))
    if activation == "smoothf":
        return float(value / (1.0 + abs(value)))
    raise ValueError(f"Unknown activation: {activation!r}")


def activation_derivative(value: float, activation: str) -> float:
    if activation == "tanh":
        output = np.tanh(value)
        return float(1.0 - output * output)
    if activation == "smoothf":
        return float(1.0 / (1.0 + abs(value)) ** 2)
    raise ValueError(f"Unknown activation: {activation!r}")


def gpi_bifurcation_diagnostics(
    patterns: np.ndarray,
    gram: np.ndarray,
    pseudoinverse: np.ndarray,
    gram_diagnostics: dict[str, float],
    cycle: tuple[int, ...],
    activation: str,
    point: CriticalPoint,
    reference_escape: dict[str, float],
) -> dict[str, Any]:
    """Verify the GPI reduction at the saddle-node for one pattern matrix."""
    n_neurons, n_memories = patterns.shape
    projector = pseudoinverse @ gram
    full_rank = int(gram_diagnostics["gram_rank"]) == n_memories

    branch_residuals: list[float] = []
    decoder_errors: list[float] = []
    stability_residuals: list[float] = []
    spectrum_errors: list[float] = []
    source_drive_errors: list[float] = []

    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        source = int(source)
        target = int(target)
        saliency = np.zeros(n_memories, dtype=float)
        saliency[source] = point.z_source_at_escape
        saliency[target] = point.z_fixed

        gamma = branch_amplitude(point.z_source_at_escape, activation)
        state = gamma * patterns[:, source]
        if activation == "tanh":
            activated = np.tanh(state)
        else:
            activated = state / (1.0 + np.abs(state))
        raw_overlap = patterns.T @ activated / float(n_neurons)
        decoded = pseudoinverse @ raw_overlap

        feedback = patterns @ (np.abs(saliency) * decoded)
        denominator = max(np.linalg.norm(state), 1e-15)
        branch_residuals.append(
            float(np.linalg.norm(-state + feedback) / denominator)
        )

        expected_decoded = np.zeros(n_memories, dtype=float)
        expected_decoded[source] = activation_value(gamma, activation)
        decoder_errors.append(
            float(
                np.linalg.norm(decoded - expected_decoded)
                / max(abs(expected_decoded[source]), 1e-15)
            )
        )
        source_drive_errors.append(
            float(
                abs(decoded[source] - expected_decoded[source])
                / max(abs(expected_decoded[source]), 1e-15)
            )
        )

        derivative = activation_derivative(gamma, activation)
        # Nonzero eigenvalues of the N-dimensional feedback Jacobian equal
        # those of C D(z) O^+, with C=Xi^T Dpsi Xi/N=psi'(gamma) O.
        reduced_feedback = derivative * gram @ np.diag(np.abs(saliency)) @ pseudoinverse
        feedback_eigenvalues = np.linalg.eigvals(reduced_feedback)
        jacobian_eigenvalues = feedback_eigenvalues - 1.0
        stability_residuals.append(
            float(np.max(np.real(jacobian_eigenvalues)))
        )

        observed = np.sort(np.real(feedback_eigenvalues))
        expected = np.sort(derivative * np.abs(saliency))
        spectrum_errors.append(float(np.max(np.abs(observed - expected))))

    valid = bool(
        full_rank
        and np.linalg.norm(projector - np.eye(n_memories), ord="fro")
        / math.sqrt(n_memories)
        < 1e-7
    )
    return {
        "gpi_theory_valid": int(valid),
        "kappa_crit_gpi": point.kappa_crit if valid else np.nan,
        "T_esc_at_kappa_crit_gpi": (
            point.T_esc_critical if valid else np.nan
        ),
        "T_esc_at_reference_kappa_gpi": (
            reference_escape["T_esc"] if valid else np.nan
        ),
        "z_fixed_at_critical_gpi": point.z_fixed if valid else np.nan,
        "z_source_at_escape_critical_gpi": (
            point.z_source_at_escape if valid else np.nan
        ),
        "z_fixed_at_reference_kappa_gpi": (
            reference_escape["z_fixed"] if valid else np.nan
        ),
        "max_relative_branch_residual": float(max(branch_residuals)),
        "max_relative_decoder_error": float(max(decoder_errors)),
        "max_relative_source_drive_error": float(max(source_drive_errors)),
        "max_abs_stability_boundary_residual": float(
            max(abs(value) for value in stability_residuals)
        ),
        "max_feedback_spectrum_error": float(max(spectrum_errors)),
    }


def run_bifurcation_sweep(
    config: SweepConfig,
    reference_kappa: float,
    integration_points: int = 8001,
) -> tuple[pd.DataFrame, pd.DataFrame, CriticalPoint, dict[str, float]]:
    """Evaluate kappa_crit and T_esc over the controlled-correlation grid."""
    point = critical_point(config.activation, integration_points)
    reference_escape = asymptotic_escape_at_kappa(
        reference_kappa,
        point,
        integration_points=max(2001, integration_points // 2),
    )
    if not np.isfinite(reference_escape["T_esc"]):
        raise ValueError(
            "reference_kappa must be greater than or equal to kappa_crit."
        )

    seed_sequence = np.random.SeedSequence(config.seed)
    correlation_children = seed_sequence.spawn(len(config.correlation_values))
    rows: list[dict[str, Any]] = []

    for correlation_index, (parameter, correlation_seed) in enumerate(
        zip(config.correlation_values, correlation_children)
    ):
        pattern_children = correlation_seed.spawn(config.pattern_sets)
        for pattern_set, child in enumerate(pattern_children):
            rng = np.random.default_rng(child)
            patterns = generate_correlated_patterns(
                rng,
                config.n_neurons,
                config.n_memories,
                parameter,
                config.correlation_model,
            ).astype(np.float64)
            gram, pseudoinverse, diagnostics = gram_and_pseudoinverse(
                patterns, config.pinv_rcond
            )
            bifurcation = gpi_bifurcation_diagnostics(
                patterns=patterns,
                gram=gram,
                pseudoinverse=pseudoinverse,
                gram_diagnostics=diagnostics,
                cycle=config.cycle,
                activation=config.activation,
                point=point,
                reference_escape=reference_escape,
            )
            row: dict[str, Any] = {
                "correlation_model": config.correlation_model,
                "correlation_parameter": float(parameter),
                "expected_pairwise_overlap": (
                    float(parameter**2)
                    if config.correlation_model == "biased"
                    else float(parameter)
                ),
                "correlation_index": correlation_index,
                "pattern_set": pattern_set,
                "activation": config.activation,
                "rule": "gpi",
                "N": config.n_neurons,
                "P": config.n_memories,
                "reference_kappa": float(reference_kappa),
            }
            row.update(diagnostics)
            row.update(bifurcation)
            rows.append(row)

    trials = pd.DataFrame.from_records(rows)
    summary_rows: list[dict[str, Any]] = []
    for parameter, group in trials.groupby("correlation_parameter", sort=True):
        valid = group[group["gpi_theory_valid"] == 1]
        summary_rows.append({
            "correlation_model": config.correlation_model,
            "correlation_parameter": float(parameter),
            "expected_pairwise_overlap": float(
                group["expected_pairwise_overlap"].iloc[0]
            ),
            "activation": config.activation,
            "N": config.n_neurons,
            "P": config.n_memories,
            "n_pattern_sets": int(len(group)),
            "full_rank_fraction": float(
                np.mean(group["gram_rank"] == config.n_memories)
            ),
            "gpi_theory_valid_fraction": float(
                group["gpi_theory_valid"].mean()
            ),
            "kappa_crit_mean": float(valid["kappa_crit_gpi"].mean()),
            "kappa_crit_std": float(valid["kappa_crit_gpi"].std(ddof=1)),
            "kappa_crit_min": float(valid["kappa_crit_gpi"].min()),
            "kappa_crit_max": float(valid["kappa_crit_gpi"].max()),
            "T_esc_critical_mean": float(
                valid["T_esc_at_kappa_crit_gpi"].mean()
            ),
            "T_esc_critical_std": float(
                valid["T_esc_at_kappa_crit_gpi"].std(ddof=1)
            ),
            "T_esc_reference_mean": float(
                valid["T_esc_at_reference_kappa_gpi"].mean()
            ),
            "T_esc_reference_std": float(
                valid["T_esc_at_reference_kappa_gpi"].std(ddof=1)
            ),
            "reference_kappa": float(reference_kappa),
            "mean_empirical_pairwise_overlap": float(
                group["pairwise_overlap_mean"].mean()
            ),
            "mean_abs_pairwise_overlap": float(
                group["pairwise_overlap_abs_mean"].mean()
            ),
            "mean_gram_condition_number": float(
                group["gram_condition_number"].mean()
            ),
            "median_gram_condition_number": float(
                group["gram_condition_number"].median()
            ),
            "max_gpi_identity_error": float(
                group["gpi_identity_error"].max()
            ),
            "max_relative_branch_residual": float(
                group["max_relative_branch_residual"].max()
            ),
            "max_relative_decoder_error": float(
                group["max_relative_decoder_error"].max()
            ),
            "max_abs_stability_boundary_residual": float(
                group["max_abs_stability_boundary_residual"].max()
            ),
            "max_feedback_spectrum_error": float(
                group["max_feedback_spectrum_error"].max()
            ),
        })

    summary = pd.DataFrame.from_records(summary_rows).sort_values(
        "correlation_parameter"
    )
    return trials, summary, point, reference_escape


def plot_bifurcation_sweep(
    summary: pd.DataFrame,
    point: CriticalPoint,
    reference_escape: dict[str, float],
    output_prefix: Path,
) -> None:
    """Plot critical gain and asymptotic escape time versus correlation."""
    x = summary["correlation_parameter"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    axes[0].plot(
        x,
        summary["kappa_crit_mean"],
        marker="o",
        linewidth=2.0,
        label="GPI implicit result",
    )
    axes[0].axhline(
        point.kappa_crit,
        linestyle="--",
        linewidth=1.3,
        label="Orthogonal-limit theory",
    )
    axes[0].set_ylabel(r"Critical gain $\kappa_{\mathrm{crit}}$")
    axes[0].set_title("Saddle-node threshold")

    axes[1].plot(
        x,
        summary["T_esc_critical_mean"],
        marker="o",
        linewidth=2.0,
        label=r"$T_{\rm esc}$ at $\kappa_{\rm crit}$",
    )
    axes[1].plot(
        x,
        summary["T_esc_reference_mean"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        label=rf"$T_{{\rm esc}}^\infty$ at $\kappa={reference_escape['kappa']:.3g}$",
    )
    axes[1].set_ylabel(r"Dimensionless escape time $T_{\mathrm{esc}}$")
    axes[1].set_title("Asymptotic saliency-map escape time")

    correlation_model = str(summary["correlation_model"].iloc[0])
    if correlation_model == "biased":
        xlabel = r"Bias parameter $\epsilon$"
    else:
        xlabel = r"Target pairwise overlap $\rho$"
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)

    if correlation_model == "biased":
        for axis in axes:
            top = axis.secondary_xaxis(
                "top",
                functions=(
                    lambda value: np.asarray(value) ** 2,
                    lambda value: np.sqrt(np.maximum(np.asarray(value), 0.0)),
                ),
            )
            top.set_xlabel(r"Expected pairwise overlap $\epsilon^2$")

    fig.suptitle(
        rf"Full-rank GPI theory, $N={int(summary['N'].iloc[0])}$, "
        rf"$P={int(summary['P'].iloc[0])}$, {point.activation} activation"
    )
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    axes[0].plot(
        x,
        summary["mean_gram_condition_number"],
        marker="o",
        linewidth=2.0,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Mean condition number of $O$")
    axes[0].set_title("Correlation changes conditioning")

    for column, label in [
        ("max_gpi_identity_error", r"$\|O^+O-I\|_F/\sqrt{P}$"),
        ("max_relative_branch_residual", "relative branch residual"),
        (
            "max_abs_stability_boundary_residual",
            "stability-boundary residual",
        ),
        ("max_feedback_spectrum_error", "feedback-spectrum error"),
    ]:
        axes[1].plot(
            x,
            np.maximum(summary[column].to_numpy(dtype=float), 1e-18),
            marker="o",
            linewidth=1.5,
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Maximum numerical residual")
    axes[1].set_title("GPI reduction identities")
    axes[1].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    diagnostic_prefix = output_prefix.with_name(
        output_prefix.name + "_conditioning"
    )
    fig.savefig(diagnostic_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(diagnostic_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numba-accelerated dynamics
# ---------------------------------------------------------------------------

@njit(cache=True)
def _activation_scalar(value: float, activation_code: int) -> float:
    if activation_code == 0:
        return math.tanh(value)
    # SmoothF: x/(1+|x|)
    return value / (1.0 + abs(value))


@njit(cache=True)
def _simulate_trial(
    patterns: np.ndarray,
    pseudoinverse: np.ndarray,
    successor: np.ndarray,
    source_initial: int,
    required_transitions: int,
    rule_code: int,
    activation_code: int,
    gamma_initial: float,
    z_initial: float,
    kappa: float,
    dt_fast: float,
    dt_slow: float,
    max_slow_time_per_transition: float,
    min_alignment: float,
    min_margin: float,
    hold_checks: int,
    wrong_hold_checks: int,
    collapse_hold_checks: int,
    check_stride: int,
    coefficient_noise: float,
    random_seed: int,
) -> tuple:
    """Simulate one full cycle and return compact trial metrics."""
    np.random.seed(random_seed)
    n_neurons, n_memories = patterns.shape

    coefficients = np.zeros(n_memories)
    saliency = np.zeros(n_memories)
    coefficients[source_initial] = gamma_initial
    saliency[source_initial] = z_initial

    # A tiny isotropic perturbation breaks invariant coordinate subspaces.
    for memory in range(n_memories):
        coefficients[memory] += coefficient_noise * np.random.randn()

    y = np.zeros(n_neurons)
    phi = np.zeros(n_neurons)
    raw_overlap = np.zeros(n_memories)
    decoded_overlap = np.zeros(n_memories)
    graph_drive = np.zeros(n_memories)
    alignment = np.zeros(n_memories)

    current_source = source_initial
    completed = 0
    target_hold = 0
    wrong_hold = 0
    collapse_hold = 0
    elapsed_edge_time = 0.0
    total_slow_time = 0.0

    sum_switch_time = 0.0
    sum_switch_time_squared = 0.0
    min_winner_alignment = 1e100
    min_winner_margin = 1e100
    max_active_saliences = 1
    max_abs_decoded_distractor = 0.0

    max_steps = int(
        math.ceil(
            required_transitions
            * max_slow_time_per_transition
            / dt_slow
        )
    ) + check_stride + 1

    for step in range(max_steps):
        # y = Xi a
        for neuron in range(n_neurons):
            value = 0.0
            for memory in range(n_memories):
                value += patterns[neuron, memory] * coefficients[memory]
            y[neuron] = value
            phi[neuron] = _activation_scalar(value, activation_code)

        # m = Xi^T psi(y)/N
        for memory in range(n_memories):
            value = 0.0
            for neuron in range(n_neurons):
                value += patterns[neuron, memory] * phi[neuron]
            raw_overlap[memory] = value / n_neurons

        # q = O^+ m for GPI; q = m for the original rule.
        if rule_code == 1:
            for row in range(n_memories):
                value = 0.0
                for column in range(n_memories):
                    value += pseudoinverse[row, column] * raw_overlap[column]
                decoded_overlap[row] = value
        else:
            for memory in range(n_memories):
                decoded_overlap[memory] = raw_overlap[memory]

        # Track leakage into memories other than the source and its intended successor.
        intended_successor = successor[current_source]
        for memory in range(n_memories):
            if memory != current_source and memory != intended_successor:
                value = abs(decoded_overlap[memory])
                if value > max_abs_decoded_distractor:
                    max_abs_decoded_distractor = value

        # Graph drive A q, with A[target, source]=1.
        for memory in range(n_memories):
            graph_drive[memory] = 0.0
        for source in range(n_memories):
            target = successor[source]
            if target >= 0:
                graph_drive[target] += decoded_overlap[source]

        # Explicit Euler on the fast dimensionless clock.
        for memory in range(n_memories):
            fast_rhs = (
                -coefficients[memory]
                + abs(saliency[memory]) * decoded_overlap[memory]
            )
            coefficients[memory] += dt_fast * fast_rhs
            coefficients[memory] += coefficient_noise * np.random.randn()

            slow_rhs = -saliency[memory] + kappa * graph_drive[memory]
            saliency[memory] += dt_slow * slow_rhs

            if not (
                math.isfinite(coefficients[memory])
                and math.isfinite(saliency[memory])
            ):
                return (
                    4,
                    completed,
                    total_slow_time,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    max_active_saliences,
                    max_abs_decoded_distractor,
                )

        elapsed_edge_time += dt_slow
        total_slow_time += dt_slow

        if (step + 1) % check_stride != 0:
            continue

        # Signed cosine alignment with every stored pattern.
        y_norm_squared = 0.0
        for neuron in range(n_neurons):
            y_norm_squared += y[neuron] * y[neuron]
        y_norm = math.sqrt(y_norm_squared)

        if y_norm < 1e-12:
            collapse_hold += 1
            if collapse_hold >= collapse_hold_checks:
                mean_switch = (
                    sum_switch_time / completed if completed > 0 else math.nan
                )
                return (
                    3,
                    completed,
                    total_slow_time,
                    mean_switch,
                    math.nan,
                    (min_winner_alignment if completed > 0 else math.nan),
                    (min_winner_margin if completed > 0 else math.nan),
                    max_active_saliences,
                    max_abs_decoded_distractor,
                )
            continue
        collapse_hold = 0

        best_index = -1
        best_value = -1e100
        second_value = -1e100
        normalization = math.sqrt(n_neurons) * y_norm
        for memory in range(n_memories):
            dot = 0.0
            for neuron in range(n_neurons):
                dot += patterns[neuron, memory] * y[neuron]
            # The supplied code measures absolute overlap, identifying +/-xi^mu.
            value = abs(dot) / normalization
            alignment[memory] = value
            if value > best_value:
                second_value = best_value
                best_value = value
                best_index = memory
            elif value > second_value:
                second_value = value

        winner_margin = best_value - second_value
        expected_target = successor[current_source]
        pure = best_value >= min_alignment and winner_margin >= min_margin

        # Diagnostic: how many saliencies are above the memory-existence level.
        active_saliences = 0
        for memory in range(n_memories):
            if abs(saliency[memory]) > 1.0:
                active_saliences += 1
        if active_saliences > max_active_saliences:
            max_active_saliences = active_saliences

        if pure and best_index == expected_target:
            target_hold += 1
        else:
            target_hold = 0

        if pure and best_index != current_source and best_index != expected_target:
            wrong_hold += 1
        else:
            wrong_hold = 0

        if wrong_hold >= wrong_hold_checks:
            mean_switch = (
                sum_switch_time / completed if completed > 0 else math.nan
            )
            switch_std = math.nan
            if completed > 1:
                variance = (
                    sum_switch_time_squared / completed
                    - mean_switch * mean_switch
                )
                switch_std = math.sqrt(max(variance, 0.0))
            return (
                2,
                completed,
                total_slow_time,
                mean_switch,
                switch_std,
                (min_winner_alignment if completed > 0 else math.nan),
                (min_winner_margin if completed > 0 else math.nan),
                max_active_saliences,
                max_abs_decoded_distractor,
            )

        if target_hold >= hold_checks:
            # Use the beginning of the persistent hold interval as switch time.
            switch_time = elapsed_edge_time - (hold_checks - 1) * (
                check_stride * dt_slow
            )
            completed += 1
            sum_switch_time += switch_time
            sum_switch_time_squared += switch_time * switch_time
            if best_value < min_winner_alignment:
                min_winner_alignment = best_value
            if winner_margin < min_winner_margin:
                min_winner_margin = winner_margin

            current_source = expected_target
            elapsed_edge_time = 0.0
            target_hold = 0
            wrong_hold = 0

            if completed >= required_transitions:
                mean_switch = sum_switch_time / completed
                switch_std = 0.0
                if completed > 1:
                    variance = (
                        sum_switch_time_squared / completed
                        - mean_switch * mean_switch
                    )
                    switch_std = math.sqrt(max(variance, 0.0))
                return (
                    0,
                    completed,
                    total_slow_time,
                    mean_switch,
                    switch_std,
                    min_winner_alignment,
                    min_winner_margin,
                    max_active_saliences,
                    max_abs_decoded_distractor,
                )

        if elapsed_edge_time >= max_slow_time_per_transition:
            mean_switch = (
                sum_switch_time / completed if completed > 0 else math.nan
            )
            switch_std = math.nan
            if completed > 1:
                variance = (
                    sum_switch_time_squared / completed
                    - mean_switch * mean_switch
                )
                switch_std = math.sqrt(max(variance, 0.0))
            return (
                1,
                completed,
                total_slow_time,
                mean_switch,
                switch_std,
                (min_winner_alignment if completed > 0 else math.nan),
                (min_winner_margin if completed > 0 else math.nan),
                max_active_saliences,
                max_abs_decoded_distractor,
            )

    mean_switch = sum_switch_time / completed if completed > 0 else math.nan
    return (
        1,
        completed,
        total_slow_time,
        mean_switch,
        math.nan,
        (min_winner_alignment if completed > 0 else math.nan),
        (min_winner_margin if completed > 0 else math.nan),
        max_active_saliences,
        max_abs_decoded_distractor,
    )


# ---------------------------------------------------------------------------
# Sweep, statistics, and plotting
# ---------------------------------------------------------------------------

def _wilson_interval(successes: int, trials: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1.0 + z_value * z_value / trials
    center = (proportion + z_value * z_value / (2.0 * trials)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value * z_value / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(reps, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def run_sweep(config: SweepConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    successor = make_successor_array(config.n_memories, config.cycle)
    source_initial = int(config.cycle[-1])
    required_transitions = len(config.cycle)
    gamma_initial = branch_amplitude(config.z_initial, config.activation)

    activation_code = ACTIVATION_NAMES.index(config.activation)
    check_stride = max(1, int(round(config.check_interval / config.dt_slow)))
    check_dt = check_stride * config.dt_slow
    hold_checks = max(1, int(math.ceil(config.hold_time / check_dt)))
    wrong_hold_checks = max(
        1, int(math.ceil(config.wrong_hold_time / check_dt))
    )
    collapse_hold_checks = max(
        1, int(math.ceil(config.collapse_hold_time / check_dt))
    )
    # Projection of iid neuron-level noise onto a normalized random memory.
    coefficient_noise = config.fast_noise_per_neuron / math.sqrt(config.n_neurons)

    seed_sequence = np.random.SeedSequence(config.seed)
    correlation_children = seed_sequence.spawn(len(config.correlation_values))

    rows: list[dict[str, Any]] = []
    for correlation_index, (parameter, correlation_seed) in enumerate(
        zip(config.correlation_values, correlation_children)
    ):
        pattern_children = correlation_seed.spawn(config.pattern_sets)
        for pattern_set, child in enumerate(pattern_children):
            rng = np.random.default_rng(child)
            patterns = generate_correlated_patterns(
                rng,
                config.n_neurons,
                config.n_memories,
                parameter,
                config.correlation_model,
            ).astype(np.float64)
            gram, pseudoinverse, diagnostics = gram_and_pseudoinverse(
                patterns, config.pinv_rcond
            )

            # A full-rank Gram matrix is the exact-recall condition for GPI.
            full_rank = diagnostics["gram_rank"] == config.n_memories

            # Same stochastic perturbation seed for paired raw/GPI runs.
            trial_seed = int(rng.integers(1, 2**31 - 1))
            for rule in config.rules:
                rule_code = RULE_NAMES.index(rule)
                result = _simulate_trial(
                    patterns,
                    pseudoinverse,
                    successor,
                    source_initial,
                    required_transitions,
                    rule_code,
                    activation_code,
                    gamma_initial,
                    config.z_initial,
                    config.kappa,
                    config.dt_fast,
                    config.dt_slow,
                    config.max_slow_time_per_transition,
                    config.min_alignment,
                    config.min_margin,
                    hold_checks,
                    wrong_hold_checks,
                    collapse_hold_checks,
                    check_stride,
                    coefficient_noise,
                    trial_seed,
                )

                (
                    outcome_code,
                    completed,
                    total_slow_time,
                    mean_switch_time,
                    switch_time_std,
                    min_winner_alignment,
                    min_winner_margin,
                    max_active_saliences,
                    max_abs_decoded_distractor,
                ) = result

                row: dict[str, Any] = {
                    "correlation_model": config.correlation_model,
                    "correlation_parameter": float(parameter),
                    "expected_pairwise_overlap": (
                        float(parameter**2)
                        if config.correlation_model == "biased"
                        else float(parameter)
                    ),
                    "correlation_index": correlation_index,
                    "pattern_set": pattern_set,
                    "rule": rule,
                    "activation": config.activation,
                    "N": config.n_neurons,
                    "P": config.n_memories,
                    "cycle_length": required_transitions,
                    "outcome": OUTCOME_NAMES[int(outcome_code)],
                    "success": int(outcome_code == 0),
                    "transitions_completed": int(completed),
                    "transition_fraction": float(completed / required_transitions),
                    "total_slow_time": float(total_slow_time),
                    "mean_switch_time": float(mean_switch_time),
                    "switch_time_std_within_cycle": float(switch_time_std),
                    "min_winner_alignment": float(min_winner_alignment),
                    "min_winner_margin": float(min_winner_margin),
                    "max_active_saliences": int(max_active_saliences),
                    "max_abs_decoded_distractor": float(
                        max_abs_decoded_distractor
                    ),
                    "gram_full_rank": int(full_rank),
                    "seed": trial_seed,
                }
                row.update(diagnostics)
                rows.append(row)

    trials = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "correlation_model",
        "correlation_parameter",
        "expected_pairwise_overlap",
        "rule",
        "activation",
        "N",
        "P",
        "cycle_length",
    ]
    for keys, group in trials.groupby(group_columns, sort=True):
        info = dict(zip(group_columns, keys))
        success_values = group["success"].to_numpy(dtype=float)
        transition_values = group["transition_fraction"].to_numpy(dtype=float)
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    config.seed,
                    int(round(float(info["correlation_parameter"]) * 10000)),
                    RULE_NAMES.index(str(info["rule"])),
                    991,
                ]
            )
        )
        success_count = int(np.sum(success_values))
        success_low, success_high = _wilson_interval(
            success_count, int(success_values.size)
        )
        transition_low, transition_high = _bootstrap_mean_interval(
            transition_values, bootstrap_rng, config.bootstrap_reps
        )

        successful = group[group["success"] == 1]
        summary = {
            **info,
            "n_pattern_sets": int(len(group)),
            "cycle_successes": success_count,
            "cycle_success_probability": float(np.mean(success_values)),
            "cycle_success_ci_low": success_low,
            "cycle_success_ci_high": success_high,
            "cycle_success_ci_method": "Wilson score, 95%",
            "mean_transition_fraction": float(np.mean(transition_values)),
            "transition_fraction_ci_low": transition_low,
            "transition_fraction_ci_high": transition_high,
            "mean_switch_time_successes": float(
                successful["mean_switch_time"].mean()
            ),
            "mean_min_winner_alignment_successes": float(
                successful["min_winner_alignment"].mean()
            ),
            "mean_min_winner_margin_successes": float(
                successful["min_winner_margin"].mean()
            ),
            "mean_max_active_saliences": float(
                group["max_active_saliences"].mean()
            ),
            "mean_max_abs_decoded_distractor": float(
                group["max_abs_decoded_distractor"].mean()
            ),
            "mean_empirical_pairwise_overlap": float(
                group["pairwise_overlap_mean"].mean()
            ),
            "mean_abs_pairwise_overlap": float(
                group["pairwise_overlap_abs_mean"].mean()
            ),
            "mean_gram_condition_number": float(
                group["gram_condition_number"].mean()
            ),
            "max_gpi_identity_error": float(
                group["gpi_identity_error"].max()
            ),
            "full_rank_fraction": float(group["gram_full_rank"].mean()),
            "timeout_probability": float(np.mean(group["outcome"] == "timeout")),
            "wrong_memory_probability": float(
                np.mean(group["outcome"] == "wrong_memory")
            ),
            "collapse_probability": float(
                np.mean(group["outcome"] == "collapse")
            ),
        }
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["rule", "correlation_parameter"]
    )
    return trials, summary



def simulate_example_history(
    patterns: np.ndarray,
    pseudoinverse: np.ndarray,
    successor: np.ndarray,
    cycle: tuple[int, ...],
    rule: str,
    activation: str,
    config: SweepConfig,
    slow_horizon: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a downsampled trajectory for qualitative inspection."""
    rng = np.random.default_rng(seed)
    n_neurons, n_memories = patterns.shape
    source = cycle[-1]
    coefficients = np.zeros(n_memories)
    saliency = np.zeros(n_memories)
    coefficients[source] = branch_amplitude(config.z_initial, activation)
    saliency[source] = config.z_initial

    coefficient_noise = config.fast_noise_per_neuron / math.sqrt(n_neurons)
    coefficients += coefficient_noise * rng.standard_normal(n_memories)
    stride = max(1, int(round(0.01 / config.dt_slow)))
    n_steps = int(math.ceil(slow_horizon / config.dt_slow))

    time_values: list[float] = []
    alignment_values: list[np.ndarray] = []
    saliency_values: list[np.ndarray] = []

    for step in range(n_steps):
        state = patterns @ coefficients
        if activation == "tanh":
            activated = np.tanh(state)
        else:
            activated = state / (1.0 + np.abs(state))
        raw = patterns.T @ activated / n_neurons
        decoded = pseudoinverse @ raw if rule == "gpi" else raw

        drive = np.zeros(n_memories)
        valid_sources = np.flatnonzero(successor >= 0)
        drive[successor[valid_sources]] += decoded[valid_sources]

        coefficients += config.dt_fast * (
            -coefficients + np.abs(saliency) * decoded
        )
        coefficients += coefficient_noise * rng.standard_normal(n_memories)
        saliency += config.dt_slow * (-saliency + config.kappa * drive)

        if (step + 1) % stride == 0:
            state = patterns @ coefficients
            denominator = math.sqrt(n_neurons) * np.linalg.norm(state)
            if denominator > 0:
                alignment = np.abs(patterns.T @ state) / denominator
            else:
                alignment = np.zeros(n_memories)
            time_values.append((step + 1) * config.dt_slow)
            alignment_values.append(alignment[np.array(cycle)])
            saliency_values.append(np.abs(saliency[np.array(cycle)]))

    return (
        np.asarray(time_values),
        np.asarray(alignment_values),
        np.asarray(saliency_values),
    )


def plot_example_trajectories(config: SweepConfig, output_directory: Path) -> None:
    """Plot paired raw/GPI trajectories at a representative correlation."""
    parameter = config.correlation_values[
        int(np.argmin(np.abs(np.asarray(config.correlation_values) - 0.6)))
    ]
    rng = np.random.default_rng(config.seed + 4049)
    patterns = generate_correlated_patterns(
        rng,
        config.n_neurons,
        config.n_memories,
        parameter,
        config.correlation_model,
    )
    _, pseudoinverse, _ = gram_and_pseudoinverse(patterns, config.pinv_rcond)
    successor = make_successor_array(config.n_memories, config.cycle)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True)
    for column, rule in enumerate(("raw", "gpi")):
        time_values, alignments, saliencies = simulate_example_history(
            patterns,
            pseudoinverse,
            successor,
            config.cycle,
            rule,
            config.activation,
            config,
            slow_horizon=12.0,
            seed=config.seed + 9127,
        )
        for index, memory in enumerate(config.cycle):
            axes[0, column].plot(
                time_values,
                alignments[:, index],
                linewidth=1.8,
                label=rf"$\xi^{{{memory}}}$",
            )
            axes[1, column].plot(
                time_values,
                saliencies[:, index],
                linewidth=1.8,
            )
        axes[0, column].set_title(f"{rule.upper()} rule")
        axes[0, column].set_ylim(-0.03, 1.05)
        axes[0, column].grid(True, alpha=0.3)
        axes[1, column].grid(True, alpha=0.3)
        axes[1, column].set_xlabel("Dimensionless slow time")

    axes[0, 0].set_ylabel("Absolute cosine alignment")
    axes[1, 0].set_ylabel(r"Saliency $|z_\mu|$")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=10)
    if config.correlation_model == "biased":
        description = rf"$\epsilon={parameter:.2f}$ (expected overlap $\epsilon^2={parameter**2:.2f}$)"
    else:
        description = rf"$\rho={parameter:.2f}$"
    fig.suptitle(f"Representative correlated-pattern trajectory: {description}")
    fig.tight_layout()
    prefix = output_directory / "correlated_retrieval_example"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_results(summary: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True)

    for rule, group in summary.groupby("rule", sort=False):
        group = group.sort_values("correlation_parameter")
        x = group["correlation_parameter"].to_numpy()
        axes[0, 0].plot(
            x,
            group["cycle_success_probability"],
            marker="o",
            linewidth=2,
            label=rule.upper(),
        )
        axes[0, 0].fill_between(
            x,
            group["cycle_success_ci_low"],
            group["cycle_success_ci_high"],
            alpha=0.18,
        )

        axes[0, 1].plot(
            x,
            group["mean_transition_fraction"],
            marker="o",
            linewidth=2,
            label=rule.upper(),
        )
        axes[0, 1].fill_between(
            x,
            group["transition_fraction_ci_low"],
            group["transition_fraction_ci_high"],
            alpha=0.18,
        )

        axes[1, 0].plot(
            x,
            group["mean_max_active_saliences"],
            marker="o",
            linewidth=2,
            label=rule.upper(),
        )
        axes[1, 1].plot(
            x,
            group["mean_max_abs_decoded_distractor"],
            marker="o",
            linewidth=2,
            label=rule.upper(),
        )

    axes[0, 0].set_ylabel("Full-cycle success probability")
    axes[0, 0].set_ylim(-0.04, 1.04)
    axes[0, 0].set_title("Retrieval of all graph-consistent transitions")

    axes[0, 1].set_ylabel("Mean completed fraction of cycle")
    axes[0, 1].set_ylim(-0.04, 1.04)
    axes[0, 1].set_title("Partial retrieval before failure")

    axes[1, 0].set_ylabel("Maximum active saliencies")
    axes[1, 0].set_title("Simultaneous memory competition")

    axes[1, 1].set_ylabel("Max. off-source decoded overlap")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Memory-coordinate crosstalk")

    correlation_model = str(summary["correlation_model"].iloc[0])
    if correlation_model == "biased":
        xlabel = r"Pattern-bias parameter $\epsilon$  (expected overlap $\epsilon^2$)"
    else:
        xlabel = r"Target pairwise overlap $\rho$"
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 1].set_xlabel(xlabel)

    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Conditioning and empirical-correlation diagnostics.
    diagnostic = (
        summary.groupby("correlation_parameter", as_index=False)
        .agg(
            mean_empirical_pairwise_overlap=(
                "mean_empirical_pairwise_overlap",
                "first",
            ),
            mean_abs_pairwise_overlap=("mean_abs_pairwise_overlap", "first"),
            mean_gram_condition_number=(
                "mean_gram_condition_number",
                "first",
            ),
            max_gpi_identity_error=("max_gpi_identity_error", "first"),
        )
        .sort_values("correlation_parameter")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(
        diagnostic["correlation_parameter"],
        diagnostic["mean_empirical_pairwise_overlap"],
        marker="o",
        linewidth=2,
        label="Signed mean",
    )
    axes[0].plot(
        diagnostic["correlation_parameter"],
        diagnostic["mean_abs_pairwise_overlap"],
        marker="s",
        linewidth=2,
        label="Absolute mean",
    )
    axes[0].set_ylabel("Empirical off-diagonal overlap")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        diagnostic["correlation_parameter"],
        diagnostic["mean_gram_condition_number"],
        marker="o",
        linewidth=2,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"Condition number of $O$")
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    diagnostic_prefix = output_prefix.parent / f"{output_prefix.name}_conditioning"
    fig.savefig(diagnostic_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(diagnostic_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("correlated_gpi_results"))
    parser.add_argument("--correlation-model", choices=("biased", "prototype"), default="biased")
    parser.add_argument(
        "--correlations",
        type=float,
        nargs="+",
        default=list(np.linspace(0.0, 0.7, 8)),
    )
    parser.add_argument("--pattern-sets", type=int, default=24)
    parser.add_argument("--rules", choices=RULE_NAMES, nargs="+", default=list(RULE_NAMES))
    parser.add_argument("--activation", choices=ACTIVATION_NAMES, default="tanh")
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--n-memories", type=int, default=15)
    parser.add_argument("--cycle", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--kappa", type=float, default=15.0)
    parser.add_argument("--z-initial", type=float, default=8.0)
    parser.add_argument("--dt-fast", type=float, default=0.05)
    parser.add_argument("--tau-fast", type=float, default=0.01)
    parser.add_argument("--tau-slow", type=float, default=1.0)
    parser.add_argument("--max-slow-time-per-transition", type=float, default=4.0)
    parser.add_argument("--fast-noise", type=float, default=1e-6)
    parser.add_argument("--bootstrap-reps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--reference-kappa",
        type=float,
        default=15.0,
        help="Supercritical gain used for the second asymptotic T_esc curve.",
    )
    parser.add_argument(
        "--bifurcation-integration-points",
        type=int,
        default=8001,
        help="Quadrature resolution for the implicit tanh return map.",
    )
    parser.add_argument(
        "--bifurcation-only",
        action="store_true",
        help="Skip trajectory retrieval and run only the critical-gain sweep.",
    )
    parser.add_argument(
        "--skip-bifurcation",
        action="store_true",
        help="Run only the original retrieval sweep.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    correlations = tuple(float(value) for value in args.correlations)
    pattern_sets = int(args.pattern_sets)
    bootstrap_reps = int(args.bootstrap_reps)
    integration_points = int(args.bifurcation_integration_points)
    if args.smoke:
        correlations = (0.0, 0.4, 0.7)
        pattern_sets = min(pattern_sets, 4)
        bootstrap_reps = min(bootstrap_reps, 500)
        integration_points = min(integration_points, 3001)

    config = SweepConfig(
        n_neurons=args.n_neurons,
        n_memories=args.n_memories,
        cycle=tuple(args.cycle),
        correlation_model=args.correlation_model,
        correlation_values=correlations,
        pattern_sets=pattern_sets,
        rules=tuple(args.rules),
        activation=args.activation,
        kappa=args.kappa,
        z_initial=args.z_initial,
        tau_fast=args.tau_fast,
        tau_slow=args.tau_slow,
        dt_fast=args.dt_fast,
        max_slow_time_per_transition=args.max_slow_time_per_transition,
        fast_noise_per_neuron=args.fast_noise,
        bootstrap_reps=bootstrap_reps,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.bifurcation_only:
        trials, summary = run_sweep(config)
        trials_path = args.output / "correlated_retrieval_trials.csv"
        summary_path = args.output / "correlated_retrieval_summary.csv"
        trials.to_csv(trials_path, index=False)
        summary.to_csv(summary_path, index=False)
        plot_results(summary, args.output / "correlated_retrieval")
        plot_example_trajectories(config, args.output)

        display_columns = [
            "correlation_parameter",
            "expected_pairwise_overlap",
            "rule",
            "cycle_success_probability",
            "mean_transition_fraction",
            "mean_max_active_saliences",
            "mean_max_abs_decoded_distractor",
            "mean_gram_condition_number",
        ]
        print(summary[display_columns].to_string(index=False))

    if not args.skip_bifurcation:
        bif_trials, bif_summary, point, reference = run_bifurcation_sweep(
            config=config,
            reference_kappa=float(args.reference_kappa),
            integration_points=integration_points,
        )
        bif_trials.to_csv(
            args.output / "correlated_bifurcation_trials.csv", index=False
        )
        bif_summary.to_csv(
            args.output / "correlated_bifurcation_summary.csv", index=False
        )
        with (args.output / "correlated_bifurcation_theory.json").open("w") as handle:
            json.dump(
                {
                    "critical_point": asdict(point),
                    "reference_escape": reference,
                    "derivation_scope": (
                        "Exact for full-column-rank GPI pattern matrices; "
                        "not a theoretical prediction for the raw-overlap rule."
                    ),
                },
                handle,
                indent=2,
            )
        plot_bifurcation_sweep(
            bif_summary,
            point,
            reference,
            args.output / "correlated_kappa_crit_Tesc",
        )
        print("\nGPI bifurcation sweep")
        print(f"  activation: {point.activation}")
        print(f"  kappa_crit: {point.kappa_crit:.10g}")
        print(f"  T_esc(kappa_crit): {point.T_esc_critical:.10g}")
        print(
            f"  T_esc(kappa={reference['kappa']:.6g}): "
            f"{reference['T_esc']:.10g}"
        )
        print(
            bif_summary[[
                "correlation_parameter",
                "expected_pairwise_overlap",
                "full_rank_fraction",
                "kappa_crit_mean",
                "T_esc_critical_mean",
                "T_esc_reference_mean",
                "mean_gram_condition_number",
                "max_gpi_identity_error",
            ]].to_string(index=False)
        )

    with (args.output / "correlated_retrieval_config.json").open("w") as handle:
        payload = asdict(config)
        payload.update({
            "reference_kappa": float(args.reference_kappa),
            "bifurcation_integration_points": integration_points,
            "bifurcation_only": bool(args.bifurcation_only),
            "skip_bifurcation": bool(args.skip_bifurcation),
        })
        json.dump(payload, handle, indent=2)

    print(f"\nWrote results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
