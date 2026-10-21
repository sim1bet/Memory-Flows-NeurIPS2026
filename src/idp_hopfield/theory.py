"""Escape equations, return maps, and activation-specific critical gains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq, minimize_scalar

from .activations import (
    branch_response,
    existence_threshold,
    stability_threshold,
)


@dataclass(frozen=True)
class CriticalPoint:
    activation: str
    kappa: float
    source_at_escape: float
    fixed_amplitude: float
    escape_time: float


def theoretical_escape_time(
    entry_amplitude: float,
    kappa: float,
    activation: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
    successor_initial: float = 0.0,
    grid_size: int = 12001,
) -> tuple[float, float, float]:
    """Solve the reduced stability-loss equation for one transition."""
    entry = float(abs(entry_amplitude))
    threshold = existence_threshold(gain_phi, gain_w)
    if entry <= threshold:
        return np.nan, np.nan, 0.0
    t_max = float(np.log(entry / threshold))
    time = np.linspace(0.0, t_max * (1.0 - 1e-10), int(grid_size))
    source = entry * np.exp(-time)
    drive = branch_response(source, activation, gain_phi, gain_w)
    boundary = stability_threshold(source, activation, gain_phi, gain_w)
    integral = cumulative_trapezoid(np.exp(time) * drive, time, initial=0.0)
    successor = np.exp(-time) * (float(abs(successor_initial)) + kappa * integral)
    margin = successor - boundary
    crossings = np.flatnonzero((margin[:-1] < 0.0) & (margin[1:] >= 0.0))
    if crossings.size == 0:
        return np.nan, np.nan, t_max
    index = int(crossings[0])
    fraction = -margin[index] / (margin[index + 1] - margin[index])
    t_escape = time[index] + fraction * (time[index + 1] - time[index])
    z_plus = successor[index] + fraction * (successor[index + 1] - successor[index])
    return float(t_escape), float(z_plus), t_max


def return_map(
    entry_amplitude: float,
    kappa: float,
    activation: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
) -> float:
    _, z_plus, _ = theoretical_escape_time(
        entry_amplitude, kappa, activation, gain_phi, gain_w
    )
    return z_plus


def fixed_points(
    kappa: float,
    activation: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
    upper: float | None = None,
) -> list[float]:
    threshold = existence_threshold(gain_phi, gain_w)
    upper = float(upper or max(1.1 * kappa, 3.0 * threshold))
    grid = np.geomspace(threshold * (1.0 + 1e-5), upper, 320)

    def residual(value: float, points: int = 5001) -> float:
        _, z_plus, _ = theoretical_escape_time(
            value,
            kappa,
            activation,
            gain_phi,
            gain_w,
            grid_size=points,
        )
        return z_plus - value if np.isfinite(z_plus) else np.nan

    values = np.array([residual(value) for value in grid])
    roots: list[float] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not (np.isfinite(f_left) and np.isfinite(f_right)):
            continue
        if f_left == 0.0:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: residual(value, 12001),
                        left,
                        right,
                        xtol=1e-9,
                        rtol=1e-9,
                    )
                )
            )
    unique: list[float] = []
    for root in sorted(roots):
        if not unique or abs(root - unique[-1]) > 1e-5:
            unique.append(root)
    return unique


def _kappa_from_escape_source(
    source_at_escape: float,
    activation: str,
    gain_phi: float,
    gain_w: float,
    integration_points: int,
) -> float:
    source = float(source_at_escape)
    threshold = existence_threshold(gain_phi, gain_w)
    if source <= threshold:
        return np.inf
    fixed_amplitude = float(
        stability_threshold(source, activation, gain_phi, gain_w)
    )
    ratio = fixed_amplitude / source
    if not np.isfinite(ratio) or ratio <= 1.0:
        return np.inf
    integration_variable = np.linspace(1.0, ratio, int(integration_points))
    response = branch_response(
        fixed_amplitude / integration_variable,
        activation,
        gain_phi,
        gain_w,
    )
    convolution = trapezoid(response, integration_variable) / ratio
    return float(fixed_amplitude / convolution) if convolution > 0.0 else np.inf


def critical_point(
    activation: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
    integration_points: int = 8001,
) -> CriticalPoint:
    threshold = existence_threshold(gain_phi, gain_w)
    lower = threshold * (1.0 + 1e-5)
    upper = 5.0 * threshold
    result = None
    for _ in range(8):
        result = minimize_scalar(
            lambda source: _kappa_from_escape_source(
                source, activation, gain_phi, gain_w, integration_points
            ),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if result.success and result.x < 0.85 * upper:
            break
        upper *= 2.0
    if result is None or not result.success:
        raise RuntimeError(f"Critical-point search failed for {activation}.")
    source = float(result.x)
    fixed_amplitude = float(
        stability_threshold(source, activation, gain_phi, gain_w)
    )
    return CriticalPoint(
        activation=activation,
        kappa=float(result.fun),
        source_at_escape=source,
        fixed_amplitude=fixed_amplitude,
        escape_time=float(np.log(fixed_amplitude / source)),
    )


def asymptotic_escape(
    kappa: float,
    activation: str,
    critical: CriticalPoint | None = None,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
    integration_points: int = 5001,
) -> dict[str, float]:
    """Return the stable fixed-point branch for kappa>=kappa_crit."""
    critical = critical or critical_point(activation, gain_phi, gain_w)
    tolerance = 1e-8 * max(1.0, critical.kappa)
    if kappa < critical.kappa - tolerance:
        return {
            "kappa": float(kappa),
            "kappa_critical": critical.kappa,
            "Z_theory_infty": np.nan,
            "z_source_at_escape": np.nan,
            "T_esc_theory_infty": np.nan,
        }
    if abs(kappa - critical.kappa) <= tolerance:
        return {
            "kappa": float(kappa),
            "kappa_critical": critical.kappa,
            "Z_theory_infty": critical.fixed_amplitude,
            "z_source_at_escape": critical.source_at_escape,
            "T_esc_theory_infty": critical.escape_time,
        }

    def residual(source: float) -> float:
        return _kappa_from_escape_source(
            source, activation, gain_phi, gain_w, integration_points
        ) - kappa

    left = critical.source_at_escape * (1.0 + 1e-8)
    right = max(left * 1.5, kappa * gain_phi * gain_w)
    while residual(right) < 0.0 and right < 1e6:
        right *= 1.6
    source = float(brentq(residual, left, right, xtol=1e-10, rtol=1e-10))
    fixed_amplitude = float(
        stability_threshold(source, activation, gain_phi, gain_w)
    )
    return {
        "kappa": float(kappa),
        "kappa_critical": critical.kappa,
        "Z_theory_infty": fixed_amplitude,
        "z_source_at_escape": source,
        "T_esc_theory_infty": float(np.log(fixed_amplitude / source)),
    }
