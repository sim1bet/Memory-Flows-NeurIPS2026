"""Componentwise activations and memory-branch quantities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

ACTIVATIONS = ("tanh", "smoothf")


def activation(x: np.ndarray | float, name: str, gain: float = 1.0):
    u = gain * np.asarray(x)
    if name == "tanh":
        value = np.tanh(u)
    elif name == "smoothf":
        value = u / (1.0 + np.abs(u))
    else:
        raise ValueError(f"Unknown activation: {name}")
    return float(value) if value.ndim == 0 else value


def activation_derivative(x: np.ndarray | float, name: str, gain: float = 1.0):
    u = gain * np.asarray(x)
    if name == "tanh":
        value = gain * (1.0 - np.tanh(u) ** 2)
    elif name == "smoothf":
        value = gain / (1.0 + np.abs(u)) ** 2
    else:
        raise ValueError(f"Unknown activation: {name}")
    return float(value) if value.ndim == 0 else value


def existence_threshold(gain_phi: float = 1.0, gain_w: float = 1.0) -> float:
    return 1.0 / (gain_phi * gain_w)


def tanh_branch_u(z: np.ndarray | float, gain_phi: float = 1.0, gain_w: float = 1.0):
    """Solve u=(gain_w gain_phi |z|) tanh(u) on the positive branch."""
    z_array = np.asarray(np.abs(z), dtype=float)
    b = gain_w * gain_phi * z_array
    result = np.zeros_like(z_array)
    active = b > 1.0 + 1e-12
    if np.any(active):
        ba = b[active]
        u = np.where(
            ba < 1.5,
            np.sqrt(np.maximum(3.0 * (ba - 1.0) / ba, 1e-18)),
            ba,
        )
        for _ in range(50):
            t = np.tanh(u)
            f = u - ba * t
            df = 1.0 - ba * (1.0 - t * t)
            candidate = u - f / df
            bad = (~np.isfinite(candidate)) | (candidate <= 0.0)
            candidate[bad] = 0.5 * u[bad]
            if np.max(np.abs(candidate - u)) < 1e-13:
                u = candidate
                break
            u = candidate
        result[active] = u
    return float(result) if result.ndim == 0 else result


def branch_amplitude(
    z: np.ndarray | float,
    name: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
):
    """Return gamma solving gamma=gain_w |z| psi(gamma)."""
    z_array = np.asarray(np.abs(z), dtype=float)
    if name == "smoothf":
        gamma = np.maximum(gain_w * z_array - 1.0 / gain_phi, 0.0)
    elif name == "tanh":
        gamma = tanh_branch_u(z_array, gain_phi, gain_w) / gain_phi
    else:
        raise ValueError(f"Unknown activation: {name}")
    return float(gamma) if np.ndim(gamma) == 0 else gamma


def branch_response(
    z: np.ndarray | float,
    name: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
):
    gamma = branch_amplitude(z, name, gain_phi, gain_w)
    return activation(gamma, name, gain_phi)


def stability_threshold(
    z_source: np.ndarray | float,
    name: str,
    gain_phi: float = 1.0,
    gain_w: float = 1.0,
):
    """Successor saliency at loss of source-branch stability."""
    gamma = branch_amplitude(z_source, name, gain_phi, gain_w)
    derivative = activation_derivative(gamma, name, gain_phi)
    threshold = np.divide(
        1.0,
        gain_w * derivative,
        out=np.full_like(np.asarray(derivative, dtype=float), np.inf),
        where=np.asarray(derivative) > 0.0,
    )
    return float(threshold) if np.ndim(threshold) == 0 else threshold
