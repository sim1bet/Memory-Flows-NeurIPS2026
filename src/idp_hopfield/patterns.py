"""Orthogonal, random, and controllably correlated pattern sets."""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import hadamard


def orthogonal_patterns(n_neurons: int, n_memories: int) -> np.ndarray:
    if n_neurons & (n_neurons - 1):
        raise ValueError("Hadamard construction requires a power-of-two N.")
    if n_memories > n_neurons - 1:
        raise ValueError("At most N-1 nonconstant Hadamard columns are available.")
    return hadamard(n_neurons, dtype=int)[:, 1 : n_memories + 1].astype(float)


def random_patterns(
    rng: np.random.Generator,
    n_neurons: int,
    n_memories: int,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    bits = rng.integers(0, 2, size=(n_neurons, n_memories), dtype=np.int8)
    return (2 * bits - 1).astype(dtype, copy=False)


def correlated_patterns(
    rng: np.random.Generator,
    n_neurons: int,
    n_memories: int,
    parameter: float,
    model: str = "biased",
) -> np.ndarray:
    """Generate patterns with expected overlap epsilon^2 or rho."""
    parameter = float(parameter)
    if not 0.0 <= parameter < 1.0:
        raise ValueError("Correlation parameter must lie in [0,1).")
    if model == "biased":
        plus = 0.5 * (1.0 + parameter)
        return np.where(
            rng.random((n_neurons, n_memories)) < plus, 1.0, -1.0
        )
    if model == "prototype":
        prototype = rng.choice(np.array([-1.0, 1.0]), size=n_neurons)
        same = 0.5 * (1.0 + math.sqrt(parameter))
        flips = np.where(
            rng.random((n_neurons, n_memories)) < same, 1.0, -1.0
        )
        return prototype[:, None] * flips
    raise ValueError(f"Unknown correlation model: {model}")


def gram_pseudoinverse(
    patterns: np.ndarray,
    rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n_neurons = patterns.shape[0]
    gram = patterns.T @ patterns / float(n_neurons)
    pinv = np.linalg.pinv(gram, rcond=rcond, hermitian=True)
    singular = np.linalg.eigvalsh(gram)
    positive = singular[singular > rcond * max(float(np.max(singular)), 1.0)]
    rank = int(positive.size)
    condition = float(np.max(positive) / np.min(positive)) if rank else np.inf
    residual = float(
        np.linalg.norm(pinv @ gram - np.eye(gram.shape[0]), ord="fro")
        / np.sqrt(gram.shape[0])
    )
    mask = ~np.eye(gram.shape[0], dtype=bool)
    diagnostics = {
        "gram_rank": rank,
        "gram_condition": condition,
        "gram_offdiag_mean": float(np.mean(gram[mask])) if mask.any() else 0.0,
        "gram_offdiag_abs_mean": float(np.mean(np.abs(gram[mask]))) if mask.any() else 0.0,
        "gram_offdiag_max_abs": float(np.max(np.abs(gram[mask]))) if mask.any() else 0.0,
        "pinv_reconstruction_residual": residual,
    }
    return gram, pinv, diagnostics
