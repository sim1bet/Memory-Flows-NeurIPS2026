"""Explicit-Euler simulation on the dimensionless fast clock."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import activation, branch_amplitude


@dataclass(frozen=True)
class DynamicsConfig:
    activation: str = "tanh"
    kappa: float = 15.0
    gain_phi: float = 1.0
    gain_w: float = 1.0
    tau_fast: float = 0.01
    tau_slow: float = 1.0
    dt_fast: float = 0.01
    slow_time_horizon: float = 8.0
    sample_dt_slow: float = 0.001
    fast_noise: float = 0.0
    slow_noise: float = 0.0
    noise_convention: str = "per_step"
    seed: int = 1

    @property
    def epsilon(self) -> float:
        return self.tau_fast / self.tau_slow

    @property
    def dt_slow(self) -> float:
        return self.dt_fast * self.epsilon


def cosine_scores(patterns: np.ndarray, state: np.ndarray, absolute: bool = True) -> np.ndarray:
    denominator = np.sqrt(patterns.shape[0]) * np.linalg.norm(state)
    if denominator == 0.0:
        return np.zeros(patterns.shape[1])
    scores = patterns.T @ state / denominator
    return np.abs(scores) if absolute else scores


def simulate_trajectory(
    patterns: np.ndarray,
    adjacency: np.ndarray,
    config: DynamicsConfig,
    initial_source: int,
    initial_saliency: float,
    decode_matrix: np.ndarray | None = None,
    initial_fast_perturbation: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Integrate the coupled model and sample at a slow-time cadence."""
    patterns = np.asarray(patterns, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    n_neurons, n_memories = patterns.shape
    if adjacency.shape != (n_memories, n_memories):
        raise ValueError("Adjacency and pattern dimensions do not match.")
    decoder = np.eye(n_memories) if decode_matrix is None else np.asarray(decode_matrix)

    dt_slow = config.dt_slow
    stride = max(1, int(round(config.sample_dt_slow / dt_slow)))
    sample_dt = stride * dt_slow
    n_steps = int(np.ceil(config.slow_time_horizon / dt_slow))
    n_steps = int(np.ceil(n_steps / stride) * stride)
    n_samples = n_steps // stride + 1

    rng = np.random.default_rng(config.seed)
    gamma = branch_amplitude(
        initial_saliency, config.activation, config.gain_phi, config.gain_w
    )
    fast = gamma * patterns[:, initial_source].copy()
    if initial_fast_perturbation > 0.0:
        perturbation = rng.standard_normal(n_neurons)
        perturbation -= patterns[:, initial_source] * (
            patterns[:, initial_source] @ perturbation / n_neurons
        )
        norm = np.linalg.norm(perturbation)
        if norm > 0.0:
            fast += initial_fast_perturbation * np.sqrt(n_neurons) * perturbation / norm

    saliency = np.zeros(n_memories)
    saliency[initial_source] = initial_saliency

    time = np.arange(n_samples) * sample_dt
    z_history = np.empty((n_memories, n_samples), dtype=float)
    activated_overlap = np.empty_like(z_history)
    cosine = np.empty_like(z_history)
    decoded_history = np.empty_like(z_history)

    def evaluate(state: np.ndarray):
        phi = activation(state, config.activation, config.gain_phi)
        raw = patterns.T @ phi / float(n_neurons)
        return phi, raw, decoder @ raw

    phi, raw, decoded = evaluate(fast)
    z_history[:, 0] = saliency
    activated_overlap[:, 0] = np.abs(raw)
    cosine[:, 0] = cosine_scores(patterns, fast)
    decoded_history[:, 0] = decoded

    fast_noise_scale = config.fast_noise
    slow_noise_scale = config.slow_noise
    if config.noise_convention == "diffusion":
        fast_noise_scale *= np.sqrt(config.dt_fast)
        slow_noise_scale *= np.sqrt(dt_slow)
    elif config.noise_convention != "per_step":
        raise ValueError("noise_convention must be 'per_step' or 'diffusion'.")

    sample_index = 1
    for step in range(1, n_steps + 1):
        # Both increments use the state at the beginning of the Euler step.
        fast_rhs = -fast + config.gain_w * patterns @ (np.abs(saliency) * decoded)
        slow_rhs = -saliency + config.kappa * (adjacency @ decoded)
        next_fast = fast + config.dt_fast * fast_rhs
        next_saliency = saliency + dt_slow * slow_rhs
        if fast_noise_scale:
            next_fast += fast_noise_scale * rng.standard_normal(n_neurons)
        if slow_noise_scale:
            next_saliency += slow_noise_scale * rng.standard_normal(n_memories)
        fast, saliency = next_fast, next_saliency
        phi, raw, decoded = evaluate(fast)

        if step % stride == 0:
            z_history[:, sample_index] = saliency
            activated_overlap[:, sample_index] = np.abs(raw)
            cosine[:, sample_index] = cosine_scores(patterns, fast)
            decoded_history[:, sample_index] = decoded
            sample_index += 1

    return {
        "time_slow": time,
        "saliency": z_history,
        "activated_overlap": activated_overlap,
        "cosine": cosine,
        "decoded_overlap": decoded_history,
        "sample_dt_slow": sample_dt,
        "realized_horizon": n_steps * dt_slow,
    }
