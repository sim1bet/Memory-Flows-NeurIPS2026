"""Numerical tools for the slow-fast IDP Hopfield model."""

from .activations import (
    activation,
    activation_derivative,
    branch_amplitude,
    branch_response,
    existence_threshold,
    stability_threshold,
)
from .simulation import DynamicsConfig, simulate_trajectory

__all__ = [
    "DynamicsConfig",
    "activation",
    "activation_derivative",
    "branch_amplitude",
    "branch_response",
    "existence_threshold",
    "simulate_trajectory",
    "stability_threshold",
]
