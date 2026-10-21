import numpy as np

from idp_hopfield.activations import activation, branch_amplitude
from idp_hopfield.graphs import cycle_adjacency
from idp_hopfield.patterns import orthogonal_patterns
from idp_hopfield.simulation import DynamicsConfig, simulate_trajectory


def test_slow_clock_conversion():
    config = DynamicsConfig(tau_fast=0.01, tau_slow=1.0, dt_fast=0.02)
    assert config.epsilon == 0.01
    assert config.dt_slow == 0.0002


def test_first_euler_step_uses_old_state():
    patterns = orthogonal_patterns(8, 3)
    adjacency = cycle_adjacency(3, np.array([0, 1, 2]))
    config = DynamicsConfig(
        activation="tanh",
        kappa=15.0,
        dt_fast=0.1,
        tau_fast=0.01,
        tau_slow=1.0,
        slow_time_horizon=0.001,
        sample_dt_slow=0.001,
        seed=1,
    )
    history = simulate_trajectory(patterns, adjacency, config, 0, 8.0)
    gamma = branch_amplitude(8.0, "tanh")
    state = gamma * patterns[:, 0]
    z = np.array([8.0, 0.0, 0.0])
    phi = activation(state, "tanh")
    projection = patterns.T @ phi / patterns.shape[0]
    expected_z = z + config.dt_slow * (-z + config.kappa * (adjacency @ projection))
    np.testing.assert_allclose(history["saliency"][:, 1], expected_z, atol=1e-12)
