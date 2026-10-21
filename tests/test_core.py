import numpy as np

from idp_hopfield.activations import (
    activation,
    branch_amplitude,
    stability_threshold,
)
from idp_hopfield.graphs import cycle_adjacency, successor_array
from idp_hopfield.patterns import (
    correlated_patterns,
    gram_pseudoinverse,
    orthogonal_patterns,
)
from idp_hopfield.theory import critical_point, theoretical_escape_time


def test_tanh_branch_equation():
    z = 8.0
    gamma = branch_amplitude(z, "tanh")
    assert abs(gamma - z * np.tanh(gamma)) < 1e-11


def test_smoothf_branch_equation():
    z = 8.0
    gamma = branch_amplitude(z, "smoothf")
    assert gamma == 7.0
    assert abs(gamma - z * activation(gamma, "smoothf")) < 1e-12


def test_stability_thresholds():
    assert abs(stability_threshold(3.0, "smoothf") - 9.0) < 1e-12
    assert stability_threshold(3.0, "tanh") > 3.0


def test_activation_specific_critical_gains():
    tanh = critical_point("tanh", integration_points=3001)
    smoothf = critical_point("smoothf", integration_points=3001)
    assert abs(tanh.kappa - 5.8914) < 3e-3
    assert abs(smoothf.kappa - 12.74) < 3e-2


def test_tanh_escape_reference():
    t_escape, _, _ = theoretical_escape_time(8.0, 15.0, "tanh")
    assert abs(t_escape - 1.40255) < 2e-3


def test_orthogonal_pseudoinverse_is_identity():
    patterns = orthogonal_patterns(64, 8)
    gram, pinv, diagnostics = gram_pseudoinverse(patterns)
    np.testing.assert_allclose(gram, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(pinv, np.eye(8), atol=1e-12)
    assert diagnostics["gram_rank"] == 8


def test_gpi_decodes_correlated_patterns():
    rng = np.random.default_rng(7)
    patterns = correlated_patterns(rng, 256, 15, 0.6, "biased")
    gram, pinv, diagnostics = gram_pseudoinverse(patterns)
    assert diagnostics["gram_rank"] == 15
    np.testing.assert_allclose(pinv @ gram, np.eye(15), atol=1e-11)


def test_biased_generator_expected_overlap():
    rng = np.random.default_rng(9)
    epsilon = 0.6
    patterns = correlated_patterns(rng, 20000, 8, epsilon, "biased")
    gram = patterns.T @ patterns / patterns.shape[0]
    mask = ~np.eye(8, dtype=bool)
    assert abs(gram[mask].mean() - epsilon**2) < 0.02


def test_cycle_orientation():
    adjacency = cycle_adjacency(6, np.array([1, 3, 5]))
    successor = successor_array(adjacency)
    assert successor[1] == 3
    assert successor[3] == 5
    assert successor[5] == 1
