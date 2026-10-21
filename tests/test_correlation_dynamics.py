from __future__ import annotations

import numpy as np

from idp_hopfield.correlation_dynamics import Config, orthogonal_fixed_point


def test_orthogonal_reference_is_supercritical_at_15():
    values = orthogonal_fixed_point("tanh", 15.0)
    assert values["orthogonal_kappa_crit"] < 15.0
    assert np.isfinite(values["orthogonal_T_esc_reference"])
    assert np.isfinite(values["orthogonal_Z_reference"])


def test_adaptive_configuration_has_consistent_slow_step():
    cfg = Config(tau_fast=0.01, tau_slow=1.0, dt_fast=0.05)
    assert np.isclose(cfg.dt_slow, 5e-4)
