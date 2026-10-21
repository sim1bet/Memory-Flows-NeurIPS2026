from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "experiments" / "run_capacity.py"
spec = importlib.util.spec_from_file_location("capacity_runner", MODULE)
capacity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capacity
assert spec.loader is not None
spec.loader.exec_module(capacity)


def test_p_grid_allows_overcomplete_fixed_cycle_loads():
    values = capacity.p_grid(256, alpha_values=[100, 120, 140])
    assert values == [25600, 30720, 35840]
    assert all(value > 256 for value in values)


def test_scaled_grid_covers_requested_low_load_range():
    values = capacity.p_grid(256, alpha_min=0.15, alpha_max=0.50, count=10)
    loads = np.asarray(values) / 256.0
    assert np.isclose(loads[0], round(0.15 * 256) / 256)
    assert np.isclose(loads[-1], round(0.50 * 256) / 256)
    assert len(values) == 10


def test_requested_fanout_degrees_are_exact():
    cfg = capacity.load_config(ROOT / "configs" / "paper" / "capacity.yml")
    assert cfg["fanout"]["outdegrees"] == [2, 6, 10, 14, 18]
    assert cfg["fanout_phase"]["outdegrees"] == list(range(2, 19))
    assert cfg["scaled_cycle"]["N_values"] == [256, 512, 1024]
    assert cfg["fixed_cycle"]["cycle_length"] == 5
