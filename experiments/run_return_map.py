#!/usr/bin/env python3
"""Evaluate activation-specific saliency return maps and fixed points."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.config import load_config
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.theory import critical_point, fixed_points, theoretical_escape_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)

    map_rows: list[dict[str, float | str]] = []
    point_rows: list[dict[str, float | str | int]] = []
    for activation in cfg["activations"]:
        critical = critical_point(activation)
        for kappa in cfg["kappas"]:
            z_grid = np.linspace(float(cfg["z_min"]), float(cfg["z_max"]), int(cfg["z_points"]))
            for z in z_grid:
                t_escape, z_plus, t_max = theoretical_escape_time(
                    float(z), float(kappa), activation
                )
                map_rows.append(
                    {
                        "activation": activation,
                        "kappa": float(kappa),
                        "Z": float(z),
                        "Z_plus": z_plus,
                        "T_esc": t_escape,
                        "T_max": t_max,
                        "transition_exists": bool(np.isfinite(z_plus)),
                    }
                )
            roots = fixed_points(float(kappa), activation)
            for index, root in enumerate(roots):
                t_escape, _, _ = theoretical_escape_time(root, float(kappa), activation)
                point_rows.append(
                    {
                        "activation": activation,
                        "kappa": float(kappa),
                        "fixed_point_index": index,
                        "fixed_point": root,
                        "T_esc": t_escape,
                        "stable": bool(index == len(roots) - 1 and len(roots) > 1),
                        "kappa_critical": critical.kappa,
                    }
                )
    x_z = np.linspace(1.0, 4.0, 300)
    z_value = 4.0
    g_zx = -x_z * (x_z - 1.0) * (x_z + 1.0 - 2.0 * z_value) / (2.0 * z_value ** 3)
    implicit_z = pd.DataFrame({"Z": z_value, "X": x_z, "G": g_zx, "inverse_kappa": 1.0 / 15.0})
    x_fixed = np.linspace(1.001, 6.0, 400)
    g_fixed = (x_fixed - 1.0) ** 2 * (2.0 * x_fixed + 1.0) / (2.0 * x_fixed ** 5)
    implicit_fixed = pd.DataFrame({"X": x_fixed, "G_fixed": g_fixed, "inverse_kappa": 1.0 / 15.0})
    write_csv(implicit_z, output / "smoothf_implicit_G_Z4.csv")
    write_csv(implicit_fixed, output / "smoothf_implicit_G_fixed.csv")
    write_csv(pd.DataFrame(map_rows), output / "return_map.csv")
    write_csv(pd.DataFrame(point_rows), output / "return_map_fixed_points.csv")
    write_json(cfg, output / "return_map_config.json")


if __name__ == "__main__":
    main()
