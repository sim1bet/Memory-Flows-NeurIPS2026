#!/usr/bin/env python3
"""Generate representative slow-fast trajectories for manuscript regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.graphs import cycle_adjacency
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import orthogonal_patterns, random_patterns
from idp_hopfield.simulation import DynamicsConfig, simulate_trajectory


def run_case(task):
    case, cfg = task
    records, summaries = [], []
    n, p = int(case["N"]), int(case["P"])
    rng = np.random.default_rng(int(case["seed"]))
    patterns = (
        orthogonal_patterns(n, p)
        if case["patterns"] == "orthogonal"
        else random_patterns(rng, n, p)
    )
    order = np.asarray(case["cycle"], dtype=int)
    adjacency = cycle_adjacency(p, order)
    dynamics = DynamicsConfig(
        activation=case["activation"],
        kappa=float(case["kappa"]),
        tau_fast=float(cfg["dynamics"]["tau_fast"]),
        tau_slow=float(cfg["dynamics"]["tau_slow"]),
        dt_fast=float(cfg["dynamics"]["dt_fast"]),
        slow_time_horizon=float(case["slow_time_horizon"]),
        sample_dt_slow=float(cfg["dynamics"]["sample_dt_slow"]),
        fast_noise=float(case.get("fast_noise", 0.0)),
        seed=int(case["seed"]),
    )
    history = simulate_trajectory(
        patterns,
        adjacency,
        dynamics,
        initial_source=int(case.get("initial_source", order[-1])),
        initial_saliency=float(case.get("initial_saliency", 8.0)),
        initial_fast_perturbation=float(case.get("initial_perturbation", 1e-8)),
    )
    time = np.asarray(history["time_slow"])
    saliency = np.asarray(history["saliency"])
    overlap = np.asarray(history["cosine"])
    for memory in range(p):
        records.append(
            pd.DataFrame(
                {
                    "case": case["id"],
                    "activation": case["activation"],
                    "patterns": case["patterns"],
                    "kappa": float(case["kappa"]),
                    "time_slow": time,
                    "memory": memory,
                    "saliency": np.abs(saliency[memory]),
                    "cosine_overlap": overlap[memory],
                }
            )
        )
    dominant = np.argmax(overlap, axis=0)
    summaries.append(
        {
            "case": case["id"],
            "activation": case["activation"],
            "patterns": case["patterns"],
            "kappa": float(case["kappa"]),
            "N": n,
            "P": p,
            "seed": int(case["seed"]),
            "realized_horizon": float(history["realized_horizon"]),
            "distinct_dominant_memories": int(np.unique(dominant).size),
            "terminal_fast_overlap": float(np.max(overlap[:, -1])),
            "terminal_max_saliency": float(np.max(np.abs(saliency[:, -1]))),
        }
    )
    return records, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)

    records: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for frames, rows in ordered_map(run_case, [(case, cfg) for case in cfg["cases"]], args.workers, "limit_cycles"):
        records.extend(frames)
        summaries.extend(rows)
    write_csv(pd.concat(records, ignore_index=True), output / "limit_cycles_timeseries.csv")
    write_csv(pd.DataFrame(summaries), output / "limit_cycles_summary.csv")
    write_json(cfg, output / "limit_cycles_config.json")


if __name__ == "__main__":
    main()
