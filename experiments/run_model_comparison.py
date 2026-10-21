#!/usr/bin/env python3
"""Paper Figure 2D-F: one- versus two-timescale retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idp_hopfield.activations import activation, branch_amplitude
from idp_hopfield.parallel import ordered_map
from idp_hopfield.config import load_config
from idp_hopfield.graphs import cycle_adjacency
from idp_hopfield.io import ensure_dir, write_csv, write_json
from idp_hopfield.patterns import orthogonal_patterns, random_patterns
from idp_hopfield.simulation import cosine_scores


def simulate_case(case: dict, common: dict) -> pd.DataFrame:
    n, p = int(common["N"]), int(common["P"])
    rng = np.random.default_rng(int(case["seed"]))
    patterns = orthogonal_patterns(n, p) if case["patterns"] == "orthogonal" else random_patterns(rng, n, p)
    order = np.asarray(common["cycle"], dtype=int)
    adjacency = cycle_adjacency(p, order)
    dt = float(common["dt_physical"])
    steps = int(np.ceil(float(common["time_horizon"]) / dt))
    stride = max(1, int(round(float(common["sample_dt"]) / dt)))
    source = int(common["initial_source"])

    if case["model"] == "two_timescale":
        state = branch_amplitude(float(common["initial_saliency"]), common["activation"]) * patterns[:, source]
        saliency = np.zeros(p)
        saliency[source] = float(common["initial_saliency"])
    else:
        state = patterns[:, source].copy()
        saliency = np.full(p, np.nan)
    state += float(common["initial_perturbation"]) * rng.standard_normal(n)

    sigma = float(case.get("fast_noise", 0.0))
    convention = str(case.get("noise_convention", "per_step"))
    if not np.isfinite(sigma) or sigma < 0 or convention not in ("per_step", "diffusion"):
        raise ValueError("fast_noise must be finite and nonnegative; use per_step or diffusion")
    noise_scale = sigma if convention == "per_step" else sigma * np.sqrt(dt / float(case.get("tau_fast", 1.0)))

    rows: list[dict[str, object]] = []
    for step in range(steps + 1):
        if step % stride == 0:
            values = cosine_scores(patterns, state)
            for memory in range(p):
                rows.append(
                    {
                        "case": case["id"],
                        "model": case["model"],
                        "patterns": case["patterns"],
                        "time": step * dt,
                        "fast_noise": sigma,
                        "noise_convention": convention,
                        "memory": memory,
                        "cosine_overlap": values[memory],
                        "saliency": abs(saliency[memory]) if case["model"] == "two_timescale" else np.nan,
                    }
                )
        if step == steps:
            break
        phi = activation(state, common["activation"])
        projection = patterns.T @ phi / float(n)
        if case["model"] == "one_timescale":
            state = state + dt * (-state + float(case["kappa"]) * patterns @ (adjacency @ projection))
        else:
            next_state = state + dt / float(case["tau_fast"]) * (
                -state + patterns @ (np.abs(saliency) * projection)
            )
            next_saliency = saliency + dt / float(case["tau_slow"]) * (
                -saliency + float(case["kappa"]) * (adjacency @ projection)
            )
            state, saliency = next_state, next_saliency
        if noise_scale:
            state += noise_scale * rng.standard_normal(n)
    return pd.DataFrame(rows)


def run_case(task):
    case, cfg = task
    return simulate_case(case, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(args.output)
    frames = list(ordered_map(run_case, [(case, cfg) for case in cfg["cases"]], args.workers, "model_comparison"))
    frame = pd.concat(frames, ignore_index=True)
    summary = (
        frame.groupby(["case", "model", "patterns"], as_index=False)
        .agg(max_overlap=("cosine_overlap", "max"), terminal_time=("time", "max"))
    )
    write_csv(frame, output / "model_comparison_timeseries.csv")
    write_csv(summary, output / "model_comparison_summary.csv")
    write_json(cfg, output / "model_comparison_config.json")


if __name__ == "__main__":
    main()
