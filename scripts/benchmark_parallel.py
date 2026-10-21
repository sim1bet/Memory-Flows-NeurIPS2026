#!/usr/bin/env python3
"""Benchmark unchanged paper-sized phase-plane trials at several worker counts."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from idp_hopfield.parallel import ordered_map, worker_count, available_cpus
from idp_hopfield.config import load_config
from idp_hopfield.patterns import correlated_patterns
from idp_hopfield.graphs import successor_array, cycle_adjacency
from idp_hopfield.correlation_dynamics import orthogonal_fixed_point, simulate_asymptotic_events
from run_correlation_phase_plane import resolved_base, run_phase_trial


def old_horizon_trial(task):
    patterns, successor, cfg, seed, raw, reference, epsilon, pattern_set, kindex, kappa = task
    return simulate_asymptotic_events(patterns, successor, replace(cfg, reference_kappa=kappa), seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', default='1,4,8')
    parser.add_argument('--epsilon-samples', type=int, default=4)
    parser.add_argument('--gain-samples', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.pop('IDP_CHECKPOINT_DIR', None)  # Timings must never use cached results.
    raw = load_config(ROOT / 'configs/paper/correlation_phase_plane.yml')
    cfg = resolved_base(raw)
    kappas = np.linspace(raw['kappa_min'], raw['kappa_max'], raw['kappa_points'])
    e_indices = set(np.linspace(0, len(cfg.epsilons)-1, args.epsilon_samples, dtype=int))
    k_indices = np.linspace(0, len(kappas)-1, args.gain_samples, dtype=int)
    successor = successor_array(cycle_adjacency(cfg.n_memories, cfg.cycle))
    jobs = []
    for index, (epsilon, child) in enumerate(zip(cfg.epsilons, np.random.SeedSequence(cfg.seed).spawn(len(cfg.epsilons)))):
        children = child.spawn(cfg.pattern_sets)
        if index not in e_indices:
            continue
        rng = np.random.default_rng(children[0])
        patterns = (np.ones((cfg.n_neurons, cfg.n_memories)) if epsilon >= 1 else
                    correlated_patterns(rng, cfg.n_neurons, cfg.n_memories, epsilon, 'biased').astype(float))
        seed = int(rng.integers(1, 2**31-1))
        for ki in k_indices:
            k = float(kappas[ki])
            reference = orthogonal_fixed_point(cfg.activation, k)['orthogonal_T_esc_reference']
            jobs.append((patterns, successor, cfg, seed, raw, reference, epsilon, 0, int(ki), k))
    measurements = []
    reference_rows = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for workers in dict.fromkeys([1] + [worker_count(int(x)) for x in args.workers.split(',')]):
        started = time.perf_counter()
        results = list(ordered_map(run_phase_trial, jobs, workers, 'phase_benchmark'))
        elapsed = time.perf_counter() - started
        rows = pd.DataFrame([item[0] for item in results])
        if reference_rows is None:
            reference_rows = rows
        else:
            pd.testing.assert_frame_equal(reference_rows, rows, check_exact=True)
        measurements.append({'workers': workers, 'seconds': elapsed,
                             'speedup': measurements[0]['seconds']/elapsed if measurements else 1.0})
        output.write_text(json.dumps({'allocated_cpus': available_cpus(), 'trials': len(jobs),
                         'measurements': measurements, 'complete': False}, indent=2))
    started = time.perf_counter()
    list(ordered_map(old_horizon_trial, jobs, 1, 'old_horizon_benchmark'))
    old_seconds = time.perf_counter() - started
    result = {'allocated_cpus': available_cpus(), 'trials': len(jobs),
              'N': cfg.n_neurons, 'P': cfg.n_memories,
              'epsilon_values': sorted({job[6] for job in jobs}),
              'kappa_values': sorted({job[-1] for job in jobs}),
              'measurements': measurements, 'old_four_unit_horizon_seconds': old_seconds,
              'horizon_cost_ratio': measurements[0]['seconds']/old_seconds,
              'outcomes': reference_rows['outcome'].value_counts().to_dict(),
              'full_grid_serial_hours_extrapolated': measurements[0]['seconds']/len(jobs)*len(cfg.epsilons)*len(kappas)*cfg.pattern_sets/3600,
              'caveat': 'Small stratified one-realization sample, not a full-campaign timing guarantee; excludes other experiments.',
              'complete': True}
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
