#!/usr/bin/env python3
"""Cost pilot: three loads per N/protocol, one pattern set and fanout source.

Retains all out-degrees and full individual trajectory horizons. Extrapolation
assumes sampled sources and realizations are representative; not a guarantee.
"""
import argparse
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
from idp_hopfield.config import load_config
from idp_hopfield.parallel import ordered_map
from run_capacity import build_tasks, run_task


def timed_task(task):
    started = time.perf_counter()
    rows = run_task(task)
    return {'experiment': task['experiment'], 'N': task['N'], 'P': task['P'],
            'seconds': time.perf_counter()-started, 'trials': len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ.pop('IDP_CHECKPOINT_DIR', None)
    cfg = load_config(ROOT / 'configs/paper/capacity.yml')
    tasks = build_tasks(cfg)
    jobs, populations = [], {}
    # Legacy fanout costs are estimated from the denser fanout phase family.
    for experiment in ('scaled_cycle', 'fixed_cycle', 'fanout_phase'):
        for n in cfg[experiment]['N_values']:
            group = [t for t in tasks if t['experiment'] == experiment and t['N'] == n]
            candidates = [t for t in group if t['pattern_set'] == 0]
            selected = np.linspace(0, len(candidates)-1, min(3, len(candidates)), dtype=int)
            for index in selected:
                task = dict(candidates[index])
                if experiment == 'fanout_phase':
                    task['fanout_sources_per_set'] = 1
                jobs.append(task)
            populations[(experiment, n)] = len(group)
    start = time.perf_counter()
    results = list(ordered_map(timed_task, jobs, args.workers, 'capacity_pilot'))
    estimates = []
    for (experiment, n), count in populations.items():
        values = [r['seconds'] for r in results if r['experiment'] == experiment and r['N'] == n]
        source_factor = cfg[experiment]['fanout_sources_per_set'] if experiment == 'fanout_phase' else 1
        estimates.append({'experiment': experiment, 'N': n, 'population_tasks': count,
                          'estimated_worker_hours': float(np.mean(values))*source_factor*count/3600})
    phase_hours = sum(r['estimated_worker_hours'] for r in estimates if r['experiment'] == 'fanout_phase')
    fanout_ratio = cfg['fanout']['p_count']/cfg['fanout_phase']['p_count'] * len(cfg['fanout']['outdegrees'])/len(cfg['fanout_phase']['outdegrees'])
    result = {'pilot_wall_seconds': time.perf_counter()-start, 'workers': args.workers,
              'samples': results, 'groups': estimates,
              'legacy_fanout_estimated_worker_hours': phase_hours*fanout_ratio,
              'total_estimated_worker_hours': sum(r['estimated_worker_hours'] for r in estimates)+phase_hours*fanout_ratio,
              'caveat': 'One pattern set, three loads per N, one source for fanout; worker times measured under concurrent execution, sensitive to CPU and memory contention. Coarse planning estimate only.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
