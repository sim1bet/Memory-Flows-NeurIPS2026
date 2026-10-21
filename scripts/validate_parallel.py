#!/usr/bin/env python3
"""Compare every pooled runner with 1 and 4 workers on complete smoke inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument("--compare-only", action="store_true", help="compare already generated 1/4-worker outputs")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('IDP_CHECKPOINT_DIR', None)
    env['PYTHONPATH'] = str(ROOT / 'src')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    experiments = ['model_comparison', 'limit_cycles', 'kappa_sweep', 'capacity',
                   'correlation', 'correlation_dynamics', 'correlation_phase_plane']
    timings = []
    for workers in (() if args.compare_only else (1, 4)):
        for name in experiments:
            destination = output / str(workers) / name
            start = time.perf_counter()
            command = [sys.executable, str(ROOT / f'experiments/run_{name}.py'),
                       '--config', str(ROOT / f'configs/smoke/{name}.yml'),
                       '--output', str(destination), '--workers', str(workers)]
            with (output / f'{name}_{workers}.log').open('w') as handle:
                subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            timings.append({'experiment': name, 'workers': workers, 'seconds': time.perf_counter()-start})
            print(timings[-1], flush=True)
    import pandas as pd
    checked = []
    for name in experiments:
        for file in sorted((output / '1' / name).glob('*.csv')):
            serial = pd.read_csv(file)
            parallel = pd.read_csv(output / '4' / name / file.name)
            exact = True
            pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
            checked.append({'experiment': name, 'file': file.name, 'exact': exact})
    (output / 'validation.json').write_text(json.dumps({'timings': timings, 'checked': checked}, indent=2))
    print(f'PASS: {len(checked)} CSV tables; all values and table order match exactly.', flush=True)


if __name__ == '__main__':
    main()
