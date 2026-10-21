#!/usr/bin/env python3
"""Run the workflow with one shared CPU budget and resumable simulation jobs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from idp_hopfield.parallel import worker_count

# These already perform one trajectory or vectorized integration; run them
# concurrently instead of changing their stochastic draw/layout conventions.
SMALL = ['escape_time', 'return_map', 'noise_sweep']
POOLED = ['model_comparison', 'limit_cycles', 'kappa_sweep', 'capacity',
          'correlation', 'correlation_dynamics', 'correlation_phase_plane']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=0, help='total CPU budget; 0 detects allocation')
    parser.add_argument('--skip-tex', action='store_true')
    parser.add_argument('--skip-tests', action='store_true')
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--no-resume', action='store_true', help='ignore completed-job checkpoints')
    parser.add_argument('--only', help='comma-separated experiment names (also skips full output contracts)')
    args = parser.parse_args()
    workers = worker_count(args.workers)
    mode = 'smoke' if args.smoke else 'paper'
    selected = set(args.only.split(',')) if args.only else set(SMALL + POOLED)
    unknown = selected - set(SMALL + POOLED)
    if unknown:
        parser.error(f'Unknown experiments: {sorted(unknown)}')
    raw = ROOT / 'data/raw' / mode
    logs = ROOT / 'logs' / mode
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    if args.smoke:
        env['IDP_SMOKE'] = '1'
    else:
        env.pop('IDP_SMOKE', None)
    if args.no_resume:
        env.pop('IDP_CHECKPOINT_DIR', None)
    else:
        env['IDP_CHECKPOINT_DIR'] = str(ROOT / '.checkpoints' / mode)
    timings = []
    lock = threading.Lock()
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    timing_file = logs / f'workflow_timing_{stamp}.json'

    def launch(name, command, allocated=1):
        print(f'[run] {name} ({allocated} CPU slots)', flush=True)
        started = time.perf_counter()
        with (logs / f'{name}.log').open('w') as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        duration = time.perf_counter() - started
        with lock:
            timings.append({'stage': name, 'workers': allocated, 'seconds': duration, 'exit_code': result.returncode})
            temp = timing_file.with_suffix('.tmp')
            temp.write_text(json.dumps({'mode': mode, 'worker_budget': workers, 'stages': timings}, indent=2))
            os.replace(temp, timing_file)
        print(f'[{"done" if result.returncode == 0 else "fail"}] {name}: {duration/60:.2f} min', flush=True)
        if result.returncode:
            tail = '\n'.join((logs / f'{name}.log').read_text().splitlines()[-30:])
            raise RuntimeError(f'{name} failed:\n{tail}')

    def experiment(name, allocated):
        command = [sys.executable, str(ROOT / 'experiments' / f'run_{name}.py'),
                   '--config', str(ROOT / 'configs' / mode / f'{name}.yml'), '--output', str(raw / name)]
        if name in POOLED:
            command += ['--workers', str(allocated)]
        launch(name, command, allocated)

    print(f'Using {workers} CPU slots; one numerical-library thread per worker.', flush=True)
    with ThreadPoolExecutor(max_workers=min(workers, len(SMALL))) as pool:
        futures = [pool.submit(experiment, name, 1) for name in SMALL if name in selected]
        for future in futures:
            future.result()
    for name in POOLED:
        if name in selected:
            experiment(name, workers)
    if not args.skip_render:
        command = [sys.executable, str(ROOT / 'scripts/render_figures.py'), '--mode', mode,
                   '--input-root', str(raw), '--output-dir', str(ROOT / 'figures' / mode)]
        if args.smoke:
            command.append('--quick')
        launch('render_figures', command)
    if not args.skip_tests:
        launch('tests', [sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests')])
    if not args.only:
        launch('output_contracts', [sys.executable, str(ROOT / 'scripts/check_outputs.py'), str(raw)])
    if mode == 'paper' and not args.skip_tex and not args.skip_render:
        if shutil.which('latexmk'):
            sources = sorted((ROOT / 'plotting/latex/generated/paper').glob('*.tex'))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(launch, 'latex_' + source.stem,
                           ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-cd', str(source)]) for source in sources]
                for future in futures:
                    future.result()
        else:
            print('[skip] latexmk unavailable; PDF/SVG/PNG figures remain available.')
    print(f'Completed {mode} workflow. Timing report: {timing_file}', flush=True)


if __name__ == '__main__':
    main()
