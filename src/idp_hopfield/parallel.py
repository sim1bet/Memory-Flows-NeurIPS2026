"""Bounded, deterministic process mapping with optional completed-job checkpoints."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.metadata
import multiprocessing as mp
import os
import pickle
import sys
import time


def available_cpus():
    counts = [len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)]
    for key in ('SLURM_CPUS_PER_TASK', 'PBS_NP'):
        if os.environ.get(key):
            counts.append(int(os.environ[key]))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            counts.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, min(counts))


def worker_count(value):
    value = int(value)
    if value < 0:
        raise ValueError('workers must be nonnegative (0 means allocated CPUs)')
    return min(value or available_cpus(), available_cpus())


def _single_thread():
    global _thread_limit
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    from threadpoolctl import threadpool_limits
    _thread_limit = threadpool_limits(limits=1)


def _execute(function, task):
    started = time.perf_counter()
    return function(task), time.perf_counter() - started


@lru_cache(maxsize=1)
def source_digest():
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256(sys.version.encode())
    for name in ('numpy', 'scipy', 'pandas', 'numba'):
        try:
            h.update((name + importlib.metadata.version(name)).encode())
        except importlib.metadata.PackageNotFoundError:
            h.update((name + ':absent').encode())
    for directory in ('src', 'experiments'):
        for path in sorted((root / directory).rglob('*.py')):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.digest()


def ordered_map(function, tasks, workers=1, label='trials'):
    """Yield input order; execute dynamically, checkpoint immediately on completion.

    Only load checkpoints created by this program in a trusted local directory.
    Seeds belong to the task, never to worker identity or completion order.
    """
    tasks = list(tasks)
    if not tasks:
        return
    workers = min(worker_count(workers), len(tasks))
    checkpoint = os.environ.get('IDP_CHECKPOINT_DIR')
    cache = Path(checkpoint) / label if checkpoint else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    results, missing, paths = {}, [], {}
    for index, task in enumerate(tasks):
        if cache:
            key = hashlib.sha256(source_digest() + function.__name__.encode() + pickle.dumps(task, protocol=5)).hexdigest()
            paths[index] = cache / (key + '.pickle')
            if paths[index].exists():
                try:
                    with paths[index].open('rb') as handle:
                        results[index] = pickle.load(handle)
                    continue
                except (EOFError, pickle.UnpicklingError):
                    pass
        missing.append(index)
    started = last_report = time.perf_counter()
    reused = len(results)
    print(f'[{label}] {len(tasks)} jobs; {workers} workers; {reused} cached', flush=True)

    def accept(index, timed_result):
        nonlocal last_report
        value, duration = timed_result
        results[index] = value
        if cache:
            temporary = paths[index].with_suffix('.tmp')
            with temporary.open('wb') as handle:
                pickle.dump(value, handle, protocol=5)
            os.replace(temporary, paths[index])
            with (cache / 'timings.jsonl').open('a') as handle:
                import json
                handle.write(json.dumps({'task': paths[index].stem, 'seconds': duration}) + '\n')
        now = time.perf_counter()
        if now - last_report >= 30 or len(results) == len(tasks):
            completed = len(results) - reused
            eta = (now - started) * (len(tasks) - len(results)) / max(1, completed)
            print(f'[{label}] {len(results)}/{len(tasks)}; elapsed {now-started:.1f}s; throughput ETA {eta/3600:.2f}h (variable trial costs)', flush=True)
            last_report = now

    if workers == 1:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            for index in missing:
                accept(index, _execute(function, tasks[index]))
    elif missing:
        # Spawn avoids inherited BLAS state and is portable across supported OSes.
        _single_thread()
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=_single_thread) as pool:
            pending = {}
            iterator = iter(missing)
            def fill():
                while len(pending) < 2 * workers:
                    index = next(iterator, None)
                    if index is None:
                        break
                    pending[pool.submit(_execute, function, tasks[index])] = index
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    accept(pending.pop(future), future.result())
                fill()
    for index in range(len(tasks)):
        yield results[index]
