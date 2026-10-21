# Parallel execution and runtime budget — 13 October 2026

This revision changes execution scheduling. It retains the 41 x 41 correlation grid, 41-load branching maps, 401 x 361 analytical fields, all replicates, integration steps, retrieval horizons, adaptive retry policy, revised capacity ranges and noise settings from 9 October.

## Execution design

| Workload | Independent work scheduled across processes |
|---|---|
| Correlation phase plane | 16,810 individual gain/correlation/realization trials |
| Correlation dynamics | 504 critical-gain searches; then 504 evaluations after the common baseline calibration |
| Controlled correlation | 400 pattern/correlation sets, retaining paired rule/activation random streams within each set |
| Capacity | 1,920 tasks: 300 scaled cycles, 90 fixed cycles, 300 fanout and 1,230 fanout-phase tasks |
| Gain sweep | Fixed batches of two gains; 13 batches in the paper configuration |
| Model comparison | Independent model cases |
| Limit cycles | Independent representative cases |
| Escape time, return map, noise sweep | Independent experiment processes run concurrently; existing vectorization and paired-noise conventions retained |
| Optional LaTeX | Independent wrappers compile concurrently |

The heavy experiments run sequentially at the workflow level, each receiving the full CPU budget. This prevents accidentally launching several full-sized pools at once. The smaller jobs share that same budget. Simulation time steps within one trajectory remain sequential because each depends on the preceding state. Figure rendering and final aggregation remain serial.

The process mapper uses portable `spawn`, dynamically schedules at most twice the worker count in flight, and restores original task order before aggregation. Random seeds are attached to scientific tasks, not worker identities. BLAS/OpenMP libraries use one thread per process, avoiding nested oversubscription. Requested workers are capped by the detected affinity, scheduler allocation and cgroup CPU quota.

The gain sweep uses fixed batch widths independent of the number of workers. This is necessary because changing BLAS matrix widths can cause tiny rounding changes in float32 stored trajectories. The fixed partition gives identical results when changing worker count. Compared with the previous single wide batch, last-digit differences in derived escape times can remain; the differential equations, input noise and integration steps are unchanged.

## Restart after a wall-time limit

Run the same command again:

```bash
./run_all.sh --workers 8 --skip-tex
```

Completed independent jobs are saved atomically under `.checkpoints/paper/`; the smoke mode uses a separate directory. Keep checkpoints on persistent storage. The same command can resume with a different worker count. Cache identity includes the task, simulation source, Python version and relevant numerical-package versions. Edited configurations or numerical code therefore cannot silently reuse stale trials.

These are completed-job checkpoints, not within-trajectory snapshots. A job interrupted in its middle is rerun. The existing escape-horizon retries still replay from the same seed; this revision does not replace the event simulator with a different continuation scheme. Small standalone stages and final summaries are recomputed. Use `--no-resume` to ignore saved jobs. Checkpoints are local pickle files and should only be reused from trusted runs of this repository.

Do not run two copies of the full workflow into the same output/checkpoint directory. Use separate repository copies for concurrent campaigns.

## Validation

- Complete smoke workflow with four workers: all ten experiments completed; all 23 CSV contracts passed.
- All seven pooled runners were executed separately with one and four workers. All 21 output CSV tables matched exactly, including row order and event outcomes.
- All 30 regression test functions passed by direct invocation; pytest is not installed in the authoring environment. These include checkpoint reuse across worker counts, cache invalidation on changed inputs, worker exception propagation and invalid worker budgets.
- The paper-sized phase-plane pilot produced exactly identical trial tables at one, four and eight workers.
- Existing rendered smoke previews, their data and atlas are retained from 9 October. This execution-only revision does not claim a new paper campaign or a new graphical design.

The authoring environment provides eight CPU slots on an Intel Xeon Platinum 8370C virtual machine. It lacks Numba, so smoke controlled-correlation timing uses that module's Python fallback. The phase-plane and capacity pilots use their ordinary NumPy implementation; neither relies on Numba. Production should install the pinned project dependencies, including Numba.

Machine-readable measurements and test records are in `reports/parallel_validation/`.

## Measured phase-plane scaling

The pilot used full paper dimensions (N=256, P=15), unchanged time steps, two five-transition cycles, the full adaptive horizons and 16 trials. It sampled four correlations (0, 0.325, 0.65, 1) and four gains (6.5, 20.6375, 34.775, 50), with one realization per combination. Ten trials timed out at the final horizon and six succeeded.

| Workers | Measured elapsed time | Measured speedup |
|---:|---:|---:|
| 1 | 125.21 s | 1.00x |
| 4 | 49.62 s | 2.52x |
| 8 | 38.14 s | 3.28x |

The same sampled trajectories with the former fixed four-unit deadline took 14.64 s. Thus, this sample's adaptive-horizon cost is approximately 8.55x the old per-cell cost, on top of the 5x denser grid. This ratio is not a whole-campaign multiplier.

Extrapolating this small sample to all 16,810 trials gives approximately 36.5 hours serial, 14.5 hours at four workers and 11.1 hours at eight workers for the phase-plane stage alone. These are planning estimates, not full-run measurements. The sample overrepresents endpoint correlations and includes only one realization; the full sweep may have a different timeout mix. Thousands of tasks should balance workers better than 16 tasks, but memory contention and hardware differences can offset that advantage.

## Capacity pilot and full-workflow estimate

The capacity pilot ran 21 tasks at the paper dimensions, spanning three loads per N for scaled cycles, fixed cycles and the fanout phase plane. It retained all configured out-degrees and full individual trajectory horizons. For fanout it sampled one of the eight sources, and extrapolated to the full source and pattern-set counts. The legacy five-degree fanout stage was estimated by its load/degree count relative to the denser phase-plane family. This is a coarse workload estimate: three loads and one pattern realization cannot characterize all runtime variability.

The pilot finished in 487.8 seconds. Its slowest task was the N=1024, P=512 scaled-cycle case: 485.1 seconds for its two activation runs. That cost is also a consequence of the requested move to higher scaled-cycle loads, not just mesh refinement.

| Capacity component | Estimated sum of worker-hours |
|---|---:|
| Scaled cycles, all N | 7.75 |
| Fixed cycles | 0.56 |
| Fanout phase planes, all N | 43.15 |
| Legacy fanout panels | 3.10 |
| **Total** | **54.55** |

These worker-hours are extrapolated from task wall times observed during concurrent execution; they are not processor-independent CPU-hour measurements. Dividing by eight gives about **6.8 hours** for a reasonably balanced eight-worker capacity run on comparable hardware. Combined with the conservative phase-plane extrapolation, the two large stages account for about **18 hours** before the remaining experiments, rendering and checks.

Recommended full-workflow planning ranges:

| Allocated CPU cores | Estimated full-run wall time | Basis |
|---:|---:|---|
| 4 | 32–42 h | Measured four-worker phase scaling; capacity work divided across four workers; allowance for remaining stages and sampling error |
| 8 | 20–28 h | Measured eight-worker phase scaling plus capacity pilot; allowance for remaining stages and sampling error |
| 16 | 12–18 h | Extrapolation beyond available hardware; reduced efficiency assumed |
| 32 | 8–14 h | Extrapolation beyond available hardware; additional scaling loss assumed |

**For a 24-hour allocation, use 16 cores rather than treating eight cores as a safe fit.** These ranges are planning allowances, not statistical confidence intervals or guarantees. They assume CPU performance comparable to this environment, sufficient RAM, no competing CPU-heavy jobs and the pinned production dependencies. The full campaign has not been timed, and the non-capacity/non-phase paper stages were not rerun here. The 16/32-core figures assume further useful scaling; they are not measurements. At 16 cores the estimate allows roughly 20% less incremental efficiency than ideal doubling; at 32, roughly 35% less than ideal quadrupling from eight. Large full grids improve load balancing compared with the small pilot, while memory bandwidth can limit scaling in the opposite direction.

The prior nine-hour measurement is not divided blindly by the worker count: the prior capacity runner already accepted multiple workers, and the worker count/hardware/stage timings for that run are unknown. Longer adaptive horizons and higher scaled-cycle loads also change the amount of work. That is why this estimate is grounded in current paper-sized samples, with explicit uncertainty, rather than promising linear speedup of the old wall time.

## Launch and monitor

```bash
# Install the declared production and test dependencies in your existing environment
python -m pip install -e '.[dev,artifacts]'

# Run with 16 allocated cores; PDF/SVG/PNG figures still render with --skip-tex
./run_all.sh --workers 16 --skip-tex

# After a wall-time interruption, issue exactly the same command to resume
./run_all.sh --workers 16 --skip-tex

# Watch the active stage's progress, for example:
tail -f logs/paper/correlation_phase_plane.log
```

Do not request more workers than allocated CPUs; the launcher caps the count. `--workers 0` (the default) detects available allocation. A multi-node allocation does not provide distributed execution: all worker processes run on one node. GPUs are not used.

To measure your own node without running the full campaign:

```bash
python scripts/benchmark_parallel.py --workers 1,4,8,16 --output logs/phase_parallel_benchmark.json
python scripts/benchmark_capacity.py --workers 16 --output logs/capacity_parallel_benchmark.json
```

The phase pilot is deliberately small and reports its sample, timeout mix, speedups and full-grid extrapolation. The capacity pilot may contain an individual trial lasting several minutes. Pilot measurements always bypass checkpoints. The normal campaign writes per-stage timing JSON and per-job timing records so subsequent estimates can use actual campaign throughput.
