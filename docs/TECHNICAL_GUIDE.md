# Technical guide

[← Back to the visual repository guide](../README.md)

This document preserves the complete numerical, protocol, and execution reference for the slow-fast sequential IDP Hopfield experiments.

## Parallel execution update - 13 October 2026

The dense meshes and all experimental settings from 9 October are retained.
The complete workflow now uses a shared CPU budget across the simulation runners:

```bash
./run_all.sh --workers 8 --skip-tex
```

Use `--workers 16` when 16 CPU cores are allocated. Omitting `--workers` now
uses the detected allocation (CPU affinity, scheduler variables and cgroup quota).
Numerical libraries use one thread per process. Do not multiply the worker count
by launching several full workflows against the same output directory.

Completed simulation jobs are checkpointed under `.checkpoints/<mode>/`.
After a wall-time stop, rerun the same command to reuse completed jobs. Worker
count may change on restart. Keep that directory on persistent storage together
with the repository. An interrupted individual job is rerun; time steps inside
an unfinished trajectory are not checkpointed. Configuration, code and numerical
package changes invalidate affected cache keys. `--no-resume` ignores the cache.

For just the phase plane:

```bash
./run_all.sh --workers 8 --only correlation_phase_plane --skip-render --skip-tests --skip-tex
```

Direct experiment runners also accept `--workers` for capacity, correlation,
correlation dynamics, correlation phase plane, model comparison, limit cycles
and the gain sweep. The launcher enables checkpointing automatically; for direct
runners set `IDP_CHECKPOINT_DIR` to a persistent directory if desired.

Per-stage times are written to `logs/<mode>/workflow_timing_*.json`; heavy-runner
logs report completed jobs and a provisional throughput ETA. That ETA is less
reliable early in a sweep because different parameter regions take different times.
The execution design, measurements and runtime assumptions are documented in
`reports/PARALLEL_EXECUTION_20261013.md`. Run the included machine-specific pilot:

```bash
python scripts/benchmark_parallel.py --workers 1,4,8 --output logs/phase_parallel_benchmark.json
python scripts/benchmark_capacity.py --workers 8 --output logs/capacity_parallel_benchmark.json
```



This repository is the reproducibility package for the numerical experiments accompanying **Memory flows: geometry and dynamics of sequential retrieval in input-driven Hopfield networks**. It consolidates the previously separate scripts for limit cycles, escape-time validation, gain and noise sweeps, capacity, branching, and correlated memories into one config-driven codebase.

The package has two objectives:

1. reproduce every numerical experiment represented in the supplied code, including the paper's principal retrieval and return-map panels;
2. reproduce the additional experiments introduced in response to the reviews: direct theory-versus-simulation escape times, asymptotic `T_esc(kappa)`, noise-dependent ejection delay, sequence-length and out-degree capacity sweeps, and pseudoinverse retrieval under controlled pattern correlation.

The repository does **not** invent missing source material. Pure schematics and manuscript-only experiments for which no numerical snippet was supplied are marked explicitly in [`FIGURE_MAP.md`](../FIGURE_MAP.md).

## Requested adjustments, 2026-10-09

This version implements denser sampled phase planes, a separate scalar-magnitude palette, x-axis labels and tick numbers on every fanout subplot, the revised capacity loads, escape-time-aware reliability horizons, and fast-state noise of 1e-4 in the orthogonal two-timescale comparison. See [`reports/REVISION_20261009.md`](../reports/REVISION_20261009.md) for exact settings and verification. Updated smoke results and their PDF/SVG/PNG previews are included.

## Editorial figure revision

The redesigned figures are in `figures/editorial/smoke/`: 19 PDF, SVG and 300 dpi PNG masters, including a new analytically derived source-branch flow portrait. The supplied numerical data are reduced software-validation data, clearly labeled in the outputs. They are not paper-scale estimates.

```bash
# Render the supplied data, without rerunning simulations
python scripts/render_figures.py --mode smoke

# Apply the same design to saved paper-scale results
python scripts/render_figures.py --mode paper
```

See [`reports/VISUAL_DESIGN.md`](../reports/VISUAL_DESIGN.md) for the scientific encodings, typography, palette, reproduction options and verification limits. The optional `artifacts` dependency group supports the PDF atlas and visual QA utilities: `python -m pip install -e ".[artifacts]"`.

The original `figures/smoke/` PNGs are retained as historical previews; use `figures/editorial/smoke/` for this revision. `run_all.sh` now generates the redesigned figures directly in its usual `figures/<mode>/` destination.

## Scientific conventions

The implementation uses `x` for the fast neural state and `z` for the slow saliency state. With pattern matrix `Xi in R^(N x P)`, decoded memory coordinate `q`, and adjacency convention `A[target, source] = 1`, the code integrates

```text
m(x) = Xi.T @ psi(x) / N
q(x) = D @ m(x)

dx/ds = -x + gain_w * Xi @ (abs(z) * q)
dz/ds = epsilon * (-z + kappa * A @ q)
```

where `s = t_physical / tau_fast`, `epsilon = tau_fast / tau_slow`, and

```text
dt_slow = dt_fast * tau_fast / tau_slow.
```

`D = I` for orthogonal/random raw-overlap experiments. For the generalized pseudoinverse experiment,

```text
O = Xi.T @ Xi / N
D = pinv(O)
```

so `q = O^+ m` decodes memory coordinates before graph routing.

### Escape-time definition

`T_esc` is the **first loss of stability of the source memory branch**, detected from

```text
abs(z_successor) - h(abs(z_source)) = 0.
```

The later time at which the successor becomes the persistent dominant memory is stored independently as `T_switch`. Their difference is the finite-timescale ejection delay. Threshold-crossing durations and inter-peak intervals are never used as substitutes for `T_esc`.

### Activations

Both componentwise activations used in the paper are implemented centrally:

```text
tanh(x)
SmoothF(x) = x / (1 + abs(x)).
```

The theory code computes activation-specific memory branches, stability thresholds, return maps, fixed points, and critical gains. The default normalization gives approximately

```text
kappa_crit(tanh)    = 5.891385
kappa_crit(SmoothF) = 12.74.
```

## Repository structure

```text
.
├── README.md
├── FIGURE_MAP.md
├── PROVENANCE.md
├── VALIDATION.md
├── environment.yml
├── pyproject.toml
├── run_all.sh
├── configs/
│   ├── paper/                 # submission-scale settings and fixed seeds
│   ├── smoke/                 # fast end-to-end software validation
│   └── exploratory/           # non-canonical retained analyses
├── experiments/              # simulations; write CSV/JSON only
├── plotting/                 # figures generated only from saved CSV files
│   └── tikz/                 # standalone pgfplots sources
├── src/idp_hopfield/         # shared model, theory, event, graph, and I/O code
├── tests/                    # unit and regression tests
├── scripts/check_outputs.py  # CSV contract validation
├── data/raw/                 # generated event/trial/summary tables
├── figures/                  # generated PDF/PNG figures
└── logs/                     # one log per command executed by run_all.sh
```

No experiment imports another experiment script. Shared definitions live under `src/idp_hopfield`, which prevents activation, clock, graph, or event conventions from drifting between figures.

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate slow-fast-idp-hopfield
```

The environment pins Python and all numerical packages. A TeX distribution containing `latexmk`, `pgfplots`, and `standalone` is needed only to compile the TikZ sources.

### Standard virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## First validation

Run the complete reduced workflow:

```bash
./run_all.sh --smoke --workers 1 --skip-tex
```

This command executes every experiment and plotting path, runs the test suite, and verifies that every required CSV exists, is nonempty, and contains no accidental index column. Smoke results test the software and file contracts; they are not paper estimates.

## Reproduce the paper-scale numerical package

```bash
./run_all.sh --workers 8
```

The master command runs, in order:

1. one- versus two-timescale comparison;
2. representative retrieval regimes;
3. direct escape-time validation;
4. SmoothF and tanh return maps;
5. asymptotic `T_esc` versus `kappa`;
6. switch-delay versus noise;
7. scaled-cycle, fixed-length, selected-outdegree, and fanout phase-plane capacity sweeps;
8. controlled-correlation raw/GPI retrieval comparison;
9. raw correlated-pattern bifurcation analysis with correlation-dependent `kappa_crit`, matched-supercritical `T_esc`, and asymptotic saliency;
10. the `(epsilon, kappa)` escape-time phase plane with failed-transition masking;
11. all plotting scripts;
12. the unit tests and CSV contract checks;
13. standalone TikZ compilation when `latexmk` is available.

The paper configurations are deliberately not replaced by smoke settings. Output is written to

```text
data/raw/paper/<experiment>/
figures/paper/
figures/data/
logs/paper/
```

An interrupted capacity computation can be divided across machines by running selected `N` values through edited copies of `configs/paper/capacity.yml`; the CSVs have explicit `N`, `P`, seed, pattern-set, source, activation, and graph columns and can be concatenated before plotting.

## Run experiments individually

All commands below write raw and summary CSV files without plotting.

| Experiment | Command |
|---|---|
| Figure 2 model comparison | `python experiments/run_model_comparison.py --config configs/paper/model_comparison.yml --output data/raw/paper/model_comparison` |
| Retrieval regimes | `python experiments/run_limit_cycles.py --config configs/paper/limit_cycles.yml --output data/raw/paper/limit_cycles` |
| Escape-time validation | `python experiments/run_escape_time.py --config configs/paper/escape_time.yml --output data/raw/paper/escape_time` |
| Return maps and implicit curves | `python experiments/run_return_map.py --config configs/paper/return_map.yml --output data/raw/paper/return_map` |
| `T_esc(kappa)` | `python experiments/run_kappa_sweep.py --config configs/paper/kappa_sweep.yml --output data/raw/paper/kappa_sweep` |
| Noise-dependent ejection delay | `python experiments/run_noise_sweep.py --config configs/paper/noise_sweep.yml --output data/raw/paper/noise_sweep` |
| Capacity and out-degree | `python experiments/run_capacity.py --config configs/paper/capacity.yml --output data/raw/paper/capacity --workers 8` |
| Correlation/gain escape-time phase plane | `python experiments/run_correlation_phase_plane.py --config configs/paper/correlation_phase_plane.yml --output data/raw/paper/correlation_phase_plane` |
| Correlated memories and GPI | `python experiments/run_correlation.py --config configs/paper/correlation.yml --output data/raw/paper/correlation` |
| Raw correlation dynamics | `python experiments/run_correlation_dynamics.py --config configs/paper/correlation_dynamics.yml --output data/raw/paper/correlation_dynamics` |

Each plotting script accepts an input directory containing the corresponding CSVs. For example:

```bash
python plotting/plot_escape_time.py \
  --input data/raw/paper/escape_time \
  --output figures/paper/escape_time_validation \
  --data-dir figures/data
```

## Output contracts

### Escape-time experiment

`escape_time_events.csv` has one row per source-to-successor event. Key columns are:

```text
Z_entry
T_esc_theory
T_esc_empirical
T_switch_empirical
switch_delay_after_stability_loss
Z_plus_theory
Z_plus_empirical
```

`escape_time_timeseries.csv` contains the time-resolved saliency and overlap data used for the trajectory panel.

### Kappa sweep

```text
kappa_sweep_events.csv       eventwise finite-epsilon stability losses
kappa_sweep_summary.csv      terminal-window plateau estimates
kappa_sweep_theory.csv       dense singular-limit stable branch
kappa_critical_point.csv     activation-specific saddle-node
```

A simulated point is marked stationary only when the terminal window contains the configured minimum event count and passes both relative-range and relative-drift tests.

### Noise sweep

`noise_sweep_trials.csv` contains every paired stochastic trial. Negative values of

```text
T_switch - T_esc
```

are retained and labeled `premature_switch`; they are not clipped to zero.

### Capacity

Four explicit protocols are reported from the same runner:

- `scaled_cycle`: `N in {256,512,1024}`, `P/N in [0.15,0.50]`, and `L ~= P/3`;
- `fixed_cycle`: `N=256`, fixed `L=5`, and the deliberately overcomplete range `P/N in {100,105,...,140}`;
- `fanout`: `N in {256,512,1024}`, `P/N in [0.05,0.30]`, and only out-degrees `d in {2,6,10,14,18}` for the line plots;
- `fanout_phase`: the same low-load grid with every integer out-degree `d=2,...,18`, used only for the transition-probability phase plane.

The fixed-cycle implementation intentionally permits `P>N`; the low-load protocols do not clip their requested graph degree when it is impossible. Cells with `d >= P` are absent because a source cannot have more than `P-1` distinct non-self successors. Several source trials sharing one memory matrix are not treated as independent pattern-set samples; confidence intervals are obtained from pattern-set fractions. The tasks measure **graph-gated sequential reliability**, not the classical all-memories-active Hopfield threshold.

### Correlation and pseudoinverse rule

The default biased generator uses

```text
Pr(xi_i^mu = +1) = (1 + epsilon) / 2,
```

so the expected pairwise overlap is `epsilon^2`. The alternative prototype generator interprets its parameter directly as expected overlap.

The GPI operator restores memory-index selectivity for full-rank correlated pattern sets. In general,

```text
Xi diag(abs(z)) O^+ Xi.T / N
```

is not symmetric because `diag(abs(z))` and `O^+` need not commute. The GPI sweep is therefore a numerical extension; it is not presented as a proof that the original symmetric energy theorem extends unchanged to arbitrary correlated memories.



### Correlation/gain escape-time phase plane

`run_correlation_phase_plane.py` evaluates the raw correlated-pattern model over a grid of `epsilon` and gains satisfying `kappa > kappa_crit` in the orthogonal baseline. Each cell stores the probability of completing the configured cycle and the mean finite `T_esc` among correct transitions. `plot_correlation_phase_plane.py` displays `T_esc` as a heatmap and masks cells with zero correct transitions in gray; the figure-ready CSV retains both `T_esc` and transition probability so the failure region is auditable.

### Raw correlated-pattern dynamics

The canonical raw-correlation dynamical analysis is `run_correlation_dynamics.py`. For each pattern realization it first estimates `kappa_crit(epsilon)` and then evaluates sustained dynamics at

```text
kappa_eval(epsilon,r) = supercritical_factor * kappa_crit(epsilon,r),
```

with a factor strictly larger than one. Unless explicitly set, the factor is calibrated so the mean evaluation gain at `epsilon=0` equals the configured reference gain (15 in the paper configuration). This prevents the high-correlation `T_esc` and asymptotic-saliency estimates from being evaluated in a subcritical regime. The orthogonal comparison is recomputed at the same matched gain. Outputs are:

```text
correlation_dynamics_trials.csv
correlation_dynamics_events.csv
correlation_dynamics_summary.csv
correlation_dynamics_theory.json
```

The `epsilon=1` endpoint is treated as rank-degenerate: all biased patterns coincide and memory-specific sequential retrieval is not identifiable.

## Randomness and statistical units

All seeds are fixed in YAML and propagated through `numpy.random.SeedSequence`. Pattern generation, graph sampling, source selection, and perturbations use independent child streams. Tanh and SmoothF receive paired memory realizations in capacity and correlation studies.

For replicated sweeps, the independently generated pattern matrix is the primary statistical unit. Trial-level CSVs are retained so alternative intervals or event criteria can be recomputed without rerunning the ODEs.

## Vector and LaTeX rendering

The canonical final figures are generated by the shared editorial Matplotlib pipeline as PDF, editable-text SVG and PNG. Figure-ready numerical CSVs remain available under `figures/data/` (or `figures/data/smoke/`).

The renderer creates standalone LaTeX wrappers in `plotting/latex/generated/<mode>/`; these include the same PDF masters and are compiled by `run_all.sh` when TeX is enabled. This keeps publication styling identical across inclusion paths. Original pgfplots sources under `plotting/tikz/` are preserved as historical editable alternatives, with their original appearances and selected-curve conventions.

```bash
python scripts/build_figure_atlas.py --figures figures/editorial/smoke --output figures/editorial/Memory_Flows_Figure_Atlas.pdf
python scripts/figure_qa.py --figures figures/editorial/smoke --output reports/visual_qa
```

The atlas and QA commands require the optional `artifacts` dependencies.

## Numerical precision and limitations

- All dynamical updates are simultaneous explicit-Euler steps evaluated at the beginning of the step.
- Paper capacity runs use `dt_fast = 0.10`; convergence subsets should be repeated at `0.05` before interpreting small differences between adjacent probability cells.
- Stability loss and visible switching are separate observables.
- The singular-limit escape theory is expected to be least accurate near the return-map saddle-node, where finite-timescale effects are amplified.
- Absolute cosine scoring identifies `xi` and `-xi`; set `score_mode: signed` in the capacity config for labeled-sign retrieval.
- The rigorous memory-manifold theory assumes orthogonal memories and out-degree at most one. Correlation and branching are intentionally labeled numerical extensions.

See [`VALIDATION.md`](../VALIDATION.md) for the executed software checks and [`PROVENANCE.md`](../PROVENANCE.md) for the mapping from the uploaded scripts to the consolidated modules.

## Merge provenance

This release merges the original reproducibility repository with four later correlation-analysis bundles. The adaptive-supercritical raw-pattern implementation is authoritative for correlation-dependent dynamical quantities. Earlier fixed-gain and raw-bifurcation implementations, and the distinct GPI-bifurcation implementation, are retained under `legacy/correlation/` rather than silently discarded. Original bundle figures and CSVs used for regression/reference are under `reference_artifacts/` and are never inputs to the canonical pipeline. See `PROVENANCE.md` and `reports/RESTRUCTURING_REPORT.md`.

## Focused smoke test for the 2026-10-07 protocol update

The capacity-grid and correlation/gain phase-plane changes can be validated quickly with:

```bash
./scripts/smoke_protocol_updates.sh
```

This executes the modified experiment runners, regenerates their plots from saved CSVs, runs the relevant regression tests, and checks the repository CSV contracts. It does not replace the complete `run_all.sh --smoke` workflow.
