# Validation record

This record describes checks executed in the delivery environment. The paper-scale Monte Carlo configurations were not substituted with reduced settings and were not run in full here; the complete smoke workflow exercises the same code paths with smaller grids.

## Environment used

```text
Python         3.13.5
NumPy          2.3.5
SciPy          1.17.0
pandas         2.2.3
Matplotlib     3.10.8
PyYAML         6.0.3
tqdm           4.67.3
Numba          0.65.1
threadpoolctl  3.6.0
pytest         9.0.2
```

These versions are pinned in `environment.yml` and `pyproject.toml`.

## Commands executed

```bash
PYTHONPATH=src python -m compileall -q src experiments plotting tests scripts
PYTHONPATH=src pytest -q
./run_all.sh --smoke --workers 1 --skip-tex
PYTHONPATH=src python scripts/check_outputs.py data/raw/smoke
```

The test suite reports:

```text
11 passed
```

The CSV contract checker validates 18 required primary tables. The complete smoke output contains 23 CSV files, 26 PDF/PNG figures, and 11 standalone TikZ sources.

All TikZ files were compiled with

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd <source.tex>
```

using smoke-generated figure data copied temporarily into the paper-data path. All 11 sources compiled successfully. Build products and temporary paper-data copies were removed afterward.

## Numerical regression checks

### Escape time

The corrected tanh experiment at `kappa = 15` produced three complete stability-loss events after the initial branch entry. Excluding the prescribed first transition when estimating the steady regime:

```text
empirical mean T_esc              1.8023170
eventwise theoretical mean T_esc  1.8022962
mean absolute relative error      0.0120 %
empirical entry amplitude          12.22635
stable theoretical fixed point     12.47372
```

The visible switch remains later than the stability crossing and is stored separately.

### Activation-specific critical gain

The tanh return-map saddle-node was recovered as

```text
kappa_crit = 5.8913851453
T_esc,crit = 0.6279989294.
```

The unit test also verifies the SmoothF value near `12.74`.

### Kappa sweep

All four smoke-grid terminal windows passed the configured stationarity criteria. The relative finite-epsilon discrepancy decreased from approximately `27.8%` immediately above the tanh saddle-node to `0.52%` at `kappa = 20`, consistent with the expected amplification of finite-timescale effects near criticality.

### Noise sweep

With common random paths and per-step fast-state noise, the median ejection gap decreased monotonically over the smoke grid:

```text
noise 1e-6   median gap  0.22795
noise 1e-3   median gap  0.16745
noise 1      median gap -0.26668
```

The negative high-noise value is retained and classified as premature switching.

### Capacity paths

The software smoke grid produced 12 trial rows and 12 grouped probability rows spanning:

```text
activations: tanh, SmoothF
experiments: L=P/3 scaled cycle, fixed-outdegree fanout
outdegrees: 2, 3
loads: two P/N values
```

The reduced replication is not a capacity estimate; it validates nested graph construction, event classification, pattern-set aggregation, and figure generation.

### Controlled correlation and GPI

The smoke run produced 24 paired trial rows and 12 aggregate rows for correlations `0.0`, `0.4`, and `0.6`, both activations, and both overlap rules. GPI completed the five-transition cycle in every smoke trial. The raw rule degraded at the highest correlation. Pseudoinverse reconstruction residuals remained near machine precision.

## Unit-test coverage

The tests verify:

1. tanh and SmoothF branch equations;
2. activation-specific stability thresholds;
3. tanh and SmoothF critical gains;
4. a reference tanh escape time;
5. the orthogonal pseudoinverse limit;
6. exact full-rank GPI decoding;
7. the expected overlap of the biased correlation generator;
8. graph orientation `A[target, source]`;
9. the fast-to-slow clock conversion;
10. simultaneous explicit-Euler use of the old state.

## Scope of validation

The executed smoke results establish software correctness, deterministic file production, and consistency of the central numerical definitions. Final paper claims should use the outputs of `configs/paper/`, with convergence subsets at finer `dt_fast` where noted in the README.

## Final merge validation (2026-10-05)

The merged repository adds the adaptive-supercritical raw-correlation workflow while preserving the earlier canonical experiments.

Executed checks:

```bash
PYTHONPATH=src python -m compileall -q src experiments plotting tests scripts
PYTHONPATH=src pytest -q
python scripts/check_outputs.py data/raw/smoke
bash -n run_all.sh
# all YAML files parsed with PyYAML
```

Results:

```text
13 tests passed
21 required primary CSV contracts validated
26 smoke CSV files present
28 smoke PDF/PNG files present
12 TikZ sources present
all YAML configurations parsed successfully
run_all.sh shell syntax passed
```

The new `correlation_dynamics.tex` source was compiled successfully with `latexmk` using smoke-generated CSV data.

### Adaptive-correlation regression

The original `raw_correlation_adaptive_kappa_bundle` and the refactored canonical implementation were executed on the same reduced configuration (`N=256`, `P=15`, correlations `0, 0.3, 0.5, 0.7, 1`, two pattern sets, `dt_fast=0.1`). On all jointly finite entries, maximum absolute differences were exactly zero for:

```text
mean_kappa_crit
mean_evaluation_kappa
plateau_fraction
mean_asymptotic_T_esc
mean_asymptotic_Z
```

This directly checks that separating simulation from plotting and replacing sibling-file imports by shared package utilities did not change the reported correlation-dynamics results.

### Master-script execution note

`./run_all.sh --smoke --workers 1 --skip-tex` was executed twice in the delivery sandbox. In both attempts it progressed through all experiment stages and all plotting stages, including the newly added correlation-dynamics plot, before the external tool wall-clock limit terminated the shell immediately after the last plotting stage. The remaining test suite and output-contract stages were then executed separately and passed as reported above. This is an execution-environment timeout, not a detected repository failure; no claim is made that the single master command completed within the sandbox time limit.

## 2026-10-07 protocol update

The capacity and correlation phase-plane extensions were smoke-tested after restructuring.

Validated directly:

- `experiments/run_capacity.py` with `configs/smoke/capacity.yml`;
- `plotting/plot_capacity.py`, including selected fanout degrees, fanout phase planes, scaled cycles, and fixed `L=5` cycles;
- `experiments/run_correlation_phase_plane.py` with `configs/smoke/correlation_phase_plane.yml`;
- `plotting/plot_correlation_phase_plane.py`;
- `pytest -q tests`;
- `python scripts/check_outputs.py data/raw/smoke` after generating all required experiment outputs.

The modified capacity smoke produced all four experiment families (`scaled_cycle`, `fixed_cycle`, `fanout`, `fanout_phase`) and both activations. The correlation phase plane produced finite `T_esc` values only in cells with correct transitions and left no-transition cells undefined for gray masking.

A single `run_all.sh --smoke --workers 1 --skip-tex` invocation was also exercised repeatedly. The execution harness terminates long container calls before the complete legacy plotting suite finishes, although every stage was then executed and validated separately. This is a harness wall-clock limitation, not a detected scientific or software failure. Paper configurations were never reduced to accommodate smoke testing.

The focused modified-path smoke command

```bash
./scripts/smoke_protocol_updates.sh
```

completed successfully end-to-end after the final code changes, including experiment generation, CSV persistence, plotting, targeted regression tests, and global CSV contract validation.


## Editorial revision, 2026-10-08

The 19 redesigned figures were rendered in PDF, SVG and 300 dpi PNG from saved inputs. The 23 existing CSV contracts passed. Seventeen zero-argument test functions passed by direct invocation, including four added scientific-encoding tests. The authoring runtime lacks pytest and tqdm, so a complete pytest execution and the three capacity protocol tests were not performed. No paper-scale numerical campaign was run. All original model, config, raw-data, reference and legacy files were byte-checked and unchanged. See `reports/VISUAL_DESIGN.md` and `reports/visual_qa/` for the visual checks, colormap measurements and limitations.


## Requested adjustments, 2026-10-09

The modified smoke capacity, correlation/gain and model comparison experiments were rerun. All 27 test functions passed by direct invocation and all 23 CSV contracts passed. Current protocol details, timing outcomes, noise conventions and mesh sizes are in `reports/REVISION_20261009.md`. This supersedes the preceding unchanged-data statement for the affected experiments. No paper-scale campaign was run.
