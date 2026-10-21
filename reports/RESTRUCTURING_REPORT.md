# Final restructuring report

## A. Repository summary

The merged release uses the existing `slow_fast_idp_hopfield` package as the canonical foundation and incorporates the later correlation-analysis bundles without duplicating model logic. Canonical experiments are config-driven, write CSV/JSON only, and are plotted in a separate stage. Shared activation, graph, pattern, theory, event, simulation, statistics, and I/O definitions live in `src/idp_hopfield/`.

The final paper path adds `correlation_dynamics`, which estimates the raw-pattern correlation-dependent critical gain first and then evaluates escape time and asymptotic saliency at a strictly supercritical, realization-matched gain. The older fixed-gain analysis is not used for paper claims near the correlation limit.

Canonical flow:

```text
configs/paper/*.yml
        -> experiments/run_*.py
        -> data/raw/paper/<experiment>/*.csv|json
        -> plotting/plot_*.py
        -> figures/data/*.csv
        -> figures/paper/*.{pdf,png}
        -> plotting/tikz/*.tex (optional final rendering)
```

`run_all.sh` executes this dependency order from the repository root.

## B. Source-bundle inventory

| Supplied bundle | Contents / original role |
|---|---|
| `slow_fast_idp_hopfield_repository` | Existing integrated repository, smoke outputs, figures, logs, configs and shared package. |
| `slow_fast_idp_hopfield_source` | Source-only duplicate of the integrated repository. All source files matched the repository archive except generated manifest/tree files. |
| `raw_correlation_bifurcation_bundle` | First raw-pattern empirical `kappa_crit` and escape-time analysis. |
| `raw_correlation_fixedpoint_bundle` | Added terminal-cycle escape and asymptotic saliency estimates, but used a fixed evaluation gain. |
| `raw_correlation_adaptive_kappa_bundle` | Corrected two-pass analysis: estimate `kappa_crit(epsilon)` and evaluate at matched relative supercriticality. |
| `correlated_gpi_bifurcation_bundle` | Separate GPI-based correlated-pattern bifurcation diagnostic. |

Generated CSV/PDF/PNG artifacts in the later bundles were classified as reference artifacts rather than canonical inputs.

## C. Consolidation decisions

- **Canonical base:** retained the structured `slow_fast_idp_hopfield` repository rather than rebuilding its already shared numerical layer.
- **Authoritative raw-correlation dynamics:** `raw_correlation_adaptive_kappa_bundle` supersedes both earlier raw-correlation variants because it enforces `kappa_eval > kappa_crit(epsilon)` for every uncensored realization.
- **Refactoring:** adaptive raw-correlation numerical logic moved to `src/idp_hopfield/correlation_dynamics.py`; YAML loading/output is in `experiments/run_correlation_dynamics.py`; plotting is in `plotting/plot_correlation_dynamics.py`.
- **Shared utilities:** sibling-file dynamic imports were removed. The canonical code now uses `activations.py`, `graphs.py`, `patterns.py`, and `theory.py`.
- **Fixed-gain raw correlation:** retained under `legacy/correlation/` because it documents an important superseded methodology; it is not on the canonical execution path.
- **Initial raw bifurcation sweep:** retained as provenance only.
- **GPI bifurcation:** retained as a scientifically distinct exploratory implementation. It is not silently equated with the raw-pattern robustness experiment and is excluded from `run_all.sh`.
- **Reference outputs:** original adaptive and GPI result artifacts are placed in `reference_artifacts/correlation/` and never consumed by canonical plotting or experiments.
- **Duplicate source archive:** no duplicate executable tree was retained because the source and repository archives matched at source-file level.

## D. Reproducibility improvements

- Added paper/smoke YAML configurations for adaptive correlation dynamics.
- Added explicit root seed `20260803` and deterministic child-seed policy.
- Removed sibling-file and working-directory dependence from the adaptive analysis.
- Separated simulation from plotting.
- Added raw trial, event, summary, resolved-config, and theory JSON outputs.
- Added output-contract validation for the new experiment.
- Added unit tests for the correlation-dynamics theory interface and fast/slow time conversion.
- Added exact regression against the original adaptive bundle.
- Added a TikZ/PGFPlots source consuming exported CSV data.
- Preserved legacy and original reference artifacts in explicitly non-canonical directories.

## E. Graphical revision report

| Figure family | Issue in source bundles | Canonical treatment | Formats | Data changed? |
|---|---|---|---|---|
| Model comparison | Independent styling in original scripts | Existing shared plotting system retained | PDF, PNG, TikZ/CSV | No |
| Limit cycles | Multiple cases needed common hierarchy | Existing consolidated multi-case plot retained | PDF, PNG, CSV | No |
| Escape-time validation | Stability loss and visible switch could be conflated | Existing corrected plot keeps them distinct | PDF, PNG, TikZ/CSV | No |
| Return maps / implicit curves | Theory curves had separate script conventions | Shared theory and plotting retained | PDF, PNG, TikZ/CSV | No |
| `T_esc(kappa)` | Theory/finite-epsilon comparison required matched axes | Existing consolidated plot retained | PDF, PNG, TikZ/CSV | No |
| Noise-dependent delay | Negative premature-switch gaps must remain visible | Existing plot retains signed delay | PDF, PNG, TikZ/CSV | No |
| Capacity / out-degree | Multiple task definitions needed explicit labeling | Existing separate cycle/fanout panels retained | PDF, PNG, TikZ/CSV | No |
| Raw/GPI correlation retrieval | Rule comparison required paired interpretation | Existing correlation plot retained | PDF, PNG, TikZ/CSV | No |
| **Correlation dynamics** | Latest bundle mixed simulation and plotting; fixed-gain predecessor could mislead | New four-panel figure separates sustained probability, `kappa_crit`/evaluation gain, matched `T_esc`, and matched asymptotic `Z`; panel labels and uncertainty retained | PDF, PNG, TikZ/CSV | No |

The new correlation-dynamics figure was visually inspected at raster resolution. The smoke plot is intentionally sparse and is not a paper estimate.

## F. Validation report

Executed successfully:

- Python compile checks for `src`, `experiments`, `plotting`, `tests`, and `scripts`.
- Shell syntax validation of `run_all.sh`.
- Parsing of all 19 YAML configurations.
- 13 pytest tests.
- 21 required CSV output contracts.
- New adaptive correlation experiment and plot in smoke mode.
- Compilation of the new `correlation_dynamics.tex` with `latexmk`.
- Exact numerical regression of the refactored adaptive workflow against the original bundle on a common reduced configuration.

Regression tolerance/result: maximum absolute difference `0.0` for the five primary summary quantities on jointly finite entries.

The paper-scale Monte Carlo configurations were not run in full in the delivery environment. The monolithic smoke master command reached the end of all plotting stages but was externally terminated by the sandbox wall-clock limit before its final tests/contracts stage; those stages were then executed independently and passed.

## G. Unresolved issues

- Manuscript-only artwork and numerical panels already marked unresolved in `FIGURE_MAP.md` remain unresolved; no data were fabricated.
- The GPI bifurcation analysis is preserved but not promoted to a paper result because it changes the synaptic/decoding rule relative to the raw-correlation question.
- Paper-scale correlation and capacity sweeps can be computationally expensive and were not executed in full here.
- `epsilon=1` in the biased pattern generator is structurally degenerate (identical memories); memory-specific `kappa_crit`, escape, and fixed-point quantities are therefore censored/undefined rather than forced to finite values.
- TeX compilation requires an external TeX distribution with `latexmk`, `standalone`, and `pgfplots`.

## H. Exact reproduction commands

Environment:

```bash
conda env create -f environment.yml
conda activate slow-fast-idp-hopfield
```

Smoke validation:

```bash
./run_all.sh --smoke --workers 1 --skip-tex
```

Paper pipeline:

```bash
./run_all.sh --workers 8
```

Adaptive raw-correlation dynamics only:

```bash
python experiments/run_correlation_dynamics.py \
  --config configs/paper/correlation_dynamics.yml \
  --output data/raw/paper/correlation_dynamics

python plotting/plot_correlation_dynamics.py \
  --input data/raw/paper/correlation_dynamics \
  --output figures/paper/correlation_dynamics \
  --data-dir figures/data
```

Compile that TikZ figure after generating paper data:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd plotting/tikz/correlation_dynamics.tex
```

Commands for every other experiment and manuscript mapping are listed in `README.md` and `FIGURE_MAP.md`.

## Protocol update: 2026-10-07

The canonical capacity workflow was refined into four explicitly configured protocols. The manuscript fanout curves now use only `d={2,6,10,14,18}`, `N={256,512,1024}`, and `P/N in [0.05,0.30]`. A separate dense fanout experiment covers every integer `d=2,...,18` for transition-probability phase planes. The scaled-cycle experiment now uses only `N={256,512,1024}` over the same low-load interval, while a separate fixed-cycle test uses `N=256`, `L=5`, and `P/N={80,85,...,120}`. The fixed-cycle integrator exploits the exact zero-saliency sparsity of disconnected memories for the fast update but still evaluates all stored memories before accepting a retrieval event.

A new raw correlated-pattern phase-plane experiment evaluates `T_esc` over `(epsilon,kappa)` with gains above the orthogonal activation-specific critical gain. Cells in which no correct transition occurs retain undefined `T_esc` and are rendered in gray rather than being assigned an artificial value. The plotting stage exports the complete transition-probability metadata with the heatmap data.

The focused end-to-end smoke workflow `scripts/smoke_protocol_updates.sh` completed successfully after these changes. The complete legacy-inclusive `run_all.sh --smoke` was also exercised stage-by-stage; the execution harness terminates a single long container invocation before all legacy plots finish, so those remaining stages were validated individually. No paper configuration was reduced for validation.
