# Source provenance and consolidation decisions

The repository was constructed from the final uploaded numerical snippets. Duplicate functions were compared before consolidation; the corrected event and clock definitions were retained.

| Uploaded source | Consolidated destination |
|---|---|
| `LimitCycle_bipartite_Tesc_corrected(2).py` | `src/idp_hopfield/activations.py`, `simulation.py`, `events.py`, `theory.py`; `experiments/run_escape_time.py` |
| `LimitCycle_bipartite_Tesc_corrected_with_kappa_sweep(1).py` | thin integration behavior replaced by `run_all.sh`; no duplicated model code retained |
| `tesc_kappa_sweep(1).py` | `experiments/run_kappa_sweep.py`, shared theory and plateau estimator |
| `LimitCycle_bipartite_Tesc_corrected_with_kappa_and_noise_sweeps(1).py` | orchestration replaced by `run_all.sh` |
| `tesc_kappa_noise_sweep(1).py` | `experiments/run_noise_sweep.py`; shared critical-gain theory |
| `LimitCycle_capacity_generalized(2).py` | scaled-cycle protocol in `experiments/run_capacity.py` |
| `LimitCycle_capacity_outdegree_2_10(1).py` | nested fixed-outdegree fanout protocol in `run_capacity.py` |
| `LimitCycle_correlated_gpi_sweep(1).py` | `experiments/run_correlation.py`, `patterns.py` |
| `test_correlated_gpi(1).py` | expanded tests in `tests/test_core.py` |

## Harmonized conventions

- `x` always denotes the fast state; `z` always denotes the slow state.
- `A[target, source] = 1` everywhere.
- `dt_fast` is a step on the dimensionless fast clock.
- `dt_slow = dt_fast * tau_fast / tau_slow` is stored in outputs.
- The slow update and fast update are simultaneous explicit-Euler increments from the old state.
- `T_esc` is the source stability crossing; `T_switch` is a separate persistent-overlap event.
- Pattern-set replication, not source-level replication, is the sampling unit for capacity intervals.
- GPI uses `pinv(Xi.T @ Xi / N, hermitian=True)` and records rank, condition number, and reconstruction residual.

The original scripts are not copied into the executable package because they contain repeated theory, plotting, and global-state code. Their scientific functionality is represented by the modules above, with configuration values externalized into YAML.

## 2026-10 correlation-bundle merge

The following additional bundles were inspected during the final merge:

| Bundle | Role | Consolidation decision |
|---|---|---|
| `slow_fast_idp_hopfield_repository` | Existing canonical repository plus generated smoke artifacts | Used as repository skeleton. |
| `slow_fast_idp_hopfield_source` | Source-only copy of the same repository | Source files matched the repository copy except generated `MANIFEST.sha256` and `REPOSITORY_TREE.txt`; no second implementation retained. |
| `raw_correlation_adaptive_kappa_bundle` | Latest raw-pattern correlation dynamics using matched supercritical gain | **Authoritative**; refactored into `src/idp_hopfield/correlation_dynamics.py`, `run_correlation_dynamics.py`, and separate plotting. |
| `raw_correlation_fixedpoint_bundle` | Earlier fixed-evaluation-gain fixed-point analysis | Superseded scientifically; retained under `legacy/correlation/`. |
| `raw_correlation_bifurcation_bundle` | Earlier empirical `kappa_crit`/escape sweep | Superseded by adaptive workflow; retained under `legacy/correlation/`. |
| `correlated_gpi_bifurcation_bundle` | GPI-specific bifurcation diagnostics | Scientifically distinct exploratory method; retained under `legacy/correlation/` and excluded from the raw-pattern paper path. |

### Numerical preservation check

A matching reduced configuration was run through both the original adaptive source and the refactored canonical implementation. The following summary columns agreed exactly (maximum absolute difference `0.0` on all jointly finite entries): `mean_kappa_crit`, `mean_evaluation_kappa`, `plateau_fraction`, `mean_asymptotic_T_esc`, and `mean_asymptotic_Z`.

## 2026-10-07 protocol refinement

The canonical capacity runner was extended rather than replaced. Its authoritative paper configuration now separates scaled-cycle, fixed-`L=5`, selected-fanout, and dense fanout-phase protocols. The fixed-cycle grid intentionally supports `P/N > 1`. A new raw-correlation phase-plane runner reuses the canonical full-Jacobian event detector from `src/idp_hopfield/correlation_dynamics.py`; it does not introduce a second escape-time definition.


## Editorial visualization revision

The ten original plotting entry points were redesigned. Shared style and scientific-encoding helpers live in `plotting/common.py`; numerical model code is unchanged. Added files include the analytical reduced-source-branch exporter/plotter, a render-only driver, vector-preserving atlas builder, accessibility diagnostics and four encoding tests. `run_all.sh` invokes the new rendering driver after the existing experiment stages. The original TikZ sources and historical simulation outputs are retained. A source-hash manifest accompanies the revised figure outputs.


## Requested adjustments, 2026-10-09

The user requested revised capacity loads, denser phase-plane sampling, a distinct scalar palette, visible fanout x axes, escape-aware correlation reliability horizons and fast noise in one model-comparison case. Those experiment configurations and affected smoke datasets are intentionally changed in this revision. The core correlation simulator adds timing diagnostics without changing numerical updates. See `reports/REVISION_20261009.md`.
