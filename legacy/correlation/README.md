# Correlation-analysis legacy sources

These files are preserved for provenance and are **not** imported by the canonical pipeline.

- `LimitCycle_raw_correlation_fixedpoint_adaptive_kappa.py` is the latest source bundle and the scientific authority for the raw-pattern adaptive-supercritical protocol. Its logic was refactored into `src/idp_hopfield/correlation_dynamics.py`, with simulation and plotting separated.
- `LimitCycle_raw_correlation_fixedpoint_sweep.py` is the superseded fixed-`kappa` implementation. It is retained because it documents why a fixed evaluation gain becomes invalid when `kappa_crit(epsilon)` rises.
- `LimitCycle_raw_correlation_bifurcation_sweep.py` is the earlier critical-gain/escape analysis, superseded by the fixed-point and adaptive implementations.
- `LimitCycle_correlated_gpi_bifurcation_sweep.py` is a scientifically distinct generalized-pseudoinverse diagnostic. It is retained as exploratory code because the paper's raw correlated-pattern robustness test does not use GPI.

Original smoke/reference outputs are under `reference_artifacts/correlation/`. They are never read by `run_all.sh`.
