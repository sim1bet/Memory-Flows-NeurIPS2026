# Adaptive-supercritical correlation sweep

The preceding implementation evaluated asymptotic escape events at one fixed
`reference_kappa` for every correlation value. This is invalid whenever that
gain is not above the empirical correlation-dependent critical gain.

The corrected workflow is two-pass:

1. Estimate `kappa_crit(epsilon, realization)` by gain scan and bisection.
2. Evaluate the long-run trajectory at

   `kappa_eval = r * kappa_crit`, with `r > 1`.

By default, `r` is calibrated from the epsilon=0 realizations so that their
mean evaluation gain equals `--reference-kappa` (default 15):

   `r = reference_kappa / mean[kappa_crit(epsilon=0)]`.

An explicit ratio can instead be supplied through `--supercritical-factor`.
Trials with censored critical gains, or gains above
`--evaluation-kappa-max`, are not assigned artificial escape-time or saliency
values.

For each valid realization, the orthogonal fixed-point-map references are
recomputed at that same `kappa_eval`. Hence the empirical and orthogonal
curves are compared at matched gain rather than against one fixed horizontal
reference.

The operational epsilon_max is now based on plateau-qualified sustained cycles
at matched relative supercriticality.

Paper-scale example:

```bash
python LimitCycle_raw_correlation_fixedpoint_adaptive_kappa.py \
  --pattern-sets 24 \
  --epsilon-points 21 \
  --reference-kappa 15 \
  --kappa-max 60 \
  --evaluation-kappa-max 150 \
  --kappa-scan-points 24 \
  --kappa-bisection-steps 8 \
  --asymptotic-cycles 4 \
  --terminal-cycles 2 \
  --dt-fast 0.05 \
  --output raw_correlation_adaptive_paper
```
