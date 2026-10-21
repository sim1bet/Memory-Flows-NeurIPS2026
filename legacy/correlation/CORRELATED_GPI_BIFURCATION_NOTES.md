# Critical gain and escape time under controlled pattern correlation

## Scope

This extension augments `LimitCycle_correlated_gpi_sweep.py` with a second,
independent experiment that computes the critical slow gain and asymptotic
escape time for correlated binary memories. The completed run uses

- `N = 256` neurons;
- `P = 15` memories;
- tanh neural activation;
- diagonal saliency activation `phi(z) = |z|`;
- biased Rademacher patterns with correlation parameter
  `epsilon = 0.0, 0.1, ..., 0.7`;
- 24 independently sampled pattern matrices per correlation;
- reference gain `kappa = 15`.

For the biased generator,

\[
\Pr(\xi_i^\mu=+1)=\frac{1+\epsilon}{2},
\qquad
\mathbb E\!\left[\frac{(\xi^\mu)^\top\xi^\nu}{N}\right]
=\epsilon^2\quad(\mu\ne\nu).
\]

## GPI dynamics

Let

\[
\Xi=[\xi^1,\ldots,\xi^P]\in\mathbb R^{N\times P},
\qquad
O=\frac{1}{N}\Xi^\top\Xi,
\]

and define the generalized-pseudoinverse memory coordinate

\[
q(x)=O^+\frac{1}{N}\Xi^\top\psi(x).
\]

The implementation uses

\[
\dot x=-x+\Xi\operatorname{diag}(|z|)q(x),
\qquad
\dot z=-z+\kappa A q(x).
\]

The raw-overlap rule remains available for retrieval experiments, but the
bifurcation calculation below is an exact statement only for the GPI rule.

## Why the GPI bifurcation curves are flat

Assume that `rank(Xi)=P`, so `O^+=O^{-1}`. On the source-memory branch

\[
x=\gamma\xi^\nu,
\]

oddness of `psi` and the binary entries of the memories imply

\[
\psi(x)=\psi(\gamma)\xi^\nu.
\]

Hence

\[
q(x)
=O^{-1}Oe_\nu\psi(\gamma)
=e_\nu\psi(\gamma).
\]

The branch equation is therefore exactly

\[
\gamma=z_\nu\psi(\gamma),
\]

and the source-successor slow flow is exactly

\[
\dot z_\nu=-z_\nu,
\qquad
\dot z_\mu=-z_\mu+\kappa\psi(\gamma(z_\nu)).
\]

These are the same scalar equations obtained in the orthogonal derivation.

At a source branch, `Dpsi(x)=psi'(gamma) I`. The fast Jacobian is

\[
J_f=-I+\psi'(\gamma)
\frac{1}{N}\Xi\operatorname{diag}(|z|)O^{-1}\Xi^\top.
\]

Its nontrivial feedback eigenvalues are those of a matrix similar to

\[
\psi'(\gamma)\operatorname{diag}(|z|).
\]

Thus the source loses stability at the same saliency threshold as in the
orthogonal model. Consequently, for every full-column-rank pattern matrix,

- the scalar saliency return map is unchanged;
- `kappa_crit` is unchanged;
- `T_esc` at the saddle-node is unchanged;
- `T_esc` on the attracting fixed-point branch at any fixed supercritical
  `kappa` is unchanged.

Correlation changes the conditioning of `O`, but not the exact bifurcation
quantities until rank loss or numerical pseudoinverse truncation occurs.

## Implicit equations used

For `psi(x)=tanh(x)`, let `s` be the source saliency at escape and let `u(s)`
be the positive solution of

\[
u=s\tanh u.
\]

The successor stability threshold is

\[
\bar Z=h(s)=\cosh^2 u(s),
\]

and

\[
X=\frac{\bar Z}{s},
\qquad
T_{\rm esc}=\log X.
\]

The fixed-point gain associated with `s` is

\[
\kappa(s)=
\frac{\bar Z}
{\displaystyle
X^{-1}\int_1^X
\tanh\!\left(u\!\left(\frac{\bar Z}{r}\right)\right)\,dr
}.
\]

The critical gain is

\[
\kappa_{\rm crit}=\min_{s>1}\kappa(s).
\]

For a prescribed `kappa > kappa_crit`, the larger solution of
`kappa(s)=kappa` gives the attracting saliency-map fixed point and its
asymptotic escape time.

## Completed tanh results

The implicit calculation gives

\[
\kappa_{\rm crit}=5.8913851431,
\]

\[
T_{\rm esc}(\kappa_{\rm crit})=0.6279989294,
\]

with

\[
s_{\rm crit}=1.3310334800,
\qquad
\bar Z_{\rm crit}=2.4941665430.
\]

At the reference gain `kappa=15`,

\[
T_{\rm esc}^{\infty}=1.8221034597,
\]

\[
s=2.0168176370,
\qquad
\bar Z=12.4737233729.
\]

All eight correlation cells and all 24 pattern matrices per cell were full
rank. Therefore both requested curves are horizontal over
`epsilon in [0, 0.7]`.

The mean Gram condition number nevertheless increases from approximately
`2.27` at `epsilon=0` to `25.95` at `epsilon=0.7`. Across the complete sweep,
GPI identity, branch, stability-boundary, and spectrum residuals remain at
approximately `1e-15` to `1e-14`.

## Interpretation

The flat curves are a mathematical result, not a failed sensitivity sweep.
The exact pseudoinverse removes the Gram matrix from both the reduced memory
coordinate and the nontrivial fast-Jacobian spectrum when `O` is invertible.
The meaningful correlation-dependent diagnostic for the exact GPI model is
therefore conditioning, not a shifted ideal `kappa_crit`.

The raw-overlap correlated model is different. Once non-source saliencies are
driven, a pure memory branch is generally no longer invariant, so the PDF's
one-dimensional implicit map does not directly define a raw-rule
`kappa_crit`. Any raw-rule threshold should be labeled empirical and estimated
from full simulations, rather than presented as a consequence of the scalar
orthogonal theorem.

## Files produced

- `correlated_bifurcation_trials.csv`: one row per pattern matrix;
- `correlated_bifurcation_summary.csv`: one row per correlation value;
- `correlated_bifurcation_theory.json`: critical point and reference-gain
  values;
- `correlated_kappa_crit_Tesc.pdf/.png`: requested curves;
- `correlated_kappa_crit_Tesc_conditioning.pdf/.png`: Gram conditioning and
  algebraic residuals.

## Commands

Run only the requested bifurcation sweep:

```bash
python LimitCycle_correlated_gpi_bifurcation_sweep.py \
  --bifurcation-only \
  --n-neurons 256 \
  --n-memories 15 \
  --activation tanh \
  --correlations 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 \
  --pattern-sets 24 \
  --reference-kappa 15 \
  --output correlated_gpi_bifurcation
```

Run the original retrieval experiment and the new bifurcation experiment:

```bash
python LimitCycle_correlated_gpi_bifurcation_sweep.py \
  --n-neurons 256 \
  --n-memories 15 \
  --activation tanh \
  --reference-kappa 15 \
  --output correlated_gpi_complete
```

The implementation also supports `--activation smoothf`. For SmoothF, the
corresponding full-rank GPI curves are likewise correlation-invariant.
