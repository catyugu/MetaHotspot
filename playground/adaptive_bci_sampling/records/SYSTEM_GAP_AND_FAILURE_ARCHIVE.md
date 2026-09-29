# System-norm gap and failure archive

> Historical record: claims below that a Robin-box frequency plan is installed
> in the production extractor refer to a reverted implementation. The current
> extractor uses the original per-port bare-kernel frequency estimator.

This record answers two questions: **what is still missing** between the
fixed-shift certificate and the full-field transfer operator the research
target asks for
(**Part A**), and **which routes are already closed** (**Part B**).  The
positive statements about the certificate itself live in
`records/PORT_CERTIFICATE_AND_BUDGET.md`; the theory lives in `THEORY.md`.
Both parts are the original per-topic records, moved here unchanged so that no
measured number was lost in the merge; only the heading levels were adjusted.

Scope labels used throughout, and the limit each one puts on what a negative
entry may conclude:

```text
STATE           full-field / A-energy / state transfer norm    the research target
PORT-SYSTEM     H2 / Hankel / impulse-response system norm     the dynamic channel, still open
PORT-FIXED-S    Z(s; mu) at one real shift                     what the box certificate covers
PORT-STEP       sampled step-response metric                   measured diagnostic only
COST            N_FOM / N_op / wall time / memory
CORRECTION      measurement bug or mis-defined quantity
```

**Conclusion limit of a scope label.**  An entry labelled `PORT-FIXED-S` or
`PORT-STEP` proves only that *that scheme fails under that diagnostic*.  It does
**not** prove the scheme cannot meet the port-Hankel / impulse-response system
norm, and it does **not** veto reusing the same parameter points or the same basis
construction under the new objective.  Reading the former as the latter is the
logical leap this file exists to prevent, and Part A is the measured reason it
is a leap: the fixed-`s` collocated square identity does **not** transfer
quadratically to dynamic port norms.

**Target restoration.**  The research target is the full-field transfer operator
`X(p, s) = A(p, s)^-1 G`, so a full-field / A-energy / state-norm failure is a
failure of the acceptance criterion itself; its fixed-shift implementation is the
collocated port defect of `THEORY.md` proposition 1.  The 2026-09 narrowing that
demoted the state-level channel to a proof device is reverted (`THEORY.md` 0.1).
No entry needs re-labelling for it: the re-review had already checked every entry
in Part B and each carries an independent port-transfer, system-norm or cost
failure, so restoring the state-level criterion can only strengthen a verdict.
Two routes stay excluded and are **not** reopened by it: the Robin
solution-manifold n-width rate proposition (sampling-size optimality is still
answered by `N_FOM` and the cost structure) and the dynamic semi-group
state-energy vendor channel (its norm is undefined in the retrieved pages, entry
A6).

Verdicts are only these four, and every entry names exactly one (or a pair):

```text
被反例证伪       mathematically wrong, or its premise fails on the measured data
界太松           the inequality holds but cannot drive a 1e-3 decision
成本不可行       correct, but the cost structure (solves, setups, wall time, memory) is unacceptable
测量更正         the conclusion came from a measurement bug or a mis-defined quantity
```

---

## Part A - the measured gap: a matching port defect does not transfer quadratically

Negative result on the quadratic bridge, from the artificial problem the bridge
has to survive before any of the heavier machinery is justified.  Scope: every
quantity below is a **port system-norm** quantity `[PORT-SYSTEM]` (`H2` and
Hankel of the port transfer), not a state or full-field error.

### A1. What is asked

The only open P0 item is the dynamic bridge: the certificate controls matching
resolvent defects `delta_j`, and the vendor guarantee is stated for the impedance
(port-Hankel) channel.  The exact MPMM space has a Hermite
structure at the matching shifts, so the *value* defect there is `O(delta^2)`.
The question is whether the *dynamics* inherit that square, or degrade to first
order.  Gates, all evaluated on a fixed problem along one controlled family
`V_theta -> U` with `delta_*(V_theta) -> 0`:

```text
quadratic bridge refuted       e_H2 / delta_* -> c > 0      (e_H2 / delta_*^2 -> inf)
sharp-baseline form refuted    excess / delta_* -> c > 0,
                               excess = max over both signs of
                                        ||H - H_V|| / ||H|| - ||H - H_U|| / ||H||
vendor impedance channel hit   e_Hankel / delta_* -> c > 0
```

`[STATE]` the run also measured the state impulse energy `e_state`; that column was
dropped in the 2026-09 scope narrowing and is not in the current output.  The
field-level fixed-shift criterion is an acceptance target again (see `THEORY.md`
0.1), but this dynamic column is not one, so it is not restored here; the verdict
below rests on the port-level counterexample, which does not depend on it.

The excess is taken over *both* rotation signs because the first-order Frechet
derivative flips sign with `Q -> -Q`: if it is nonzero, at least one branch must
raise the total error above the exact-MPMM baseline.

### A2. Model and definitions

Mass-whitened so the pencil has one symmetric parameter matrix,

```text
B = B^T > 0,   H(s) = f^T (s I + B)^-1 f,   x(sigma) = (B + sigma I)^-1 f,
```

with `C = I`, eigenvalues `lambda_i = kappa^u_i` for `u` from the preregistered
families, and the delivered elliptic plan's shifts for `[1, kappa]` at
`epsilon = 1e-3` (`mpmm_elliptic_shift_count` / `mpmm_elliptic_shifts`, the same
generator the box plan uses).  The 4x4 problem takes the plan's highest and
lowest shift, the 8x8 problem four shifts spread over the band.

```text
delta_j^2(V) = ||x_j - x_V,j||^2_(B + sigma_j I) / ||x_j||^2_(B + sigma_j I),
delta_*      = max_j delta_j,
U            = orth[x(sigma_j)],       V_pm(theta) = U cos theta +/- Q sin theta,
e_H2         = ||H_U - H_V||_H2 / ||H||_H2,
e_Hankel     = ||H_U - H_V||_Hankel / ||H||_Hankel.
```

`Q` is a fixed orthonormal basis of the orthogonal complement of `U` (2 columns
for 4x4, 4 for 8x8, so `V_pm` has the same dimension as `U`); `theta` runs over
`1e-1 ... 1e-6`.  The moment space itself comes from a truncated SVD of the
snapshot matrix, not a bare QR, so a rank-deficient snapshot set (repeated
eigenvalues) does not silently contribute padding directions.

All dynamic quantities are exact and port-level: the `H2` distance from a Lyapunov
solve on the block system that carries both semigroups, the Hankel norm from the
largest Hankel singular value of the difference realization (controllability and
observability Lyapunov equations).  No frequency grid and no time stepping enters
any verdict.

Spectrum families (`u_i`, `lambda_i = kappa^u_i`): log-uniform, endpoint-cluster,
low-cluster, repeated.  Sources: `f` flat, and `f_i ~ lambda_i^(1/4)`, which
balances the per-mode `f_i^4 / (2 lambda_i)` H2 contribution.

### A3. Results at the delivered box condition number

Actual 5 mm box plan: `kappa = 4.1400e+01 / 4.2795e-05 = 9.674e+05`.  Orders are
the fitted local exponents over the smallest rotations above the double-precision
floor, for the `+` branch (the `-` branch behaves the same); the ratio columns are
the smallest-`theta` values of the sharp-baseline excess.

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.30e-05 .. 9.58e-01 | 1.96 | 0.33 | 1.41 | 5.361e-05 | 2.572e-02 | 3.633e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.25e-05 .. 9.54e-01 | 1.72 | 2.14 | 1.57 | 1.844e-04 | 1.210e-02 | 1.640e-05 |
| 4x4 | log-uniform | flat | 2/2 | 3.94e-05 .. 9.71e-01 | 1.00 | 0.83 | 1.01 | 1.336e-02 | 6.598e-02 | 4.330e-04 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 5.64e-06 .. 4.18e-01 | 1.00 | 1.00 | 1.00 | 4.516e-01 | 1.384e+00 | 2.166e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.90e-07 .. 9.88e-02 | 0.36 | 1.34 | 1.00 | 1.595e-02 | 1.093e-02 | 5.166e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.87e-02 | 0.72 | 1.36 | 1.00 | 1.193e-02 | 9.101e-01 | 3.473e-04 |
| 4x4 | repeated | flat | 2/2 | 3.24e-05 .. 9.55e-01 | 1.00 | 0.72 | 2.53 | 4.955e-04 | 7.343e-03 | -3.237e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.23e-05 .. 9.53e-01 | 0.79 | 0.26 | 2.26 | 6.170e-04 | 2.575e-02 | -6.348e-05 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.44e-04 .. 9.99e-01 | 2.00 | 1.68 | 2.02 | 6.874e-04 | 7.121e-03 | 6.534e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.43e-04 .. 9.96e-01 | 2.00 | 1.72 | 2.00 | 4.854e-04 | 8.607e-04 | 4.862e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.31e-04 .. 8.96e-01 | 1.00 | 0.99 | 0.92 | 3.383e-02 | 6.429e-02 | 6.826e-04 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.44e-04 .. 8.22e-01 | 1.00 | 1.01 | 0.99 | 3.276e-02 | 1.248e-01 | 6.462e-03 |
| 8x8 | low-cluster | flat | 4/4 | 1.05e-06 .. 1.06e-01 | 1.00 | 1.21 | 1.00 | 1.487e-01 | 1.909e-01 | 1.019e-02 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.17e-07 .. 9.18e-02 | 0.98 | 0.94 | 1.00 | 3.995e-02 | 6.000e-02 | 1.198e-03 |
| 8x8 | repeated | flat | 2/4 | 5.73e-04 .. 1.00e+00 | 2.00 | 1.48 | 2.00 | 8.805e-04 | 4.963e-03 | 8.812e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.72e-04 .. 9.98e-01 | 2.00 | 2.14 | 2.00 | 4.538e-04 | 1.080e-03 | 4.516e-04 |

Two full series, to show that the plateaus are asymptotic and not a single lucky
point:

4x4 log-uniform flat

| theta | delta_* | e_H2 | e_H2/delta_* | e_Hankel | e_Hankel/delta_* | excess | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-01 | 9.712e-01 | 9.550e-01 | 9.834e-01 | 9.942e-01 | 1.024e+00 | 8.506e-01 | 8.758e-01 |
| 3e-02 | 7.836e-01 | 6.946e-01 | 8.864e-01 | 8.738e-01 | 1.115e+00 | 5.952e-01 | 7.596e-01 |
| 1e-02 | 3.681e-01 | 1.904e-01 | 5.172e-01 | 3.044e-01 | 8.271e-01 | 1.168e-01 | 3.173e-01 |
| 3e-03 | 1.239e-01 | 2.392e-02 | 1.931e-01 | 4.105e-02 | 3.313e-01 | 3.337e-03 | 2.693e-02 |
| 1e-03 | 3.942e-02 | 2.732e-03 | 6.930e-02 | 4.808e-03 | 1.220e-01 | 9.861e-05 | 2.501e-03 |
| 3e-04 | 1.247e-02 | 3.736e-04 | 2.996e-02 | 6.691e-04 | 5.365e-02 | 1.047e-05 | 8.399e-04 |
| 1e-04 | 3.944e-03 | 7.185e-05 | 1.822e-02 | 1.266e-04 | 3.210e-02 | 2.167e-06 | 5.496e-04 |
| 3e-05 | 1.247e-03 | 1.851e-05 | 1.485e-02 | 3.160e-05 | 2.534e-02 | 5.841e-07 | 4.684e-04 |
| 1e-05 | 3.944e-04 | 5.462e-06 | 1.385e-02 | 9.432e-06 | 2.392e-02 | 1.748e-07 | 4.433e-04 |
| 3e-06 | 1.247e-04 | 1.689e-06 | 1.355e-02 | 2.865e-06 | 2.297e-02 | 5.430e-08 | 4.354e-04 |
| 1e-06 | 3.944e-05 | 5.306e-07 | 1.346e-02 | 6.623e-07 | 1.679e-02 | 1.707e-08 | 4.330e-04 |

8x8 low-cluster flat

| theta | delta_* | e_H2 | e_H2/delta_* | e_Hankel | e_Hankel/delta_* | excess | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-01 | 1.065e-01 | 1.885e-02 | 1.770e-01 | 1.909e-02 | 1.793e-01 | 6.124e-03 | 5.752e-02 |
| 3e-02 | 3.335e-02 | 4.990e-03 | 1.496e-01 | 7.568e-03 | 2.270e-01 | 7.381e-04 | 2.213e-02 |
| 1e-02 | 1.051e-02 | 1.544e-03 | 1.469e-01 | 2.547e-03 | 2.424e-01 | 1.459e-04 | 1.388e-02 |
| 3e-03 | 3.320e-03 | 4.870e-04 | 1.467e-01 | 8.208e-04 | 2.472e-01 | 3.770e-05 | 1.136e-02 |
| 1e-03 | 1.050e-03 | 1.539e-04 | 1.467e-01 | 2.611e-04 | 2.488e-01 | 1.108e-05 | 1.056e-02 |
| 3e-04 | 3.319e-04 | 4.868e-05 | 1.467e-01 | 8.272e-05 | 2.493e-01 | 3.420e-06 | 1.031e-02 |
| 1e-04 | 1.049e-04 | 1.539e-05 | 1.467e-01 | 2.620e-05 | 2.497e-01 | 1.073e-06 | 1.023e-02 |
| 3e-05 | 3.319e-05 | 4.868e-06 | 1.467e-01 | 8.485e-06 | 2.557e-01 | 3.385e-07 | 1.020e-02 |
| 1e-05 | 1.049e-05 | 1.540e-06 | 1.467e-01 | 5.203e-06 | 4.958e-01 | 1.070e-07 | 1.019e-02 |
| 3e-06 | 3.319e-06 | 4.877e-07 | 1.470e-01 | 3.582e-06 | 1.079e+00 | 3.382e-08 | 1.019e-02 |
| 1e-06 | 1.049e-06 | 1.542e-07 | 1.469e-01 | 2.005e-07 | 1.911e-01 | 1.069e-08 | 1.019e-02 |

### A4. Other condition numbers

The same run covers `kappa = 1e2, 1e4, 1e6`; the order of `e_H2` does not depend
on `kappa` for the two non-degenerate families:

`kappa = 1e2`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 1.04e-06 .. 1.04e-01 | 0.12 | 0.14 | 1.10 | 1.368e-02 | 1.764e-02 | -2.220e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 9.65e-07 .. 9.63e-02 | 0.22 | 1.00 | 0.86 | 2.460e-02 | 8.231e-01 | 3.530e-04 |
| 4x4 | log-uniform | flat | 2/2 | 1.60e-06 .. 1.59e-01 | 1.00 | 1.09 | 1.00 | 3.769e-01 | 5.500e-01 | 2.412e-02 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 1.15e-06 .. 1.09e-01 | 1.00 | 0.58 | 1.00 | 5.097e-01 | 2.522e+00 | 2.170e-01 |
| 4x4 | low-cluster | flat | 2/2 | 9.93e-07 .. 9.91e-02 | 0.31 | 1.51 | 1.01 | 1.615e-02 | 1.479e-01 | 1.230e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.64e-07 .. 9.62e-02 | 0.65 | 0.96 | 1.00 | 1.935e-02 | 1.381e-01 | 1.151e-04 |
| 4x4 | repeated | flat | 2/2 | 1.04e-06 .. 1.04e-01 | 0.05 | 0.69 | -0.04 | 1.234e-02 | 6.290e-03 | 1.619e-02 |
| 4x4 | repeated | h2-balanced | 2/2 | 9.65e-07 .. 9.63e-02 | 2.01 | -0.01 | 0.09 | 3.331e-04 | 6.009e-03 | 3.377e-04 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.64e-06 .. 4.18e-01 | 0.18 | 1.68 | 0.24 | 2.636e-03 | 2.232e-03 | 0.000e+00 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.31e-06 .. 3.86e-01 | 0.51 | -0.35 | 0.10 | 2.865e-03 | 4.396e-03 | 2.933e-03 |
| 8x8 | log-uniform | flat | 4/4 | 2.94e-06 .. 2.81e-01 | 0.99 | 0.15 | 1.00 | 3.074e-02 | 5.810e-01 | 1.783e-03 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 2.26e-06 .. 2.24e-01 | 0.99 | 0.73 | 1.00 | 4.783e-02 | 2.031e-01 | 2.946e-03 |
| 8x8 | low-cluster | flat | 4/4 | 9.72e-07 .. 9.70e-02 | 1.16 | 0.76 | 1.00 | 1.775e-03 | 4.294e-01 | 1.357e-03 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 8.95e-07 .. 8.95e-02 | 0.98 | 0.22 | 1.00 | 4.797e-03 | 4.083e-02 | 2.044e-03 |
| 8x8 | repeated | flat | 2/4 | 5.84e-06 .. 5.04e-01 | 2.00 | 2.62 | 0.24 | 2.741e-04 | 3.503e-04 | 2.729e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.12e-06 .. 4.43e-01 | 0.67 | 0.79 | 0.51 | 1.634e-03 | 6.123e-02 | 4.222e-04 |

`kappa = 1e4`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.47e-06 .. 3.30e-01 | 0.47 | -1.00 | 1.02 | 6.477e-03 | 1.166e-01 | 1.979e-04 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.41e-06 .. 3.21e-01 | 1.02 | 1.59 | 1.00 | 1.168e-03 | 7.039e-04 | 1.625e-04 |
| 4x4 | log-uniform | flat | 2/2 | 7.33e-06 .. 6.12e-01 | 1.00 | 1.12 | 1.00 | 1.132e-01 | 1.384e-01 | 7.708e-03 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 2.51e-06 .. 2.35e-01 | 1.00 | 0.93 | 1.00 | 5.045e-01 | 2.002e+00 | 4.512e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.93e-07 .. 9.91e-02 | 1.03 | -0.05 | 1.00 | 2.470e-03 | 4.694e-03 | 3.634e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.88e-02 | 0.54 | -0.50 | 1.00 | 1.860e-02 | 7.530e-01 | 2.808e-04 |
| 4x4 | repeated | flat | 2/2 | 3.44e-06 .. 3.24e-01 | 1.09 | 1.81 | 2.05 | 4.164e-04 | 3.671e-04 | -4.359e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.40e-06 .. 3.20e-01 | -0.18 | 1.72 | 2.39 | 4.459e-03 | 4.319e-03 | -1.664e-03 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.52e-05 .. 8.52e-01 | 2.03 | -0.38 | 2.11 | 1.984e-04 | 7.743e-03 | 1.714e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.47e-05 .. 9.64e-01 | 2.08 | 0.25 | 1.22 | 1.224e-04 | 1.399e-02 | 1.658e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.71e-05 .. 7.62e-01 | 1.00 | 0.97 | 1.00 | 5.386e-02 | 9.238e-02 | 1.354e-03 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.87e-05 .. 6.57e-01 | 1.00 | 0.92 | 1.00 | 5.559e-02 | 2.798e-01 | 1.027e-02 |
| 8x8 | low-cluster | flat | 4/4 | 9.81e-07 .. 9.82e-02 | 0.98 | 1.00 | 1.00 | 6.050e-02 | 7.890e-02 | 1.317e-03 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.06e-07 .. 9.04e-02 | 0.85 | 1.77 | 1.00 | 2.833e-02 | 8.335e-02 | 3.174e-03 |
| 8x8 | repeated | flat | 2/4 | 5.82e-05 .. 9.85e-01 | 1.88 | 0.72 | 2.06 | 1.466e-04 | 9.673e-03 | -9.599e-05 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.72e-05 .. 9.69e-01 | 2.27 | 1.73 | 2.01 | 7.168e-05 | 6.422e-03 | 1.442e-04 |

`kappa = 1e6`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.35e-05 .. 9.59e-01 | 2.05 | 1.90 | 1.41 | 3.451e-04 | 4.516e-04 | 4.893e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.31e-05 .. 9.55e-01 | 1.89 | 0.73 | 1.58 | 4.816e-05 | 1.555e-02 | 9.860e-06 |
| 4x4 | log-uniform | flat | 2/2 | 3.99e-05 .. 9.72e-01 | 1.00 | 0.47 | 1.01 | 1.315e-02 | 6.947e-02 | 4.261e-04 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 5.68e-06 .. 4.20e-01 | 1.00 | 1.01 | 1.00 | 4.512e-01 | 1.164e+00 | 2.131e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.90e-07 .. 9.88e-02 | 0.86 | 1.01 | 1.00 | 1.079e-02 | 7.350e-03 | 5.259e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.87e-02 | 0.55 | 0.56 | 1.00 | 5.556e-03 | 1.204e-02 | 3.491e-04 |
| 4x4 | repeated | flat | 2/2 | 3.29e-05 .. 9.57e-01 | 1.14 | -0.05 | 1.44 | 3.648e-04 | 6.945e-03 | 1.705e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.29e-05 .. 9.54e-01 | 1.72 | 3.21 | 1.97 | 2.147e-04 | 1.291e-02 | 1.208e-04 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.51e-04 .. 9.99e-01 | 2.00 | 1.38 | 2.00 | 6.995e-04 | 5.347e-03 | 6.995e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.50e-04 .. 9.96e-01 | 2.00 | 1.73 | 2.00 | 4.926e-04 | 5.828e-03 | 4.943e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.35e-04 .. 8.97e-01 | 1.00 | 0.99 | 0.92 | 3.368e-02 | 6.241e-02 | 6.946e-04 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.47e-04 .. 8.23e-01 | 1.00 | 1.01 | 0.99 | 3.261e-02 | 1.245e-01 | 6.420e-03 |
| 8x8 | low-cluster | flat | 4/4 | 1.05e-06 .. 1.07e-01 | 1.00 | 0.10 | 1.00 | 1.456e-01 | 2.254e+00 | 1.026e-02 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.17e-07 .. 9.18e-02 | 0.95 | 1.34 | 1.00 | 4.287e-02 | 6.216e-02 | 1.185e-03 |
| 8x8 | repeated | flat | 2/4 | 5.82e-04 .. 1.00e+00 | 2.00 | 2.14 | 2.00 | 8.956e-04 | 1.097e-03 | 8.953e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.81e-04 .. 9.98e-01 | 2.00 | 2.14 | 2.02 | 4.611e-04 | 1.098e-03 | 4.372e-04 |

### A5. What this does and does not refute

Refuted, on a fixed problem along one controlled family with `delta_* -> 0`:

* `||H_U - H_V||_H2 <= C delta_*^2` with a constant that may depend on the fixed
  `(B, f, Sigma)`: the transfer exponent is 1 and `e_H2 / delta_*` plateaus at a
  positive constant (`1.34e-02` for 4x4 log-uniform flat, `1.49e-01` for 8x8
  low-cluster flat) while `e_H2 / delta_*^2` grows by orders of magnitude.
* the sharp-baseline form `Err(V) <= Err(U) + C delta_*^2`: the excess over the
  exact-MPMM baseline also has exponent 1 with a positive constant ratio.
* the vendor impedance channel: `e_Hankel / delta_*` plateaus at a positive
  constant as well, so a Hankel-norm guarantee does not fare better than H2 here.
* any quadratic dynamic statement assembled from matching-defect information
  alone by triangle inequality.

Not refuted:

* quadratic statements that use structure beyond the matching defects (Hermite /
  derivative / cross-shift information).  The experiment shows such structure
  cannot be *recovered* from the defects; it does not show that no theorem using
  that structure exists.
* a **linear** port bridge: the surviving candidate
  `delta(t; V) <= gamma_Sigma(t) + K_vec(t) delta_*` of A7 is linear in
  `delta_*` and is exactly the shape the counterexample leaves standing.
* a cancellation between `H - H_U` and `H_U - H_V`: measured and excluded by the
  excess columns above.

Mechanism, consistent with the numbers: at a matching shift the value error is
`||x - x_V||^2_(B + sigma I) = O(delta^2)`, but the derivative is
`H'(sigma) = -x^T x` and Galerkin only gives `x_V^T (B + sigma I) e = 0`, not
`x_V^T e = 0`, so the derivative mismatch is generically `O(delta)`.

### A6. Limits

* One controlled perturbation family (`V_theta` rotating out of `U`).  It settles
  the order question for the general defect-only statement, by counterexample; it
  is not a sweep over inexact moment spaces, and the plateau constants are
  problem specific (`B`, `f`, `Sigma` fixed) rather than universal.
* `delta_* -> 0` is driven by `theta -> 0` at fixed `[a, b]` and fixed shifts.
* 4x4 uses two shifts and 8x8 four, not the full 13-shift plan.
* `rank` in the tables is the numerical rank of the snapshot matrix; for the
  `repeated` family at 8x8 it is `2` out of 4 shifts, so those rows measure a
  2-dimensional moment space and cannot be read as "degeneracy restores the
  square" in the same sense as a full-rank clustered spectrum.  `endpoint-cluster`
  is likewise not uniformly second order (`kappa = 1e2` gives exponent 0.1 - 0.2).
* The vendor material also states a whole-space-time temperature energy as
  `< 2 sqrt(epsilon)`, but does not define that norm in the retrieved pages.  It is
  `[STATE-DYN]` and is **not** an acceptance target: a norm that is undefined in the
  retrieved pages cannot be a testable criterion (see `THEORY.md` 0.1).  The
  field-level criterion that *is* an acceptance target is the fixed-shift one, and
  no number in this record is claimed against the dynamic channel; the decisive
  evidence here stays on the port side (the `< 2 epsilon` impedance Hankel
  statement).
* Only relative ROM-to-ROM differences are reported; how large `delta_*` actually
  is in the real model stays the certificate's job.

### A7. The skeleton layer: invariants, rho_Sigma, K_sample and K_vec

Second mode of the same bench (`--mode skeleton`), on the delivered 13-shift box
plan.  The cardinal functions are the symmetric skeleton choice `lambda_j = t_j =
sigma_j`,

```text
ell_j(s) = 2 sigma_j/(s + sigma_j) prod_{k != j} (s - sigma_k)/(s + sigma_k)
           * (sigma_j + sigma_k)/(sigma_j - sigma_k),
r_Sigma(z) = prod_k (z - sigma_k)/(z + sigma_k),
```

so that the Massei--Robol identity holds: `1 - (s + lambda) I_Sigma[(. + lambda)^-1](s)
= r_Sigma(lambda)/r_Sigma(-s)`.  The four implementation invariants, checked before
anything is measured (mixed relative tolerance, and an 80-digit mpmath reference for
the cardinal values themselves):

```text
kappa                        100 | 1e+04 | 9.674e+05 | 1e+06
ell_j(sigma_k) = delta_jk    4.4e-16 | 4.4e-16 | 8.9e-16 | 6.7e-16
skeleton identity            6.7e-16 | 8.9e-16 | 9.1e-16 | 9.7e-16
|r_Sigma(-i omega)| = 1      6.7e-16 | 1.1e-15 | 1.1e-15 | 8.9e-16
mpmath 80-digit reference    1.3e-16 | 1.9e-16 | 1.5e-16 | 3.0e-16
partial fractions rebuild    1.6e-12 | 2.3e-12 | 1.0e-11 | 1.5e-11
min eigenvalue of Q          1.4e+00 | 1.7e+00 | 1.7e+00 | 1.7e+00
```

All six hold at machine precision, so the plan geometry below is that of the
documented construction and not of a private convention.  The partial fractions have
both signs with magnitudes around `2e+07` and cancel, which is why the rebuild test
matters: with mixed relative error `1e-11` the double-precision residues are
usable.

| kappa | shifts | rho_Sigma | rho_Sigma^2 | q_m | K_vec = sup_t K_vec(t) | gamma_Sigma | K_sample range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 6 | 1.428271e-02 | 2.039957e-04 | 2.040041e-04 | 8.886572e+01 | 1.4283e-02 | 14.679 .. 18.650 |
| 1e+04 | 9 | 3.025451e-02 | 9.153356e-04 | 9.153357e-04 | 9.059954e+01 | 3.0255e-02 | 6.966 .. 10.330 |
| 9.674e+05 | 13 | 2.912612e-02 | 8.483307e-04 | 8.483203e-04 | 1.440550e+02 | 2.9126e-02 | 7.210 .. 10.691 |
| 1e+06 | 13 | 2.940478e-02 | 8.646413e-04 | 8.641083e-04 | 1.430984e+02 | 2.9405e-02 | 7.146 .. 10.593 |

`rho_Sigma^2` and the elliptic construction `q_m = 4 exp(-m pi^2/log(4 kappa))` agree
to four or five digits at every condition number, which is the expected relation
(`8.483307e-04` against `8.483189e-04` at the delivered `kappa = 9.674e+05`) - checked
as a relation, not as an equality to roundoff.

The two constants that decide whether a linear dynamic bridge can be attempted:

* `K_sample`, the exact SISO worst-case amplification of matching-value errors
  (`D_j <= delta_*^2 h_j`) through the skeleton cardinal functions, from the
  partial-fraction Gram `Q_ij = sum_{k,l} a_ik a_jl/(sigma_k + sigma_l)` with all
  `2^13` box vertices enumerated: **6.97 .. 18.65** over all
  condition numbers, families and sources, i.e. mild.  So sample-value
  propagation does not explain the first-order transfer error of A3.
* `K_vec(t) = sum_j |ell_j(t)| c_j(t)` with `c_j(t) = max_{lambda in [a,b]}
  (lambda + t)/(lambda + sigma_j)` and `[a, b]` the **box spectral interval**
  `[1, kappa]` (not the extreme shifts), the constant of the linear candidate
  `delta(t; V) <= gamma_Sigma(t) + K_vec(t) delta_*`.  It is reported at its
  supremum: `|ell_j|` and the branch of `c_j` only break at the shifts, so each
  open interval of `t` carries a smooth rational `K_vec`, and the maximum is
  located per interval and cross-checked at 80 digits, as is the analytic tail.  The
  value reported is therefore a **numerically resolved** supremum, not a
  computer-certified one: the 80-digit pass verifies the candidate value, not the
  absence of another stationary point, and `rho_Sigma` is likewise a grid maximum.
  Should it enter a theorem constant, the extremum needs root isolation or a proof
  that it sits at `t -> 0`.  **88.87 (kappa=1e2) ->
  90.60 (1e4) -> 144.06 (9.674e5) -> 143.10 (1e6)**, i.e. bounded by a few hundred,
  not growing with the condition number, and attained at `t -> 0` on the delivered
  plan.  `sup_{t >= 0} gamma_Sigma(t) = rho_Sigma` holds in closed form because
  `|r_Sigma(-t)| = prod_j (t + sigma_j)/|t - sigma_j| >= 1`, so the purely skeleton
  part needs no sampling either (`2.9e-02` on the delivered plan).

Consequence: the linear route is not killed by an exploding constant, which is what
the RHS budget question needs; the responsibility for the first-order transfer error
is localized to the subspace perturbation destroying the Hermite structure, not to
cardinal-function amplification.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_dynamic_bridge_toy.py \\
  --mode skeleton --size 4 --output <path>.json
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_dynamic_bridge_toy.py \\
  --mode skeleton --size 8 --output <path>.json
```

### A8. The linear candidate survives the real axis

Layer 3 of the same bench (`--mode linear`) tests the derived inequality

```text
delta(t; V) <= gamma_Sigma(t) + K_vec(t) delta_*,        t >= 0,
```

on the **whole** 13-shift plan rather than on a two- or four-shift subset: the toy
dimension is raised to 26 so that `U` (spanned by the 13 exact snapshots) keeps rank
13 and a 13-dimensional `Q` complement exists, and the rotated space is still
`V = U cos theta + Q sin theta`, so `delta_* = max_{j=1..13} delta_j` is the maximum
matching defect of the complete plan.  The time grid is not a scan: it contains
`t = 0`, every shift `sigma_j`, the geometric midpoint of each neighbouring pair, 40
logarithmic points across `[1e-6, 1e6 kappa]` and the far tail `1e9 kappa`.  Both
terms of the bound are also measured separately, against their own bounds:

```text
skeleton piece  ||e_skel(t)||_(A_t) / ||x(t)||_(A_t)  against  gamma_Sigma(t),
sample piece    sum_j |ell_j(t)| ||x_j - x_V,j||_(A_t) / ||x(t)||_(A_t)
                                                      against  K_vec(t) delta_*.
```

402 rows (two spectrum families, `theta = 1e-4, 1e-6, 1e-8`, 67 time arguments
each) give **no violation** of the total inequality and no violation of either
piece.  The maximum of `R(t) = delta(t;V) / U_vec(t)` is exactly `1.000000`, and it
sits at `t = sigma_min`: there the skeleton is exact (`gamma_Sigma = 0`,
`e_skel = 0`) and `K_vec(sigma_min) = 1`, so `delta(t;V) = delta_* = U_vec(t)` by
construction - the bound is attained at the worst matching point, as it must be.
A strict `>` comparison had reported this equality as a violation; the test now
carries a `1e-12` relative tolerance.

Tightness of the bound away from that point, as `U_vec / delta`:

```text
family        theta     delta_*      median     max      max location
log-uniform   1e-8      1.781e-06    2.22       1.77e+02  t = 1.0e+09 kappa
low-cluster   1e-8      1.528e-06    2.63       1.45e+02  t = 1.0e+09 kappa
log-uniform   1e-6      1.781e-04    4.14       1.77e+02  t = 1.0e+09 kappa
low-cluster   1e-6      1.528e-04    5.00       1.45e+02  t = 1.0e+09 kappa
log-uniform   1e-4      1.781e-02    87.82      2.76e+02  t = 1.0e+09 kappa
low-cluster   1e-4      1.528e-02    93.45      4.94e+02  t = 1.0e+09 kappa
```

In the asymptotic regime a median factor of `2.2 .. 5.0` separates the bound from
the measured defect; the largest ratios occur in the far tail, where
`gamma_Sigma = rho_Sigma` alone dominates and the measured `delta(t;V)` has already
decayed below the exact-skeleton baseline - that is the baseline being conservative,
not the sample term being slack.

The imaginary axis is deliberately **not** part of this test: the derivation uses
`A_t = B + tI > 0` and the `A_t`-energy Galerkin optimality, neither of which
survives at `t = i omega`.  Frequency-response error on `t = i omega` remains a
separate diagnostic.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_dynamic_bridge_toy.py \\
  --mode linear --size 26 --kappas 9.674e5 --families log-uniform,low-cluster \\
  --sources flat --thetas 1e-4,1e-6,1e-8 --output <path>.json
```

Reproduce (layer 1, JSON outside the repository):

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_dynamic_bridge_toy.py \
  --mode rotation --size 4 --output <path>.json
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_dynamic_bridge_toy.py \
  --mode rotation --size 8 --output <path>.json
```

---

## Part B - closed routes and measurement corrections

本文件是本目录**唯一**的失败路线档案。此前的 27 份逐条记录（`RESULTS.md`、
`ZOLOTAREV_RESULTS.md`、`TANGENT_CORNER_SAMPLING.md`、`TRANSIENT_COMPARISON.md`、
`CONTINUOUS_RESIDUAL_CERTIFICATE.md`、`PARAMETER_BOX_TAYLOR_CERTIFICATE.md`、
`CHEBYSHEV_BERNSTEIN_FALSIFICATION.md`、`NEUMANN_BERNSTEIN_CASE1_5MM.md`、
`CHEAP_RESIDUAL_BOUND.md`、`TRANSIENT_CERTIFICATE_RESEARCH.md`、`POST_SVD_EFFECT.md`、
`BOX_FREQUENCY_PLAN_AUDIT.md`、`HANKEL_IMPULSE_GUARANTEE.md`、
`INEXACT_MOMENT_DYNAMIC_BRIDGE.md`、`INEXACT_MOMENT_MARGINS_CASE1.md`、
`INEXACT_RK_DEFECT_CASE1.md`、`ACTIVE_SET_EXPERIMENT.md`、`ACTIVE_SET_CASE1_VALIDATION.md`、
`REDUCED_RESIDUAL_FALSIFICATION.md`、`RESIDUAL_COMPRESSION_AND_STABILITY.md`、
`GLOBAL_CROSS_COUPLING.md`、`NEXT_STEPS.md`、`GREEDY_TRANSIENT_RESULTS.md`、
`OUTPUT_GREEDY_TRANSIENT_RESULTS.md`、`FREQUENCY_DEPENDENT_HTC_SAMPLING.md`、
`TWO_GROUP_FREQUENCY_FACES.md`、`ONE_MM_FREQUENCY_FACES.md`）已按本次收束删除，
结论与判据式数字全部并入本文。仍然有效的正面结论见 `README.md` 与 `THEORY.md`。

判定只取四类，每条明确归属：

```text
被反例证伪      数学上错误，或前提在实测数据上直接失效
界太松          不等式本身成立，但数值上不可能驱动 1e-3 级判断
成本不可行      结论正确，但成本结构（求解数、分解数、墙钟、内存）不可接受
测量更正        结论来自实测 bug 或定义错位，原数字撤回，不是方法失败
```

**每条条目还必须带一个判定作用域标签**，说明它的失败证据是在哪个误差度量下取得的：

```text
STATE           全场 / A-能量 / 状态传递算子范数
PORT-SYSTEM     H2 / Hankel / 冲激响应系统范数（动态口径，P0 未闭合）
PORT-FIXED-S    单个实频移上的 Z(s;mu)
PORT-STEP       采样 step-response 度量（BDF1 步进 / 观测步响应）
COST            N_RHS / setup / 墙钟 / 内存
CORRECTION      实测 bug 或定义错位
```

**作用域标签的结论上限（本文件最重要的纪律）.** 标着 `PORT-FIXED-S` 或 `PORT-STEP` 的条目
只证明“该方案在这个固定实移 / 采样步进诊断下失败”。它们**不**证明该方案不可能满足
port-Hankel / 冲激响应系统范数，也**不**否定同一套参数点、同一套基底构造思想在系统范数目标下
重新采用。把前者读成后者是本文件历史上最需要纠正的逻辑跨越，依据是
`records/SYSTEM_GAP_AND_FAILURE_ARCHIVE.md` 的一阶反例：fixed-`s` 的同址平方关系**不**推出动态系统范数
误差也平方。反过来，标着 `PORT-SYSTEM` 的条目（A1、A10、C1、C2、C4）是直接对动态系统范数
成立的判决。

**目标恢复（2026-09 重新定位）.** 研究目标是全场传递算子 `X(p,s) = A(p,s)^-1 G`，因此仅凭
全场 / A-能量 / 状态范数判据的失败条目**同样是**当前目标下的负结果；它在定频上的实现就是
`THEORY.md` 命题 1 的共址端口缺陷。此前把 state 级通道降级为“仅证明装置”的作用域收紧已撤回，
见 `THEORY.md` 第 0 节。第 1--3 节原有条目无需重新标注：它们各自都含独立的 port 传递、系统
范数或成本失败，加上场级判据只会让结论更强。

**同时保持排除的两条路线.** 本次重定位**不**重开：Robin 解流形 n-width 速率命题（采样规模
最优性仍由 `N_FOM` 与成本结构回答）与动态半群 state-energy 的 vendor 通道（其范数在可取得
材料中未定义，见 A6）。

**两条边界（沿用 GPT 审计确定的规则）**：

* **正文撤回**：原文档陈述不准确、定义写错、范围写大了，但方法本身没有被失败证据否定。这种条目留在 `README.md` / `THEORY.md` 的正文里，写明“旧说法撤回”，**不进本文件**。
* **本文件（负结果）**：存在明确的数学反例、前提在目标数据上失效、严格界在数值上不可用、或成本结构使路线无法达到研究目标。

这样“错误表述”不会被误记成“算法失败”。

四类是**判定**，不是流程阶段；一条路线可以同时属于两类（写作“A + B”）。**不使用“被取代”
“范围不足”“成本未确定”“诊断保留”这类状态描述**：所有条目都必须落到上面四类之一，否则它
就不是负结果。因此本文件不收录两类东西：

* 仍是开放候选的路线（先验闭式采样的 handoff，见 `README.md` 第 6 节）；
* 尚未证明的目标保证（vendor 的 `2*eps` / `2*sqrt(eps)` 常数，属于 `THEORY.md` 的 P0 动态
  传递定理，不是实验失败）。同理，“某条路线关闭”绝不等于“它所属的上位问题关闭”。

**关于复现命令.** 许多条目引用的 bench/probe 脚本已在收束中退役（它们只服务于
已被否定的一次性实验）。命令按原样保留为**出处记录**，说明该数字是怎样产生的；
仍然在维护、可直接运行的脚本见 `README.md` 的文件清单。凡是记录里没有印出命令的，
本条照实写“记录未给出”。

---

### B0. 撤回声明与理论交叉引用（不是失败路线）

以下条目曾经以“负结果”的形式出现在旧记录里，但按上面的边界它们属于**正文撤回**或**未证明的目标**，因此不占用负结果编号：

* **A9 误差目标的重估（原编号 A9）.** all-input 定义 `eta_E(p) = sqrt(lambda_max(E(p), E0))`、`E0 = (1/2) G^T C^-1 G` 已确立；厂商的 `2*eps` / `2*sqrt(eps)` 常数属于**尚未证明的目标保证**
  （引文未写归一化、2019 年误差界论文无法完整核对），在 `THEORY.md` 的 P0 动态传递定理下作为开放缺口跟踪。需要撤回的是文档层面的错误说法：“一个 Gramian 同时给出两个 all-input 指标”为**错**
  （`tr(D^T T D P)` 只给 `tr(E)`），先前的 `sqrt(tr(E)/tr(E0))` 只测四个脉冲能量之和、会掩盖同时激励下的坏组合。参数单元树 + 精确中心解 + 半群 Lipschitz 常数 + Gramian 余项界这一整套机制
  （`semigroup_box_bounds`）因大单元常数按 `1/alpha^2` 放大且未认证整个 Case 1 盒而被移除。
* **C10 先验闭式参数采样（原编号 C10）**：开放候选，见 `README.md` 第 6 节第 7 条。旧记录把它挂在 `THEORY.md` 的 P1「Robin 解流形的 n-width 衰减」上，该命题仍是排除项，不在主理论线上（state-space 复杂度问题由 `N_FOM` 与成本结构回答）。

---

### B1. 认证与误差界路线

#### A1  全局 matrix-fractional Bernstein 残差证书 + 均匀 Neumann 界

**判定作用域.** `PORT-SYSTEM + PORT-FIXED-S + COST`

主张：不细分参数域，用同一张量 Bernstein 基的系数级 Schur 不等式与均匀 Neumann
余项，在整个 HTC 域证明最终 1e-3 SVD ROM 的 impedance Hankel 与 impulse-energy
相对误差不超过容差。
决定性测量：degree-1/2/3 reduced Taylor witness 的 DC 绝对界 `703.51 / 699.60 /
695.76`（记录判为 unusably loose）；均匀 Neumann 界要压到 1e-3 需要约 2627 阶
(5 mm) / 2950 阶 (2.5 mm)；局部连续盒证书只覆盖 ±1%..±20%（±1% 盒 5 mm 为
`0.00176891 / 0.05891111`，目标 `0.002 / 0.0632455532`），单个 ±50% 盒不过
（`0.03499529 / 0.02809764`）；2.5 mm 分片 42 单元不认证、82 单元 Hankel
`0.00224561` 仍差约 12%；证书本身耗时 `246 s`，而同一次提取只需 `2.75 s`。
判定：界太松（系数级不等式正确，数值无用）；局部路线另有成本不可行。
复现：`probe_uniform_certificate.py --mesh-mm 5 --half-width 0.01 --stock` 等（脚本已退役）。

#### A2  全盒 Taylor jet 参数盒证书（旧成本结构）

**判定作用域.** `PORT-STEP + COST`

**作用域.** 本条只否掉当时那种“每个盒都要全阶中心轨迹与灵敏度、每盒 2.6--16.4 s”的
**全盒**实现，以及“全局多项式-矩阵分式不等式能给出可用界”的期望。当前路线用的
**cell-local polynomial trial + Riesz--Bernstein 单元证书**（`THEORY.md` 命题 5 / 5M）不受
影响，不要把它读成“多项式 trial 被否掉”。

主张：在单个锚点展开 Taylor jet 加严格余项，为未改动的最终 1e-3 SVD ROM 在连续
二维 HTC 盒上证明 BDF1 前 40 步、16 个热源到结温传递的相对误差不超过 1e-3。
决定性测量：2.5 mm、92 快照 / 47 阶、中心 HTC `(100,100)`：一阶只覆盖 ±0.25%，
二阶 ±2%（界 `2.7238e-4 / 7.1558e-5`），三阶 ±5%，四阶 ±10%；难中心
`(5000,2)` 要到 7 阶才证明 ±10%（`3.5485e-4 / 1.9808e-4`），而它的 4 阶 ±10%
界是 `1.2090e-1`。每盒 `2.6..16.4 s`，且每锚点新增大量全阶灵敏度求解；两个中心
都没有给出整个 `[1,10000]^2` 的盒数、运行时间或全域误差上界。
判定：成本不可行（对已离散的有限维模型严格，但可证明盒极小、无全域成本估计）。
复现：`probe_taylor_box.py --mesh-mm 2.5 --steps 40 --fractions ...`、`--order 7
--center-h 5000 2`（脚本已退役）。

#### A3  单元 Neumann/Chebyshev 逆多项式 + 张量 Bernstein 盒（旧加权与旧实现）

**判定作用域.** `COST`

**作用域.** 否掉的是当时的**代价结构**：每个单元的全阶中心求值、旧的 Neumann 加权、旧的
tensor 实现。这不否定“单元上的多项式 trial + Bernstein 包络”本身 —— 现行的矩阵型单元
证书仍然使用它。

主张：用逆预解式的 Chebyshev/Neumann 多项式与张量 Bernstein 盒在单元上给出可比界，
再用廉价的 de Casteljau 细分闭合 1e-3 残差门限下的小区，目标低于 100 次昂贵局部
中心求解、单线程约 1 s。
决定性测量：原 Neumann 树 1555 次局部中心求值、776 个失败单元（其中 746 个仅余项
R 就超过阈值）；把 Neumann 升到 12 阶闭合 `0` 个；真实 Chebyshev 树 356 叶、
709 次中心求值、`10.90 s`，336 个失败单元里可廉价细分 17 个、细分闭合 `0` 个；
另试 `107+637+608 = 1352` 次 de Casteljau 细分仍闭合 `0` 个。两项门限均未过
（709 对 <100，10.90 s 对约 1 s）。
判定：成本不可行（记录明确写为 negative cost result，不是对精确算术包围的否证）。
复现：`bench_polynomial_anatomy_case1.py --output .../polynomial_anatomy_seed20260805_rank63.json`
（脚本已退役）。

#### A4  Reduced Neumann--Bernstein 残差盒原型

**判定作用域.** `PORT-FIXED-S + COST`

主张：用约化 Neumann 多项式 + 张量 Bernstein 系数 + 解析尾界，在连续二维 HTC 盒上
认证仓库式欧氏相对残差不超过 1e-3，且每次调用零额外全阶求解、成本低于亚秒门限。
决定性测量：三阶时 seed 20260805 在秩 49/60/63 上 `11/11` 个频移全部闭合，单元数
`1038/930/903`，但**界本身耗时 `18.75/21.76/27.50 s`**（stock 单次提取仅
`1.4..1.6 s`），effectivity 约 `4.94`；一阶只闭合 `5/11`（最小频移界 `0.1098`）、
二阶 `8/11`（`0.01476`），在 256 单元上限下仍未解析；提高秩几乎不减少单元数
（`1038 -> 903`）。41x41 网格上无一点超界，但秩 49 的网格最大值仅低于阈值约 0.9%。
判定：成本不可行（数学成立，实现是普通浮点、未外向舍入，且远超亚秒门限）。
复现：`bench_residual_compression.py --grid 9 --max-order 80 --greedy-target 1e-4`、
`bench_neumann_bernstein_case1.py --ranks 49 60 63 --orders 3 --max-cells 512 --audit-grid 41`
（脚本已退役）。

#### A5  单一全局谱标量的廉价盒残差界

**判定作用域.** `PORT-FIXED-S`

主张：用 `(lambda_min(K-,C) + s)^-1 * R^T C^-1 R`（只需一次稀疏乘法、不需任何逆作用）
作为联合 (HTC, frequency, input) greedy 的停止证书，认证 1e-3 甚至 1e-4。
决定性测量：恒等式与包围本身被确认（3 个基底 x 1053 点，Loewner 违反 `0`、逐元违反
`0`），但 effectivity 中位数 `11.0 / 29.4 / 125.3`、最大达 `2.114e5`；在设定误差的
点上界给出 `0.9088 / 8.752 / 6.745`，而真值是 `4.573e-5 / 2.851e-4 / 5.010e-5`
（比 `2.0e4 / 3.1e4 / 1.3e5`）。换成逐点最优尺度 `lambda_min(K(h),C)` 仍为
`2.5e4 / 4.0e4 / 1.4e5`：残差落在 C 对 A 并不是好度量的方向上，任何单一谱数都丢掉
四到五个数量级。该量只保留作排序优先级分数，绝不作为停止规则。
判定：界太松（且属结构性：见上句）。
复现：`bench_cheap_residual_bound.py 5 --grid 9 --local`（脚本已退役）。

#### A6  BDF1 输出残差证书（K_min^-1 全局 Gram + 见证空间）

**判定作用域.** `PORT-STEP`

主张：用步进增量作暂态伴随、单次 `K_min^-1` Gram 收缩，给出 SVD 压缩后每个 BDF1 步、
每个通道的输出误差上界，并据此选点与停止以认证 1e-3。
恒等式本身成立（`|Y - Yhat| <= sqrt(sum r K(p)^-1 r) * sqrt(sum d K(p)^-1 d)`），但界
比实测松约 40 倍：`7.079e-2` 对实测 `1.757e-3`，`1.932e-2` 对 `9.719e-4`；把
`K_min` 局部化为精确 `K(p)^-1` 后仍有 `2.121e-2 / 3.106e-2` 的松弛，即“换多个参数
锚点救不了这个停止测试”。dt = 50 s 时 `lambda_max(D, K_min+D) ~ 0.997862`，40 步几何
和约 `38.4`，界因此无法在 1e-3 上退休任何候选，反而强制额外富集。
判定：界太松（记录原文 valid algebraically but too loose to retire a candidate）。
复现：记录未给出命令。

#### A7  post-SVD 压缩效应诊断：raw majorant 不能认证交付 ROM

**判定作用域.** `PORT-FIXED-S + PORT-STEP`

原以为可以在 SVD 之前的 raw 快照张成空间上用凸角点二次 majorant 认证最终 ROM。
决定性测量：同一快照矩阵下 raw 对 post-SVD，2.5 mm 为 92 列 / raw 93 阶 / final 47 阶，
raw 最坏步 `3.7455e-6` 对 final `5.5055e-4`；1 mm 为 100 列 / 101 阶 / 57 阶，
`1.7758e-5` 对 `5.6538e-4`；压缩放大步误差约 `31.8` 倍（1 mm）与 `147` 倍
（2.5 mm）、稳态约 `18.4` 倍（1 mm）。raw majorant 把 top-high/top-high 排第一，
而 post-SVD 最大误差出在 top-high/bottom-low 训练角。
判定：被反例证伪 —— “raw 空间的 convex-corner majorant 可以认证 post-SVD 交付 ROM”是一次
错误的**数学迁移**，不是某个数字算错；由于该错误，记录里若干报告数字随之错位，这一部分才是
测量更正。raw 空间的界本身有效，但固定 1e-3 SVD 截断主导最终误差；压缩作为证明义务已被
“认证最终交付 V 本身”替代（见 `THEORY.md`）。
复现：`probe_post_svd.py --mesh-mm 2.5`、`--mesh-mm 1`（脚本已退役）。

#### A8  频率计划谱区间缺陷（修复已撤回）

**判定作用域.** `PORT-FIXED-S + CORRECTION`

生产频率计划曾用裸传导核 K 的广义谱区间，而 ROM 需要在 HTC 盒上求逆的算子谱包含
被 Robin 提升抬回的 Neumann 常数模。决定性测量：低 HTC 角 `(1.000, 0.992)` 的 SISO
误差 `6.07e-03`（容差的 6 倍），盒合法计划在同一点为 `1.89e-05`（改善 320 倍）；
裸 K 的 `lambda_min = 4.4716e-04` 对盒包围 `4.2795e-05`，计划下端点高 10.45 倍。
判定：被反例证伪（“旧裸 K 计划的谱区间覆盖整个 Robin 盒”这一假设是错的，低 HTC 角实测
6 倍超差）＋ 后来曾修复、现已按要求撤回的实现变更。其中
撤回旧的 `2.61e-2` first-term failure 那部分才属于测量更正 —— 那是 bench 的 LU 缓存 bug
（键只按 shift 值，第一个点之后复用了别的点的分解），纠正后为 `6.07e-03` (SISO) /
`3.78e-04` (MIMO)。FANTASTIC 精确矩参考本身未被证伪。
代价：盒合法计划 +18% RHS（116 -> 137）与两个额外频移；步误差不随计划同向改善
（5 mm seed 20260805 `1.262e-03 -> 1.785e-03` 变差，seed 7 `1.680e-03 -> 1.410e-03`
变好）。以上仅为历史测量，不代表当前生产路径。
原 `test_box_frequency_plan.py` 已随实现撤回。

#### A10  频率轴（Hankel）整盒证书

**判定作用域.** `PORT-SYSTEM`

把 README 第 3.2 节的锚定 Riesz 论证搬到 `W(p,omega) = A(p) + omega*C` 上。实测
（5 mm）传递误差上确界 `5.81e2`，而整模型 Hankel 下界只有 `0.185`，相对界 `3.1e3`。
原因是锚点 Riesz 算子必须同时上控整个 HTC 范围（四个数量级）与频率单元，Loewner 比率
损失约六个数量级；要修好需要按 (参数单元 x 频率单元) 逐块 Gram，即约 `1e4..1e5` 次
稀疏求解。
判定：界太松 + 成本不可行。`certified_box.py` 因此只保留固定实频移版本，频率轴明确
列为未决问题。

---

### B2. 采样、选点与富集路线

#### B1  先验闭式点集（tensor Chebyshev、二维 Padua、嵌套 Clenshaw--Curtis Smolyak）

**判定作用域.** `PORT-STEP + PORT-FIXED-S`

主张：参数样本可先验、确定性地选取，且张量 Chebyshev 插值对全纯响应的指数衰减界经
未截断快照空间继承，从而给出无随机种子的可认证提取器。
决定性测量：完整闭式 coercivity 界需要张量 63 阶（4096 个参数点）才能认证 0.1% 相对
junction 误差；换成广义特征值下界也只降到 34 阶（1225 点）。该保证只对**未截断**快照
空间成立，1e-3 SVD 既未计入证书也未再验证（记录原文 the present singular-value cutoff
alone does neither）。
判定：界太松（记录原文 the direct rigorous bound is too conservative）。
复现：`compare_sampling.py 2.5 1 2 3 --validation-grid 21 --random-holdout 128`、
`compare_extractors.py 2.5 --degrees 1 2 --random-seeds 5`（脚本已退役）。

#### B2  tensor corners（只用四个 HTC 角点）

**判定作用域.** `PORT-STEP`

主张：覆盖参数箱的四个角点即可代表整个箱内的响应。
决定性测量：4 点 / 16 个 source RHS / 阶 9，最坏 junction 误差 `0.17265%`，而三个点的
log-Padua degree 1 只有 `0.00553%`；角点响应张成更低秩的空间并漏掉重要的混合方向。
判定：被反例证伪。
复现：同 B1。

#### B3  log-Padua degree 2 与嵌套 level-3 Smolyak（加密确定性采样应带来单调改善）

**判定作用域.** `PORT-STEP`

决定性测量：全 12 个位移下 degree 2 的 `0.19403%` **差于** degree 1 的 `0.09239%`，
SVD 保留 45 而非 47 个模态；只把 closing cutoff 收紧到 1e-4 后阶数升到 67--68、误差
降到 `0.00744% / 0.00978%`；level-2 的 5 点嵌套 Smolyak 为 `0.07890%`，level-3 的
13 点在同一 cutoff 下并未改善。
判定：被反例证伪 —— 采样族不是非单调性的原因（记录原文 Thus the sampling family is
not the cause），失效环节是 closing SVD 压缩。
复现：同 B1。

#### B4  raw tensor Zolotarev 计数作为采样规模规则

**判定作用域.** `PORT-FIXED-S`

主张：有限区间 Zolotarev 理论的闭式点位与先验 resolvent 计数（8 x 7 = 56）可直接作为
提取器的采样规模与放置规则。
决定性测量：2.5 mm 等预算表里 Zolotarev 要到 4x4（16 点 / 64 RHS）才得 `1.543e-7`，
而 Zolotarev-seeded certified greedy 只用 5 点 / 20 RHS / 阶 21 就得候选网格相对证书
`3.882e-6`；同预算下 Padua 三点 `1.106e-3`、十点 `2.555e-9`、十五点 `9.93e-12`。
一维坐标的 Zolotarev 界不证明张量积满足每个 cross transfer 的相对误差。
判定：界太松（记录原文 its direct tensor count is too conservative）。
复现：`compare_zolotarev.py 2.5 --junction-tolerances --count-pairs 2x2 3x3 4x4 5x5 6x6 ...`
（脚本已退役；同族的 `test_zolotarev.py` 逻辑已并入 `test_certified_sampling.py`）。

#### B5  Zolotarev-seeded weak greedy 的停止规则（有限候选网格证书）

**判定作用域.** `PORT-FIXED-S`

最强结果：2.5 mm 用 5 个参数点 / 20 个 source RHS / 阶 21，41x41 候选网格最坏相对证书
`3.882e-6`；1 mm（122400 单元）同样 5 点 / 阶 21，候选网格证书 `3.523e-5`，场规模从
9072 涨到 122400 单元都不增加选点数，全阶解从约 150 降到 20。
失败点：1 mm 那一次带 `--skip-validation`，结论只是候选网格证书；连续箱保证仍需在
`(log h1, log h2)` 上做外圆整区间分支定界，记录称之为 remaining mathematical gap。
1 mm 的离线墙钟仍高于 stock 提取的约 60 s（谱区间 `90.6 s` + 五步贪心 `236.1 s`）。
判定：被反例证伪 —— 被否掉的是**“可把有限候选集上的最大值迁移成连续参数盒停止判据”**
这一说法（逻辑作用域不够，不是数值 effectivity 差）；在有限候选集上取最大化本身仍然正确。
复现：同 B4。

#### B6  seed + 排名最高的单个切角（2 个 HTC 点）

**判定作用域.** `PORT-STEP`

决定性测量：96 个 dynamic RHS、阶 45，12 点与 70 点 holdout 最坏 step transfer 均为
`2.704e-3`，比 stock 差（seed 20260805 `1.757e-3` / 126 RHS，seed 7 `1.924e-3` /
127 RHS）；被 seed + 前两个切角（144 RHS、阶 48、`6.189e-4`）取代。
判定：被反例证伪 —— 单角点只保护暴露边的一端。
复现：`probe_tangent_corners.py 2.5 --tolerance 1e-3 --maximum-points 3 --random-holdout 64`
（脚本已退役）。

#### B7  seed + 三个切角（4 个 HTC 点）

**判定作用域.** `PORT-STEP`

决定性测量：4 点 / 阶 48，12 点 holdout 最坏 step `7.302e-4`，比三点（三点预算的
`6.189e-4`）更差；两个最终空间都是阶 48，但三点基投影到四点空间之外的 Frobenius 范数
为 `0.124`，说明两个子空间明显不同。
判定：被反例证伪 —— 压缩后的 SVD 空间一般不嵌套，增加快照可使误差上升（记录原文
individual transient errors are not guaranteed to improve as snapshots are added）。
复现：同 B6。

#### B8  BDF1 输出残差界 greedy

**判定作用域.** `PORT-STEP + COST`

决定性测量：4 个 HTC 点 / 192 个 dynamic RHS，12 点 holdout `7.302e-4`；选择 + 提取
共 `44.32 s`（对比三点顶点规则的端到端 `10..11 s`），而第四阶段的误差界仍为 `1.738e-2`，
远高于 1e-3，无法停止。
判定：界太松（记录原文 too loose to stop at 1e-3）。
复现：记录未给出命令。

#### B9  conditional Zolotarev 暴露边规则

**判定作用域.** `PORT-STEP + PORT-FIXED-S`

主张：按 Massei--Robol 定理在暴露边 `p1 = p1_max` 上最小化一维场 resolvent 比。
决定性测量：三点预算 / 144 个 dynamic RHS / 阶 46，12 点 holdout 最坏 step
`1.855e-3`、最坏 steady `1.309e-2`；该规则自己的 degree-two 有理界为 `0.158`（远高于
1e-3）；两类设计的最坏误差都出现在暴露端点 `(p1_max, p2_min)` —— 最小化一维场
resolvent 比会把快照推离主导归一化输出的角点。
判定：被反例证伪（以实际目标衡量；一维放置原理本身成立）。
复现：`probe_conditional_zolotarev.py 2.5 --random-holdout 64`（脚本已退役）。

#### B10  四顶点凸二次 majorant 作为选择规则/误差量

**判定作用域.** `PORT-FIXED-S`

决定性测量：全局 majorant 极悲观 —— 单端口二维二次的特征值约 `8.55e4 / 2.09e2`，
而种子处精确局部 Ritz Hessian 只有 `7.73e1 / 1.86e-3`，主方向相反。它只对含 `X(p0)`
的**未压缩**稳态快照空间成立，而动态提取器采样正频移并施加 1e-3 全局 SVD，交付空间
不必含 `X(p0)`，正频域界也不直接是 40 步 BDF1 输出界。
判定：界太松（记录原文 its numerical value must not be read as the actual ROM error）。
作为**选择规则**它是有效的：排序得到的角点 `(p1_max,p2_max)`、`(p1_max,p2_min)` 加种子
共三点，12 点 holdout `6.189e-4`。
复现：`probe_tangent_corners.py 2.5 --tolerance 1e-3 --maximum-points 3`（脚本已退役）。

#### B11  场残差 greedy（1x1 Zolotarev 种子 + 9x9 候选网格）

**判定作用域.** `PORT-FIXED-S + PORT-STEP`

决定性测量：tol 1e-3 用 140 RHS / 阶 44 / `11.25 s`，完成残差 `8.04e-4`、step
`4.83e-5`、transfer `1.44e-3`（对 stock 的 `1.76e-3` 改善约 18%）；tol 1e-4 用
222 RHS / 阶 65 / `18.90 s`、transfer `1.26e-4`。两轮都**不满足每个 transfer entry**
的容差（1e-3 时最坏 entry `1.44e-3`，1e-4 时 `1.26e-4`），只有 nominal-power step 通过；
代价约多 28% / 21% 提取时间。
判定：被反例证伪（原主张“1x1 场残差 score 能代表目标多参数/动态误差”被本条的
entrywise 失败直接否掉）＋ 界太松（预 SVD 残差界本身合法，但只针对有限候选网格上的
场残差/源范数，不控制 junction 相对 cross transfer，也不是全时间或连续箱的界）。
复现：`compare_transient.py 2.5 --tolerances 1e-3 1e-4 --counts --seeds 20260805
--greedy-seed-counts 1 2 --greedy-grid 9 --max-extra-per-shift 12 --random-holdout 6`（脚本已退役）。

#### B12  Zolotarev 2x2 种子 + 同样的场残差 greedy

**判定作用域.** `PORT-FIXED-S + PORT-STEP`

决定性测量：tol 1e-3 为 227 RHS / 阶 45 / `15.60 s`、transfer `1.83e-3`；tol 1e-4 为
330 RHS / 阶 64 / `24.59 s`、`2.79e-4`。两轮的 transfer 都劣于 1x1 臂（`1.44e-3`、
`1.26e-4`），RHS 与时间更高。
判定：被反例证伪（记录原文 The 2x2 seed is dominated in these runs）。
复现：同 B11。

#### B13  固定 Zolotarev 4x4 张量替代随机 HTC 循环

**判定作用域.** `COST`

决定性测量：2.5 mm @1e-3 为 768 个 full source RHS / 阶 45 / `37.37 s`，worst transfer
`1.59e-3`，对 stock 的 `1.76e-3` 只改善约 11%，却用 6.1 倍 RHS 与 4.4 倍提取时间；
1 mm 同一次比较中 fixed 2x2 为 224 RHS / 阶 54 / `427 s`，stock 为 159 RHS / 阶 52 /
`193 s`。该 1 mm 运行在极慢的瞬态角点参考中被中断，记录声明 No 1 mm transient errors
are claimed。
判定：成本不可行。
复现：`compare_transient.py 2.5 --tolerances 1e-3 --counts 2 3 4 --seeds 20260805 7
--random-holdout 6 --dt 50 --duration 2000`（脚本已退役）。

#### B14  候选网格 17x17

**判定作用域.** `PORT-STEP`

决定性测量：选出 128 RHS（比 9x9 的 140 少）/ 阶 44 / 完成残差 `9.45e-4` / `14.89 s`，
但 holdout transfer `1.58e-3` **劣于** 9x9 的 `1.44e-3`；更密的网格改变贪心路径，
两种情况的最坏 transfer 参数都是物理角点 `(10000,1)`。
判定：被反例证伪 —— 加密候选网格不改善输出误差。
复现：`compare_transient.py 2.5 --tolerances 1e-3 --greedy-seed-counts 1 --greedy-grid 17 ...`
（脚本已退役）。

#### B15  场残差 greedy + 收紧 closing SVD cutoff

**判定作用域.** `PORT-STEP + COST`

决定性测量：tol 1e-3 + cutoff 1e-5 得到 140 RHS / 阶 84 / `11.14 s` / step `3.45e-7` /
transfer `5.49e-6`；tol 1e-4 + cutoff 1e-5 为 222 RHS / 阶 92 / `18.66 s` / `2.38e-6`，
均达标，但模态数几乎翻倍（cutoff 1e-3 时只需 44 / 65）；中间一档
（tol 1e-3 + cutoff 1e-4）阶 61 仍有 `1.10e-3`，略超容差。
判定：成本不可行 —— 记录原文 The mode count, rather than HTC placement alone, is
decisive for the final error。
复现：对应的 `greedy_transient_svd_1e4.json` / `_svd_1e5.json` / `_tol1e4_svd_1e5.json`
一族运行（脚本已退役）。

#### B16  稳态输出 greedy 直接用于动态四端口频移压缩 ROM

**判定作用域.** `PORT-STEP + PORT-FIXED-S + COST`

决定性测量：1e-3 臂选 5 点 / 260 RHS / 阶 47 / `18.95 s`，是那次 9x9 运行中唯一在全
cutoff 下通过全部 step transfer 的臂（`9.719e-4`），但最坏 steady transfer
`2.959e-3` 超出容差近 3 倍；1e-4 臂稳态 `1.427e-4` 仍失败；最终 ROM 上的稳态网格
证书在每个等 cutoff 行都失败（1e-3 时 `4.91e-3`，而原始稳态选择证书是 `3.66e-6`，
81 个候选中 50 个未解决）。
判定：成本不可行 ＋ 被反例证伪 —— “原始稳态选择证书可以迁移到压缩后的频移 ROM”是一次
错误的保证迁移（记录原文 The raw steady selection certificate is not a certificate for the
compressed frequency-shifted ROM）；稳态界也不界定整个瞬态，有限 2000 s 步进检查不能替代
稳态输出保证。
复现：`compare_transient.py 2.5 --tolerances 1e-3 1e-4 --output-greedy-grid 9
--output-max-points 12 ...`（脚本已退役）。

#### B17  稳态输出 greedy + 17x17 候选网格

**判定作用域.** `PORT-STEP`

决定性测量：仍选 5 点 / 260 RHS / 阶 47，选点证书 `3.81e-6`，但最坏观测 step
`1.003e-3` **略高于** 1e-3（9x9 为 `9.719e-4`），最终稳态网格证书 `5.36e-3`，
289 个候选中 183 个未解决。
判定：被反例证伪 —— 记录原文 even the observed passing result is sensitive to
changing the candidate grid。
复现：`compare_transient.py 2.5 --tolerances 1e-3 --output-greedy-grid 17 ...`（脚本已退役）。

#### B18  稳态输出 greedy + 收紧 closing cutoff

**判定作用域.** `PORT-STEP + COST`

决定性测量：目标 1e-3 + cutoff 1e-4 得到 260 RHS / 阶 70 / `20.10 s`、稳态网格证书
`1.64e-4`；目标 1e-4 + cutoff 1e-5 为 320 RHS / 阶 96 / `23.59 s`、`5.06e-6`。阶数从
47 升到 70 再升到 96，且这些行改变了 closing 阈值，不构成与 stock 的等 cutoff 对比。
判定：成本不可行 —— 记录原文 The data do not establish a small-order, all-output
guaranteed dynamic ROM。
复现：对应的 `bridge_transient_svd_1e4.json` / `_svd_1e5.json`（脚本已退役）。

#### B19  八组 HTC 的 17 点低频面设计

**判定作用域.** `PORT-STEP + PORT-FIXED-S + COST`

决定性测量：304 个 full RHS / 阶 56 / `8.01 s`，最坏 step entry `8.2647e-4`（通过
1e-3 有限时间目标），比 seed 20260805（466 RHS / 阶 51 / `28.98 s` / `2.2577e-3`）
少 35% 解、约快 3.6 倍。但最坏 steady entry `4.5775e-3`，**未通过** 1e-3 稳态目标；
`s <= 1/dt` 只是结构性假设，不是 40 步 transfer 的先验界；bottom 暴露面的选择是
经验性的。
判定：被反例证伪（该设计未达到它自己的 1e-3 稳态判据：最坏 steady entry `4.5775e-3`）。
复现：`explore_frequency_faces.py --include-stock`（脚本已退役）。

#### B20  八组 HTC：仅在单个 Zolotarev 种子处加 DC

**判定作用域.** `PORT-STEP + PORT-FIXED-S`

决定性测量：308 次求解 / 阶 56 / `8.22 s`，292 个验证参数上最坏 step `6.2203e-4`、
最坏 steady `4.5643e-3`（未过 1e-3）；三种设计中只有把 DC 铺满 bottom 面（372 RHS /
阶 60 / `9.36 s`，step `6.2230e-4`、steady `5.6313e-4`）才把稳态压到 1e-3 以下。
判定：被反例证伪。
复现：`explore_frequency_faces.py --include-dc --compare-seed-dc`（脚本已退役）。

#### B21  多组数早期探索：全局稳态切向风险排名的 5 点，与欠分辨暴露面加密

**判定作用域.** `PORT-STEP + PORT-FIXED-S`

决定性测量：5 点排名在 4 组最坏瞬态 `8.255e-4`、6 组 `9.805e-4`，到 8 组升到
`8.662e-3` 而失效；8 组的欠分辨七点面为 `7.681e-2`，加两个棋盘格点后仍只有
`2.504e-2`。这些初步脚本与结果在一次 workspace 重置中丢失，记录声明只有 292 点全集
比较是可复现主张。
判定：被反例证伪（维数升高时低点数全局面排名失效：8 组 `8.662e-3`）。“脚本与结果在一次
workspace 重置中丢失”是证据质量注释，不是分类。
复现：记录未给出这些初步运行的命令。

#### B22  两组 Case 1：低频段三点面（80 解）与三切点铺满全部频移（144 解）

**判定作用域.** `PORT-STEP + PORT-FIXED-S + COST`

决定性测量：80 解版为 70 参数上最坏 step `7.9102e-4`（通过），但最坏 steady
`3.4789e-3`（70 参数）/ `4.3345e-3`（359 评估，最坏点 `(10000, 17.7828)`）未过 1e-3；
三切点全频移版在 17x17 网格上为 `9.3908e-4` step / `3.9497e-3` steady，被低频面设计的
`5.4916e-4 / 8.2245e-4` 支配。记录原文 The exposed edge is a finite geometric sampling
rule, not an established two-dimensional minimax Zolotarev rule。
判定：被反例证伪（两个变体都未达到它们自己的判据：80 解版最坏 steady `3.4789e-3`、
三切点全频移版 `3.9497e-3`，均超 1e-3）。
复现：`explore_two_group_frequency_faces.py`（脚本已退役）。

#### B23  两组 Case 1：三低频点面 + 仅种子处 DC（84 解）

**判定作用域.** `PORT-STEP + PORT-FIXED-S`

决定性测量：70 点验证集上看似达标（最坏 step `8.2083e-4`），但在 70 点 + 17x17 网格
共 359 个评估上实际为 `1.0378e-3`（最坏点 `(10000, 17.7828)`）；65 点/边的高 HTC 边
审计 200 个评估上更达 `1.0454e-3`。三处全 DC 的 92 解版在同样集合上是
`5.4916e-4 / 8.2245e-4`。
判定：被反例证伪 —— 更密网格与边扫描发现了 70 点集漏掉的误差；有限验证集不能当连续箱
保证。
复现：`explore_two_group_frequency_faces.py --grid 17`、`--edge-grid 65`（脚本已退役）。

#### B24  1 mm 上用稀疏直接 LU 做谱准备与验证参考

**判定作用域.** `COST`

决定性测量：直接 LU 版谱准备 `101.03 s`（峰值约 `4.37 GB`），总提取 `138.29 s`，
比 stock 的 `67.95 s` 更慢（尽管解更少）；单次 1 mm 稀疏 LU 分解约 `47 s` / `4 GB`。
改用 AMG-CG 逆作用后谱准备 `18.91 s`、总提取 `55.93 s`、`668 MB`，两种方法节点相对差
约 `1.2e-13` 与 `1.6e-13`，6 个代表性参数上误差差最大 `6.4e-12` (step) 与
`9.4e-12` (steady)。
判定：成本不可行 —— 直接 LU 在 1 mm 上的时间与内存代价不可接受，被 AMG 预条件 CG
（`rtol = 1e-10`）取代；记录不把迭代逆声明为正式认证的谱包围。
复现：`explore_one_mm_frequency_faces.py --output ...`（直接版）与 `--fast-spectrum`
（AMG 版）（脚本已退役）。

---

### B3. 压缩、缺陷传播与 inexact-moment 路线

#### C1  inexact-moment 动态桥

**判定作用域.** `PORT-SYSTEM`

主张：把 inexact Galerkin moments 经“最小对称邻近 SPD 系统 -> 局部精确 Hermite moments
-> 三项 H2 界”接到 FANCTASTIC 动态误差定理，相对 H2 误差不超过
`eta_F + gamma_f (1 + eta_F) + d_WV / norm(H_V)`，其中 `delta = norm(S_R Y^+)` 只需残差
QR 之后的小稠密 SVD。
决定性测量：代数核心在真实数据上被证实（`delta` 两种形式十二位一致，`V = I` 时
`1.4e-13`，H2 与 80 点 Gauss--Legendre 相对差 `5.9e-5`），但**定理假设在交付基上被直接
违反**：`q = 44 > m = 35/36` 使 X 秩亏，兼容缺陷 `norm(R - R X^+ X)/norm(R)` 达 `0.45`，
此时根本不存在满足 `F X = R` 的对称 F；H2 项要求 `delta < lambda_min(B)` 却在四个箱角
全部失败 `4.9e3 .. 1.8e5` 倍，`eta_F = inf`。`d_WV` 在交付基上恒为零（实测 `<= 2.8e-8`）。
判定：被反例证伪 —— 所需前提在交付基的实测数据上直接失效（`q = 44 > m = 35/36` 使 X 秩亏、
兼容缺陷达 `0.45`、四个箱角 `delta / lambda_min` 全部超 `4.9e3` 倍），`eta_F = inf`。第 9 节
的结论是该路线应在进一步认证工作之前被否掉。
复现：`bench_inexact_moment_margins.py --mesh 5 --grid 9 --full-h2 --subset-sizes 4 13 26 44`
（bench 已退役；同族的 `test_inexact_moment_theory.py` 仍在维护）。

#### C2  在交付基上实测桥量裕度（判决实验）

**判定作用域.** `PORT-SYSTEM`

决定性测量：`sup_h delta(h)` 在管线列数下无法便宜地变小 —— `delta / norm(B)` 达
`8.1 .. 5.1e1`（stock）与 `5.2e4 .. 2.0e5`（design），`delta / lambda_min` 达
`2.5e5 .. 9.3e6` 甚至 `1.3e9 .. 1.3e11`；`sup_h d_WV(h)` 在交付基上恒零、在 pre-SVD span
上 `<= 3.5e-4`，与交付 ROM 自身的 H2 误差同量级，无法分离它本要刻画的压缩效应。抽取
循环实际执行的逐列欧氏残差判据（训练探针上 `<= 1e-3`）比它想控制的块量低 2..7 个数量级，
差距来自 `Y^+` 放大而不是度量变换。
判定：被反例证伪 —— 两个必要的桥量在交付基上都不成立：块条件数让 `delta` 比逐列残差高
2..7 个数量级，`delta < lambda_min` 在该模型内不可修复；判决实验的结论就是在进一步认证
之前否掉该路线。
复现：同 C1。

#### C3  LU cache 测量 bug（INEXACT_RK_DEFECT_CASE1.md 的更正段）

**判定作用域.** `CORRECTION`

原报告：ideal local exact-moment 空间 U 在 `(0.99999, 118.58)` 的相对 H2 误差
`2.61e-2`，据此判定缺陷传播桥“第一项就失败”。更正后：同一角上 U 的真实误差是
`2.8e-6`（比交付 ROM 好 117 倍），全网格 U 最差 `3.8e-4`，四个共享同一频率计划的基底上
ideal-space 误差逐位相同。被污染的恰好是所有走分解路径的量（U、`gap g_defect`、逐 shift
缺陷 `eta` 与权重 c）；两个 bench 起初“相互印证”只是因为它们共享同一个 bug。
根因：每个 run 只建一个 LU cache，而 `factorize_shifts` 只用 shift 值作键，于是第一个
HTC 点之后的所有点都在用别的点的分解。
判定：测量更正 —— `2.61e-2` 与“第一项失败”的读法全部撤回。
复现：`bench_inexact_rk_defect_case1.py --mesh 5 --grid 5 --linearity`（脚本已退役）。

#### C4  逐 shift 加性 defect 预算

**判定作用域.** `PORT-SYSTEM`

更正后仍然成立的判决：该加性预算被否，不是因为第一项失败，而是因为**太松** ——
`sum_k c_k` 稳定高估联合缺陷 gap `1.3..25` 倍、高估它本要界的交付误差 `1.2..22` 倍；
在固定 h 下唯一既适定又占主导的项是 pre-SVD -> delivered 的压缩（四角 `38..368` 倍，
独立复现了 A7 在 2.5 mm 上的约 147 倍）。预算坐在 `5.9e-3`，无法认证 ROM 实际达到的
`3.3e-4`，更无法认证 `2e-3` 的工业阻抗目标。逐列权重继承 moment 块在 35 维空间里的
病态冗余，重新加权不是修复（可逆右乘不改变 `F X = R` 的可行性与最小扰动）。
判定：界太松。被否掉的只是“逐 shift 加性聚合”，不是 defect-aware 桥本身。
复现：同 C3。

#### C5  active-set 原型：合成先行 + 真实 Case 1 验证

**判定作用域.** `COST`

**合成先行（20/40/80 未知数人工模型）.** 最终上界在全部 12 个 shift/size 组合上都超过独立
21x21 稠密探针的观测绝对误差；新算子 `5..6` 个、RHS 解 `10..12` 次，而同一组参数的全张量
需要 `20..24` 次；证书工作总量 `884 / 586 / 416` 次单元评估，耗时 `1.90 / 1.16 / 1.22 s`。
它只证明 operator sparsity，**不构成成本结论** —— 比较遗漏了每次迭代/每个 shift 都要重建
的 Woodbury 表准备，模型也只有 20..80 个未知数。

**真实 Case 1（5 mm）.** 决定性测量：绝对阈值那轮确实通过（箱界 `0.20165 K/W` 对阈值
`0.20603 K/W`，只用 20 个富化 RHS / 5 个共享算子），但其同点最差相对误差是 `4.279e-3`，
说明过绝对阈值不保证同点相对容差；相对目标那轮用 24 个富化 RHS / 6 个共享算子 / 阶 29，
端到端 `118.10 s`（stock `1.12 s`，慢约 100 倍），界停在 `1.908e-2`，**未解决**。根因是
exact-map oracle 每次迭代、每个 shift 都重建 shift 相关的稀疏 LU 与 Woodbury 表（至少
20160 / 23520 次边界列回代）。记录结论：仅“更少的 FOM 富化解”不构成更便宜的提取器。
判定：成本不可行。
复现：`bench_active_set_case1.py --max-solves 20 --max-cells 128 --probe-grid 5`
（脚本已退役）。

#### C7  在未改动的 stock 最终基上对连续 HTC 箱做残差认证（零新增快照）

**判定作用域.** `PORT-FIXED-S`

决定性测量：两个种子的四个端口在低 HTC 角全部超过 1e-3（最差 `1.5582e-2`，
seed 20260805 port 3 shift `5.2453e-4`；`8.9471e-3`，seed 7 port 1 shift `1.3135e-3`），
而直接范数与 Gram 表达在同一批点上一致到 `5.0e-9` 与 `6.6e-11`，所以违反不是约化求值的
假象；未截断快照在同一 9x9 探针上只有 `4.7287e-5 / 1.0732e-4`。
判定：被反例证伪 —— 生产 SVD 截断按快照方差而非方程残差排序，它优化传递误差而把方程
残差判据留在 1e-3 之上；全箱认证只需一个点违反，因此任何只放在抽取之后的接受证书都不可
能成立。
复现：`bench_reduced_residual_stock.py --mesh 5 --grid 1`（及 `--grid 9`）

#### C9  低秩跨边界耦合（参数无关截断 SVD + 轴流形）

**判定作用域.** `PORT-FIXED-S + COST`

最强结果：轴张成的精确恒等式在真实模型上被数值确认（相对差 `3.8e-16` 与 `1.3e-15`）；
只把 `Phi_12` 换成 rank-4 近似时，五个参数点上的最大相对 DC 传递差为 `3.01e-5` (5 mm) /
`4.76e-5` (2.5 mm)，相对于 rank-0 的 `599 / 598` 是决定性的。
失败点：全局传递界本身不可用 —— r = 4 时 5 mm 约 `1.04e4`、2.5 mm 约 `1.32e4`
（相对），绝不能当 1e-3 认证；脚本要形成整个参考 Green 块（5 mm 280 个边界解、2.5 mm
1072 个），比 stock 随机抽取更贵，所以测得的秩不是经济采样器；精确轴张成定理在有限轴
采样与不变的 1e-3 closing SVD 之前就适用。
判定：界太松 + 成本不可行。前置诊断同时否掉两条近似路线：全箱居中 Neumann 比
`0.9974 (5 mm) / 0.9977 (2.5 mm)` 使低阶全局 Taylor 场不适用；DC 场的四角多线性插值
在 11x11 对数 HTC 审计下最大相对谱阻抗误差 `5.14e-2`，其全局 Bernstein 包络超过上角点
阻抗的 `2e5` 倍。
复现：`probe_cross_coupling.py 5`（及 `2.5`）（脚本已退役）。

#### C10  已移出：先验闭式采样的 handoff

该路线是**开放候选，不是失败路线**，因此不再作为条目留在本文件。已被验证的部分
（Woodbury 在 ONE 参数处精确给出整个参数族、射线极点快速求值器、已知的三个计数陷阱）
连同实测数字记在 `README.md` 第 6 节第 7 条。它原先挂在 `THEORY.md` 的 P1「Robin 解
流形的 n-width 衰减」上，该命题仍是排除项，不在主理论线上：采样规模的最优性应由
`N_FOM` 与成本结构回答，而不是由 state-space n-width 速率回答。

---

### B4. 支撑诊断（不是失败路线）

以下内容曾被当作候选交付准则，测量结果**支持**其技术部分，但都不构成交付准则，
因此按上面的规则不属于四类负结果，单独留档。

#### D1  span 内残差感知压缩 + 稳定的正交余量残差表示

正面部分：残差贪心在两个种子上都在阶 49 通过（`9.5627e-4 / 9.5764e-4`），而 leading-SVD
要到 67 / 66 才首次通过；投影 QR 把与直接全向量残差的最大绝对偏差从原始 Gram 的
`1.17e-8 / 1.20e-5` 降到 `4.72e-11 / 1.24e-13`。
代价与限制：阶数上升 13--14（49 对 35/36），基从约 `0.37/0.38 MB` 涨到约 `0.52 MB`；
最差残差在 Galerkin 富化下不单调（seed 20260805 的 SVD rank 35 与 40 分别给
`1.558e-2` 与 `1.673e-2`）；本记录只是离散诊断，不是连续箱证书，也不是原
Hankel/冲激误差定理。更强的方程残差判据**并不证明**会改善原 FANSTIC 动态界。
复现：`bench_residual_compression.py --grid 9 --max-order 100`、
`bench_reduced_residual_stock.py --grid 9`（脚本已退役）。

---

### B5. 仍然在维护的对象

被否掉的是一次性实验，不是这套工具。当前保留并在 `README.md` 中登记的有：

```text
certified_box.py            连续参数盒证书（单元 Bernstein 包络、分支定界、选择规则）
deterministic_design.py     Zolotarev 种子 + 确定性贪心选点 + 快照组装 + 频率计划
residual_certificate.py     A(h_min)-Riesz 逐点残差证书
exact_error.py              精确参数映射与误差见证（全阶对照）
zolotarev.py                有限区间 Zolotarev 规则与每群谱区间
certify_extraction.py       驱动：设计 + stock 基线 + 证书 + 全阶验证
bench_certificate.py （tightness / bandb 两个 mode）/ bench_pareto_budget.py /
bench_dynamic_bridge_toy.py / test_certified_sampling.py
records/PORT_CERTIFICATE_AND_BUDGET.md
```

`records/PORT_CERTIFICATE_AND_BUDGET.md` 是**当前仍在使用的正面结论**（证书的有效性、
盒分支定界、提取预算计数约定），因此保留在本目录而不是并入本文；
动态桥在人工问题上的一阶否证已经并入本文 Part A。
尚未证明的端口系统范数保证见 `THEORY.md` 的 P0 动态传递定理。
