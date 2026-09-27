# Dynamic bridge toy: the transfer error is first order in the moment defect

Negative result on the quadratic bridge, from the artificial problem the bridge
has to survive before any of the heavier machinery is justified.

## 1. What is asked

The only open P0 item is the dynamic bridge: the certificate controls matching
resolvent defects `delta_j`, and the vendor guarantee is stated for the impedance
and for a space-time impulse energy.  The exact MPMM space has a Hermite
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
linear state bridge refuted    e_state / delta_* -> inf
```

The excess is taken over *both* rotation signs because the first-order Frechet
derivative flips sign with `Q -> -Q`: if it is nonzero, at least one branch must
raise the total error above the exact-MPMM baseline.

## 2. Model and definitions

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
e_Hankel     = ||H_U - H_V||_Hankel / ||H||_Hankel,
e_state      = ||U e^(-B_U t) f_U - V e^(-B_V t) f_V||_L2(0, inf; B)
               / ||e^(-B t) f||_L2(0, inf; B).
```

`Q` is a fixed orthonormal basis of the orthogonal complement of `U` (2 columns
for 4x4, 4 for 8x8, so `V_pm` has the same dimension as `U`); `theta` runs over
`1e-1 ... 1e-6`.  The moment space itself comes from a truncated SVD of the
snapshot matrix, not a bare QR, so a rank-deficient snapshot set (repeated
eigenvalues) does not silently contribute padding directions.

All dynamic quantities are exact: `H2` and `L2` distances from Lyapunov solves on
the block system that carries both semigroups, the Hankel norm from the largest
Hankel singular value of the difference realization (controllability and
observability Lyapunov equations), the impulse energy likewise.  No frequency
grid and no time stepping enters any verdict.

Spectrum families (`u_i`, `lambda_i = kappa^u_i`): log-uniform, endpoint-cluster,
low-cluster, repeated.  Sources: `f` flat, and `f_i ~ lambda_i^(1/4)`, which
balances the per-mode `f_i^4 / (2 lambda_i)` H2 contribution.

## 3. Results at the delivered box condition number

Actual 5 mm box plan: `kappa = 4.1400e+01 / 4.2795e-05 = 9.674e+05`.  Orders are
the fitted local exponents over the smallest rotations above the double-precision
floor, for the `+` branch (the `-` branch behaves the same); the ratio columns are
the smallest-`theta` values of the sharp-baseline excess.

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order e_state | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.30e-05 .. 9.58e-01 | 1.96 | 0.33 | 1.00 | 1.41 | 5.361e-05 | 2.572e-02 | 3.633e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.25e-05 .. 9.54e-01 | 1.72 | 2.14 | 1.00 | 1.57 | 1.844e-04 | 1.210e-02 | 1.640e-05 |
| 4x4 | log-uniform | flat | 2/2 | 3.94e-05 .. 9.71e-01 | 1.00 | 0.83 | 1.00 | 1.01 | 1.336e-02 | 6.598e-02 | 4.330e-04 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 5.64e-06 .. 4.18e-01 | 1.00 | 1.00 | 1.00 | 1.00 | 4.516e-01 | 1.384e+00 | 2.166e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.90e-07 .. 9.88e-02 | 0.36 | 1.34 | 1.00 | 1.00 | 1.595e-02 | 1.093e-02 | 5.166e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.87e-02 | 0.72 | 1.36 | 1.00 | 1.00 | 1.193e-02 | 9.101e-01 | 3.473e-04 |
| 4x4 | repeated | flat | 2/2 | 3.24e-05 .. 9.55e-01 | 1.00 | 0.72 | 1.00 | 2.53 | 4.955e-04 | 7.343e-03 | -3.237e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.23e-05 .. 9.53e-01 | 0.79 | 0.26 | 1.00 | 2.26 | 6.170e-04 | 2.575e-02 | -6.348e-05 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.44e-04 .. 9.99e-01 | 2.00 | 1.68 | 1.00 | 2.02 | 6.874e-04 | 7.121e-03 | 6.534e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.43e-04 .. 9.96e-01 | 2.00 | 1.72 | 1.00 | 2.00 | 4.854e-04 | 8.607e-04 | 4.862e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.31e-04 .. 8.96e-01 | 1.00 | 0.99 | 1.00 | 0.92 | 3.383e-02 | 6.429e-02 | 6.826e-04 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.44e-04 .. 8.22e-01 | 1.00 | 1.01 | 1.00 | 0.99 | 3.276e-02 | 1.248e-01 | 6.462e-03 |
| 8x8 | low-cluster | flat | 4/4 | 1.05e-06 .. 1.06e-01 | 1.00 | 1.21 | 1.00 | 1.00 | 1.487e-01 | 1.909e-01 | 1.019e-02 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.17e-07 .. 9.18e-02 | 0.98 | 0.94 | 0.99 | 1.00 | 3.995e-02 | 6.000e-02 | 1.198e-03 |
| 8x8 | repeated | flat | 2/4 | 5.73e-04 .. 1.00e+00 | 2.00 | 1.48 | 1.00 | 2.00 | 8.805e-04 | 4.963e-03 | 8.812e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.72e-04 .. 9.98e-01 | 2.00 | 2.14 | 1.00 | 2.00 | 4.538e-04 | 1.080e-03 | 4.516e-04 |

Two full series, to show that the plateaus are asymptotic and not a single lucky
point:

4x4 log-uniform flat

| theta | delta_* | e_H2 | e_H2/delta_* | e_Hankel | e_Hankel/delta_* | e_state | e_state/delta_* | excess | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-01 | 9.712e-01 | 9.550e-01 | 9.834e-01 | 9.942e-01 | 1.024e+00 | 7.386e-01 | 7.606e-01 | 8.506e-01 | 8.758e-01 |
| 3e-02 | 7.836e-01 | 6.946e-01 | 8.864e-01 | 8.738e-01 | 1.115e+00 | 5.790e-01 | 7.389e-01 | 5.952e-01 | 7.596e-01 |
| 1e-02 | 3.681e-01 | 1.904e-01 | 5.172e-01 | 3.044e-01 | 8.271e-01 | 2.741e-01 | 7.447e-01 | 1.168e-01 | 3.173e-01 |
| 3e-03 | 1.239e-01 | 2.392e-02 | 1.931e-01 | 4.105e-02 | 3.313e-01 | 9.251e-02 | 7.466e-01 | 3.337e-03 | 2.693e-02 |
| 1e-03 | 3.942e-02 | 2.732e-03 | 6.930e-02 | 4.808e-03 | 1.220e-01 | 2.944e-02 | 7.468e-01 | 9.861e-05 | 2.501e-03 |
| 3e-04 | 1.247e-02 | 3.736e-04 | 2.996e-02 | 6.691e-04 | 5.365e-02 | 9.313e-03 | 7.467e-01 | 1.047e-05 | 8.399e-04 |
| 1e-04 | 3.944e-03 | 7.185e-05 | 1.822e-02 | 1.266e-04 | 3.210e-02 | 2.945e-03 | 7.467e-01 | 2.167e-06 | 5.496e-04 |
| 3e-05 | 1.247e-03 | 1.851e-05 | 1.485e-02 | 3.160e-05 | 2.534e-02 | 9.312e-04 | 7.467e-01 | 5.841e-07 | 4.684e-04 |
| 1e-05 | 3.944e-04 | 5.462e-06 | 1.385e-02 | 9.432e-06 | 2.392e-02 | 2.945e-04 | 7.467e-01 | 1.748e-07 | 4.433e-04 |
| 3e-06 | 1.247e-04 | 1.689e-06 | 1.355e-02 | 2.865e-06 | 2.297e-02 | 9.312e-05 | 7.467e-01 | 5.430e-08 | 4.354e-04 |
| 1e-06 | 3.944e-05 | 5.306e-07 | 1.346e-02 | 6.623e-07 | 1.679e-02 | 2.945e-05 | 7.467e-01 | 1.707e-08 | 4.330e-04 |

8x8 low-cluster flat

| theta | delta_* | e_H2 | e_H2/delta_* | e_Hankel | e_Hankel/delta_* | e_state | e_state/delta_* | excess | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-01 | 1.065e-01 | 1.885e-02 | 1.770e-01 | 1.909e-02 | 1.793e-01 | 8.838e-02 | 8.301e-01 | 6.124e-03 | 5.752e-02 |
| 3e-02 | 3.335e-02 | 4.990e-03 | 1.496e-01 | 7.568e-03 | 2.270e-01 | 2.738e-02 | 8.210e-01 | 7.381e-04 | 2.213e-02 |
| 1e-02 | 1.051e-02 | 1.544e-03 | 1.469e-01 | 2.547e-03 | 2.424e-01 | 8.601e-03 | 8.184e-01 | 1.459e-04 | 1.388e-02 |
| 3e-03 | 3.320e-03 | 4.870e-04 | 1.467e-01 | 8.208e-04 | 2.472e-01 | 2.714e-03 | 8.176e-01 | 3.770e-05 | 1.136e-02 |
| 1e-03 | 1.050e-03 | 1.539e-04 | 1.467e-01 | 2.611e-04 | 2.488e-01 | 8.578e-04 | 8.173e-01 | 1.108e-05 | 1.056e-02 |
| 3e-04 | 3.319e-04 | 4.868e-05 | 1.467e-01 | 8.272e-05 | 2.493e-01 | 2.712e-04 | 8.172e-01 | 3.420e-06 | 1.031e-02 |
| 1e-04 | 1.049e-04 | 1.539e-05 | 1.467e-01 | 2.620e-05 | 2.497e-01 | 8.576e-05 | 8.172e-01 | 1.073e-06 | 1.023e-02 |
| 3e-05 | 3.319e-05 | 4.868e-06 | 1.467e-01 | 8.485e-06 | 2.557e-01 | 2.712e-05 | 8.172e-01 | 3.385e-07 | 1.020e-02 |
| 1e-05 | 1.049e-05 | 1.540e-06 | 1.467e-01 | 5.203e-06 | 4.958e-01 | 8.576e-06 | 8.172e-01 | 1.070e-07 | 1.019e-02 |
| 3e-06 | 3.319e-06 | 4.877e-07 | 1.470e-01 | 3.582e-06 | 1.079e+00 | 2.712e-06 | 8.172e-01 | 3.382e-08 | 1.019e-02 |
| 1e-06 | 1.049e-06 | 1.542e-07 | 1.469e-01 | 2.005e-07 | 1.911e-01 | 8.577e-07 | 8.173e-01 | 1.069e-08 | 1.019e-02 |

## 4. Other condition numbers

The same run covers `kappa = 1e2, 1e4, 1e6`; the order of `e_H2` does not depend
on `kappa` for the two non-degenerate families:

`kappa = 1e2`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order e_state | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 1.04e-06 .. 1.04e-01 | 0.12 | 0.14 | 1.00 | 1.10 | 1.368e-02 | 1.764e-02 | -2.220e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 9.65e-07 .. 9.63e-02 | 0.22 | 1.00 | 1.00 | 0.86 | 2.460e-02 | 8.231e-01 | 3.530e-04 |
| 4x4 | log-uniform | flat | 2/2 | 1.60e-06 .. 1.59e-01 | 1.00 | 1.09 | 1.00 | 1.00 | 3.769e-01 | 5.500e-01 | 2.412e-02 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 1.15e-06 .. 1.09e-01 | 1.00 | 0.58 | 1.00 | 1.00 | 5.097e-01 | 2.522e+00 | 2.170e-01 |
| 4x4 | low-cluster | flat | 2/2 | 9.93e-07 .. 9.91e-02 | 0.31 | 1.51 | 1.00 | 1.01 | 1.615e-02 | 1.479e-01 | 1.230e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.64e-07 .. 9.62e-02 | 0.65 | 0.96 | 1.00 | 1.00 | 1.935e-02 | 1.381e-01 | 1.151e-04 |
| 4x4 | repeated | flat | 2/2 | 1.04e-06 .. 1.04e-01 | 0.05 | 0.69 | 1.00 | -0.04 | 1.234e-02 | 6.290e-03 | 1.619e-02 |
| 4x4 | repeated | h2-balanced | 2/2 | 9.65e-07 .. 9.63e-02 | 2.01 | -0.01 | 1.00 | 0.09 | 3.331e-04 | 6.009e-03 | 3.377e-04 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.64e-06 .. 4.18e-01 | 0.18 | 1.68 | 1.00 | 0.24 | 2.636e-03 | 2.232e-03 | 0.000e+00 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.31e-06 .. 3.86e-01 | 0.51 | -0.35 | 1.00 | 0.10 | 2.865e-03 | 4.396e-03 | 2.933e-03 |
| 8x8 | log-uniform | flat | 4/4 | 2.94e-06 .. 2.81e-01 | 0.99 | 0.15 | 1.00 | 1.00 | 3.074e-02 | 5.810e-01 | 1.783e-03 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 2.26e-06 .. 2.24e-01 | 0.99 | 0.73 | 1.00 | 1.00 | 4.783e-02 | 2.031e-01 | 2.946e-03 |
| 8x8 | low-cluster | flat | 4/4 | 9.72e-07 .. 9.70e-02 | 1.16 | 0.76 | 1.00 | 1.00 | 1.775e-03 | 4.294e-01 | 1.357e-03 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 8.95e-07 .. 8.95e-02 | 0.98 | 0.22 | 1.00 | 1.00 | 4.797e-03 | 4.083e-02 | 2.044e-03 |
| 8x8 | repeated | flat | 2/4 | 5.84e-06 .. 5.04e-01 | 2.00 | 2.62 | 1.00 | 0.24 | 2.741e-04 | 3.503e-04 | 2.729e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.12e-06 .. 4.43e-01 | 0.67 | 0.79 | 1.00 | 0.51 | 1.634e-03 | 6.123e-02 | 4.222e-04 |

`kappa = 1e4`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order e_state | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.47e-06 .. 3.30e-01 | 0.47 | -1.00 | 1.00 | 1.02 | 6.477e-03 | 1.166e-01 | 1.979e-04 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.41e-06 .. 3.21e-01 | 1.02 | 1.59 | 1.00 | 1.00 | 1.168e-03 | 7.039e-04 | 1.625e-04 |
| 4x4 | log-uniform | flat | 2/2 | 7.33e-06 .. 6.12e-01 | 1.00 | 1.12 | 1.00 | 1.00 | 1.132e-01 | 1.384e-01 | 7.708e-03 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 2.51e-06 .. 2.35e-01 | 1.00 | 0.93 | 1.00 | 1.00 | 5.045e-01 | 2.002e+00 | 4.512e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.93e-07 .. 9.91e-02 | 1.03 | -0.05 | 1.00 | 1.00 | 2.470e-03 | 4.694e-03 | 3.634e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.88e-02 | 0.54 | -0.50 | 1.00 | 1.00 | 1.860e-02 | 7.530e-01 | 2.808e-04 |
| 4x4 | repeated | flat | 2/2 | 3.44e-06 .. 3.24e-01 | 1.09 | 1.81 | 1.00 | 2.05 | 4.164e-04 | 3.671e-04 | -4.359e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.40e-06 .. 3.20e-01 | -0.18 | 1.72 | 1.00 | 2.39 | 4.459e-03 | 4.319e-03 | -1.664e-03 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.52e-05 .. 8.52e-01 | 2.03 | -0.38 | 1.00 | 2.11 | 1.984e-04 | 7.743e-03 | 1.714e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.47e-05 .. 9.64e-01 | 2.08 | 0.25 | 1.00 | 1.22 | 1.224e-04 | 1.399e-02 | 1.658e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.71e-05 .. 7.62e-01 | 1.00 | 0.97 | 1.00 | 1.00 | 5.386e-02 | 9.238e-02 | 1.354e-03 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.87e-05 .. 6.57e-01 | 1.00 | 0.92 | 1.00 | 1.00 | 5.559e-02 | 2.798e-01 | 1.027e-02 |
| 8x8 | low-cluster | flat | 4/4 | 9.81e-07 .. 9.82e-02 | 0.98 | 1.00 | 1.00 | 1.00 | 6.050e-02 | 7.890e-02 | 1.317e-03 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.06e-07 .. 9.04e-02 | 0.85 | 1.77 | 1.00 | 1.00 | 2.833e-02 | 8.335e-02 | 3.174e-03 |
| 8x8 | repeated | flat | 2/4 | 5.82e-05 .. 9.85e-01 | 1.88 | 0.72 | 1.00 | 2.06 | 1.466e-04 | 9.673e-03 | -9.599e-05 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.72e-05 .. 9.69e-01 | 2.27 | 1.73 | 1.00 | 2.01 | 7.168e-05 | 6.422e-03 | 1.442e-04 |

`kappa = 1e6`

| size | spectrum | source | rank | delta_* range | order e_H2 | order e_Hankel | order e_state | order excess | e_H2/delta_* | e_Hankel/delta_* | excess/delta_* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4x4 | endpoint-cluster | flat | 2/2 | 3.35e-05 .. 9.59e-01 | 2.05 | 1.90 | 1.00 | 1.41 | 3.451e-04 | 4.516e-04 | 4.893e-05 |
| 4x4 | endpoint-cluster | h2-balanced | 2/2 | 3.31e-05 .. 9.55e-01 | 1.89 | 0.73 | 1.00 | 1.58 | 4.816e-05 | 1.555e-02 | 9.860e-06 |
| 4x4 | log-uniform | flat | 2/2 | 3.99e-05 .. 9.72e-01 | 1.00 | 0.47 | 1.00 | 1.01 | 1.315e-02 | 6.947e-02 | 4.261e-04 |
| 4x4 | log-uniform | h2-balanced | 2/2 | 5.68e-06 .. 4.20e-01 | 1.00 | 1.01 | 1.00 | 1.00 | 4.512e-01 | 1.164e+00 | 2.131e-02 |
| 4x4 | low-cluster | flat | 2/2 | 9.90e-07 .. 9.88e-02 | 0.86 | 1.01 | 1.00 | 1.00 | 1.079e-02 | 7.350e-03 | 5.259e-04 |
| 4x4 | low-cluster | h2-balanced | 2/2 | 9.89e-07 .. 9.87e-02 | 0.55 | 0.56 | 1.01 | 1.00 | 5.556e-03 | 1.204e-02 | 3.491e-04 |
| 4x4 | repeated | flat | 2/2 | 3.29e-05 .. 9.57e-01 | 1.14 | -0.05 | 1.00 | 1.44 | 3.648e-04 | 6.945e-03 | 1.705e-04 |
| 4x4 | repeated | h2-balanced | 2/2 | 3.29e-05 .. 9.54e-01 | 1.72 | 3.21 | 1.00 | 1.97 | 2.147e-04 | 1.291e-02 | 1.208e-04 |
| 8x8 | endpoint-cluster | flat | 4/4 | 4.51e-04 .. 9.99e-01 | 2.00 | 1.38 | 1.00 | 2.00 | 6.995e-04 | 5.347e-03 | 6.995e-04 |
| 8x8 | endpoint-cluster | h2-balanced | 4/4 | 4.50e-04 .. 9.96e-01 | 2.00 | 1.73 | 1.00 | 2.00 | 4.926e-04 | 5.828e-03 | 4.943e-04 |
| 8x8 | log-uniform | flat | 4/4 | 2.35e-04 .. 8.97e-01 | 1.00 | 0.99 | 1.00 | 0.92 | 3.368e-02 | 6.241e-02 | 6.946e-04 |
| 8x8 | log-uniform | h2-balanced | 4/4 | 1.47e-04 .. 8.23e-01 | 1.00 | 1.01 | 1.00 | 0.99 | 3.261e-02 | 1.245e-01 | 6.420e-03 |
| 8x8 | low-cluster | flat | 4/4 | 1.05e-06 .. 1.07e-01 | 1.00 | 0.10 | 1.00 | 1.00 | 1.456e-01 | 2.254e+00 | 1.026e-02 |
| 8x8 | low-cluster | h2-balanced | 4/4 | 9.17e-07 .. 9.18e-02 | 0.95 | 1.34 | 1.00 | 1.00 | 4.287e-02 | 6.216e-02 | 1.185e-03 |
| 8x8 | repeated | flat | 2/4 | 5.82e-04 .. 1.00e+00 | 2.00 | 2.14 | 1.00 | 2.00 | 8.956e-04 | 1.097e-03 | 8.953e-04 |
| 8x8 | repeated | h2-balanced | 2/4 | 5.81e-04 .. 9.98e-01 | 2.00 | 2.14 | 1.00 | 2.02 | 4.611e-04 | 1.098e-03 | 4.372e-04 |

## 5. What this does and does not refute

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
* the linear state bridge: `e_state / delta_*` plateaus at a positive constant
  (`0.75` for 4x4 log-uniform flat, `0.82` for 8x8 low-cluster flat), so a linear
  `Phi_state` remains possible and is compatible with the vendor's
  `epsilon` / `sqrt(epsilon)` split.
* a cancellation between `H - H_U` and `H_U - H_V`: measured and excluded by the
  excess columns above.

Mechanism, consistent with the numbers: at a matching shift the value error is
`||x - x_V||^2_(B + sigma I) = O(delta^2)`, but the derivative is
`H'(sigma) = -x^T x` and Galerkin only gives `x_V^T (B + sigma I) e = 0`, not
`x_V^T e = 0`, so the derivative mismatch is generically `O(delta)`.

## 6. Limits

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
* `e_state` is a diagnostic for the state impulse energy.  The vendor material
  states the impedance Hankel error as `< 2 epsilon` and a space-time impulse
  energy as `< 2 sqrt(epsilon)` but does not define that energy norm, so it is not
  claimed as the vendor norm here.
* Only relative ROM-to-ROM differences are reported; how large `delta_*` actually
  is in the real model stays the certificate's job.

## 7. The skeleton layer: invariants, rho_Sigma, K_sample and K_vec

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
  propagation does not explain the first-order transfer error of section 3.
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

## 8. The linear candidate survives the real axis

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
