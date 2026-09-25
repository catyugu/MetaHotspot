# Reassessing the FANTASTIC error target

The [Flotherm validation note](https://assets.ctfassets.net/dww76w587oxz/1ePeFrdaZXfoq0E1cFuM0T/e71cbcd824d7962c90319da1db28548a/Simcenter-Flotherm-BCI-ROM-Validation-2020.2.pdf)
states an impedance Hankel error below `2ε` and impulse-field energy error
below `2√ε`. Per the stated research goal, **relative** here means normalized
by the full model at the same Robin parameter. The validation note itself
does not spell out this normalization in the quoted passage. The
[Extended FANTASTIC paper](https://ieeexplore.ieee.org/document/9507439)
uses elliptic frequency shifts and a random HTC/residual loop; its cited
[2019 error-bound paper](https://ieeexplore.ieee.org/document/8777138)
has not been available in full for checking precisely which assumptions
yield those constants. In particular, a finite number of random passing
HTCs cannot prove a maximum over the continuous HTC domain.

## A necessary correction: all power inputs, not four energies added together

For `C xdot + A(p)x = G u`, with final Galerkin basis `V` **after** the
unmodified `1e-3` SVD, denote the exact and ROM temperature fields from an
impulse of amplitude vector `w` by `x(t;w)` and `x_r(t;w)`. Their error-energy
matrices satisfy

\[
w^T E(p)w=\int_0^\infty
\|x(t;w)-x_r(t;w)\|_{A(p)}^2\,dt,\qquad
E_0=\tfrac12G^TC^{-1}G.
\]

The **all-input** relative error is

\[
\eta_E(p)=\sqrt{\lambda_{\max}(E(p),E_0)}.
\]

The ratio `sqrt(tr(E)/tr(E0))` reported earlier only measures the *sum*
of four separate impulse energies. It can hide a bad linear combination
of simultaneously powered sources and does **not** justify an all-input
claim. `exact_impulse_energy` now returns the `4×4` matrix `E` and its
worst generalized eigenvalue. On the original, 2.5 mm, two-HTC Case 1:

| Physical HTCs | 92 solves / order 47, worst input | Random 126 solves / order 42, worst input |
| --- | ---: | ---: |
| `(100,100)` | `0.01560148` | `0.01564479` |
| `(5000,2)` | `0.00705458` | `0.00720730` |
| `(10000,1)` | `0.00529391` | `0.00546189` |
| `(10000,10000)` | `0.00531231` | `0.00547840` |
| `(1,1)` | `0.01625148` | `0.01693903` |

Both arms are below `2 sqrt(1e-3) = 0.06324555` **at these five HTCs**.
At `(100,100)` the deterministic arm's worst-input improvement is only
about 0.28%, appreciably less than suggested by the aggregated energies.

## The simple exact identities are useful diagnostics, not a sampling rule

Whiten the mass matrix, writing `T=C^-1/2 A C^-1/2`, `F=C^-1/2 G`, and
`Q=C^1/2 V(V.T C V)^-1/2`. For the augmented error realization define
`L=diag(T,Q.T T Q)`, `B=[F;Q.T F]`, `J=diag(I,-I)` and `D=[I,-Q]`.
Two standard continuous-time Lyapunov equations give the precise norms:

\[
LP+PL=BB^T,\quad LO+OL=D^TTD,\qquad
\|\mathcal H_{Z-Z_r}\|=\rho(P^{1/2}JP^{1/2}),\quad E=B^TOB.
\]

The full-model Hankel norm is the largest eigenvalue of the full-model
block of `P`. Both ratios are evaluated at the *same* `p`. The previous
report's claim that **one** Gramian supplies both requested all-input
metrics was wrong: `tr(D.T T D P)` supplies only `tr(E)`. The second
Lyapunov equation, or an equivalent small `4×4` matrix assembled through
shifted sparse solves, is needed for worst-input energy.

We audited the Hankel ratio with enlarged local resolvent witness spaces:

| Physical HTCs | Witness orders | Deterministic Hankel ratio | Random Hankel ratio |
| --- | --- | ---: | ---: |
| `(100,100)` | `140 / 180` | `2.89361e-4 / 2.89743e-4` | `3.82403e-4 / 3.82154e-4` |
| `(10000,1)` | `140 / 187` | `2.36511e-4 / 2.35356e-4` | `2.63807e-4 / 2.62501e-4` |
| `(1,1)` | `123 / 160` | `3.77295e-5 / 3.77267e-5` | `7.94780e-4 / 7.94791e-4` |

These are **projected-reference estimates**, not certified full-model
Hankel bounds. They do not establish `2ε` even at the displayed points.

## What should be removed from the algorithmic proposal

The prior attempt added a parameter-cell tree, exact center solves,
semigroup Lipschitz constants, Gramian residual bounds, and a final SVD
check. This is too much machinery to answer **where to take a few HTC
snapshots**. Its large-cell constants scale like `1/alpha^2` with the
slowest thermal decay rate `alpha`; the bound did not certify the full
Case-1 box. We have removed the corresponding `semigroup_box_bounds`
implementation. It should not be represented as an effective sampling
algorithm or as a proof that the final ROM meets the target.

## A cleaner research question

Hold the original elliptic frequency shifts and final SVD fixed. Study the
**one object actually sampled**, the parameter-dependent resolvent

\[
R(p,s)=(A(p)+sC)^{-1}G.
\]

For every complex `s` in the closed right half-plane, put
`S=A+sC` and `e=R-R_V`. Galerkin orthogonality gives the exact
**bilinear** matrix identity (transpose, not complex conjugation)

\[
Z(p,s)-Z_V(p,s)=
e(p,s)^TS(p,s)e(p,s).
\]

For real `s >= 0`, this matrix is positive semidefinite. At imaginary
frequencies `s=iω`, set `W_ω=A+|ω|C`. Since
`||W_ω^-1/2(A+iω C)W_ω^-1/2||_2 <= 1`, and the Hankel operator is a
compression of the transfer multiplication operator, we get

\[
\frac{\|\mathcal H_{Z-Z_V}\|}{\|\mathcal H_Z\|}
\le \frac{\sup_{\omega\in\mathbb R}
\|W_\omega^{1/2}e(p,i\omega)\|_2^2}{\|\mathcal H_Z\|}.
\]

Parseval gives an equally direct, but **different**, route to impulse
energy for *every* power vector `w`:

\[
w^TE(p)w=\frac1{2\pi}\int_{-\infty}^{\infty}
\|A^{1/2}e(p,i\omega)w\|_2^2\,d\omega.
\]

This is the clean structural reason to seek a resolvent approximation
with size `O(sqrt(ε))`: the impedance error is **quadratic** in the
Galerkin resolvent defect, while the field-energy norm is **linear** in
its size. It does **not** by itself provide the numerical constants `2`
and `2`, because the two normalizations and their frequency weights
must be related uniformly over `p` and the final SVD.

Thus a **single approximation problem on the joint parameter-frequency
domain** could guide HTC samples: the Robin resolvent is analytic in each
parameter until an operator pole, and those poles move away as `s` grows.
This gives a principled reason to allocate richer HTC interpolation at
low frequencies and fewer HTC points at high frequencies. Candidate
next steps are explicit parameter-domain complex analyticity bounds and
their resulting interpolation-degree estimates, followed by a theorem
relating the *final SVD basis's* uniform resolvent error to **both**
continuous-time relative norms.

**That sharp implication has not been proved.** Positivity of the transfer
error for real `s` alone does not control the needed imaginary axis. Until a
sound bridge to both norms, including the unchanged final SVD, exists,
the three-point frequency-conditioned design is an empirical improvement
over the random baseline, not a new FANTASTIC-style guaranteed algorithm.

Reproduction (results outside the repository):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/test_impulse_hankel_metrics.py
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_impulse_energy.py --output /tmp/bci-continuous-impulse-energy.json
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_impulse_hankel_witness.py --htc 100 100 --output /tmp/bci-hankel-witness-100.json
```
