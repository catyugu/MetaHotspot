# Residual certificate for the final 1e-3 SVD basis

## Research status: local boxes are not the sampling strategy

The certificates below concern small neighborhoods of one HTC point. They do
not certify the original whole HTC domain and should not be extended into a
parameter-space subdivision algorithm. This approach is retained only as a
diagnostic of a final SVD basis; the intended sampling principle must work on
the entire domain at once.

A global improvement of the Bernstein argument is available in exact
arithmetic. If `W(p,omega)` and a *polynomial trial residual* `r(p,omega)`
are expressed in the same tensor Bernstein basis with coefficients `W_alpha`
and `r_alpha`, then the matrix-fractional function is jointly convex:

```
    r(p,omega)^* W(p,omega)^-1 r(p,omega)
        <= max_alpha lambda_max(r_alpha^* W_alpha^-1 r_alpha).
```

Indeed each coefficient obeys the block positive-semidefinite inequality
`[W_alpha, r_alpha; r_alpha^*, t I] >= 0` for the maximum `t`; a convex
combination preserves it, and the Schur complement gives the claim. This
replaces the single pessimistic `W_min` by coefficient-dependent operators,
without subdividing the parameter domain. It still requires a good *global*
trial residual. On the 5 mm original two-HTC domain, the degree-one,
degree-two, and degree-three reduced Taylor witnesses gave absolute DC
transfer bounds of about `703.51`, `699.60`, and `695.76` under this sharper
inequality. These are unusably loose bounds, not observed ROM errors. Thus
the global matrix inequality alone does not rescue a low-degree Taylor
witness, and its exploratory implementation was discarded.

There is also a direct full-domain obstruction to a simple polynomial
expansion. Set `A_min=A(p_min)`, `A_max=A(p_max)`, `A_c=(A_min+A_max)/2`, and
`D=(A_max-A_min)/2`. The exact spectral radius of the centered perturbation
over the whole box is
`q=lambda_max(D, A_c)`; the worst corner attains it because all boundary
perturbations are positive semidefinite. Direct generalized eigenvalue solves
gave `q=0.99737344` at 5 mm and `q=0.99766085` at 2.5 mm. A uniform Neumann
remainder based on `q^m` therefore needs approximately 2,627 and 2,950
orders, respectively, even to make this factor `1e-3`. This is an argument
against this **particular uniform polynomial bound**, not a lower bound on
the number of required snapshots or on an output-weighted rational method.

The next theoretical target is a global source-weighted *bivariate rational*
trial field with a bound on its residual after the **unchanged** closing SVD.
One-dimensional Zolotarev estimates along individual rays do not control
the unsampled mixed HTC directions. Until such a bound is derived, the
small-node design has empirical comparisons but no claimed whole-domain
guarantee for either relative Hankel or impulse-energy error.

This experiment keeps the original two-HTC Case 1, the extractor's elliptic
time shifts, the column normalization, and the closing `1e-3` snapshot SVD.
The fixed Zolotarev-seed/high-top-face design samples three HTCs at low
shifts, only the seed at higher shifts, and three DC snapshots per source.
The stock comparison uses `probe_rounds=10`, seed `20260805` and the same
closing SVD. Sampling is faster in the reported runs, but there is no claim
that these three HTC points are a minimax solution to the two-variable
rational approximation problem.

## Precise relative metrics

For `C xdot + A(p)x = G u`, use the *same* `p` in both the full and reduced
operators. Write `S=A+iω C`, `X=S^-1 G`, `X_V=V(V^T S V)^-1 V^T G` and
`Z=G^T X`. The requested impedance ratio is
`||H_(Z-Z_V)(p)||/||H_Z(p)||`. For every impulse amplitude vector `w`, the
relative field-energy ratio is the supremum of

```
    [integral_0^infty ||(x(t)-x_V(t)) w||_A^2 dt /
     (w^T E0 w)]^1/2,         E0 = (1/2) G^T C^-1 G.
```

The target values at `epsilon=1e-3` are `0.002` and `0.0632455532`.
The 4-by-4 generalized eigenvalue from `exact_impulse_energy` evaluates
the actual **worst input combination** at individual parameters; adding
four separate impulse energies does not evaluate this metric.

## A common post-SVD resolvent certificate

For `ω>=0` let `W(p,ω)=A(p)+ω C` and `r=G-S X_V`. One can bound the final
Galerkin ROM directly, without transferring a guarantee through the SVD:

```
    Z-Z_V = (X-X_V)^T S (X-X_V),
    ||Z-Z_V||_2 <= ||W^1/2 (X-X_V)||_2^2.
```

The transpose is bilinear, not the complex conjugate transpose. For any
explicit trial field `V Q_trial`, set `r_trial=G-S V Q_trial`. The weighted
resolvent has norm at most `sqrt(2)`; the oblique Galerkin projection and
its complementary projection both have `W`-operator norm at most `sqrt(2)`.
Consequently,

```
    ||W^1/2 (X-X_V)||_2 <= 2 ||W^-1/2 r_trial||_2.
```

At exactly zero frequency this factor `2` improves to `1`. A uniform
bound on the right therefore bounds *both* the impedance and the
temperature field after SVD. The Hankel operator is a compression of the
frequency-domain transfer multiplier, so its error norm is no larger than
the maximum impedance transfer error along the imaginary axis.

### Finite parameter/frequency cells

Given a rectangular cell in the two HTCs and `ω`, solve the **small**
reduced system at its center. Construct a degree-`m` Taylor polynomial in
the three centered coordinates for its reduced solution; this polynomial
is only a *trial field*. Its full residual has coordinate degree at most
`m+1`. The residual at any point in the box is a convex combination of its
tensor Bernstein coefficient vectors. With
`W_min=A(p_lower)+ω_lower C`, Loewner monotonicity gives the computable
box bound

```
    sup_cell ||W^-1/2 r_trial||_2
       <= max_Bernstein_coefficients ||W_min^-1/2 coefficient||_2.
```

No convergence assumption on the Taylor series is required: even if its
polynomial trial solution is poor, the residual inequality remains valid.
The degree-two trial uses 64 coefficient matrices for this three-dimensional
cell. `residual_cell_bound.py` implements this enclosure. **SciPy 1.17
guesses `assume_a='her'` incorrectly for our complex symmetric reduced
matrix; its solves must explicitly use `assume_a='gen'`.** The tests check
both the sparse and dense operators and independently solve complex cells.

There is a tighter, still explicit option when the **reduced** Taylor
expansion converges in the cell. Let `J=S_r(center)^-1 Delta S_r` and
`rho=sum_j ||S_r(center)^-1 Delta_j S_r||_2`. For `rho<1`,

```
   ||Q_actual-Q_m||_2 <= rho^(m+1) ||Q_center||_2 / (1-rho).
```

Let `L` bound `||W_min^-1/2 S(p,iω)V||_2` by the sum of the center and
half-width contributions, each computed from one small Gram matrix after
a sparse solve. The **actual reduced residual** is then bounded by the
trial Bernstein residual plus `L rho^(m+1)||Q_center||/(1-rho)`.
Its weighted field error uses the factor `sqrt(2)` rather than `2`.
For large cells where `rho>=1`, simply use the first, unconditional
Bernstein bound. This remainder is a consequence of the small reduced
resolvent identity, not a fit to validation errors.

### Two denominators and the infinite-frequency tail

For the impulse metric, replace `G` by `G E0^-1/2`, whose full impulse
energy matrix is the identity. Parseval and the cell bound give a
contribution of at most
`(ω_upper-ω_lower) B_cell^2/π` to the squared **worst-input** relative
error. If all generalized eigenvalues of `(A(p_max),C)` are below `Lambda`,
the improved factor `Lambda/(Lambda+ω_lower)` applies to each interval.

For the full-model Hankel denominator, the whitened thermal stiffness is
a symmetric M-matrix and its semigroup is nonnegative. Increasing either
HTC decreases the full controllability Gramian entrywise. Perron-Frobenius
therefore lower-bounds the full Hankel norm on a parameter cell by its
value at the cell's **upper** HTC corner. A cheaper, certified-in-exact-
arithmetic Rayleigh witness for that corner is, for any `a>0`,

```
    ||H_Z(p_upper)|| >=
    2a lambda_max[G^T (A(p_upper)+a C)^-1 C
                    (A(p_upper)+a C)^-1 G].
```

Finally, for frequencies at least `Omega`, choose the rational trial
`Q_trial=(iω)^-1 (V^T C V)^-1 V^T G`. Its residual is exactly
`r_inf+i d(p)/ω`, where `r_inf=G-CV(V^TCV)^-1 V^TG` and
`d(p)=A(p)V(V^TCV)^-1V^TG`. The spectral norms of `C^-1/2 r_inf` and
`C^-1/2 d(p)` are bounded by `a` and the maximum `b` over **parameter
corners** (the norm of an affine function is convex). The transfer tail is
at most `4(a+b/Omega)^2/Omega`. For energy, integrating the extra
`Lambda/ω` weight gives the explicit squared-error tail

```
   4 Lambda / pi * [a^2/Omega + a*b/Omega^2 + b^2/(3 Omega^3)],
```

using `a,b` calculated with the energy-normalized ports. This covers all
frequencies, including arbitrarily high ones. The remaining finite cells
cover the specified **continuous parameter box**, not just grid points.

## Three mesh sizes: actual all-input impulse energy

The following figures are actual infinite-time errors at the indicated
HTCs after the unchanged SVD, obtained by positive shifted full-order
solves. Values are ratios, not percent. The 1 mm solve uses AMG-preconditioned
CG at relative tolerance `1e-11`; the coarser meshes use sparse direct LU.

| Grid | Fixed solves/order | Random solves/order | HTC | Fixed worst input | Random worst input |
| --- | ---: | ---: | --- | ---: | ---: |
| 5 mm | 88 / 42 | 116 / 35 | (100,100) | 0.00733649 | 0.01115763 |
| 5 mm | 88 / 42 | 116 / 35 | (1,1) | 0.00787494 | 0.01318418 |
| 5 mm | 88 / 42 | 116 / 35 | (10000,1) | 0.00114329 | 0.00171997 |
| 2.5 mm | 92 / 47 | 126 / 42 | (100,100) | 0.01560148 | 0.01564479 |
| 2.5 mm | 92 / 47 | 126 / 42 | (1,1) | 0.01625148 | 0.01693903 |
| 2.5 mm | 92 / 47 | 126 / 42 | (10000,1) | 0.00529391 | 0.00546189 |
| 1 mm | 100 / 57 | 159 / 52 | (100,100) | 0.02138859 | 0.03110211 |

All displayed actual errors are under `0.06324555`, **at the displayed
HTCs only**. This empirical comparison does not prove a full HTC-box
guarantee. Earlier 1 mm step-response tests on twelve parameter points
also found `5.6538e-4` versus `4.9537e-3` for the same random seed; see
`ONE_MM_FREQUENCY_FACES.md` for reference-solver checks.

## Local continuous-box certificate at 5 mm

With degree-two reduced trial fields, 42 frequency cells from zero to
`1000 s^-1`, the explicit infinite tail, and the five exponential Hankel
witness rates `1e-5,...,1e-1 s^-1`, the **final** 5 mm fixed basis has:

| Physical HTC box around (100,100) | Fixed Hankel upper | Fixed energy upper | 1e-3 target passed? |
| --- | ---: | ---: | --- |
| Each effective HTC within ±1% | 0.00176891 | 0.05891111 | Both |
| Each effective HTC within ±5% | 0.00184367 | 0.05895552 | Both |
| Each effective HTC within ±10% | 0.00194021 | 0.05901500 | Both |

Using the degree-three reduced Taylor witness **and** its analytic reduced
remainder, the wider **±20%** box also passes: Hankel `0.00167982`, energy
`0.02645967`. The single ±50% box does not pass (`0.03499529` and
`0.02809764`); splitting it is necessary to remove its low-frequency
parameter enclosure overestimate.

On the *same* ±1% box and frequency partition the stock random basis
returns Hankel `0.00381695` and energy `0.06574380`. Those are **upper
bounds**, so failing them says the stock basis is *uncertified by this
partition*, not that its true errors exceed target. The 5 mm fixed basis
needed 88 full RHS solves versus the random baseline's 116.

On the 2.5 mm ±1% box, the 42-cell frequency partition gives `0.00333870`
and `0.07957745` for the fixed basis. This partition does not certify it;
an 82-cell partition with the same degree-two witness yields `0.00224561`
and `0.05242399`, passing energy but missing Hankel by about 12%.
Retrying only the ten failing frequency cells with a degree-three witness
and the reduced Neumann remainder gives **`0.00197991` Hankel and
`0.05242399` energy**, passing both targets on this continuous ±1% HTC
box. The certificate takes 246 seconds after a 2.75-second extraction:
its cost is a verification cost and must not be disguised as part of the
sampling-time advantage. Its slowest interval is `0.00891–0.01059 s^-1`.
The bounds are necessarily
conservative and differ from the smaller measured errors.

The 2.5 mm Hankel witness used only five round-number exponential rates.
At the same upper HTC corner, independently evaluating its Rayleigh formula
at `a=0.0021544346900 s^-1` raises the full Hankel lower bound from
`15.18655` to `16.41853`. Rescaling the **unchanged** maximal absolute
transfer enclosure therefore tightens its relative bound to `0.00183135`.
This extra one-dimensional denominator witness changes no HTC snapshot or
reduced model, and does not alter the energy bound.

## Limits of the result

1. The locally passing 5 mm boxes are **not** the complete allowed HTC
   rectangle. A full box guarantee requires tiling that rectangle and
   satisfying both inequalities for every tile. The current three-point
   sampling strategy has no demonstrated full-box guarantee on any mesh.
2. The mathematical enclosure is exact-arithmetic rigorous. Numerical
   positive definiteness, sparse factorizations, eigenvalues, and Bernstein
   coefficient norms are presently evaluated in ordinary floating point.
   The output explicitly sets `floating_point_certified=False`.
3. Certifying a wide three-dimensional parameter/frequency box by its
   polynomial coefficients is conservative. The demonstrated improvement
   comes from a **higher-order analytic witness**, not from changing the
   extractor, the random baseline, or the final SVD cutoff. A practical
   full-box scheme must keep certificate cost under control.

Reproduce the 5 mm local comparison without checking data into Git:

```
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python playground/adaptive_bci_sampling/probe_uniform_certificate.py \
    --mesh-mm 5 --half-width 0.01 --stock --output /tmp/bci-cert-5mm.json
```

For the 2.5 mm local certificate, construct the frequency grid separately
and retry only the failing Hankel intervals:

```
python -c 'import json,numpy as np; x=np.r_[0,np.geomspace(1e-7,1e-4,12),np.geomspace(1e-4,.1,41)[1:],np.geomspace(.1,10,21)[1:],np.geomspace(10,1000,11)[1:]];open("/tmp/bci-frequency-edges-82.json","w").write(json.dumps(x.tolist()))'
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python playground/adaptive_bci_sampling/probe_uniform_certificate.py \
    --mesh-mm 2.5 --frequency-edges-file /tmp/bci-frequency-edges-82.json \
    --retry-order 3 --output /tmp/bci-cert-2p5mm.json
```

The approach is motivated by the [Extended FANTASTIC
article](https://ieeexplore.ieee.org/document/9507439) and the
[Flotherm validation note](https://assets.ctfassets.net/dww76w587oxz/1ePeFrdaZXfoq0E1cFuM0T/e71cbcd824d7962c90319da1db28548a/Simcenter-Flotherm-BCI-ROM-Validation-2020.2.pdf).
Those sources state the familiar `2epsilon`/`2sqrt(epsilon)` targets but
do not prove that random HTC probes provide a relative continuous-parameter
box guarantee with the normalization used above.
