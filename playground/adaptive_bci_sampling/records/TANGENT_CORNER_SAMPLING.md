# Output tangent geometry for three HTC sampling points at `1e-3`

This experiment holds the production Extended FANTASTIC frequency plan, four
independent source ports, and **closing SVD cutoff `1e-3` fixed**. It replaces
only random HTC selection. The objective is the worst of 16 junction step
transfers, divided by each transfer's exact same-parameter steady rise.
The 2.5 mm Case 1 has 9072 cells. Each holdout contains four physical HTC
corners, `(50,1000)`, `(100,100)`, and fixed-seed log-uniform points. BDF1
uses 40 steps of 50 s. Full-order validation solves are excluded from
extraction timing; sampling preparation is reported separately.

## An exact matrix identity and parameter tangency

Let `A(p)=K+sum_k p_k H_k`, `X(p)=A(p)^-1 G`, and
`Y(p)=G^T X(p)`. For a *co-located* Galerkin model with trial space `V`,
`X_V=V(V^T A V)^-1 V^T G` and `R=G-A X_V`. Galerkin orthogonality gives

```text
Y(p)-Y_V(p) = (X-X_V)^T A(p) (X-X_V)
            = R(p)^T A(p)^-1 R(p)  >= 0  (Loewner order).
```

This also holds for the positive real frequency operator `A(p)+sC`.
If the **uncompressed** trial space contains every source solution at a seed
`p0`, then `R(p0)=0`: all transfer values and all first HTC derivatives match
there. In particular, `Y-Y_V=O(||p-p0||^2)` locally. The off-diagonal
entries of the error matrix may have either sign even though the matrix is
positive semidefinite.

This is a **rational Hermite** fact, not an assumption that junction values
determine the field span. Along any ray `p(t)=p0+t d`, write
`D=sum_k d_k H_k` and diagonalize
`A(p0)^(-1/2) D A(p0)^(-1/2)=U diag(mu_l) U^T`. Then, with
`z_i=U^T A(p0)^(-1/2) g_i`,

```text
Y_ij(t) = sum_l z_i,l z_j,l / (1 + t mu_l).
```

Diagonal transfer weights are nonnegative; cross-transfer weights can have
either sign. All poles lie outside a ray on which `A(p(t))` stays SPD.
Snapshot inclusion makes the Galerkin rational transfer interpolate the
value and first derivative at `t=0`. The question is **which rays** deserve
further sampling in the two-dimensional HTC box; the quadratic error
geometry below supplies a computable way to rank them.

## A four-vertex bound for all steady transfer entries

Write `x0,i=A(p0)^-1 g_i`, `d=p-p0`, and
`A_min=A(p_min)`. Since the Ritz solution minimizes energy error in its
space, trying `x0,i` at the new parameter and using
`A(p)^-1 <= A_min^-1` gives a computable *raw-space* bound:

```text
0 <= E_ii(p) <= q_i(d) = d^T Q_i d,
(Q_i)_ab = (H_a x0,i)^T A_min^-1 (H_b x0,i).
```

Each `Q_i` is positive semidefinite. The Case 1 operators have nonpositive
off-diagonal entries, all `H_k` are nonnegative diagonal, and `G` is
nonnegative. Hence `A(p)` is an SPD M-matrix, its inverse is nonnegative,
and

```text
dY_ij/dp_k = -x_i(p)^T H_k x_j(p) <= 0.
```

The **exact** four-by-four steady transfer at the upper HTC corner,
`Y_max=Y(p_max)`, is therefore a positive entrywise lower bound for every
`Y(p)` in the box. Positive semidefiniteness and the arithmetic-geometric
inequality give, for every source/output pair,

```text
|Y_ij(p)-Y_V,ij(p)| / Y_ij(p)
  <= sqrt(q_i(d) q_j(d)) / Y_max,ij
  <= d^T [(Q_i+Q_j)/(2 Y_max,ij)] d.
```

This is a **box-wide bound for the raw steady seed snapshot space**, whose
normalization uses each transfer's own steady value. The maximum over the
16 convex quadratic upper bounds is itself convex, so its maximum over a
rectangle occurs at one of its **four vertices**. This reduces the search
for the most exposed seed directions to four closed-form evaluations; it
does not require a candidate grid or random parameter probes. Computing
the matrices needs four steady solves at `p0`, four at `p_max`, and eight
applications of one `A_min` factor (2 groups times 4 source ports).

Use the degree-one Zolotarev product as `p0` and rank the four corners by
that maximum transfer majorant. For the 2.5 mm case the descending order is
`(p1_max,p2_max)`, `(p1_max,p2_min)`, `(p1_min,p2_max)`,
`(p1_min,p2_min)`. Selecting the first two protects both ends of the
strongly exposed `p1_max` edge with a total of **three parameter points**.
Their effective values are approximately `(346.032738, 14.526334)`,
`(9285.714286, 234.375000)`, and `(9285.714286, 0.995851)`.
The choice to stop at two extra corners is a low-budget design decision,
not a proved minimal count or a transient-error theorem.
In this four-die geometry, extending the ranking from the four diagonal
transfers to all 16 entries leaves the corner order unchanged; the
off-diagonal formula nevertheless states the correct output quantity and
allows a future asymmetric testcase to expose different rankings.
The seed's finite-interval rational placement is described in
[Massei and Robol, *Rational Krylov for Stieltjes Matrix Functions:
Convergence and Pole Selection*](https://arxiv.org/pdf/1908.02032).

## Numerical comparison with fixed SVD cutoff

Reproduce the sampling and holdout with:

```text
.venv/bin/python playground/adaptive_bci_sampling/probe_tangent_corners.py 2.5 \
  --tolerance 1e-3 --maximum-points 3 --random-holdout 64 \
  --output playground/adaptive_bci_sampling/tangent_corners_entrywise_2p5_tol1e3_holdout70.json
```

For a direct three-point build without the two-point diagnostic, add
`--minimum-points 3 --maximum-points 3`. That direct run measured 3.09 s
for selection and 7.95 s for dynamic extraction, totaling **11.04 s**;
it reproduced `6.189e-4` on the 12-point holdout.

Both the frequency shift count and the normalized snapshot SVD threshold
are *identical* to the stock `1e-3` extraction. The four-port frequency
snapshots for three parameters use `3 × 48 = 144` full-order RHS. The
corner-selection stage costs eight steady source RHS plus eight Riesz
back-solves, in addition to the 144 dynamic RHS; its wall time is separate.

| Method | Dynamic RHS | ROM order | Dynamic extraction | Selection preparation | 12-point worst step transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock, seed 20260805 | 126 | 42 | 9.22 s | included | `1.757e-3` |
| Stock, seed 7 | 127 | 42 | 8.63 s | included | `1.924e-3` |
| Seed + highest-ranked corner | 96 | 45 | 5.74 s | 3.32 s | `2.704e-3` |
| **Seed + top two tangent corners** | **144** | 48 | **7.83 s** | 3.32 s | **`6.189e-4`** |

The 70-parameter holdout uses the same initial 12 points plus 58 additional
independent log-uniform points. It confirms the same worst error for each arm:

| Method | Dynamic RHS | ROM order | Dynamic extraction | Selection preparation | 70-point worst step transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock, seed 20260805 | 126 | 42 | 9.17 s | included | `1.757e-3` |
| Stock, seed 7 | 127 | 42 | 9.27 s | included | `1.924e-3` |
| Seed + highest-ranked corner | 96 | 45 | 6.02 s | 2.87 s | `2.704e-3` |
| **Seed + top two tangent corners** | **144** | 48 | **7.26 s** | **2.87 s** | **`6.189e-4`** |

The end-to-end three-point cost is approximately 10--11 s, **slower than
stock when selection is included**. Its accuracy is better at the same SVD
cutoff and its dynamic extraction alone is faster on these runs. Adding a
third corner raises the measured 12-point error from `6.189e-4` to
`7.302e-4`: the finally compressed SVD spaces are not generally nested,
and individual transient errors are not guaranteed to improve as snapshots
are added. In this run both final spaces have order 48, and the Frobenius
norm of projecting the three-point basis outside the four-point space is
`0.124`; they are demonstrably different subspaces. Thus the three-point
result is the meaningful low-budget arm.
Against the earlier **five-point steady-output greedy** at this same cutoff,
which uses 260 RHS, a 47-dimensional ROM and 18.95 s total extraction for
`9.719e-4` on the 12-point set, this three-point rule reduces solves and
end-to-end time while achieving a smaller measured transfer error (at the
cost of one additional retained mode).

As a control, [probe_tangent_greedy.py](probe_tangent_greedy.py) chooses the
next HTC by maximizing the previously derived **BDF1 output residual bound**
over a 9x9 candidate grid, always rebuilding the ROM at `1e-3` before
scoring. It needs four HTC points and 192 final dynamic RHS to reach
`7.302e-4` on the 12-point holdout; because the diagnostic rebuilds prior
snapshots at each stage, actual selection plus extraction takes 44.32 s.
Its computed error bound is `1.738e-2` even at the fourth stage and is too
loose to stop at `1e-3`. The exact vertex reduction of the seed's parameter
error geometry is therefore a substantive mathematical simplification:
it replaces repeated transient scoring with four quadratic evaluations and
selects a smaller final ROM family. The two bounds answer different
questions, and the vertex bound is **not** a transient stopping certificate.

## Why the proof stops before the delivered ROM

The quadratic bounds above are **valid for the raw, uncompressed steady
snapshot span containing `X(p0)`**. The dynamic extractor samples positive
frequency shifts rather than `s=0`, then applies a global SVD at `1e-3`.
Its delivered space need not contain `X(p0)`, so neither the exact
interpolation nor the quadratic bound carries over automatically.
Likewise a frequency-domain bound at positive `s` is not directly a
40-step BDF1 output bound. The holdout results are measurements of the
actual delivered ROM, not a corollary of the vertex theorem.

The present global `A_min^-1` matrix is very pessimistic: its scaled
two-dimensional quadratic has eigenvalues about `8.55e4` and `2.09e2`
for one port, exposing the first HTC direction. The *exact local Ritz*
Hessian at the seed has eigenvalues about `7.73e1` and `1.86e-3` and mainly
exposes the second direction. Thus local curvature cannot be extrapolated
over this huge parameter interval; the loose global majorant is useful
for selecting directions, but its numerical value must not be read as the
actual ROM error. A sharper mathematically motivated next step is to
replace the single `A_min` inverse by parameter-dependent rational
resolvent bounds while preserving the four-vertex or low-dimensional
extremization argument.

More precisely, the local second-order coefficient for source `i` is

```text
Qlocal_i,ab = (H_a x0,i)^T
  [A(p0)^-1 - V (V^T A(p0) V)^-1 V^T] (H_b x0,i).
```

The bracket is positive semidefinite and removes derivatives already
represented by `V`. The global `Q_i` above keeps a valid box bound but
does not perform this projection. The reversal between the two dominant
directions is a concrete reason to investigate parameter-dependent
resolvents, rather than replacing the global bound with an unjustified
local Taylor extrapolation.

The use of coercive output residual bounds and offline/online reduced basis
evaluation is established in [Sen et al., *Natural norm a posteriori error
estimators for reduced basis approximations* (2006)](https://www.mit.edu/~cuongng/publication/pub7/pub7.pdf).
The exact collocated error identity, M-matrix transfer monotonicity, and
convex vertex rule are specialized here to the BCI sampling question; their
combination should be judged by the controlled comparison above, rather
than treated as a new general rational approximation theorem.

## Conditional Zolotarev edge experiment

Massei and Robol's Theorem 3.4 bounds rational Krylov approximation of the
family `(A+tI)^-1 v` for a **fixed** positive definite `A`; Theorem 3.6
extends their pole analysis to exponentials. On the exposed edge
`p1=p1_max`, eliminating cells outside the support of `H2` gives a Schur
complement `S` and a positive diagonal `D=H2|support`. The remaining
parameter dependence is `(S+p2 D)^-1`: after symmetric scaling it has the
scalar resolvent kernels `1/(lambda+p2)`, where `lambda` belongs to the
finite generalized spectrum `(S,D)`. The resolvent skeleton identity makes
the error proportional to `r(lambda)/r(-p2)`, so the conditional
finite-interval Zolotarev rule chooses two interior `p2` values to minimize
the worst ratio on those two spectral and HTC intervals. This is a valid
one-dimensional **raw resolvent** placement principle. It provides no
uniform bound on the other HTC directions or on the compressed transient
transfer ROM.

To test whether the principle improves the actual objective, hold the
three-parameter budget, the original frequency shifts, closing SVD cutoff
`1e-3`, and the 70-parameter holdout fixed:

```text
.venv/bin/python playground/adaptive_bci_sampling/probe_conditional_zolotarev.py 2.5 \
  --random-holdout 64 \
  --output playground/adaptive_bci_sampling/conditional_zolotarev_2p5_tol1e3_holdout70.json
```

The conditional spectrum estimate on the exposed edge is `[6.183,1190.563]`;
the two interior effective HTC nodes are `5.9165` and `126.2610` at
`p1=9285.7143`. The degree-two rational bound returned by the rule is
`0.158`: it is far above our `1e-3` target even for the raw one-dimensional
problem. The alternate points `(seed, 9285.7143/5.9165,
9285.7143/126.2610)` are compared directly to the previously selected
endpoint corners in one run:

| Three-point placement | Dynamic RHS | Final ROM order | Dynamic extraction | Shared selection preparation | Worst step transfer | Worst steady transfer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conditional edge Zolotarev | 144 | 46 | 9.14 s | 4.81 s | `1.855e-3` | `1.309e-2` |
| Tangent-selected edge endpoints | 144 | 48 | 7.99 s | 4.81 s | `6.189e-4` | `3.080e-3` |

Both worst errors occur at the exposed endpoint `(p1_max,p2_min)`.
The shared preparation computes both full-box and conditional spectra,
and is therefore **not** the cheaper `2.87 s` preparation measured for
the standalone endpoint rule above. Run-to-run extraction times can vary.
The experiment gives a useful limit to the paper's application here:
minimizing a raw one-dimensional field-resolvent ratio can move the
snapshots away from the parameter corner that dominates our normalized
output and final post-SVD transient error. **Keep the seed plus two
tangent-selected corners** for the current `1e-3` comparison. Neither
the 70-point result nor the conditional bound proves a box-wide transient
tolerance certificate.
