# A discrete-time output certificate for BCI ROMs: derivation and feasibility check

For the controlled `1e-3` parameter experiment with **unchanged closing
SVD**, see [TANGENT_CORNER_SAMPLING.md](TANGENT_CORNER_SAMPLING.md). The
nested-SVD results below deliberately change the closing cutoff and are
separate diagnostic experiments.

This note follows [OUTPUT_GREEDY_TRANSIENT_RESULTS.md](OUTPUT_GREEDY_TRANSIENT_RESULTS.md).
It is an investigated **research route**, not a claim that the present
extractor meets a certified transient tolerance.

## Reuse the four source ports as transient adjoints

For a fixed effective HTC vector `p`, let `K(p)` and `C` be symmetric positive
definite, `D=C/dt`, and `B(p)=K(p)+D`. The exact full-order response to a
unit step at source `j`, from zero initial temperature rise, is

```text
B x[j,n] = G[j] + D x[j,n-1],     x[j,0] = 0.
```

The usual Galerkin ROM gives states `xhat[j,n] = V a[j,n]`. Its *full-order*
time residual is

```text
r[j,n] = G[j] - B xhat[j,n] + D xhat[j,n-1].
```

Write `w[i,l] = x[i,l] - x[i,l-1]` for the exact **step increments** at
output port `i`. Then

```text
B w[i,1] = G[i],
B w[i,l] = D w[i,l-1]       for l > 1.
```

These increments are exactly the backward adjoint impulse kernels of the
co-located observation `G[i]^T x`: symmetric `B,D` make their action on the
primal residual identical to that of the adjoint propagator. Hence the
dual is already represented by the four original source ports; separate
full-order dual trajectories are unnecessary.

The reduced step increments `what[i,l]` lie in `range(V)`, and every
Galerkin residual `r[j,k]` is orthogonal to that range. Let `e_w=w-what`.
The **exact** output-error identity is

```text
Y[i,j,n] - Yhat[i,j,n]
  = sum_{k=1}^n e_w[i,n-k+1]^T r[j,k].
```

Define `r[i,0]=0`, `d[i,l]=r[i,l]-r[i,l-1]`. The impulse error solves
`D(e_w[l]-e_w[l-1])+K(p)e_w[l]=d[i,l]`; the usual discrete energy inequality
and Cauchy--Schwarz therefore imply, for **each discrete time n**,

```text
|Y[i,j,n] - Yhat[i,j,n]|
  <= sqrt(sum_{k=1}^n r[j,k]^T K(p)^-1 r[j,k])
     sqrt(sum_{l=1}^n d[i,l]^T K(p)^-1 d[i,l]).
```

Both factors can be upper-bounded using one fixed `K_min^-1` because
`K(p) >= K_min=K(p_min)`. As `K(p)` is affine and the ROM time states are
small, every residual is a linear combination of columns in the **fixed
span** `[G, KV, (C/dt)V, H_1V, H_2V]`. Precomputing its `K_min^-1` Gram
matrix permits evaluation for every parameter, channel, and step using
only reduced matrices. The estimate applies **after** SVD compression,
which is precisely where the steady-only selection in the previous
experiment loses its stopping claim.

If the required denominator is the same-parameter exact steady transfer,
obtain a *lower* bound for its magnitude from the existing corrected
steady primal-dual certificate:

```text
|Y_steady[i,j](p)| >= |Yhat_steady[i,j](p)| - Delta_steady[i,j](p).
```

Only if the right side is positive does division give a certified relative
step error. Otherwise the relative certificate is infinite. This is
stronger and more honest than dividing by a reduced steady value.

This derivation concerns BDF1 with `dt=50 s` and the enumerated time steps;
extension to arbitrary continuous times, power histories, or the entire
HTC box needs an additional argument. Numerical `K_min^-1` applications
and Gram contractions also have solver/roundoff error and are not
outward-rounded validated computations.

## Numerical feasibility check at 2.5 mm

We computed the bound from a single `K_min^-1` residual Gram for two
existing `1e-3` bases: stock seed 20260805, order 42; steady-output greedy,
order 47. The six physical HTC points were the four corners, `(50,1000)`,
and `(100,100)`. All 40 BDF1 times and 16 transfer entries were checked at
each point. Actual errors use the exact *same-parameter* steady transfer
entry. These are diagnostic computations, **not** a box-wide certificate.

| Basis | Largest observed relative step error | Largest computed relative bound | Sampled violations |
| --- | ---: | ---: | ---: |
| Stock, order 42 | `1.757e-3` | `7.079e-2` | 0 |
| Steady-output greedy, order 47 | `9.719e-4` | `1.932e-2` | 0 |

The maximum bound and maximum true error need not occur at the same
entry/time. The bound is valid algebraically but too loose to retire a
candidate at `1e-3`. It would compel substantial extra enrichment even
for the ROM whose measured step response already passes the requested
tolerance.

We replaced `K_min^-1` with the *exact local* `K(p)^-1` at two stock-basis
corners to isolate the global-norm effect (this uses a new full-order
factorization at each point and is **not** a cheap online algorithm):

| Physical HTC | True maximum | Global bound | Local bound |
| --- | ---: | ---: | ---: |
| `(10000,1)` | `1.757e-3` | `2.359e-2` | `2.121e-2` |
| `(10000,10000)` | `1.485e-3` | `7.079e-2` | `3.106e-2` |

Thus tightening the Riesz operator helps at the fully cooled corner but
still leaves roughly an order of magnitude of slack at both corners.
Replacing `K_min` with many parameter anchors alone will not rescue this
particular stopping test. A separate generalized-eigenvalue check at
`dt=50 s` found `lambda_max(D,K_min+D) ≈ 0.997862`, giving a 40-step
geometric sum of about 38.4. A simple worst-case residual recursion with
that global contraction factor would be similarly pessimistic.

## Separate witness space and three deterministic HTC points

Let `W` be an SVD basis from **the same** dynamic snapshots as `V`, retained
at a tighter cutoff. No extra full-order solve is needed. Its step increments
`what_W` generally have a nonzero pairing with the `V` residual. Thus

```text
Y - Yhat_V = correction_W + remainder,
correction_W[n] = sum_k what_W[n-k+1]^T r_V[k],
|Y - Yhat_V| <= |correction_W|
  + sqrt(sum_k ||r_V[k]||_(K_min^-1)^2)
    sqrt(sum_l ||r_W[l]-r_W[l-1]||_(K_min^-1)^2).
```

The bound is for the **uncorrected delivered ROM**, with the correction
included as an error term. The same-parameter steady denominator is bounded
below by the compressed ROM's steady output minus its output error bound.
An unsafe or zero lower bound produces infinite relative error. The
implementation checks all 40 BDF1 steps and all 16 four-port transfers.

For parameter points, take the product of the one-point Zolotarev rule as a
seed, then the *opposite two corners* of the effective two-dimensional HTC
rectangle, `(p1_min,p2_max)` and `(p1_max,p2_min)`. This three-point rule is
a tested heuristic, not a proven optimal rational tensor sampling formula.
Its selected points in the 2.5 mm test are approximately `(346.03,14.53)`,
`(0.9999923,234.375)`, `(9285.71,0.995851)` in effective parameter units.

The following rows compare **the same frequency-shift tolerance** and the
**same 12-parameter holdout**, with 40 BDF1 steps and the worst of 16 junction
transfers, normalized by each transfer's exact same-parameter steady rise.
The three-point arm deliberately uses a ten-times-tighter **closing SVD**
cutoff than stock; its larger ROM order and that SVD time are included.
Stock timings and results come from
[OUTPUT_GREEDY_TRANSIENT_RESULTS.md](OUTPUT_GREEDY_TRANSIENT_RESULTS.md).
The new extraction timing includes Zolotarev spectral enclosures and
retaining all three SVD cuts; it excludes full-order holdout validation and
the optional offline certificate preparation.

| Extraction cutoff | Method | Dynamic + selection RHS | Delivered ROM order | Extraction seconds | Worst held-out step error | 9x9 maximum computed relative bound | Witness order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | Stock, seed 20260805 | 126 | 42 | 9.22 | `1.757e-3` | — | — |
| `1e-3` | Stock, seed 7 | 127 | 42 | 8.63 | `1.924e-3` | — | — |
| `1e-3` | Three fixed points, SVD cutoff `1e-4` | 144 | 66 | 10.86 | `1.817e-4` | `8.905e-4` | 82 |
| `1e-4` | Stock, seed 20260805 | 232 | 62 | 16.51 | `1.805e-4` | — | — |
| `1e-4` | Stock, seed 7 | 227 | 62 | 14.99 | `2.892e-4` | — | — |
| `1e-4` | Three fixed points, SVD cutoff `1e-5` | 180 | 90 | 13.19 | `4.677e-6` | `8.041e-5` | 112 |

At `1e-4`, the fixed three-point extraction is faster and takes fewer RHS
than either stock seed, while its measured worst step error is much smaller.
Running only the delivered 90-dimensional basis, without extra witness SVDs,
measured `13.16 s` (including spectral enclosures) and the same 180 RHS.
At `1e-3`, it produces a markedly more accurate, higher-order ROM but takes
more time and RHS than stock; the stock ROMs themselves do not reach `1e-3`
on this holdout. The certificate's offline setup for **five** nested space
pairs costs `7.20 s` and `9.74 s` respectively; scoring 81 grid points for
all five pairs costs another `9.49 s` and `15.20 s`. These are extra costs
if certification is required. Building just the chosen pair would cost
less, but this has not been separately timed. All tested holdout
`(parameter,time,channel)` errors stayed below the computed absolute bounds.
Refining the `1e-3` candidate grid from 9x9 to 17x17 raised the chosen
66/82 bound only from `8.905e-4` to `8.936e-4`. Relaxing that delivered
SVD cutoff from `1e-4` to approximately `3.33e-4` gave order 54, but its
held-out step error increased to `1.094e-3`, so this smaller ROM was rejected.
For `1e-4`, the 90/112 maximum bound likewise rose from `8.041e-5` to
`8.473e-5` on the 17x17 grid and remained below the target.

A 2x2 fixed Zolotarev tensor at `1e-3` needs 192 RHS, and its 58-dimensional
ROM attains a measured `8.169e-4` error; its best tested 73-dimensional
witness gives a `4.604e-3` grid bound, failing certification. Thus the two
cross-corners are consequential even though the three-point rule uses fewer
snapshots. Five steady-output greedy HTC points need 260 RHS at `1e-3`:
their 70-dimensional ROM / 90-dimensional witness yields a `3.122e-4`
grid bound, but extraction takes `22.16 s` before certification.

This finite-grid bound does **not** prove tolerance over the continuous HTC
box. It is an algebraic bound for every enumerated parameter and BDF1 time,
subject to floating-point factorization and Gram roundoff, and separately
validated against full-order solves at the 12 holdout parameters. The paper
reproduced as Extended FANTASTIC picks random HTC vectors in its Algorithm 1;
its printed claim of accuracy over its chosen continuous parameter set does
not itself turn that finite random stopping check into a worst-case proof.

## Next practical improvements

Tune the closing SVD cutoff *on the existing three-point snapshots* to find
the smallest delivered order meeting a measured output tolerance. Try a
larger, fixed holdout set and another mesh before treating the three points
as robust. If selection cost matters, use the dual certificate only on a
shortlist of high-error candidates and spend extra full-order solves at the
worst parameter/channel/shift, rather than taking a Cartesian product of
all selected HTC values and shifts. A continuous-box claim would additionally
require certified maximization of the small-matrix bound, for example
interval branch-and-bound; it is separate from the present performance win.

The general space-time primal-dual output-bound and greedy framework is
established by [Grepl and Patera (2005)](https://www.numdam.org/item/M2AN_2005__39_1_157_0.pdf).
The reuse of step increments for co-located heat-source/temperature outputs
and the six-point effectivity check above are specializations examined here.
