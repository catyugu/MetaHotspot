# Rational parameter sampling with output certificates

This experiment follows `RESULTS.md` and uses the full four-source response
block at every parameter.  It therefore measures both the nominal-power
junction vector and every entry of the 4 by 4 source-to-junction transfer
matrix.  No closing SVD is used: every reported basis is the untruncated,
incrementally orthonormalized snapshot span plus the constant mode.

## Result in one sentence

Finite-interval Zolotarev theory gives a closed-form, seed-free placement rule
and an a priori scalar resolvent count, but its direct tensor count is too
conservative.  The best tested algorithm is a Zolotarev-seeded, output-certified
weak greedy: it needs 5 parameter points (20 source RHS, order 21) at both
2.5 mm and 1 mm.  Its stopping statement is currently rigorous only on the
deterministic candidate set, not over the continuous parameter box.

## What is proved and what is not

For one coordinate,

```text
A(h) = A_0 + h H,       spectrum(A_0, H) subset [alpha, beta],
h in [h_min, h_max].
```

The scalar kernel is `1 / (lambda + h)`.  The two-real-interval Zolotarev
construction separates `[alpha,beta]` from `[-h_max,-h_min]` and supplies
explicit sample points.  With

```text
gamma = (alpha + h_max)(beta + h_min)
        / ((alpha + h_min)(beta + h_max)),
Z_n <= 4 exp(-pi^2 n / log(16 gamma)),
```

`zolotarev.py` returns the elliptic-function nodes and the smallest `n` meeting
a requested scalar tolerance.  A dense scalar skeleton test verifies the bound
directly.

For Case 1 at 2.5 mm, the conservative coordinate spectral intervals are

| group | spectral interval | gamma | `n` for `Z_n <= 1e-3` |
| --- | ---: | ---: | ---: |
| die crowns | `[13.1980, 371605]` | 638.98 | 8 |
| remaining ambient | `[0.0660725, 1190.56]` | 184.61 | 7 |

The upper endpoints use the active principal block as a Schur-complement upper
bound.  This is slightly wider than the explicitly formed envelopes from the
local handoff (`290582` and `1179.48`), but reduces the 1 mm envelope time from
919.6 s to about 91--99 s by replacing thousands of blocked back-solves with
Krylov extreme-eigenvalue iterations.

The one-coordinate Zolotarev result does **not**, by itself, prove that the
smallest tensor product meeting a split scalar tolerance also meets the desired
relative error of every cross transfer entry.  The experiment intentionally
does not make that claim.  A continuous-box output guarantee still requires a
verified maximization of the primal-dual certificate.

## Equal-budget placement with block source solves

Command:

```text
python playground/adaptive_bci_sampling/compare_zolotarev.py 2.5 \
  --junction-tolerances \
  --count-pairs 2x2 3x3 4x4 5x5 6x6 \
  --padua-degrees 1 3 4 6 7 \
  --tensor-log-counts 2 3 4 5 6 \
  --greedy-tolerances 1e-3 1e-4 \
  --greedy-metric entrywise --greedy-grid 41 \
  --validation-grid 21 --random-holdout 128
```

The holdout is a 21 by 21 log grid plus 128 fixed random points (569 total).
The table reports the worst relative error of the complete 4 by 4 transfer
matrix.  `RHS = 4 * points`.

| rule | points | RHS | order | worst entrywise error |
| --- | ---: | ---: | ---: | ---: |
| Zolotarev 2x2 | 4 | 16 | 17 | `3.107e-2` |
| CL-log 2x2 | 4 | 16 | 17 | `3.070e-2` |
| Zolotarev 3x3 | 9 | 36 | 37 | `8.797e-5` |
| CL-log 3x3 | 9 | 36 | 37 | `1.123e-4` |
| Zolotarev 4x4 | 16 | 64 | 65 | `1.543e-7` |
| CL-log 4x4 | 16 | 64 | 65 | `1.261e-7` |
| Zolotarev 5x5 | 25 | 100 | 96 | `4.149e-10` |
| CL-log 5x5 | 25 | 100 | 93 | `5.779e-10` |
| Zolotarev 6x6 | 36 | 144 | 114 | `2.714e-12` |
| CL-log 6x6 | 36 | 144 | 117 | `2.330e-12` |

Zolotarev and Chebyshev--Lobatto in log space are statistically tied at equal
square budgets; neither uniformly dominates.  Both converge rapidly and reach
the full-solve numerical floor by 36 points.  This reproduces the local
placement ranking after fixing its combined-power-snapshot limitation.

Padua remains very competitive at its triangular budgets: 3 points give
`1.106e-3`, 10 points give `2.555e-9`, and 15 points reach `9.93e-12`.
Therefore the data do not support a universal claim that tensor Zolotarev is
the smallest deterministic set.  They support using rational information as a
seed for adaptive certification instead of paying the tensor product count.

## Zolotarev-seeded certified weak greedy

For a fixed raw basis `V`, let `r_i(h)` be the Galerkin residual for source
`i`.  Since `A(h) >= A_min = A(h_min)`, every transfer entry satisfies

```text
|Y_ij(h) - Yhat_ij(h)|
  <= sqrt(r_i^T A_min^-1 r_i) sqrt(r_j^T A_min^-1 r_j).
```

`certified_greedy.py` precomputes the Riesz Gram matrix for all affine residual
components.  Online evaluation then uses only a small dense solve.  Relative
bounds use `Delta / (|Yhat| - Delta)` when the denominator is positive.  While
it is not positive, the greedy selects the largest absolute bound; this avoids
an ordering-dependent tie between infinite relative scores.

At 2.5 mm, a 41 by 41 deterministic log grid gives:

| quantity | result |
| --- | ---: |
| selected parameter points | 5 |
| full source RHS | 20 |
| raw ROM order | 21 |
| worst candidate-grid relative certificate | `3.882e-6` |
| worst holdout nominal-power error | `3.855e-7` |
| worst holdout entrywise error | `3.173e-6` |
| worst holdout exact energy-product bound | `3.457e-6` |

Thus the same 5-point space passes both `1e-3` and `1e-4`; the certificate
drops across both thresholds in one enrichment step.  It is much smaller than
the direct `8 * 7 = 56` Zolotarev tensor budget.

At 1 mm (122400 cells), the corrected entrywise run also selects 5 parameters
(20 RHS, order 21) and reaches a `3.523e-5` relative certificate on the same
41 by 41 candidate grid.  The two spectral intervals take 90.6 s and the
five-step greedy takes 236.1 s on this machine.  This run used
`--skip-validation`; its statement is therefore only the candidate-grid
certificate.  The local handoff's independent 185-point experiment remains
the available large-mesh error check.

The important scaling result is that increasing the field size from 9072 to
122400 cells does not increase the selected parameter count.  Offline wall
time is still above the approximately 60 s stock extraction at 1 mm, but the
spectral stage is about ten times faster than the previously formed blocked
envelope and the number of full source solves falls from roughly 150 to 20.

## Recommendation

Do not replace the production sampler with the raw tensor Zolotarev count.  Use
this staged design instead:

1. compute conservative coordinate spectral intervals with the Krylov method;
2. use the degree-one coordinate Zolotarev product as the deterministic seed;
3. enrich with the maximum full-transfer primal-dual certificate, always using
   all source RHS;
4. preserve the raw span, or certify any subsequent SVD compression;
5. label the current stop as a finite-candidate guarantee;
6. before calling it a continuous-box guarantee, maximize the reduced
   certificate with outward-rounded interval branch-and-bound in
   `(log h_1, log h_2)`.

The last item is the remaining mathematical gap.  It is now a small dense
two-variable verification problem; it no longer requires additional full-order
solves.  A useful implementation should interval-enclose the reduced SPD solve
on each box, prune boxes whose certificate upper bound is below tolerance, and
subdivide the box with the largest unresolved upper bound.

## Theory references

- S. Massei and L. Robol,
  [Rational Krylov for Stieltjes Matrix Functions: Convergence and Pole Selection](https://arxiv.org/abs/1908.02032).
  The resolvent skeleton identity, rational-Krylov exactness, Möbius reduction,
  Zolotarev poles, and explicit convergence rates used here are in Sections
  3.1--3.4.  The paper also warns that a priori degree bounds commonly
  overestimate the useful degree, motivating adaptive pole selection.
- S. Zhang,
  [Primal-Dual Reduced Basis Methods for Convex Minimization Variational Problems](https://arxiv.org/abs/1810.04073).
  This supplies the broader certified reduced-basis and adaptive-greedy
  framework for symmetric coercive problems.  The implementation here uses
  the simpler linear residual-product specialization available because the
  thermal operator is SPD and the outputs are source functionals.

## Tests

```text
python playground/adaptive_bci_sampling/test_zolotarev.py
```

The focused tests cover node containment, the scalar Zolotarev error bound,
minimal count selection, spectral containment on an exactly reduced problem,
power-weighted and entrywise primal-dual certificates, and deterministic
tie-breaking while relative certificates are infinite.
