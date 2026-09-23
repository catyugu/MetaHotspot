# Deterministic affine-parameter sampling: interim result

This note continues `NEXT_STEPS.md`.  It separates two questions which the
original random extraction conflates:

1. can the parameter samples be selected a priori and deterministically?
2. does the *compressed* Galerkin space retain the guarantee of those samples?

The answer to the first question is mathematically yes, but the direct rigorous
bound is too conservative to be a practical extractor.  The answer to the
second question is no for the current unconstrained closing SVD.

## Closed-form sample sets

For each effective HTC range `[a_i,b_i]`, introduce a log coordinate

```text
h_i(x_i) = exp(c_i + d_i x_i),
c_i = (log(a_i) + log(b_i))/2,
d_i = (log(b_i) - log(a_i))/2,
x_i in [-1,1].
```

Three deterministic point sets are implemented in `compare_sampling.py`:

- tensor Chebyshev--Lobatto points;
- two-dimensional Padua points, with `(n+1)(n+2)/2` points for total degree
  `n`;
- nested Clenshaw--Curtis Smolyak points.

All are explicit and seed-free.  Padua is the smallest of the tested sets for
two parameters; Smolyak is nested and therefore better suited to incremental
enrichment and to more than two parameters.

## What can actually be guaranteed

For one source and frequency shift, let

```text
A(x) u(x) = g,
A(x) = K + shift*C + sum_i h_i(x_i) H_i.
```

`u(x)` is holomorphic in every complex polyellipse on which the Hermitian part
of `A(x)` remains positive definite.  Tensor Chebyshev interpolation therefore
has an explicit exponentially decaying uniform error bound.  Since the
interpolant is a linear combination of the sampled full responses, it belongs
to their snapshot span.  Galerkin projection onto the *untruncated* snapshot
span is energy-best, so it inherits that bound.  Junction-transfer error is
then bounded by the product of the primal and reciprocal-source energy errors.

This is a real a priori guarantee, not a Monte-Carlo statement.  It has two
important limitations:

- the tensor count grows exponentially in the number of HTC groups;
- the coercivity bound is very pessimistic across a four-decade HTC range.

For the 2.5 mm Case 1 model, the simple fully closed-form coercivity bound needs
tensor degree 63, hence 4096 parameter points, to certify a 0.1% relative
junction error.  Replacing the coefficient-wise coercivity lower bound by
computed generalized eigenvalue bounds reduces the estimate to roughly degree
34, or 1225 points.  Both are orders of magnitude above what is needed in the
actual model, so this is not a competitive production rule.

The guarantee applies to the raw snapshot span.  A subsequent SVD truncation
must either be included in the certificate or followed by revalidation; the
present singular-value cutoff alone does neither.

## Reproducible experiments

Commands:

```text
python playground/adaptive_bci_sampling/compare_sampling.py \
  2.5 1 2 3 --validation-grid 21 --random-holdout 128 \
  --equal-cost-random-seeds 10

python playground/adaptive_bci_sampling/compare_extractors.py \
  2.5 --degrees 1 2 --random-seeds 5 \
  --validation-grid 21 --random-holdout 128
```

The holdout contains a 21 by 21 log-uniform grid plus 128 fixed-seed random
points, 569 points in total.  The metric is the maximum, over the four dies and
all holdout parameters, of the absolute junction error divided by that same
junction's full-order temperature rise.

### One steady shift only

| design | parameter points | source RHS | ROM order | worst junction error |
| --- | ---: | ---: | ---: | ---: |
| log-Padua degree 1 | 3 | 12 | 13 | **0.00553%** |
| random log-uniform, 10 seeds | 3 | 12 | 8--13 | 0.00472% / 0.09357% / 4.39765% (min/median/max) |
| log-Padua degree 2 | 6 | 24 | 14 | 0.01033% |
| random log-uniform, 10 seeds | 6 | 24 | 12--14 | 0.00157% / 0.00441% / 0.00713% |
| tensor corners | 4 | 16 | 9 | 0.17265% |

The three-point Padua rule removes the catastrophic random-seed tail.  Merely
sampling all four corners is not enough: those responses produce a lower-rank
space and miss the important mixed direction.

### Full 12-shift Extended FANTASTIC extraction

All cases below use the production normalized-snapshot SVD cutoff `1e-3` and
preserve the constant mode.

| design | full response snapshots | residual probes | ROM order | worst junction error |
| --- | ---: | ---: | ---: | ---: |
| log-Padua degree 1 | 144 | 0 | 47 | **0.09239%** |
| log-Padua degree 2 | 288 | 0 | 45 | 0.19403% |
| current random extractor, 5 seeds | 126--134 | 683--705 | 41--42 | 0.10661%--0.17119% |

The degree-1 deterministic extraction is seed-free and slightly more accurate
than all five random runs, at 14% more full response solves than the nominal
126-solve run.  It also removes hundreds of random residual probes.

Degree 2 being worse than degree 1 is the decisive negative result.  More
samples changed the empirical singular-vector weighting; the `1e-3` SVD kept
45 rather than 47 modes.  Tightening only the deterministic closing cutoff to
`1e-4` raised the orders to 67--68 and reduced the worst errors to 0.00744% and
0.00978%, respectively.  Thus the sampling family is not the cause of the
non-monotonicity.

A nested five-point level-2 log-Smolyak rule gave 240 snapshots, order 45, and
0.07890% worst junction error on a smaller 15 by 15 plus 64-point holdout.  A
13-point level-3 rule again failed to improve after the same SVD cutoff.  This
is consistent with the closing-compression diagnosis.

## Decision

Do not replace the current extractor by "fixed Padua samples followed by the
existing SVD" and call it certified.  That would remove the random seed but the
claimed interpolation guarantee would be destroyed in the last step.

The defensible next algorithm is:

1. use nested log-Smolyak points as the deterministic candidate hierarchy;
2. append only the new hierarchical shell;
3. compress the accumulated responses;
4. re-evaluate the residual on every retained sparse-grid node and keep more
   singular vectors until the compressed space passes;
5. stop from a certified analytic tail/hierarchical-surplus bound, or report
   the sparse-grid residual as a deterministic finite-set statement rather
   than a box-wide guarantee.

This retains the useful part of the closed-form construction while making the
remaining source of non-certification explicit.
