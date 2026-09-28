# Matrix-valued HTC box certificate: valid and (at order 3) tight

> **Plan provenance.**  Every number in this record was produced with the
> **legacy bare-`K` frequency plan** (`shared_frequency_plan`: 11 shifts on
> `lambda in [4.4716e-04, 3.9001e+01]` at 5 mm, 12 shifts at 2.5 mm).  The
> delivered basis uses the **box-corrected plan** instead
> (`box_spectral_interval`: 13 shifts on `lambda in [4.2795e-05, 4.1400e+01]` at
> 5 mm, 14 at 2.5 mm); the benches now take `--plan box|legacy` and default to
> `box`.  The statements below are about the certificate itself; the box plan is
> certified in records/BOX_BRANCH_AND_BOUND.md.

Positive result, and the replacement for the rejected `C`-metric scalar.  At one
fixed real shift the delivered basis is certified against *every* HTC vector of
the box and *every* input combination by a relative matrix statement, and on
Case 1 / 5 mm a 16x16 log partition with jet order 3 makes that statement 0.8 %
tight while costing no additional full-order extraction solve.

## 1. Statement

For one shift `s`, `A(h) = K + s C + sum_i h_i H_i`, residual
`R(h) = G - A(h) V q(h)`, and

```text
E(h) = R(h)^T A(h)^-1 R(h),      Y(h) = G^T A(h)^-1 G,
```

the exact relative collocated port defect of the delivered space is

```text
sup_h lambda_max(E(h), Y(h)) = sup_h sup_w w^T (Y(h) - Y_V(h)) w / w^T Y(h) w,
```

(its square root is the equal `A`-energy state error, `[STATE-AUX]` and not
reported), with the worst input `w` a 4x4 generalized eigenvector of `(E, Y)`.  On a cell `Q` with
lower corner `a` and upper corner `b`, the certificate is

```text
sup_{h in Q} lambda_max(E(h), Y(h)) <= max_nu lambda_max(Xi_nu^T S_a Xi_nu, Y(b)),
S_a = Z^T A(a)^-1 Z,
```

from three exact facts: `H_i >= 0` gives `A(h) >= A(a)` hence
`A(h)^-1 <= A(a)^-1` (Loewner); Galerkin optimality gives `E(h) <= R_q(h)^T
A(h)^-1 R_q(h)` for *any* trial `q`, so a Taylor jet of the reduced solve is
allowed to be bad without breaking the bound; and `X -> X^T S_a X` is matrix
convex, so the tensor Bernstein enclosure `R_q = Z Xi(h) = Z sum_nu beta_nu(h)
Xi_nu` with `beta_nu >= 0`, `sum_nu beta_nu = 1` bounds the quadratic form by
the largest coefficient.  The denominator uses the *upper* corner, which is a
lower bound of `Y` on the whole cell because the family is entrywise decreasing
in `h`.

Note what this fixes relative to the entrywise certificate: no per-entry
denominator and no entrywise maximum over coefficients, so weak coupling cannot
blow the statement up, and the result is already normalized by the exact
transfer of the same cell.

## 2. Falsification against the exact map (Case 1, 5 mm, design basis, 42 vectors, s = 0)

650-point box reference from the exact map: `1.824841e-04`.  Uniform log
partitions, both anchor rules, zero Loewner violations at every configuration:

```text
cells/axis  anchor  order   bound        bound/exact  median  violations
2           local   2       7.435e+01    4.08e+05     29762   0
4           local   2       2.817e+00    1.54e+04     4265    0
8           local   2       5.860e-02    3.21e+02     158     0
8           local   3       1.557e-02    8.53e+01     30      0
8           local   4       4.135e-03    2.27e+01     5.2     0
8           block   2       1.819e-01    9.97e+02     260     0
8           block   3       4.832e-02    2.65e+02     41      0
8           block   4       1.283e-02    7.03e+01     6.8     0
16          block   2       4.449e-03    2.44e+01     4.5     0
16          block   3       3.432e-04    1.88e+00     1.2     0
16          local   2       8.114e-04    4.45e+00     2.7     0
16          local   3       1.8396e-04   1.008e+00    1.17    0
```

The last row is the headline: at 16 cells per axis with jet order 3 the bound is
`1.8396e-04` against the exact `1.8248e-04`, i.e. 0.8 % above the true box
maximum, and the median cell is 17 % above its own exact value.  In port terms
the certified statement is: over the whole box at this shift the worst relative
all-input collocated port defect `sup_w w^T (Y - Y_V) w / w^T Y w` of the
delivered basis is `1.8396e-04`, against a true `1.8248e-04`.  No square root is
taken: this is a port quantity, and the A-energy state error is *not* reported
anywhere (`[STATE-AUX]`, see records/NEGATIVE_RESULTS.md).  The pre-registered
gate was "after refinement the cells that
dominate the maximum must be within 10x": the dominating cells are at `1.06x`
(local anchor) and `1.06x`-`1.62x` (block anchor, order 2/3), so the gate is
passed with margin.

Both refinement directions help and they are not interchangeable:

* raising the jet order at a fixed partition buys about 4x per order at 8
  cells/axis, which shows the enclosure of the *trial* is a real loss;
* refining the partition buys more, but each cell that gets its own anchor costs
  one Riesz Gram (130 sparse solves at 5 mm), which is where the Woodbury
  sharing belongs: a shared reference factorization plus `m_b` backsolves makes
  every cell anchor a small dense update (see the plan below).

Block anchors (16 anchors for a 4x4 log block structure, reused by 256 cells)
reach `1.88x` at order 3, so the cheapest configuration that still passes the
gate does not need a Gram per cell.

The same configuration certifies a different delivered basis: the stock
extractor (seed 20260805, 36 vectors, 137 right-hand sides) has exact box
maximum `1.0905e-04` on the 650-point reference and `1.0978e-04` on the in-cell
sampling (the finer of the two, so the truth is at least that large), and at 16
cells/axis with order 3 its bound is `1.125e-04` (`1.025x` tight against that
sampled worst) with cell-local anchors and `3.433e-04` (`3.13x`) with shared
block anchors, again with zero violations.  Its exact maximum is *smaller* than
the design basis's `1.8248e-04` at this shift, which repeats the earlier
observation that the deterministic design buys the continuous statement rather
than raw accuracy.

The tabulated design-basis numbers above combine two runs: the partitions up to
16 cells/axis with orders 2/3 from one run and orders 4 at 8 cells/axis from
another; the exact reference is identical in both (`1.824841e-04` on 650 points),
which is the check that the two runs are comparable.

## 2b. Nonzero shifts, denser sampling, a second mesh

The steady shift is the binding one.  On the same 16x16 partition with order 3:

```text
operator                        exact box maximum  bound (block)  bound (local)  violations
s = 0                           1.8248e-04         3.4317e-04     1.8396e-04     0
s = plan shifts[5] = 1.3206e-01 2.8403e-07         1.9703e-05     8.0723e-06     0
s = plan shifts[0] = 3.3248e+01  5.7470e-09         5.7473e-09     5.7471e-09     0
```

* The defect falls monotonically in the shift (`1.82e-04`, `2.84e-07`, `5.75e-09`):
  the HTC dependence weakens with real frequency, so the steady operator is the
  binding one and certifying `s = 0` tightly is what the frequency plan costs.
  At `s = 33.25` the bound is `1.02x` tight even with block anchors; at
  `s = 0.1321` it is `69x` (block) or `28x` (local) while its dominating cells
  are still at `1.07x` / `1.06x`.  A stopping rule keyed on the *dominating*
  cells therefore reduces to a fine certification near the steady operator,
  while a rule keyed on the largest bound over all cells would be dominated by
  cells that are certified far from their own maximum - which is why the greedy
  has to keep the exact witness `L` separate from the certificate `U`.
* Denser sampling (11 points per axis inside each cell, about `3.1e4` exact
  evaluations) leaves the s = 0 numbers unchanged and still reports zero
  violations, so the earlier evidence was not an artifact of the sampling.
* Second mesh, 2.5 mm (9072 dof), same design basis recipe, 8 cells/axis and
  order 3: exact box maximum `3.4798e-04`, bound `1.5587e-02` (44.8x) with local
  anchors and `4.8373e-02` (139x) with 4x4 block anchors, zero violations,
  dominating cells at `6.81x`.  The 2.5 mm certification is therefore close to
  but not inside the 10x gate at 8 cells/axis, and needs either a finer
  partition or a higher trial order there.
* Denominator: the reduced `Y_V(b)` reproduces the exact-`Y(b)` bound to six
  significant digits at every configuration tried, with zero fallbacks to the
  exact denominator.  It removes one factorization and `k` solves per cell.

## 2c. Trial portfolio and p-saturation

16 cells/axis, 4x4 block anchors (16 Gram matrices for 256 cells), reduced
denominator, exact box maximum `1.824841e-04`:

```text
trial                  p=2       p=3       p=4       p=5
taylor                 4.449e-03 3.432e-04 1.8396e-04 1.8396e-04
chebyshev              7.795e-03 8.501e-04 1.8396e-04 1.8396e-04
logfit                 8.377e-03 7.572e-04 1.8396e-04 1.8396e-04
portfolio (min)        4.449e-03 3.432e-04 1.8396e-04 1.8396e-04
violations             0         0         0          0
```

* The bound saturates at `1.8396e-04`, i.e. `1.008x` the true box maximum, from
  order 4 on, and every trial converges to that same value: the remaining 0.8 %
  is the enclosure and denominator gap of the cell that carries the maximum, not
  trial error.
* Which trial wins is not universal, exactly as expected without a theorem: the
  Taylor jets win at p = 2 and p = 3 on this partition (the Chebyshev and
  log-spread fits are 2.2-2.5x worse there), while at p >= 4 they agree to five
  digits.  The portfolio costs no Gram matrix and still helps where it matters:
  at p = 2 it takes the dominating-cell effectivity from `1.623x` to `1.058x`.
* Consequence for the design: `p = 4` Taylor jets with a 4x4 block structure is
  enough, so the Woodbury backend and finer anchors are frozen rather than built.
  Cost per shift is 16 factorizations plus `16 * 130 = 2080` large right-hand-side
  solves, no cell denominators at all (they are reduced) and no AMG-CG solve.

## 2d. Second mesh, p-refinement only (2.5 mm, 8 cells/axis)

The question this run had to answer was whether the 2.5 mm cost at 8 cells/axis
comes from an unconverged trial or from cells that are simply too wide, so only
`p` was changed: 4x4 block anchors, reduced denominator, `s = 0`, exact box
maximum `3.479831e-04`.

```text
trial                p=3        p=4        p=5        dominating cells (p=5)
taylor               4.837e-02  1.285e-02  3.416e-03  1.086x
chebyshev            1.847e-01  4.603e-02  1.713e-02  1.086x
portfolio (min)      4.837e-02  1.285e-02  3.416e-03  1.086x
U*/L*                139x       36.9x      9.82x      -
U*/L_Q at the U* cell 3313x      880x       234x       -
L_Q/L* at the U* cell 0.042      0.042      0.042      -
violations           0          0          0          -
```

* Both effects are present but they separate cleanly.  `U*` is still falling by
  about `3.8x` per order with no saturation, so `p`-refinement alone has not
  converged - unlike 5 mm / 16 cells where order 4 already saturated.  That is
  the signature of cells that are too wide, and the answer is `h`-refinement,
  not a higher order.
* The cell that carries `U*` is *not* the cell that carries `L*`: its true defect
  stays `0.042 x L*` at every order, i.e. 24x below the global maximum, while its
  own certificate is still `234x` above its own value.  So it is both a false
  positive as a witness of the maximum and a loose bound in its own right, and no
  order fixes it - it is a wide cell.  The cells that do carry the maximum are
  already certified at `1.086x`.  This is exactly the configuration where
  adaptive splitting beats uniform refinement, because refining 8x8 -> 16x16
  uniformly would spend all its cells on the same improvement that splitting one
  false-positive cell buys.
* Taylor still beats Chebyshev at every order here (`3.8x` at p = 5), the same
  coarse-cell ranking seen on 5 mm, and the portfolio never loses: it is at most
  the better of the two by construction.

## 3. Cost

```text
configuration                 Grams  Gram solves  cell denominators  bound solves
16 cells/axis, local anchor    256    33280        0                  0
16 cells/axis, block anchor    16     2080         0                  0
```

No AMG-CG extraction solve is added by the certificate and no full-order cell
denominator is left: the 33280 Riesz solves are large-RHS back-substitutions
against the anchor factorizations, and the cell denominators are the *reduced*
Galerkin transfers `Y_V(b)`, which are small dense solves in the basis.  Both are
reported separately from the enrichment budget, as required.

## 4. Pitfall found while implementing

The matrix path must not clip the coefficient matrices entrywise to
non-negativity - that is correct for the entrywise bound (only diagonals are
used) but it changes the Loewner order: at a degenerate cell the first version
reported `1.7039e-04` against an exact `1.8248e-04`, i.e. an actual 6.6 %
violation.  After removing the clip the degenerate cell reproduces the exact
value to 13 digits (`1.824840729e-04` vs `1.824840732e-04`), which is the
regression test for the machinery.

## 5. Limits

* Inequalities are exact in real arithmetic; the Gram matrices, factorizations
  and small dense solves are ordinary floating point
  (`floating_point_certified=False`).
* Falsification samples the interior of every cell, so it is evidence, not a
  proof: the reported violation count is "no counterexample found on the
  sampled points".
* One real shift (`s = 0`) and one mesh (5 mm) so far; the box-plan shifts, the
  2.5 mm mesh and the stock bases are the immediate next runs.
* "The steady shift is the binding one" is an observation on the shifts measured
  here, not a theorem: for a fixed `V` both the numerator `E(s)` and the
  denominator `Y(s)` move with the shift, and their ratio need not be monotone in
  `s`.  A finite-plan certificate still has to cover every matching shift; it is
  only the *frequency axis* that does not need to be continuous, because the plan
  has finitely many shifts.
* The refinement chain is not asserted to be monotone in the trial.
  `U_Q(W_n)` is computed from a *fresh* Taylor/Chebyshev trial at each space, so a
  larger space can in principle produce a worse bound even when the true defect
  fell.  The clean fix, not yet implemented because there is no enrichment yet,
  is to carry the previous trial forward as one more portfolio candidate:
  keeping `W_n`'s columns gives `q^{carry}_{n+1}(h) = [q_n(h); 0]`, and after an
  orthogonalising rotation the same physical trial is `T q_n(h)`, so
  `U_Q^{n+1} = min(U_Q^{fresh}, U_Q^{carry}) <= U_Q^n` in pure reduced algebra.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_matrix_cell_certificate.py 5 \
  --cells 8 --orders 2,3,4 --samples 5 --reference-cells 24 --output <path>.json
```
