# The fixed-shift port certificate, its refinement, and the extraction budget

> Historical record: references below to a delivered Robin-box frequency plan
> and its performance describe a reverted implementation. The current extractor
> uses the original bare-kernel frequency estimator. Fixed-shift certificates
> for a given final basis remain independent of how its shifts were selected.

One record for the three positive statements about the certificate machinery,
each a part of this file:

* **Part A** - at one fixed real shift the delivered basis is certified over the
  **whole** HTC box and **every** input combination, and at a sufficient partition
  the statement is tight.  This part also carries the partition setting that makes
  a certificate number readable, the trial portfolio, the nonzero-shift and second
  mesh evidence, the cost and the implementation pitfall.
* **Part B** - the continuous-box statement can be decided by branch and bound
  without enrichment, and the shared anchor span makes it cheap.
* **Part C** - the same certificate closes the `N_FOM`-versus-accuracy trade-off
  into a Pareto table.

The parts are the original per-topic records, moved here unchanged so that no
measured number was lost in the merge; only the heading levels were adjusted.

Everything here is `[PORT-FIXED-S]`: the certified quantity is
`lambda_max(Z(h) - Z_V(h), Z(h))`, the relative all-input collocated port defect at
one real shift.  No square root is taken anywhere, and the equal `A`-energy state
error is `[STATE]`; this record reports the port quantity only.  The step from a fixed shift to the
port system norm is the open `[PORT-SYSTEM]` gap, which lives in
`records/SYSTEM_GAP_AND_FAILURE_ARCHIVE.md`, not here.

---

## Part A - the matrix cell certificate: validity, tightness and the partition setting

> **Plan provenance.**  Every number in this record was produced with the
> **legacy bare-`K` frequency plan** (`shared_frequency_plan`: 11 shifts on
> `lambda in [4.4716e-04, 3.9001e+01]` at 5 mm, 12 shifts at 2.5 mm).  The
> delivered basis uses the **box-corrected plan** instead
> (`box_spectral_interval`: 13 shifts on `lambda in [4.2795e-05, 4.1400e+01]` at
> 5 mm, 14 at 2.5 mm); the benches now take `--plan box|legacy` and default to
> `box`.  The statements below are about the certificate itself; the box plan is
> certified in records/PORT_CERTIFICATE_AND_BUDGET.md.

Positive result, and the replacement for the rejected `C`-metric scalar.  At one
fixed real shift the delivered basis is certified against *every* HTC vector of
the box and *every* input combination by a relative matrix statement, and on
Case 1 / 5 mm a 16x16 log partition with jet order 3 makes that statement 0.8 %
tight while costing no additional full-order extraction solve.

### A1. Statement

For one shift `s`, `A(h) = K + s C + sum_i h_i H_i`, residual
`R(h) = G - A(h) V q(h)`, and

```text
E(h) = R(h)^T A(h)^-1 R(h),      Y(h) = G^T A(h)^-1 G,
```

the exact relative collocated port defect of the delivered space is

```text
sup_h lambda_max(E(h), Y(h)) = sup_h sup_w w^T (Y(h) - Y_V(h)) w / w^T Y(h) w,
```

(its square root is the equal `A`-energy state error, `[STATE]`, which this
record does not report), with the worst input `w` a 4x4 generalized eigenvector of `(E, Y)`.  On a cell `Q` with
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

### A2. Falsification against the exact map (Case 1, 5 mm, design basis, 42 vectors, s = 0)

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
taken: this is the port quantity, and the A-energy state error (`[STATE]`, see
records/SYSTEM_GAP_AND_FAILURE_ARCHIVE.md) is reported separately by the
field-level metrics.  The pre-registered
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

### A2b. Nonzero shifts, denser sampling, a second mesh

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

### A2c. Trial portfolio and p-saturation

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

### A2d. Second mesh, p-refinement only (2.5 mm, 8 cells/axis)

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

### A2e. Basis discrimination on the delivered box plan (16 cells/axis, order 3)

The 8-cell rows of A2 are enclosure-limited and therefore cannot separate bases at
all.  Raising the partition to 16 cells per axis with the same order 3 and
cell-local anchors makes the certificate discriminating, and it reverses the
picture the enclosure-limited run suggested:

```text
Case 1, 5 mm, box plan, s = 0, 16 cells/axis, order 3, cell-local anchors
exact grid = 441 points; exact@centres = the same map at the 256 cell centres

basis                    dim V   exact grid   exact@centres   certified    cert/exact
deterministic (design)      41    3.3471e-04   2.7502e-04      3.3671e-04   1.006
stock_20260805              36    1.0861e-04   1.0434e-04      1.1252e-04   1.036
stock_legacy_20260805       35    9.8431e-04   7.9589e-04      9.9328e-04   1.009
stock_7                     37    1.6382e-04   1.1148e-04      1.6512e-04   1.008
stock_legacy_7              36    1.3102e-03   1.0551e-03      1.3260e-03   1.012
```

* The certificate is `1.006`-`1.036` tight on **all five** delivered bases, so the
  tightness statement is not an artifact of the design basis.
* On the delivered box plan the deterministic design basis is about **3x worse**
  than the stock extractor at the steady shift (`3.347e-04` against `1.086e-04`).
  The design buys the continuous statement, not steady accuracy - the same
  trade-off A2 records from the other direction.
* The two legacy-plan bases are a further order of magnitude out
  (`9.84e-04`, `1.31e-03`), independently reproducing the box-plan defect A8.
* **Consequence for reading any certificate table**: a run at 8 cells/axis returns
  `1.5566e-02` for every good basis (A2) while the same bases span
  `1.09e-04 .. 3.35e-04` exactly.  A certificate number is only interpretable next
  to the partition it was computed on, which is why the partition setting is part
  of the statement and not a benchmark knob.

### A3. Cost

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

### A4. Pitfall found while implementing

The matrix path must not clip the coefficient matrices entrywise to
non-negativity - that is correct for the entrywise bound (only diagonals are
used) but it changes the Loewner order: at a degenerate cell the first version
reported `1.7039e-04` against an exact `1.8248e-04`, i.e. an actual 6.6 %
violation.  After removing the clip the degenerate cell reproduces the exact
value to 13 digits (`1.824840729e-04` vs `1.824840732e-04`), which is the
regression test for the machinery.

### A5. Limits

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
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py tightness 5 \
  --cells 8 --orders 2,3,4 --samples 5 --reference-cells 24 --output <path>.json
```

---

## Part B - deciding the continuous box without enrichment

The continuous-stop statement is `forall leaf Q: u_Q <= tau_port`, where `U_Q`
is the matrix cell certificate of records/PORT_CERTIFICATE_AND_BUDGET.md.  This note
records the refinement loop that decides it without enrichment, and the numbers
that decide whether it beats a uniform partition.

### B1. Algorithm

`BoxCertificate.branch_and_bound(threshold, ...)` refines cheapest first:

* a leaf tries the trial orders in turn at its fixed cell and anchor
  (p-refinement, no new Gram matrix), and stops at the first order that is below
  the threshold, so the order used varies per leaf;
* a leaf still above the threshold is h-refined, by log-bisecting the widest axis
  of the leaf that carries the largest bound - one split per round;
* the anchor is the shared 4x4 block anchor, so any number of leaves still costs
  at most 16 Gram matrices.  A leaf may build its own anchor behind
  `upgrade_anchor`, with the attempt cached because an unsplit leaf has the same
  certificate next round.

Correctness and stopping use `U` alone.  The exact witness `L` at sampled points
of a leaf - one full-order solve each - is behind `oracle=True` and exists to
study the refinement and to locate enrichment witnesses later.  A certificate
that did full-order work on every leaf would already have spent more than the
enrichment it is meant to justify, so it is not the production path.

### B2. Does it beat the uniform partition? (5 mm, design basis, s = 0)

Exact box maximum `L* = 1.824841e-04` on 650 reference points.

```text
scheme                                  leaves  Grams  bound      U*/L*   seconds
adaptive B&B, threshold 5e-04           79      16     4.7185e-04  2.59    29.4
adaptive B&B, threshold 2.5e-04         88      16     2.4543e-04  1.35    42.7
uniform 16 cells/axis, order 4          256     16     1.8396e-04  1.008   47.6
uniform 16 cells/axis, order 4 local    256     256    1.8396e-04  1.008   23.5
```

* The adaptive loop reaches `2.5e-04` with `88` leaves against the uniform
  partition's `256` cells, at the same 16 Gram matrices: the value of B&B here is
  a `2.9x` smaller cell count for a comparably tight certificate, not a bound the
  uniform partition cannot reach at all.
* It ends looser (`1.35x` against `1.008x`) only because it stops at the
  threshold instead of pushing the bound to its floor.
* No anchor upgrade is needed: the shared block anchors are enough at this mesh,
  so the Woodbury backend and finer anchors stay frozen.

### B3. Plan provenance: legacy bare-`K` versus box-corrected

Every number in Part B's table was produced with the **legacy bare-`K` plan**
(`shared_frequency_plan` encloses the generalized spectrum of `K` per source
port).  A later, now-reverted implementation used the **box-corrected plan**
instead. At 5 mm and tolerance `1e-3`, the historical plans were:

```text
plan     shifts  lambda_min   lambda_max
legacy   11      4.4716e-04   3.9001e+01
box      13      4.2795e-05   4.1400e+01
```

The two plans are not interchangeable, so the benches now take
`--plan box|legacy` (default `box`), record the kind, interval and count in the
report, and print them.  The runs in Part B are therefore a **legacy-plan
regression**: they are valid statements about that plan and say nothing about the
delivered one.

### B4. The delivered box plan, all thirteen shifts (threshold 2.5e-04)

Design basis rebuilt on the box plan (order 42, exact box maximum `1.763102e-04`
at `s = 0`), certificate mode, 4x4 block anchors:

```text
                              per-shift span        shift-free common span
shift        leaves  Grams    bound        leaves   bound        Grams
3.5232e+01   4       4        1.0544e-08   4        2.9414e-05   4
1.3964e+01   4       4        3.2055e-08   4        1.5272e-05   4
4.4658e+00   4       4        2.2779e-06   4        1.4198e-05   4
1.3941e+00   4       4        2.1987e-05   4        1.7176e-04   4
4.3418e-01   6       6        5.1249e-05   12       7.3249e-05   8
1.3519e-01   12      8        4.4008e-05   12       1.3682e-04   8
4.2092e-02   24      8        7.8550e-05   24       1.4050e-04   8
1.3105e-02   24      8        2.3009e-04   24       2.1429e-04   8
4.0805e-03   30      10       1.8938e-04   30       2.4658e-04   10
1.2708e-03   45      13       2.4875e-04   45       2.4790e-04   13
3.9672e-04   54      13       2.2727e-04   57       2.0779e-04   15
1.2688e-04   63      16       2.4459e-04   65       2.4068e-04   16
5.0286e-05   71      16       2.4165e-04   73       2.4061e-04   16
             -------                        -------
             114 Grams                      16 Grams (shared)
```

* The whole delivered plan certifies at `2.5e-04` in about nine seconds of
  certificate work, and the low-frequency end is again the expensive part.
* The shift-free common span `Z = [G, K V, C V, H_1 V, ...]` certifies every
  shift from a single anchor Gram `Z^T A(a,0)^-1 Z`, because `A(h, s) >= A(a, 0)`
  for every `s >= 0` and the shift enters the residual only through the
  coefficient `s`.  It needs **16 Gram matrices for the entire plan instead of
  114**, with the same leaf counts to within `3` and the same total runtime: the
  cheaply-shared span is not looser in any way that costs refinement.
* The Gram matrices are shared across shifts through an explicit cache
  (`gram_cache=`), which is exactly what makes the common span pay off.

### B5. Measurement cost: persistent leaves and a heap

The first loop re-measured every leaf in every round while splitting one leaf per
round: 84 splits over a growing leaf set cost about `sum_{n=4}^{88} n ≈ 3900`
cell evaluations.  Leaves are now measured once, at creation, and kept in a
max-heap keyed by their bound, so a split only measures its two children and
acceptance is the pop of the first leaf below the threshold:

```text
threshold  scheme            splits  leaves  created  measures  bound        seconds
5e-04      rounds (old)      75      79      154      ~2600     4.7185e-04   29.4
5e-04      heap (new)        75      79      154      154       4.7185e-04   1.6
2.5e-04    rounds (old)      84      88      172      ~3900     2.4543e-04   42.7
2.5e-04    heap (new)        84      88      172      172       2.4543e-04   1.6
```

Identical bounds and leaf counts, `18x` to `27x` less wall clock and `~23x` fewer
certificate evaluations.  The 42.7 s figure quoted above was therefore an
implementation cost, not an algorithmic one.

### B6. Pitfall: a round cap looks like an anchor floor

With `max_rounds = 32` the same run stalls at `2.97e-03` with 36 leaves and the
last eight splits move it only from `3.4e-03` to `2.97e-03`.  That looks exactly
like an anchor-mismatch floor, and it is not one: the block anchor does not get
looser as the cell shrinks, so the plateau reads as a property of the anchor
rule.  Raising the cap to 400 rounds brings the same configuration to
`2.45e-04`, i.e. 12x lower, with the same 16 Gram matrices.

Rule: a branch-and-bound result only measures the refinement policy if the loop
terminated by exceeding its threshold.  Report `max_rounds` and `max_cells`
next to every number, and never read a capped run as a structural limit.

### B7. Cost

Per accepted run at 5 mm: one shared Gram matrix per anchor, i.e. `16 * 172 =
2752` large RHS solves for the **whole frequency plan** with the shift-free common
span (against `114` Grams, one set per shift, with the per-shift span), one
reduced solve per leaf for its reduced denominator, and reduced algebra for the
Bernstein enclosure.  No cell denominator solve, no AMG-CG solve, no full-order
solve outside the anchor Grams in certificate mode, and a whole-plan run costs
under ten seconds once the loop keeps its leaves.  That shared-span saving is
not free: see B8 for the refinement-independent floor it introduces on
the large shifts.

### B8. The measured anchor floor of the shared shift-free span

The heap branch and bound stalls at a fixed bound on a few large shifts, and the
single-shot sweep of records/PORT_CERTIFICATE_AND_BUDGET.md does not see that because
it never pushes a cell to the point where refinement stops paying.  The plateau is
now measured directly: for 2.5 mm and the plan's second largest shift
`s = 4.3517e+01`, the worst leaf after 198 splits has bound `3.8026125e-04`, and
the exact parameter value of the same cell form (the anchor floor, see
`BoxCertificate.point_floor`) at the centre and the four corners of that leaf is

```text
span      floor over the worst leaf        worst leaf bound
common    3.8026111e-04 .. 3.8026112e-04   3.8026125e-04
shift     3.3581499e-07 .. 3.3581499e-07   3.8026125e-04
```

So the plateau *is* the floor of the shift-free span, to seven digits, and the same
leaf under the per-shift span has no floor at all (its floor equals the exact
defect there).  The mechanism is the sharing itself: the common span takes its
Riesz Gram from the shift-free operator `A(a, 0)`, and for `s = 4.35e+01` the shift
term dominates `A(a, s)`, so `A(a, 0)^-1` overstates the residual energy that the
certificate is allowed to charge.  The floor is independent of the partition, which
is why more splits never helped, and it is above the `3.5e-04` used in the
experiment while the per-shift span needs no partition at all there.

Consequence, stated without reopening the frozen design: the 16-gram shared span is
correct but not free - it costs a floor on the large shifts, and any threshold below
that floor is unreachable *with that span*, no matter how many cells are spent.  The
per-shift span has no such floor, at 114 grams for the plan instead of 16.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py bandb 2.5 \
  --plan box --shifts 1 --span common --orders 2,3,4,5 --samples 2 --max-cells 400 \
  --max-rounds 500 --thresholds 3.5e-4 --floor-probe --output <path>.json
```

Reproduce (the plan sweep):

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py bandb 5 \
  --plan box --shifts all --span common --orders 2,3,4,5 --samples 2 \
  --max-cells 300 --max-rounds 400 --thresholds 2.5e-4 --output <path>.json
```

### B9. Limits

* Thresholds are demonstration levels, not adopted tolerances: `tau_port` stays
  symbolic until the dynamic bridge `Phi_Sigma` fixes it.  Because `U` is an upper
  bound, no threshold below `L*` can ever be accepted, which is why the ladder is
  chosen above `L*`, and `L*` is an exact-map *sampled* reference rather than a
  continuous maximum.
* Inequalities are exact in real arithmetic; Gram matrices, factorizations and
  the small dense solves are floating point (`floating_point_certified=False`).
* Certificate mode is evidence about the algorithm, not about the exact defect:
  without `oracle=True` the run reports no witness, and the acceptance is only as
  good as the certificate it is built on.
* Reported for one mesh (5 mm).  The common-span statement is proved per anchor
  from `A(h, s) >= A(a, 0)` and needs `s >= 0`; the box plan's shifts are all
  positive, and the steady operator `s = 0` is certified separately.
* A leaf's own anchor is allowed but unused by default; the observed data does not
  justify it.  The asymptotic mismatch floor
  `F_B(h) = lambda_max(R_V(h)^T A(a_B)^-1 R_V(h), Y_V(h))` was written here as the
  criterion to revisit if some mesh stalls; it has now been triggered and measured
  (B8): the stall is the floor of the *shared shift-free* span, not of the
  spatial block anchors.  No certificate design is reopened for it.

---

## Part C - the extraction budget: N_FOM against the certified defect

The dynamic bridge is not closed, so there is no vendor-derived stopping threshold
for the matching defect; what can be stated today is a Pareto table.  For every
combination of

```text
(parameter points requested, SVD cutoff)
```

the delivered design basis is rebuilt from the **same** cached snapshots on the
box-corrected plan - lowering the cutoff therefore costs no further full-order
solve - and certified shift by shift over the whole box.  Four quantities are
reported together, never separately:

```text
N_RHS        full-order inverse actions
D_cert       max_j max_Q U_{Q,j}(V)     certified, [PORT-FIXED-S]
D_wit        max over sampled (shift, point) of the same quantity
dim(V)       delivered ROM order
```

**No square root is taken**: `D_cert` and `D_wit` are the port defect itself, and
the square root is the state-amplitude quantity that is no longer reported.

> **Convention change.**  The table in C1 was measured before the square root was
> removed from `bench_pareto_budget.py`, so its `sqrt(U*)` / `sqrt(L*)` columns are
> the square roots of the quantities the script now calls `D_cert` / `D_wit`.  The
> readings below are unchanged - a monotone transform does not move a plateau - but
> the numbers are not the current output: `D_cert` is `sqrt(U*)^2`, e.g.
> `1.5863e-04` where the table reads `1.2595e-02` for five points at cutoff `1e-3`,
> which the current script reproduces directly.  A rerun with the current code and
> the current greedy scorer is in flight and its raw JSON is the table to use.

### C0. Counting convention

One *operator block* is one factorization of `A(h, s)`, and every block serves one
right-hand side per port, so

```text
N_RHS = number_of_ports * (selection_blocks + build_blocks),      ports = 4 here.
```

Only `N_RHS` may be compared with the stock extractor's counter, whose numbers
(137/138 at 5 mm, box-corrected) already use this convention: the three-point
deterministic design is `(13 + 1) * 3 * 4 = 168` RHS, which is what this bench
reproduces.  The selection's residual certificate and the box certificate itself
are certification overhead - 256 Gram matrices per shift here - and are reported
separately; they are never added to `N_RHS`.  `tau_port` stays symbolic.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_pareto_budget.py 5 \\
  --points 2,3,8 --cutoffs 1e-3,3e-4,1e-4,3e-5,1e-5 --cells 16 --order 3 \\
  --witness --witness-cells 2 --output <path>.json
```

### C1. The table, Case 1 at 5 mm, 16 cells per axis, order 3

`fresh` is what that row cost; `cumulative` is the total the script has spent so
far.  Requesting 8 points selects 5, because the greedy scorer stops at its own
`1e-5` tolerance.

```text
points   cutoff    fresh RHS  cum RHS  dim V  sqrt(U*)     sqrt(L*)     U*/L*
2        1e-03     112        112      39     6.274697e-02 6.239355e-02 1.0057
2        3e-04       0        112      45     4.219707e-02 4.199702e-02 1.0048
2        1e-04       0        112      50     3.719702e-02 3.702670e-02 1.0046
2        3e-05       0        112      57     3.357565e-02 3.342541e-02 1.0045
2        1e-05       0        112      63     3.353020e-02 3.338020e-02 1.0045
3        1e-03      56        168      42     1.333271e-02 1.327819e-02 1.0041
3        3e-04       4        168      48     9.198663e-03 9.161398e-03 1.0041
3        1e-04       4        168      56     7.910539e-03 4.162093e-03 1.9006
3        3e-05       4        168      66     7.910058e-03 1.110416e-03 7.1235
3        1e-05       4        168      75     7.910037e-03 9.932350e-04 7.9639
5        1e-03     112        280      42     1.259468e-02 1.254322e-02 1.0041
5        3e-04       8        280      48     1.167809e-02 1.163050e-02 1.0041
5        1e-04       8        280      59     7.910301e-03 1.369527e-03 5.7759
5        3e-05       8        280      71     7.910034e-03 4.270972e-04 18.520
5        1e-05       8        280      80     7.910025e-03 4.225131e-04 18.721
```

Three readings, each of which is a different verdict:

1. **Lowering the cutoff is free in full-order solves and is not free in rank.** From
   `1e-3` to `1e-5` the fresh cost of every further row is exactly zero (all
   snapshots are already cached) while the delivered order rises from 39 to 63 (two
   points), 42 to 75 (three) and 42 to 80 (five).  The 1e-3 truncation was discarding
   most of the information the extraction had already paid for.
2. **The exact defect follows the rank, the certified bound does not.** The witness
   falls by more than an order of magnitude - five points: `1.254e-02` to
   `4.225e-04` - while `delta_cert` sticks at `7.9100e-03` in every row whose rank
   exceeds about 50.  That plateau is neither the basis nor the plan: it is the
   order-3 cell enclosure of one specific middle-band `(shift, cell)`, and the
   effectivity `U*/L*` rises from `1.004` to `18.7` as the true defect leaves it
   behind.  **The certificate's middle band is now the binding limit on what can be
   stated**, which is exactly the trigger condition recorded for upgrading a single
   shift's certificate: only the dominating `(shift, cell)` needs a higher trial
   order or a portfolio, not the whole plan.
3. **Under the stock gate the two-point family is the only option, and it does not
   reach the three-point accuracy.** `112 < 137 < 168` RHS: the best two-point row
   gives `sqrt(L*) = 3.34e-02` against `9.16e-03` for three points at `1e-3` (168
   RHS).  Compression alone recovers a factor 1.9 within the two-point budget
   (`6.27e-02` to `3.35e-02`) and then saturates, so under the gate the remaining
   lever is not the cutoff but *which* points are chosen.

The certificate cost stays out of all of this: 256 anchors and 50-190 s per row at
16 cells per axis, i.e. a research-mode price, not a production one.

### C2. Per-shift split (five points, cutoff 1e-3, 280 RHS)

```text
shift          bound sqrt(U*)   witness sqrt(L*)   U*/L*
0.000000e+00   1.2595e-02       1.2543e-02         1.004
5.028615e-05   1.0998e-02       1.0958e-02         1.004
1.268782e-04   9.3533e-03       9.3242e-03         1.003
3.967216e-04   7.5467e-03       6.9739e-03         1.082
1.270821e-03   7.1433e-03       5.6344e-03         1.268
4.080540e-03   6.5839e-03       4.4732e-03         1.472
1.310543e-02   5.8396e-03       2.7913e-03         2.092
4.209151e-02   4.6600e-03       1.3861e-03         3.362
1.351879e-01   2.8065e-03       6.2327e-04         4.503
4.341816e-01   8.8611e-04       3.4218e-04         2.590
1.394134e+00   2.3460e-04       2.3120e-04         1.015
4.465840e+00   1.3393e-04       1.3392e-04         1.000
1.396375e+01   1.0819e-04       1.0819e-04         1.000
3.523227e+01   9.4516e-05       9.4515e-05         1.000
```

On this basis the low-frequency end (`s <= 1.3e-4`) carries the defect and the
certificate is essentially exact there (`1.003 .. 1.004`); the middle band
(`4e-4 <= s <= 4e-1`) has the smaller exact defect but a bound `1.1x .. 4.5x` loose;
the large shifts are again exactly tight with defects near `1e-4`.  The middle band
is known to be loose and, on the bases measured here, non-binding - up to the row
that pushes the rank past 50, where it becomes binding (C1, reading 2).

### C3. Superseded first version, and what is still missing

An earlier run of this script (`--points 3,6`, 8 cells, no witness) reported
`42` and `70` as "N_RHS".  Those were **operator blocks, not RHS solves** - the
underlying numbers were `168` and `280` RHS - and the point count was the requested
one rather than the selected one.  That table, and the `delta_cert = 1.2476e-01` of
the 8-cell run, are superseded by C1; the 8-cell number mixed the enclosure
slack of a coarse partition with a real saturation.

Still missing from the table: the two box-corrected stock rows on the same
`delta_cert` convention, and the same table at 2.5 mm.  The stock rows must be
produced by this script (`--stock-seeds`) so that the whole table shares one
counting convention, and the 2.5 mm stock counter must come from that run rather
than from the legacy-plan numbers in `README.md`.
