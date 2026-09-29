# The fixed-shift collocated certificate, its refinement, and the extraction budget

Everything in this record is `[PORT-FIXED-S]`: the certified quantity is
`lambda_max(E(h), Y(h))`, the all-input collocated defect of the delivered basis at one
real shift, with `E(h) = R(h)^T A(h)^-1 R(h)`, `Y(h) = G^T A(h)^-1 G`, `A(h) = K + s C
+ sum_i h_i H_i`.  That quantity is the **fixed-shift realization of the field-level
acceptance criterion**: by `THEORY.md` proposition 1 it equals the squared relative
`A`-energy error of the delivered full-field transfer operator `X(h) = A(h)^-1 G`, so
no square root is taken anywhere here and the state amplitude is reported by the
field-level metrics of `certify_extraction.py`.  The step from one real shift to the
port system norm is the open `[PORT-SYSTEM]` gap and lives in
`records/FAILURE_ARCHIVE.md`, not here.

* **Part A** - at one fixed real shift the delivered basis is certified over the **whole**
  HTC box and **every** input combination, and at a sufficient partition the statement
  is tight.
* **Part B** - the continuous-box statement can be decided by branch and bound without
  enrichment, and the shared anchor span makes it cheap.
* **Part C** - the same certificate closes the `N_FOM`-versus-accuracy trade-off into a
  Pareto table.

> **Plan provenance.**  Numbers marked `legacy plan` were produced with the original
> bare-`K` frequency plan (`shared_frequency_plan`: 11 shifts on
> `lambda in [4.4716e-04, 3.9001e+01]` at 5 mm, 12 at 2.5 mm).  A later, now reverted
> implementation used a box-corrected plan (13 shifts on
> `lambda in [4.2795e-05, 4.1400e+01]` at 5 mm, 14 at 2.5 mm).  The benches take
> `--plan box|legacy`; a certificate for a given delivered basis is valid under either
> plan, because it never uses how the shifts were selected.

---

## Part A - the matrix cell certificate: validity, tightness and the partition setting

### A1. Statement

On a cell `Q` with lower corner `a` and upper corner `b`,

```text
sup_{h in Q} lambda_max(E(h), Y(h)) <= max_nu lambda_max(Xi_nu^T S_a Xi_nu, Y(b)),
S_a = Z^T A(a)^-1 Z,     Z = [G, (K + sC)V, H_1 V, ..., H_d V],
```

from three exact facts: `H_i >= 0` gives `A(h) >= A(a)` hence `A(h)^-1 <= A(a)^-1`
(Loewner); Galerkin optimality gives `E(h) <= R_q(h)^T A(h)^-1 R_q(h)` for *any* trial
`q`, so a Taylor jet of the reduced solve is allowed to be bad without breaking the
bound; and `X -> X^T S_a X` is matrix convex, so the tensor Bernstein enclosure
`R_q = Z Xi(h) = Z sum_nu beta_nu(h) Xi_nu` with `beta_nu >= 0`, `sum_nu beta_nu = 1`
bounds the quadratic form by the largest coefficient.  The denominator uses the *upper*
corner, which lower-bounds `Y` on the whole cell because the family is entrywise
decreasing in `h`.  No per-entry denominator and no entrywise maximum over
coefficients enter, so weak coupling cannot blow the statement up.

### A2. Falsification against the exact map (Case 1, 5 mm, design basis, 42 vectors, s = 0)

650-point box reference from the exact map: `1.824841e-04`.  Uniform log partitions,
zero Loewner violations at every configuration:

```text
cells/axis  anchor  order   bound        bound/exact  median
8           local   2       5.860e-02    3.21e+02     158
16          block   3       3.432e-04    1.88e+00     1.2
16          local   3       1.8396e-04   1.008e+00    1.17
```

At 16 cells per axis with jet order 3 the bound is `1.8396e-04` against the exact
`1.8248e-04`, i.e. 0.8 % above the true box maximum, and the dominating cells are at
`1.06x` (local anchor) and `1.06x`-`1.62x` (block anchor, orders 2/3), inside the
pre-registered 10x gate.  Refinement works in both directions and they are not
interchangeable: raising the jet order at a fixed partition buys about 4x per order at
8 cells/axis (trial enclosure is a real loss), and refining the partition buys more but
gives each new cell its own Riesz Gram (130 sparse solves at 5 mm).  Block anchors
(16 Gram matrices for a 4x4 log block structure, reused by 256 cells) reach `1.88x` at
order 3, so the cheapest passing configuration does not need a Gram per cell.

The same configuration certifies the stock extractor (seed 20260805, 36 vectors,
137 right-hand sides): exact box maximum `1.0905e-04`, bound `1.125e-04` (`1.025x`)
with cell-local anchors and `3.433e-04` (`3.13x`) with shared block anchors, zero
violations.  Its exact maximum is *smaller* than the design basis's `1.8248e-04` at this
shift: the deterministic design buys the continuous statement, not raw accuracy.

### A3. Nonzero shifts, denser sampling, a second mesh, trial portfolio

The steady shift is the binding one.  Same 16x16 partition, order 3:

```text
operator                          exact box max  bound (block)  bound (local)  violations
s = 0                             1.8248e-04     3.4317e-04     1.8396e-04     0
s = plan shifts[5] = 1.3206e-01   2.8403e-07     1.9703e-05     8.0723e-06     0
s = plan shifts[0] = 3.3248e+01    5.7470e-09     5.7473e-09     5.7471e-09     0
```

The defect falls monotonically in the shift, so the HTC dependence weakens with real
frequency and certifying `s = 0` tightly is what the plan costs.  A stopping rule keyed
on the *dominating* cells reduces to a fine certification near the steady operator; a
rule keyed on the largest bound over all cells would be dominated by cells certified
far from their own maximum, which is why the greedy keeps the exact witness `L` separate
from the certificate `U`.  Denser sampling (11 points per axis inside each cell, about
`3.1e4` exact evaluations) leaves the `s = 0` numbers unchanged and still reports zero
violations.

Second mesh, 2.5 mm (9072 dof), 8 cells/axis, order 3: exact box maximum `3.4798e-04`,
bound `1.5587e-02` (44.8x) with local anchors and `4.8373e-02` (139x) with 4x4 block
anchors, zero violations, dominating cells at `6.81x` - close to but not inside the 10x
gate, and `p`-refinement alone does not converge there (`3.8x` per order with no
saturation, and the cell carrying `U*` has a true defect 24x below `L*` while its own
bound is still `234x` above it): the binding effect is a too-wide cell, so `h`-refinement
is the answer and adaptive splitting beats a uniform 8x8 -> 16x16 refinement.

Trial portfolio at 16 cells/axis, 4x4 block anchors, exact box maximum `1.824841e-04`:
`taylor` gives `4.449e-03 / 3.432e-04 / 1.8396e-04` at `p = 2/3/4`, `chebyshev`
`7.795e-03 / 8.501e-04 / 1.8396e-04`, and every trial converges to the same `1.8396e-04`
from order 4 on: the remaining 0.8 % is the enclosure and denominator gap of the cell
that carries the maximum, not trial error.  Which trial wins is not universal, which is
exactly why the portfolio (min over trials) is used; it costs no Gram matrix and takes
the dominating-cell effectivity from `1.623x` to `1.058x` at `p = 2`.

### A4. Basis discrimination on the delivered box plan (16 cells/axis, order 3)

```text
basis                    dim V   exact grid   certified    cert/exact
deterministic (design)      41    3.3471e-04   3.3671e-04   1.006
stock_20260805              36    1.0861e-04   1.1252e-04   1.036
stock_legacy_20260805       35    9.8431e-04   9.9328e-04   1.009
stock_7                     37    1.6382e-04   1.6512e-04   1.008
stock_legacy_7              36    1.3102e-03   1.3260e-03   1.012
```

The certificate is `1.006`-`1.036` tight on **all five** delivered bases, so tightness
is not an artifact of the design basis.  On the box plan the deterministic design is
about 3x worse than the stock extractor at the steady shift, and both legacy-plan bases
are a further order of magnitude out.  **A certificate number is only interpretable next
to the partition it was computed on** (a run at 8 cells/axis returns `1.5566e-02` for
every good basis while the same bases span `1.09e-04 .. 3.35e-04` exactly), which is why
the partition setting is part of the statement and not a benchmark knob.

### A5. Cost, pitfall, limits

Certificate cost, per shift, no AMG-CG extraction solve added and no full-order cell
denominator (the cell denominators are the *reduced* Galerkin transfers `Y_V(b)`, small
dense solves in the basis):

```text
configuration                 Grams  Gram solves  cell denominators  bound solves
16 cells/axis, local anchor    256    33280        0                  0
16 cells/axis, block anchor    16     2080         0                  0
```

The matrix path must not clip the coefficient matrices entrywise to non-negativity:
that is correct for the entrywise bound (only diagonals are used) but it changes the
Loewner order.  With the clip a degenerate cell reported `1.7039e-04` against an exact
`1.8248e-04` (a 6.6 % violation); without it the same cell reproduces the exact value to
13 digits (`1.824840729e-04` vs `1.824840732e-04`).

Limits: inequalities are exact in real arithmetic, while the Gram matrices,
factorizations and small dense solves are ordinary floating point
(`floating_point_certified=False`).  Falsification samples cell interiors, so a
violation count of zero is "no counterexample found on the sampled points", not a proof.
"The steady shift is the binding one" is an observation on the shifts measured here, not
a theorem: for a fixed `V` both the numerator `E(s)` and the denominator `Y(s)` move with
the shift and their ratio need not be monotone; a finite-plan certificate still has to
cover every matching shift (the *frequency axis* needs no continuity because the plan is
finite).  The refinement chain is not asserted to be monotone in the trial, because
`U_Q(W_n)` is computed from a fresh trial at each space; the clean fix, not implemented,
is to carry the previous trial forward as one more portfolio candidate.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py tightness 5 --cells 8 --orders 2,3,4 --samples 5 --reference-cells 24 --output <path>.json
```

---

## Part B - deciding the continuous box without enrichment

### B1. Algorithm and its cost

`BoxCertificate.branch_and_bound(threshold, ...)` refines cheapest first: a leaf tries the
trial orders in turn at its fixed cell and anchor (p-refinement, no new Gram matrix) and
stops at the first order below the threshold, so the order varies per leaf; a leaf still
above the threshold is h-refined by log-bisecting the widest axis of the leaf carrying the
largest bound, one split per round; the anchor is the shared 4x4 block anchor, so any
number of leaves costs at most 16 Gram matrices.  Correctness and stopping use `U` alone;
the exact witness `L` at sampled points of a leaf (one full-order solve each) is behind
`oracle=True` and exists to study refinement and locate enrichment witnesses - a
certificate that did full-order work on every leaf would already have spent more than the
enrichment it is meant to justify.

Cost per accepted run at 5 mm with the shift-free common span: 16 Gram matrices, i.e.
`16 * 172 = 2752` large RHS solves for the **whole** frequency plan, one reduced solve per
leaf for its denominator, and reduced algebra for the Bernstein enclosure.  The first loop
re-measured every leaf in every round while splitting one leaf per round (`84` splits over
a growing leaf set, about `3900` cell evaluations, 42.7 s); keeping leaves in a max-heap
keyed by their bound gives identical bounds and leaf counts with `154`/`172` measures and
1.6 s, i.e. `18x`-`27x` less wall clock.  **A branch-and-bound result only measures the
refinement policy if the loop terminated by exceeding its threshold**: with
`max_rounds = 32` the same run stalls at `2.97e-03` and looks exactly like an
anchor-mismatch floor; raising the cap to 400 rounds reaches `2.45e-04` with the same 16
Gram matrices.

### B2. Does it beat the uniform partition? (5 mm, design basis, s = 0, legacy plan)

Exact box maximum `L* = 1.824841e-04` on 650 reference points.

```text
scheme                                  leaves  Grams  bound      U*/L*
adaptive B&B, threshold 2.5e-04         88      16     2.4543e-04  1.35
uniform 16 cells/axis, order 4          256     16     1.8396e-04  1.008
```

The adaptive loop reaches `2.5e-04` with 88 leaves against the uniform partition's 256
cells at the same 16 Gram matrices: the value of B&B here is a `2.9x` smaller cell count
for a comparably tight certificate, not a bound the uniform partition cannot reach at all.
It ends looser only because it stops at the threshold instead of pushing the bound to its
floor.  No anchor upgrade is needed on this mesh.

### B3. The delivered box plan, all thirteen shifts

Design basis rebuilt on the box plan (order 42, exact box maximum `1.763102e-04` at
`s = 0`), certificate mode, 4x4 block anchors, threshold `2.5e-04`: the whole plan
certifies in about nine seconds of certificate work, with the low-frequency end the
expensive part (leaf counts `4, 4, 4, 4, 6, 12, 24, 24, 30, 45, 54, 63, 71` from the
largest to the smallest shift).  The shift-free common span
`Z = [G, K V, C V, H_1 V, ...]` certifies every shift from a single anchor Gram
`Z^T A(a,0)^-1 Z`, because `A(h, s) >= A(a, 0)` for every `s >= 0` and the shift enters
the residual only through the coefficient `s`: **16 Gram matrices for the entire plan
instead of 114**, with the same leaf counts to within `3` and the same runtime.

That sharing is correct but not free.  For 2.5 mm and the plan's second largest shift
`s = 4.3517e+01` the worst leaf after 198 splits has bound `3.8026125e-04`, and the exact
parameter value of the same cell form (the anchor floor) is
`3.8026111e-04 .. 3.8026112e-04` - the plateau *is* the floor of the shift-free span, to
seven digits, while the same leaf under the per-shift span has no floor at all
(`3.3581499e-07`).  The mechanism is the sharing itself: the common span takes its Riesz
Gram from `A(a, 0)`, and for `s = 4.35e+01` the shift term dominates `A(a, s)`, so
`A(a, 0)^-1` overstates the residual energy the certificate may charge.  The floor is
independent of the partition, so more splits never help, and any threshold below it is
unreachable **with that span**; the per-shift span has no such floor, at 114 Grams for the
plan instead of 16.

Limits: thresholds are demonstration levels, not adopted tolerances - `tau_port` stays
symbolic until the dynamic bridge of `THEORY.md` P0 fixes it.  Because `U` is an upper
bound, no threshold below `L*` can be accepted, and `L*` is a sampled reference rather
than a continuous maximum.  Certificate mode is evidence about the algorithm, not about
the exact defect: without `oracle=True` the run reports no witness.  Reported for one mesh
(5 mm) except where 2.5 mm is named.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py bandb 2.5 --plan box --shifts 1 --span common --orders 2,3,4,5 --samples 2 --max-cells 400 --max-rounds 500 --thresholds 3.5e-4 --floor-probe --output <path>.json
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_certificate.py bandb 5 --plan box --shifts all --span common --orders 2,3,4,5 --samples 2 --max-cells 300 --max-rounds 400 --thresholds 2.5e-4 --output <path>.json
```

---

## Part C - the extraction budget: N_FOM against the certified defect

The dynamic bridge is not closed, so no vendor-derived stopping threshold exists for the
matching defect; what can be stated today is a Pareto table.  For every combination of
(parameter points requested, SVD cutoff) the delivered design basis is rebuilt from the
**same** cached snapshots, so lowering the cutoff costs no further full-order solve, and
certified shift by shift over the whole box.  Four quantities are reported together,
never separately: `N_RHS` (full-order inverse actions), `D_cert` (max over shifts and
cells of the certified value), `D_wit` (max over sampled shift/point of the same quantity)
and `dim(V)`.

Counting convention: one *operator block* is one factorization of `A(h, s)` and every
block serves one right-hand side per port, so
`N_RHS = number_of_ports * (selection_blocks + build_blocks)`, `ports = 4` here.  Only
`N_RHS` may be compared with the stock extractor's counter (137/138 at 5 mm box-corrected
already use this convention; the three-point deterministic design is
`(13 + 1) * 3 * 4 = 168` RHS, which this bench reproduces).  The selection's residual
certificate and the box certificate itself are certification overhead (256 Gram matrices
per shift here) and are never added to `N_RHS`.  `tau_port` stays symbolic.

Case 1 at 5 mm, 16 cells per axis, order 3; `fresh` is what that row cost and `cumulative`
the total spent so far.  Requesting 8 points selects 5, because the greedy scorer stops at
its own `1e-5` tolerance.

```text
points  cutoff  fresh RHS  cum RHS  dim V  D_cert       D_wit        U*/L*
2       1e-03   112        112      39     3.9372e-03   3.8930e-03   1.0057
2       1e-05   0          112      63     1.1243e-03   1.1142e-03   1.0045
3       1e-03   56         168      42     1.7776e-04   1.7631e-04   1.0041
5       1e-03   112        280      42     1.5863e-04   1.5733e-04   1.0041
```

`D_cert` and `D_wit` are the squares of the `sqrt(U*)`/`sqrt(L*)` columns of the first
version of this table, which are the numbers the earlier prose quoted (`1.2595e-02` at
five points and cutoff `1e-3` is `sqrt(1.5863e-04)`); a monotone transform does not move a
plateau or a ratio, so the three readings below are unchanged.

Three readings, each a different verdict:

1. **Lowering the cutoff is free in full-order solves and is not free in rank.** From
   `1e-3` to `1e-5` the fresh cost of every further row is exactly zero (all snapshots are
   already cached) while the delivered order rises from 39 to 63 (two points), 42 to 75
   (three) and 42 to 80 (five).  The `1e-3` truncation was discarding most of the
   information the extraction had already paid for.
2. **The exact defect follows the rank, the certified bound does not.** The witness falls
   by more than an order of magnitude (five points: `1.254e-02` to `4.225e-04` on the
   `sqrt` scale) while `D_cert` sticks at `7.9100e-03` in every row whose rank exceeds
   about 50.  That plateau is neither the basis nor the plan: it is the order-3 cell
   enclosure of one specific middle-band `(shift, cell)`, and the effectivity `U*/L*`
   rises from `1.004` to `18.7` as the true defect leaves it behind.  **The certificate's
   middle band is the binding limit on what can be stated**, and only the dominating
   `(shift, cell)` needs a higher trial order or a portfolio, not the whole plan.
3. **Under the stock gate the two-point family is the only option, and it does not reach
   the three-point accuracy.** `112 < 137 < 168` RHS: the best two-point row gives
   `sqrt(L*) = 3.34e-02` against `1.33e-02` for three points at `1e-3` (168 RHS), a factor
   2.5.  Compression alone recovers a factor 1.9 within the two-point budget
   (`6.27e-02` to `3.35e-02`) and then saturates, so under the gate the remaining lever is
   not the cutoff but *which* points are chosen.

Per-shift split (five points, cutoff `1e-3`, 280 RHS): the low-frequency end
(`s <= 1.3e-4`) carries the defect with an essentially exact certificate (`1.003 .. 1.004`),
the middle band (`4e-4 <= s <= 4e-1`) has the smaller exact defect but a bound
`1.1x .. 4.5x` loose, and the large shifts are again exactly tight with defects near
`1e-4`.  The middle band is non-binding up to the row that pushes the rank past 50.

Certificate cost stays outside all of this (256 anchors and 50-190 s per row at 16 cells
per axis, a research-mode price).  An earlier run of this script reported `42` and `70`
as "N_RHS"; those were operator blocks, not right-hand sides (the underlying numbers were
`168` and `280`), and its `delta_cert = 1.2476e-01` mixed the enclosure slack of a coarse
8-cell partition with a real saturation.  Still missing from the table: the two
box-corrected stock rows, and the same table at 2.5 mm.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_pareto_budget.py 5 --points 2,3,8 --cutoffs 1e-3,3e-4,1e-4,3e-5,1e-5 --cells 16 --order 3 --witness --witness-cells 2 --output <path>.json
```
