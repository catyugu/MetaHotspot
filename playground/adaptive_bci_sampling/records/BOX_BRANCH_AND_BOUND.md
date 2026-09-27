# Certified branch and bound over the HTC box

The continuous-stop statement is `forall Q: U_Q(V) <= tau_moment^2`, where `U_Q`
is the matrix cell certificate of records/MATRIX_CELL_CERTIFICATE.md.  This note
records the refinement loop that decides it without enrichment, and the numbers
that decide whether it beats a uniform partition.

## 1. Algorithm

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

## 2. Does it beat the uniform partition? (5 mm, design basis, s = 0)

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

## 3. Plan provenance: legacy bare-`K` versus box-corrected

Every number in section 2's table was produced with the **legacy bare-`K` plan**
(`shared_frequency_plan` encloses the generalized spectrum of `K` per source
port).  The delivered basis and `build_parametric_basis` use the **box-corrected
plan** instead (`box_spectral_interval` encloses the whole affine family by
Loewner monotonicity).  At 5 mm and tolerance `1e-3`:

```text
plan     shifts  lambda_min   lambda_max
legacy   11      4.4716e-04   3.9001e+01
box      13      4.2795e-05   4.1400e+01
```

The two plans are not interchangeable, so the benches now take
`--plan box|legacy` (default `box`), record the kind, interval and count in the
report, and print them.  The runs in section 2 are therefore a **legacy-plan
regression**: they are valid statements about that plan and say nothing about the
delivered one.

## 4. The delivered box plan, all thirteen shifts (threshold 2.5e-04)

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

## 5. Measurement cost: persistent leaves and a heap

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

## 6. Pitfall: a round cap looks like an anchor floor

With `max_rounds = 32` the same run stalls at `2.97e-03` with 36 leaves and the
last eight splits move it only from `3.4e-03` to `2.97e-03`.  That looks exactly
like an anchor-mismatch floor, and it is not one: the block anchor does not get
looser as the cell shrinks, so the plateau reads as a property of the anchor
rule.  Raising the cap to 400 rounds brings the same configuration to
`2.45e-04`, i.e. 12x lower, with the same 16 Gram matrices.

Rule: a branch-and-bound result only measures the refinement policy if the loop
terminated by exceeding its threshold.  Report `max_rounds` and `max_cells`
next to every number, and never read a capped run as a structural limit.

## 7. Cost

Per accepted run at 5 mm: one shared Gram matrix per anchor, i.e. `16 * 172 =
2752` large RHS solves for the **whole frequency plan** with the shift-free common
span (against `114` Grams, one set per shift, with the per-shift span), one
reduced solve per leaf for its reduced denominator, and reduced algebra for the
Bernstein enclosure.  No cell denominator solve, no AMG-CG solve, no full-order
solve outside the anchor Grams in certificate mode, and a whole-plan run costs
under ten seconds once the loop keeps its leaves.  That shared-span saving is
not free: see section 8 for the refinement-independent floor it introduces on
the large shifts.

## 8. The measured anchor floor of the shared shift-free span

The heap branch and bound stalls at a fixed bound on a few large shifts, and the
single-shot sweep of records/MATRIX_CELL_CERTIFICATE.md does not see that because
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
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_box_branch_and_bound.py 2.5 \
  --plan box --shifts 1 --span common --orders 2,3,4,5 --samples 2 --max-cells 400 \
  --max-rounds 500 --thresholds 3.5e-4 --floor-probe --output <path>.json
```

Reproduce (the plan sweep):

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_box_branch_and_bound.py 5 \
  --plan box --shifts all --span common --orders 2,3,4,5 --samples 2 \
  --max-cells 300 --max-rounds 400 --thresholds 2.5e-4 --output <path>.json
```

## 9. Limits

* Thresholds are demonstration levels, not adopted tolerances: `tau_moment` stays
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
  (section 8): the stall is the floor of the *shared shift-free* span, not of the
  spatial block anchors.  No certificate design is reopened for it.
