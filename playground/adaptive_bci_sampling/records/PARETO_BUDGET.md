# Extraction budget, snapshot compression and the certified matching defect

The dynamic bridge is not closed, so there is no vendor-derived stopping threshold
for the matching defect; what can be stated today is a Pareto table.  For every
combination of

```text
(parameter points requested, SVD cutoff)
```

the delivered design basis is rebuilt from the **same** cached snapshots on the
box-corrected plan - lowering the cutoff therefore costs no further full-order
solve - and certified shift by shift over the whole box.  Three quantities are
reported together, never separately:

```text
N_RHS -> sqrt(U*)   the certified bound, delta_cert = sqrt(max_j max_Q U_Qj),
N_RHS -> sqrt(L*)   the exact sampled witness, a 3x3 lattice per shift,
N_RHS -> dim V      the delivered ROM order.
```

## 0. Counting convention

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
separately; they are never added to `N_RHS`.  `tau_moment` stays symbolic.

Reproduce:

```text
PYTHONPATH=python python playground/adaptive_bci_sampling/bench_pareto_budget.py 5 \\
  --points 2,3,8 --cutoffs 1e-3,3e-4,1e-4,3e-5,1e-5 --cells 16 --order 3 \\
  --witness --witness-cells 2 --output <path>.json
```

## 1. The table, Case 1 at 5 mm, 16 cells per axis, order 3

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

## 2. Per-shift split (five points, cutoff 1e-3, 280 RHS)

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
that pushes the rank past 50, where it becomes binding (section 1, reading 2).

## 3. Superseded first version, and what is still missing

An earlier run of this script (`--points 3,6`, 8 cells, no witness) reported
`42` and `70` as "N_RHS".  Those were **operator blocks, not RHS solves** - the
underlying numbers were `168` and `280` RHS - and the point count was the requested
one rather than the selected one.  That table, and the `delta_cert = 1.2476e-01` of
the 8-cell run, are superseded by section 1; the 8-cell number mixed the enclosure
slack of a coarse partition with a real saturation.

Still missing from the table: the two box-corrected stock rows on the same
`delta_cert` convention, and the same table at 2.5 mm.  The stock rows must be
produced by this script (`--stock-seeds`) so that the whole table shares one
counting convention, and the 2.5 mm stock counter must come from that run rather
than from the legacy-plan numbers in `README.md`.
