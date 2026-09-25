# Original two-HTC Case 1: frequency-dependent parameter samples

## Controlled experiment

This is the **original Case 1 with two boundary groups**, die crowns and FR4
bottom, on the 2.5 mm, 9,072-cell grid. The reference extractor uses the
Extended FANTASTIC random HTC loop with 10 consecutive passing probes; both
arms use the *same four-source, 12-shift-per-source elliptic frequency plan*,
the unmodified unit-column snapshot SVD at **1e-3**, and the constant mode.
No field error is used for selection or validation.

For each parameter, compare all 16 heat-source/junction BDF1 step-transfer
entries during 40 steps of 50 s and all 16 steady-transfer entries; normalize
each by the exact steady rise for that entry **at the same parameter**. The
holdout has all four physical HTC corners, the reproduction parameter,
geometric center, and 64 log-random pairs, totaling 70. `--grid 17` adds a
17-by-17 deterministic physical-HTC log grid to these same 70 parameters.

## Parameter and frequency rule

The degree-one finite-interval Zolotarev nodes yield the effective-HTC seed
`(346.0327384, 14.5263339)`. The bottom exposed edge at the largest die-crown
HTC contributes its two vertices:

```text
(9285.7142857, 0.9958506), (9285.7142857, 234.375)
```

For each heat source, take these **three** points at the four original
elliptic shifts `s <= 1/dt = 0.02 s^-1`, and only the seed at the eight
higher shifts: `4 sources * (4*3 + 8*1) = 80` full-order RHS solves. An
additional exact `s=0` snapshot at each of the three points costs 12 solves
(92 total). A control adding DC only at the seed costs four (84 total).
Neither the elliptic shift positions nor the SVD threshold is modified.

For the original two-group model, at this seed the largest source-output
scaled mixed HTC derivative `p_top*p_bottom*|d²Y/dp_top dp_bottom|` falls from
`0.91283` at `s=0`, through `0.16312` at `s=0.001`, to `3.2276e-5` at
`s=0.02`. With the thermal M-matrix inverse, nonnegative source/diagonal
boundary terms and nonnegative diagonal capacity, **every entry of every
absolute raw mixed HTC derivative is nonincreasing in frequency**. The
derivation and its post-SVD limitations are in
[FREQUENCY_DEPENDENT_HTC_SAMPLING.md](FREQUENCY_DEPENDENT_HTC_SAMPLING.md).
The exposed edge is a finite geometric sampling rule, not an established
two-dimensional minimax Zolotarev rule.

## Same-run results on 70 parameters

| Method | Full RHS solves | ROM order | Extraction including selection (s) | Worst step entry | Worst steady entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three points at low shifts, seed at high shifts | **80** | 45 | **2.83** | **7.9102e-4** | 3.4789e-3 |
| Same plus DC at seed only | 84 | 46 | 2.97 | 8.2083e-4 | 2.8655e-3 |
| **Same plus DC at all three points** | **92** | **47** | **3.19** | **5.2323e-4** | **1.9112e-4** |
| Random stock, seed 20260805 | 126 | 42 | 3.99 | 1.7569e-3 | 6.0915e-3 |
| Random stock, seed 7 | 127 | 42 | 4.00 | 1.9243e-3 | 6.7696e-3 |
| Tangent-selected three points at *all* shifts | 144 | 48 | 4.80 | 6.1893e-4 | 3.0796e-3 |

The 92-solve design uses 27% fewer full RHS solves than the first random
seed and has about 3.4 times lower worst step error and 32 times lower
worst steady error on this validation set. It extracts faster on this run;
its ROM has five more modes. The 80-solve version already meets the 1e-3
**40-step** criterion but misses the 1e-3 **steady** criterion. Adding just
the seed at DC does not bring steady below 1e-3. These figures are observed
errors on finite parameter sets, not rigorous bounds over the whole HTC box.

## Dense two-dimensional check: 70 parameters plus 17-by-17 log grid

The additional grid gives **359 evaluations** (four vertices appear in both
sets). It identifies errors missed by the 70-point set. All results below
come from a fresh *same-run* extraction and validation; times vary slightly
between runs.

| Method | RHS | Order | Extraction (s) | Worst step entry | Worst steady entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-point low-frequency face | 80 | 45 | 2.76 | 8.2295e-4 | 4.3345e-3 |
| Face + seed DC only | 84 | 46 | 2.82 | **1.0378e-3** | 3.6608e-3 |
| **Face + all three DC snapshots** | **92** | **47** | **2.99** | **5.4916e-4** | **8.2245e-4** |
| Random stock, seed 20260805 | 126 | 42 | 3.80 | 1.7569e-3 | 6.0915e-3 |
| Random stock, seed 7 | 127 | 42 | 3.52 | 1.9243e-3 | 6.7696e-3 |
| Three tangent-ranked points at all shifts | 144 | 48 | 4.35 | 9.3908e-4 | 3.9497e-3 |

The **84-solve** version, seemingly good on 70 points, actually **misses
1e-3 transient tolerance** at physical HTCs `(10000, 17.7828)` on the
grid. The 92-solve version's largest step error occurs at
`(1778.2794, 10000)` and its largest steady error at `(10000, 17.7828)`.
It uses 27% fewer full-order RHS solves and 21% less measured extraction
time than stock seed 20260805, while its ROM retains five extra modes. The
nearby continuum has not been bounded: an output-oriented a posteriori
bound is needed for a continuous-box claim.

To probe the grid's most sensitive directions, we also tested **65
log-spaced points on each of the two high-HTC edges**, together with the 70
original holdout points. On these 200 evaluations, the 92-solve version's
worst step entry was **5.5055e-4** at physical HTCs
`(2053.5250, 10000)` and its worst steady entry was **8.3450e-4** at
`(10000, 20.5353)`. This edge audit strengthens the empirical observation,
but the continuous box is still not certified. The 84-solve control instead
reached `1.0454e-3` step error at `(10000, 20.5353)`.

## Reproduction

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_two_group_frequency_faces.py --output /tmp/bci-two-group-frequency-face-results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_two_group_frequency_faces.py --grid 17 --output /tmp/bci-two-group-frequency-face-grid17.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_two_group_frequency_faces.py --edge-grid 65 --output /tmp/bci-two-group-frequency-face-edge65.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_mixed_frequency.py --original-two-groups
```

Generated numerical JSON stays outside the repository. The experiments
neither change the production extractor nor its SVD truncation rule.

The corresponding 122,400-cell, 1 mm experiment and the full derivation of
the three-point ranking are in
[ONE_MM_FREQUENCY_FACES.md](ONE_MM_FREQUENCY_FACES.md).
