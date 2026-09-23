# Fixed Zolotarev HTC tensors versus Extended FANTASTIC reproduction

## Setup

The reproduction is the repository's `build_parametric_basis` used by
`playground/bci_rom_testcase1/reproduce_case1.py`.  The other arm replaces its
random HTC loop with a fixed tensor product of finite-interval Zolotarev
nodes.  Everything else follows the reproduction: the same four individual
source ports, the same per-port elliptic frequency-shift formula, the same
`spd_solve` and requested extraction tolerance, the same normalized-snapshot
closing SVD cutoff, and the same preserved constant mode.  The fixed arm
includes its generalized spectral-envelope computation in extraction time.

The 2.5 mm Case 1 has 9072 cells and physical HTC ranges `[1, 1e4]` for both
groups.  At requested tolerance `1e-3`, each port needs 12 frequency shifts;
at `1e-4`, each port needs 15.  Thus an `m x m` HTC tensor requires
`4 * shifts_per_port * m**2` source RHS solves.  Reproduction adaptively
chooses a random number of full solves and then tests successive random HTCs.
Its candidate residual is a field residual, while the accuracy metrics below
are junction outputs.

The independent holdout contains all four physical HTC corners, the
reproduction point `(50, 1000)`, the physical log center, and six fixed-seed
log-uniform interior points.  Each case compares four independent *unit-power*
step responses.  Both full and reduced systems start at ambient and use the
same BDF1 steps `dt=50 s` through `2000 s`.  Direct sparse solves supply the
full-order reference.  The nominal power vector is `(0.1, 0.2, 0.3, 0.4) W`.
For each parameter and junction, the maximum error over sampled times is
divided by that same junction's **full-order steady rise**.  The entrywise
metric instead divides each source-to-junction step-response error by the
corresponding same-parameter steady transfer entry.  Neither metric uses a
full-field norm or a transient rise near zero as denominator.

The 12 cases are a deterministic *validation set*, not a continuous-box
certificate.  Independent unit steps also characterize arbitrary discrete
power histories by linear superposition, although the maxima in this note
refer only to the measured step histories.

## Same requested tolerance and same closing SVD cutoff

All errors below are dimensionless relative fractions; `step` is the worst
nominal-power junction history, `transfer` is the worst entry of the full 4x4
step-response matrix, and `steady` is the worst nominal-power steady error.
The wall time includes spectral preparation in each fixed Zolotarev row.

| Requested tolerance | Extraction | Full source RHS | ROM order | Extract s | Step | Transfer | Steady |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | reproduction seed 20260805 | 126 | 42 | 8.47 | `5.20e-5` | `1.76e-3` | `1.19e-3` |
| `1e-3` | reproduction seed 7 | 127 | 42 | 8.39 | `6.11e-5` | `1.92e-3` | `1.22e-3` |
| `1e-3` | Zolotarev 2x2 | 192 | 45 | 11.00 | `7.10e-5` | `1.89e-3` | `8.13e-4` |
| `1e-3` | Zolotarev 3x3 | 432 | 46 | 21.69 | `3.89e-5` | `1.78e-3` | `1.11e-3` |
| `1e-3` | Zolotarev 4x4 | 768 | 45 | 37.37 | `4.02e-5` | `1.59e-3` | `1.43e-3` |
| `1e-4` | reproduction seed 20260805 | 232 | 62 | 15.22 | `3.74e-6` | `1.81e-4` | `2.88e-5` |
| `1e-4` | reproduction seed 7 | 227 | 62 | 14.60 | `2.95e-6` | `2.89e-4` | `3.42e-5` |
| `1e-4` | Zolotarev 2x2 | 240 | 58 | 13.02 | `4.65e-5` | `7.25e-4` | `3.90e-4` |
| `1e-4` | Zolotarev 3x3 | 540 | 64 | 26.79 | `5.82e-6` | `1.56e-4` | `9.02e-5` |
| `1e-4` | Zolotarev 4x4 | 960 | 65 | 47.18 | `4.10e-6` | `1.67e-4` | `5.37e-5` |

At these two requested tolerances, every arm passes the *nominal step*
tolerance, but no arm passes it for *every transfer entry*.  In the
`1e-3` round, fixed 4x4 improves the worst transfer error only about 11%
against reproduction seed 20260805, while using 6.1 times the RHS and 4.4
times the extraction time.  At `1e-4`, fixed 3x3 has the smallest measured
transfer error, but takes 1.8 times the extraction time of that reproduction
seed.  These comparisons keep extraction and SVD tolerance identical;
they do not compare a raw steady snapshot span with a compressed dynamic ROM.

The most difficult step-response cases are often physical HTC corners.  The
nominal step and steady worst cases differ: after 2000 s a weakly cooled case
need not have reached its steady state, so a small finite-duration step error
does not imply a small steady error.

## Closing SVD diagnosis

To isolate truncation, the next experiment keeps the `1e-3` frequency-shift
plan and exactly the same fixed parameter samples and RHS solves.  It changes
only the fixed arm's closing SVD cutoff:

| Fixed nodes | SVD cutoff | RHS | ROM order | Nominal step | Transfer step | Nominal steady |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zolotarev 3x3 | `1e-3` | 432 | 46 | `3.89e-5` | `1.78e-3` | `1.11e-3` |
| Zolotarev 3x3 | `1e-4` | 432 | 64 | `6.41e-6` | `1.67e-4` | `8.01e-5` |
| Zolotarev 4x4 | `1e-3` | 768 | 45 | `4.02e-5` | `1.59e-3` | `1.43e-3` |
| Zolotarev 4x4 | `1e-4` | 768 | 66 | `3.12e-6` | `1.38e-4` | `4.07e-5` |
| Zolotarev 4x4 | `1e-5` | 768 | 86 | `3.51e-7` | `4.33e-6` | `1.97e-6` |

This confirms that closing SVD, rather than missing tensor points alone,
controls much of the reported error.  With a `1e-3` SVD cutoff, adding
parameters from 3x3 to 4x4 even decreases order 46 to 45.  A tighter
cutoff retains more information and recovers the expected accuracy, at a
substantial order and extraction-cost premium.  The diagnostic changes the
closing threshold, so it is **not** part of the like-for-like table above.

## 1 mm extraction-only check

On the original reproduction resolution of 122400 cells, a separate run at
`1e-3` measured reproduction seed 20260805 at **159 RHS, order 52, 193 s**.
Fixed Zolotarev 2x2 took **224 RHS, order 54, 427 s**, of which **236 s** was
spectral preparation.  This run was interrupted during an exceptionally slow
full-order transient corner reference; no 1 mm transient errors are claimed.
The previous independent spectral-only runs on the same mesh took about
91--99 s, so elapsed times vary materially across runs.  Only the timings
from the same 1 mm comparison run should be divided to form the 2.2x ratio.

## Reproduce

```text
python playground/adaptive_bci_sampling/test_compare_transient.py
python playground/adaptive_bci_sampling/test_zolotarev.py
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 --counts 2 3 4 --seeds 20260805 7 \
  --random-holdout 6 --dt 50 --duration 2000 \
  --output playground/adaptive_bci_sampling/transient_comparison_2p5.json
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-4 --counts 2 3 4 --seeds 20260805 7 \
  --random-holdout 6 --dt 50 --duration 2000 \
  --output playground/adaptive_bci_sampling/transient_comparison_2p5_tol1e4.json
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 --counts 3 4 --seeds \
  --fixed-closing-tolerance 1e-4 --random-holdout 6 \
  --output playground/adaptive_bci_sampling/transient_comparison_svd_diagnostic.json
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 --counts 4 --seeds \
  --fixed-closing-tolerance 1e-5 --random-holdout 6 \
  --output playground/adaptive_bci_sampling/transient_comparison_svd_1e5.json
```

Generated JSON (ignored by Git) includes per-metric worst parameter locations and the nominal
step's worst time.  A full source RHS count is the number of computed
snapshots; cheap random residual probes are recorded separately for the
reproduction.  Validation time is recorded separately and is excluded from
both extraction columns.

## Decision

Fixed tensor Zolotarev HTC sampling does **not** improve the current full
Extended FANTASTIC extraction on this case at the same frequency and SVD
tolerance.  It removes randomness, but it buys few modes of practical
transient accuracy for many more full solves.  The useful next algorithmic
change is an output-aware stopping and SVD retention rule that checks the
individual source-to-junction transfer responses; steady-only rational
placement does not supply a transient output guarantee.  Any continuous-box
guarantee additionally needs a certified maximization over HTC and a
time/frequency-domain error bound.
