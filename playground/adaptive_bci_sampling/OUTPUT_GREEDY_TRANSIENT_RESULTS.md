# Steady output greedy selection in a dynamic Extended FANTASTIC extraction

## Question and controlled experiment

Does the steady, Zolotarev-seeded **junction-output certificate** from
[ZOLOTAREV_RESULTS.md](ZOLOTAREV_RESULTS.md) improve the *dynamic* ROM at the
same extraction tolerance and with the same test metric as the reproduced
Extended FANTASTIC extractor? The newly advanced `agent/work` experiment
reports small **steady** ROMs, but its result cannot be compared directly to a
four-port, frequency-shifted, compressed ROM's step response.

At each tolerance, select HTC parameters using the maximum corrected
primal-dual relative certificate of all 16 steady junction transfer entries
on a 9x9 log-spaced grid in the effective HTC box. The fixed starting parameter
is the 1x1 Zolotarev tensor point. Full-order selection snapshots use **all
four individual source vectors**. No closing SVD is used when scoring the
steady selection basis. The corrected relative bound is
`Delta / (abs(Yhat_ij) - Delta)`, with `inf` for each entry with a nonpositive
denominator. This selects five parameters at both tolerances; the completed
selection certificate is `3.66e-6` on the candidate grid.

For every selected parameter, solve **the same 12 or 15 elliptic frequency
shifts per heat source** as the reproduction, using the same `spd_solve` and
the same normalized closing SVD. The input dimension is four in every arm.
Reevaluate the *steady* certificate on the actual final compressed dynamic
ROM; this matters because the selection certificate applies to a different,
uncompressed steady space. Count the 20 steady source solves for parameter
selection **in addition to** the 240 or 300 frequency-shifted source solves.
The spectral precomputation, selection, full solves, SVD and final steady
certificate audit are all included in the output arm's extraction time.

The 2.5 mm Case 1 has 9072 cells and physical HTC ranges `[1, 10000]^2`.
Each of the same 12 independent holdout parameters contains four corners,
`(50,1000)`, `(100,100)`, and six fixed log-uniform points. BDF1 runs
`dt=50 s` through `2000 s` from zero initial rise. The *step* metric is
the nominal-power junction rise error normalized by each junction's own
same-parameter exact steady rise. The *transfer* metric instead measures
every one of the 16 source-to-junction step entries and normalizes each by
its **own** same-parameter exact steady transfer entry. This is stricter
than dividing all entries by the largest transfer entry. Exact full-order
step and steady solutions use sparse direct solves. Every arm in the following
table shares those reference solutions. Validation time is excluded from
extraction time.

## Same extraction and closing SVD tolerance

`RHS` is the number of full-order **individual source** solves. This is a
same-run comparison; timings can vary across independent runs. `Steady`
and `steady transfer` measure errors at the steady state, which is not
necessarily reached by the end of the 2000 s step window.

| Tolerance | Parameter sampling | RHS | ROM order | Extract s | Step | Transfer | Steady transfer |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | Stock seed 20260805 | 126 | 42 | 9.22 | `5.20e-5` | `1.757e-3` | `6.092e-3` |
| `1e-3` | Stock seed 7 | 127 | 42 | 8.63 | `6.11e-5` | `1.924e-3` | `6.770e-3` |
| `1e-3` | Fixed Zolotarev 2x2 | 192 | 45 | 11.68 | `7.10e-5` | `1.888e-3` | `5.540e-3` |
| `1e-3` | Fixed Zolotarev 3x3 | 432 | 46 | 22.76 | `3.89e-5` | `1.776e-3` | `2.026e-3` |
| `1e-3` | Field-residual greedy 1x1 | 140 | 44 | 11.30 | `4.83e-5` | `1.437e-3` | `1.369e-3` |
| `1e-3` | **Steady output greedy 1x1** | **260 (20+240)** | **47** | **18.95** | **`4.84e-5`** | **`9.719e-4`** | **`2.959e-3`** |
| `1e-4` | Stock seed 20260805 | 232 | 62 | 16.51 | `3.74e-6` | `1.805e-4` | `3.138e-4` |
| `1e-4` | Stock seed 7 | 227 | 62 | 14.99 | `2.95e-6` | `2.892e-4` | `7.848e-4` |
| `1e-4` | Fixed Zolotarev 2x2 | 240 | 58 | 13.31 | `4.65e-5` | `7.253e-4` | `7.551e-4` |
| `1e-4` | Fixed Zolotarev 3x3 | 540 | 64 | 28.19 | `5.82e-6` | `1.557e-4` | `3.032e-4` |
| `1e-4` | Field-residual greedy 1x1 | 222 | 65 | 18.67 | `3.61e-6` | `1.259e-4` | `2.403e-4` |
| `1e-4` | **Steady output greedy 1x1** | **320 (20+300)** | **70** | **20.86** | **`4.44e-6`** | **`5.812e-5`** | **`1.427e-4`** |

The output-guided arm has the smallest *measured step transfer* error at
both tolerances and is the only same-cutoff arm passing every tested step
transfer at both levels on this 9x9 selection run. However, the `1e-3`
result has almost no margin, consumes roughly twice the stock RHS and time,
and fails the independent steady-transfer tolerance by almost a factor of
three. At `1e-4` it still fails the steady-transfer tolerance. A finite
2000 s step check cannot replace a steady-output guarantee.

## Does the final ROM retain the selection certificate?

| Extraction tolerance | Candidate grid | Selection certificate on raw steady space | Final steady grid certificate after SVD | Final unresolved candidates |
| ---: | ---: | ---: | ---: | ---: |
| `1e-3` | 9x9 | `3.66e-6` | `4.91e-3` | 50/81 |
| `1e-4` | 9x9 | `3.66e-6` | `1.83e-4` | 21/81 |
| `1e-3` | 17x17 | `3.81e-6` | `5.36e-3` | 183/289 |

The same 17x17 run selects five points, costs 260 RHS, and gives order 47.
Its worst observed step transfer is **`1.003e-3`**, just above the `1e-3`
target, whereas the 9x9 result is `9.719e-4`. Thus even the *observed*
passing result is sensitive to changing the candidate grid. The final
steady certificate also fails in every equal-SVD-cutoff row. The raw
steady selection certificate is not a certificate for the compressed
frequency-shifted ROM.

## Separately diagnosed closing SVD

The following output-greedy runs retain the same steady selection and
frequency-shift plan but deliberately use a *tighter* closing SVD than the
extraction tolerance. They are not same-cutoff comparisons against the
stock extractor. The steady certificate is evaluated **after** the SVD on
the final ROM, on the 9x9 candidate grid.

| Extraction target | SVD cutoff | RHS | ROM order | Extract s | Final steady grid certificate | Step transfer | Steady transfer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | `1e-3` | 260 | 47 | 18.95 | `4.91e-3` | `9.72e-4` | `2.96e-3` |
| `1e-3` | `1e-4` | 260 | 70 | 20.10 | `1.64e-4` | `7.04e-5` | `1.34e-4` |
| `1e-3` | `1e-5` | 260 | 90 | 20.31 | `6.06e-6` | `2.86e-6` | `3.94e-6` |
| `1e-4` | `1e-4` | 320 | 70 | 20.86 | `1.83e-4` | `5.81e-5` | `1.43e-4` |
| `1e-4` | `1e-5` | 320 | 96 | 23.59 | `5.06e-6` | `1.63e-6` | `2.33e-6` |

At extraction target `1e-3`, a `1e-4` closing cutoff is sufficient to
make the final *grid* steady certificate and both measured transfers pass,
but raises order 47 to 70. At `1e-4`, the tested `1e-5` cutoff raises order
70 to 96 and passes the final grid certificate. The data do not establish a
small-order, all-output guaranteed dynamic ROM.

## Consequences

The valid steady primal-dual absolute bound is useful for deciding **which
HTC parameter to add**. To make it a stopping rule for the final dynamic
model, selection should be coupled to the *compressed* frequency-shifted
space: recalculate the output bound after every SVD and enrich or retain
modes until it meets the tolerance. A steady bound still does not bound the
entire transient. The next mathematical step is a time-discrete
output-residual bound (or a suitable transfer-function-to-time bound) for
all 16 source/junction channels, followed by verified maximization over
the continuous HTC box. Candidate-grid certificates and a 12-point holdout
are not continuous-box or all-time guarantees.

The tested dynamic ROM builds 240 or 300 additional snapshots after the
steady selection. That makes it slower than stock despite selecting only
five parameters. An output-aware **per-shift** sampling and a final
output-preserving compression could reduce redundant frequency solves;
they still need to be compared at equal extraction and output tolerance.

## Reproduce

```text
python playground/adaptive_bci_sampling/test_compare_transient.py
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 1e-4 --counts 2 3 --seeds 20260805 7 \
  --greedy-seed-counts 1 --greedy-grid 9 \
  --output-greedy-grid 9 --output-max-points 12 \
  --random-holdout 6 --dt 50 --duration 2000 \
  --output playground/adaptive_bci_sampling/bridge_transient_2p5.json
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 --counts --seeds --output-greedy-grid 17 \
  --random-holdout 6 \
  --output playground/adaptive_bci_sampling/bridge_transient_grid17.json
```

The diagnostics regenerated as `bridge_transient_svd_1e4.json` and
`bridge_transient_svd_1e5.json` use
`--counts --seeds --output-greedy-grid 9 --output-closing-tolerance 1e-4`
at `1e-3`, and `--output-closing-tolerance 1e-5` at `1e-3 1e-4`
respectively. Output JSON includes selected effective HTC coefficients,
per-arm worst physical parameters, and the separate phase times.
