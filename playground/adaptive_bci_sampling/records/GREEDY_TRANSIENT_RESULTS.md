# Zolotarev seeds with residual-greedy HTC enrichment

## Experiment

This extends [TRANSIENT_COMPARISON.md](TRANSIENT_COMPARISON.md) on the same
2.5 mm, 9072-cell Case 1. At each frequency shift of each of the four heat
source ports, start with a fixed 1x1 or 2x2 Zolotarev HTC tensor. Evaluate
the *Galerkin field residual* divided by the source norm on a deterministic
logarithmic HTC candidate grid, solve the full model at its maximizer, and
repeat until every grid point has residual at most the requested extraction
tolerance. The basis accumulates within each port across its frequency shifts.
As in the reproduced extractor, all selected snapshots are passed through
one normalized closing SVD and the constant mode is restored. The greedy
scoring uses reduced solves and an affine residual Gram matrix: candidate
evaluation does not call the full-order solver.

Both extractors use identical four-port frequency-shift formulas, sparse
solver, physical HTC range `[1, 10000]^2`, and closing SVD threshold unless
explicitly labeled otherwise. The full-order holdout consists of all four
corners, `(50, 1000)`, `(100, 100)`, and six fixed random log-uniform points.
Each parameter uses all 16 source-to-junction unit step transfers, BDF1 from
zero initial rise at `dt=50 s` through `2000 s`. The nominal step error uses
the physical power vector `(0.1, 0.2, 0.3, 0.4) W`. Each junction error is
normalized by the **same-parameter full-order steady junction rise**; an
individual transfer error uses the same-parameter steady transfer entry.
Extraction time excludes validation and includes Zolotarev spectral setup.

## Matching extraction and SVD tolerances

`RHS` counts full-order source solves. The numbers in parentheses for greedy
rows are initial Zolotarev solves plus greedy additions. `Grid residual` is
the maximum at the completed shifts **before** the closing SVD. `Step` is
the maximum nominal-power junction step error; `Transfer` is the worst of
all 16 individually normalized step transfer errors over the holdout.

| Tolerance | Extractor | RHS | ROM order | Extract s | Grid residual | Step | Transfer |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | Reproduction seed 20260805 | 126 | 42 | 8.79 | — | `5.20e-5` | `1.76e-3` |
| `1e-3` | Zolotarev 1x1 + greedy, 9x9 | 140 (48+92) | 44 | 11.25 | `8.04e-4` | `4.83e-5` | `1.44e-3` |
| `1e-3` | Zolotarev 2x2 + greedy, 9x9 | 227 (192+35) | 45 | 15.60 | `9.21e-4` | `3.87e-5` | `1.83e-3` |
| `1e-4` | Reproduction seed 20260805 | 232 | 62 | 15.59 | — | `3.74e-6` | `1.81e-4` |
| `1e-4` | Zolotarev 1x1 + greedy, 9x9 | 222 (60+162) | 65 | 18.90 | `9.98e-5` | `3.61e-6` | `1.26e-4` |
| `1e-4` | Zolotarev 2x2 + greedy, 9x9 | 330 (240+90) | 64 | 24.59 | `9.96e-5` | `3.06e-6` | `2.79e-4` |

In both tolerance rounds, all 48 or 60 greedy port-shifts reached the
candidate-grid field-residual threshold. The 1x1 seed improves the observed
worst transfer error over the stock seed by about 18% at `1e-3` and 30% at
`1e-4`. It costs about 28% and 21% more extraction time respectively.
Its RHS count at `1e-4` is slightly smaller than the stock count, but the
reduced candidate solves and spectral preparation still cost time. The 2x2
seed is dominated in these runs. None of the resulting compressed ROMs meets
the requested tolerance for **every** transfer entry, although all meet it
for the nominal-power step history.

This is a comparison with one reproduced random seed. See the earlier
comparison for a second seed and the non-greedy fixed Zolotarev tensor results.
For example, at `1e-3`, fixed 4x4 requires 768 RHS, 45 modes, and 37.37 s
for `1.59e-3` transfer error; greedy 1x1 uses much less time and improves
that holdout error. The stock and greedy rows in the table above were measured
in the *same run*. Wall times from separate runs are not exact speed ratios.

## Grid density and closing SVD

With only the greedy 1x1 arm, increasing candidates from 9x9 to 17x17 at
`1e-3` changed the selected set from 140 to 128 RHS, while keeping order 44.
The completed 17x17 residual was `9.45e-4`; extraction took 14.89 s and
holdout transfer error was `1.58e-3`. A denser candidate grid changes the
greedy path and increases score-evaluation time. Both candidate grids fail
the final transfer tolerance. The worst transfer parameter in both cases is
the physical corner `(10000, 1)`.

To isolate the closing SVD, the next rows hold the 9x9 candidate grid and
greedy 1x1 sampling rule fixed but change the SVD cutoff. Each row re-runs
the same deterministic 140- or 222-RHS extraction. These rows have **different
SVD tolerances** and therefore are not like-for-like alternatives to the
table above.

| Greedy target | Closing cutoff | RHS | ROM order | Extract s | Step | Transfer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-3` | `1e-3` | 140 | 44 | 11.25 | `4.83e-5` | `1.44e-3` |
| `1e-3` | `1e-4` | 140 | 61 | 11.67 | `1.15e-5` | `1.10e-3` |
| `1e-3` | `1e-5` | 140 | 84 | 11.14 | `3.45e-7` | `5.49e-6` |
| `1e-4` | `1e-4` | 222 | 65 | 18.90 | `3.61e-6` | `1.26e-4` |
| `1e-4` | `1e-5` | 222 | 92 | 18.66 | `1.02e-7` | `2.38e-6` |

At the tighter `1e-5` cutoff the measured transfer tolerances pass, but
the ROM needs 84 or 92 modes. At `1e-3` target, cutoff `1e-4` still leaves
the worst entry slightly above tolerance. The mode count, rather than HTC
placement alone, is decisive for the final error. The elapsed times across
the separate SVD experiments fluctuate and do not establish that a tighter
cutoff reduces extraction time.

## Scope of the stopping rule

The pre-SVD residual bound applies to the finite deterministic candidate
grid at the selected frequency shifts, in the *field residual/source norm*.
It is not an error bound on junction transfers, on all times, or on the
continuous HTC box. A smaller residual does not directly control a small
relative cross transfer, especially when its steady normalization is small.
Closing SVD can remove modes that satisfied the grid test. The 12 holdout
parameters are an observed accuracy comparison, **not a guarantee**.

The earlier uncompressed steady transfer experiment in
[ZOLOTAREV_RESULTS.md](ZOLOTAREV_RESULTS.md) uses a primal-dual energy
certificate to control junction outputs on its candidate set. That result
does not automatically extend to these compressed dynamic ROMs. For a
tolerance-driven production extractor, certify *each retained final ROM's*
four-by-four junction transfer over a prescribed frequency/time domain and
parameter box; retain or add modes where the certificate fails. A verified
box-wide maximum and a bound connecting frequency samples to the entire
time interval are still required for a continuous, transient guarantee.

## Reproduce

```text
python playground/adaptive_bci_sampling/test_adaptive_zolotarev.py
python playground/adaptive_bci_sampling/test_compare_transient.py
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 1e-4 --counts --seeds 20260805 \
  --greedy-seed-counts 1 2 --greedy-grid 9 --max-extra-per-shift 12 \
  --random-holdout 6 \
  --output playground/adaptive_bci_sampling/greedy_transient_2p5.json
python playground/adaptive_bci_sampling/compare_transient.py 2.5 \
  --tolerances 1e-3 --counts --seeds --greedy-seed-counts 1 \
  --greedy-grid 17 --max-extra-per-shift 12 --random-holdout 6 \
  --output playground/adaptive_bci_sampling/greedy_transient_grid17.json
```

The closing SVD checks can be regenerated as
`greedy_transient_svd_1e4.json`, `greedy_transient_svd_1e5.json`, and
`greedy_transient_tol1e4_svd_1e5.json` (generated JSON is ignored by Git).
Their commands use the matching `--tolerances`,
`--greedy-closing-tolerance`, and `--output` arguments with
`--counts --seeds --greedy-seed-counts 1 --greedy-grid 9`.
