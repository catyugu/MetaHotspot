# Two-HTC Case 1 on the 1 mm mesh: results and sampling rationale

## Exact comparison being made

The original two-group Case 1 mesh has **122,400 cells** at 1 mm. All arms
use four heat ports, the same per-port **14 elliptic shifts**, the original
unit-column snapshot SVD threshold **1e-3**, and a final constant mode.
The two stock seeds use the original random HTC enrichment and ten passing
probes. The three-point method chooses HTC deterministically and also includes
one steady snapshot at each of its three points per source. It does not
change the closing SVD or elliptic matching points.

At each tested parameter compare all 16 source-to-junction BDF1 transfers
for 40 steps of 50 s and all 16 steady transfers. Each entry is divided by
its **same-parameter exact steady junction rise**. Validation uses the four
physical HTC corners, the reproduction point `(50,1000)`, the `(100,100)`
center, four independently seeded log-random pairs, and the two difficult
physical-HTC edge points `(2053.5250,10000)` and `(10000,20.5353)`: **12
parameter vectors in total**. This is not a continuous-box guarantee.

## Why these three HTC points?

Let `A_s(p)=K+sC+p_1 H_1+p_2 H_2` and
`Y_s(p)=G^T A_s(p)^-1 G`. The thermal discretization has an inverse-positive
M-matrix, nonnegative diagonal `H_1,H_2,C`, and nonnegative four-source
matrix `G`. We use two different mathematical facts:

1. **One rational seed.** With the other HTC fixed, eliminating all cells
   outside the active boundary cells turns the response into a resolvent
   with scalar kernel `1/(lambda+p_i)` and finite generalized spectrum in a
   coordinate-dependent interval `[alpha_i,beta_i]`. The degree-one
   finite-interval Zolotarev rule places one point on each HTC axis to
   minimize its associated *one-dimensional* rational separation ratio.
   Numerical, box-wide boundary Schur spectral enclosures on the 1 mm mesh
   give the effective-HTC seed
   `p_*=(356.2744296,22.5080361)`. This guarantee does **not** cover two
   parameters or the final compressed time response.

2. **Output-aware corner ranking.** For the *uncompressed steady* Galerkin
   space containing all four exact source responses `x_{*,a}=A_0(p_*)^-1 g_a`,
   the transfer error is exactly

   `E=Y-Y_V=(X-X_V)^T A_0(p)(X-X_V)=R^T A_0(p)^-1 R`.

   Put `d=p-p_*` and `A_min=A_0(p_min)`. The Ritz best-approximation
   property and `A_0(p)^-1 <= A_min^-1` give

   `E_aa <= d^T Q_a d`, with
   `(Q_a)_ij=(H_i x_{*,a})^T A_min^-1(H_j x_{*,a})`.

   Each `Q_a` is positive semidefinite. Since `E` is positive semidefinite
   and M-matrix monotonicity gives `Y_ab(p)>=Y_ab(p_max)>0`, every relative
   junction transfer error has the computable **raw-space** majorant

   `|E_ab|/Y_ab(p) <= d^T (Q_a+Q_b) d / (2 Y_ab(p_max))`.

   The maximum over all 16 entries is convex in `p`, hence its maximum on
   the two-dimensional box is attained at a vertex. Computing its four
   scores on the actual 1 mm operators gives, from largest to smallest:

   | Effective HTC corner `(top,bottom)` | Relative-error majorant score |
   | --- | ---: |
   | `(9629.6296,566.0377)` | `5.6874e7` |
   | `(9629.6296,0.998336)` | `3.7314e7` |
   | `(0.999996,566.0377)` | `1.5628e6` |
   | `(0.999996,0.998336)` | `8.4499e4` |

   Therefore both endpoints of the high-top-HTC edge are the two most
   exposed **seed directions**. Together with `p_*` they are the three
   sampled HTC points. These very loose scores justify *ranking*, not the
   error magnitude or the claim that three points are minimal. The same
   two vertices were ranked first on the 2.5 mm mesh.

## Why concentrate the three points at low frequency and include DC?

For `R_s=A_s(p)^-1`, `dR_s/ds=-R_s C R_s`. Every product of nonnegative
`R_s,H_i,C,G` is entrywise nonnegative. In particular,

`d²Y_s/(dp_1 dp_2) = G^T(R_s H_1 R_s H_2 R_s + R_s H_2 R_s H_1 R_s)G >= 0`,

and differentiating with respect to `s` inserts one `-R_s C R_s` into
each product. Thus **every entry of this raw cross-HTC derivative decreases
as positive frequency increases**. At the 1 mm seed the maximum over all
16 transfer entries of `p_1 p_2 |d²Y_s/(dp_1 dp_2)|` is:

| `s` (1/s) | Maximum scaled cross derivative |
| ---: | ---: |
| `0` | `0.76953` |
| `0.001` | `0.16583` |
| `0.02` | `3.0584e-5` |
| `0.1` | `8.6679e-11` |

The first BDF1 solve uses the resolvent `A_0(p)+C/dt`, so `1/dt=0.02
s^-1` is a physically defined split point. The original four elliptic
shifts below that value are approximately `0.01259`, `0.004100`,
`0.001372`, `0.0005702 s^-1`; ten others lie above it. Each source uses
the three HTC points at these four low shifts and the seed at ten high
shifts. The **exact DC transfer** is `Y_0(p)`, whereas all 14 elliptic shifts
are positive; sampling `s=0` at the three points addresses steady behavior
explicitly. Counting full-order right-hand-side solves:

`4 sources * (4 low shifts * 3 points + 10 high shifts * 1 point + 3 DC points)
= 100`.

The sign/monotonicity argument justifies placing more *raw* interaction
information at low frequency. Neither `s<=1/dt` nor the choice of two
additional corners proves a finite-time error bound **after** SVD. At 2.5
mm, testing showed that adding DC only at the seed missed 1e-3 transient
tolerance on a denser two-dimensional grid; that ablation was not repeated
on the 1 mm mesh.

## Measured 1 mm extraction and errors

The ordinary direct-LU Schur spectral preparation took 101.03 s and consumed
about 4.37 GB peak RSS. The same Schur/Lanczos eigenproblem with AMG-CG
inverse actions took **18.91 s** and returned the two Zolotarev nodes with
relative differences of about `1.2e-13` and `1.6e-13` versus LU. Six
representative parameters checked with both constructions had final errors
that differed by at most `6.4e-12` (step) and `9.4e-12` (steady). The AMG
version took another 35.09 s for full snapshots and 1.93 s for SVD, totaling
**55.93 s** and 668 MB peak RSS for its extraction-plus-check run. A small
Lanczos Ritz residual and an accurate inner CG solve make the two node
estimates numerically interchangeable here; the iterative inverse is *not*
claimed to turn the spectral estimate into a formally certified enclosure.

| Method | Full RHS solves | Final ROM order | Extraction (s) | Worst step entry | Worst steady entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Three points at low shifts + three DC; AMG spectral preparation** | **100** | 57 | **55.93** | **5.6538e-4** | **4.0223e-4** |
| Same snapshots, direct-LU spectral preparation | 100 | 57 | 138.29 | 5.6538e-4 | 4.0223e-4 |
| Random stock, seed 20260805 | 159 | 52 | 67.95 | 4.9537e-3 | 5.3136e-3 |
| Random stock, seed 7 | 165 | 50 | 69.78 | 1.6812e-2 | 1.3848e-2 |

Stock and the LU-spectrum deterministic version were extracted and evaluated
in the same 12-parameter run. The faster AMG-spectrum version was rerun on
the four corners and the two historically difficult edges: its basis order
and six worst-case metric pairs agreed with the LU-spectrum version to the
differences above. The table uses the LU version's **12-parameter error
maxima** for the numerically equivalent AMG version; it does not present
the 6-point rerun as an independent 12-point validation. All times include
the full spectral setup for the deterministic arm and the stock algorithm's
probe work. For this tested set, the deterministic design uses 37% fewer
full RHS solves than the first stock seed and is approximately 18% faster
with the AMG spectral solve, while retaining five more modes.

Because a single 1 mm sparse direct LU factor took about 47 s and 4 GB,
full-order validation used AMG-preconditioned CG (`rtol=1e-10`) for both
steady and 40-step BDF1 operators, reusing one preconditioner per operator
and parameter. On 2.5 mm, comparing to the direct reference changed the
normalized step metric by at most `8.36e-10`; on 1 mm, tightening the
reference tolerance from `1e-10` to `1e-12` changed it by at most
`1.71e-9` at the difficult `(10000,10000)` corner. The uncertainty is
small compared with the reported ROM errors.

## Reproduction

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --check-reference
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --output /tmp/bci-1mm-direct.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --check-spectrum-from /tmp/bci-1mm-direct.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --fast-spectrum --output /tmp/bci-1mm-amg.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --rank-corners-from /tmp/bci-1mm-direct.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_one_mm_frequency_faces.py --probe-interactions-from /tmp/bci-1mm-direct.json
```

The main two-group report covers a 17-by-17 parameter grid and two
65-point edge scans at 2.5 mm. This 1 mm run checks 12 parameters; its
observed success does not establish worst-box or other-time-step tolerance.
Generated JSON files belong outside git.

The direct comparison of this design's raw snapshot span with its **actual
post-SVD ROM** is reported in [POST_SVD_EFFECT.md](POST_SVD_EFFECT.md).
