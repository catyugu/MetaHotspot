# Frequency-dependent HTC samples: an eight-parameter experiment

## Question and fixed comparison

Can the parameter samples used for BCI extraction be chosen deterministically,
with fewer full-order solves and lower junction error than the reproduced
FANTASTIC random-parameter loop? This experiment partitions the four silicon
die crowns and the FR4 bottom into four quadrants each, giving **eight
independent HTC groups**. It keeps the same Case 1 geometry at 2.5 mm (9,072
cells), the same four individual heat ports, the original per-port 12-point
elliptic frequency plan, the unmodified column-normalized snapshot SVD with
**relative cutoff 1e-3**, and the final constant basis vector.

For each parameter, compare all 16 entries of the 4-by-4 step transfer
against full-order BDF1 at `dt=50 s`, 40 steps. Each entry is divided by its
own exact, *same-parameter* steady junction rise. The finite validation set
contains **all 256 parameter-box vertices**, 12 paired-log parameters and 24
independent log-random parameters. It is an audit of these points and times,
not a bound over the continuous eight-dimensional box.

## Construction

Let `p=(p_top[4],p_bottom[4])` denote the effective HTC values. Take the
one-point two-group Zolotarev nodes, replicated over the four groups on each
side, as the common seed `p_*`. The other 16 points fill the *entire bottom
four-cube* at the largest HTC on all four die crowns:

\[
 \mathcal P_{17}=\{p_*\}\;\cup\;
 \{(p_{\mathrm{top,max}},b):
        b\in\{p_{\mathrm{bottom,min}},p_{\mathrm{bottom,max}}\}^{4}\}.
\]

Each of the four lowest positive shifts per heat port, those `s <= 1/dt =
0.02 s^-1`, uses all 17 points. Each of the other eight shifts uses just
`p_*`. Thus the total is `4 ports * (4 * 17 + 8 * 1) = 304` full-order right
hand-side solves. This frequency cut is motivated by the first BDF1 resolvent
`(A+C/dt)^-1`; it is a structural hypothesis, **not** an a priori bound on
the 40-step transfer. In particular, selection of the exposed bottom face is
empirical; Zolotarev's one-dimensional guarantee does not extend to the
entire eight-dimensional face-selection rule.

### Why high-frequency cross interactions decay

Write `A(s,p)=K+s C+sum_i p_i H_i`, `R=A^-1`, and `Y=G^T R G`. In this
thermal discretization `A` is an invertible M-matrix, `C` and the `H_i` are
nonnegative diagonal matrices, and `G` is nonnegative. Thus `R` is entrywise
nonnegative. For any multi-index of HTC coordinates,

\[
 (-1)^m\partial_{i_1}\cdots\partial_{i_m}Y
 =\sum_{\pi\in S_m}G^T R H_{i_{\pi(1)}}R\cdots
 H_{i_{\pi(m)}}R G\quad\text{(entrywise nonnegative)}.
\]

Since `dR/ds=-R C R`, the derivative with respect to `s` of *each* summand
is entrywise nonpositive. Therefore every absolute mixed HTC derivative of
the **raw transfer** decreases entrywise as frequency increases. This
statement does not automatically hold for a derivative divided by `Y(s)`, or
for the error after the fixed SVD cutoff and BDF1 propagation.

`probe_mixed_frequency.py` evaluates the maximum over 16 top/bottom HTC
pairs and all 16 transfer entries of
`p_*[i]*p_*[j]*abs(d^2Y/dp_i dp_j)`:

| `s` (1/s) | maximum scaled mixed derivative | maximum divided by `abs(Y)` |
| ---: | ---: | ---: |
| 0 | 4.7440e-1 | 5.4508e-2 |
| 0.001 | 1.0233e-1 | 1.6500e-2 |
| 0.02 | 2.8931e-5 | 2.9073e-5 |
| 0.1 | 3.7529e-10 | 2.0298e-9 |

## Completed full-box-vertex comparison

The two stock seeds are `20260805` and `7`. All three ROMs are validated at
the **same** 292 parameters. Times below are wall clock from one run and
include the stock algorithm's residual checks; face extraction time includes
its spectral preparation, solves and SVD. Small timing variation between
runs is expected.

| Design | Full RHS solves | Final ROM order | Extraction (s) | Worst step entry | Worst steady entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17-point low-frequency face | 304 | 56 | 8.01 | **8.2647e-4** | **4.5775e-3** |
| Random stock, seed 20260805 | 466 | 51 | 28.98 | 2.2577e-3 | 7.2487e-3 |
| Random stock, seed 7 | 468 | 51 | 28.34 | 2.8276e-3 | 7.9458e-3 |

The worst step error for the face design occurs at vertex 242: all four top
groups at maximum and three of the four bottom groups at minimum. Its worst
steady error occurs at vertex 4: all four top groups at minimum and exactly
one bottom group at maximum. The tested finite-time 1e-3 goal passes, while
the steady 1e-3 goal does **not** pass. The face uses 35% fewer solves than
seed 20260805, extracts about 3.6 times faster on this run, and has about
2.7 times smaller worst observed transient error; it does retain five more
ROM modes. This is a comparison with two random seeds, not a statistical
claim over all seeds or a worst-box guarantee.

### Adding the mathematically distinguished DC endpoint

The steady transfer is exactly `Y(0,p)`, whereas every shift in the original
elliptic plan is positive (the smallest is approximately `0.0005538 s^-1`).
As a separate, fixed-SVD experiment, also take `s=0` at all 17 points for
each source. This adds `4 * 17 = 68` full-order right-hand-side solves and
does **not** change any of the original 12 elliptic frequency points or the
1e-3 SVD threshold. The same 292 full-order validation cases give:

| Design | Full RHS solves | Final ROM order | Extraction (s) | Worst step entry | Worst steady entry |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17-point face + DC | **372** | 60 | **9.36** | **6.2230e-4** | **5.6313e-4** |
| 17-point face without DC | 304 | 56 | 8.06 | 8.2647e-4 | 4.5775e-3 |
| Random stock, seed 20260805 (preceding run) | 466 | 51 | 28.98 | 2.2577e-3 | 7.2487e-3 |
| Random stock, seed 7 (preceding run) | 468 | 51 | 28.34 | 2.8276e-3 | 7.9458e-3 |

The first two rows were run together, and the two stock rows are from the
same machine under the same single-thread BLAS setting in the preceding run.
The DC variant uses about 20% fewer full-order solves than the first random
seed, with approximately 3.6 times lower observed worst transient error and
13 times lower observed worst steady error. The worst DC-variant steady case
is independent log-random parameter 275; its worst step case is box vertex
249. The additional four ROM modes and the 68 extra solves are the cost of
improving the steady transfer.

A control adds DC **only at the single Zolotarev seed**: four extra solves in
place of 68. On all 292 validation parameters it has 308 solves, ROM order
56, extraction 8.22 s, worst step error **6.2203e-4**, and worst steady error
**4.5643e-3**. Its steady worst case is still vertex 4. Thus one DC sample
per source suffices for the *measured 40-step criterion*, while zero-frequency
coverage of the bottom face is necessary among these three tested designs
to bring the steady criterion below 1e-3. The data do not establish that
17 DC points are mathematically minimal.

## Other exploratory outcomes and cautions

Earlier experiments in the same conversation tested 2, 4, 6 and 8 HTC
groups under the same frequency/SVD plan. Five points chosen from a global
steady tangent risk ranking gave observed worst transient errors `8.255e-4`
in 4 groups and `9.805e-4` in 6 groups, but `8.662e-3` in 8 groups. An
underresolved seven-point exposed-face design in eight groups reached
`7.681e-2` and adding two checkerboard points gave `2.504e-2` on their
targeted validation set. These preliminary scripts/results were lost in a
workspace reset, so the primary reproducible claim here is the full
292-point comparison above.

Adding snapshot columns can make the *compressed* ROM worse: the existing
SVD keeps singular values `sigma_j >= 1e-3 sigma_1`. With unit-normalized
columns, `sigma_1` may increase with sample count, and SVD-compressed spaces
from distinct sample sets need not be nested. Earlier 21-point variants did
indeed sometimes have larger errors than the 17-point variant. The cutoff is
deliberately held fixed here; no additional selection based on SVD rank or
validation errors is built into the design.

## Reproduction

From repository root (with the project and its Python dependencies installed):

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_frequency_faces.py --include-stock --output /tmp/bci-frequency-face-results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_frequency_faces.py --include-dc --output /tmp/bci-frequency-face-dc-results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/explore_frequency_faces.py --include-dc --compare-seed-dc --output /tmp/bci-frequency-face-dc-ablation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_mixed_frequency.py
python -m unittest discover -s playground/adaptive_bci_sampling -p 'test_*.py' -v
```

Generated result JSON belongs outside the repository. For a stronger
theoretical result, relate an *output-weighted* mixed-derivative majorant and
the rational frequency interpolation error to the final finite-time transfer
**after** the fixed SVD. The exposed-face choice then needs a computable
remainder for the four unsampled die-crown directions; the M-matrix theorem
above supplies a frequency ordering of raw interactions, but no such
eight-dimensional remainder yet.
