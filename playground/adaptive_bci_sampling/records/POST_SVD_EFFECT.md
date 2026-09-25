# What the final 1e-3 SVD does to the selected HTC snapshots

## Controlled comparison

This diagnostic uses the original two-HTC Case 1 with the previously chosen
Zolotarev seed, two top-high HTC corners, four low-frequency shifts per heat
source, one seed at every higher shift, and three DC snapshots per source.
There is **one identical snapshot matrix** per mesh. From it we make:

- `raw`: the orthonormal span of **all** numerically independent snapshot
  columns plus the constant mode;
- `post_svd`: the production **unit-column-normalized snapshot SVD with
  unchanged relative cutoff 1e-3**, plus the constant mode.

No new full-order HTC solves or alternative cutoffs are used. Both Galerkin
ROMs are evaluated at the exact same four heat sources and 40 BDF1 steps of
50 s. Each of the 16 junction transfers is divided by its exact steady
transfer **at the same parameter**. The nine audit parameters consist of
the three sampled DC HTC vectors, the two unselected corners, two previously
difficult edges, and two interior pairs. They are **not** a worst-box search.
The 1 mm full-order reference uses AMG-CG at relative tolerance 1e-10,
previously checked against direct LU on 2.5 mm and against 1e-12 on 1 mm.

## Measured effects

| Mesh | Snapshot columns | Raw order | Final order | First rejected `sigma/sigma_1` | Raw worst step | Final worst step | Raw worst steady | Final worst steady |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.5 mm | 92 | 93 | 47 | 8.8124e-4 | 3.7455e-6 | 5.5055e-4 | 5.6856e-6 | 8.3450e-4 |
| **1 mm** | **100** | **101** | **57** | **9.5353e-4** | **1.7758e-5** | **5.6538e-4** | **2.1826e-5** | **4.0223e-4** |

The raw diagnostic keeps the full numerical snapshot span (the last retained
singular ratio is approximately `2.56e-7` at 1 mm); those smallest modes
are of comparable scale to the `1e-6` relative tolerance used for snapshot
linear solves. Its errors therefore describe what the computed full span
can accomplish; its 101-state order and sensitivity to low singular modes
make it a **diagnostic**, not the recommended extractor. The final 57-state
model still meets the observed 1e-3 criterion and retains the advantage
over the stock random models reported in
[ONE_MM_FREQUENCY_FACES.md](ONE_MM_FREQUENCY_FACES.md).

For the *same* nine 1 mm points, compression changes the maximum observed
step error by a factor of about `31.8`, and maximum observed steady error
by about `18.4`. At 2.5 mm these factors are both approximately `147`.
Although the upper bounds used for parameter selection concern the raw
snapshot space, the final errors are largely set by the fixed SVD cutoff.

The training points show this directly on 1 mm:

| HTC point | Raw step | Final step | Raw steady | Final steady |
| --- | ---: | ---: | ---: | ---: |
| Degree-one Zolotarev seed | 1.2993e-7 | 5.6360e-5 | 3.1297e-12 | 2.5998e-5 |
| Top high, bottom low | 1.2758e-6 | **5.6538e-4** | 1.6077e-10 | 8.3353e-6 |
| Top high, bottom high | 5.9423e-7 | 3.1576e-4 | 2.1694e-10 | 2.8047e-5 |

The raw DC Galerkin space nearly interpolates the steady transfer at all
three sampled parameters. The final SVD discards directions needed for
that interpolation, so even a *sampled* corner has a nonzero error. Its
largest observed step error is at the **top-high/bottom-low training
corner**, although the raw seed-space quadratic majorant ranked the
top-high/bottom-high corner first. The majorant chooses exposed seed
directions; it does not predict the post-SVD error ordering.

At the difficult `(10000,20.5353)` physical-HTC edge, the 1 mm steady
error is `2.1826e-5` before SVD and `4.0223e-4` after SVD. The step errors
are `1.7758e-5` and `3.0789e-4`, respectively. These errors are computed
against the **same** full-order reference and normalization.

## Exact way to separate truncation from sampling error

Write `W` for the raw span and `V subset W` for the final SVD span,
including the constant mode in both. For every positive real frequency
`s` and every HTC vector `p`, Galerkin energy orthogonality gives the
*exact, full-order-free* identity

`Y_W(s,p)-Y_V(s,p) = (X_W(s,p)-X_V(s,p))^T A_s(p)
                    (X_W(s,p)-X_V(s,p)) >= 0` (positive semidefinite).

Thus the steady or frequency-domain loss due **solely to the final SVD**
can be calculated from the two ROMs. Its diagonal junction transfers are
nonnegative; off-diagonal entries can have either sign, with
`|DeltaY_ab|<=sqrt(DeltaY_aa DeltaY_bb)`. Moreover,

`Y-Y_V = (Y-Y_W) + (Y_W-Y_V)`.

This splits the actual transfer error into raw sampling error and SVD
compression error without changing the cutoff. The two reduced BDF1
trajectories can likewise be differenced directly; the time-domain
transfer difference is **not** claimed to be sign-definite at each time.
The observed before/after errors above show the second term is substantial
on this benchmark, so the original convex corner-ranking bound for `W`
alone cannot certify the final ROM.

A principled next step is to rank or audit parameter candidates by
**the post-SVD output transfer gap and the residual of the final basis**,
while keeping the production 1e-3 threshold fixed. An a priori continuous
box guarantee still requires bounding the raw error and its propagation
through the SVD and BDF1 dynamics. Adding snapshots alone need not decrease
the error after separately recomputing a nonnested truncated SVD space.

## Reproduction

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_post_svd.py --mesh-mm 2.5 --output /tmp/bci-post-svd-2p5.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python playground/adaptive_bci_sampling/probe_post_svd.py --mesh-mm 1 --output /tmp/bci-post-svd-1mm.json
```

Only code and this short report are kept in git; generated numerical JSON
belongs outside the repository.
