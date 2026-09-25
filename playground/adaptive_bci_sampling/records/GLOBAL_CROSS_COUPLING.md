# A global two-HTC structure: low-rank cross-boundary coupling

## Why examine the cross block

The original two-group Case 1 has two *disjoint* sets of boundary cells. The
full-box centered Neumann ratio is about 0.9974 at 5 mm and 0.9977 at 2.5 mm;
low-degree global Taylor fields are therefore unsuitable. Four-corner
multilinear interpolation of the **DC field** is also insufficient: on the
5 mm mesh its uncompressed 16-column Galerkin span has a maximum relative
spectral impedance error of `5.14e-2` over an 11-by-11 log HTC audit. The
associated global Bernstein transfer enclosure exceeds `2e5` relative to
the upper-corner impedance. These are diagnostic failures of polynomial
interpolation, not lower bounds on rational sampling.

There is a different *global* structure. At a fixed **nonnegative real**
frequency `s`, let

```
 A_0 = K+s C+p_min,1 H_1+p_min,2 H_2,
 A(p) = A_0+E_1 D_1(p_1) E_1^T+E_2 D_2(p_2) E_2^T,
 T_i = A_0^-1 E_i,   Phi_ij = E_i^T A_0^-1 E_j,
 x_0 = A_0^-1 G,     c_i = E_i^T x_0.
```

The `D_i` are diagonal, positive semidefinite, and vanish at the lower HTC
endpoint. Nothing is approximated in this representation. Woodbury gives
`x(p)=x_0-[T_1,T_2] D^(1/2) S(p)^-1 D^(1/2) c`, where

```
 S(p) = [ I+D_1^1/2 Phi_11 D_1^1/2,  D_1^1/2 Phi_12 D_2^1/2;
          D_2^1/2 Phi_21 D_1^1/2, I+D_2^1/2 Phi_22 D_2^1/2 ].
```

Only `Phi_12` conveys interaction between the two HTC groups. Let
`Phi_ii=L_i L_i^T` be Cholesky factors and
`Gamma=L_1^-1 Phi_12 L_2^-T`. Since the full `Phi` is positive definite,
`gamma=||Gamma||_2<1`. Its truncated SVD `Gamma_r` gives a *single*,
parameter-independent cross-block approximation
`Phi_12^(r)=L_1 Gamma_r L_2^T`; define `delta_r=sigma_(r+1)(Gamma)`.

## Exact axis-span statement

**Claim.** If `Phi_12=B_1 B_2^T` has rank `r`, the entire two-parameter
solution family at this frequency lies in the sum of two *one-parameter*
solution manifolds. Besides the ordinary source `G`, each axis requires at
most `r` extra right-hand sides of the form `E_i Phi_ii^-1 B_i`.

To see this, let `S_0` be the block diagonal part of `S`. The remaining
off-diagonal block has rank at most `2r`. Applying Woodbury again to
`S_0+S_cross`, every mixed-parameter correction is a combination of

```
 T_i W_i(p_i) B_i,       W_i(p_i)=(D_i(p_i)^-1+Phi_ii)^-1,
```

with the continuous limit `W_i=0` when `D_i=0`. These correction directions
are available **on coordinate axes alone**, because, writing
`A_i(p_i)=A_0+E_i D_i(p_i) E_i^T`, the exact identity is

```
 T_i W_i(p_i) B_i
    = A_0^-1 E_i Phi_ii^-1 B_i
      - A_i(p_i)^-1 E_i Phi_ii^-1 B_i.
```

The block diagonal part is the sum of `A_i(p_i)^-1 G` minus `x_0`.
Consequently, an exact rank-`r` cross interaction requires **axis
manifolds**, rather than a tensor product of HTC samples. The same statement
holds *exactly* for the surrogate obtained by replacing `Gamma` with
`Gamma_r`; its discrepancy from the original model still needs a bound.
The rank reduction concerns interactions between groups; it does not assert
that each univariate axis is resolved by one or two parameter samples.
Checking the displayed one-axis identity on the 5 mm Case 1 with the first
four canonical channels and an interior parameter gave relative differences
`3.8e-16` and `1.3e-15` for the two groups.

## A global bound exists, but is not yet useful

Put `S_i=I+D_i^1/2 Phi_ii D_i^1/2` and normalize the cross block of `S` by
`S_i^-1/2`. The two factors surrounding `Gamma` are contractions: for
example `||S_i^-1/2 D_i^1/2 L_i||_2<=1`. Thus the normalized full cross
block has norm at most `gamma` and the **truncation discrepancy is at most
`delta_r` for every HTC in the original continuous box**. In particular,
the truncated block system remains positive definite globally.

Let `v=D^1/2 c` and `w=S_0^-1/2 v`. The inverse resolvent identity gives
an absolute spectral transfer bound at every HTC:

```
 ||Z(p)-Z_r(p)||_2
   <= delta_r ||w(p)||_2^2 / [(1-gamma)(1-||Gamma_r||_2)].
```

Here `||w||_2` is the matrix spectral norm. Because
`D_i^1/2 S_i^-1 D_i^1/2=(D_i^-1+Phi_ii)^-1` is monotone in `D_i`, its
maximum over the box occurs at `p_max`. Also `Z(p)>=Z(p_max)` in Loewner
order, so division by `||Z(p_max)||_2` gives one explicit *global relative*
bound. At 5 mm with `r=4` it is about **`1.04e4`**, however; at 2.5 mm
it is about **`1.32e4`**. Source directions almost orthogonal to the
discarded coupling modes are lost in this norm bound. These figures must
not be presented as a useful `1e-3` certification.

## What the actual two-group model shows

`probe_cross_coupling.py` constructs the exact reference Green blocks and
diagonalizes the canonical cross block. The table records counts of
singular values above `0.01`; the threshold is a way to *describe the
spectrum*, not a proposed algorithmic tolerance.

| Mesh | Cells | Group cell counts | `sigma_1` at DC | `sigma_5` at DC | Count at DC | Count at `s=0.02` | `sigma_1` at `s=0.1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 mm | 1,320 | 16 / 264 | 0.951764 | 0.019001 | 8 | 4 | 0.000614 |
| 2.5 mm | 9,072 | 64 / 1,008 | 0.952547 | 0.023569 | 8 | 1 | 0.000105 |

Replacing only `Phi_12` by its rank-four approximation, while retaining
the **exact** diagonal blocks and source solutions, gave the following
maximum relative spectral DC transfer differences on the *same five*
parameter points: `p_max`, the 25% and 50% affine diagonal points, and
the two mixed corners. This is **not** a maximum over the continuous box
and is **not** an error of a final SVD ROM.

| Mesh | Cross rank 0 | Cross rank 1 | Cross rank 2 | Cross rank 4 | Cross rank 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 mm | 599 | 5.46 | 4.81 | `3.01e-5` | `2.70e-5` |
| 2.5 mm | 598 | 4.91 | 4.28 | `4.76e-5` | `3.40e-5` |

The large discontinuity between ranks two and four reflects the four
dominant cross-boundary channels in this geometry, not a generic theorem
that four modes always suffice. The lower rank-eight errors need not be
monotone with rank when measured on a specific output and five points.
At `s=0.02`, coupling has already weakened considerably; this offers a
structural explanation for concentrating HTC diversity at low frequencies,
but does **not** establish the final transient-error target.

### Cost and proof gap

The script forms the entire reference Green block: **280 boundary solves**
at 5 mm or **1,072 boundary solves** at 2.5 mm, plus the source solves,
at each frequency. This is more work than the stock random extraction.
An algorithm must obtain the few important cross directions with a small
number of matrix-free inverse actions; these measured ranks alone are not
an economical sampler. Crucially, the present global transfer bound is
much too loose, and the exact axis-span theorem applies before finite axis
sampling and before the **unchanged `1e-3` closing SVD**. There is no claim
here of the relative Hankel `<2epsilon` or impulse-energy `<2sqrt(epsilon)`
guarantee. A useful next theorem has to incorporate source-weighted cross
tails and SVD projection error, then extend real-frequency bounds to the
time domain without a parameter-box subdivision.

For mathematical context, [Angleitner, Faustmann, and Melenk,
*Exponential Meshes and H-Matrices*](https://arxiv.org/pdf/2203.09925)
proves low-rank approximability of off-diagonal inverse blocks for specified
elliptic FEM discretizations. Applying its rate directly to this
heterogeneous finite-volume stack requires checking its geometric and
discretization assumptions; the spectral data above are measurements.

Reproduce, from the repository root:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python playground/adaptive_bci_sampling/probe_cross_coupling.py 5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python playground/adaptive_bci_sampling/probe_cross_coupling.py 2.5
```
