# Source-independent inverse compression: principal-part and jump matching

This is a NEW isolated study, not an extension of hotspot bounds, commutator
banks, shared-Gram fitting or finite-pool spectral cubature. Base production
sources are unchanged. Its objects are shifted diffusion inverses, not
source-trained state-space ROMs. No full transient or device claim is made.

## Question and exact finite-matrix gate

Can a fast full-rank principal-part inverse absorb the mesh complexity while
only a small correction handles smooth nonseparable coefficients? Does keeping
known flat material jumps in that backbone materially reduce correction rank?

For a positive backbone P=Q Q^T define H=Q^T A Q. If H u_i=lambda_i u_i,

    R_r = P + (Q U_r)(Lambda_r^-1-I)(Q U_r)^T.

Selecting the r largest |lambda_i-1| minimizes

    ||I - A^(1/2) R_r A^(1/2)||_2

among arbitrary rank-r updates to P; the optimum is the largest omitted
|lambda_i-1|. This is the standard best-rank approximation principle in energy
coordinates, NOT a new theorem. It allows us to test the approximation class
before blaming a candidate-bank heuristic. One/two/three stationary inverse
applications have energy-error matrix E, E^2, E^3. Extra applications can
DIVERGE if ||E||>1. This polynomial identity is also established algebra.

The continuum Liouville identity motivates coefficient scaling:

    -div(k grad) = sqrt(k) [-Delta + Delta(sqrt(k))/sqrt(k)] sqrt(k).

For harmonic FVM faces, g_ij/sqrt(k_i*k_j)=sech((log k_i-log k_j)/2).
Smooth variation gives a small principal-part defect; jumps need not. The
continuum identity is not silently substituted for the actual FVM matrix:
all oracle/error calculations use the explicitly assembled discrete operator.

## Predeclared cases and controls

Research seeds: 20261001, 20261002, 20261003. Only seed42 small smoke runs are
used locally. Domain [0,1]^2, cell-centred harmonic FVM, homogeneous Dirichlet
on all four sides; capacity normalized to I. These are dimensionless 2-D
research systems, NOT the production 3-D assembler or calibrated devices.

Three analytic coefficient families use the same physical field under mesh
refinement. `smooth` is exp of two fixed smooth trigonometric fields (amplitudes
0.65,0.35, phase set by seed). `layered` multiplies it by known y layers
1/30/3 at y=1/3,2/3. `inclusion` adds an unrepresented 20-fold rectangular jump
[.25,.5) x [.25,.75). This is an explicit negative-control geometry, retained
whether or not the method works. Shifts are 0,20,2000; no shift/geometry is
selected after seeing results.

Four backbones:
- homogeneous geometric-mean conductivity;
- exact layer conductivity, without coefficient scaling (`raw_layer`);
- homogeneous diffusion after scaling by sqrt(full k) (`scaled_uniform`);
- exact layer diffusion after scaling only by sqrt(k/layer) (`jump_aware`).

No split is fitted using a source or reference solution. Flat layers come from
the declared model metadata. All backbones use the y-profile of the x-averaged
transformed mass term. The exact factor is computed by an orthonormal x DST-II
and per-mode y tridiagonal Cholesky. No dense inverse is used in the online
construction or application.

Dense **oracle-only** grids: n=12,24,36. Compute the complete spectrum for each
backbone and required correction ranks for tolerances .1,.01,.001, with inverse
polynomial degrees 1,2,3. Dense diagonalization is a diagnostic reference, never
hidden in practical timings or provided to the practical algorithm.

Practical grids: n=48,96. Matrix-free Lanczos on H-I, starting with k=8 and
doubling to cap64, stops when an observed Ritz magnitude is at most .09.
Retain magnitudes >.09. All repeated eigensolver actions and postprocessing
are charged. No A^-1 solve is used to construct the correction. Rayleigh-Ritz
re-evaluation defines a positive small matrix. Approximate invariance residuals
and failure to observe a small tail are recorded. This is a numerical tail
criterion, not a rigorously verified completeness bound for Lanczos.

Primary candidate: degree2 corrected inverse (nominal .09^2<.01 energy target).
Degree1/3 and uncorrected degree2 are ablations, not post-hoc replacements.
Strong controls: raw-layer PCG, jump-aware PCG, Ruge-Stuben AMG-PCG, each with
rtol=1e-2 and 1e-6, plus sparse LU. Including loose PCG and LU prevents artificial
wins against needlessly accurate or poorly chosen full-order solvers.

Twelve held-out RHSs per case: three smooth positive, three signed random and
six single-cell concentrated sources. They are generated AFTER construction,
and no physical RHS enters the construction API. Tight LU reference residual
must be <1e-9. Full-vector energy and relative infinity errors are evaluated
outside timing. Do not call small energy error a peak-temperature bound.

## Decision gates, costs and limitations

Structural signal: on layered cases, degree2 oracle rank at eps=.01 grows by
at most a factor two from n12 to n36, and is at most half the scaled-uniform
rank at n36. Report each shift/seed, including counterexamples, not just means.
This is a finite-grid screening criterion, not an asymptotic proof.

Practical signal: the primary candidate must have <=1% maximum energy error
on all twelve RHSs and >=2x online speed versus the cheapest accuracy-qualified
iterative control. Also report sparse LU separately and cold/100/1000-query
costs. A win against iterative methods is not a win against LU. Finite RHS
validation is NOT an all-input guarantee at the larger grids.

Three timing repeats, one untimed warmup of EVERY method, alternating order;
all timed methods solve each RHS individually (no batch-only advantage).
Online includes transforms, triangular solves, full-vector reconstruction,
low-rank multiplications, residuals and refinement. Setup includes each method's
actual factor/hierarchy/basis construction. Shared assembly and source generation
are excluded; raw times, setup, 12-query cold cost and break-even data are saved.
Rank construction starts from scratch for each shift/geometry; no uncharged reuse.

Known limitations: scalar diffusion, prescribed flat interfaces, constant
capacity, fixed shifted systems, finite meshes. No nonlinear Jacobian, tensor
anisotropy, arbitrary geometry, moving interface, continuous-time error,
formal interval arithmetic, or 3-D production scaling is claimed. The low-rank
update is not a pure small-state ROM: online work is still O(N r) plus transforms.
Process RSS includes oracle/reference storage, not per-method memory.

## Literature boundary

These ingredients have direct prior art. We are testing whether a useful
structure-specific approximation principle merits deeper investigation, NOT
claiming that diagonal scaling, Lanczos, low-rank updates, or Richardson
polynomials are new algorithms.

- Borcea, Guevara Vasquez, Mamonov (2017), A discrete Liouville identity for
  numerical reconstruction of Schrodinger potentials. DOI 10.3934/ipi.2017029.
- Li and Saad (2017), Low-Rank Correction Methods for Algebraic Domain
  Decomposition Preconditioners. DOI 10.1137/16M110486X.
- Axelsson, Karatson, Magoules (2022), Robust Superlinear Krylov Convergence
  for Complex Noncoercive Compact-Equivalent Operator Preconditioners.
  DOI 10.1137/21M1466955.
- Lai and Tseng (2005), A fast iterative solver for the variable coefficient
  diffusion equation on a disk. DOI 10.1016/j.jcp.2005.02.005.

Other new-looking routes were screened out before coding: singularity
subtraction plus RB (Kweyu et al., arXiv:2103.00245), and materialwise Kirchhoff
transformation plus nonlinear transmission (Berninger, Kornhuber, Sander,
DOI 10.1007/s10596-014-9461-8). They are not presented as new inventions.

## Reproduction

From the existing project-compatible Python environment:

    python -m unittest discover -s playground/defect_operator -p 'test_*.py' -v
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python playground/defect_operator/run.py --seed 20261001 --output-dir /tmp/defect-results

`--smoke --seed 42` uses small debug cases and omits unavailable PyAMG; it must
not be used for research conclusions. CI always runs the full predeclared study.
The workflow also rebuilds unchanged native code and runs all documented tests.
Generated scientific outputs belong outside the working tree.
