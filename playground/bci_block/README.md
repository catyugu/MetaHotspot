# Block-response BCI extraction: predeclared comparison protocol

## Scope and provenance

Production baseline: agent/work at 0b4e5874d27571488d62568d12b7a68df7656aca.
This experiment branches from the already existing, unfinished comparison branch
at 2b6efde0d25537777f677bf4d3de6fea36b3e02e. It reuses its native Case1 exports,
SharedSpace, closing response compressor, and reference solver; it does not modify
those files. Prior CI 35371039445 has successful coarse jobs but failed mid-grid
16-port extraction at the artificial snapshot cap1024 and a timed-out fine job.
Those are NOT reported as completed scientific comparisons. This screen uses the
library's actual MAX_ORDER=2048, identically for all algorithms, and records any
remaining failure or resource limit without substituting an inaccurate model.

BCI is the repository/paper terminology, not BXI. The following stock functions
are imported without patching: build_parametric_basis, port_eigenvalue_bounds,
mpmm_elliptic_shifts, project_bci, assemble_reduced_k, solve_rom_transient and
accuracy_summary. Original sources, geometry and material laws remain unchanged.

## Research questions and prior art

Full-block: does grouping residual singular input directions reduce repeated
source-wise extraction work, beyond an already shared and cached scalar tangent?
Ritz-block: can an energy-optimal rank-limited correction of all current residuals
replace converged response snapshots and reduce TOTAL offline cost?

Known ingredients, not originality claims:
- Druskin, Simoncini, Zaslavsky (2014), Adaptive Tangential Interpolation in
  Rational Krylov Subspaces for MIMO Dynamical Systems, DOI 10.1137/120898784.
- Kaouane, Jbilou (2019), A block tangential Lanczos method for model reduction
  of large-scale first and second order dynamical systems, arXiv:1903.06876.
- Li, Zikatanov, Zuo (2025), Reduced Krylov Basis Methods for Parametric PDEs,
  DOI 10.1137/24M1661236. Its stated convergence theorem has two coefficients
  and an exact fixed-parameter inverse preconditioner, not this four-term BCI
  family with multiple sources and approximate AMG.

The local Ritz energy-gain identity and response POD are standard linear algebra.
A positive timing screen would NOT establish a new theorem, optimal global ROM,
or priority over the above literature. These are new-to-project constructions
that must earn further analysis; old inverse/certificate studies are not extended.

## Algorithms

stock_fantastic: exact original library extraction, each source's spectral plan,
independent response space, random HTC probes, AMG-CG and final normalized SVD.
shared_tangent_cached: unchanged previous comparison implementation; common space,
scalar residual-SVD tangent, fully converged solve and same-sample AMG reuse.
full_block: same shared framework, take the fewest residual singular directions
capturing 90% squared singular mass, capped at four, then fully solve them using
the shared same-sample AMG. This is not a block-CG implementation; column CGs
are charged individually. Gain comes only from selection/projection batching.
ritz_block: precondition each normalized source residual once, A-deflate these
vectors against the current basis, whiten the trial space in A, then SVD W.T R.
Keep 90% of its squared singular values, capped at four. The resulting vectors
maximize the summed reduction of squared A-errors WITHIN THIS trial space at
THIS parameter/shift. Full-state errors need not be evaluated for the selection.
ritz_single: exactly the same all-source work but retains at most one direction.

Every candidate pays original spectral estimates for all ports. All candidates
use the same union spectral interval, stock elliptic points and random parameter
stream, rechecking every rejected sample until the maximum individual relative
residual is <=epsilon. Ten successful probes advance the frequency; a failure
resets the counter. Global state compression uses the same construction-only
response POD as the shared control and preserves the uniform mode. Boundary SVD
is stock epsilon=1e-3 for everyone. No test input or held-out state trains a model.
Random acceptance and final SVD do not certify continuous-parameter accuracy.

## Exact physical comparison contract

Native C++ Case1Model: original 60x100x20 mm domain, air, FR4, E-10,
anisotropic interconnect and four silicon dies. No synthetic SPD example is used
for the scientific comparison. Mesh2.5 mm has N9072; mesh1 mm has N122400 and is
the original reproduce_case1.py mesh. Mesh5 mm only establishes the baseline.
Four original source columns are primary; the 16-port extension splits each die
into four quadrants for BOTH competitors and preserves nominal total heating.

Physical HTC box [1,1e4]^2 uses the original groups (die crowns, FR4 bottom).
Use the exact repository conversion p=h/(1+h*half_cell/k) both for extraction
ranges and validation. Identical K,C,G,H and zero initial temperature RISE are
used by all models. Relative errors never include the 308.15 K ambient offset.

Mid-grid: three new seeds 20261101/02/03, independently for 4 and16 ports,
all five methods, epsilon=1e-2/1e-3/1e-4, seven HTC vectors (four corners,
nominal [50,1000], two independent log-uniform draws).
Fine grid: original seed20260805, four original ports, all methods except the
ritz_single ablation, epsilon=1e-2/1e-3, four corners plus nominal HTC.
Seeds are RNG integers, not calendar dates. One geometry is not many devices.

All cases have all-source independent steady and unit-step fields, BDF1 dt50 s
for2000 s; independent slow mixed powers use the same grid. Fine fast pulses
(dt0.1 s for10 s) are checked at nominal HTC; mid-grid fast pulses at every HTC.
Slow mixed references are exact discrete superposition of the full-order step
responses, tested against direct BDF1. Other references are sparse LU or, above
N50000, checked warm-started AMG-CG at rtol1e-10. Individual sampled residuals
must be <=1e-9. These are time-discrete FOM references, not continuum truth.

## Metrics and accounting

Primary error = maximum over ALL tests of field and per-output/input junction
relative errors. Step errors use that input's steady maximum field rise and
THAT junction/input pair's steady rise respectively. Mixed errors use actual
trace peaks, not averaged cellwise maxima. Denominators below1e-14 K are counted.
Also record peak error in K, relative peak and C-weighted L2. The nominal original
four-port combined-power steady/final metrics use stock accuracy_summary itself.

All methods use the SAME dense factored BDF1 backend for fair online timing.
The original stock AMG reduced backend is separately timed and checked at nominal
h for EVERY method/tolerance; changing this backend is not an extraction gain.
Record closure, integration, low-dimensional junction output, full recovery and
peak scan separately. Never compare junction-only time with full-field time.

Two extractions (forward/reverse method order), three online timing repeats.
Offline = all spectral estimation, AMG setup, solves/preconditioner applications,
residual checks, orthogonalization, images/projections, closing compression and
BCI projection. Each extraction runs in an isolated spawned process to bound
memory; interpreter/startup/serialization wall time is separately recorded.
Identical per-extraction wall budget:180 s mid-grid,360 s fine. If any extraction
fails, retain the failure, do not repeat that configuration, validate all OTHER
finished models, and make CI fail. The benchmark must not discard prior records.
Caches keep only one full basis per configuration; artifacts keep reduced arrays,
basis hashes and full source plus matrix/input hashes. Full bases are regenerable.

Predeclared exploration gate: <=1% worst field AND junction error; compare the
cheapest TESTED tolerance achieving this with the cheapest qualifying stock
model. Require >=2x offline improvement, <=1.25x rank/basis bytes and <=1.25x
online time under BOTH junction-only and full-field+peak contracts. This is a
fixed-grid Pareto summary, not a new training tolerance selected from holdouts.
Against the shared/cached control report paired benefits separately. A win over
stock but not this control is not evidence for the new block/energy rule.

No production merge, no hardware extrapolation, no untested geometry generality,
no physical-device guarantee, and no claim of superiority to all MOR algorithms.

## Reproduction

The workflow runs the original CMake/test/case commands and unchanged stock
numerical baseline before seven independent study jobs. Existing validated
Python3.13/NumPy2.5.3/SciPy1.18.1/PyAMG5.3.0 versions are installed only in fresh
CI venvs, with one BLAS/OMP thread. No local or project dependencies are changed.

    python -m unittest discover -s playground/bci_block -p 'test_*.py' -v
    python playground/bci_block/benchmark.py --data PATH/case1_2.5 --ports 4 --seed 20261101 --output OUT --cache CACHE

Each successful job must pass git diff --check and have an empty worktree.
