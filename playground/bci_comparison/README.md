# Direct BCI FANTASTIC comparison: two new-to-project extraction directions

## Baseline first, not an inverse-solver surrogate

Base is agent/work at 0b4e5874d27571488d62568d12b7a68df7656aca. The name in the
repository and supplied papers is BCI (boundary-condition independent), not BXI.
The baseline invokes `metahotspot.macromodel.utils.build_parametric_basis`
WITHOUT replacing its eigensolvers, frequency selection, local response spaces,
random probes, AMG-CG solves, closing SVD or constant-mode preservation. All
methods use the unchanged `project_bci` and `assemble_reduced_k` with boundary
SVD tolerance 1e-3. The stock `solve_rom_transient` is separately timed and
compared numerically with the common dense BDF1 backend for EVERY method.
Swapping a small-system solver is not credited as an extraction innovation.

The first CI 35368950186 runs unchanged native/C++/Python/case regressions, then
extracts the stock four-port model (epsilon=1e-3, 10 probes, seed20260805) on the
coarse native Case1 mesh. Its baseline has N=1320, 116 full response solves,
rank35 and 0.6798% nominal steady full-field rise error. These numbers are a
measured starting point, not the tolerance or outcome of the new experiments.

## Literature and originality boundary

Two actual algorithmic questions are tested:

1. Can a globally shared response space use residual singular directions to
   compress input-port work *while it is generated*, rather than performing
   independent source loops followed by the stock closing SVD?
2. Can adaptive, multi-parameter reduced Krylov bases retain useful directions
   from only four PCG iterations and avoid fully converged snapshot solves?

These are new experiments in this project, NOT a claim that tangential Krylov,
shared bases, inexact solves, response POD, caching or PCG basis reuse is new.
The close existing work includes:

- Druskin, Simoncini, Zaslavsky, Adaptive Tangential Interpolation in Rational
  Krylov Subspaces for MIMO Dynamical Systems, 2014, DOI 10.1137/120898784.
- Li, Zikatanov, Zuo, A Reduced Conjugate Gradient Basis Method for Fractional
  Diffusion, 2024, DOI 10.1137/23M1575913, arXiv:2305.18038.
- Reduced Krylov Basis Methods for Parametric Partial Differential Equations,
  2025, DOI 10.1137/24M1661236. Its abstract's stated convergence result assumes
  two parameter coefficients and an inverse-at-a-fixed-parameter preconditioner;
  this study has K+sC+p1H1+p2H2, multiple sources, and AMG approximations. We do
  not claim its theorem transfers automatically.
- Baur et al., Interpolatory Projection Methods for Parameterized Model
  Reduction, 2011, DOI 10.1137/090776925.
- Gosea, Gugercin, Unger, Parametric model reduction via rational interpolation
  along parameters, arXiv:2104.01016. Parameter-dependent spaces were screened
  as existing methodology rather than relabeled as an original idea.

The intended research frontier would be an efficient, rank-adaptive construction
and convergence/work analysis for many-port BCI families. This finite experiment
cannot establish such a theorem or worldwide originality. Even a positive
speed comparison would not by itself constitute the intended academic result.
The earlier hotspot, commutator/Gram/cubature and inverse-factor experiments are
not extended, merged or used as baselines here.

## Implemented algorithms and controls

`stock_fantastic`: exact existing library call, epsilon in {1e-2,1e-3,1e-4},
probe_rounds=10 (the Case1 reproduction script's setting), max_order=1024.

The four shared-space methods all pay for the stock per-source spectral
estimation, then use the SAME union spectral interval, stock elliptic shifts,
all-port residual acceptance and common closing response compression. At a
sample (s,p), solve the small projected system for every normalized source and
form R=G_normalized-A V Y. Accept only when the maximum individual column norm
is <=epsilon. This remains random parameter probing, not a domain certificate.
Otherwise enrich at that same sample until acceptable; after failure reset the
10-success counter, as in the stock state-machine intent.

- `shared_column`: choose the worst individual residual column and fully solve.
  This isolates source sharing from singular-direction selection.
- `shared_tangent`: choose a leading right singular vector of R, then fully
  solve for that source combination at stock ENRICH_RTOL=1e-6.
- `shared_tangent_cached`: identical selection and full-solve accuracy, but
  reuse the same AMG hierarchy while a sample is being repaired. This prevents
  crediting mere hierarchy reuse as an advantage of incomplete Krylov solves.
- `krylov_tangent`: same leading residual direction, but retain up to four PCG
  search vectors for the correction equation. No exact snapshot is claimed.
  Ritz projection in the enlarged common space determines whether to continue.

All shared methods use normalized projected responses over their own recorded
CONSTRUCTION parameter/shift points for a closing POD. Covariance is assembled
in the common space, so there are no hidden full-order solves. All compression,
operator images, orthogonalization and projections are timed. The uniform mode
is preserved. This compression is not the stock snapshot SVD; the shared-column
control has the same compressor, preventing an attribution solely to tangents
or Krylov directions. No validation point enters construction/compression.

## Physical comparison contract

Use the existing `playground/bci_rom_testcase1/model_case1.py` with its native
C++ assembler, materials, geometry, air background, source positions, and
boundary groups. Model both 5 mm and 2.5 mm cell limits in x/y/z (N=1320/9072).
The original reproduction script uses 1 mm, so this first screen is NOT a claim
to have rerun that exact fine-mesh reproduction. No synthetic random SPD matrix
is used as the scientific thermal benchmark.

The primary case has the four ORIGINAL independent source shapes. A separate
many-port extension splits each die into four independent quadrants, hence 16
sources. Both competitors receive all sixteen shapes, and their nominal sum
is asserted equal to the original heat load. This is not an unannounced unseen
source or a restriction of the baseline's input information.

Physical HTCs are [1,1e4] W/m2/K for die crowns and FR4 bottom. Native geometry
supplies half_cell/k; all cells in each group are asserted to share that ratio.
Every solver uses p=h/(1+h*half_cell/k), the repository's effective-coefficient
contract, for both training ranges and reference queries. A heat source f=G P
and zero initial temperature RISE are identical for all methods. No ambient
Kelvin offset may enter a relative-error denominator.

Seeds 20260931,20260932,20260933 are arbitrary RNG integers, not dates. Every
seed uses five fixed physical HTC vectors (all four corners plus [50,1000]) and
five separate log-uniform holdouts. Training and validation RNG streams differ.
Only three construction/input seeds on one physical geometry are tested.

For every HTC, compare:
- All single-source steady fields, including weak sources individually.
- All single-source step response fields: dt=50 s, duration=2000 s, as in the
  stock Case1 reproduction script's transient grid.
- Simultaneous independent slow mixed powers on the same grid.
- Independent fast pulsed powers: dt=.1 s, duration=10 s.

References use native-exported K,C,H,G and sparse-LU backward Euler, with sampled
linear residual <=1e-9. This is the same time-discrete equation as the stock
BDF1 ROM, NOT continuous-time ground truth. Common model assembly is excluded;
reference setup and solve are recorded but are not the main speed comparator.
The imported commercial ROM is not treated as truth: a different discretization
or geometry would confound the reduction error. We compare the actual repository
Extended-FANTASTIC implementation against the SAME native FOM.

## Metrics, cost, and predeclared gate

Report every configuration, not only favorable seeds. A source-step field error
is max over time/cells of absolute rise error divided by that source's maximum
steady reference rise; then take the maximum over sources. Junction responses
are normalized analogously, so stronger source ports do not hide weaker ones.
Mixed-profile errors use the maximum reference rise during that profile.
Also record peak-temperature error in K, capacity-weighted L2 error, steady
boundary heat-flow error, rank, boundary rank and explicit basis bytes.

Each extraction is run twice, in forward then reverse method order. Report both
orders/ranks and the median total extraction+BCI projection cost. Every method
uses the same dense, factored BDF1 online solver; online parameter closure,
small solve, low-dimensional junction output, full-field recovery and maximum
scan are recorded separately (three repeats). A speedup that removes full-field
recovery from one method but includes it for another is prohibited. The native
stock small-system backend is a separate compatibility/control measurement.

Primary application target is <=1% worst full-field AND junction error across
all listed steady/transient cases. For each method choose the cheapest *tested*
extraction tolerance that meets this target, then compare against the cheapest
qualifying STOCK tolerance, not necessarily stock epsilon=1e-3. This is a stated
Pareto summary of a fixed grid, not holdout-driven parameter tuning and retraining.
A promising gate requires >=2x total offline improvement, rank/basis storage
<=1.25x the qualifying stock model, and no >25% online regression for either
junction-only or full-field output contracts. If no stock or candidate qualifies,
report it explicitly rather than treating a coarse inaccurate model as a win.

No automatic production merge and no claim of continuous-parameter assurance,
nonlinear material capability, full-world originality, or many-device validation.

## Reproduction

The workflow exports native matrices, then six isolated jobs run:

    python playground/bci_comparison/benchmark.py --data PATH/case1_2.5 --seed 20260931 --output OUT
    python -m unittest discover -s playground/bci_comparison -p 'test_*.py' -v

CI uses the already validated Python3.13 / NumPy2.5.3 / SciPy1.18.1 / PyAMG5.3.0
versions in a fresh venv; no production dependency file is changed. Source,
operator hashes, individual extraction logs, reduced bases, raw error/timing
records, environment and clean-worktree audits are included in artifacts.
