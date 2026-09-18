# Source-independent operator compression: CI research results

## Decision

This study does not extend the abandoned hotspot certificates, commutator banks,
shared-Gram fits or finite-pool spectral cubature. It investigates full-rank
approximate inverses, independent of a prescribed source snapshot space.

There is one restricted positive representation result, NOT a demonstrated fast
new solver: interpolating aligned inverse factors before forming their Gram
product preserves nonlocal cross terms and sharply reduces the optimal spectral
correction rank on the tested smooth/layered operators, across four shifts.
The five-node primary passes all of its declared small-matrix structural gates.
An unrepresented rectangular material interface remains a clear failure case.

The first practical low-rank-corrected inverse fails its speed/accuracy gate in
all 54 configurations. The later coherent representation has not yet been
implemented as a practical large-grid corrected solver. Its rank result must
not be substituted for a measured acceleration or a worldwide novelty claim.

## Provenance and repository scope

Repository: catyugu/MetaHotspot. Base agent/work is unchanged at
0b4e5874d27571488d62568d12b7a68df7656aca. Branch:
agent/defect-operator-20260918. No previous experiment was merged. Production
src/python/tests/cmake/CMakeLists.txt and dependency files are unchanged.

| Stage | Code commit | Actions run |
| --- | --- | ---: |
| Native baseline only | fa59fa7331379332eac2ab3b44f33467c5dc507e | 35331633058 |
| Initial inverse study | b8f74b6d0aef197931405051686e51ac30071045 | 35332654116 |
| Integer-input regression fix | f1b72bc506855b3901d4c7e61ee1363f3e41bc66 | 35332861187 |
| Positive inverse-mixture follow-up | e78e285fd535c39794986c10fc4739599303ca97 | 35333937196 |
| Coherent factor follow-up; all earlier studies rerun | d053164b059e05c3bce1dbf003678ff9dc7a40ed | 35334811911 |

Final numerical tree: 96416afb5795b32e15cce7827a8bb0d2ff9bdf15. The final run's ten
jobs succeeded: one native regression job and three seeds for each construction.
It passed 93 C++ tests, 12 Python API tests, eight existing case regressions and
28 new algebraic tests. Every final job passed diff/worktree checks. All ten
final artifact ZIP hashes were checked against GitHub digests; every source
archive was byte-compared with the staged source (219 files per archive).

Local GitHub DNS is unavailable. Staging was reconstructed from the exact CI
source archive; git write-tree matched the remote tree at each code commit.
This is not a conventional local clone with the remote history. No dependencies
were installed or upgraded locally. Local tests and reduced-size seed42 smoke
runs preceded the scientific CI. All added requirements had failing tests before
implementation. CI used isolated Python 3.13 venvs with previously validated
NumPy 2.5.3, SciPy 1.18.1, PyAMG 5.3.0 and pytest 9.1.1; BLAS/OMP threads were one.

The one-line integer-input fix makes the FVM diagonal explicitly floating point.
All scientific shifts were already floats, so no research matrix changed.
Each follow-up used NEW coefficient seeds and predeclared its primary, controls,
accuracy/rank gates and failure cases before CI. Earlier code and results remain.

## Literature boundary and alternative routes screened before implementation

These are established ingredients, not claims of original algorithms:

- Discrete Liouville transformations: Borcea, Guevara Vasquez and Mamonov,
  DOI 10.3934/ipi.2017029 (inverse-problem setting).
- Low-rank correction with Lanczos for approximate inverse preconditioners:
  Li and Saad, DOI 10.1137/16M110486X.
- Compact-equivalent operator preconditioning and superlinear Krylov convergence:
  Axelsson, Karatson and Magoules, DOI 10.1137/21M1466955.
- Adaptive product-convolution approximations of full-rank operators:
  Alger et al., DOI 10.1137/18M1189324.
- Piecewise interpolated approximate inverses evaluated by DST:
  DOI 10.1016/j.cam.2022.114088.
- Factorized approximate inverse preconditioners: DOI 10.1137/S1064827599356900.

Other initially attractive directions were screened out rather than implemented
and presented as new: singularity subtraction plus reduced basis already appears
in Kweyu et al., arXiv:2103.00245; subdomain Kirchhoff transformations with nonlinear
transmission appear in Berninger et al., DOI 10.1007/s10596-014-9461-8; interface-
enriched FFT thermal homogenization appears in Gehrig and Schneider,
DOI 10.1002/nme.70022. These papers do not exhaust their subjects, but they rule
out claiming their basic combinations as new simply by changing the application.

The remaining research target is much narrower: a shift-robust, coherent inverse
factor approximation with an efficient residual correction and operator-level
analysis. We have not established that this construction has no close prior art.

## Common mathematical object and exact small-matrix gate

For an SPD shifted diffusion matrix A and a fast inverse backbone P=Q Q^T,
let H=Q^T A Q. With exact eigenpairs H U=U Lambda, define

    R_r = P + (Q U_r)(Lambda_r^-1-I)(Q U_r)^T.

Selecting the r largest |lambda_i-1| gives the best rank-r update in the
energy-scaled inverse norm, with error max_omitted |lambda_i-1|. This is standard
best-rank approximation in energy coordinates, NOT a new theorem. Two stationary
inverse applications have error matrix (I-A^(1/2) R_r A^(1/2))^2. Thus a 1% target
requires removing eigenvalue deviations larger than 0.1. Full numerical spectra
on small meshes determine optimal ranks, rather than testing one vector heuristic.
They are CORRECTION RANKS, not dimensions of a source-trained ROM.

All systems are dimensionless 2-D cell-centred harmonic FVM on [0,1]^2 with
homogeneous Dirichlet conditions on four sides and capacity I. They are not the
native 3-D assembler or calibrated devices. Conductivity is a fixed smooth
analytic field a(x,y), optionally multiplied by flat layers 1/30/3 at y=1/3,2/3.
The inclusion control adds an unrepresented 20-fold rectangular jump. Coefficients
and interfaces are fixed under refinement; seed changes two smooth field phases.
README.md specifies the exact formula and discretization.

A separable layered inverse factor uses orthonormal x DST-II and y tridiagonal
Cholesky. A square-root conductivity scaling matches the smooth principal part,
while declared layers are retained explicitly. The Liouville identity motivates
this but never replaces the actual FVM matrix in a comparison.

## 1. Principal-part and jump-matched inverse

Protocol: seeds 20261001-03, shifts 0/20/2000. Full-spectrum meshes n=12/24/36;
practical meshes n=48/96. Four backbones isolate homogeneous reference, raw
layers, uniform-reference square-root scaling, and scaled known layers (primary).

The following are three-seed medians of optimal correction rank for 1% energy
error after TWO inverse applications, on layered coefficients.

| Shift | n / unknowns | Raw layers | Scaled uniform | Scaled known layers |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 12 / 144 | 120 | 49 | 7 |
| 0 | 24 / 576 | 490 | 98 | 7 |
| 0 | 36 / 1296 | 1100 | 146 | 7 |
| 20 | 12 / 144 | 120 | 53 | 10 |
| 20 | 24 / 576 | 490 | 101 | 10 |
| 20 | 36 / 1296 | 1099 | 149 | 10 |
| 2000 | 12 / 144 | 88 | 102 | 86 |
| 2000 | 24 / 576 | 456 | 296 | 241 |
| 2000 | 36 / 1296 | 1066 | 481 | 362 |

All six low/moderate-shift seed cases pass the structural gate; the three
high-shift cases fail. The inclusion at shift zero instead needs ranks 35,73,109
as n increases. A material interface cannot simply be treated as a harmless
low-rank residual. Finite-grid rank growth at fixed large shift does NOT prove
asymptotic divergence as mesh size tends to zero.

### Actual practical costs, not oracle costs

Matrix-free Lanczos on H-I starts at 8 vectors, doubles to cap64, and retains
observed deviations >0.09. All repeated operator actions are charged; no A^-1
source-training solves are used. Rayleigh-Ritz re-evaluation defines the small
inverse update. An observed tail is NOT a verified completeness certificate.
Primary: two inverse applications; one/three and no-correction variants are
predeclared ablations, not replacement winners.

Each case has 12 held-out RHSs generated after construction: three smooth,
three signed random and six point sources. Point positions are not nested across
meshes because their RNG follows mesh-sized random vectors; mesh consistency
refers to coefficient fields. Reference sparse-LU residual must be <1e-9.
Controls are raw-layer, scaled-layer and AMG-PCG with rtol=1e-2/1e-6, plus sparse LU.
Three timing repeats, alternating order, warmup for every method; RHSs solved
individually. Setup, full-vector actions, corrections and residuals are charged.
Shared assembly and independent reference validation are excluded.

Tables in this subsection use run 35332861187. NONE of 54 cases passes both
<=1% maximum energy error and >=2x speed over the cheapest accuracy-qualified
iterative control. The 24 smooth/layered low/moderate-shift cases pass accuracy,
but their median paired speed is 1.233x, maximum 1.420x. All 30 remaining cases
fail accuracy with cap64. Cold 12-query cost loses in every case.

On n96, representative three-seed medians are:

| Family | Shift | Rank | Max RHS energy error | Iterative speed | LU speed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smooth | 0 | 5 | 0.2504% | 1.232x | 0.341x |
| Layered | 0 | 10 | 0.4563% | 1.369x | 0.330x |
| Layered | 20 | 12 | 0.2874% | 1.391x | 0.342x |
| Layered | 2000 | 64 | 1.7411% | 1.096x | 0.304x |
| Inclusion | 0 | 64 | 2.8023% | 3.415x | 0.308x |
| Inclusion | 2000 | 64 | 18.0428% | 1.199x | 0.312x |

Apparent inclusion speed is not a success because accuracy fails. Sparse LU
wins at these 2-D sizes. Across n96 smooth/layered cases, median per-RHS costs
are 2.636 ms (primary), 3.178 ms (loose AMG-PCG), and 0.871 ms (LU); setup medians
are 0.149 s, 0.00976 s and 0.01974 s respectively. No hypothetical compiled
speedup is counted. Extrapolated 100/1000-query totals are labeled as such.

## 2. Positive inverse mixture: scalar accuracy can destroy nonlocal coupling

Independent seeds 20261011-13, same three families and n=12/24/36, shifts
0/20/2000/200000. Write k=a*layer. With geometric positive nodes a_j and
nonnegative weights sum(w_j)=1, sum(w_j/a_j)=1/a(x), test

    P_s = sum_j sqrt(W_j) (a_j K_layer+sI)^-1 sqrt(W_j).

Five-node harmonic interpolation is primary. Arithmetic five-node and harmonic
nine-node controls are fixed before results. Frozen scalar interpolation matches
two limits and has a directly derived relative error bound; it is NOT a spatial
operator bound. Dense inverse formation and Cholesky below are oracle costs only.

At n36, three-seed median optimal ranks (two applications, 1% target) are:

| Family | Shift | Single scaled | Harmonic5 | Arithmetic5 | Harmonic9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smooth | 0 | 4 | 92 | 112 | 312 |
| Smooth | 2000 | 475 | 0 | 0 | 77 |
| Layered | 0 | 6 | 90 | 110 | 308 |
| Layered | 2000 | 262 | 26 | 37 | 187 |
| Layered | 200000 | 914 | 0 | 0 | 0 |
| Inclusion | 0 | 108 | 117 | 1015 | 146 |
| Inclusion | 2000 | 377 | 107 | 793 | 114 |

The complete high-shift gate is passed in 5/6 smooth/layered seed cases, not all:
one layered case grows from rank11 to25, above the predeclared factor-two limit.
Five factors cost approximately five times the single backbone per application.
Nine nodes make the low-shift operator worse despite refining the scalar fit.

This has an exact explanation. At s=0, let G=K_layer^-1. The i,k entry is

    G_ik * sum_j sqrt(w_j(i)*w_j(k))/a_j.

Disjoint coefficient interpolation supports force this entry to ZERO although
elliptic propagation is nonlocal. Refining intervals can eliminate additional
cross-coefficient couplings. This obstruction is algebraic, not an optimizer
failure. It motivates changing the representation rather than raising rank.

## 3. Coherent inverse-factor interpolation: restricted structural success

Use constant-coefficient inverse factors in a COMMON DST/triangular gauge,
(a_j K_layer+sI)^-1=Q_j Q_j^T. Interpolate in 1/sqrt(a), obtaining positive
weights sum(v_j)=1 and sum(v_j/sqrt(a_j))=1/sqrt(a(x)). Define

    Q_s = sum_j diag(v_j) Q_j(s),       P_s = Q_s Q_s^T.

Unlike the mixture, this retains cross-factor products. Applications use J
factor/adjoint pairs, not J^2 explicit cross matrices. It is PSD by construction;
strict positivity was checked numerically, not proved for arbitrary coefficients.

Two exact limits hold: at s=0, Q_j=a_j^-1/2 Q_layer, so P_s equals the original
scaled layered inverse for ANY node count; as s tends to infinity, all factors
approach the same gauge times s^-1/2, so s*P_s tends to I. These limits alone do
not establish accuracy at intermediate shifts. They concern the approximate
backbone, not exact equality to the original variable-coefficient FVM inverse.

New seeds 20261021-23; n=12/24/36 and all four shifts retained. Primary is five-node
coherent root-coordinate interpolation. Matched controls isolate coherence and
coordinate choice separately: incoherent/root weights, coherent/harmonic weights,
the old incoherent/harmonic mixture, single scaled inverse, and coherent/root9.
No source snapshots, held-out solution, or tuned spectrum selects the weights.

At n36=1296 unknowns, the three-seed median optimal correction ranks are:

| Family | Shift | Single | Incoherent harmonic5 | Incoherent root5 | Coherent harmonic5 | Coherent root5 | Coherent root9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smooth | 0 | 4 | 89 | 93 | 5 | 4 | 4 |
| Smooth | 20 | 7 | 85 | 89 | 2 | 3 | 3 |
| Smooth | 2000 | 550 | 0 | 0 | 0 | 0 | 0 |
| Smooth | 200000 | 1028 | 0 | 0 | 0 | 0 | 0 |
| Layered | 0 | 8 | 87 | 90 | 7 | 8 | 8 |
| Layered | 20 | 9 | 85 | 89 | 7 | 8 | 8 |
| Layered | 2000 | 321 | 33 | 36 | 2 | 2 | 2 |
| Layered | 200000 | 949 | 0 | 0 | 0 | 0 | 0 |
| Inclusion | 0 | 109 | 116 | 123 | 111 | 109 | 109 |
| Inclusion | 2000 | 425 | 106 | 103 | 109 | 104 | 106 |
| Inclusion | 200000 | 1105 | 62 | 56 | 65 | 65 | 64 |

All SIX predeclared smooth/layered seed-family gates pass: maximum primary rank
across four shifts is <=8 (gate15), high-shift rank reduction is >=4x, and the
zero-shift limit agrees with the single-scaled construction to roundoff. Across
all 24 primary smooth/layered n36 cases, ranks range from zero to eight. At s2000,
layered ranks are 2/2/1 versus single ranks321/367/300. These are complete numerical
spectra, not a favorable selection of test source responses.

Primary median ranks across n12,n24,n36:
- smooth s0: 5,4,4; s20: 2,2,3; s2000: 0,0,0;
- layered s0: 8,8,8; s20: 7,8,8; s2000: 1,2,2;
- inclusion s0: 36,73,109; s2000: 24,65,104.

The matched controls show that COHERENCE accounts for the major improvement;
root-coordinate weights are not uniformly more accurate than harmonic weights.
Their specific advantage is exact retention of the zero-shift scaled backbone.
Do not claim a general accuracy superiority for the new interpolation coordinate.

Cost/guarantee limits are material. At n36 the five-node primary factor application
median is 1.385 ms, versus 1.366 ms for the five-node old mixture, 0.274 ms for the
single inverse and 2.517 ms for nine nodes. This is roughly five times the single
application, not a free rank reduction. No practical matrix-free corrected solve,
setup break-even, or CG/LU speed comparison was run for this third representation.
The dense oracle is not an efficient construction algorithm.

Without the spectral correction, the primary's worst-case two-application energy
error at n36 is approximately 10.6-12.1% for smooth s0 and 20.8-25.9% for layered
s0. It is therefore incorrect to claim that the five-factor inverse by itself
already achieves 1% on these cases. The low optimal correction rank is what was
established. At s2000 the uncorrected errors are 0.098-0.124% (smooth) and
1.36-2.06% (layered). The inclusion remains badly represented and can diverge
under uncorrected stationary refinement.

## Auditing, counts, and limits of interpretation

Final run 35334811911 reran all studies. Original oracle ranks and primary inverse
errors agree with their earlier runs to numerical precision; the blend spectra
also reproduce. One seed's jump-aware PCG controls on inclusion cases differ in
tolerance-stopped trajectories: maximum absolute relative-energy-error difference
is 5.06e-4. This did not change the 0/54 practical-gate result. The precise
floating-point cause was not isolated. Do not claim all iterative outputs are
bitwise reproducible across runner hardware. Raw data from both runs are retained.

There are 324 first-study oracle rows, 594 practical method/configuration rows,
432 mixture-oracle rows and 648 coherent-oracle rows. These are 1998 configuration
rows, NOT 1998 independent devices. First-study practical rows each contain 12
source checks. Only three analytic family templates and three seeds per stage
are tested. Seed changes are smooth-field phases, not arbitrary geometries.

The final ten artifacts each include source, environment, logs and clean worktree
records; detailed hashes and aggregation scripts are in the downloadable bundle.
Actions artifacts have 30-day retention. The committed report preserves all
negative results alongside the restricted positive result.

No 3-D scaling, million-DoF computation, temperature-dependent material law,
arbitrary tensor conductivity, moving/non-flat interface treatment, physical
maximum principle, continuous-time bound or nonlinear trajectory guarantee has
been established. Unit-square shifts are dimensionless, not a device bandwidth.
Oracle eigenvalues use floating point, not formal interval arithmetic. General
sublinear-in-N online complexity is not claimed; full-vector transforms remain.

Recommended next decision: retain coherent factor interpolation as a narrowly
supported ALGORITHMIC REPRESENTATION candidate, not a production merge or a
six-month commitment. Before scaling it, it needs an intermediate-shift operator
error analysis, explicit factor-gauge assumptions, comparison to close interpolated
inverse/factorized-inverse literature, and a practical corrected implementation
that beats equally accurate strong solvers after setup and every factor action
are charged. More nodes or more modes alone would not establish originality.
