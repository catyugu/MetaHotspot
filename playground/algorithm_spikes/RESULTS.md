# Algorithm feasibility results - 2026-09-18

## Decision

None of the three TESTED CONSTRUCTIONS passes its predeclared advantage gate.
Do not promote these implementations into production or claim a new algorithmic
advantage. This conclusion is narrower than rejecting every possible algorithm
in the three research directions. The implemented primitives are:

1. A fixed second-order filtered-commutator candidate bank, not a complete
   adaptive noncommutation-defect algorithm or a new MOR complexity theorem.
2. Fixed-rank nonconvex shared-Gram-factor fitting, not convex positive-kernel
   completion, certified rank revelation, or a minimal realization algorithm.
3. Spectral support/weight exchange over a finite candidate-state pool, not a
   state-domain-wide separation oracle or uniform spectral certificate.

The baseline methods often work better. No experimental holdout was used to
change model parameters, budgets, objectives or source selection rules.

## Repository and execution provenance

Repository: `catyugu/MetaHotspot`.
Base: `agent/work` at `0b4e5874d27571488d62568d12b7a68df7656aca`.
Branch: `agent/algorithm-spikes-20260918`.
Final numerical code: `9e6fdbc2c5e70fa65758ab2f435adfec19af7010`.
Final code tree: `189383937bae9f9658dcbd23936b20f9b9cea889`.

| Stage | Actions run | Outcome |
| --- | ---: | --- |
| Unchanged production baseline | 35326129802 | success |
| Initial three-direction screen | 35327156691 | one spectral job failed numerically |
| LP retry plus primal checks | 35327920151 | three spectral jobs failed primal checks |
| Column-equilibrated LP, actual epigraph reconstruction | 35328615673 | all ten jobs succeeded |

The final run contains one native regression job and nine numerical jobs:
three directions times seeds 20260921, 20260922 and 20260923. It passed 93 C++
tests, 12 Python API tests, eight existing case regressions, and 22 new
algebra/Jacobian/solver regression tests. Every successful final job passed
`git diff --check` and ended with a clean worktree. The native job verified
that src, python, tests, cmake and CMakeLists.txt are unchanged against the base.
The shared `agent/work` branch has not been modified or merged.

CI: Ubuntu 24.04, Python 3.13.15, NumPy 2.5.3 and SciPy 1.18.1; native regression
also uses pyamg 5.3.0 and pytest 9.1.1. BLAS/OMP thread counts are one. Each job
uses an isolated venv, reusing previously validated versions without changing
project dependency files. Full source archives in all ten final artifacts were
byte-compared with the locally staged, Git-tree-verified source. Each downloaded
artifact SHA256 matched the digest returned by GitHub.

There are 804 raw method/configuration rows: 540 commutator, 48 Gram and 216
spectral. These are not 804 independent physical systems. They are controlled
synthetic algebra-level tests, with only three random seeds per direction.
Numerical portions of individual jobs took about 3.5 s (commutator), 17-19 s
(Gram), and 7-8 s (spectral); native builds and runner setup are separate.

## 1. Filtered commutator bank: no advantage

### Controlled question

A 48-state symmetric dissipative affine family has two input ports. Fixed
individual spectra and common inputs isolate the effect of operator rotation.
The experiments include commuting controls, low-rank commutators with full-rank
parameter differences, and dense-commutator controls. Final ROM ranks are 10
and 14. Snapshot variants share a six-vector frozen-response core.

Commutator, ordered-action and symmetric-action candidates come from exactly
the same computed bank. Each is charged 96 column solves, 96 operator-column
actions and 24 factorizations. The further-frozen-response comparator has the
same 96 column solves but 48 factorizations and no extra operator actions;
these are not identical flop budgets. A generalized bilinear Gramian comparator
is independent and not cost-matched. Its conservative bilinear input scaling is
recorded, while all validation uses the original unscaled family.

Two previously unseen forward/reversed 24-segment paths are evaluated with
three segment lengths and exact symmetric-exponential/phi-function steps.
There are also 18 held-out frozen frequency/parameter pairs. Homogeneous path
reversal is measured separately, with no changing forcing, to isolate ordering.

### Results

The following are medians over 72 matched NONCOMMUTING configurations (three
seeds, two structural families, two nonzero rotation amplitudes, two ROM ranks,
three segment lengths). Each configuration's error is the worse normalized
L2 trajectory error of its two test paths. These percentages concern states
and collocated outputs, not peak physical temperature.

| Method | State relative L2 error | Output relative L2 error |
| --- | ---: | ---: |
| Filtered commutator bank | 10.8282% | 1.35968% |
| Same-bank ordered actions | 9.15271% | 1.09954% |
| Same-bank symmetric actions | 9.03757% | 1.01421% |
| More frozen responses | 3.56740% | 0.177305% |
| Generalized bilinear Gramian subspace | 2.38494% | 0.0754281% |

The commutator method beats the ordered-action control in only 7/72 matched
configurations and the further-frozen-response control in 0/72. Its paired
median error ratios are 1.1291 versus ordered actions and 3.2249 versus more
frozen responses. It wins 0/72 against the best applicable control in each
configuration, with paired median ratio 5.0997. This is a ratio of matched
errors, not a ratio of the medians in the table.

Even in the low-rank-commutator family, the median error ratio is 1.1646 versus
ordered actions and 3.2249 versus more frozen responses. The algebraic structure
alone did not translate into useful additional state directions. At rotation
amplitude 0.8 the filtered commutator bank has measured rank 16 at relative
cutoff 1e-6 in both structural families; frequency filtering/concatenation does
not automatically retain the raw commutator's low rank. The commuting control's
largest homogeneous reversal effect is 8.63e-17, validating the negative control.

At N=48, median construction times are about 3.78 ms for the commutator bank,
3.93 ms for ordered actions, 5.82 ms for more frozen responses, and 1.96 ms for
the dense generalized Gramian. These tiny-matrix timings are not evidence about
large sparse offline scaling. Basis storage is 3,840 or 5,376 bytes.

### Interpretation and limits

This rejects the tested rule of selecting dominant filtered commutator columns
as a superior allocation of the additional basis budget. It does NOT establish
that a fully adaptive defect-driven construction cannot help. In particular,
the six-vector frozen core was fixed rather than first converged to a small
common frozen-response error. Thus this is not a clean theorem-level test of
"extra order after all frozen responses are already accurate." The current
construction does not use an adaptive nested-commutator stopping rule.

BIRKA, optimized nice selections and low-rank envelope-system implementations
were not reproduced. The generalized-Gramian comparator is not relabeled as
BIRKA. Failing the current controls is already enough to avoid promoting this
specific candidate-bank heuristic; it is not a comprehensive literature ranking.
The matrices are dimensionless SPD examples, not calibrated thermal M-matrices.

## 2. Shared positive Gram factors: cost and conditioning problems

### Controlled question

An eight-state positive SISO pencil supplies only 24 transfer-function samples.
Both competitors fit the same K0=L0 L0.T+1e-6 I, K1=L1 L1.T, C=I structure;
neither can use full-state responses or true matrices. The kernel objective adds
latent response coordinates Y and fits the common Gram identities. The direct
control fits transfer values using the SAME positive pencil parameterization.

Two ranks (2,4), two noise levels (0,1e-4 relative), two initializations and a
200-function-evaluation cap per start are fixed. The selected start minimizes
that method's own training objective, never the 180-point validation error.
Analytic Jacobians are finite-difference tested. All-start time is charged.
Equal evaluation caps are not equal arithmetic work: the kernel method has
56 versus 8 variables at rank 2 and 120 versus 24 at rank 4.

### Two-dimensional-grid results

Each error below is the median across three seeds of the MAXIMUM relative
error over the independent off-grid holdout. Fitting time also includes both
initializations. These medians should not hide the severe seed-specific failure
reported immediately below.

| Rank | Relative noise | Direct error | Gram-factor error | Direct fit | Gram-factor fit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 1.54935% | 2.66688% | 0.2144 s | 0.6859 s |
| 2 | 1e-4 | 1.57318% | 2.66239% | 0.1291 s | 0.7980 s |
| 4 | 0 | 0.198491% | 0.604977% | 0.8312 s | 3.1209 s |
| 4 | 1e-4 | 0.181556% | 0.874132% | 0.8181 s | 3.2905 s |

Across the 12 matched two-dimensional-grid configurations, the kernel construction
wins only 2/12 on maximum error (also 2/12 on RMS error). Its paired median
maximum-error ratio is 1.8377 and its paired median all-start cost ratio is
3.7656. Neither global optimality nor fully converged rank-4 fits is claimed:
all selected rank-4 runs hit the evaluation cap for BOTH methods.

Seed 20260921 is particularly informative. At rank 4, kernel maximum error is
27.10% without noise and 35.32% with noise, while the direct control gives
0.1568% and 0.1682%. The fitted latent matrices have condition numbers about
4.75e4 and 4.88e4; relative latent equation defects are 1.823 and 2.081. This
shows why a small projected/Gram discrepancy cannot be silently equated with a
small state-equation discrepancy when the latent rows become nearly dependent.
It is consistent with a conditioning/optimization difficulty, not a proof of a
fundamental approximation limitation.

### Correlated-data control

The two-dimensional grid is not a proof of global parameter identifiability.
Both methods can fit a correlated frequency/parameter path while extrapolating
badly off it. In the separate exact two-state counterexample, the two systems'
diagonal responses agree to 2.22e-16, while their off-diagonal responses differ
by up to 23.78%. Positive structure does not resolve absent information.
This control is not counted as evidence that either fitter should infer an
unidentifiable model. The kernel has no reliable advantage on these fits either.

### Interpretation and limits

The naive latent-factor Gram formulation is not a competitive numerical
replacement for direct positive-pencil fitting under the tested budget.
Passivity is not a distinguishing success because the direct control has the
same PSD parameterization. A genuine positive-kernel completion/rank-revelation
algorithm has NOT been implemented or invalidated here. No claims are made
about full structured-Loewner baselines, global minimal order, MIMO performance,
or full-field temperature recovery. Sample acquisition is common and excluded
from fitting times; no end-to-end simulator speedup is claimed.

## 3. Finite-pool spectral exchange: compression works, new selection does not

### Controlled question

A fixed six-dimensional smooth basis is applied to square grounded graphs with
312, 1,200 and 4,704 edges. Positive endpoint conductivity laws are nonlinear;
harmonic averaging gives positive non-affine edge conductances. Mild and strong
three-material slope regimes are used. These are synthetic laws, not calibrated
materials, and the grids are not a mesh-convergence sequence for identical
random coefficient fields.

All methods share the same 48 candidate states. Holdout contains 192 independent
interior states plus all 64 vertices of [-1,1]^6. No holdout/vertex is used for
selection. Edge budgets are 12,24,48, with actual positive support recorded.
The strongest baseline fits all whitened stiffness entries with positive
weights. A fixed-support spectral-minimax refit isolates a change of objective
from a change of edge selection. Nominal-leverage/NNLS is an additional control.

### Results on the largest graph

The following are medians across three seeds of maximum held-out generalized
spectral relative error. The error is the largest absolute generalized
eigenvalue of (Ktilde(q)-K(q),K(q)), maximized over held-out states; it is not a
trajectory or maximum-temperature error.

| Construction | Mild, 24 edges | Mild, 48 edges | Strong, 24 edges | Strong, 48 edges |
| --- | ---: | ---: | ---: | ---: |
| Full-matrix positive cubature | 7.827% | 1.409% | 37.554% | 14.097% |
| Same support, spectral minimax weights | 8.556% | 4.365% | 63.859% | 31.429% |
| Spectral support exchange | 9.135% | 6.452% | 66.249% | 38.076% |
| Nominal leverage plus NNLS | 35.138% | 6.698% | 88.759% | 37.011% |

Across the 24 predeclared larger-grid (nx>=24), larger-budget (>=24) cases,
spectral exchange reaches <=5% held-out spectral error only once. It beats the
fixed-support minimax control in 6/24 cases and never reduces that control's
error by a factor two. Its paired median ratio to that control is 1.1626.
At nx=48 and budget 48 it retains only 32-38 positive edges, but the reduced
support is not an accuracy success: the same-budget full-matrix control is
substantially more accurate. All sampled approximate matrices remained positive;
the minimum recorded eigenvalue over all methods/configurations was 0.01256.

Online evaluation at nx=48 is indeed cheaper. At the 48-edge budget, exchange
has median speedups 7.89x (mild) and 7.76x (strong); the full-matrix positive
cubature control achieves 7.61x and 7.40x while being more accurate. Across both
regimes the full already-reduced operator takes a median 182.0 microseconds per
evaluation, versus 23.3 microseconds for exchange. These timings include local
state evaluation, conductances and assembly on selected edges. They are NOT
speedups over a full PDE solve or a nonlinear transient simulation.

Exchange's median offline cost at nx=48/budget48 is 0.1085 s (mild) or 0.1037 s
(strong), corresponding to about 678 or 656 operator calls before amortization.
The full-matrix positive control costs 0.1269 s or 0.1340 s and amortizes in about
868 or 880 calls. An approximately 0.02-0.03 s setup reduction does not establish
a useful tradeoff when held-out errors grow by several times. Small-edge online
storage and speed gains are shared by the baseline; they are not attributable
to the proposed spectral-selection principle.

### Numerical failures were not removed from the record

The first run had one HiGHS failure on an explicitly feasible minimax LP.
The next patch added a retry and exposed further solver-reported epigraph
residuals. Final code equilibrates columns (the same LP under positive variable
scaling), reconstructs a feasible epigraph on original coefficients, and records
any repair. In the complete final run: zero solver retries were needed, maximum
epigraph repair was 4.57e-10, and maximum negative-weight projection was zero.
Both failed-run logs and all three added numerical regression tests are retained.
No case, edge budget or nonlinear law was dropped to make the run pass.

### Interpretation and limits

Finite-pool spectral support exchange did not outperform the existing-style
positive full-matrix cubature control. Changing to a minimax objective on the
training pool also did not reliably help the unseen corners. This does NOT
establish a uniform-domain spectral inequality, its sample complexity, control
of coefficient derivatives, or a nonlinear trajectory-error bound. No such
claims should be inferred from sampled positivity or from fast edge evaluation.
A future state-domain oracle with constitutive-law guarantees would be a
materially different algorithm, not an already demonstrated result here.

## Overall limitations and next decision

These screens provide reasons to stop the current three constructions, not a
proof that their mathematical themes are unproductive. The most important gaps
between the proposals and implemented prototypes are explicit above. More modes,
more optimizer iterations or more candidate states, without a new construction,
would not by themselves establish the intended algorithmic contribution.

Peak process RSS across jobs was about 66-67 MiB (commutator), 85 MiB (Gram),
and 198 MiB (spectral). These include validation arrays and fit workspaces and
are NOT per-method memory measurements. No million-DoF, physical-device,
continuous-domain guarantee or production-backend timing has been established.

Recommendation: archive this screen on its experiment branch; do not merge or
start larger FVM benchmarking solely because one of the formulations looks
mathematically attractive. Any further research should first supply the missing
algorithmic step (adaptive action selection, rank-revealing kernel completion,
or a non-enumerative state-domain separation oracle), then test that step rather
than relabeling these prototypes as the complete proposal.

## Artifacts and reproduction

Workflow: `.github/workflows/algorithm-spikes.yml`.
Commands and complete experimental definitions are in README.md.
All final artifacts retain 30 days on Actions; the downloadable research bundle
also preserves the source snapshot, raw JSON/CSV, logs, environment, aggregation
script and artifact digests. No attached reference PDF is redistributed.

| Final artifact | Artifact ID |
| --- | ---: |
| algorithm-native-baseline | 10539943919 |
| algorithm-commutator-20260921 | 10540077373 |
| algorithm-commutator-20260922 | 10539394914 |
| algorithm-commutator-20260923 | 10540277202 |
| algorithm-gram-20260921 | 10540262220 |
| algorithm-gram-20260922 | 10539724360 |
| algorithm-gram-20260923 | 10540570005 |
| algorithm-spectral-20260921 | 10540307181 |
| algorithm-spectral-20260922 | 10539584540 |
| algorithm-spectral-20260923 | 10539429792 |

Results are taken from final run 35328615673 only; the failed earlier runs are
not pooled into the reported medians. Exact per-seed values, starts, supports,
weights, operation counts and metadata are retained instead of only this summary.
