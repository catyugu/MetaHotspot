# Asymmetric heat-sink model: spectrum, extraction and compression

## Research question

We investigate whether a shared spectral interval covering the full admissible
HTC family improves the new asymmetric Case-1 ROM relative to the existing
per-port spectral calculation. The primary hypothesis is that replacing the
adiabatic first-positive eigenvalue by the lowest eigenvalue at the weakest
HTC corner extends low-frequency coverage and reduces the worst steady junction
error by at least a factor of two. This is a hypothesis to test, not a promised
outcome. The second hypothesis is that snapshot truncation or boundary projection
can dominate the final error even when the extraction residual has passed.
The experiment must be capable of rejecting either explanation.

## Data and scope

The only geometry is the newly committed heat-sink stack in dev revision
32eb633. It contains aluminum, copper, two TIM layers, and four silicon dies
with different footprints. This is the user's current failure case, so replacing
it with an easier or older symmetric example would not answer the question.
The default adapter mesh has a maximum cell dimension of 0.5 mm and preserves
all material boundaries and thin films. Its actual active cell count is measured
rather than assumed from the previous experiment. Three independently variable
ambient groups cover heat-sink sides, heat-sink bottom, and die crowns. All
physical coefficients range from 1 to 10000 W/(m² K).

The source and access details are recorded in data_contract.md. No external
dataset is required because the experiment is a controlled numerical comparison
on a supplied physical model. All reference operators and full solutions are
computed from that same adapter. FloTHERM's 35-state export provides a second
comparison but is not substituted for a full-order truth solution. In particular,
agreement with FloTHERM can be limited by mesh differences that no reduction
algorithm can correct. We report that distinction explicitly.

## Baselines and controlled variants

The baseline is the repository's actual build_parametric_basis implementation,
including AMG-preconditioned LOBPCG, port-specific high-end eigensolves, random
HTC probes, warm starts, local per-port response spaces, normalized snapshot SVD,
and the added constant mode. We keep the requested 0.001 tolerance and ten
acceptance rounds. We do not replace AMG by a diagonal approximation. This
avoids transferring a failure demonstrated with a different preconditioner to
the present implementation without evidence.

The main variant calls that identical extraction routine with a single shared
spectral interval. The low endpoint is obtained from the minimum-HTC matrix,
using inverse Lanczos with AMG-CG solves and an explicit original-pencil
residual. A positive Ritz vector also gives componentwise M-matrix envelopes.
The upper endpoint is the maximum absolute row sum of the capacitance-scaled
maximum-HTC matrix. This conservative upper bound costs one sparse traversal.
Nonnegative boundary terms make the corner construction monotone in HTC.
We report interval width and residual rather than accepting a bare Ritz number.

Ablations use the same accepted response snapshots for a given extraction.
They vary only the normalized SVD threshold and compare exact Galerkin boundary
terms against project_bci's compressed boundary representation. An uncompressed
orthogonalized snapshot space is a useful control if truncation loses accuracy.
The constant vector is retained in every case. We use positive real shifts
throughout and no Schur complement or interface elimination.

## Metrics and error decomposition

We record six kinds of evidence. First is the steady junction error in kelvin
against the identical full-order model. Second is full-field maximum error in
kelvin, because internal recovery matters for this application. Third is the
maximum transient junction error over the nominal 1000-second step response,
using the same backward Euler time grid. Fourth is the number of actual full
response solves, which directly measures extraction effort. Fifth is elapsed
extraction time, including spectral planning and closing SVD. Sixth is final
state dimension, to expose an improvement obtained merely by a larger space.

The nominal junction temperature table also contains the independent FloTHERM
ROM and full-order temperatures. This distinguishes errors in reduction from
errors between the discretized models. Solver residuals, orthogonality error,
and accepted probe residuals are diagnostics, not alternative accuracy metrics.
No empirical H-infinity guarantee is claimed from a finite collection of tests.

## Protocol and replication

The nominal HTC vector is (100,1000,30) in the adapter's declared group order.
Evaluation additionally includes all eight corners of the physical HTC box and
independent log-uniform interior points drawn from seed 1113. Evaluation is
held out from enrichment. For reproducibility, the default extraction seed
20260805 is retained and two neighboring seeds are planned. If runtime limits
prevent a complete three-seed comparison, the report labels the completed
subset and does not invent a standard deviation. Method comparisons use the
same extraction seed, although their differing shift plans can consume different
numbers of random draws. Therefore pairing does not imply identical samples.

Sparse and dense thread counts are fixed for reproducible CPU measurements.
The environment, dependency versions, source hashes, command, and elapsed time
are saved with each run. Full responses use the baseline's 0.0001 relative
linear tolerance; held-out full solutions use tighter tolerances and their
independent residual is checked. We save the basis and snapshots locally so
compression ablations need not repeat expensive response solves. Large arrays
and raw JSON results are excluded from commits, as requested.

## Limitations and stopping decisions

This stage concerns a single asymmetric geometry. It cannot establish
performance on arbitrary packages, parameter boxes, or non-M-matrix operators.
A positive-vector eigenvalue envelope relies on a connected symmetric diffusion
matrix with nonpositive off-diagonals and diagonal positive capacitance. We
check those properties. Floating-point bounds are diagnostic enclosures, not
interval-arithmetic certificates. Spectral convergence does not by itself prove
that a compressed parametric ROM meets a specified output tolerance.

The shared interval may add response solves or produce a larger ROM. That
counts as a tradeoff, and a result without improved error per unit cost would
reject the practical motivation for this change. If exact projection still
agrees poorly with the full model, compression and random acceptance require
further investigation. If the full model itself disagrees with FloTHERM, a
separate mesh and boundary audit is required. We will not attribute all three
failure classes to spectral planning.

## Budget

Initial compilation is allowed several minutes. One nominal full reference and
one extraction run establish actual cost before launching the paired variants.
The intended CPU budget is about one hour for three baseline and three global
runs plus held-out validation, with snapshot-based ablations reusing completed
work. Progress and results are written after each method, so partial results
remain auditable if a method fails or reaches the extraction cap.
