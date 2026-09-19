# Resolvent design: two new-to-project screens against stock BCI FANTASTIC

## Status and reproducible baseline

Base: agent/work at 0b4e5874d27571488d62568d12b7a68df7656aca. The separately
existing bci_comparison, bci_block, bci_state_geometry and algorithm_spikes
branches were inspected, not modified or reported as new experiments here.

Native baseline run 35431145215 completed before these algorithms were added.
It runs the documented native build, run_tests.py, run_cases.py and python/tests,
exports the ORIGINAL Case1 at three mesh sizes, and invokes the stock extractor
on the 5 mm mesh. At epsilon=1e-3, probes=10, seed20260805, the stock result is
1320 cells, rank35, 116 exact response solves and 0.679787738% nominal steady
full-field rise error. This is a baseline measurement, not an acceptance gate.

The stock baseline calls build_parametric_basis, project_bci and
assemble_reduced_k from python/metahotspot/macromodel/utils.py without replacing
its spectral estimation, elliptic shifts, per-source spaces, random probes,
AMG-CG solves, closing SVD or constant-mode retention. Its complete source hash
is checked by both tests and the scientific runner. No project dependency or
production source changes are made. MAX_ORDER=1024 and probes=10 match the
original reproduce_case1.py protocol; every stock tolerance is run unmodified.

## Research and originality boundary

The low-fidelity and parameter-derivative ideas themselves are established:

- Hampton et al., Parametric/Stochastic Model Reduction: Low-Rank
  Representation, Non-Intrusive Bi-Fidelity Approximation, and Convergence
  Analysis, arXiv:1709.03661: low-rank bi-fidelity representation/error analysis.
- Chetry et al., An iterative multi-fidelity approach for model order reduction
  of multidimensional input parametric PDE systems, IJNME2023,
  DOI10.1002/nme.7333: low-fidelity-driven selection of high-fidelity samples,
  including a heat-conduction example. Merely moving selection to a coarse mesh
  is not a new algorithmic principle.
- Feng et al., Multi-fidelity error estimation accelerates greedy model
  reduction of complex dynamical systems, IJNME2023, DOI10.1002/nme.7348.
- Baur, Beattie, Benner, Gugercin, Interpolatory Projection Methods for
  Parameterized Model Reduction, SISC2011, DOI10.1137/090776925: interpolation
  of parameter gradients/Hessians is already covered.
- Codecasa et al., Extended FANTASTIC, DOI10.1109/TCPMT.2021.3102657; FANTASTIC
  2014 already diagonalizes fixed-parameter ROMs. A small-system backend swap
  therefore cannot be credited as a new extraction algorithm.

The narrower hypotheses tested are (A) whether a cheap fine-grid discrepancy
sketch fixes missing directions in low-fidelity sample selection, and (B)
whether cost/dependency-aware derivative selection beats equal-cost secants.
These are finite feasibility constructions, NOT established new theories,
uniform error certificates, complete reproductions of every cited method, or
claims of worldwide originality. Positive results require stronger analysis.

## Algorithms and same-budget controls

Let A(s,h)=K+sC+sum_j p_j(h_j) H_j, with p=h/(1+beta*h). A response column is
x=A^-1 g. Every method retains the uniform-temperature direction explicitly.

Construction pool: physical HTC grid {1,100,10000}^2 plus eight log-uniform
points (seed+101), crossed with zero and 13 logarithmic shifts between 1e-5
and 2*max_i K_ii/C_ii from the fine model. Every source appears at each point.
The upper shift is a common planning bound, not a bound for all Robin-loaded
operators. Four value responses at h=(100,100),s=0 initialize ALL methods.
The pool, derivative agenda and hashes are saved before physical validation.

A. coarse_qr uses exact coarse sparse-LU response columns, scales them by their
capacity-L2 norms and removes their uniform component, then greedily pivots
orthogonal residual features. defect_qr augments those same features with

    CountSketch[ sqrt(Cf) diag(Af)^-1 (g_f - A_f P x_c) ] / ||sqrt(Cc)x_c||.

P is nonnegative cell-center trilinear interpolation with boundary clamping.
The sketch has 128 rows, with fixed seed+303. No fine inverse solve is used in
this score, and it is NOT a reliable full-state error estimate. All fine sparse
matrix actions and all interpolation/sketching costs are charged. A random
agenda is the further equal-fine-solve-budget control; it does NOT pay for a
coarse bank it never needs. The coarse operator is not substituted for fine
physics. Fine snapshots always solve the actual native fine equations.

B. jet constructs coarse values and derivatives w.r.t. log PHYSICAL h:

    dp/dlog(h)=h/(1+beta*h)^2,
    A dx/dlog(h_j)=-(dp_j/dlog(h_j)) H_j x.

Derivatives use the VALUE response norm, not an independent unit normalization,
when selecting actions. Each derivative requires a previously paid base-value
solve; the pivot score accounts for this dependency. A fixed 96-solve agenda is
constructed. At budgets24,48,72,96, a prefix can end after a newly required base
value without falsely counting an unpaid derivative.

secant uses EXACTLY the same agenda and the same anchor AMG preconditioner,
but computes a finite log-parameter displacement of magnitude .25, directed
inside the physical box, and uses (x(h_new)-x(h))/delta as the snapshot. It pays
one perturbed fine solve for every derivative action; no extra sample is free.
Both value/derivative parent solves count. This isolates derivative information
from extra work and from warm-start/cache implementation changes.

ALL candidates use the same four-entry AMG cache, AMG-CG rtol1e-6, projected
warm start, orthogonalization, Euclidean column-normalized closing SVD, and
stock boundary projection epsilon1e-3. Budgets are24/48/72/96 and closing
cutoffs1e-3/1e-4. The complete 96-action coarse planning cost is charged to each
prefix. Earlier checkpoint compression is excluded from later prefix times;
each prefix pays its own closing/projection. Fine work is cumulative. No
candidate claims the stock extraction tolerance semantics or an adaptive
stopping certificate. A smaller fixed sample budget can be inaccurate.

## Physical comparison contract

Use the unchanged native Case1 geometry/materials/air background/four original
independent sources. Low-fidelity mesh limit5 mm has1320 cells; the screening
fine mesh2.5 mm has9072. Native1 mm data (122400 cells, original reproduction
mesh) is exported but is NOT a completed fine comparison. A primary must pass
both declared gates across all three seeds before a further expensive fine
confirmation is justified. Failure means stopping rather than tuning holdouts.

Original physical HTC ranges are[1,1e4] W/m2/K for die crowns and FR4 bottom.
Every mesh has its own native half-cell resistance beta, which is asserted
constant within each group. Its p=h/(1+beta*h) map is used consistently in all
training, projection and FOM queries. No boundary hA shortcut or geometry
change is used to favor a candidate. Ambient is removed: all states, source
fields and error denominators are temperature RISES, initially zero.

Seeds20261301/02/03 are RNG integers, not calendar dates. All three jobs use the
same one geometry, not three independently calibrated devices. The stock
extractor is run at epsilon=.01,.001,.0001,.00001, twice in opposite order.
Candidates likewise have two forward/reverse construction timing repeats.
Subspace agreement and actual ranks are checked; no eigensolver is replaced.

Validation per seed: four physical range corners, (50,1000), and five independent
log-uniform pairs from seed+1717. Corners overlap construction, so they are not
independent holdouts. At each HTC compare each unit source's steady field and
step field (dt50 s,40 steps,original2000 s grid), slow independently varying
powers on that grid, and fast independent powers(dt.1 s,100 steps). Power RNG
seed+717 is separate. No validation result chooses a sampling action or basis.

Fine reference is sparse-LU backward Euler on the SAME native fine K,C,H,G,
with sampled relative linear residual<=1e-9. This controls algebraic reduction
error, not spatial/time-discretization error or physical-device calibration.
Every ROM is compared to the FOM, not merely a parent ROM or coarse prediction.

Independent-source field errors normalize by each source's own maximum steady
rise, then maximize over sources. Junction errors use each unit experiment's
steady maximum over source averages. Mixed-profile errors use the maximum of
the ACTUAL time trace, not the average of cellwise temporal maxima. Record
worst field/junction/peak relative error, peak error K, capacity-weighted L2 and
steady boundary heat-flow discrepancy in W for unit inputs.

## Timings, attribution and predeclared gates

All scientific configurations are reported, not only favorable seeds. Offline
cost includes coarse solves, scoring, fine solves, matrix actions, warm starts,
closing SVD and stock BCI projection. Common native assembly/reference validation
and file serialization are excluded and documented; coarse native assembly is
also excluded, so an observed offline failure cannot be repaired by charging
its omitted extra work. Per-query non-asymptotic and repeated-use costs must
not be confused. Total process peak RSS is not per-method working memory;
explicit persistent ROM and decoder bytes are recorded.

Online comparisons include parameter closure and, when applicable, generalized
eigendecomposition and field decoder rotation. EVERY model gets the same dense
factored BDF1 and diagonalized BDF1 backends; three repeats after warmup are
recorded at nominal HTC for unit-step and fast mixed queries. The faster common
backend is selected independently for every competitor/profile/output contract.
Ports-only and full-field-plus-peak costs are separate. Full-field reconstruction
cannot disappear from one competitor's timing. Common backends are checked to
agree to1e-8. The actual stock solve_rom_transient wrapper is ALSO executed on
every model and checked against dense BDF1 to1e-5, but replacing that wrapper
is not counted as extraction innovation.

Accuracy gate: <=1% worst field AND junction error across all ten HTC settings,
steady and every time history. A separate junction-only gate uses only junction
error, preventing a full-field requirement from inflating a ports-only baseline.
Choose the cheapest accuracy-qualified STOCK tolerance from the fixed grid,
compare rank to the smallest qualified stock model, and online time to the
fastest qualified stock model. If stock or candidates fail accuracy, state it;
never claim speed over an inaccurate comparator.

Practical gate: >=2x offline speedup, rank<=1.25x smallest qualified stock,
and no>25% query-time regression (both outputs for the full-field contract).
New-mechanism gate: at the selected candidate's SAME budget and closing cutoff,
defect_qr must halve coarse_qr's worst error, and jet must halve secant's worst
error. Generic savings from planning/caching are not credited to these additions.
Both gates must hold across all three seeds before recommending a primary.
The conventional controls can win; that alone does not establish new research.

## Reproduction and audit

    python -m unittest discover -s playground/bci_resolvent_design -p 'test_*.py' -v
    python playground/bci_resolvent_design/run.py --native NATIVE_EXPORT --seed 20261301 --output OUT

CI reuses previously validated Python3.13/NumPy2.5.3/SciPy1.18.1/PyAMG5.3.0
versions in fresh isolated environments, no project dependency change. Native
regressions run before studies. Local tests use the installed NumPy/SciPy;
real PyAMG is absent locally and the native fine-solve test is explicitly
reserved for CI, never substituted by a dummy module or a direct solver.
Tests were red before implementation (missing design module) and green after.

Local source is reconstructed from CI archives because GitHub DNS is unavailable
in the container, not a conventional remote-history clone. Its original tree
matches701f2d6891a16c55a2436fab39f6e7926f4039d4; its baseline workflow tree matches
10634c844dd344d672d8dcdb0abcc1765b1dea0b. Source archives, native hashes, all raw
construction/metric/timing rows, agendas and original stock extraction summaries
are retained in GitHub artifacts. No automatic merge into agent/work.
