# State-space geometry: direct comparison with stock BCI FANTASTIC

## Scope and unchanged baseline

Base: catyugu/MetaHotspot agent/work at
0b4e5874d27571488d62568d12b7a68df7656aca. Existing bci-comparison and
bci-block-screen branches were inspected, not overwritten or represented as
new work. No hotspot bounds, inverse-factor or commutator code is reused.

The first native baseline CI, 35427909542, passed 93 C++ tests, 12 Python API
tests and all eight run_cases.py cases. The actual unmodified library extractor
on original Case1 with 5 mm mesh, epsilon=1e-3, seed20260805 and 10 probes yields
N=1320, rank35, 116 full responses. The new experiments invoke the same library
build_parametric_basis, project_bci and assemble_reduced_k. No replacement of
spectral estimation, random probes, AMG-CG, per-port loops or closing SVD is
allowed in the stock baseline. MAX_ORDER is read from the library (2048).

Two new-to-project questions are screened. Neither is claimed to be a complete
new MOR theory. Both first pay for the stock epsilon=1e-4 BCI parent, then alter
its state-space representation. They CANNOT claim cheaper extraction than
that prerequisite. Unmodified stock epsilon=1e-2 and 1e-3 models are competing
baselines, not artificially excluded because they were not used as parents.

## Literature boundary

- Codecasa et al., Extended FANTASTIC, DOI 10.1109/TCPMT.2021.3102657:
  multi-source, parametric Robin conditions, whole temperature fields.
- Codecasa et al., FANTASTIC 2014 (supplied project paper): generalized
  eigencoordinates already diagonalize fixed-boundary ROM dynamics. Therefore
  a dense or modal small-system solver improvement must be offered to EVERY
  model and cannot be credited to the new state-space construction.
- Schwerdtner and Schaller, Structured Optimization-Based Model Order Reduction
  for Parametric Systems, SISC 2025, DOI 10.1137/22M1524928: existing structured
  optimization across frequency and parameters. Generic minimax PMOR is not new.
- Hund et al., H2 x L2 optimal parametric MOR, DOI 10.1137/21M140290X.
- Castagnotto et al., model-function/H2 framework,
  DOI 10.1080/13873954.2018.1464030: optimization on intermediate models is known.
- Gosea, Gugercin and Unger, arXiv:2104.01016: parameter-dependent rational
  interpolation spaces are known. Parameter-dependent projection is not new.
- Liljegren-Sailer and Gosea, SISC 2024, DOI 10.1137/23M155791X: balanced singular
  perturbation and its low-frequency advantages are established. This study
  does not claim static condensation, moment preservation or harmonic extension
  as inventions. It compares simple DC augmentation and local Gramian reduction
  explicitly rather than presenting them as the new proposal.

The narrower algorithm hypotheses are whether a constrained spectral-deficit
objective improves a global fixed-rank space beyond response POD, and whether
parameter-local harmonic transport yields a useful dynamical space beyond merely
adding static solutions. No worldwide originality or H-infinity bound follows
from these experiments. The search has not ruled out all close prior art.

## A. Spectral-deficit optimization of a global subspace

Whiten the parent capacity to I. At each construction (s,h), let A=K(h)+sI and
normalize every source by its parent self-impedance, yielding B. For an
orthonormal W, Y=(W^T A W)^-1 W^T B and R=B-A W Y. Then

    D = B^T A^-1 B - B^T W Y = (A^-1 B-WY)^T A (A^-1 B-WY) >= 0.

Minimize a smooth maximum of all eigenvalues of these D matrices across the
fixed construction bank, with the constant mode constrained to remain in W.
An analytic gradient -2 sum R S Y^T is used, where S is the spectral softmax
weight. It is finite-difference tested. Unpivoted-QR retraction, Armijo backtracking
(max14), 10 steps at each tau=.05,.01,.002, and normalized-response POD
initialization are fixed before scientific CI. No validation point tunes these.
POD is an equal-rank control using precisely the same response bank and constant
mode. Its leading singular subspaces are specifically regression tested.

This is a finite positive-real-frequency objective, NOT an H-infinity norm,
all-parameter certificate or pointwise temperature bound. All operators, ports
and the full-field decoder are Galerkin projected; the parent's BCI boundary
representation is retained, avoiding a changed boundary-SVD error budget.

## B. Harmonic parameter-local spaces, with exact parent static response

For constant h within a query, choose k=r-m-1 nonconstant leading POD columns
V and an orthogonal complement Z. Let D=Z^T K(h) Z. Form

    T = V - Z D^-1 Z^T K(h) V,
    L = Z D^-1 Z^T F,
    W = orth([constant,T,L]).

The parent static solution belongs to this space. A first-order Galerkin model
on W retains its projected mass and zero initial response. It does not replace
eliminated dynamics with algebraic feedthrough. Both SPD and exact parent DC
are tested. Source directions m and the constant count toward the rank budget.

Matched controls use the same rank and POD bank:
- dc_augment: orth([constant,V,K(h)^-1 F]); isolates transport from static moments.
- local_balanced: leading local symmetric-system controllability Gramian
  directions with the constant preserved. It is a conventional control, not
  a complete reproduction of every balanced-truncation or SPA variant.

All parent matrices and decoder must remain available. Query-local state rank
is NOT persistent decoder memory reduction. Every parameter query pays for
local construction, projection and decoder multiplication. Time-varying HTC
state transfer and its derivative/connection terms are NOT implemented; no
claim of unrestricted drop-in equivalence to a global BCI state space is made.

## Physical comparison contract

Use the native original playground/bci_rom_testcase1/model_case1.py. Original
geometry, air background, materials, four die source shapes, ambient and the
two boundary groups remain unchanged. Three meshes are exported by CI: cell
limits 5,2.5,1 mm on all axes, with N=1320,9072,122400 respectively. The 1 mm
mesh matches the original reproduction setting; the main screen uses 2.5 mm.
The native matrices, source files and library files are hashed.

Main screen: seeds20261201/20261202/20261203 x source counts4/16, all on 2.5 mm.
Six independent jobs; the seeds are integers, not calendar dates. For 16 ports,
each original source is split into four quadrants and BOTH methods get all
shapes. The nominal sum is asserted identical to the original four-source load.
This is one geometry, not six independently validated physical packages.

All methods use physical HTC ranges [1,1e4] W/m2/K for both groups. Boundary
series resistance is handled with p=h/(1+h*half_cell/k), identical in stock,
new models, training and FOM. Zero initial temperature RISE is used; absolute
ambient Kelvin is excluded from relative-error denominators.

Stock tolerances are .01,.001,.0001; probes10, original spectral estimates and
library default enrichment accuracy. Each is extracted twice, forward then
reverse tolerance order. Candidate ranks are ceil(.35R),ceil(.5R),ceil(.7R),
clamped below R and above m+2; actual numerical ranks are recorded. All candidates
use the epsilon1e-4 parent. The bank has physical HTC grid {1,100,10000}^2 and
13 shifts: zero plus 12 logarithmic shifts from parent min eigenvalue/10 to
parent max eigenvalue*10 over that grid. Parent solves are dense and counted.

Validation has four range corners, [50,1000], and five separate log-uniform HTC
pairs per seed. Corners overlap the construction bank and are NOT independent
holdouts. Random HTC and power RNG streams differ from extraction. All models
see the same validation inputs, never use them for construction, and are checked
against the native FOM, not merely against the parent ROM. At every HTC:
- all independent unit-source steady and step fields: dt50s,40steps (2000s);
- slow independent mixed power: same grid;
- fast independent pulse power: dt.1s,100steps (10s).

Sparse-LU FOM below 50000 cells, real AMG-CG rtol1e-10 above, with explicit
residual audit <=1e-9. The reference is the SAME backward-Euler discrete system
as the stock BDF1 ROM, not the continuum PDE or a commercial-model ground truth.
Heat-source average outputs, temperature fields and boundary flow are retained.

## Costs, metrics and declared decision gates

For independent sources, each field error is normalized by that source's own
maximum steady temperature rise before taking a maximum over sources. Junction
errors likewise keep weak inputs visible. Mixed histories normalize by their
own maximum reference rise. Report maximum field/junction relative errors,
peak error in K and relative, capacity L2 and steady boundary-flow discrepancy.
One percent means the stated sampled physical metrics, not extraction epsilon.

Main accuracy gate: worst field AND junction errors (steady and all histories)
<=1% across all ten HTC settings. No speed claim for an inaccurate model.
From the PREDECLARED stock tolerance grid report separately the cheapest
accuracy-qualified offline model, smallest qualified rank, and fastest qualified
online model. Do not compare only to an unnecessarily accurate large parent.

Global primary: spectral_minimax. It must reach accuracy at <=75% of the smallest
qualified stock rank, use <=75% of stock persistent field-basis bytes, and have
>=1.5x full-field query speed vs the fastest qualified stock backend. At equal
rank it must reduce worst physical error by >=2x relative to POD, otherwise any
compression advantage belongs to recompression, not the proposed objective.
Local primary: harmonic. It must reach accuracy at <=75% of qualified stock rank
and >=1.5x junction-query speed including parameter preparation, without >25%
full-field regression. At equal rank its error must be <=half of the better DC
augmentation/local Gramian control. It cannot claim reduced persistent decoder
storage or reduced initial extraction cost. All gates must hold in all three
seeds for a source count before recommending that construction.

Three timing repeats after warmup. All models use the same dense factored BDF1
and the same diagonalized BDF1 backend; eigendecomposition/input rotation and
field-decoder rotation count. Per-case minimum of the two common backends is
used for every competitor. Parameter closure, local projection, port output,
field recovery and maximum scan are included in corresponding query costs.
The actual original solve_rom_transient is separately run on EVERY model at
nominal HTC and compared to the common equation; wrapper replacement is not
an algorithmic success. Full-field generation costs cannot be hidden.

Additional offline = whitening + bank/POD + optimization + projection. Total
setup also includes the complete parent extraction and boundary projection.
Report repeated-use break-even rather than claiming offline savings. Query-local
models must pay for retained parent data and per-query construction. Process RSS
is explicitly NOT per-method working memory. Shared native assembly/reference
validation/serialization are excluded from model timings.

Conditional fine confirmation: only a primary that passes the above gates for
four ports is eligible for the 1 mm case, using seed20261211 and the same fixed
parameters. No primary pass means no expensive fine run. A passing coarse test
alone is not a claim of success on the original 1 mm reproduction or other devices.
The exact fine export remains available for future independently approved work.

## Reproduction and audits

    python -m unittest discover -s playground/bci_state_geometry -p 'test_*.py' -v
    python playground/bci_state_geometry/run.py --data NATIVE/case1_2.5 --ports 4 --seed 20261201 --output OUT

Use the pinned isolated CI Python3.13/NumPy2.5.3/SciPy1.18.1/PyAMG5.3.0 environment
already used in this repository; no project dependency files change. Stock
source SHA256 is asserted unchanged. Pure NumPy/SciPy algebra tests run locally;
local PyAMG is unavailable and no fake module or solver substitute is used.
The new numerical studies run only on real GitHub CI.

Validation helper functions and memory-bounded FOM reference were inspected and
reused from bci_comparison at d29bb6d3ecb15bc547bd689c4e219a74b7ca05ea, with
attribution, not counted as new algorithms. Production files and prior experiments
remain unchanged. Local staging is reconstructed from exact CI source archives
because GitHub DNS is unavailable in the local container; its git write-tree
must match each remote tree. It is not a conventional remote-history clone.
