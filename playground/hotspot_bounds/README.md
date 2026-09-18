# Certified transient hotspot feasibility experiment

This is an isolated research experiment, not a production solver change and
not a new connecting-ROM variant. It asks whether a reliable interval for the
maximum temperature can cost less than solving every backward-Euler system to
a conventional fixed residual tolerance.

## Reproduce

Use the repository's existing Python package/dependencies and documented native
build. Do not upgrade an existing environment just for this experiment.

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
python run_tests.py
python run_cases.py
python -m pytest python/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s playground/hotspot_bounds -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python playground/hotspot_bounds/run.py --output-dir /tmp/hotspot-results
```

`--smoke --steps 16 --repeats 1` runs a small algebraic graph without the native
library or PyAMG. Smoke results must never be represented as native device
results. CI uses native assembly without a fallback.

## Model and predeclared controls

Native MetaHotspot assembles a 20 x 20 x 1.6 mm layered package containing two
6 x 6 mm silicon dies in mould, a 0.1 mm TIM, an anisotropic substrate, and a
copper spreader. Materials are synthetic benchmark inputs, not measured
properties of a real product. Top/bottom HTC are 2000/1000 W/(m2 K), side faces
adiabatic, ambient 300 K. There are 6,912 and 98,304 cell-centered DoFs. Native
material mapping, cell count, ambient equilibrium, diagonal heat capacity and
M-matrix assumptions are explicitly checked.

Two predeclared workloads share exactly the same operator, predictor and
certificate: independent switching of the trained bulk sources, and the same
switching with a localized source that was NOT used for extraction. The fixed
12-vector Ritz predictor uses only the two bulk sources at shifts
0, 0.1, 1, 10, 100, 1000 1/s. No validation trajectory is used in training.

The time step is 0.025 s, 120 steps, seed 20260918. A 350 K decision threshold
is fixed before the native results are inspected. The optional `--bottleneck`
changes TIM conductivity from 2 to 0.2 W/(m K); it is not part of the default
comparison. No claim of boundary-condition independence is made.

Controls are warm-started AMG-CG at rtol=1e-11 and rtol=1e-6, an unchecked ROM,
and the same ROM with a comparison bound but no corrections. Two adaptive
policies are compared: a 1 K maximum interval width, and threshold decisions
with a 0.1 K stopping width for unresolved cases. Threshold equality remains
UNKNOWN, never SAFE. The latter is a decision policy, not a 0.1 K width promise
for already-decisive intervals.

## Enclosure and temporal history

For the stored linear system `A x_n = D x_(n-1) + f_n`, where `D=C/dt` is
positive diagonal and A is a symmetric strictly row-diagonally-dominant
Z-matrix, its inverse is nonnegative. Given a preceding componentwise error
radius b, candidate x and residual r, any positive w with A w > 0 gives

```
beta = max_i (D_i b_i + abs(r_i)) / (A w)_i
abs(error_n) <= beta w
```

We intersect the valid componentwise bounds from w=1 and a positive approximate
Poisson response. A w is checked explicitly; accuracy of the Poisson solve is
NOT assumed by the argument. The maximum-temperature endpoints are the maxima
of the lower and upper fields separately, so the hottest cell is not fixed.

The prototype uses full-length residuals and O(N) comparison-vector scans. It
does NOT establish sublinear-in-N certification. Extra work consists of PCG
iterations only while an interval is inadequate. Importantly, solving a current
equation more accurately does not erase uncertainty in the preceding state.
If inherited uncertainty blocks progress, the method replays from its last
tight checkpoint and charges ALL replayed steps. It never resets error to zero
using the current approximate state. A regression test captures this failure.

Floating-point residual and A w evaluations include conservative roundoff
buffers. This implementation is not fully verified interval arithmetic. The
mathematical enclosure concerns the stored FVM/backward-Euler system at its
sample times. It does not certify the continuum PDE, between-sample maxima,
spatial/time-discretization errors, nonlinear material laws, input uncertainty
or model mismatch.

## Measurement

All methods reuse the same time-step AMG hierarchy. The reference uses the
same assembled matrices, time step and warm starts. Prediction, field
reconstruction, residuals, bound evaluation, corrections, and replay are all
included in online timing. Reference comparisons and artifact serialization
are audit-only and excluded from every method's timing. Two online timing
repeats are reported with reversed method order. CI runner timings are not
hardware-independent performance guarantees.

The 12 extraction solves, their AMG setups, SVD/projection, and the extra
Poisson barrier solve are reported as offline cost. Report online speedup,
one-query speedup INCLUDING setup, and break-even query count separately.
A practical rtol=1e-6 FOM control prevents using an unnecessarily accurate
reference as the only performance comparator. The peak process RSS includes
validation histories and is not a fair per-method working-memory comparison;
explicit predictor/certificate storage sizes are recorded separately.

CSV traces contain every interval, reference peak, refinement count and replay
length. JSON summaries include enclosure violations, false decisions,
uncertainty widths, timing samples, setup cost and solver work. Empirical
coverage is necessary testing, not a proof of continuous or physical safety.

## Follow-up: spatial majorant transport and controls

CI run 35303895189 completed the original four-case comparison. Its results
showed that the decision-only policy can save current-step corrections yet
lose overall time to history replay. The next controlled experiment retains
the 98,304-DoF model, both workloads, predictor, time step, seed and threshold:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python playground/hotspot_bounds/followup.py --output-dir /tmp/hotspot-followup
```

`transport.py` computes one or two AMG V-cycle approximations z to A^-1 q,
where q is the nonnegative temporal error drive. It repairs the remaining
componentwise defect using the checked positive comparison vectors, so no
positivity or contraction assumption about AMG is required. The repaired
spatial upper bound is intersected with the original bound. This tests a
stronger majorant, not a new predictor. All certificate AMG cycles, including
those during PCG checks and history replay, are counted and timed. They are
base certification cost; only additional state corrections/replays remain
decision-driven. The floating-point and discrete-model limitations above
are unchanged.

The follow-up also compares a predictor that merely copies the preceding
state (zero ROM extraction cost) and a fixed-rtol=1e-3 FOM. The latter has no
certified maximum-temperature interval, but its measured error is reported to
check whether the apparent speedup is only against unnecessarily tight solves.
The original width and decision policies are remeasured on the same runner
as controls. The workflow now runs this follow-up; the original experiment
remains reproducible using run.py. Three added tests verify repaired majorants
for inaccurate, negative and exact approximate inverses and over time.
