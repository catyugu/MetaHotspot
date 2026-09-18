# Transient hotspot intervals: CI findings, 2026-09-18

## Conclusion

The discrete interval prototype passed its enclosure checks, but these trials
DO NOT establish a substantially cheaper decision-driven thermal simulator.
A 1 K interval policy is 1.14-1.68 times faster online than the rtol=1e-6 FOM
in the first run, but loses that advantage after one-query setup costs. In the
follow-up, rtol=1e-3 FOM solves are faster than every certified method while
showing less than 0.04 K peak-temperature error on these two trajectories.
Those loose FOM solves do not themselves supply a certified interval.

Waiting until a threshold decision becomes ambiguous can cause costly history
replay. One/two AMG cycles for spatial majorant transport did not fix the cost
problem. These are feasibility and negative-control results, not an originality
claim or a production solver recommendation.

## Provenance and validation

Base: `agent/work` at `0b4e5874d27571488d62568d12b7a68df7656aca`.
Experiment branch: `agent/hotspot-bounds-20260918`; agent/work was not changed.

| Stage | Code commit | GitHub Actions run | Result |
| --- | --- | --- | --- |
| Unchanged production baseline | `5ba062dc31865c445636a0ce0a07eb8ef56ef1e5` | `35303339848` | success |
| Initial four-case experiment | `39ae154fdbc09a48e50c79c0a3f2d723796afd16` | `35303895189` | success |
| Controlled transport / no-ROM follow-up | `ea2dc7ad951460e6b009f39c2dcfa840162e28b1` | `35304621120` | success |

The final numerical CI passed 93 C++ tests, 12 Python API tests, all eight cases
covered by run_cases.py, and 17 new algebraic/history tests. The first numerical
run had 14 new tests; the follow-up added three. Both numerical runs passed
`git diff --check`, verified no differences in production src/python/tests/cmake
or CMakeLists.txt against the base, and ended with clean worktrees.

Artifacts are named `hotspot-interval-audit`. Initial experiment artifact ID:
`10530487119`; SHA256:
`e2674ad291789301ad30086574e7ecf16d90fb411b6e14b561ae9cf74b702a09`.
Follow-up artifact ID: `10531568086`; SHA256:
`b77a666fb5c286af83795cf5d706ef803c88c13ae81e73781b6b4ad9a8895277`.
Each contains the exact source archive, environment, build/test logs, CPU
metadata, complete CSV traces and JSON summaries. Actions retention is 30 days.

## Experiment definition

One synthetic layered two-die package, two meshes (6,912 and 98,304 cells), and
two prescribed source histories: bulk-source switching, and the same switching
plus an untrained localized die source. This is four distinct discretized
trajectories, not four independent real devices. The follow-up repeats only
the larger-mesh pair; it does not add independent physical validation cases.

MetaHotspot C++ supplies the native FVM matrices. Timed FOM and correction solves
use SciPy/PyAMG, not the C++ AMGCL backend. Every method shares the same time-step
matrix, warm-start convention and AMG hierarchy. The predictor has 12 vectors,
extracted with 12 shifted solves using only the two bulk sources. No validation
trajectory or localized-source snapshot is used for training.

Backward Euler: dt=0.025 s, 120 steps, 3 s duration. Ambient: 300 K. The 350 K
research decision threshold, seed 20260918, grids and source histories were
fixed before the first native results. The request is a result at every sampled
time, not an existential overtemperature test that stops at the first violation.

Online timing includes prediction, full-field reconstruction, residuals,
certification, corrections and ALL replayed steps. Validation against stored
reference states and artifact serialization are excluded. Two timing repeats
with reversed method order are reported. Shared model assembly is excluded
from the solver timings; cold comparisons include common time-step AMG setup
and all additional extraction/barrier setup. Timings are runner-specific.

## Initial run: 1 K interval-width policy

All 480 sampled times across the four trajectories passed peak and full-field
enclosure checks. Every interval-width-policy result stayed below 1 K width;
no history replay was needed for this policy. The test uses a 1e-6 K violation
tolerance plus the independently bounded numerical reference error. The largest
reference error radius was 2.277e-7 K. No false threshold labels were observed
for the certified policies at the prescribed 350 K threshold.

| DoF | Load | FOM rtol=1e-6 (s) | 1 K interval policy (s) | Online speedup | Max width (K) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 6,912 | switching | 0.6030 | 0.4446 | 1.356 | 0.99834 |
| 6,912 | unseen hotspot | 0.5982 | 0.5226 | 1.145 | 0.99834 |
| 98,304 | switching | 10.9224 | 6.4991 | 1.681 | 0.99977 |
| 98,304 | unseen hotspot | 10.8127 | 7.8150 | 1.384 | 0.99977 |

For the larger mesh, the extra predictor/barrier setup was 4.6094 s. Including
this cost, one-query speedups against rtol=1e-6 FOM are only 0.983 and 0.871.
The corresponding two small-mesh cold speedups are 0.921 and 0.823. Repeated
queries can amortize setup, but this is not a cold-start acceleration result.

The larger-mesh 1 K policy still corrected 113/120 switching steps and 117/120
unseen-source steps, using respectively 236 and 302 PCG iterations. Its actual
peak errors were 0.00127 and 0.00184 K. It is a full-vector hybrid algorithm
with a 12-vector predictor, not a 12-state online system independent of N.

## Initial run: important failure modes

On the larger switching case, the unchecked predictor's maximum peak error was
only 0.02493 K, yet certifying it without corrections produced intervals up to
131.07 K wide. For the untrained source, predictor peak error rose to 64.69 K,
and the uncorrected interval reached 6428.84 K. Mathematical enclosure alone
is not useful accuracy; these cheap-but-loose results are not speedup successes.

The decision-only policy accepts any strict 350 K decision, regardless of width;
otherwise it refines until decisive or at most 0.1 K wide, returning UNKNOWN if
still straddling the threshold. Thus 0.1 K is NOT a width promise for already
decisive intervals. On the larger switching case, 107/120 steps needed no current
correction, but six replay events recomputed 113 historical steps. Total time
was 25.3802 s, versus 10.9224 s for the practical FOM. On the larger unseen-source
case, four replays recomputed 64 steps; total time was 15.7180 s.

This is not repaired by solving the current approximate equation accurately:
its initial state is itself uncertain. The implementation propagates the prior
radius and replays from a trustworthy checkpoint when necessary; it never
silently resets the radius to zero. A dedicated scalar regression test checks
this trap.

## Follow-up: same-run controls on 98,304 DoF

All follow-up enclosure checks passed, with no observed false decisions. The
following are second-run medians; compare methods within this table rather
than interpreting machine-to-machine absolute time differences as progress.

| Method | Required result | Switching (s) | Unseen hotspot (s) |
| --- | --- | ---: | ---: |
| FOM, rtol=1e-6 | point solution, no interval | 9.9486 | 10.0875 |
| FOM, rtol=1e-3 | point solution, no interval | 3.8077 | 3.8161 |
| Original predictor + comparison vectors | width <= 1 K | 6.1312 | 7.6289 |
| No ROM: preceding-state predictor | width <= 1 K | 10.9669 | 11.2216 |
| Spatial transport, 1 AMG cycle per certificate | width <= 1 K | 12.4316 | 14.9448 |
| Spatial transport, 2 AMG cycles per certificate | width <= 1 K | 15.8406 | 18.7568 |
| Original predictor + comparison vectors | threshold decision | 23.4097 | 14.9466 |
| Spatial transport, 1 AMG cycle per certificate | threshold decision | 29.9688 | 20.2704 |
| Spatial transport, 2 AMG cycles per certificate | threshold decision | 28.3454 | 23.4584 |

The rtol=1e-3 FOM maximum peak errors were 0.03808 K and 0.02764 K. It had no
certified bound, so this is not an equal-guarantee comparison. Nevertheless,
it prevents claiming a robust acceleration over ordinary adequately accurate
full-order calculation. The original 1 K policy was 1.61 and 2.00 times SLOWER
than this loose FOM, even before additional offline costs.

The no-ROM control shows that simply replacing the residual stopping tolerance
with the comparison bound is not a winning shortcut here. Prediction helps,
but removing its extraction cost does not compensate for the extra online work.

For switching, one-cycle transport used the same 236 state PCG iterations as
the original width policy, plus 238 certification AMG cycles. Two-cycle transport
saved only two state iterations (234 versus 236) while adding 474 certification
cycles. For the unseen source the analogous counts were 302+271 and 300+540.
Retaining some spatial shape did not recover enough work to pay for certification.

Transport constructs an approximate z to A^-1 q for the nonnegative error drive
q. For each checked positive w, it repairs the defect using
`gamma=max(0,max_i((q-A*z)_i/(A*w)_i))`, making z+gamma*w a supersolution. No AMG
positivity or contraction assumption is needed. All such cycles are base
certification work and are charged, including those during correction/replay.
This is a tested stronger comparison baseline, not a claimed new theorem.

## Limits and next research question

The mathematical bounds concern the stored linear FVM/backward-Euler system
at output times. Guarded double precision is used, not formally verified interval
arithmetic. The experiments do not certify the continuum PDE, between-output
maxima, spatial/time-discretization errors, uncertain inputs, nonlinear material
feedback or physical-device safety. Synthetic material values are not calibration.

In fact, the two meshes produce unseen-source maximum temperatures of 464.0104 K
and 456.7990 K, a 7.2114 K difference. The reported 1 K algebraic interval must
not be mistaken for a 1 K physical or mesh-converged uncertainty budget.

The first-run basis occupies 9 MiB at the larger mesh; the explicitly recorded
comparison arrays occupy 3.75 MiB, excluding sparse operators, state vectors and
temporaries. Process peak RSS was 289.16 MiB including reference histories, so
it is not a per-method working-memory comparison. No sublinear-in-N certificate,
C++ AMGCL timing comparison, broad parameter sweep or continuous-time guarantee
has been established.

The next useful target is a tighter, cheaper space-time/local-output certificate,
not more full-order AMG work in a second error equation. Specifically, hold the
predictor fixed and try to remove most full-vector corrections on the switching
case, where the predictor is already accurate to 0.025 K but the cheap enclosure
is wider than 100 K. Preserving signed residual cancellation and limiting temporal
wrapping are hypotheses to investigate, not conclusions proven by these trials.
A successful next method must beat the same-run loose-FOM control after charging
certification and setup, and must still detect the withheld local hotspot.

Reproduction commands, mathematical assumptions and source definitions are in
README.md. run.py reproduces the initial experiment; followup.py reproduces the
controlled follow-up. The final workflow runs followup.py. No production code,
project dependency file or shared branch history was modified.
