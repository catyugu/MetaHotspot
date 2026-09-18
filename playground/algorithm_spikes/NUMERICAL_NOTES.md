# Numerical robustness follow-up

Initial numerical run 35327156691 passed the native suite, 19 algebra tests,
all commutator/Gram jobs and two spectral jobs. The third spectral job (seed
20260921, nx=24, mild law, 48-edge budget) reported a HiGHS numerical failure in
a minimax LP. This LP always has the feasible point w=0,t=1. The failed run and
its logs are retained; it is not discarded as an unfavorable scientific case.

A twentieth regression test forces the primary solver failure and was red
before the fix. The implementation retries the SAME LP once with highs-ipm,
presolve disabled, and independently checks primal feasibility on the original
constraints. No seeds, states, budgets, weights objective or other numerical
method were tuned. All retry work is charged to offline time; lp_calls and
lp_retries are included in the selection audit. All nine research jobs are
rerun on the fixed source version, without examining holdouts for tuning.

The first retry patch (run 35327920151) exposed additional solver epigraph
residuals even when both methods reported success. Those failures are also
retained. The final patch column-equilibrates the same LP (an invertible positive
variable scaling), projects returned roundoff-negative weights onto nonnegative
weights and recomputes the epigraph on the ORIGINAL constraints. Maximum
lp_max_epigraph_repair and lp_max_bound_projection are explicitly recorded.
This is a feasible numerical weight fit, not a formally certified optimal LP.
Two additional regressions cover extreme column scaling and rounded epigraphs;
both failed before this patch. All 22 tests then pass locally. A training-only
reproduction of the failed seed/configuration was attempted locally after the
failure; local SciPy 1.17 did not reproduce the CI 1.18 residual condition.
No held-out results were used to tune the algorithm or this numerical guard.
