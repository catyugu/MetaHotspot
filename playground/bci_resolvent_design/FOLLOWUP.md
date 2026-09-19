# Follow-up after run35431777808: stronger stock control and derivative coverage

The initial three-seed screen succeeded and is preserved at411eb216e6e9ed368c5e52db8f04728556a30d06.
Two qualifications prevent overinterpreting it:

1. Stock epsilon1e-4 misses the1% full-field gate by only0.0614% and0.0607%
   (percentage points) in two seeds, forcing selection of1e-5 from the original
   coarse tolerance grid. This does NOT prove that the larger1e-5 model is the
   cheapest possible qualifying stock model. Strengthen the competing grid.
2. The cost-aware jet policy selects0/0/2 derivative actions at budgets24/48/72
   in seed20261301, and zero at all those budgets in the other two seeds. By96
   it selects only5/2/2. Similarity to secants cannot invalidate derivative
   information in general when it is barely exercised.

Add stock epsilon=.005,.003,.0003,.00008,.00006,.00003. Preserve ALL original
stock models, candidate definitions, budgets, cutoffs, geometry, powers and
validation points. All are re-executed in the same job so ratios are paired
within-run, never combined across runners. No hidden tuning of candidates.

Add quota_jet and quota_secant as a diagnostic allocation: exactly one quarter
of fine solve slots at each reported budget is assigned to derivative/secant
actions. At quota steps only already-paid parents are eligible, with the same
coarse feature pivot choosing the location/direction. At other steps only
value actions are eligible. The four initial source values stay mandatory.
Both methods use the same agenda, anchor preconditioner, initial guesses and
fine RHS budget. Secant displacement remains .25 in log physical h. All coarse
planning and the extra derivative RHS solves remain charged. This quota is
an additional construction, not an optimal allocation theorem or a rewrite of
the original adaptive policy. No validation error selects its quota or points.

Three new regression tests were red before implementation. Initial code paths
remain available with run.py without --followup; the updated workflow supplies
--followup. Compare to the strengthened stock controls and the SAME-budget
secant control before interpreting any derivative benefit. Initial results are
not overwritten or represented as independent device replications.
