# Independent follow-up: match the diffusion and mass limits simultaneously

This follow-up is motivated by the first study's high-shift rank growth. It
changes the approximate inverse representation rather than increasing its
rank cap. The original study, datasets and code are retained unchanged.

Let k(x)=a(x)*layer(y). Choose positive coefficient nodes a_j and nonnegative
weights w_j(x) with sum_j w_j=1 and sum_j w_j/a_j=1/a(x). Define

    P_s = sum_j sqrt(W_j) (a_j K_layer+s I)^-1 sqrt(W_j).

This is SPD. For frozen scalar coefficients its symbol matches both the
large-s and large-spatial-frequency limits. Linear interpolation in rho=1/a
has relative scalar resolvent error between zero and

    ((sqrt(a_hi/a_lo)-1)/(sqrt(a_hi/a_lo)+1))^2

on each coefficient interval, uniformly over positive shift/frequency ratios.
The error formula follows directly by interpolating rho/(lambda+s*rho).
Crucially, this scalar inequality is NOT an operator inequality when the
weights vary in space. Weight multiplication does not commute with diffusion.
The exact finite-matrix spectrum below tests that missing part instead of
assuming it. This also differs from fitting source response snapshots.

This is a feasibility screen, not a claim of invention of positive sums,
product-convolution or interpolated inverse preconditioners. Close prior art:
Alger et al., SIAM J Sci Comput 2019, DOI 10.1137/18M1189324; and the 2022
paper on tau-matrix approximate inverse preconditioning for diagonal-plus-
Toeplitz fractional diffusion, DOI 10.1016/j.cam.2022.114088. The latter already
uses piecewise interpolation and DST. A publishable contribution would need
substantially more than applying those ingredients here.

## Fixed protocol, before follow-up results

New seeds 20261011,20261012,20261013. Same analytic coefficient family definitions
as the first study, including its unrepresented rectangular jump. Meshes
n=12,24,36; shifts 0,20,2000,200000. The added shift probes the mass-dominated
limit, not a replacement for the failed s=2000 case.

Four representations: original jump-aware single inverse; harmonic five-node
interpolation (PRIMARY); arithmetic five-node interpolation on the same nodes
(control for changing the interpolation coordinate); harmonic nine-node
interpolation (predeclared cost/accuracy ablation, not a fallback winner).
Nodes are geometric between the actual coefficient extrema. No validation
source, solution or tuned spectrum enters node/weight construction.

All small-grid spectra are computed fully. Dense inverse formation and Cholesky
are oracle work; there is NO practical low-rank construction in this follow-up.
Report optimal correction ranks for 1% and 0.1% energy accuracy with one, two,
and three fixed stationary inverse applications. Report actual factor-only
application times, including every backbone solve and weighting. Comparing
ranks alone must not hide a five/ninefold backbone application count.

Screening gate for the PRIMARY at s=2000, n=36 on the smooth and layered
families: at least fourfold optimal-rank reduction for two inverse applications
and 1% energy target, in every new seed; rank growth from n12 to n36 at most
factor two. Report s=0,20 and the inclusion even if they regress. Passing would
justify a better matrix-free factorization study, NOT prove overall speedup.
No source accuracy, full PDE solve acceleration, uniform-domain certificate,
new complexity theorem or 3-D result is inferred from this screen.

## Reproduction

    python -m unittest discover -s playground/defect_operator -p 'test_*.py' -v
    OPENBLAS_NUM_THREADS=1 python playground/defect_operator/blend.py \
      --seed 20261011 --output-dir /tmp/blend-results

Only seed42 n6 smoke ran locally before these CI cases. Six new tests verify
constant coefficients, positivity, two scalar moments, the scalar bound, and
the mass-dominated limit. All 23 algebra tests pass before commit.
