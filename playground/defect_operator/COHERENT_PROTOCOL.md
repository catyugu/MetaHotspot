# Coherent inverse-factor interpolation: a distinct construction

The positive sum in BLEND_PROTOCOL improves the mass-dominated limit but loses
low-frequency propagation. This has an exact algebraic explanation, not merely
a disappointing benchmark. At s=0 its i,j inverse entry is the layered Green
entry times sum_l sqrt(w_l(i)*w_l(j))/a_l. If the interpolation supports of two
coefficient values do not overlap, that entry is exactly zero, despite global
elliptic coupling. Refining local coefficient intervals can remove MORE such
couplings even while improving the frozen scalar interpolant.

We test a different construction, preserving cross-factor products. Put all
constant-coefficient inverse factors Q_j(s) in the SAME DST/triangular gauge,
with (a_j K_layer+sI)^-1=Q_j Q_j^T. Select nonnegative weights v_j(x) by linear
interpolation in 1/sqrt(a), so

    sum_j v_j=1,   sum_j v_j/sqrt(a_j)=1/sqrt(a(x)).

Define

    Q_s = sum_j diag(v_j) Q_j(s),   P_s = Q_s Q_s^T.

This preserves the cross terms Q_i Q_j^T that the positive sum discards.
The product is PSD; strict positivity is checked in the finite-matrix screen,
not asserted for every conceivable variable coefficient. Shared factor gauge
is essential. This is not independent interpolation of arbitrary Cholesky
coordinates or merely a change of source snapshots.

Two exact identities motivate it:
- At s=0, Q_j=a_j^-1/2 Q_layer, so Q_s=D^-1 Q_layer and P_s equals the
  original scaled layered inverse for ANY interpolation resolution.
- At s approaching infinity, sqrt(s)*Q_j approaches the same F^T, so
  s*P_s approaches I because the weights sum to one.

These limit identities do not imply uniform intermediate-shift accuracy.
A five-test algebra suite checks both limits, adjoints, the interpolation moment
and the exact disjoint-support failure of the earlier positive sum. Tests were
red before implementation; all 28 tests pass before the scientific CI.

## Fixed experiment before coherent results

New independent seeds 20261021,20261022,20261023. Same three analytic families,
meshes n=12,24,36 and shifts 0,20,2000,200000. The unrepresented 20-fold
inclusion remains. Primary is five-node coherent inverse-square-root-coordinate
interpolation. Controls: original scaled inverse; five-node incoherent harmonic
blend; five-node incoherent ROOT-coordinate blend (same weights as primary);
five-node coherent HARMONIC-coordinate blend (same weights as earlier blend).
Nine-node coherent-root interpolation is a predeclared sensitivity check.
Thus cross-factor coherence and coordinate choice are isolated separately.

Measure complete small-matrix energy spectra, optimal rank for 1%/0.1% after
one/two/three inverse applications, uncorrected two-application error, and full
factor application time. Both five-node coherent and incoherent versions use
five pairs of triangular factor actions; no cross terms are formed explicitly
at application time. Dense oracle work is separate and is NOT a practical
low-rank construction. No speedup over CG or LU is claimed from this screen.

Structural gate: on n36 smooth/layered, primary two-application/1% correction
rank <=15 for all four shifts and new seeds; at s2000 a >=4x rank reduction
against the original scaled inverse in every seed; at s0 equality with the
original representation up to roundoff. Report every case, including failures.
Nine-node results cannot replace the five-node primary. No retuning after data.

## Literature and remaining research burden

Interpolated inverses (DOI 10.1016/j.cam.2022.114088), product-convolution
(DOI 10.1137/18M1189324), and factorized approximate inverses (e.g.
DOI 10.1137/S1064827599356900) are established. Preserving cross terms and matching
two limits is a tested construction, not a worldwide originality guarantee.
To become a research contribution it would require a uniform operator-error
analysis, a more efficient implementation with fair controls, and wider
coefficient/interface classes. No claim of physical maximum principles,
nonlinear stability, arbitrary geometry or full transient accuracy is made.
