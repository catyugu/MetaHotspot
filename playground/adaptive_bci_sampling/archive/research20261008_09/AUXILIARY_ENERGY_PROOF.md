# Common auxiliary trajectories and deterministic dynamic tails

2026-10-09. Exact-arithmetic proof, evaluated in ordinary floating point by the
drivers. This is not an outward-rounded machine certificate. All guarantees
below concern the supplied affine effective-Robin matrix box, not an independently
validated native physical-HTC model. The original pre-SVD V is certified, without
doubling either tolerance. Input vectors are arbitrary real signed combinations.

## 1. Why snapshot tails do not enter the acceptance condition

Let C be positive diagonal, K(h)=K0+sum h_i H_i symmetric positive definite,
H_i PSD, and V^T C V=I. For x(0)=z(0)=0,

    C x' + K(h)x = G u,     z' + A(h)z = B u,
    A=V^T K V, B=V^T G, D=KV-CVA, R0=G-CVB.

The exact error satisfies C e'+K e=R0 u-Dz. A common C-orthonormal W is chosen
from input-driven error snapshots. It is subsequently only a trial space:

    eta' + Aw eta = W^T R0 u - W^T D z,    Aw=W^T K W.

Set y=[z;eta], b=[B;W^T R0],

    L(h) = [[A,0],[W^T D,Aw]],
    Da(h) = [D-CW W^T D, KW-CW Aw], Rbar=(I-CW W^T)R0.

Then y'+Ly=bu and r=e-W eta obeys the exact equation

    C r' + K r = Rbar u-Da y,    r(0)=0.

Every neglected direction, including errors of the chosen W, enters this
equation. No sum of discarded snapshot singular values is asserted to bound r.
W need not be C-orthogonal to V. The block-triangular L has positive eigenvalues
because A and Aw are SPD; its eigenvectors need not be orthogonal or well
conditioned. The current implementation refuses a modal condition >1e10 and
records modal defects, rather than treating a bad decomposition as success.

## 2. Matrix energy bound and the infinite-time split

For C r'+K r=d(t)u and any positive definite Kstar <= K,

    d/dt ||r||_C^2 <= -||r||_K^2 + u^T d(t)^T K^-1 d(t) u
                  <= u^T d(t)^T Kstar^-1 d(t) u.

Thus ||r(t)||_C^2 <= u^T J(t)u, where
J(t)=integral_0^t d(s)^T Kstar^-1 d(s) ds. This is an INPUT Gram, not a largest
operator norm on all reduced-state directions. The small input dimension is
preserved in time integration and inverse-defect correction.

For fixed h, yinf=L^-1 b and

    dinf=Rbar-Da yinf,
    d(t)=dinf+Da exp(-Lt) yinf.

Define rinf=K^-1 dinf and v=r-rinf. Then v(0)=-rinf and its forcing is the
decaying second term. If R bounds rinf^T C rinf and Jinf bounds the integrated
dual Gram of the decaying term, for every theta>0,

    r(t)^T C r(t) <= (1+theta) R + (1+1/theta)(R+Jinf).

The inequality is interpreted in input coordinates. It follows by first applying
the energy inequality to v, then Young's inequality to rinf+v. Constant dinf is
NEVER integrated to infinity. At fixed h, Jinf=yinf^T P yinf, where
L^T P+P L=Da^T K^-1 Da. A computed Lyapunov equation defect is included via a
positive Lyapunov repair, in ordinary floats. Finite-prefix and infinite-envelope
bounds are both valid; the smaller generalized norm bound may be used.

`auxiliary_energy.py` verifies all t>0 at each supplied FIXED parameter. Checking
five or 25 parameters is not a continuous-parameter certificate.

## 3. Continuous-parameter reconstruction, without a snapshot remainder assumption

Use tensor Chebyshev-Lobatto interpolation nodes on a box. At every node j solve
only the small auxiliary system y_j'+L_j y_j=b. Let ell_j(h) be its Lagrange
polynomial and construct

    P(t,h) = [V,W] sum_j ell_j(h)y_j(t),
    p(t,h) = [I,0] sum_j ell_j(h)y_j(t),
    etaP(t,h) = [0,I] sum_j ell_j(h)y_j(t).

These are trials, not assumed close to the actual trajectories. Their exact
equation defects are

    RF = G-C P'-K(h)P
       = Rbar + sum_j ell_j(h) (C[V,W]L_j-K(h)[V,W])y_j,
    RR = B-p'-A(h)p
       = sum_j ell_j(h)(A_j-A(h))[I,0]y_j.

They are parameter polynomials of tensor degree at most interpolation degree+1.
Represent them by Bernstein controls; RF/RR contain ALL interpolation error.
In particular an affine K does not imply an affine trajectory. Both defects must
be retained: leaving RR out would certify a different ROM than the original V.

Set tF=x-Pu and tR=pu-z. Both start at zero and obey their respective error
equations. The ORIGINAL error has the exact decomposition

    x-Vz = W etaP u + tF + V tR.

For the full tail use Kstar=K(box lower). For the reduced tail use Astar=V^T
Kstar V. If a defect is sum_c beta_c(h) d_c(t), beta_c>=0 and sum beta_c=1,

    integral d(h)^T Kstar^-1 d(h)
       <= sum_c beta_c(h) integral d_c^T Kstar^-1 d_c.

This follows from matrix convexity of D -> D^T Kstar^-1 D and applies pointwise
in time before integration. Consequently verified bounds of every control cover
EVERY h in the box. Spatial compression and parameter interpolation are both
charged to a deterministic equation defect; no empirical interpolation accuracy
or saturation assumption is used.

## 4. A fixed joint inverse span and actual iterative-solver defects

RF controls are in the fixed column span

    M = [Rbar, C[V,W], Kcenter[V,W], H1[V,W], ...].

The input coefficients are parameter polynomials and small-system trajectories.
QR of M is a coordinate change, without truncation. It costs at most
m+(d+2)(r+k) full inverse RHS per box, possibly fewer when the span is redundant.
No full dynamic eigendecomposition or direct factorization is required.

With `--global-inverse`, instead form M using K0[V,W] and invert a single lower
operator of the ORIGINAL whole parameter box. This span is independent of the
cell: Kcenter[V,W] is reconstructed by its affine coefficients. The same dual
Gram therefore bounds every sub-box, with zero additional full inverse RHS per
cell. Local reduced dual norms and verified coercivity bounds are still used.
A global inverse can be looser than local inverses; the drivers measure this
tradeoff rather than assume the shared inverse is free and equally tight.

For a trial inverse Y and E=M-Kstar Y,

    Qminus = M^T Y+Y^T M-Y^T Kstar Y,
    Qminus <= M^T Kstar^-1 M
           <= Qminus + E^T C^-1 E / alpha,

provided Kstar >= alpha C. This is the exact square-completion identity plus a
coercivity bound. The correction is a MATRIX, not ||E||^2 I. The driver uses
AMG-CG and records actual E. Its dual coordinate root then includes every cross
term between Rbar, temporal terms, and parameter terms.

For this conservative symmetric thermal M-matrix, a computed positive w with
Kstar w >= alpha Cw verifies the coercivity bound by Collatz-Wielandt. A Ritz
minimum is not used as a lower bound. The structural assumptions must hold.

## 5. Time integration with an explicit remainder

At a Bernstein control, the dual residual is a constant plus a finite sum of
real decaying exponentials. On [a,b], htime=(b-a)/2, expand at its midpoint to
degree p=12. For modal coefficient norms w_j and rates lambda_j>0,

    sup ||remainder|| <= sum_j w_j exp(-lambda_j a)
                              (lambda_j htime)^(p+1)/(p+1)!.

Constant terms are included in the polynomial exactly. A p+1-point Gauss rule
integrates the polynomial's squared Gram EXACTLY (degree 2p); it is not applied
to the original exponentials without a remainder. If Gpoly is the polynomial
Gram integral, rem is the above uniform norm bound, and theta>0,

    Jinterval <= (1+theta) Gpoly
                 +(1+1/theta)(b-a) rem^2 I.

For an integrated norm, Minkowski and time Cauchy-Schwarz give
integral ||d|| <= sqrt((b-a) lambda_max(Gpoly))+(b-a) rem.
This avoids allocating a dense (parameter nodes * modes)^2 modal Gram. It does
not remove parameter approximation errors, which are still RF and RR.

For the infinite horizon, split each control into its stationary value and a
decaying part. If K>=alpha C and A>=beta I, define

    sF=max_control ||RF_inf||_(Kstar^-1) / sqrt(alpha),
    sR=max_control ||RR_inf||_(Astar^-1) / sqrt(beta).

The all-time full-tail norm is at most sF+sqrt(sF^2+EF), where EF is a verified
infinite transient dual-energy integral; the analogous reduced bound uses sR.
The omitted integral after T is bounded in L2 by
sum_j w_j exp(-lambda_j T)/sqrt(2 lambda_j). All quantities act on unit input
vectors, and matrix energy integrals are retained until their comparison.

## 6. Denominators, every time, and the original two tolerances

Early times use the direct original-error estimate: after B^T B=I,

    ||e||_C / ||Vz||_C <= (t Dmax/2+||C^-1/2 R0||) exp(bmax t).

For a finite time interval, trial response Bernstein controls p_c(a) enclose
p(a,h). At the parameter midpoint let pc be the Bernstein-weighted trial.
The exact ROM norm is bounded below by sigma_min(pc), minus the largest
parameter-control deviation, the verified time variation of p, and the reduced
tail. This uses computed INPUT responses, not a global reduced-state Lipschitz
operator norm. A second valid lower bound is

    d(t)=lambda_min(t B^T (I+t Aupper)^-1 B),

using ||B||=1. It follows from scalar functional calculus and an inverse Loewner
inequality followed by Cauchy-Schwarz; no matrix inequality is squared. For all
t>=T use d(T), NEVER the larger limiting value d(infinity).

If main+full_tail+reduced_tail <= q * denominator everywhere, then
||e||_C/||x||_C <= q/(1-q). The original step threshold is therefore
q<=sqrt(0.001)/(1+sqrt(0.001)).

At steady state, the full and reduced stationary tails have energy bounds equal
to their respective dual residual norms. The W etaP term uses W^T Kupper W.
Let s be their sum divided by sqrt(lambda_min(B^T Aupper^-1 B)). Galerkin
K-orthogonality of the ORIGINAL V gives the sharper exact conversion

    ||einf||_K / ||xinf||_K <= s/sqrt(1+s^2).

Both step and steady conditions must pass. No relaxed factor-two tolerances,
post-SVD basis, nonnegative-input restriction, or sampled acceptance is used.

## 7. Limits and the rejected velocity-lift alternative

`velocity_defect.py` separately tests the identity
e=-Yz+integral exp(-C^-1 K(t-s))Yz'(s)ds, Y=K^-1D, with an affine trial T and
defect F=D-KT. Interpolating velocity rather than step trajectories makes its
equation defect integrable to infinity. Its remaining term, however, bounds
Y times the velocity interpolation error by 2||Y|| integral ||qv|| / beta.
This is a correct sufficient inequality but acts on arbitrary reduced states.
It can fail even when the input-driven error space is excellent. The main
continuous auxiliary method deliberately does not use that operator norm.

The current certificates use floating eigensolves, QR, matrix products and CG
without outward rounding. Positive-gram clipping and modal condition checks are
diagnostics, not a verified interval arithmetic implementation. The analytic
inequalities are continuous-domain statements; a `passed` JSON value only means
their ordinary floating evaluation passed. No academic novelty or cost victory
is inferred from a successful small-model evaluation.

The fixed-parameter driver also reuses the joint inverse to bound the residual
tail by an input-driven Riesz velocity. For Ya=K^-1 Da,

    r=-Ya y + integral exp(-C^-1 K(t-s)) Ya y'(s)ds
      + integral exp(-C^-1 K(t-s)) C^-1 Rbar u ds.

An approximate Ya uses its actual inverse equation defect, charged as
2/alpha times the integrated C^-1 dual norm of that defect times y'. It never
uses ||Ya|| on arbitrary augmented states. Both this bound and the matrix-energy
bound are valid, so the smaller may be selected on each time interval. The
initial analytic interval is chosen by its explicit error inequality, rather
than an unnecessarily tiny cutover that magnifies a sqrt(t) enclosure of
numerical inverse defects.
