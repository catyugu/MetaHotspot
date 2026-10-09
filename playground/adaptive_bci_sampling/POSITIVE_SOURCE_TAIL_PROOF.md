# Source-driven positive tails and a modal reachable enclosure

Date: 2026-10-09. Baseline: `2f7a709`. This is a new research attempt, **not a
successful certified extraction algorithm**. Production extraction is unchanged.
All statements in this note are exact-arithmetic statements. Implementations use
ordinary floating point, actual AMG-CG defects, and no outward rounding. Case1
is still the matrix-only reconstruction (`native_validated=false`).

## 1. Contract and additional structure

Let C be positive diagonal, K(h)=K0+sum hi Hi SPD on a positive parameter box,
K0 a symmetric graph Laplacian, and Hi nonnegative diagonal. Let G>=0 have m
columns. Positive diagonal input rescaling is allowed; we choose
G^T C^-1 G=I, possible here because the four sources have disjoint support.
This preserves positivity and the entire arbitrary signed input span.
Parameters are constant in time; x(0)=0 and Cx'+K(h)x=Gu.

K(h) is an irreducible nonsingular M-matrix. Consequently its inverse and
exp(-C^-1 K(h)t) are entrywise nonnegative. Matrix inequalities written with
entrywise absolute values below are **entrywise**, not Loewner inequalities.
The C norm of a field is the Euclidean norm after multiplying by sqrt(C).

The target remains uniform relative K-energy steady error <=0.001 and uniform
relative C-energy step error <=sqrt(0.001) for the SAME pre-SVD Galerkin V.
Neither a parameter-kernel tail alone nor a sampled error spectrum meets it.

## 2. A source-seeded parameterized dynamic reachable family

On a cell [ell,b] choose the UPPER endpoint Kb=K(b), and let
Delta(h)=Kb-K(h)>=0 be diagonal. Set Hb(t)=exp(-C^-1 Kb t) C^-1.
Define matrix-valued, actual-input-driven step terms

    X0(t)=integral_0^t Hb(v) G dv,
    Xj(t,h)=integral_0^t Hb(t-v) Delta(h) X(j-1)(v,h) dv.

Every column of every Xj is nonnegative. The finite-dimensional
Dyson--Phillips/variation-of-constants expansion gives

    X(t,h)=sum_{j>=0} Xj(t,h).

Each term is an integral of a positive impulse kernel, so it increases
entrywise in t and converges to its steady value. Write
Delta(h)=sum pi Di, pi in [0,1], Di=(bi-elli)Hi. Then Xj is the sum of
all noncommuting words of length j in the Di, multiplied by the corresponding
parameter monomials. Its initial forcing is always G, never an unrestricted
boundary or energy-ball input. The closed span of the ranges of these word
kernels for all times is a common source-driven reachable space. This is a
definition of the full space; it does NOT assert that it is small.

In particular, static word snapshots do not span all dynamic word kernels in
general. Omitting words and compressing their spatial ranges are two different
operations and must be certified separately.

## 3. Deterministic terminal-response tail, without rho^N/(1-rho)

Let D=Kb-K(ell), S0=Kb^-1 G, Sj=Kb^-1 D S(j-1), and

    ThetaN = K(ell)^-1 D SN.

Then, for every h in the cell and every t>=0,

    0 <= EN(t,h):=X(t,h)-sum_{j=0}^N Xj(t,h) <= ThetaN.

Proof: positivity makes every word monotone in every pi; its maximum is at
pi=1. Integrating the positive omitted impulse kernels over [0,t] is at most
their integral over [0,infinity). At pi=1 those integrals are Sj. Since
T=Kb^-1 D>=0 and I-T=Kb^-1 K(ell), rho(T)<1 and

    sum_{j>N} Sj = (I-T)^-1 T S_N = K(ell)^-1 D S_N.

The last identity follows directly by multiplication, so noncommutation does
not invalidate it. It is also verified independently by subtraction on a
small model. For ANY signed input combination u,

    |EN(t,h)u| <= ThetaN |u|,
    ||EN(t,h)u||_C <= aN ||u||, aN=||sqrt(C) ThetaN||_2.

Only four RHS are needed for each terminal solve, after source-seeded recurrence
construction. The bound pays for the actual last forcing D SN and its actual
propagation, rather than for an arbitrary direction. The terminal response is
the exact worst-cell steady parameter remainder before numerical enclosure.
Its C operator norm need not be the best possible signed, finite-time bound.

The same recurrence with Kb+sC and K(ell)+sC proves a continuous-cell
resolvent tail for s>=0. A real-shift tail alone is not a step certificate.

## 4. Close the near-zero relative normalization

Let d=max diag(D)/diag(C). For the graph Markov generator -C^-1 K0,
Feynman--Kac represents the upper-endpoint heat kernel with diagonal killing.
Replacing Kb by K(h) multiplies each path weight by exp(J), where
0<=J<=dt over a path of duration at most t. Keeping j=0..N insertion terms
keeps the first N+1 powers of J. Hence

    0 <= EN(t,h) <= pN(dt) X(t,h),
    pN(v)=1-exp(-v) sum_{j=0}^N v^j/j!.

This can also be proved directly by expanding the path occupation functional
and ordering its integration variables. It is not a probabilistic certificate:
the Poisson survival function is only a deterministic scalar remainder formula.

Since the full semigroup contracts in C norm and G^T C^-1 G=I,
||X(t,h)|u|||_C<=t||u||. Combining with Section 3 gives the absolute bound

    ||EN(t,h)u||_C <= min(aN, t pN(dt)) ||u||.

Now let g=C^-1/2 G u/||u|| and A=C^-1/2 K(h) C^-1/2.
The scalar function ft(lambda)=(1-exp(-t lambda))/lambda is positive, decreasing
and convex (it is integral_0^t exp(-s lambda) ds). Scalar spectral Jensen and
Cauchy--Schwarz give

    ||X(t,h)u||_C /||u|| >= g^T ft(A) g
        >= ft(g^T A g) >= ft(L),
    L=lambda_max(G^T C^-1 Kb C^-1 G).

Only a four-input Gram is used; this is not a full-FOM maximum eigenvalue.
Therefore the uniform relative PARAMETER tail is bounded by

    sup_{t>0} min(aN, t pN(dt))/ft(L).

The increasing branch t pN(dt)/ft(L) and the decreasing branch aN/ft(L)
cross at the unique t* satisfying t* pN(dt*)=aN. Monotonicity of the first
branch follows from monotonicity of pN and of t/ft(L). The supremum is exactly
aN/ft*(L) for this scalar majorant. Degenerate d=0 or aN=0 gives zero.
This includes t->0 and t->infinity without finite time sampling.

This solves a missing normalization for the parameter remainder. It does NOT
make the finite term realization low dimensional or certify its projection.

## 5. A common, input-driven modal cone for the actual Galerkin ROM

Let V^T C V=I, Ar(h)=V^T K(h)V, B=V^T G, and z'+Ar(h)z=B u.
Use Ar(b)=U Lambda U^T, Lambda>0, and q=U^T z. Define

    Pabs=sum_i (bi-elli) |U^T (V^T Hi V) U|,
    M=Lambda-Pabs.

Pabs is symmetric and entrywise nonnegative. If
lambda_max(Lambda^-1/2 Pabs Lambda^-1/2)<1, M is an SPD M-matrix.
For the matrix of source coefficients Q(t,h), the exact differential comparison

    d|Q|/dt <= -M |Q| + |U^T B|

holds in the usual upper Dini-derivative sense. Therefore

    |Q(t,h)| <= integral_0^t exp(-M s)|U^T B| ds
              <= Qbar:=M^-1 |U^T B|.

This treats the entire reduced parameter feedback, with no Taylor truncation,
and keeps the source columns. The image VU of this parameter-dependent cone
is a common reachable enclosure. It does not cover an arbitrary coefficient
ball. Taking absolute values still removes correlations and can be very loose.

## 6. Deterministic discarded-spatial-dynamics bound for this V

Set R0=G-CVB and D(h)=K(h)V-CV Ar(h). The exact Galerkin error obeys

    Ce'+K(h)e = R0-D(h)z, e(0)=0.

Let Dabs be the entrywise maximum of |D(h)U| over cell vertices. Since D is
affine, convexity of scalar absolute value gives |D(h)U|<=Dabs throughout
the cell. Put bsrc=|R0|+Dabs Qbar and Y=K(ell)^-1 bsrc.
The full semigroup is positive, and increasing diagonal killing makes it
entrywise smaller. Consequently

    |e(t,h)u| <= Y |u|, ||e(t,h)u||_C <= a ||u||,
    a=||sqrt(C)Y||_2.

This is a bound on ALL omitted spatial/dynamic directions of the specified
Galerkin V, not a snapshot singular-value surrogate. It needs just m terminal
full-order RHS per cell after Qbar is obtained. No m-by-n unrestricted input
is introduced. The bound can be useless even though it is mathematically valid.

For near zero, define eta0=||C^-1/2 R0||_2 and
eta1=0.5 max_vertex ||C^-1/2 D(h)||_2 ||B||_2.
Because ||z(s,h)||<=s ||B|| ||u|| and the FOM semigroup contracts in C norm,

    ||e(t,h)u||_C <= (eta0 t+eta1 t^2)||u||.

Using the Jensen lower bound from Section 4 gives the continuous-cell,
all-time relative Galerkin certificate

    sup_{t>0} min(a,eta0 t+eta1 t^2)/ft(L)
      = a/ft*(L),
    eta0 t*+eta1 t*^2=a.

The degenerate cases can be handled by continuity. The scripts evaluate the
positive cases encountered here with the stable quadratic root formula.
This is a sufficient step certificate only. A K-energy steady certificate
must still be proved and pass separately for the same V.

## 7. Enclose actual linear-solve defects without hiding them

For any nonsingular M-matrix A used above, retain a positive supersolution
w>0 with Aw>0. Given approximate Y and F=B-AY, set

    eta_j=max_i |F_ij|/(Aw)_i.

Inverse positivity proves |A^-1 B-Y|<=w eta^T. For a nonnegative right-hand
side, max(Y+w eta^T,0) is an upper enclosure. Propagating these upper
solutions through the positive recurrence and the terminal solve preserves
every previous upper inequality. A single w from the global lower-HTC
supersolution works for all cells and s>=0 since A w>=K(global_lo)w>0.

The scripts verify positivity by actual products, not by a Ritz minimum.
This is a defect enclosure in exact arithmetic evaluated in ordinary float;
outward rounding of all products/solves remains unimplemented.

## 8. Algorithm, cost, and stopping rule

1. Assemble C,K0,Hi,G; verify the diagonal/nonnegative/M-matrix assumptions.
2. Normalize the sources by positive diagonal scaling and obtain one positive
   supersolution by AMG-CG, checking the actual product.
3. Build a candidate common V from C^-1G and source-only resolvents at nine
   HTC nodes and four real shifts; C-orthogonalize, no final SVD. This is a
   candidate-selection rule, NOT the continuous-domain argument.
4. For each cell, attempt the modal M-matrix condition. If it fails, report
   no certificate for this enclosure; never silently use a sampled maximum.
5. If it holds, compute the input cone Qbar, the four-column residual forcing,
   its enclosed terminal response, and the all-time relative majorant.
6. In parallel, diagnose parameter-word truncation with its exact source
   terminal tail and Poisson/Jensen relative majorant. Do not add it to the
   actual-ROM bound unless the finite-series realization is actually used.
7. Only accept extraction after both the step bound and a separate steady
   K-energy bound pass on a covering of the original box. That acceptance
   was NOT reached in this experiment.

The four-shift candidate costs 145 full-order RHS and 37 AMG setups at every
mesh. Modal certification charges its separate supersolution and four RHS per
successful cell. Positive recurrence costs grow with retained degree and cell
count. The whole-box slow-mode failure makes high-degree/large-cover rescue
expensive; that is a negative result, not a proposed innovation.

The classical ingredients (positive semigroups, Feynman--Kac, comparison
systems, Neumann expansion, scalar Jensen) are not claimed as novel. The
source-terminal-tail composition is a useful independently proved tool for
this project, but no academic novelty or improved accepted extraction cost
has been established. See the companion research record for evidence.

## 9. Independent error-space reconstruction and its discarded-direction audit

`input_error_reachability.py` additionally constructs a common error trial U
directly from the FOUR source columns of

    E(s,h)=(K(h)+sC)^-1G-V(Ar(h)+sI)^-1B.

The training fields are normalized in input coordinates by the inverse square
root of X(s,h)^T C X(s,h). They are sampled on four endpoint HTC combinations
and three positive real shifts. The resulting SVD only selects U; it does not
certify a continuous manifold. Disjoint HTC nodes and disjoint shifts audit it.
For any selected C-orthonormal U, define the practical residual-driven proxy

    R=G-(K(h)+sC)V(Ar(h)+sI)^-1B,
    Ehat=U[U^T(K(h)+sC)U]^-1 U^T R,
    F=G-(K(h)+sC)(X_ROM+Ehat).

Then the EXACT omitted error is (K(h)+sC)^-1 F. Inverse positivity and
K(h)+sC>=K(global_low) entrywise on the diagonal imply

    |E-Ehat| <= K(global_low)^-1 |F|.

The terminal solve uses the actual four-input F and the defect enclosure in
Section 7. This is a deterministic bound for all discarded spatial directions
at the stated h,s. It makes no parameter-continuum or time-domain claim.

Reference solves used to measure reconstruction error also carry a defect
enclosure: if delta=||sqrt(C) reference_correction||/sigma_min(sqrt(C)Xref)<1,
a measured relative q yields the true upper (q+delta)/(1-delta). The positive
terminal numerator is divided by the same certified minimum response norm.
No reference solve is assumed exact. The best C-projection error is reported
separately to distinguish a deficient common space from proxy coefficient error.
