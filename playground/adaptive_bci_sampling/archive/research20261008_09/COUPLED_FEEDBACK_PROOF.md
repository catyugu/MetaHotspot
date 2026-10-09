# Input-coupled reachable enclosures and composite tails

Exact-arithmetic statements, evaluated in ordinary floating point. No outward
rounding or independently validated native HTC model is supplied. Inputs are
arbitrary signed constant combinations, and V is the pre-SVD space.

## 1. An enclosure of the coefficient matrix, not the whole state energy ball

At a fixed real s>=0, set M=K(h_low)+sC, Q as in BOUNDARY_FEEDBACK_PROOF.md,
S=Q^T M^-1 Q, and group-separated orthonormal Z. Put Sz=Z^T S Z,
Fz=Z^T Q^T M^-1 G. The entire input-response coefficient MATRIX is

    c(Delta)=(I+Delta_r Sz)^-1 Delta_r Fz.

It is well-defined including zero Delta entries. All group parameters multiply
the same physical input u; coefficients are never allowed to vary independently
of u in the retained part of the enclosure.

On a parameter cell let dc be its midpoint, H=Sz^{1/2},
A=I+H dc H, J(delta)=A^-1/2 H(delta-dc)H A^-1/2,
g(delta)=A^-1/2 H(delta-dc)(Fz-Sz c0), and c0=c(dc).
An exact subtraction of the two equations gives

    c(delta)=c0+Sz^-1/2 A^-1/2 (I+J(delta))^-1 g(delta).

The affine input-coupled trial is
c_aff=c0+Sz^-1/2 A^-1/2 g(delta).
If rho=max_vertex ||J||<1, then

    c-c_aff=Sz^-1/2 A^-1/2 y,
    ||y u|| <= rho/(1-rho) max_vertex ||g|| ||u||.

Norm maxima at vertices are sufficient because J and g are affine. This is a
continuum enclosure of the source-coupled matrix family; its affine part retains
parameter/input cross terms and its norm ball covers only a SECOND-ORDER inverse
remainder. It does not claim to retain all input coupling in that remainder.
The enclosure shrinks quadratically with cell width, without increasing
polynomial degree. Subdivision is charged as small-matrix work, not free work.

For an output B(delta), one can avoid the extra norm split rho||B||:

    ||B(c-c_aff)|| <= max_(v,w) ||B(v) Sz^-1/2 A^-1/2 J(w)||
                         max_vertex ||g|| /(1-rho).

This follows from (I+J)^-1-I=-J(I+J)^-1 and convex vertex norm bounds.
It is used in the direct Galerkin tail to retain directional cancellation.

## 2. Exact omitted-direction residual, including CG and symmetry defects

Let computed F and T approximate M^-1G and M^-1QZ. Define
Fp=Q^T F, Tp=Q^T T, Szc=sym(Z^T Tp), Fzc=Z^T Fp. Define the proxy
Xhat=F-T c, with c solving the coefficient equation using Szc.
It is a trial field: no assumption that its diffusion solve is exact is needed.
For P=I-ZZ^T, dF=G-MF and dT=MT-QZ, its exact original equation defect is

    r=G-(M+Q Delta Q^T)Xhat
     =dF+dT c-Q Delta P(Fp-Tp c)
        +QZ Delta_r (Z^T Tp-Szc)c.

The last term charges symmetry repair. Group separation permits expressing this
as E times a small input-coefficient matrix, with

    E=[dF, U1,U2, dT,W1,W2],
    U_i=Q Pi P Fp,
    W_i=Q Pi P Tp+QZ Di(Z^T Tp-Szc),
    a(delta,c)=[I; -delta1 I; -delta2 I; c; delta1 c; delta2 c].

The coefficient trial a(delta,c_aff) is tensor degree two. Its nine Bernstein
controls, computed by a value-to-control transformation, retain every cross term
in the INPUT Gram. With a common dual Gram upper J_E,

    ||r||_(M^-1) <= max_control ||J_E^{1/2} a_control||
                    +max_vertex ||J_E^{1/2} a_c(delta) Lc|| epsilon_c,

where Lc=Sz^-1/2 A^-1/2. No singular-value tail of snapshots enters this bound.
Using the unsimplified defect G-A Xhat creates large artificial residuals of the
coefficient trial in RETAINED directions. The projected identity above removes
those by their exact reduced equation, rather than assuming they are small.

## 3. One fixed inverse Gram and a local feedback metric with no new RHS

For E, a computed inverse Y and D=E-MY, verified M>=alpha_s C gives

    J_lower=E^T Y+Y^T E-Y^T M Y,
    J_lower <= E^T M^-1 E <= J_lower+D^T C^-1 D/alpha_s.

This is square completion with the actual iterative defect. A single global
inverse is shared by every cell. Normal floating eigensolves and PSD clipping
are not an outward-rounded proof implementation; negative eigenvalues are
recorded. Analytic guarantees remain exact-arithmetic statements.

For the joint columns [E,QZ], compute the same inverse Gram upper J and split it
into EE, EQ, QQ blocks. For the cell's lower compressed feedback Dl>=0 define

    Phi(J)=J_EE-J_EQ(Dl^-1+J_QQ)^-1 J_QE.

Use the square-root formula at zero entries. For each input vector, Phi is the
minimum over b of [u;-b]^T J [u;-b]+b^T Dl^-1 b. Therefore Phi is monotone in J
in Loewner order, and bounds E^T (M+QZ Dl Z^T Q^T)^-1 E from above.
The actual operator on the cell is larger than this compressed lower operator,
so Phi also bounds the actual dual norm. The update costs small dense algebra,
zero new FOM RHS. It retains the physical source cross terms of the Gram.

A positive computed w with K(cell lower)w>=alpha_cell Cw supplies a Collatz
lower bound because the ACTUAL thermal operator is a symmetric M-matrix.
The compressed Robin operator need not be an M-matrix and is never used to
justify Collatz. A reconstructed w is only a candidate; positivity and the
actual product determine whether its lower bound can replace alpha_s.

## 4. Composite proxy omission and Galerkin reconstruction tail

Let V^T C V=I, Ar=V^T K(h)V+sI, B=V^T G, Xv=V Ar^-1B.
For the trial Xhat set zout=V^T C Xhat and Eout=Xhat-V zout. Then exactly

    Xhat-Xv=Eout-V Ar^-1 q,
    q=V^T r+V^T (K(h)+sC)Eout.

q is a coupled defect MATRIX. Bounding two independently enclosed response
coefficients instead would throw away their cancellation. Eout and q share the
same affine c trial and inverse remainder. q's trial is degree two; its control
matrices can be acted on by the center inverse Ar_c^-1, with a directional
inverse variation bound for Ar on the cell. This yields a dynamic reconstruction
upper g_C. The proxy full-equation residual gives

    ||x-Xhat||_C <= ||r||_(K(h)+sC)^-1 / sqrt(alpha_cell).

If p_C is that upper and d0 is sigma_min(C^1/2 Xhat_center) minus a verified
proxy variation, then ||x||_C >= (d0-p_C)||u||. Whenever d0>p_C,

    ||x-Xv||_C/||x||_C <= (p_C+g_C)/(d0-p_C).

This covers every parameter in the cell at the listed fixed real shift.
For the steady K error, Galerkin best approximation instead gives
||x-Xv||_K <= ||x-Xhat||_K+||Eout||_K. The denominator can use a variational
lower of G^T K(cell upper)^-1 G. No surrogate-error theorem is silently used
as a theorem for the specified V.

## 5. Direct Galerkin contrast and original steady acceptance

Define R0=G-CVB, D0=M V-CV Ar_low and Di=Hi V-CV Ai. Since the equation of
z=Ar^-1B is exact, the original Galerkin defect is

    rV=R0-(D0+sum delta_i Di)z.

This identity removes represented-state residual directions. The same
coefficient subtraction, degree-two controls, directional remainder and fixed
inverse Gram apply, now directly to the ORIGINAL V. It costs at most
m+(d+1)dim(V) inverse RHS once, not per cell. This extra cost is explicitly
reported, not attributed to port compression.

If a cell bound gives ||rV||_(K^-1)<=tau||u|| and
b0=lambda_min(B^T Ar_upper^-1 B), Galerkin orthogonality gives

    ||x-Xv||_K / ||x||_K <= tau/sqrt(b0+tau^2).

A complete leaf covering of the original box proves the steady condition when
every leaf passes epsilon=0.001. Checking centers does not. The archived leaf
coordinates and bounds describe that covering, and the computational cap may
leave a valid covering whose bound still fails.

## 6. All-time composition: exact theorem and remaining numerical obligation

Real-shift coverage is not a step guarantee. Here is a way to compose it without
assuming that frequency selection automatically proves the time contract.
Write A=C^-1/2 K C^-1/2, U=C^1/2 V, g=C^-1/2 G, b=U^Tg. Let [a,beta]
contain the full and reduced spectra for EVERY parameter. Let
f_t(lambda)=(1-exp(-t lambda))/lambda and choose, for every t,

    R_t(lambda)=w0(t)+sum_j wj(t)/(lambda+sj),  sj>=0.

Suppose eta(t) bounds |f_t-R_t| on the WHOLE spectral interval and dj bounds
|| (A+sj I)^-1g-U(Ar+sj I)^-1b || uniformly on the parameter box. Then functional
calculus and the triangle inequality give, for every t>0 and signed input,

    ||error_step(t)||_C <= [eta(t)(||g||+||b||)
          +|w0(t)| ||g-Ub||+sum_j |wj(t)| dj] ||u||.

The constant term vanishes if the physical input velocity is represented.
Let this upper be N(t). If a verified reduced step denominator D(t) satisfies
N(t)<=q D(t) for all t>0, the original relative full-field C error is <=q/(1-q).
The threshold is q=sqrt(0.001)/(1+sqrt(0.001)). Early time and infinite-time
limits must both be included. This is an explicit dynamic composition tail,
but not a claim that the present three real shifts provide its inputs.

The current experiments implement the continuous-parameter coefficient and
omission/reconstruction bounds at fixed real shifts and the steady contrast.
They do NOT yet verify scalar eta(t), all required shifts, and the denominator
on the entire time half-line. Sampled scalar approximation or step times would
not close this obligation. No all-time dual acceptance is declared.

## 7. Input energy congruence and a nested coefficient trial

The scalar b0 in section 5 can lose coupling between input directions. On each
cell instead set H_B=B^T Ar_upper^-1 B and change input coordinates by
T_B=H_B^-1/2. Replace EVERY input matrix by its product with T_B, including
R0, B, z0 and every trial control. The represented energy is now >=I. If the
normalized dual tail is tau, the steady relative bound is

    tau/sqrt(1+tau^2).

This is an invertible congruence preserving all signed physical input combinations.
It must not be applied only to the numerator or only to the denominator.

As a separate cost contrast, build U from input-driven coefficient solutions of
the PARENT Galerkin system, at cheap reduced-only parameter points. The child
field space VU is nested in the unchanged parent V. Its exact full equation
defect, not its snapshot tail, can be bounded by the same direct-tail construction
with m+(d+1)dim(U) full inverse RHS. Galerkin best approximation yields

    ||x-X_parent||_K <= ||x-X_child||_K.

Hence a child steady bound is also a bound for the original parent V, when the
normalizing full-response denominator is retained. This implication is steady
K only; it does not justify transferring a child step bound to the parent.
The implementation refuses a parent C step claim via this contrast.

A 16-direction trial failed even sampled steady checks. A 32-direction trial
improved samples but still failed the continuous tail at the tested covering
budget. This is evidence that input snapshot compression and an economical
strict continuous tail remain different problems.
