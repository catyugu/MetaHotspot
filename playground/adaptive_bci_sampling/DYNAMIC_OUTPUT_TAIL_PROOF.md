# Continuous-HTC, input-driven dynamic output tail (2026-10-09)

This note proves two **integrated reconstruction-tail** bounds. Neither is a
complete certificate of the original full-field relative step error. The model
is matrix-only Case1, native_validated=false. Parameters are fixed in time;
inputs are arbitrary signed constant four-source combinations, zero initial
state. Statements below are exact-arithmetic statements; scripts evaluate them
in ordinary floating point, without outward rounding.

## 1. Exact cancellation and the quantity that matters

Assume C>0, K(h)>0, V^T C V=I, A(h)=V^T K(h)V, B=V^T G,
D(h)=K(h)V-CVA(h), R0=G-CVB. The ROM trajectory z obeys
z'+A(h)z=B, z(0)=0. For ANY proxy p, put w=z-p and q=B-p'-A(h)p.
Then w'+A(h)w=q. If y=x-Vp and e=x-Vz=y-Vw, direct subtraction gives

    Ce'+K(h)e=R0-D(h)p-D(h)w.

The CVq term cancels exactly. Bounding y and Vw separately can destroy that
cancellation. The required dynamic output is D(h)w, not the unweighted w.
For K(h)>=alpha C and r=R0-Dp-Dw, Young's inequality gives

    (||e||_C^2)' + alpha ||e||_C^2 <= ||r||_{K(h)^-1}^2.

The exponentially damped time kernel must be kept when composing a full error
bound. An undamped infinite integral of a nonzero steady residual diverges.
Also, a bound on ||e|| alone does not supply a relative bound near t=0.

## 2. Proxy with the exact reduced steady response

On a continuous two-parameter cell choose its arithmetic center hc and Ac=A(hc).
Let E(h)=A(h)-Ac and zinf(h)=A(h)^-1 B. Use

    p(t,h)=(I-exp(-Ac t)) zinf(h),
    w(t,h)=(exp(-Ac t)-exp(-A(h)t)) zinf(h).

Then w(0)=w(infinity)=0 and w'+A(h)w=E(h)exp(-Ac t)zinf(h).
This is the stable augmented system with

    L(h) = [[ A(h), -E(h) ], [ 0, Ac ]],
    v(0,h) = [0; zinf(h)],  output = [D(h),0] v.

Thus the relevant Gram is zinf(h)^T P22(h) zinf(h), a FOUR by FOUR
input Gram, not the largest eigenvalue of an unrestricted state covariance.
The steady-calibrated proxy is a certificate device, not a replacement ROM.

## 3. Certified inverse Gram, including AMG-CG defects

Let Klo=K(h_domain_low)>=alpha C. Let M=[D0,D1,D2], with D(h)=M T(h)
and T(h)=[I;h1 I;h2 I]. Solve Klo Y approximately and set F=M-Klo Y.
The identity

    M^T Klo^-1 M = M^T Y + Y^T M - Y^T Klo Y + F^T Klo^-1 F

implies the upper Gram Q=M^T Y+Y^T M-Y^T Klo Y+F^T C^-1 F/alpha.
Therefore N(h)=T(h)^T Q T(h)>=D(h)^T K(h)^-1 D(h).
All 3r AMG-CG RHS are charged. No large-FOM spectrum or dense direct solve is
used. The alpha evaluation and its RHS are charged in candidate preparation.

## 4. Common storage baseline: valid but very conservative

If P>=0 and L(h)^T P+P L(h)>=diag(N(h),0) throughout the cell, integration gives

    integral_0^infinity ||D(h)w(t,h)u||_{K(h)^-1}^2 dt
       <= u^T zinf(h)^T P22 zinf(h) u.

The script first solves the CENTER Lyapunov equation for P0. With beta=min eig
A(cell_low), bc=min eig Ac, Emax=max_vertex ||E||, take
S=diag(I,kappa I), kappa=1+Emax^2/(beta bc), d=min(beta,bc).
A Schur-complement calculation gives L^T S+SL>=d S throughout the cell.
The deficit diag(N,0)-L^T P0-P0 L is tensor degree two. Its nine Bernstein
controls dominate the entire cell by convexity; PSD checks at parameter
vertices alone would NOT suffice. Set

    delta=max(0,max_control lambda_max(S^-1/2 deficit S^-1/2))/d,
    P=P0+delta S.

This is sufficient in exact arithmetic. Our experiments show the repair once
again pays for unreached states and is catastrophically conservative. This
failure applies to THIS center-storage-and-isotropic-repair construction, not
to every optimized parameter-dependent storage function.

To enclose zinf(h) in any output metric R, use the exact resolvent identity

    zinf = zc - Ac^-1 E zc + Ac^-1 E A(h)^-1 E zc,  zc=Ac^-1 B.

The affine-output norm is maximized at vertices. A remainder bound is
max_vertex ||R Ac^-1 E|| * max_vertex ||E zc|| / beta. This is an
inverse remainder, not evidence that the entire coefficient ball is reachable.

## 5. Input-driven first variation with a deterministic Neumann remainder

Define Fh=Ac^-1/2 E Ac^-1/2 and rho=max_vertex ||Fh||. Assume rho<1.
For x=Ac^1/2 w and vc=Ac^1/2 exp(-Ac t)zinf, the exact half-line
convolution equation is

    x = T Fh (vc-x),
    T(s)=Ac^1/2(sI+Ac)^-1 Ac^1/2, ||T||_(L2->L2)<=1.

Keep the first variation driven by the ACTUAL source matrix zc:
w1'+Ac w1=E exp(-Ac t)zc, w1(0)=0. Since E=delta1 A1+delta2 A2,
use a fixed 3r-state realization [w11;w12;exp(-Ac t)zc] with generator

    [[Ac,0,-A1],[0,Ac,-A2],[0,0,Ac]].

Its initial columns are [0;0;zc]. The output D(h)(delta1 w11+delta2 w12)
is degree two in h. Convert its OPERATOR coefficients to nine Bernstein
controls O_j. Each exact Lyapunov solve with O_j^T Q O_j produces a source
Gram J_j (four by four). Convexity in the Hilbert space L2 then proves

    ||D(h)w1 u||_(L2,Klo^-1) <= a ||u||,
    a=max_j sqrt(lambda_max(J_j)).

This is a continuous-cell bound retaining parameter and input coupling in the
first variation. It is not merely interpolation of sampled tail norms.

The current prototype deliberately exposes the conservative remainder:
let bc=min eig Ac, Mout=max_vertex ||Q^1/2 T(h) Ac^-1/2||, and

    dz = max_vertex ||Ac^-1/2 E zc|| / (sqrt(bc)(1-rho)).

Then ||zinf-zc||<=dz, and ||Ac^1/2 exp(-Ac t)Z||_L2=||Z||/sqrt(2).
Writing x1=T Fh Ac^1/2 exp(-Ac t)zc gives

    ||x-x1||_L2 <= [rho dz + rho^2/(1-rho)(||zc||+dz)]/sqrt(2).

Hence b=Mout times the right side and

    integral ||D(h)w u||^2 <= (a+b)^2 ||u||^2

uniformly on the continuous cell. Mout is vertex-bounded because D(h) is affine.
This tail is deterministic but still replaces higher-order reachable directions
with an induced-norm ball. The experiments show that b, not a, is the bottleneck.
No tight whole-domain guarantee is claimed.

## 6. Independent fixed-point step audit without FOM eigendecomposition

For N>=2 backward Euler steps, z=t lambda, the spectral multiplier difference is
(t/z)[(1+z/N)^(-N)-exp(-z)]. Since
z-N log(1+z/N)<=z^2/(2N), its magnitude is at most

    t/(2N) ((N-1)/N)^(N-1).

This follows by maximizing z(1+z/N)^(-N), whose maximizer is N/(N-1).
Multiply by ||C^-1/2 G|| to get an absolute C-norm reference error bound.
For each actual linear solve defect dj, propagate the additional bound
eta_j=(eta_(j-1)+||C^-1/2 dj||)/(1+dt alpha).
This includes CG defects instead of assuming exact BE solves.
If dmin=sigma_min(C^1/2 X_BE), delta=(BE_bound+eta)/dmin and
q=||X_BE-X_ROM|| relative to X_BE, the true relative error lies in

    [max(0,q-delta)/(1+delta), (q+delta)/(1-delta)]

when delta<1. Small-FOM spectra independently audit this inequality.
Formal results remain FIXED h and FIXED t, not continuous or all-time acceptance.

## 7. Literature and scope of originality

Grepl & Patera (2005), DOI 10.1051/m2an:2005006, develops rigorous parabolic
reduced-basis output estimators and parameter/time sampling. Rigorous parabolic
residual estimation is established methodology, not our novelty.
https://numdam.org/articles/10.1051/m2an:2005006/

Kojima (2019), DOI 10.9746/sicetr.55.429, uses Bernstein bases to turn polynomial
parameter-dependent LMIs into finite sufficient tests and domain subdivision.
Bernstein continuum checks and parameter-dependent Lyapunov functions are
established tools, not our novelty.
https://www.jstage.jst.go.jp/article/sicetr/55/7/55_429/_article/-char/ja

Benner et al., A bilinear H2 model order reduction approach to linear
parameter-varying systems (2019), develops Volterra-kernel reachability/
observability Gramians and generalized Lyapunov equations. See Sec. 2,
particularly recurrence (2.6) and generalized equations (2.9), (2.11).
https://pure.tue.nl/ws/files/145627013/Benner2019_Article_ABilinear_2MathcalH2ModelOrder.pdf

A potentially useful next research hypothesis is a matrix-valued majorant of the
entire output-weighted Volterra tail, tested only after restriction to the actual
source columns and coupled back to R0-Dp. The published multi-time bilinear H2
kernel norm is NOT automatically a certificate for our single-clock,
constant-parameter step trajectory. That bridge and the pointwise relative
normalization still need proof. No academic novelty is asserted here.
