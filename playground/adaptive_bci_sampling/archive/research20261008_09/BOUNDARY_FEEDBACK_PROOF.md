# Robin feedback compression: exact identities and deterministic tails

This is a structural prototype, not the original continuous-parameter/all-time
Galerkin acceptance certificate. All statements below are in exact arithmetic.
Computed bounds use ordinary floating point, without outward rounding.

## Model and group constraint

Let C>0, K*=K(h_low)>0, K(h)=K*+Q Delta Q^T. The columns of Q are
face coordinate vectors weighted by square-root face area; Delta has a scalar
nonnegative value per Robin group. For each fixed s>=0 put
M=K*+sC, R=M^{-1}, S=Q^T R Q>0, F=Q^T R G.
Choose Euclidean orthonormal, group-separated Z. Thus P=I-ZZ^T commutes
with Delta and Delta Z=Z Delta_r. This group constraint is essential.

The compressed feedback model is
Khat(h)=K*+QZ Delta_r Z^T Q^T>0.
It keeps the full reference diffusion; it is not yet a cheap dynamic ROM.
Its response is xhat=RG u-RQZ c, with
c=(Delta_r^{-1}+S_z)^{-1} Z^T F u and S_z=Z^T S Z.
Inverse notation at zero coefficients means the continuous limit; the code
uses the equivalent square-root formula. There is no parameter polynomial.

## Exact source-driven innovation identity

The full feedback coefficient is a=(Delta^{-1}+S)^{-1} F u. Define
r=P(F u-SZ c). The reduced feedback equation implies
F u-SZ c-Z Delta_r^{-1}c=r. Consequently

\[
a-Zc=(\Delta^{-1}+S)^{-1}r,\qquad
x-\widehat x=-RQ(\Delta^{-1}+S)^{-1}r.
\]

The identities extend to zero Delta entries by continuity. They expose what
snapshots omit: both the direct source innovation PF and the feedback
innovation PSZ c(h,u). Small PF alone does not bound the second term.

## Passive deterministic innovation enclosure

Write T=(Delta^{-1}+S)^{-1}. Since 0<=T<=S^{-1},
T S T<=S^{-1}. Thus ||x-xhat||_M^2=r^T T S T r<=r^T S^{-1}r.
Set b=S_z^{-1/2}Z^T F u and y=S_z^{1/2}c. Passivity gives ||y||<=||b||:
the map from b to y is a symmetric positive contraction.
Let L=PSZ S_z^{-1/2}, R0=PF, and k=||S^{-1/2}L||_2.
For any tau>0 and any signed u,

\[
\|x-\widehat x\|_M^2\le u^T B_\tau u,\quad
B_\tau=(1+\tau)R_0^T S^{-1}R_0
 +(1+\tau^{-1})k^2 F^T Z S_z^{-1}Z^T F.
\]

This holds for all nonnegative Delta, not just a finite HTC grid. The code
chooses the smallest bound over a finite scalar tau grid; each choice is safe
in exact arithmetic. It does not claim an optimal enclosure.

For G^T C^{-1}G=I, let
 gamma=||C^{1/2}RQ S^{-1/2}|| and
 d=lambda_min(G^T(K(h_high)+sC)^{-1}G)>0.
Cauchy-Schwarz in the C metric and Loewner order give
||C^{1/2}(K(h)+sC)^{-1}G u||>=d||u||.
The same identity yields

\[
\frac{\|x-\widehat x\|_C}{\|x\|_C}
\le \gamma\sqrt{\lambda_{\max}(B_\tau)}/d.
\]

This is a continuous parameter bound at each specified real s. It is not a
bound over a frequency interval, complex frequencies, or every step time.
The current implementation evaluates it only on the small model.

The enclosure replaces the actual input-driven set of c(h,u) by an entire
energy ball. Its k need not decrease when Z grows. This loss, not snapshot
rank, is the experimentally identified obstacle to a useful source-driven tail.

## Independent, uniform steady operator domination

The omitted Robin matrix is L(h)=QP Delta P Q^T>=0, so K=Khat+L.
Let Lmax=QP Delta_max P Q^T and
 theta=lambda_max(K*^{-1/2}Lmax K*^{-1/2}).
Then L<=Lmax<=theta K*<=theta Khat.
For x=K^{-1}Gu and xhat=Khat^{-1}Gu, xhat-x=Khat^{-1}Lx.
In Khat coordinates this map has eigenvalues in [0,theta]; the K metric
is I plus the same symmetric matrix. Hence

\[
\|x-\widehat x\|_{K(h)}/\|x\|_{K(h)}\le\theta
\]

uniformly over the original continuous HTC box and arbitrary signed inputs.
This theorem is source-independent and steady only. It certifies the Robin
surrogate, not the separately constructed Galerkin V. In this experiment its
bound is far too large to satisfy epsilon=0.001.

## What a complete extractor would still need

1. A tighter deterministic enclosure of PSZ c(h,u) preserving its physical
input coupling, rather than permitting every reduced coefficient direction.
2. A fixed, economical reference diffusion approximation with its own tail.
3. A composition theorem for the specified Galerkin V, steady K error and
relative C step error for every t>0 on the same parameter box.

Real-resolvent checks and sampled step oracles cannot replace item 3. The
current work isolates these obligations instead of declaring snapshots a tail.
