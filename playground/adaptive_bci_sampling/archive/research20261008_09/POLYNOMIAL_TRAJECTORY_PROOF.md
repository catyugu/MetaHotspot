# Parameter-polynomial trajectories with equation-defect certification

This is a sufficient continuous-parameter, all-time certificate for the same
pre-SVD step C-energy contract as RESIDUAL_RECONSTRUCTION_PROOF.md. It is not
a proof of novelty, a new extraction algorithm, or an outward-rounded numerical
certificate. The model and positive diagonal mass, conservative symmetric
M-matrix conduction, nonnegative diagonal Robin, and positive fixed HTC
assumptions are unchanged. All inequalities below refer to exact arithmetic.

## 1. Why change the enclosure object

The previous certificate bounds the parameter perturbation of the reduced
semigroup using an input-independent operator norm, and bounds the unretained
lift by another scalar norm. Both can be large even when their action on the
actual driven trajectory is small. Instead use a polynomial surrogate for the
small reduced trajectory, and certify that surrogate using its equation defect.
The surrogate does not replace the exported Galerkin ROM.

Let V be C-orthonormal, A(h)=VᵀK(h)V, B=VᵀG, and D(h)=K(h)V−CVA(h).
Use an invertible input change to make BᵀB=I, and retain R0=G−CVB explicitly.
For the exact reduced step z, z'+A(h)z=B, z(0)=0. In the following, z and p
are r-by-m matrices mapping arbitrary constant input vectors to states.

At tensor Chebyshev–Lobatto nodes h_j in the supplied box compute only the
reduced trajectories

    z_j(t) = f_t(A(h_j)) B,     f_t(λ) = (1−exp(−λt))/λ.

If ℓ_j(h) are the tensor Lagrange polynomials, define

    p(t,h) = Σ_j ℓ_j(h) z_j(t),
    q(t,h) = B−p'−A(h)p = Σ_j ℓ_j(h)[A(h_j)−A(h)]z_j(t).       (1)

The last identity follows exactly from Σℓ_j=1 and the node ODEs. No parameter
sampling assumption or unknown interpolation remainder is used: q is an
explicit polynomial in h with known matrix exponential time dependence.
It is zero at interpolation nodes, but it is its continuous enclosure that
certifies p between nodes.

## 2. Affine lift and its exact defect

At the arithmetic center c solve with the existing AMG-CG solver

    Y ≈ Kc⁻¹ Dc,                   d0 = Dc−KcY,
    Zi ≈ Kc⁻¹(Di−HiY),             di = Di−HiY−KcZi.

Keep T(h)=Y+Σδ_i Zi, δ=h−c. Its exact lifting defect is

    E(h) = D(h)−K(h)T(h)
         = d0+Σδ_i di−Σ_i,j δ_iδ_j HiZj.                      (2)

Unlike the previous construction, do not turn E into a global operator norm
and divide it by a coercivity bound to estimate L−T. Apply it to p directly.
Every actual CG defect is one of the explicit blocks in (2).

## 3. Error identity and bound

Let S_h(t)=exp(−C⁻¹K(h)t). With verified α>0 and β>0 such that
K(h)≥αC and A(h)≥βI throughout the box, the corresponding semigroups satisfy
||S_h(t)||_C≤exp(−αt) and ||exp(−A(h)t)||≤exp(−βt).
The full lower bound is obtained from the same positive supersolution/Collatz
test as the previous certificate: α=(1−ρ)min_i(Kc w)_i/(Cw)_i,
ρ=max_i(box width_i/(2c_i))<1. The reduced β is the smallest eigenvalue
of A at the all-low corner, using Loewner monotonicity of the Robin terms.

For y=x−Vp and ρ_p=−Tp, direct substitution gives

    C y'+K y = Kρ_p−E p+CVq+R0,    y(0)=ρ_p(0)=0.

Integration by parts therefore yields

    y = ρ_p−∫₀ᵗ S_h(t−s)ρ_p'(s) ds
        +∫₀ᵗ S_h(t−s)C⁻¹[−E p+CVq+R0] ds.

Also w=p−z satisfies w'+A(h)w=−q, w(0)=0. Since V is C-isometric,
for every input vector u the original ROM error obeys

    ||(x−Vz)u||_C / ||u|| ≤ N(t),

    N(t) = ||T p(t)||_C + ∫₀ᵗ ||T p'(s)||_C ds
           +∫₀ᵗ exp(−α(t−s)) ||C⁻¹/² E p(s)|| ds
           +∫₀ᵗ [exp(−α(t−s))+exp(−β(t−s))] ||q(s)|| ds
           + min(t,1/α) ||C⁻¹/²R0||.                         (3)

All matrix norms in (3) are spectral operator norms, so the bound covers
all input combinations, not only the four coordinate sources. The two q
convolutions arise from different equations, not an unexplained factor-two
relaxation of the original acceptance tolerance. Bounds below use the
undamped variation integral in (3), a deliberate remaining source of slack.

## 4. Continuous parameter enclosure

For tensor interpolation degree k in each coordinate:

| Object | Degree in each coordinate | Stored spatial blocks |
|---|---:|---|
| T p, T p' | ≤k+1 | Y and Zi, with their joint C Gram |
| E p | ≤k+2 | d0, di, HiZj, each with its C⁻¹ Gram |
| q | ≤k+1 | small reduced matrices from (1) |

Convert these exact power polynomials to tensor Bernstein controls. If
F(h)=Σ_ν b_ν(h)F_ν, the weights b_ν are nonnegative and sum to one, hence

    ||F(h)|| ≤ max_ν ||F_ν||.

The implementation retains all cross terms within Tp through its joint
Gram. For Ep it takes a triangle inequality between the blocks of (2),
while retaining all state and input directions within each block. This is
weaker than the full joint Ep Gram but is still a continuous-domain upper
bound. No spatial inverse is needed to raise k: only small ROM node
eigendecompositions and Bernstein algebra change.

## 5. All-time enclosure, including the infinite tail

Every control has the form F_ν(t)=Σ_j M_νj f_t(Λ_j) B_j, after the node
eigendecompositions. For modal rank-one weights

    w_νjℓ = ||(M_νj)_:ℓ||₂ ||(B_j)_ℓ:||₂,

the norm of the (s+1)-st derivative on [a,b], s≥0, is at most

    L_ν,s(a) = Σ_j,ℓ w_νjℓ λ_jℓ^s exp(−λ_jℓ a).

Thus, with μ=(a+b)/2,

    sup_[a,b] ||F_ν^(s)(t)||
      ≤ ||F_ν^(s)(μ)|| + (b−a)L_ν,s(a)/2.                   (4)

This encloses every time in the interval; midpoint evaluation is not a
sampling-based acceptance test. The implementation uses (4) for the separate
Ep defect blocks. For Tp and q, the modal absolute sums proved too loose in
the first large-model experiment, so instead form their exact integrated
derivative Grams. Stack all node modal columns of M into Mflat, all node
input rows into Bflat, and all positive node rates into λ. Then

    Q_ν(a,b) = Bflatᵀ [(Mflatᵀ Mflat) ∘ κ(a,b)] Bflat,
    κ_ij = exp(−(λ_i+λ_j)a)
           [1−exp(−(λ_i+λ_j)(b−a))]/(λ_i+λ_j).

Q_ν=∫_a^b F_ν'(s)ᵀF_ν'(s)ds, including cross terms between *different*
parameter nodes, not only within each node. For each fixed input vector,
time Cauchy–Schwarz gives

    ∫_a^b ||F_ν'(s)u|| ds ≤ sqrt((b−a)λmax(Q_ν)) ||u||.

The maximum over controls bounds the integral for a fixed parameter too:
its Bernstein weights do not depend on time, so convexity can be applied
before integration. Consequently both ∫||Tp' u|| and the variation of
Tp or q on the interval have such an input-uniform bound. This is an
upper bound on the supremum of input-specific variation, not generally
on the integral of the operator norm; (3) may safely be relaxed input by
input using this bound. The interval norm enclosure is the control norm
at a plus its Cauchy–Schwarz variation bound. For the infinite tail the
modal absolute sums above remain valid.

For a forcing upper bound g on [a,b], propagate
the scalar comparison ODE v'=−αv+g exactly:

    v_b = exp(−αΔ)v_a + f_Δ(α)g,
    sup_[a,b] v ≤ max(v_a,v_b).

Apply this separately to Ep, q with α, and q with β. The cumulative
variation at b and the interval bounds for Tp and the convolutions enclose
N(t) throughout [a,b].

For t≥T, f_t(λ)=1/λ−exp(−λt)/λ gives the tail enclosure

    ||F(t)|| ≤ ||F(∞)|| + Σ w exp(−λT)/λ.

The same modal sum bounds ∫_T^∞||Tp'||. A convolution beyond T is bounded
by max(v(T),g_tail/α). These statements certify the infinite tail.

For 0<t≤tiny use the original error equation directly:
||x−Vz||_C ≤ Dmax t²/2+r0 t, since ||z(s)||≤s under BᵀB=I.
The represented state satisfies ||z(t)u||≥t exp(−bmax t)||u||.
Consequently (Dmax tiny/2+r0)exp(bmax tiny) bounds error over ROM
throughout this initial interval; bmax is the maximum reduced decay at the
all-high corner. This covers times that would otherwise have a zero
denominator at the origin.

## 6. Relative full-state contract

For every parameter in the box, scalar functional calculus and inverse
Loewner monotonicity give

    Bᵀ f_t(A(h)) B ≥ t Bᵀ[I+t A(high)]⁻¹B.

The smallest eigenvalue of the right-hand side, d(t), bounds
||z(t)u||/||u|| from below. d(t) is nondecreasing, so d(a) can be used
over [a,b] and d(T) over the infinite tail. Let q_* be the resulting
uniform upper bound on N(t)/d(t). If q_*<1, the reverse triangle inequality
gives relative error over the full state ≤q_* /(1−q_*). If q_*≥1 this
certificate is unresolved; it does not imply that the ROM is inaccurate.

The implementation also retains the previous optional center/perturbation
denominator when it is sharper. Duhamel gives

    ||z_h(t)−z_c(t)|| ≤ ΔA f_t(β) f_t(λmin(Ac)),

where ΔA=max_vertices||A(h)−Ac||. This follows by bounding the convolution
with z_c(s), using ||z_c(s)||≤f_t(λmin(Ac)) for s≤t. Therefore over [a,b]
the center smallest singular value at a minus ΔA f_b(β)f_b(λmin(Ac))
is another valid lower bound; over the infinite tail use ΔA/(βλmin(Ac)).
Take the maximum with the positive rational lower bound. The choice changes
only the denominator, not the new equation-defect construction.

## 7. Steady K-energy from the same lift solves

At t=∞ the same surrogate error identity is algebraic:

    y∞ = −T p∞ + K(h)⁻¹[−E p∞+CVq∞+R0],
    p∞−z∞ = −A(h)⁻¹q∞.

Since K(h)≥αC and A(h)≥βI, inverse maps into the corresponding
energy norms satisfy

    ||K(h)⁻¹ r||_K ≤ ||C⁻¹/²r||/sqrt(α),
    ||V A(h)⁻¹ q||_K ≤ ||q||/sqrt(β).

Also K(h)≤(1+ρ)Kc, so Bernstein controls of Tp∞ with the joint Kc
Gram imply a bound on ||Tp∞||_K. Combining these observations bounds
the steady error numerator by

    Nsteady = sqrt(1+ρ) max_controls ||Tp∞||_Kc
              + [max_controls ||C⁻¹/²Ep∞||
                 +max_controls ||q∞||+r0]/sqrt(α)
              +max_controls ||q∞||/sqrt(β).                  (5)

The same block triangle for Ep as above can relax its control maximum.
The represented steady K-energy denominator is bounded below by
sqrt(λmin(Bᵀ A(high)⁻¹ B)). Divide (5) by this denominator and apply the
same q/(1−q) conversion to obtain relative error over the full steady
state. This certificate requires a Kc Gram of the retained lifts but **no
additional full inverse RHS**. Both steady≤0.001 and step≤sqrt(0.001)
must pass to satisfy the complete extraction contract. Old experiment
stages testing only the step condition are explicitly labeled as such.

## 8. Cost and limitations

The full inverse cost remains **1+(d+1)r RHS per box** (one supersolution,
r center lifts, d*r directional lifts). Increasing the interpolation degree
does not increase this cost, but its small-matrix/time-envelope work and
memory grow with (k+1)^d nodes and Bernstein controls. No full eigensolve
or full sparse direct factorization is used. A passing step certificate alone
does not suffice; the independent inequality (5) must pass as well.

This is a new prototype combination within this repository, not a verified
academic innovation. Elliptic reconstruction, equation-defect estimators,
polynomial interpolation, and Bernstein convex-hull bounds are established
tools. Relevant primary literature includes
[Makridakis–Nochetto (2003)](https://www.math.umd.edu/~rhn/papers/pdf/reconst.pdf),
[Ohlberger–Rave–Schindler (2017)](https://arxiv.org/abs/1606.09216), and
[Chellappa–Feng–Benner's corrected-ROM residual estimator](https://arxiv.org/abs/2307.11138).
The latter uses a different data-enhanced construction and does not establish
the present all-time, continuous-box conclusion. This limited search neither
proves priority nor rules out closer existing constructions.

Remaining obstacles include the elliptic total variation term, the minimum
decay used on unretained Ep, dimension-dependent inverse count, wide-box
polynomial degree and Bernstein slack, and lack of outward rounding.
