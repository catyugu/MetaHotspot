> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# Continuous affine HTC bridge for all-time full-field step error

## Scope and correction of the literature metric

The target remains the user's full-field metric, not a port metric: for every
constant input vector u, every fixed h in a parameter box, and every t>0,

    ||(X_h(t)-X_V,h(t))u||_C / ||X_h(t)u||_C <= 2 sqrt(epsilon).

The companion steady K-energy target is 2 epsilon. The limit at t=0 is understood
from the right. C is positive definite, K(h) is symmetric positive definite,
G has independent columns, and V is fixed after SVD. No time-varying HTC claim
is made. All certificates below are exact-arithmetic statements; the prototype
uses ordinary floating-point eigensolves, without outward rounding.

Siemens' **Simcenter Flotherm BCI-ROM Validation, version 2020.2, December 2020**,
section 1, states a Hankel-norm impedance error threshold 2 epsilon and a
space-time energy error threshold 2 sqrt(epsilon) for power impulses. Those are
not a pointwise full-field step relative error statement. Its numerical validation
uses monitor temperatures normalized by the full-model steady temperature.
Source: https://assets.ctfassets.net/dww76w587oxz/1ePeFrdaZXfoq0E1cFuM0T/e71cbcd824d7962c90319da1db28548a/Simcenter-Flotherm-BCI-ROM-Validation-2020.2.pdf

A related author paper, *Circuit-Based Electrothermal Simulation of Multicellular
SiC Power MOSFETs Using FANTASTIC*, Energies 13 (2020), 4563, in its algorithm discussion, writes an
H2-labelled impulse-response integral and calls it a Hankel norm. Its displayed
integral is the Frobenius impulse L2/H2 quantity; it should not silently be
replaced with the induced Hankel-operator norm. Norm definitions, output maps,
and input aggregation must be checked when importing a theorem.
Source: https://www.mdpi.com/1996-1073/13/17/4563/html

Frequency selection and rational interpolation supply temporal approximation
under their hypotheses. Uniform parameter-domain interpolation/residual control
and preservation after SVD are still required. The stock source-dependent spectral
estimates and randomized HTC tests do not themselves establish those hypotheses
for the whole affine family. This bridge accepts any valid center time bound;
therefore a compatible rational-interpolation theorem can replace the prototype's
center spectral audit without changing the parameter proof.

## 1. Relative form radius for an entire parameter box

Transform to C-normal coordinates:

    A_h=C^(-1/2) K(h) C^(-1/2), B=C^(-1/2)G,
    F_h(t)=f_t(A_h)B, f_t(lambda)=(1-exp(-lambda t))/lambda.

Let A=A_center and Delta=A_h-A. Define

    rho=max_{corners h} ||A^(-1/2) Delta A^(-1/2)||_2 < 1.

Convexity of the spectral norm and affine dependence extend the corner bound to
all interior parameters, including signed affine terms. Thus

    (1-rho) A <= A_h <= (1+rho) A,
    lambda_min(A_h) >= alpha=(1-rho) lambda_min(A).

The implementation uses generalized eigenvectors of (K_center,C), which evaluates
the same norm without explicitly forming C^(-1/2). The corner step requires
2^d norm evaluations, not PDE inverse actions.

Put eta=rho/sqrt(1-rho). Then

    Delta=A_h^(1/2) L A^(1/2), ||L||_2 <= eta.

Proof: write Delta=A^(1/2) E A^(1/2), ||E||<=rho, and use
||A_h^(-1/2) A^(1/2)||<=1/sqrt(1-rho).

## 2. Uniform short-time perturbation

Duhamel gives, with arbitrary input coordinates Z,

    (F_h(t)-F(t))Z
      = - integral_0^t exp(-A_h(t-s)) Delta F(s)Z ds.

For any positive operator T and q>0,

    ||T^(1/2) exp(-T q)|| <= 1/sqrt(2 e q),
    ||A^(1/2) f_s(A)|| <= sqrt(s).

The second inequality follows from 1-exp(-x)<=min(x,1)<=sqrt(x).
Consequently

    ||(F_h(t)-F(t))Z||
      <= c_s eta t ||BZ||, c_s=pi/(2 sqrt(2 e)),

because integral_0^t sqrt(s/(t-s)) ds=pi t/2. This controls the
relative error as t approaches zero, with no stiffness-maximum factor.

## 3. Uniform long-time perturbation

Let X=A^(-1) B and X_h=A_h^(-1) B. The inverse identity yields

    ||(X_h-X)Z|| <= eta/sqrt(alpha) ||A^(1/2) XZ||.

A second Duhamel identity, applied to the semigroups, gives

    ||exp(-A_h t)-exp(-A t)|| <= c_l eta,
    c_l=pi/(2e),

since integral_0^t ds/sqrt(s(t-s))=pi. Splitting the step difference,

    F_h-F=(I-exp(-A_h t))(X_h-X)
                  +(exp(-A t)-exp(-A_h t))X,

and ||I-exp(-A_h t)||<=1 proves

    ||(F_h-F)Z|| <= eta/sqrt(alpha) ||A^(1/2) XZ||
                                      + c_l eta ||XZ||.

These are relative-form estimates. They avoid the absolute perturbation factor
||Delta||/alpha^2 typical of a direct stiffness Lipschitz bound. They retain the
input-dependent operator norms; no Frobenius sum over inputs is required.

## 4. A finite certificate covering every time

Q(t)=F(t)^T F(t)=B^T f_t(A)^2 B is increasing in Loewner order.
For a finite interval [a,b], choose Z=Q(a)^(-1/2). Every input combination is
bounded on that interval by the smaller of:

    c_s eta b ||BZ||,
    eta/sqrt(alpha)||A^(1/2)XZ||+c_l eta||XZ||.

The implementation obtains Z by Cholesky whitening; it need not be the symmetric
square root. All statements are invariant under invertible input coordinates.

For 0<t<=t0, f_t(lambda)/t decreases with t. Hence the initial interval bound is
c_s eta ||B Z0||, where Z0 whitens Q(t0)/t0^2. For all t>=T use the long-time
formula with ZT whitening Q(T). The intervening finite log intervals cover all
remaining times. Their number changes tightness, not validity: no time samples
are being accepted in place of intervals. t0 and T need not bound the true
dynamics asymptotically; the two endpoint formulas hold on their entire tails.

Denote the maximum of these finitely many upper bounds by d_F. This proves

    ||(F_h(t)-F(t))u|| <= d_F ||F(t)u||

simultaneously for every box parameter, time and input. The same construction
in reduced C-normal coordinates proves a variation d_R for the Galerkin state.

## 5. Transfer the center certificate to the continuous cell

Suppose the final SVD basis has a proven center bound

    ||F(t)u-R(t)u|| <= e ||F(t)u|| for all t>0,u.

Then ||R(t)u||<=(1+e)||F(t)u||. Combining both parameter variation estimates and
using ||F_h(t)u||>=(1-d_F)||F(t)u|| proves, when d_F<1,

    sup_{h,t,u} ||F_h(t)u-R_h(t)u||/||F_h(t)u||
      <= [e+d_F+(1+e)d_R]/(1-d_F).

Reduced C-normalization is an isometry when lifting through V. This is why the
reduced bound applies to the full-field error. SVD effects are included by using
the final basis in the center audit and the reduced affine family.

This theorem separates temporal approximation from the affine-domain extension.
It does not assume that a port Hankel bound is already a full-field step bound.

## Algorithm and limitations

1. Fix the final basis and a candidate parameter cell.
2. Obtain a center all-time bound in the desired full-field norm.
3. Compute full/reduced relative form radii from parameter corners.
4. Evaluate the two analytic variation envelopes on covering time intervals.
5. Apply the bridge inequality; accept only below 2 sqrt(epsilon).
6. Require the independent continuous steady certificate below 2 epsilon.

The parameter radii and positive definiteness are exact sufficient conditions.
Failure of this bound is unresolved, not evidence that the ROM violates the target.
The theorem currently certifies local cells, not the original entire HTC box.
The prototype is dense and not an extraction-cost optimization; reducing its
spectral setup cost is separate future work. No publication priority is claimed.

## Updated application: pre-SVD certification

The user subsequently chose the uncompressed snapshot span as the certification
object, with thresholds epsilon and sqrt(epsilon). The theorem above applies
unchanged to that fixed QR basis; the factor two is not used in acceptance.
See PRE_SVD_ALGORITHM.md and records/PRE_SVD_CERTIFICATION_20261007.md. The earlier
final-SVD scope is retained as the historical application, not the current policy.
