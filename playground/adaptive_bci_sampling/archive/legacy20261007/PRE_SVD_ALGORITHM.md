> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# Current algorithm: certify before tolerance-based SVD compression

This supersedes the certificate-object and tolerance choices in the earlier
final-SVD experiment. It does not change the relative-form parameter theorem.

## Object and targets

Let S contain all extracted shifted response snapshots. Include the normalized
constant-temperature direction. Construct V0 as an orthonormal basis for this
uncompressed span. The implementation uses pivoted QR and removes numerical
linear dependencies at a machine-precision threshold independent of epsilon.
There is no epsilon-based rank selection. For the three recorded cases QR keeps
all snapshot directions plus the constant direction.

Certify the Galerkin model on V0 against the original full system:

- steady, all inputs, continuous HTC cell, relative K(h)-energy <= epsilon;
- step, all inputs, continuous HTC cell, every t>0, relative C-energy <= sqrt(epsilon).

SVD compression is a subsequent stage. These pre-SVD acceptance tests never use
2 epsilon or 2 sqrt(epsilon). The stored certified basis is explicitly named
`raw_basis`; the report has `post_svd_certified=false`. A snapshot singular-value
cutoff alone is not a proof of a subsequent response error budget.

## Algorithm

1. Select rational shifts and collect affine-parametric response snapshots using
   the existing extraction procedure. Treat extraction as candidate generation.
2. Normalize snapshots, append the constant direction, and orthogonalize by QR.
   Retain the full numerically independent snapshot span V0.
3. On each candidate HTC cell, apply the Bernstein steady residual certificate
   with V=V0. Require its bound to be <= epsilon.
4. At the cell arithmetic center, obtain an all-time full-field step bound ec
   for V0. The current prototype uses the analytic spectral/time-interval audit;
   a compatible rational-interpolation theorem can supply ec instead.
5. Compute full and reduced relative-form radii rho_F and rho_R from affine
   parameter corners. Require both <1. Compute the analytic short/long-time
   parameter variation envelopes d_F and d_R as in AFFINE_STEP_BRIDGE_PROOF.md.
6. Require d_F<1 and

       [ec + d_F + (1+ec)d_R] / (1-d_F) <= sqrt(epsilon).

7. Accept only when both steady and step tests pass. Otherwise report the cell
   as unresolved, enrich snapshots or refine the parameter cover as appropriate.
   Do not discard raw directions to make a certificate pass.
8. Report the accepted continuous cell together with its uncompressed basis,
   thresholds, bounds and costs. A full-domain declaration requires every cell
   in a covering partition to pass.

The center audit threshold is set to 0.1 sqrt(epsilon) in the experiment to leave
room for parameter variations. This is an internal sufficient-condition choice;
the final acceptance threshold remains exactly sqrt(epsilon).

## Proof and implementation

All proofs in AFFINE_STEP_BRIDGE_PROOF.md hold for any fixed independent basis,
so replacing its final-SVD basis with V0 leaves the proof unchanged. In particular,
C-normalized reduced coordinates still give an isometric full-field lift, and
all-input error metrics remain invariant under input coordinate changes.

For PSD Robin terms H_j, the existing steady Bernstein proof also applies
unchanged with the smaller threshold epsilon. No new factor-two argument is used.

The prototype is `pre_svd_audit.py`. It intercepts snapshots before the existing
extractor's compression call, constructs its own QR basis, and never uses the
extractor's compressed basis for certification. The legacy extractor still
computes a compressed comparator; its order is recorded only for context.
Production extraction can return S directly and defer that SVD until after the
raw-space certificate. No production macromodel API is changed in this prototype.

As before, analytic scope is continuous parameters and all times, but ordinary
floating-point operations are not outward-rounded certificates. Dense full
spectral setup remains a prototype limitation; certifying an uncompressed space
alone does not remove that cost.
