# Signed-challenge randomized discovery: preliminary gate (2026-10-11)

This is a **research option, not a production replacement**. The branch's pre-existing all-time pre-SVD, signed-input joint risk certificate remains the only acceptance authority. Activation is explicit:
```bash
python playground/adaptive_bci_sampling/random_task_extraction.py \
  --seed 20261031 --certificate-mode scalar --inverse-lower-kind bipartite \
  --sampling-policy signed-challenge --signed-exploration .25 \
  --max-rounds 240 --audit 2 \
  --output playground/adaptive_bci_sampling/results/signed_small.json
```
For a comparison, change `--sampling-policy` to `tournament` (other flags unchanged). The new proposal draws a log-uniform HTC, input, and preset real shift with probability .25. Otherwise it perturbs a previously rejected HTC (clipped in log parameter coordinates), chooses an input with a signed-residual Gram eigenvector weighting, and chooses a residual-maximizing local shift or failed interval's inverse time. No symmetry or common training space is assumed. All entries of the proposal law retain full support from the uniform component. The candidate is only a discovery proposal, never a certificate.

The signed residual direction solves `(R.T @ B^-1 @ R)u=lambda (G.T @ C^-1 @ G)u`, with `R` the actual port-specific physical residuals and `B` the proven MIC or graph lower operator. Source weights are proportional to `(u_j ||B^-1/2 r_j||)^2`, with a small positive floor; this is *not* a theorem saying a single input is the worst signed input. A full-rank input Gram and the inverse lower operator hypotheses are required. A reference all-time challenge is still evaluated from a **fresh independent HTC stream after freezing the combined SVD-before Galerkin basis**, retaining original epoch-wise `delta_k`. Previously rejected HTC are used only to propose training points.

## Independent exploratory prototype, not production numbers

A self-contained matrix-only Case1-like reconstruction with four deliberately asymmetrically reweighted source footprints was tested. These test results are **steady/resolvent risk only, not all-time transient**, and use a separate, offline MIC-whitened, full-input-Gram steady certificate (no `splu` on the FOM). The prototype is not identical to the branch's production and all-time workflow.

Parameters: 364 DOF, log-uniform effective HTC, epsilon=.001, rho=.01, delta=1e-6, each accepted basis tested with a new independent zero-failure HTC stream. Samples were not re-used across adaptive epochs. Methods share model, tolerances, certificate, and initial input responses. AMG was a custom x-y aggregation V-cycle with CG; the same sparse iterative solver was used for each method.

| Seed | Pure random (200-snapshot cap) | 4-proposal tournament | Certificate-directed mixture |
| --- | --- | --- | --- |
| 30 | no acceptance, 201 total RHS | accepted, 145 total RHS | accepted, 97 total RHS |
| 31 | no acceptance, 201 total RHS | accepted, 133 total RHS | accepted, 97 total RHS |
| 32 | no acceptance, 201 total RHS | accepted, 157 total RHS | accepted, 121 total RHS |

Here total RHS includes one extra positive-witness solve. This is a preliminary reduction in **full solves**, not a fair proof of **total wall-clock acceleration**: the directed sampler has extra certificate/feedback evaluations and the three seeds are too few for a population statement. Independent true small-grid steady errors at two untrained HTC did not exceed the corresponding MIC Gram bounds in these tests; that numerical check is not a formal floating-point proof.

A fixed-budget large-grid control (different experimental environment and separate certificate) is deliberately not declared accepted:

| DOF / budget | Directed MIC signed Gram q90 | Pure random MIC signed Gram q90 | Both passed? |
| --- | ---: | ---: | --- |
| 47,085 / 61 total RHS | 0.242 | 0.956 | No |
| 122,400 / 37 total RHS | 0.9811 | 0.9818 | No |

The earlier cruder source-separated surrogate showed a stronger difference at 47k; that surrogate did **not** faithfully measure the final signed-input certificate, so it must not be used as the headline result. No all-time joint-risk acceptance or official Case1 reproduction has been obtained for the newly integrated option at formal grid size. The standalone Case1-like input masks and fallback aggregation AMG are exploratory, not native assembly verification.

## Mathematical boundary

For any fixed `h` and `s>=0`, let `x_j=(A(h)+sC)^-1 g_j` and `w_j` be locally approximated responses in the separate input trial spaces. Let `E=[x_j-w_j]`, `W=[w_j]`, `R=G-(A+sC)W`. If a proven `0<B<=A+sC` exists, then
`E.T (A+sC) E = R.T (A+sC)^-1 R <= R.T B^-1 R=:D_R`.
This controls arbitrary signed `u`, not just source-wise positive inputs. But it does **not** immediately bound the error of the *combined* Galerkin projection unless the Galerkin optimality and denominator comparison are carried through. In the numerical prototype a stronger offline signed Gram estimator was used; it was independently spot checked, not outward-rounded or used as a substitute for full-time certification. Do not interpret any finite collection of `s` values as an all-time step theorem.

The integration has not been run end-to-end in this runtime: PyAMG/native project libraries are unavailable locally. Its syntax and behavior require a subsequent branch-environment smoke test before any publication claim. Existing default behavior remains `tournament`.

## Required next acceptance gate

1. Run new option against same-guarantee `--sampling-policy tournament` and `--tournament 1` at 364, 47,085 and 122,400 DOF, 5+ paired seeds, holding the all-time MIC certificate and risk-test settings fixed.
2. Report SVD-before rank, accepted joint risk, transient audit at independent times, FOM solves, all auxiliary triangular applications, certificate queries, peak memory and isolated wall/CPU time.
3. Verify native matrix assembly and floating-point enclosures; ordinary floating point plus a mathematical exact-arithmetic proof is not an end-to-end machine-certified implementation.
4. Re-evaluate novelty against randomized greedy reduced-basis construction, weighted residual estimators, adaptive importance sampling, and sequential testing literature. Full-support mixtures and the MIC factorization alone are established techniques.

**Status: promising small-grid cost signal; official large-grid and all-time superiority not demonstrated.**