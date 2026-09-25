# Next steps: choosing the affine parameter params in a BCI extraction

This is the original handoff, retained for the exact Woodbury and ray-pole
checks and their pitfalls. For the current `1e-3` sampling comparison, see
[`TANGENT_CORNER_SAMPLING.md`](TANGENT_CORNER_SAMPLING.md). Generated JSON
outputs are deliberately omitted from version control; the reports contain
their numerical summaries and reproduction commands.

Report: `E:\code\ObsidianNotepad\research\MetaHotspot Weekly Reports\weekly_report_0923.md`,
section "实验二、改善仿射参数的采样方法".

## The problem

The stock extractor (`metahotspot.macromodel.utils.build_parametric_basis`)
proposes the next affine boundary parameter `h` by drawing it log-uniformly at
random, and retires a shift after `probe_rounds` consecutive draws pass the
residual test. Both halves are Monte-Carlo statements: the snapshot set depends
on a seed, and the stopping rule says nothing about the worst point of the
parameter box.

Question to answer: **is there an a priori, closed-form way to choose the
parameter samples, with a guarantee?**

Model: `playground/bci_rom_testcase1/model_case1.py` (3-layer stack, 4 dies, two
ambient groups, 4 heat sources). Metric: worst junction error relative to the
junction rise at the same parameter. Do NOT use the field norm -- 98.8% of it is
air and it says nothing about the dies.

## What is verified (do not redo)

Reproduce with:

```text
python playground/adaptive_bci_sampling/verify.py 2.5
```

```text
mesh=2.5 mm  cells=9072  groups=2  m_b=1072  reference solves=1076

worst relative field error                5.86e-07
worst relative junction error             5.10e-08
cond(M) at the hardest corner             2.46e+04
worst cond(M) over probes                 2.92e+13   (group 1, 1-1e-10 of p_ref)
field error at that worst cond            9.50e-08
pole series vs full-order solve           6.84e-07
```

Meaning:

1. The parameter enters only through a diagonal perturbation of the boundary
   cells (`H_k = diag(a_k)`), so Woodbury gives the whole family exactly from
   `m_b + n_src` solves at ONE parameter. Nothing is fitted; the residuals above
   are solver accuracy.
2. The small system `M = Phi + diag(1/lambda)` is ill-conditioned near the
   reference point (2.92e13) but the field error stays at solver accuracy
   (9.50e-08). Ill-conditioning here is benign, and this was checked rather than
   assumed.
3. Along any ray `p(t) = p_ref + t (target - p_ref)` the whole parameter vector
   enters through the single scalar `t`, and the solution is a sum of simple
   poles at `t = 1/mu_j` with `mu_j` the positive eigenvalues of
   `Psi = B^{1/2} Phi B^{1/2}`, `B = diag(-b_c)`. The series sums back to the
   exact solution.

Files kept at the time of this handoff (the directory now also contains the
subsequent comparison experiments):

```text
rational_surrogate.py   BoundaryRationalModel: exact rational parametrization,
                        ray poles, pole weights
verify.py               the three checks above
```

## What is NOT settled

**At the time of this handoff, the sampling scheme itself.** Several attempts
had been made and none was finished.
What is known about the landscape:

## Traps (each of these cost real time)

1. **Counting on the junction vector is useless.** It has `n_src` rows, so its
   rank saturates at `n_src` whatever the tolerance. A "count = 4" result was
   exactly that artefact. Count on the field, or on the boundary vector `s`.
2. **A span built from the diagonal ray alone stalls.** It reaches about 2%
   junction error and stops, no matter how many modes it is given, because the
   corner directions are not in its span. All rays out of the reference point
   must be covered -- but see the next item.
3. **Unweighted Gram matrices inflate the count as the grid refines.** `sigma_1`
   grows like `sqrt(n_samples)`, so a fixed threshold raises the count purely
   from refining. Use quadrature weights summing to one. After weighting, counts
   at 15, 25 and 35 points per axis agreed exactly, which is the check that the
   count is a property of the family.
4. **The fast evaluator's sign convention is not the one you would guess.** The
   pole series is `x(t) = x_ref + G_b sum_j coefs_j phi_j(t)` with
   `phi_j(t) = t t_j / (t_j - t)`. The `+` was determined numerically against the
   exact solve; the Markov-formula route (`(I + t Psi)^{-1}`) and the
   Schur-complement route (`t B (I - t Psi)^{-1}`) differ by a sign that is easy
   to get wrong on paper. Verify numerically, do not derive and trust.
5. **The evaluator is the bottleneck, not the SVD.** The dense solve in
   `response` costs `O(m_b^3)` per parameter; at 1 mm `m_b = 3335`, which is
   hundreds of milliseconds per point. The pole series costs `O(N q)` per point
   and `q = 156` already gives 6.4e-09 at a 1e-4 target. For the junction alone,
   precompute `R = G^T G_b` and never touch the N-vector field at all.
6. **Do not use `sleep` to wait for a background run.** Use the process wait.
