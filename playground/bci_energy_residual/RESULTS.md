# Certified energy-residual experiment

Date: 2026-09-19

Base: `agent/work` at `0b4e5874d27571488d62568d12b7a68df7656aca`.

This experiment does not modify production code. It isolates Algorithm 1's
probe acceptance criterion in Extended FANTASTIC.

## Criterion

For the SPD matching-point system

[
A x=b,qquad A=K+sigma C+sum_j h_j H_j,
]

with Galerkin estimate (widehat x) and residual
(r=b-Awidehat x), the tested exact indicator is

[
ho_E = rac{r^T A^{-1}r}{b^T A^{-1}b}.
]

This is the squared relative (A)-energy error. The experiment follows the
explicit convention (ho_Eleqarepsilon), with
(arepsilon=10^{-3}).

Let (q=r^TA^{-1}r) and
(g=2b^Twidehat x-widehat x^TAwidehat x). Then exactly

[
b^TA^{-1}b=g+q,qquad
ho_Elearepsilon
iff
qlerac{arepsilon}{1-arepsilon}g.
]

Thus only one inverse quadratic form must be classified.

## Certificates

Two certificates were tested.

1. A mass-scaled CG interval using
   (Asucceq sigma C).
2. A tighter streaming Lanczos/Gauss-Radau interval using
   [
   P=sigma C+sum_j h_jH_j,qquad A=P+Ksucceq P.
   ]
   Hence (P^{-1/2}AP^{-1/2}succeq I), supplying an exact known lower
   spectral endpoint for left Gauss-Radau bounds.

The final implementation stores only the current two Lanczos vectors.
A rejected probe falls back to the ordinary ROM warm start.

## Exact-oracle check

On the 1320-cell Case1 model, for both seeds 20260805 and 20260931,
the Gauss-Radau extraction and an oracle that explicitly solves
(A^{-1}r) had exactly the same:

- number of full enrichments,
- pre-SVD snapshot count,
- final basis order.

The minimum principal cosine between the final subspaces was
(0.9999999999999988) or larger.

## Final CI scaling study

All rows use the repository's stock `build_parametric_basis` as the
baseline with identical Case1 model, matching points, HTC sampling,
(arepsilon=10^{-3}), ten probe rounds, and random seed.

| cells | seed | stock full solves | energy full solves | stock order | energy order | stock extraction | energy extraction | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,320 | 20260805 | 116 | 45 | 35 | 36 | 0.837 s | 0.702 s | 0.839 |
| 1,320 | 20260931 | 121 | 47 | 35 | 37 | 1.304 s | 1.164 s | 0.892 |
| 9,072 | 20260805 | 126 | 62 | 42 | 41 | 2.332 s | 2.074 s | 0.890 |
| 9,072 | 20260931 | 130 | 58 | 43 | 42 | 4.346 s | 4.311 s | 0.992 |
| 122,400 | 20260805 | 159 | 72 | 52 | 49 | 56.982 s | 62.923 s | 1.104 |

The exact energy criterion consistently removes roughly half to three fifths
of the expensive enrichments:

- 61.2% fewer full solves at 1,320 cells;
- 50.8% and 55.4% fewer at 9,072 cells;
- 54.7% fewer at 122,400 cells.

However the current unpreconditioned rigorous certificate does not scale
well enough. At 122,400 cells it used 22,308 Lanczos iterations over 679
validation probes (32.85 per probe); certification alone cost 35.39 s.
The saved full solves therefore did not compensate for certification and
total extraction was 10.4% slower.

## Holdout observations

Independent random matching-point holdouts had zero violations of
(ho_Ele10^{-3}) for both stock and energy-certified models.

For the energy-certified model, maximum observed holdout energy ratios were:

- (2.02	imes10^{-4}) at 1,320 cells, seed 20260805;
- (3.28	imes10^{-5}) at 1,320 cells, seed 20260931;
- (4.07	imes10^{-5}) and (9.60	imes10^{-5}) at 9,072 cells;
- (5.29	imes10^{-4}) at 122,400 cells.

These are sample-based checks, not a proof over the continuous
frequency/HTC domain or after every possible closing compression.

Pointwise steady full-field maximum error was generally worse for the
energy-certified extraction (for example 3.06% versus 1.45% on the
122,400-cell run). This is expected: the tested criterion controls the
energy metric, not the spatial infinity norm. The Euclidean residual is
therefore conservative with respect to the tested energy criterion but can
incidentally retain directions useful for pointwise error.

## Interpretation

The experiment supports two separate conclusions.

1. The current Euclidean residual causes substantial over-enrichment relative
   to the exact inverse-weighted energy criterion: the strict criterion needs
   only about 39-49% as many full solves on the tested Case1 meshes.
2. The first rigorous matrix-free certificate is not yet the final algorithm.
   Unpreconditioned certification becomes the bottleneck as the mesh is
   refined.

The next algorithmic problem is therefore narrower and better justified than
before: construct a **certified preconditioned inverse-quadratic-form
decision** whose spectral bounds remain rigorous for the affine diffusion
family. A heuristic AMG-preconditioned residual is not sufficient unless
spectral-equivalence constants are also certified.

## Validation

GitHub Actions run `35442044829` passed:

- native build,
- existing native regression suite,
- existing case suite,
- Python tests,
- exact-energy and Gauss-Radau unit tests,
- experiment scope/diff checks,
- all five numerical study jobs.

The experimental branch changes only files under
`playground/bci_energy_residual/` and the dedicated GitHub workflow.
`agent/work` remains unchanged.
