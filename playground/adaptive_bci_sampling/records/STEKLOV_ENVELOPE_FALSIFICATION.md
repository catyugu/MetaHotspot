# Whole-boundary Steklov envelope: falsification experiment

Base revision: `77ff6dbece655cd6579f0995492da9363eaee925`.

The proposed stop rule was to include the dominant eigenvectors of
`H_delta phi = mu (K + s C + sum p_low H_i) phi` until
`mu[m+1] * kappa_s <= 1e-3`, where `kappa_s` is the largest generalized
eigenvalue of the four-port lower/upper corner transfer matrices. This is
only a *real-shift resolvent* target: no derivation from it to the two
continuous-time FANTASTIC error guarantees has been established.

Run `probe_steklov_envelope.py` from the repository root with the project's
existing Python environment, `PYTHONPATH=python`, a compatible built C API
(`MHS_CAPI_PATH`), and single-threaded BLAS. For example:

```sh
python playground/adaptive_bci_sampling/probe_steklov_envelope.py 5 --full-dc-spectrum --output steklov_5.json
python playground/adaptive_bci_sampling/probe_steklov_envelope.py 2.5 --full-dc-spectrum --output steklov_2_5.json
python playground/adaptive_bci_sampling/probe_steklov_envelope.py 1 --steady-only --count 30
```

The operator is assembled from the original Case-1 model with the same
`h_ranges()` and shared `1e-3` elliptic shift plan as the deterministic
baseline. All four sources are solved together at each corner. The symmetric
boundary-only eigenproblem is
`diag(sqrt(H_delta)) [A_low^-1]_{boundary,boundary} diag(sqrt(H_delta))`.
ARPACK finds 31 dominant eigenpairs (the 31st is the tail after 30 retained
modes); the full DC eigenvalue range at 5 and 2.5 mm is checked separately
with a dense symmetric eigensolve of the boundary Gram matrix. Numbers below
are ordinary double-precision diagnostics, **not** interval enclosures.

| Mesh | Full DOFs | Boundary DOFs | DC kappa | DC mu_31 * kappa | Smallest DC mu | Modes required by the 1e-3 rule at DC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 mm | 1,320 | 280 | 670.15 | 822.96 | 0.18412 | 280 / 280 |
| 2.5 mm | 9,072 | 1,072 | 664.07 | 1,924.14 | 0.031952 | 1,072 / 1,072 |
| 1 mm | 122,400 | 6,520 | 673.3 | 5,413.83 | not computed | more than 30 |

The 1 mm run was stopped after its DC result (56.8 s for that shift); its
three positive shifts and complete spectrum were deliberately not computed.
At DC, even the **last** boundary eigenvector has `mu * kappa` approximately
123 at 5 mm and 21.2 at 2.5 mm. Hence the proposed rule cannot terminate
before the entire boundary space is included. This is a decisive negative
result for this *global, source-agnostic spectral envelope*.

| Mesh | Shift s | kappa | Tail after 30 modes | Inverse RHS for 31 eigenpairs |
| --- | ---: | ---: | ---: | ---: |
| 5 mm | 0 | 670.1 | 822.96 | 80 |
| 5 mm | 0.0005245 | 82.17 | 97.49 | 80 |
| 5 mm | 0.1321 | 5.655 | 0.8426 | 144 |
| 5 mm | 33.25 | 1.058 | 0.0007082 | 176 |
| 2.5 mm | 0 | 664.1 | 1,924.14 | 96 |
| 2.5 mm | 0.0005538 | 74.09 | 208.59 | 96 |
| 2.5 mm | 0.1370 | 4.823 | 2.2877 | 232 |
| 2.5 mm | 107.77 | 1.022 | 0.01778 | 96 |

The eigenprobe alone took 480 and 520 applications of a sparse inverse
respectively over the four displayed shifts, in addition to eight source
right-hand sides per shift. These are repeated right-hand sides of two
factorizations per shift; they are **not** 480 or 520 distinct factorizations.
The repository's stock extraction previously used 116–118 and 126–127
source right-hand sides at 5 and 2.5 mm respectively. Computing the *full*
DC spectrum cost an additional 280 / 1,072 inverse right-hand sides in this
probe. A dense full-spectrum solve is only a falsification diagnostic, not
a proposed extraction step.

To distinguish an inadequate ROM from a loose envelope, we also projected
the four lower-corner source snapshots plus the leading boundary modes and
measured the largest same-parameter relative **real-shift transfer** error at
the four physical corners:

| Mesh | 0 modes | 3 modes | 8 modes | 30 modes |
| --- | ---: | ---: | ---: | ---: |
| 5 mm | 0.9043 | 0.4928 | 0.01874 | 4.21e-5 |
| 2.5 mm | 0.9478 | 0.6752 | 0.03655 | 2.44e-5 |

These finite-corner errors neither certify the full box nor imply the
Hankel/impulse-energy guarantees. They show why adding a few globally dominant
boundary modes can *look* successful on test parameters while the proposed
uniform, source-agnostic tail remains unusably conservative.

## Proof caveat

The original short proof asserts that operator monotonicity of
`f(S)=S(I+S)^-1` permits truncating in the eigenbasis of `S_delta` with
error exactly `sqrt(mu[m+1])`. Since `S` need not commute with `S_delta`,
that assertion does not follow from Loewner monotonicity alone. One safe
elementary estimate in whitened coordinates is

```
||P_tail f(S)||^2 <= ||P_tail S P_tail|| <= mu[m+1],
energy error^2 <= (1 + mu[m+1]) * mu[m+1] * ||A_low^-1/2 g||^2.
```

It uses `f(S)^2 <= S`, `S <= S_delta`, and Galerkin best approximation.
This valid but slightly weaker bound is also decisively too large in the
measured regime. The sharper constant `mu[m+1]`, and any bridge from real
matching shifts to both continuous-time error targets, must be proved
independently before being advertised as guarantees.

**Decision:** do not integrate the global Steklov-envelope stop rule into
the production extraction. A source-aware method would be a separate
hypothesis requiring its own uniform proof and cost comparison; the present
probe makes no claim about that possibility.
