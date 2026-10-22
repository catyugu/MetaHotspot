# Connecting MOR models

Experiments for whole-PoP BCI reduction, contour-element boundary
representation, and embeddable thermal ROM connections. Run the commands below
from this directory with NumPy, SciPy and PyAMG installed, and the repository's
`python/` package on `PYTHONPATH`. Native model examples also require the built
C API library, selected with `MHS_CAPI_PATH`.

## Files

```text
experiments/
  mor_common.py                    FVM operators, conservative coupling and contour elements
  pop.py                           whole-PoP two-source BCI/contour comparison
  simple_erom.py                   reusable interface-enriched ROM vs FloTHERM
  test_contour_elements.py         independent contour quadrature checks
  transfer_boundary_basis.py       matrix-based boundary transfer directions
  transfer_boundary_enrichment.py  manual extra-boundary-mode sweep
baseline/
  flotherm_erom/                   curated FloTHERM EROM reference data
  metahotspot/                     reference-run configuration and provenance
results/                            generated locally; ignored
```

Generated numerical results stay under ignored `results/` directories or in CI
artifacts; the repository contains experiment code, baseline notes and the
curated FloTHERM comparison CSV.

## Whole-PoP boundary-representation comparison

```bash
python experiments/test_contour_elements.py
python experiments/pop.py --cases paper_fig2 paper_fig4 range_low range_high
```

`pop.py` extracts one BCI ROM for the entire surrogate PoP, including two
independent die heat sources. Its six external surfaces use three independent
HTC parameters: one shared by the four lateral faces, one for the bottom, and
one for the top. The basis is built with
`metahotspot.macromodel.utils.build_parametric_basis`.

The script compares two boundary representations of the same interior basis:

1. `full_trace`: complete modal trace on every FVM boundary face.
2. `contour`: the trace projected onto the contour elements of Codecasa et al.,
   *Boundary Condition Independent Compact Thermal Models Enhanced by Contour
   Elements*, THERMINIC 2023.

Each rectangular surface uses products of continuous Legendre polynomials with
total degree `p`, normalized in the physical-area inner product. Integrals are
evaluated from face rectangles; contour traces are recovered as exact rectangle
averages. Degrees are 2 on each lateral face (6 coefficients each), 4 on the
bottom (15), and 10 on the top (66), for 105 boundary coefficients. The interior
ROM order is unchanged by this representation choice.

The default surrogate resolution is `--nxy 43 --nz 36`, with extraction tolerance
`1e-2` and three residual-probe rounds. Geometry and material values define a
surrogate, not an exact reconstruction of the paper's detailed model. Steady
field comparisons and independent source/shift transfer checks are written to
`results/pop/`; `--vtu` also writes the `paper_fig2` contour field when that case
is selected. Reference-run settings and reported limitations are in
[baseline/metahotspot/README.md](baseline/metahotspot/README.md).

Conservative non-conforming common-patch coupling is provided separately by
`mor_common.py` and `metahotspot.macromodel.embeddable`; `pop.py` compares whole-PoP
operators rather than connecting a reduced bottom package to a detailed top.

## FloTHERM simple-EROM comparison

```bash
python experiments/simple_erom.py \
  --cases baseline_copper bottom_htc_strong external_source_200w all_stress layered_extreme_source
```

This uses `playground/simple_erom_case1/` and compares one reusable
interface-enriched MetaHotspot cube ROM against the frozen FloTHERM EROM stress
baseline in `baseline/flotherm_erom/simple_case1_stress_baseline.csv`. The cube
ROM is extracted once using its internal source and interface heat-flux
training directions, then reused for every attachment case. Results are written
to `results/simple_erom/`.
