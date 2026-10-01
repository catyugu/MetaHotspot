# Connecting MOR models

Reproducible experiments for connecting embeddable BCI thermal ROMs. Generated
outputs under `results/` are ignored; experiment code and curated comparison
data are versioned.

## Files

```text
experiments/
  mor_common.py             conservative FVM coupling and standard contour-element utilities
  pop.py                    two-source PoP BCI/contour comparison
  simple_erom.py            MetaHotspot vs FloTHERM simple-EROM comparison
  test_contour_elements.py  independent quadrature checks for contour elements
baseline/
  flotherm_erom/            curated FloTHERM EROM reference data
  metahotspot/              curated results from the current reproducible rerun
results/                     generated locally; ignored
```

## PoP boundary-representation comparison

```bash
python experiments/test_contour_elements.py
python experiments/pop.py --method both --cases nominal strong_board_cooling lowk_board
```

The surrogate PoP contains two independent die heat sources in total: one in
the bottom package and one in the top package. The reduced bottom package is
extracted directly with the repository Extended-FANTASTIC implementation in
`python/metahotspot/macromodel/utils.py`; its two connectable faces are ordinary
affine BCI Robin groups during extraction.

The comparison deliberately keeps that *same* BCI basis and changes only the
exported connection-surface representation:

1. `affine_full_trace`: use the complete modal boundary trace on every FVM face.
2. `affine_contour`: represent the same trace with the contour elements of
   Codecasa et al., *Boundary Condition Independent Compact Thermal Models
   Enhanced by Contour Elements*, THERMINIC 2023.

The contour implementation follows the paper's Eqs. (13)-(17), rather than
using contour modes as extra heat-source snapshots. Each rectangular surface
uses products of continuous one-dimensional Legendre polynomials with total
degree `p`, analytically normalized in the physical-area inner product. The
FVM implementation evaluates the integrals in Eq. (15) exactly over every
boundary face and evaluates Eq. (16) by exact rectangle averages on the common
patches. The symmetric coupling uses the same map in transpose for heat flux,
which is the discrete Eq. (17).

For the two connection surfaces the default orders follow the 2023 PoP example:
`p_top = 10` (66 coefficients) and `p_bottom = 4` (15 coefficients). The
interior ROM order is therefore unchanged by contour elements; only the
boundary representation is compressed.

Non-conforming interfaces are integrated on conservative common patches. The
massless patch temperatures are analytically condensed, preserving a symmetric
coupling and heat-flux balance to roundoff.

## FloTHERM simple-EROM comparison

```bash
python experiments/simple_erom.py \
  --cases baseline_copper bottom_htc_strong external_source_200w all_stress layered_extreme_source
```

This reuses `playground/simple_erom_case1/` and compares the current
MetaHotspot BCI extraction against the frozen FloTHERM EROM stress baseline in
`baseline/flotherm_erom/simple_case1_stress_baseline.csv`. Detailed case
configuration notes are kept in the Obsidian weekly report rather than in this
repository.

The committed rerun tables in `baseline/metahotspot/` were produced by GitHub
Actions from the validated implementation; their README records the exact run
and extraction settings.
