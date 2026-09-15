# MetaHotspot ROM experiment baseline — 2026-09-16

This directory documents the current experiment baseline only. Generated numerical result data is **not committed to the repository**. CSV/JSON tables, basis arrays, VTU files, and other generated outputs must stay under ignored `results/` directories or be retained as GitHub Actions artifacts.

The extraction implementation remains the project-standard `metahotspot.macromodel.utils.build_parametric_basis()` path with its existing closing SVD. Full-order thermal solves use AMG-preconditioned CG.

## Whole-PoP baseline configuration

The whole-PoP surrogate has 562,176 cells/DoFs, two independent die heat sources, six external surfaces, and three independent HTC parameters: the four lateral faces share one HTC, while bottom and top are independent. The MetaHotspot baseline uses extraction tolerance `1e-3` and 20 residual-probe rounds. The tighter tolerance is a project baseline choice; the 2023 THERMINIC paper itself reports `1e-2`.

Reproduction command:

```bash
python playground/connecting_mor_models/experiments/pop.py \
  --out playground/connecting_mor_models/results/pop_baseline \
  --nxy 89 --nz 53 --probe-rounds 20 --tolerance 0.001 \
  --cases paper_fig2 paper_fig4 range_low range_high paper_fig3_extrapolation \
  --validation-shifts 0 0.1 10
```

The validated reference run produced 105 exact response snapshots, 23 closing-SVD modes plus the uniform null mode, final ROM order 24, maximum accepted residual `9.576780432562378e-4`, tested in-range full-trace steady maximum global infinity error 0.5334%, and independent source/shift transfer maximum 0.5131%.

The fixed paper contour degrees remain inadequate on this surrogate: boundary-trace projection errors are about 24.23% on each side face, 68.96% on the bottom, and 28.55% on the top; the corresponding contour-system maximum in-range steady and transfer errors are 23.56% and 24.38%. Contour compression therefore remains a separate follow-up problem from the 24-state interior ROM.

Reproducibility provenance for the surviving CI branch: `work/connecting-roms-ci`, successful GitHub Actions run `34996751057`, artifact `10407883601`, Ubuntu 24.04 / Python 3.12. The artifact contains the generated result tables; those files are intentionally not versioned in Git.

## Reusable simple EROM baseline

The simple EROM configuration remains unchanged: one copper-cube ROM is extracted once and reused for all attachment cases, with one physical internal source plus one constant `z-` interface heat-flux training direction. Extraction tolerance is `1e-3`; the validated final ROM order is 11, matching the frozen FloTHERM EROM order 11.

Generated simple-EROM comparison tables are likewise experiment outputs and must not be committed. Keep them local under ignored output directories or publish them as CI artifacts when reproducibility requires retention.
