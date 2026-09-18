# Original 1 mm reproduction-mesh check

Registered before inspecting screen results.

A seventh job uses the original Case1 reproduction mesh (1 mm in x/y/z), its
four original sources and seed20260805. All FIVE methods, all THREE tolerances
and both extraction repeats remain; there is no post-result winning-method
selection. It checks all four HTC corners, the nominal case and two new random
HTCs with the same three profiles. The independent validation stream is still
seed+700000. This is the exact native geometry/mesh/configuration used by the
stock reproduction, not the separate imported commercial ROM discretization.

For N>50000, reference systems use warm-started AMG-CG at rtol1e-10 instead of
memory-intensive sparse LU. Every such RHS is independently residual-checked
against the original matrix at <=1e-9. This backend is used ONLY for FOM truth;
all ROMs still use the same dense BDF1 and stock-backend compatibility checks.
The coarse jobs retain sparse LU and their unchanged data/cases/budgets.
Three new reference-backend tests were red before implementation. No extractor,
source shape, h mapping, tolerance or output metric changed in this extension.

For the fine mesh, the artifact records basis shapes and SHA256 fingerprints
instead of duplicating fifteen large full bases. Reduced matrices, all native
operators, construction seeds and source are retained for regeneration. Coarse
artifacts still retain every full basis. Basis storage metrics always count the
actual in-memory full basis, not the smaller artifact representation.
