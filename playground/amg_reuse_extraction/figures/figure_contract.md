# Figure contract

The figures will test whether saved AMG setup time survives full extraction and whether error remains comparable.

1. Quantitative comparison: paired end-to-end extraction seconds for Fresh, Port cache, and Shift cache. Plot all three individual seeds and mean/sample-standard-deviation bars. This is the primary performance evidence; do not label setup speedup as total speedup.
2. Quantitative decomposition: stacked mean extraction time, partitioned into spectrum, setup, PCG, probes/basis/other, snapshot SVD, and final projection. Setup count and iteration count are supplied in the report. This tests attribution and exposes remaining bottlenecks.
3. Accuracy and response cost: held-out per-point steady field errors and transient field/junction errors for all methods, together with per-frequency setup/CG time and iterations. Use data-supported panels only; the primary validation view will show each physical HTC point and all three seeds, with errors normalized by steady rise and absolute kelvin given in the report.

Export PDF, editable SVG, and PNG previews. Use embedded TrueType fonts, eight-point minimum labels, a colorblind-safe palette, and a caption defining three paired seeds. Inspect the rendered output before delivery. Reviewer risks: only one geometry; adaptive full-response trajectories can diverge under inexact solves; trial order and three repeats do not eliminate host scheduling noise; validation error is relative to the same discrete FOM, not an external experiment.
