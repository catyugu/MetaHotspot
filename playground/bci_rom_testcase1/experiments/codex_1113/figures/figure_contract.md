# Figure contract

Core conclusion: A parameter-domain spectral interval can improve the asymmetric
model's output accuracy, but must be judged against response solves and final
compression error.
Archetype: quantitative grid. Exports: vector PDF, editable SVG, preview PNG.

1. Paired comparison: maximum steady junction error over all 17 evaluation
   points, and response solve counts, with individual seeds and mean +/- sample
   standard deviation (n=3). No inference about unrelated packages.
2. Compression ablation: accuracy against final state dimension for thresholds
   1e-3..1e-6 using identical first-seed snapshots. Include fixed-order energy POD
   if actually executed, as an explicitly exploratory result. This isolates
   compression from additional response solves; it is not independent model
   selection on a new test dataset.
3. Transient validation: per-source junction error against the same full-order
   BDF1 trajectory, nominal HTC, dt=20 s, duration=1000 s. Overlay baseline,
   global and FloTHERM ROM errors with common axes and units. This does not
   validate sub-20-second dynamics or claim continuous-time error bounds.

Reviewer risks: different shift plans consume different random draws; comparing
errors to FloTHERM alone confounds different full discretizations; the weakest
HTC corner has a very large linear-model rise, so relative and absolute errors
must both be available in the report. Avoid an absolute-temperature bar chart
that hides millikelvin differences on a 300 K baseline.
