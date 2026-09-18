# Strict code-path control after the initial screen

The initial screen's full_block implementation reuses the assembled sparse
matrix at a rejected sample, while its inherited cached scalar comparator
reassembles that matrix even though it reuses AMG. This small difference must
not be silently credited as a block-selection benefit.

This follow-up runs the SAME block extractor with block_cap=1 and block_cap=4,
plus the unchanged legacy scalar comparator. All other code, source/HTC data,
epsilon=1e-4,16 ports, mesh2.5 mm, seeds20261101/02/03, planning, compression,
BCI projection and two forward/reverse timing repeats are identical. Both
scalar constructions must produce numerically identical subspaces. Their
basis hashes allow matching to the fully validated primary screen.

No acceptance target, candidate rule, tolerance, source, held-out case, or
production file is changed. This is a post-screen ATTRIBUTION check, not an
independent accuracy validation or a newly optimized method. No timing is
combined across runner machines: only paired ratios within each follow-up
job are interpreted. The first primary screen remains untouched and retained.

CI downloads the exact native matrix artifact from run35382733713, records
source/environment, runs the existing19 new unit tests, and checks Git scope.
Only this protocol, its runner and a separate workflow are added. The primary
workflow is not retriggered by this folder.
