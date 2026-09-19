## Additional accuracy-qualified stock control (after the first run)

The initial run 35428739226 used .01/.001/.0001 as planned. In four-port seed
20261203 the tightest stock model has 1.0324% maximum physical field error,
while the rank44 harmonic construction has 0.9886%. Thus the original grid
contains no all-contract accuracy-qualified speed comparator in that case.
Do not compare to a failing control or silently round both to one percent.

Add ONLY stock epsilon=.00001, using the unchanged library and the same boundary
projection epsilon. Rerun all six jobs in the same environment so timing is
within-run, not compared across runners. The old stock controls remain. Parent
selection stays epsilon=.0001; no candidate rank, optimization, source, parameter,
validation sample or metric changes. New helper configuration is unit tested;
this is a baseline sufficiency repair, not holdout tuning of the proposed methods.
All initial data and commits are retained. Output-only qualification is additionally
reported using junction error alone; a full-field requirement must not be used
to force an unnecessarily large stock model in a junction-only comparison.
