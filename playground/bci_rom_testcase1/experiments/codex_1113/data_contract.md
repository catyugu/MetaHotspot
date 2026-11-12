# Data contract

This experiment uses the user-supplied model and FloTHERM export from
`catyugu/MetaHotspot`, dev commit `32eb633` (2026-11-13).
The geometry is `../../model_case1.py`, its FloTHERM specification is
`../../case1.ecxml`, and the reference reduced operators are in `../../MATRICES`.
The repository is MIT licensed; the exported commercial ROM is treated as
user-provided comparison data, with no claim about Siemens redistribution rights.

The default new-model mesh is 0.5 mm. Its actual active cell count is recorded
at runtime; it is not the old symmetric 122,400-cell model. We preserve every
material, heat source, and boundary group. HTC vectors passed to the reference
are physical W/(m² K); the extraction samples the adapter's effective ranges.
Source columns integrate to one watt, and all junction probes use the same
source-weighted average. Temperatures are rises above 308.15 K.

Training: original random parameter draws and MPMM positive real shifts, seeds
20260805, 20260806, 20260807. Evaluation: nominal HTC (100,1000,30), eight
physical HTC corners, and independent log-uniform points with seed 1113.
No evaluation response is added to the trial space. Compression ablations reuse
training snapshots, not evaluation solutions. FloTHERM is an independent ROM,
not a full-field truth model. A difference from its output includes discretization
and reduction differences, and cannot alone measure extraction accuracy.

Numbers are measured executions of a numerical thermal experiment, not physical
measurements. Raw outputs and arrays are generated into ignored `results/`;
only source, protocol, and a compact report are intended for git.
