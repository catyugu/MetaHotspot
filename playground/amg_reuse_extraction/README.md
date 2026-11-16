# Same-frequency AMG reuse during parametric ROM extraction

This experiment measures complete Extended-FANTASTIC extraction on the repository's original 146704-cell Case-1. All methods use common global spectral endpoints, independently generated at the start of each extraction. The production extractor's per-port response-space logic, tolerances, random probes, warm starts, and final projection are retained. `extractor.py` moves the frequency loop outside the port loop in all three methods. Each port keeps its own evolving basis and Galerkin projection.

- `fresh`: actual sampled matrix, new AMG for every full response.
- `port_cache`: lower-HTC reference AMG per (source port, frequency).
- `shift_cache`: lower-HTC reference AMG per frequency, shared across all ports.

The reference is `K_min + s*C`. A cached hierarchy always preconditions the actual sampled matrix; no matrix or RHS approximation is introduced. The cache has exactly one slot. A completed frequency is released before building the next hierarchy; the port-cache ablation also replaces the slot when switching ports. A cache is cold at the start of every extraction. Only frequencies needing a full response construct an AMG hierarchy.

## Reproduce

Use repository revision `deb981297d1353d9005e0a042657e1d8eef0e595`. The recorded measurements used the matching Linux CI wheel from workflow `43612616388`, artifact `14121435121`. Native C API assembly comes from that wheel; its Python `utils.py` was byte-compared against the checkout.

```bash
uv venv
uv pip install --python .venv/bin/python -r playground/amg_reuse_extraction/requirements.txt
# Install the matching wheel, or build this checkout using the documented native toolchain.
uv pip install --python .venv/bin/python /path/to/metahotspot.whl
.venv/bin/python playground/amg_reuse_extraction/run_all.py
```

Each complete extraction takes about half a minute to one minute on the measured AMD EPYC CPU with one BLAS/OpenMP thread. All nine runs execute sequentially, in an alternating paired order. `run_all.py` resumes completed JSON outputs; use an empty results directory for a cold complete rerun. Do not invoke `uv run` without `--no-sync` when using an installed wheel, because syncing the repository may trigger a native rebuild.

`config.yaml` records the measured settings; the runnable benchmark exposes method, seed, output path, and mesh size as command-line arguments. The other extraction settings are fixed in `benchmark.py` to match this protocol.

For individual runs:

```bash
.venv/bin/python playground/amg_reuse_extraction/benchmark.py --method shift_cache --seed 20260805
.venv/bin/python playground/amg_reuse_extraction/evaluate.py
.venv/bin/python playground/amg_reuse_extraction/summarize.py
```

Raw timing records and basis arrays are written under ignored `results/`; logs are ignored as well. The report, small source files, and figure source can be committed. Basis arrays are intermediate checkpoints and are omitted from the downloadable evidence package to keep it small. Rerunning extraction recreates them.

`benchmark.py` temporarily adapts `port_eigenvalue_bounds` and `spd_solve` in memory for the frequency-major extractor, restoring them when extraction finishes. The solver adapter reads the active extractor's `port`, `shift`, and `h_vec` local variables, so it is intentionally bound to the recorded source revision. It is an experiment harness, not a proposed production API. Final BCI projection is included in extraction time; geometry assembly, validation, and writing evidence files are reported separately.

Validation covers eight physical HTC corners, the nominal point, and four held-out random points, plus three 1000 s transients using the same BDF1 discretization for FOM and ROM. This measures reduction error against the discrete affine FOM. No claim about physical measurement accuracy or other thermal geometries is implied.

The earlier port-major experiment retained multiple frequency hierarchies. Its records are archived separately and excluded from the reported single-cache results.

## Measured result

On 146704-cell Case-1, three paired seeds with frequency-major scheduling and a single overwriteable AMG slot:

| Method | Complete extraction, mean ± sample SD | Response AMG setups | Final order |
|---|---:|---:|---:|
| Fresh | 48.16 ± 1.08 s | 143–147 | 40–41 |
| HTC reuse within port | 35.15 ± 1.62 s | 58–61 | 41 |
| HTC and port reuse | 26.63 ± 0.54 s | 17 | 41 |

Paired end-to-end speedup is 1.809 ± 0.037×, with 44.71% lower mean wall time. Peak process RSS is approximately 1.5 GiB in all methods. The two reuse methods produce byte-identical bases for each seed.

Over 13 independent steady points and three 1000 s transient cases, the shift-cache ROM's worst steady full-field/junction errors are 0.024794/0.000486 K; worst transient errors are 0.024778/0.000367 K. Fresh's corresponding full-field errors are 0.018983/0.012578 K. Worst normalized steady full-field errors are 0.1754% for Fresh and 0.2283% for reuse. Inexact solves can change the adaptive sampling path and SVD truncation; the third paired seed has order 40 for Fresh and 41 for reuse. These measurements apply to this benchmark and configuration.

Source code and protocol are versioned here. Raw per-response timing data, validation records and exported figures are retained as experiment evidence, outside git. Large basis checkpoints are excluded from both the commit and evidence package.
