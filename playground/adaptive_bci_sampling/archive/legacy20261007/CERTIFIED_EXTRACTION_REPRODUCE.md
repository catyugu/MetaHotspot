> 历史归档：保留当时的目标、结果与局限；“最新”等措辞仅指当时状态。当前验收和研究入口见 [当前 README](../../README.md)。

# Reproduce end-to-end extraction and comparison

Repository branch: `agent/experiment20261007`. All code and organized results are
tracked; generated data/operator files are not. Run from repository root with
Python 3.12, numpy, scipy and pyamg (recorded versions are in the report).
No native C API is required for these matrix reconstruction benchmarks.

## Sequential comparisons

Run these commands sequentially, without another benchmark process, to avoid
contaminating timing. Use `set -e` in a script so a failed run stops the sequence.
Cloud scheduling and BLAS versions still affect wall-clock measurements.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python
python playground/adaptive_bci_sampling/archive/legacy20261007/benchmark_certified_extraction.py \
  --model case1-local --matched-direct \
  --output playground/adaptive_bci_sampling/results/extraction_benchmark_case1_local_20261008.json
python playground/adaptive_bci_sampling/archive/legacy20261007/benchmark_certified_extraction.py \
  --model chain-local --matched-direct \
  --output playground/adaptive_bci_sampling/results/extraction_benchmark_chain_local_20261008.json
python playground/adaptive_bci_sampling/archive/legacy20261007/benchmark_certified_extraction.py \
  --model case1-mesh8-local --seeds 20260805 --repeats 1 --matched-direct \
  --output playground/adaptive_bci_sampling/results/extraction_benchmark_case1_mesh8_local_20261008.json
python playground/adaptive_bci_sampling/archive/legacy20261007/benchmark_certified_extraction.py \
  --model case1-full --seeds 20260805 --repeats 1 --max-cells 8 \
  --output playground/adaptive_bci_sampling/results/extraction_benchmark_case1_full_20261008.json
# Organized comparison is preserved in records/EXTRACTION_COMPARISON_20261008.md; the document-rewriting summarizer is retired.
```

Default epsilon is .001, new repeats are 3, and random seeds are 20260805,
20260806 and 20261007. The independent validation uses parameter seed 20261008.
Local domains are explicitly constructed in `model()` and printed in the report;
full Case1 uses the untouched original affine range. All algorithms receive the
same matrices and domain within each benchmark. Mesh 8 means n=1200 and mesh 10
means n=364 for this reconstruction.

Random default extraction stays unchanged, including AMG-CG and stock SVD.
`--matched-direct` adds an otherwise identical random extractor using the new
method's sparse-direct thermal snapshot solver. When the default raw space does
not meet the target, a preset internal tolerance search [.0001,.00005,.00001]
(at epsilon=.001) generates the same-target random comparison. It uses the same
raw-space certificate and constrained SVD guard as the new method. The report
separates passing-run costs from failed/default plus tuning costs. Full-range
geometry failure is not repaired merely by tightening random residual tolerance.

The script exports `*_new_operators.npz`. The file contains only reduced operators,
V, and metadata; it is reloadable without original matrices. A failed full-range
run also produces a diagnostic archive whose `final_certified` flag is false.
Do not use that archive as a model satisfying the requested tolerance.

## Direct API

```python
# PYTHONPATH=python:playground/adaptive_bci_sampling
from certified_extraction import extract_certified_rom, ExtractedROM

rom = extract_certified_rom(K0, C, G, H, ranges, epsilon=1e-3)
assert rom.report['raw_certified'] and rom.report['final_certified']
rom.save('playground/adaptive_bci_sampling/results/operators.npz')
loaded = ExtractedROM.load('playground/adaptive_bci_sampling/results/operators.npz')
x = loaded.steady(h, u)
trajectory = loaded.step(h, u, times)
```

The algorithm description/proofs are in CERTIFIED_EXTRACTION_ALGORITHM.md;
results are in records/EXTRACTION_COMPARISON_20261008.md. Existing pre-SVD,
final-SVD and Poisson–Loewner archives remain intact. New optional cache/witness
arguments on research audit functions preserve the old default return format.
Production code and its native binary/file formats are unchanged.
