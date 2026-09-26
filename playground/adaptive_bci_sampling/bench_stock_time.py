"""Stock extractor wall clock at 1 mm under its own AMG-CG settings."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402

mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None

model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
source = np.asarray(model.source_shape, dtype=np.float64)
terms = [term.tocsc() for term in model.boundary_terms]
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
print(f"mesh={mesh} n={model.core.K.shape[0]} ports={source.shape[1]}", flush=True)

record = {}
for seed in (20260805, 7):
    clock = time.perf_counter()
    basis, stats = build_parametric_basis(
        model.core, source, terms, ranges,
        tolerance=1e-3, max_order=4096, probe_rounds=10, seed=seed,
    )
    seconds = time.perf_counter() - clock
    record[f"stock_{seed}"] = {
        "seconds": seconds,
        "full_rhs_solves": int(stats["pre_svd_order"]),
        "validation_count": int(stats["validation_count"]),
        "basis_order": int(np.asarray(basis).shape[1]),
    }
    print(f"stock seed={seed}: {seconds:.1f}s  solves={stats['pre_svd_order']} "
          f"probes={stats['validation_count']} order={np.asarray(basis).shape[1]}", flush=True)

if out:
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("wrote", out, flush=True)
