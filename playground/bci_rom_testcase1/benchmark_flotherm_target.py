#!/usr/bin/env python3
"""Benchmark BCI-ROM extraction on case1 at FloTHERM's 1 mm air-domain spec.

Mirrors `case1.ecxml`: solutionDomain 60x100x22 mm filled with ambient-air
background, ~132000 cells, tolerance 1e-3, affine h range [1, 1e4].  Reports the
total extraction wall-clock plus the (sequentially-measurable) Krylov enrichment
solves.  The per-port spectral bounds are parallelised inside
build_parametric_basis, so they are reported by difference (wall minus solves).

Python: must run in the numerical conda env with PYTHONPATH at the bindings.
    cd playground/bci_rom_testcase1
    PYTHONPATH='E:/code/cpp/MetaHotspot/python' \
      /e/env/miniconda3/envs/numerical/python.exe benchmark_flotherm_target.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_MACRO = Path(__file__).resolve().parent.parent / "macromodel"
if str(_MACRO) not in sys.path:
    sys.path.insert(0, str(_MACRO))

import utils  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402


def main() -> None:
    cfg = Case1Config(
        max_xy_cell_mm=1.0,
        max_z_cell_mm=1.0,
        h_ranges=((1.0, 1.0e4), (1.0, 1.0e4)),
    )
    model = Case1Model(cfg)
    print(f"model      : {model.name}")
    print(f"nx x ny x nz = {cfg.nx} x {cfg.ny} x {cfg.nz} -> {cfg.nx*cfg.ny*cfg.nz} cells")

    t0 = time.perf_counter()
    ops = model.core_operators()
    t_compile = time.perf_counter() - t0
    K, C, f = ops.K, ops.C, ops.f
    print(f"assemble   : {t_compile:6.2f}s  (K nnz={K.nnz}, C nnz={C.nnz}, dof={K.shape[0]})")

    G = model.source_shape()
    terms = model.boundary_terms()

    # ---- monkeypatched timer for the sequential enrich solves -------------
    timing = {"solve": 0.0, "solve_calls": 0}

    orig_spd = utils.spd_solve

    def timed_spd(*a, **kw):
        t0 = time.perf_counter()
        out = orig_spd(*a, **kw)
        dt = time.perf_counter() - t0
        timing["solve"] += dt
        timing["solve_calls"] += 1
        return out

    utils.spd_solve = timed_spd
    try:
        t0 = time.perf_counter()
        basis, summary = utils.build_parametric_basis(
            ops,
            G,
            terms,
            model.h_ranges(),
            residual_tolerance=1.0e-3,
            target_relative_epsilon=1.0e-3,
        )
        t_basis = time.perf_counter() - t0
    finally:
        utils.spd_solve = orig_spd

    print(f"\nbuild_parametric_basis (wall): {t_basis:6.2f}s")
    print(f"  ├─ AMG-CG enrich solves     : {timing['solve']:6.2f}s "
          f"({timing['solve_calls']} calls, sequential)")
    print(f"  └─ bounds(parallel)+config  : "
          f"{t_basis - timing['solve']:6.2f}s  (by difference)")
    print(f"  basis_order={summary['basis_order']} pre_svd={summary['pre_svd_order']}")
    print(f"  candidate_count={summary['candidate_count']} processed={summary['processed_candidate_count']}")
    print(f"  outer_count={summary['outer_count']} converged={summary['converged']}")
    print(f"  worst relative response error={summary['relative_response_error']:.3e}")


if __name__ == "__main__":
    main()