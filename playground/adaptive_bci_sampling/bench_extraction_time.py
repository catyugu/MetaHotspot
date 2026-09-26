"""Where the 1 mm deterministic extraction spends its time (phase split)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

import pyamg  # noqa: E402

import deterministic_design as dd  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402

mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
maximum_points = int(sys.argv[2]) if len(sys.argv) > 2 else 3
out = Path(sys.argv[3]) if len(sys.argv) > 3 else None

model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
kernel = model.core.K.tocsc()
mass = model.core.C.tocsc()
source = np.asarray(model.source_shape, dtype=np.float64)
terms = [term.tocsc() for term in model.boundary_terms]
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
power = np.asarray(model.nominal_power(), dtype=np.float64)
record = {"mesh_mm": mesh, "cells": int(kernel.shape[0]), "ports": int(source.shape[1])}
print(f"mesh={mesh} n={kernel.shape[0]} ports={source.shape[1]}", flush=True)

stats = {"setups": 0, "setup_seconds": 0.0, "cg_seconds": 0.0, "rhs": 0}


def amg_response_block(cache, kernel, mass, terms, source, point, shift=0.0):
    key = (float(f"{shift:.14e}"),) + tuple(
        float(f"{value:.14e}") for value in np.asarray(point, float)
    )
    block = cache.get(key)
    if block is None:
        operator = kernel if not shift else kernel + float(shift) * mass
        matrix = dd.full_operator(operator, terms, point).tocsr()
        started = time.perf_counter()
        ml = pyamg.ruge_stuben_solver(matrix, interpolation="direct")
        preconditioner = ml.aspreconditioner(cycle="V")
        stats["setups"] += 1
        stats["setup_seconds"] += time.perf_counter() - started
        block = np.empty((matrix.shape[0], source.shape[1]), dtype=np.float64)
        started = time.perf_counter()
        for column in range(source.shape[1]):
            block[:, column], info = spla.cg(
                matrix, np.ascontiguousarray(source[:, column]), rtol=1e-6,
                atol=0.0, maxiter=2000, M=preconditioner,
            )
            if info != 0:
                raise RuntimeError(f"AMG-CG did not converge: info={info}")
        stats["cg_seconds"] += time.perf_counter() - started
        stats["rhs"] += source.shape[1]
        cache[key] = block
    return block


original = dd.response_block
dd.response_block = amg_response_block
try:
    clock = time.perf_counter()
    plan = dd.shared_frequency_plan(kernel, mass, source, 1e-3)
    record["seconds_plan"] = time.perf_counter() - clock
    cache = {}
    clock = time.perf_counter()
    points, selection_certificate, selection = dd.certified_greedy_points(
        kernel, terms, source, ranges, None, tolerance=1e-5,
        maximum_points=maximum_points, grid=41, power=power, metric="entrywise",
        cache=cache,
    )
    record["seconds_selection"] = time.perf_counter() - clock
    clock = time.perf_counter()
    basis, snapshots, info = dd.build_basis(
        kernel, mass, terms, source, points, plan=plan, tolerance=1e-3,
        include_dc=True, cache=cache,
    )
    record["seconds_build"] = time.perf_counter() - clock
finally:
    dd.response_block = original

record.update(
    setups=stats["setups"],
    setup_seconds=stats["setup_seconds"],
    cg_seconds=stats["cg_seconds"],
    rhs=stats["rhs"],
    operators=int(info["operators"]),
    basis_order=int(info["basis_order"]),
    shift_count=int(plan["count"]),
    selection_certificate=float(selection_certificate),
)
record["seconds_solve"] = record["setup_seconds"] + record["cg_seconds"]
record["seconds_total"] = (
    record["seconds_plan"] + record["seconds_selection"] + record["seconds_build"]
)
print(
    "phases: plan {seconds_plan:.1f}s + selection {seconds_selection:.1f}s + "
    "build {seconds_build:.1f}s = {seconds_total:.1f}s".format(**record), flush=True)
print(
    "solves: {setups} setups x {setup_mean:.3f}s = {setup_seconds:.1f}s + CG "
    "{cg_seconds:.1f}s for {rhs} rhs  => solve total {seconds_solve:.1f}s, "
    "non-solve {non_solve:.1f}s".format(
        setup_mean=record["setup_seconds"] / max(record["setups"], 1),
        non_solve=record["seconds_total"] - record["seconds_solve"],
        **record,
    ), flush=True)
print(f"operators {record['operators']}, order {record['basis_order']}, "
      f"shift_count {record['shift_count']}, selection certificate "
      f"{record['selection_certificate']:.3e}", flush=True)

if out:
    out.write_text(json.dumps(record, indent=2) + "\n")
    print("wrote", out, flush=True)
