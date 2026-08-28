#!/usr/bin/env python3
"""Compare production ``embeddable.extract_rom`` with detailed FVM.

The test uses the same active, broad external-source scenarios as
stress_active_external_sources.py. The lower cube is unforced during the test;
its ROM basis is nevertheless extracted from the simple_case1 source and the
production embeddable closure. Detailed FVM/FVM is the reference and the
production EmbeddableRom/FVM coupling is the candidate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "playground" / "simple_erom_case1"
sys.path[:0] = [str(ROOT / "python"), str(CASE)]

from experiment_simple_case1_integrated import (  # noqa: E402
    AMBIENT_K,
    DT,
    DURATION,
    assemble_nonconforming,
    build_external,
    identity_trace,
    make_port,
    rectangles_from_edges,
    read_diag,
    read_links,
    solve_transient,
)
from evaluate_simple_erom_standalone import build_detailed as build_lower  # noqa: E402
from metahotspot.macromodel.embeddable import (  # noqa: E402
    FacePort,
    Subdomain,
    extract_rom,
)

OUT = CASE / "results" / "production_embeddable_active_stress.json"
EXTERNAL_POWER_W = 1.0
EXTERNAL_K = 21.0
EXTERNAL_H = 1000.0
PATTERNS = (
    "uniform",
    "x_left",
    "x_right",
    "y_front",
    "y_back",
    "diagonal_quadrants",
    "anti_diagonal_quadrants",
    "center_square",
    "outer_ring",
)


def pattern_weights(name, centers):
    x, y = centers[:, 0], centers[:, 1]
    if name == "uniform":
        mask = np.ones(x.size, dtype=bool)
    elif name == "x_left":
        mask = x < 0.05
    elif name == "x_right":
        mask = x >= 0.05
    elif name == "y_front":
        mask = y < 0.05
    elif name == "y_back":
        mask = y >= 0.05
    elif name == "diagonal_quadrants":
        mask = (x < 0.05) == (y < 0.05)
    elif name == "anti_diagonal_quadrants":
        mask = (x < 0.05) != (y < 0.05)
    elif name == "center_square":
        mask = (x >= 0.025) & (x < 0.075) & (y >= 0.025) & (y < 0.075)
    elif name == "outer_ring":
        mask = ~((x >= 0.025) & (x < 0.075) & (y >= 0.025) & (y < 0.075))
    else:
        raise KeyError(name)
    w = mask.astype(np.float64)
    return w / w.sum()


def external_source(compiled, name):
    return EXTERNAL_POWER_W * pattern_weights(name, compiled.cells.centers)


def run_one(
    grid_name,
    pattern,
    x,
    y,
    lower,
    erom,
    lower_port,
    lower_trace,
    lower_g,
    erom_port,
    erom_trace,
    erom_g,
    Krom,
    Mrom,
):
    ext_ops, ext_K, ext_compiled, ext_bottom, ext_g = build_external(x, y)
    ext_port = make_port(rectangles_from_edges(x, y), EXTERNAL_K, 0.025, "z+", 1)
    dK, dC, dr, dV, dHe, dHf, drows, darea = assemble_nonconforming(
        lower.K,
        lower.C,
        np.zeros(lower.K.shape[0]),
        lower_port,
        lower_trace,
        lower_g,
        ext_K,
        ext_ops.C,
        ext_port,
        ext_g,
    )
    rK, rC, rr, rV, rHe, rHf, rrows, rarea = assemble_nonconforming(
        sp.csc_matrix(Krom),
        sp.csc_matrix(Mrom),
        np.zeros(Krom.size),
        erom_port,
        erom_trace,
        erom_g,
        ext_K,
        ext_ops.C,
        ext_port,
        ext_g,
    )
    src = external_source(ext_compiled, pattern)
    nd, ne = lower.K.shape[0], Krom.size
    dr[nd + darea.size :] += src
    rr[ne + rarea.size :] += src
    ds, dh, times = solve_transient(dK, dC, dr)
    rs, rh, times_r = solve_transient(rK, rC, rr)
    assert np.array_equal(times, times_r)
    d_if, r_if = ds[nd : nd + darea.size], rs[ne : ne + rarea.size]
    d_ext, r_ext = ds[nd + darea.size :], rs[ne + rarea.size :]
    d_if_h, r_if_h = dh[:, nd : nd + darea.size], rh[:, ne : ne + rarea.size]
    d_ext_h, r_ext_h = dh[:, nd + darea.size :], rh[:, ne + rarea.size :]
    dtrace, rtrace = dV @ ds[:nd], rV @ rs[:ne]
    dtrace_h, rtrace_h = dh[:, :nd] @ dV.T, rh[:, :ne] @ rV.T
    qe = rHe @ (rtrace - r_if)
    ext_idx = np.asarray(rrows) - (ne + rarea.size)
    qf = rHf @ (r_ext[ext_idx] - r_if)
    return {
        "grid": grid_name,
        "pattern": pattern,
        "external_cells": int(ext_K.shape[0]),
        "common_patches": int(rarea.size),
        "external_source_power_W": float(src.sum()),
        "steady_global_shared_max_abs_K": float(
            np.max(np.abs(np.r_[r_if, r_ext] - np.r_[d_if, d_ext]))
        ),
        "transient_global_shared_max_abs_K": float(
            np.max(np.abs(np.c_[r_if_h, r_ext_h] - np.c_[d_if_h, d_ext_h]))
        ),
        "steady_external_fvm_max_abs_K": float(np.max(np.abs(r_ext - d_ext))),
        "transient_external_fvm_max_abs_K": float(np.max(np.abs(r_ext_h - d_ext_h))),
        "steady_interface_node_max_abs_K": float(np.max(np.abs(r_if - d_if))),
        "transient_interface_node_max_abs_K": float(np.max(np.abs(r_if_h - d_if_h))),
        "steady_rom_trace_max_abs_K": float(np.max(np.abs(rtrace - dtrace))),
        "transient_rom_trace_max_abs_K": float(np.max(np.abs(rtrace_h - dtrace_h))),
        "steady_flux_balance_max_W": float(np.max(np.abs(qe + qf))),
        "steady_flux_total_rom_W": float(qe.sum()),
        "steady_flux_total_fvm_W": float(qf.sum()),
    }


def main():
    started = time.perf_counter()
    links = read_links()
    x = np.unique(np.r_[links[:, 5], links[:, 6]])
    y = np.unique(np.r_[links[:, 7], links[:, 8]])
    rects = rectangles_from_edges(x, y)
    lower_ops, compiled = build_lower(x, y)
    lower_cells = compiled.cells
    top = np.flatnonzero(lower_cells.ijk[:, 2] == lower_cells.nz - 1)
    top = top[
        np.lexsort((compiled.cells.centers[top, 1], compiled.cells.centers[top, 0]))
    ]
    top_area = compiled.cells.cell_sizes[top, 0] * compiled.cells.cell_sizes[top, 1]
    kz = compiled.eval_materials()["conductivity_z"][top]
    lower_g = kz * top_area / (compiled.cells.cell_sizes[top, 2] / 2.0)
    lower_trace = identity_trace(top, lower_ops.K.shape[0])
    lower_port = FacePort(
        label="z-",
        axis=2,
        direction=-1,
        cells=np.asarray(top, dtype=np.int64),
        areas=top_area,
        k=np.asarray(kz, dtype=np.float64),
        half=np.asarray(lower_cells.cell_sizes[top, 2] / 2.0, dtype=np.float64),
        t1=0,
        t2=1,
        rects=rects,
    )
    lower_source = lower_ops.f.reshape(-1, 1)
    ambient_top = sp.csc_matrix((top_area, (top, top)), shape=lower_ops.K.shape)
    lower = Subdomain(
        name="simple_case1_lower",
        cells=np.arange(lower_ops.K.shape[0]),
        K=lower_ops.K,
        C=lower_ops.C,
        source=lower_source,
        ports=[lower_port],
        # The top surface is retained as the connectable port, but its
        # extraction-time BCI Robin term is also supplied to Algorithm 1.
        boundary_ports=[lower_port],
        ambient_terms=[ambient_top],
        ambient_ranges=np.asarray([[1.0, 1.0e4]]),
        effective_p=np.asarray([1000.0]),
    )
    extract_started = time.perf_counter()
    rom = extract_rom(
        lower, tolerance=1.0e-3, max_order=64, probe_rounds=2, seed=20260825
    )
    extraction_seconds = time.perf_counter() - extract_started
    erom_trace = rom.boundary_traces["z-"]
    erom_g = rom.boundary_conductances["z-"]
    Krom, Mrom = rom.K0_hat, rom.C_hat
    erom_port = lower_port
    scenarios = {}
    for grid_name, xe, ye in (
        ("conforming", x, y),
        (
            "nonconforming",
            np.unique(np.r_[x[::2], x[-1]]),
            np.unique(np.r_[y[::2], y[-1]]),
        ),
    ):
        scenarios[grid_name] = {}
        for pattern in PATTERNS:
            scenarios[grid_name][pattern] = run_one(
                grid_name,
                pattern,
                xe,
                ye,
                lower,
                rom,
                lower_port,
                lower_trace,
                lower_g,
                erom_port,
                erom_trace,
                erom_g,
                Krom,
                Mrom,
            )
    result = {
        "case": "simple_case1",
        "algorithm": "python.metahotspot.macromodel.embeddable.extract_rom",
        "coupling": "Codecasa Section 4/5 common patches and massless interface nodes",
        "rom_order": int(rom.order),
        "extraction_seconds": extraction_seconds,
        "extraction_summary": rom.summary,
        "source_power_in_basis_W": float(lower_source.sum()),
        "external_source_power_W": EXTERNAL_POWER_W,
        "external_htc_W_m2K": EXTERNAL_H,
        "dt_s": DT,
        "duration_s": DURATION,
        "scenarios": scenarios,
        "elapsed_s": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
