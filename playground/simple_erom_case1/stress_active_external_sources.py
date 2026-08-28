#!/usr/bin/env python3
"""Active low-frequency external-source stress test for simple_case1 EROM.

The lower simple_case1 domain is unforced. Heat is injected only in the
external detailed FVM domain using broad spatial patterns. For each pattern,
a detailed-FVM/detailed-FVM solution is compared with an EROM/FVM solution
through Codecasa Section 4/5 common interface nodes. This isolates whether the
exported EROM boundary trace can represent large-scale nonuniform external
loading; the external FVM itself is identical in both calculations.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "playground" / "simple_erom_case1"
sys.path[:0] = [str(ROOT / "python"), str(CASE)]

from experiment_simple_case1_integrated import (  # noqa: E402
    AMBIENT_K,
    DT,
    DURATION,
    assemble_nonconforming,
    build_external,
    build_lower,
    identity_trace,
    make_port,
    read_diag,
    read_links,
    read_matrix,
    rectangles_from_edges,
    solve_transient,
)
from metahotspot.macromodel.embeddable import FacePort  # noqa: E402

DATA = CASE / "simple_case1_EROM"
OUT = CASE / "results" / "simple_erom_active_external_stress.json"
EXTERNAL_POWER_W = 1.0
HTC = 1000.0


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
    weights = mask.astype(np.float64)
    if not weights.any():
        raise RuntimeError(f"empty external source pattern: {name}")
    return weights / weights.sum()


def external_source(compiled, pattern):
    weights = pattern_weights(pattern, compiled.cells.centers)
    return EXTERNAL_POWER_W * weights


def add_right_rhs(rhs, n_left, n_if, source_right):
    result = np.asarray(rhs, dtype=np.float64).copy()
    result[n_left + n_if :] += source_right
    return result


def run_scenario(
    name,
    x_ext,
    y_ext,
    lower_port,
    lower_trace,
    lower_g,
    lower_K,
    lower_C,
    erom_port,
    erom_trace,
    erom_g,
    Krom,
    Mrom,
    pattern,
):
    ext_ops, ext_K, ext_compiled, ext_bottom, ext_g = build_external(x_ext, y_ext)
    ext_rects = rectangles_from_edges(x_ext, y_ext)
    ext_port = make_port(ext_rects, 21.0, 0.025, "z+", 1)
    lower_rhs = np.zeros(lower_K.shape[0], dtype=np.float64)
    source_right = external_source(ext_compiled, pattern)

    Kd, Cd, rd, Vd, Hd, Hfd, d_rows, d_areas = assemble_nonconforming(
        lower_K,
        lower_C,
        lower_rhs,
        lower_port,
        lower_trace,
        lower_g,
        ext_K,
        ext_ops.C,
        ext_port,
        ext_g,
    )
    Ke, Ce, re, Ve, He, Hfe, e_rows, e_areas = assemble_nonconforming(
        sp.diags(Krom, format="csc"),
        sp.diags(Mrom, format="csc"),
        np.zeros(Krom.size),
        erom_port,
        erom_trace,
        erom_g,
        ext_K,
        ext_ops.C,
        ext_port,
        ext_g,
    )
    rd = add_right_rhs(rd, lower_K.shape[0], d_areas.size, source_right)
    re = add_right_rhs(re, Krom.size, e_areas.size, source_right)
    dss, dh, times = solve_transient(Kd, Cd, rd)
    ess, eh, times_e = solve_transient(Ke, Ce, re)
    assert np.array_equal(times, times_e)

    nd, ne = lower_K.shape[0], Krom.size
    d_if = dss[nd : nd + d_areas.size]
    e_if = ess[ne : ne + e_areas.size]
    d_ext = dss[nd + d_areas.size :]
    e_ext = ess[ne + e_areas.size :]
    d_if_h = dh[:, nd : nd + d_areas.size]
    e_if_h = eh[:, ne : ne + e_areas.size]
    d_ext_h = dh[:, nd + d_areas.size :]
    e_ext_h = eh[:, ne + e_areas.size :]
    d_trace = Vd @ dss[:nd]
    e_trace = Ve @ ess[:ne]
    d_trace_h = dh[:, :nd] @ Vd.T
    e_trace_h = eh[:, :ne] @ Ve.T

    qe = He @ (e_trace - e_if)
    qf = Hfe @ (e_ext[np.asarray(e_rows) - (ne + e_areas.size)] - e_if)
    result = {
        "pattern": pattern,
        "external_cells": int(ext_K.shape[0]),
        "common_patches": int(e_areas.size),
        "external_source_power_W": float(source_right.sum()),
        "steady_global_shared_max_abs_K": float(
            np.max(np.abs(np.r_[e_if, e_ext] - np.r_[d_if, d_ext]))
        ),
        "transient_global_shared_max_abs_K": float(
            np.max(np.abs(np.c_[e_if_h, e_ext_h] - np.c_[d_if_h, d_ext_h]))
        ),
        "steady_external_fvm_max_abs_K": float(np.max(np.abs(e_ext - d_ext))),
        "transient_external_fvm_max_abs_K": float(np.max(np.abs(e_ext_h - d_ext_h))),
        "steady_interface_node_max_abs_K": float(np.max(np.abs(e_if - d_if))),
        "transient_interface_node_max_abs_K": float(np.max(np.abs(e_if_h - d_if_h))),
        "steady_erom_trace_max_abs_K": float(np.max(np.abs(e_trace - d_trace))),
        "transient_erom_trace_max_abs_K": float(np.max(np.abs(e_trace_h - d_trace_h))),
        "steady_flux_balance_max_W": float(np.max(np.abs(qe + qf))),
        "steady_flux_total_erom_W": float(qe.sum()),
        "steady_flux_total_fvm_W": float(qf.sum()),
        "reference_external_steady_max_rise_K": float(np.max(d_ext)),
        "rom_external_steady_max_rise_K": float(np.max(e_ext)),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    started = time.perf_counter()
    links = read_links()
    x = np.unique(np.r_[links[:, 5], links[:, 6]])
    y = np.unique(np.r_[links[:, 7], links[:, 8]])
    rects = rectangles_from_edges(x, y)
    Vb = read_matrix("Vb")
    Krom = read_diag("K_bci_hat")
    Mrom = read_diag("M_bci_hat")
    R = read_diag("Xresistances")
    areas = (links[:, 6] - links[:, 5]) * (links[:, 8] - links[:, 7])
    erom_port = FacePort(
        label="z-",
        axis=2,
        direction=-1,
        cells=np.arange(areas.size),
        areas=areas,
        k=np.ones(areas.size),
        half=R,
        t1=0,
        t2=1,
        rects=rects,
    )
    lower_ops, lower_compiled = build_lower(x, y)
    top = np.flatnonzero(lower_compiled.cells.ijk[:, 2] == lower_compiled.cells.nz - 1)
    top = top[
        np.lexsort(
            (lower_compiled.cells.centers[top, 1], lower_compiled.cells.centers[top, 0])
        )
    ]
    top_area = (
        lower_compiled.cells.cell_sizes[top, 0]
        * lower_compiled.cells.cell_sizes[top, 1]
    )
    kz = lower_compiled.eval_materials()["conductivity_z"][top]
    lower_g = kz * top_area / (lower_compiled.cells.cell_sizes[top, 2] / 2.0)
    lower_port = make_port(rects, 1.0, 1.0, "z-", -1)
    lower_trace = identity_trace(top, lower_ops.K.shape[0])
    scenarios = {}
    patterns = (
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
    for grid_name, xe, ye in (
        ("conforming", x, y),
        (
            "nonconforming",
            np.unique(np.r_[x[::2], x[-1]]),
            np.unique(np.r_[y[::2], y[-1]]),
        ),
    ):
        scenarios[grid_name] = {}
        for pattern in patterns:
            scenarios[grid_name][pattern] = run_scenario(
                grid_name,
                xe,
                ye,
                lower_port,
                lower_trace,
                lower_g,
                lower_ops.K,
                lower_ops.C,
                erom_port,
                Vb,
                areas / R,
                Krom,
                Mrom,
                pattern,
            )
    result = {
        "case": "simple_case1",
        "method": "active external low-frequency broad-source stress test",
        "coupling": "Codecasa 2017 Section 4/5 common patches and massless interface nodes",
        "ambient_K": AMBIENT_K,
        "external_htc_W_m2K": HTC,
        "external_source_power_W": EXTERNAL_POWER_W,
        "dt_s": DT,
        "duration_s": DURATION,
        "rom_order": int(Krom.size),
        "interface_faces": int(Vb.shape[0]),
        "scenarios": scenarios,
        "elapsed_s": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
