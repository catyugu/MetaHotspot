#!/usr/bin/env python3
"""Integrated simple_case1 EROM--FVM steady/transient experiment.

The EROM payload is read from simple_case1_EROM.  A detailed FVM lower cube is
coupled to a detailed FVM exterior as reference, then the same exterior is
coupled to the 14-state EROM.  Both conforming and non-conforming exterior
lattices use Codecasa et al. Section 4/5 common interface patches:

    V_if = E_if @ V_b
    h_if = xi * (E @ h_b)

The private EROM interior field is not exported.  Consequently the reported
"global" ROM error is over the common interface nodes and all external FVM
cells, which are observable in both calculations.  The EROM-side trace error
is reported separately.
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
CASE = ROOT / "playground" / "simple_erom_case1"
sys.path[:0] = [str(PYTHON), str(CASE)]

import metahotspot  # noqa: E402
from metahotspot.enums import GeometryOp, LengthUnit, Study  # noqa: E402
from metahotspot.macromodel.embeddable import FacePort, common_patches  # noqa: E402
from metahotspot.macromodel.utils import normalized_operators  # noqa: E402

DATA = CASE / "simple_case1_EROM"
OUT = CASE / "results" / "simple_erom_fvm_coupling.json"
AMBIENT_K = 308.15
DT = 1.0
DURATION = 100.0
EXTERNAL_K = 21.0
EXTERNAL_H = 1000.0


def read_matrix(name: str) -> np.ndarray:
    data = (DATA / name).read_bytes()
    rows, cols = struct.unpack_from("<2I", data)
    return np.frombuffer(data, dtype="<f8", offset=8, count=rows * cols).reshape(
        rows, cols
    )


def read_diag(name: str) -> np.ndarray:
    data = (DATA / name).read_bytes()
    n = struct.unpack_from("<I", data)[0]
    return np.frombuffer(data, dtype="<f8", offset=4, count=n).copy()


def read_links() -> np.ndarray:
    data = (DATA / "XresLink").read_bytes()
    n = struct.unpack_from("<I", data)[0]
    rows = []
    for i in range(n):
        rec = data[4 + 68 * i : 4 + 68 * (i + 1)]
        rows.append(struct.unpack_from("<5I", rec) + struct.unpack_from("<6d", rec, 20))
    return np.asarray(rows, dtype=np.float64)


def rectangles_from_edges(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (x[i], x[i + 1], y[j], y[j + 1])
            for i in range(x.size - 1)
            for j in range(y.size - 1)
        ],
        dtype=np.float64,
    )


def make_port(rects, k, half, label, direction):
    rects = np.asarray(rects, dtype=np.float64)
    areas = (rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2])
    return FacePort(
        label=label,
        axis=2,
        direction=direction,
        cells=np.arange(rects.shape[0], dtype=np.int64),
        areas=areas,
        k=np.full(rects.shape[0], k, dtype=np.float64),
        half=np.full(rects.shape[0], half, dtype=np.float64),
        t1=0,
        t2=1,
        rects=rects,
    )


def build_lower(x, y):
    """Detailed 100 mm cube from simple_case1.ecxml."""
    model = metahotspot.Model()
    model.set_settings(
        study=Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=AMBIENT_K,
    )
    model.set_mesh(x * 1000.0, y * 1000.0, np.linspace(0.0, 100.0, 16))
    model.add_material("Copper (Pure)", "385", "385", "385", "8930", "385")
    model.add_material("Titanium (Pure)", "21", "21", "21", "4508", "536")
    layer = model.add_layer("100")
    copper = model.add_block(layer, "Copper (Pure)")
    model.add_rect(copper, GeometryOp.ADD, "0", "0", "50", "100")
    titanium = model.add_block(layer, "Titanium (Pure)")
    model.add_rect(titanium, GeometryOp.ADD, "50", "0", "50", "100")
    volume = 0.025 * 0.05 * 0.05
    source_l = model.add_block(layer, "Copper (Pure)", heat_source=str(0.5 / volume))
    model.add_rect(source_l, GeometryOp.ADD, "25", "25", "25", "50")
    source_r = model.add_block(layer, "Titanium (Pure)", heat_source=str(0.5 / volume))
    model.add_rect(source_r, GeometryOp.ADD, "50", "25", "25", "50")
    model.set_default_neumann("0")
    compiled = model.compile()
    return normalized_operators(*compiled.assemble()), compiled


def build_external(x, y):
    """Detailed 50 mm exterior block with h=1000 on its top."""
    model = metahotspot.Model()
    model.set_settings(
        study=Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=AMBIENT_K,
    )
    model.set_mesh(x * 1000.0, y * 1000.0, np.array([0.0, 50.0]))
    model.add_material(
        "external", str(EXTERNAL_K), str(EXTERNAL_K), str(EXTERNAL_K), "4508", "536"
    )
    layer = model.add_layer("50")
    block = model.add_block(layer, "external")
    model.add_rect(block, GeometryOp.ADD, "0", "0", "100", "100")
    model.set_default_neumann("0")
    compiled = model.compile()
    ops = normalized_operators(*compiled.assemble())
    cells = compiled.cells
    bottom = np.flatnonzero(cells.ijk[:, 2] == 0)
    top = np.flatnonzero(cells.ijk[:, 2] == cells.nz - 1)
    bottom = bottom[np.lexsort((cells.centers[bottom, 1], cells.centers[bottom, 0]))]
    top = top[np.lexsort((cells.centers[top, 1], cells.centers[top, 0]))]
    bottom_area = cells.cell_sizes[bottom, 0] * cells.cell_sizes[bottom, 1]
    bottom_g = EXTERNAL_K * bottom_area / (cells.cell_sizes[bottom, 2] / 2.0)
    K = ops.K.tolil()
    top_area = cells.cell_sizes[top, 0] * cells.cell_sizes[top, 1]
    K[top, top] += EXTERNAL_H * top_area
    return ops, K.tocsc(), compiled, bottom, bottom_g


def identity_trace(rows: np.ndarray, n_cell: int) -> sp.csc_matrix:
    return sp.coo_matrix(
        (np.ones(rows.size), (np.arange(rows.size), rows)),
        shape=(rows.size, n_cell),
    ).tocsc()


def assemble_conforming(
    left_K, left_C, left_rhs, left_trace, left_g, right_K, right_C, right_rows, right_g
):
    n_left, n_if, n_right = left_K.shape[0], left_g.size, right_K.shape[0]
    V = sp.csc_matrix(left_trace)
    He, Hf = sp.diags(left_g, format="csc"), sp.diags(right_g, format="csc")
    A = identity_trace(right_rows, n_right)
    K = sp.lil_matrix((n_left + n_if + n_right, n_left + n_if + n_right))
    K[:n_left, :n_left] = left_K + V.T @ He @ V
    K[n_left : n_left + n_if, n_left : n_left + n_if] = He + Hf
    K[n_left + n_if :, n_left + n_if :] = right_K + A.T @ Hf @ A
    node = np.arange(n_left, n_left + n_if)
    right_node_rows = n_left + n_if + right_rows
    K[:n_left, node] = -(V.T @ He)
    K[node, :n_left] = -(He @ V)
    for i, row in enumerate(right_node_rows):
        K[node[i], row] = -Hf[i, i]
        K[row, node[i]] = -Hf[i, i]
    C = sp.block_diag((left_C, sp.csc_matrix((n_if, n_if)), right_C), format="csc")
    rhs = np.r_[np.asarray(left_rhs).reshape(-1), np.zeros(n_if), np.zeros(n_right)]
    return K.tocsc(), C, rhs, V, He, Hf, right_node_rows, None


def assemble_nonconforming(
    left_K,
    left_C,
    left_rhs,
    left_port,
    left_trace,
    left_g,
    right_K,
    right_C,
    right_port,
    right_g,
):
    areas, E_l, E_r, xi_l, xi_r, _, right_owner = common_patches(left_port, right_port)
    V = sp.csc_matrix(E_l @ left_trace)
    hl = xi_l * np.asarray(E_l @ left_g).ravel()
    hr = xi_r * np.asarray(E_r @ right_g).ravel()
    n_left, n_if, n_right = left_K.shape[0], areas.size, right_K.shape[0]
    He, Hf = sp.diags(hl, format="csc"), sp.diags(hr, format="csc")
    A = sp.csc_matrix(E_r)
    K = sp.lil_matrix((n_left + n_if + n_right, n_left + n_if + n_right))
    K[:n_left, :n_left] = left_K + V.T @ He @ V
    K[n_left : n_left + n_if, n_left : n_left + n_if] = He + Hf
    K[n_left + n_if :, n_left + n_if :] = right_K + A.T @ Hf @ A
    node = np.arange(n_left, n_left + n_if)
    right_node_rows = n_left + n_if + np.asarray(right_owner, dtype=np.int64)
    K[:n_left, node] = -(V.T @ He)
    K[node, :n_left] = -(He @ V)
    for i, row in enumerate(right_node_rows):
        K[node[i], row] = -Hf[i, i]
        K[row, node[i]] = -Hf[i, i]
    C = sp.block_diag((left_C, sp.csc_matrix((n_if, n_if)), right_C), format="csc")
    rhs = np.r_[np.asarray(left_rhs).reshape(-1), np.zeros(n_if), np.zeros(n_right)]
    return K.tocsc(), C, rhs, V, He, Hf, right_node_rows, areas


def solve_transient(K, C, rhs):
    times = np.arange(0.0, DURATION + 0.5 * DT, DT)
    steady = spla.spsolve(K, rhs)
    factor = spla.factorized((K + C / DT).tocsc())
    state = np.zeros(K.shape[0])
    history = [state.copy()]
    for _ in times[1:]:
        state = factor(C @ state / DT + rhs)
        history.append(state.copy())
    return steady, np.asarray(history), times


def symmetry_error(K):
    data = (K - K.T).data
    return float(np.max(np.abs(data))) if data.size else 0.0


def run_scenario(
    name,
    x_ext,
    y_ext,
    lower_port,
    lower_trace,
    lower_g,
    lower_K,
    lower_C,
    lower_rhs,
    erom_port,
    erom_trace,
    erom_g,
    Krom,
    Mrom,
    gsrc,
):
    ext_ops, ext_K, ext_compiled, ext_bottom, ext_g = build_external(x_ext, y_ext)
    ext_port = make_port(
        rectangles_from_edges(x_ext, y_ext), EXTERNAL_K, 0.025, "z+", 1
    )
    # Detailed lower/FVM reference.
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
    # EROM/FVM candidate.
    Ke, Ce, re, Ve, He, Hfe, e_rows, e_areas = assemble_nonconforming(
        sp.diags(Krom, format="csc"),
        sp.diags(Mrom, format="csc"),
        gsrc[:, 0],
        erom_port,
        erom_trace,
        erom_g,
        ext_K,
        ext_ops.C,
        ext_port,
        ext_g,
    )
    sd, hd, times = solve_transient(Kd, Cd, rd)
    se, he, times_e = solve_transient(Ke, Ce, re)
    assert np.array_equal(times, times_e)
    nd, ne, nif, nr = lower_K.shape[0], Krom.size, e_areas.size, ext_K.shape[0]
    d_if, e_if = sd[nd : nd + d_areas.size], se[ne : ne + e_areas.size]
    d_ext, e_ext = sd[nd + d_areas.size :], se[ne + e_areas.size :]
    d_if_h, e_if_h = hd[:, nd : nd + d_areas.size], he[:, ne : ne + e_areas.size]
    d_ext_h, e_ext_h = hd[:, nd + d_areas.size :], he[:, ne + e_areas.size :]
    # Both scenarios use the same physical interface area and same number of
    # common patches here, so direct observable comparisons are valid.
    ref_ss = np.r_[d_if, d_ext]
    rom_ss = np.r_[e_if, e_ext]
    ref_h = np.c_[d_if_h, d_ext_h]
    rom_h = np.c_[e_if_h, e_ext_h]
    erom_face_ss = Ve @ se[:ne]
    detail_face_ss = Vd @ sd[:nd]
    erom_face_h = he[:, :ne] @ Ve.T
    detail_face_h = hd[:, :nd] @ Vd.T
    qe = He @ (erom_face_ss - e_if)
    qf = Hfe @ (e_ext[np.asarray(e_rows) - (ne + e_areas.size)] - e_if)
    result = {
        "common_patches": int(e_areas.size),
        "external_cells": int(nr),
        "matrix_shape_detailed": list(Kd.shape),
        "matrix_shape_erom": list(Ke.shape),
        "symmetry_error_detailed": symmetry_error(Kd),
        "symmetry_error_erom": symmetry_error(Ke),
        "area_sum_m2": float(e_areas.sum()),
        "steady_global_shared_max_abs_K": float(np.max(np.abs(rom_ss - ref_ss))),
        "transient_global_shared_max_abs_K": float(np.max(np.abs(rom_h - ref_h))),
        "steady_external_fvm_max_abs_K": float(np.max(np.abs(e_ext - d_ext))),
        "transient_external_fvm_max_abs_K": float(np.max(np.abs(e_ext_h - d_ext_h))),
        "steady_interface_node_max_abs_K": float(np.max(np.abs(e_if - d_if))),
        "transient_interface_node_max_abs_K": float(np.max(np.abs(e_if_h - d_if_h))),
        "steady_erom_trace_max_abs_K": float(
            np.max(np.abs(erom_face_ss - detail_face_ss))
        ),
        "transient_erom_trace_max_abs_K": float(
            np.max(np.abs(erom_face_h - detail_face_h))
        ),
        "steady_flux_balance_max_W": float(np.max(np.abs(qe + qf))),
        "steady_flux_total_erom_W": float(qe.sum()),
        "steady_flux_total_fvm_W": float(qf.sum()),
    }
    print(name, json.dumps(result, indent=2))
    return result


def main():
    started = time.perf_counter()
    Vb = read_matrix("Vb")
    gsrc = read_matrix("g_bci_hat")
    hsrc = read_matrix("h_bci_hat")
    Krom = read_diag("K_bci_hat")
    Mrom = read_diag("M_bci_hat")
    R = read_diag("Xresistances")
    links = read_links()
    x = np.unique(np.r_[links[:, 5], links[:, 6]])
    y = np.unique(np.r_[links[:, 7], links[:, 8]])
    rects = rectangles_from_edges(x, y)
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
    lower_cells = lower_compiled.cells
    top = np.flatnonzero(lower_cells.ijk[:, 2] == lower_cells.nz - 1)
    top = top[np.lexsort((lower_cells.centers[top, 1], lower_cells.centers[top, 0]))]
    top_area = lower_cells.cell_sizes[top, 0] * lower_cells.cell_sizes[top, 1]
    kz = lower_compiled.eval_materials()["conductivity_z"][top]
    lower_g = kz * top_area / (lower_cells.cell_sizes[top, 2] / 2.0)
    lower_port = make_port(rects, 1.0, 1.0, "z-", -1)
    lower_trace = identity_trace(top, lower_ops.K.shape[0])
    detailed = run_scenario(
        "conforming",
        x,
        y,
        lower_port,
        lower_trace,
        lower_g,
        lower_ops.K,
        lower_ops.C,
        lower_ops.f.reshape(-1, 1),
        erom_port,
        Vb,
        areas / R,
        Krom,
        Mrom,
        gsrc,
    )
    xc = np.unique(np.r_[x[::2], x[-1]])
    yc = np.unique(np.r_[y[::2], y[-1]])
    nonconforming = run_scenario(
        "nonconforming",
        xc,
        yc,
        lower_port,
        lower_trace,
        lower_g,
        lower_ops.K,
        lower_ops.C,
        lower_ops.f.reshape(-1, 1),
        erom_port,
        Vb,
        areas / R,
        Krom,
        Mrom,
        gsrc,
    )
    result = {
        "case": "simple_case1",
        "rom_dir": str(DATA),
        "dt_s": DT,
        "duration_s": DURATION,
        "rom_order": int(Krom.size),
        "interface_faces": int(Vb.shape[0]),
        "source_output_vectors": {
            "g_bci_hat": gsrc[:, 0].tolist(),
            "h_bci_hat": hsrc[:, 0].tolist(),
        },
        "scenarios": {"conforming": detailed, "nonconforming": nonconforming},
        "elapsed_s": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
