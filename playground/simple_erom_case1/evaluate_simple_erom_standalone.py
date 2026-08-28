#!/usr/bin/env python3
"""Standalone simple_case1 EROM versus detailed-FVM evaluation.

No external FVM is attached here. The detailed model is the 100-mm cube from
simple_case1.ecxml. The EROM is closed against its own top boundary using the
exported trace Vb and the exported center-to-face resistances. Several possible
meanings of the two exported 14-vectors are reported explicitly; no vector is
silently assigned a semantic role that the private format does not establish.
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
CASE = ROOT / "playground" / "simple_erom_case1"
PYTHON = ROOT / "python"
sys.path[:0] = [str(PYTHON)]

import metahotspot  # noqa: E402
from metahotspot.enums import GeometryOp, LengthUnit, Study  # noqa: E402
from metahotspot.macromodel.utils import normalized_operators  # noqa: E402

DATA = CASE / "simple_case1_EROM"
OUT = CASE / "results" / "simple_erom_standalone.json"
AMBIENT_K = 308.15
DT = 1.0
DURATION = 100.0
H_VALUES = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def read_matrix(name):
    data = (DATA / name).read_bytes()
    r, c = struct.unpack_from("<2I", data)
    return np.frombuffer(data, dtype="<f8", offset=8, count=r * c).reshape(r, c)


def read_diag(name):
    data = (DATA / name).read_bytes()
    n = struct.unpack_from("<I", data)[0]
    return np.frombuffer(data, dtype="<f8", offset=4, count=n).copy()


def read_links():
    data = (DATA / "XresLink").read_bytes()
    n = struct.unpack_from("<I", data)[0]
    rows = []
    for i in range(n):
        rec = data[4 + 68 * i : 4 + 68 * (i + 1)]
        rows.append(struct.unpack_from("<5I", rec) + struct.unpack_from("<6d", rec, 20))
    return np.asarray(rows, dtype=np.float64)


def build_detailed(x, y):
    model = metahotspot.Model()
    model.set_settings(
        study=Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=AMBIENT_K,
    )
    model.set_mesh(x * 1000.0, y * 1000.0, np.linspace(0.0, 100.0, 17))
    model.add_material("Copper (Pure)", "385", "385", "385", "8930", "385")
    model.add_material("Titanium (Pure)", "21", "21", "21", "4508", "536")
    # Layers are added top-first. The source occupies z=25..75 mm, matching
    # simple_case1.ecxml; the two passive layers complete the 100-mm cube.
    top_layer = model.add_layer("25")
    copper = model.add_block(top_layer, "Copper (Pure)")
    model.add_rect(copper, GeometryOp.ADD, "0", "0", "50", "100")
    titanium = model.add_block(top_layer, "Titanium (Pure)")
    model.add_rect(titanium, GeometryOp.ADD, "50", "0", "50", "100")

    layer = model.add_layer("50")
    copper = model.add_block(layer, "Copper (Pure)")
    model.add_rect(copper, GeometryOp.ADD, "0", "0", "50", "100")
    titanium = model.add_block(layer, "Titanium (Pure)")
    model.add_rect(titanium, GeometryOp.ADD, "50", "0", "50", "100")
    volume = 0.025 * 0.05 * 0.05
    source_l = model.add_block(layer, "Copper (Pure)", heat_source=str(0.5 / volume))
    model.add_rect(source_l, GeometryOp.ADD, "25", "25", "25", "50")
    source_r = model.add_block(layer, "Titanium (Pure)", heat_source=str(0.5 / volume))
    model.add_rect(source_r, GeometryOp.ADD, "50", "25", "25", "50")

    bottom_layer = model.add_layer("25")
    copper = model.add_block(bottom_layer, "Copper (Pure)")
    model.add_rect(copper, GeometryOp.ADD, "0", "0", "50", "100")
    titanium = model.add_block(bottom_layer, "Titanium (Pure)")
    model.add_rect(titanium, GeometryOp.ADD, "50", "0", "50", "100")
    model.set_default_neumann("0")
    compiled = model.compile()
    return normalized_operators(*compiled.assemble()), compiled


def solve_transient(K, C, rhs):
    steady = spla.spsolve(K.tocsc(), rhs)
    times = np.arange(0.0, DURATION + 0.5 * DT, DT)
    state = np.zeros(K.shape[0])
    history = [state.copy()]
    factor = spla.factorized((K + C / DT).tocsc())
    for _ in times[1:]:
        state = factor(C @ state / DT + rhs)
        history.append(state.copy())
    return steady, np.asarray(history), times


def metrics(reference_ss, candidate_ss, reference_hist, candidate_hist):
    return {
        "steady_max_abs_K": float(np.max(np.abs(candidate_ss - reference_ss))),
        "transient_max_abs_K": float(np.max(np.abs(candidate_hist - reference_hist))),
        "steady_final_abs_K": float(
            np.max(np.abs(candidate_hist[-1] - reference_hist[-1]))
        ),
    }


def evaluate_at_h(
    Vb, ghat, hhat, Khat, Mhat, R, areas, detail_ops, detail_cells, h_phys
):
    n = detail_ops.K.shape[0]
    top = np.flatnonzero(detail_cells.cells.ijk[:, 2] == detail_cells.cells.nz - 1)
    top = top[
        np.lexsort(
            (detail_cells.cells.centers[top, 1], detail_cells.cells.centers[top, 0])
        )
    ]
    top_area = (
        detail_cells.cells.cell_sizes[top, 0] * detail_cells.cells.cell_sizes[top, 1]
    )
    material = detail_cells.eval_materials()
    top_k = np.asarray(material["conductivity_z"])[top]
    top_half = detail_cells.cells.cell_sizes[top, 2] / 2.0
    p = top_k * h_phys / (top_k + h_phys * top_half)
    Hdetail = sp.csc_matrix((p * top_area, (top, top)), shape=(n, n))
    source = detail_ops.f.reshape(-1)
    detail_ss, detail_hist, _ = solve_transient(
        detail_ops.K + Hdetail, detail_ops.C, source
    )
    w = source / source.sum()
    detail_j_ss = float(w @ detail_ss)
    detail_j_hist = detail_hist @ w
    detail_face_ss = detail_ss[top]
    detail_face_hist = detail_hist[:, top]
    g_series = areas / (R + 1.0 / h_phys)
    Hrom = sp.csc_matrix(Vb.T @ (g_series[:, None] * Vb))
    rom_K = sp.diags(Khat, format="csc") + Hrom
    rom_ss, rom_hist, _ = solve_transient(rom_K, sp.diags(Mhat, format="csc"), ghat)
    return {
        "h_W_m2K": h_phys,
        "source_junction_g_steady_error_K": float(ghat @ rom_ss - detail_j_ss),
        "source_junction_g_transient_max_error_K": float(
            np.max(np.abs(rom_hist @ ghat - detail_j_hist))
        ),
        "source_junction_h_steady_error_K": float(hhat @ rom_ss - detail_j_ss),
        "source_junction_h_transient_max_error_K": float(
            np.max(np.abs(rom_hist @ hhat - detail_j_hist))
        ),
        "interface_trace_steady_max_error_K": float(
            np.max(np.abs(Vb @ rom_ss - detail_face_ss))
        ),
        "interface_trace_transient_max_error_K": float(
            np.max(np.abs(rom_hist @ Vb.T - detail_face_hist))
        ),
        "detail_source_steady_rise_K": detail_j_ss,
        "rom_g_source_steady_rise_K": float(ghat @ rom_ss),
    }


def main():
    started = time.perf_counter()
    Vb = read_matrix("Vb")
    ghat = read_matrix("g_bci_hat")[:, 0]
    hhat = read_matrix("h_bci_hat")[:, 0]
    Khat = read_diag("K_bci_hat")
    Mhat = read_diag("M_bci_hat")
    R = read_diag("Xresistances")
    links = read_links()
    x = np.unique(np.r_[links[:, 5], links[:, 6]])
    y = np.unique(np.r_[links[:, 7], links[:, 8]])
    areas = (links[:, 6] - links[:, 5]) * (links[:, 8] - links[:, 7])
    h_phys = 1000.0

    detail_ops, detail_cells = build_detailed(x, y)
    n = detail_ops.K.shape[0]
    top = np.flatnonzero(detail_cells.cells.ijk[:, 2] == detail_cells.cells.nz - 1)
    top = top[
        np.lexsort(
            (detail_cells.cells.centers[top, 1], detail_cells.cells.centers[top, 0])
        )
    ]
    top_area = (
        detail_cells.cells.cell_sizes[top, 0] * detail_cells.cells.cell_sizes[top, 1]
    )
    detail_material = detail_cells.eval_materials()
    top_k = np.asarray(detail_material["conductivity_z"])[top]
    top_half = detail_cells.cells.cell_sizes[top, 2] / 2.0
    detail_h_eff = top_k * h_phys / (top_k + h_phys * top_half)
    Hdetail = sp.csc_matrix((detail_h_eff * top_area, (top, top)), shape=(n, n))
    detail_K = detail_ops.K + Hdetail
    source = detail_ops.f.reshape(-1)
    detail_ss, detail_hist, times = solve_transient(detail_K, detail_ops.C, source)
    source_weight = source / source.sum()
    detail_source_ss = float(source_weight @ detail_ss)
    detail_source_hist = detail_hist @ source_weight
    top_trace = detail_ss[top]
    detail_top_ss = float(np.sum(top_area * top_trace) / np.sum(top_area))
    detail_top_hist = (detail_hist[:, top] @ top_area) / np.sum(top_area)

    # Xresistances stores the area-normalized half-cell resistance R=half/k
    # (units m*K/W). Combining it with h over a face area A gives
    #   g = 1 / (R/A + 1/(h*A)) = A / (R + 1/h).
    g_series = areas / (R + 1.0 / h_phys)
    g_raw = h_phys * areas
    candidates = {}
    for closure_name, g_boundary in (
        ("series_R_plus_convection", g_series),
        ("raw_h_area", g_raw),
    ):
        H = sp.csc_matrix(Vb.T @ (g_boundary[:, None] * Vb))
        erom_K = sp.diags(Khat, format="csc") + H
        erom_C = sp.diags(Mhat, format="csc")
        for input_name, input_vec in (("g_bci_hat", ghat), ("h_bci_hat", hhat)):
            ss, hist, _ = solve_transient(erom_K, erom_C, input_vec)
            # Both exported vectors are possible source/probe outputs in the
            # private format; report both against the detailed source junction.
            for output_name, output_vec in (("g_bci_hat", ghat), ("h_bci_hat", hhat)):
                key = f"{closure_name}__input_{input_name}__output_{output_name}"
                out_ss = float(output_vec @ ss)
                out_hist = hist @ output_vec
                candidates[key] = {
                    **metrics(
                        np.array([detail_source_ss]),
                        np.array([out_ss]),
                        detail_source_hist[:, None],
                        out_hist[:, None],
                    ),
                    "steady_rise_K": out_ss,
                    "input_norm": float(np.linalg.norm(input_vec)),
                    "output_norm": float(np.linalg.norm(output_vec)),
                }

        # Trace error: the EROM connection-face temperatures versus the detailed
        # top cell temperatures, in the export face order.
        ss, hist, _ = solve_transient(erom_K, erom_C, ghat)
        trace_ss = Vb @ ss
        trace_hist = hist @ Vb.T
        candidates[f"{closure_name}__trace_from_g_input"] = {
            "steady_max_abs_K": float(np.max(np.abs(trace_ss - top_trace))),
            "transient_max_abs_K": float(
                np.max(np.abs(trace_hist - detail_hist[:, top]))
            ),
        }

    h_sweep = [
        evaluate_at_h(
            Vb,
            ghat,
            hhat,
            Khat,
            Mhat,
            R,
            areas,
            detail_ops,
            detail_cells,
            float(h),
        )
        for h in H_VALUES
    ]

    result = {
        "case": "simple_case1",
        "mode": "standalone EROM versus detailed FVM; no external FVM attached",
        "rom_order": int(Khat.size),
        "detailed_cells": int(n),
        "interface_faces": int(Vb.shape[0]),
        "dt_s": DT,
        "duration_s": DURATION,
        "boundary_htc_W_m2K": h_phys,
        "detailed_reference": {
            "source_junction_steady_rise_K": detail_source_ss,
            "source_junction_transient_final_rise_K": float(detail_source_hist[-1]),
            "top_area_average_steady_rise_K": detail_top_ss,
            "top_area_average_transient_final_rise_K": float(detail_top_hist[-1]),
            "source_power_W": float(source.sum()),
        },
        "diagnostics": {
            "x_faces": int(x.size - 1),
            "y_faces": int(y.size - 1),
            "top_area_m2": float(areas.sum()),
            "series_conductance_range_W_K": [
                float(g_series.min()),
                float(g_series.max()),
            ],
            "raw_h_area_range_W_K": [float(g_raw.min()), float(g_raw.max())],
            "g_h_dot_product": float(ghat @ hhat),
        },
        "candidates": candidates,
        "h_sweep": h_sweep,
        "elapsed_s": time.perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
