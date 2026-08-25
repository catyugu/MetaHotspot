#!/usr/bin/env python3
"""Interface-node coupling experiments for the Case-1 BCI-ROM model."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = ROOT / "playground" / "bci_rom_testcase1"
MACRO_DIR = ROOT / "playground" / "macromodel"
sys.path[:0] = [str(CASE_DIR), str(MACRO_DIR), str(ROOT / "python")]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from utils import build_parametric_basis, normalized_operators  # noqa: E402

AMBIENT_K = 308.15
BOUNDARY_H = (5.0e1, 1.0e3)
POWER_W = np.array([0.1, 0.2, 0.3, 0.4])
INTERFACE_Z = 10.0e-3
ROM_TOLERANCE = 1.0e-3
ROM_MAX_ORDER = 512
ROM_SEED = 20260825
DT_S = 5.0
DURATION_S = 100.0
RESULT_PATH = ROOT / "results" / "experiments" / "interface_coupling_experiments.json"


@dataclass
class Side:
    cells: np.ndarray
    stiffness: sp.csc_matrix
    capacitance: sp.csc_matrix
    source: np.ndarray
    interface_cells: np.ndarray
    interface_rectangles: np.ndarray
    interface_conductivity: np.ndarray
    boundary_terms: list[sp.csc_matrix]


@dataclass
class Interface:
    areas: np.ndarray
    upper_cells: np.ndarray
    lower_cells: np.ndarray
    upper_e: sp.csc_matrix
    lower_e: sp.csc_matrix
    upper_xi: sp.csc_matrix
    lower_xi: sp.csc_matrix


def build_full_system(model):
    core = model.core_operators()
    effective_h = model.physical_to_effective(BOUNDARY_H)
    stiffness = core.K.tocsc().copy()
    for coefficient, term in zip(effective_h, model.boundary_terms()):
        stiffness += float(coefficient) * term.tocsc()
    stiffness = (0.5 * (stiffness + stiffness.T)).tocsc()
    return stiffness, core.C.tocsc(), model.source_shape()


def split_cells(model, upper):
    z = model.cell_layout.centers[:, 2]
    if upper:
        return np.flatnonzero(z >= INTERFACE_Z - 1.0e-12)
    return np.flatnonzero(z < INTERFACE_Z - 1.0e-12)


def interface_geometry(model, cells):
    layout = model.cell_layout
    cells = np.asarray(cells, dtype=np.int64)
    distance = np.abs(layout.centers[cells, 2] - INTERFACE_Z)
    adjacent = distance <= layout.half_sizes[cells, 2] + 1.0e-12
    cells = cells[adjacent]
    if cells.size == 0:
        raise RuntimeError("no cells found at the requested interface")

    nearest = np.abs(layout.centers[cells, 2] - INTERFACE_Z).min()
    cells = cells[np.abs(layout.centers[cells, 2] - INTERFACE_Z) <= nearest + 1.0e-12]
    center = layout.centers[cells, :2]
    half = layout.half_sizes[cells, :2]
    rectangles = np.column_stack(
        (
            center[:, 0] - half[:, 0],
            center[:, 0] + half[:, 0],
            center[:, 1] - half[:, 1],
            center[:, 1] + half[:, 1],
        )
    )
    return cells, rectangles


def build_side(model, stiffness, capacitance, source, upper):
    cells = np.asarray(split_cells(model, upper), dtype=np.int64)
    outside = np.setdiff1d(np.arange(stiffness.shape[0]), cells)
    cross = stiffness[cells, :][:, outside].tocoo()
    interface_diagonal = np.zeros(cells.size)
    for row, value in zip(cross.row, cross.data):
        if value < 0.0:
            interface_diagonal[row] -= value

    local_index = {int(cell): row for row, cell in enumerate(cells)}
    interface_cells, rectangles = interface_geometry(model, cells)
    local_interface = np.array(
        [local_index[int(cell)] for cell in interface_cells], dtype=np.int64
    )
    layout = model.cell_layout
    normal_conductivity = layout.conductivity[interface_cells, 2]
    half_distance = layout.half_sizes[interface_cells, 2]
    terms = [
        term.tocsc()[cells, :][:, cells].tocsc() for term in model.boundary_terms()
    ]
    return Side(
        cells=cells,
        stiffness=(
            stiffness[cells, :][:, cells] - sp.diags(interface_diagonal)
        ).tocsc(),
        capacitance=capacitance[cells, :][:, cells].tocsc(),
        source=np.asarray(source[cells], dtype=np.float64),
        interface_cells=local_interface,
        interface_rectangles=rectangles,
        interface_conductivity=np.column_stack((normal_conductivity, half_distance)),
        boundary_terms=terms,
    )


def contains(rectangles, xl, xr, yl, yr):
    return np.flatnonzero(
        (rectangles[:, 0] <= xl + 1.0e-12)
        & (rectangles[:, 1] >= xr - 1.0e-12)
        & (rectangles[:, 2] <= yl + 1.0e-12)
        & (rectangles[:, 3] >= yr - 1.0e-12)
    )


def build_interface(upper, lower):
    upper_rect = upper.interface_rectangles
    lower_rect = lower.interface_rectangles
    x_edges = np.unique(np.r_[upper_rect[:, (0, 1)], lower_rect[:, (0, 1)]])
    y_edges = np.unique(np.r_[upper_rect[:, (2, 3)], lower_rect[:, (2, 3)]])

    areas = []
    upper_cells = []
    lower_cells = []
    for xl, xr in zip(x_edges[:-1], x_edges[1:]):
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            if xr <= xl or yr <= yl:
                continue
            upper_match = contains(upper_rect, xl, xr, yl, yr)
            lower_match = contains(lower_rect, xl, xr, yl, yr)
            if upper_match.size and lower_match.size:
                areas.append((xr - xl) * (yr - yl))
                upper_cells.append(upper_match[0])
                lower_cells.append(lower_match[0])

    if not areas:
        raise RuntimeError("interface grids do not overlap")
    areas = np.asarray(areas, dtype=np.float64)
    upper_cells = np.asarray(upper_cells, dtype=np.int64)
    lower_cells = np.asarray(lower_cells, dtype=np.int64)
    patch_index = np.arange(areas.size)
    upper_e = sp.coo_matrix(
        (np.ones(areas.size), (patch_index, upper_cells)),
        shape=(areas.size, upper.interface_cells.size),
    ).tocsc()
    lower_e = sp.coo_matrix(
        (np.ones(areas.size), (patch_index, lower_cells)),
        shape=(areas.size, lower.interface_cells.size),
    ).tocsc()
    upper_face_area = upper.interface_conductivity[upper_cells, 1]
    lower_face_area = lower.interface_conductivity[lower_cells, 1]
    upper_xi = sp.diags(areas / upper_face_area) @ upper_e
    lower_xi = sp.diags(areas / lower_face_area) @ lower_e
    return Interface(
        areas, upper_cells, lower_cells, upper_e, lower_e, upper_xi, lower_xi
    )


def build_rom(side, interface_area_by_cell):
    area = np.asarray(interface_area_by_cell, dtype=np.float64)
    boundary_terms = [side.boundary_terms[0], sp.diags(area)]
    operators = normalized_operators(
        side.stiffness, side.capacitance, np.zeros(side.cells.size)
    )
    source = side.source
    if not np.any(source):
        source = np.zeros((side.cells.size, 1))
        source[side.interface_cells, 0] = np.maximum(
            area[side.interface_cells], 1.0e-30
        )
        source /= source[:, 0].sum()
    return build_parametric_basis(
        operators,
        source,
        boundary_terms,
        [[1.0, 1.0e4], [1.0, 1.0e4]],
        tolerance=ROM_TOLERANCE,
        max_order=ROM_MAX_ORDER,
        probe_rounds=2,
        seed=ROM_SEED,
    )


def identity_basis(side):
    return np.eye(side.cells.size), {"seconds": 0.0}


def assemble_coupled(upper, lower, interface, upper_basis, lower_basis):
    upper_order = upper_basis.shape[1]
    lower_order = lower_basis.shape[1]
    upper_trace = upper_basis[upper.interface_cells[interface.upper_cells], :]
    lower_trace = lower_basis[lower.interface_cells[interface.lower_cells], :]
    upper_g = upper.interface_conductivity[interface.upper_cells, 0] * interface.areas
    lower_g = lower.interface_conductivity[interface.lower_cells, 0] * interface.areas
    upper_diag = sp.diags(upper_g)
    lower_diag = sp.diags(lower_g)
    upper_k = (
        upper_basis.T @ upper.stiffness @ upper_basis
        + upper_trace.T @ upper_diag @ upper_trace
    )
    lower_k = (
        lower_basis.T @ lower.stiffness @ lower_basis
        + lower_trace.T @ lower_diag @ lower_trace
    )
    interface_count = interface.areas.size
    stiffness = sp.bmat(
        [
            [upper_k, -upper_trace.T @ upper_diag, None],
            [
                -upper_diag @ upper_trace,
                upper_diag + lower_diag,
                -lower_diag @ lower_trace,
            ],
            [None, -lower_trace.T @ lower_diag, lower_k],
        ],
        format="csc",
    )
    capacitance = sp.block_diag(
        (
            upper_basis.T @ upper.capacitance @ upper_basis,
            sp.csc_matrix((interface_count, interface_count)),
            lower_basis.T @ lower.capacitance @ lower_basis,
        ),
        format="csc",
    )
    upper_source = upper_basis.T @ upper.source
    lower_source = lower_basis.T @ lower.source
    rhs = np.r_[
        upper_source @ POWER_W, np.zeros(interface_count), lower_source @ POWER_W
    ]
    return stiffness, capacitance, rhs, upper_source, lower_source


def solve_system(stiffness, capacitance, rhs):
    steady = np.asarray(spla.spsolve(stiffness, rhs)).ravel()
    lhs = (stiffness + capacitance / DT_S).tocsc()
    solver = spla.splu(lhs)
    state = np.zeros(stiffness.shape[0])
    history = [state.copy()]
    for _ in range(round(DURATION_S / DT_S)):
        state = solver.solve(capacitance @ state / DT_S + rhs)
        history.append(state.copy())
    return steady, np.asarray(history)


def run_case(upper, lower, interface, upper_basis, lower_basis, reference):
    stiffness, capacitance, rhs, upper_source, lower_source = assemble_coupled(
        upper, lower, interface, upper_basis, lower_basis
    )
    steady, history = solve_system(stiffness, capacitance, rhs)
    upper_order = upper_basis.shape[1]
    lower_order = lower_basis.shape[1]
    upper_junction = AMBIENT_K + upper_source.T @ steady[:upper_order]
    lower_junction = AMBIENT_K + lower_source.T @ steady[-lower_order:]
    junction = np.r_[upper_junction, lower_junction]
    interface_state = steady[upper_order : upper_order + interface.areas.size]
    return {
        "basis_order": [int(upper_order), int(lower_order)],
        "interface_patches": int(interface.areas.size),
        "steady_junction_K": junction.tolist(),
        "reference_junction_K": reference.tolist(),
        "steady_interface_peak_rise_K": float(interface_state.max()),
        "max_junction_error_K": float(np.max(np.abs(junction - reference))),
        "transient_steps": int(history.shape[0]),
    }


def make_model(cell_size_mm):
    return Case1Model(
        Case1Config(
            max_xy_cell_mm=cell_size_mm,
            max_z_cell_mm=2.5,
            dt_s=DT_S,
            duration_s=DURATION_S,
        )
    )


def detailed_reference(upper, lower, interface):
    upper_basis, lower_basis = identity_basis(upper)[0], identity_basis(lower)[0]
    stiffness, capacitance, rhs, upper_source, lower_source = assemble_coupled(
        upper, lower, interface, upper_basis, lower_basis
    )
    steady, _ = solve_system(stiffness, capacitance, rhs)
    upper_count = upper.cells.size
    return (
        AMBIENT_K
        + np.r_[
            upper_source.T @ steady[:upper_count],
            lower_source.T @ steady[-lower.cells.size :],
        ]
    )


def run():
    fine_model = make_model(2.5)
    coarse_model = make_model(3.0)
    fine_stiffness, fine_capacitance, fine_source = build_full_system(fine_model)
    coarse_stiffness, coarse_capacitance, coarse_source = build_full_system(
        coarse_model
    )
    fine_upper = build_side(
        fine_model, fine_stiffness, fine_capacitance, fine_source, True
    )
    fine_lower = build_side(
        fine_model, fine_stiffness, fine_capacitance, fine_source, False
    )
    coarse_lower = build_side(
        coarse_model, coarse_stiffness, coarse_capacitance, coarse_source, False
    )

    conforming = build_interface(fine_upper, fine_lower)
    nonconforming = build_interface(fine_upper, coarse_lower)
    conforming_reference = detailed_reference(fine_upper, fine_lower, conforming)
    nonconforming_reference = detailed_reference(
        fine_upper, coarse_lower, nonconforming
    )

    fine_area = np.bincount(
        fine_upper.interface_cells[conforming.upper_cells],
        weights=conforming.areas,
        minlength=fine_upper.cells.size,
    )
    upper_rom, upper_summary = build_rom(fine_upper, fine_area)
    identity_upper, identity_summary = identity_basis(fine_upper)
    identity_lower, _ = identity_basis(fine_lower)

    results = {
        "conforming": run_case(
            fine_upper,
            fine_lower,
            conforming,
            identity_upper,
            identity_lower,
            conforming_reference,
        ),
        "nonconforming": run_case(
            fine_upper,
            coarse_lower,
            nonconforming,
            identity_upper,
            identity_basis(coarse_lower)[0],
            nonconforming_reference,
        ),
        "rom_fvm": run_case(
            fine_upper,
            fine_lower,
            conforming,
            upper_rom,
            identity_lower,
            conforming_reference,
        ),
    }
    payload = {
        "method": "independent interface nodes with common-patch area weights",
        "area_weight_matrices": {
            "conforming": {
                "upper_E_shape": list(conforming.upper_e.shape),
                "lower_E_shape": list(conforming.lower_e.shape),
                "upper_Xi_nnz": int(conforming.upper_xi.nnz),
                "lower_Xi_nnz": int(conforming.lower_xi.nnz),
            },
            "nonconforming": {
                "upper_E_shape": list(nonconforming.upper_e.shape),
                "lower_E_shape": list(nonconforming.lower_e.shape),
                "upper_Xi_nnz": int(nonconforming.upper_xi.nnz),
                "lower_Xi_nnz": int(nonconforming.lower_xi.nnz),
            },
        },
        "rom_extraction": {
            "basis_order": int(upper_rom.shape[1]),
            "seconds": float(upper_summary["seconds"]),
            "relative_response_error": float(upper_summary["relative_response_error"]),
        },
        "results": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
