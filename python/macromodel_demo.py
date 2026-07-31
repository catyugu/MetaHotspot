#!/usr/bin/env python3
"""Transient BCI-ROM benchmark using MetaHotspot C++ solves."""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple
import numpy as np

try:
    import scipy.linalg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:
    raise SystemExit("SciPy is required for the BCI-ROM benchmark") from exc
import metahotspot
from metahotspot.compiled import SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortCoupling, PortModel, solve as solve_macromodel


@dataclass(frozen=True)
class PackageConfig:
    nx: int = 24
    ny: int = 24
    width_mm: float = 40.0
    height_mm: float = 40.0
    ambient_K: float = 300.0
    substrate_mm: float = 1.2
    bump_mm: float = 0.24
    die_mm: float = 0.6
    tim_mm: float = 0.18
    spreader_mm: float = 1.2
    cold_plate_mm: float = 1.5
    substrate_cells: int = 4
    bump_cells: int = 2
    die_cells: int = 3
    tim_cells: int = 1
    spreader_cells: int = 3
    cold_plate_cells: int = 3
    bump_rows: int = 8
    bump_columns: int = 8
    bump_width_mm: float = 0.9
    chiplet_width_mm: float = 12.0
    chiplet_height_mm: float = 12.0
    chiplet_power_W: float = 25.0

    @property
    def detailed_nz(self) -> int:
        return self.substrate_cells + self.bump_cells + self.die_cells

    @property
    def total_nz(self) -> int:
        return (
            self.detailed_nz
            + self.tim_cells
            + self.spreader_cells
            + self.cold_plate_cells
        )

    @property
    def physical_ports(self) -> int:
        return self.nx * self.ny

    @property
    def total_height_mm(self) -> float:
        return (
            self.substrate_mm
            + self.bump_mm
            + self.die_mm
            + self.tim_mm
            + self.spreader_mm
            + self.cold_plate_mm
        )


@dataclass(frozen=True)
class ExperimentConfig:
    error_limit_K: float = 0.1
    port_modes: int = 484
    fixed_interface_modes: int = 224
    rational_modes: int = 224
    source_residual_modes: int = 560
    rational_block: int = 16
    rational_shifts: int = 6
    duration_s: float = 0.5
    time_step_s: float = 0.025
    nominal_h_W_m2K: float = 2500.0
    boundary_h_values_W_m2K: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    preprocess_budget_s: float = 30.0
    random_seed: int = 20260731


@dataclass
class MethodResult:
    name: str
    physical_ports: int
    port_modes: int
    rom_order: int
    preprocess_s: float
    steady_error_K: float
    transient_error_K: float
    transient_records: int
    passed: bool


@dataclass
class BoundaryResult:
    h_W_m2K: float
    steady_error_K: float
    transient_error_K: float
    passed: bool


class PackageData(NamedTuple):
    full_adiabatic: object
    detailed_steady: object
    detailed_transient: object
    K_full: sp.csc_matrix
    C_full: sp.csc_matrix
    f_full: np.ndarray
    K_detailed: sp.csc_matrix
    C_detailed: sp.csc_matrix
    f_detailed: np.ndarray
    full_detailed_cells: np.ndarray
    full_macro_cells: np.ndarray
    detailed_interface_cells: np.ndarray
    full_top_cells: np.ndarray
    macro_interface_local_cells: np.ndarray
    macro_top_local_cells: np.ndarray
    detailed_half_conductance: np.ndarray
    macro_half_conductance: np.ndarray
    K_macro_cells: sp.csc_matrix
    C_macro_cells: sp.csc_matrix
    f_macro_cells: np.ndarray
    top_face_area_m2: float
    top_half_length_m: float
    top_conductivity_W_mK: float


class MacroCore(NamedTuple):
    Kpp: sp.csc_matrix
    Kpi: sp.csc_matrix
    Kip: sp.csc_matrix
    Kii: sp.csc_matrix
    Cii: sp.csc_matrix
    fi: np.ndarray
    constraint_map: np.ndarray
    transfer_vectors: np.ndarray
    transfer_values: np.ndarray
    fixed_interface_modes: np.ndarray
    preprocess_s: float


class MethodBasis(NamedTuple):
    name: str
    V: np.ndarray
    physical_basis: np.ndarray
    port_modes: int
    preprocess_s: float


class ReducedMacro(NamedTuple):
    K: sp.csc_matrix
    C: sp.csc_matrix
    f: np.ndarray
    V: np.ndarray
    physical_basis: np.ndarray


class CppReference(NamedTuple):
    steady_temperature: np.ndarray
    transient_times: np.ndarray
    transient_temperature: np.ndarray


def _axis_vertices(length_mm: float, cells: int) -> np.ndarray:
    return np.linspace(0.0, length_mm, cells + 1, dtype=np.float64)


def _layered_z_vertices(cfg: PackageConfig, include_macro: bool) -> np.ndarray:
    layers = [
        (cfg.substrate_mm, cfg.substrate_cells),
        (cfg.bump_mm, cfg.bump_cells),
        (cfg.die_mm, cfg.die_cells),
    ]
    if include_macro:
        layers.extend(
            [
                (cfg.tim_mm, cfg.tim_cells),
                (cfg.spreader_mm, cfg.spreader_cells),
                (cfg.cold_plate_mm, cfg.cold_plate_cells),
            ]
        )
    vertices = [0.0]
    cursor = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            cursor += thickness / cells
            vertices.append(cursor)
    return np.asarray(vertices, dtype=np.float64)


def _add_materials(model) -> None:
    model.add_material("organic", "0.65", "0.65", "0.55", "1900", "1100")
    model.add_material("underfill", "0.80", "0.80", "0.80", "1550", "1000")
    model.add_material("copper", "390", "390", "390", "8960", "385")
    model.add_material("mold", "0.85", "0.85", "0.75", "1850", "1000")
    model.add_material("silicon", "130", "130", "115", "2330", "700")
    model.add_material("tim", "4.0", "4.0", "3.0", "2500", "900")
    model.add_material("aluminum", "180", "180", "180", "2700", "900")


def _add_full_rect(model, block: int, cfg: PackageConfig) -> None:
    model.add_rect(
        block, GeometryOp.ADD, "0", "0", f"{cfg.width_mm:.17g}", f"{cfg.height_mm:.17g}"
    )


def _chiplet_positions(cfg: PackageConfig) -> tuple[tuple[float, float], ...]:
    margin_x = 5.0
    margin_y = 5.0
    return (
        (margin_x, margin_y),
        (cfg.width_mm - margin_x - cfg.chiplet_width_mm, margin_y),
        (margin_x, cfg.height_mm - margin_y - cfg.chiplet_height_mm),
        (
            cfg.width_mm - margin_x - cfg.chiplet_width_mm,
            cfg.height_mm - margin_y - cfg.chiplet_height_mm,
        ),
    )


def _chiplet_heat_source(cfg: PackageConfig) -> float:
    volume_m3 = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1e-09
    return cfg.chiplet_power_W / volume_m3


def _power_points(duration_s: float) -> np.ndarray:
    return np.asarray(
        [
            (0.0, 0.0),
            (0.2 * duration_s, 1.0),
            (0.55 * duration_s, 1.0),
            (0.75 * duration_s, 0.55),
            (duration_s, 0.85),
        ],
        dtype=np.float64,
    )


def build_package_model(
    cfg: PackageConfig,
    *,
    include_macro: bool,
    study: Study,
    duration_s: float = 0.0,
    output_interval_s: float = 0.0,
    convection_h_W_m2K: float | None = None,
):
    model = metahotspot.Model()
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=duration_s,
        output_interval=output_interval_s,
    )
    model.set_mesh(
        _axis_vertices(cfg.width_mm, cfg.nx),
        _axis_vertices(cfg.height_mm, cfg.ny),
        _layered_z_vertices(cfg, include_macro),
    )
    _add_materials(model)
    transient = study == Study.TRANSIENT
    if transient:
        model.add_function_piecewise("power_scale", _power_points(duration_s))
    source_value = _chiplet_heat_source(cfg)
    source_expression = (
        f"{source_value:.17g}*power_scale(x)" if transient else f"{source_value:.17g}"
    )
    if include_macro:
        cold_plate_layer = model.add_layer(f"{cfg.cold_plate_mm:.17g}")
        cold_plate = model.add_block(cold_plate_layer, "aluminum")
        _add_full_rect(model, cold_plate, cfg)
        spreader_layer = model.add_layer(f"{cfg.spreader_mm:.17g}")
        spreader = model.add_block(spreader_layer, "copper")
        _add_full_rect(model, spreader, cfg)
        tim_layer = model.add_layer(f"{cfg.tim_mm:.17g}")
        tim = model.add_block(tim_layer, "tim")
        _add_full_rect(model, tim, cfg)
    die_layer = model.add_layer(f"{cfg.die_mm:.17g}")
    mold = model.add_block(die_layer, "mold")
    _add_full_rect(model, mold, cfg)
    for x, y in _chiplet_positions(cfg):
        chiplet = model.add_block(die_layer, "silicon", heat_source=source_expression)
        model.add_rect(
            chiplet,
            GeometryOp.ADD,
            f"{x:.17g}",
            f"{y:.17g}",
            f"{cfg.chiplet_width_mm:.17g}",
            f"{cfg.chiplet_height_mm:.17g}",
        )
    bump_layer = model.add_layer(f"{cfg.bump_mm:.17g}")
    underfill = model.add_block(bump_layer, "underfill")
    _add_full_rect(model, underfill, cfg)
    pitch_x = cfg.width_mm / cfg.bump_columns
    pitch_y = cfg.height_mm / cfg.bump_rows
    for iy in range(cfg.bump_rows):
        for ix in range(cfg.bump_columns):
            x = (ix + 0.5) * pitch_x - 0.5 * cfg.bump_width_mm
            y = (iy + 0.5) * pitch_y - 0.5 * cfg.bump_width_mm
            bump = model.add_block(bump_layer, "copper")
            model.add_rect(
                bump,
                GeometryOp.ADD,
                f"{x:.17g}",
                f"{y:.17g}",
                f"{cfg.bump_width_mm:.17g}",
                f"{cfg.bump_width_mm:.17g}",
            )
    substrate_layer = model.add_layer(f"{cfg.substrate_mm:.17g}")
    substrate = model.add_block(substrate_layer, "organic")
    _add_full_rect(model, substrate, cfg)
    model.set_default_neumann("0")
    if include_macro and convection_h_W_m2K is not None:
        model.add_convection(
            f"{convection_h_W_m2K:.17g}",
            f"{cfg.ambient_K:.17g}",
            regions=[
                (Axis.Z, cfg.total_height_mm, 0.0, cfg.width_mm, 0.0, cfg.height_mm)
            ],
        )
    return model


def _grid_cell(compiled, ix: int, iy: int, iz: int) -> int:
    grid = (ix * compiled.ny + iy) * compiled.nz + iz
    cell = int(compiled.grid_to_cell[grid])
    if cell == np.iinfo(np.uintp).max:
        raise RuntimeError(f"inactive grid cell at ({ix}, {iy}, {iz})")
    return cell


def _ordered_cells(compiled, z_begin: int, z_end: int) -> np.ndarray:
    return np.asarray(
        [
            _grid_cell(compiled, ix, iy, iz)
            for ix in range(compiled.nx)
            for iy in range(compiled.ny)
            for iz in range(z_begin, z_end)
        ],
        dtype=np.int64,
    )


def _max_sparse_abs(matrix: sp.spmatrix) -> float:
    return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))


def assemble_package(cfg: PackageConfig, exp: ExperimentConfig) -> PackageData:
    full_model = build_package_model(cfg, include_macro=True, study=Study.STEADY)
    detailed_steady_model = build_package_model(
        cfg, include_macro=False, study=Study.STEADY
    )
    detailed_transient_model = build_package_model(
        cfg,
        include_macro=False,
        study=Study.TRANSIENT,
        duration_s=exp.duration_s,
        output_interval_s=exp.time_step_s,
    )
    full = full_model.compile()
    detailed_steady = detailed_steady_model.compile()
    detailed_transient = detailed_transient_model.compile()
    K_full, C_full, f_full = full.assemble()
    K_detailed, C_detailed, f_detailed = detailed_steady.assemble()
    K_full = K_full.tocsc()
    C_full = C_full.tocsc()
    K_detailed = K_detailed.tocsc()
    C_detailed = C_detailed.tocsc()
    full_detailed_cells = _ordered_cells(full, 0, cfg.detailed_nz)
    full_macro_cells = _ordered_cells(full, cfg.detailed_nz, cfg.total_nz)
    detailed_order = _ordered_cells(detailed_steady, 0, cfg.detailed_nz)
    if not np.array_equal(detailed_order, np.arange(detailed_steady.cell_count)):
        raise RuntimeError("unexpected detailed compact-cell ordering")
    detailed_interface_cells = np.asarray(
        [
            _grid_cell(detailed_steady, ix, iy, cfg.detailed_nz - 1)
            for ix in range(cfg.nx)
            for iy in range(cfg.ny)
        ],
        dtype=np.int64,
    )
    full_detailed_interface = np.asarray(
        [
            _grid_cell(full, ix, iy, cfg.detailed_nz - 1)
            for ix in range(cfg.nx)
            for iy in range(cfg.ny)
        ],
        dtype=np.int64,
    )
    full_macro_interface = np.asarray(
        [
            _grid_cell(full, ix, iy, cfg.detailed_nz)
            for ix in range(cfg.nx)
            for iy in range(cfg.ny)
        ],
        dtype=np.int64,
    )
    full_top_cells = np.asarray(
        [
            _grid_cell(full, ix, iy, cfg.total_nz - 1)
            for ix in range(cfg.nx)
            for iy in range(cfg.ny)
        ],
        dtype=np.int64,
    )
    macro_position = {int(cell): pos for pos, cell in enumerate(full_macro_cells)}
    macro_interface_local = np.asarray(
        [macro_position[int(cell)] for cell in full_macro_interface], dtype=np.int64
    )
    macro_top_local = np.asarray(
        [macro_position[int(cell)] for cell in full_top_cells], dtype=np.int64
    )
    g_series = np.asarray(
        [
            -float(K_full[detailed_cell, macro_cell])
            for detailed_cell, macro_cell in zip(
                full_detailed_interface, full_macro_interface, strict=True
            )
        ],
        dtype=np.float64,
    )
    g_detailed = detailed_steady.half_conductance(
        detailed_interface_cells, Face.ZP, cfg.ambient_K, 0.0
    )
    g_macro = full.half_conductance(full_macro_interface, Face.ZM, cfg.ambient_K, 0.0)
    expected_series = g_detailed * g_macro / (g_detailed + g_macro)
    series_error = float(np.max(np.abs(g_series - expected_series)))
    series_scale = max(1.0, float(np.max(np.abs(expected_series))))
    if series_error > 1e-09 * series_scale:
        raise RuntimeError(
            f"C/C++ interface contract mismatch: monolithic and half-conductance series operators differ by {series_error:.6e}"
        )
    K_macro = K_full[full_macro_cells, :][:, full_macro_cells].tocsc()
    macro_diag = np.zeros(full_macro_cells.size, dtype=np.float64)
    macro_diag[macro_interface_local] = g_series
    K_macro = K_macro - sp.diags(macro_diag, format="csc")
    C_macro = C_full[full_macro_cells, :][:, full_macro_cells].tocsc()
    f_macro = np.asarray(f_full[full_macro_cells], dtype=np.float64)
    K_detail_from_full = K_full[full_detailed_cells, :][:, full_detailed_cells].tocsc()
    detailed_position = {int(cell): pos for pos, cell in enumerate(full_detailed_cells)}
    detailed_diag = np.zeros(full_detailed_cells.size, dtype=np.float64)
    for cell, conductance in zip(full_detailed_interface, g_series, strict=True):
        detailed_diag[detailed_position[int(cell)]] = conductance
    K_detail_from_full = K_detail_from_full - sp.diags(detailed_diag, format="csc")
    operator_error = _max_sparse_abs(K_detail_from_full - K_detailed)
    operator_scale = max(1.0, _max_sparse_abs(K_detailed))
    if operator_error > 1e-09 * operator_scale:
        raise RuntimeError(
            f"independent detailed compile differs from stripped full operator: {operator_error:.6e}"
        )
    dx_m = cfg.width_mm * 0.001 / cfg.nx
    dy_m = cfg.height_mm * 0.001 / cfg.ny
    top_dz_m = cfg.cold_plate_mm * 0.001 / cfg.cold_plate_cells
    return PackageData(
        full_adiabatic=full,
        detailed_steady=detailed_steady,
        detailed_transient=detailed_transient,
        K_full=K_full,
        C_full=C_full,
        f_full=np.asarray(f_full, dtype=np.float64),
        K_detailed=K_detailed,
        C_detailed=C_detailed,
        f_detailed=np.asarray(f_detailed, dtype=np.float64),
        full_detailed_cells=full_detailed_cells,
        full_macro_cells=full_macro_cells,
        detailed_interface_cells=detailed_interface_cells,
        full_top_cells=full_top_cells,
        macro_interface_local_cells=macro_interface_local,
        macro_top_local_cells=macro_top_local,
        detailed_half_conductance=np.asarray(g_detailed, dtype=np.float64),
        macro_half_conductance=np.asarray(g_macro, dtype=np.float64),
        K_macro_cells=K_macro,
        C_macro_cells=C_macro,
        f_macro_cells=f_macro,
        top_face_area_m2=dx_m * dy_m,
        top_half_length_m=0.5 * top_dz_m,
        top_conductivity_W_mK=180.0,
    )


def build_macro_core(data: PackageData, exp: ExperimentConfig) -> MacroCore:
    start = time.perf_counter()
    n_ports = data.detailed_interface_cells.size
    n_macro = data.full_macro_cells.size
    rows = data.macro_interface_local_cells
    columns = np.arange(n_ports, dtype=np.int64)
    Kpp = sp.diags(data.macro_half_conductance, format="csc")
    Kip = sp.coo_matrix(
        (-data.macro_half_conductance, (rows, columns)), shape=(n_macro, n_ports)
    ).tocsc()
    Kpi = Kip.T.tocsc()
    diagonal = np.zeros(n_macro, dtype=np.float64)
    diagonal[rows] = data.macro_half_conductance
    Kii = data.K_macro_cells + sp.diags(diagonal, format="csc")
    Cii = data.C_macro_cells.tocsc()
    factor = spla.splu(Kii)
    constraint_map = -factor.solve(Kip.toarray())
    steklov = Kpp.toarray() + np.asarray(Kpi @ constraint_map)
    steklov = 0.5 * (steklov + steklov.T)
    values, vectors = scipy.linalg.eigh(steklov, check_finite=False, driver="evr")
    order = np.argsort(values)
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    mode_count = min(exp.fixed_interface_modes, n_macro - 2)
    if mode_count > 0:
        fixed_values, fixed_modes = spla.eigsh(
            Kii, k=mode_count, M=Cii, sigma=0.0, which="LM", tol=1e-08
        )
        fixed_modes = np.asarray(
            fixed_modes[:, np.argsort(fixed_values)], dtype=np.float64
        )
    else:
        fixed_modes = np.empty((n_macro, 0), dtype=np.float64)
    return MacroCore(
        Kpp=Kpp,
        Kpi=Kpi,
        Kip=Kip,
        Kii=Kii,
        Cii=Cii,
        fi=data.f_macro_cells,
        constraint_map=constraint_map,
        transfer_vectors=vectors,
        transfer_values=values,
        fixed_interface_modes=fixed_modes,
        preprocess_s=time.perf_counter() - start,
    )


def dct_port_basis(nx: int, ny: int, count: int) -> np.ndarray:
    modes = sorted(
        ((kx, ky) for kx in range(nx) for ky in range(ny)),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, item[0], item[1]),
    )[:count]
    x = np.arange(nx, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    columns = []
    for kx, ky in modes:
        vx = np.cos(math.pi * kx * x / nx)
        vy = np.cos(math.pi * ky * y / ny)
        vx *= math.sqrt((1.0 if kx == 0 else 2.0) / nx)
        vy *= math.sqrt((1.0 if ky == 0 else 2.0) / ny)
        columns.append(np.kron(vx, vy))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def source_weighted_port_basis(cfg: PackageConfig, count: int) -> np.ndarray:
    full = dct_port_basis(cfg.nx, cfg.ny, cfg.physical_ports)
    x_centres = (np.arange(cfg.nx, dtype=np.float64) + 0.5) * (cfg.width_mm / cfg.nx)
    y_centres = (np.arange(cfg.ny, dtype=np.float64) + 0.5) * (cfg.height_mm / cfg.ny)
    source_maps = []
    for x0, y0 in _chiplet_positions(cfg):
        source_maps.append(
            np.asarray(
                [
                    (
                        1.0
                        if x0 <= x <= x0 + cfg.chiplet_width_mm
                        and y0 <= y <= y0 + cfg.chiplet_height_mm
                        else 0.0
                    )
                    for x in x_centres
                    for y in y_centres
                ],
                dtype=np.float64,
            )
        )
    relevance = np.linalg.norm(np.abs(full.T @ np.column_stack(source_maps)), axis=1)
    ranked = np.argsort(-relevance, kind="stable")
    selected = [0]
    selected_set = {0}
    for index in ranked:
        index = int(index)
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        if len(selected) == count:
            break
    return np.ascontiguousarray(full[:, selected], dtype=np.float64)


def orthonormal_range(matrix: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    q, r, _ = scipy.linalg.qr(
        matrix, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    rank = int(np.count_nonzero(diagonal > tolerance * max(1.0, float(diagonal.max()))))
    return np.ascontiguousarray(q[:, :rank], dtype=np.float64)


def randomized_left_basis(
    snapshots: np.ndarray, rank: int, seed: int, oversampling: int = 32
) -> np.ndarray:
    rank = min(rank, snapshots.shape[0], snapshots.shape[1])
    if rank <= 0:
        return np.empty((snapshots.shape[0], 0), dtype=np.float64)
    sample_count = min(snapshots.shape[1], rank + oversampling)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((snapshots.shape[1], sample_count))
    sample = snapshots @ omega
    sample = snapshots @ (snapshots.T @ sample)
    q = np.linalg.qr(sample, mode="reduced")[0]
    compressed = q.T @ snapshots
    u_hat, _, _ = scipy.linalg.svd(
        compressed, full_matrices=False, check_finite=False, lapack_driver="gesdd"
    )
    return np.ascontiguousarray(q @ u_hat[:, :rank], dtype=np.float64)


def _guyan(name: str, phi: np.ndarray, core: MacroCore, elapsed: float) -> MethodBasis:
    psi = core.constraint_map @ phi
    V = np.vstack((phi, psi))
    return MethodBasis(name, V, phi, phi.shape[1], elapsed)


def _craig_bampton(phi: np.ndarray, core: MacroCore, elapsed: float) -> MethodBasis:
    psi = core.constraint_map @ phi
    modes = core.fixed_interface_modes
    zeros = np.zeros((phi.shape[0], modes.shape[1]), dtype=np.float64)
    V = np.block([[phi, zeros], [psi, modes]])
    physical = np.hstack((phi, zeros))
    return MethodBasis("transfer_craig_bampton", V, physical, phi.shape[1], elapsed)


def _rational_krylov(
    phi: np.ndarray, core: MacroCore, exp: ExperimentConfig, elapsed: float
) -> MethodBasis:
    rng = np.random.default_rng(exp.random_seed)
    block = min(exp.rational_block, phi.shape[1])
    shifts = np.logspace(
        math.log10(1.0 / max(exp.duration_s, exp.time_step_s)),
        math.log10(1.0 / exp.time_step_s),
        exp.rational_shifts,
    )
    forcing = core.Kip @ phi
    snapshots = []
    for shift in shifts:
        directions = rng.standard_normal((phi.shape[1], block))
        directions = np.linalg.qr(directions, mode="reduced")[0]
        factor = spla.splu((core.Kii + shift * core.Cii).tocsc())
        snapshots.append(factor.solve(-(forcing @ directions)))
    modes = orthonormal_range(np.hstack(snapshots))[:, : exp.rational_modes]
    psi = core.constraint_map @ phi
    zeros = np.zeros((phi.shape[0], modes.shape[1]), dtype=np.float64)
    V = np.block([[phi, zeros], [psi, modes]])
    physical = np.hstack((phi, zeros))
    return MethodBasis("transfer_rational_krylov", V, physical, phi.shape[1], elapsed)


def _source_aware_bci(
    cfg: PackageConfig, data: PackageData, core: MacroCore, exp: ExperimentConfig
) -> MethodBasis:
    start = time.perf_counter()
    phi = source_weighted_port_basis(cfg, exp.port_modes)
    psi = core.constraint_map @ phi
    top_diagonal = np.zeros(core.Kii.shape[0], dtype=np.float64)
    top_diagonal[data.macro_top_local_cells] = data.top_face_area_m2
    unit_top_operator = sp.diags(top_diagonal, format="csc")
    static_factor = spla.splu(core.Kii)
    boundary_modes = orthonormal_range(-static_factor.solve(unit_top_operator @ psi))
    forcing = core.Kip @ phi
    shift_scale = 0.025 / exp.time_step_s
    shifts = shift_scale * np.asarray(
        (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0), dtype=np.float64
    )
    residual_blocks = []
    for shift in shifts:
        factor = spla.splu((core.Kii + float(shift) * core.Cii).tocsc())
        residual = factor.solve(-forcing) - psi
        if boundary_modes.shape[1] > 0:
            residual -= boundary_modes @ (boundary_modes.T @ residual)
        residual_blocks.append(residual)
    snapshots = np.hstack(residual_blocks)
    dynamic_modes = randomized_left_basis(
        snapshots, exp.source_residual_modes, seed=exp.random_seed
    )
    if boundary_modes.shape[1] > 0:
        dynamic_modes -= boundary_modes @ (boundary_modes.T @ dynamic_modes)
        dynamic_modes = orthonormal_range(dynamic_modes)
    interior_only = np.hstack((boundary_modes, dynamic_modes))
    zeros = np.zeros((cfg.physical_ports, interior_only.shape[1]), dtype=np.float64)
    physical = np.hstack((phi, zeros))
    interior = np.hstack((psi, interior_only))
    V = np.vstack((physical, interior))
    return MethodBasis(
        "source_aware_bci_krylov",
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(physical, dtype=np.float64),
        phi.shape[1],
        core.preprocess_s + time.perf_counter() - start,
    )


def build_method_bases(
    cfg: PackageConfig, data: PackageData, core: MacroCore, exp: ExperimentConfig
) -> list[MethodBasis]:
    if not 0 < exp.port_modes < cfg.physical_ports:
        raise ValueError(f"port_modes must be in [1, {cfg.physical_ports - 1}]")
    methods: list[MethodBasis] = []
    start = time.perf_counter()
    phi_dct = dct_port_basis(cfg.nx, cfg.ny, exp.port_modes)
    methods.append(
        _guyan(
            "dct_guyan", phi_dct, core, core.preprocess_s + time.perf_counter() - start
        )
    )
    phi_transfer = core.transfer_vectors[:, : exp.port_modes]
    methods.append(_guyan("transfer_guyan", phi_transfer, core, core.preprocess_s))
    methods.append(_craig_bampton(phi_transfer, core, core.preprocess_s))
    start = time.perf_counter()
    methods.append(
        _rational_krylov(
            phi_transfer, core, exp, core.preprocess_s + time.perf_counter() - start
        )
    )
    methods.append(_source_aware_bci(cfg, data, core, exp))
    return methods


def _top_convection_conductance(data: PackageData, h_W_m2K: float) -> float:
    k = data.top_conductivity_W_mK
    half = data.top_half_length_m
    return k * h_W_m2K * data.top_face_area_m2 / (k + h_W_m2K * half)


def project_macro(
    data: PackageData,
    core: MacroCore,
    method: MethodBasis,
    cfg: PackageConfig,
    h_W_m2K: float,
) -> ReducedMacro:
    V = method.V
    n_ports = cfg.physical_ports
    Vp = V[:n_ports, :]
    Vi = V[n_ports:, :]
    K = np.asarray(Vp.T @ (core.Kpp @ Vp), dtype=np.float64)
    K += np.asarray(Vp.T @ (core.Kpi @ Vi), dtype=np.float64)
    K += np.asarray(Vi.T @ (core.Kip @ Vp), dtype=np.float64)
    K += np.asarray(Vi.T @ (core.Kii @ Vi), dtype=np.float64)
    C = np.asarray(Vi.T @ (core.Cii @ Vi), dtype=np.float64)
    rhs = np.asarray(Vi.T @ core.fi, dtype=np.float64)
    conductance = _top_convection_conductance(data, h_W_m2K)
    Vtop = Vi[data.macro_top_local_cells, :]
    K += Vtop.T @ (conductance * Vtop)
    rhs += Vtop.T @ np.full(
        Vtop.shape[0], conductance * cfg.ambient_K, dtype=np.float64
    )
    K = 0.5 * (K + K.T)
    C = 0.5 * (C + C.T)
    return ReducedMacro(
        sp.csc_matrix(K),
        sp.csc_matrix(C),
        np.asarray(rhs, dtype=np.float64),
        V,
        method.physical_basis,
    )


def _solve_options(exp: ExperimentConfig, transient: bool) -> SolveOptions:
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1e-12,
        linear_max_iterations=5000,
        underrelaxation=1.0,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1e-11,
        nonlinear_absolute_tolerance=1e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1e-09,
        min_dt=exp.time_step_s if transient else 1e-12,
        max_dt=exp.time_step_s if transient else 1.0,
        fixed_dt=exp.time_step_s if transient else 1.0,
    )


def _macro_initial_coordinates(reduced: ReducedMacro, ambient_K: float) -> np.ndarray:
    target = np.full(reduced.V.shape[0], ambient_K, dtype=np.float64)
    coordinates, _, _, _ = scipy.linalg.lstsq(
        reduced.V, target, cond=1e-12, lapack_driver="gelsy", check_finite=False
    )
    residual = float(np.max(np.abs(reduced.V @ coordinates - target)))
    if residual > 1e-08:
        raise RuntimeError(
            f"ROM basis cannot represent uniform initial temperature: {residual:.3e} K"
        )
    return np.asarray(coordinates, dtype=np.float64)


def build_cpp_reference(
    cfg: PackageConfig, exp: ExperimentConfig, h_W_m2K: float
) -> CppReference:
    steady_model = build_package_model(
        cfg, include_macro=True, study=Study.STEADY, convection_h_W_m2K=h_W_m2K
    ).compile()
    transient_model = build_package_model(
        cfg,
        include_macro=True,
        study=Study.TRANSIENT,
        duration_s=exp.duration_s,
        output_interval_s=exp.time_step_s,
        convection_h_W_m2K=h_W_m2K,
    ).compile()
    steady = steady_model.solve(opts=_solve_options(exp, transient=False))
    transient = transient_model.solve(opts=_solve_options(exp, transient=True))
    return CppReference(
        np.asarray(steady.temperature, dtype=np.float64).copy(),
        np.asarray(transient.history_times, dtype=np.float64).copy(),
        np.asarray(transient.temperature_history, dtype=np.float64).copy(),
    )


def _solve_reduced_cpp(
    data: PackageData,
    reduced: ReducedMacro,
    cfg: PackageConfig,
    exp: ExperimentConfig,
    transient: bool,
):
    compiled = data.detailed_transient if transient else data.detailed_steady
    initial = np.concatenate(
        (
            np.full(compiled.cell_count, cfg.ambient_K, dtype=np.float64),
            _macro_initial_coordinates(reduced, cfg.ambient_K),
        )
    )
    model = PortModel(
        operators=(reduced.K, reduced.C, reduced.f),
        basis=reduced.physical_basis,
        physical_port_count=cfg.physical_ports,
    )
    coupling = PortCoupling(
        model_cells=data.detailed_interface_cells.copy(), model_face=int(Face.ZP)
    )
    return solve_macromodel(
        compiled, model, coupling, initial, _solve_options(exp, transient=transient)
    )


def _recover_history(
    data: PackageData,
    reduced: ReducedMacro,
    reduced_history: np.ndarray,
    cfg: PackageConfig,
) -> np.ndarray:
    detailed_count = data.detailed_steady.cell_count
    recovered = np.empty(
        (reduced_history.shape[0], data.K_full.shape[0]), dtype=np.float64
    )
    recovered[:, data.full_detailed_cells] = reduced_history[:, :detailed_count]
    augmented_macro = reduced_history[:, detailed_count:] @ reduced.V.T
    recovered[:, data.full_macro_cells] = augmented_macro[:, cfg.physical_ports :]
    return recovered


def evaluate_method(
    data: PackageData,
    core: MacroCore,
    method: MethodBasis,
    reference: CppReference,
    cfg: PackageConfig,
    exp: ExperimentConfig,
    h_W_m2K: float,
) -> tuple[float, float, int, ReducedMacro]:
    reduced = project_macro(data, core, method, cfg, h_W_m2K)
    steady_solution = _solve_reduced_cpp(data, reduced, cfg, exp, transient=False)
    steady_history = np.asarray(steady_solution.state, dtype=np.float64)[None, :]
    recovered_steady = _recover_history(data, reduced, steady_history, cfg)[0]
    steady_error = float(
        np.max(np.abs(recovered_steady - reference.steady_temperature))
    )
    transient_solution = _solve_reduced_cpp(data, reduced, cfg, exp, transient=True)
    times = np.asarray(transient_solution.history_times, dtype=np.float64)
    if times.shape != reference.transient_times.shape or not np.allclose(
        times, reference.transient_times, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            "full and reduced C++ solvers returned different output time grids"
        )
    reduced_history = np.asarray(transient_solution.state_history, dtype=np.float64)
    recovered_transient = _recover_history(data, reduced, reduced_history, cfg)
    transient_error = float(
        np.max(np.abs(recovered_transient - reference.transient_temperature))
    )
    return (steady_error, transient_error, len(times), reduced)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--port-modes", type=int)
    parser.add_argument("--interior-modes", type=int)
    parser.add_argument("--source-residual-modes", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument(
        "--output", type=Path, default=Path("results/bci_rom_final_results.json")
    )
    return parser.parse_args(argv)


def _configs_from_args(
    args: argparse.Namespace,
) -> tuple[PackageConfig, ExperimentConfig]:
    package = PackageConfig()
    experiment = ExperimentConfig()
    if args.quick:
        package = PackageConfig(nx=16, ny=16, bump_rows=6, bump_columns=6)
        experiment = ExperimentConfig(
            port_modes=208,
            fixed_interface_modes=96,
            rational_modes=96,
            source_residual_modes=240,
            duration_s=0.2,
            time_step_s=0.025,
        )
    package_values = asdict(package)
    experiment_values = asdict(experiment)
    if args.nx is not None:
        package_values["nx"] = args.nx
    if args.ny is not None:
        package_values["ny"] = args.ny
    if args.port_modes is not None:
        experiment_values["port_modes"] = args.port_modes
    if args.interior_modes is not None:
        experiment_values["fixed_interface_modes"] = args.interior_modes
        experiment_values["rational_modes"] = args.interior_modes
    if args.source_residual_modes is not None:
        experiment_values["source_residual_modes"] = args.source_residual_modes
    if args.duration is not None:
        experiment_values["duration_s"] = args.duration
    if args.dt is not None:
        experiment_values["time_step_s"] = args.dt
    return (PackageConfig(**package_values), ExperimentConfig(**experiment_values))


def _print_method(result: MethodResult) -> None:
    print(
        f"  {result.name:<30} order={result.rom_order:4d} port={result.port_modes:4d}/{result.physical_ports:<4d} prep={result.preprocess_s:7.3f}s steady={result.steady_error_K:9.5f}K transient={result.transient_error_K:9.5f}K {('PASS' if result.passed else 'FAIL')}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg, exp = _configs_from_args(args)
    print("=" * 88)
    print("Transient BCI-ROM package benchmark — C++ solve path")
    print("=" * 88)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.total_nz} = {cfg.nx * cfg.ny * cfg.total_nz:,} thermal cells"
    )
    print(
        f"Physical interface ports: {cfg.physical_ports}; requested port modes: {exp.port_modes}"
    )
    build_start = time.perf_counter()
    data = assemble_package(cfg, exp)
    build_s = time.perf_counter() - build_start
    print(f"Model build + C++ assembly + interface validation: {build_s:.3f}s")
    core = build_macro_core(data, exp)
    methods = build_method_bases(cfg, data, core, exp)
    print(
        f"Common isolated-macro preprocessing: {core.preprocess_s:.3f}s; Steklov spectrum[0:3]={core.transfer_values[:3]}"
    )
    print("Building nominal full-order C++ references...")
    nominal_reference = build_cpp_reference(cfg, exp, exp.nominal_h_W_m2K)
    nominal_results: list[MethodResult] = []
    print("\nNominal-boundary method comparison:")
    for method in methods:
        steady, transient, records, _ = evaluate_method(
            data, core, method, nominal_reference, cfg, exp, exp.nominal_h_W_m2K
        )
        passed = (
            steady <= exp.error_limit_K
            and transient <= exp.error_limit_K
            and (method.preprocess_s <= exp.preprocess_budget_s)
        )
        result = MethodResult(
            name=method.name,
            physical_ports=cfg.physical_ports,
            port_modes=method.port_modes,
            rom_order=method.V.shape[1],
            preprocess_s=method.preprocess_s,
            steady_error_K=steady,
            transient_error_K=transient,
            transient_records=records,
            passed=passed,
        )
        nominal_results.append(result)
        _print_method(result)
    candidates = [result for result in nominal_results if result.passed]
    if candidates:
        winner = min(
            candidates,
            key=lambda item: (
                item.rom_order,
                max(item.steady_error_K, item.transient_error_K),
                item.preprocess_s,
            ),
        )
    else:
        winner = min(
            nominal_results,
            key=lambda item: max(item.steady_error_K, item.transient_error_K),
        )
        print(
            f"\nERROR: no method met {exp.error_limit_K:.3f} K and {exp.preprocess_budget_s:.1f}s preprocessing limits",
            file=sys.stderr,
        )
    print(f"\nSelected method: {winner.name}")
    method_by_name = {method.name: method for method in methods}
    winning_method = method_by_name[winner.name]
    boundary_results: list[BoundaryResult] = []
    print("\nBoundary-independence reuse sweep (basis is not re-extracted):")
    reference_cache = {exp.nominal_h_W_m2K: nominal_reference}
    for h_value in exp.boundary_h_values_W_m2K:
        reference = reference_cache.get(h_value)
        if reference is None:
            reference = build_cpp_reference(cfg, exp, h_value)
            reference_cache[h_value] = reference
        steady, transient, _, _ = evaluate_method(
            data, core, winning_method, reference, cfg, exp, h_value
        )
        passed = steady <= exp.error_limit_K and transient <= exp.error_limit_K
        boundary_results.append(BoundaryResult(h_value, steady, transient, passed))
        print(
            f"  h={h_value:8.1f} W/(m2 K): steady={steady:9.5f} K, transient={transient:9.5f} K {('PASS' if passed else 'FAIL')}"
        )
    report = {
        "schema_version": 2,
        "solver_backend": "MetaHotspot C++ for full and reduced solves",
        "package": asdict(cfg),
        "experiment": asdict(exp),
        "thermal_cell_count": int(data.K_full.shape[0]),
        "physical_port_count": cfg.physical_ports,
        "selected_method": winner.name,
        "selected_rom_order": winner.rom_order,
        "selected_port_modes": winner.port_modes,
        "nominal_methods": [asdict(result) for result in nominal_results],
        "boundary_reuse": [asdict(result) for result in boundary_results],
        "model_build_and_validation_s": build_s,
        "common_preprocess_s": core.preprocess_s,
        "passed": bool(
            winner.passed and all((item.passed for item in boundary_results))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nReport: {args.output}")
    if args.strict and (not report["passed"]):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
