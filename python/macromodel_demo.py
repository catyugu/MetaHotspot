#!/usr/bin/env python3
"""Transient BCI-ROM benchmark using the MetaHotspot C++ solve path."""

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
except ImportError as exc:  # pragma: no cover - dependency diagnosis
    raise SystemExit("SciPy is required for the BCI-ROM benchmark") from exc

import metahotspot
from metahotspot.compiled import SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import (
    DtNModel,
    PortMap,
    PortPatch,
    solve as solve_macromodel,
)


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
    error_limit_K: float = 0.2
    port_energy_target: float = 0.9999
    port_growth_factor: float = 1.15
    port_growth_minimum: int = 8
    source_residual_modes: int = 560
    duration_s: float = 0.5
    time_step_s: float = 0.025
    nominal_h_W_m2K: float = 2500.0
    boundary_h_values_W_m2K: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    preprocess_budget_s: float = 30.0
    random_seed: int = 20260731
    report_path: Path = Path("results/bci_rom_final_results.json")


@dataclass
class AdaptationResult:
    port_modes: int
    source_energy_fraction: float
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
    detailed_ports_steady: PortMap
    detailed_ports_transient: PortMap
    macro_adiabatic: object
    macro_ports: PortMap
    macro_operators: tuple
    boundary_delta_Kii: sp.csc_matrix
    full_detailed_cells: np.ndarray
    full_macro_cells: np.ndarray


class MacroCore(NamedTuple):
    Kpp: sp.csc_matrix
    Kpi: sp.csc_matrix
    Kip: sp.csc_matrix
    Kii: sp.csc_matrix
    Cii: sp.csc_matrix
    fi: np.ndarray
    constraint_map: np.ndarray
    preprocess_s: float


class PortSpectrum(NamedTuple):
    basis: np.ndarray
    cumulative_energy: np.ndarray


class MethodBasis(NamedTuple):
    V: np.ndarray
    physical_basis: np.ndarray
    port_modes: int
    source_energy_fraction: float
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
        block,
        GeometryOp.ADD,
        "0",
        "0",
        f"{cfg.width_mm:.17g}",
        f"{cfg.height_mm:.17g}",
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
    volume_m3 = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1.0e-9
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


def _macro_z_vertices(cfg: PackageConfig) -> np.ndarray:
    vertices = [0.0]
    cursor = 0.0
    for thickness, cells in (
        (cfg.tim_mm, cfg.tim_cells),
        (cfg.spreader_mm, cfg.spreader_cells),
        (cfg.cold_plate_mm, cfg.cold_plate_cells),
    ):
        for _ in range(cells):
            cursor += thickness / cells
            vertices.append(cursor)
    return np.asarray(vertices, dtype=np.float64)


def build_macro_model(cfg: PackageConfig, convection_h_W_m2K: float | None = None):
    model = metahotspot.Model()
    model.set_settings(
        study=Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
    )
    model.set_mesh(
        _axis_vertices(cfg.width_mm, cfg.nx),
        _axis_vertices(cfg.height_mm, cfg.ny),
        _macro_z_vertices(cfg),
    )
    _add_materials(model)
    cold_plate_layer = model.add_layer(f"{cfg.cold_plate_mm:.17g}")
    cold_plate = model.add_block(cold_plate_layer, "aluminum")
    _add_full_rect(model, cold_plate, cfg)
    spreader_layer = model.add_layer(f"{cfg.spreader_mm:.17g}")
    spreader = model.add_block(spreader_layer, "copper")
    _add_full_rect(model, spreader, cfg)
    tim_layer = model.add_layer(f"{cfg.tim_mm:.17g}")
    tim = model.add_block(tim_layer, "tim")
    _add_full_rect(model, tim, cfg)
    model.set_default_neumann("0")
    if convection_h_W_m2K is not None:
        macro_height = cfg.tim_mm + cfg.spreader_mm + cfg.cold_plate_mm
        model.add_convection(
            f"{convection_h_W_m2K:.17g}",
            f"{cfg.ambient_K:.17g}",
            regions=[
                (Axis.Z, macro_height, 0.0, cfg.width_mm, 0.0, cfg.height_mm)
            ],
        )
    return model


def _interface_patches(cfg: PackageConfig, face: Face, coordinate_m: float) -> list[PortPatch]:
    dx = cfg.width_mm * 1.0e-3 / cfg.nx
    dy = cfg.height_mm * 1.0e-3 / cfg.ny
    return [
        PortPatch(
            face=int(face),
            coordinate=coordinate_m,
            rectangle=(ix * dx, (ix + 1) * dx, iy * dy, (iy + 1) * dy),
        )
        for ix in range(cfg.nx)
        for iy in range(cfg.ny)
    ]


def _assemble_macro_dtn(
    cfg: PackageConfig, h_W_m2K: float | None
) -> tuple[object, PortMap, tuple]:
    compiled = build_macro_model(cfg, h_W_m2K).compile()
    ports = PortMap(compiled, _interface_patches(cfg, Face.ZM, 0.0))
    return compiled, ports, ports.assemble()


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


def assemble_package(cfg: PackageConfig, exp: ExperimentConfig) -> PackageData:
    full = build_package_model(cfg, include_macro=True, study=Study.STEADY).compile()
    detailed_steady = build_package_model(
        cfg, include_macro=False, study=Study.STEADY
    ).compile()
    detailed_transient = build_package_model(
        cfg,
        include_macro=False,
        study=Study.TRANSIENT,
        duration_s=exp.duration_s,
        output_interval_s=exp.time_step_s,
    ).compile()

    detail_coordinate = (cfg.substrate_mm + cfg.bump_mm + cfg.die_mm) * 1.0e-3
    detailed_patches = _interface_patches(cfg, Face.ZP, detail_coordinate)
    detailed_ports_steady = PortMap(detailed_steady, detailed_patches)
    detailed_ports_transient = PortMap(detailed_transient, detailed_patches)

    macro_adiabatic, macro_ports, macro_operators = _assemble_macro_dtn(cfg, None)
    _, _, nominal_operators = _assemble_macro_dtn(cfg, exp.nominal_h_W_m2K)
    p = cfg.physical_ports
    boundary_delta = (
        nominal_operators.K[p:, p:] - macro_operators.K[p:, p:]
    ).tocsc()

    full_detailed_cells = _ordered_cells(full, 0, cfg.detailed_nz)
    full_macro_cells = _ordered_cells(full, cfg.detailed_nz, cfg.total_nz)
    if detailed_ports_steady.port_count != cfg.physical_ports:
        raise RuntimeError("detailed port-map size does not match physical port count")
    if macro_ports.port_count != cfg.physical_ports:
        raise RuntimeError("macro port-map size does not match physical port count")
    if macro_operators.K.shape[0] != cfg.physical_ports + macro_adiabatic.cell_count:
        raise RuntimeError("C++ DtN operator has an unexpected state dimension")

    return PackageData(
        full_adiabatic=full,
        detailed_steady=detailed_steady,
        detailed_transient=detailed_transient,
        detailed_ports_steady=detailed_ports_steady,
        detailed_ports_transient=detailed_ports_transient,
        macro_adiabatic=macro_adiabatic,
        macro_ports=macro_ports,
        macro_operators=macro_operators,
        boundary_delta_Kii=boundary_delta,
        full_detailed_cells=full_detailed_cells,
        full_macro_cells=full_macro_cells,
    )


def build_macro_core(data: PackageData) -> MacroCore:
    start = time.perf_counter()
    K, C, f = data.macro_operators
    p = data.macro_ports.port_count
    K = K.tocsc()
    C = C.tocsc()
    Kpp = K[:p, :p].tocsc()
    Kpi = K[:p, p:].tocsc()
    Kip = K[p:, :p].tocsc()
    Kii = K[p:, p:].tocsc()
    Cii = C[p:, p:].tocsc()
    fi = np.asarray(f[p:], dtype=np.float64)
    constraint_map = -spla.splu(Kii).solve(Kip.toarray())
    return MacroCore(
        Kpp=Kpp,
        Kpi=Kpi,
        Kip=Kip,
        Kii=Kii,
        Cii=Cii,
        fi=fi,
        constraint_map=constraint_map,
        preprocess_s=time.perf_counter() - start,
    )


def _complete_dct_basis(nx: int, ny: int) -> np.ndarray:
    modes = sorted(
        ((kx, ky) for kx in range(nx) for ky in range(ny)),
        key=lambda item: (item[0] ** 2 + item[1] ** 2, item[0], item[1]),
    )
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


def build_source_port_spectrum(cfg: PackageConfig) -> PortSpectrum:
    full_dct = _complete_dct_basis(cfg.nx, cfg.ny)
    x_centres = (np.arange(cfg.nx, dtype=np.float64) + 0.5) * (
        cfg.width_mm / cfg.nx
    )
    y_centres = (np.arange(cfg.ny, dtype=np.float64) + 0.5) * (
        cfg.height_mm / cfg.ny
    )
    source_maps = []
    for x0, y0 in _chiplet_positions(cfg):
        source_maps.append(
            np.asarray(
                [
                    1.0
                    if x0 <= x <= x0 + cfg.chiplet_width_mm
                    and y0 <= y <= y0 + cfg.chiplet_height_mm
                    else 0.0
                    for x in x_centres
                    for y in y_centres
                ],
                dtype=np.float64,
            )
        )
    maps = np.column_stack(source_maps)
    coefficient_energy = np.sum((full_dct.T @ maps) ** 2, axis=1)
    ranked = np.argsort(-coefficient_energy, kind="stable")
    ranked = np.asarray([0, *[int(index) for index in ranked if index != 0]])
    ranked_energy = coefficient_energy[ranked]
    total_energy = float(np.sum(ranked_energy))
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise RuntimeError("source port spectrum has zero or invalid energy")
    cumulative = np.cumsum(ranked_energy) / total_energy
    return PortSpectrum(
        np.ascontiguousarray(full_dct[:, ranked], dtype=np.float64),
        np.asarray(cumulative, dtype=np.float64),
    )


def _orthonormal_range(matrix: np.ndarray, tolerance: float = 1.0e-11) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    q, r, _ = scipy.linalg.qr(
        matrix, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    rank = int(
        np.count_nonzero(diagonal > tolerance * max(1.0, float(diagonal.max())))
    )
    return np.ascontiguousarray(q[:, :rank], dtype=np.float64)


def _randomized_left_basis(
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
        compressed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return np.ascontiguousarray(q @ u_hat[:, :rank], dtype=np.float64)


def build_source_aware_bci(
    data: PackageData,
    core: MacroCore,
    spectrum: PortSpectrum,
    port_modes: int,
    exp: ExperimentConfig,
) -> MethodBasis:
    start = time.perf_counter()
    if not 0 < port_modes <= spectrum.basis.shape[1]:
        raise ValueError("adaptive port mode count is outside the physical port range")

    phi = spectrum.basis[:, :port_modes]
    psi = core.constraint_map @ phi
    static_factor = spla.splu(core.Kii)
    boundary_modes = _orthonormal_range(
        -static_factor.solve(data.boundary_delta_Kii @ psi)
    )

    forcing = core.Kip @ phi
    shift_scale = 0.025 / exp.time_step_s
    shifts = shift_scale * np.asarray(
        (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
        dtype=np.float64,
    )
    residual_blocks = []
    for shift in shifts:
        factor = spla.splu((core.Kii + float(shift) * core.Cii).tocsc())
        residual = factor.solve(-forcing) - psi
        if boundary_modes.shape[1] > 0:
            residual -= boundary_modes @ (boundary_modes.T @ residual)
        residual_blocks.append(residual)

    dynamic_modes = _randomized_left_basis(
        np.hstack(residual_blocks), exp.source_residual_modes, seed=exp.random_seed
    )
    if boundary_modes.shape[1] > 0:
        dynamic_modes -= boundary_modes @ (boundary_modes.T @ dynamic_modes)
        dynamic_modes = _orthonormal_range(dynamic_modes)

    interior_only = np.hstack((boundary_modes, dynamic_modes))
    zeros = np.zeros((spectrum.basis.shape[0], interior_only.shape[1]), dtype=np.float64)
    physical = np.hstack((phi, zeros))
    V = np.vstack((physical, np.hstack((psi, interior_only))))
    return MethodBasis(
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(physical, dtype=np.float64),
        port_modes,
        float(spectrum.cumulative_energy[port_modes - 1]),
        core.preprocess_s + time.perf_counter() - start,
    )


def project_macro(
    data: PackageData,
    core: MacroCore,
    method: MethodBasis,
    cfg: PackageConfig,
    h_W_m2K: float,
) -> ReducedMacro:
    del data, core
    _, _, operators = _assemble_macro_dtn(cfg, h_W_m2K)
    K, C, f = operators
    V = method.V
    reduced_K = np.asarray(V.T @ (K @ V), dtype=np.float64)
    reduced_C = np.asarray(V.T @ (C @ V), dtype=np.float64)
    reduced_f = np.asarray(V.T @ f, dtype=np.float64)
    reduced_K = 0.5 * (reduced_K + reduced_K.T)
    reduced_C = 0.5 * (reduced_C + reduced_C.T)
    return ReducedMacro(
        sp.csc_matrix(reduced_K),
        sp.csc_matrix(reduced_C),
        reduced_f,
        V,
        method.physical_basis,
    )


def _solve_options(exp: ExperimentConfig, transient: bool) -> SolveOptions:
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1.0e-12,
        linear_max_iterations=5000,
        underrelaxation=1.0,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1.0e-11,
        nonlinear_absolute_tolerance=1.0e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1.0e-9,
        min_dt=exp.time_step_s if transient else 1.0e-12,
        max_dt=exp.time_step_s if transient else 1.0,
        fixed_dt=exp.time_step_s if transient else 1.0,
    )


def _macro_initial_coordinates(reduced: ReducedMacro, ambient_K: float) -> np.ndarray:
    target = np.full(reduced.V.shape[0], ambient_K, dtype=np.float64)
    coordinates, _, _, _ = scipy.linalg.lstsq(
        reduced.V,
        target,
        cond=1.0e-12,
        lapack_driver="gelsy",
        check_finite=False,
    )
    residual = float(np.max(np.abs(reduced.V @ coordinates - target)))
    if residual > 1.0e-8:
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
    ports = data.detailed_ports_transient if transient else data.detailed_ports_steady
    initial = np.concatenate(
        (
            np.full(compiled.cell_count, cfg.ambient_K, dtype=np.float64),
            _macro_initial_coordinates(reduced, cfg.ambient_K),
        )
    )
    model = DtNModel(
        operators=(reduced.K, reduced.C, reduced.f),
        port_basis=reduced.physical_basis,
    )
    return solve_macromodel(
        compiled, model, ports, initial, _solve_options(exp, transient=transient)
    )


def _recover_history(
    data: PackageData,
    reduced: ReducedMacro,
    reduced_history: np.ndarray,
    cfg: PackageConfig,
) -> np.ndarray:
    detailed_count = data.detailed_steady.cell_count
    recovered = np.empty(
        (reduced_history.shape[0], data.full_adiabatic.cell_count), dtype=np.float64
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
    steady_state = np.asarray(steady_solution.state, dtype=np.float64)[None, :]
    recovered_steady = _recover_history(data, reduced, steady_state, cfg)[0]
    steady_error = float(
        np.max(np.abs(recovered_steady - reference.steady_temperature))
    )

    transient_solution = _solve_reduced_cpp(data, reduced, cfg, exp, transient=True)
    times = np.asarray(transient_solution.history_times, dtype=np.float64)
    if times.shape != reference.transient_times.shape or not np.allclose(
        times, reference.transient_times, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(
            "full and reduced C++ solvers returned different output time grids"
        )
    reduced_history = np.asarray(transient_solution.state_history, dtype=np.float64)
    recovered_transient = _recover_history(data, reduced, reduced_history, cfg)
    transient_error = float(
        np.max(np.abs(recovered_transient - reference.transient_temperature))
    )
    return steady_error, transient_error, len(times), reduced


def _initial_port_count(spectrum: PortSpectrum, target: float) -> int:
    count = int(np.searchsorted(spectrum.cumulative_energy, target, side="left")) + 1
    return min(max(1, count), spectrum.basis.shape[1])


def _next_port_count(current: int, total: int, exp: ExperimentConfig) -> int:
    if current >= total:
        return total
    grown = max(
        current + exp.port_growth_minimum,
        int(math.ceil(current * exp.port_growth_factor)),
    )
    return min(total, grown)


def adapt_port_space(
    data: PackageData,
    core: MacroCore,
    spectrum: PortSpectrum,
    reference: CppReference,
    cfg: PackageConfig,
    exp: ExperimentConfig,
) -> tuple[MethodBasis, AdaptationResult, list[AdaptationResult]]:
    count = _initial_port_count(spectrum, exp.port_energy_target)
    attempts: list[AdaptationResult] = []
    selected_basis: MethodBasis | None = None

    while True:
        method = build_source_aware_bci(data, core, spectrum, count, exp)
        steady, transient, records, _ = evaluate_method(
            data, core, method, reference, cfg, exp, exp.nominal_h_W_m2K
        )
        passed = (
            steady <= exp.error_limit_K
            and transient <= exp.error_limit_K
            and method.preprocess_s <= exp.preprocess_budget_s
        )
        result = AdaptationResult(
            port_modes=count,
            source_energy_fraction=method.source_energy_fraction,
            rom_order=method.V.shape[1],
            preprocess_s=method.preprocess_s,
            steady_error_K=steady,
            transient_error_K=transient,
            transient_records=records,
            passed=passed,
        )
        attempts.append(result)
        print(
            f"  port={count:4d}/{cfg.physical_ports:<4d} "
            f"source-energy={method.source_energy_fraction:.8f} "
            f"order={method.V.shape[1]:4d} prep={method.preprocess_s:7.3f}s "
            f"steady={steady:9.5f}K transient={transient:9.5f}K "
            f"{'PASS' if passed else 'EXPAND'}"
        )
        if passed:
            selected_basis = method
            break
        next_count = _next_port_count(count, cfg.physical_ports, exp)
        if next_count == count:
            selected_basis = method
            break
        count = next_count

    assert selected_basis is not None
    return selected_basis, attempts[-1], attempts


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="run the reduced smoke case")
    mode.add_argument("--strict", action="store_true", help="run the full release case")
    return parser.parse_args(argv)


def _configs_from_args(args: argparse.Namespace) -> tuple[PackageConfig, ExperimentConfig]:
    if args.quick:
        return (
            PackageConfig(nx=16, ny=16, bump_rows=6, bump_columns=6),
            ExperimentConfig(
                source_residual_modes=240,
                duration_s=0.2,
                time_step_s=0.025,
            ),
        )
    return PackageConfig(), ExperimentConfig()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg, exp = _configs_from_args(args)

    print("=" * 88)
    print("Transient BCI-ROM package benchmark — adaptive source-aware C++ path")
    print("=" * 88)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.total_nz} = "
        f"{cfg.nx * cfg.ny * cfg.total_nz:,} thermal cells"
    )
    print(f"Physical interface ports: {cfg.physical_ports}; port rank: adaptive")

    build_start = time.perf_counter()
    data = assemble_package(cfg, exp)
    build_s = time.perf_counter() - build_start
    print(f"Model build + C++ assembly + interface validation: {build_s:.3f}s")

    core = build_macro_core(data)
    spectrum = build_source_port_spectrum(cfg)
    initial_count = _initial_port_count(spectrum, exp.port_energy_target)
    print(
        f"Common isolated-macro preprocessing: {core.preprocess_s:.3f}s; "
        f"source-spectrum initial rank={initial_count} at "
        f"{exp.port_energy_target:.5f} cumulative energy"
    )
    print("Building nominal full-order C++ references...")
    nominal_reference = build_cpp_reference(cfg, exp, exp.nominal_h_W_m2K)

    print("\nAdaptive source_aware_bci_krylov search:")
    winning_method, winner, attempts = adapt_port_space(
        data, core, spectrum, nominal_reference, cfg, exp
    )
    print(
        f"\nSelected source_aware_bci_krylov: port={winner.port_modes}/"
        f"{cfg.physical_ports}, order={winner.rom_order}"
    )

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
            f"  h={h_value:8.1f} W/(m2 K): steady={steady:9.5f} K, "
            f"transient={transient:9.5f} K {'PASS' if passed else 'FAIL'}"
        )

    report = {
        "schema_version": 3,
        "mode": "quick" if args.quick else "strict",
        "solver_backend": "MetaHotspot C++ for full and reduced solves",
        "reduction_method": "source_aware_bci_krylov",
        "package": asdict(cfg),
        "experiment": {
            **asdict(exp),
            "report_path": str(exp.report_path),
        },
        "thermal_cell_count": int(data.full_adiabatic.cell_count),
        "physical_port_count": cfg.physical_ports,
        "selected_port_modes": winner.port_modes,
        "selected_source_energy_fraction": winner.source_energy_fraction,
        "selected_rom_order": winner.rom_order,
        "adaptation_attempts": [asdict(item) for item in attempts],
        "boundary_reuse": [asdict(item) for item in boundary_results],
        "model_build_and_validation_s": build_s,
        "common_preprocess_s": core.preprocess_s,
        "passed": bool(winner.passed and all(item.passed for item in boundary_results)),
    }
    exp.report_path.parent.mkdir(parents=True, exist_ok=True)
    exp.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nReport: {exp.report_path}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
