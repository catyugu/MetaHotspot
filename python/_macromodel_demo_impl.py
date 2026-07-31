#!/usr/bin/env python3
"""Transient boundary-condition-independent thermal ROM research benchmark.

This is the final, single-script experiment for the MetaHotspot macromodel
stack.  It deliberately replaces the old steady-only Guyan demonstration.

The script performs, in order:

1. Build a representative heterogeneous package with the public Python API.
2. Assemble an isolated, boundary-condition-independent upper-package macro.
3. Validate the cell/face interface contract used by the C/C++ macromodel API.
4. Compare four reduction strategies:
   - geometric DCT port basis + Guyan constraint modes,
   - transfer/Steklov port basis + Guyan constraint modes,
   - transfer/Steklov port basis + Craig-Bampton fixed-interface modes,
   - transfer/Steklov port basis + rational Krylov interior modes.
5. Require both steady and transient maximum absolute temperature errors to be
   at most the configured threshold (0.1 K by default).
6. Reuse the same extracted basis under several convection boundary operators.
7. Call ``metahotspot.macromodel.solve`` and compare the C/C++ end-side result
   against an independently assembled Python BDF1 system.
8. Write a machine-readable JSON report.

The macro basis is extracted from an adiabatic isolated component.  External
convection is represented as an affine operator and projected only after basis
extraction, so changing the boundary condition does not trigger re-extraction.
Physical interface face temperatures are retained through a reduced port basis;
the first macro cell layer is connected to those face temperatures by the
macro-side half conductance.  This matches the C++ modal-port coupling contract,
which connects detailed FVM cell centres to interface face temperatures by the
detailed-side half conductance.

Typical release run (after building/installing the Python package and DLLs):

    python python/macromodel_demo.py --strict

A smaller development smoke test is available with ``--quick``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np

try:
    import scipy.linalg
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:  # pragma: no cover - dependency diagnosis
    raise SystemExit("SciPy is required for the BCI-ROM experiment") from exc

import metahotspot
from metahotspot.compiled import SolveOptions
from metahotspot.enums import Face, GeometryOp, LengthUnit, Study


# ---------------------------------------------------------------------------
# Configuration and result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageConfig:
    nx: int = 24
    ny: int = 24
    width_mm: float = 40.0
    height_mm: float = 40.0
    ambient_K: float = 300.0
    substrate_mm: float = 1.20
    bump_mm: float = 0.24
    die_mm: float = 0.60
    tim_mm: float = 0.18
    spreader_mm: float = 1.20
    cold_plate_mm: float = 1.50
    substrate_cells: int = 4
    bump_cells: int = 2
    die_cells: int = 3
    tim_cells: int = 1
    spreader_cells: int = 3
    cold_plate_cells: int = 3
    bump_rows: int = 8
    bump_columns: int = 8
    bump_width_mm: float = 0.90
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


@dataclass(frozen=True)
class ExperimentConfig:
    error_limit_K: float = 0.1
    port_modes: int = 441
    interior_modes: int = 224
    rational_modes: int = 224
    rational_block: int = 16
    rational_shifts: int = 6
    duration_s: float = 0.50
    time_step_s: float = 0.025
    cpp_duration_s: float = 0.10
    nominal_h_W_m2K: float = 2500.0
    boundary_h_values_W_m2K: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    preprocess_budget_s: float = 30.0
    random_seed: int = 20260731


@dataclass
class MethodResult:
    name: str
    port_modes: int
    rom_order: int
    preprocess_s: float
    steady_error_K: float
    transient_error_K: float
    transient_steps: int
    passed: bool


@dataclass
class BoundaryResult:
    h_W_m2K: float
    steady_error_K: float
    transient_error_K: float
    passed: bool


class MethodBasis(NamedTuple):
    name: str
    V: np.ndarray
    physical_basis: np.ndarray
    preprocess_s: float


class PackageData(NamedTuple):
    full: object
    detailed: object
    K_full: sp.csc_matrix
    C_full: sp.csc_matrix
    f_full: np.ndarray
    K_detailed: sp.csc_matrix
    C_detailed: sp.csc_matrix
    f_detailed: np.ndarray
    full_detailed_cells: np.ndarray
    full_macro_cells: np.ndarray
    detailed_interface_cells: np.ndarray
    full_macro_interface_cells: np.ndarray
    macro_interface_local_cells: np.ndarray
    full_top_cells: np.ndarray
    macro_top_local_cells: np.ndarray
    detailed_half_conductance: np.ndarray
    macro_half_conductance: np.ndarray
    interface_series_conductance: np.ndarray
    K_macro_cells: sp.csc_matrix
    C_macro_cells: sp.csc_matrix
    f_macro_cells: np.ndarray
    top_face_area_m2: float


# ---------------------------------------------------------------------------
# Package construction through the repository's public model API
# ---------------------------------------------------------------------------


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
    z = 0.0
    for thickness, cells in layers:
        dz = thickness / cells
        for _ in range(cells):
            z += dz
            vertices.append(z)
    return np.asarray(vertices, dtype=np.float64)


def _add_full_rect(model, block: int, cfg: PackageConfig) -> None:
    model.add_rect(
        block,
        GeometryOp.ADD,
        "0",
        "0",
        f"{cfg.width_mm:.17g}",
        f"{cfg.height_mm:.17g}",
    )


def _add_materials(model) -> None:
    # kx, ky, kz [W/(m K)], rho [kg/m3], c [J/(kg K)]
    model.add_material("organic", "0.65", "0.65", "0.55", "1900", "1100")
    model.add_material("underfill", "0.80", "0.80", "0.80", "1550", "1000")
    model.add_material("copper", "390", "390", "390", "8960", "385")
    model.add_material("mold", "0.85", "0.85", "0.75", "1850", "1000")
    model.add_material("silicon", "130", "130", "115", "2330", "700")
    model.add_material("tim", "4.0", "4.0", "3.0", "2500", "900")
    model.add_material("aluminum", "180", "180", "180", "2700", "900")


def _chiplet_heat_source(cfg: PackageConfig) -> str:
    volume_m3 = (
        cfg.chiplet_width_mm
        * cfg.chiplet_height_mm
        * cfg.die_mm
        * 1.0e-9
    )
    return f"{cfg.chiplet_power_W / volume_m3:.17g}"


def build_package_model(
    cfg: PackageConfig,
    *,
    include_macro: bool,
    study: Study = Study.STEADY,
    duration_s: float = 0.0,
    output_interval_s: float = 0.0,
):
    """Build a heterogeneous package using only the public scripting API."""
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

    # Organic substrate.
    substrate_layer = model.add_layer(f"{cfg.substrate_mm:.17g}")
    substrate = model.add_block(substrate_layer, "organic")
    _add_full_rect(model, substrate, cfg)

    # Underfill with an 8x8 copper bump array.  Later blocks override the
    # underfill background in the cells covered by a bump.
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

    # Mold background with four active silicon chiplets.
    die_layer = model.add_layer(f"{cfg.die_mm:.17g}")
    mold = model.add_block(die_layer, "mold")
    _add_full_rect(model, mold, cfg)
    q_chiplet = _chiplet_heat_source(cfg)
    margin_x = 5.0
    margin_y = 5.0
    positions = (
        (margin_x, margin_y),
        (cfg.width_mm - margin_x - cfg.chiplet_width_mm, margin_y),
        (margin_x, cfg.height_mm - margin_y - cfg.chiplet_height_mm),
        (
            cfg.width_mm - margin_x - cfg.chiplet_width_mm,
            cfg.height_mm - margin_y - cfg.chiplet_height_mm,
        ),
    )
    for x, y in positions:
        chiplet = model.add_block(die_layer, "silicon", heat_source=q_chiplet)
        model.add_rect(
            chiplet,
            GeometryOp.ADD,
            f"{x:.17g}",
            f"{y:.17g}",
            f"{cfg.chiplet_width_mm:.17g}",
            f"{cfg.chiplet_height_mm:.17g}",
        )

    if include_macro:
        tim_layer = model.add_layer(f"{cfg.tim_mm:.17g}")
        tim = model.add_block(tim_layer, "tim")
        _add_full_rect(model, tim, cfg)

        spreader_layer = model.add_layer(f"{cfg.spreader_mm:.17g}")
        spreader = model.add_block(spreader_layer, "copper")
        _add_full_rect(model, spreader, cfg)

        cold_plate_layer = model.add_layer(f"{cfg.cold_plate_mm:.17g}")
        cold_plate = model.add_block(cold_plate_layer, "aluminum")
        _add_full_rect(model, cold_plate, cfg)

    # Adiabatic isolation is essential: external boundary operators are added
    # after extraction and may be changed without recomputing the basis.
    model.set_default_neumann("0")
    return model


# ---------------------------------------------------------------------------
# Grid ordering and interface-contract validation
# ---------------------------------------------------------------------------


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
    if matrix.nnz == 0:
        return 0.0
    return float(np.max(np.abs(matrix.data)))


def assemble_package(cfg: PackageConfig, exp: ExperimentConfig) -> PackageData:
    full_model = build_package_model(cfg, include_macro=True)
    detailed_model = build_package_model(
        cfg,
        include_macro=False,
        study=Study.TRANSIENT,
        duration_s=exp.cpp_duration_s,
        output_interval_s=exp.time_step_s,
    )
    full = full_model.compile()
    detailed = detailed_model.compile()

    K_full, C_full, f_full = full.assemble()
    K_detailed, C_detailed, f_detailed = detailed.assemble()
    K_full = K_full.tocsc()
    C_full = C_full.tocsc()
    K_detailed = K_detailed.tocsc()
    C_detailed = C_detailed.tocsc()

    full_detailed_cells = _ordered_cells(full, 0, cfg.detailed_nz)
    full_macro_cells = _ordered_cells(full, cfg.detailed_nz, cfg.total_nz)
    detailed_order = _ordered_cells(detailed, 0, cfg.detailed_nz)
    if not np.array_equal(detailed_order, np.arange(detailed.cell_count)):
        raise RuntimeError("unexpected compact-cell ordering in detailed model")

    detailed_interface_cells = np.asarray(
        [
            _grid_cell(detailed, ix, iy, cfg.detailed_nz - 1)
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

    # The monolithic FVM reference contains a series conductance between the two
    # adjacent cell centres.  The C++ modal-port contract instead introduces a
    # face temperature and attaches each side by its own half conductance.
    g_series = np.asarray(
        [
            -float(K_full[dcell, mcell])
            for dcell, mcell in zip(
                full_detailed_interface, full_macro_interface, strict=True
            )
        ],
        dtype=np.float64,
    )
    g_d = detailed.half_conductance(
        detailed_interface_cells, Face.ZP, cfg.ambient_K, 0.0
    )
    g_m = full.half_conductance(
        full_macro_interface, Face.ZM, cfg.ambient_K, 0.0
    )
    expected_series = g_d * g_m / (g_d + g_m)
    series_error = float(np.max(np.abs(g_series - expected_series)))
    series_scale = max(1.0, float(np.max(np.abs(expected_series))))
    if series_error > 1.0e-9 * series_scale:
        raise RuntimeError(
            "C/C++ interface contract mismatch: monolithic and half-conductance "
            f"series operators differ by {series_error:.6e}"
        )

    # Remove the monolithic interface contribution from the macro cell block.
    K_macro_cells = K_full[full_macro_cells, :][:, full_macro_cells].tocsc()
    interface_diag = np.zeros(full_macro_cells.size, dtype=np.float64)
    interface_diag[macro_interface_local] = g_series
    K_macro_cells = K_macro_cells - sp.diags(interface_diag, format="csc")
    C_macro_cells = C_full[full_macro_cells, :][:, full_macro_cells].tocsc()
    f_macro_cells = np.asarray(f_full[full_macro_cells], dtype=np.float64)

    # Verify that independently compiling the detailed component produces the
    # same isolated operator as stripping the monolithic interface.
    K_detail_from_full = K_full[full_detailed_cells, :][:, full_detailed_cells].tocsc()
    detail_interface_diag = np.zeros(full_detailed_cells.size, dtype=np.float64)
    full_detail_position = {
        int(cell): pos for pos, cell in enumerate(full_detailed_cells)
    }
    for cell, conductance in zip(
        full_detailed_interface, g_series, strict=True
    ):
        detail_interface_diag[full_detail_position[int(cell)]] = conductance
    K_detail_from_full = K_detail_from_full - sp.diags(
        detail_interface_diag, format="csc"
    )
    operator_error = _max_sparse_abs(K_detail_from_full - K_detailed)
    operator_scale = max(1.0, _max_sparse_abs(K_detailed))
    if operator_error > 1.0e-9 * operator_scale:
        raise RuntimeError(
            "independent detailed compile does not match the stripped full model: "
            f"max operator difference={operator_error:.6e}"
        )

    dx_m = cfg.width_mm * 1.0e-3 / cfg.nx
    dy_m = cfg.height_mm * 1.0e-3 / cfg.ny
    return PackageData(
        full=full,
        detailed=detailed,
        K_full=K_full,
        C_full=C_full,
        f_full=np.asarray(f_full, dtype=np.float64),
        K_detailed=K_detailed,
        C_detailed=C_detailed,
        f_detailed=np.asarray(f_detailed, dtype=np.float64),
        full_detailed_cells=full_detailed_cells,
        full_macro_cells=full_macro_cells,
        detailed_interface_cells=detailed_interface_cells,
        full_macro_interface_cells=full_macro_interface,
        macro_interface_local_cells=macro_interface_local,
        full_top_cells=full_top_cells,
        macro_top_local_cells=macro_top_local,
        detailed_half_conductance=np.asarray(g_d, dtype=np.float64),
        macro_half_conductance=np.asarray(g_m, dtype=np.float64),
        interface_series_conductance=g_series,
        K_macro_cells=K_macro_cells,
        C_macro_cells=C_macro_cells,
        f_macro_cells=f_macro_cells,
        top_face_area_m2=dx_m * dy_m,
    )


# ---------------------------------------------------------------------------
# Boundary-face augmented macro and port bases
# ---------------------------------------------------------------------------


class MacroCore(NamedTuple):
    Kpp: sp.csc_matrix
    Kpi: sp.csc_matrix
    Kip: sp.csc_matrix
    Kii: sp.csc_matrix
    Cii: sp.csc_matrix
    fi: np.ndarray
    constraint_map: np.ndarray
    steklov: np.ndarray
    transfer_vectors: np.ndarray
    transfer_values: np.ndarray
    fixed_interface_modes: np.ndarray
    fixed_interface_values: np.ndarray
    preprocess_s: float


def build_macro_core(data: PackageData, exp: ExperimentConfig) -> MacroCore:
    start = time.perf_counter()
    n_ports = data.detailed_interface_cells.size
    n_macro = data.full_macro_cells.size
    rows = data.macro_interface_local_cells
    cols = np.arange(n_ports, dtype=np.int64)

    # p = physical face-temperature ports, i = macro FVM cell temperatures.
    Kpp = sp.diags(data.macro_half_conductance, format="csc")
    Kip = sp.coo_matrix(
        (-data.macro_half_conductance, (rows, cols)),
        shape=(n_macro, n_ports),
    ).tocsc()
    Kpi = Kip.T.tocsc()
    interior_diag = np.zeros(n_macro, dtype=np.float64)
    interior_diag[rows] = data.macro_half_conductance
    Kii = data.K_macro_cells + sp.diags(interior_diag, format="csc")
    Cii = data.C_macro_cells.tocsc()

    factor = spla.splu(Kii)
    # W = -Kii^{-1} Kip.  Solving all interface columns in one block is faster
    # than one Python-level solve per port.
    W = -factor.solve(Kip.toarray())
    S = Kpp.toarray() + np.asarray(Kpi @ W)
    S = 0.5 * (S + S.T)
    transfer_values, transfer_vectors = scipy.linalg.eigh(
        S, check_finite=False, driver="evr"
    )
    order = np.argsort(transfer_values)
    transfer_values = transfer_values[order]
    transfer_vectors = transfer_vectors[:, order]

    k_modes = min(exp.interior_modes, n_macro - 2)
    if k_modes <= 0:
        fixed_values = np.empty(0, dtype=np.float64)
        fixed_modes = np.empty((n_macro, 0), dtype=np.float64)
    else:
        fixed_values, fixed_modes = spla.eigsh(
            Kii,
            k=k_modes,
            M=Cii,
            sigma=0.0,
            which="LM",
            tol=1.0e-8,
        )
        mode_order = np.argsort(fixed_values)
        fixed_values = np.asarray(fixed_values[mode_order], dtype=np.float64)
        fixed_modes = np.asarray(fixed_modes[:, mode_order], dtype=np.float64)

    return MacroCore(
        Kpp=Kpp,
        Kpi=Kpi,
        Kip=Kip,
        Kii=Kii,
        Cii=Cii,
        fi=data.f_macro_cells,
        constraint_map=W,
        steklov=S,
        transfer_vectors=transfer_vectors,
        transfer_values=transfer_values,
        fixed_interface_modes=fixed_modes,
        fixed_interface_values=fixed_values,
        preprocess_s=time.perf_counter() - start,
    )


def dct_port_basis(nx: int, ny: int, count: int) -> np.ndarray:
    """Return the lowest spatial-frequency orthonormal 2-D cosine modes."""
    modes: list[tuple[int, int]] = sorted(
        ((kx, ky) for kx in range(nx) for ky in range(ny)),
        key=lambda item: (item[0] * item[0] + item[1] * item[1], item[0], item[1]),
    )
    modes = modes[:count]
    x = np.arange(nx, dtype=np.float64) + 0.5
    y = np.arange(ny, dtype=np.float64) + 0.5
    columns = []
    for kx, ky in modes:
        vx = np.cos(math.pi * kx * x / nx)
        vy = np.cos(math.pi * ky * y / ny)
        vx *= math.sqrt((1.0 if kx == 0 else 2.0) / nx)
        vy *= math.sqrt((1.0 if ky == 0 else 2.0) / ny)
        columns.append(np.kron(vx, vy))
    return np.column_stack(columns)


def _build_guyan(name: str, phi: np.ndarray, core: MacroCore, elapsed: float) -> MethodBasis:
    psi = core.constraint_map @ phi
    V = np.vstack((phi, psi))
    return MethodBasis(name, V, phi, elapsed)


def _build_craig_bampton(
    phi: np.ndarray, core: MacroCore, elapsed: float
) -> MethodBasis:
    psi = core.constraint_map @ phi
    modes = core.fixed_interface_modes
    zeros = np.zeros((phi.shape[0], modes.shape[1]), dtype=np.float64)
    V = np.block([[phi, zeros], [psi, modes]])
    physical = np.hstack((phi, zeros))
    return MethodBasis("transfer_craig_bampton", V, physical, elapsed)


def _orthonormal_columns(matrix: np.ndarray, limit: int) -> np.ndarray:
    if matrix.size == 0 or limit <= 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    keep = diagonal > max(1.0, diagonal.max()) * 1.0e-11
    return q[:, keep][:, :limit]


def _build_rational_krylov(
    phi: np.ndarray,
    core: MacroCore,
    exp: ExperimentConfig,
    elapsed: float,
) -> MethodBasis:
    rng = np.random.default_rng(exp.random_seed)
    r_port = phi.shape[1]
    block = min(exp.rational_block, r_port)
    shifts = np.logspace(
        math.log10(1.0 / max(exp.duration_s, exp.time_step_s)),
        math.log10(1.0 / exp.time_step_s),
        exp.rational_shifts,
    )
    snapshots = []
    port_forcing = core.Kip @ phi
    for shift in shifts:
        directions = rng.standard_normal((r_port, block))
        directions, _ = np.linalg.qr(directions, mode="reduced")
        factor = spla.splu((core.Kii + shift * core.Cii).tocsc())
        snapshots.append(factor.solve(-(port_forcing @ directions)))
    q = _orthonormal_columns(np.hstack(snapshots), exp.rational_modes)
    psi = core.constraint_map @ phi
    zeros = np.zeros((phi.shape[0], q.shape[1]), dtype=np.float64)
    V = np.block([[phi, zeros], [psi, q]])
    physical = np.hstack((phi, zeros))
    return MethodBasis("transfer_rational_krylov", V, physical, elapsed)


def build_method_bases(
    cfg: PackageConfig, core: MacroCore, exp: ExperimentConfig
) -> list[MethodBasis]:
    n_ports = cfg.physical_ports
    port_modes = min(exp.port_modes, n_ports)
    if port_modes >= n_ports:
        raise ValueError(
            "port reduction is required: --port-modes must be smaller than the "
            f"physical port count ({n_ports})"
        )

    methods: list[MethodBasis] = []

    start = time.perf_counter()
    phi_dct = dct_port_basis(cfg.nx, cfg.ny, port_modes)
    methods.append(
        _build_guyan(
            "dct_guyan",
            phi_dct,
            core,
            core.preprocess_s + time.perf_counter() - start,
        )
    )

    phi_transfer = core.transfer_vectors[:, :port_modes]
    methods.append(
        _build_guyan(
            "transfer_guyan", phi_transfer, core, core.preprocess_s
        )
    )

    start = time.perf_counter()
    methods.append(
        _build_craig_bampton(
            phi_transfer, core, core.preprocess_s + time.perf_counter() - start
        )
    )

    start = time.perf_counter()
    methods.append(
        _build_rational_krylov(
            phi_transfer,
            core,
            exp,
            core.preprocess_s + time.perf_counter() - start,
        )
    )
    return methods


# ---------------------------------------------------------------------------
# Projection, coupled assembly, steady and transient reference solves
# ---------------------------------------------------------------------------


class ReducedMacro(NamedTuple):
    K: sp.csc_matrix
    C: sp.csc_matrix
    f_heat: np.ndarray
    f_boundary: np.ndarray
    V: np.ndarray
    physical_basis: np.ndarray


def project_macro(
    data: PackageData,
    core: MacroCore,
    method: MethodBasis,
    cfg: PackageConfig,
    h_W_m2K: float,
) -> ReducedMacro:
    V = method.V
    np_ = cfg.physical_ports
    Vp = V[:np_, :]
    Vi = V[np_:, :]

    KVi = core.Kii @ Vi
    K = np.asarray(Vp.T @ (core.Kpp @ Vp), dtype=np.float64)
    K += np.asarray(Vp.T @ (core.Kpi @ Vi), dtype=np.float64)
    K += np.asarray(Vi.T @ (core.Kip @ Vp), dtype=np.float64)
    K += np.asarray(Vi.T @ KVi, dtype=np.float64)
    C = np.asarray(Vi.T @ (core.Cii @ Vi), dtype=np.float64)
    f_heat = np.asarray(Vi.T @ core.fi, dtype=np.float64)

    weights = np.full(
        data.macro_top_local_cells.size,
        h_W_m2K * data.top_face_area_m2,
        dtype=np.float64,
    )
    Vtop = Vi[data.macro_top_local_cells, :]
    K += Vtop.T @ (weights[:, None] * Vtop)
    f_boundary = Vtop.T @ (weights * cfg.ambient_K)

    # Numerical projection can introduce tiny asymmetry.  Symmetrising avoids
    # method-dependent sparse factorisation behaviour.
    K = 0.5 * (K + K.T)
    C = 0.5 * (C + C.T)
    return ReducedMacro(
        K=sp.csc_matrix(K),
        C=sp.csc_matrix(C),
        f_heat=f_heat,
        f_boundary=np.asarray(f_boundary, dtype=np.float64),
        V=V,
        physical_basis=method.physical_basis,
    )


def full_boundary_system(
    data: PackageData, cfg: PackageConfig, h_W_m2K: float
) -> tuple[sp.csc_matrix, np.ndarray]:
    diagonal = np.zeros(data.K_full.shape[0], dtype=np.float64)
    diagonal[data.full_top_cells] = h_W_m2K * data.top_face_area_m2
    rhs = np.zeros(data.K_full.shape[0], dtype=np.float64)
    rhs[data.full_top_cells] = (
        h_W_m2K * data.top_face_area_m2 * cfg.ambient_K
    )
    return data.K_full + sp.diags(diagonal, format="csc"), rhs


def coupled_reduced_system(
    data: PackageData, reduced: ReducedMacro
) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    nd = data.K_detailed.shape[0]
    nr = reduced.K.shape[0]
    np_ = data.detailed_interface_cells.size
    B = reduced.physical_basis
    if B.shape != (np_, nr):
        raise RuntimeError(
            f"physical basis has shape {B.shape}, expected {(np_, nr)}"
        )

    interface_diag = np.zeros(nd, dtype=np.float64)
    interface_diag[data.detailed_interface_cells] = data.detailed_half_conductance
    Kdd = data.K_detailed + sp.diags(interface_diag, format="csc")

    cross_values = -data.detailed_half_conductance[:, None] * B
    rows = np.repeat(data.detailed_interface_cells, nr)
    cols = np.tile(np.arange(nr, dtype=np.int64), np_)
    Kdr = sp.coo_matrix(
        (cross_values.ravel(), (rows, cols)), shape=(nd, nr)
    ).tocsc()
    Krr = reduced.K + sp.csc_matrix(
        B.T @ (data.detailed_half_conductance[:, None] * B)
    )
    K = sp.bmat([[Kdd, Kdr], [Kdr.T, Krr]], format="csc")
    C = sp.block_diag((data.C_detailed, reduced.C), format="csc")
    return K, C


def _macro_initial_coordinates(
    reduced: ReducedMacro, ambient_K: float
) -> np.ndarray:
    target = np.full(reduced.V.shape[0], ambient_K, dtype=np.float64)
    gram = reduced.V.T @ reduced.V
    rhs = reduced.V.T @ target
    return scipy.linalg.solve(
        gram, rhs, assume_a="sym", check_finite=False
    )


def _recover_cell_state(
    data: PackageData,
    reduced: ReducedMacro,
    reduced_state: np.ndarray,
) -> np.ndarray:
    nd = data.K_detailed.shape[0]
    result = np.empty(data.K_full.shape[0], dtype=np.float64)
    result[data.full_detailed_cells] = reduced_state[:nd]
    augmented_macro = reduced.V @ reduced_state[nd:]
    np_ = data.detailed_interface_cells.size
    result[data.full_macro_cells] = augmented_macro[np_:]
    return result


def power_scale(time_s: float, duration_s: float) -> float:
    """A nontrivial deterministic transient power waveform."""
    if time_s <= 0.0:
        return 0.0
    x = time_s / duration_s
    if x < 0.20:
        return x / 0.20
    if x < 0.55:
        return 1.0
    if x < 0.75:
        return 0.55
    return 0.85 + 0.15 * math.sin(8.0 * math.pi * x)


def solve_bdf1(
    K: sp.csc_matrix,
    C: sp.csc_matrix,
    initial: np.ndarray,
    rhs_at: Callable[[float], np.ndarray],
    duration_s: float,
    dt_s: float,
) -> Iterable[tuple[float, np.ndarray]]:
    steps = int(round(duration_s / dt_s))
    if not math.isclose(steps * dt_s, duration_s, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("duration must be an integer multiple of time step")
    mass_over_dt = C * (1.0 / dt_s)
    factor = spla.splu((K + mass_over_dt).tocsc())
    state = np.asarray(initial, dtype=np.float64).copy()
    for step in range(1, steps + 1):
        t = step * dt_s
        state = factor.solve(rhs_at(t) + mass_over_dt @ state)
        yield t, state


def evaluate_method(
    data: PackageData,
    core: MacroCore,
    method: MethodBasis,
    cfg: PackageConfig,
    exp: ExperimentConfig,
    h_W_m2K: float,
) -> tuple[float, float, int, ReducedMacro]:
    reduced = project_macro(data, core, method, cfg, h_W_m2K)
    K_full_bc, f_full_boundary = full_boundary_system(data, cfg, h_W_m2K)
    K_red, C_red = coupled_reduced_system(data, reduced)

    full_steady = spla.spsolve(
        K_full_bc, data.f_full + f_full_boundary
    )
    red_rhs_steady = np.concatenate(
        (data.f_detailed, reduced.f_heat + reduced.f_boundary)
    )
    red_steady = spla.spsolve(K_red, red_rhs_steady)
    recovered_steady = _recover_cell_state(data, reduced, red_steady)
    steady_error = float(np.max(np.abs(recovered_steady - full_steady)))

    full_initial = np.full(
        data.K_full.shape[0], cfg.ambient_K, dtype=np.float64
    )
    z_initial = _macro_initial_coordinates(reduced, cfg.ambient_K)
    red_initial = np.concatenate(
        (
            np.full(data.K_detailed.shape[0], cfg.ambient_K, dtype=np.float64),
            z_initial,
        )
    )

    full_history = solve_bdf1(
        K_full_bc,
        data.C_full,
        full_initial,
        lambda t: power_scale(t, exp.duration_s) * data.f_full
        + f_full_boundary,
        exp.duration_s,
        exp.time_step_s,
    )
    red_history = solve_bdf1(
        K_red,
        C_red,
        red_initial,
        lambda t: np.concatenate(
            (
                power_scale(t, exp.duration_s) * data.f_detailed,
                power_scale(t, exp.duration_s) * reduced.f_heat
                + reduced.f_boundary,
            )
        ),
        exp.duration_s,
        exp.time_step_s,
    )

    transient_error = 0.0
    steps = 0
    for (time_full, state_full), (time_red, state_red) in zip(
        full_history, red_history, strict=True
    ):
        if not math.isclose(time_full, time_red, abs_tol=1.0e-12):
            raise RuntimeError("reference and reduced time grids differ")
        recovered = _recover_cell_state(data, reduced, state_red)
        transient_error = max(
            transient_error, float(np.max(np.abs(recovered - state_full)))
        )
        steps += 1
    return steady_error, transient_error, steps, reduced


# ---------------------------------------------------------------------------
# C/C++ end-side contract and numerical consistency check
# ---------------------------------------------------------------------------


def run_cpp_consistency_check(
    data: PackageData,
    reduced: ReducedMacro,
    cfg: PackageConfig,
    exp: ExperimentConfig,
) -> float:
    """Compare the plugin C/C++ solve with independent Python BDF1 assembly."""
    try:
        from metahotspot.macromodel import PortCoupling, PortModel, solve
    except Exception as exc:  # pragma: no cover - runtime installation issue
        raise RuntimeError(
            "the macromodel extension is unavailable; build/install mhs_c_api "
            "before running the release experiment"
        ) from exc

    K, C = coupled_reduced_system(data, reduced)
    initial_macro = _macro_initial_coordinates(reduced, cfg.ambient_K)
    initial = np.concatenate(
        (
            np.full(data.K_detailed.shape[0], cfg.ambient_K, dtype=np.float64),
            initial_macro,
        )
    )
    rhs = np.concatenate(
        (data.f_detailed, reduced.f_heat + reduced.f_boundary)
    )

    # Independent reference using exactly the same fixed-step BDF1 equation.
    state_python = initial.copy()
    for _, state_python in solve_bdf1(
        K,
        C,
        initial,
        lambda _t: rhs,
        exp.cpp_duration_s,
        exp.time_step_s,
    ):
        pass

    options = SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1.0e-12,
        linear_max_iterations=4000,
        underrelaxation=1.0,
        nonlinear_max_iterations=20,
        nonlinear_relative_tolerance=1.0e-11,
        nonlinear_absolute_tolerance=1.0e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1.0e-8,
        min_dt=exp.time_step_s,
        max_dt=exp.time_step_s,
        fixed_dt=exp.time_step_s,
    )
    port_model = PortModel(
        operators=(reduced.K, reduced.C, reduced.f_heat + reduced.f_boundary),
        basis=reduced.physical_basis,
        physical_port_count=data.detailed_interface_cells.size,
    )
    coupling = PortCoupling(
        model_cells=data.detailed_interface_cells.copy(),
        model_face=int(Face.ZP),
    )
    solution = solve(data.detailed, port_model, coupling, initial, options)
    state_cpp = np.asarray(solution.state, dtype=np.float64).copy()
    if state_cpp.shape != state_python.shape:
        raise RuntimeError(
            f"C++ state shape {state_cpp.shape} != Python state shape {state_python.shape}"
        )
    error = float(np.max(np.abs(state_cpp - state_python)))
    if error > 1.0e-7:
        raise RuntimeError(
            "C/C++ macromodel solve disagrees with independent BDF1 assembly: "
            f"max error={error:.6e}"
        )
    return error


# ---------------------------------------------------------------------------
# Driver and reporting
# ---------------------------------------------------------------------------


def _print_method(result: MethodResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"  {result.name:<28} order={result.rom_order:4d} "
        f"port={result.port_modes:4d} prep={result.preprocess_s:7.3f}s "
        f"steady={result.steady_error_K:9.5f}K "
        f"transient={result.transient_error_K:9.5f}K {status}"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller smoke-test grid")
    parser.add_argument("--strict", action="store_true", help="fail if no method passes")
    parser.add_argument(
        "--skip-cpp-check",
        action="store_true",
        help="skip the C/C++ extension consistency check",
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--port-modes", type=int, default=None)
    parser.add_argument("--interior-modes", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bci_rom_final_results.json"),
    )
    return parser.parse_args(argv)


def _configs_from_args(args: argparse.Namespace) -> tuple[PackageConfig, ExperimentConfig]:
    package = PackageConfig()
    experiment = ExperimentConfig()
    if args.quick:
        package = PackageConfig(nx=16, ny=16, bump_rows=6, bump_columns=6)
        experiment = ExperimentConfig(
            port_modes=196,
            interior_modes=96,
            rational_modes=96,
            duration_s=0.20,
            time_step_s=0.025,
            cpp_duration_s=0.10,
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
        experiment_values["interior_modes"] = args.interior_modes
        experiment_values["rational_modes"] = args.interior_modes
    if args.duration is not None:
        experiment_values["duration_s"] = args.duration
    if args.dt is not None:
        experiment_values["time_step_s"] = args.dt
    return PackageConfig(**package_values), ExperimentConfig(**experiment_values)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cfg, exp = _configs_from_args(args)

    print("=" * 88)
    print("Transient BCI-ROM package benchmark")
    print("=" * 88)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.total_nz} = "
        f"{cfg.nx * cfg.ny * cfg.total_nz:,} thermal cells"
    )
    print(
        f"Physical interface ports: {cfg.physical_ports}; requested port modes: "
        f"{exp.port_modes}"
    )

    build_start = time.perf_counter()
    data = assemble_package(cfg, exp)
    build_s = time.perf_counter() - build_start
    print(f"Model build + assembly + interface validation: {build_s:.3f}s")

    core = build_macro_core(data, exp)
    methods = build_method_bases(cfg, core, exp)
    print(
        f"Common isolated-macro preprocessing: {core.preprocess_s:.3f}s; "
        f"Steklov spectrum[0:3]={core.transfer_values[:3]}"
    )

    nominal_results: list[MethodResult] = []
    reduced_by_name: dict[str, ReducedMacro] = {}
    print("\nNominal-boundary method comparison:")
    for method in methods:
        steady, transient, steps, reduced = evaluate_method(
            data, core, method, cfg, exp, exp.nominal_h_W_m2K
        )
        passed = (
            steady <= exp.error_limit_K
            and transient <= exp.error_limit_K
            and method.preprocess_s <= exp.preprocess_budget_s
        )
        result = MethodResult(
            name=method.name,
            port_modes=exp.port_modes,
            rom_order=method.V.shape[1],
            preprocess_s=method.preprocess_s,
            steady_error_K=steady,
            transient_error_K=transient,
            transient_steps=steps,
            passed=passed,
        )
        nominal_results.append(result)
        reduced_by_name[method.name] = reduced
        _print_method(result)

    candidates = [result for result in nominal_results if result.passed]
    if not candidates:
        message = (
            f"no method met {exp.error_limit_K:.3f} K and "
            f"{exp.preprocess_budget_s:.1f}s preprocessing limits"
        )
        print(f"\nERROR: {message}", file=sys.stderr)
        if args.strict:
            return 2
        winner = min(
            nominal_results,
            key=lambda item: max(item.steady_error_K, item.transient_error_K),
        )
    else:
        winner = min(
            candidates,
            key=lambda item: (
                item.rom_order,
                max(item.steady_error_K, item.transient_error_K),
                item.preprocess_s,
            ),
        )
    print(f"\nSelected method: {winner.name}")

    method_by_name = {method.name: method for method in methods}
    winning_method = method_by_name[winner.name]

    boundary_results: list[BoundaryResult] = []
    winning_reduced = reduced_by_name[winner.name]
    print("\nBoundary-independence reuse sweep (basis is not re-extracted):")
    for h in exp.boundary_h_values_W_m2K:
        steady, transient, _, reduced = evaluate_method(
            data, core, winning_method, cfg, exp, h
        )
        passed = steady <= exp.error_limit_K and transient <= exp.error_limit_K
        boundary_results.append(
            BoundaryResult(
                h_W_m2K=h,
                steady_error_K=steady,
                transient_error_K=transient,
                passed=passed,
            )
        )
        if math.isclose(h, exp.nominal_h_W_m2K):
            winning_reduced = reduced
        print(
            f"  h={h:8.1f} W/(m2 K): steady={steady:9.5f} K, "
            f"transient={transient:9.5f} K "
            f"{'PASS' if passed else 'FAIL'}"
        )

    cpp_error = None
    if not args.skip_cpp_check:
        print("\nC/C++ end-side consistency check:")
        cpp_error = run_cpp_consistency_check(
            data, winning_reduced, cfg, exp
        )
        print(f"  max|state_cpp - state_python| = {cpp_error:.6e}")

    all_boundary_passed = all(result.passed for result in boundary_results)
    report = {
        "schema_version": 1,
        "package": asdict(cfg),
        "experiment": asdict(exp),
        "thermal_cell_count": int(data.K_full.shape[0]),
        "physical_port_count": int(cfg.physical_ports),
        "selected_method": winner.name,
        "selected_rom_order": int(winner.rom_order),
        "selected_port_modes": int(winner.port_modes),
        "nominal_methods": [asdict(result) for result in nominal_results],
        "boundary_reuse": [asdict(result) for result in boundary_results],
        "cpp_max_state_error": cpp_error,
        "model_build_and_validation_s": build_s,
        "common_preprocess_s": core.preprocess_s,
        "passed": bool(winner.passed and all_boundary_passed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nReport: {args.output}")

    if args.strict and not report["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
