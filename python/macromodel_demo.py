#!/usr/bin/env python3
"""Sparse transient BCI-ROM benchmark for an unequal-footprint package.

The detailed substrate/bump/die domain is coupled to a reusable TIM/spreader/
cold-plate macro model. Four cold-plate convection quadrants form an affine
boundary family. The macro model is reduced with independent vertical-column
bases so the projected operators retain the sparse nearest-neighbour structure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import metahotspot
from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortMap, PortPatch, solve as solve_macro


POWER_MAP = np.asarray(
    (
        (0.10, 0.15, 0.20, 0.15),
        (0.15, 0.50, 1.20, 0.20),
        (0.10, 0.80, 8.55, 0.25),
        (0.10, 0.20, 1.20, 0.45),
    ),
    dtype=np.float64,
)
POWER_MAP /= POWER_MAP.mean()
CHIPLET_POWER_SCALE = (1.00, 0.72, 1.25, 0.55)


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    h_W_m2K: tuple[float, float, float, float]


@dataclass(frozen=True)
class Package:
    ambient_K: float = 300.0
    cold_plate_size_mm: float = 60.0
    spreader_size_mm: float = 50.0
    substrate_size_mm: float = 50.0
    bump_region_size_mm: float = 36.0
    die_size_mm: float = 32.0
    tim_size_mm: float = 32.0
    substrate_mm: float = 1.2
    bump_mm: float = 0.24
    die_mm: float = 0.60
    tim_mm: float = 0.18
    spreader_mm: float = 1.2
    cold_plate_mm: float = 1.5
    substrate_cells: int = 6
    bump_cells: int = 2
    die_cells: int = 4
    tim_cells: int = 2
    spreader_cells: int = 6
    cold_plate_cells: int = 8
    max_xy_cell_mm: float = 4.0
    bump_rows: int = 12
    bump_columns: int = 12
    bump_width_mm: float = 0.60
    chiplet_size_mm: float = 12.0
    chiplet_power_W: float = 25.0

    @property
    def detail_layers(self):
        return (
            (self.substrate_mm, self.substrate_cells),
            (self.bump_mm, self.bump_cells),
            (self.die_mm, self.die_cells),
        )

    @property
    def macro_layers(self):
        return (
            (self.tim_mm, self.tim_cells),
            (self.spreader_mm, self.spreader_cells),
            (self.cold_plate_mm, self.cold_plate_cells),
        )

    @property
    def detail_nz(self) -> int:
        return sum(cells for _, cells in self.detail_layers)

    @property
    def macro_nz(self) -> int:
        return sum(cells for _, cells in self.macro_layers)

    @property
    def nz(self) -> int:
        return self.detail_nz + self.macro_nz

    @property
    def detail_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.detail_layers)

    @property
    def macro_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.macro_layers)

    @property
    def total_height_mm(self) -> float:
        return self.detail_height_mm + self.macro_height_mm

    @property
    def chiplet_origins_mm(self) -> tuple[tuple[float, float], ...]:
        low = -self.die_size_mm / 2.0 + 2.0
        high = self.die_size_mm / 2.0 - 2.0 - self.chiplet_size_mm
        return ((low, low), (high, low), (low, high), (high, high))

    @property
    def x_vertices_mm(self) -> np.ndarray:
        return package_axis_vertices(self)

    @property
    def y_vertices_mm(self) -> np.ndarray:
        return package_axis_vertices(self)

    @property
    def nx(self) -> int:
        return self.x_vertices_mm.size - 1

    @property
    def ny(self) -> int:
        return self.y_vertices_mm.size - 1

    @property
    def port_x_indices(self) -> np.ndarray:
        return footprint_cell_indices(self.x_vertices_mm, self.tim_size_mm)

    @property
    def port_y_indices(self) -> np.ndarray:
        return footprint_cell_indices(self.y_vertices_mm, self.tim_size_mm)

    @property
    def port_shape(self) -> tuple[int, int]:
        return self.port_x_indices.size, self.port_y_indices.size

    @property
    def ports(self) -> int:
        nx, ny = self.port_shape
        return nx * ny

    @property
    def nominal_power_W(self) -> float:
        return self.chiplet_power_W * float(sum(CHIPLET_POWER_SCALE))

    @property
    def peak_to_mean_tile_density(self) -> float:
        return float(POWER_MAP.max())


@dataclass(frozen=True)
class Run:
    error_K: float = 0.25
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    local_dynamic_modes: int = 2
    bdf1_shifts: tuple[float, ...] = (1.0, 2.0)
    speedup_target: float = 1.5
    compression_target: float = 2.5
    report: Path = Path("results/bci_rom_sparse_column_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s


class MacroAffine(NamedTuple):
    compiled: object
    ports: PortMap
    anchor_h: float
    base: Operators
    components: tuple[Operators, ...]
    c_relative_change: float
    ambient_residual: float
    seconds: float

    def at(self, h_values: Sequence[float]) -> Operators:
        h = np.asarray(h_values, dtype=np.float64)
        if h.shape != (len(self.components),):
            raise ValueError("one convection coefficient is required per affine region")
        return combine_many(self.base, self.components, h / self.anchor_h)


class Data(NamedTuple):
    full_layout: object
    detail_steady: object
    detail_transient: object
    detail_ports_steady: PortMap
    detail_ports_transient: PortMap
    macro: MacroAffine


class Column(NamedTuple):
    cells: np.ndarray
    port: int | None


class CellMaps(NamedTuple):
    detail_to_full: np.ndarray
    macro_to_full: np.ndarray


class Basis(NamedTuple):
    W: sp.csc_matrix
    column_count: int
    port_columns: int
    orders: np.ndarray
    ambient_error: float
    orthogonality_error: float
    seconds: float


@dataclass(frozen=True)
class ReducedAffine:
    anchor_h: float
    base: Operators
    components: tuple[Operators, ...]
    seconds: float

    def at(self, h_values: Sequence[float]) -> tuple[Operators, float]:
        started = time.perf_counter()
        h = np.asarray(h_values, dtype=np.float64)
        operators = combine_many(self.base, self.components, h / self.anchor_h)
        return operators, time.perf_counter() - started


class Reference(NamedTuple):
    steady: np.ndarray
    times: np.ndarray
    transient: np.ndarray
    compile_s: float
    steady_solve_s: float
    transient_solve_s: float
    order: int
    k_nnz: int
    c_nnz: int


def centered_rect(size_mm: float) -> tuple[float, float, float, float]:
    half = 0.5 * size_mm
    return -half, -half, size_mm, size_mm


def refined_breakpoints(points: Sequence[float], max_step: float) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64))
    output = [float(unique[0])]
    for left, right in zip(unique[:-1], unique[1:]):
        pieces = max(1, int(math.ceil((right - left) / max_step)))
        output.extend(np.linspace(left, right, pieces + 1)[1:].tolist())
    return np.asarray(output, dtype=np.float64)


def package_axis_vertices(cfg: Package) -> np.ndarray:
    outer = cfg.cold_plate_size_mm / 2.0
    spreader = cfg.spreader_size_mm / 2.0
    bump = cfg.bump_region_size_mm / 2.0
    die = cfg.die_size_mm / 2.0
    points = [-outer, -spreader, -bump, -die, 0.0, die, bump, spreader, outer]
    tile = cfg.chiplet_size_mm / 4.0
    for x0, _ in cfg.chiplet_origins_mm:
        points.extend(x0 + tile * np.arange(5, dtype=np.float64))
    return refined_breakpoints(points, cfg.max_xy_cell_mm)


def footprint_cell_indices(vertices_mm: np.ndarray, size_mm: float) -> np.ndarray:
    half = 0.5 * size_mm
    tolerance = 1.0e-10 * max(1.0, size_mm)
    return np.asarray(
        [
            index
            for index in range(vertices_mm.size - 1)
            if vertices_mm[index] >= -half - tolerance
            and vertices_mm[index + 1] <= half + tolerance
        ],
        dtype=np.int64,
    )


def z_vertices(layers) -> np.ndarray:
    output = [0.0]
    z = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            output.append(z)
    return np.asarray(output, dtype=np.float64)


def add_materials(model) -> None:
    for args in (
        ("organic", ".65", ".65", ".55", "1900", "1100"),
        ("underfill", ".8", ".8", ".8", "1550", "1000"),
        ("copper", "390", "390", "390", "8960", "385"),
        ("silicon", "130", "130", "115", "2330", "700"),
        ("tim", "4", "4", "3", "2500", "900"),
        ("aluminum", "180", "180", "180", "2700", "900"),
    ):
        model.add_material(*args)


def add_centered_rect(model, block: int, size_mm: float) -> None:
    x, y, width, height = centered_rect(size_mm)
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{x:.17g}",
        f"{y:.17g}",
        f"{width:.17g}",
        f"{height:.17g}",
    )


def configure(model, cfg: Package, layers, study: Study, run: Run | None) -> None:
    transient = study == Study.TRANSIENT
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=run.duration_s if transient else 0.0,
        output_interval=run.dt_s if transient else 0.0,
    )
    model.set_mesh(cfg.x_vertices_mm, cfg.y_vertices_mm, z_vertices(layers))
    add_materials(model)


def add_macro(model, cfg: Package) -> None:
    for thickness, material, size in (
        (cfg.cold_plate_mm, "aluminum", cfg.cold_plate_size_mm),
        (cfg.spreader_mm, "copper", cfg.spreader_size_mm),
        (cfg.tim_mm, "tim", cfg.tim_size_mm),
    ):
        layer = model.add_layer(str(thickness))
        add_centered_rect(model, model.add_block(layer, material), size)


def add_activity_functions(model, run: Run) -> None:
    duration = run.duration_s
    traces = (
        ((0.00, 0.20), (0.10, 1.00), (0.35, 0.65), (0.58, 1.30), (0.82, 0.40), (1.00, 0.90)),
        ((0.00, 0.75), (0.18, 1.20), (0.40, 0.30), (0.64, 1.05), (0.88, 0.55), (1.00, 0.80)),
        ((0.00, 0.10), (0.08, 1.45), (0.28, 0.50), (0.52, 1.15), (0.76, 0.25), (1.00, 1.00)),
        ((0.00, 0.55), (0.22, 0.35), (0.44, 1.25), (0.70, 0.60), (0.90, 1.10), (1.00, 0.70)),
    )
    for index, trace in enumerate(traces):
        model.add_function_piecewise(
            f"activity_{index}",
            np.asarray([(fraction * duration, value) for fraction, value in trace]),
        )


def add_detail(model, cfg: Package, study: Study, run: Run) -> None:
    die = model.add_layer(str(cfg.die_mm))
    add_centered_rect(model, model.add_block(die, "silicon"), cfg.die_size_mm)
    if study == Study.TRANSIENT:
        add_activity_functions(model, run)

    tile_size = cfg.chiplet_size_mm / 4.0
    tile_volume_m3 = tile_size * tile_size * cfg.die_mm * 1.0e-9
    for chiplet, ((x0, y0), chiplet_scale) in enumerate(
        zip(cfg.chiplet_origins_mm, CHIPLET_POWER_SCALE)
    ):
        for iy in range(4):
            for ix in range(4):
                tile_power = (
                    cfg.chiplet_power_W
                    * chiplet_scale
                    * POWER_MAP[iy, ix]
                    / POWER_MAP.size
                )
                source = f"{tile_power / tile_volume_m3:.17g}"
                if study == Study.TRANSIENT:
                    activity = (chiplet + 2 * ix + iy) % 4
                    source += f"*activity_{activity}(x)"
                block = model.add_block(die, "silicon", heat_source=source)
                model.add_rect(
                    block,
                    GeometryOp.ADD,
                    f"{x0 + ix * tile_size:.17g}",
                    f"{y0 + iy * tile_size:.17g}",
                    f"{tile_size:.17g}",
                    f"{tile_size:.17g}",
                )

    bump = model.add_layer(str(cfg.bump_mm))
    add_centered_rect(model, model.add_block(bump, "underfill"), cfg.bump_region_size_mm)
    pitch_x = cfg.die_size_mm / cfg.bump_columns
    pitch_y = cfg.die_size_mm / cfg.bump_rows
    origin = -0.5 * cfg.die_size_mm
    for iy in range(cfg.bump_rows):
        for ix in range(cfg.bump_columns):
            x = origin + (ix + 0.5) * pitch_x - 0.5 * cfg.bump_width_mm
            y = origin + (iy + 0.5) * pitch_y - 0.5 * cfg.bump_width_mm
            block = model.add_block(bump, "copper")
            model.add_rect(
                block,
                GeometryOp.ADD,
                f"{x:.17g}",
                f"{y:.17g}",
                f"{cfg.bump_width_mm:.17g}",
                f"{cfg.bump_width_mm:.17g}",
            )

    substrate = model.add_layer(str(cfg.substrate_mm))
    add_centered_rect(
        model,
        model.add_block(substrate, "organic"),
        cfg.substrate_size_mm,
    )


def convection_regions(cfg: Package, z_mm: float):
    half = 0.5 * cfg.cold_plate_size_mm
    return (
        (Axis.Z, z_mm, -half, 0.0, -half, 0.0),
        (Axis.Z, z_mm, 0.0, half, -half, 0.0),
        (Axis.Z, z_mm, -half, 0.0, 0.0, half),
        (Axis.Z, z_mm, 0.0, half, 0.0, half),
    )


def add_convection_family(
    model,
    cfg: Package,
    z_mm: float,
    h_values: Sequence[float] | None,
) -> None:
    if h_values is None:
        return
    values = tuple(float(value) for value in h_values)
    if len(values) != 4:
        raise ValueError("four quadrant convection coefficients are required")
    for coefficient, region in zip(values, convection_regions(cfg, z_mm)):
        if coefficient < 0.0:
            raise ValueError("convection coefficients must be non-negative")
        if coefficient > 0.0:
            model.add_convection(str(coefficient), str(cfg.ambient_K), [region])


def build_package(
    cfg: Package,
    run: Run,
    include_macro: bool,
    study: Study,
    h_values: Sequence[float] | None = None,
):
    model = metahotspot.Model()
    layers = (
        (*cfg.detail_layers, *cfg.macro_layers)
        if include_macro
        else cfg.detail_layers
    )
    configure(model, cfg, layers, study, run)
    if include_macro:
        add_macro(model, cfg)
    add_detail(model, cfg, study, run)
    model.set_default_neumann("0")
    if include_macro:
        add_convection_family(model, cfg, cfg.total_height_mm, h_values)
    return model


def build_macro(cfg: Package, h_values: Sequence[float] | None = None):
    model = metahotspot.Model()
    configure(model, cfg, cfg.macro_layers, Study.STEADY, None)
    add_macro(model, cfg)
    model.set_default_neumann("0")
    add_convection_family(model, cfg, cfg.macro_height_mm, h_values)
    return model


def port_patches(cfg: Package, face: Face, z_m: float) -> list[PortPatch]:
    x = cfg.x_vertices_mm * 1.0e-3
    y = cfg.y_vertices_mm * 1.0e-3
    return [
        PortPatch(int(face), z_m, (x[ix], x[ix + 1], y[iy], y[iy + 1]))
        for ix in cfg.port_x_indices
        for iy in cfg.port_y_indices
    ]


def normalized(operators: Operators) -> Operators:
    K = sp.csc_matrix(operators.K)
    C = sp.csc_matrix(operators.C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(operators.f, dtype=np.float64).copy())


def subtract(a: Operators, b: Operators) -> Operators:
    K = (a.K - b.K).tocsc()
    C = (a.C - b.C).tocsc()
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(a.f) - np.asarray(b.f))


def combine_many(
    base: Operators,
    components: Sequence[Operators],
    coordinates: Sequence[float],
) -> Operators:
    theta = np.asarray(coordinates, dtype=np.float64)
    if theta.shape != (len(components),):
        raise ValueError("affine coordinate count does not match component count")
    K = base.K.copy().tocsc()
    C = base.C.copy().tocsc()
    f = np.asarray(base.f, dtype=np.float64).copy()
    for value, component in zip(theta, components):
        if value == 0.0:
            continue
        K = K + value * component.K
        C = C + value * component.C
        f += value * np.asarray(component.f)
    K = K.tocsc()
    C = C.tocsc()
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, f)


def relative_norm(matrix, reference) -> float:
    denominator = max(float(spla.norm(reference)), np.finfo(float).tiny)
    return float(spla.norm(matrix)) / denominator


def extract_affine_macro(cfg: Package, anchor_h: float) -> MacroAffine:
    if anchor_h <= 0.0:
        raise ValueError("affine_anchor_h must be positive")
    started = time.perf_counter()
    patches = port_patches(cfg, Face.ZM, 0.0)
    compiled = build_macro(cfg).compile()
    ports = PortMap(compiled, patches)
    base = normalized(ports.assemble())
    components = []
    c_changes = []
    ambient_residuals = []
    for quadrant in range(4):
        h_values = [0.0] * 4
        h_values[quadrant] = anchor_h
        anchor_compiled = build_macro(cfg, h_values).compile()
        anchor_ports = PortMap(anchor_compiled, patches)
        try:
            anchor = normalized(anchor_ports.assemble())
            if anchor.K.shape != base.K.shape:
                raise RuntimeError("convection changed the macro state ordering")
            component = subtract(anchor, base)
        finally:
            anchor_ports.close()
            anchor_compiled.close()
        ambient = np.full(base.K.shape[0], cfg.ambient_K)
        defect = np.asarray(component.K @ ambient).ravel() - component.f
        scale = max(np.linalg.norm(component.f), np.finfo(float).tiny)
        components.append(component)
        c_changes.append(relative_norm(component.C, base.C))
        ambient_residuals.append(float(np.linalg.norm(defect) / scale))
    return MacroAffine(
        compiled,
        ports,
        anchor_h,
        base,
        tuple(components),
        float(max(c_changes)),
        float(max(ambient_residuals)),
        time.perf_counter() - started,
    )


def assemble(cfg: Package, run: Run) -> Data:
    full_layout = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    detail_patches = port_patches(
        cfg,
        Face.ZP,
        cfg.detail_height_mm * 1.0e-3,
    )
    macro = extract_affine_macro(cfg, run.affine_anchor_h)
    if macro.ports.port_count != cfg.ports:
        raise RuntimeError("configured interface port count is inconsistent")
    return Data(
        full_layout,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, detail_patches),
        PortMap(detail_transient, detail_patches),
        macro,
    )


def grid_cell(compiled, ix: int, iy: int, iz: int) -> int:
    index = (ix * compiled.ny + iy) * compiled.nz + iz
    return int(compiled.grid_to_cell[index])


def coordinate_cell_map(source, target, target_z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: source/target lateral meshes differ")
    if target_z_offset < 0 or target_z_offset + source.nz > target.nz:
        raise RuntimeError(f"{label}: target z offset is out of range")

    mapping = np.full(source.cell_count, -1, dtype=np.int64)
    for ix in range(source.nx):
        for iy in range(source.ny):
            for iz in range(source.nz):
                source_cell = grid_cell(source, ix, iy, iz)
                target_cell = grid_cell(target, ix, iy, target_z_offset + iz)
                if (source_cell >= 0) != (target_cell >= 0):
                    raise RuntimeError(
                        f"{label}: geometry occupancy differs at ({ix}, {iy}, {iz})"
                    )
                if source_cell < 0:
                    continue
                previous = mapping[source_cell]
                if previous >= 0 and previous != target_cell:
                    raise RuntimeError(f"{label}: source cell maps to multiple targets")
                mapping[source_cell] = target_cell

    if np.any(mapping < 0):
        raise RuntimeError(f"{label}: source cells were not completely mapped")
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(f"{label}: target mapping is not one-to-one")
    return mapping


def build_cell_maps(data: Data, cfg: Package) -> CellMaps:
    detail_steady = coordinate_cell_map(
        data.detail_steady,
        data.full_layout,
        0,
        "detail-steady/full",
    )
    detail_transient = coordinate_cell_map(
        data.detail_transient,
        data.full_layout,
        0,
        "detail-transient/full",
    )
    if not np.array_equal(detail_steady, detail_transient):
        raise RuntimeError("steady and transient detail cell orderings differ")
    macro = coordinate_cell_map(
        data.macro.compiled,
        data.full_layout,
        cfg.detail_nz,
        "macro/full",
    )
    combined = np.r_[detail_steady, macro]
    if combined.size != data.full_layout.cell_count:
        raise RuntimeError("detail and macro maps do not cover the full model")
    if np.unique(combined).size != data.full_layout.cell_count:
        raise RuntimeError("detail and macro maps overlap or omit full cells")
    return CellMaps(detail_steady, macro)


def operator_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, :ports].tocsc(),
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, :ports].tocsc(),
        operators.C[ports:, ports:].tocsc(),
    )


def macro_columns(compiled, cfg: Package) -> tuple[Column, ...]:
    port_pairs = [
        (int(ix), int(iy))
        for ix in cfg.port_x_indices
        for iy in cfg.port_y_indices
    ]
    port_lookup = {pair: index for index, pair in enumerate(port_pairs)}
    output = []
    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = [
                grid_cell(compiled, ix, iy, iz)
                for iz in range(compiled.nz)
                if grid_cell(compiled, ix, iy, iz) >= 0
            ]
            if cells:
                output.append(
                    Column(
                        np.asarray(cells, dtype=np.int64),
                        port_lookup.get((ix, iy)),
                    )
                )
    if sum(column.port is not None for column in output) != cfg.ports:
        raise RuntimeError("interface-port/column mapping is inconsistent")
    return tuple(output)


def range_basis(matrix: np.ndarray) -> np.ndarray:
    q, r, _ = scipy.linalg.qr(
        np.asarray(matrix, dtype=np.float64),
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    tolerance = np.finfo(float).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, diagonal > tolerance])


def build_basis(macro: MacroAffine, cfg: Package, run: Run) -> Basis:
    started = time.perf_counter()
    ports = macro.ports.port_count
    Kip0, Kii0, Cip0, Cii0 = operator_blocks(macro.base, ports)
    component_blocks = [operator_blocks(component, ports) for component in macro.components]
    columns = macro_columns(macro.compiled, cfg)

    rows = []
    basis_columns = []
    values = []
    orders = []
    offset = 0

    for column in columns:
        cells = column.cells
        k0 = Kii0[cells, :][:, cells].toarray()
        c0 = Cii0[cells, :][:, cells].toarray()
        candidates = [np.ones(cells.size)]

        eigenvalues, eigenvectors = scipy.linalg.eigh(k0, c0, check_finite=False)
        dynamic = np.flatnonzero(eigenvalues <= run.modal_cutoff_per_s)[
            : run.local_dynamic_modes
        ]
        candidates.extend(eigenvectors[:, index] for index in dynamic)

        if column.port is not None:
            port = column.port
            b0 = Kip0[cells, port].toarray().ravel()
            cp0 = Cip0[cells, port].toarray().ravel()
            static = -scipy.linalg.solve(
                k0,
                b0,
                assume_a="sym",
                check_finite=False,
            )
            candidates.append(static)

            for Kip1, Kii1, _, _ in component_blocks:
                k1 = Kii1[cells, :][:, cells].toarray()
                b1 = Kip1[cells, port].toarray().ravel()
                rhs = k1 @ static + b1
                if np.linalg.norm(rhs) > 1.0e-14 * max(np.linalg.norm(b0), 1.0):
                    candidates.append(
                        -scipy.linalg.solve(
                            k0,
                            rhs,
                            assume_a="sym",
                            check_finite=False,
                        )
                    )

            for multiplier in run.bdf1_shifts:
                shift = multiplier / run.dt_s
                A = k0 + shift * c0
                rhs = b0 + shift * cp0
                response = -scipy.linalg.solve(
                    A,
                    rhs,
                    assume_a="sym",
                    check_finite=False,
                )
                candidates.append(response - static)

        local = range_basis(np.column_stack(candidates))
        orders.append(local.shape[1])
        for local_row, cell in enumerate(cells):
            nonzero = np.flatnonzero(np.abs(local[local_row]) > 1.0e-14)
            rows.extend([int(cell)] * nonzero.size)
            basis_columns.extend((offset + nonzero).tolist())
            values.extend(local[local_row, nonzero].tolist())
        offset += local.shape[1]

    W = sp.csc_matrix(
        (values, (rows, basis_columns)),
        shape=(Kii0.shape[0], offset),
    )
    ones = np.ones(W.shape[0])
    ambient_error = float(
        np.linalg.norm(W @ (W.T @ ones) - ones) / math.sqrt(ones.size)
    )
    orthogonality = float(
        spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc"))
    )
    if ambient_error > 1.0e-10:
        raise RuntimeError("macro basis does not preserve a uniform temperature field")
    if orthogonality > 1.0e-10:
        raise RuntimeError("macro basis lost orthogonality")
    return Basis(
        W,
        len(columns),
        sum(column.port is not None for column in columns),
        np.asarray(orders),
        ambient_error,
        orthogonality,
        time.perf_counter() - started,
    )


def project(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    transform = sp.block_diag(
        (sp.eye(ports, format="csc"), W),
        format="csc",
    )

    def project_matrix(matrix):
        reduced = (transform.T @ matrix @ transform).tocsc()
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project_matrix(operators.K),
        project_matrix(operators.C),
        np.asarray(transform.T @ operators.f).ravel(),
    )


def project_affine(macro: MacroAffine, basis: Basis) -> ReducedAffine:
    started = time.perf_counter()
    ports = macro.ports.port_count
    return ReducedAffine(
        macro.anchor_h,
        project(macro.base, ports, basis.W),
        tuple(project(component, ports, basis.W) for component in macro.components),
        time.perf_counter() - started,
    )


def solve_options(run: Run, transient: bool) -> SolveOptions:
    dt = run.dt_s if transient else 1.0
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1.0e-12,
        linear_max_iterations=5000,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1.0e-11,
        nonlinear_absolute_tolerance=1.0e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1.0e-9,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def reference(cfg: Package, run: Run, boundary: BoundaryCase) -> Reference:
    started = time.perf_counter()
    steady_compiled = build_package(
        cfg,
        run,
        True,
        Study.STEADY,
        boundary.h_W_m2K,
    ).compile()
    transient_compiled = build_package(
        cfg,
        run,
        True,
        Study.TRANSIENT,
        boundary.h_W_m2K,
    ).compile()
    compile_s = time.perf_counter() - started
    operators = transient_compiled.assemble()
    try:
        started = time.perf_counter()
        with steady_compiled.solve(opts=solve_options(run, False)) as solution:
            steady = np.asarray(solution.temperature).copy()
        steady_solve_s = time.perf_counter() - started

        started = time.perf_counter()
        with transient_compiled.solve(opts=solve_options(run, True)) as solution:
            times = np.asarray(solution.history_times).copy()
            transient = np.asarray(solution.temperature_history).copy()
        transient_solve_s = time.perf_counter() - started
    finally:
        steady_compiled.close()
        transient_compiled.close()
    return Reference(
        steady,
        times,
        transient,
        compile_s,
        steady_solve_s,
        transient_solve_s,
        operators.K.shape[0],
        operators.K.nnz,
        operators.C.nnz,
    )


def evaluate(
    data: Data,
    maps: CellMaps,
    cfg: Package,
    run: Run,
    basis: Basis,
    reduced: ReducedAffine,
    boundary: BoundaryCase,
    ref: Reference,
):
    operators, online_assembly_s = reduced.at(boundary.h_W_m2K)
    internal0 = np.asarray(
        basis.W.T @ np.full(basis.W.shape[0], cfg.ambient_K)
    ).ravel()

    def run_solve(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count + cfg.ports, cfg.ambient_K),
            internal0,
        ]
        started = time.perf_counter()
        with solve_macro(
            compiled,
            operators,
            ports,
            state,
            solve_options(run, transient),
        ) as solution:
            elapsed = time.perf_counter() - started
            if transient:
                return (
                    np.asarray(solution.history_times).copy(),
                    np.asarray(solution.state_history).copy(),
                    elapsed,
                )
            return np.asarray(solution.state).copy(), elapsed

    steady_state, steady_solve_s = run_solve(False)
    times, transient_states, transient_solve_s = run_solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        recovered = np.empty((states.shape[0], data.full_layout.cell_count))
        recovered[:, maps.detail_to_full] = states[:, :detail_n]
        recovered[:, maps.macro_to_full] = (
            basis.W @ states[:, detail_n + cfg.ports :].T
        ).T
        return recovered

    if times.shape != ref.times.shape or not np.allclose(
        times,
        ref.times,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("full and reduced output times differ")

    steady_diff = np.abs(recover(steady_state)[0] - ref.steady)
    transient_diff = np.abs(recover(transient_states) - ref.transient)
    return {
        "name": boundary.name,
        "h_quadrants_W_m2K": list(boundary.h_W_m2K),
        "steady_error_K": float(steady_diff.max()),
        "transient_error_K": float(transient_diff.max()),
        "detail_steady_error_K": float(steady_diff[maps.detail_to_full].max()),
        "macro_steady_error_K": float(steady_diff[maps.macro_to_full].max()),
        "detail_transient_error_K": float(
            transient_diff[:, maps.detail_to_full].max()
        ),
        "macro_transient_error_K": float(
            transient_diff[:, maps.macro_to_full].max()
        ),
        "online_reduced_assembly_s": online_assembly_s,
        "full_compile_s": ref.compile_s,
        "full_steady_solve_s": ref.steady_solve_s,
        "reduced_steady_solve_s": steady_solve_s,
        "full_transient_solve_s": ref.transient_solve_s,
        "reduced_transient_solve_s": transient_solve_s,
        "transient_speedup": ref.transient_solve_s
        / max(transient_solve_s, np.finfo(float).tiny),
        "full_order": ref.order,
        "reduced_online_order": int(detail_n + operators.K.shape[0]),
        "full_k_nnz": ref.k_nnz,
        "full_c_nnz": ref.c_nnz,
        "reduced_macro_k_nnz": int(operators.K.nnz),
        "reduced_macro_c_nnz": int(operators.C.nnz),
    }


def close_data(data: Data) -> None:
    data.detail_ports_steady.close()
    data.detail_ports_transient.close()
    data.macro.ports.close()
    data.detail_steady.close()
    data.detail_transient.close()
    data.macro.compiled.close()
    data.full_layout.close()


def configs(quick: bool):
    if quick:
        cfg = Package(
            substrate_cells=3,
            bump_cells=1,
            die_cells=2,
            tim_cells=1,
            spreader_cells=3,
            cold_plate_cells=4,
            max_xy_cell_mm=6.0,
            bump_rows=8,
            bump_columns=8,
        )
        run = Run(
            error_K=0.35,
            duration_s=0.20,
            local_dynamic_modes=1,
            bdf1_shifts=(1.0,),
            speedup_target=1.0,
            compression_target=2.0,
        )
        boundaries = (
            BoundaryCase("uniform-low", (500.0,) * 4),
            BoundaryCase("uniform-high", (8000.0,) * 4),
            BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
        )
        return cfg, run, boundaries

    cfg = Package()
    run = Run()
    boundaries = (
        BoundaryCase("uniform-low", (500.0,) * 4),
        BoundaryCase("uniform-medium", (2500.0,) * 4),
        BoundaryCase("uniform-high", (8000.0,) * 4),
        BoundaryCase("x-gradient", (500.0, 8000.0, 500.0, 8000.0)),
        BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
    )
    return cfg, run, boundaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    cfg, run, boundaries = configs(args.quick)
    print("=" * 100)
    print("Transient BCI-ROM - sparse irregular-column local thermal basis")
    print("=" * 100)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(
        f"Nominal die power={cfg.nominal_power_W:.2f} W; "
        f"tile peak/mean density={cfg.peak_to_mean_tile_density:.2f}x"
    )

    started = time.perf_counter()
    data = assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    try:
        maps = build_cell_maps(data, cfg)
        basis = build_basis(data.macro, cfg, run)
        reduced = project_affine(data.macro, basis)
        full_macro_order = cfg.ports + basis.W.shape[0]
        reduced_macro_order = cfg.ports + basis.W.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; exact ports={cfg.ports}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
            f"({compression:.2f}x)"
        )

        results = []
        offline_s = assembly_s + basis.seconds + reduced.seconds
        for boundary in boundaries:
            result = evaluate(
                data,
                maps,
                cfg,
                run,
                basis,
                reduced,
                boundary,
                reference(cfg, run, boundary),
            )
            accuracy = (
                max(result["steady_error_K"], result["transient_error_K"])
                <= run.error_K
            )
            speed = (
                result["transient_speedup"] >= run.speedup_target
                if args.strict
                else True
            )
            result.update(
                accuracy_passed=accuracy,
                speedup_passed=speed,
                passed=accuracy and speed,
            )
            results.append(result)
            print(
                f"{boundary.name:>16s}: error steady/transient="
                f"{result['steady_error_K']:.5f}/"
                f"{result['transient_error_K']:.5f} K; "
                f"full/ROM={result['full_transient_solve_s']:.3f}/"
                f"{result['reduced_transient_solve_s']:.3f}s, "
                f"speedup={result['transient_speedup']:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= run.compression_target
        report = {
            "schema_version": 15,
            "mode": "quick" if args.quick else "strict",
            "method": (
                "sparse irregular-column static/affine/BDF1/local-mode "
                "Galerkin ROM"
            ),
            "package": {
                **asdict(cfg),
                "nx": cfg.nx,
                "ny": cfg.ny,
                "ports": cfg.ports,
                "port_shape": list(cfg.port_shape),
                "nominal_power_W": cfg.nominal_power_W,
                "power_map_normalized": POWER_MAP.tolist(),
                "peak_to_mean_tile_density": cfg.peak_to_mean_tile_density,
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "experiment": {**asdict(run), "report": str(run.report)},
            "affine_boundary": {
                "family": "A(h)=A0+sum_q(h_q/anchor)*DeltaA_q",
                "regions": [
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast",
                ],
                "anchor_h_W_m2K": run.affine_anchor_h,
                "full_order_offline_assemblies": 5,
                "full_order_online_assemblies_per_case": 0,
                "capacitance_relative_change": data.macro.c_relative_change,
                "ambient_consistency_residual": data.macro.ambient_residual,
                "extraction_s": data.macro.seconds,
                "projection_s": reduced.seconds,
            },
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "compression_ratio": compression,
                "compression_target": run.compression_target,
                "compression_passed": compression_passed,
                "column_count": basis.column_count,
                "port_columns": basis.port_columns,
                "local_order_min": int(basis.orders.min()),
                "local_order_mean": float(basis.orders.mean()),
                "local_order_max": int(basis.orders.max()),
                "basis_nnz": basis.W.nnz,
                "basis_density": basis.W.nnz
                / max(1, basis.W.shape[0] * basis.W.shape[1]),
                "ambient_reconstruction_error": basis.ambient_error,
                "orthogonality_error": basis.orthogonality_error,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence",
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(item["passed"] for item in results) and compression_passed
            ),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print("Passivity: preserved structurally by symmetric Galerkin congruence")
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
