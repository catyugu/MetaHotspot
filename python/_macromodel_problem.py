#!/usr/bin/env python3
"""Transient boundary-condition-independent affine DtN ROM benchmark.

This experiment deliberately separates the package into a detailed lower domain
(substrate, bump/underfill and die) and a reusable upper thermal macro domain
(TIM, spreader and cold plate). The upper domain is extracted once as an exact
physical-port DtN system. Four independent cold-plate convection quadrants are
identified as an affine boundary family,

    A(h) = A0 + sum_q h_q / h_anchor * DeltaA_q,

and are projected offline. Online boundary changes therefore require only a
small reduced-coordinate affine combination; no full-order macro assembly is
performed.

Compared with the earlier column-local basis, the internal macro basis here is
built by block rational transfer snapshots over boundary and frequency samples,
compressed by SVD, and adaptively enriched with worst-case residual correction
vectors. Exact interface temperatures remain unreduced. The geometry uses
realistic unequal footprints and the die carries a strongly non-uniform tiled
power map with several independent transient activity traces.
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

    @property
    def mean_h(self) -> float:
        return float(np.mean(self.h_W_m2K))


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
        return sum(t for t, _ in self.detail_layers)

    @property
    def macro_height_mm(self) -> float:
        return sum(t for t, _ in self.macro_layers)

    @property
    def total_height_mm(self) -> float:
        return self.detail_height_mm + self.macro_height_mm

    @property
    def chiplet_origins_mm(self) -> tuple[tuple[float, float], ...]:
        offset = self.die_size_mm / 2.0 - 2.0 - self.chiplet_size_mm
        return (
            (-self.die_size_mm / 2.0 + 2.0, -self.die_size_mm / 2.0 + 2.0),
            (offset, -self.die_size_mm / 2.0 + 2.0),
            (-self.die_size_mm / 2.0 + 2.0, offset),
            (offset, offset),
        )

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
        px, py = self.port_shape
        return px * py

    @property
    def nominal_power_W(self) -> float:
        return self.chiplet_power_W * float(sum(CHIPLET_POWER_SCALE))

    @property
    def peak_to_mean_tile_density(self) -> float:
        return float(POWER_MAP.max())


@dataclass(frozen=True)
class Run:
    error_K: float = 0.05
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    expansion_points: int = 6
    input_modes: int = 36
    svd_relative_tolerance: float = 1.0e-8
    residual_tolerance: float = 2.0e-6
    enrichment_block: int = 12
    max_internal_order: int = 240
    speedup_target: float = 2.0
    compression_target: float = 10.0
    report: Path = Path("results/bci_rom_rational_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s

    @property
    def expansion_points_per_s(self) -> tuple[float, ...]:
        if self.expansion_points < 2:
            return (0.0,)
        low = max(0.25 / self.duration_s, np.finfo(float).tiny)
        positive = np.geomspace(low, self.modal_cutoff_per_s, self.expansion_points - 1)
        return (0.0, *(float(v) for v in positive))


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
    detail_cells: np.ndarray
    macro_cells: np.ndarray


class Basis(NamedTuple):
    W: sp.csc_matrix
    initial_order: int
    final_order: int
    singular_values: np.ndarray
    snapshot_columns: int
    input_modes: int
    expansion_points_per_s: np.ndarray
    training_boundary_count: int
    residual_history: np.ndarray
    worst_residual: float
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
    bytes: int


class ValidationPoint(NamedTuple):
    label: str
    s_per_s: float
    A: sp.csc_matrix
    B: sp.csc_matrix
    lu: object


def centered_rect(size_mm: float) -> tuple[float, float, float, float]:
    half = 0.5 * size_mm
    return -half, -half, size_mm, size_mm


def refined_breakpoints(points: Sequence[float], max_step: float) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64))
    output = [float(unique[0])]
    for left, right in zip(unique[:-1], unique[1:]):
        pieces = max(1, int(math.ceil((right - left) / max_step)))
        output.extend(np.linspace(left, right, pieces + 1, endpoint=True)[1:].tolist())
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
    indices = [
        i
        for i in range(vertices_mm.size - 1)
        if vertices_mm[i] >= -half - tolerance
        and vertices_mm[i + 1] <= half + tolerance
    ]
    return np.asarray(indices, dtype=np.int64)


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
    t = run.duration_s
    traces = (
        ((0.00, 0.20), (0.10, 1.00), (0.35, 0.65), (0.58, 1.30), (0.82, 0.40), (1.00, 0.90)),
        ((0.00, 0.75), (0.18, 1.20), (0.40, 0.30), (0.64, 1.05), (0.88, 0.55), (1.00, 0.80)),
        ((0.00, 0.10), (0.08, 1.45), (0.28, 0.50), (0.52, 1.15), (0.76, 0.25), (1.00, 1.00)),
        ((0.00, 0.55), (0.22, 0.35), (0.44, 1.25), (0.70, 0.60), (0.90, 1.10), (1.00, 0.70)),
    )
    for index, trace in enumerate(traces):
        model.add_function_piecewise(
            f"activity_{index}",
            np.asarray([(fraction * t, value) for fraction, value in trace]),
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
        model, model.add_block(substrate, "organic"), cfg.substrate_size_mm
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
    model, cfg: Package, z_mm: float, h_values: Sequence[float] | None
) -> None:
    if h_values is None:
        return
    values = tuple(float(v) for v in h_values)
    if len(values) != 4:
        raise ValueError("four quadrant convection coefficients are required")
    for coefficient, region in zip(values, convection_regions(cfg, z_mm)):
        if coefficient < 0.0:
            raise ValueError("convection coefficients must be non-negative")
        if coefficient == 0.0:
            continue
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
        (*cfg.detail_layers, *cfg.macro_layers) if include_macro else cfg.detail_layers
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
    K, C = sp.csc_matrix(operators.K), sp.csc_matrix(operators.C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(operators.f, dtype=np.float64).copy())


def subtract(a: Operators, b: Operators) -> Operators:
    K, C = (a.K - b.K).tocsc(), (a.C - b.C).tocsc()
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(a.f) - np.asarray(b.f))


def combine_many(
    base: Operators, components: Sequence[Operators], coordinates: Sequence[float]
) -> Operators:
    theta = np.asarray(coordinates, dtype=np.float64)
    if theta.shape != (len(components),):
        raise ValueError("affine coordinate count does not match component count")
    K, C = base.K.copy().tocsc(), base.C.copy().tocsc()
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
        h = [0.0] * 4
        h[quadrant] = anchor_h
        anchor_compiled = build_macro(cfg, h).compile()
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


def valid_zone_cells(compiled, z0: int, z1: int) -> np.ndarray:
    cells = []
    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            for iz in range(z0, z1):
                cell = int(
                    compiled.grid_to_cell[(ix * compiled.ny + iy) * compiled.nz + iz]
                )
                if cell >= 0:
                    cells.append(cell)
    return np.asarray(cells, dtype=np.int64)


def assemble(cfg: Package, run: Run) -> Data:
    full_layout = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
    detail_cells = valid_zone_cells(full_layout, 0, cfg.detail_nz)
    macro_cells = valid_zone_cells(full_layout, cfg.detail_nz, cfg.nz)
    macro = extract_affine_macro(cfg, run.affine_anchor_h)
    if detail_cells.size != detail_steady.cell_count:
        raise RuntimeError("detail/full cell ordering is inconsistent")
    if macro_cells.size != macro.compiled.cell_count:
        raise RuntimeError("macro/full cell ordering is inconsistent")
    if macro.ports.port_count != cfg.ports:
        raise RuntimeError("configured interface port count is inconsistent")
    return Data(
        full_layout,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, patches),
        PortMap(detail_transient, patches),
        macro,
        detail_cells,
        macro_cells,
    )


def internal_dynamic_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, :ports].tocsc(),
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, :ports].tocsc(),
        operators.C[ports:, ports:].tocsc(),
    )


def dct_matrix(order: int) -> np.ndarray:
    x = np.arange(order, dtype=np.float64)[:, None]
    k = np.arange(order, dtype=np.float64)[None, :]
    matrix = np.cos(math.pi * (x + 0.5) * k / order)
    matrix[:, 0] /= math.sqrt(order)
    if order > 1:
        matrix[:, 1:] *= math.sqrt(2.0 / order)
    return matrix


def interface_input_modes(cfg: Package, count: int) -> np.ndarray:
    nx, ny = cfg.port_shape
    dx, dy = dct_matrix(nx), dct_matrix(ny)
    pairs = sorted(
        ((ix, iy) for ix in range(nx) for iy in range(ny)),
        key=lambda item: (
            (item[0] / max(nx - 1, 1)) ** 2 + (item[1] / max(ny - 1, 1)) ** 2,
            item[0] + item[1],
            item[0],
        ),
    )
    columns = [np.kron(dx[:, ix], dy[:, iy]) for ix, iy in pairs[:count]]
    return np.ascontiguousarray(np.column_stack(columns))


def training_boundaries(run: Run) -> tuple[tuple[float, ...], ...]:
    a = run.affine_anchor_h
    high = 3.2 * a
    low = 0.2 * a
    unit = [tuple(a if i == q else 0.0 for i in range(4)) for q in range(4)]
    return (
        (0.0, 0.0, 0.0, 0.0),
        (a, a, a, a),
        (high, low, 0.6 * a, 1.8 * a),
        (low, high, 1.8 * a, 0.6 * a),
        *unit,
    )


def validation_boundaries(run: Run) -> tuple[tuple[float, ...], ...]:
    a = run.affine_anchor_h
    return (
        (0.08 * a, 0.08 * a, 0.08 * a, 0.08 * a),
        (0.55 * a, 1.65 * a, 2.85 * a, 0.32 * a),
        (3.4 * a, 0.25 * a, 0.45 * a, 2.1 * a),
        (0.35 * a, 2.6 * a, 3.0 * a, 0.18 * a),
    )


def normalized_snapshot_block(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float64)
    norms = np.linalg.norm(block, axis=0)
    keep = norms > np.finfo(float).eps * max(1.0, float(norms.max(initial=0.0)))
    if not np.any(keep):
        return np.empty((block.shape[0], 0), dtype=np.float64)
    return block[:, keep] / norms[keep]


def svd_basis(
    snapshots: np.ndarray, relative_tolerance: float, max_order: int
) -> tuple[np.ndarray, np.ndarray]:
    if snapshots.size == 0:
        raise RuntimeError("rational sampling produced no usable snapshots")
    U, singular_values, _ = scipy.linalg.svd(
        snapshots,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    threshold = relative_tolerance * singular_values[0]
    order = int(np.count_nonzero(singular_values >= threshold))
    order = max(1, min(order, max_order))
    return np.ascontiguousarray(U[:, :order]), singular_values


def transfer_snapshot(
    operators: Operators, ports: int, s_per_s: float, inputs: np.ndarray
) -> np.ndarray:
    Kip, Kii, Cip, Cii = internal_dynamic_blocks(operators, ports)
    A = (Kii + s_per_s * Cii).tocsc()
    B = (Kip + s_per_s * Cip).tocsc()
    lu = spla.splu(A)
    return lu.solve(-np.asarray(B @ inputs))


def validation_points(
    macro: MacroAffine, run: Run, ports: int
) -> tuple[ValidationPoint, ...]:
    frequencies = run.expansion_points_per_s
    selected_frequencies = tuple(
        dict.fromkeys((frequencies[0], *frequencies[1::2], frequencies[-1]))
    )
    points = []
    for boundary_index, h in enumerate(validation_boundaries(run)):
        operators = macro.at(h)
        Kip, Kii, Cip, Cii = internal_dynamic_blocks(operators, ports)
        for s in selected_frequencies:
            A = (Kii + s * Cii).tocsc()
            B = (Kip + s * Cip).tocsc()
            points.append(
                ValidationPoint(
                    f"boundary-{boundary_index}/s={s:.6g}",
                    float(s),
                    A,
                    B,
                    spla.splu(A),
                )
            )
    return tuple(points)


def residual_and_correction(
    point: ValidationPoint, W: np.ndarray, correction: bool
) -> tuple[float, np.ndarray | None]:
    reduced_A = np.asarray(W.T @ (point.A @ W))
    reduced_B = np.asarray(W.T @ point.B)
    q = scipy.linalg.solve(
        reduced_A, -reduced_B, assume_a="sym", check_finite=False
    )
    residual = np.asarray(point.A @ (W @ q) + point.B.toarray())
    relative = float(
        np.linalg.norm(residual, ord="fro")
        / max(np.linalg.norm(point.B.data), np.finfo(float).tiny)
    )
    if not correction:
        return relative, None
    return relative, point.lu.solve(-residual)


def orthogonal_enrichment(
    W: np.ndarray, correction: np.ndarray, block_size: int, max_order: int
) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.float64)
    correction -= W @ (W.T @ correction)
    U, singular_values, _ = scipy.linalg.svd(
        correction,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if not singular_values.size or singular_values[0] == 0.0:
        return W
    usable = int(np.count_nonzero(singular_values >= 1.0e-10 * singular_values[0]))
    room = max_order - W.shape[1]
    add = min(block_size, usable, room)
    if add <= 0:
        return W
    candidate = np.column_stack((W, U[:, :add]))
    Q, _ = scipy.linalg.qr(candidate, mode="economic", check_finite=False)
    return np.ascontiguousarray(Q)


def build_basis(macro: MacroAffine, cfg: Package, run: Run) -> Basis:
    started = time.perf_counter()
    ports = macro.ports.port_count
    inputs = interface_input_modes(cfg, min(run.input_modes, ports))
    blocks = []
    boundaries = training_boundaries(run)
    for h in boundaries:
        operators = macro.at(h)
        for s in run.expansion_points_per_s:
            block = normalized_snapshot_block(
                transfer_snapshot(operators, ports, s, inputs)
            )
            if block.shape[1]:
                blocks.append(block)
    snapshots = np.ascontiguousarray(np.column_stack(blocks))
    initial_limit = max(1, run.max_internal_order - 2 * run.enrichment_block)
    W, singular_values = svd_basis(
        snapshots, run.svd_relative_tolerance, initial_limit
    )
    initial_order = W.shape[1]
    residual_history = []
    points = validation_points(macro, run, ports)
    while True:
        residuals = [residual_and_correction(point, W, False)[0] for point in points]
        worst_index = int(np.argmax(residuals))
        worst = float(residuals[worst_index])
        residual_history.append(worst)
        if worst <= run.residual_tolerance or W.shape[1] >= run.max_internal_order:
            break
        _, correction = residual_and_correction(points[worst_index], W, True)
        previous_order = W.shape[1]
        W = orthogonal_enrichment(
            W, correction, run.enrichment_block, run.max_internal_order
        )
        if W.shape[1] == previous_order:
            break
    orthogonality = float(
        np.linalg.norm(W.T @ W - np.eye(W.shape[1]), ord="fro")
    )
    sparse_W = sp.csc_matrix(W)
    return Basis(
        sparse_W,
        initial_order,
        W.shape[1],
        singular_values,
        snapshots.shape[1],
        inputs.shape[1],
        np.asarray(run.expansion_points_per_s),
        len(boundaries),
        np.asarray(residual_history),
        float(residual_history[-1]),
        orthogonality,
        time.perf_counter() - started,
    )


def project(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    Kpp = operators.K[:ports, :ports].tocsc()
    Kpi = (operators.K[:ports, ports:] @ W).tocsc()
    Kip = (W.T @ operators.K[ports:, :ports]).tocsc()
    Kii = (W.T @ operators.K[ports:, ports:] @ W).tocsc()
    Cpp = operators.C[:ports, :ports].tocsc()
    Cpi = (operators.C[:ports, ports:] @ W).tocsc()
    Cip = (W.T @ operators.C[ports:, :ports]).tocsc()
    Cii = (W.T @ operators.C[ports:, ports:] @ W).tocsc()

    def symmetric_block(a, b, c, d):
        matrix = sp.bmat(((a, b), (c, d)), format="csc")
        matrix = (0.5 * (matrix + matrix.T)).tocsc()
        matrix.eliminate_zeros()
        return matrix

    transform_f = np.r_[
        np.asarray(operators.f[:ports]),
        np.asarray(W.T @ operators.f[ports:]).ravel(),
    ]
    return Operators(
        symmetric_block(Kpp, Kpi, Kip, Kii),
        symmetric_block(Cpp, Cpi, Cip, Cii),
        transform_f,
    )


def project_affine(macro: MacroAffine, W: sp.csc_matrix) -> ReducedAffine:
    started = time.perf_counter()
    ports = macro.ports.port_count
    return ReducedAffine(
        macro.anchor_h,
        project(macro.base, ports, W),
        tuple(project(component, ports, W) for component in macro.components),
        time.perf_counter() - started,
    )


def solve_options(run: Run, transient: bool) -> SolveOptions:
    dt = run.dt_s if transient else 1.0
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1e-12,
        linear_max_iterations=5000,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1e-11,
        nonlinear_absolute_tolerance=1e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1e-9,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def csc_bytes(matrix) -> int:
    matrix = matrix.tocsc()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def reference(cfg: Package, run: Run, boundary: BoundaryCase) -> Reference:
    started = time.perf_counter()
    steady_compiled = build_package(
        cfg, run, True, Study.STEADY, boundary.h_W_m2K
    ).compile()
    transient_compiled = build_package(
        cfg, run, True, Study.TRANSIENT, boundary.h_W_m2K
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
        csc_bytes(operators.K) + csc_bytes(operators.C),
    )


def minimum_symmetric_eigenvalue(matrix: sp.csc_matrix) -> float:
    dense = np.asarray(matrix.toarray())
    return float(scipy.linalg.eigvalsh(dense, subset_by_index=(0, 0))[0])


def passivity_metrics(
    reduced: ReducedAffine, boundaries: Sequence[BoundaryCase]
) -> dict[str, float | bool]:
    minimum_k = math.inf
    minimum_c = math.inf
    maximum_asymmetry = 0.0
    scale = 0.0
    for boundary in boundaries:
        operators, _ = reduced.at(boundary.h_W_m2K)
        minimum_k = min(minimum_k, minimum_symmetric_eigenvalue(operators.K))
        minimum_c = min(minimum_c, minimum_symmetric_eigenvalue(operators.C))
        maximum_asymmetry = max(
            maximum_asymmetry,
            float(spla.norm(operators.K - operators.K.T)),
            float(spla.norm(operators.C - operators.C.T)),
        )
        scale = max(scale, float(spla.norm(operators.K)), float(spla.norm(operators.C)))
    tolerance = 1.0e-10 * max(scale, 1.0)
    return {
        "minimum_K_eigenvalue": minimum_k,
        "minimum_C_eigenvalue": minimum_c,
        "maximum_symmetry_defect": maximum_asymmetry,
        "tolerance": tolerance,
        "passed": bool(min(minimum_k, minimum_c) >= -tolerance),
    }


def evaluate(
    data: Data,
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
            np.full(compiled.cell_count, cfg.ambient_K),
            np.full(cfg.ports, cfg.ambient_K),
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
        recovered[:, data.detail_cells] = states[:, :detail_n]
        recovered[:, data.macro_cells] = (
            basis.W @ states[:, detail_n + cfg.ports :].T
        ).T
        return recovered

    recovered_steady = recover(steady_state)[0]
    steady_error = float(np.max(np.abs(recovered_steady - ref.steady)))
    steady_rise = max(
        float(np.max(np.abs(ref.steady - cfg.ambient_K))), np.finfo(float).tiny
    )
    if times.shape != ref.times.shape or not np.allclose(
        times, ref.times, atol=1e-12, rtol=0
    ):
        raise RuntimeError("full and reduced solvers returned different output times")
    recovered_transient = recover(transient_states)
    transient_error = float(np.max(np.abs(recovered_transient - ref.transient)))
    transient_rise = max(
        float(np.max(np.abs(ref.transient - cfg.ambient_K))), np.finfo(float).tiny
    )
    return {
        "name": boundary.name,
        "h_quadrants_W_m2K": list(boundary.h_W_m2K),
        "h_mean_W_m2K": boundary.mean_h,
        "affine_coordinates": [
            value / reduced.anchor_h for value in boundary.h_W_m2K
        ],
        "steady_error_K": steady_error,
        "steady_relative_rise_error": steady_error / steady_rise,
        "transient_error_K": transient_error,
        "transient_relative_rise_error": transient_error / transient_rise,
        "online_reduced_assembly_s": online_assembly_s,
        "online_full_order_macro_assemblies": 0,
        "full_compile_s": ref.compile_s,
        "full_steady_solve_s": ref.steady_solve_s,
        "reduced_steady_solve_s": steady_solve_s,
        "steady_speedup": ref.steady_solve_s
        / max(steady_solve_s, np.finfo(float).tiny),
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
        "full_operator_bytes": ref.bytes,
        "reduced_macro_bytes": csc_bytes(operators.K) + csc_bytes(operators.C),
    }


def close_data(data: Data) -> None:
    data.detail_ports_steady.close()
    data.detail_ports_transient.close()
    data.macro.ports.close()
    data.detail_steady.close()
    data.detail_transient.close()
    data.macro.compiled.close()
    data.full_layout.close()


def algebraic_self_test() -> dict[str, float | int | bool]:
    rng = np.random.default_rng(7)
    ports, internal = 9, 180
    gradient = sp.diags(
        (-np.ones(internal), np.ones(internal)),
        (0, 1),
        shape=(internal - 1, internal),
    )
    Kii = (gradient.T @ gradient + 0.08 * sp.eye(internal)).tocsc()
    Cii = sp.diags(0.8 + rng.random(internal), format="csc")
    Kip = sp.csc_matrix(rng.normal(size=(internal, ports)) * 0.02)
    Kpp = sp.diags(
        np.asarray(np.abs(Kip).sum(axis=0)).ravel() + 0.2, format="csc"
    )
    zero_pi = sp.csc_matrix((ports, internal))
    zero_ip = sp.csc_matrix((internal, ports))
    K = sp.bmat(((Kpp, Kip.T), (Kip, Kii)), format="csc")
    C = sp.bmat(
        ((sp.csc_matrix((ports, ports)), zero_pi), (zero_ip, Cii)), format="csc"
    )
    base = Operators(K, C, np.zeros(ports + internal))
    boundary_diag = sp.diags(
        np.r_[np.ones(ports), np.zeros(internal)], format="csc"
    )
    component = Operators(
        boundary_diag, sp.csc_matrix(K.shape), np.zeros(ports + internal)
    )
    snapshots = []
    inputs = np.eye(ports)
    for theta in (0.0, 0.5, 2.0):
        operators = combine_many(base, (component,), (theta,))
        for s in (0.0, 1.0, 10.0, 100.0):
            snapshots.append(
                normalized_snapshot_block(
                    transfer_snapshot(operators, ports, s, inputs)
                )
            )
    W, singular_values = svd_basis(np.column_stack(snapshots), 1.0e-9, 60)
    reduced = project(base, ports, sp.csc_matrix(W))
    symmetry = max(
        float(spla.norm(reduced.K - reduced.K.T)),
        float(spla.norm(reduced.C - reduced.C.T)),
    )
    orthogonality = float(np.linalg.norm(W.T @ W - np.eye(W.shape[1])))
    passed = symmetry < 1.0e-10 and orthogonality < 1.0e-10 and W.shape[1] < internal
    return {
        "full_internal_order": internal,
        "reduced_internal_order": W.shape[1],
        "leading_singular_value": float(singular_values[0]),
        "symmetry_defect": symmetry,
        "orthogonality_error": orthogonality,
        "passed": bool(passed),
    }


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
            error_K=0.15,
            duration_s=0.20,
            expansion_points=4,
            input_modes=20,
            svd_relative_tolerance=3.0e-7,
            residual_tolerance=2.0e-5,
            enrichment_block=8,
            max_internal_order=96,
            speedup_target=1.0,
            compression_target=5.0,
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
    mode.add_argument("--algebraic-self-test", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.algebraic_self_test:
        result = algebraic_self_test()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 3

    cfg, run, boundaries = configs(args.quick)
    print("=" * 100)
    print("Transient BCI-ROM - affine quadrant convection + adaptive rational SVD")
    print("=" * 100)
    print(
        f"Footprints cold plate/spreader/substrate/bump/die/TIM="
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
        basis = build_basis(data.macro, cfg, run)
        reduced = project_affine(data.macro, basis.W)
        compression = basis.W.shape[0] / max(basis.W.shape[1], 1)
        print(
            f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; exact interface ports={cfg.ports} "
            f"({cfg.port_shape[0]}x{cfg.port_shape[1]}); macro internal "
            f"{basis.W.shape[0]:,}->{basis.W.shape[1]:,} "
            f"({compression:.2f}x compression)"
        )
        print(
            f"Rational snapshots={basis.snapshot_columns}; SVD order "
            f"{basis.initial_order}->{basis.final_order}; residual history="
            + ", ".join(f"{value:.3e}" for value in basis.residual_history)
        )
        passivity = passivity_metrics(reduced, boundaries)

        results = []
        offline_s = assembly_s + basis.seconds + reduced.seconds
        for boundary in boundaries:
            result = evaluate(
                data,
                cfg,
                run,
                basis,
                reduced,
                boundary,
                reference(cfg, run, boundary),
            )
            result["accuracy_passed"] = (
                max(result["steady_error_K"], result["transient_error_K"])
                <= run.error_K
            )
            result["speedup_passed"] = (
                result["transient_speedup"] >= run.speedup_target
                if args.strict
                else True
            )
            result["passed"] = result["accuracy_passed"] and result["speedup_passed"]
            result["rom_offline_s"] = offline_s
            results.append(result)
            print(
                f"{boundary.name:>16s}: h={boundary.h_W_m2K}; error steady/transient="
                f"{result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K; "
                f"full/ROM transient={result['full_transient_solve_s']:.3f}/"
                f"{result['reduced_transient_solve_s']:.3f}s, "
                f"speedup={result['transient_speedup']:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= run.compression_target
        report = {
            "schema_version": 12,
            "mode": "quick" if args.quick else "strict",
            "method": (
                "exact-port BCI-DtN with four affine convection regions, "
                "block rational transfer sampling, SVD compression and "
                "residual-driven enrichment"
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
                "regions": ["southwest", "southeast", "northwest", "northeast"],
                "anchor_h_W_m2K": run.affine_anchor_h,
                "full_order_offline_assemblies": 5,
                "full_order_online_assemblies_per_case": 0,
                "capacitance_relative_change": data.macro.c_relative_change,
                "ambient_consistency_residual": data.macro.ambient_residual,
                "extraction_s": data.macro.seconds,
                "projection_s": reduced.seconds,
            },
            "reduction": {
                "physical_ports": cfg.ports,
                "full_internal_order": basis.W.shape[0],
                "initial_internal_order": basis.initial_order,
                "reduced_internal_order": basis.final_order,
                "internal_compression_ratio": compression,
                "compression_target": run.compression_target,
                "compression_passed": compression_passed,
                "basis_nnz": basis.W.nnz,
                "basis_density": basis.W.nnz
                / max(1, basis.W.shape[0] * basis.W.shape[1]),
                "snapshot_columns": basis.snapshot_columns,
                "input_modes": basis.input_modes,
                "training_boundary_count": basis.training_boundary_count,
                "expansion_points_per_s": basis.expansion_points_per_s.tolist(),
                "singular_value_head": basis.singular_values[:20].tolist(),
                "residual_history": basis.residual_history.tolist(),
                "worst_validation_residual": basis.worst_residual,
                "residual_target": run.residual_tolerance,
                "orthogonality_error": basis.orthogonality_error,
            },
            "passivity": passivity,
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(item["passed"] for item in results)
                and compression_passed
                and passivity["passed"]
                and basis.worst_residual <= run.residual_tolerance
            ),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            f"Passivity min eig(K/C)={passivity['minimum_K_eigenvalue']:.3e}/"
            f"{passivity['minimum_C_eigenvalue']:.3e}; "
            f"{'PASS' if passivity['passed'] else 'FAIL'}"
        )
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
