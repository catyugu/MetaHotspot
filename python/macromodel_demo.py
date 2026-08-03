#!/usr/bin/env python3
"""Transient boundary-condition-independent ROM benchmark for a package.

The detailed substrate/bump/die domain is coupled to a reusable TIM/spreader/
cold-plate macro model. The cold-plate top uses one uniform convection
coefficient. A column-local Galerkin basis preserves exact interface ports and
sparse nearest-neighbor coupling while remaining independent of that coefficient.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from functools import cached_property
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
MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("underfill", ".8", ".8", ".8", "1550", "1000"),
    ("copper", "390", "390", "390", "8960", "385"),
    ("silicon", "130", "130", "115", "2330", "700"),
    ("tim", "4", "4", "3", "2500", "900"),
    ("aluminum", "180", "180", "180", "2700", "900"),
)
ACTIVITY_TRACES = (
    (
        (0.00, 0.20),
        (0.10, 1.00),
        (0.35, 0.65),
        (0.58, 1.30),
        (0.82, 0.40),
        (1.00, 0.90),
    ),
    (
        (0.00, 0.75),
        (0.18, 1.20),
        (0.40, 0.30),
        (0.64, 1.05),
        (0.88, 0.55),
        (1.00, 0.80),
    ),
    (
        (0.00, 0.10),
        (0.08, 1.45),
        (0.28, 0.50),
        (0.52, 1.15),
        (0.76, 0.25),
        (1.00, 1.00),
    ),
    (
        (0.00, 0.55),
        (0.22, 0.35),
        (0.44, 1.25),
        (0.70, 0.60),
        (0.90, 1.10),
        (1.00, 0.70),
    ),
)


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    h_W_m2K: float


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
    def nz(self) -> int:
        return self.detail_nz + sum(cells for _, cells in self.macro_layers)

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

    @cached_property
    def axis_vertices_mm(self) -> np.ndarray:
        half_sizes = (
            self.cold_plate_size_mm / 2.0,
            self.spreader_size_mm / 2.0,
            self.bump_region_size_mm / 2.0,
            self.die_size_mm / 2.0,
        )
        points = [-value for value in half_sizes]
        points.extend((0.0, *reversed(half_sizes)))
        tile = self.chiplet_size_mm / 4.0
        for origin, _ in self.chiplet_origins_mm:
            points.extend(origin + tile * np.arange(5, dtype=np.float64))
        return refined_breakpoints(points, self.max_xy_cell_mm)

    @cached_property
    def port_indices(self) -> np.ndarray:
        return footprint_cell_indices(self.axis_vertices_mm, self.tim_size_mm)

    @property
    def nx(self) -> int:
        return self.axis_vertices_mm.size - 1

    ny = nx

    @property
    def port_shape(self) -> tuple[int, int]:
        count = self.port_indices.size
        return count, count

    @property
    def ports(self) -> int:
        return self.port_indices.size**2

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
    report: Path = Path("results/bci_rom_uniform_convection_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s


class MacroAffine(NamedTuple):
    compiled: object
    ports: PortMap
    anchor_h: float
    base: Operators
    convection: Operators
    seconds: float

    def at(self, h_W_m2K: float) -> Operators:
        return interpolate_operators(
            self.base, self.convection, float(h_W_m2K) / self.anchor_h
        )


class Data(NamedTuple):
    full_layout: object
    detail_steady: object
    detail_transient: object
    detail_ports_steady: PortMap
    detail_ports_transient: PortMap
    macro: MacroAffine
    detail_to_full: np.ndarray
    macro_to_full: np.ndarray


class Column(NamedTuple):
    cells: np.ndarray
    port: int | None


class Basis(NamedTuple):
    W: sp.csc_matrix
    orders: np.ndarray
    initial_internal: np.ndarray
    seconds: float


class ReducedAffine(NamedTuple):
    anchor_h: float
    base: Operators
    convection: Operators
    seconds: float

    def at(self, h_W_m2K: float) -> tuple[Operators, float]:
        started = time.perf_counter()
        operators = interpolate_operators(
            self.base, self.convection, float(h_W_m2K) / self.anchor_h
        )
        return operators, time.perf_counter() - started


class Reference(NamedTuple):
    steady: np.ndarray
    times: np.ndarray
    transient: np.ndarray
    compile_s: float
    steady_solve_s: float
    transient_solve_s: float
    order: int


def refined_breakpoints(points: Sequence[float], max_step: float) -> np.ndarray:
    unique = np.unique(np.asarray(points, dtype=np.float64))
    output = [float(unique[0])]
    for left, right in zip(unique[:-1], unique[1:]):
        pieces = max(1, math.ceil((right - left) / max_step))
        output.extend(np.linspace(left, right, pieces + 1)[1:])
    return np.asarray(output)


def footprint_cell_indices(vertices: np.ndarray, size: float) -> np.ndarray:
    half = size / 2.0
    tolerance = 1.0e-10 * max(1.0, size)
    return np.flatnonzero(
        (vertices[:-1] >= -half - tolerance) & (vertices[1:] <= half + tolerance)
    ).astype(np.int64)


def z_vertices(layers) -> np.ndarray:
    output = [0.0]
    z = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            output.append(z)
    return np.asarray(output)


def add_rect(model, block: int, size: float) -> None:
    half = size / 2.0
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{-half:.17g}",
        f"{-half:.17g}",
        f"{size:.17g}",
        f"{size:.17g}",
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
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, z_vertices(layers))
    for material in MATERIALS:
        model.add_material(*material)


def add_macro(model, cfg: Package) -> None:
    for thickness, material, size in (
        (cfg.cold_plate_mm, "aluminum", cfg.cold_plate_size_mm),
        (cfg.spreader_mm, "copper", cfg.spreader_size_mm),
        (cfg.tim_mm, "tim", cfg.tim_size_mm),
    ):
        layer = model.add_layer(str(thickness))
        add_rect(model, model.add_block(layer, material), size)


def add_detail(model, cfg: Package, study: Study, run: Run) -> None:
    die = model.add_layer(str(cfg.die_mm))
    add_rect(model, model.add_block(die, "silicon"), cfg.die_size_mm)
    if study == Study.TRANSIENT:
        for index, trace in enumerate(ACTIVITY_TRACES):
            model.add_function_piecewise(
                f"activity_{index}",
                np.asarray(
                    [(fraction * run.duration_s, value) for fraction, value in trace]
                ),
            )

    tile = cfg.chiplet_size_mm / 4.0
    tile_volume = tile * tile * cfg.die_mm * 1.0e-9
    for chiplet, ((x0, y0), scale) in enumerate(
        zip(cfg.chiplet_origins_mm, CHIPLET_POWER_SCALE)
    ):
        for iy in range(4):
            for ix in range(4):
                source = cfg.chiplet_power_W * scale * POWER_MAP[iy, ix]
                expression = f"{source / POWER_MAP.size / tile_volume:.17g}"
                if study == Study.TRANSIENT:
                    expression += f"*activity_{(chiplet + 2 * ix + iy) % 4}(x)"
                block = model.add_block(die, "silicon", heat_source=expression)
                model.add_rect(
                    block,
                    GeometryOp.ADD,
                    f"{x0 + ix * tile:.17g}",
                    f"{y0 + iy * tile:.17g}",
                    f"{tile:.17g}",
                    f"{tile:.17g}",
                )

    bump = model.add_layer(str(cfg.bump_mm))
    add_rect(model, model.add_block(bump, "underfill"), cfg.bump_region_size_mm)
    pitch_x = cfg.die_size_mm / cfg.bump_columns
    pitch_y = cfg.die_size_mm / cfg.bump_rows
    origin = -cfg.die_size_mm / 2.0
    for iy in range(cfg.bump_rows):
        for ix in range(cfg.bump_columns):
            x = origin + (ix + 0.5) * pitch_x - cfg.bump_width_mm / 2.0
            y = origin + (iy + 0.5) * pitch_y - cfg.bump_width_mm / 2.0
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
    add_rect(model, model.add_block(substrate, "organic"), cfg.substrate_size_mm)


def add_convection(model, cfg: Package, z: float, h_W_m2K: float | None) -> None:
    if h_W_m2K is None:
        return
    h = float(h_W_m2K)
    if h < 0.0:
        raise ValueError("convection coefficient must be non-negative")
    if h == 0.0:
        return
    half = cfg.cold_plate_size_mm / 2.0
    region = (Axis.Z, z, -half, half, -half, half)
    model.add_convection(str(h), str(cfg.ambient_K), [region])


def build_model(
    cfg: Package,
    study: Study,
    *,
    run: Run | None = None,
    detail: bool,
    macro: bool,
    h_W_m2K: float | None = None,
):
    if detail and run is None:
        raise ValueError("detail models require a Run configuration")
    model = metahotspot.Model()
    layers = (
        (*cfg.detail_layers, *cfg.macro_layers)
        if detail and macro
        else (cfg.detail_layers if detail else cfg.macro_layers)
    )
    configure(model, cfg, layers, study, run)
    if macro:
        add_macro(model, cfg)
    if detail:
        add_detail(model, cfg, study, run)
    model.set_default_neumann("0")
    if macro:
        z = cfg.total_height_mm if detail else cfg.macro_height_mm
        add_convection(model, cfg, z, h_W_m2K)
    return model


def port_patches(cfg: Package, face: Face, z: float) -> list[PortPatch]:
    vertices = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(face),
            z,
            (vertices[ix], vertices[ix + 1], vertices[iy], vertices[iy + 1]),
        )
        for ix in cfg.port_indices
        for iy in cfg.port_indices
    ]


def clean_operators(operators: Operators) -> Operators:
    K, C = sp.csc_matrix(operators.K), sp.csc_matrix(operators.C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(operators.f, dtype=np.float64).copy())


def operator_delta(a: Operators, b: Operators) -> Operators:
    return clean_operators(Operators(a.K - b.K, a.C - b.C, np.asarray(a.f) - b.f))


def interpolate_operators(
    base: Operators, convection: Operators, coordinate: float
) -> Operators:
    if not np.isfinite(coordinate) or coordinate < 0.0:
        raise ValueError("normalized convection coordinate must be non-negative")
    if coordinate == 0.0:
        return clean_operators(base)
    return clean_operators(
        Operators(
            base.K + coordinate * convection.K,
            base.C + coordinate * convection.C,
            np.asarray(base.f) + coordinate * convection.f,
        )
    )


def extract_affine_macro(cfg: Package, anchor_h: float) -> MacroAffine:
    if anchor_h <= 0.0:
        raise ValueError("affine_anchor_h must be positive")
    started = time.perf_counter()
    patches = port_patches(cfg, Face.ZM, 0.0)
    compiled = build_model(cfg, Study.STEADY, detail=False, macro=True).compile()
    ports = PortMap(compiled, patches)
    anchor_compiled = anchor_ports = None
    try:
        base = clean_operators(ports.assemble())
        anchor_compiled = build_model(
            cfg,
            Study.STEADY,
            detail=False,
            macro=True,
            h_W_m2K=anchor_h,
        ).compile()
        anchor_ports = PortMap(anchor_compiled, patches)
        anchor = clean_operators(anchor_ports.assemble())
        if anchor.K.shape != base.K.shape:
            raise RuntimeError("convection changed macro state ordering")
        convection = operator_delta(anchor, base)

        ambient = np.full(base.K.shape[0], cfg.ambient_K)
        defect = convection.K @ ambient - convection.f
        scale = max(np.linalg.norm(convection.f), np.finfo(float).tiny)
        if spla.norm(convection.C) > 1.0e-11 * max(spla.norm(base.C), 1.0):
            raise RuntimeError("convection unexpectedly changed macro capacitance")
        if np.linalg.norm(defect) > 1.0e-10 * scale:
            raise RuntimeError("affine convection component violates ambient balance")
        return MacroAffine(
            compiled,
            ports,
            anchor_h,
            base,
            convection,
            time.perf_counter() - started,
        )
    except Exception:
        ports.close()
        compiled.close()
        raise
    finally:
        if anchor_ports is not None:
            anchor_ports.close()
        if anchor_compiled is not None:
            anchor_compiled.close()


def cell_grid(compiled) -> np.ndarray:
    return compiled.grid_to_cell.reshape(compiled.nx, compiled.ny, compiled.nz)


def coordinate_cell_map(source, target, z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: lateral meshes differ")
    source_grid = cell_grid(source)
    target_grid = cell_grid(target)[:, :, z_offset : z_offset + source.nz]
    if target_grid.shape != source_grid.shape:
        raise RuntimeError(f"{label}: z range differs")
    valid = source_grid >= 0
    if not np.array_equal(valid, target_grid >= 0):
        raise RuntimeError(f"{label}: geometry occupancy differs")
    source_ids, target_ids = source_grid[valid], target_grid[valid]
    if (
        source_ids.size != source.cell_count
        or np.unique(source_ids).size != source.cell_count
    ):
        raise RuntimeError(f"{label}: source cell IDs are incomplete")
    mapping = np.empty(source.cell_count, dtype=np.int64)
    mapping[source_ids] = target_ids
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(f"{label}: target mapping is not one-to-one")
    return mapping


def assemble(cfg: Package, run: Run) -> Data:
    full = steady = transient = None
    steady_ports = transient_ports = None
    macro = None
    try:
        full = build_model(
            cfg, Study.STEADY, run=run, detail=True, macro=True
        ).compile()
        steady = build_model(
            cfg, Study.STEADY, run=run, detail=True, macro=False
        ).compile()
        transient = build_model(
            cfg, Study.TRANSIENT, run=run, detail=True, macro=False
        ).compile()
        patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
        steady_ports = PortMap(steady, patches)
        transient_ports = PortMap(transient, patches)
        macro = extract_affine_macro(cfg, run.affine_anchor_h)
        if macro.ports.port_count != cfg.ports:
            raise RuntimeError("configured interface port count is inconsistent")
        detail_map = coordinate_cell_map(steady, full, 0, "detail/full")
        transient_map = coordinate_cell_map(transient, full, 0, "transient/full")
        if not np.array_equal(detail_map, transient_map):
            raise RuntimeError("steady and transient detail orderings differ")
        macro_map = coordinate_cell_map(
            macro.compiled, full, cfg.detail_nz, "macro/full"
        )
        combined = np.r_[detail_map, macro_map]
        if (
            combined.size != full.cell_count
            or np.unique(combined).size != full.cell_count
        ):
            raise RuntimeError("detail and macro maps do not partition the full model")
        return Data(
            full,
            steady,
            transient,
            steady_ports,
            transient_ports,
            macro,
            detail_map,
            macro_map,
        )
    except Exception:
        if steady_ports is not None:
            steady_ports.close()
        if transient_ports is not None:
            transient_ports.close()
        if macro is not None:
            macro.ports.close()
            macro.compiled.close()
        for compiled in (steady, transient, full):
            if compiled is not None:
                compiled.close()
        raise


def internal_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, :ports].tocsc(),
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, :ports].tocsc(),
        operators.C[ports:, ports:].tocsc(),
    )


def macro_columns(compiled, cfg: Package) -> tuple[Column, ...]:
    pairs = ((int(ix), int(iy)) for ix in cfg.port_indices for iy in cfg.port_indices)
    port_lookup = {pair: index for index, pair in enumerate(pairs)}
    grid = cell_grid(compiled)
    output = []
    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = grid[ix, iy]
            cells = cells[cells >= 0]
            if cells.size:
                output.append(Column(cells.astype(np.int64), port_lookup.get((ix, iy))))
    if sum(column.port is not None for column in output) != cfg.ports:
        raise RuntimeError("interface-port/column mapping is inconsistent")
    return tuple(output)


def range_basis(candidates: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(candidates)
    q, r, _ = scipy.linalg.qr(
        matrix, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0))
    keep = diagonal > np.finfo(float).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def build_basis(macro: MacroAffine, cfg: Package, run: Run) -> Basis:
    started = time.perf_counter()
    ports = macro.ports.port_count
    Kip0, Kii0, Cip0, Cii0 = internal_blocks(macro.base, ports)
    Kip1, Kii1, _, _ = internal_blocks(macro.convection, ports)
    columns = macro_columns(macro.compiled, cfg)
    rows, cols, values, orders = [], [], [], []
    offset = 0

    for column in columns:
        cells = column.cells
        k0 = Kii0[cells][:, cells].toarray()
        c0 = Cii0[cells][:, cells].toarray()
        candidates = [np.ones(cells.size)]

        mode_count = min(run.local_dynamic_modes, cells.size)
        if mode_count:
            eigenvalues, modes = scipy.linalg.eigh(
                k0,
                c0,
                subset_by_index=(0, mode_count - 1),
                check_finite=False,
            )
            candidates.extend(modes[:, eigenvalues <= run.modal_cutoff_per_s].T)

        if column.port is not None:
            port = column.port
            b0 = Kip0[cells, port].toarray().ravel()
            cp0 = Cip0[cells, port].toarray().ravel()
            static = scipy.linalg.solve(k0, -b0, assume_a="sym", check_finite=False)
            candidates.append(static)

            sensitivity_rhs = (
                Kii1[cells][:, cells] @ static + Kip1[cells, port].toarray().ravel()
            )
            if np.linalg.norm(sensitivity_rhs) > 1.0e-14 * max(np.linalg.norm(b0), 1.0):
                candidates.append(
                    scipy.linalg.solve(
                        k0,
                        -sensitivity_rhs,
                        assume_a="sym",
                        check_finite=False,
                    )
                )

            for multiplier in run.bdf1_shifts:
                shift = multiplier / run.dt_s
                response = scipy.linalg.solve(
                    k0 + shift * c0,
                    -(b0 + shift * cp0),
                    assume_a="sym",
                    check_finite=False,
                )
                candidates.append(response - static)

        local = range_basis(candidates)
        orders.append(local.shape[1])
        for local_row, cell in enumerate(cells):
            nonzero = np.flatnonzero(np.abs(local[local_row]) > 1.0e-14)
            rows.extend([int(cell)] * nonzero.size)
            cols.extend((offset + nonzero).tolist())
            values.extend(local[local_row, nonzero].tolist())
        offset += local.shape[1]

    W = sp.csc_matrix((values, (rows, cols)), shape=(Kii0.shape[0], offset))
    ones = np.ones(W.shape[0])
    if np.linalg.norm(W @ (W.T @ ones) - ones) > 1.0e-10 * math.sqrt(ones.size):
        raise RuntimeError("macro basis does not preserve uniform temperature")
    gram_error = spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc"))
    if gram_error > 1.0e-10:
        raise RuntimeError("macro basis lost orthogonality")
    initial = np.asarray(W.T @ np.full(W.shape[0], cfg.ambient_K)).ravel()
    return Basis(W, np.asarray(orders), initial, time.perf_counter() - started)


def project_matrix(matrix, ports: int, W: sp.csc_matrix) -> sp.csc_matrix:
    port = matrix[:ports, :ports].tocsc()
    upper = (matrix[:ports, ports:] @ W).tocsc()
    lower = (W.T @ matrix[ports:, :ports]).tocsc()
    internal = (W.T @ matrix[ports:, ports:] @ W).tocsc()
    reduced = sp.bmat(((port, upper), (lower, internal)), format="csc")
    reduced = (0.5 * (reduced + reduced.T)).tocsc()
    reduced.eliminate_zeros()
    return reduced


def project(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    return Operators(
        project_matrix(operators.K, ports, W),
        project_matrix(operators.C, ports, W),
        np.r_[operators.f[:ports], np.asarray(W.T @ operators.f[ports:]).ravel()],
    )


def project_affine(macro: MacroAffine, basis: Basis) -> ReducedAffine:
    started = time.perf_counter()
    ports = macro.ports.port_count
    return ReducedAffine(
        macro.anchor_h,
        project(macro.base, ports, basis.W),
        project(macro.convection, ports, basis.W),
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
    steady = transient = None
    try:
        steady = build_model(
            cfg,
            Study.STEADY,
            run=run,
            detail=True,
            macro=True,
            h_W_m2K=boundary.h_W_m2K,
        ).compile()
        transient = build_model(
            cfg,
            Study.TRANSIENT,
            run=run,
            detail=True,
            macro=True,
            h_W_m2K=boundary.h_W_m2K,
        ).compile()
        compile_s = time.perf_counter() - started
        started = time.perf_counter()
        with steady.solve(opts=solve_options(run, False)) as solution:
            steady_temperature = np.asarray(solution.temperature).copy()
        steady_s = time.perf_counter() - started
        started = time.perf_counter()
        with transient.solve(opts=solve_options(run, True)) as solution:
            times = np.asarray(solution.history_times).copy()
            history = np.asarray(solution.temperature_history).copy()
        transient_s = time.perf_counter() - started
        return Reference(
            steady_temperature,
            times,
            history,
            compile_s,
            steady_s,
            transient_s,
            transient.cell_count,
        )
    finally:
        if steady is not None:
            steady.close()
        if transient is not None:
            transient.close()


def evaluate(
    data: Data,
    cfg: Package,
    run: Run,
    basis: Basis,
    reduced: ReducedAffine,
    boundary: BoundaryCase,
    ref: Reference,
):
    operators, assembly_s = reduced.at(boundary.h_W_m2K)

    def run_reduced(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count + cfg.ports, cfg.ambient_K),
            basis.initial_internal,
        ]
        started = time.perf_counter()
        with solve_macro(
            compiled, operators, ports, state, solve_options(run, transient)
        ) as solution:
            elapsed = time.perf_counter() - started
            if transient:
                return (
                    np.asarray(solution.history_times).copy(),
                    np.asarray(solution.state_history).copy(),
                    elapsed,
                )
            return np.asarray(solution.state).copy(), elapsed

    steady_state, reduced_steady_s = run_reduced(False)
    times, transient_states, reduced_transient_s = run_reduced(True)
    if times.shape != ref.times.shape or not np.allclose(
        times, ref.times, atol=1.0e-12, rtol=0.0
    ):
        raise RuntimeError("full and reduced output times differ")

    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        output = np.empty((states.shape[0], data.full_layout.cell_count))
        output[:, data.detail_to_full] = states[:, :detail_n]
        output[:, data.macro_to_full] = (
            basis.W @ states[:, detail_n + cfg.ports :].T
        ).T
        return output

    steady_error = float(np.max(np.abs(recover(steady_state)[0] - ref.steady)))
    transient_error = float(np.max(np.abs(recover(transient_states) - ref.transient)))
    return {
        "name": boundary.name,
        "h_W_m2K": boundary.h_W_m2K,
        "steady_error_K": steady_error,
        "transient_error_K": transient_error,
        "online_reduced_assembly_s": assembly_s,
        "full_compile_s": ref.compile_s,
        "full_steady_solve_s": ref.steady_solve_s,
        "reduced_steady_solve_s": reduced_steady_s,
        "full_transient_solve_s": ref.transient_solve_s,
        "reduced_transient_solve_s": reduced_transient_s,
        "transient_speedup": ref.transient_solve_s
        / max(reduced_transient_s, np.finfo(float).tiny),
        "full_order": ref.order,
        "reduced_online_order": detail_n + operators.K.shape[0],
        "reduced_macro_k_nnz": operators.K.nnz,
        "reduced_macro_c_nnz": operators.C.nnz,
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
        return (
            cfg,
            run,
            (
                BoundaryCase("uniform-low", 500.0),
                BoundaryCase("uniform-high", 8000.0),
            ),
        )
    return (
        Package(),
        Run(),
        (
            BoundaryCase("uniform-low", 500.0),
            BoundaryCase("uniform-medium", 2500.0),
            BoundaryCase("uniform-high", 8000.0),
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run, boundaries = configs(args.quick)

    print("=" * 100)
    print("Transient BCI-ROM - uniform cold-plate convection")
    print("=" * 100)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/"
        f"{cfg.bump_region_size_mm:g}/{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
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
        for boundary in boundaries:
            result = evaluate(
                data, cfg, run, basis, reduced, boundary, reference(cfg, run, boundary)
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
            results.append(result)
            print(
                f"{boundary.name:>16s}: h={boundary.h_W_m2K:g} W/(m^2 K), "
                f"error steady/transient={result['steady_error_K']:.5f}/"
                f"{result['transient_error_K']:.5f} K; full/ROM="
                f"{result['full_transient_solve_s']:.3f}/"
                f"{result['reduced_transient_solve_s']:.3f}s, "
                f"speedup={result['transient_speedup']:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= run.compression_target
        report = {
            "schema_version": 17,
            "mode": "quick" if args.quick else "strict",
            "method": (
                "uniform-convection column-local static/sensitivity/BDF1 "
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
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "experiment": {**asdict(run), "report": str(run.report)},
            "affine_boundary": {
                "family": "A(h)=A0+(h/anchor_h)*DeltaA_h",
                "region": "entire cold-plate top surface",
                "anchor_h_W_m2K": run.affine_anchor_h,
                "full_order_offline_assemblies": 2,
                "full_order_online_assemblies_per_case": 0,
                "extraction_s": data.macro.seconds,
                "projection_s": reduced.seconds,
            },
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "compression_ratio": compression,
                "compression_target": run.compression_target,
                "compression_passed": compression_passed,
                "column_count": basis.orders.size,
                "port_columns": cfg.ports,
                "local_order_min": int(basis.orders.min()),
                "local_order_mean": float(basis.orders.mean()),
                "local_order_max": int(basis.orders.max()),
                "basis_nnz": basis.W.nnz,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence",
            },
            "offline_s": assembly_s + basis.seconds + reduced.seconds,
            "boundary_reuse": results,
            "passed": bool(
                all(item["passed"] for item in results) and compression_passed
            ),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
