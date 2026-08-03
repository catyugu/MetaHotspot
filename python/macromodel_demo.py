#!/usr/bin/env python3
"""Extract and validate a transient boundary-condition-independent thermal ROM.

The detailed substrate/bump/die model is coupled to a reusable
TIM/spreader/cold-plate macro model. The ROM keeps every interface temperature
as an exact physical port and reduces only the macro internal cells. Uniform
cold-plate convection is represented by an affine operator, so the basis is
extracted once and reused for every tested convection coefficient.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path

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
BOUNDARIES = (
    ("uniform-low", 500.0),
    ("uniform-medium", 2500.0),
    ("uniform-high", 8000.0),
)


@dataclass(frozen=True)
class Config:
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

        unique = np.unique(np.asarray(points, dtype=np.float64))
        vertices = [float(unique[0])]
        for left, right in zip(unique[:-1], unique[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_xy_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @cached_property
    def port_indices(self) -> np.ndarray:
        vertices = self.axis_vertices_mm
        half = self.tim_size_mm / 2.0
        tolerance = 1.0e-10 * max(1.0, self.tim_size_mm)
        return np.flatnonzero(
            (vertices[:-1] >= -half - tolerance) & (vertices[1:] <= half + tolerance)
        ).astype(np.int64)

    @property
    def nx(self) -> int:
        return self.axis_vertices_mm.size - 1

    @property
    def ports(self) -> int:
        return self.port_indices.size**2

    @property
    def nominal_power_W(self) -> float:
        return self.chiplet_power_W * float(sum(CHIPLET_POWER_SCALE))


QUICK_OVERRIDES = dict(
    substrate_cells=3,
    bump_cells=1,
    die_cells=2,
    tim_cells=1,
    spreader_cells=3,
    cold_plate_cells=4,
    max_xy_cell_mm=6.0,
    bump_rows=8,
    bump_columns=8,
    error_K=0.35,
    duration_s=0.20,
    local_dynamic_modes=1,
    bdf1_shifts=(1.0,),
    speedup_target=1.0,
    compression_target=2.0,
)


def z_vertices(layers) -> np.ndarray:
    vertices = [0.0]
    z = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            vertices.append(z)
    return np.asarray(vertices)


def add_square(model, block: int, size_mm: float) -> None:
    half = size_mm / 2.0
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{-half:.17g}",
        f"{-half:.17g}",
        f"{size_mm:.17g}",
        f"{size_mm:.17g}",
    )


def build_model(
    cfg: Config,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    convection_h: float | None = None,
):
    """Build a full, detail-only, or macro-only model on the shared mesh."""
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    if convection_h is not None and convection_h < 0.0:
        raise ValueError("convection coefficient must be non-negative")

    model = metahotspot.Model()
    layers = (
        (*cfg.detail_layers, *cfg.macro_layers)
        if detail and macro
        else (cfg.detail_layers if detail else cfg.macro_layers)
    )
    transient = study == Study.TRANSIENT
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=cfg.duration_s if transient else 0.0,
        output_interval=cfg.dt_s if transient else 0.0,
    )
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, z_vertices(layers))
    for material in MATERIALS:
        model.add_material(*material)

    if macro:
        for thickness, material, size in (
            (cfg.cold_plate_mm, "aluminum", cfg.cold_plate_size_mm),
            (cfg.spreader_mm, "copper", cfg.spreader_size_mm),
            (cfg.tim_mm, "tim", cfg.tim_size_mm),
        ):
            layer = model.add_layer(str(thickness))
            add_square(model, model.add_block(layer, material), size)

    if detail:
        die = model.add_layer(str(cfg.die_mm))
        add_square(model, model.add_block(die, "silicon"), cfg.die_size_mm)
        if transient:
            for index, trace in enumerate(ACTIVITY_TRACES):
                model.add_function_piecewise(
                    f"activity_{index}",
                    np.asarray(
                        [
                            (fraction * cfg.duration_s, value)
                            for fraction, value in trace
                        ]
                    ),
                )

        tile = cfg.chiplet_size_mm / 4.0
        tile_volume_m3 = tile * tile * cfg.die_mm * 1.0e-9
        for chiplet, ((x0, y0), scale) in enumerate(
            zip(cfg.chiplet_origins_mm, CHIPLET_POWER_SCALE)
        ):
            for iy in range(4):
                for ix in range(4):
                    tile_power = (
                        cfg.chiplet_power_W * scale * POWER_MAP[iy, ix] / POWER_MAP.size
                    )
                    source = f"{tile_power / tile_volume_m3:.17g}"
                    if transient:
                        source += f"*activity_{(chiplet + 2 * ix + iy) % 4}(x)"
                    block = model.add_block(die, "silicon", heat_source=source)
                    model.add_rect(
                        block,
                        GeometryOp.ADD,
                        f"{x0 + ix * tile:.17g}",
                        f"{y0 + iy * tile:.17g}",
                        f"{tile:.17g}",
                        f"{tile:.17g}",
                    )

        bump = model.add_layer(str(cfg.bump_mm))
        add_square(
            model,
            model.add_block(bump, "underfill"),
            cfg.bump_region_size_mm,
        )
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
        add_square(
            model,
            model.add_block(substrate, "organic"),
            cfg.substrate_size_mm,
        )

    model.set_default_neumann("0")
    if macro and convection_h:
        half = cfg.cold_plate_size_mm / 2.0
        top_z = cfg.total_height_mm if detail else cfg.macro_height_mm
        model.add_convection(
            str(float(convection_h)),
            str(cfg.ambient_K),
            [(Axis.Z, top_z, -half, half, -half, half)],
        )
    return model


def port_patches(cfg: Config, face: Face, z_m: float) -> list[PortPatch]:
    vertices = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(face),
            z_m,
            (vertices[ix], vertices[ix + 1], vertices[iy], vertices[iy + 1]),
        )
        for ix in cfg.port_indices
        for iy in cfg.port_indices
    ]


def normalized_operators(K, C, f) -> Operators:
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(f, dtype=np.float64).copy())


def affine_operators(base: Operators, delta: Operators, alpha: float) -> Operators:
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("normalized convection coordinate must be non-negative")
    return normalized_operators(
        base.K + alpha * delta.K,
        base.C + alpha * delta.C,
        np.asarray(base.f) + alpha * delta.f,
    )


def grid_cells(compiled) -> np.ndarray:
    return compiled.grid_to_cell.reshape(compiled.nx, compiled.ny, compiled.nz)


def coordinate_map(source, target, z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: lateral meshes differ")
    source_grid = grid_cells(source)
    target_grid = grid_cells(target)[:, :, z_offset : z_offset + source.nz]
    if target_grid.shape != source_grid.shape:
        raise RuntimeError(f"{label}: z range differs")
    valid = source_grid >= 0
    if not np.array_equal(valid, target_grid >= 0):
        raise RuntimeError(f"{label}: geometry occupancy differs")

    source_ids = source_grid[valid]
    target_ids = target_grid[valid]
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


def column_basis(
    compiled,
    cfg: Config,
    base: Operators,
    delta: Operators,
    port_count: int,
):
    """Build a block-local basis without source or boundary-response snapshots."""
    started = time.perf_counter()
    K_ip = base.K[port_count:, :port_count].tocsc()
    K_ii = base.K[port_count:, port_count:].tocsc()
    C_ip = base.C[port_count:, :port_count].tocsc()
    C_ii = base.C[port_count:, port_count:].tocsc()
    dK_ip = delta.K[port_count:, :port_count].tocsc()
    dK_ii = delta.K[port_count:, port_count:].tocsc()

    port_lookup = {
        (int(ix), int(iy)): port
        for port, (ix, iy) in enumerate(
            (ix, iy) for ix in cfg.port_indices for iy in cfg.port_indices
        )
    }
    grid = grid_cells(compiled)
    seen_ports = 0
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    orders: list[int] = []
    offset = 0

    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = grid[ix, iy]
            cells = cells[cells >= 0].astype(np.int64)
            if not cells.size:
                continue

            k = K_ii[cells][:, cells].toarray()
            c = C_ii[cells][:, cells].toarray()
            candidates = [np.ones(cells.size)]

            mode_count = min(cfg.local_dynamic_modes, cells.size)
            if mode_count:
                eigenvalues, modes = scipy.linalg.eigh(
                    k,
                    c,
                    subset_by_index=(0, mode_count - 1),
                    check_finite=False,
                )
                cutoff = math.pi / cfg.dt_s
                candidates.extend(modes[:, eigenvalues <= cutoff].T)

            port = port_lookup.get((ix, iy))
            if port is not None:
                seen_ports += 1
                b = K_ip[cells, port].toarray().ravel()
                cp = C_ip[cells, port].toarray().ravel()
                static = scipy.linalg.solve(k, -b, assume_a="sym", check_finite=False)
                candidates.append(static)

                sensitivity_rhs = (
                    dK_ii[cells][:, cells] @ static
                    + dK_ip[cells, port].toarray().ravel()
                )
                if np.linalg.norm(sensitivity_rhs) > 1.0e-14 * max(
                    np.linalg.norm(b), 1.0
                ):
                    candidates.append(
                        scipy.linalg.solve(
                            k,
                            -sensitivity_rhs,
                            assume_a="sym",
                            check_finite=False,
                        )
                    )

                for multiplier in cfg.bdf1_shifts:
                    shift = multiplier / cfg.dt_s
                    response = scipy.linalg.solve(
                        k + shift * c,
                        -(b + shift * cp),
                        assume_a="sym",
                        check_finite=False,
                    )
                    candidates.append(response - static)

            matrix = np.column_stack(candidates)
            q, r, _ = scipy.linalg.qr(
                matrix, mode="economic", pivoting=True, check_finite=False
            )
            diagonal = np.abs(np.diag(r))
            if not diagonal.size or diagonal[0] == 0.0:
                local = np.empty((cells.size, 0))
            else:
                keep = diagonal > (
                    np.finfo(float).eps * max(matrix.shape) * diagonal[0]
                )
                local = np.ascontiguousarray(q[:, keep])

            orders.append(local.shape[1])
            for local_row, cell in enumerate(cells):
                nonzero = np.flatnonzero(np.abs(local[local_row]) > 1.0e-14)
                rows.extend([int(cell)] * nonzero.size)
                cols.extend((offset + nonzero).tolist())
                values.extend(local[local_row, nonzero].tolist())
            offset += local.shape[1]

    if seen_ports != port_count:
        raise RuntimeError("interface-port/column mapping is inconsistent")

    W = sp.csc_matrix((values, (rows, cols)), shape=(K_ii.shape[0], offset))
    ones = np.ones(W.shape[0])
    if np.linalg.norm(W @ (W.T @ ones) - ones) > 1.0e-10 * math.sqrt(ones.size):
        raise RuntimeError("macro basis does not preserve uniform temperature")
    if spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc")) > 1.0e-10:
        raise RuntimeError("macro basis lost orthogonality")

    initial = np.asarray(W.T @ np.full(W.shape[0], cfg.ambient_K)).ravel()
    return W, np.asarray(orders), initial, time.perf_counter() - started


def project_operators(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    def project_matrix(matrix):
        reduced = sp.bmat(
            (
                (
                    matrix[:ports, :ports],
                    matrix[:ports, ports:] @ W,
                ),
                (
                    W.T @ matrix[ports:, :ports],
                    W.T @ matrix[ports:, ports:] @ W,
                ),
            ),
            format="csc",
        )
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project_matrix(operators.K),
        project_matrix(operators.C),
        np.r_[
            operators.f[:ports],
            np.asarray(W.T @ operators.f[ports:]).ravel(),
        ],
    )


def solve_options(cfg: Config, transient: bool) -> SolveOptions:
    dt = cfg.dt_s if transient else 1.0
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


def full_reference(cfg: Config, convection_h: float):
    started = time.perf_counter()
    with ExitStack() as stack:
        steady = build_model(
            cfg,
            Study.STEADY,
            detail=True,
            macro=True,
            convection_h=convection_h,
        ).compile()
        stack.callback(steady.close)
        transient = build_model(
            cfg,
            Study.TRANSIENT,
            detail=True,
            macro=True,
            convection_h=convection_h,
        ).compile()
        stack.callback(transient.close)
        compile_s = time.perf_counter() - started

        started = time.perf_counter()
        with steady.solve(opts=solve_options(cfg, False)) as solution:
            steady_temperature = np.asarray(solution.temperature).copy()
        steady_s = time.perf_counter() - started

        started = time.perf_counter()
        with transient.solve(opts=solve_options(cfg, True)) as solution:
            times = np.asarray(solution.history_times).copy()
            history = np.asarray(solution.temperature_history).copy()
        transient_s = time.perf_counter() - started

        return (
            steady_temperature,
            times,
            history,
            compile_s,
            steady_s,
            transient_s,
            transient.cell_count,
        )


def run_experiment(cfg: Config, boundaries, strict: bool) -> dict:
    offline_started = time.perf_counter()
    with ExitStack() as stack:
        full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
        stack.callback(full_layout.close)
        detail_steady = build_model(
            cfg, Study.STEADY, detail=True, macro=False
        ).compile()
        stack.callback(detail_steady.close)
        detail_transient = build_model(
            cfg, Study.TRANSIENT, detail=True, macro=False
        ).compile()
        stack.callback(detail_transient.close)

        detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
        detail_ports_steady = PortMap(detail_steady, detail_patches)
        stack.callback(detail_ports_steady.close)
        detail_ports_transient = PortMap(detail_transient, detail_patches)
        stack.callback(detail_ports_transient.close)

        macro_started = time.perf_counter()
        macro_compiled = build_model(
            cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        stack.callback(macro_compiled.close)
        macro_ports = PortMap(macro_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(macro_ports.close)
        base = normalized_operators(*macro_ports.assemble())

        anchor_compiled = build_model(
            cfg,
            Study.STEADY,
            detail=False,
            macro=True,
            convection_h=cfg.affine_anchor_h,
        ).compile()
        stack.callback(anchor_compiled.close)
        anchor_ports = PortMap(anchor_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(anchor_ports.close)
        anchor = normalized_operators(*anchor_ports.assemble())
        if anchor.K.shape != base.K.shape:
            raise RuntimeError("convection changed macro state ordering")
        delta = normalized_operators(
            anchor.K - base.K,
            anchor.C - base.C,
            np.asarray(anchor.f) - base.f,
        )
        macro_extraction_s = time.perf_counter() - macro_started

        ambient = np.full(base.K.shape[0], cfg.ambient_K)
        balance_error = np.linalg.norm(delta.K @ ambient - delta.f)
        balance_scale = max(np.linalg.norm(delta.f), np.finfo(float).tiny)
        if spla.norm(delta.C) > 1.0e-11 * max(spla.norm(base.C), 1.0):
            raise RuntimeError("convection unexpectedly changed macro capacitance")
        if balance_error > 1.0e-10 * balance_scale:
            raise RuntimeError("affine convection component violates ambient balance")

        port_count = macro_ports.port_count
        if port_count != cfg.ports:
            raise RuntimeError("configured interface port count is inconsistent")

        detail_to_full = coordinate_map(detail_steady, full_layout, 0, "detail/full")
        transient_to_full = coordinate_map(
            detail_transient, full_layout, 0, "transient/full"
        )
        if not np.array_equal(detail_to_full, transient_to_full):
            raise RuntimeError("steady and transient detail orderings differ")
        macro_to_full = coordinate_map(
            macro_compiled, full_layout, cfg.detail_nz, "macro/full"
        )
        combined = np.r_[detail_to_full, macro_to_full]
        if (
            combined.size != full_layout.cell_count
            or np.unique(combined).size != combined.size
        ):
            raise RuntimeError("detail and macro maps do not partition the full model")

        W, orders, initial_internal, basis_s = column_basis(
            macro_compiled, cfg, base, delta, port_count
        )
        projection_started = time.perf_counter()
        reduced_base = project_operators(base, port_count, W)
        reduced_delta = project_operators(delta, port_count, W)
        projection_s = time.perf_counter() - projection_started
        offline_s = time.perf_counter() - offline_started

        full_macro_order = port_count + W.shape[0]
        reduced_macro_order = port_count + W.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={port_count}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
            f"({compression:.2f}x)"
        )

        results = []
        detail_n = detail_steady.cell_count
        for name, convection_h in boundaries:
            reference = full_reference(cfg, convection_h)
            (
                reference_steady,
                reference_times,
                reference_history,
                full_compile_s,
                full_steady_s,
                full_transient_s,
                full_order,
            ) = reference

            assembly_started = time.perf_counter()
            reduced = affine_operators(
                reduced_base,
                reduced_delta,
                convection_h / cfg.affine_anchor_h,
            )
            online_assembly_s = time.perf_counter() - assembly_started

            def solve_reduced(transient: bool):
                compiled = detail_transient if transient else detail_steady
                ports = detail_ports_transient if transient else detail_ports_steady
                state = np.r_[
                    np.full(compiled.cell_count + port_count, cfg.ambient_K),
                    initial_internal,
                ]
                started = time.perf_counter()
                with solve_macro(
                    compiled,
                    reduced,
                    ports,
                    state,
                    solve_options(cfg, transient),
                ) as solution:
                    elapsed = time.perf_counter() - started
                    if transient:
                        return (
                            np.asarray(solution.history_times).copy(),
                            np.asarray(solution.state_history).copy(),
                            elapsed,
                        )
                    return np.asarray(solution.state).copy(), elapsed

            steady_state, reduced_steady_s = solve_reduced(False)
            times, transient_states, reduced_transient_s = solve_reduced(True)
            if times.shape != reference_times.shape or not np.allclose(
                times, reference_times, atol=1.0e-12, rtol=0.0
            ):
                raise RuntimeError("full and reduced output times differ")

            def recover(states):
                states = np.atleast_2d(states)
                temperature = np.empty((states.shape[0], full_layout.cell_count))
                temperature[:, detail_to_full] = states[:, :detail_n]
                temperature[:, macro_to_full] = (
                    W @ states[:, detail_n + port_count :].T
                ).T
                return temperature

            steady_error = float(
                np.max(np.abs(recover(steady_state)[0] - reference_steady))
            )
            transient_error = float(
                np.max(np.abs(recover(transient_states) - reference_history))
            )
            speedup = full_transient_s / max(reduced_transient_s, np.finfo(float).tiny)
            accuracy_passed = max(steady_error, transient_error) <= cfg.error_K
            speedup_passed = speedup >= cfg.speedup_target if strict else True
            result = {
                "name": name,
                "h_W_m2K": convection_h,
                "steady_error_K": steady_error,
                "transient_error_K": transient_error,
                "online_reduced_assembly_s": online_assembly_s,
                "full_compile_s": full_compile_s,
                "full_steady_solve_s": full_steady_s,
                "reduced_steady_solve_s": reduced_steady_s,
                "full_transient_solve_s": full_transient_s,
                "reduced_transient_solve_s": reduced_transient_s,
                "transient_speedup": speedup,
                "full_order": full_order,
                "reduced_online_order": detail_n + reduced.K.shape[0],
                "reduced_macro_k_nnz": reduced.K.nnz,
                "reduced_macro_c_nnz": reduced.C.nnz,
                "accuracy_passed": accuracy_passed,
                "speedup_passed": speedup_passed,
                "passed": accuracy_passed and speedup_passed,
            }
            results.append(result)
            print(
                f"{name:>16s}: h={convection_h:g} W/(m^2 K), "
                f"error steady/transient={steady_error:.5f}/{transient_error:.5f} K; "
                f"full/ROM={full_transient_s:.3f}/{reduced_transient_s:.3f}s, "
                f"speedup={speedup:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= cfg.compression_target
        return {
            "schema_version": 18,
            "method": "exact-port affine-convection column-local Galerkin BCI-ROM",
            "configuration": {
                **asdict(cfg),
                "report": str(cfg.report),
                "nx": cfg.nx,
                "ny": cfg.nx,
                "nz": cfg.nz,
                "ports": cfg.ports,
                "port_shape": [cfg.port_indices.size, cfg.port_indices.size],
                "nominal_power_W": cfg.nominal_power_W,
                "power_map_normalized": POWER_MAP.tolist(),
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "affine_boundary": {
                "family": "A(h)=A0+(h/anchor_h)*DeltaA_h",
                "region": "entire cold-plate top surface",
                "anchor_h_W_m2K": cfg.affine_anchor_h,
                "full_order_offline_assemblies": 2,
                "full_order_online_assemblies_per_case": 0,
                "extraction_s": macro_extraction_s,
                "projection_s": projection_s,
            },
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "compression_ratio": compression,
                "compression_target": cfg.compression_target,
                "compression_passed": compression_passed,
                "column_count": int(orders.size),
                "port_columns": port_count,
                "local_order_min": int(orders.min()),
                "local_order_mean": float(orders.mean()),
                "local_order_max": int(orders.max()),
                "basis_nnz": W.nnz,
                "basis_extraction_s": basis_s,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence with exact ports",
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(result["passed"] for result in results) and compression_passed
            ),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="small smoke experiment")
    mode.add_argument("--strict", action="store_true", help="full benchmark gates")
    args = parser.parse_args(argv)

    cfg = replace(Config(), **QUICK_OVERRIDES) if args.quick else Config()
    boundaries = (BOUNDARIES[0], BOUNDARIES[-1]) if args.quick else BOUNDARIES

    print("=" * 96)
    print("Transient BCI-ROM extraction - uniform cold-plate convection")
    print("=" * 96)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(
        f"Nominal die power={cfg.nominal_power_W:.2f} W; "
        f"tile peak/mean density={POWER_MAP.max():.2f}x"
    )

    report = run_experiment(cfg, boundaries, args.strict)
    report["mode"] = "quick" if args.quick else "strict"
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report: {cfg.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
