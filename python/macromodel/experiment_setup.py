#!/usr/bin/env python3
"""Shared model construction and numerical helpers for macromodel experiments."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np
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
BOUNDARIES = (500.0, 2500.0, 8000.0)
MAX_RELATIVE_RISE_ERROR = 0.01


@dataclass(frozen=True)
class BaseConfig:
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

    # About 0.18-0.30 mm per vertical cell. Thin layers no longer receive
    # artificial over-resolution; lateral resolution is increased instead.
    substrate_cells: int = 4
    bump_cells: int = 1
    die_cells: int = 2
    tim_cells: int = 1
    spreader_cells: int = 4
    cold_plate_cells: int = 5
    max_xy_cell_mm: float = 1.75

    bump_rows: int = 12
    bump_columns: int = 12
    bump_width_mm: float = 0.60
    chiplet_size_mm: float = 12.0
    chiplet_power_W: float = 25.0
    duration_s: float = 100.0
    dt_s: float = 10.0
    affine_anchor_h: float = 2500.0
    speedup_target: float = 1.5
    compression_target: float = 2.5

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

        fixed = np.unique(np.asarray(points, dtype=np.float64))
        vertices = [float(fixed[0])]
        for left, right in zip(fixed[:-1], fixed[1:]):
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

    def report_dict(self) -> dict:
        return {
            **asdict(self),
            "nx": self.nx,
            "ny": self.nx,
            "nz": self.nz,
            "ports": self.ports,
            "port_shape": [self.port_indices.size, self.port_indices.size],
            "nominal_power_W": self.nominal_power_W,
        }


def temperature_error_metrics(reference, approximation, ambient_K: float) -> dict:
    reference = np.asarray(reference)
    approximation = np.asarray(approximation)
    absolute_error = float(np.max(np.abs(approximation - reference)))
    reference_rise = float(np.max(np.abs(reference - ambient_K)))
    relative_error = (
        absolute_error / reference_rise
        if reference_rise
        else float(absolute_error != 0.0)
    )
    return {
        "reference_temperature_range_K": [
            float(reference.min()),
            float(reference.max()),
        ],
        "max_absolute_rise_error_K": absolute_error,
        "max_relative_rise_error": relative_error,
        "passed": relative_error < MAX_RELATIVE_RISE_ERROR,
    }


def accuracy_summary(
    reference_steady,
    reduced_steady,
    reference_history,
    reduced_history,
    ambient_K: float,
) -> dict:
    steady = temperature_error_metrics(reference_steady, reduced_steady, ambient_K)
    transient = temperature_error_metrics(
        reference_history[-1], reduced_history[-1], ambient_K
    )
    return {
        "steady_reference_temperature_range_K": steady["reference_temperature_range_K"],
        "transient_final_reference_temperature_range_K": transient[
            "reference_temperature_range_K"
        ],
        "steady_max_absolute_rise_error_K": steady["max_absolute_rise_error_K"],
        "steady_max_relative_rise_error": steady["max_relative_rise_error"],
        "transient_final_max_absolute_rise_error_K": transient[
            "max_absolute_rise_error_K"
        ],
        "transient_final_max_relative_rise_error": transient["max_relative_rise_error"],
        "accuracy_passed": steady["passed"] and transient["passed"],
    }


def format_accuracy(summary: dict) -> str:
    steady_range = summary["steady_reference_temperature_range_K"]
    transient_range = summary["transient_final_reference_temperature_range_K"]
    return (
        f"reference range steady={steady_range[0]:.3f}..{steady_range[1]:.3f} K, "
        f"transient final={transient_range[0]:.3f}..{transient_range[1]:.3f} K; "
        f"rise error steady={summary['steady_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['steady_max_relative_rise_error']:.3%}, transient final="
        f"{summary['transient_final_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['transient_final_max_relative_rise_error']:.3%}"
    )


def z_vertices(layers) -> np.ndarray:
    vertices = [0.0]
    for thickness, cells in layers:
        vertices.extend(vertices[-1] + thickness * np.arange(1, cells + 1) / cells)
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
    cfg: BaseConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    convection_h: float | None = None,
):
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
        add_square(model, model.add_block(bump, "underfill"), cfg.bump_region_size_mm)
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
        add_square(model, model.add_block(substrate, "organic"), cfg.substrate_size_mm)

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


def port_patches(cfg: BaseConfig, face: Face, z_m: float) -> list[PortPatch]:
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


def full_face_patches(cfg: BaseConfig, face: Face, z_m: float) -> list[PortPatch]:
    """One PortPatch per exposed FVM cell over the full lateral extent.

    Unlike :func:`port_patches`, which restricts to the TIM/interface region,
    this spans every cell of the compiled face so that a boundary-port closure
    (``h*A`` added at each boundary port) reproduces the native convection
    discretization of the same face exactly.
    """
    vertices = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(face),
            z_m,
            (vertices[ix], vertices[ix + 1], vertices[iy], vertices[iy + 1]),
        )
        for ix in range(vertices.size - 1)
        for iy in range(vertices.size - 1)
    ]


def patch_areas(cfg: BaseConfig, patches: list[PortPatch]) -> np.ndarray:
    """SI face area (m^2) of each patch, in patch order."""
    areas = np.empty(len(patches), dtype=np.float64)
    for index, patch in enumerate(patches):
        a_min, a_max, b_min, b_max = patch.rectangle
        areas[index] = (a_max - a_min) * (b_max - b_min)
    return areas


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


def project_exact_ports(
    operators: Operators, ports: int, basis, ambient_K: float | None = None
) -> Operators:
    source = np.asarray(operators.f, dtype=np.float64)
    if ambient_K is not None:
        offset = np.full(operators.K.shape[0] - ports, ambient_K)
        source = np.asarray(source - operators.K[:, ports:] @ offset).ravel()

    def project(matrix):
        reduced = sp.bmat(
            (
                (
                    sp.csc_matrix(matrix[:ports, :ports]),
                    sp.csc_matrix(matrix[:ports, ports:] @ basis),
                ),
                (
                    sp.csc_matrix(basis.T @ matrix[ports:, :ports]),
                    sp.csc_matrix(basis.T @ matrix[ports:, ports:] @ basis),
                ),
            ),
            format="csc",
        )
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project(operators.K),
        project(operators.C),
        np.r_[source[:ports], np.asarray(basis.T @ source[ports:]).ravel()],
    )


def validate_affine_macro(base: Operators, delta: Operators, ambient_K: float) -> None:
    ambient = np.full(base.K.shape[0], ambient_K)
    balance_error = np.linalg.norm(delta.K @ ambient - delta.f)
    balance_scale = max(np.linalg.norm(delta.f), np.finfo(float).tiny)
    if spla.norm(delta.C) > 1.0e-11 * max(spla.norm(base.C), 1.0):
        raise RuntimeError("convection unexpectedly changed macro capacitance")
    if balance_error > 1.0e-10 * balance_scale:
        raise RuntimeError("affine convection component violates ambient balance")


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


def solve_options(cfg: BaseConfig, transient: bool) -> SolveOptions:
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
        error_rel_tol=1.0e-3,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def solve_reduced(
    compiled,
    ports: PortMap,
    operators: Operators,
    state: np.ndarray,
    cfg: BaseConfig,
    transient: bool,
):
    started = time.perf_counter()
    with solve_macro(
        operators, ports, state, solve_options(cfg, transient)
    ) as solution:
        elapsed = time.perf_counter() - started
        if transient:
            return solution.history_times, solution.state_history, elapsed
        return solution.state, elapsed


def recover_temperature(
    states,
    *,
    full_count: int,
    detail_map,
    macro_map,
    detail_count: int,
    ports: int,
    basis,
    ambient_K: float | None,
):
    states = np.atleast_2d(states)
    temperature = np.empty((states.shape[0], full_count))
    temperature[:, detail_map] = states[:, :detail_count]
    internal = (basis @ states[:, detail_count + ports :].T).T
    temperature[:, macro_map] = internal if ambient_K is None else ambient_K + internal
    return temperature


def full_reference(cfg: BaseConfig, convection_h: float):
    started = time.perf_counter()
    steady = build_model(
        cfg, Study.STEADY, detail=True, macro=True, convection_h=convection_h
    ).compile()
    transient = build_model(
        cfg, Study.TRANSIENT, detail=True, macro=True, convection_h=convection_h
    ).compile()
    compile_s = time.perf_counter() - started

    started = time.perf_counter()
    with steady.solve(opts=solve_options(cfg, False)) as solution:
        steady_temperature = solution.temperature
    steady_s = time.perf_counter() - started

    started = time.perf_counter()
    with transient.solve(opts=solve_options(cfg, True)) as solution:
        times = solution.history_times
        history = solution.temperature_history
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
