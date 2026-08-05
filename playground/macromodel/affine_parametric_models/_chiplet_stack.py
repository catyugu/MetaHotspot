"""Concrete affine parametric model: the chiplet+spreader+cold-plate stack.

This module is *private* — it is reachable only through the factory under the
registered name ``"chiplet_stack"``.  It adapts the geometry/material/source
construction previously living in ``experiment_setup.py`` to the
:class:`AffineParametricModel` interface.

Layout (z from 0 up): substrate (organic) / bump (underfill+Cu pillars) / die
(silicon, chiplet heat sources) at the bottom (the *detail* domain), then TIM /
spreader (copper) / cold plate (aluminum) on top (the *macro* domain).  The
macro block bottom face is the interface (= die top); its top face is the
single parametric boundary group (uniform heat-exchange coefficient ``h``).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

import metahotspot
from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortMap, PortPatch, solve as solve_macro

from affine_parametric_models._interfaces import (
    AffineParametricModel,
    AffineSolveResult,
    BoundaryGroup,
    StateLayout,
)
from utils import (
    extract_boundary_groups,
    normalized_operators,
)


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
DEFAULT_H = 2500.0  # W/m^2 K


@dataclass(frozen=True)
class ChipletStackConfig:
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
    # nominal heat-exchange coefficient on the top face (W/m^2 K)
    top_h: float = DEFAULT_H

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


def _boundary_regions(cfg: ChipletStackConfig) -> dict[str, tuple]:
    """Face regions (macro frame, z=0 at macro base) for the top boundary group.

    The single chiplet-stack group covers the macro top face over the full
    cold-plate lateral extent.
    """
    half = cfg.cold_plate_size_mm / 2.0
    return {
        "top": ((Axis.Z, cfg.macro_height_mm, -half, half, -half, half),),
    }


def build_geometry(
    cfg: ChipletStackConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    boundary_h: dict[str, float] | None = None,
):
    """Assemble the chiplet stack (no boundary conditions unless requested).

    Returns a model with its default Neumann BC set.  ``boundary_h`` (if given)
    maps boundary-group name -> coefficient and applies the group's convection
    via :func:`apply_boundary_convection` — the group region table
    (:attr:`_chiplet_stack._boundary_regions`) is the model's parameterization,
    so nothing here hard-codes a specific face.  Used only for the native full
    reference.
    """
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    if boundary_h is not None and any(h < 0.0 for h in boundary_h.values()):
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

    if macro and boundary_h:
        from affine_parametric_models._interfaces import apply_boundary_convection

        apply_boundary_convection(
            model,
            _boundary_regions(cfg),
            boundary_h,
            cfg.ambient_K,
            z_offset=cfg.detail_height_mm if detail else 0.0,
        )
    return model


def port_patches(cfg: ChipletStackConfig, face: Face, z_m: float) -> list[PortPatch]:
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


def full_face_patches(cfg: ChipletStackConfig, face: Face, z_m: float) -> list[PortPatch]:
    """One PortPatch per exposed FVM cell over the full lateral extent."""
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


def patch_areas(cfg: ChipletStackConfig, patches: list[PortPatch]) -> np.ndarray:
    """SI face area (m^2) of each patch, in patch order."""
    areas = np.empty(len(patches), dtype=np.float64)
    for index, patch in enumerate(patches):
        a_min, a_max, b_min, b_max = patch.rectangle
        areas[index] = (a_max - a_min) * (b_max - b_min)
    return areas


class _ChipletStack(AffineParametricModel):
    """Private concrete implementation registered as ``"chiplet_stack"``."""

    def __init__(self, cfg: ChipletStackConfig):
        self._cfg = cfg

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return "chiplet_stack"

    @property
    def ambient_K(self) -> float:
        return self._cfg.ambient_K

    # --------------------------------------------------- DtN core extraction

    @property
    def port_count(self) -> int:
        return self._cfg.ports

    @cached_property
    def _core(self) -> Operators:
        macro = build_geometry(
            self._cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        interface = port_patches(self._cfg, Face.ZM, 0.0)
        pm_core = PortMap(macro, interface)
        return normalized_operators(*pm_core.assemble())

    def core_operators(self) -> Operators:
        return self._core

    def merged_operators(self, group_sizes) -> Operators:
        macro = build_geometry(
            self._cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        interface = port_patches(self._cfg, Face.ZM, 0.0)
        boundary = full_face_patches(
            self._cfg, Face.ZP, self._cfg.macro_height_mm * 1.0e-3
        )
        pm_merged = PortMap(macro, interface + boundary)
        return normalized_operators(*pm_merged.assemble())

    @cached_property
    def _boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        interface = port_patches(self._cfg, Face.ZM, 0.0)
        boundary = full_face_patches(
            self._cfg, Face.ZP, self._cfg.macro_height_mm * 1.0e-3
        )
        merged = self.merged_operators([len(boundary)])
        (cells, g), = extract_boundary_groups(
            merged, len(interface), [len(boundary)]
        )
        areas = patch_areas(self._cfg, boundary)
        return (
            BoundaryGroup(
                name="top",
                cells=cells,
                g=g,
                areas=areas,
                h_default=self._cfg.top_h,
            ),
        )

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        return self._boundary_groups

    @property
    def macro_cell_count(self) -> int:
        return self._core.K.shape[0] - self.port_count

    @property
    def macro_nx(self) -> int:
        return self._cfg.nx

    @property
    def macro_ny(self) -> int:
        return self._cfg.nx

    @cached_property
    def macro_grid(self) -> np.ndarray:
        return self._macro.grid_to_cell.reshape(
            self._macro.nx, self._macro.ny, self._macro.nz
        )

    @property
    def dt(self) -> float:
        return self._cfg.dt_s

    @cached_property
    def port_lookup(self) -> dict[tuple[int, int], int]:
        return {
            (int(ix), int(iy)): port
            for port, (ix, iy) in enumerate(
                (ix, iy) for ix in self._cfg.port_indices for iy in self._cfg.port_indices
            )
        }

    # --------------------------------------- native reference + recovery

    @cached_property
    def _full_layout(self):
        return build_geometry(
            self._cfg, Study.STEADY, detail=True, macro=True
        ).compile()

    @cached_property
    def _detail_steady(self):
        return build_geometry(
            self._cfg, Study.STEADY, detail=True, macro=False
        ).compile()

    @cached_property
    def _detail_transient(self):
        return build_geometry(
            self._cfg, Study.TRANSIENT, detail=True, macro=False
        ).compile()

    @cached_property
    def _macro(self):
        return build_geometry(
            self._cfg, Study.STEADY, detail=False, macro=True
        ).compile()

    def full_reference(self, h_vec) -> AffineSolveResult:
        if len(h_vec) != 1:
            raise ValueError("chiplet_stack has exactly one boundary group")
        convection_h = float(h_vec[0])
        started = time.perf_counter()
        steady = build_geometry(
            self._cfg,
            Study.STEADY,
            detail=True,
            macro=True,
            boundary_h={"top": convection_h},
        ).compile()
        transient = build_geometry(
            self._cfg,
            Study.TRANSIENT,
            detail=True,
            macro=True,
            boundary_h={"top": convection_h},
        ).compile()
        compile_s = time.perf_counter() - started

        from affine_parametric_models._interfaces import native_solve_timing

        return native_solve_timing(steady, transient, self.solver_options, compile_s)

    def state_layout(self, internal_count: int) -> StateLayout:
        return StateLayout(
            detail_count=self.detail_cell_count,
            port_count=self.port_count,
            internal_count=internal_count,
        )

    @cached_property
    def _detail_ports(self) -> tuple[PortMap, PortMap]:
        detail_patches = port_patches(
            self._cfg, Face.ZP, self._cfg.detail_height_mm * 1.0e-3
        )
        return (
            PortMap(self._detail_steady, detail_patches),
            PortMap(self._detail_transient, detail_patches),
        )

    def solve_reduced(self, operators: Operators, state, transient: bool):
        detail_ports_steady, detail_ports_transient = self._detail_ports
        if transient:
            started = time.perf_counter()
            with solve_macro(
                operators,
                detail_ports_transient,
                state,
                self.solver_options(True),
            ) as solution:
                elapsed = time.perf_counter() - started
                return solution.history_times, solution.state_history, elapsed
        started = time.perf_counter()
        with solve_macro(
            operators,
            detail_ports_steady,
            state,
            self.solver_options(False),
        ) as solution:
            elapsed = time.perf_counter() - started
            return solution.state, elapsed

    @cached_property
    def _detail_to_full(self) -> np.ndarray:
        from utils import coordinate_map, grid_cells

        mapping = coordinate_map(
            self._detail_steady, self._full_layout, 0, "detail/full"
        )
        if not np.array_equal(
            mapping,
            coordinate_map(self._detail_transient, self._full_layout, 0, "transient/full"),
        ):
            raise RuntimeError("steady and transient detail orderings differ")
        return mapping

    @cached_property
    def _macro_to_full(self) -> np.ndarray:
        from utils import coordinate_map

        return coordinate_map(
            self._macro, self._full_layout, self._cfg.detail_nz, "macro/full"
        )

    def detail_to_full(self) -> np.ndarray:
        return self._detail_to_full

    def macro_to_full(self) -> np.ndarray:
        return self._macro_to_full

    @property
    def full_cell_count(self) -> int:
        return self._full_layout.cell_count

    @property
    def detail_cell_count(self) -> int:
        return self._detail_steady.cell_count

    def recover_temperature(
        self, states, *, basis, ports: int, ambient_K: float | None
    ) -> np.ndarray:
        states = np.atleast_2d(states)
        temperature = np.empty((states.shape[0], self.full_cell_count))
        temperature[:, self.detail_to_full()] = states[:, : self.detail_cell_count]
        internal = (basis @ states[:, self.detail_cell_count + ports :].T).T
        temperature[:, self.macro_to_full()] = (
            internal if ambient_K is None else ambient_K + internal
        )
        return temperature

    def monitor_cells(self) -> np.ndarray:
        # Chiplet stack has no dedicated monitor semantic; default to the
        # detail model's full cell range (the experiments that use this model
        # score full-field accuracy, not per-monitor curves).
        return np.arange(self.detail_cell_count, dtype=np.int64)

    def monitor_full(self, detail_cells: np.ndarray) -> np.ndarray:
        return self.detail_to_full()[np.asarray(detail_cells, dtype=np.int64)]

    def report_dict(self) -> dict:
        return self._cfg.report_dict()

    def solver_options(self, transient: bool) -> SolveOptions:
        dt = self._cfg.dt_s if transient else 1.0
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


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = ChipletStackConfig(**(overrides or {}))
    return _ChipletStack(cfg)
