"""Concrete affine parametric model: Flotherm-flavoured package (die + substrate + cap).

This module is *private* — reachable only through the factory under the
registered name ``"bci_pkg"``.  It adapts the geometry previously inlined in
``bci_fantastic_reproduction.py`` to the :class:`AffineParametricModel`
interface.

Layout (z from 0 up): die (silicon, detail, two heat zones) at the bottom,
then substrate (organic) + cap (silicon) on top (the *macro* domain).  The
macro block bottom face is the interface (= die top); its top face and four
side walls are two parametric boundary groups {top, side}, each with an
independent heat-exchange coefficient (FANTASTIC BCI 2015: boundary partitioned
into groups, one scalar ``p_k`` per group).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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


MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("silicon", "130", "130", "115", "2330", "700"),
)
DEFAULT_H_RANGE = (1.0, 1.0e6)  # Flotherm default 1..10,000 W/m2K


@dataclass(frozen=True)
class BciPkgConfig:
    ambient_K: float = 300.0
    size_mm: float = 14.0  # all layers share lateral extent
    die_h_mm: float = 0.6
    substrate_h_mm: float = 1.0
    cap_h_mm: float = 1.0
    die_cells: int = 2
    substrate_cells: int = 3
    cap_cells: int = 3
    max_xy_cell_mm: float = 1.0
    source_power_W: float = 1.0
    left_power_frac: float = 0.6  # fraction of total power in the left heat zone
    duration_s: float = 600.0
    dt_s: float = 30.0
    h_ranges: tuple = (DEFAULT_H_RANGE, DEFAULT_H_RANGE)  # [top, side]
    top_h: float = 10000.0
    side_h: float = 10000.0

    @property
    def macro_layers(self):
        return (
            (self.substrate_h_mm, self.substrate_cells),
            (self.cap_h_mm, self.cap_cells),
        )

    @property
    def detail_layers(self):
        return ((self.die_h_mm, self.die_cells),)

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
    def macro_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.macro_layers)

    @property
    def total_height_mm(self) -> float:
        return self.die_h_mm + self.macro_height_mm

    @cached_property
    def axis_vertices_mm(self) -> np.ndarray:
        half = self.size_mm / 2.0
        fixed = np.unique([-half, 0.0, half])
        vertices = [float(fixed[0])]
        for left, right in zip(fixed[:-1], fixed[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_xy_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @property
    def nx(self) -> int:
        return self.axis_vertices_mm.size - 1

    def report_dict(self) -> dict:
        return {
            **{k: v for k, v in self.__dict__.items() if not k.startswith("_")},
            "nx": self.nx,
            "nz": self.nz,
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


def build_geometry(
    cfg: BciPkgConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    boundary_h: dict[str, float] | None = None,
):
    """Assemble geometry.  Layers bottom-up: die (detail), then substrate+cap (macro).

    ``boundary_h`` (if given) maps boundary-group name -> coefficient and
    applies the group's convection via :func:`apply_boundary_convection` — the
    group region table (:func:`_boundary_regions`) is the model's
    parameterization, so nothing here special-cases a face.  Used only for the
    native full reference.
    """
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    model = metahotspot.Model()
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=cfg.duration_s if study == Study.TRANSIENT else 0.0,
        output_interval=cfg.dt_s if study == Study.TRANSIENT else 0.0,
    )
    # Physical z from bottom to top: [die, substrate, cap].  add_layer pushes
    # each new layer to the bottom (last added sits at z=0), so layers are
    # added top-first: cap, then substrate, then die.
    zv = [0.0]
    if detail:
        zv.extend(z_vertices(cfg.detail_layers)[1:])
    if macro:
        macro_z = z_vertices(cfg.macro_layers)[1:] + (cfg.die_h_mm if detail else 0.0)
        zv.extend(macro_z)
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, np.asarray(zv))
    for material in MATERIALS:
        model.add_material(*material)

    if macro:
        for thickness, name in (
            (cfg.cap_h_mm, "silicon"),
            (cfg.substrate_h_mm, "organic"),
        ):
            layer = model.add_layer(str(thickness))
            add_square(model, model.add_block(layer, name), cfg.size_mm)

    if detail:
        die = model.add_layer(str(cfg.die_h_mm))
        add_square(model, model.add_block(die, "silicon"), cfg.size_mm)
        half = cfg.size_mm / 2.0
        zone_volume = half * cfg.size_mm * cfg.die_h_mm * 1.0e-9  # m^3
        for x0, frac in (
            (-half, cfg.left_power_frac),
            (0.0, 1.0 - cfg.left_power_frac),
        ):
            block = model.add_block(
                die,
                "silicon",
                heat_source=f"{frac * cfg.source_power_W / zone_volume:.17g}",
            )
            model.add_rect(
                block,
                GeometryOp.ADD,
                f"{x0:.17g}",
                f"{-half:.17g}",
                f"{half:.17g}",
                f"{cfg.size_mm:.17g}",
            )

    model.set_default_neumann("0")

    if macro and boundary_h:
        from affine_parametric_models._interfaces import apply_boundary_convection

        apply_boundary_convection(
            model,
            _boundary_regions(cfg),
            boundary_h,
            cfg.ambient_K,
            z_offset=cfg.die_h_mm if detail else 0.0,
        )
    return model


def _boundary_regions(cfg: BciPkgConfig) -> dict[str, tuple]:
    """Face regions (macro frame, z=0 at macro base) for the {top, side} groups.

    Top = cap top face (macro z = macro_height_mm) over the full extent.
    Side = the four side walls over the full macro height.
    """
    half = cfg.size_mm / 2.0
    macro_h = cfg.macro_height_mm
    z0, z1 = 0.0, macro_h
    side = []
    for axis, coord in (
        (Axis.X, -half),
        (Axis.X, half),
        (Axis.Y, -half),
        (Axis.Y, half),
    ):
        side.append((axis, coord, -half, half, z0, z1))
    return {
        "top": ((Axis.Z, macro_h, -half, half, -half, half),),
        "side": tuple(side),
    }


def macro_interface_patches(cfg: BciPkgConfig) -> list[PortPatch]:
    """Interface ports on the macro block bottom (= die top, macro z=0)."""
    verts = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(Face.ZM), 0.0, (verts[ix], verts[ix + 1], verts[iy], verts[iy + 1])
        )
        for ix in range(verts.size - 1)
        for iy in range(verts.size - 1)
    ]


def macro_boundary_groups(cfg: BciPkgConfig):
    """Boundary port groups [top, side] with per-group patch areas.

    Top = cap top face (macro z = macro_h), one patch per exposed cell.
    Side = the four side walls over the full macro height, one patch per cell.
    """
    verts = cfg.axis_vertices_mm * 1.0e-3
    zv = z_vertices(cfg.macro_layers) * 1.0e-3
    top_z = zv[-1]

    top = [
        PortPatch(
            int(Face.ZP), top_z, (verts[ix], verts[ix + 1], verts[iy], verts[iy + 1])
        )
        for ix in range(verts.size - 1)
        for iy in range(verts.size - 1)
    ]
    top_areas = np.asarray(
        [
            (verts[ix + 1] - verts[ix]) * (verts[iy + 1] - verts[iy])
            for ix in range(verts.size - 1)
            for iy in range(verts.size - 1)
        ]
    )

    half = cfg.size_mm / 2.0 * 1e-3
    side, side_areas = [], []
    for axis_face, coord, a_verts in (
        (Face.XM, -half, verts),
        (Face.XP, half, verts),
        (Face.YM, -half, verts),
        (Face.YP, half, verts),
    ):
        for a0, a1 in zip(a_verts[:-1], a_verts[1:]):
            for b0, b1 in zip(zv[:-1], zv[1:]):
                side.append(PortPatch(int(axis_face), float(coord), (a0, a1, b0, b1)))
                side_areas.append((a1 - a0) * (b1 - b0))
    return [top, side], [top_areas, np.asarray(side_areas)]


def detail_monitor_cells(cfg: BciPkgConfig, detail_compiled) -> np.ndarray:
    """Die-top cell indices (in the detail model) above each heat zone centre."""
    grid = detail_compiled.grid_to_cell.reshape(
        detail_compiled.nx, detail_compiled.ny, detail_compiled.nz
    )
    top = grid[:, :, -1]
    verts = cfg.axis_vertices_mm * 1e-3
    centres = (verts[:-1] + verts[1:]) / 2.0
    quarter = cfg.size_mm / 4.0 * 1e-3
    cells = []
    for xc in (-quarter, quarter):
        ix = int(np.argmin(np.abs(centres - xc)))
        iy = int(np.argmin(np.abs(centres - 0.0)))
        cells.append(int(top[ix, iy]))
    return np.asarray(cells)


class _BciPkg(AffineParametricModel):
    """Private concrete implementation registered as ``"bci_pkg"``."""

    def __init__(self, cfg: BciPkgConfig):
        self._cfg = cfg

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return "bci_pkg"

    @property
    def ambient_K(self) -> float:
        return self._cfg.ambient_K

    # --------------------------------------------------- DtN core extraction

    @property
    def port_count(self) -> int:
        interface = macro_interface_patches(self._cfg)
        return len(interface)

    @cached_property
    def _core(self) -> Operators:
        macro = build_geometry(
            self._cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        interface = macro_interface_patches(self._cfg)
        pm_core = PortMap(macro, interface)
        return normalized_operators(*pm_core.assemble())

    def core_operators(self) -> Operators:
        return self._core

    def merged_operators(self, group_sizes) -> Operators:
        macro = build_geometry(
            self._cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        interface = macro_interface_patches(self._cfg)
        groups, _areas = macro_boundary_groups(self._cfg)
        all_boundary = [p for g in groups for p in g]
        pm_merged = PortMap(macro, interface + all_boundary)
        return normalized_operators(*pm_merged.assemble())

    @cached_property
    def _boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        interface = macro_interface_patches(self._cfg)
        groups, group_areas = macro_boundary_groups(self._cfg)
        group_sizes = [len(g) for g in groups]
        merged = self.merged_operators(group_sizes)
        extracted = extract_boundary_groups(merged, len(interface), group_sizes)
        return (
            BoundaryGroup(
                name="top",
                cells=extracted[0][0],
                g=extracted[0][1],
                areas=group_areas[0],
                h_default=self._cfg.top_h,
                h_range=self._cfg.h_ranges[0],
            ),
            BoundaryGroup(
                name="side",
                cells=extracted[1][0],
                g=extracted[1][1],
                areas=group_areas[1],
                h_default=self._cfg.side_h,
                h_range=self._cfg.h_ranges[1],
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
                (ix, iy) for ix in range(self._cfg.nx) for iy in range(self._cfg.nx)
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
        if len(h_vec) != 2:
            raise ValueError("bci_pkg has exactly two boundary groups")
        h_top, h_side = float(h_vec[0]), float(h_vec[1])
        boundary_h = {"top": h_top, "side": h_side}
        started = time.perf_counter()
        steady = build_geometry(
            self._cfg,
            Study.STEADY,
            detail=True,
            macro=True,
            boundary_h=boundary_h,
        ).compile()
        transient = build_geometry(
            self._cfg,
            Study.TRANSIENT,
            detail=True,
            macro=True,
            boundary_h=boundary_h,
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
        detail_interface = [
            PortPatch(int(Face.ZP), self._cfg.die_h_mm * 1e-3, p.rectangle)
            for p in macro_interface_patches(self._cfg)
        ]
        return (
            PortMap(self._detail_steady, detail_interface),
            PortMap(self._detail_transient, detail_interface),
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
        dg = self._detail_steady.grid_to_cell.reshape(
            self._detail_steady.nx, self._detail_steady.ny, self._detail_steady.nz
        )
        fg = self._full_layout.grid_to_cell.reshape(
            self._full_layout.nx, self._full_layout.ny, self._full_layout.nz
        )
        dz = self._cfg.detail_nz
        valid = dg >= 0
        assert np.array_equal(valid, fg[:, :, :dz] >= 0)
        mapping = np.empty(self._detail_steady.cell_count, dtype=np.int64)
        mapping[dg[valid]] = fg[:, :, :dz][valid]
        assert np.unique(mapping).size == mapping.size
        return mapping

    @cached_property
    def _macro_to_full(self) -> np.ndarray:
        mg = self._macro.grid_to_cell.reshape(
            self._macro.nx, self._macro.ny, self._macro.nz
        )
        fg = self._full_layout.grid_to_cell.reshape(
            self._full_layout.nx, self._full_layout.ny, self._full_layout.nz
        )
        dz = self._cfg.detail_nz
        valid = mg >= 0
        assert np.array_equal(valid, fg[:, :, dz:] >= 0)
        mapping = np.empty(self._macro.cell_count, dtype=np.int64)
        mapping[mg[valid]] = fg[:, :, dz:][valid]
        assert np.unique(mapping).size == mapping.size
        return mapping

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
        return detail_monitor_cells(self._cfg, self._detail_steady)

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
    cfg = BciPkgConfig(**(overrides or {}))
    return _BciPkg(cfg)
