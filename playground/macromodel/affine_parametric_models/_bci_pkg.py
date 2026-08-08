"""Concrete affine parametric model: Flotherm-flavoured package (die + substrate + cap).

This module is *private* — reachable only through the factory under the
registered name ``"bci_pkg"``.  It supplies the geometry hooks consumed by the
shared :class:`AffineParametricModel` base.

Layout (z from 0 up): die (silicon, detail, two heat zones) at the bottom,
then substrate (organic) + cap (silicon) on top (the *macro* domain).  The
macro block bottom face is the interface (= die top); its top face and four
side walls are two parametric boundary groups {top, side}, each with an
independent heat-exchange coefficient (FANTASTIC BCI 2015: boundary partitioned
into groups, one scalar ``p_k`` per group).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

import metahotspot
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study

from affine_parametric_models._interfaces import (
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
    surface_exposed_cells,
)

MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("silicon", "130", "130", "115", "2330", "700"),
)
DEFAULT_H_RANGE = (1.0, 1.0e6)

QUICK_OVERRIDES = {
    "max_xy_cell_mm": 2.0,
}


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


def _boundary_regions(cfg: BciPkgConfig) -> dict[str, tuple]:
    """Face regions (macro frame, z=0 at macro base) for the {top, side} groups."""
    half = cfg.size_mm / 2.0
    macro_h = cfg.macro_height_mm
    side = [
        (axis, coord, -half, half, 0.0, macro_h)
        for axis, coord in (
            (Axis.X, -half),
            (Axis.X, half),
            (Axis.Y, -half),
            (Axis.Y, half),
        )
    ]
    return {
        "top": ((Axis.Z, macro_h, -half, half, -half, half),),
        "side": tuple(side),
    }


def build_geometry(
    cfg: BciPkgConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    boundary_h: dict[str, float] | None = None,
    source_sink: list | None = None,
):
    """Assemble geometry.  Layers bottom-up: die (detail), then substrate+cap (macro).

    ``boundary_h`` (if given) maps boundary-group name -> coefficient and
    applies the group's convection via :func:`apply_boundary_convection`.
    When ``source_sink`` is given, each heat-source block appends
    ``(block_id, power_W)`` to it (in add order), so the model can recover
    per-source cells from the compiled ``block_ids``.
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
        # die-layer block id counter: 0 = die base block, then one per zone
        die_block_count = 1
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
            if source_sink is not None:
                source_sink.append((die_block_count, float(frac * cfg.source_power_W)))
            die_block_count += 1

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


class _BciPkg(AffineParametricModel):
    """Private concrete implementation registered as ``"bci_pkg"``."""

    def __init__(self, cfg: BciPkgConfig):
        self.config = cfg
        self._source_sink: list = []

    @property
    def name(self) -> str:
        return "bci_pkg"

    # ------------------------------------------------- geometry hooks

    def build_geometry(self, study, *, detail, macro, boundary_h=None):
        self._source_sink.clear()
        return build_geometry(
            self.config,
            study,
            detail=detail,
            macro=macro,
            boundary_h=boundary_h,
            source_sink=self._source_sink,
        )

    def source_ports(self) -> list[SourcePort]:
        """One :class:`SourcePort` per heat-source block (two die zones)."""
        f = np.asarray(self._core.f, dtype=np.float64)
        source_cells = np.flatnonzero(f > 0.0)
        block = self._full.block_ids[source_cells]
        order = np.argsort(block, kind="stable")
        block_sorted = block[order]
        cell_sorted = source_cells[order]
        boundaries = np.flatnonzero(np.diff(block_sorted) != 0) + 1
        groups = np.split(cell_sorted, boundaries)

        if len(groups) != len(self._source_sink):
            raise RuntimeError(
                f"source block count mismatch: {len(groups)} groups vs "
                f"{len(self._source_sink)} sink entries"
            )
        ports = []
        for cells, (block_id, power_W) in zip(groups, self._source_sink):
            ports.append(
                SourcePort(cells=np.asarray(cells, dtype=np.int64), power_W=power_W)
            )
        return ports

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        full = self._full
        grid = full.grid_to_cell.reshape(full.nx, full.ny, full.nz)
        x = self.config.axis_vertices_mm * 1.0e-3
        y = x
        z = z_vertices((*self.config.detail_layers, *self.config.macro_layers)) * 1.0e-3
        half = self.config.size_mm / 2.0 * 1.0e-3
        total_h = self.config.total_height_mm * 1.0e-3

        # top group: cap top face (ZP at full height)
        top_cells, top_areas = surface_exposed_cells(grid, x, y, z, Face.ZP, total_h)

        # side group: four lateral walls of the macro block (z in [die_h, total_h])
        side_cells, side_areas = [], []
        die_h = self.config.die_h_mm * 1.0e-3
        for face, coord in (
            (Face.XM, -half),
            (Face.XP, half),
            (Face.YM, -half),
            (Face.YP, half),
        ):
            cells, areas = surface_exposed_cells(
                grid, x, y, z, face, coord, z_range=(die_h, total_h)
            )
            side_cells.append(cells)
            side_areas.append(areas)
        side_cells = (
            np.concatenate(side_cells) if side_cells else np.empty(0, dtype=np.int64)
        )
        side_areas = (
            np.concatenate(side_areas) if side_areas else np.empty(0, dtype=np.float64)
        )

        return (
            BoundaryGroup(
                cells=top_cells, areas=top_areas, h_range=self.config.h_ranges[0]
            ),
            BoundaryGroup(
                cells=side_cells, areas=side_areas, h_range=self.config.h_ranges[1]
            ),
        )

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 2:
            raise ValueError("bci_pkg has exactly two boundary groups")
        return {"top": float(h_vec[0]), "side": float(h_vec[1])}

    def group_h_ranges(self):
        return self.config.h_ranges

    def parameter_points(self, count: int = 5) -> list[tuple[float, ...]]:
        """Product grid over {top, side}: ``count`` points per axis.

        The two boundary groups (top / side) are independent, so the validation
        points span their full ``(h_top, h_side)`` space — which is exactly the
        grid the BCI-FANTASTIC report's ``error_vs_h`` scatter needs.
        """
        top_range, side_range = self.config.h_ranges
        top_axis = np.geomspace(top_range[0], top_range[1], count)
        side_axis = np.geomspace(side_range[0], side_range[1], count)
        return [
            (float(h_top), float(h_side)) for h_top in top_axis for h_side in side_axis
        ]

    @property
    def detail_nz(self) -> int:
        return self.config.detail_nz


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = BciPkgConfig(**(overrides or {}))
    return _BciPkg(cfg)
