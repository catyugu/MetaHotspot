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
from metahotspot.macromodel import PortPatch

from affine_parametric_models._interfaces import AffineParametricModel

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
):
    """Assemble geometry.  Layers bottom-up: die (detail), then substrate+cap (macro).

    ``boundary_h`` (if given) maps boundary-group name -> coefficient and
    applies the group's convection via :func:`apply_boundary_convection`.
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


def _full_face_patches(cfg: BciPkgConfig, face: Face, z_m: float) -> list[PortPatch]:
    verts = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(int(face), z_m, (verts[ix], verts[ix + 1], verts[iy], verts[iy + 1]))
        for ix in range(verts.size - 1)
        for iy in range(verts.size - 1)
    ]


def _boundary_patch_groups(cfg: BciPkgConfig):
    """Boundary port groups [top, side] with per-group patch areas."""
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


def _detail_monitor_cells(cfg: BciPkgConfig, detail_compiled) -> np.ndarray:
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
        self.config = cfg

    @property
    def name(self) -> str:
        return "bci_pkg"

    # ------------------------------------------------- geometry hooks

    def build_geometry(self, study, *, detail, macro, boundary_h=None):
        return build_geometry(
            self.config, study, detail=detail, macro=macro, boundary_h=boundary_h
        )

    def interface_patches(self) -> list[PortPatch]:
        # Macro block bottom (= die top, macro z=0): full lateral extent.
        return _full_face_patches(self.config, Face.ZM, 0.0)

    def boundary_patch_groups(self):
        return _boundary_patch_groups(self.config)

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 2:
            raise ValueError("bci_pkg has exactly two boundary groups")
        return {"top": float(h_vec[0]), "side": float(h_vec[1])}

    def group_h_ranges(self):
        return self.config.h_ranges

    def detail_interface_patches(self) -> list[PortPatch]:
        # Interface on the detail model = die top, same lateral extent.
        return _full_face_patches(self.config, Face.ZP, self.config.die_h_mm * 1e-3)

    @property
    def detail_nz(self) -> int:
        return self.config.detail_nz

    def monitor_cells(self) -> np.ndarray:
        return _detail_monitor_cells(self.config, self._detail_steady)

    @cached_property
    def port_lookup(self) -> dict[tuple[int, int], int]:
        return {
            (int(ix), int(iy)): port
            for port, (ix, iy) in enumerate(
                (ix, iy) for ix in range(self.config.nx) for iy in range(self.config.nx)
            )
        }


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = BciPkgConfig(**(overrides or {}))
    return _BciPkg(cfg)
