#!/usr/bin/env python3
"""Case-1 thermal model: 3-layer stack + 4 silicon dies (FloTHERM BCI ROM case).

Geometry mirrors ``playground/bci_rom_testcase1/case1.ecxml``:

    layer        material    thickness (mm)   lateral (mm)            power
    ----------   ----------  --------------   --------------------    ----
    FR4          FR4         10.0             60 x 100                -
    aluminum     Aluminum    5.0              60 x  60                -
    E-10         E-10        5.0              40 x  40                -
    dies         Silicon     2.0              4 x (10 x 10)           4 sources

Four 10x10x2 mm silicon dies sit on top (z 20..22 mm) at:
    S0  x[ 5,15]  y[ 5,15]  0.1 W
    S1  x[-15,-5] y[ 5,15]  0.2 W
    S2  x[ 5,15]  y[-15,-5] 0.3 W
    S3  x[-15,-5] y[-15,-5] 0.4 W

Boundary: two ambient groups, side faces adiabatic —

    die-crown faces  (z = 22 mm, 4 dies, area 4e-4 m2)  h = 5e1   (Ambient:0)
    FR4 bottom face  (z =  0 mm, area 6e-3 m2)          h = 1e3   (Ambient:1)

``h_ranges`` keeps the model group order [ZP crowns, ZM FR4]; both are the
*physical* HTC range [1, 1e4] (FloTHERM extraction range).  The public
boundary-parameter space is the physical HTC ``h`` (W/m²·K): callers pass
physical values to :meth:`~AffineParametricModel.full_reference` /
``parameter_points``; the model maps them internally through
:meth:`~AffineParametricModel.physical_to_effective` to the
surface-consistent effective coefficient ``p`` before assembling the affine
``K``.  The ROM is trained over the same effective ``p``
(:func:`~utils.assemble_reduced_k` fed ``model.physical_to_effective(h)``;
:meth:`~AffineParametricModel.h_ranges` returns the effective training
ranges).  In FloTHERM's own z-orientation the die side is its "bottom"
(Ambient:0, small area) and the FR4 side is its "top" (Ambient:1, large
area).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

import metahotspot
from metahotspot.enums import Face, GeometryOp, LengthUnit, Study

import sys
from pathlib import Path

_MACRO = Path(__file__).resolve().parent.parent / "macromodel"
if str(_MACRO) not in sys.path:
    sys.path.insert(0, str(_MACRO))

from affine_parametric_models._interfaces import (  # noqa: E402
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
    surface_exposed_cells,
)

# (kx, ky, kz, rho, c) SI — the four solids present in this model
MATERIALS = {
    "FR4": (".3", ".3", ".3", "1200", "880"),
    "Aluminum (Pure)": ("201", "201", "201", "2710", "913"),
    "E-10": (".56", ".56", ".56", "1140", "3330"),
    "Silicon": ("130", "130", "130", "2330", "700"),
}

AIR_MATERIAL = "air"
AIR_K = "0.02643"
AIR_RHO = "1.149"
AIR_C = "1007"

A2_MIN_X = -30.0
A2_MAX_X = 30.0
A2_MIN_Y = -50.0
A2_MAX_Y = 50.0

# bottom-up (thickness_mm, material, size_x_mm, size_y_mm, center_x_mm, center_y_mm, power_W)
LAYERS = (
    (10.0, "FR4", 60.0, 100.0, 0.0, 0.0, 0.0),
    (5.0, "Aluminum (Pure)", 60.0, 60.0, 0.0, 0.0, 0.0),
    (5.0, "E-10", 40.0, 40.0, 0.0, 0.0, 0.0),
)

DIES = (
    (0.1, (5.0, 15.0), (5.0, 15.0)),  # S0
    (0.2, (-15.0, -5.0), (5.0, 15.0)),  # S1
    (0.3, (5.0, 15.0), (-15.0, -5.0)),  # S2
    (0.4, (-15.0, -5.0), (-15.0, -5.0)),  # S3
)
DIE_THICKNESS_MM = 2.0
DIE_MATERIAL = "Silicon"

DEFAULT_H_RANGE = (1.0, 1.0e4)


@dataclass(frozen=True)
class Case1Config:
    ambient_K: float = 308.15  # 35 C = Tambient (rom_parameters.m)
    duration_s: float = 100.0
    dt_s: float = 5.0
    max_xy_cell_mm: float = 2.5
    max_z_cell_mm: float = 2.5
    # group 0 = die crowns (Ambient:0), group 1 = FR4 bottom (Ambient:1),
    # both physical [1, 1e4] (FloTHERM extraction range); see module docstring
    h_ranges: tuple = (DEFAULT_H_RANGE, DEFAULT_H_RANGE)

    @property
    def nz(self) -> int:
        return 4  # 3 layers + dies

    @property
    def total_height_mm(self) -> float:
        return 22.0

    def _all_cuts_mm(self) -> np.ndarray:
        """Every lateral cut (block edges + package/domain extents), unique.

        FR4/Al x-edges at +-30, FR4 y-edges at +-50, E-10 at +-20, die edges at
        +-15/+-5.  This is the *superset* of cuts; the x and y meshes each keep
        only the cuts inside their own solution-domain extent so no empty cells
        stretch beyond the domain (case1.ecxml solutionDomain 60x100 mm).
        """
        pts = [
            -30.0,
            30.0,  # FR4 / aluminum x edges
            -50.0,
            50.0,  # FR4 y edges
            -20.0,
            20.0,  # E-10 edges
            -15.0,
            -5.0,
            5.0,
            15.0,  # die edges (x & y)
        ]
        return np.unique(np.asarray(pts, dtype=np.float64))

    def _subdivide(self, fixed) -> np.ndarray:
        vertices = [float(fixed[0])]
        for left, right in zip(fixed[:-1], fixed[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_xy_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @property
    def x_vertices_mm(self) -> np.ndarray:
        """x mesh: solution-domain cuts inside [-30, 30] mm (60 mm wide)."""
        full = self._all_cuts_mm()
        return self._subdivide(
            np.unique(full[(full >= A2_MIN_X - 1e-12) & (full <= A2_MAX_X + 1e-12)])
        )

    @property
    def y_vertices_mm(self) -> np.ndarray:
        """y mesh mirroring the FloTHERM base-grid export.

        Central span [-30, 30] mm (aluminum/E-10/dies extent; same cut set
        as x, incl. the aluminum y-edges at +-30) is subdivided to
        ``max_xy_cell_mm``.  The FR4-only overhang [30, 50] / [-50, -30]
        (20 mm per side) is split into ``ceil(20 / max_xy_cell_mm) + 1``
        equal cells — the FloTHERM auto-mesh rule, which at the 1 mm
        nominal gives 21 x (20/21) mm cells per side, exactly reproducing
        the reference ``Root Assembly_Temperature_Base Grid.csv``
        (60 x 1mm central + 21 x (20/21) mm per side = 102 y rows).
        """
        full = self._all_cuts_mm()
        central = np.unique(
            full[(full >= A2_MIN_X - 1e-12) & (full <= A2_MAX_X + 1e-12)]
        )
        inner = self._subdivide(central)
        overhang = A2_MAX_Y - A2_MAX_X  # 20 mm of FR4 beyond the aluminum span
        pieces = max(1, math.ceil(overhang / self.max_xy_cell_mm) + 1)
        outer = np.linspace(A2_MAX_X, A2_MAX_Y, pieces + 1)[1:]  # (30, 50]
        return np.concatenate([-outer[::-1], inner, outer])

    @property
    def z_vertices_mm(self) -> np.ndarray:
        """Cumulative z breakpoints (mm): 0,10,15,20,22, subdivided."""
        fixed = [0.0, 10.0, 15.0, 20.0, 22.0]
        vertices = [0.0]
        for left, right in zip(fixed[:-1], fixed[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_z_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @property
    def nx(self) -> int:
        return self.x_vertices_mm.size - 1

    @property
    def ny(self) -> int:
        return self.y_vertices_mm.size - 1

    def report_dict(self) -> dict:
        return {
            **asdict(self),
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "source_power_W": [d[0] for d in DIES],
            "solid_domain_mm": [
                self.x_vertices_mm.min(),
                self.x_vertices_mm.max(),
            ],
        }


def build_geometry(cfg: Case1Config, study: Study, *, detail: bool, macro: bool):
    """Assemble the Case-1 geometry (no boundary conditions; default Neumman)."""
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    model = metahotspot.Model()
    transient = study == Study.TRANSIENT
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=cfg.duration_s if transient else 0.0,
        output_interval=cfg.dt_s if transient else 0.0,
    )
    model.set_mesh(cfg.x_vertices_mm, cfg.y_vertices_mm, cfg.z_vertices_mm)
    for name, (kx, ky, kz, rho, c) in MATERIALS.items():
        model.add_material(name, kx, ky, kz, rho, c)
    model.add_material(AIR_MATERIAL, AIR_K, AIR_K, AIR_K, AIR_RHO, AIR_C)

    def add_air_background(layer):
        xlo, xhi = A2_MIN_X, A2_MAX_X
        ylo, yhi = A2_MIN_Y, A2_MAX_Y
        air_block = model.add_block(layer, AIR_MATERIAL)
        model.add_rect(
            air_block,
            GeometryOp.ADD,
            f"{xlo:.17g}",
            f"{ylo:.17g}",
            f"{xhi - xlo:.17g}",
            f"{yhi - ylo:.17g}",
        )

    layer = model.add_layer(str(DIE_THICKNESS_MM))
    add_air_background(layer)
    for power, (xlo, xhi), (ylo, yhi) in DIES:
        volume_m3 = (xhi - xlo) * (yhi - ylo) * DIE_THICKNESS_MM * 1.0e-9
        block = model.add_block(
            layer, DIE_MATERIAL, heat_source=f"{power / volume_m3:.17g}"
        )
        model.add_rect(
            block,
            GeometryOp.ADD,
            f"{xlo:.17g}",
            f"{ylo:.17g}",
            f"{xhi - xlo:.17g}",
            f"{yhi - ylo:.17g}",
        )

    for thickness, material, sx, sy, cx, cy, power in reversed(LAYERS):
        layer = model.add_layer(str(thickness))
        add_air_background(layer)
        if power > 0.0:
            volume_m3 = sx * sy * thickness * 1.0e-9
            block = model.add_block(
                layer, material, heat_source=f"{power / volume_m3:.17g}"
            )
        else:
            block = model.add_block(layer, material)
        half_x, half_y = sx / 2.0, sy / 2.0
        model.add_rect(
            block,
            GeometryOp.ADD,
            f"{cx - half_x:.17g}",
            f"{cy - half_y:.17g}",
            f"{sx:.17g}",
            f"{sy:.17g}",
        )

    model.set_default_neumann("0")
    return model


class Case1Model(AffineParametricModel):
    """Concrete affine parametric model: BCI ROM Case 1."""

    def __init__(self, cfg: Case1Config | None = None):
        self.config = cfg or Case1Config()

    @property
    def name(self) -> str:
        return "bci_case1"

    # ------------------------------------------------ geometry hooks

    def build_geometry(self, study, *, detail, macro):
        return build_geometry(self.config, study, detail=detail, macro=macro)

    def source_ports(self) -> list[SourcePort]:
        f = np.asarray(self._core.f, dtype=np.float64)
        source_cells = np.flatnonzero(f > 0.0)
        # All four dies sit in the same top z slab; split them into per-die
        # ports by x/y footprint (block ids do not distinguish them).
        return self._gate_by_geometry(source_cells)

    def _gate_by_geometry(self, source_cells) -> list[SourcePort]:
        cell_x = self.cell_layout.centers[:, 0]
        cell_y = self.cell_layout.centers[:, 1]
        ports = []
        for power, (xlo, xhi), (ylo, yhi) in DIES:
            mask = (
                (cell_x >= xlo * 1.0e-3 - 1.0e-12)
                & (cell_x <= xhi * 1.0e-3 + 1.0e-12)
                & (cell_y >= ylo * 1.0e-3 - 1.0e-12)
                & (cell_y <= yhi * 1.0e-3 + 1.0e-12)
            )
            cells = source_cells[mask[source_cells]]
            ports.append(
                SourcePort(cells=np.asarray(cells, dtype=np.int64), power_W=power)
            )
        return ports

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        full = self._full
        grid = full.grid_to_cell.reshape(full.nx, full.ny, full.nz)
        x = self.config.x_vertices_mm * 1.0e-3
        y = self.config.y_vertices_mm * 1.0e-3
        z = self.config.z_vertices_mm * 1.0e-3
        total_h = self.config.total_height_mm * 1.0e-3

        top_cells, top_areas = surface_exposed_cells(grid, x, y, z, Face.ZP, total_h)
        # The top convective BC applies only to cells carrying real silicon fill
        # (the die crowns); air-only cells on the top layer are dropped so no h
        # is applied to the air domain around the dies.
        cell_x = self.cell_layout.centers[:, 0]
        cell_y = self.cell_layout.centers[:, 1]
        silicon = self._silicon_footprint(cell_x, cell_y)[top_cells]
        top_cells, top_areas = top_cells[silicon], top_areas[silicon]
        bot_cells, bot_areas = surface_exposed_cells(grid, x, y, z, Face.ZM, 0.0)

        return (
            BoundaryGroup(
                cells=top_cells,
                areas=top_areas,
                h_range=self.config.h_ranges[0],
            ),
            BoundaryGroup(
                cells=bot_cells,
                areas=bot_areas,
                h_range=self.config.h_ranges[1],
            ),
        )

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 2:
            raise ValueError("bci_case1 has exactly two boundary groups")
        return {"top": float(h_vec[0]), "bottom": float(h_vec[1])}

    def group_h_ranges(self):
        return self.config.h_ranges

    # ------------------------------------------------ cell geometry helpers

    def _axis_vertices(self, axis: str) -> np.ndarray:
        """SI vertex array along ``axis`` (shared geometry hook)."""
        if axis == "x":
            return self.config.x_vertices_mm * 1.0e-3
        if axis == "y":
            return self.config.y_vertices_mm * 1.0e-3
        return self.config.z_vertices_mm * 1.0e-3

    def _physical_stack(self) -> tuple[tuple[float, float, float, float], ...]:
        """Bottom-up ``(thickness_mm, kx, ky, kz)`` physical layer stack.

        case1's stack is [FR4 10, Aluminum 5, E-10 5] bottom-up plus the
        Silicon die layer (2 mm) on top.  Single ground truth for layer
        layout; the base derives ``_layer_conductivity`` (layer_id 0 = top =
        die) and asserts every ``layer_id``'s compiled z-band against it.
        """
        material_k = {
            name: (float(kx), float(ky), float(kz))
            for name, (kx, ky, kz, _, _) in MATERIALS.items()
        }
        return tuple((float(t), *material_k[m]) for t, m, *_ in LAYERS) + (
            (DIE_THICKNESS_MM, *material_k[DIE_MATERIAL]),
        )

    def _silicon_footprint(self, cell_x, cell_y) -> np.ndarray:
        """Boolean mask over cells whose x/y centre sits inside a die footprint."""
        mask = np.zeros(cell_x.shape, dtype=bool)
        for _, (xlo, xhi), (ylo, yhi) in DIES:
            mask |= (
                (cell_x >= xlo * 1.0e-3 - 1.0e-12)
                & (cell_x <= xhi * 1.0e-3 + 1.0e-12)
                & (cell_y >= ylo * 1.0e-3 - 1.0e-12)
                & (cell_y <= yhi * 1.0e-3 + 1.0e-12)
            )
        return mask


def builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = Case1Config(**(overrides or {}))
    return Case1Model(cfg)
