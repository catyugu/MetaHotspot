#!/usr/bin/env python3
"""Case-1 thermal model: heat-sink stack + 4 silicon dies (FloTHERM BCI ROM case).

Geometry mirrors ``playground/bci_rom_testcase1/case1.ecxml`` (FloTHERM
solution domain 50 x 50 x 8 mm, origin at (-25, -25, 0) mm):

    block        material   z (mm)          lateral         power
    ----------   --------   --------------  -------------   -----
    Heat Sink    Aluminum   0.00 - 6.00     50 x 50         -
    TIM2         TIM        6.00 - 6.05     38 x 38         -
    IHS          Copper     6.05 - 7.55     38 x 38         -
    TIM          TIM        7.55 - 7.60     24 x 24         -
    dies         Silicon    7.60 - 8.00     see DIES        4 x 5 W

Four 0.4 mm silicon dies sit on the 24 x 24 mm TIM (all at z 7.6 .. 8.0 mm):

    S0  x[  6,  9]  y[-5, 10]  5 W
    S1  x[-10,  5]  y[-5, 10]  5 W
    S2  x[  6,  9]  y[-9, -6]  5 W
    S3  x[-10,  5]  y[-9, -6]  5 W

Boundary: three ambient groups, every other exterior facet adiabatic.  The
groups are named and ordered exactly as the FloTHERM export
(``MATRICES/report.rom``, ``MATRICES/xData.txt``), so index ``k`` of the ``h``
vector is index ``k`` of ``delta_H_bci_hat[k]`` and ``Areas[k]``:

    heat-sink sides   (4 x 50 x 6 mm = 1200 mm2)   h = 1.0e2   (Ambient:2)
    heat-sink bottom  (50 x 50 mm    = 2500 mm2)   h = 1.0e3   (Ambient:1)
    die crowns        (4 die tops    =  324 mm2)   h = 3.0e1   (Ambient:0)

Those areas are exactly the exported ``Areas[k].mtx`` values
(1.2e-3 / 2.5e-3 / 3.24e-4 m2) and the reference HTCs are ``in.h`` of
``MATRICES/rom_parameters.m``.  Only solid blocks carry cells; the unfilled
volume of the FloTHERM solution domain (around the TIM/IHS/dies) is void, as in
the extracted ROM, so the ambient groups see only real solid surface.

Mesh: every block edge is a vertex and every axis is subdivided so that no cell
exceeds ``max_cell_mm`` (0.5 mm).  The 0.05 mm TIM films therefore stay one cell
thick while the dies, the IHS and the heat sink are resolved laterally and
vertically.

``h_ranges`` keeps the group order and gives every group the *physical* HTC
range [1, 1e4] W/m2.K reported by ``report.rom``.  The public
boundary-parameter space is the physical HTC ``h``: callers pass physical values
to :meth:`~AffineParametricModel.full_reference`; the model maps them internally
through :meth:`~AffineParametricModel.physical_to_effective` to the
surface-consistent effective coefficient ``p`` before assembling the affine
``K``.  The ROM is trained over the same effective ``p``
(:func:`~metahotspot.macromodel.utils.assemble_reduced_k` fed
``model.physical_to_effective(h)``; :meth:`~AffineParametricModel.h_ranges`
returns the effective training ranges).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

import metahotspot
from metahotspot.enums import Face
from metahotspot.macromodel.affine import (
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
)
from metahotspot.macromodel.geometry import BoundarySurface

# (kx, ky, kz, rho, c) SI — the four solids present in this model
MATERIALS = {
    "Aluminum": ("200", "200", "200", "2700", "900"),
    "TIM": ("8", "8", "8", "3000", "900"),
    "Copper": ("400", "400", "400", "8960", "385"),
    "Silicon": ("130", "130", "130", "2330", "700"),
}

# bottom-up (name, material, thickness_mm, size_x_mm, size_y_mm, center_x, center_y)
STACK = (
    ("Heat Sink", "Aluminum", 6.0, 50.0, 50.0, 0.0, 0.0),
    ("TIM2", "TIM", 0.05, 38.0, 38.0, 0.0, 0.0),
    ("IHS", "Copper", 1.5, 38.0, 38.0, 0.0, 0.0),
    ("TIM", "TIM", 0.05, 24.0, 24.0, 0.0, 0.0),
)

# (name, (xlo, xhi), (ylo, yhi)) in mm — the assembly "Core" blocks of case1.ecxml
DIES = (
    ("S0", (6.0, 9.0), (-5.0, 10.0)),
    ("S1", (-10.0, 5.0), (-5.0, 10.0)),
    ("S2", (6.0, 9.0), (-9.0, -6.0)),
    ("S3", (-10.0, 5.0), (-9.0, -6.0)),
)
DIE_POWER_W = (5.0, 5.0, 5.0, 5.0)
DIE_THICKNESS_MM = 0.4
DIE_MATERIAL = "Silicon"

DEFAULT_H_RANGE = (1.0, 1.0e4)


@dataclass(frozen=True)
class Case1Config:
    ambient_K: float = 308.15  # 35 C = in.Tamb / in.Tinitial (rom_parameters.m)
    duration_s: float = 1000.0  # in.tscan = [0, 1e3]
    dt_s: float = 20.0
    max_cell_mm: float = 0.5  # every axis is subdivided to at most this cell
    # group order = FloTHERM export order (Ambient:2, Ambient:1, Ambient:0);
    # every group is trained over the reported physical range [1, 1e4]
    h_ranges: tuple = (DEFAULT_H_RANGE, DEFAULT_H_RANGE, DEFAULT_H_RANGE)

    @property
    def layer_count(self) -> int:
        return len(STACK) + 1  # four stacked blocks + the die slab

    @property
    def total_height_mm(self) -> float:
        stack_mm = sum(thickness for _, _, thickness, _, _, _, _ in STACK)
        return stack_mm + DIE_THICKNESS_MM

    @staticmethod
    def _cuts_mm(axis: int) -> np.ndarray:
        """Every lateral block edge on ``axis`` (0 = x, 1 = y), unique."""
        edges = []
        for _, _, _, size_x, size_y, center_x, center_y in STACK:
            size, center = (size_x, center_x) if axis == 0 else (size_y, center_y)
            edges.extend((center - size / 2, center + size / 2))
        for _, (xlo, xhi), (ylo, yhi) in DIES:
            edges.extend((xlo, xhi) if axis == 0 else (ylo, yhi))
        return np.unique(np.asarray(edges, dtype=np.float64))

    def _subdivide(self, fixed) -> np.ndarray:
        vertices = [float(fixed[0])]
        for left, right in pairwise(fixed):
            pieces = max(1, math.ceil((right - left) / self.max_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @property
    def x_vertices_mm(self) -> np.ndarray:
        return self._subdivide(self._cuts_mm(0))

    @property
    def y_vertices_mm(self) -> np.ndarray:
        return self._subdivide(self._cuts_mm(1))

    @property
    def z_vertices_mm(self) -> np.ndarray:
        """Cumulative z breakpoints of the stack: 0, 6, 6.05, 7.55, 7.6, 8."""
        fixed = [0.0]
        for _, _, thickness, _, _, _, _ in STACK:
            fixed.append(fixed[-1] + thickness)
        fixed.append(fixed[-1] + DIE_THICKNESS_MM)
        return self._subdivide(np.asarray(fixed))

    @property
    def nx(self) -> int:
        return self.x_vertices_mm.size - 1

    @property
    def ny(self) -> int:
        return self.y_vertices_mm.size - 1

    @property
    def nz(self) -> int:
        return self.z_vertices_mm.size - 1

    def report_dict(self) -> dict:
        return {
            **asdict(self),
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "source_power_W": list(DIE_POWER_W),
            "domain_mm": (
                float(self.x_vertices_mm.min()),
                float(self.x_vertices_mm.max()),
                float(self.y_vertices_mm.min()),
                float(self.y_vertices_mm.max()),
                self.total_height_mm,
            ),
        }


def build_geometry(cfg: Case1Config, *, detail: bool, macro: bool):
    """Assemble the Case-1 geometry (no boundary conditions; default Neumann)."""
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    model = metahotspot.Model()
    model.set_reference_temperature(cfg.ambient_K)
    model.set_cartesian_mesh(
        cfg.x_vertices_mm * 1e-3, cfg.y_vertices_mm * 1e-3, cfg.z_vertices_mm * 1e-3
    )
    materials = {
        name: model.add_material(name, *props) for name, props in MATERIALS.items()
    }

    def add_box(material, lo, hi, power=0.0):
        source = (
            model.add_source(f"{power / (np.prod(np.array(hi) - lo) * 1e-9):.17g}")
            if power > 0
            else 0xFFFFFFFF
        )
        volume = model.add_volume(material, source)
        model.add_box(volume, np.array(lo) * 1e-3, np.array(hi) * 1e-3)

    z = 0.0
    for _, material, thickness, sx, sy, cx, cy in STACK:
        add_box(
            materials[material],
            [cx - sx / 2, cy - sy / 2, z],
            [cx + sx / 2, cy + sy / 2, z + thickness],
        )
        z += thickness
    for (_, (xlo, xhi), (ylo, yhi)), power in zip(DIES, DIE_POWER_W, strict=True):
        add_box(
            materials[DIE_MATERIAL],
            [xlo, ylo, z],
            [xhi, yhi, z + DIE_THICKNESS_MM],
            power,
        )
    model.set_default_neumann("0")
    return model


def merge_surfaces(*surfaces: BoundarySurface) -> BoundarySurface:
    """One boundary surface from several face selections of the same group."""
    return BoundarySurface(
        cell_ids=np.concatenate([s.cell_ids for s in surfaces]),
        areas=np.concatenate([s.areas for s in surfaces]),
        facet_ids=np.concatenate([s.facet_ids for s in surfaces]),
    )


class Case1Model(AffineParametricModel):
    """Concrete affine parametric model: BCI ROM Case 1."""

    def __init__(self, cfg: Case1Config | None = None):
        self.config = cfg or Case1Config()

    @property
    def name(self) -> str:
        return "bci_case1"

    # ------------------------------------------------ geometry hooks

    def build_geometry(self, *, detail, macro):
        return build_geometry(self.config, detail=detail, macro=macro)

    def source_ports(self) -> list[SourcePort]:
        f = np.asarray(self.core.f, dtype=np.float64)
        source_cells = np.flatnonzero(f > 0.0)
        # All four dies sit in the same top z slab; split them into per-die
        # ports by x/y footprint (block ids do not distinguish them).
        return self._gate_by_geometry(source_cells)

    def _gate_by_geometry(self, source_cells) -> list[SourcePort]:
        centers = self.cell_layout.centers
        ports = []
        for (name, (xlo, xhi), (ylo, yhi)), power in zip(
            DIES, DIE_POWER_W, strict=True
        ):
            mask = self._footprint(centers, xlo, xhi, ylo, yhi)
            cells = source_cells[mask[source_cells]]
            if not cells.size:
                raise ValueError(f"no source cells inside die {name}")
            ports.append(
                SourcePort(cells=np.asarray(cells, dtype=np.int64), power_W=power)
            )
        return ports

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        """Sides, bottom, crowns — the FloTHERM export (Ambient:2/1/0) order."""
        return (
            BoundaryGroup(
                surface=merge_surfaces(
                    self.geometry.surface(Face.XM),
                    self.geometry.surface(Face.XP),
                    self.geometry.surface(Face.YM),
                    self.geometry.surface(Face.YP),
                ),
                h_range=self.config.h_ranges[0],
            ),
            BoundaryGroup(
                surface=self.geometry.surface(Face.ZM, coordinate=0.0),
                h_range=self.config.h_ranges[1],
            ),
            BoundaryGroup(
                surface=self.geometry.surface(Face.ZP),
                h_range=self.config.h_ranges[2],
            ),
        )

    @staticmethod
    def _footprint(centers, xlo, xhi, ylo, yhi) -> np.ndarray:
        """Boolean mask over cells whose x/y centre sits inside a die footprint."""
        return (
            (centers[:, 0] >= xlo * 1.0e-3 - 1.0e-12)
            & (centers[:, 0] <= xhi * 1.0e-3 + 1.0e-12)
            & (centers[:, 1] >= ylo * 1.0e-3 - 1.0e-12)
            & (centers[:, 1] <= yhi * 1.0e-3 + 1.0e-12)
        )
