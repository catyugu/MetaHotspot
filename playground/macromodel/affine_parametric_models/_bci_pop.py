"""Concrete affine parametric model: Flotherm §4.1 Package-on-Package (PoP).

This module is *private* — reachable only through the factory under the
registered name ``"bci_pop"``.  It supplies the geometry hooks consumed by the
shared :class:`AffineParametricModel` base.

The geometry mirrors the Package-on-Package validation case of the Simcenter
Flotherm BCI-ROM Validation (v2020.2, Sec. 4.1): two stacked wire-bonded BGA
packages — a *bottom logic* package (14 x 14 mm, die 3 W) and a *top memory*
package (10 x 10 mm, die 2 W) — joined by a solder-ball interconnect.  The
boundary is partitioned into the three Flotherm ambient groups {top, sides,
bottom}, each with an independent heat-exchange coefficient drawn from an
admissible range (default 1 .. 1e4 W/m^2K, Flotherm's default extraction
range).  Transient horizon 700 s (Flotherm §4.1).

Layer stack (physical z from bottom to top):

    layer               material    thickness (mm)   lateral (mm)   power
    ------------------  ----------  --------------   -----------   ----
    bottom substrate    organic     0.60             14 x 14       -
    bottom die-attach   underfill   0.05              8 x  8       -
    bottom die          silicon     0.40              8 x  8       3 W
    bottom mold         mold        0.90             14 x 14       -
    solder balls        solder      0.40             11 x 11       -
    top substrate       organic     0.60             10 x 10       -
    top die-attach      underfill   0.05              6 x  6       -
    top die             silicon     0.40              6 x  6       2 W
    top mold            mold        0.90             10 x 10       -

The public boundary-parameter space is the *physical* HTC ``h``
(W/m²·K): callers pass physical values to
:meth:`~AffineParametricModel.full_reference` / ``parameter_points``; the
model maps them internally through
:meth:`~AffineParametricModel.physical_to_effective` to the
surface-consistent effective coefficient ``p`` before assembling the affine
``K``.  The ROM is trained over the same effective ``p``
(:func:`~metahotspot.macromodel.utils.assemble_reduced_k` fed ``model.physical_to_effective(h)``;
:meth:`~AffineParametricModel.h_ranges` returns the effective training
ranges).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

import metahotspot
from metahotspot.enums import Face, GeometryOp, LengthUnit, Study

from metahotspot.macromodel.affine import (
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
    surface_exposed_cells,
)

MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("underfill", ".8", ".8", ".8", "1550", "1000"),
    ("silicon", "130", "130", "115", "2330", "700"),
    ("mold", ".8", ".8", ".6", "1900", "1000"),
    ("solder", "50", "50", "50", "7400", "230"),
)
DEFAULT_H_RANGE = (1.0, 1.0e4)

QUICK_OVERRIDES = {
    "max_xy_cell_mm": 1.0,
}


@dataclass(frozen=True)
class PopConfig:
    ambient_K: float = 300.0

    # bottom package (logic)
    bottom_size_mm: float = 14.0
    bottom_substrate_mm: float = 0.60
    bottom_dieattach_mm: float = 0.05
    bottom_die_mm: float = 0.40
    bottom_die_size_mm: float = 8.0
    bottom_mold_mm: float = 0.90
    bottom_power_W: float = 3.0

    # interconnect
    solder_mm: float = 0.40
    solder_size_mm: float = 11.0

    # top package (memory)
    top_size_mm: float = 10.0
    top_substrate_mm: float = 0.60
    top_dieattach_mm: float = 0.05
    top_die_mm: float = 0.40
    top_die_size_mm: float = 6.0
    top_mold_mm: float = 0.90
    top_power_W: float = 2.0

    max_xy_cell_mm: float = 0.5
    duration_s: float = 700.0
    dt_s: float = 25.0
    h_ranges: tuple = (
        DEFAULT_H_RANGE,
        DEFAULT_H_RANGE,
        DEFAULT_H_RANGE,
    )  # [top, sides, bottom]

    @property
    def layers(self):
        """(thickness, material, size_mm, power_W) bottom-up."""
        return (
            (self.bottom_substrate_mm, "organic", self.bottom_size_mm, 0.0),
            (self.bottom_dieattach_mm, "underfill", self.bottom_die_size_mm, 0.0),
            (
                self.bottom_die_mm,
                "silicon",
                self.bottom_die_size_mm,
                self.bottom_power_W,
            ),
            (self.bottom_mold_mm, "mold", self.bottom_size_mm, 0.0),
            (self.solder_mm, "solder", self.solder_size_mm, 0.0),
            (self.top_substrate_mm, "organic", self.top_size_mm, 0.0),
            (self.top_dieattach_mm, "underfill", self.top_die_size_mm, 0.0),
            (self.top_die_mm, "silicon", self.top_die_size_mm, self.top_power_W),
            (self.top_mold_mm, "mold", self.top_size_mm, 0.0),
        )

    @property
    def nz(self) -> int:
        return len(self.layers)

    @property
    def total_height_mm(self) -> float:
        return sum(th for th, _, _, _ in self.layers)

    @cached_property
    def axis_vertices_mm(self) -> np.ndarray:
        """Lateral breakpoints: all block edges + package extents, subdivided."""
        half_b = self.bottom_size_mm / 2.0
        half_t = self.top_size_mm / 2.0
        half_s = self.solder_size_mm / 2.0
        half_db = self.bottom_die_size_mm / 2.0
        half_dt = self.top_die_size_mm / 2.0
        points = [
            -half_b,
            -half_s,
            -half_t,
            -half_db,
            -half_dt,
            0.0,
            half_dt,
            half_db,
            half_t,
            half_s,
            half_b,
        ]
        fixed = np.unique(np.asarray(points, dtype=np.float64))
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
            **asdict(self),
            "nx": self.nx,
            "ny": self.nx,
            "nz": self.nz,
            "layers": [(m, th, sz, pw) for th, m, sz, pw in self.layers],
            "source_power_W": [self.bottom_power_W, self.top_power_W],
        }


def z_vertices(layers) -> np.ndarray:
    """Cumulative z breakpoints (mm) for the bottom-up layer stack."""
    vertices = [0.0]
    for thickness, _, _, _ in layers:
        vertices.append(vertices[-1] + thickness)
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
    cfg: PopConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
):
    """Assemble the PoP geometry (no boundary conditions).

    Layers are added *top-first*: metahotspot's ``add_layer`` pushes each new
    layer to the bottom (the last added sits at z=0), so the physical bottom-up
    stack ``cfg.layers`` is added in reverse order.  Boundary conditions are
    deliberately *not* applied here — all boundary terms enter through the
    affine :meth:`boundary_terms`, so the BCI basis/closure is h-free.
    """
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
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, z_vertices(cfg.layers))
    for material in MATERIALS:
        model.add_material(*material)

    # top-first: iterate layers reversed, each add_layer pushes to the bottom.
    # A heat-source layer's single block *is* the die and carries the source;
    # plain layers get one unpowered block.
    for thickness, material, size, power in reversed(cfg.layers):
        layer = model.add_layer(str(thickness))
        if power > 0.0:
            volume_m3 = size * size * thickness * 1.0e-9
            block = model.add_block(
                layer,
                material,
                heat_source=f"{power / volume_m3:.17g}",
            )
        else:
            block = model.add_block(layer, material)
        add_square(model, block, size)

    model.set_default_neumann("0")
    return model


class _BciPop(AffineParametricModel):
    """Private concrete implementation registered as ``"bci_pop"``."""

    def __init__(self, cfg: PopConfig):
        self.config = cfg

    @property
    def name(self) -> str:
        return "bci_pop"

    # ------------------------------------------------- geometry hooks

    def build_geometry(self, study, *, detail, macro):
        return build_geometry(self.config, study, detail=detail, macro=macro)

    def source_ports(self) -> list[SourcePort]:
        """Two die heat sources, split by z (bottom die vs top die).

        The two dies sit in different z slabs but are the same material, so
        block ids do not distinguish them; split the ``f > 0`` cells by their
        z-centre across the solder layer.
        """
        f = np.asarray(self._core.f, dtype=np.float64)
        source_cells = np.flatnonzero(f > 0.0)
        zc = self.cell_layout.centers[:, 2]
        bottom_hi = (
            self.config.bottom_substrate_mm
            + self.config.bottom_dieattach_mm
            + self.config.bottom_die_mm
            + self.config.bottom_mold_mm
        )
        mid_z = (bottom_hi + 0.5 * self.config.solder_mm) * 1.0e-3  # mid-solder
        bottom = source_cells[zc[source_cells] < mid_z]
        top = source_cells[zc[source_cells] >= mid_z]
        if bottom.size == 0 or top.size == 0:
            raise RuntimeError("bci_pop source z-split failed (empty die slab)")
        return [
            SourcePort(
                cells=np.asarray(bottom, dtype=np.int64),
                power_W=self.config.bottom_power_W,
            ),
            SourcePort(
                cells=np.asarray(top, dtype=np.int64), power_W=self.config.top_power_W
            ),
        ]

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        full = self._full
        cells = full.cells
        half_b = self.config.bottom_size_mm / 2.0 * 1.0e-3
        half_t = self.config.top_size_mm / 2.0 * 1.0e-3
        half_s = self.config.solder_size_mm / 2.0 * 1.0e-3
        total_h = cells.z_vertices[-1]

        # physical z-intervals (m) for the exposed-wall ranges.
        bottom_hi = (
            self.config.bottom_substrate_mm
            + self.config.bottom_dieattach_mm
            + self.config.bottom_die_mm
            + self.config.bottom_mold_mm
        ) * 1.0e-3
        solder_lo = bottom_hi
        solder_hi = bottom_hi + self.config.solder_mm * 1.0e-3
        top_lo = solder_hi

        # top group: top-mold top face (ZP at full height)
        top_cells, top_areas = surface_exposed_cells(cells, Face.ZP, total_h)

        # bottom group: bottom-substrate bottom face (ZM at z=0)
        bot_cells, bot_areas = surface_exposed_cells(cells, Face.ZM, 0.0)

        # sides group: all exposed lateral walls.  The bottom package outer
        # walls (x/y = ±half_b) run z [0, bottom_hi]; the solder overhang adds
        # walls at ±half_s over [solder_lo, solder_hi]; the smaller top package
        # adds walls at ±half_t over [top_lo, total_h].  surface_exposed_cells'
        # neighbour check picks up exactly the cells that are truly exposed.
        side_cells, side_areas = [], []
        for axis, coord, lo, hi in (
            (Face.XM, -half_b, 0.0, bottom_hi),
            (Face.XP, half_b, 0.0, bottom_hi),
            (Face.YM, -half_b, 0.0, bottom_hi),
            (Face.YP, half_b, 0.0, bottom_hi),
            (Face.XM, -half_s, solder_lo, solder_hi),
            (Face.XP, half_s, solder_lo, solder_hi),
            (Face.YM, -half_s, solder_lo, solder_hi),
            (Face.YP, half_s, solder_lo, solder_hi),
            (Face.XM, -half_t, top_lo, total_h),
            (Face.XP, half_t, top_lo, total_h),
            (Face.YM, -half_t, top_lo, total_h),
            (Face.YP, half_t, top_lo, total_h),
        ):
            wall_cells, wall_areas = surface_exposed_cells(
                cells, axis, coord, z_range=(lo, hi)
            )
            side_cells.append(wall_cells)
            side_areas.append(wall_areas)
        side_cells = (
            np.concatenate(side_cells) if side_cells else np.empty(0, dtype=np.int64)
        )
        side_areas = (
            np.concatenate(side_areas) if side_areas else np.empty(0, dtype=np.float64)
        )

        return (
            BoundaryGroup(
                cells=top_cells,
                areas=top_areas,
                h_range=self.config.h_ranges[0],
            ),
            BoundaryGroup(
                cells=side_cells,
                areas=side_areas,
                h_range=self.config.h_ranges[1],
            ),
            BoundaryGroup(
                cells=bot_cells,
                areas=bot_areas,
                h_range=self.config.h_ranges[2],
            ),
        )

    def _boundary_axis_per_group(self) -> tuple[int, ...]:
        """Face-normal axis for each boundary group: (top=z, sides=x, bottom=z).

        Used by :meth:`~AffineParametricModel.physical_to_effective` /
        :meth:`~AffineParametricModel.h_ranges` to pick the per-cell (kx, ky,
        kz) and cell-side half-distance for the FloTHERM ThirdType series
        condensation.  The sides group faces sideways (x/y), so its effective
        coefficient uses the lateral conductivity/half rather than the z-axis.
        """
        return (2, 0, 2)

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 3:
            raise ValueError("bci_pop has exactly three boundary groups")
        return {
            "top": float(h_vec[0]),
            "sides": float(h_vec[1]),
            "bottom": float(h_vec[2]),
        }

    def group_h_ranges(self):
        return self.config.h_ranges

    def parameter_points(self, count: int = 36) -> list[tuple[float, ...]]:
        """Log-uniform random 3-vectors over the physical {top, sides, bottom}.

        The three ambient groups are independent, so validation points span
        their full physical ``(h_top, h_sides, h_bottom)`` space.  ``count``
        log-uniform random draws (deterministic seed) mirror the Flotherm
        DoE-style scenario set (40 scenarios in the validation doc).  The
        returned points are *physical* HTC validation scenarios;
        :meth:`~AffineParametricModel.full_reference` maps them internally,
        and the ROM must be assembled with
        ``model.physical_to_effective(h)``.
        """
        ranges = np.asarray(
            [g.h_range for g in self.boundary_groups()], dtype=np.float64
        )  # physical HTC
        if ranges.size == 0:
            return []
        lows = np.log10(ranges[:, 0])
        highs = np.log10(ranges[:, 1])
        rng = np.random.default_rng(20260805)
        draws = 10.0 ** rng.uniform(lows, highs, size=(count, ranges.shape[0]))
        return [tuple(float(v) for v in row) for row in draws]


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = PopConfig(**(overrides or {}))
    return _BciPop(cfg)
