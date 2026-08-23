"""Concrete affine parametric model: the chiplet+spreader+cold-plate stack.

This module is *private* — it is reachable only through the factory under the
registered name ``"chiplet_stack"``.  It supplies the geometry hooks consumed by
the shared :class:`AffineParametricModel` base (which carries the full-domain
operator assembly, source-port extraction, cell-geometry helpers and boundary
affine terms).

Layout (z from 0 up): substrate (organic) / bump (underfill+Cu pillars) / die
(silicon, chiplet heat sources) at the bottom, then TIM / spreader (copper) /
cold plate (aluminum) on top.  Each of the four chiplets is one uniform
heat-source port (FANTASTIC); the cold-plate top face is the single parametric
boundary group.  The public boundary-parameter space is the *physical* HTC ``h``
(W/m²·K): callers pass physical values to
:meth:`~AffineParametricModel.full_reference` / ``parameter_points``; the
model maps them internally through
:meth:`~AffineParametricModel.physical_to_effective` to the
surface-consistent effective coefficient ``p`` before assembling the affine
``K``.  The ROM is trained over the same effective ``p``
(:func:`~utils.assemble_reduced_k` fed ``model.physical_to_effective(h)``;
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

from affine_parametric_models._interfaces import (
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
    surface_exposed_cells,
)

DEFAULT_H_RANGE = (1.0, 1.0e6)

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

QUICK_OVERRIDES = {
    "max_xy_cell_mm": 4.0,
}


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
    def source_port_count(self) -> int:
        return 4  # one uniform heat-source block per chiplet

    @property
    def nominal_power_W(self) -> float:
        return self.chiplet_power_W * float(sum(CHIPLET_POWER_SCALE))

    def report_dict(self) -> dict:
        return {
            **asdict(self),
            "nx": self.nx,
            "ny": self.nx,
            "nz": self.nz,
            "source_ports": self.source_port_count,
            "source_port_shape": [4, 1],
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


def build_geometry(
    cfg: ChipletStackConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    source_sink: list | None = None,
):
    """Assemble the chiplet stack (no boundary conditions; default Neumann).

    When ``source_sink`` is given, each heat-source block appends
    ``(block_id, power_W, activity_index)`` to it (in add order), so the model
    can recover per-source cells from the compiled ``block_ids``.
    """
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")

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

        # One heat-source block per chiplet (uniform power over the whole
        # 12x12 mm chiplet), instead of 4x4 = 16 tiles per chiplet.  Each
        # chiplet carries its own activity trace.
        chiplet_volume_m3 = (
            cfg.chiplet_size_mm * cfg.chiplet_size_mm * cfg.die_mm * 1.0e-9
        )
        # die-layer block id counter: 0 = die base block, then one per chiplet
        die_block_count = 1
        for chiplet, ((x0, y0), scale) in enumerate(
            zip(cfg.chiplet_origins_mm, CHIPLET_POWER_SCALE)
        ):
            chiplet_power = cfg.chiplet_power_W * scale
            source = f"{chiplet_power / chiplet_volume_m3:.17g}"
            activity_idx = chiplet % 4
            if transient:
                source += f"*activity_{activity_idx}(x)"
            block = model.add_block(die, "silicon", heat_source=source)
            model.add_rect(
                block,
                GeometryOp.ADD,
                f"{x0:.17g}",
                f"{y0:.17g}",
                f"{cfg.chiplet_size_mm:.17g}",
                f"{cfg.chiplet_size_mm:.17g}",
            )
            if source_sink is not None:
                source_sink.append(
                    (die_block_count, float(chiplet_power), activity_idx)
                )
            die_block_count += 1

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

    return model


class _ChipletStack(AffineParametricModel):
    """Private concrete implementation registered as ``"chiplet_stack"``."""

    def __init__(self, cfg: ChipletStackConfig):
        self.config = cfg
        self._source_sink: list = []

    @property
    def name(self) -> str:
        return "chiplet_stack"

    # ------------------------------------------------- geometry hooks

    def build_geometry(self, study, *, detail, macro):
        self._source_sink.clear()
        return build_geometry(
            self.config,
            study,
            detail=detail,
            macro=macro,
            source_sink=self._source_sink,
        )

    def _axis_vertices(self, axis: str) -> np.ndarray:
        """SI vertex array along x/y/z (shared geometry hook)."""
        if axis in ("x", "y"):
            return self.config.axis_vertices_mm * 1.0e-3
        return (
            z_vertices((*self.config.detail_layers, *self.config.macro_layers)) * 1.0e-3
        )

    def source_ports(self) -> list[SourcePort]:
        """One :class:`SourcePort` per heat-source block (4 chiplet regions).

        Source cells are located directly from the compiled model: the cells
        with non-zero constant heat-source RHS.  They are grouped by block id
        (each heat-source block has exactly one block id), and the sink (in
        the same add order) supplies the power and activity trace.
        """
        f = np.asarray(self._core.f, dtype=np.float64)
        source_cells = np.flatnonzero(f > 0.0)
        block = self._full.block_ids[source_cells]
        # group by block id, preserving block-id order
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
        for cells, (block_id, power_W, activity_idx) in zip(groups, self._source_sink):
            if activity_idx is not None:
                ports.append(
                    SourcePort(
                        cells=np.asarray(cells, dtype=np.int64),
                        power_W=power_W,
                        activity=lambda t, idx=activity_idx: self._activity_value(
                            idx, t
                        ),
                    )
                )
            else:
                ports.append(
                    SourcePort(cells=np.asarray(cells, dtype=np.int64), power_W=power_W)
                )
        return ports

    def _activity_value(self, index: int, t: float) -> float:
        """Piecewise-linear activity trace ``index`` evaluated at time ``t``."""
        points = np.asarray(ACTIVITY_TRACES[index], dtype=np.float64)
        xs = points[:, 0] * self.config.duration_s
        ys = points[:, 1]
        return float(np.interp(t, xs, ys))

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        full = self._full
        grid = full.grid_to_cell.reshape(full.nx, full.ny, full.nz)
        x = self.config.axis_vertices_mm * 1.0e-3
        y = x
        z = self._axis_vertices("z")
        top_z = (self.config.detail_height_mm + self.config.macro_height_mm) * 1.0e-3
        cells, areas = surface_exposed_cells(grid, x, y, z, Face.ZP, top_z)
        return (
            BoundaryGroup(
                cells=cells,
                areas=areas,
                h_range=DEFAULT_H_RANGE,
            ),
        )

    def _physical_stack(self) -> tuple[tuple[float, float, float, float], ...]:
        """Bottom-up ``(thickness_mm, kx, ky, kz)`` physical layer stack.

        substrate(organic) / bump(underfill) / die(silicon) at the bottom,
        then tim / spreader(copper) / cold-plate(aluminum) on top.  Single
        ground truth for layer layout; the base derives
        ``_layer_conductivity`` (layer_id 0 = top = cold plate) and asserts
        every ``layer_id``'s compiled z-band against it.
        """
        material_k = {
            name: (float(kx), float(ky), float(kz))
            for name, kx, ky, kz, _, _ in MATERIALS
        }
        return (
            (self.config.substrate_mm, *material_k["organic"]),
            (self.config.bump_mm, *material_k["underfill"]),
            (self.config.die_mm, *material_k["silicon"]),
            (self.config.tim_mm, *material_k["tim"]),
            (self.config.spreader_mm, *material_k["copper"]),
            (self.config.cold_plate_mm, *material_k["aluminum"]),
        )

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 1:
            raise ValueError("chiplet_stack has exactly one boundary group")
        return {"top": float(h_vec[0])}

    def group_h_ranges(self):
        return (DEFAULT_H_RANGE,)

    def source_power(self, t: float) -> np.ndarray:
        ports = self.source_ports()
        return np.asarray(
            [p.power_W * (p.activity(t) if p.activity else 1.0) for p in ports],
            dtype=np.float64,
        )

    @property
    def detail_nz(self) -> int:
        return self.config.detail_nz


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = ChipletStackConfig(**(overrides or {}))
    return _ChipletStack(cfg)
