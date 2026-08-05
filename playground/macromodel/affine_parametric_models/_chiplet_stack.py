"""Concrete affine parametric model: the chiplet+spreader+cold-plate stack.

This module is *private* — it is reachable only through the factory under the
registered name ``"chiplet_stack"``.  It supplies the geometry hooks consumed by
the shared :class:`AffineParametricModel` base (which carries the DtN/PortMap
plumbing).

Layout (z from 0 up): substrate (organic) / bump (underfill+Cu pillars) / die
(silicon, chiplet heat sources) at the bottom (the *detail* domain), then TIM /
spreader (copper) / cold plate (aluminum) on top (the *macro* domain).  The
macro block bottom face is the interface (= die top); its top face is the
single parametric boundary group (uniform heat-exchange coefficient ``h``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

import metahotspot
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortPatch

from affine_parametric_models._interfaces import AffineParametricModel

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
    """Face regions (macro frame, z=0 at macro base) for the top boundary group."""
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

    ``boundary_h`` (if given) maps boundary-group name -> coefficient and
    applies the group's convection via :func:`apply_boundary_convection`.
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


def _patches(
    cfg: ChipletStackConfig, face: Face, z_m: float, indices=None
) -> list[PortPatch]:
    vertices = cfg.axis_vertices_mm * 1.0e-3
    if indices is None:
        indices = range(vertices.size - 1)
    return [
        PortPatch(
            int(face),
            z_m,
            (vertices[ix], vertices[ix + 1], vertices[iy], vertices[iy + 1]),
        )
        for ix in indices
        for iy in indices
    ]


def _patch_areas(cfg: ChipletStackConfig, patches: list[PortPatch]) -> np.ndarray:
    areas = np.empty(len(patches), dtype=np.float64)
    for index, patch in enumerate(patches):
        a_min, a_max, b_min, b_max = patch.rectangle
        areas[index] = (a_max - a_min) * (b_max - b_min)
    return areas


class _ChipletStack(AffineParametricModel):
    """Private concrete implementation registered as ``"chiplet_stack"``."""

    def __init__(self, cfg: ChipletStackConfig):
        self.config = cfg

    @property
    def name(self) -> str:
        return "chiplet_stack"

    # ------------------------------------------------- geometry hooks

    def build_geometry(self, study, *, detail, macro, boundary_h=None):
        return build_geometry(
            self.config, study, detail=detail, macro=macro, boundary_h=boundary_h
        )

    def interface_patches(self) -> list[PortPatch]:
        return _patches(self.config, Face.ZM, 0.0, self.config.port_indices)

    def boundary_patch_groups(self):
        boundary = _patches(self.config, Face.ZP, self.config.macro_height_mm * 1.0e-3)
        return [boundary], [_patch_areas(self.config, boundary)]

    def boundary_h(self, h_vec) -> dict[str, float]:
        if len(h_vec) != 1:
            raise ValueError("chiplet_stack has exactly one boundary group")
        return {"top": float(h_vec[0])}

    def group_h_ranges(self):
        return ((1.0, 1.0e5),)

    def detail_interface_patches(self) -> list[PortPatch]:
        return _patches(
            self.config,
            Face.ZP,
            self.config.detail_height_mm * 1.0e-3,
            self.config.port_indices,
        )

    @property
    def detail_nz(self) -> int:
        return self.config.detail_nz

    def monitor_cells(self) -> np.ndarray:
        # Chiplet stack has no dedicated monitor semantic; the experiments
        # score full-field accuracy, not per-monitor curves.
        return np.arange(self.detail_cell_count, dtype=np.int64)

    @cached_property
    def port_lookup(self) -> dict[tuple[int, int], int]:
        return {
            (int(ix), int(iy)): port
            for port, (ix, iy) in enumerate(
                (ix, iy)
                for ix in self.config.port_indices
                for iy in self.config.port_indices
            )
        }


def _builder(overrides: dict | None = None, **_kwargs) -> AffineParametricModel:
    cfg = ChipletStackConfig(**(overrides or {}))
    return _ChipletStack(cfg)
