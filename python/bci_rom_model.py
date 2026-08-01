"""Transient boundary-condition-independent sparse DtN ROM benchmark.

The macro basis is extracted once from a homogeneous-Neumann macro domain. It
uses only the assembled macro operators and the geometric port/column mapping:

1. every physical interface port is retained exactly as a leading DtN state;
2. every lateral mesh column gets a local static-constraint shape;
3. a constant shape and the lowest fixed-interface thermal eigenmodes enrich
   each column; and
4. disjoint column support makes the global basis and projected K/C sparse.

No heat-source distribution, source time waveform, port excitation pattern,
port locality prior, random probe, or response snapshot is used for extraction.
Convection coefficients and applied power histories are introduced only after
the basis has been built, when projected operators are instantiated and tested.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import metahotspot
from metahotspot.compiled import SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import DtNModel, PortMap, PortPatch, solve as solve_macro


@dataclass(frozen=True)
class Package:
    nx: int = 28
    ny: int = 28
    width_mm: float = 40.0
    height_mm: float = 40.0
    ambient_K: float = 300.0
    substrate_mm: float = 1.2
    bump_mm: float = 0.24
    die_mm: float = 0.6
    tim_mm: float = 0.18
    spreader_mm: float = 1.2
    cold_plate_mm: float = 1.5
    substrate_cells: int = 6
    bump_cells: int = 2
    die_cells: int = 4
    tim_cells: int = 2
    spreader_cells: int = 6
    cold_plate_cells: int = 8
    bump_rows: int = 10
    bump_columns: int = 10
    bump_width_mm: float = 0.75
    chiplet_width_mm: float = 12.0
    chiplet_height_mm: float = 12.0
    chiplet_power_W: float = 25.0

    @property
    def detail_nz(self) -> int:
        return self.substrate_cells + self.bump_cells + self.die_cells

    @property
    def macro_nz(self) -> int:
        return self.tim_cells + self.spreader_cells + self.cold_plate_cells

    @property
    def nz(self) -> int:
        return self.detail_nz + self.macro_nz

    @property
    def ports(self) -> int:
        return self.nx * self.ny

    @property
    def total_height_mm(self) -> float:
        return sum(
            (
                self.substrate_mm,
                self.bump_mm,
                self.die_mm,
                self.tim_mm,
                self.spreader_mm,
                self.cold_plate_mm,
            )
        )


@dataclass(frozen=True)
class Run:
    error_K: float = 0.5
    duration_s: float = 0.5
    dt_s: float = 0.025
    nominal_h: float = 2500.0
    h_values: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    dynamic_modes_per_column: int = 2
    speedup_target: float = 1.5
    residual_block_size: int = 32
    report: Path = Path("results/bci_rom_sparse_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s


class Sample(NamedTuple):
    h: float | None
    compiled: object
    ports: PortMap
    K: sp.csc_matrix
    C: sp.csc_matrix
    f: np.ndarray


class Data(NamedTuple):
    full_layout: object
    detail_steady: object
    detail_transient: object
    detail_ports_steady: PortMap
    detail_ports_transient: PortMap
    samples: tuple[Sample, ...]
    detail_cells: np.ndarray
    macro_cells: np.ndarray


class Basis(NamedTuple):
    W: sp.csc_matrix
    column_orders: np.ndarray
    retained_eigenvalues_per_s: np.ndarray
    local_static_residual: float
    projected_static_residual: float
    orthogonality_error: float
    seconds: float


class Reduced(NamedTuple):
    K: sp.csc_matrix
    C: sp.csc_matrix
    f: np.ndarray
    W: sp.csc_matrix
    projection_s: float


class Reference(NamedTuple):
    steady: np.ndarray
    times: np.ndarray
    transient: np.ndarray
    compile_s: float
    steady_solve_s: float
    transient_solve_s: float
    operator_order: int
    operator_k_nnz: int
    operator_c_nnz: int
    operator_bytes: int


def vertices(length: float, cells: int) -> np.ndarray:
    return np.linspace(0.0, length, cells + 1)


def z_vertices(layers) -> np.ndarray:
    out, z = ([0.0], 0.0)
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            out.append(z)
    return np.asarray(out)


def add_materials(model) -> None:
    for args in (
        ("organic", ".65", ".65", ".55", "1900", "1100"),
        ("underfill", ".8", ".8", ".8", "1550", "1000"),
        ("copper", "390", "390", "390", "8960", "385"),
        ("mold", ".85", ".85", ".75", "1850", "1000"),
        ("silicon", "130", "130", "115", "2330", "700"),
        ("tim", "4", "4", "3", "2500", "900"),
        ("aluminum", "180", "180", "180", "2700", "900"),
    ):
        model.add_material(*args)


def full_rect(model, block: int, cfg: Package) -> None:
    model.add_rect(
        block, GeometryOp.ADD, "0", "0", str(cfg.width_mm), str(cfg.height_mm)
    )


def chiplets(cfg: Package):
    x1 = cfg.width_mm - 5.0 - cfg.chiplet_width_mm
    y1 = cfg.height_mm - 5.0 - cfg.chiplet_height_mm
    return ((5.0, 5.0), (x1, 5.0), (5.0, y1), (x1, y1))


def power_density(cfg: Package) -> float:
    volume = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1e-09
    return cfg.chiplet_power_W / volume


def build_package(cfg: Package, run: Run, include_macro: bool, study: Study, h=None):
    model = metahotspot.Model()
    detail_layers = (
        (cfg.substrate_mm, cfg.substrate_cells),
        (cfg.bump_mm, cfg.bump_cells),
        (cfg.die_mm, cfg.die_cells),
    )
    macro_layers = (
        (cfg.tim_mm, cfg.tim_cells),
        (cfg.spreader_mm, cfg.spreader_cells),
        (cfg.cold_plate_mm, cfg.cold_plate_cells),
    )
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=run.duration_s if study == Study.TRANSIENT else 0.0,
        output_interval=run.dt_s if study == Study.TRANSIENT else 0.0,
    )
    layers = (*detail_layers, *macro_layers) if include_macro else detail_layers
    model.set_mesh(
        vertices(cfg.width_mm, cfg.nx),
        vertices(cfg.height_mm, cfg.ny),
        z_vertices(layers),
    )
    add_materials(model)
    if study == Study.TRANSIENT:
        t = run.duration_s
        model.add_function_piecewise(
            "power_scale",
            np.asarray(
                (
                    (0.0, 0.0),
                    (0.12 * t, 1.0),
                    (0.42 * t, 0.65),
                    (0.58 * t, 1.15),
                    (0.82 * t, 0.35),
                    (t, 0.85),
                )
            ),
        )
    source = f"{power_density(cfg):.17g}" + (
        "*power_scale(x)" if study == Study.TRANSIENT else ""
    )
    if include_macro:
        for thickness, material in (
            (cfg.cold_plate_mm, "aluminum"),
            (cfg.spreader_mm, "copper"),
            (cfg.tim_mm, "tim"),
        ):
            layer = model.add_layer(str(thickness))
            full_rect(model, model.add_block(layer, material), cfg)
    die = model.add_layer(str(cfg.die_mm))
    full_rect(model, model.add_block(die, "mold"), cfg)
    for x, y in chiplets(cfg):
        block = model.add_block(die, "silicon", heat_source=source)
        model.add_rect(
            block,
            GeometryOp.ADD,
            str(x),
            str(y),
            str(cfg.chiplet_width_mm),
            str(cfg.chiplet_height_mm),
        )
    bump = model.add_layer(str(cfg.bump_mm))
    full_rect(model, model.add_block(bump, "underfill"), cfg)
    px, py = (cfg.width_mm / cfg.bump_columns, cfg.height_mm / cfg.bump_rows)
    for iy in range(cfg.bump_rows):
        for ix in range(cfg.bump_columns):
            x = (ix + 0.5) * px - 0.5 * cfg.bump_width_mm
            y = (iy + 0.5) * py - 0.5 * cfg.bump_width_mm
            block = model.add_block(bump, "copper")
            model.add_rect(
                block,
                GeometryOp.ADD,
                str(x),
                str(y),
                str(cfg.bump_width_mm),
                str(cfg.bump_width_mm),
            )
    substrate = model.add_layer(str(cfg.substrate_mm))
    full_rect(model, model.add_block(substrate, "organic"), cfg)
    model.set_default_neumann("0")
    if include_macro and h is not None:
        model.add_convection(
            str(h),
            str(cfg.ambient_K),
            [(Axis.Z, cfg.total_height_mm, 0, cfg.width_mm, 0, cfg.height_mm)],
        )
    return model


def build_macro(cfg: Package, h=None):
    model = metahotspot.Model()
    layers = (
        (cfg.tim_mm, cfg.tim_cells),
        (cfg.spreader_mm, cfg.spreader_cells),
        (cfg.cold_plate_mm, cfg.cold_plate_cells),
    )
    model.set_settings(
        study=Study.STEADY,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
    )
    model.set_mesh(
        vertices(cfg.width_mm, cfg.nx),
        vertices(cfg.height_mm, cfg.ny),
        z_vertices(layers),
    )
    add_materials(model)
    for thickness, material in (
        (cfg.cold_plate_mm, "aluminum"),
        (cfg.spreader_mm, "copper"),
        (cfg.tim_mm, "tim"),
    ):
        layer = model.add_layer(str(thickness))
        full_rect(model, model.add_block(layer, material), cfg)
    model.set_default_neumann("0")
    if h is not None:
        model.add_convection(
            str(h),
            str(cfg.ambient_K),
            [(Axis.Z, sum((x[0] for x in layers)), 0, cfg.width_mm, 0, cfg.height_mm)],
        )
    return model


def patches(cfg: Package, face: Face, z: float) -> list[PortPatch]:
    dx, dy = (cfg.width_mm * 0.001 / cfg.nx, cfg.height_mm * 0.001 / cfg.ny)
    return [
        PortPatch(int(face), z, (i * dx, (i + 1) * dx, j * dy, (j + 1) * dy))
        for i in range(cfg.nx)
        for j in range(cfg.ny)
    ]


def macro_sample(cfg: Package, h=None) -> Sample:
    compiled = build_macro(cfg, h).compile()
    port_map = PortMap(compiled, patches(cfg, Face.ZM, 0.0))
    K, C, f = port_map.assemble()
    return Sample(h, compiled, port_map, K.tocsc(), C.tocsc(), np.asarray(f))


def grid_cells(compiled, z0: int, z1: int) -> np.ndarray:
    return np.asarray(
        [
            int(compiled.grid_to_cell[(i * compiled.ny + j) * compiled.nz + k])
            for i in range(compiled.nx)
            for j in range(compiled.ny)
            for k in range(z0, z1)
        ]
    )


def macro_columns(compiled) -> tuple[np.ndarray, ...]:
    columns = []
    for i in range(compiled.nx):
        for j in range(compiled.ny):
            cells = np.asarray(
                [
                    int(compiled.grid_to_cell[(i * compiled.ny + j) * compiled.nz + k])
                    for k in range(compiled.nz)
                ],
                dtype=np.int64,
            )
            if np.any(cells < 0):
                raise RuntimeError(
                    "macro basis requires a complete rectangular macro grid"
                )
            columns.append(cells)
    return tuple(columns)


def assemble(cfg: Package, run: Run) -> Data:
    full_layout = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    z = (cfg.substrate_mm + cfg.bump_mm + cfg.die_mm) * 0.001
    detail_patches = patches(cfg, Face.ZP, z)
    samples = tuple((macro_sample(cfg, h) for h in (None, *run.h_values)))
    return Data(
        full_layout,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, detail_patches),
        PortMap(detail_transient, detail_patches),
        samples,
        grid_cells(full_layout, 0, cfg.detail_nz),
        grid_cells(full_layout, cfg.detail_nz, cfg.nz),
    )


def split(sample: Sample):
    p = sample.ports.port_count
    return (
        sample.K[:p, :p].tocsc(),
        sample.K[:p, p:].tocsc(),
        sample.K[p:, :p].tocsc(),
        sample.K[p:, p:].tocsc(),
        sample.C[p:, p:].tocsc(),
        sample.f[:p],
        sample.f[p:],
    )
