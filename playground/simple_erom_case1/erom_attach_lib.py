#!/usr/bin/env python3
"""Shared machinery for the "EROM attached to a real external FVM body" experiments.

Single public entry point: :func:`run_attached(cfg, outdir)` solves the coupled
ROM(EROM)-FVM system and returns a report dict, exporting VTU field files and a
probe-temperature trajectory.  The stacked reference geometry
(:class:`AttachConfig`, :class:`AttachModel`) and the small VTU/CSV writers live
next to it here so the per-case experiment scripts stay thin.

Geometry (one stacked model so the monolithic detailed reference is exact):

    z  [0, H_m]  external body   (material / footprint / bottom HTC / internal source)
    z [H_m, ...] copper cube     (centre 100 W source, top +Z HTC face)

The cube cells are reduced with ``extract_rom`` into a FloTHERM-style
``EmbeddableRom`` (top +Z HTC = affine ambient group; -Z face = connectable
interface port).  The external cells stay full FVM (``build_subdomain``); the
two sides are joined through independent interface nodes by ``connect``.  The
monolithic reference is the same model solved unsplit with the same physical
HTC, so identity coupling is exact by construction.  Interface accuracy is
reported as separate observables (junction rise, interface-face trace, EROM
interior, external field, interface heat-flux balance).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import metahotspot
from metahotspot.compiled import CellFields
from metahotspot.enums import Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel.affine import (
    AffineParametricModel,
    BoundaryGroup,
    SourcePort,
    surface_exposed_cells,
)
from metahotspot.macromodel.embeddable import (
    build_subdomain,
    common_patches,
    connect,
    extract_rom,
    interface_trace,
    side_junction_rise,
    solve_system,
)

# copper cube (matches simple_case1.ecxml)
COPPER = ("385.0", "385.0", "385.0", "8930.0", "385.0")
CUBE_MM = 100.0  # cube side (mm)
TOP_HTC = 1000.0  # physical HTC on the cube +Z face (W/m2.K)
CUBE_SOURCE_W = 100.0  # cube centre volumetric source (W)
SOURCE_VOL_M3 = 0.05**3  # centre 50x50x50 mm source volume (m^3)

OUT = Path(__file__).resolve().parent / "results"


def cube_axis_mm() -> np.ndarray:
    """Fixed 15-cell FloTHERM grid used by the current EROM demonstration (mm)."""
    return np.r_[
        np.linspace(0.0, 25.0, 5),
        np.linspace(25.0, 75.0, 8)[1:],
        np.linspace(75.0, 100.0, 5)[1:],
    ]


@dataclass(frozen=True)
class AttachConfig:
    ambient_K: float = 308.15
    duration_s: float = 200.0  # thermal capture window (s)
    dt_s: float = 4.0
    # --- external body (a second BCI-ROM-like 100x100x100 cube below) ---
    ext_thickness_mm: float = 100.0  # external cube side (mm); = cube side
    ext_k: float = 385.0  # external material conductivity (W/m.K)
    ext_rho: float = 8930.0
    ext_c: float = 385.0
    ext_bottom_h: float | None = None  # external bottom HTC (None = adiabatic)
    ext_source_w: float = 0.0  # external centre source power (W; 0 = none)
    ext_vox_mm: float = 10.0  # external z mesh target cell size (mm)

    @property
    def n_ext(self) -> int:
        return max(2, math.ceil(self.ext_thickness_mm / self.ext_vox_mm))

    @property
    def total_height_mm(self) -> float:
        return self.ext_thickness_mm + CUBE_MM

    def report_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# geometry
# -----------------------------------------------------------------------------


def _z_full_mm(cfg: AttachConfig) -> np.ndarray:
    ext = np.linspace(0.0, cfg.ext_thickness_mm, cfg.n_ext + 1)
    cube = cfg.ext_thickness_mm + cube_axis_mm()
    return np.unique(np.r_[ext, cube])


def _add_solid(model, layer, material, xlo, xhi, ylo, yhi):
    block = model.add_block(layer, material)
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{xlo:.17g}",
        f"{ylo:.17g}",
        f"{xhi - xlo:.17g}",
        f"{yhi - ylo:.17g}",
    )
    return block


def build_geometry(cfg: AttachConfig, study: Study, *, detail: bool, macro: bool):
    """Assemble the full stacked model (no BC; default Neumann)."""
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
    model.set_mesh(cube_axis_mm(), cube_axis_mm(), _z_full_mm(cfg))
    model.add_material("Copper (Pure)", *COPPER)
    model.add_material(
        "External",
        f"{cfg.ext_k:.17g}",
        f"{cfg.ext_k:.17g}",
        f"{cfg.ext_k:.17g}",
        f"{cfg.ext_rho:.17g}",
        f"{cfg.ext_c:.17g}",
    )

    H = cfg.ext_thickness_mm  # external cube height (mm); ROM cube sits above it
    cube_src_density = CUBE_SOURCE_W / SOURCE_VOL_M3
    ext_src_density = (
        (cfg.ext_source_w / SOURCE_VOL_M3) if cfg.ext_source_w > 0 else 0.0
    )

    z_full = _z_full_mm(cfg)
    for i in range(z_full.size - 2, -1, -1):  # add top layer first (top-down stacking)
        z = 0.5 * (z_full[i] + z_full[i + 1])
        layer = model.add_layer(f"{z_full[i + 1] - z_full[i]:.17g}")
        if z >= H:  # upper cube = the EROM side
            _add_solid(model, layer, "Copper (Pure)", 0.0, CUBE_MM, 0.0, CUBE_MM)
            if (H + 25.0) < z < (H + 75.0):
                block = model.add_block(
                    layer, "Copper (Pure)", heat_source=f"{cube_src_density:.17g}"
                )
                model.add_rect(block, GeometryOp.ADD, "25", "25", "50", "50")
        else:  # lower cube = the external body (full 100x100x100, centre source)
            _add_solid(model, layer, "External", 0.0, CUBE_MM, 0.0, CUBE_MM)
            if cfg.ext_source_w > 0 and (H / 2.0 - 25.0) < z < (H / 2.0 + 25.0):
                block = model.add_block(
                    layer, "External", heat_source=f"{ext_src_density:.17g}"
                )
                model.add_rect(block, GeometryOp.ADD, "25", "25", "50", "50")
    model.set_default_neumann("0")
    return model


class AttachModel(AffineParametricModel):
    """Affine parametric model of the cube + external attachment."""

    def __init__(self, cfg: AttachConfig | None = None):
        self.config = cfg or AttachConfig()

    @property
    def name(self) -> str:
        return "erom_cube_attach"

    def build_geometry(self, study, *, detail, macro):
        return build_geometry(self.config, study, detail=detail, macro=macro)

    def source_ports(self) -> list[SourcePort]:
        f = np.asarray(self._core.f, dtype=np.float64)
        zc = self.cell_layout.centers[:, 2]
        H = self.config.ext_thickness_mm * 1.0e-3
        src = np.flatnonzero(f > 0.0)
        ports = [SourcePort(cells=src[zc[src] >= H - 1.0e-12], power_W=CUBE_SOURCE_W)]
        if self.config.ext_source_w > 0:
            ports.append(
                SourcePort(
                    cells=src[zc[src] < H - 1.0e-12], power_W=self.config.ext_source_w
                )
            )
        return ports

    def boundary_groups(self) -> tuple[BoundaryGroup, ...]:
        cells = self._full.cells
        top_cells, top_areas = surface_exposed_cells(
            cells, Face.ZP, cells.z_vertices[-1]
        )
        groups = [BoundaryGroup(cells=top_cells, areas=top_areas, h_range=(1.0, 1.0e4))]
        if self.config.ext_bottom_h is not None:
            bot_cells, bot_areas = surface_exposed_cells(
                cells, Face.ZM, cells.z_vertices[0]
            )
            groups.append(
                BoundaryGroup(cells=bot_cells, areas=bot_areas, h_range=(1.0, 1.0e4))
            )
        return tuple(groups)

    def boundary_h(self, h_vec) -> dict[str, float]:
        out = {"top": float(h_vec[0])}
        if self.config.ext_bottom_h is not None:
            out["bottom"] = float(h_vec[1])
        return out

    def group_h_ranges(self):
        return tuple(g.h_range for g in self.boundary_groups())


# -----------------------------------------------------------------------------
# IO (small VTU + CSV writers)
# -----------------------------------------------------------------------------


def _grid_vertices(cells: CellFields):
    """(V,3) vertex coordinates from the structured cell layout (SI metres)."""
    pts = np.empty(
        (cells.x_vertices.size * cells.y_vertices.size * cells.z_vertices.size, 3)
    )
    k = 0
    for iz in range(cells.z_vertices.size):
        for iy in range(cells.y_vertices.size):
            for ix in range(cells.x_vertices.size):
                pts[k] = (
                    cells.x_vertices[ix],
                    cells.y_vertices[iy],
                    cells.z_vertices[iz],
                )
                k += 1
    return pts


def write_vtu(
    path, cells: CellFields, cell_field: np.ndarray, name: str = "Temperature_K"
) -> str:
    """Write a cell-centred scalar field on the structured grid to ``path`` (SI)."""
    NX, NY, NZ = cells.nx, cells.ny, cells.nz

    def vid(ix, iy, iz):
        return iz * (NY * NX) + iy * NX + ix

    ijk = cells.ijk
    n_cells = ijk.shape[0]
    conn = np.empty((n_cells, 8), dtype=np.int64)
    for c in range(n_cells):
        ix, iy, iz = int(ijk[c, 0]), int(ijk[c, 1]), int(ijk[c, 2])
        conn[c] = [
            vid(ix, iy, iz),
            vid(ix + 1, iy, iz),
            vid(ix + 1, iy + 1, iz),
            vid(ix, iy + 1, iz),
            vid(ix, iy, iz + 1),
            vid(ix + 1, iy, iz + 1),
            vid(ix + 1, iy + 1, iz + 1),
            vid(ix, iy + 1, iz + 1),
        ]

    cell_offsets = 8 * np.arange(1, n_cells + 1, dtype=np.int64)
    cell_types = np.full(n_cells, 12, dtype=np.uint8)  # VTK_HEXAHEDRON

    def fmt(a, prec=12):
        return " ".join(f"{v:.{prec}e}" for v in a)

    def fmt_int(a):
        return " ".join(str(int(v)) for v in a)

    pts_arr = _grid_vertices(cells)
    pts_str = " ".join(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in pts_arr)
    cells_block = "".join(
        f"{n:d} " + " ".join(str(v) for v in conn[i]) + "\n"
        for i, n in enumerate(cell_offsets)
    )
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "  <UnstructuredGrid>\n"
        f'    <Piece NumberOfPoints="{pts_arr.shape[0]}" NumberOfCells="{n_cells}">\n'
        "      <Points>\n"
        f'        <DataArray type="Float64" NumberOfComponents="3" format="ascii">{pts_str}</DataArray>\n'
        "      </Points>\n"
        "      <Cells>\n"
        f'        <DataArray type="Int64" Name="connectivity" format="ascii">\n{cells_block}        </DataArray>\n'
        f'        <DataArray type="Int64" Name="offsets" format="ascii">{fmt_int(cell_offsets)}</DataArray>\n'
        f'        <DataArray type="UInt8" Name="types" format="ascii">{fmt_int(cell_types)}</DataArray>\n'
        "      </Cells>\n"
        "      <CellData>\n"
        f'        <DataArray type="Float64" Name="{name}" format="ascii">{fmt(np.asarray(cell_field, dtype=np.float64))}</DataArray>\n'
        "      </CellData>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n"
    )
    Path(path).write_text(xml, encoding="utf-8")
    return str(Path(path).resolve())


def write_trajectory_csv(path, header: list[str], rows) -> str:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.10g}" for v in r])
    return str(Path(path).resolve())


# -----------------------------------------------------------------------------
# coupled solve
# -----------------------------------------------------------------------------


def _probe_cells(cells: CellFields, points_m) -> list[int]:
    c = cells.centers
    return [
        int(np.argmin((c[:, 0] - px) ** 2 + (c[:, 1] - py) ** 2 + (c[:, 2] - pz) ** 2))
        for (px, py, pz) in points_m
    ]


def _maxerr(a, b) -> float:
    return float(np.max(np.abs(a - b)))


def run_attached(cfg: AttachConfig, outdir: Path) -> dict:
    """Solve the coupled ROM-FVM system, export VTU + trajectory, return report."""
    outdir.mkdir(parents=True, exist_ok=True)
    model = AttachModel(cfg)
    cells = model._full.cells
    zc = cells.centers[:, 2]
    H_m = cfg.ext_thickness_mm * 1.0e-3

    cube_idx = np.flatnonzero(zc >= H_m - 1.0e-12)
    ext_idx = np.flatnonzero(zc < H_m - 1.0e-12)
    hvec = [TOP_HTC]
    if cfg.ext_bottom_h is not None:
        hvec.append(cfg.ext_bottom_h)

    erom_sub = build_subdomain(model, cube_idx, name="erom", physical_h=hvec)
    ext_sub = build_subdomain(model, ext_idx, name="external", physical_h=hvec)

    def trim_zero_sources(sub):
        src = np.asarray(sub.source, dtype=np.float64)
        nz = np.flatnonzero(np.abs(src).sum(axis=0) > 0.0)
        if nz.size and nz.size < src.shape[1]:
            sub.source = src[:, nz]
        return sub

    # each side exposes exactly one source column (the other source is zero there
    # and would break the modal extraction / coupling scalar-power contract).
    trim_zero_sources(erom_sub)
    trim_zero_sources(ext_sub)

    erom = extract_rom(erom_sub, tolerance=1.0e-3)
    m = erom.m
    erom.F_hat = np.asarray(erom.F_hat, dtype=np.float64) * CUBE_SOURCE_W
    if cfg.ext_source_w > 0:
        ext_sub.source = np.asarray(ext_sub.source, dtype=np.float64).copy()
        ext_sub.source[:, 0] *= cfg.ext_source_w

    K, C, rhs, _ldof, _rdof, n_interface = connect(
        erom, ext_sub, erom.port("z-"), ext_sub.port("z+")
    )
    ext_offset = m + n_interface
    symmetry = float(np.max(np.abs((K - K.T))))
    eigmin = float(np.min(np.linalg.eigvalsh(K.toarray())))

    steady, history = solve_system(K, C, rhs, dt=cfg.dt_s, duration=cfg.duration_s)

    ref = model.full_reference(hvec)  # monolithic detailed reference
    ref_rise = np.asarray(ref.steady_temperature - cfg.ambient_K)

    coup_rise = np.zeros(model.full_cell_count, dtype=np.float64)
    coup_rise[erom.cells] = erom.basis @ np.asarray(steady[:m], dtype=np.float64)
    coup_rise[ext_sub.cells] = np.asarray(steady[ext_offset:], dtype=np.float64)
    coupled_field = cfg.ambient_K + coup_rise

    probe_labels = ["junction", "cube_top", "interface", "ext_src", "ext_bottom"]
    probe_cells = _probe_cells(
        cells,
        [
            (0.05, 0.05, H_m + 0.05),  # cube source centre (junction)
            (0.05, 0.05, H_m + 0.095),  # cube top centre
            (0.05, 0.05, H_m),  # interface centre
            (0.05, 0.05, H_m / 2.0),  # external source centre
            (0.05, 0.05, 0.0),  # external bottom centre
        ],
    )
    nprobe = len(probe_labels)

    n_steps = history.shape[0]
    step_times = np.arange(n_steps) * cfg.dt_s
    coup_traj = np.empty((n_steps, nprobe), dtype=np.float64)
    basis, erom_cells, ext_sub_cells = erom.basis, erom.cells, ext_sub.cells
    for t in range(n_steps):
        rise = np.zeros(model.full_cell_count, dtype=np.float64)
        rise[erom_cells] = basis @ np.asarray(history[t, :m], dtype=np.float64)
        rise[ext_sub_cells] = np.asarray(history[t, ext_offset:], dtype=np.float64)
        coup_traj[t] = [cfg.ambient_K + rise[c] for c in probe_cells]
    ref_traj = np.stack(
        [np.interp(step_times, ref.times, ref.history[:, c]) for c in probe_cells],
        axis=1,
    )

    # --- exports --------------------------------------------------------------
    vtu_coupled = write_vtu(outdir / "coupled_field.vtu", cells, coupled_field)
    vtu_reference = write_vtu(
        outdir / "reference_field.vtu", cells, ref.steady_temperature
    )
    header = (
        ["t_s"]
        + [f"{lb}_coupled_K" for lb in probe_labels]
        + [f"{lb}_ref_K" for lb in probe_labels]
    )
    rows = [
        [
            float(step_times[t]),
            *[float(coup_traj[t, i]) for i in range(nprobe)],
            *[float(ref_traj[t, i]) for i in range(nprobe)],
        ]
        for t in range(n_steps)
    ]
    traj_csv = write_trajectory_csv(outdir / "probe_trajectories.csv", header, rows)

    # --- error observables (steady, rise coordinates, separate) ---------------
    erom_trace = erom.boundary_trace("z-") @ np.asarray(steady[:m], dtype=np.float64)
    cube_bot_full = erom_sub.cells[np.asarray(erom.port("z-").cells, dtype=np.int64)]

    G = model.source_shape()
    cube_src_rows = np.flatnonzero((G[:, 0] > 0) & (zc >= H_m - 1.0e-12))
    w = G[cube_src_rows, 0] / G[cube_src_rows, 0].sum()
    ref_junction = float(w @ ref_rise[cube_src_rows])
    coup_junction = float(
        np.asarray(side_junction_rise(steady, erom, 0)).ravel()[0] / CUBE_SOURCE_W
    )

    top_cells, top_areas = surface_exposed_cells(cells, Face.ZP, cells.z_vertices[-1])
    kz = model.cell_layout.conductivity[top_cells, 2]
    half = cells.half_sizes[top_cells, 2]
    p = kz * TOP_HTC / (kz + TOP_HTC * half)
    top_flux_ref = float(np.sum(p * top_areas * ref_rise[top_cells]))
    top_flux_coupled = float(np.sum(p * top_areas * coup_rise[top_cells]))
    injected = float(
        np.sum(np.asarray(model._core.f).ravel())
    )  # actual RHS source power

    _areas, E_l, E_r, xi_l, xi_r, _li, _ri = common_patches(
        erom.port("z-"), ext_sub.port("z+")
    )
    Vl, hl = interface_trace(erom, erom.port("z-"), E_l, xi_l)
    Vr, hr = interface_trace(ext_sub, ext_sub.port("z+"), E_r, xi_r)
    q_rom = np.asarray(steady[:m], dtype=np.float64)
    q_ext = np.asarray(steady[ext_offset:], dtype=np.float64)
    T_if = np.asarray(steady[m:ext_offset], dtype=np.float64)
    flux_erom = float(np.sum(hl * (np.asarray(Vl @ q_rom) - T_if)))
    flux_ext = float(np.sum(hr * (np.asarray(Vr @ q_ext) - T_if)))

    traj_maxerr = [_maxerr(coup_traj[:, i], ref_traj[:, i]) for i in range(nprobe)]

    metrics = {
        "rom_order": m,
        "n_interface_nodes": int(n_interface),
        "detailed_cells": int(model.full_cell_count),
        "matrix_symmetry": symmetry,
        "matrix_PD_min_eig": eigmin,
        "injected_W": injected,
        "top_flux_reference_W": top_flux_ref,
        "top_flux_coupled_W": top_flux_coupled,
        "interface_flux_erom_W": flux_erom,
        "interface_flux_external_W": flux_ext,
        "interface_flux_balance_W": flux_erom + flux_ext,
        "junction_coupled_K": coup_junction,
        "junction_reference_K": ref_junction,
        "junction_error_K": coup_junction - ref_junction,
        "junction_error_pct": 100.0 * (coup_junction - ref_junction) / ref_junction,
        "interface_trace_maxerr_K": _maxerr(erom_trace, ref_rise[cube_bot_full]),
        "erom_field_maxerr_K": _maxerr(coup_rise[erom.cells], ref_rise[erom.cells]),
        "external_field_maxerr_K": _maxerr(
            coup_rise[ext_sub.cells], ref_rise[ext_sub.cells]
        ),
        "global_field_maxerr_K": _maxerr(coup_rise, ref_rise),
        "junction_traj_maxerr_K": float(traj_maxerr[0]),
        "probe_traj_maxerr_K": traj_maxerr,
    }

    report = {
        "cfg": cfg.report_dict(),
        "metrics": metrics,
        "probes": {
            lb: {
                "ref_K": float(ref.steady_temperature[c]),
                "coupled_K": float(coupled_field[c]),
            }
            for lb, c in zip(probe_labels, probe_cells)
        },
        "artifacts": {
            "vtu_coupled": str(vtu_coupled),
            "vtu_reference": str(vtu_reference),
            "trajectory_csv": str(traj_csv),
        },
    }
    (outdir / "report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    return report


# library registry convenience (idempotent; harmless on import)
try:
    from metahotspot.macromodel.affine import register as _register

    def _builder(overrides: dict | None = None, **_kw):
        return AttachModel(AttachConfig(**(overrides or {})))

    _register("erom_cube_attach", _builder)
except Exception:  # pragma: no cover
    pass
