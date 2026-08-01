#!/usr/bin/env python3
"""Transient boundary-condition-independent sparse DtN ROM benchmark.

The macro basis is extracted once from a homogeneous-Neumann macro domain. It
uses only assembled macro operators and the geometric port/column mapping. The
physical interface ports remain exact leading states; localized static and
fixed-interface spectral modes reduce only the macro's internal cells.
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
from metahotspot.compiled import Operators, SolveOptions
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study
from metahotspot.macromodel import PortMap, PortPatch, solve as solve_macro


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
    output = [0.0]
    z = 0.0
    for thickness, cells in layers:
        for _ in range(cells):
            z += thickness / cells
            output.append(z)
    return np.asarray(output)


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
    volume = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1e-9
    return cfg.chiplet_power_W / volume


def build_package(
    cfg: Package,
    run: Run,
    include_macro: bool,
    study: Study,
    h: float | None = None,
):
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
    px = cfg.width_mm / cfg.bump_columns
    py = cfg.height_mm / cfg.bump_rows
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


def build_macro(cfg: Package, h: float | None = None):
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
            [(Axis.Z, sum(x[0] for x in layers), 0, cfg.width_mm, 0, cfg.height_mm)],
        )
    return model


def port_patches(cfg: Package, face: Face, z: float) -> list[PortPatch]:
    dx = cfg.width_mm * 1e-3 / cfg.nx
    dy = cfg.height_mm * 1e-3 / cfg.ny
    return [
        PortPatch(int(face), z, (i * dx, (i + 1) * dx, j * dy, (j + 1) * dy))
        for i in range(cfg.nx)
        for j in range(cfg.ny)
    ]


def macro_sample(cfg: Package, h: float | None = None) -> Sample:
    compiled = build_macro(cfg, h).compile()
    ports = PortMap(compiled, port_patches(cfg, Face.ZM, 0.0))
    operators = ports.assemble()
    return Sample(
        h,
        compiled,
        ports,
        operators.K.tocsc(),
        operators.C.tocsc(),
        np.asarray(operators.f),
    )


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
                raise RuntimeError("macro basis requires a complete rectangular grid")
            columns.append(cells)
    return tuple(columns)


def assemble(cfg: Package, run: Run) -> Data:
    full_layout = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    z = (cfg.substrate_mm + cfg.bump_mm + cfg.die_mm) * 1e-3
    patches = port_patches(cfg, Face.ZP, z)
    samples = tuple(macro_sample(cfg, h) for h in (None, *run.h_values))
    return Data(
        full_layout,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, patches),
        PortMap(detail_transient, patches),
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


def orthonormal_range(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))
    q, r, _ = scipy.linalg.qr(
        matrix, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0))
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, diagonal > tolerance])


def projected_static_residual(
    Kii: sp.csc_matrix,
    Kip: sp.csc_matrix,
    W: sp.csc_matrix,
    block_size: int,
) -> float:
    lu = spla.splu((W.T @ Kii @ W).tocsc())
    reduced_rhs = -(W.T @ Kip).tocsc()
    numerator = 0.0
    denominator = float(np.dot(Kip.data, Kip.data))
    for start in range(0, Kip.shape[1], block_size):
        stop = min(Kip.shape[1], start + block_size)
        coordinates = lu.solve(reduced_rhs[:, start:stop].toarray())
        residual = Kii @ (W @ coordinates) + Kip[:, start:stop].toarray()
        numerator += float(np.linalg.norm(residual, ord="fro") ** 2)
    return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))


def build_basis(sample: Sample, run: Run) -> Basis:
    started = time.perf_counter()
    _, _, Kip, Kii, Cii, _, _ = split(sample)
    columns = macro_columns(sample.compiled)
    if len(columns) != sample.ports.port_count:
        raise RuntimeError("one exact physical port is required per macro column")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    column_orders: list[int] = []
    retained_eigenvalues: list[float] = []
    local_residuals: list[float] = []
    offset = 0
    for port, cells in enumerate(columns):
        local_k = Kii[cells, :][:, cells].toarray()
        local_c = Cii[cells, :][:, cells].toarray()
        local_coupling = Kip[cells, port].toarray().ravel()
        static_shape = -scipy.linalg.solve(
            local_k, local_coupling, assume_a="sym", check_finite=False
        )
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            local_k, local_c, check_finite=False
        )
        eligible = np.flatnonzero(eigenvalues <= run.modal_cutoff_per_s)
        dynamic = eligible[: run.dynamic_modes_per_column]
        local_basis = orthonormal_range(
            np.column_stack(
                (
                    static_shape,
                    np.ones(cells.size, dtype=np.float64),
                    eigenvectors[:, dynamic],
                )
            )
        )
        if local_basis.shape[1] == 0:
            raise RuntimeError("empty local basis")
        coupling_scale = max(np.linalg.norm(local_coupling), np.finfo(float).tiny)
        local_residuals.append(
            float(
                np.linalg.norm(local_k @ static_shape + local_coupling) / coupling_scale
            )
        )
        retained_eigenvalues.extend(eigenvalues[dynamic].tolist())
        column_orders.append(local_basis.shape[1])
        for local_row, cell in enumerate(cells):
            nonzero = np.flatnonzero(np.abs(local_basis[local_row]) > 1e-14)
            rows.extend([int(cell)] * nonzero.size)
            cols.extend((offset + nonzero).tolist())
            values.extend(local_basis[local_row, nonzero].tolist())
        offset += local_basis.shape[1]

    W = sp.csc_matrix((values, (rows, cols)), shape=(Kii.shape[0], offset))
    orthogonality_error = float(
        spla.norm((W.T @ W).tocsc() - sp.eye(W.shape[1], format="csc"))
    )
    return Basis(
        W,
        np.asarray(column_orders, dtype=np.int64),
        np.asarray(retained_eigenvalues, dtype=np.float64),
        max(local_residuals),
        projected_static_residual(Kii, Kip, W, run.residual_block_size),
        orthogonality_error,
        time.perf_counter() - started,
    )


def project(sample: Sample, W: sp.csc_matrix) -> tuple[Operators, float]:
    started = time.perf_counter()
    Kpp, Kpi, Kip, Kii, Cii, fp, fi = split(sample)
    p, r = Kpp.shape[0], W.shape[1]
    Krr = (W.T @ Kii @ W).tocsc()
    Crr = (W.T @ Cii @ W).tocsc()
    Krr = (0.5 * (Krr + Krr.T)).tocsc()
    Crr = (0.5 * (Crr + Crr.T)).tocsc()
    zero_pp = sp.csc_matrix((p, p))
    zero_pr = sp.csc_matrix((p, r))
    K = sp.bmat(
        ((Kpp, (Kpi @ W).tocsc()), ((W.T @ Kip).tocsc(), Krr)),
        format="csc",
    )
    C = sp.bmat(((zero_pp, zero_pr), (zero_pr.T, Crr)), format="csc")
    K.eliminate_zeros()
    C.eliminate_zeros()
    f = np.r_[fp, np.asarray(W.T @ fi).ravel()]
    return Operators(K, C, f), time.perf_counter() - started


def solve_options(run: Run, transient: bool) -> SolveOptions:
    dt = run.dt_s if transient else 1.0
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1e-12,
        linear_max_iterations=5000,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1e-11,
        nonlinear_absolute_tolerance=1e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_abs_tol=1e-9,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def csc_bytes(matrix: sp.csc_matrix) -> int:
    matrix = matrix.tocsc()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def reference(cfg: Package, run: Run, h: float) -> Reference:
    compile_started = time.perf_counter()
    steady_compiled = build_package(cfg, run, True, Study.STEADY, h).compile()
    transient_compiled = build_package(cfg, run, True, Study.TRANSIENT, h).compile()
    compile_s = time.perf_counter() - compile_started
    operators = transient_compiled.assemble()

    started = time.perf_counter()
    with steady_compiled.solve(opts=solve_options(run, False)) as solution:
        steady = np.asarray(solution.temperature).copy()
    steady_s = time.perf_counter() - started

    started = time.perf_counter()
    with transient_compiled.solve(opts=solve_options(run, True)) as solution:
        times = np.asarray(solution.history_times).copy()
        transient = np.asarray(solution.temperature_history).copy()
    transient_s = time.perf_counter() - started
    steady_compiled.close()
    transient_compiled.close()

    return Reference(
        steady,
        times,
        transient,
        compile_s,
        steady_s,
        transient_s,
        operators.K.shape[0],
        operators.K.nnz,
        operators.C.nnz,
        csc_bytes(operators.K) + csc_bytes(operators.C),
    )


def evaluate(
    data: Data,
    cfg: Package,
    run: Run,
    W: sp.csc_matrix,
    h: float,
    ref: Reference,
):
    sample = next(item for item in data.samples if item.h == h)
    operators, projection_s = project(sample, W)
    p = cfg.ports
    internal0 = np.asarray(W.T @ np.full(W.shape[0], cfg.ambient_K)).ravel()

    def run_solve(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count, cfg.ambient_K),
            np.full(p, cfg.ambient_K),
            internal0,
        ]
        started = time.perf_counter()
        with solve_macro(
            compiled,
            operators,
            ports,
            state,
            solve_options(run, transient),
        ) as solution:
            elapsed = time.perf_counter() - started
            if transient:
                return (
                    np.asarray(solution.history_times).copy(),
                    np.asarray(solution.state_history).copy(),
                    elapsed,
                )
            return np.asarray(solution.state).copy(), elapsed

    steady_state, steady_s = run_solve(False)
    times, transient_states, transient_s = run_solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        macro = states[:, detail_n:]
        output = np.empty((states.shape[0], data.full_layout.cell_count))
        output[:, data.detail_cells] = states[:, :detail_n]
        output[:, data.macro_cells] = (W @ macro[:, p:].T).T
        return output

    steady_error = float(np.max(np.abs(recover(steady_state)[0] - ref.steady)))
    if times.shape != ref.times.shape or not np.allclose(
        times, ref.times, atol=1e-12, rtol=0
    ):
        raise RuntimeError("full and reduced solvers returned different output times")
    transient_error = float(np.max(np.abs(recover(transient_states) - ref.transient)))
    reduced_bytes = csc_bytes(operators.K) + csc_bytes(operators.C)
    return {
        "h_W_m2K": h,
        "steady_error_K": steady_error,
        "transient_error_K": transient_error,
        "transient_records": int(times.size),
        "projection_s": projection_s,
        "full_compile_s": ref.compile_s,
        "full_steady_solve_s": ref.steady_solve_s,
        "reduced_steady_solve_s": steady_s,
        "steady_speedup": ref.steady_solve_s / max(steady_s, np.finfo(float).tiny),
        "full_transient_solve_s": ref.transient_solve_s,
        "reduced_transient_solve_s": transient_s,
        "transient_speedup": ref.transient_solve_s
        / max(transient_s, np.finfo(float).tiny),
        "full_operator_order": ref.operator_order,
        "full_operator_k_nnz": ref.operator_k_nnz,
        "full_operator_c_nnz": ref.operator_c_nnz,
        "full_operator_bytes": ref.operator_bytes,
        "reduced_macro_order": int(operators.K.shape[0]),
        "reduced_online_order": int(detail_n + operators.K.shape[0]),
        "reduced_macro_k_nnz": int(operators.K.nnz),
        "reduced_macro_c_nnz": int(operators.C.nnz),
        "reduced_macro_operator_bytes": reduced_bytes,
    }


def configs(quick: bool):
    if quick:
        return (
            Package(
                nx=12,
                ny=12,
                substrate_cells=4,
                bump_cells=2,
                die_cells=3,
                tim_cells=2,
                spreader_cells=4,
                cold_plate_cells=5,
                bump_rows=6,
                bump_columns=6,
            ),
            Run(duration_s=0.2, dt_s=0.025, speedup_target=1.0),
        )
    return Package(), Run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run = configs(args.quick)

    print("=" * 108)
    print("Transient BCI-ROM benchmark - sparse localized spectral DtN reduction")
    print("=" * 108)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.nz} = "
        f"{cfg.nx * cfg.ny * cfg.nz:,} full cells; detail z={cfg.detail_nz}, "
        f"macro z={cfg.macro_nz}, exact ports={cfg.ports}"
    )

    started = time.perf_counter()
    data = assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    training_sample = next(item for item in data.samples if item.h is None)
    basis = build_basis(training_sample, run)
    basis_density = basis.W.nnz / max(1, basis.W.shape[0] * basis.W.shape[1])
    print(
        f"Basis: internal {basis.W.shape[0]:,} -> {basis.W.shape[1]:,}; "
        f"column order min/mean/max={basis.column_orders.min()}/"
        f"{basis.column_orders.mean():.2f}/{basis.column_orders.max()}; "
        f"nnz={basis.W.nnz:,}, density={basis_density:.3e}"
    )
    print(
        f"Extraction: {basis.seconds:.3f}s; local static residual="
        f"{basis.local_static_residual:.3e}; projected static transfer residual="
        f"{basis.projected_static_residual:.3e}; orthogonality="
        f"{basis.orthogonality_error:.3e}"
    )

    boundary = []
    nominal_ref = reference(cfg, run, run.nominal_h)
    for h in run.h_values:
        ref = nominal_ref if h == run.nominal_h else reference(cfg, run, h)
        result = evaluate(data, cfg, run, basis.W, h, ref)
        accuracy_passed = (
            max(result["steady_error_K"], result["transient_error_K"]) <= run.error_K
        )
        speed_passed = (
            result["transient_speedup"] >= run.speedup_target if args.strict else True
        )
        result["accuracy_passed"] = accuracy_passed
        result["speedup_passed"] = speed_passed
        result["passed"] = accuracy_passed and speed_passed
        online_savings = (
            result["full_transient_solve_s"] - result["reduced_transient_solve_s"]
        )
        result["rom_offline_s"] = assembly_s + basis.seconds + result["projection_s"]
        result["offline_break_even_transient_runs"] = (
            result["rom_offline_s"] / online_savings if online_savings > 0.0 else None
        )
        boundary.append(result)
        print(
            f"h={h:7.1f}: error steady/transient="
            f"{result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K; "
            f"transient full/ROM={result['full_transient_solve_s']:.3f}/"
            f"{result['reduced_transient_solve_s']:.3f}s, "
            f"speedup={result['transient_speedup']:.2f}x "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    nominal = next(item for item in boundary if item["h_W_m2K"] == run.nominal_h)
    detail_operators = data.detail_transient.assemble()
    coupled_k_nnz_upper = (
        detail_operators.K.nnz + nominal["reduced_macro_k_nnz"] + 4 * cfg.ports
    )
    coupled_c_nnz_upper = detail_operators.C.nnz + nominal["reduced_macro_c_nnz"]
    coupled_bytes_estimate = (
        csc_bytes(detail_operators.K)
        + csc_bytes(detail_operators.C)
        + nominal["reduced_macro_operator_bytes"]
        + 4 * cfg.ports * (8 + 4)
    )
    print(
        f"Online order: {nominal['full_operator_order']:,} -> "
        f"{nominal['reduced_online_order']:,} "
        f"({nominal['reduced_online_order'] / nominal['full_operator_order']:.3f}x); "
        f"K nnz full/ROM-upper={nominal['full_operator_k_nnz']:,}/"
        f"{coupled_k_nnz_upper:,}"
    )
    print(
        f"Operator memory full/ROM-estimate="
        f"{nominal['full_operator_bytes'] / 2**20:.2f}/"
        f"{coupled_bytes_estimate / 2**20:.2f} MiB; nominal break-even="
        f"{nominal['offline_break_even_transient_runs']} transient runs"
    )

    report = {
        "schema_version": 9,
        "mode": "quick" if args.quick else "strict",
        "reduction_method": "localized_static_constraint_fixed_interface_spectral",
        "training_boundary": "homogeneous_neumann",
        "input_training": "none",
        "port_coordinates": "exact_leading_states",
        "package": asdict(cfg),
        "experiment": {
            **asdict(run),
            "report": str(run.report),
            "modal_cutoff_per_s": run.modal_cutoff_per_s,
        },
        "full_cell_count": cfg.nx * cfg.ny * cfg.nz,
        "detail_cell_count": data.detail_steady.cell_count,
        "macro_full_internal_order": basis.W.shape[0],
        "physical_port_count": cfg.ports,
        "reduced_internal_order": basis.W.shape[1],
        "reduced_macro_total_order": cfg.ports + basis.W.shape[1],
        "basis_nnz": basis.W.nnz,
        "basis_density": basis_density,
        "local_column_order": {
            "minimum": int(basis.column_orders.min()),
            "mean": float(basis.column_orders.mean()),
            "maximum": int(basis.column_orders.max()),
        },
        "retained_eigenvalue_range_per_s": (
            [
                float(basis.retained_eigenvalues_per_s.min()),
                float(basis.retained_eigenvalues_per_s.max()),
            ]
            if basis.retained_eigenvalues_per_s.size
            else []
        ),
        "local_static_constraint_residual": basis.local_static_residual,
        "projected_static_transfer_residual": basis.projected_static_residual,
        "basis_orthogonality_error": basis.orthogonality_error,
        "model_assembly_s": assembly_s,
        "basis_extraction_s": basis.seconds,
        "nominal_online_structure": {
            "reduced_coupled_k_nnz_upper_bound": int(coupled_k_nnz_upper),
            "reduced_coupled_c_nnz_upper_bound": int(coupled_c_nnz_upper),
            "reduced_coupled_operator_bytes_estimate": int(coupled_bytes_estimate),
        },
        "boundary_reuse": boundary,
        "passed": bool(all(item["passed"] for item in boundary)),
    }
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {run.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
