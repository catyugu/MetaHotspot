#!/usr/bin/env python3
"""Transient boundary-condition-independent affine DtN ROM benchmark.

The macro basis is extracted from a homogeneous-Neumann domain. One additional
full-order assembly at ``affine_anchor_h`` identifies the convection increment

    A(h) = A(0) + (h / affine_anchor_h) [A(anchor) - A(0)].

Both components are projected once. Every queried ``h`` is then assembled only
in reduced coordinates; the full-order macro is never rebuilt online. Steady
and transient full-order solutions are retained as complementary ROM checks.
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
    def macro_nz(self) -> int:
        return sum(cells for _, cells in self.macro_layers)

    @property
    def nz(self) -> int:
        return self.detail_nz + self.macro_nz

    @property
    def ports(self) -> int:
        return self.nx * self.ny

    @property
    def total_height_mm(self) -> float:
        return sum(t for t, _ in (*self.detail_layers, *self.macro_layers))


@dataclass(frozen=True)
class Run:
    error_K: float = 0.5
    duration_s: float = 0.5
    dt_s: float = 0.025
    h_values: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    affine_anchor_h: float = 2500.0
    dynamic_modes_per_column: int = 2
    speedup_target: float = 1.5
    residual_block_size: int = 32
    report: Path = Path("results/bci_rom_sparse_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s


class MacroAffine(NamedTuple):
    compiled: object
    ports: PortMap
    anchor_h: float
    base: Operators
    delta: Operators
    c_relative_change: float
    ambient_residual: float
    seconds: float

    def at(self, h: float) -> Operators:
        return combine(self.base, self.delta, h / self.anchor_h)


class Data(NamedTuple):
    full_layout: object
    detail_steady: object
    detail_transient: object
    detail_ports_steady: PortMap
    detail_ports_transient: PortMap
    macro: MacroAffine
    detail_cells: np.ndarray
    macro_cells: np.ndarray


class Basis(NamedTuple):
    W: sp.csc_matrix
    column_orders: np.ndarray
    eigenvalues_per_s: np.ndarray
    local_static_residual: float
    local_parameter_residual: float
    projected_residual_base: float
    projected_residual_anchor: float
    orthogonality_error: float
    seconds: float


@dataclass(frozen=True)
class ReducedAffine:
    anchor_h: float
    base: Operators
    delta: Operators
    seconds: float

    def at(self, h: float) -> tuple[Operators, float]:
        started = time.perf_counter()
        operators = combine(self.base, self.delta, h / self.anchor_h)
        return operators, time.perf_counter() - started


class Reference(NamedTuple):
    steady: np.ndarray
    times: np.ndarray
    transient: np.ndarray
    compile_s: float
    steady_solve_s: float
    transient_solve_s: float
    order: int
    k_nnz: int
    c_nnz: int
    bytes: int


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


def configure(model, cfg: Package, layers, study: Study, run: Run | None) -> None:
    transient = study == Study.TRANSIENT
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=run.duration_s if transient else 0.0,
        output_interval=run.dt_s if transient else 0.0,
    )
    model.set_mesh(
        vertices(cfg.width_mm, cfg.nx),
        vertices(cfg.height_mm, cfg.ny),
        z_vertices(layers),
    )
    add_materials(model)


def add_macro(model, cfg: Package) -> None:
    for thickness, material in (
        (cfg.cold_plate_mm, "aluminum"),
        (cfg.spreader_mm, "copper"),
        (cfg.tim_mm, "tim"),
    ):
        layer = model.add_layer(str(thickness))
        full_rect(model, model.add_block(layer, material), cfg)


def add_detail(model, cfg: Package, source: str) -> None:
    die = model.add_layer(str(cfg.die_mm))
    full_rect(model, model.add_block(die, "mold"), cfg)
    x1 = cfg.width_mm - 5.0 - cfg.chiplet_width_mm
    y1 = cfg.height_mm - 5.0 - cfg.chiplet_height_mm
    for x, y in ((5.0, 5.0), (x1, 5.0), (5.0, y1), (x1, y1)):
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
    px, py = cfg.width_mm / cfg.bump_columns, cfg.height_mm / cfg.bump_rows
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


def build_package(
    cfg: Package,
    run: Run,
    include_macro: bool,
    study: Study,
    h: float | None = None,
):
    model = metahotspot.Model()
    layers = (
        (*cfg.detail_layers, *cfg.macro_layers) if include_macro else cfg.detail_layers
    )
    configure(model, cfg, layers, study, run)
    volume = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1e-9
    source = f"{cfg.chiplet_power_W / volume:.17g}"
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
        source += "*power_scale(x)"
    if include_macro:
        add_macro(model, cfg)
    add_detail(model, cfg, source)
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
    configure(model, cfg, cfg.macro_layers, Study.STEADY, None)
    add_macro(model, cfg)
    model.set_default_neumann("0")
    if h is not None:
        height = sum(t for t, _ in cfg.macro_layers)
        model.add_convection(
            str(h),
            str(cfg.ambient_K),
            [(Axis.Z, height, 0, cfg.width_mm, 0, cfg.height_mm)],
        )
    return model


def port_patches(cfg: Package, face: Face, z: float) -> list[PortPatch]:
    dx, dy = cfg.width_mm * 1e-3 / cfg.nx, cfg.height_mm * 1e-3 / cfg.ny
    return [
        PortPatch(int(face), z, (i * dx, (i + 1) * dx, j * dy, (j + 1) * dy))
        for i in range(cfg.nx)
        for j in range(cfg.ny)
    ]


def normalized(operators: Operators) -> Operators:
    K, C = sp.csc_matrix(operators.K), sp.csc_matrix(operators.C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(operators.f, dtype=np.float64).copy())


def subtract(a: Operators, b: Operators) -> Operators:
    K, C = (a.K - b.K).tocsc(), (a.C - b.C).tocsc()
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(a.f) - np.asarray(b.f))


def combine(base: Operators, delta: Operators, theta: float) -> Operators:
    K, C = (base.K + theta * delta.K).tocsc(), (base.C + theta * delta.C).tocsc()
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(base.f) + theta * np.asarray(delta.f))


def relative_norm(matrix, reference) -> float:
    denominator = max(float(spla.norm(reference)), np.finfo(float).tiny)
    return float(spla.norm(matrix)) / denominator


def extract_affine_macro(cfg: Package, anchor_h: float) -> MacroAffine:
    if anchor_h <= 0.0:
        raise ValueError("affine_anchor_h must be positive")
    started = time.perf_counter()
    patches = port_patches(cfg, Face.ZM, 0.0)
    compiled = build_macro(cfg).compile()
    ports = PortMap(compiled, patches)
    base = normalized(ports.assemble())

    anchor_compiled = build_macro(cfg, anchor_h).compile()
    anchor_ports = PortMap(anchor_compiled, patches)
    try:
        anchor = normalized(anchor_ports.assemble())
        if anchor.K.shape != base.K.shape:
            raise RuntimeError("convection changed the macro state ordering")
        delta = subtract(anchor, base)
    finally:
        anchor_ports.close()
        anchor_compiled.close()

    ambient = np.full(base.K.shape[0], cfg.ambient_K)
    defect = np.asarray(delta.K @ ambient).ravel() - delta.f
    ambient_scale = max(np.linalg.norm(delta.f), np.finfo(float).tiny)
    return MacroAffine(
        compiled,
        ports,
        anchor_h,
        base,
        delta,
        relative_norm(delta.C, base.C),
        float(np.linalg.norm(defect) / ambient_scale),
        time.perf_counter() - started,
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
                raise RuntimeError("macro basis requires a rectangular grid")
            columns.append(cells)
    return tuple(columns)


def assemble(cfg: Package, run: Run) -> Data:
    full_layout = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    z = sum(t for t, _ in cfg.detail_layers) * 1e-3
    patches = port_patches(cfg, Face.ZP, z)
    return Data(
        full_layout,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, patches),
        PortMap(detail_transient, patches),
        extract_affine_macro(cfg, run.affine_anchor_h),
        grid_cells(full_layout, 0, cfg.detail_nz),
        grid_cells(full_layout, cfg.detail_nz, cfg.nz),
    )


def internal_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, :ports].tocsc(),
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
    )


def orthonormal_range(matrix: np.ndarray) -> np.ndarray:
    q, r, _ = scipy.linalg.qr(
        np.asarray(matrix), mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0))
    tolerance = np.finfo(float).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, diagonal > tolerance])


def projected_residual(
    operators: Operators, ports: int, W: sp.csc_matrix, block_size: int
) -> float:
    Kip, Kii, _ = internal_blocks(operators, ports)
    lu = spla.splu((W.T @ Kii @ W).tocsc())
    rhs = -(W.T @ Kip).tocsc()
    numerator = 0.0
    denominator = float(np.dot(Kip.data, Kip.data))
    for start in range(0, ports, block_size):
        stop = min(ports, start + block_size)
        q = lu.solve(rhs[:, start:stop].toarray())
        residual = Kii @ (W @ q) + Kip[:, start:stop].toarray()
        numerator += float(np.linalg.norm(residual, ord="fro") ** 2)
    return math.sqrt(numerator / max(denominator, np.finfo(float).tiny))


def build_basis(macro: MacroAffine, run: Run) -> Basis:
    started = time.perf_counter()
    ports = macro.ports.port_count
    Kip0, Kii0, Cii = internal_blocks(macro.base, ports)
    Kip1, Kii1, _ = internal_blocks(macro.delta, ports)
    columns = macro_columns(macro.compiled)
    if len(columns) != ports:
        raise RuntimeError("one exact port is required per macro column")

    rows, cols, values = [], [], []
    orders, eigenvalues, static_errors, parameter_errors = [], [], [], []
    offset = 0
    for port, cells in enumerate(columns):
        k0 = Kii0[cells, :][:, cells].toarray()
        c0 = Cii[cells, :][:, cells].toarray()
        k1 = Kii1[cells, :][:, cells].toarray()
        b0 = Kip0[cells, port].toarray().ravel()
        b1 = Kip1[cells, port].toarray().ravel()
        static = -scipy.linalg.solve(k0, b0, assume_a="sym", check_finite=False)
        sensitivity_rhs = k1 @ static + b1
        sensitivity = -scipy.linalg.solve(
            k0, sensitivity_rhs, assume_a="sym", check_finite=False
        )
        eig, vectors = scipy.linalg.eigh(k0, c0, check_finite=False)
        dynamic = np.flatnonzero(eig <= run.modal_cutoff_per_s)[
            : run.dynamic_modes_per_column
        ]
        local = orthonormal_range(
            np.column_stack(
                (static, sensitivity, np.ones(cells.size), vectors[:, dynamic])
            )
        )
        static_errors.append(
            np.linalg.norm(k0 @ static + b0)
            / max(np.linalg.norm(b0), np.finfo(float).tiny)
        )
        parameter_errors.append(
            np.linalg.norm(k0 @ sensitivity + sensitivity_rhs)
            / max(np.linalg.norm(sensitivity_rhs), np.finfo(float).tiny)
        )
        eigenvalues.extend(eig[dynamic].tolist())
        orders.append(local.shape[1])
        for local_row, cell in enumerate(cells):
            nonzero = np.flatnonzero(np.abs(local[local_row]) > 1e-14)
            rows.extend([int(cell)] * nonzero.size)
            cols.extend((offset + nonzero).tolist())
            values.extend(local[local_row, nonzero].tolist())
        offset += local.shape[1]

    W = sp.csc_matrix((values, (rows, cols)), shape=(Kii0.shape[0], offset))
    return Basis(
        W,
        np.asarray(orders),
        np.asarray(eigenvalues),
        float(max(static_errors)),
        float(max(parameter_errors)),
        projected_residual(macro.base, ports, W, run.residual_block_size),
        projected_residual(macro.at(macro.anchor_h), ports, W, run.residual_block_size),
        float(spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc"))),
        time.perf_counter() - started,
    )


def project(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    T = sp.block_diag((sp.eye(ports, format="csc"), W), format="csc")

    def matrix(A):
        reduced = (T.T @ A @ T).tocsc()
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        matrix(operators.K),
        matrix(operators.C),
        np.asarray(T.T @ operators.f).ravel(),
    )


def project_affine(macro: MacroAffine, W: sp.csc_matrix) -> ReducedAffine:
    started = time.perf_counter()
    ports = macro.ports.port_count
    base, delta = project(macro.base, ports, W), project(macro.delta, ports, W)
    return ReducedAffine(macro.anchor_h, base, delta, time.perf_counter() - started)


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


def csc_bytes(matrix) -> int:
    matrix = matrix.tocsc()
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def reference(cfg: Package, run: Run, h: float) -> Reference:
    started = time.perf_counter()
    steady_compiled = build_package(cfg, run, True, Study.STEADY, h).compile()
    transient_compiled = build_package(cfg, run, True, Study.TRANSIENT, h).compile()
    compile_s = time.perf_counter() - started
    operators = transient_compiled.assemble()

    started = time.perf_counter()
    with steady_compiled.solve(opts=solve_options(run, False)) as solution:
        steady = np.asarray(solution.temperature).copy()
    steady_solve_s = time.perf_counter() - started

    started = time.perf_counter()
    with transient_compiled.solve(opts=solve_options(run, True)) as solution:
        times = np.asarray(solution.history_times).copy()
        transient = np.asarray(solution.temperature_history).copy()
    transient_solve_s = time.perf_counter() - started
    steady_compiled.close()
    transient_compiled.close()
    return Reference(
        steady,
        times,
        transient,
        compile_s,
        steady_solve_s,
        transient_solve_s,
        operators.K.shape[0],
        operators.K.nnz,
        operators.C.nnz,
        csc_bytes(operators.K) + csc_bytes(operators.C),
    )


def evaluate(
    data: Data,
    cfg: Package,
    run: Run,
    basis: Basis,
    reduced: ReducedAffine,
    h: float,
    ref: Reference,
):
    operators, online_assembly_s = reduced.at(h)
    internal0 = np.asarray(basis.W.T @ np.full(basis.W.shape[0], cfg.ambient_K)).ravel()

    def run_solve(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count, cfg.ambient_K),
            np.full(cfg.ports, cfg.ambient_K),
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

    steady_state, steady_solve_s = run_solve(False)
    times, transient_states, transient_solve_s = run_solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        recovered = np.empty((states.shape[0], data.full_layout.cell_count))
        recovered[:, data.detail_cells] = states[:, :detail_n]
        recovered[:, data.macro_cells] = (
            basis.W @ states[:, detail_n + cfg.ports :].T
        ).T
        return recovered

    steady_error = float(np.max(np.abs(recover(steady_state)[0] - ref.steady)))
    if times.shape != ref.times.shape or not np.allclose(
        times, ref.times, atol=1e-12, rtol=0
    ):
        raise RuntimeError("full and reduced solvers returned different output times")
    transient_error = float(np.max(np.abs(recover(transient_states) - ref.transient)))
    return {
        "h_W_m2K": h,
        "affine_coordinate": h / reduced.anchor_h,
        "steady_error_K": steady_error,
        "transient_error_K": transient_error,
        "online_reduced_assembly_s": online_assembly_s,
        "online_full_order_macro_assemblies": 0,
        "full_compile_s": ref.compile_s,
        "full_steady_solve_s": ref.steady_solve_s,
        "reduced_steady_solve_s": steady_solve_s,
        "steady_speedup": ref.steady_solve_s
        / max(steady_solve_s, np.finfo(float).tiny),
        "full_transient_solve_s": ref.transient_solve_s,
        "reduced_transient_solve_s": transient_solve_s,
        "transient_speedup": ref.transient_solve_s
        / max(transient_solve_s, np.finfo(float).tiny),
        "full_order": ref.order,
        "reduced_online_order": int(detail_n + operators.K.shape[0]),
        "full_k_nnz": ref.k_nnz,
        "full_c_nnz": ref.c_nnz,
        "reduced_macro_k_nnz": int(operators.K.nnz),
        "reduced_macro_c_nnz": int(operators.C.nnz),
        "full_operator_bytes": ref.bytes,
        "reduced_macro_bytes": csc_bytes(operators.K) + csc_bytes(operators.C),
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
            Run(duration_s=0.2, speedup_target=1.0),
        )
    return Package(), Run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run = configs(args.quick)

    print("=" * 100)
    print("Transient BCI-ROM - affine-parametric sparse DtN reduction")
    print("=" * 100)
    started = time.perf_counter()
    data = assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    basis = build_basis(data.macro, run)
    reduced = project_affine(data.macro, basis.W)
    density = basis.W.nnz / max(1, basis.W.shape[0] * basis.W.shape[1])
    print(
        f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; exact ports={cfg.ports}; "
        f"macro internal {basis.W.shape[0]:,}->{basis.W.shape[1]:,}"
    )
    print(
        f"Basis column order min/mean/max={basis.column_orders.min()}/"
        f"{basis.column_orders.mean():.2f}/{basis.column_orders.max()}, "
        f"density={density:.3e}; residual base/anchor="
        f"{basis.projected_residual_base:.3e}/{basis.projected_residual_anchor:.3e}"
    )

    results = []
    offline_s = assembly_s + basis.seconds + reduced.seconds
    for h in run.h_values:
        result = evaluate(data, cfg, run, basis, reduced, h, reference(cfg, run, h))
        result["accuracy_passed"] = (
            max(result["steady_error_K"], result["transient_error_K"]) <= run.error_K
        )
        result["speedup_passed"] = (
            result["transient_speedup"] >= run.speedup_target if args.strict else True
        )
        result["passed"] = result["accuracy_passed"] and result["speedup_passed"]
        result["rom_offline_s"] = offline_s
        results.append(result)
        print(
            f"h={h:7.1f}: error steady/transient="
            f"{result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K; "
            f"transient full/ROM={result['full_transient_solve_s']:.3f}/"
            f"{result['reduced_transient_solve_s']:.3f}s, "
            f"speedup={result['transient_speedup']:.2f}x "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    report = {
        "schema_version": 11,
        "mode": "quick" if args.quick else "strict",
        "method": "BCI-DtN with affine convection and local sensitivity enrichment",
        "package": asdict(cfg),
        "experiment": {**asdict(run), "report": str(run.report)},
        "affine_boundary": {
            "family": "A(h)=A0+(h/anchor)*(A_anchor-A0)",
            "anchor_h_W_m2K": run.affine_anchor_h,
            "full_order_offline_assemblies": 2,
            "full_order_online_assemblies_per_h": 0,
            "capacitance_relative_change": data.macro.c_relative_change,
            "ambient_consistency_residual": data.macro.ambient_residual,
            "extraction_s": data.macro.seconds,
            "projection_s": reduced.seconds,
        },
        "reduction": {
            "physical_ports": cfg.ports,
            "full_internal_order": basis.W.shape[0],
            "reduced_internal_order": basis.W.shape[1],
            "basis_nnz": basis.W.nnz,
            "basis_density": density,
            "column_order_min_mean_max": [
                int(basis.column_orders.min()),
                float(basis.column_orders.mean()),
                int(basis.column_orders.max()),
            ],
            "local_static_residual": basis.local_static_residual,
            "local_parameter_residual": basis.local_parameter_residual,
            "projected_residual_base": basis.projected_residual_base,
            "projected_residual_anchor": basis.projected_residual_anchor,
            "orthogonality_error": basis.orthogonality_error,
            "retained_eigenvalue_range_per_s": (
                [
                    float(basis.eigenvalues_per_s.min()),
                    float(basis.eigenvalues_per_s.max()),
                ]
                if basis.eigenvalues_per_s.size
                else []
            ),
        },
        "offline_s": offline_s,
        "boundary_reuse": results,
        "passed": bool(all(item["passed"] for item in results)),
    }
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report: {run.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
