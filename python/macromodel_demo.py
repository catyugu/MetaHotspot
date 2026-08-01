#!/usr/bin/env python3
"""Transient boundary-condition-independent DtN macro-model benchmark.

The macro domain is reduced with a deterministic component-mode basis:

1. all physical interface ports remain exact algebraic variables;
2. the complete zero-frequency port response is retained through static
   constraint modes; and
3. fixed-interface thermal modes are retained up to the angular Nyquist rate
   pi / dt of the requested transient output grid.

The basis is extracted once from the macro domain with homogeneous external
Neumann conditions. Applied convection coefficients are used only when the
already-built basis is projected and validated. No heat-source distribution,
port locality, boundary excitation, random probe, or response snapshot is used
for training.
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
    nx: int = 24
    ny: int = 24
    width_mm: float = 40.0
    height_mm: float = 40.0
    ambient_K: float = 300.0
    substrate_mm: float = 1.2
    bump_mm: float = 0.24
    die_mm: float = 0.6
    tim_mm: float = 0.18
    spreader_mm: float = 1.2
    cold_plate_mm: float = 1.5
    substrate_cells: int = 4
    bump_cells: int = 2
    die_cells: int = 3
    tim_cells: int = 1
    spreader_cells: int = 3
    cold_plate_cells: int = 3
    bump_rows: int = 8
    bump_columns: int = 8
    bump_width_mm: float = 0.9
    chiplet_width_mm: float = 12.0
    chiplet_height_mm: float = 12.0
    chiplet_power_W: float = 25.0

    @property
    def detail_nz(self) -> int:
        return self.substrate_cells + self.bump_cells + self.die_cells

    @property
    def nz(self) -> int:
        return (
            self.detail_nz
            + self.tim_cells
            + self.spreader_cells
            + self.cold_plate_cells
        )

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
    error_K: float = 0.2
    duration_s: float = 0.5
    dt_s: float = 0.025
    nominal_h: float = 2500.0
    h_values: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    report: Path = Path("results/bci_rom_final_results.json")

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
    full: object
    detail_steady: object
    detail_transient: object
    detail_ports_steady: PortMap
    detail_ports_transient: PortMap
    samples: tuple[Sample, ...]
    detail_cells: np.ndarray
    macro_cells: np.ndarray


class Basis(NamedTuple):
    W: np.ndarray
    static_modes: int
    dynamic_modes: int
    eigenvalues_per_s: np.ndarray
    residual: float
    seconds: float


class Reduced(NamedTuple):
    K: sp.csc_matrix
    C: sp.csc_matrix
    f: np.ndarray
    W: np.ndarray
    port_basis: np.ndarray


def vertices(length: float, cells: int) -> np.ndarray:
    return np.linspace(0.0, length, cells + 1)


def z_vertices(layers) -> np.ndarray:
    out, z = [0.0], 0.0
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
    volume = cfg.chiplet_width_mm * cfg.chiplet_height_mm * cfg.die_mm * 1e-9
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
                ((0, 0), (0.2 * t, 1), (0.55 * t, 1), (0.75 * t, 0.55), (t, 0.85))
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
            [(Axis.Z, sum(x[0] for x in layers), 0, cfg.width_mm, 0, cfg.height_mm)],
        )
    return model


def patches(cfg: Package, face: Face, z: float) -> list[PortPatch]:
    dx, dy = cfg.width_mm * 1e-3 / cfg.nx, cfg.height_mm * 1e-3 / cfg.ny
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


def assemble(cfg: Package, run: Run) -> Data:
    full = build_package(cfg, run, True, Study.STEADY).compile()
    detail_steady = build_package(cfg, run, False, Study.STEADY).compile()
    detail_transient = build_package(cfg, run, False, Study.TRANSIENT).compile()
    z = (cfg.substrate_mm + cfg.bump_mm + cfg.die_mm) * 1e-3
    detail_patches = patches(cfg, Face.ZP, z)
    samples = tuple(macro_sample(cfg, h) for h in (None, *run.h_values))
    return Data(
        full,
        detail_steady,
        detail_transient,
        PortMap(detail_steady, detail_patches),
        PortMap(detail_transient, detail_patches),
        samples,
        grid_cells(full, 0, cfg.detail_nz),
        grid_cells(full, cfg.detail_nz, cfg.nz),
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


def orthonormal_range(matrix: np.ndarray, against=None) -> np.ndarray:
    """Return a deterministic numerical basis using only machine-rank truncation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0))
    if against is not None and against.shape[1]:
        matrix = matrix - against @ (against.T @ matrix)
        matrix = matrix - against @ (against.T @ matrix)
    q, r, _ = scipy.linalg.qr(
        matrix, mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((matrix.shape[0], 0))
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, diagonal > tolerance])


def low_frequency_modes(
    Kii: sp.csc_matrix,
    Cii: sp.csc_matrix,
    lu,
    cutoff: float,
    initial_count: int,
):
    """Compute every fixed-interface thermal pole not exceeding ``cutoff``."""
    n = Kii.shape[0]
    if n <= 2:
        return np.empty(0), np.empty((n, 0))
    k = min(n - 2, max(1, initial_count))
    inverse = spla.LinearOperator(
        Kii.shape,
        matvec=lu.solve,
        matmat=lu.solve,
        dtype=np.float64,
    )
    v0 = np.sin(math.sqrt(2.0) * np.arange(1, n + 1, dtype=np.float64))
    v0 /= np.linalg.norm(v0)
    while True:
        values, vectors = spla.eigsh(
            Kii,
            k=k,
            M=Cii,
            sigma=0.0,
            which="LM",
            OPinv=inverse,
            v0=v0,
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        if values[-1] > cutoff:
            break
        if k == n - 2:
            raise RuntimeError(
                "the time grid retains essentially the full macro domain; "
                "reduce the output bandwidth or skip model reduction"
            )
        k = min(n - 2, 2 * k)
    tolerance = cutoff * (1.0 + 64.0 * np.finfo(np.float64).eps)
    keep = values <= tolerance
    return values[keep], vectors[:, keep]


def build_basis(sample: Sample, run: Run) -> Basis:
    """Build a boundary-condition-independent KMS component-mode basis."""
    started = time.perf_counter()
    _, _, Kip, Kii, Cii, _, _ = split(sample)
    lu = spla.splu(Kii)

    dense_coupling = Kip.toarray()
    static_response = -lu.solve(dense_coupling)
    static_basis = orthonormal_range(
        np.column_stack((static_response, np.ones(Kii.shape[0])))
    )
    eigenvalues, eigenvectors = low_frequency_modes(
        Kii,
        Cii,
        lu,
        run.modal_cutoff_per_s,
        initial_count=sample.ports.port_count,
    )
    dynamic_basis = orthonormal_range(eigenvectors, against=static_basis)
    W = np.ascontiguousarray(np.column_stack((static_basis, dynamic_basis)))

    scale = max(np.linalg.norm(dense_coupling, ord="fro"), np.finfo(float).tiny)
    residual = float(
        np.linalg.norm(Kii @ static_response + dense_coupling, ord="fro") / scale
    )
    return Basis(
        W,
        static_basis.shape[1],
        dynamic_basis.shape[1],
        np.asarray(eigenvalues),
        residual,
        time.perf_counter() - started,
    )


def project(sample: Sample, W: np.ndarray) -> Reduced:
    Kpp, Kpi, Kip, Kii, Cii, fp, fi = split(sample)
    p, r = Kpp.shape[0], W.shape[1]
    Kpr = sp.csc_matrix(Kpi @ W)
    Krp = sp.csc_matrix(W.T @ Kip)
    Krr = np.asarray(W.T @ (Kii @ W))
    Crr = np.asarray(W.T @ (Cii @ W))
    Krr = 0.5 * (Krr + Krr.T)
    Crr = 0.5 * (Crr + Crr.T)
    zero_pp = sp.csc_matrix((p, p))
    zero_pr = sp.csc_matrix((p, r))
    K = sp.bmat(((Kpp, Kpr), (Krp, sp.csc_matrix(Krr))), format="csc")
    C = sp.bmat(((zero_pp, zero_pr), (zero_pr.T, sp.csc_matrix(Crr))), format="csc")
    f = np.r_[fp, np.asarray(W.T @ fi)]
    port_basis = np.zeros((p, p + r))
    port_basis[:, :p] = np.eye(p)
    return Reduced(K, C, f, W, port_basis)


def options(run: Run, transient: bool) -> SolveOptions:
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


def reference(cfg: Package, run: Run, h: float):
    steady = (
        build_package(cfg, run, True, Study.STEADY, h)
        .compile()
        .solve(opts=options(run, False))
    )
    transient = (
        build_package(cfg, run, True, Study.TRANSIENT, h)
        .compile()
        .solve(opts=options(run, True))
    )
    return (
        np.asarray(steady.temperature).copy(),
        np.asarray(transient.history_times).copy(),
        np.asarray(transient.temperature_history).copy(),
    )


def evaluate(data: Data, cfg: Package, run: Run, W: np.ndarray, h: float, ref):
    sample = next(x for x in data.samples if x.h == h)
    reduced = project(sample, W)
    p = cfg.ports
    internal0 = W.T @ np.full(W.shape[0], cfg.ambient_K)

    def solve(transient):
        compiled = data.detail_transient if transient else data.detail_steady
        ports_ = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count, cfg.ambient_K),
            np.full(p, cfg.ambient_K),
            internal0,
        ]
        model = DtNModel((reduced.K, reduced.C, reduced.f), reduced.port_basis)
        return solve_macro(compiled, model, ports_, state, options(run, transient))

    steady, transient = solve(False), solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        macro = states[:, detail_n:]
        out = np.empty((states.shape[0], data.full.cell_count))
        out[:, data.detail_cells] = states[:, :detail_n]
        out[:, data.macro_cells] = macro[:, p:] @ W.T
        return out

    steady_ref, times_ref, transient_ref = ref
    steady_error = float(
        np.max(np.abs(recover(np.asarray(steady.state))[0] - steady_ref))
    )
    times = np.asarray(transient.history_times)
    if times.shape != times_ref.shape or not np.allclose(
        times, times_ref, atol=1e-12, rtol=0
    ):
        raise RuntimeError("full and reduced solvers returned different output times")
    transient_error = float(
        np.max(np.abs(recover(np.asarray(transient.state_history)) - transient_ref))
    )
    return steady_error, transient_error, len(times)


def configs(quick: bool):
    if quick:
        return Package(nx=16, ny=16, bump_rows=6, bump_columns=6), Run(
            duration_s=0.2, dt_s=0.025
        )
    return Package(), Run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run = configs(args.quick)

    print("=" * 92)
    print("Transient BCI-ROM benchmark - deterministic static-constraint KMS DtN")
    print("=" * 92)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.nz} = {cfg.nx*cfg.ny*cfg.nz:,} cells; "
        f"physical ports={cfg.ports}"
    )
    started = time.perf_counter()
    data = assemble(cfg, run)
    assembly_s = time.perf_counter() - started

    training_sample = next(sample for sample in data.samples if sample.h is None)
    basis = build_basis(training_sample, run)
    print(
        f"Basis: static={basis.static_modes}, dynamic={basis.dynamic_modes}, "
        f"internal order={basis.W.shape[1]}, cutoff={run.modal_cutoff_per_s:.6g}/s, "
        f"static residual={basis.residual:.3e}, extraction={basis.seconds:.3f}s"
    )

    boundary = []
    nominal_ref = reference(cfg, run, run.nominal_h)
    for h in run.h_values:
        ref = nominal_ref if h == run.nominal_h else reference(cfg, run, h)
        steady, transient, records = evaluate(data, cfg, run, basis.W, h, ref)
        passed = max(steady, transient) <= run.error_K
        boundary.append(
            dict(
                h_W_m2K=h,
                steady_error_K=steady,
                transient_error_K=transient,
                transient_records=records,
                passed=passed,
            )
        )
        print(
            f"h={h:7.1f}: steady={steady:.5f}K transient={transient:.5f}K "
            f"{'PASS' if passed else 'FAIL'}"
        )

    report = dict(
        schema_version=7,
        mode="quick" if args.quick else "strict",
        reduction_method="static_constraint_fixed_interface_kms",
        training_boundary="homogeneous_neumann",
        input_training="none",
        package=asdict(cfg),
        experiment={
            **asdict(run),
            "report": str(run.report),
            "modal_cutoff_per_s": run.modal_cutoff_per_s,
        },
        physical_port_count=cfg.ports,
        static_constraint_modes=basis.static_modes,
        fixed_interface_modes=basis.dynamic_modes,
        reduced_internal_order=basis.W.shape[1],
        reduced_total_order=cfg.ports + basis.W.shape[1],
        retained_eigenvalue_range_per_s=(
            [float(basis.eigenvalues_per_s[0]), float(basis.eigenvalues_per_s[-1])]
            if basis.eigenvalues_per_s.size
            else []
        ),
        static_constraint_residual=basis.residual,
        model_assembly_s=assembly_s,
        basis_extraction_s=basis.seconds,
        boundary_reuse=boundary,
        passed=bool(all(item["passed"] for item in boundary)),
    )
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Report: {run.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
