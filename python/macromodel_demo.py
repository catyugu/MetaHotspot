#!/usr/bin/env python3
"""Transient package ROM trained from global physical-port responses.

All physical interface patches remain algebraic port variables. Only the
nonlocal macro response is reduced: a global multi-shift basis is learned from
randomized port excitations, while the local port conductance stays sparse and
exact. No source geometry or spatial localization is used.
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
    response_energy: float = 0.9999
    response_holdout: float = 0.03
    probe_count: int = 48
    holdout_count: int = 8
    max_response_modes: int = 1024
    duration_s: float = 0.5
    dt_s: float = 0.025
    nominal_h: float = 2500.0
    h_values: tuple[float, ...] = (500.0, 2500.0, 8000.0)
    seed: int = 20260731
    report: Path = Path("results/bci_rom_final_results.json")

    @property
    def shifts(self) -> tuple[float, ...]:
        scale = 0.025 / self.dt_s
        return tuple(scale * x for x in (0.0, 0.5, 1, 2, 5, 10, 20, 40, 80))


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


class TestCase(NamedTuple):
    A: sp.csc_matrix
    Kip: sp.csc_matrix
    Kpi: sp.csc_matrix
    probes: np.ndarray
    exact_flux: np.ndarray


class Family(NamedTuple):
    W: np.ndarray
    energy: np.ndarray
    tests: tuple[TestCase, ...]
    seconds: float


class Reduced(NamedTuple):
    K: sp.csc_matrix
    C: sp.csc_matrix
    f: np.ndarray
    W: np.ndarray
    basis: np.ndarray


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
    ds = build_package(cfg, run, False, Study.STEADY).compile()
    dt = build_package(cfg, run, False, Study.TRANSIENT).compile()
    z = (cfg.substrate_mm + cfg.bump_mm + cfg.die_mm) * 1e-3
    detail_patches = patches(cfg, Face.ZP, z)
    samples = tuple(macro_sample(cfg, h) for h in (None, *run.h_values))
    return Data(
        full,
        ds,
        dt,
        PortMap(ds, detail_patches),
        PortMap(dt, detail_patches),
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


def response_cases(data: Data, run: Run):
    for sample in data.samples:
        _, Kpi, Kip, Kii, Cii, _, _ = split(sample)
        for shift in run.shifts:
            yield (Kii + shift * Cii).tocsc(), Kip, Kpi


def randomized_left_basis(X: np.ndarray, cap: int, seed: int):
    cap = min(cap, min(X.shape))
    rng = np.random.default_rng(seed)
    sketch = min(X.shape[1], cap + 32)
    Q = np.linalg.qr(X @ rng.standard_normal((X.shape[1], sketch)), mode="reduced")[0]
    Q = np.linalg.qr(X @ (X.T @ Q), mode="reduced")[0]
    U0, sigma, _ = scipy.linalg.svd(Q.T @ X, full_matrices=False, check_finite=False)
    return Q @ U0[:, :cap], sigma[:cap]


def build_family(data: Data, run: Run) -> Family:
    """Learn a global internal realization from all physical-port responses."""
    started = time.perf_counter()
    p = data.samples[0].ports.port_count
    n = data.samples[0].compiled.cell_count
    rng = np.random.default_rng(run.seed)
    test_rng = np.random.default_rng(run.seed + 1)
    snapshots, tests = [], []

    for A, Kip, Kpi in response_cases(data, run):
        factor = spla.splu(A)
        probes = rng.standard_normal((p, min(p, run.probe_count)))
        X = -factor.solve(Kip @ probes)
        snapshots.append(X / max(np.linalg.norm(X, "fro"), np.finfo(float).tiny))

        holdout = test_rng.standard_normal((p, min(p, run.holdout_count)))
        exact = np.asarray(Kpi @ (-factor.solve(Kip @ holdout)))
        tests.append(TestCase(A, Kip, Kpi, holdout, exact))

    uniform = np.ones(n)
    uniform /= np.linalg.norm(uniform)
    X = np.hstack(snapshots)
    X -= uniform[:, None] * (uniform @ X)[None, :]
    cap = min(run.max_response_modes - 1, min(X.shape))
    U, sigma = randomized_left_basis(X, cap, run.seed)
    W, _ = np.linalg.qr(np.column_stack((uniform, U)), mode="reduced")
    energy = np.r_[
        0.0, np.cumsum(sigma**2) / max(np.sum(sigma**2), np.finfo(float).tiny)
    ]
    return Family(
        np.ascontiguousarray(W),
        energy[: W.shape[1]],
        tuple(tests),
        time.perf_counter() - started,
    )


def holdout_error(family: Family, rank: int) -> float:
    W = family.W[:, :rank]
    worst = 0.0
    for A, Kip, Kpi, probes, exact in family.tests:
        Ar = np.asarray(W.T @ (A @ W))
        Ar = 0.5 * (Ar + Ar.T)
        reduced_state = -scipy.linalg.solve(
            Ar, np.asarray(W.T @ (Kip @ probes)), assume_a="pos"
        )
        approximate = np.asarray(Kpi @ (W @ reduced_state))
        scale = max(np.linalg.norm(exact, "fro"), np.finfo(float).tiny)
        worst = max(worst, float(np.linalg.norm(approximate - exact, "fro") / scale))
    return worst


def project(sample: Sample, W: np.ndarray) -> Reduced:
    Kpp, Kpi, Kip, Kii, Cii, fp, fi = split(sample)
    p, r = Kpp.shape[0], W.shape[1]
    Kpr = sp.csc_matrix(Kpi @ W)
    Krp = Kpr.T.tocsc()
    Krr = np.asarray(W.T @ (Kii @ W))
    Crr = np.asarray(W.T @ (Cii @ W))
    Krr = 0.5 * (Krr + Krr.T)
    Crr = 0.5 * (Crr + Crr.T)
    K = sp.bmat(((Kpp, Kpr), (Krp, sp.csc_matrix(Krr))), format="csc")
    C = sp.bmat(
        (
            (sp.csc_matrix((p, p)), sp.csc_matrix((p, r))),
            (sp.csc_matrix((r, p)), sp.csc_matrix(Crr)),
        ),
        format="csc",
    )
    f = np.r_[fp, np.asarray(W.T @ fi)]
    basis = np.zeros((p, p + r))
    basis[:, :p] = np.eye(p)
    return Reduced(K, C, f, W, basis)


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
    interior0 = W.T @ np.full(W.shape[0], cfg.ambient_K)

    def solve(transient):
        compiled = data.detail_transient if transient else data.detail_steady
        ports_ = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count, cfg.ambient_K),
            np.full(p, cfg.ambient_K),
            interior0,
        ]
        model = DtNModel((reduced.K, reduced.C, reduced.f), reduced.basis)
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

    print(f"steady ref T range: [{min(steady_ref)}, {max(steady_ref)}]")

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


def rank_for(curve: np.ndarray, target: float) -> int:
    return min(len(curve), int(np.searchsorted(curve, target, side="left")) + 1)


def configs(quick: bool):
    if quick:
        return Package(nx=16, ny=16, bump_rows=6, bump_columns=6), Run(
            duration_s=0.2, dt_s=0.025, probe_count=40, max_response_modes=768
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
    print("Transient BCI-ROM benchmark — global physical-port response realization")
    print("=" * 92)
    print(
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.nz} = {cfg.nx*cfg.ny*cfg.nz:,} cells; ports={cfg.ports} exact"
    )
    t0 = time.perf_counter()
    data = assemble(cfg, run)
    build_s = time.perf_counter() - t0
    family = build_family(data, run)
    rank = max(1, rank_for(family.energy, run.response_energy))
    nominal = reference(cfg, run, run.nominal_h)
    attempts = []

    while True:
        response_error = holdout_error(family, rank)
        steady, transient, records = evaluate(
            data, cfg, run, family.W[:, :rank], run.nominal_h, nominal
        )
        passed = (
            max(steady, transient) <= run.error_K
            and response_error <= run.response_holdout
        )
        attempts.append(
            dict(
                response_modes=rank,
                rom_order=cfg.ports + rank,
                holdout_response_error=response_error,
                steady_error_K=steady,
                transient_error_K=transient,
                transient_records=records,
                passed=passed,
            )
        )
        print(
            f"response={rank:4d}/{family.W.shape[1]:<4d} holdout={response_error:.3e} "
            f"order={cfg.ports+rank:4d} steady={steady:.5f}K transient={transient:.5f}K "
            f"{'PASS' if passed else 'EXPAND'}"
        )
        if passed or rank == family.W.shape[1]:
            break
        rank = min(family.W.shape[1], max(rank + 48, math.ceil(1.25 * rank)))

    W = family.W[:, :rank]
    boundary = []
    for h in run.h_values:
        ref = nominal if h == run.nominal_h else reference(cfg, run, h)
        steady, transient, _ = evaluate(data, cfg, run, W, h, ref)
        ok = max(steady, transient) <= run.error_K
        boundary.append(
            dict(
                h_W_m2K=h, steady_error_K=steady, transient_error_K=transient, passed=ok
            )
        )
        print(
            f"h={h:7.1f}: steady={steady:.5f}K transient={transient:.5f}K {'PASS' if ok else 'FAIL'}"
        )

    report = dict(
        schema_version=6,
        mode="quick" if args.quick else "strict",
        reduction_method="global_physical_port_response_realization",
        package=asdict(cfg),
        experiment={
            **asdict(run),
            "report": str(run.report),
            "shifts": list(run.shifts),
        },
        physical_port_count=cfg.ports,
        selected_response_modes=rank,
        selected_rom_order=cfg.ports + rank,
        transfer_training_s=family.seconds,
        model_build_s=build_s,
        attempts=attempts,
        boundary_reuse=boundary,
        passed=bool(attempts[-1]["passed"] and all(x["passed"] for x in boundary)),
    )
    run.report.parent.mkdir(parents=True, exist_ok=True)
    run.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Report: {run.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
