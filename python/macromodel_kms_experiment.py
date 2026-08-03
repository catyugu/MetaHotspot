#!/usr/bin/env python3
"""Experimental BCI-ROM with coarse interface ports and thermal KMS modes.

This is intentionally separate from ``macromodel_demo.py``. It tests whether
port over-resolution and all-port equal weighting are the dominant causes of the
poor accuracy/runtime trade-off before the final demo is changed again.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _macromodel_problem as problem
import macromodel_demo as legacy
from metahotspot.compiled import Operators
from metahotspot.enums import Face
from metahotspot.macromodel import PortPatch

BoundaryCase = problem.BoundaryCase


@dataclass(frozen=True)
class Package(problem.Package):
    port_grid: int = 8

    @property
    def x_vertices_mm(self):
        return axis_vertices(self)

    @property
    def y_vertices_mm(self):
        return axis_vertices(self)

    @property
    def port_shape(self):
        return self.port_grid, self.port_grid

    @property
    def ports(self):
        return self.port_grid**2

    @property
    def fine_interface_shape(self):
        x = problem.footprint_cell_indices(self.x_vertices_mm, self.tim_size_mm)
        y = problem.footprint_cell_indices(self.y_vertices_mm, self.tim_size_mm)
        return int(x.size), int(y.size)


def axis_vertices(cfg: Package):
    points = [
        -cfg.cold_plate_size_mm / 2,
        -cfg.spreader_size_mm / 2,
        -cfg.bump_region_size_mm / 2,
        -cfg.die_size_mm / 2,
        0.0,
        cfg.die_size_mm / 2,
        cfg.bump_region_size_mm / 2,
        cfg.spreader_size_mm / 2,
        cfg.cold_plate_size_mm / 2,
    ]
    tile = cfg.chiplet_size_mm / 4
    for x0, _ in cfg.chiplet_origins_mm:
        points.extend(x0 + tile * np.arange(5))
    points.extend(
        np.linspace(
            -cfg.tim_size_mm / 2,
            cfg.tim_size_mm / 2,
            cfg.port_grid + 1,
        )
    )
    return problem.refined_breakpoints(points, cfg.max_xy_cell_mm)


def coarse_port_patches(cfg: Package, face: Face, z_m: float):
    bounds = (
        np.linspace(
            -cfg.tim_size_mm / 2,
            cfg.tim_size_mm / 2,
            cfg.port_grid + 1,
        )
        * 1e-3
    )
    return [
        PortPatch(
            int(face),
            z_m,
            (bounds[ix], bounds[ix + 1], bounds[iy], bounds[iy + 1]),
        )
        for ix in range(cfg.port_grid)
        for iy in range(cfg.port_grid)
    ]


problem.port_patches = coarse_port_patches


@dataclass(frozen=True)
class Run:
    error_K: float = 0.05
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    expansion_points: int = 6
    modal_modes_per_anchor: int = 48
    input_modes: int = 49
    modes_per_snapshot: int = 8
    relative_tolerance: float = 1e-8
    transfer_tolerance: float = 5e-4
    state_tolerance: float = 3e-3
    enrichment_block: int = 12
    max_dynamic_order: int = 144
    speedup_target: float = 2.0
    compression_target: float = 10.0
    report: Path = Path("results/bci_rom_kms_experiment.json")

    @property
    def modal_cutoff_per_s(self):
        return math.pi / self.dt_s

    @property
    def expansion_points_per_s(self):
        low = max(0.25 / self.duration_s, np.finfo(float).tiny)
        return (
            0.0,
            *np.geomspace(
                low,
                self.modal_cutoff_per_s,
                self.expansion_points - 1,
            ),
        )


class Basis(NamedTuple):
    lifting: np.ndarray
    V: np.ndarray
    capacity: np.ndarray
    initial_order: int
    final_order: int
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    transfer_history: np.ndarray
    state_history: np.ndarray
    residual_history: np.ndarray
    static_residual: float
    unity_residual: float
    orthogonality_error: float
    seconds: float


class Scan(NamedTuple):
    transfer: float
    state: float
    residual: float
    correction: np.ndarray
    label: str


def blocks(op: Operators, ports: int):
    return (
        op.K[:ports, :ports].tocsc(),
        op.K[:ports, ports:].tocsc(),
        op.K[ports:, :ports].tocsc(),
        op.K[ports:, ports:].tocsc(),
        op.C[:ports, :ports].tocsc(),
        op.C[:ports, ports:].tocsc(),
        op.C[ports:, :ports].tocsc(),
        op.C[ports:, ports:].tocsc(),
    )


def boundaries(run: Run):
    anchor = run.affine_anchor_h
    return (
        (0.2 * anchor,) * 4,
        (anchor,) * 4,
        (3.2 * anchor,) * 4,
        (3.2 * anchor, 0.2 * anchor, 0.6 * anchor, 1.8 * anchor),
        (0.2 * anchor, 3.2 * anchor, 1.8 * anchor, 0.6 * anchor),
    )


def validation_boundaries(run: Run):
    anchor = run.affine_anchor_h
    return (
        (0.08 * anchor,) * 4,
        (0.55 * anchor, 1.65 * anchor, 2.85 * anchor, 0.32 * anchor),
        (3.4 * anchor, 0.25 * anchor, 0.45 * anchor, 2.1 * anchor),
        (0.35 * anchor, 2.6 * anchor, 3.0 * anchor, 0.18 * anchor),
    )


def dct(order: int):
    x = np.arange(order)[:, None]
    k = np.arange(order)[None, :]
    matrix = np.cos(np.pi * (x + 0.5) * k / order)
    matrix[:, 0] /= np.sqrt(order)
    matrix[:, 1:] *= np.sqrt(2 / order)
    return matrix


def port_modes(port_grid: int, count: int):
    matrix = dct(port_grid)
    pairs = sorted(
        ((i, j) for i in range(port_grid) for j in range(port_grid)),
        key=lambda pair: (
            (pair[0] / max(port_grid - 1, 1)) ** 2
            + (pair[1] / max(port_grid - 1, 1)) ** 2,
            pair[0] + pair[1],
        ),
    )[:count]
    columns = []
    for i, j in pairs:
        wave_number = (
            (i / max(port_grid - 1, 1)) ** 2
            + (j / max(port_grid - 1, 1)) ** 2
        )
        columns.append(
            np.kron(matrix[:, i], matrix[:, j])
            / (1 + 12 * wave_number) ** 1.5
        )
    return np.ascontiguousarray(np.column_stack(columns))


def static_lifting(op: Operators, ports: int):
    _, _, Kip, Kii, *_ = blocks(op, ports)
    lifting = np.ascontiguousarray(
        spla.splu(Kii).solve(-Kip.toarray())
    )
    residual = np.asarray(Kii @ lifting + Kip.toarray())
    return (
        lifting,
        float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(Kip.data), np.finfo(float).tiny)
        ),
        float(
            np.linalg.norm(lifting @ np.ones(ports) - 1)
            / np.sqrt(lifting.shape[0])
        ),
    )


def capacity(op: Operators, ports: int):
    matrix = op.C[ports:, ports:].tocsc()
    off_diagonal = matrix - sp.diags(matrix.diagonal(), format="csc")
    if spla.norm(off_diagonal) > 1e-11 * max(spla.norm(matrix), 1.0):
        raise RuntimeError("expected diagonal FVM capacity matrix")
    diagonal = np.asarray(matrix.diagonal())
    if np.any(diagonal <= 0):
        raise RuntimeError("non-positive thermal capacity")
    return diagonal


def append_modes(V, block, capacities, tolerance, max_add, max_order):
    if block.size == 0 or V.shape[1] >= max_order:
        return V, np.empty(0)
    block = np.asarray(block, dtype=float)
    if V.shape[1]:
        block -= V @ (V.T @ (capacities[:, None] * block))
        block -= V @ (V.T @ (capacities[:, None] * block))
    weighted = np.sqrt(capacities)[:, None] * block
    norms = np.linalg.norm(weighted, axis=0)
    keep = norms > np.finfo(float).eps * max(
        1.0,
        norms.max(initial=0.0),
    )
    if not np.any(keep):
        return V, np.empty(0)
    U, singular_values, _ = scipy.linalg.svd(
        weighted[:, keep],
        full_matrices=False,
        check_finite=False,
    )
    count = min(
        np.count_nonzero(
            singular_values >= tolerance * singular_values[0]
        ),
        max_add,
        max_order - V.shape[1],
    )
    if count <= 0:
        return V, np.empty(0)
    candidates = U[:, :count] / np.sqrt(capacities)[:, None]
    if V.shape[1]:
        candidates -= V @ (
            V.T @ (capacities[:, None] * candidates)
        )
    gram = candidates.T @ (capacities[:, None] * candidates)
    values, vectors = scipy.linalg.eigh(gram, check_finite=False)
    good = values > 1e-12 * max(values.max(initial=0.0), 1.0)
    candidates = candidates @ (
        vectors[:, good] / np.sqrt(values[good])[None, :]
    )
    return (
        np.ascontiguousarray(np.column_stack((V, candidates))),
        singular_values[: candidates.shape[1]],
    )


def fixed_modes(op: Operators, ports: int, count: int):
    *_, Kii, _Cpp, _Cpi, _Cip, Cii = blocks(op, ports)
    count = min(count, Kii.shape[0] - 1)
    eigenvalues, modes = spla.eigsh(
        Kii,
        M=Cii,
        k=count,
        sigma=0.0,
        which="LM",
        tol=1e-9,
    )
    order = np.argsort(eigenvalues)
    return np.asarray(modes[:, order]), np.asarray(eigenvalues[order])


def response(op: Operators, ports: int, frequency: float, inputs):
    Kpp, Kpi, Kip, Kii, Cpp, Cpi, Cip, Cii = blocks(op, ports)
    A = (Kii + frequency * Cii).tocsc()
    B = (Kip + frequency * Cip).tocsc()
    H = (Kpi + frequency * Cpi).tocsc()
    P = (Kpp + frequency * Cpp).tocsc()
    exact = np.ascontiguousarray(
        spla.splu(A).solve(-np.asarray(B @ inputs))
    )
    return exact, np.asarray(P @ inputs + H @ exact), A, B, H, P


def approximate(A, B, lifting, V, inputs):
    static = lifting @ inputs
    if not V.shape[1]:
        return static
    reduced_A = np.asarray(V.T @ (A @ V))
    rhs = -np.asarray(V.T @ (A @ static + B @ inputs))
    return static + V @ scipy.linalg.solve(
        reduced_A,
        rhs,
        assume_a="sym",
        check_finite=False,
    )


def scan(macro, run, lifting, V, capacities):
    ports = macro.ports.port_count
    port_grid = int(round(math.sqrt(ports)))
    if port_grid * port_grid != ports:
        raise RuntimeError("KMS experiment requires a square port grid")
    inputs = port_modes(port_grid, run.input_modes)
    frequencies = tuple(
        dict.fromkeys(
            (
                *run.expansion_points_per_s,
                *np.geomspace(
                    0.35 / run.duration_s,
                    run.modal_cutoff_per_s,
                    5,
                ),
            )
        )
    )
    worst = None
    for boundary_index, h_values in enumerate(validation_boundaries(run)):
        op = macro.at(h_values)
        for frequency in frequencies:
            exact, exact_port, A, B, H, P = response(
                op,
                ports,
                float(frequency),
                inputs,
            )
            reduced = approximate(A, B, lifting, V, inputs)
            reduced_port = np.asarray(P @ inputs + H @ reduced)
            correction = exact - reduced
            transfer_error = float(
                np.linalg.norm(reduced_port - exact_port)
                / max(np.linalg.norm(exact_port), np.finfo(float).tiny)
            )
            state_error = float(
                np.linalg.norm(
                    np.sqrt(capacities)[:, None] * correction
                )
                / max(
                    np.linalg.norm(
                        np.sqrt(capacities)[:, None] * exact
                    ),
                    np.finfo(float).tiny,
                )
            )
            residual = float(
                np.linalg.norm(A @ reduced + B @ inputs)
                / max(
                    np.linalg.norm(B @ inputs),
                    np.finfo(float).tiny,
                )
            )
            candidate = Scan(
                transfer_error,
                state_error,
                residual,
                correction,
                f"b{boundary_index}/s={frequency:.5g}",
            )
            score = max(
                transfer_error / run.transfer_tolerance,
                state_error / run.state_tolerance,
            )
            if worst is None:
                worst = candidate
            else:
                worst_score = max(
                    worst.transfer / run.transfer_tolerance,
                    worst.state / run.state_tolerance,
                )
                if score > worst_score:
                    worst = candidate
    return worst


def build_basis(macro, run: Run):
    started = time.perf_counter()
    ports = macro.ports.port_count
    reference = macro.at((run.affine_anchor_h,) * 4)
    lifting, static_residual, unity_residual = static_lifting(
        reference,
        ports,
    )
    capacities = capacity(reference, ports)
    V = np.empty((lifting.shape[0], 0))
    eigenvalues = []
    singular_values = []
    initial_limit = run.max_dynamic_order - 2 * run.enrichment_block
    modal_limit = max(1, initial_limit // 2)
    for h_values in boundaries(run):
        modes, values = fixed_modes(
            macro.at(h_values),
            ports,
            run.modal_modes_per_anchor,
        )
        keep = values <= 4 * run.modal_cutoff_per_s
        modes, values = modes[:, keep], values[keep]
        V, added = append_modes(
            V,
            modes,
            capacities,
            run.relative_tolerance,
            modes.shape[1],
            modal_limit,
        )
        eigenvalues.extend(values.tolist())
        singular_values.extend(added.tolist())
        if V.shape[1] >= modal_limit:
            break

    port_grid = int(round(math.sqrt(ports)))
    inputs = port_modes(port_grid, run.input_modes)
    points = 0
    columns = 0
    for h_values in boundaries(run):
        op = macro.at(h_values)
        for frequency in run.expansion_points_per_s:
            exact, _, _, _, _, _ = response(
                op,
                ports,
                float(frequency),
                inputs,
            )
            V, added = append_modes(
                V,
                exact - lifting @ inputs,
                capacities,
                run.relative_tolerance,
                run.modes_per_snapshot,
                initial_limit,
            )
            singular_values.extend(added.tolist())
            points += 1
            columns += inputs.shape[1]
            if V.shape[1] >= initial_limit:
                break
        if V.shape[1] >= initial_limit:
            break

    initial_order = V.shape[1]
    transfer_history = []
    state_history = []
    residual_history = []
    while True:
        worst = scan(macro, run, lifting, V, capacities)
        transfer_history.append(worst.transfer)
        state_history.append(worst.state)
        residual_history.append(worst.residual)
        if (
            worst.transfer <= run.transfer_tolerance
            and worst.state <= run.state_tolerance
        ) or V.shape[1] >= run.max_dynamic_order:
            break
        previous = V.shape[1]
        V, added = append_modes(
            V,
            worst.correction,
            capacities,
            min(run.relative_tolerance, 1e-10),
            run.enrichment_block,
            run.max_dynamic_order,
        )
        singular_values.extend(added.tolist())
        if V.shape[1] == previous:
            break

    return (
        Basis(
            lifting,
            V,
            capacities,
            initial_order,
            V.shape[1],
            np.asarray(eigenvalues),
            np.asarray(singular_values),
            np.asarray(transfer_history),
            np.asarray(state_history),
            np.asarray(residual_history),
            static_residual,
            unity_residual,
            float(
                np.linalg.norm(
                    V.T @ (capacities[:, None] * V)
                    - np.eye(V.shape[1])
                )
            ),
            time.perf_counter() - started,
        ),
        points,
        columns,
    )


def project_matrix(matrix, ports, lifting, V):
    App = matrix[:ports, :ports].toarray()
    Api = matrix[:ports, ports:].tocsc()
    Aip = matrix[ports:, :ports].tocsc()
    Aii = matrix[ports:, ports:].tocsc()
    Aii_lifting = np.asarray(Aii @ lifting)
    Aii_V = np.asarray(Aii @ V)
    residual = np.asarray(Aip.toarray() + Aii_lifting)
    reduced = np.block(
        [
            [
                App
                + np.asarray(Api @ lifting)
                + lifting.T @ residual,
                np.asarray(Api @ V) + lifting.T @ Aii_V,
            ],
            [V.T @ residual, V.T @ Aii_V],
        ]
    )
    return sp.csc_matrix(0.5 * (reduced + reduced.T))


def project(op, ports, basis):
    port_rhs = np.asarray(op.f[:ports])
    internal_rhs = np.asarray(op.f[ports:])
    return Operators(
        project_matrix(
            op.K,
            ports,
            basis.lifting,
            basis.V,
        ),
        project_matrix(
            op.C,
            ports,
            basis.lifting,
            basis.V,
        ),
        np.r_[
            port_rhs + basis.lifting.T @ internal_rhs,
            basis.V.T @ internal_rhs,
        ],
    )


@dataclass(frozen=True)
class Reduced:
    anchor_h: float
    base: Operators
    components: tuple[Operators, ...]
    seconds: float

    def at(self, h_values: Sequence[float]):
        started = time.perf_counter()
        return (
            problem.combine_many(
                self.base,
                self.components,
                np.asarray(h_values) / self.anchor_h,
            ),
            time.perf_counter() - started,
        )


def project_affine(macro, basis):
    started = time.perf_counter()
    ports = macro.ports.port_count
    return Reduced(
        macro.anchor_h,
        project(macro.base, ports, basis),
        tuple(
            project(component, ports, basis)
            for component in macro.components
        ),
        time.perf_counter() - started,
    )


def evaluate(data, cfg, run, basis, reduced, boundary, reference):
    proxy = SimpleNamespace(lifting=basis.lifting, V=basis.V)
    return legacy.evaluate(
        data,
        cfg,
        run,
        proxy,
        reduced,
        boundary,
        reference,
    )


def configs(quick: bool):
    if quick:
        cfg = Package(
            substrate_cells=3,
            bump_cells=1,
            die_cells=2,
            tim_cells=1,
            spreader_cells=3,
            cold_plate_cells=4,
            max_xy_cell_mm=6.0,
            bump_rows=8,
            bump_columns=8,
            port_grid=8,
        )
        run = Run(
            error_K=0.15,
            duration_s=0.20,
            expansion_points=4,
            modal_modes_per_anchor=24,
            input_modes=36,
            modes_per_snapshot=6,
            relative_tolerance=3e-7,
            transfer_tolerance=8e-4,
            state_tolerance=5e-3,
            enrichment_block=10,
            max_dynamic_order=88,
            speedup_target=1.0,
            compression_target=12.0,
        )
        cases = (
            BoundaryCase("uniform-low", (500.0,) * 4),
            BoundaryCase("uniform-high", (8000.0,) * 4),
            BoundaryCase(
                "diagonal-skew",
                (8000.0, 700.0, 1200.0, 6000.0),
            ),
        )
        return cfg, run, cases
    return (
        Package(port_grid=8),
        Run(),
        (
            BoundaryCase("uniform-low", (500.0,) * 4),
            BoundaryCase("uniform-medium", (2500.0,) * 4),
            BoundaryCase("uniform-high", (8000.0,) * 4),
            BoundaryCase(
                "x-gradient",
                (500.0, 8000.0, 500.0, 8000.0),
            ),
            BoundaryCase(
                "diagonal-skew",
                (8000.0, 700.0, 1200.0, 6000.0),
            ),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true")
    group.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run, cases = configs(args.quick)
    print("=" * 100)
    print(
        "Experimental BCI-ROM - 8x8 aggregated ports + "
        "fixed-interface thermal KMS"
    )
    print("=" * 100)
    print(
        f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; "
        f"fine interface={cfg.fine_interface_shape}; "
        f"physical ports={cfg.ports}"
    )
    data = problem.assemble(cfg, run)
    try:
        basis, points, columns = build_basis(data.macro, run)
        reduced = project_affine(data.macro, basis)
        dynamic_compression = basis.lifting.shape[0] / basis.final_order
        total_compression = (
            cfg.ports + basis.lifting.shape[0]
        ) / (cfg.ports + basis.final_order)
        print(
            f"Macro internal {basis.lifting.shape[0]}->"
            f"{basis.final_order}; dynamic/total compression="
            f"{dynamic_compression:.2f}x/{total_compression:.2f}x"
        )
        print(
            f"KMS order {basis.initial_order}->{basis.final_order}; "
            f"rational points/columns={points}/{columns}"
        )
        print(
            "Validation transfer/state/residual="
            + " | ".join(
                f"{a:.3e}/{b:.3e}/{c:.3e}"
                for a, b, c in zip(
                    basis.transfer_history,
                    basis.state_history,
                    basis.residual_history,
                )
            )
        )
        passivity = legacy.passivity_metrics(reduced, cases)
        results = []
        for case in cases:
            result = evaluate(
                data,
                cfg,
                run,
                basis,
                reduced,
                case,
                problem.reference(cfg, run, case),
            )
            result["passed"] = (
                max(
                    result["steady_error_K"],
                    result["transient_error_K"],
                )
                <= run.error_K
            )
            results.append(result)
            print(
                f"{case.name:>16s}: error steady/transient="
                f"{result['steady_error_K']:.5f}/"
                f"{result['transient_error_K']:.5f} K; "
                f"full/ROM={result['full_transient_solve_s']:.3f}/"
                f"{result['reduced_transient_solve_s']:.3f}s; "
                f"speedup={result['transient_speedup']:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )
        report = {
            "schema_version": 1,
            "experimental": True,
            "method": (
                "8x8 piecewise-constant ports + fixed-interface thermal "
                "modes + low-pass DCT rational corrections"
            ),
            "package": {
                **asdict(cfg),
                "nx": cfg.nx,
                "ny": cfg.ny,
                "fine_interface_shape": cfg.fine_interface_shape,
            },
            "run": {**asdict(run), "report": str(run.report)},
            "reduction": {
                "full_internal_order": basis.lifting.shape[0],
                "dynamic_order": basis.final_order,
                "dynamic_compression": dynamic_compression,
                "total_compression": total_compression,
                "transfer_history": basis.transfer_history.tolist(),
                "state_history": basis.state_history.tolist(),
                "residual_history": basis.residual_history.tolist(),
                "static_residual": basis.static_residual,
                "unity_residual": basis.unity_residual,
                "orthogonality_error": basis.orthogonality_error,
            },
            "passivity": passivity,
            "cases": results,
            "passed": bool(
                all(result["passed"] for result in results)
                and passivity["passed"]
            ),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        problem.close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
