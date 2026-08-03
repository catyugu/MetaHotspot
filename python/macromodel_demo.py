#!/usr/bin/env python3
"""Transient BCI-ROM experiment using affine Craig-Bampton reduction.

Geometry, power maps, full-order references, and affine DtN extraction live in
``_macromodel_problem.py``. This runner applies the transformation

    [Tp]   [I  0] [Tp]
    [Ti] = [L  V] [ q],

where L is the exact homogeneous-Neumann static port lifting and V contains only
dynamic and convection-parameter corrections. L has one column per physical
port but introduces no reduced state. This removes the artificial lower bound
that forced the earlier block-diagonal basis to use roughly one internal state
per port.
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
from metahotspot.compiled import Operators, SolveOptions
from metahotspot.macromodel import solve as solve_macro

BoundaryCase = problem.BoundaryCase
Package = problem.Package
MacroAffine = problem.MacroAffine
Data = problem.Data
Reference = problem.Reference
POWER_MAP = problem.POWER_MAP
CHIPLET_POWER_SCALE = problem.CHIPLET_POWER_SCALE


@dataclass(frozen=True)
class Run:
    error_K: float = 0.05
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    expansion_points: int = 6
    snapshot_relative_tolerance: float = 1.0e-8
    snapshot_modes_per_point: int = 24
    transfer_error_tolerance: float = 1.0e-5
    enrichment_block: int = 40
    max_dynamic_order: int = 320
    speedup_target: float = 2.0
    compression_target: float = 10.0
    report: Path = Path("results/bci_rom_constraint_results.json")

    @property
    def modal_cutoff_per_s(self) -> float:
        return math.pi / self.dt_s

    @property
    def expansion_points_per_s(self) -> tuple[float, ...]:
        if self.expansion_points < 2:
            return (0.0,)
        low = max(0.25 / self.duration_s, np.finfo(float).tiny)
        positive = np.geomspace(low, self.modal_cutoff_per_s, self.expansion_points - 1)
        return (0.0, *(float(v) for v in positive))


class Basis(NamedTuple):
    lifting: np.ndarray
    V: np.ndarray
    initial_order: int
    final_order: int
    snapshot_columns: int
    snapshot_points: int
    added_singular_values: np.ndarray
    transfer_error_history: np.ndarray
    residual_history: np.ndarray
    worst_transfer_error: float
    worst_residual: float
    static_residual: float
    unity_residual: float
    orthogonality_error: float
    seconds: float


@dataclass(frozen=True)
class ReducedAffine:
    anchor_h: float
    base: Operators
    components: tuple[Operators, ...]
    seconds: float

    def at(self, h_values: Sequence[float]) -> tuple[Operators, float]:
        started = time.perf_counter()
        h = np.asarray(h_values, dtype=np.float64)
        operators = problem.combine_many(
            self.base, self.components, h / self.anchor_h
        )
        return operators, time.perf_counter() - started


class TransferScan(NamedTuple):
    label: str
    transfer_error: float
    residual: float
    correction: np.ndarray


def internal_dynamic_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, :ports].tocsc(),
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, :ports].tocsc(),
        operators.C[ports:, ports:].tocsc(),
    )


def static_training_boundaries(run: Run) -> tuple[tuple[float, ...], ...]:
    a = run.affine_anchor_h
    high = 3.2 * a
    low = 0.2 * a
    unit = [tuple(a if i == q else 0.0 for i in range(4)) for q in range(4)]
    return (
        (a, a, a, a),
        (high, high, high, high),
        (high, low, 0.6 * a, 1.8 * a),
        (low, high, 1.8 * a, 0.6 * a),
        *unit,
    )


def dynamic_training_boundaries(run: Run) -> tuple[tuple[float, ...], ...]:
    a = run.affine_anchor_h
    return (
        (a, a, a, a),
        (3.2 * a, 0.2 * a, 0.6 * a, 1.8 * a),
        (0.2 * a, 3.2 * a, 1.8 * a, 0.6 * a),
    )


def validation_boundaries(run: Run) -> tuple[tuple[float, ...], ...]:
    a = run.affine_anchor_h
    return (
        (0.08 * a, 0.08 * a, 0.08 * a, 0.08 * a),
        (0.55 * a, 1.65 * a, 2.85 * a, 0.32 * a),
        (3.4 * a, 0.25 * a, 0.45 * a, 2.1 * a),
        (0.35 * a, 2.6 * a, 3.0 * a, 0.18 * a),
    )


def validation_frequencies(run: Run) -> tuple[float, ...]:
    frequencies = run.expansion_points_per_s
    return tuple(
        dict.fromkeys((frequencies[0], *frequencies[1::2], frequencies[-1]))
    )


def static_lifting(operators: Operators, ports: int):
    Kip, Kii, _, _ = internal_dynamic_blocks(operators, ports)
    lifting = np.ascontiguousarray(spla.splu(Kii).solve(-Kip.toarray()))
    residual = np.asarray(Kii @ lifting + Kip.toarray())
    static_residual = float(
        np.linalg.norm(residual, ord="fro")
        / max(np.linalg.norm(Kip.data), np.finfo(float).tiny)
    )
    unity = lifting @ np.ones(ports)
    unity_residual = float(
        np.linalg.norm(unity - np.ones(lifting.shape[0]))
        / math.sqrt(lifting.shape[0])
    )
    return lifting, static_residual, unity_residual


def transfer_map(operators: Operators, ports: int, s_per_s: float):
    Kip, Kii, Cip, Cii = internal_dynamic_blocks(operators, ports)
    A = (Kii + s_per_s * Cii).tocsc()
    B = (Kip + s_per_s * Cip).tocsc()
    exact = np.ascontiguousarray(spla.splu(A).solve(-B.toarray()))
    return exact, A, B


def append_modes(
    V: np.ndarray,
    correction: np.ndarray,
    reference: np.ndarray,
    relative_tolerance: float,
    max_add: int,
    max_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    correction = np.asarray(correction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    reference_norms = np.linalg.norm(reference, axis=0)
    correction_norms = np.linalg.norm(correction, axis=0)
    scale_floor = np.finfo(float).eps * max(
        1.0, float(reference_norms.max(initial=0.0))
    )
    keep = correction_norms > scale_floor
    if not np.any(keep) or V.shape[1] >= max_order:
        return V, np.empty(0)

    block = correction[:, keep] / np.maximum(reference_norms[keep], scale_floor)
    if V.shape[1]:
        block -= V @ (V.T @ block)
        block -= V @ (V.T @ block)
    U, singular_values, _ = scipy.linalg.svd(
        block,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if not singular_values.size or singular_values[0] == 0.0:
        return V, np.empty(0)
    usable = int(
        np.count_nonzero(singular_values >= relative_tolerance * singular_values[0])
    )
    add = min(usable, max_add, max_order - V.shape[1])
    if add <= 0:
        return V, np.empty(0)
    Q, _ = scipy.linalg.qr(
        np.column_stack((V, U[:, :add])), mode="economic", check_finite=False
    )
    return np.ascontiguousarray(Q), singular_values[:add].copy()


def transfer_scan(
    macro: MacroAffine,
    lifting: np.ndarray,
    V: np.ndarray,
    run: Run,
) -> TransferScan:
    ports = macro.ports.port_count
    worst: TransferScan | None = None
    for boundary_index, h in enumerate(validation_boundaries(run)):
        operators = macro.at(h)
        for s in validation_frequencies(run):
            exact, A, B = transfer_map(operators, ports, s)
            if V.shape[1]:
                AV = np.asarray(A @ V)
                reduced_A = np.asarray(V.T @ AV)
                rhs = -np.asarray(V.T @ (A @ lifting + B.toarray()))
                q = scipy.linalg.solve(
                    reduced_A, rhs, assume_a="sym", check_finite=False
                )
                approximation = lifting + V @ q
            else:
                approximation = lifting
            residual_matrix = np.asarray(A @ approximation + B.toarray())
            correction = exact - approximation
            transfer_error = float(
                np.linalg.norm(correction, ord="fro")
                / max(np.linalg.norm(exact, ord="fro"), np.finfo(float).tiny)
            )
            residual = float(
                np.linalg.norm(residual_matrix, ord="fro")
                / max(np.linalg.norm(B.data), np.finfo(float).tiny)
            )
            candidate = TransferScan(
                f"boundary-{boundary_index}/s={s:.6g}",
                transfer_error,
                residual,
                correction,
            )
            if worst is None or transfer_error > worst.transfer_error:
                worst = candidate
    if worst is None:
        raise RuntimeError("no transfer validation point was generated")
    return worst


def build_basis(macro: MacroAffine, run: Run) -> Basis:
    started = time.perf_counter()
    ports = macro.ports.port_count
    lifting, static_residual, unity_residual = static_lifting(macro.base, ports)
    V = np.empty((lifting.shape[0], 0), dtype=np.float64)
    singular_values = []
    snapshot_columns = 0
    snapshot_points = 0
    initial_limit = max(1, run.max_dynamic_order - 2 * run.enrichment_block)

    for h in static_training_boundaries(run):
        exact, _, _ = transfer_map(macro.at(h), ports, 0.0)
        V, added = append_modes(
            V,
            exact - lifting,
            exact,
            run.snapshot_relative_tolerance,
            run.snapshot_modes_per_point,
            initial_limit,
        )
        singular_values.extend(added.tolist())
        snapshot_columns += ports
        snapshot_points += 1

    for h in dynamic_training_boundaries(run):
        operators = macro.at(h)
        for s in run.expansion_points_per_s[1:]:
            exact, _, _ = transfer_map(operators, ports, s)
            V, added = append_modes(
                V,
                exact - lifting,
                exact,
                run.snapshot_relative_tolerance,
                run.snapshot_modes_per_point,
                initial_limit,
            )
            singular_values.extend(added.tolist())
            snapshot_columns += ports
            snapshot_points += 1

    initial_order = V.shape[1]
    transfer_history = []
    residual_history = []
    while True:
        scan = transfer_scan(macro, lifting, V, run)
        transfer_history.append(scan.transfer_error)
        residual_history.append(scan.residual)
        if (
            scan.transfer_error <= run.transfer_error_tolerance
            or V.shape[1] >= run.max_dynamic_order
        ):
            break
        previous = V.shape[1]
        exact = lifting + scan.correction
        V, added = append_modes(
            V,
            scan.correction,
            exact,
            min(run.snapshot_relative_tolerance, 1.0e-10),
            run.enrichment_block,
            run.max_dynamic_order,
        )
        singular_values.extend(added.tolist())
        if V.shape[1] == previous:
            break

    orthogonality = float(
        np.linalg.norm(V.T @ V - np.eye(V.shape[1]), ord="fro")
    )
    return Basis(
        lifting,
        V,
        initial_order,
        V.shape[1],
        snapshot_columns,
        snapshot_points,
        np.asarray(singular_values),
        np.asarray(transfer_history),
        np.asarray(residual_history),
        float(transfer_history[-1]),
        float(residual_history[-1]),
        static_residual,
        unity_residual,
        orthogonality,
        time.perf_counter() - started,
    )


def project_matrix(
    matrix: sp.csc_matrix, ports: int, lifting: np.ndarray, V: np.ndarray
) -> sp.csc_matrix:
    App = matrix[:ports, :ports].toarray()
    Api = matrix[:ports, ports:].tocsc()
    Aip = matrix[ports:, :ports].tocsc()
    Aii = matrix[ports:, ports:].tocsc()
    Aii_lifting = np.asarray(Aii @ lifting)
    Aii_V = np.asarray(Aii @ V)
    port_internal = np.asarray(Aip.toarray() + Aii_lifting)
    A00 = App + np.asarray(Api @ lifting) + lifting.T @ port_internal
    A01 = np.asarray(Api @ V) + lifting.T @ Aii_V
    A10 = V.T @ port_internal
    A11 = V.T @ Aii_V
    reduced = np.block([[A00, A01], [A10, A11]])
    reduced = 0.5 * (reduced + reduced.T)
    return sp.csc_matrix(reduced)


def project(
    operators: Operators, ports: int, lifting: np.ndarray, V: np.ndarray
) -> Operators:
    f_ports = np.asarray(operators.f[:ports], dtype=np.float64)
    f_internal = np.asarray(operators.f[ports:], dtype=np.float64)
    return Operators(
        project_matrix(operators.K, ports, lifting, V),
        project_matrix(operators.C, ports, lifting, V),
        np.r_[f_ports + lifting.T @ f_internal, V.T @ f_internal],
    )


def project_affine(macro: MacroAffine, basis: Basis) -> ReducedAffine:
    started = time.perf_counter()
    ports = macro.ports.port_count
    return ReducedAffine(
        macro.anchor_h,
        project(macro.base, ports, basis.lifting, basis.V),
        tuple(
            project(component, ports, basis.lifting, basis.V)
            for component in macro.components
        ),
        time.perf_counter() - started,
    )


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


def minimum_symmetric_eigenvalue(matrix: sp.csc_matrix) -> float:
    dense = np.asarray(matrix.toarray())
    return float(scipy.linalg.eigvalsh(dense, subset_by_index=(0, 0))[0])


def passivity_metrics(
    reduced: ReducedAffine, boundaries: Sequence[BoundaryCase]
) -> dict[str, float | bool]:
    minimum_k = math.inf
    minimum_c = math.inf
    maximum_asymmetry = 0.0
    scale = 0.0
    for boundary in boundaries:
        operators, _ = reduced.at(boundary.h_W_m2K)
        minimum_k = min(minimum_k, minimum_symmetric_eigenvalue(operators.K))
        minimum_c = min(minimum_c, minimum_symmetric_eigenvalue(operators.C))
        maximum_asymmetry = max(
            maximum_asymmetry,
            float(spla.norm(operators.K - operators.K.T)),
            float(spla.norm(operators.C - operators.C.T)),
        )
        scale = max(scale, float(spla.norm(operators.K)), float(spla.norm(operators.C)))
    tolerance = 1.0e-10 * max(scale, 1.0)
    return {
        "minimum_K_eigenvalue": minimum_k,
        "minimum_C_eigenvalue": minimum_c,
        "maximum_symmetry_defect": maximum_asymmetry,
        "tolerance": tolerance,
        "passed": bool(min(minimum_k, minimum_c) >= -tolerance),
    }


def evaluate(
    data: Data,
    cfg: Package,
    run: Run,
    basis: Basis,
    reduced: ReducedAffine,
    boundary: BoundaryCase,
    ref: Reference,
):
    operators, online_assembly_s = reduced.at(boundary.h_W_m2K)
    initial_ports = np.full(cfg.ports, cfg.ambient_K)
    initial_internal = np.full(basis.lifting.shape[0], cfg.ambient_K)
    initial_q = basis.V.T @ (initial_internal - basis.lifting @ initial_ports)

    def run_solve(transient: bool):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[
            np.full(compiled.cell_count, cfg.ambient_K),
            initial_ports,
            initial_q,
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
        port_state = states[:, detail_n : detail_n + cfg.ports]
        q_state = states[:, detail_n + cfg.ports :]
        recovered[:, data.macro_cells] = (
            port_state @ basis.lifting.T + q_state @ basis.V.T
        )
        return recovered

    recovered_steady = recover(steady_state)[0]
    steady_error = float(np.max(np.abs(recovered_steady - ref.steady)))
    steady_rise = max(
        float(np.max(np.abs(ref.steady - cfg.ambient_K))), np.finfo(float).tiny
    )
    if times.shape != ref.times.shape or not np.allclose(
        times, ref.times, atol=1e-12, rtol=0
    ):
        raise RuntimeError("full and reduced solvers returned different output times")
    recovered_transient = recover(transient_states)
    transient_error = float(np.max(np.abs(recovered_transient - ref.transient)))
    transient_rise = max(
        float(np.max(np.abs(ref.transient - cfg.ambient_K))), np.finfo(float).tiny
    )
    return {
        "name": boundary.name,
        "h_quadrants_W_m2K": list(boundary.h_W_m2K),
        "h_mean_W_m2K": boundary.mean_h,
        "affine_coordinates": [
            value / reduced.anchor_h for value in boundary.h_W_m2K
        ],
        "steady_error_K": steady_error,
        "steady_relative_rise_error": steady_error / steady_rise,
        "transient_error_K": transient_error,
        "transient_relative_rise_error": transient_error / transient_rise,
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
        "reduced_macro_bytes": problem.csc_bytes(operators.K)
        + problem.csc_bytes(operators.C),
    }


def synthetic_macro() -> MacroAffine:
    nx, nz = 12, 14
    ports = nx
    internal = nx * nz
    Kii = sp.lil_matrix((internal, internal), dtype=np.float64)
    for iz in range(nz):
        for ix in range(nx):
            cell = iz * nx + ix
            for dx, dz, conductance in (
                (-1, 0, 2.0),
                (1, 0, 2.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
            ):
                xx, zz = ix + dx, iz + dz
                if 0 <= xx < nx and 0 <= zz < nz:
                    neighbor = zz * nx + xx
                    Kii[cell, cell] += conductance
                    Kii[cell, neighbor] -= conductance
    Kip = sp.lil_matrix((internal, ports), dtype=np.float64)
    Kpp = sp.lil_matrix((ports, ports), dtype=np.float64)
    for ix in range(nx):
        Kip[ix, ix] = -8.0
        Kii[ix, ix] += 8.0
        Kpp[ix, ix] += 8.0
    Kii = Kii.tocsc()
    Kip = Kip.tocsc()
    Kpp = Kpp.tocsc()
    Cii = sp.diags(np.linspace(0.5, 2.0, internal), format="csc")
    zero_pp = sp.csc_matrix((ports, ports))
    zero_pi = sp.csc_matrix((ports, internal))
    base = Operators(
        sp.bmat(((Kpp, Kip.T), (Kip, Kii)), format="csc"),
        sp.bmat(((zero_pp, zero_pi), (zero_pi.T, Cii)), format="csc"),
        np.zeros(ports + internal),
    )
    components = []
    for quadrant in range(4):
        diagonal = np.zeros(ports + internal)
        begin = quadrant * nx // 4
        end = (quadrant + 1) * nx // 4
        for ix in range(begin, end):
            diagonal[ports + (nz - 1) * nx + ix] = 5.0
        components.append(
            Operators(
                sp.diags(diagonal, format="csc"),
                sp.csc_matrix(base.C.shape),
                np.zeros(ports + internal),
            )
        )
    return MacroAffine(
        None,
        SimpleNamespace(port_count=ports),
        1.0,
        base,
        tuple(components),
        0.0,
        0.0,
        0.0,
    )


def algebraic_self_test() -> dict[str, float | int | bool]:
    macro = synthetic_macro()
    run = Run(
        affine_anchor_h=1.0,
        duration_s=1.0,
        dt_s=0.02,
        expansion_points=5,
        snapshot_relative_tolerance=1.0e-9,
        snapshot_modes_per_point=16,
        transfer_error_tolerance=1.0e-5,
        enrichment_block=16,
        max_dynamic_order=160,
    )
    basis = build_basis(macro, run)
    reduced = project_affine(macro, basis)
    base_exact, _, _ = transfer_map(macro.base, macro.ports.port_count, 0.0)
    base_dc_error = float(
        np.linalg.norm(base_exact - basis.lifting, ord="fro")
        / np.linalg.norm(base_exact, ord="fro")
    )
    test_boundaries = (
        BoundaryCase("holdout-a", (0.2, 0.2, 0.2, 0.2)),
        BoundaryCase("holdout-b", (0.55, 1.65, 2.85, 0.32)),
        BoundaryCase("holdout-c", (3.4, 0.25, 0.45, 2.1)),
    )
    passivity = passivity_metrics(reduced, test_boundaries)
    passed = bool(
        base_dc_error < 1.0e-11
        and basis.worst_transfer_error < 1.0e-5
        and basis.final_order < basis.lifting.shape[0]
        and basis.orthogonality_error < 1.0e-10
        and passivity["passed"]
    )
    return {
        "full_internal_order": basis.lifting.shape[0],
        "physical_ports": basis.lifting.shape[1],
        "dynamic_order": basis.final_order,
        "base_dc_transfer_error": base_dc_error,
        "holdout_transfer_error": basis.worst_transfer_error,
        "holdout_residual": basis.worst_residual,
        "orthogonality_error": basis.orthogonality_error,
        "passivity_passed": passivity["passed"],
        "passed": passed,
    }


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
        )
        run = Run(
            error_K=0.15,
            duration_s=0.20,
            expansion_points=4,
            snapshot_relative_tolerance=1.0e-7,
            snapshot_modes_per_point=12,
            transfer_error_tolerance=5.0e-5,
            enrichment_block=32,
            max_dynamic_order=192,
            speedup_target=1.0,
            compression_target=12.0,
        )
        boundaries = (
            BoundaryCase("uniform-low", (500.0,) * 4),
            BoundaryCase("uniform-high", (8000.0,) * 4),
            BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
        )
        return cfg, run, boundaries
    cfg = Package()
    run = Run()
    boundaries = (
        BoundaryCase("uniform-low", (500.0,) * 4),
        BoundaryCase("uniform-medium", (2500.0,) * 4),
        BoundaryCase("uniform-high", (8000.0,) * 4),
        BoundaryCase("x-gradient", (500.0, 8000.0, 500.0, 8000.0)),
        BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
    )
    return cfg, run, boundaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    mode.add_argument("--algebraic-self-test", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.algebraic_self_test:
        result = algebraic_self_test()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 3

    cfg, run, boundaries = configs(args.quick)
    print("=" * 100)
    print("Transient BCI-ROM - affine Craig-Bampton + adaptive all-port rational correction")
    print("=" * 100)
    print(
        f"Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(
        f"Nominal die power={cfg.nominal_power_W:.2f} W; "
        f"tile peak/mean density={cfg.peak_to_mean_tile_density:.2f}x"
    )

    started = time.perf_counter()
    data = problem.assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    try:
        basis = build_basis(data.macro, run)
        reduced = project_affine(data.macro, basis)
        dynamic_compression = basis.lifting.shape[0] / max(basis.final_order, 1)
        state_compression = (
            cfg.ports + basis.lifting.shape[0]
        ) / max(cfg.ports + basis.final_order, 1)
        print(
            f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; exact interface ports={cfg.ports} "
            f"({cfg.port_shape[0]}x{cfg.port_shape[1]}); macro dynamic internal "
            f"{basis.lifting.shape[0]:,}->{basis.final_order:,} "
            f"({dynamic_compression:.2f}x), total macro state compression="
            f"{state_compression:.2f}x"
        )
        print(
            f"Static lifting columns={cfg.ports} (non-state); snapshot points/columns="
            f"{basis.snapshot_points}/{basis.snapshot_columns}; dynamic order "
            f"{basis.initial_order}->{basis.final_order}"
        )
        print(
            "Greedy transfer error history="
            + ", ".join(f"{value:.3e}" for value in basis.transfer_error_history)
            + "; residual history="
            + ", ".join(f"{value:.3e}" for value in basis.residual_history)
        )
        passivity = passivity_metrics(reduced, boundaries)

        results = []
        offline_s = assembly_s + basis.seconds + reduced.seconds
        for boundary in boundaries:
            result = evaluate(
                data,
                cfg,
                run,
                basis,
                reduced,
                boundary,
                problem.reference(cfg, run, boundary),
            )
            result["accuracy_passed"] = (
                max(result["steady_error_K"], result["transient_error_K"])
                <= run.error_K
            )
            result["speedup_passed"] = (
                result["transient_speedup"] >= run.speedup_target
                if args.strict
                else True
            )
            result["passed"] = result["accuracy_passed"] and result["speedup_passed"]
            result["rom_offline_s"] = offline_s
            results.append(result)
            print(
                f"{boundary.name:>16s}: h={boundary.h_W_m2K}; error steady/transient="
                f"{result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K; "
                f"full/ROM transient={result['full_transient_solve_s']:.3f}/"
                f"{result['reduced_transient_solve_s']:.3f}s, "
                f"speedup={result['transient_speedup']:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = dynamic_compression >= run.compression_target
        report = {
            "schema_version": 13,
            "mode": "quick" if args.quick else "strict",
            "method": (
                "exact-port affine BCI-DtN with Craig-Bampton static constraint "
                "lifting, all-port rational correction modes and greedy transfer "
                "error enrichment"
            ),
            "package": {
                **asdict(cfg),
                "nx": cfg.nx,
                "ny": cfg.ny,
                "ports": cfg.ports,
                "port_shape": list(cfg.port_shape),
                "nominal_power_W": cfg.nominal_power_W,
                "power_map_normalized": POWER_MAP.tolist(),
                "peak_to_mean_tile_density": cfg.peak_to_mean_tile_density,
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "experiment": {**asdict(run), "report": str(run.report)},
            "affine_boundary": {
                "family": "A(h)=A0+sum_q(h_q/anchor)*DeltaA_q",
                "regions": ["southwest", "southeast", "northwest", "northeast"],
                "anchor_h_W_m2K": run.affine_anchor_h,
                "full_order_offline_assemblies": 5,
                "full_order_online_assemblies_per_case": 0,
                "capacitance_relative_change": data.macro.c_relative_change,
                "ambient_consistency_residual": data.macro.ambient_residual,
                "extraction_s": data.macro.seconds,
                "projection_s": reduced.seconds,
            },
            "reduction": {
                "transformation": "T=[[I,0],[L,V]]",
                "physical_ports": cfg.ports,
                "full_internal_order": basis.lifting.shape[0],
                "static_lifting_columns_non_state": basis.lifting.shape[1],
                "initial_dynamic_order": basis.initial_order,
                "reduced_dynamic_order": basis.final_order,
                "dynamic_compression_ratio": dynamic_compression,
                "total_macro_state_compression_ratio": state_compression,
                "compression_target": run.compression_target,
                "compression_passed": compression_passed,
                "lifting_bytes": int(basis.lifting.nbytes),
                "dynamic_basis_bytes": int(basis.V.nbytes),
                "snapshot_points": basis.snapshot_points,
                "snapshot_columns": basis.snapshot_columns,
                "all_physical_ports_trained": True,
                "expansion_points_per_s": list(run.expansion_points_per_s),
                "added_singular_value_head": basis.added_singular_values[:20].tolist(),
                "transfer_error_history": basis.transfer_error_history.tolist(),
                "residual_history": basis.residual_history.tolist(),
                "worst_validation_transfer_error": basis.worst_transfer_error,
                "transfer_error_target": run.transfer_error_tolerance,
                "worst_validation_residual": basis.worst_residual,
                "static_constraint_residual": basis.static_residual,
                "uniform_lifting_residual": basis.unity_residual,
                "orthogonality_error": basis.orthogonality_error,
            },
            "passivity": passivity,
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(item["passed"] for item in results)
                and compression_passed
                and passivity["passed"]
                and basis.worst_transfer_error <= run.transfer_error_tolerance
            ),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            f"Passivity min eig(K/C)={passivity['minimum_K_eigenvalue']:.3e}/"
            f"{passivity['minimum_C_eigenvalue']:.3e}; "
            f"{'PASS' if passivity['passed'] else 'FAIL'}"
        )
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        problem.close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
