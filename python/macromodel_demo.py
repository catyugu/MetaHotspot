#!/usr/bin/env python3
"""Sparse irregular-column transient BCI-ROM benchmark."""
from __future__ import annotations

import argparse, json, math, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _macromodel_problem as problem
from metahotspot.compiled import Operators
from metahotspot.macromodel import solve as solve_macro

BoundaryCase = problem.BoundaryCase
Package = problem.Package


@dataclass(frozen=True)
class Run:
    error_K: float = 0.25
    duration_s: float = 0.5
    dt_s: float = 0.025
    affine_anchor_h: float = 2500.0
    local_dynamic_modes: int = 2
    bdf1_shifts: tuple[float, ...] = (1.0, 2.0)
    residual_block_size: int = 32
    speedup_target: float = 1.5
    compression_target: float = 2.5
    report: Path = Path("results/bci_rom_sparse_column_results.json")

    @property
    def modal_cutoff_per_s(self):
        return math.pi / self.dt_s


class Column(NamedTuple):
    cells: np.ndarray
    port: int | None


class Basis(NamedTuple):
    W: sp.csc_matrix
    column_count: int
    port_columns: int
    orders: np.ndarray
    projected_residuals: dict[str, float]
    local_residuals: tuple[float, float, float]
    ambient_error: float
    orthogonality_error: float
    seconds: float


@dataclass(frozen=True)
class ReducedAffine:
    anchor_h: float
    base: Operators
    components: tuple[Operators, ...]
    seconds: float

    def at(self, h_values):
        started = time.perf_counter()
        h = np.asarray(h_values, dtype=float)
        op = problem.combine_many(self.base, self.components, h / self.anchor_h)
        return op, time.perf_counter() - started


def blocks(op, ports):
    return (
        op.K[ports:, :ports].tocsc(),
        op.K[ports:, ports:].tocsc(),
        op.C[ports:, :ports].tocsc(),
        op.C[ports:, ports:].tocsc(),
    )


def columns(compiled, cfg):
    port_pairs = [
        (int(ix), int(iy))
        for ix in cfg.port_x_indices
        for iy in cfg.port_y_indices
    ]
    port_lookup = {pair: index for index, pair in enumerate(port_pairs)}
    result = []
    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = []
            for iz in range(compiled.nz):
                cell = int(compiled.grid_to_cell[(ix * compiled.ny + iy) * compiled.nz + iz])
                if cell >= 0:
                    cells.append(cell)
            if cells:
                result.append(Column(np.asarray(cells, dtype=np.int64), port_lookup.get((ix, iy))))
    if sum(item.port is not None for item in result) != cfg.ports:
        raise RuntimeError("interface-port/column mapping is inconsistent")
    return tuple(result)


def range_basis(matrix):
    q, r, _ = scipy.linalg.qr(
        np.asarray(matrix, dtype=float), mode="economic", pivoting=True, check_finite=False
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0:
        return np.empty((matrix.shape[0], 0))
    keep = diagonal > np.finfo(float).eps * max(matrix.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def vector_residual(A, x, b):
    return float(np.linalg.norm(A @ x + b) / max(np.linalg.norm(b), np.finfo(float).tiny))


def projected_residual(op, ports, W, block_size):
    Kip, Kii, _, _ = blocks(op, ports)
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


def residual_boundaries(run):
    a = run.affine_anchor_h
    return {
        "base": (0.0,) * 4,
        "anchor": (a,) * 4,
        "high": (3.2 * a,) * 4,
        "skew": (3.2 * a, 0.28 * a, 0.48 * a, 2.4 * a),
    }


def build_basis(macro, cfg, run):
    started = time.perf_counter()
    ports = macro.ports.port_count
    Kip0, Kii0, Cip0, Cii0 = blocks(macro.base, ports)
    component_blocks = [blocks(item, ports) for item in macro.components]
    macro_columns = columns(macro.compiled, cfg)
    rows, cols, values, orders = [], [], [], []
    static_errors, parameter_errors, bdf_errors = [], [], []
    offset = 0

    for column in macro_columns:
        cells = column.cells
        k0 = Kii0[cells, :][:, cells].toarray()
        c0 = Cii0[cells, :][:, cells].toarray()
        candidates = [np.ones(cells.size)]
        eigenvalues, eigenvectors = scipy.linalg.eigh(k0, c0, check_finite=False)
        dynamic = np.flatnonzero(eigenvalues <= run.modal_cutoff_per_s)[: run.local_dynamic_modes]
        candidates.extend(eigenvectors[:, index] for index in dynamic)

        if column.port is not None:
            port = column.port
            b0 = Kip0[cells, port].toarray().ravel()
            cp0 = Cip0[cells, port].toarray().ravel()
            static = -scipy.linalg.solve(k0, b0, assume_a="sym", check_finite=False)
            candidates.append(static)
            static_errors.append(vector_residual(k0, static, b0))
            for Kip1, Kii1, _, _ in component_blocks:
                k1 = Kii1[cells, :][:, cells].toarray()
                b1 = Kip1[cells, port].toarray().ravel()
                rhs = k1 @ static + b1
                if np.linalg.norm(rhs) > 1e-14 * max(np.linalg.norm(b0), 1.0):
                    sensitivity = -scipy.linalg.solve(k0, rhs, assume_a="sym", check_finite=False)
                    candidates.append(sensitivity)
                    parameter_errors.append(vector_residual(k0, sensitivity, rhs))
            for multiplier in run.bdf1_shifts:
                shift = multiplier / run.dt_s
                A = k0 + shift * c0
                rhs = b0 + shift * cp0
                response = -scipy.linalg.solve(A, rhs, assume_a="sym", check_finite=False)
                candidates.append(response - static)
                bdf_errors.append(vector_residual(A, response, rhs))

        local = range_basis(np.column_stack(candidates))
        orders.append(local.shape[1])
        for local_row, cell in enumerate(cells):
            nz = np.flatnonzero(np.abs(local[local_row]) > 1e-14)
            rows.extend([int(cell)] * nz.size)
            cols.extend((offset + nz).tolist())
            values.extend(local[local_row, nz].tolist())
        offset += local.shape[1]

    W = sp.csc_matrix((values, (rows, cols)), shape=(Kii0.shape[0], offset))
    ones = np.ones(W.shape[0])
    ambient_error = float(np.linalg.norm(W @ (W.T @ ones) - ones) / math.sqrt(ones.size))
    residuals = {
        name: projected_residual(macro.at(h), ports, W, run.residual_block_size)
        for name, h in residual_boundaries(run).items()
    }
    return Basis(
        W,
        len(macro_columns),
        sum(item.port is not None for item in macro_columns),
        np.asarray(orders),
        residuals,
        (
            max(static_errors, default=0.0),
            max(parameter_errors, default=0.0),
            max(bdf_errors, default=0.0),
        ),
        ambient_error,
        float(spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc"))),
        time.perf_counter() - started,
    )


def project(op, ports, W):
    T = sp.block_diag((sp.eye(ports, format="csc"), W), format="csc")
    def matrix(A):
        value = (T.T @ A @ T).tocsc()
        value = (0.5 * (value + value.T)).tocsc()
        value.eliminate_zeros()
        return value
    return Operators(matrix(op.K), matrix(op.C), np.asarray(T.T @ op.f).ravel())


def project_affine(macro, basis):
    started = time.perf_counter()
    ports = macro.ports.port_count
    return ReducedAffine(
        macro.anchor_h,
        project(macro.base, ports, basis.W),
        tuple(project(item, ports, basis.W) for item in macro.components),
        time.perf_counter() - started,
    )


def evaluate(data, cfg, run, basis, reduced, boundary, reference):
    op, online_s = reduced.at(boundary.h_W_m2K)
    internal0 = np.asarray(basis.W.T @ np.full(basis.W.shape[0], cfg.ambient_K)).ravel()

    def run_solve(transient):
        compiled = data.detail_transient if transient else data.detail_steady
        ports = data.detail_ports_transient if transient else data.detail_ports_steady
        state = np.r_[np.full(compiled.cell_count + cfg.ports, cfg.ambient_K), internal0]
        started = time.perf_counter()
        with solve_macro(compiled, op, ports, state, problem.solve_options(run, transient)) as solution:
            elapsed = time.perf_counter() - started
            if transient:
                return np.asarray(solution.history_times), np.asarray(solution.state_history), elapsed
            return np.asarray(solution.state), elapsed

    steady_state, steady_s = run_solve(False)
    times, transient_states, transient_s = run_solve(True)
    detail_n = data.detail_steady.cell_count

    def recover(states):
        states = np.atleast_2d(states)
        result = np.empty((states.shape[0], data.full_layout.cell_count))
        result[:, data.detail_cells] = states[:, :detail_n]
        result[:, data.macro_cells] = (basis.W @ states[:, detail_n + cfg.ports :].T).T
        return result

    if times.shape != reference.times.shape or not np.allclose(times, reference.times, atol=1e-12, rtol=0):
        raise RuntimeError("full and reduced output times differ")
    steady_diff = np.abs(recover(steady_state)[0] - reference.steady)
    transient_diff = np.abs(recover(transient_states) - reference.transient)
    return {
        "name": boundary.name,
        "h_quadrants_W_m2K": list(boundary.h_W_m2K),
        "steady_error_K": float(steady_diff.max()),
        "transient_error_K": float(transient_diff.max()),
        "detail_transient_error_K": float(transient_diff[:, data.detail_cells].max()),
        "macro_transient_error_K": float(transient_diff[:, data.macro_cells].max()),
        "full_transient_solve_s": reference.transient_solve_s,
        "reduced_transient_solve_s": transient_s,
        "transient_speedup": reference.transient_solve_s / max(transient_s, np.finfo(float).tiny),
        "online_reduced_assembly_s": online_s,
        "reduced_macro_k_nnz": int(op.K.nnz),
        "reduced_macro_c_nnz": int(op.C.nnz),
    }


def configs(quick):
    if quick:
        cfg = Package(
            substrate_cells=3, bump_cells=1, die_cells=2, tim_cells=1,
            spreader_cells=3, cold_plate_cells=4, max_xy_cell_mm=6.0,
            bump_rows=8, bump_columns=8,
        )
        run = Run(
            error_K=0.35, duration_s=0.20, local_dynamic_modes=1,
            bdf1_shifts=(1.0,), speedup_target=1.0, compression_target=2.0,
        )
        cases = (
            BoundaryCase("uniform-low", (500.0,) * 4),
            BoundaryCase("uniform-high", (8000.0,) * 4),
            BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
        )
        return cfg, run, cases
    cfg, run = Package(), Run()
    cases = (
        BoundaryCase("uniform-low", (500.0,) * 4),
        BoundaryCase("uniform-medium", (2500.0,) * 4),
        BoundaryCase("uniform-high", (8000.0,) * 4),
        BoundaryCase("x-gradient", (500.0, 8000.0, 500.0, 8000.0)),
        BoundaryCase("diagonal-skew", (8000.0, 700.0, 1200.0, 6000.0)),
    )
    return cfg, run, cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--strict", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg, run, boundaries = configs(args.quick)
    print("=" * 100)
    print("Transient BCI-ROM - sparse irregular-column local thermal basis")
    print("=" * 100)
    print(
        f"Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/{cfg.substrate_size_mm:g}/"
        f"{cfg.bump_region_size_mm:g}/{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(f"Nominal die power={cfg.nominal_power_W:.2f} W; tile peak/mean density={cfg.peak_to_mean_tile_density:.2f}x")

    started = time.perf_counter()
    data = problem.assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    try:
        basis = build_basis(data.macro, cfg, run)
        reduced = project_affine(data.macro, basis)
        full_order = cfg.ports + basis.W.shape[0]
        reduced_order = cfg.ports + basis.W.shape[1]
        compression = full_order / reduced_order
        print(
            f"Grid {cfg.nx}x{cfg.ny}x{cfg.nz}; exact ports={cfg.ports}; "
            f"macro states {full_order:,}->{reduced_order:,} ({compression:.2f}x)"
        )
        print(
            f"Columns={basis.column_count} (port/non-port={basis.port_columns}/"
            f"{basis.column_count-basis.port_columns}); local order min/mean/max="
            f"{basis.orders.min()}/{basis.orders.mean():.2f}/{basis.orders.max()}"
        )
        print("Projected residuals=" + ", ".join(f"{k}:{v:.3e}" for k, v in basis.projected_residuals.items()))
        print(
            f"Local residual static/parameter/BDF1={basis.local_residuals[0]:.3e}/"
            f"{basis.local_residuals[1]:.3e}/{basis.local_residuals[2]:.3e}; "
            f"ambient={basis.ambient_error:.3e}"
        )

        results = []
        offline_s = assembly_s + basis.seconds + reduced.seconds
        for boundary in boundaries:
            result = evaluate(data, cfg, run, basis, reduced, boundary, problem.reference(cfg, run, boundary))
            accuracy = max(result["steady_error_K"], result["transient_error_K"]) <= run.error_K
            speed = result["transient_speedup"] >= run.speedup_target if args.strict else True
            result.update(accuracy_passed=accuracy, speedup_passed=speed, passed=accuracy and speed)
            results.append(result)
            print(
                f"{boundary.name:>16s}: error steady/transient="
                f"{result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K "
                f"[detail {result['detail_transient_error_K']:.5f}, macro {result['macro_transient_error_K']:.5f}]; "
                f"full/ROM={result['full_transient_solve_s']:.3f}/{result['reduced_transient_solve_s']:.3f}s, "
                f"speedup={result['transient_speedup']:.2f}x {'PASS' if result['passed'] else 'FAIL'}"
            )

        report = {
            "schema_version": 14,
            "mode": "quick" if args.quick else "strict",
            "method": "sparse irregular-column static/affine/BDF1/local-mode Galerkin ROM",
            "package": {**asdict(cfg), "nx": cfg.nx, "ny": cfg.ny, "ports": cfg.ports},
            "experiment": {**asdict(run), "report": str(run.report)},
            "reduction": {
                "full_macro_order": full_order,
                "reduced_macro_order": reduced_order,
                "compression_ratio": compression,
                "basis_nnz": basis.W.nnz,
                "basis_density": basis.W.nnz / max(1, basis.W.shape[0] * basis.W.shape[1]),
                "projected_residuals": basis.projected_residuals,
                "ambient_reconstruction_error": basis.ambient_error,
                "orthogonality_error": basis.orthogonality_error,
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(all(item["passed"] for item in results) and compression >= run.compression_target),
        }
        run.report.parent.mkdir(parents=True, exist_ok=True)
        run.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print("Passivity: preserved structurally by symmetric Galerkin congruence")
        print(f"Report: {run.report}")
        return 0 if report["passed"] else 3
    finally:
        problem.close_data(data)


if __name__ == "__main__":
    raise SystemExit(main())
