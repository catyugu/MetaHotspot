#!/usr/bin/env python3
"""Run the tangential rational Krylov thermal macromodel reduction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macromodel.experiment_setup import *


@dataclass(frozen=True)
class Config(BaseConfig):
    krylov_parameter_samples: int = 3
    krylov_frequency_samples: int = 6
    krylov_residual_tolerance: float = 2.0e-3
    krylov_block_size: int = 16
    krylov_max_order: int = 512
    report: Path = Path("results/bci_rom_parametric_krylov_results.json")


QUICK_OVERRIDES = dict(
    substrate_cells=3,
    bump_cells=1,
    die_cells=2,
    tim_cells=1,
    spreader_cells=3,
    cold_plate_cells=4,
    max_xy_cell_mm=6.0,
    bump_rows=8,
    bump_columns=8,
    krylov_frequency_samples=4,
    krylov_residual_tolerance=1.0e-2,
    krylov_block_size=24,
    krylov_max_order=384,
    speedup_target=1.0,
    compression_target=2.0,
)


def symmetric_dense(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def eigenpairs_descending(matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-negative eigenpairs of a symmetric Gram matrix."""
    values, vectors = scipy.linalg.eigh(
        symmetric_dense(matrix),
        check_finite=False,
    )
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


def training_points(cfg: Config, boundaries):
    h_min = min(map(float, boundaries))
    h_max = max(map(float, boundaries))
    h_values = np.geomspace(h_min, h_max, cfg.krylov_parameter_samples)
    h_values = np.unique(np.r_[h_values, cfg.affine_anchor_h])

    low = 1.0 / cfg.duration_s
    high = 2.0 / cfg.dt_s
    interior_count = max(0, cfg.krylov_frequency_samples - 2)
    interior = (
        np.geomspace(low, high, interior_count + 2)[1:-1]
        if interior_count
        else np.empty(0)
    )
    shifts = np.unique(np.r_[0.0, low, interior, high])
    return h_values, shifts


def internal_blocks(operators: Operators, ports: int):
    return (
        operators.K[ports:, ports:].tocsc(),
        operators.C[ports:, ports:].tocsc(),
        operators.K[ports:, :ports].tocsc(),
        operators.C[ports:, :ports].tocsc(),
    )


def orthonormalize_block(basis: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Remove the current space and return an orthonormal independent block."""
    block = np.asarray(vectors, dtype=np.float64).copy()
    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)
    q, r, _ = scipy.linalg.qr(
        block,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def build_krylov_basis(
    cfg: Config,
    boundaries,
    base: Operators,
    delta: Operators,
    ports: int,
):
    """Build a global basis by block residual-greedy rational interpolation.

    At each training pair (s, h), the exact internal response block is

        X(s, h) = -(K_ii(h) + s C_ii(h))^-1
                    (K_ip(h) + s C_ip(h)).

    Errors are normalized separately at every training point.  The worst point
    contributes several dominant tangential error directions per enrichment,
    rather than one direction at a time.
    """
    started = time.perf_counter()
    K0, C0, B0, D0 = internal_blocks(base, ports)
    K1, C1, B1, D1 = internal_blocks(delta, ports)
    h_values, shifts = training_points(cfg, boundaries)
    candidates = []

    for h_value in h_values:
        mu = float(h_value / cfg.affine_anchor_h)
        for shift in shifts:
            A = (K0 + mu * K1 + shift * (C0 + mu * C1)).tocsc()
            A = (0.5 * (A + A.T)).tocsc()
            B = (B0 + mu * B1 + shift * (D0 + mu * D1)).tocsc()
            response = np.asarray(spla.splu(A).solve(-B.toarray()))
            gram = symmetric_dense(response.T @ (A @ response))
            reference_values, _ = eigenpairs_descending(gram)
            candidates.append(
                {
                    "h_W_m2K": float(h_value),
                    "shift_per_s": float(shift),
                    "A": A,
                    "B": B,
                    "response": response,
                    "reference_eigenvalue": max(
                        float(reference_values[0]), np.finfo(float).tiny
                    ),
                }
            )

    internal_order = K0.shape[0]
    max_order = min(cfg.krylov_max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    converged = False

    while True:
        best = None
        for candidate in candidates:
            if basis.shape[1]:
                reduced_A = symmetric_dense(basis.T @ (candidate["A"] @ basis))
                reduced_B = basis.T @ candidate["B"]
                reduced_response = scipy.linalg.solve(
                    reduced_A,
                    -reduced_B,
                    assume_a="sym",
                    check_finite=False,
                )
                error_response = candidate["response"] - basis @ reduced_response
            else:
                error_response = candidate["response"]

            error_gram = symmetric_dense(
                error_response.T @ (candidate["A"] @ error_response)
            )
            error_values, tangents = eigenpairs_descending(error_gram)
            score = math.sqrt(
                float(error_values[0]) / candidate["reference_eigenvalue"]
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "candidate": candidate,
                    "error_response": error_response,
                    "error_values": error_values,
                    "tangents": tangents,
                }

        entry = {
            "order": basis.shape[1],
            "relative_response_error": float(best["score"]),
            "h_W_m2K": best["candidate"]["h_W_m2K"],
            "shift_per_s": best["candidate"]["shift_per_s"],
            "added_directions": 0,
        }
        history.append(entry)
        if best["score"] <= cfg.krylov_residual_tolerance:
            converged = True
            break
        if basis.shape[1] >= max_order:
            break

        relative_directions = np.sqrt(
            best["error_values"] / best["candidate"]["reference_eigenvalue"]
        )
        requested = int(
            np.count_nonzero(relative_directions > cfg.krylov_residual_tolerance)
        )
        count = min(
            max(1, requested),
            cfg.krylov_block_size,
            max_order - basis.shape[1],
        )
        vectors = best["error_response"] @ best["tangents"][:, :count]
        block = orthonormalize_block(basis, vectors)
        if not block.shape[1]:
            raise RuntimeError("rational Krylov block enrichment stalled")
        remaining = max_order - basis.shape[1]
        block = block[:, :remaining]
        basis = np.column_stack((basis, block))
        entry["added_directions"] = int(block.shape[1])

    orthogonality_error = np.linalg.norm(
        basis.T @ basis - np.eye(basis.shape[1]), ord=2
    )
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    summary = {
        "parameter_samples_W_m2K": h_values.tolist(),
        "frequency_shifts_per_s": shifts.tolist(),
        "candidate_count": len(candidates),
        "full_port_tangential_search": True,
        "error_normalization": "relative at each parameter-frequency point",
        "block_size": cfg.krylov_block_size,
        "basis_order": basis.shape[1],
        "maximum_order": max_order,
        "orthogonality_error": orthogonality_error,
        "relative_response_error": history[-1]["relative_response_error"],
        "residual_tolerance": cfg.krylov_residual_tolerance,
        "converged": converged,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    return basis, summary


def project_operators(
    operators: Operators,
    ports: int,
    basis: np.ndarray,
    ambient_K: float,
) -> Operators:
    """Project internal temperature rise while retaining absolute port states."""
    internal_offset = np.full(operators.K.shape[0] - ports, ambient_K)
    shifted_f = np.asarray(
        operators.f - operators.K[:, ports:] @ internal_offset
    ).ravel()

    def project_matrix(matrix):
        reduced = sp.bmat(
            (
                (
                    matrix[:ports, :ports].tocsc(),
                    sp.csc_matrix(matrix[:ports, ports:] @ basis),
                ),
                (
                    sp.csc_matrix(basis.T @ matrix[ports:, :ports]),
                    sp.csc_matrix(basis.T @ matrix[ports:, ports:] @ basis),
                ),
            ),
            format="csc",
        )
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    return Operators(
        project_matrix(operators.K),
        project_matrix(operators.C),
        np.r_[shifted_f[:ports], basis.T @ shifted_f[ports:]],
    )


def verify_ambient_balance(
    operators: Operators,
    ports: int,
    reduced_order: int,
    ambient_K: float,
    label: str,
) -> None:
    state = np.r_[np.full(ports, ambient_K), np.zeros(reduced_order)]
    defect = np.asarray(operators.K @ state - operators.f).ravel()
    scale = max(
        np.linalg.norm(operators.K @ state),
        np.linalg.norm(operators.f),
        1.0,
    )
    if np.linalg.norm(defect) > 1.0e-10 * scale:
        raise RuntimeError(f"{label} reduced operator violates ambient balance")


def run_experiment(cfg: Config, boundaries, strict: bool) -> dict:
    offline_started = time.perf_counter()
    with ExitStack() as stack:
        full_layout = build_model(cfg, Study.STEADY, detail=True, macro=True).compile()
        stack.callback(full_layout.close)
        detail_steady = build_model(
            cfg, Study.STEADY, detail=True, macro=False
        ).compile()
        stack.callback(detail_steady.close)
        detail_transient = build_model(
            cfg, Study.TRANSIENT, detail=True, macro=False
        ).compile()
        stack.callback(detail_transient.close)

        detail_patches = port_patches(cfg, Face.ZP, cfg.detail_height_mm * 1.0e-3)
        detail_ports_steady = PortMap(detail_steady, detail_patches)
        stack.callback(detail_ports_steady.close)
        detail_ports_transient = PortMap(detail_transient, detail_patches)
        stack.callback(detail_ports_transient.close)

        macro_started = time.perf_counter()
        macro_compiled = build_model(
            cfg, Study.STEADY, detail=False, macro=True
        ).compile()
        stack.callback(macro_compiled.close)
        macro_ports = PortMap(macro_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(macro_ports.close)
        base = normalized_operators(*macro_ports.assemble())

        anchor_compiled = build_model(
            cfg,
            Study.STEADY,
            detail=False,
            macro=True,
            convection_h=cfg.affine_anchor_h,
        ).compile()
        stack.callback(anchor_compiled.close)
        anchor_ports = PortMap(anchor_compiled, port_patches(cfg, Face.ZM, 0.0))
        stack.callback(anchor_ports.close)
        anchor = normalized_operators(*anchor_ports.assemble())
        if anchor.K.shape != base.K.shape:
            raise RuntimeError("convection changed macro state ordering")
        delta = normalized_operators(
            anchor.K - base.K,
            anchor.C - base.C,
            np.asarray(anchor.f) - base.f,
        )
        macro_extraction_s = time.perf_counter() - macro_started

        ambient = np.full(base.K.shape[0], cfg.ambient_K)
        balance_error = np.linalg.norm(delta.K @ ambient - delta.f)
        balance_scale = max(np.linalg.norm(delta.f), np.finfo(float).tiny)
        if spla.norm(delta.C) > 1.0e-11 * max(spla.norm(base.C), 1.0):
            raise RuntimeError("convection unexpectedly changed macro capacitance")
        if balance_error > 1.0e-10 * balance_scale:
            raise RuntimeError("affine convection component violates ambient balance")

        port_count = macro_ports.port_count
        if port_count != cfg.ports:
            raise RuntimeError("configured interface port count is inconsistent")

        detail_to_full = coordinate_map(detail_steady, full_layout, 0, "detail/full")
        transient_to_full = coordinate_map(
            detail_transient, full_layout, 0, "transient/full"
        )
        if not np.array_equal(detail_to_full, transient_to_full):
            raise RuntimeError("steady and transient detail orderings differ")
        macro_to_full = coordinate_map(
            macro_compiled, full_layout, cfg.detail_nz, "macro/full"
        )
        combined = np.r_[detail_to_full, macro_to_full]
        if (
            combined.size != full_layout.cell_count
            or np.unique(combined).size != combined.size
        ):
            raise RuntimeError("detail and macro maps do not partition the full model")

        basis, basis_summary = build_krylov_basis(
            cfg, boundaries, base, delta, port_count
        )
        if not basis_summary["converged"]:
            raise RuntimeError(
                "Krylov extraction did not converge: "
                f"order={basis_summary['basis_order']}, "
                "worst relative response error="
                f"{basis_summary['relative_response_error']:.3e}, "
                f"target={basis_summary['residual_tolerance']:.3e}"
            )
        projection_started = time.perf_counter()
        reduced_base = project_operators(base, port_count, basis, cfg.ambient_K)
        reduced_delta = project_operators(delta, port_count, basis, cfg.ambient_K)
        projection_s = time.perf_counter() - projection_started
        verify_ambient_balance(
            reduced_base,
            port_count,
            basis.shape[1],
            cfg.ambient_K,
            "base",
        )
        verify_ambient_balance(
            reduced_delta,
            port_count,
            basis.shape[1],
            cfg.ambient_K,
            "convection increment",
        )
        offline_s = time.perf_counter() - offline_started

        full_macro_order = port_count + basis.shape[0]
        reduced_macro_order = port_count + basis.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={port_count}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
            f"({compression:.2f}x); Krylov residual="
            f"{basis_summary['relative_response_error']:.3e}"
        )

        results = []
        detail_n = detail_steady.cell_count
        for convection_h in boundaries:
            reference = full_reference(cfg, convection_h)
            (
                reference_steady,
                reference_times,
                reference_history,
                full_compile_s,
                full_steady_s,
                full_transient_s,
                full_order,
            ) = reference

            assembly_started = time.perf_counter()
            reduced = affine_operators(
                reduced_base,
                reduced_delta,
                convection_h / cfg.affine_anchor_h,
            )
            online_assembly_s = time.perf_counter() - assembly_started

            def solve_reduced(transient: bool):
                compiled = detail_transient if transient else detail_steady
                ports = detail_ports_transient if transient else detail_ports_steady
                state = np.r_[
                    np.full(compiled.cell_count + port_count, cfg.ambient_K),
                    np.zeros(basis.shape[1]),
                ]
                started = time.perf_counter()
                with solve_macro(
                    compiled,
                    reduced,
                    ports,
                    state,
                    solve_options(cfg, transient),
                ) as solution:
                    elapsed = time.perf_counter() - started
                    if transient:
                        return (
                            np.asarray(solution.history_times).copy(),
                            np.asarray(solution.state_history).copy(),
                            elapsed,
                        )
                    return np.asarray(solution.state).copy(), elapsed

            steady_state, reduced_steady_s = solve_reduced(False)
            times, transient_states, reduced_transient_s = solve_reduced(True)
            if times.shape != reference_times.shape or not np.allclose(
                times, reference_times, atol=1.0e-12, rtol=0.0
            ):
                raise RuntimeError("full and reduced output times differ")

            def recover(states):
                states = np.atleast_2d(states)
                temperature = np.empty((states.shape[0], full_layout.cell_count))
                temperature[:, detail_to_full] = states[:, :detail_n]
                reduced_internal = states[:, detail_n + port_count :]
                temperature[:, macro_to_full] = (
                    cfg.ambient_K + (basis @ reduced_internal.T).T
                )
                return temperature

            accuracy = accuracy_summary(
                reference_steady,
                recover(steady_state)[0],
                reference_history,
                recover(transient_states),
                cfg.ambient_K,
            )
            speedup = full_transient_s / max(reduced_transient_s, np.finfo(float).tiny)
            speedup_passed = speedup >= cfg.speedup_target if strict else True
            result = {
                "h_W_m2K": convection_h,
                **accuracy,
                "online_reduced_assembly_s": online_assembly_s,
                "full_compile_s": full_compile_s,
                "full_steady_solve_s": full_steady_s,
                "reduced_steady_solve_s": reduced_steady_s,
                "full_transient_solve_s": full_transient_s,
                "reduced_transient_solve_s": reduced_transient_s,
                "transient_speedup": speedup,
                "full_order": full_order,
                "reduced_online_order": detail_n + reduced.K.shape[0],
                "reduced_macro_k_nnz": reduced.K.nnz,
                "reduced_macro_c_nnz": reduced.C.nnz,
                "speedup_passed": speedup_passed,
                "passed": accuracy["accuracy_passed"] and speedup_passed,
            }
            results.append(result)
            print(
                f"h={convection_h:g} W/(m^2 K): {format_accuracy(accuracy)}; full/ROM="
                f"{full_transient_s:.3f}/{reduced_transient_s:.3f}s, "
                f"speedup={speedup:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= cfg.compression_target
        basis_passed = basis_summary["converged"]
        return {
            "schema_version": 21,
            "method": (
                "exact-port affine-parametric adaptive tangential rational "
                "Krylov BCI-ROM"
            ),
            "configuration": {
                **asdict(cfg),
                "report": str(cfg.report),
                "nx": cfg.nx,
                "ny": cfg.nx,
                "nz": cfg.nz,
                "ports": cfg.ports,
                "port_shape": [
                    cfg.port_indices.size,
                    cfg.port_indices.size,
                ],
                "nominal_power_W": cfg.nominal_power_W,
                "power_map_normalized": POWER_MAP.tolist(),
                "chiplet_power_scale": list(CHIPLET_POWER_SCALE),
            },
            "affine_boundary": {
                "family": "A(h)=A0+(h/anchor_h)*DeltaA_h",
                "region": "entire cold-plate top surface",
                "anchor_h_W_m2K": cfg.affine_anchor_h,
                "full_order_offline_assemblies": 2,
                "full_order_online_assemblies_per_case": 0,
                "extraction_s": macro_extraction_s,
                "projection_s": projection_s,
            },
            "reduction": {
                "full_macro_order": full_macro_order,
                "reduced_macro_order": reduced_macro_order,
                "internal_full_order": basis.shape[0],
                "internal_reduced_order": basis.shape[1],
                "compression_ratio": compression,
                "compression_target": cfg.compression_target,
                "compression_passed": compression_passed,
                "basis_dense": True,
                "temperature_coordinates": (
                    "absolute physical port temperatures and internal "
                    "temperature rise above ambient"
                ),
                "krylov": basis_summary,
                "basis_passed": basis_passed,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence with exact ports",
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(result["passed"] for result in results)
                and compression_passed
                and basis_passed
            ),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="small smoke experiment")
    mode.add_argument("--strict", action="store_true", help="full benchmark gates")
    args = parser.parse_args(argv)

    cfg = replace(Config(), **QUICK_OVERRIDES) if args.quick else Config()

    print("=" * 96)
    print("Transient BCI-ROM extraction - adaptive parametric rational Krylov")
    print("=" * 96)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )

    report = run_experiment(cfg, BOUNDARIES, args.strict)
    report["mode"] = "quick" if args.quick else "strict"
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {cfg.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
