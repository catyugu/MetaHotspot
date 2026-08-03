#!/usr/bin/env python3
"""Run the column-localized thermal macromodel reduction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macromodel.experiment_setup import *


@dataclass(frozen=True)
class Config(BaseConfig):
    local_dynamic_modes: int = 2
    bdf1_shifts: tuple[float, ...] = (1.0, 2.0)
    report: Path = Path("results/bci_rom_uniform_convection_results.json")


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
    local_dynamic_modes=1,
    bdf1_shifts=(1.0,),
    speedup_target=1.0,
    compression_target=2.0,
)


def column_basis(
    compiled,
    cfg: Config,
    base: Operators,
    delta: Operators,
    port_count: int,
):
    """Build a block-local basis without source or boundary-response snapshots."""
    started = time.perf_counter()
    K_ip = base.K[port_count:, :port_count].tocsc()
    K_ii = base.K[port_count:, port_count:].tocsc()
    C_ip = base.C[port_count:, :port_count].tocsc()
    C_ii = base.C[port_count:, port_count:].tocsc()
    dK_ip = delta.K[port_count:, :port_count].tocsc()
    dK_ii = delta.K[port_count:, port_count:].tocsc()

    port_lookup = {
        (int(ix), int(iy)): port
        for port, (ix, iy) in enumerate(
            (ix, iy) for ix in cfg.port_indices for iy in cfg.port_indices
        )
    }
    grid = grid_cells(compiled)
    seen_ports = 0
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    orders: list[int] = []
    offset = 0

    for ix in range(compiled.nx):
        for iy in range(compiled.ny):
            cells = grid[ix, iy]
            cells = cells[cells >= 0].astype(np.int64)
            if not cells.size:
                continue

            k = K_ii[cells][:, cells].toarray()
            c = C_ii[cells][:, cells].toarray()
            candidates = [np.ones(cells.size)]

            mode_count = min(cfg.local_dynamic_modes, cells.size)
            if mode_count:
                eigenvalues, modes = scipy.linalg.eigh(
                    k,
                    c,
                    subset_by_index=(0, mode_count - 1),
                    check_finite=False,
                )
                cutoff = math.pi / cfg.dt_s
                candidates.extend(modes[:, eigenvalues <= cutoff].T)

            port = port_lookup.get((ix, iy))
            if port is not None:
                seen_ports += 1
                b = K_ip[cells, port].toarray().ravel()
                cp = C_ip[cells, port].toarray().ravel()
                static = scipy.linalg.solve(k, -b, assume_a="sym", check_finite=False)
                candidates.append(static)

                sensitivity_rhs = (
                    dK_ii[cells][:, cells] @ static
                    + dK_ip[cells, port].toarray().ravel()
                )
                if np.linalg.norm(sensitivity_rhs) > 1.0e-14 * max(
                    np.linalg.norm(b), 1.0
                ):
                    candidates.append(
                        scipy.linalg.solve(
                            k,
                            -sensitivity_rhs,
                            assume_a="sym",
                            check_finite=False,
                        )
                    )

                for multiplier in cfg.bdf1_shifts:
                    shift = multiplier / cfg.dt_s
                    response = scipy.linalg.solve(
                        k + shift * c,
                        -(b + shift * cp),
                        assume_a="sym",
                        check_finite=False,
                    )
                    candidates.append(response - static)

            matrix = np.column_stack(candidates)
            q, r, _ = scipy.linalg.qr(
                matrix, mode="economic", pivoting=True, check_finite=False
            )
            diagonal = np.abs(np.diag(r))
            if not diagonal.size or diagonal[0] == 0.0:
                local = np.empty((cells.size, 0))
            else:
                keep = diagonal > (
                    np.finfo(float).eps * max(matrix.shape) * diagonal[0]
                )
                local = np.ascontiguousarray(q[:, keep])

            orders.append(local.shape[1])
            for local_row, cell in enumerate(cells):
                nonzero = np.flatnonzero(np.abs(local[local_row]) > 1.0e-14)
                rows.extend([int(cell)] * nonzero.size)
                cols.extend((offset + nonzero).tolist())
                values.extend(local[local_row, nonzero].tolist())
            offset += local.shape[1]

    if seen_ports != port_count:
        raise RuntimeError("interface-port/column mapping is inconsistent")

    W = sp.csc_matrix((values, (rows, cols)), shape=(K_ii.shape[0], offset))
    ones = np.ones(W.shape[0])
    if np.linalg.norm(W @ (W.T @ ones) - ones) > 1.0e-10 * math.sqrt(ones.size):
        raise RuntimeError("macro basis does not preserve uniform temperature")
    if spla.norm(W.T @ W - sp.eye(W.shape[1], format="csc")) > 1.0e-10:
        raise RuntimeError("macro basis lost orthogonality")

    initial = np.asarray(W.T @ np.full(W.shape[0], cfg.ambient_K)).ravel()
    return W, np.asarray(orders), initial, time.perf_counter() - started


def project_operators(operators: Operators, ports: int, W: sp.csc_matrix) -> Operators:
    def project_matrix(matrix):
        reduced = sp.bmat(
            (
                (
                    matrix[:ports, :ports],
                    matrix[:ports, ports:] @ W,
                ),
                (
                    W.T @ matrix[ports:, :ports],
                    W.T @ matrix[ports:, ports:] @ W,
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
        np.r_[
            operators.f[:ports],
            np.asarray(W.T @ operators.f[ports:]).ravel(),
        ],
    )


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

        W, orders, initial_internal, basis_s = column_basis(
            macro_compiled, cfg, base, delta, port_count
        )
        projection_started = time.perf_counter()
        reduced_base = project_operators(base, port_count, W)
        reduced_delta = project_operators(delta, port_count, W)
        projection_s = time.perf_counter() - projection_started
        offline_s = time.perf_counter() - offline_started

        full_macro_order = port_count + W.shape[0]
        reduced_macro_order = port_count + W.shape[1]
        compression = full_macro_order / reduced_macro_order
        print(
            f"Grid {cfg.nx}x{cfg.nx}x{cfg.nz}; exact ports={port_count}; "
            f"macro states {full_macro_order:,}->{reduced_macro_order:,} "
            f"({compression:.2f}x)"
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
                    initial_internal,
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
                temperature[:, macro_to_full] = (
                    W @ states[:, detail_n + port_count :].T
                ).T
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
                f"h={convection_h:g} W/(m^2 K): {format_accuracy(accuracy)}; "
                f"full/ROM={full_transient_s:.3f}/{reduced_transient_s:.3f}s, "
                f"speedup={speedup:.2f}x "
                f"{'PASS' if result['passed'] else 'FAIL'}"
            )

        compression_passed = compression >= cfg.compression_target
        return {
            "schema_version": 19,
            "method": "exact-port affine-convection column-local Galerkin BCI-ROM",
            "configuration": {
                **asdict(cfg),
                "report": str(cfg.report),
                "nx": cfg.nx,
                "ny": cfg.nx,
                "nz": cfg.nz,
                "ports": cfg.ports,
                "port_shape": [cfg.port_indices.size, cfg.port_indices.size],
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
                "compression_ratio": compression,
                "compression_target": cfg.compression_target,
                "compression_passed": compression_passed,
                "column_count": int(orders.size),
                "port_columns": port_count,
                "local_order_min": int(orders.min()),
                "local_order_mean": float(orders.mean()),
                "local_order_max": int(orders.max()),
                "basis_nnz": W.nnz,
                "basis_extraction_s": basis_s,
            },
            "passivity": {
                "preserved_structurally": True,
                "reason": "symmetric Galerkin congruence with exact ports",
            },
            "offline_s": offline_s,
            "boundary_reuse": results,
            "passed": bool(
                all(result["passed"] for result in results) and compression_passed
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
    print("Transient BCI-ROM extraction - uniform cold-plate convection")
    print("=" * 96)
    print(
        "Footprints cold plate/spreader/substrate/bump/die/TIM="
        f"{cfg.cold_plate_size_mm:g}/{cfg.spreader_size_mm:g}/"
        f"{cfg.substrate_size_mm:g}/{cfg.bump_region_size_mm:g}/"
        f"{cfg.die_size_mm:g}/{cfg.tim_size_mm:g} mm"
    )
    print(
        f"Nominal die power={cfg.nominal_power_W:.2f} W; "
        f"tile peak/mean density={POWER_MAP.max():.2f}x"
    )

    report = run_experiment(cfg, BOUNDARIES, args.strict)
    report["mode"] = "quick" if args.quick else "strict"
    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    cfg.report.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Report: {cfg.report}")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
