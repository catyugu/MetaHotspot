#!/usr/bin/env python3
"""CLI entry point for the sparse transient BCI-ROM DtN benchmark."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

from bci_rom_model import *
from bci_rom_reduction import *


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
    return (Package(), Run())


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
        f"Grid: {cfg.nx} x {cfg.ny} x {cfg.nz} = {cfg.nx * cfg.ny * cfg.nz:,} full cells; detail z={cfg.detail_nz}, macro z={cfg.macro_nz}, exact ports={cfg.ports}"
    )
    started = time.perf_counter()
    data = assemble(cfg, run)
    assembly_s = time.perf_counter() - started
    training_sample = next((sample for sample in data.samples if sample.h is None))
    basis = build_basis(training_sample, run)
    basis_density = basis.W.nnz / max(1, basis.W.shape[0] * basis.W.shape[1])
    print(
        f"Basis: internal {basis.W.shape[0]:,} -> {basis.W.shape[1]:,}; column order min/mean/max={basis.column_orders.min()}/{basis.column_orders.mean():.2f}/{basis.column_orders.max()}; nnz={basis.W.nnz:,}, density={basis_density:.3e}"
    )
    print(
        f"Extraction: {basis.seconds:.3f}s; local static residual={basis.local_static_residual:.3e}; projected static transfer residual={basis.projected_static_residual:.3e}; orthogonality={basis.orthogonality_error:.3e}"
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
            f"h={h:7.1f}: error steady/transient={result['steady_error_K']:.5f}/{result['transient_error_K']:.5f} K; transient full/ROM={result['full_transient_solve_s']:.3f}/{result['reduced_transient_solve_s']:.3f}s, speedup={result['transient_speedup']:.2f}x {('PASS' if result['passed'] else 'FAIL')}"
        )
    nominal = next((item for item in boundary if item["h_W_m2K"] == run.nominal_h))
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
        f"Online order: {nominal['full_operator_order']:,} -> {nominal['reduced_online_order']:,} ({nominal['reduced_online_order'] / nominal['full_operator_order']:.3f}x); K nnz full/ROM-upper={nominal['full_operator_k_nnz']:,}/{coupled_k_nnz_upper:,}"
    )
    print(
        f"Operator memory full/ROM-estimate={nominal['full_operator_bytes'] / 2 ** 20:.2f}/{coupled_bytes_estimate / 2 ** 20:.2f} MiB; nominal break-even={nominal['offline_break_even_transient_runs']} transient runs"
    )
    report = {
        "schema_version": 8,
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
        "passed": bool(all((item["passed"] for item in boundary))),
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
