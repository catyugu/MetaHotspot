"""Check a fixed thermal component: static lift and rational enrichment."""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from assembly import (
    BOTTOM_HTC,
    CHIP_RECTS_MM,
    DEFAULT_MAX_CELL_SIZE,
    FIXED_HTC,
    LATERAL_SIZE,
    LAYER_SPECS,
    MATERIAL_SCALING,
    MM,
    assemble,
)
from cms import BLOCK_SIZE, extract, factorize, patterns, port_inputs
from transient import simulate
from visualization import plot_connections, write_outputs

ROOT = Path(__file__).resolve().parent
SAMPLES = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CHANNELS = ("interface", "cross", "power")


def require(name, value, tolerance=1e-9):
    if not np.isfinite(value) or value > tolerance:
        raise RuntimeError(f"{name}: error {value:.3e} exceeds {tolerance:.1e}")


def relative(actual, expected):
    return float(
        la.norm(actual - expected) / max(la.norm(expected), np.finfo(float).tiny)
    )


def validate_model(model):
    """Check conservation, source units and the fixed Robin boundary."""
    require("stiffness symmetry", spl.norm(model.K - model.K.T) / spl.norm(model.K))
    expected_capacity = model.cell_volume @ model.cell_rhoc
    require("total heat capacity", abs(model.capacity.sum() / expected_capacity - 1.0))
    require("unit power sources", float(np.max(np.abs(model.B_P.sum(axis=0) - 1.0))))
    require(
        "interface area",
        abs(
            model.D_gamma.sum()
            / (sum((c - a) * (d - b) for a, b, c, d in CHIP_RECTS_MM) * MM**2)
            - 1.0
        ),
    )
    losses = model.K @ np.ones(model.N)
    require("conduction conservation", relative(losses, model.robin))
    expected_loss = LATERAL_SIZE**2 * BOTTOM_HTC
    require("Robin heat loss", abs(losses.sum() / expected_loss - 1.0))
    for axis, maximum in zip(model.axes, model.max_cell_size):
        require("maximum cell size", max(0.0, float(np.diff(axis).max() / maximum - 1)))
    if not np.array_equal(np.unique(model.cells), np.arange(model.N)):
        raise RuntimeError("unused DOFs survived active-cell compaction")
    occupied_volume = sum(
        s[1] * (LATERAL_SIZE**2 if i == 0 else model.D_gamma.sum())
        for i, s in enumerate(LAYER_SPECS)
    )
    require("occupied volume", abs(model.cell_volume.sum() / occupied_volume - 1.0))
    if np.any(model.capacity <= 0.0):
        raise RuntimeError("heat capacities must be positive")


def reference_response(model, inputs, shift):
    """Q^T (K+sC)^(-1) Q, evaluated directly on the requested load space.

    Linearity makes this identical to forming the full interface response and
    compressing it afterwards. One sparse factorization per sample; bounded RHS
    batches also keep the optional all-interface audit from allocating N x m.
    """
    factor = factorize(model.K + sp.diags(shift * model.capacity, format="csc"))
    response = np.empty((inputs.shape[1], inputs.shape[1]))
    for first in range(0, inputs.shape[1], BLOCK_SIZE):
        cols = slice(first, min(first + BLOCK_SIZE, inputs.shape[1]))
        response[:, cols] = inputs.T @ factor.solve(inputs[:, cols].toarray())
    return response


def reduced_response(capacity, stiffness, inputs, shift):
    factor = la.cho_factor(stiffness + shift * capacity)
    return inputs.T @ la.cho_solve(factor, inputs)


def blocks(response, interface_count):
    k = interface_count
    return (response[:k, :k], response[:k, k:], response[k:, k:])


def evaluate(model, capacity, stiffness, inputs, reduced_inputs, ends):
    """Stream reference samples, reusing each one for all nested spaces."""
    k = inputs.shape[1] - model.B_P.shape[1]
    errors = np.empty((len(ends), len(SAMPLES), len(CHANNELS)))
    reference_seconds, evaluation_seconds = 0.0, 0.0
    for index, shift in enumerate(SAMPLES):
        started = time.perf_counter()
        reference = reference_response(model, inputs, shift)
        reference_seconds += time.perf_counter() - started
        if shift == 0.0:
            scales = np.array([la.norm(b) for b in blocks(reference, k)])
        started = time.perf_counter()
        for level, count in enumerate(ends):
            order = model.m + count
            reduced = reduced_response(
                capacity[:order, :order],
                stiffness[:order, :order],
                reduced_inputs[:order],
                shift,
            )
            errors[level, index] = [
                la.norm(b) for b in blocks(reduced - reference, k)
            ] / scales
        evaluation_seconds += time.perf_counter() - started
    # Source DC is exact because A^-1 B_power is in the zero-shift block.
    require("steady transfer", float(errors[:, 0].max()))
    for channel in (0, 2):
        increase = np.diff(errors[:, :, channel], axis=0)
        require(
            "nested collocated convergence", max(0.0, float(increase.max(initial=0.0)))
        )
    return errors, reference_seconds, evaluation_seconds


def study(model, patches, epsilon, full_interface):
    validate_model(model)
    started = time.perf_counter()
    basis = extract(model, patches=patches, epsilon=epsilon)
    extraction_seconds = time.perf_counter() - started
    require("static lift residual", basis.lift_residual)
    require("zero-shift moment", basis.zero_moment_error)

    started = time.perf_counter()
    capacity, stiffness = basis.project()
    la.cholesky(capacity)
    la.cholesky(stiffness)
    m = model.m
    require(
        "capacity orthogonality",
        relative(capacity[m:, m:], np.eye(capacity.shape[0] - m)),
    )
    inputs = port_inputs(model, patterns(model, patches))
    reduced_inputs = basis.project_inputs(inputs)
    projection_seconds = time.perf_counter() - started
    errors, reference_seconds, evaluation_seconds = evaluate(
        model, capacity, stiffness, inputs, reduced_inputs, basis.ends
    )
    report = {
        "material": model.material,
        "max_cell_size_m": model.max_cell_size,
        "full_order": model.N,
        "interface_dofs": m,
        "patches": patches,
        "shift_tolerance": epsilon,
        "spectrum_per_s": basis.spectrum,
        "matching_shifts_per_s": basis.shifts.tolist(),
        "sample_shifts_per_s": SAMPLES,
        "reference_loads": inputs.shape[1],
        "seconds": {
            "extraction": extraction_seconds,
            "projection": projection_seconds,
            "reference": reference_seconds,
            "evaluation": evaluation_seconds,
        },
        "checks": {
            "lift_residual": basis.lift_residual,
            "zero_moment_error": basis.zero_moment_error,
        },
        "convergence": [
            {
                "interior_directions": count,
                "max_error": dict(zip(CHANNELS, values.max(axis=0).tolist())),
            }
            for count, values in zip(basis.ends, errors)
        ],
        "final_sample_errors": dict(zip(CHANNELS, errors[-1].T.tolist())),
    }
    if full_interface:
        full_inputs = port_inputs(model)
        audit, ref_s, eval_s = evaluate(
            model,
            capacity,
            stiffness,
            full_inputs,
            basis.project_inputs(full_inputs),
            [basis.ends[-1]],
        )
        report["all_interface_audit"] = {
            "reference_loads": full_inputs.shape[1],
            "max_error": dict(zip(CHANNELS, audit[0].max(axis=0).tolist())),
            "reference_seconds": ref_s,
            "evaluation_seconds": eval_s,
        }
    return report, errors, basis, capacity, stiffness


def make_figure(report, errors, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    counts = [row["interior_directions"] for row in report["convergence"]]
    for index, name in enumerate(CHANNELS):
        axes[0].semilogy(
            counts, np.maximum(errors[:, :, index].max(axis=1), 1e-16), ".-", label=name
        )
        axes[1].plot(SAMPLES, np.maximum(errors[-1, :, index], 1e-16), ".-", label=name)
    axes[0].set_xlabel("interior directions")
    axes[0].set_title("Convergence over the sampled points")
    axes[1].set_xscale("symlog", linthresh=0.1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("real Laplace sample [1/s], including DC")
    axes[1].set_title("Final model on the patch load space")
    for axis in axes:
        axis.set_ylabel("Frobenius error / corresponding DC block norm")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    fig.suptitle(
        f"{report['material']}: all {report['interface_dofs']} interface DOFs retained"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dx-mm",
        type=float,
        default=DEFAULT_MAX_CELL_SIZE[0] / MM,
        help="maximum x cell size [mm]",
    )
    parser.add_argument(
        "--dy-mm",
        type=float,
        default=DEFAULT_MAX_CELL_SIZE[1] / MM,
        help="maximum y cell size [mm]",
    )
    parser.add_argument(
        "--dz-mm",
        type=float,
        default=DEFAULT_MAX_CELL_SIZE[2] / MM,
        help="maximum z cell size [mm]",
    )
    parser.add_argument(
        "--materials", nargs="+", choices=list(MATERIAL_SCALING), default=["isotropic"]
    )
    parser.add_argument("--patches", type=int, default=2)
    parser.add_argument(
        "--shift-tolerance",
        type=float,
        default=1e-3,
        help="elliptic sampling budget; not an achieved ROM error bound",
    )
    parser.add_argument(
        "--full-interface",
        action="store_true",
        help="also check every interface flux direction on the final model",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.005,
        help="maximum backward-Euler step [s]; jumps restart at 20 us",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    sizes = np.array([args.dx_mm, args.dy_mm, args.dz_mm]) * MM
    if np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        parser.error("dx/dy/dz must be finite and positive")
    if args.patches < 1:
        parser.error("patches must be positive")
    if not 0.0 < args.shift_tolerance < 1.0:
        parser.error("shift tolerance must lie strictly between zero and one")
    if not np.isfinite(args.time_step) or args.time_step < 1e-8:
        parser.error("time step must be finite and at least 1e-8 s")
    args.output.mkdir(parents=True, exist_ok=True)
    for material in args.materials:
        model = assemble(tuple(sizes), material)
        print(
            f"{material}: {model.N} temperatures, {model.m} interface DOFs", flush=True
        )
        report, errors, basis, capacity, stiffness = study(
            model, args.patches, args.shift_tolerance, args.full_interface
        )
        print("  Simulating switched powers and exporting fields...", flush=True)
        transient, fields = simulate(model, basis, capacity, stiffness, args.time_step)
        require(
            "transient energy balance",
            transient["transient_energy_balance_relative"],
            1e-7,
        )
        require(
            "steady power balance", transient["steady_power_balance_relative"], 1e-9
        )
        require("steady field reconstruction", transient["steady_field_error_K"], 1e-7)
        report["transient"] = transient
        connected, connected_fields = simulate(
            model,
            basis,
            capacity,
            stiffness,
            args.time_step,
            cooling_scale=0.5,
        )
        require(
            "coupled transient energy balance",
            connected["transient_energy_balance_relative"],
            1e-7,
        )
        require(
            "coupled steady power balance",
            connected["steady_power_balance_relative"],
            1e-9,
        )
        require(
            "coupled steady reconstruction", connected["steady_field_error_K"], 1e-7
        )
        report["reduced_cooling_transient"] = connected
        print(
            "  Full TIM/lid, half cooling: "
            f"peak {connected['snapshot_peak_FOM_C']:.2f} °C; "
            f"max probe error {connected['max_probe_error_K']:.3e} K",
            flush=True,
        )
        write_outputs(
            args.output / material / "reduced_cooling", model, connected_fields
        )
        plot_connections(
            fields,
            connected_fields,
            args.output / material / "external_connections.png",
        )
        report["package"] = {
            "lateral_size_m": LATERAL_SIZE,
            "layers": LAYER_SPECS,
            "layer_z_cells": model.layer_cells,
            "actual_max_cell_size_m": [float(np.diff(a).max()) for a in model.axes],
            "external_lid_top_htc_W_m2K": FIXED_HTC,
            "substrate_bottom_htc_W_m2K": BOTTOM_HTC,
            "cell_count": len(model.cells),
            "capacity_J_K": float(model.capacity.sum()),
            "chip_edges_mesh_aligned": True,
            "actual_lateral_cells": [len(axis) - 1 for axis in model.axes[:2]],
            "empty_cell_count": int(
                (len(model.axes[0]) - 1)
                * (len(model.axes[1]) - 1)
                * (len(model.axes[2]) - 1)
                - len(model.cells)
            ),
            "interface": "two complete silicon top faces",
        }
        write_outputs(args.output / material, model, fields)
        stem = args.output / material
        stem.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        make_figure(report, errors, stem.with_suffix(".png"))
        print(
            f"  Peak stored FOM: {transient['snapshot_peak_FOM_C']:.2f} °C; "
            f"steady FOM: {transient['steady_peak_FOM_C']:.2f} °C; "
            f"max probe error: {transient['max_probe_error_K']:.3e} K; "
            f"snapshot field error: {transient['max_snapshot_field_error_K']:.3e} K",
            flush=True,
        )
        final = report["convergence"][-1]
        print(
            f"  Interior directions: {final['interior_directions']}; "
            f"sampled errors: "
            + ", ".join(f"{n}={v:.2e}" for n, v in final["max_error"].items())
        )
        print(
            "  Time: "
            + ", ".join(f"{n} {v:.2f}s" for n, v in report["seconds"].items())
        )
        if args.full_interface:
            print(
                "  All-interface audit: "
                + ", ".join(
                    f"{n}={v:.2e}"
                    for n, v in report["all_interface_audit"]["max_error"].items()
                )
            )
    print(f"Checks passed. Results: {args.output}")


if __name__ == "__main__":
    main()
