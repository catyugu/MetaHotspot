#!/usr/bin/env python3
"""Audit the stock Extended BCI FANTASTIC basis against full-field responses.

This is a fixed-real-shift audit, not a new extractor or a dynamic guarantee.
The certificate bounds the squared relative A-energy error for every input
combination. Direct field solves independently check the bound on cell interiors.
All large solves use AMG-CG. Certificate and reference RHS costs are reported
separately and included in the total audit cost.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.linalg as la

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

from certified_box import BoxCertificate, logarithmic_edges  # noqa: E402
from sparse_solve import AmgSolver  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import build_parametric_basis  # noqa: E402


def field_errors(operator, source, basis):
    """Measure all-input field energy and the induced l2-input/linf-field error.

    The reference denominator is X.T A X. G must have independent columns.
    Use the direct field error Gram, avoiding subtraction of two close transfers.
    Node error has units K per unit input; it is a measured value, not an energy
    certificate or a componentwise relative temperature claim.
    """
    reference = AmgSolver(operator).solve(source)
    coefficients = la.solve(basis.T @ (operator @ basis), basis.T @ source,
                            assume_a="pos", check_finite=False)
    approximation = basis @ coefficients
    error = reference - approximation
    gram = error.T @ (operator @ error)
    weight = reference.T @ (operator @ reference)
    gram = 0.5 * (gram + gram.T)
    weight = 0.5 * (weight + weight.T)
    squared = float(la.eigvalsh(gram, weight, check_finite=False)[-1])
    defect = source.T @ error
    scale = max(float(la.norm(weight)), np.finfo(float).tiny)
    return {
        "relative_energy_error": float(np.sqrt(max(squared, 0.0))),
        "squared_relative_energy_error": max(squared, 0.0),
        "max_absolute_field_error_per_unit_input": float(np.max(la.norm(error, axis=1))),
        "galerkin_identity_relative_gap": float(la.norm(defect - gram) / scale),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_mm", nargs="?", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--htc-ranges", nargs=4, type=float,
                        metavar=("H1_MIN", "H1_MAX", "H2_MIN", "H2_MAX"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = Case1Model(Case1Config(max_xy_cell_mm=args.mesh_mm,
                                  max_z_cell_mm=args.mesh_mm))
    kernel, mass = model.core.K.tocsc(), model.core.C.tocsc()
    source = np.asarray(model.source_shape, dtype=float)
    terms = [term.tocsc() for term in model.boundary_terms]
    ranges = np.asarray(model.h_ranges() if args.htc_ranges is None
                        else np.reshape(args.htc_ranges, (2, 2)), dtype=float)
    if not 0 < args.tolerance < 1:
        parser.error("tolerance must lie in (0, 1)")
    if np.any(ranges <= 0) or np.any(ranges[:, 1] < ranges[:, 0]):
        parser.error("HTC intervals must have positive, ordered endpoints")

    # Fixed audit resolution, unrelated to the construction/stopping of a MOR
    # method. No candidate grid, enrichment cap, or separate compression knob.
    cells, order, blocks, samples = 8, 3, 4, 2
    report = {
        "target": "full-field transfer operator",
        "shift_per_s": 0.0,
        "mesh_mm": args.mesh_mm,
        "cells": int(kernel.shape[0]),
        "tolerance": args.tolerance,
        "htc_ranges": ranges.tolist(),
        "audit_resolution": {"cells_per_axis": cells, "trial_order": order,
                             "anchor_blocks_per_axis": blocks,
                             "samples_per_cell": samples},
        "floating_point_certified": False,
        "dynamic_certified": False,
        "baselines": {},
    }
    for seed in (20260805, 7):
        started = time.perf_counter()
        basis, stats = build_parametric_basis(model.core, source, terms, ranges,
                                             tolerance=args.tolerance, seed=seed)
        extraction_seconds = time.perf_counter() - started
        counts = {"certificate": 0, "reference": 0}
        phase = "certificate"
        original_column = AmgSolver._column

        def counted_column(solver, rhs):
            counts[phase] += 1
            return original_column(solver, rhs)

        certificate = BoxCertificate(kernel, terms, source, ranges, basis, blocks=blocks)
        measured = []
        edges = [logarithmic_edges(low, high, cells) for low, high in ranges]
        rng = np.random.default_rng(20260930)
        with patch.object(AmgSolver, "_column", counted_column):
            for index in itertools.product(range(cells), repeat=len(ranges)):
                low = np.array([edges[axis][i] for axis, i in enumerate(index)])
                high = np.array([edges[axis][i + 1] for axis, i in enumerate(index)])
                phase = "certificate"
                gram = certificate.anchor_gram(
                    certificate.block_lower(certificate.block_index(low)))
                weight, _ = certificate.cell_weight(high, "reduced")
                bound = float(certificate.cell_bound(gram, low, high, weight, order))
                # Corners are adversarial diagnostics. Interiors are independent
                # of the stock extractor's frozen random stream.
                points = [low, high] + [np.exp(rng.uniform(np.log(low), np.log(high)))
                                        for _ in range(samples)]
                phase = "reference"
                for parameter in points:
                    operator = kernel.copy()
                    for value, term in zip(parameter, terms):
                        operator = operator + value * term
                    metrics = field_errors(operator, source, basis)
                    if metrics["squared_relative_energy_error"] > bound + 1e-9:
                        raise RuntimeError(f"field certificate violated at {parameter}")
                    measured.append({"htc": parameter.tolist(),
                                     "certified_squared_energy_bound": bound, **metrics})

        extraction_rhs = int(stats["pre_svd_order"])
        record = {
            "stock_settings": stats,
            "basis_order": int(basis.shape[1]),
            "extraction_seconds": extraction_seconds,
            "extraction_rhs": extraction_rhs,
            "certificate_rhs": counts["certificate"],
            "reference_rhs": counts["reference"],
            "total_rhs": extraction_rhs + sum(counts.values()),
            "cost_scope": "source extraction plus certificate and reference inverse actions; "
                          "spectral matvecs and AMG cycles are not RHS solves",
            "audit_seconds": time.perf_counter() - started - extraction_seconds,
            "certified_relative_energy_bound": float(np.sqrt(max(
                point["certified_squared_energy_bound"] for point in measured))),
            "sampled_relative_energy_error": max(p["relative_energy_error"] for p in measured),
            "sampled_max_absolute_field_error_per_unit_input": max(
                p["max_absolute_field_error_per_unit_input"] for p in measured),
            "sampled_galerkin_identity_relative_gap": max(
                p["galerkin_identity_relative_gap"] for p in measured),
            "samples": measured,
        }
        report["baselines"][str(seed)] = record
        print(f"stock {seed}: order={basis.shape[1]}, extraction RHS={extraction_rhs}, "
              f"certificate RHS={counts['certificate']}, reference RHS={counts['reference']}, "
              f"field energy={record['sampled_relative_energy_error']:.6g}, "
              f"bound={record['certified_relative_energy_bound']:.6g}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
