#!/usr/bin/env python3
"""Paper-aligned whole-PoP Extended-FANTASTIC / contour-element experiment.

The 2023 THERMINIC paper reports one BCI CTM for the *entire* PoP, with two
independent die heat sources, four lateral boundary surfaces, one bottom and one
top surface. This experiment follows that reduction problem directly instead
of reducing only the bottom package and attaching unreduced package/board
models.

The paper does not publish enough geometry/material detail to reproduce its
561,408-DoF commercial detailed model exactly. The structured FVM model below
is therefore an explicitly documented surrogate constrained by the paper's
published topology, source count, HTC ranges, contour degrees and tolerance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from mor_common import (
    Domain,
    FacePort,
    SourceBox,
    build_domain,
    contour_trace_coefficients,
    refined_edges,
    write_domains_vtu,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
from metahotspot.macromodel.utils import (  # noqa: E402
    build_parametric_basis,
    normalized_operators,
    spd_solve,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pop"
AMBIENT_K = 300.0
PAPER_TOLERANCE = 1.0e-2
PAPER_DETAILED_DOF = 561_408
PAPER_BCI_ORDER = 42
PAPER_FULL_BOUNDARY_ROWS = 41_208
PAPER_CONTOUR_ROWS_PRINTED = 110

SIDE_HTC_RANGE = (0.1, 200.0)
BOTTOM_HTC_RANGE = (0.1, 1.0e3)
TOP_HTC_RANGE = (0.1, 1.0e4)
FACE_DEGREES = {
    "x-": 2,
    "x+": 2,
    "y-": 2,
    "y+": 2,
    "z-": 4,
    "z+": 10,
}
SIDE_FACES = ("x-", "x+", "y-", "y+")
BOTTOM_FACE = "z-"
TOP_FACE = "z+"

# Material values are surrogate inputs, not values published in the 2023 paper.
MAT = {
    "mold": (0.84, 1900.0, 1000.0, 1),
    "si": (140.0, 2330.0, 700.0, 2),
    "substrate": (5.0, 1900.0, 1000.0, 3),
    "die_attach": (0.7, 1500.0, 1000.0, 4),
    "solder": (50.0, 7400.0, 230.0, 5),
}


@dataclass(frozen=True)
class BoundaryCase:
    side_h: float
    bottom_h: float
    top_h: float
    in_declared_range: bool = True


CASES = {
    # Fig. 2 and Fig. 4 captions in the 2023 paper.
    "paper_fig2": BoundaryCase(20.0, 500.0, 200.0),
    "paper_fig4": BoundaryCase(20.0, 200.0, 50.0),
    # Declared extraction-domain corners.
    "range_low": BoundaryCase(0.1, 0.1, 0.1),
    "range_high": BoundaryCase(200.0, 1.0e3, 1.0e4),
    # Fig. 3 prints 5,000 W/m2K on the sides, outside the paper's stated
    # 0.1-200 W/m2K extraction range; keep it only as an extrapolation stress.
    "paper_fig3_extrapolation": BoundaryCase(5.0e3, 100.0, 200.0, False),
}


def _inside_square(x: float, y: float, half: float) -> bool:
    return abs(x) < half and abs(y) < half


def _is_bump(
    x: float,
    y: float,
    *,
    pitch: float = 2.0e-3,
    half: float = 0.30e-3,
) -> bool:
    """Simple cube-like BGA array used only to enrich the surrogate geometry."""
    gx = round(x / pitch) * pitch
    gy = round(y / pitch) * pitch
    if abs(gx) > 13.0e-3 or abs(gy) > 13.0e-3:
        return False
    # Leave a central keep-out around the dies.
    if abs(gx) < 5.0e-3 and abs(gy) < 5.0e-3:
        return False
    return abs(x - gx) < half and abs(y - gy) < half


def build_whole_pop(nxy: int = 43, nz: int = 36) -> Domain:
    """Build one connected whole-PoP surrogate with two independent die sources."""
    x = refined_edges(
        -0.015,
        0.015,
        nxy,
        [-0.014, -0.008, -0.0045, 0.0, 0.0045, 0.008, 0.014],
    )
    y = x.copy()
    z = refined_edges(
        0.0,
        3.0e-3,
        nz,
        [
            0.25e-3,
            0.35e-3,
            0.55e-3,
            0.75e-3,
            1.05e-3,
            1.25e-3,
            1.35e-3,
            1.60e-3,
        ],
    )

    def material(xx, yy, zz):
        # Lower package active footprint is intentionally smaller than the upper
        # package footprint, as stated by the paper; the outer cuboid remains
        # filled by low-k encapsulant so the six external contour surfaces are
        # well-defined rectangles.
        if zz < 0.25e-3:
            return (
                MAT["substrate"] if _inside_square(xx, yy, 8.0e-3) else MAT["mold"]
            )
        if zz < 0.35e-3:
            return (
                MAT["die_attach"]
                if _inside_square(xx, yy, 4.5e-3)
                else MAT["mold"]
            )
        if zz < 0.55e-3:
            if _inside_square(xx, yy, 4.5e-3):
                return MAT["si"]
            return (
                MAT["die_attach"]
                if _inside_square(xx, yy, 8.0e-3)
                else MAT["mold"]
            )
        if zz < 0.75e-3:
            return MAT["mold"]
        if zz < 1.05e-3:
            return MAT["solder"] if _is_bump(xx, yy) else MAT["die_attach"]
        if zz < 1.25e-3:
            return (
                MAT["substrate"]
                if _inside_square(xx, yy, 14.0e-3)
                else MAT["mold"]
            )
        if zz < 1.35e-3:
            return (
                MAT["die_attach"]
                if _inside_square(xx, yy, 4.5e-3)
                else MAT["mold"]
            )
        if zz < 1.60e-3 and _inside_square(xx, yy, 4.5e-3):
            return MAT["si"]
        return MAT["mold"]

    return build_domain(
        "whole_pop",
        x,
        y,
        z,
        material,
        sources=[
            SourceBox(
                "die_bottom",
                -0.0045,
                0.0045,
                -0.0045,
                0.0045,
                0.35e-3,
                0.55e-3,
            ),
            SourceBox(
                "die_top",
                -0.0045,
                0.0045,
                -0.0045,
                0.0045,
                1.35e-3,
                1.60e-3,
            ),
        ],
        source_powers=[1.0, 0.5],
    )


def boundary_port(domain: Domain, side: str) -> FacePort:
    """Return one of the six rectangular external FVM face sets."""
    nx, ny, nz = len(domain.x) - 1, len(domain.y) - 1, len(domain.z) - 1
    dx, dy, dz = np.diff(domain.x), np.diff(domain.y), np.diff(domain.z)
    ids: list[int] = []
    half: list[float] = []
    area: list[float] = []
    rects: list[tuple[float, float, float, float]] = []

    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    if side in {"x-", "x+"}:
        i = 0 if side == "x-" else nx - 1
        for j, k in product(range(ny), range(nz)):
            q = idx(i, j, k)
            a = dy[j] * dz[k]
            ids.append(q)
            area.append(a)
            half.append(domain.k[q] * a / (0.5 * dx[i]))
            rects.append(
                (domain.y[j], domain.y[j + 1], domain.z[k], domain.z[k + 1])
            )
    elif side in {"y-", "y+"}:
        j = 0 if side == "y-" else ny - 1
        for i, k in product(range(nx), range(nz)):
            q = idx(i, j, k)
            a = dx[i] * dz[k]
            ids.append(q)
            area.append(a)
            half.append(domain.k[q] * a / (0.5 * dy[j]))
            rects.append(
                (domain.x[i], domain.x[i + 1], domain.z[k], domain.z[k + 1])
            )
    elif side in {"z-", "z+"}:
        k = 0 if side == "z-" else nz - 1
        for i, j in product(range(nx), range(ny)):
            q = idx(i, j, k)
            a = dx[i] * dy[j]
            ids.append(q)
            area.append(a)
            half.append(domain.k[q] * a / (0.5 * dz[k]))
            rects.append(
                (domain.x[i], domain.x[i + 1], domain.y[j], domain.y[j + 1])
            )
    else:
        raise KeyError(side)

    return FacePort(
        np.asarray(ids, dtype=np.int64),
        np.asarray(half, dtype=np.float64),
        np.asarray(area, dtype=np.float64),
        np.asarray(rects, dtype=np.float64),
    )


def face_term(domain: Domain, port: FacePort) -> sp.csc_matrix:
    """Affine Robin boundary matrix H = integral phi_i phi_j dA for cell traces."""
    return sp.csc_matrix(
        (port.area, (port.ids, port.ids)), shape=domain.K.shape
    )


def boundary_data(domain: Domain):
    ports = {side: boundary_port(domain, side) for side in FACE_DEGREES}
    terms = {side: face_term(domain, port) for side, port in ports.items()}
    side_group = sp.csc_matrix(domain.K.shape)
    for side in SIDE_FACES:
        side_group = side_group + terms[side]
    groups = [side_group, terms[BOTTOM_FACE], terms[TOP_FACE]]
    ranges = np.asarray([SIDE_HTC_RANGE, BOTTOM_HTC_RANGE, TOP_HTC_RANGE])
    return ports, terms, groups, ranges


def extract_whole_pop_basis(
    domain: Domain,
    boundary_groups,
    h_ranges,
    *,
    tolerance: float,
    probe_rounds: int,
):
    if domain.source_matrix is None or domain.source_matrix.shape[1] != 2:
        raise ValueError("2023 whole-PoP experiment requires exactly two source ports")
    ops = normalized_operators(domain.K, domain.C, domain.rhs)
    return build_parametric_basis(
        ops,
        domain.source_matrix,
        boundary_groups,
        h_ranges,
        tolerance=tolerance,
        probe_rounds=probe_rounds,
        seed=20260915,
    )


def _boundary_grams(ports, basis):
    full, contour, diagnostics = {}, {}, {}
    for side, port in ports.items():
        trace = np.asarray(basis[port.ids, :], dtype=np.float64)
        full[side] = trace.T @ (port.area[:, None] * trace)
        elements, coefficients, relative = contour_trace_coefficients(
            port, basis, FACE_DEGREES[side]
        )
        contour[side] = coefficients.T @ coefficients
        diagnostics[side] = {
            "degree": int(FACE_DEGREES[side]),
            "full_rows": int(port.ids.size),
            "contour_rows": int(elements.size),
            "projection_error_pct": float(100.0 * relative),
        }
    return full, contour, diagnostics


def _assemble_from_grams(K0_hat, grams, case: BoundaryCase):
    K = np.asarray(K0_hat, dtype=np.float64).copy()
    for side in SIDE_FACES:
        K += case.side_h * grams[side]
    K += case.bottom_h * grams[BOTTOM_FACE]
    K += case.top_h * grams[TOP_FACE]
    return K


def _full_operator(
    domain: Domain,
    terms,
    case: BoundaryCase,
    shift: float = 0.0,
):
    K = domain.K.copy()
    for side in SIDE_FACES:
        K = K + case.side_h * terms[side]
    K = K + case.bottom_h * terms[BOTTOM_FACE]
    K = K + case.top_h * terms[TOP_FACE]
    if shift:
        K = K + shift * domain.C
    return K.tocsc()


def _field_metrics(domain: Domain, reference: np.ndarray, approx: np.ndarray):
    diff = np.asarray(approx) - np.asarray(reference)
    scale = max(float(np.max(np.abs(reference))), 1.0e-30)
    vol = domain.volumes
    return {
        "global_inf_pct": float(100.0 * np.max(np.abs(diff)) / scale),
        "global_volume_l2_pct": float(
            100.0
            * math.sqrt(
                float(np.sum(vol * diff * diff))
                / max(float(np.sum(vol * reference * reference)), 1.0e-300)
            )
        ),
    }


def _junctions(domain: Domain, field: np.ndarray) -> np.ndarray:
    G = np.asarray(domain.source_matrix, dtype=np.float64)
    return np.asarray([G[:, j] @ field for j in range(G.shape[1])])


def evaluate_reduced_case(
    domain: Domain,
    basis: np.ndarray,
    K0_hat,
    F_hat,
    grams,
    case: BoundaryCase,
    reference: np.ndarray,
):
    K_hat = _assemble_from_grams(K0_hat, grams, case)
    q = scipy.linalg.solve(
        K_hat,
        F_hat @ domain.source_powers,
        assume_a="pos",
    )
    reduced = basis @ q
    metrics = _field_metrics(domain, reference, reduced)
    jref = _junctions(domain, reference)
    jrom = _junctions(domain, reduced)
    for i, (a, b) in enumerate(zip(jref, jrom), start=1):
        metrics[f"die{i}_junction_pct"] = float(
            100.0 * abs(b - a) / max(abs(a), 1.0e-30)
        )
    return metrics, reduced


def evaluate_transfer_holdouts(
    domain: Domain,
    terms,
    basis: np.ndarray,
    K0_hat,
    C_hat,
    F_hat,
    full_grams,
    contour_grams,
    *,
    shifts: tuple[float, ...],
):
    """Independent source/shift validation of dynamic response-space coverage."""
    rows = []
    selected = ("paper_fig2", "paper_fig4", "range_low", "range_high")
    for case_name in selected:
        case = CASES[case_name]
        for shift in shifts:
            A = _full_operator(domain, terms, case, shift=shift)
            for port in range(domain.source_matrix.shape[1]):
                ref = spd_solve(A, domain.source_matrix[:, port])
                for method, grams in (
                    ("full_trace", full_grams),
                    ("contour", contour_grams),
                ):
                    Ahat = _assemble_from_grams(K0_hat, grams, case) + shift * C_hat
                    q = scipy.linalg.solve(
                        Ahat,
                        F_hat[:, port],
                        assume_a="pos",
                    )
                    approx = basis @ q
                    metrics = _field_metrics(domain, ref, approx)
                    rows.append(
                        {
                            "method": method,
                            "case": case_name,
                            "shift_per_s": float(shift),
                            "source_port": int(port + 1),
                            **metrics,
                        }
                    )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tolerance", type=float, default=PAPER_TOLERANCE)
    parser.add_argument("--probe-rounds", type=int, default=3)
    parser.add_argument("--nxy", type=int, default=43)
    parser.add_argument("--nz", type=int, default=36)
    parser.add_argument(
        "--cases", nargs="*", choices=tuple(CASES), default=list(CASES)
    )
    parser.add_argument(
        "--validation-shifts",
        type=float,
        nargs="*",
        default=[0.0, 1.0e-1, 10.0],
    )
    parser.add_argument("--vtu", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    domain = build_whole_pop(args.nxy, args.nz)
    ports, terms, boundary_groups, h_ranges = boundary_data(domain)

    started = time.time()
    basis, extraction = extract_whole_pop_basis(
        domain,
        boundary_groups,
        h_ranges,
        tolerance=args.tolerance,
        probe_rounds=args.probe_rounds,
    )
    extraction_s = time.time() - started
    np.save(args.out / "basis_whole_pop.npy", basis)

    K0_hat = np.asarray(basis.T @ (domain.K @ basis), dtype=np.float64)
    C_hat = np.asarray(basis.T @ (domain.C @ basis), dtype=np.float64)
    F_hat = np.asarray(basis.T @ domain.source_matrix, dtype=np.float64)
    full_grams, contour_grams, contour_diag = _boundary_grams(ports, basis)

    steady_rows = []
    nominal_artifact = None
    for case_name in args.cases:
        case = CASES[case_name]
        reference = spd_solve(_full_operator(domain, terms, case), domain.rhs)
        for method, grams in (
            ("full_trace", full_grams),
            ("contour", contour_grams),
        ):
            metrics, reduced = evaluate_reduced_case(
                domain,
                basis,
                K0_hat,
                F_hat,
                grams,
                case,
                reference,
            )
            row = {
                "method": method,
                "case": case_name,
                "order": int(basis.shape[1]),
                "full_dofs": int(domain.n_cells),
                "in_declared_range": bool(case.in_declared_range),
                **metrics,
            }
            steady_rows.append(row)
            print(json.dumps(row), flush=True)
            if args.vtu and case_name == "paper_fig2" and method == "contour":
                nominal_artifact = (reference, reduced)

    transfer_rows = evaluate_transfer_holdouts(
        domain,
        terms,
        basis,
        K0_hat,
        C_hat,
        F_hat,
        full_grams,
        contour_grams,
        shifts=tuple(args.validation_shifts),
    )
    _write_csv(args.out / "comparison.csv", steady_rows)
    _write_csv(args.out / "transfer_validation.csv", transfer_rows)

    contour_rows_formula = int(
        sum(v["contour_rows"] for v in contour_diag.values())
    )
    summary = {
        "paper_reference": {
            "detailed_dofs": PAPER_DETAILED_DOF,
            "bci_order": PAPER_BCI_ORDER,
            "tolerance": PAPER_TOLERANCE,
            "sources": 2,
            "side_htc_range_W_m2K": list(SIDE_HTC_RANGE),
            "bottom_htc_range_W_m2K": list(BOTTOM_HTC_RANGE),
            "top_htc_range_W_m2K": list(TOP_HTC_RANGE),
            "full_boundary_rows": PAPER_FULL_BOUNDARY_ROWS,
            "contour_rows_printed": PAPER_CONTOUR_ROWS_PRINTED,
        },
        "surrogate": {
            "full_dofs": int(domain.n_cells),
            "sources": int(domain.source_matrix.shape[1]),
            "source_powers_W": domain.source_powers.tolist(),
            "boundary_surfaces": 6,
            "independent_htc_parameters": 3,
            "side_faces_share_one_htc": True,
            "contour_rows_formula": contour_rows_formula,
            "contour_degrees": FACE_DEGREES,
            "note": (
                "The 2023 paper does not publish sufficient geometry/material data "
                "for an exact 561408-DoF reproduction; this model follows the "
                "published reduction problem and boundary/source specification."
            ),
        },
        "extraction_s": float(extraction_s),
        "extraction": extraction,
        "contour": contour_diag,
        "steady_max_in_range_inf_pct": {
            method: float(
                max(
                    row["global_inf_pct"]
                    for row in steady_rows
                    if row["method"] == method and row["in_declared_range"]
                )
            )
            for method in ("full_trace", "contour")
        },
        "transfer_max_in_range_inf_pct": {
            method: float(
                max(
                    row["global_inf_pct"]
                    for row in transfer_rows
                    if row["method"] == method
                )
            )
            for method in ("full_trace", "contour")
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if nominal_artifact is not None:
        reference, reduced = nominal_artifact
        write_domains_vtu(
            args.out / "whole_pop_paper_fig2_contour.vtu",
            [domain],
            {
                "T_ref_K": [AMBIENT_K + reference],
                "T_rom_K": [AMBIENT_K + reduced],
                "abs_error_K": [np.abs(reference - reduced)],
                "material_id": [domain.material_id],
            },
        )

    expected_formula_rows = 4 * 6 + 15 + 66
    if contour_rows_formula != expected_formula_rows:
        raise RuntimeError(
            f"contour row count {contour_rows_formula} != formula "
            f"{expected_formula_rows}"
        )


if __name__ == "__main__":
    main()
