#!/usr/bin/env python3
"""Compare minimax rational HTC samples on BCI Case 1.

This experiment deliberately keeps the raw, incrementally orthonormalized
snapshot span.  There is no closing SVD, so a result cannot get worse merely
because a larger sample set changes singular-value weighting.

The primary metric is the maximum entrywise error of the four-source junction
transfer matrix.  The nominal-power junction error is also reported.  Exact
energy-product bounds on the fixed holdout are diagnostics and are distinct
from the online residual certificate used by the greedy selector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "bci_rom_testcase1"))

from compare_sampling import (  # noqa: E402
    chebyshev_lobatto,
    full_operator,
    map_box,
    padua_points,
    response,
    validation_points,
)
from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel.utils import orthonormalize_block  # noqa: E402
from certified_greedy import (  # noqa: E402
    prepare_residual_certificate,
    select_worst_certificate,
)
from zolotarev import (  # noqa: E402
    CoordinateSpectrum,
    coordinate_spectral_enclosures,
    zolotarev_count,
    zolotarev_rule,
)


@dataclass
class Design:
    name: str
    points: np.ndarray
    basis: np.ndarray | None = None
    extraction_seconds: float = 0.0
    worst_error: float = -1.0
    worst_bound: float = -1.0
    worst_entrywise_error: float = -1.0
    worst_entrywise_bound: float = -1.0
    worst_parameter: tuple[float, ...] | None = None
    selection_certificate: float = np.nan


def _point_key(point: np.ndarray) -> tuple[float, ...]:
    return tuple(float(f"{value:.14e}") for value in point)


def tensor_product(axes: list[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([coordinate.ravel() for coordinate in mesh])


def logarithmic_tensor_grid(ranges: np.ndarray, count: int) -> np.ndarray:
    unit_axes = [np.linspace(0.0, 1.0, count) for _ in range(ranges.shape[0])]
    unit = tensor_product(unit_axes)
    logarithms = np.log10(ranges)
    return 10.0 ** (
        logarithms[:, 0] + unit * (logarithms[:, 1] - logarithms[:, 0])
    )


def raw_snapshot_basis(
    K,
    terms,
    source_shape,
    points: np.ndarray,
    response_cache: dict[tuple[float, ...], np.ndarray],
) -> np.ndarray:
    """Build the untruncated union of all block response snapshots."""
    basis = np.empty((K.shape[0], 0), dtype=np.float64)
    for point in points:
        key = _point_key(point)
        block = response_cache.get(key)
        if block is None:
            block = response(K, terms, source_shape, point)
            response_cache[key] = block
        addition = orthonormalize_block(basis, block)
        if addition.shape[1]:
            basis = np.ascontiguousarray(np.column_stack((basis, addition)))

    # Preserve the steady uniform-temperature mode exactly, as the production
    # extractor does.  Its addition is rank-revealing and untruncated.
    constant = np.ones((K.shape[0], 1), dtype=np.float64)
    addition = orthonormalize_block(basis, constant)
    if addition.shape[1]:
        basis = np.ascontiguousarray(np.column_stack((basis, addition)))
    return basis


def certified_greedy_design(
    K,
    terms,
    source_shape,
    power,
    ranges,
    spectra,
    candidates,
    tolerance: float,
    maximum_points: int,
    response_cache,
    metric: str,
) -> Design:
    """Select full solves by the maximum finite-grid residual certificate."""
    seed = np.asarray(
        [
            zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(ranges[index]), 1
            ).parameter_nodes[0]
            for index, spectrum in enumerate(spectra)
        ]
    )
    selected = [seed]
    minimum_operator = K.copy()
    for value, term in zip(ranges[:, 0], terms):
        minimum_operator = minimum_operator + float(value) * term
    minimum_factor = spla.splu(minimum_operator.tocsc())

    started = time.perf_counter()
    final_score = np.inf
    basis = None
    while True:
        basis = raw_snapshot_basis(
            K, terms, source_shape, np.asarray(selected), response_cache
        )
        certificate = prepare_residual_certificate(
            K,
            terms,
            source_shape,
            ranges,
            basis,
            minimum_factor=minimum_factor,
        )
        scores = np.empty(candidates.shape[0], dtype=np.float64)
        absolute_scores = np.empty(candidates.shape[0], dtype=np.float64)
        for index, point in enumerate(candidates):
            if metric == "entrywise":
                _transfer, absolute, relative = certificate.evaluate_entrywise(point)
            else:
                _junction, absolute, relative = certificate.evaluate(point, power)
            scores[index] = float(np.max(relative))
            absolute_scores[index] = float(np.max(absolute))

        # A relative certificate does not exist while |y_hat| <= Delta.  In
        # that phase many candidates have score=inf, so np.argmax would make
        # the result depend on their enumeration order.  Break the tie using
        # the largest absolute output bound, which is itself certified and has
        # common physical units across every junction-transfer entry.
        worst_index = select_worst_certificate(scores, absolute_scores)
        final_score = float(scores[worst_index])
        print(
            f"  greedy tolerance={tolerance:g} points={len(selected)} "
            f"order={basis.shape[1]} max-certificate={final_score:.6e} "
            f"at={tuple(float(v) for v in candidates[worst_index])}",
            flush=True,
        )
        if final_score <= tolerance or len(selected) >= maximum_points:
            break
        next_point = candidates[worst_index]
        if any(np.array_equal(next_point, existing) for existing in selected):
            raise RuntimeError("greedy certificate selected an existing point")
        selected.append(next_point.copy())

    assert basis is not None
    return Design(
        name=f"greedy-{metric}-grid-tol-{tolerance:.0e}",
        points=np.asarray(selected),
        basis=basis,
        extraction_seconds=time.perf_counter() - started,
        selection_certificate=final_score,
    )


def _make_designs(
    spectra: list[CoordinateSpectrum],
    ranges: np.ndarray,
    junction_tolerances: list[float],
    exploratory_pairs: list[tuple[int, int]],
    padua_degrees: list[int],
    tensor_log_counts: list[int],
) -> list[Design]:
    designs: list[Design] = []
    seen: set[tuple[int, int]] = set()

    for counts in exploratory_pairs:
        if counts in seen:
            continue
        seen.add(counts)
        axes = [
            zolotarev_rule(
                (spectrum.lower, spectrum.upper),
                tuple(ranges[index]),
                counts[index],
            ).parameter_nodes
            for index, spectrum in enumerate(spectra)
        ]
        designs.append(
            Design(name=f"zolo-{counts[0]}x{counts[1]}", points=tensor_product(axes))
        )

    dimensions = ranges.shape[0]
    for tolerance in junction_tolerances:
        # For a compliant output the output error is the square of the energy
        # error.  Splitting sqrt(tol) equally among coordinates is conservative
        # for the first-order telescoping estimate; the continuous two-variable
        # guarantee is intentionally not asserted by this experiment.
        coordinate_tolerance = np.sqrt(tolerance) / dimensions
        counts = tuple(
            zolotarev_count(
                (spectrum.lower, spectrum.upper),
                tuple(ranges[index]),
                coordinate_tolerance,
            )
            for index, spectrum in enumerate(spectra)
        )
        if counts in seen:
            continue
        seen.add(counts)
        rules = [
            zolotarev_rule(
                (spectrum.lower, spectrum.upper),
                tuple(ranges[index]),
                counts[index],
            )
            for index, spectrum in enumerate(spectra)
        ]
        label = f"zolo-tol-{tolerance:.0e}-{counts[0]}x{counts[1]}"
        designs.append(
            Design(
                name=label,
                points=tensor_product([rule.parameter_nodes for rule in rules]),
            )
        )

    for degree in padua_degrees:
        designs.append(
            Design(
                name=f"padua-log-{degree}",
                points=map_box(padua_points(degree), ranges, logarithmic=True),
            )
        )
    for count in tensor_log_counts:
        unit_axis = chebyshev_lobatto(count - 1)
        designs.append(
            Design(
                name=f"cl-log-{count}x{count}",
                points=map_box(
                    tensor_product([unit_axis, unit_axis]),
                    ranges,
                    logarithmic=True,
                ),
            )
        )
    return designs


def _evaluate_at_parameter(
    K,
    terms,
    source_shape,
    power,
    point,
    exact_responses,
    designs: list[Design],
) -> None:
    operator = full_operator(K, terms, point)
    exact_junction = np.asarray(source_shape.T @ (exact_responses @ power)).ravel()
    exact_transfer = np.ascontiguousarray(source_shape.T @ exact_responses)
    tiny = np.finfo(float).tiny

    for design in designs:
        basis = design.basis
        assert basis is not None
        projected_operator = basis.T @ (operator @ basis)
        projected_source = basis.T @ source_shape
        coefficients = scipy.linalg.solve(
            projected_operator,
            projected_source,
            assume_a="pos",
            check_finite=False,
        )
        approximate_junction = np.asarray(
            projected_source.T @ (coefficients @ power)
        ).ravel()
        approximate_transfer = np.ascontiguousarray(
            projected_source.T @ coefficients
        )
        relative = np.abs(approximate_junction - exact_junction) / np.maximum(
            np.abs(exact_junction), tiny
        )
        local_error = float(relative.max())

        # Since every junction functional is one column of source_shape, the
        # same block basis supplies both primal and dual Galerkin solutions.
        reduced_responses = basis @ coefficients
        response_errors = exact_responses - reduced_responses
        operator_errors = operator @ response_errors
        dual_energy_squared = np.einsum(
            "ij,ij->j", response_errors, operator_errors
        )
        primal_error = response_errors @ power
        primal_energy_squared = float(primal_error @ (operator @ primal_error))
        absolute_bound = np.sqrt(
            np.maximum(dual_energy_squared, 0.0)
            * max(primal_energy_squared, 0.0)
        )
        relative_bound = absolute_bound / np.maximum(np.abs(exact_junction), tiny)
        local_bound = float(relative_bound.max())

        entrywise_relative = np.abs(approximate_transfer - exact_transfer) / np.maximum(
            np.abs(exact_transfer), tiny
        )
        local_entrywise_error = float(entrywise_relative.max())
        energy_norms = np.sqrt(np.maximum(dual_energy_squared, 0.0))
        entrywise_absolute_bound = np.outer(energy_norms, energy_norms)
        entrywise_relative_bound = entrywise_absolute_bound / np.maximum(
            np.abs(exact_transfer), tiny
        )
        local_entrywise_bound = float(entrywise_relative_bound.max())

        if local_error > design.worst_error:
            design.worst_error = local_error
            design.worst_parameter = tuple(float(value) for value in point)
        design.worst_bound = max(design.worst_bound, local_bound)
        design.worst_entrywise_error = max(
            design.worst_entrywise_error, local_entrywise_error
        )
        design.worst_entrywise_bound = max(
            design.worst_entrywise_bound, local_entrywise_bound
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", nargs="?", type=float, default=2.5)
    parser.add_argument(
        "--junction-tolerances",
        nargs="*",
        type=float,
        default=[1.0e-2, 1.0e-3, 1.0e-4],
    )
    parser.add_argument(
        "--count-pairs",
        nargs="*",
        default=["2x2", "3x2", "4x3"],
        help="additional exploratory Zolotarev tensor counts, e.g. 3x2",
    )
    parser.add_argument("--padua-degrees", nargs="*", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--tensor-log-counts", nargs="*", type=int, default=[])
    parser.add_argument(
        "--greedy-tolerances",
        nargs="*",
        type=float,
        default=[1.0e-2, 1.0e-3, 1.0e-4],
    )
    parser.add_argument("--greedy-grid", type=int, default=21)
    parser.add_argument("--greedy-max-points", type=int, default=12)
    parser.add_argument(
        "--greedy-metric",
        choices=("entrywise", "power"),
        default="entrywise",
    )
    parser.add_argument("--validation-grid", type=int, default=21)
    parser.add_argument("--random-holdout", type=int, default=128)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="stop after construction and finite-candidate certification",
    )
    args = parser.parse_args()

    exploratory_pairs = []
    for item in args.count_pairs:
        fields = item.lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid count pair {item!r}")
        exploratory_pairs.append((int(fields[0]), int(fields[1])))

    model = Case1Model(
        Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm)
    )
    K = model.core.K.tocsc()
    terms = [term.tocsc() for term in model.boundary_terms]
    source_shape = np.asarray(model.source_shape, dtype=np.float64)
    power = np.asarray(model.nominal_power(), dtype=np.float64)
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    if ranges.shape[0] != 2:
        raise ValueError("this comparison currently expects exactly two HTC groups")

    print(
        f"mesh={args.mesh_mm:g} mm cells={K.shape[0]} sources={source_shape.shape[1]} "
        f"groups={ranges.shape[0]}",
        flush=True,
    )
    spectra = coordinate_spectral_enclosures(K, terms, ranges)
    for index, spectrum in enumerate(spectra):
        print(
            f"spectrum group={index} active={spectrum.active_cells} "
            f"interval=[{spectrum.lower:.9g}, {spectrum.upper:.9g}] "
            f"residuals=({spectrum.lower_ritz_residual:.2e},"
            f" {spectrum.upper_ritz_residual:.2e}) seconds={spectrum.seconds:.3f}",
            flush=True,
        )

    designs = _make_designs(
        spectra,
        ranges,
        list(args.junction_tolerances),
        exploratory_pairs,
        list(args.padua_degrees),
        list(args.tensor_log_counts),
    )
    response_cache: dict[tuple[float, ...], np.ndarray] = {}
    for design in designs:
        started = time.perf_counter()
        design.basis = raw_snapshot_basis(
            K, terms, source_shape, design.points, response_cache
        )
        design.extraction_seconds = time.perf_counter() - started
        print(
            f"built {design.name} points={design.points.shape[0]} "
            f"rhs={design.points.shape[0] * source_shape.shape[1]} "
            f"order={design.basis.shape[1]} seconds={design.extraction_seconds:.3f}",
            flush=True,
        )

    greedy_candidates = logarithmic_tensor_grid(ranges, args.greedy_grid)
    for tolerance in args.greedy_tolerances:
        design = certified_greedy_design(
            K,
            terms,
            source_shape,
            power,
            ranges,
            spectra,
            greedy_candidates,
            tolerance,
            args.greedy_max_points,
            response_cache,
            args.greedy_metric,
        )
        designs.append(design)
        print(
            f"built {design.name} points={design.points.shape[0]} "
            f"rhs={design.points.shape[0] * source_shape.shape[1]} "
            f"order={design.basis.shape[1]} seconds={design.extraction_seconds:.3f} "
            f"grid-certificate={design.selection_certificate:.6e}",
            flush=True,
        )

    started = time.perf_counter()
    if args.skip_validation:
        holdout = np.empty((0, ranges.shape[0]), dtype=np.float64)
    else:
        holdout = validation_points(
            ranges, args.validation_grid, args.random_holdout
        )
        for index, point in enumerate(holdout, start=1):
            exact = response(K, terms, source_shape, point)
            _evaluate_at_parameter(
                K, terms, source_shape, power, point, exact, designs
            )
            if index % 100 == 0:
                print(f"validated {index}/{holdout.shape[0]}", flush=True)
    validation_seconds = time.perf_counter() - started

    print("\nresult")
    print(
        "design points rhs order selection-certificate worst-junction "
        "worst-exact-energy-bound worst-entrywise worst-entrywise-bound parameter"
    )
    for design in designs:
        assert design.basis is not None
        print(
            f"{design.name} {design.points.shape[0]} "
            f"{design.points.shape[0] * source_shape.shape[1]} "
            f"{design.basis.shape[1]} {design.selection_certificate:.9e} "
            f"{design.worst_error:.9e} "
            f"{design.worst_bound:.9e} {design.worst_entrywise_error:.9e} "
            f"{design.worst_entrywise_bound:.9e} {design.worst_parameter}"
        )
    print(
        f"unique-sample-points={len(response_cache)} holdout={holdout.shape[0]} "
        f"validation-seconds={validation_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
