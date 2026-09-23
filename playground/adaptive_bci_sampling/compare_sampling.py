#!/usr/bin/env python3
"""Compare deterministic affine-parameter samples on BCI Case 1.

This experiment isolates the open question in ``NEXT_STEPS.md``: how to choose
the HTC vector at one fixed frequency shift.  It deliberately counts the full
response fields when constructing the space, but judges the result only by the
junction temperatures under the nominal four-source power vector.

The deterministic candidates are

* tensor Chebyshev--Lobatto points, in either physical or log10 HTC;
* Padua points, an explicit unisolvent set for total-degree polynomials in two
  variables, again in either physical or log10 HTC.

Run a cheap exploratory case with::

    python playground/adaptive_bci_sampling/compare_sampling.py 5 1 2 3 4 5

The default mesh is 2.5 mm and the default degrees are 1 through 8.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import splu

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "bci_rom_testcase1"))

from model_case1 import Case1Config, Case1Model  # noqa: E402


@dataclass(frozen=True)
class Result:
    family: str
    degree: int
    parameter_points: int
    rhs_solves: int
    basis_order: int
    worst_relative_junction_error: float
    worst_parameter: tuple[float, ...]


def chebyshev_lobatto(degree: int) -> np.ndarray:
    """Chebyshev--Lobatto points on ``[-1, 1]`` in increasing order."""
    if degree < 1:
        raise ValueError("degree must be positive")
    return np.sort(np.cos(np.pi * np.arange(degree + 1) / degree))


def tensor_points(degree: int) -> np.ndarray:
    """Tensor Chebyshev--Lobatto points for a two-parameter box."""
    x = chebyshev_lobatto(degree)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel()))


def padua_points(degree: int) -> np.ndarray:
    """First-family Padua points of total degree ``degree`` on ``[-1, 1]^2``.

    The parity form is explicit and has exactly ``(n+1)(n+2)/2`` points.
    """
    if degree < 1:
        raise ValueError("degree must be positive")
    points = [
        (np.cos(i * np.pi / degree), np.cos(j * np.pi / (degree + 1)))
        for i in range(degree + 1)
        for j in range(degree + 2)
        if (i + j) % 2 == 0
    ]
    result = np.asarray(points, dtype=np.float64)
    expected = (degree + 1) * (degree + 2) // 2
    if result.shape != (expected, 2):
        raise AssertionError(f"Padua cardinality {result.shape[0]} != {expected}")
    return result


def smolyak_points(level: int) -> np.ndarray:
    """Nested two-dimensional Clenshaw--Curtis Smolyak point set.

    Level 1 is the box centre, level 2 is the five-point axis cross, and level
    3 adds the first corner/mixed-direction shell (13 points total).
    """
    if level < 1:
        raise ValueError("level must be positive")

    def one_dimensional(one_d_level: int) -> np.ndarray:
        if one_d_level == 1:
            return np.array([0.0])
        degree = 2 ** (one_d_level - 1)
        return chebyshev_lobatto(degree)

    points = set()
    for left_level in range(1, level + 1):
        for right_level in range(1, level + 2 - left_level):
            for left in one_dimensional(left_level):
                for right in one_dimensional(right_level):
                    points.add((round(float(left), 15), round(float(right), 15)))
    return np.asarray(sorted(points), dtype=np.float64)


def map_box(points: np.ndarray, ranges: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    """Map points from ``[-1,1]^k`` to the physical/effective HTC box."""
    bounds = np.log10(ranges) if logarithmic else ranges
    mapped = bounds[:, 0] + 0.5 * (points + 1.0) * (bounds[:, 1] - bounds[:, 0])
    return 10.0**mapped if logarithmic else mapped


def full_operator(K, terms, h):
    A = K.copy()
    for value, term in zip(h, terms):
        A = A + float(value) * term
    return A.tocsc()


def response(K, terms, G, h) -> np.ndarray:
    """All source responses at one parameter point (one sparse factorization)."""
    return np.asarray(splu(full_operator(K, terms, h)).solve(G), dtype=np.float64)


def snapshot_basis(snapshot_blocks: list[np.ndarray], tolerance: float) -> np.ndarray:
    """Use the production extractor's normalized-column closing SVD."""
    snapshots = np.column_stack(snapshot_blocks)
    norms = np.linalg.norm(snapshots, axis=0)
    snapshots = snapshots / norms
    U, singular_values, _ = scipy.linalg.svd(
        snapshots, full_matrices=False, check_finite=False
    )
    keep = singular_values >= tolerance * singular_values[0]
    basis = np.ascontiguousarray(U[:, keep])

    constant = np.ones((basis.shape[0], 1), dtype=np.float64)
    constant /= np.linalg.norm(constant)
    constant -= basis @ (basis.T @ constant)
    constant -= basis @ (basis.T @ constant)
    norm = float(np.linalg.norm(constant))
    if norm > np.finfo(float).eps * basis.shape[0]:
        basis = np.column_stack((basis, constant / norm))
    return np.ascontiguousarray(basis)


def reduced_junction(K, terms, G, power, basis, h) -> np.ndarray:
    A = full_operator(K, terms, h)
    A_hat = basis.T @ (A @ basis)
    G_hat = basis.T @ G
    coefficients = scipy.linalg.solve(
        A_hat, G_hat @ power, assume_a="pos", check_finite=False
    )
    return np.asarray(G_hat.T @ coefficients).ravel()


def validation_points(ranges: np.ndarray, count: int, random_count: int) -> np.ndarray:
    """Deterministic log grid plus a fixed-seed log-uniform holdout."""
    axis = np.linspace(0.0, 1.0, count)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    unit = np.column_stack((xx.ravel(), yy.ravel()))
    rng = np.random.default_rng(20260923)
    unit = np.vstack((unit, rng.random((random_count, 2))))
    logs = np.log10(ranges)
    return 10.0 ** (logs[:, 0] + unit * (logs[:, 1] - logs[:, 0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_mm", nargs="?", type=float, default=2.5)
    parser.add_argument("degrees", nargs="*", type=int, default=list(range(1, 9)))
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--validation-grid", type=int, default=21)
    parser.add_argument("--random-holdout", type=int, default=128)
    parser.add_argument(
        "--equal-cost-random-seeds",
        type=int,
        default=0,
        help="also test this many fixed-seed log-uniform designs per degree",
    )
    args = parser.parse_args()

    model = Case1Model(
        Case1Config(max_xy_cell_mm=args.mesh_mm, max_z_cell_mm=args.mesh_mm)
    )
    K = model.core.K.tocsc()
    terms = [term.tocsc() for term in model.boundary_terms]
    G = np.asarray(model.source_shape, dtype=np.float64)
    power = np.asarray(model.nominal_power(), dtype=np.float64)
    ranges = np.asarray(model.h_ranges(), dtype=np.float64)
    holdout = validation_points(ranges, args.validation_grid, args.random_holdout)

    print(
        f"mesh={args.mesh_mm:g} mm cells={K.shape[0]} sources={G.shape[1]} "
        f"groups={ranges.shape[0]} holdout={holdout.shape[0]}"
    )
    print("effective ranges:", ranges.tolist())

    started = time.perf_counter()
    exact_junctions = []
    for index, h in enumerate(holdout, start=1):
        exact_junctions.append(G.T @ (response(K, terms, G, h) @ power))
        if index % 100 == 0:
            print(f"  exact holdout {index}/{holdout.shape[0]}", flush=True)
    exact_junctions = np.asarray(exact_junctions)
    print(f"exact holdout seconds={time.perf_counter() - started:.2f}")

    cache: dict[tuple[float, ...], np.ndarray] = {}
    results: list[Result] = []
    families = (
        ("tensor-linear", tensor_points, False),
        ("tensor-log", tensor_points, True),
        ("padua-linear", padua_points, False),
        ("padua-log", padua_points, True),
        ("smolyak-log", smolyak_points, True),
    )

    for degree in args.degrees:
        for family, point_rule, logarithmic in families:
            unit_points = point_rule(degree)
            samples = map_box(unit_points, ranges, logarithmic=logarithmic)
            blocks = []
            for h in samples:
                key = tuple(float(value) for value in h)
                if key not in cache:
                    cache[key] = response(K, terms, G, h)
                blocks.append(cache[key])
            basis = snapshot_basis(blocks, args.tolerance)

            worst = -1.0
            worst_h = None
            for h, exact_junction in zip(holdout, exact_junctions):
                approximate = reduced_junction(K, terms, G, power, basis, h)
                relative = np.max(
                    np.abs(approximate - exact_junction)
                    / np.maximum(np.abs(exact_junction), np.finfo(float).tiny)
                )
                if relative > worst:
                    worst = float(relative)
                    worst_h = tuple(float(value) for value in h)
            result = Result(
                family=family,
                degree=degree,
                parameter_points=samples.shape[0],
                rhs_solves=samples.shape[0] * G.shape[1],
                basis_order=basis.shape[1],
                worst_relative_junction_error=worst,
                worst_parameter=worst_h,
            )
            results.append(result)
            print(
                f"{family:14s} degree={degree:2d} points={result.parameter_points:3d} "
                f"rhs={result.rhs_solves:4d} order={result.basis_order:3d} "
                f"worst_junction={100.0 * worst:9.5f}% at {result.worst_parameter}",
                flush=True,
            )

        point_count = padua_points(degree).shape[0]
        random_errors = []
        random_orders = []
        for seed_offset in range(args.equal_cost_random_seeds):
            rng = np.random.default_rng(20260923 + seed_offset)
            unit_points = 2.0 * rng.random((point_count, 2)) - 1.0
            samples = map_box(unit_points, ranges, logarithmic=True)
            blocks = []
            for h in samples:
                key = tuple(float(value) for value in h)
                if key not in cache:
                    cache[key] = response(K, terms, G, h)
                blocks.append(cache[key])
            basis = snapshot_basis(blocks, args.tolerance)
            worst = 0.0
            for h, exact_junction in zip(holdout, exact_junctions):
                approximate = reduced_junction(K, terms, G, power, basis, h)
                relative = np.max(
                    np.abs(approximate - exact_junction)
                    / np.maximum(np.abs(exact_junction), np.finfo(float).tiny)
                )
                worst = max(worst, float(relative))
            random_errors.append(worst)
            random_orders.append(basis.shape[1])
        if random_errors:
            print(
                f"random-log[{len(random_errors):2d}] degree={degree:2d} "
                f"points={point_count:3d} rhs={point_count * G.shape[1]:4d} "
                f"order={min(random_orders)}..{max(random_orders)} "
                f"worst_junction min/median/max="
                f"{100.0 * min(random_errors):.5f}%/"
                f"{100.0 * np.median(random_errors):.5f}%/"
                f"{100.0 * max(random_errors):.5f}%",
                flush=True,
            )

    print(f"total seconds={time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
