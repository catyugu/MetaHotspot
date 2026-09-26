#!/usr/bin/env python3
"""Deterministic, certificate-driven parameter sampling for BCI extraction.

The stock Extended FANTASTIC extractor proposes the next boundary parameter by
drawing it log-uniformly at random and retires a shift after a fixed number of
random residual probes.  Both halves are Monte-Carlo statements: the snapshot
set depends on a seed, and a finite number of passing probes says nothing about
the worst point of the parameter box.

This module replaces the random loop by three deterministic stages.

1. ``zolotarev_seed``: one interior point per HTC group from the finite-interval
   Zolotarev rule of :mod:`zolotarev`, which is the minimax rational placement
   for the scalar resolvent kernels of that group.
2. ``certified_greedy_points``: a weak greedy that appends the parameter of the
   largest *certified* output residual on a deterministic log candidate grid.
   The certificate is the A(h_min)-Riesz residual product of
   :mod:`residual_certificate`; every score is an upper bound of the transfer error
   at that candidate, so the stopping statement is deterministic.
3. ``build_basis``: one shared elliptic frequency plan for every port (the union
   of the ports' spectral intervals, so a single plan is valid for all of them
   and each operator is factorized once), every selected parameter at the low
   shifts, the seed at the high shifts, and the steady (DC) endpoint, followed
   by the unchanged normalized-snapshot SVD.

Every full-order solve is counted; no random number is drawn anywhere.  One
sparse factorization serves every source port that asks for it, and one cache
shared between the greedy scorer and ``build_basis`` makes the steady
endpoints cache hits instead of a second solve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from residual_certificate import prepare_residual_certificate, select_worst_certificate
from metahotspot.macromodel.utils import (
    _snapshot_svd_basis,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    port_eigenvalue_bounds,
)
from zolotarev import coordinate_spectral_enclosures, zolotarev_rule


@dataclass
class Design:
    """One deterministic sampling design and its delivered basis."""

    name: str
    points: np.ndarray
    basis: np.ndarray | None = None
    snapshots: np.ndarray | None = None
    full_rhs_solves: int = 0
    selection_certificate: float = np.nan
    shift_plan: list = field(default_factory=list)
    seconds: float = 0.0
    comments: dict = field(default_factory=dict)


def logarithmic_tensor_grid(ranges, count):
    """Deterministic log-uniform candidate grid on a rectangular box."""
    ranges = np.asarray(ranges, dtype=np.float64)
    axes = [np.linspace(0.0, 1.0, int(count)) for _ in range(ranges.shape[0])]
    mesh = np.meshgrid(*axes, indexing="ij")
    unit = np.column_stack([coordinate.ravel() for coordinate in mesh])
    logarithms = np.log10(ranges)
    grid = 10.0 ** (logarithms[:, 0] + unit * (logarithms[:, 1] - logarithms[:, 0]))
    return np.clip(grid, ranges[:, 0], ranges[:, 1])


def full_operator(kernel, terms, parameter):
    operator = sp.csc_matrix(kernel)
    for value, term in zip(parameter, terms):
        operator = operator + float(value) * term
    return operator.tocsc()


def response_block(cache, kernel, mass, terms, source, point, shift=0.0):
    """Get-or-solve ``(K + shift*C + sum_i p_i H_i) x = source``, all ports.

    One sparse factorization serves every source port, and one cache serves
    the greedy scorer (``shift = 0``), the steady endpoint and every frequency
    shift of :func:`build_basis`, so a stationary block solved while selecting
    parameters becomes a cache hit instead of a second solve.
    """
    key = (float(f"{shift:.14e}"),) + tuple(
        float(f"{value:.14e}") for value in np.asarray(point, float)
    )
    block = cache.get(key)
    if block is None:
        operator = kernel if not shift else kernel + float(shift) * mass
        factor = spla.splu(
            full_operator(operator, terms, point), permc_spec="MMD_AT_PLUS_A"
        )
        block = np.asarray(factor.solve(source), dtype=np.float64)
        cache[key] = block
    return block


def zolotarev_seed(kernel, terms, ranges):
    """Degree-one finite-interval Zolotarev product, one node per group."""
    spectra = coordinate_spectral_enclosures(kernel, terms, np.asarray(ranges, float))
    nodes = np.array(
        [
            zolotarev_rule(
                (spectrum.lower, spectrum.upper), tuple(extent), 1
            ).parameter_nodes[0]
            for spectrum, extent in zip(spectra, np.asarray(ranges, float))
        ]
    )
    return nodes, spectra


def certified_greedy_points(
    kernel,
    terms,
    source,
    ranges,
    spectra,
    *,
    tolerance=1e-5,
    maximum_points=8,
    grid=41,
    power=None,
    metric="entrywise",
    cache=None,
    progress=False,
):
    """Append parameters by the largest certified output residual.

    The candidate set is the deterministic ``grid`` x ``grid`` log tensor grid,
    the seed is the Zolotarev product, and the score of a candidate is the
    ``A(h_min)``-Riesz residual bound of the *raw* snapshot span built from the
    parameters selected so far.  The result is reproducible bit for bit.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    power = np.ones(source.shape[1]) if power is None else np.asarray(power, float)
    candidates = logarithmic_tensor_grid(ranges, grid)
    selected = [np.asarray(zolotarev_seed(kernel, terms, ranges)[0], dtype=np.float64)]
    response_cache = {} if cache is None else cache
    cached_before = len(response_cache)

    minimum_operator = sp.csc_matrix(kernel)
    for value, term in zip(ranges[:, 0], terms):
        minimum_operator = minimum_operator + float(value) * term
    minimum_factor = spla.splu(minimum_operator.tocsc())
    started = time.perf_counter()

    basis = None
    score = np.inf
    history = []
    while True:
        blocks = [
            response_block(response_cache, kernel, None, terms, source, point)
            for point in selected
        ]
        basis = np.empty((kernel.shape[0], 0))
        for block in blocks:
            addition = orthonormalize_block(basis, block)
            if addition.shape[1]:
                basis = np.ascontiguousarray(np.column_stack((basis, addition)))
        constant = np.ones((kernel.shape[0], 1)) / np.sqrt(kernel.shape[0])
        addition = orthonormalize_block(basis, constant)
        if addition.shape[1]:
            basis = np.column_stack((basis, addition))

        certificate = prepare_residual_certificate(
            kernel, terms, source, ranges, basis, minimum_factor=minimum_factor
        )
        relative_scores = np.empty(candidates.shape[0])
        absolute_scores = np.empty(candidates.shape[0])
        for index, point in enumerate(candidates):
            if metric == "entrywise":
                _transfer, absolute, relative = certificate.evaluate_entrywise(point)
            else:
                _junction, absolute, relative = certificate.evaluate(point, power)
            relative_scores[index] = float(np.max(relative))
            absolute_scores[index] = float(np.max(absolute))
        worst = select_worst_certificate(relative_scores, absolute_scores)
        score = float(relative_scores[worst])
        history.append(
            {
                "points": len(selected),
                "order": int(basis.shape[1]),
                "certificate": score,
                "worst_parameter": candidates[worst].tolist(),
            }
        )
        if progress:
            print(
                f"  deterministic greedy points={len(selected)} order={basis.shape[1]} "
                f"certificate={score:.6e}",
                flush=True,
            )
        if score <= tolerance or len(selected) >= maximum_points:
            break
        next_point = candidates[worst].copy()
        if any(np.allclose(next_point, existing, rtol=1e-12) for existing in selected):
            raise RuntimeError("deterministic greedy selected an existing point")
        selected.append(next_point)

    return (
        np.asarray(selected, dtype=np.float64),
        score,
        {
            "history": history,
            "candidate_grid": int(grid),
            "tolerance": float(tolerance),
            "factorizations": int(len(response_cache) - cached_before),
            "fresh_rhs_solves": int(len(response_cache) - cached_before)
            * source.shape[1],
            "seconds": time.perf_counter() - started,
        },
    )


def shared_frequency_plan(kernel, mass, source, tolerance):
    """One MPMM elliptic plan that is valid for every source port.

    The elliptic rule only needs an interval containing the port's generalized
    spectrum of ``(K, C)``; the union of the ports' intervals contains all of
    them, so a single plan serves every port with the same closed-form bound.
    Without it, ports whose eigenvalue estimators agree to fifteen digits still
    get eleven of twelve shifts split by the last bit, and the same operator is
    factorized two or three times.
    """
    source = np.asarray(source, dtype=np.float64)
    intervals = [
        port_eigenvalue_bounds(kernel, mass, source[:, port])
        for port in range(source.shape[1])
    ]
    lower = min(interval[0] for interval in intervals)
    upper = max(interval[1] for interval in intervals)
    count = mpmm_elliptic_shift_count(tolerance, lower, upper)
    return {
        "lower": float(lower),
        "upper": float(upper),
        "count": int(count),
        "shifts": [float(value) for value in mpmm_elliptic_shifts(count, upper, upper / lower)],
        "per_port_intervals": [[float(a), float(b)] for a, b in intervals],
    }


def build_basis(
    kernel,
    mass,
    terms,
    source,
    points,
    *,
    plan,
    tolerance=1e-3,
    low_shift_threshold=None,
    include_dc=True,
    low_shift_points=None,
    constant=True,
    cache=None,
):
    """Assemble snapshots on one shared elliptic plan and compress them.

    ``plan`` is the shared frequency plan of :func:`shared_frequency_plan`.
    ``points`` are the selected effective HTC parameters.  Shifts at or below
    ``low_shift_threshold`` use ``low_shift_points`` (default: all selected
    points); higher shifts use the first selected point (the Zolotarev seed)
    alone.  The steady endpoint is sampled at the low-shift points when
    ``include_dc`` is set.  The closing compression is the production
    unit-column-normalized SVD.

    One factorization per distinct ``(shift, parameter)`` operator serves every
    port that asks for it, and a ``cache`` shared with
    :func:`certified_greedy_points` turns the steady endpoints into cache hits:
    they are exactly the blocks the greedy scorer already solved.
    ``factorizations`` counts the sparse factorizations this call performed and
    ``cached_blocks`` the snapshot requests it answered without one.
    """
    kernel = sp.csc_matrix(kernel)
    mass = sp.csc_matrix(mass)
    source = np.asarray(source, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    low_shift_points = points if low_shift_points is None else np.asarray(low_shift_points, float)
    if points.ndim != 2:
        raise ValueError("points must be a two-dimensional array")

    cache = {} if cache is None else cache
    threshold = 1.0 if low_shift_threshold is None else float(low_shift_threshold)
    shifts = [float(value) for value in plan["shifts"]]
    low_shifts = [shift for shift in shifts if shift <= threshold]
    entries = [
        (shift, point)
        for shift in shifts
        for point in (points if shift <= threshold else points[:1])
    ]
    if include_dc:
        entries.extend((0.0, point) for point in low_shift_points)

    known = len(cache)
    started = time.perf_counter()
    snapshots = [
        response_block(cache, kernel, mass, terms, source, point, shift)[:, port]
        for port in range(source.shape[1])
        for shift, point in entries
    ]
    factorizations = len(cache) - known
    matrix = np.column_stack(snapshots)
    basis, singular_values = _snapshot_svd_basis(matrix, tolerance)
    if constant:
        uniform = np.ones((kernel.shape[0], 1)) / np.sqrt(kernel.shape[0])
        addition = orthonormalize_block(basis, uniform)
        if addition.shape[1]:
            basis = np.column_stack((basis, addition))
    info = {
        "points": points.tolist(),
        "full_rhs_solves": int(matrix.shape[1]),
        "factorizations": int(factorizations),
        "cached_blocks": int(matrix.shape[1] - factorizations),
        "snapshot_columns": int(matrix.shape[1]),
        "svd_cutoff": float(tolerance),
        "svd_kept_order": int(np.count_nonzero(singular_values >= tolerance * singular_values[0])),
        "basis_order": int(basis.shape[1]),
        "low_shift_threshold": float(threshold),
        "low_shifts": low_shifts,
        "include_dc": bool(include_dc),
        "frequency_plan": {
            "lower": float(plan["lower"]),
            "upper": float(plan["upper"]),
            "count": int(plan["count"]),
            "per_port_intervals": plan["per_port_intervals"],
        },
        "seconds": time.perf_counter() - started,
    }
    return np.ascontiguousarray(basis), matrix, info
