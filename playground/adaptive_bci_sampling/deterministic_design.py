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
3. ``build_basis``: the production elliptic frequency plan per port, with every
   selected parameter at the low shifts, the seed at the high shifts, and the
   steady (DC) endpoint, followed by the unchanged normalized-snapshot SVD.

Every full-order solve is counted; no random number is drawn anywhere.
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


def responses(kernel, terms, source, parameter):
    """All source responses at one parameter (one sparse factorization)."""
    factor = spla.splu(full_operator(kernel, terms, parameter), permc_spec="MMD_AT_PLUS_A")
    return np.asarray(factor.solve(source), dtype=np.float64)


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

    minimum_operator = sp.csc_matrix(kernel)
    for value, term in zip(ranges[:, 0], terms):
        minimum_operator = minimum_operator + float(value) * term
    minimum_factor = spla.splu(minimum_operator.tocsc())
    started = time.perf_counter()

    basis = None
    score = np.inf
    history = []
    while True:
        blocks = []
        for point in selected:
            key = tuple(float(f"{value:.14e}") for value in point)
            block = response_cache.get(key)
            if block is None:
                block = responses(kernel, terms, source, point)
                response_cache[key] = block
            blocks.append(block)
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
            "seconds": time.perf_counter() - started,
        },
    )


def build_basis(
    kernel,
    mass,
    terms,
    source,
    points,
    *,
    tolerance=1e-3,
    low_shift_threshold=None,
    include_dc=True,
    low_shift_points=None,
    constant=True,
):
    """Assemble snapshots on the production elliptic plan and compress them.

    ``points`` are the selected effective HTC parameters.  Every port uses the
    stock elliptic shift plan.  Shifts at or below ``low_shift_threshold`` use
    ``low_shift_points`` (default: all selected points); higher shifts use the
    first selected point (the Zolotarev seed) alone.  The steady endpoint is
    sampled at the low-shift points when ``include_dc`` is set.  The closing
    compression is the production unit-column-normalized SVD.
    """
    kernel = sp.csc_matrix(kernel)
    mass = sp.csc_matrix(mass)
    source = np.asarray(source, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    low_shift_points = points if low_shift_points is None else np.asarray(low_shift_points, float)
    if points.ndim != 2:
        raise ValueError("points must be a two-dimensional array")

    snapshots = []
    plan = []
    started = time.perf_counter()
    for port in range(source.shape[1]):
        response = source[:, port]
        lower, upper = port_eigenvalue_bounds(kernel, mass, response)
        count = mpmm_elliptic_shift_count(tolerance, lower, upper)
        shifts = mpmm_elliptic_shifts(count, upper, upper / lower)
        threshold = 1.0 if low_shift_threshold is None else float(low_shift_threshold)
        low_shifts = [float(shift) for shift in shifts if shift <= threshold]
        plan.append({"port": port, "shift_count": int(count), "low_shifts": low_shifts})
        for shift in shifts:
            shifted = kernel + float(shift) * mass
            used = points if float(shift) <= threshold else points[:1]
            for point in used:
                snapshots.append(
                    np.asarray(
                        spla.splu(full_operator(shifted, terms, point)).solve(response),
                        dtype=np.float64,
                    )
                )
        if include_dc:
            for point in low_shift_points:
                snapshots.append(
                    np.asarray(
                        spla.splu(full_operator(kernel, terms, point)).solve(response),
                        dtype=np.float64,
                    )
                )

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
        "snapshot_columns": int(matrix.shape[1]),
        "svd_cutoff": float(tolerance),
        "svd_kept_order": int(np.count_nonzero(singular_values >= tolerance * singular_values[0])),
        "basis_order": int(basis.shape[1]),
        "low_shift_threshold": float(threshold),
        "include_dc": bool(include_dc),
        "per_port_plan": plan,
        "seconds": time.perf_counter() - started,
    }
    return np.ascontiguousarray(basis), matrix, info
