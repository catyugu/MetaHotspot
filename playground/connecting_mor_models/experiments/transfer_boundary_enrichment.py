#!/usr/bin/env python3
"""Manual extra-boundary-mode sweep for transfer-enriched Extended FANTASTIC.

The transfer directions come only from matrices/vectors (see
``transfer_boundary_basis.py``). Every extraction includes one explicit uniform
boundary heat-flow-density training direction. The user-facing
``extra_boundary_modes`` count adds exactly that many transfer-optimal
nonuniform directions; there is no residual-based stopping rule and no cluster
completion.

Two deterministic holdout ensembles are reported:

* ``transfer_weighted``: random combinations of the uniform direction and the
  computed extra transfer directions, weighted by their transfer gains.
* ``isotropic``: random vectors in the full conductance-normalized boundary
  space. This is deliberately harsh and remains only a stress diagnostic.

No surface coordinates, contour modes, polynomial basis, or attachment geometry
is used to choose a training direction. Euclidean residual is intentionally not
used as a boundary-enrichment quality or stopping metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg

from metahotspot.macromodel.utils import build_parametric_basis, normalized_operators
from mor_common import AMGPCGSolver, boundary_port
from transfer_boundary_basis import (
    PHYSICAL_H_RANGE,
    TransferSample,
    affine_operator,
    boundary_training_rhs,
    build_simple_cube,
    default_samples,
    effective_surface_coefficient,
    normalized_boundary_injection,
    surface_term,
    transfer_boundary_basis,
)


def effective_range(domain, port) -> tuple[float, float]:
    lo, hi = PHYSICAL_H_RANGE
    a = effective_surface_coefficient(domain, port, lo)
    b = effective_surface_coefficient(domain, port, hi)
    return min(a, b), max(a, b)


def unit_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("training source contains a zero column")
    return matrix / norms


def extract_fantastic(domain, top, interface, boundary_rhs, *, seed: int):
    """Run Extended FANTASTIC with physical sources plus boundary training ports."""
    physical = np.asarray(domain.source_matrix, dtype=float)
    boundary_rhs = np.asarray(boundary_rhs, dtype=float)
    if boundary_rhs.ndim != 2 or boundary_rhs.shape[1] < 1:
        raise ValueError("boundary training must include the uniform heat-flow mode")
    sources = np.column_stack((physical, unit_columns(boundary_rhs)))
    terms = [
        surface_term(domain.n_cells, top),
        surface_term(domain.n_cells, interface),
    ]
    ranges = np.asarray(
        [effective_range(domain, top), effective_range(domain, interface)],
        dtype=float,
    )
    ops = normalized_operators(domain.K, domain.C, domain.rhs)
    basis, summary = build_parametric_basis(
        ops,
        sources,
        terms,
        ranges,
        tolerance=1.0e-3,
        max_order=256,
        probe_rounds=1,
        seed=seed,
    )
    return basis, summary


def holdout_samples(domain, top, *, count: int, seed: int):
    """Matrix-parameter holdouts with the exported interface left unclosed."""
    del domain, top
    rng = np.random.default_rng(seed)
    shifts = np.r_[0.0, 10.0 ** rng.uniform(-3.0, 2.0, max(1, count - 1))]
    rng.shuffle(shifts)
    result = []
    for i in range(count):
        top_h = float(10.0 ** rng.uniform(0.0, 4.0))
        # interface_h=0 is the exported/embedded condition: the artificial
        # Robin training term is absent and an external model supplies flux.
        result.append(TransferSample(top_h, 0.0, float(shifts[i])))
    return tuple(result)


def make_probe_vectors(transfer, *, count: int, seed: int):
    rng = np.random.default_rng(seed)
    m = transfer.uniform_normalized_mode.size

    isotropic = rng.standard_normal((m, count))
    isotropic /= np.linalg.norm(isotropic, axis=0)

    directions = np.column_stack(
        (transfer.uniform_normalized_mode, transfer.extra_normalized_modes)
    )
    gains = np.r_[transfer.uniform_gain, transfer.extra_eigenvalues]
    weights = np.sqrt(gains / max(gains[0], np.finfo(float).tiny))
    coeff = rng.standard_normal((directions.shape[1], count))
    coeff *= weights[:, None]
    weighted = directions @ coeff
    weighted /= np.linalg.norm(weighted, axis=0)
    return {"transfer_weighted": weighted, "isotropic": isotropic}


def precompute_references(domain, top, interface, transfer, *, count: int, seed: int):
    B = normalized_boundary_injection(domain.n_cells, interface).tocsr()
    probes = make_probe_vectors(transfer, count=count, seed=seed + 1)
    params = holdout_samples(domain, top, count=count, seed=seed + 2)
    refs = {name: [] for name in probes}

    for name, vectors in probes.items():
        for i, sample in enumerate(params):
            A = affine_operator(domain, top, interface, sample)
            b = np.asarray(B @ vectors[:, i]).ravel()
            x = AMGPCGSolver(A).solve(b)
            refs[name].append((A, b, x))
    return refs


def basis_metrics(domain, basis, references):
    C = domain.C.tocsr()
    state_errors = []
    for A, b, x in references:
        AV = A @ basis
        Ar = np.asarray(basis.T @ AV, dtype=float)
        br = np.asarray(basis.T @ b, dtype=float)
        theta = scipy.linalg.solve(Ar, br, assume_a="pos", check_finite=False)
        xr = np.asarray(basis @ theta).ravel()
        error = xr - x
        num = float(error @ (C @ error))
        den = float(x @ (C @ x))
        state_errors.append(float(np.sqrt(num / max(den, np.finfo(float).tiny))))

    values = np.asarray(state_errors, dtype=float)
    return {
        "relative_C_error": {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    }


def run(max_extra_modes: int, probe_count: int, seed: int) -> dict:
    domain = build_simple_cube()
    top = boundary_port(domain, "top")
    interface = boundary_port(domain, "bottom")
    transfer = transfer_boundary_basis(
        domain,
        top,
        interface,
        samples=default_samples(),
        max_extra_modes=max_extra_modes,
    )
    references = precompute_references(
        domain,
        top,
        interface,
        transfer,
        count=probe_count,
        seed=seed,
    )

    stages = []
    for extra_boundary_modes in range(max_extra_modes + 1):
        rhs = boundary_training_rhs(transfer, extra_boundary_modes)
        basis, summary = extract_fantastic(
            domain,
            top,
            interface,
            rhs,
            seed=seed,
        )
        stages.append(
            {
                "extra_boundary_modes": extra_boundary_modes,
                "boundary_training_modes": int(rhs.shape[1]),
                "rom_order": int(basis.shape[1]),
                "pre_svd_order": int(summary["pre_svd_order"]),
                "max_accepted_fantastic_residual": float(
                    summary["max_accepted_residual"]
                ),
                "holdout": {
                    name: basis_metrics(domain, basis, refs)
                    for name, refs in references.items()
                },
            }
        )

    ratios = transfer.extra_eigenvalues / max(
        transfer.uniform_gain, np.finfo(float).tiny
    )
    return {
        "full_order_dofs": int(domain.n_cells),
        "interface_dofs": int(interface.ids.size),
        "default_extra_boundary_modes": 0,
        "default_boundary_training_modes": 1,
        "uniform_transfer_gain": transfer.uniform_gain,
        "extra_transfer_eigenvalue_ratios_to_uniform_gain": ratios.tolist(),
        "probe_count_per_ensemble": int(probe_count),
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-extra-modes", type=int, default=4)
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = run(args.max_extra_modes, args.probe_count, args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
