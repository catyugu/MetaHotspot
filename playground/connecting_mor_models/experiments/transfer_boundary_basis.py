#!/usr/bin/env python3
"""Matrix-only transfer-optimal boundary basis diagnostic.

This experiment intentionally separates *which boundary inputs should enrich the
state basis* from *how an exported boundary trace is represented*.  No contour
basis, polynomial surface basis, partition, or coordinate-dependent boundary
mode enters the transfer calculation.

For a full-order thermal system

    A_mu x = B_gamma q,       A_mu = K + sum_j p_j H_j + s C,

we measure boundary heat-flow inputs in the natural half-cell conductance norm
``q.T @ G_gamma^{-1} @ q``.  With ``q = G_gamma**1/2 z`` the normalized input
operator is ``B = P_gamma.T @ G_gamma**1/2``.  The C-weighted transfer Gramian is

    T = sum_mu w_mu B.T A_mu^{-1} C A_mu^{-1} B.

Its dominant eigenvectors are the boundary input directions that produce the
largest bulk thermal response per unit boundary-input norm.  They are intended
as *training-only* source directions for Extended FANTASTIC; they are not an
exported boundary representation.

Running this file builds the same 100 mm copper-cube discretization used by
``playground/simple_erom_case1`` with the repository's pure-Python FVM helper,
then reports the transfer spectrum and the overlap of the first transfer mode
with the hand-written uniform-flux direction used by the current simple EROM
experiment.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mor_common import AMGPCGSolver, SourceBox, boundary_port, build_domain


COPPER_K = 385.0
COPPER_RHO = 8930.0
COPPER_CP = 385.0
PHYSICAL_H_RANGE = (1.0, 1.0e4)


@dataclass(frozen=True)
class TransferSample:
    """One affine/frequency point used only to define the boundary transfer map."""

    top_h: float
    interface_h: float
    shift: float


@dataclass(frozen=True)
class TransferBasis:
    """Dominant normalized and physical boundary directions."""

    eigenvalues: np.ndarray
    normalized_modes: np.ndarray
    heat_flow_modes: np.ndarray
    rhs_modes: np.ndarray


def cube_axis_m() -> np.ndarray:
    """The fixed 15-cell axis used by ``simple_erom_case1`` in SI metres."""
    mm = np.r_[
        np.linspace(0.0, 25.0, 5),
        np.linspace(25.0, 75.0, 8)[1:],
        np.linspace(75.0, 100.0, 5)[1:],
    ]
    return mm * 1.0e-3


def build_simple_cube():
    """Pure-Python FVM reconstruction of the reusable simple-EROM cube."""

    def copper(_x, _y, _z):
        return COPPER_K, COPPER_RHO, COPPER_CP, 0

    axis = cube_axis_m()
    source = SourceBox("cube", 0.025, 0.075, 0.025, 0.075, 0.025, 0.075)
    return build_domain(
        "simple_erom_cube",
        axis,
        axis,
        axis,
        copper,
        sources=(source,),
        source_powers=(100.0,),
    )


def surface_term(n_cells: int, port) -> sp.csc_matrix:
    """Affine Robin term ``H = diag(face area)`` for one boundary port."""
    return sp.csc_matrix(
        (np.asarray(port.area, dtype=float), (port.ids, port.ids)),
        shape=(n_cells, n_cells),
    )


def effective_surface_coefficient(domain, port, physical_h: float) -> float:
    """Area-weighted series-condensed coefficient used by the BCI affine term."""
    ids = np.asarray(port.ids, dtype=np.int64)
    area = np.asarray(port.area, dtype=float)
    g = np.asarray(port.half_conductance, dtype=float)
    k = np.asarray(domain.k[ids], dtype=float)
    half = k * area / g
    h = float(physical_h)
    p = k * h / (k + h * half)
    return float(np.dot(area, p) / area.sum())


def normalized_boundary_injection(n_cells: int, port) -> sp.csc_matrix:
    """Return ``P_gamma.T @ G_gamma**1/2`` for heat-flow input coordinates.

    A physical face heat-flow vector ``q`` is measured in the dual interface
    norm ``q.T @ G_gamma^{-1} @ q``.  Writing ``q = G_gamma**1/2 z`` makes the
    normalized coordinate ``z`` Euclidean, hence the injection below.
    """
    g = np.asarray(port.half_conductance, dtype=float)
    m = g.size
    return sp.csc_matrix(
        (np.sqrt(g), (np.asarray(port.ids, dtype=np.int64), np.arange(m))),
        shape=(n_cells, m),
    )


def affine_operator(domain, top_port, interface_port, sample: TransferSample):
    """Assemble one SPD matrix ``K + p_top H_top + p_if H_if + s C``."""
    H_top = surface_term(domain.n_cells, top_port)
    H_if = surface_term(domain.n_cells, interface_port)
    p_top = effective_surface_coefficient(domain, top_port, sample.top_h)
    p_if = effective_surface_coefficient(domain, interface_port, sample.interface_h)
    return (
        domain.K
        + p_top * H_top
        + p_if * H_if
        + float(sample.shift) * domain.C
    ).tocsc()


def default_samples() -> tuple[TransferSample, ...]:
    """Small deterministic affine/frequency cover for the diagnostic.

    The values are not boundary basis functions.  They merely evaluate the
    matrix transfer operator at low/mid/high thermal time scales and at two
    crossed Robin corners.  The boundary modes themselves depend only on the
    assembled matrices.
    """
    return (
        TransferSample(100.0, 100.0, 0.0),
        TransferSample(1000.0, 1000.0, 0.1),
        TransferSample(10000.0, 10000.0, 10.0),
        TransferSample(100.0, 10000.0, 0.1),
        TransferSample(10000.0, 100.0, 0.1),
    )


def transfer_boundary_basis(
    domain,
    top_port,
    interface_port,
    *,
    samples: tuple[TransferSample, ...],
    max_modes: int = 8,
    eig_tol: float = 1.0e-7,
) -> TransferBasis:
    """Compute dominant C-weighted boundary transfer directions matrix-free."""
    if not samples:
        raise ValueError("at least one transfer sample is required")
    B = normalized_boundary_injection(domain.n_cells, interface_port).tocsr()
    m = B.shape[1]
    if not 1 <= max_modes < m:
        raise ValueError(f"max_modes must satisfy 1 <= max_modes < {m}")

    solvers = [
        AMGPCGSolver(affine_operator(domain, top_port, interface_port, sample))
        for sample in samples
    ]
    C = domain.C.tocsr()
    weight = 1.0 / len(solvers)

    def gramian_matvec(z):
        z = np.asarray(z, dtype=float)
        rhs = np.asarray(B @ z).ravel()
        result = np.zeros(m, dtype=float)
        for solver in solvers:
            x = solver.solve(rhs)
            adjoint_rhs = np.asarray(C @ x).ravel()
            y = solver.solve(adjoint_rhs)
            result += weight * np.asarray(B.T @ y).ravel()
        return result

    gramian = spla.LinearOperator((m, m), matvec=gramian_matvec, dtype=float)
    values, modes = spla.eigsh(
        gramian,
        k=max_modes,
        which="LA",
        tol=eig_tol,
        v0=np.ones(m, dtype=float) / np.sqrt(m),
    )
    order = np.argsort(values)[::-1]
    values = np.maximum(np.asarray(values[order], dtype=float), 0.0)
    modes = np.asarray(modes[:, order], dtype=float)

    g_sqrt = np.sqrt(np.asarray(interface_port.half_conductance, dtype=float))
    heat_flow_modes = g_sqrt[:, None] * modes
    rhs_modes = np.asarray(B @ modes, dtype=float)
    return TransferBasis(values, modes, heat_flow_modes, rhs_modes)


def eigen_clusters(values: np.ndarray, relative_gap: float = 5.0e-3) -> list[list[int]]:
    """Group nearly degenerate adjacent transfer eigenvalues.

    Keeping an entire cluster avoids an orientation-dependent truncation inside
    a repeated or nearly repeated eigenspace.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size:
        return []
    clusters = [[0]]
    for i in range(1, values.size):
        scale = max(abs(values[i - 1]), abs(values[i]), np.finfo(float).tiny)
        gap = abs(values[i - 1] - values[i]) / scale
        if gap <= relative_gap:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def normalized_overlap(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def diagnostic(max_modes: int) -> dict:
    domain = build_simple_cube()
    top = boundary_port(domain, "top")
    interface = boundary_port(domain, "bottom")
    transfer = transfer_boundary_basis(
        domain,
        top,
        interface,
        samples=default_samples(),
        max_modes=max_modes,
    )
    ratios = transfer.eigenvalues / transfer.eigenvalues[0]
    captured = np.cumsum(transfer.eigenvalues) / transfer.eigenvalues.sum()

    # Constant heat-flux density gives per-face heat flow proportional to area.
    uniform_flux = np.asarray(interface.area, dtype=float)
    first_overlap = normalized_overlap(transfer.heat_flow_modes[:, 0], uniform_flux)
    clusters = eigen_clusters(transfer.eigenvalues)

    return {
        "full_order_dofs": int(domain.n_cells),
        "interface_dofs": int(interface.ids.size),
        "sample_count": len(default_samples()),
        "eigenvalues": transfer.eigenvalues.tolist(),
        "eigenvalue_ratios": ratios.tolist(),
        "captured_energy_over_computed_modes": captured.tolist(),
        "clusters": clusters,
        "first_mode_uniform_flux_overlap": first_overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-modes", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = diagnostic(args.max_modes)
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
