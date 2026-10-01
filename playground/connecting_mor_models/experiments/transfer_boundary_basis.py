#!/usr/bin/env python3
"""Matrix-only transfer-optimal boundary basis diagnostic.

This experiment intentionally separates *which boundary inputs should enrich the
state basis* from *how an exported boundary trace is represented*. No contour
basis, polynomial surface basis, partition, or coordinate-dependent boundary
mode enters the transfer calculation.

For a full-order thermal system

    A_mu x = B_gamma q,       A_mu = K + sum_j p_j H_j + s C,

we measure boundary heat-flow inputs in the natural half-cell conductance norm
``q.T @ G_gamma^{-1} @ q``. With ``q = G_gamma**1/2 z`` the normalized input
operator is ``B = P_gamma.T @ G_gamma**1/2``. The C-weighted transfer Gramian is

    T = sum_mu w_mu B.T A_mu^{-1} C A_mu^{-1} B.

Every embeddable ROM is trained with one explicit uniform boundary heat-flow
*density* direction: integrated face heat flow is proportional to face area.
The user-facing ``extra_boundary_modes`` count then selects dominant transfer
modes from the orthogonal complement of that uniform direction. Therefore
``extra_boundary_modes=0`` means "uniform boundary heat flow only", never an
adiabatic/source-only training policy.

The transfer modes are training-only source directions for Extended FANTASTIC;
they are not an exported boundary representation.
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
    """Uniform baseline plus dominant transfer-optimal extra boundary directions."""

    uniform_gain: float
    uniform_normalized_mode: np.ndarray
    uniform_heat_flow_mode: np.ndarray
    uniform_rhs_mode: np.ndarray
    extra_eigenvalues: np.ndarray
    extra_normalized_modes: np.ndarray
    extra_heat_flow_modes: np.ndarray
    extra_rhs_modes: np.ndarray


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
    norm ``q.T @ G_gamma^{-1} @ q``. Writing ``q = G_gamma**1/2 z`` makes the
    normalized coordinate ``z`` Euclidean, hence the injection below.
    """
    g = np.asarray(port.half_conductance, dtype=float)
    m = g.size
    return sp.csc_matrix(
        (np.sqrt(g), (np.asarray(port.ids, dtype=np.int64), np.arange(m))),
        shape=(n_cells, m),
    )


def uniform_boundary_mode(port) -> tuple[np.ndarray, np.ndarray]:
    """Normalized constant-flux-density boundary heat-flow direction.

    ``q_face`` is integrated heat flow, so a spatially uniform heat-flow density
    has ``q_face proportional to face area``. The returned normalized coordinate
    has unit Euclidean norm after ``q = G_gamma**1/2 z``.
    """
    g = np.asarray(port.half_conductance, dtype=float)
    area = np.asarray(port.area, dtype=float)
    z = area / np.sqrt(g)
    norm = float(np.linalg.norm(z))
    if norm <= np.finfo(float).tiny:
        raise ValueError("uniform boundary heat-flow direction has zero norm")
    z /= norm
    q = np.sqrt(g) * z
    return z, q


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

    The values are not boundary basis functions. They merely evaluate the
    matrix transfer operator at low/mid/high thermal time scales and at two
    crossed Robin corners. The boundary modes themselves depend only on the
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
    max_extra_modes: int = 8,
    eig_tol: float = 1.0e-7,
) -> TransferBasis:
    """Compute the uniform baseline plus C-weighted transfer-optimal extra modes.

    Extra eigenvectors are computed in the Euclidean orthogonal complement of
    the normalized uniform boundary direction. Thus they are optimal *given
    that the uniform heat-flow mode is always already part of training*.
    """
    if not samples:
        raise ValueError("at least one transfer sample is required")
    B = normalized_boundary_injection(domain.n_cells, interface_port).tocsr()
    m = B.shape[1]
    if not 0 <= max_extra_modes < m:
        raise ValueError(f"max_extra_modes must satisfy 0 <= value < {m}")

    uniform_z, uniform_q = uniform_boundary_mode(interface_port)
    uniform_rhs = np.asarray(B @ uniform_z).ravel()
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

    uniform_gain = float(uniform_z @ gramian_matvec(uniform_z))
    if max_extra_modes == 0:
        empty_modes = np.empty((m, 0), dtype=float)
        empty_rhs = np.empty((domain.n_cells, 0), dtype=float)
        return TransferBasis(
            uniform_gain,
            uniform_z,
            uniform_q,
            uniform_rhs,
            np.empty(0, dtype=float),
            empty_modes,
            empty_modes.copy(),
            empty_rhs,
        )

    def project(z):
        z = np.asarray(z, dtype=float)
        return z - uniform_z * float(uniform_z @ z)

    def projected_gramian_matvec(z):
        z_perp = project(z)
        return project(gramian_matvec(z_perp))

    gramian = spla.LinearOperator(
        (m, m), matvec=projected_gramian_matvec, dtype=float
    )
    rng = np.random.default_rng(20260916)
    v0 = project(rng.standard_normal(m))
    v0 /= np.linalg.norm(v0)
    values, modes = spla.eigsh(
        gramian,
        k=max_extra_modes,
        which="LA",
        tol=eig_tol,
        v0=v0,
    )
    order = np.argsort(values)[::-1]
    values = np.maximum(np.asarray(values[order], dtype=float), 0.0)
    modes = np.asarray(modes[:, order], dtype=float)
    modes -= uniform_z[:, None] * (uniform_z @ modes)[None, :]
    modes /= np.linalg.norm(modes, axis=0)[None, :]

    g_sqrt = np.sqrt(np.asarray(interface_port.half_conductance, dtype=float))
    heat_flow_modes = g_sqrt[:, None] * modes
    rhs_modes = np.asarray(B @ modes, dtype=float)
    return TransferBasis(
        uniform_gain,
        uniform_z,
        uniform_q,
        uniform_rhs,
        values,
        modes,
        heat_flow_modes,
        rhs_modes,
    )


def boundary_training_rhs(
    transfer: TransferBasis, extra_boundary_modes: int
) -> np.ndarray:
    """Return uniform boundary training plus the requested number of extras."""
    available = transfer.extra_rhs_modes.shape[1]
    if not 0 <= extra_boundary_modes <= available:
        raise ValueError(
            "extra_boundary_modes must satisfy "
            f"0 <= value <= {available}; got {extra_boundary_modes}"
        )
    uniform = transfer.uniform_rhs_mode[:, None]
    if extra_boundary_modes == 0:
        return np.ascontiguousarray(uniform)
    return np.ascontiguousarray(
        np.column_stack(
            (uniform, transfer.extra_rhs_modes[:, :extra_boundary_modes])
        )
    )


def normalized_overlap(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def diagnostic(max_extra_modes: int) -> dict:
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

    uniform_flux = np.asarray(interface.area, dtype=float)
    uniform_overlap = normalized_overlap(transfer.uniform_heat_flow_mode, uniform_flux)
    if transfer.extra_eigenvalues.size:
        extra_ratios = transfer.extra_eigenvalues / max(
            transfer.uniform_gain, np.finfo(float).tiny
        )
        extra_uniform_overlaps = [
            normalized_overlap(
                transfer.extra_normalized_modes[:, i],
                transfer.uniform_normalized_mode,
            )
            for i in range(transfer.extra_normalized_modes.shape[1])
        ]
    else:
        extra_ratios = np.empty(0, dtype=float)
        extra_uniform_overlaps = []

    return {
        "full_order_dofs": int(domain.n_cells),
        "interface_dofs": int(interface.ids.size),
        "sample_count": len(default_samples()),
        "default_extra_boundary_modes": 0,
        "default_boundary_training_modes": 1,
        "uniform_transfer_gain": transfer.uniform_gain,
        "uniform_mode_uniform_flux_overlap": uniform_overlap,
        "extra_eigenvalues": transfer.extra_eigenvalues.tolist(),
        "extra_eigenvalue_ratios_to_uniform_gain": extra_ratios.tolist(),
        "extra_mode_uniform_coordinate_overlaps": extra_uniform_overlaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-extra-modes", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = diagnostic(args.max_extra_modes)
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
