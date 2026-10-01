"""Interface-response enrichment for embeddable thermal ROM extraction.

The baseline :func:`metahotspot.macromodel.embeddable.extract_rom` trains the
Extended-FANTASTIC state space from physical heat-source ports while varying a
single scalar Robin coefficient over the connectable interface.  That is enough
for source-driven responses, but it does not explicitly train the state space
against spatially varying heat injection from an attached model.

This module adds low-order orthonormal polynomial heat-flux directions on each
connectable interface as *training-only* ports.  They enrich the state basis but
are not exported as physical heat sources.  The idea mirrors the contour-element
representation used for BCI boundary fields: a small polynomial trace/flux space
is used to cover boundary-driven response directions reproducibly.
"""

from __future__ import annotations

import copy

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from numpy.polynomial.legendre import Legendre

from metahotspot.macromodel.embeddable import EmbeddableRom, Subdomain
from metahotspot.macromodel.utils import (
    build_parametric_basis,
    normalized_operators,
)


def _contour_orders(degree: int) -> tuple[tuple[int, int], ...]:
    if degree < 0:
        raise ValueError("interface_contour_degree must be non-negative")
    return tuple(
        (i, total - i)
        for total in range(degree + 1)
        for i in range(total + 1)
    )


def _integrals_1d(
    intervals: np.ndarray, lo: float, hi: float, max_order: int
) -> np.ndarray:
    length = float(hi - lo)
    if length <= 0.0:
        raise ValueError("interface contour bounds must have positive extent")
    u0 = 2.0 * (intervals[:, 0] - lo) / length - 1.0
    u1 = 2.0 * (intervals[:, 1] - lo) / length - 1.0
    result = np.empty((intervals.shape[0], max_order + 1), dtype=np.float64)
    for n in range(max_order + 1):
        anti = Legendre.basis(n).integ()
        integral_u = anti(u1) - anti(u0)
        result[:, n] = (
            np.sqrt((2.0 * n + 1.0) / length) * (length / 2.0) * integral_u
        )
    return result


def _port_contour_integrals(
    rects: np.ndarray, degree: int
) -> tuple[np.ndarray, tuple]:
    """Exact face integrals of a continuous L2-orthonormal contour basis."""
    rects = np.asarray(rects, dtype=np.float64)
    if rects.ndim != 2 or rects.shape[1] != 4 or rects.shape[0] == 0:
        raise ValueError("interface rects must have shape (n, 4)")
    orders = _contour_orders(degree)
    xlo, xhi = float(rects[:, 0].min()), float(rects[:, 1].max())
    ylo, yhi = float(rects[:, 2].min()), float(rects[:, 3].max())
    max_i = max(i for i, _ in orders)
    max_j = max(j for _, j in orders)
    ix = _integrals_1d(rects[:, :2], xlo, xhi, max_i)
    iy = _integrals_1d(rects[:, 2:], ylo, yhi, max_j)
    return np.column_stack([ix[:, i] * iy[:, j] for i, j in orders]), orders


def _selected_ports(subdomain: Subdomain, interface_labels):
    if interface_labels is None:
        return list(subdomain.ports)
    labels = tuple(interface_labels)
    available = {p.label for p in subdomain.ports}
    missing = sorted(set(labels) - available)
    if missing:
        raise KeyError(f"unknown interface port labels: {missing}")
    return [p for p in subdomain.ports if p.label in labels]


def interface_flux_training_sources(
    subdomain: Subdomain,
    *,
    degree: int = 2,
    interface_labels=None,
) -> tuple[np.ndarray, list[dict]]:
    """Build signed unit-norm interface heat-flux directions for training only.

    For each selected connectable face port and each total-degree polynomial
    contour mode, the full-order right-hand side receives
    ``integral_face psi dA`` on the owning boundary cell.  The columns are
    normalized only for numerical conditioning; response-space span is
    invariant to that scaling.
    """
    n = subdomain.cells.size
    columns: list[np.ndarray] = []
    metadata: list[dict] = []
    for port in _selected_ports(subdomain, interface_labels):
        integrals, orders = _port_contour_integrals(port.rects, degree)
        cells = np.asarray(port.cells, dtype=np.int64)
        for local_col, order in enumerate(orders):
            g = np.zeros(n, dtype=np.float64)
            np.add.at(g, cells, integrals[:, local_col])
            norm = float(np.linalg.norm(g))
            if norm <= np.finfo(float).tiny:
                continue
            g /= norm
            columns.append(g)
            metadata.append(
                {
                    "port": port.label,
                    "polynomial_order": [int(order[0]), int(order[1])],
                }
            )
    if not columns:
        return np.empty((n, 0), dtype=np.float64), metadata
    return np.ascontiguousarray(np.column_stack(columns)), metadata


def _active_ambient_data(subdomain: Subdomain):
    """Drop affine groups whose restriction to this subdomain is identically zero."""
    active_indices: list[int] = []
    terms: list[sp.csc_matrix] = []
    ranges: list[list[float]] = []
    effective: list[float] = []
    for idx, (term, h_range) in enumerate(
        zip(subdomain.ambient_terms, subdomain.ambient_ranges)
    ):
        diagonal = np.asarray(term.diagonal(), dtype=np.float64)
        if not np.any(np.abs(diagonal) > 0.0):
            continue
        active_indices.append(idx)
        terms.append(sp.csc_matrix(term))
        ranges.append([float(h_range[0]), float(h_range[1])])
        if np.asarray(subdomain.effective_p).size:
            effective.append(float(subdomain.effective_p[idx]))
    return (
        active_indices,
        terms,
        np.asarray(ranges, dtype=np.float64).reshape((-1, 2)),
        np.asarray(effective, dtype=np.float64),
    )


def extract_interface_enriched_rom(
    subdomain: Subdomain,
    *,
    tolerance: float = 1.0e-3,
    max_order: int = 512,
    probe_rounds: int = 3,
    seed: int = 20260915,
    interface_h_range: tuple[float, float] = (1.0, 1.0e4),
    interface_contour_degree: int = 2,
    interface_labels=None,
) -> EmbeddableRom:
    """Extract one reusable EROM enriched for spatial interface excitation.

    Physical source ports remain the only exported ``F_hat`` columns.  Contour
    flux ports exist solely during Extended-FANTASTIC basis construction.
    Zero-support ambient affine terms are removed before random parameter
    sampling, making the extraction invariant to boundary groups that live only
    on the attached external body.
    """
    cells = np.arange(subdomain.cells.size, dtype=np.int64)
    ops = normalized_operators(subdomain.K, subdomain.C, np.zeros(cells.size))
    physical_sources = np.asarray(subdomain.source, dtype=np.float64)
    if physical_sources.ndim != 2 or physical_sources.shape[1] == 0:
        raise ValueError(
            "interface-enriched extraction requires a physical source port"
        )

    active_indices, boundary_terms, ambient_ranges, effective_p = (
        _active_ambient_data(subdomain)
    )
    h_ranges = [list(r) for r in ambient_ranges]

    training_ports = _selected_ports(subdomain, interface_labels)
    interface_areas = np.zeros(cells.size, dtype=np.float64)
    for port in training_ports:
        np.add.at(
            interface_areas,
            np.asarray(port.cells, dtype=np.int64),
            np.asarray(port.areas, dtype=np.float64),
        )
    if training_ports:
        boundary_terms.append(sp.diags(interface_areas, format="csc"))
        h_ranges.append(
            [float(interface_h_range[0]), float(interface_h_range[1])]
        )

    interface_sources, interface_metadata = interface_flux_training_sources(
        subdomain,
        degree=interface_contour_degree,
        interface_labels=interface_labels,
    )
    training_sources = np.ascontiguousarray(
        np.column_stack((physical_sources, interface_sources))
    )

    basis, summary = build_parametric_basis(
        ops,
        training_sources,
        boundary_terms,
        np.asarray(h_ranges, dtype=np.float64),
        tolerance=tolerance,
        max_order=max_order,
        probe_rounds=probe_rounds,
        seed=seed,
    )

    C_hat = sp.csc_matrix(basis.T @ ops.C @ basis)
    K0_hat = sp.csc_matrix(basis.T @ ops.K @ basis)
    F_hat = np.asarray(basis.T @ physical_sources, dtype=np.float64)
    ambient_hat = [
        sp.csc_matrix(basis.T @ H @ basis)
        for H in boundary_terms[: len(active_indices)]
    ]

    modal_k, modal_q = scipy.linalg.eigh(
        K0_hat.toarray(), C_hat.toarray(), check_finite=False
    )
    modal_q = np.asarray(modal_q, dtype=np.float64)
    basis = np.asarray(basis @ modal_q, dtype=np.float64)
    C_hat = sp.eye(modal_k.size, format="csc")
    K0_hat = sp.diags(modal_k, format="csc")
    F_hat = np.asarray(modal_q.T @ F_hat, dtype=np.float64)
    ambient_hat = [
        sp.csc_matrix(modal_q.T @ H.toarray() @ modal_q) for H in ambient_hat
    ]
    boundary_traces = {
        p.label: np.asarray(basis[np.asarray(p.cells), :], dtype=np.float64)
        for p in subdomain.boundary_ports
    }
    boundary_conductances = {
        p.label: np.asarray(p.g, dtype=np.float64)
        for p in subdomain.boundary_ports
    }

    summary = dict(summary)
    summary.update(
        {
            "physical_source_count": int(physical_sources.shape[1]),
            "interface_training_source_count": int(interface_sources.shape[1]),
            "training_source_count": int(training_sources.shape[1]),
            "interface_contour_degree": int(interface_contour_degree),
            "interface_labels": [p.label for p in training_ports],
            "interface_training_sources": interface_metadata,
            "active_ambient_indices": active_indices,
            "dropped_zero_ambient_count": int(
                len(subdomain.ambient_terms) - len(active_indices)
            ),
        }
    )

    return EmbeddableRom(
        name=subdomain.name,
        cells=subdomain.cells,
        basis=np.ascontiguousarray(basis),
        C_hat=C_hat,
        K0_hat=K0_hat,
        F_hat=np.asarray(F_hat, dtype=np.float64),
        ambient_hat=ambient_hat,
        boundary_traces=boundary_traces,
        boundary_conductances=boundary_conductances,
        ambient_ranges=ambient_ranges,
        effective_p=effective_p,
        ports=subdomain.ports,
        summary=summary,
    )


def clone_embeddable_rom(rom: EmbeddableRom) -> EmbeddableRom:
    """Deep-enough clone for per-case solves that mutate projected source scaling."""
    return EmbeddableRom(
        name=rom.name,
        cells=np.asarray(rom.cells, dtype=np.int64).copy(),
        basis=np.asarray(rom.basis, dtype=np.float64).copy(),
        C_hat=rom.C_hat.copy(),
        K0_hat=rom.K0_hat.copy(),
        F_hat=np.asarray(rom.F_hat, dtype=np.float64).copy(),
        ambient_hat=[H.copy() for H in rom.ambient_hat],
        boundary_traces={
            k: np.asarray(v, dtype=np.float64).copy()
            for k, v in rom.boundary_traces.items()
        },
        boundary_conductances={
            k: np.asarray(v, dtype=np.float64).copy()
            for k, v in rom.boundary_conductances.items()
        },
        ambient_ranges=np.asarray(rom.ambient_ranges, dtype=np.float64).copy(),
        effective_p=np.asarray(rom.effective_p, dtype=np.float64).copy(),
        ports=copy.deepcopy(rom.ports),
        summary=copy.deepcopy(rom.summary),
    )
