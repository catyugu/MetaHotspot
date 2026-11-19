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
from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from metahotspot.macromodel.contour import contour_training_sources

from legacy_embeddable import EmbeddableRom, Subdomain, _project_rom
from metahotspot.macromodel.utils import (
    build_parametric_basis,
    normalized_operators,
)


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
    ports = _selected_ports(subdomain, interface_labels)
    return contour_training_sources(
        subdomain.cells.size, ((p.label, p.cells, p.rects) for p in ports), degree
    )


def _active_ambient_data(subdomain: Subdomain):
    """Drop affine groups whose restriction to this subdomain is identically zero."""
    active_indices: list[int] = []
    terms: list[sp.csc_matrix] = []
    ranges: list[list[float]] = []
    effective: list[float] = []
    for idx, (term, h_range) in enumerate(
        zip(subdomain.ambient_terms, subdomain.ambient_ranges, strict=True)
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

    active_indices, boundary_terms, ambient_ranges, effective_p = _active_ambient_data(
        subdomain
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
        h_ranges.append([float(interface_h_range[0]), float(interface_h_range[1])])

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

    projected_domain = replace(
        subdomain,
        ambient_terms=boundary_terms[: len(active_indices)],
        ambient_ranges=ambient_ranges,
        effective_p=effective_p,
    )
    return _project_rom(projected_domain, basis, summary)


def clone_embeddable_rom(rom: EmbeddableRom) -> EmbeddableRom:
    """Clone all mutable ROM data for independent per-case solves."""
    return copy.deepcopy(rom)
