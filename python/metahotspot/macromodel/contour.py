"""Polynomial flux training directions on planar patches."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import Legendre


def _contour_orders(degree: int) -> tuple[tuple[int, int], ...]:
    if degree < 0:
        raise ValueError("interface_contour_degree must be non-negative")
    return tuple(
        (i, total - i) for total in range(degree + 1) for i in range(total + 1)
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
        result[:, n] = np.sqrt((2.0 * n + 1.0) / length) * (length / 2.0) * integral_u
    return result


def _port_contour_integrals(rects: np.ndarray, degree: int) -> tuple[np.ndarray, tuple]:
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


def contour_training_sources(n, patches, degree) -> tuple[np.ndarray, list[dict]]:
    """Integrate contour modes on planar patches and accumulate by owning cell."""
    columns: list[np.ndarray] = []
    metadata: list[dict] = []
    for label, cells, rects in patches:
        integrals, orders = _port_contour_integrals(rects, degree)
        threshold = (
            np.finfo(float).eps * max(integrals.shape) * np.linalg.norm(integrals[:, 0])
        )
        for local_col, order in enumerate(orders):
            g = np.zeros(n, dtype=np.float64)
            np.add.at(g, cells, integrals[:, local_col])
            norm = float(np.linalg.norm(g))
            if norm <= threshold:
                continue
            g /= norm
            columns.append(g)
            metadata.append(
                {
                    "port": label,
                    "polynomial_order": [int(order[0]), int(order[1])],
                }
            )
    if not columns:
        return np.empty((n, 0), dtype=np.float64), metadata
    return np.ascontiguousarray(np.column_stack(columns)), metadata
