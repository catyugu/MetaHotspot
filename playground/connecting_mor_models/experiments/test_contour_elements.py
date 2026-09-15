#!/usr/bin/env python3
"""Numerical checks for the THERMINIC-2023 contour-element implementation."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss

from mor_common import ContourElements


def uniform_rects(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-0.7, 1.3, nx + 1)
    y = np.linspace(0.2, 2.7, ny + 1)
    return np.asarray(
        [(x[i], x[i + 1], y[j], y[j + 1]) for i in range(nx) for j in range(ny)]
    )


def numerical_integrals(elements: ContourElements, rects: np.ndarray) -> np.ndarray:
    """Independent tensor Gauss quadrature for exact polynomial integration."""
    nodes, weights = leggauss(elements.degree + 2)
    xlo, xhi, ylo, yhi = elements.bounds
    lx, ly = xhi - xlo, yhi - ylo
    result = np.empty((rects.shape[0], elements.size))
    for r, (xa, xb, ya, yb) in enumerate(rects):
        x = 0.5 * (xa + xb) + 0.5 * (xb - xa) * nodes
        y = 0.5 * (ya + yb) + 0.5 * (yb - ya) * nodes
        wx = 0.5 * (xb - xa) * weights
        wy = 0.5 * (yb - ya) * weights
        ux = 2.0 * (x - xlo) / lx - 1.0
        uy = 2.0 * (y - ylo) / ly - 1.0
        for c, (i, j) in enumerate(elements.orders):
            px = np.sqrt((2.0 * i + 1.0) / lx) * Legendre.basis(i)(ux)
            py = np.sqrt((2.0 * j + 1.0) / ly) * Legendre.basis(j)(uy)
            result[r, c] = (wx @ px) * (wy @ py)
    return result


def main() -> None:
    rects = uniform_rects(11, 9)
    degree = 4
    elements = ContourElements.from_rects(rects, degree)
    assert elements.size == (degree + 1) * (degree + 2) // 2

    analytic = elements.integrals(rects)
    numeric = numerical_integrals(elements, rects)
    integral_error = float(np.max(np.abs(analytic - numeric)))
    if integral_error > 2.0e-14:
        raise AssertionError(f"contour rectangle integral error: {integral_error:.3e}")

    area = (rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2])
    averages = elements.averages(rects)
    if not np.allclose(analytic, area[:, None] * averages, rtol=2e-14, atol=2e-14):
        raise AssertionError("contour rectangle integral/average identity failed")

    # Integrating the continuous orthonormal basis over the whole surface gives
    # sqrt(area) for the constant mode and zero for every higher mode.
    total = analytic.sum(axis=0)
    surface_area = (1.3 - (-0.7)) * (2.7 - 0.2)
    expected = np.zeros(elements.size)
    expected[0] = np.sqrt(surface_area)
    if not np.allclose(total, expected, rtol=0.0, atol=3.0e-14):
        raise AssertionError("global contour normalization/integral check failed")

    # Independent Gauss quadrature of pairwise products checks the claimed L2
    # orthonormality of the continuous Legendre-product contour basis.
    nodes, weights = leggauss(2 * degree + 3)
    x = 0.5 * ((-0.7) + 1.3) + 0.5 * (1.3 - (-0.7)) * nodes
    y = 0.5 * (0.2 + 2.7) + 0.5 * (2.7 - 0.2) * nodes
    wx = 0.5 * (1.3 - (-0.7)) * weights
    wy = 0.5 * (2.7 - 0.2) * weights
    ux = 2.0 * (x - (-0.7)) / (1.3 - (-0.7)) - 1.0
    uy = 2.0 * (y - 0.2) / (2.7 - 0.2) - 1.0
    values = []
    for i, j in elements.orders:
        px = np.sqrt((2.0 * i + 1.0) / (1.3 - (-0.7))) * Legendre.basis(i)(ux)
        py = np.sqrt((2.0 * j + 1.0) / (2.7 - 0.2)) * Legendre.basis(j)(uy)
        values.append(np.outer(px, py))
    gram = np.empty((elements.size, elements.size))
    W = np.outer(wx, wy)
    for i in range(elements.size):
        for j in range(elements.size):
            gram[i, j] = np.sum(W * values[i] * values[j])
    gram_error = float(np.max(np.abs(gram - np.eye(elements.size))))
    if gram_error > 2.0e-14:
        raise AssertionError(f"continuous contour L2 Gram error: {gram_error:.3e}")

    print(
        f"contour elements: degree={degree}, modes={elements.size}, "
        f"integral max error={integral_error:.3e}, Gram max error={gram_error:.3e}"
    )


if __name__ == "__main__":
    main()
