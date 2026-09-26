"""Exact parameter-dependent Galerkin output error for the affine HTC family.

The whole-box certificate of :mod:`certified_box` evaluates the residual of the
delivered basis in the fixed Riesz map ``A(h_min)^-1`` and transfers the result
to ``A(p)^-1`` by Loewner monotonicity.  That one substitution is the entire
source of the gap between its bound and the measured error: on Case 1 it costs
a factor of about ``2.3e1`` in the absolute bound and ``2.9e6`` in the
entrywise-relative bound, and the certificate pays 1024 factorisations for it
(one per cell corner, for the same-parameter denominator).

The substitution is unnecessary.  Every boundary term is a nonnegative
diagonal, so about the lower HTC corner

    A(p)   = A_ref + B diag(delta(p)) B^T,     A_ref = A(p_low),
    A(p)^-1 = A_ref^-1 - A_ref^-1 B (Delta(p)^-1 + Sigma)^-1 B^T A_ref^-1,
    Sigma  = B^T A_ref^-1 B                    (boundary block of the inverse),

and the residual of any trial space ``V`` lies in the fixed span
``Z = [G, (K + sC)V, H_1 V, ..., H_d V]``, i.e. ``r_i(p) = Z zeta_i(p)`` with
``zeta_i`` built from the small reduced solve at ``p``.  Hence

    (Y(p) - Y_V(p))_ij = r_i(p)^T A(p)^-1 r_j(p)
                       = zeta_i(p)^T [ Z^T A(p)^-1 Z ] zeta_j(p),

and the bracket follows from the two precomputed tables ``Z^T A_ref^-1 Z`` and
``Z^T A_ref^-1 B`` by dense algebra.  This is the exact Galerkin output error at
every ``p``, not an upper bound of it, and no term in it is an extraction solve.

Rigour over the continuous box comes from the same tables.  For a cell
``[low, high]`` and *any* trial ``q``, ``A(p) >= A(low)`` gives

    |Y(p) - Y_V(p)| <= r_q(p)^T A(low)^-1 r_q(p),

and with a constant trial the right-hand side is a convex quadratic in ``p``,
so its maximum over the cell is attained at a vertex.  ``cell_bound`` evaluates
exactly those vertices, so a grid maximum is bracketed by a cell-wide bound
that uses no additional full-order solve either.

Cost: one factorisation plus ``m_b + span_columns`` back-substitutions, once
per fixed shift.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from certified_box import BoxCertificate, logarithmic_edges, sparse_factor


class AffineErrorMap:
    """Exact steady output error of one fixed trial space on the HTC box.

    Parameters
    ----------
    kernel, terms:
        ``K`` and the boundary matrices ``H_i``; the reference operator is
        ``K + shift*C + sum_i p_low,i H_i``, so every ``H_i`` must be positive
        semidefinite for the box to be a nonnegative perturbation.
    source:
        ``(n, k)`` co-located source/output shape.
    ranges:
        ``(d, 2)`` HTC box.
    basis:
        ``(n, m)`` trial space; the error reported is the Galerkin error of
        this space, whatever produced it.
    shift, mass:
        fixed real frequency shift and the capacity matrix it needs.
    """

    def __init__(self, kernel, terms, source, ranges, basis, *, shift=0.0, mass=None):
        self.kernel = sp.csc_matrix(kernel)
        self.terms = [sp.csc_matrix(term) for term in terms]
        self.source = np.asarray(source, dtype=np.float64)
        self.ranges = np.asarray(ranges, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64)
        self.shift = float(shift)
        self.mass = None if mass is None else sp.csc_matrix(mass)
        self.dimension = len(self.terms)
        if self.ranges.shape != (self.dimension, 2):
            raise ValueError("ranges must have one row per boundary term")
        if self.basis.ndim != 2 or self.basis.shape[0] != self.kernel.shape[0]:
            raise ValueError("basis must match the kernel size")
        if self.shift and self.mass is None:
            raise ValueError("a nonzero shift requires the mass matrix")

        self.source_count = int(self.source.shape[1])
        self.order = int(self.basis.shape[1])
        self.base = self.kernel if not self.shift else (
            self.kernel + self.shift * self.mass
        )
        diagonal = sum(np.asarray(term.diagonal()).ravel() for term in self.terms)
        self.boundary = np.flatnonzero(diagonal > 0.0)
        self.weights = np.stack(
            [np.asarray(term.diagonal()).ravel()[self.boundary] for term in self.terms]
        )
        reference = sp.csc_matrix(self.base)
        for lower, term in zip(self.ranges[:, 0], self.terms):
            reference = reference + float(lower) * term
        self.factor = sparse_factor(reference)

        blocks = [self.source, self.base @ self.basis]
        blocks.extend(term @ self.basis for term in self.terms)
        self.span = np.ascontiguousarray(np.column_stack(blocks))
        self.span_columns = int(self.span.shape[1])
        # The Bernstein cell bound of certified_box only needs the cell's
        # Riesz Gram, which the Woodbury tables supply; sharing the class
        # keeps one implementation of the polynomial enclosure.
        self.certificate = BoxCertificate(
            self.kernel, self.terms, self.source, self.ranges, self.basis,
            shift=self.shift, mass=self.mass, blocks=(2,) * self.dimension,
        )

        self._prepare()

    def _prepare(self):
        """The two Woodbury tables and the reference Gram."""
        started = time.perf_counter()
        selector = np.zeros((self.kernel.shape[0], self.boundary.size))
        selector[self.boundary, np.arange(self.boundary.size)] = 1.0
        images = np.asarray(self.factor.solve(selector), dtype=np.float64)
        self.boundary_gram = images[self.boundary, :]
        self.boundary_gram = 0.5 * (self.boundary_gram + self.boundary_gram.T)
        self.span_boundary = np.ascontiguousarray(self.span.T @ images)
        gram = self.span.T @ self.factor.solve(self.span)
        self.span_gram = np.ascontiguousarray(0.5 * (gram + gram.T))
        self.preparation_seconds = time.perf_counter() - started

    # ------------------------------------------------------------- operators

    def operator(self, parameter):
        operator = self.base
        for value, term in zip(parameter, self.terms):
            operator = operator + float(value) * term
        return sp.csc_matrix(operator)

    def augmented_gram(self, parameter):
        """``Z^T A(parameter)^-1 Z`` from the tables, dense only."""
        delta = self.weights.T @ (np.asarray(parameter, float) - self.ranges[:, 0])
        active = delta > 0.0
        if not np.any(active):
            return self.span_gram
        matrix = self.boundary_gram[np.ix_(active, active)].copy()
        matrix[np.diag_indices_from(matrix)] += 1.0 / delta[active]
        solved = la.solve(
            matrix, self.span_boundary[:, active].T, assume_a="pos", check_finite=False
        )
        return self.span_gram - self.span_boundary[:, active] @ solved

    def residual_coefficients(self, parameter, coefficients=None):
        """Coefficient vectors of the residuals ``r_i(p) = Z zeta_i(p)``."""
        if coefficients is None:
            operator = self.operator(parameter)
            reduced = self.basis.T @ (operator @ self.basis)
            coefficients = [
                la.solve(
                    reduced,
                    self.basis.T @ self.source[:, port],
                    assume_a="pos",
                    check_finite=False,
                )
                for port in range(self.source_count)
            ]
        columns = []
        for port, values in enumerate(coefficients):
            zeta = np.zeros((self.span_columns, 1))
            zeta[port, 0] = 1.0
            zeta[self.source_count : self.source_count + self.order, 0] = -values
            start = self.source_count + self.order
            for axis, value in enumerate(parameter):
                zeta[start + axis * self.order : start + (axis + 1) * self.order, 0] = (
                    -float(value) * values
                )
            columns.append(zeta)
        return np.hstack(columns), coefficients

    # ------------------------------------------------------------ statements

    def error_matrix(self, parameter):
        """Exact ``Y(p) - Y_V(p)``, the full source-to-source transfer error."""
        zeta, _coefficients = self.residual_coefficients(parameter)
        return np.ascontiguousarray(zeta.T @ (self.augmented_gram(parameter) @ zeta))

    def absolute_error(self, parameter):
        """Exact entrywise absolute error at one parameter."""
        return np.abs(self.error_matrix(parameter))

    def transfer(self, parameter):
        """Exact full-order transfer at one parameter (reference only)."""
        return np.ascontiguousarray(
            self.source.T @ sparse_factor(self.operator(parameter)).solve(self.source)
        )

    def worst_on_grid(self, cells_per_axis=41):
        """Exact worst entrywise error over a uniform log grid of the box."""
        if np.isscalar(cells_per_axis):
            cells_per_axis = (int(cells_per_axis),) * self.dimension
        axes = [np.linspace(0.0, 1.0, int(count)) for count in cells_per_axis]
        mesh = np.meshgrid(*axes, indexing="ij")
        unit = np.column_stack([coordinate.ravel() for coordinate in mesh])
        logarithm = np.log10(self.ranges)
        grid = 10.0 ** (
            logarithm[:, 0] + unit * (logarithm[:, 1] - logarithm[:, 0])
        )
        started = time.perf_counter()
        worst = 0.0
        location = None
        for parameter in grid:
            value = float(np.max(self.absolute_error(parameter)))
            if value > worst:
                worst = value
                location = parameter.copy()
        return {
            "points": int(grid.shape[0]),
            "worst_absolute_error": worst,
            "location": None if location is None else location.tolist(),
            "seconds": time.perf_counter() - started,
        }

    def cell_bound(self, low, high, order=2):
        """Rigorous entrywise bound of the error on one HTC cell.

        The residual is enclosed by a tensor Bernstein polynomial in the cell's
        *own* Riesz map ``Z^T A(low)^-1 Z``: Galerkin optimality admits any
        polynomial trial, ``A(p) >= A(low)`` gives the Loewner transfer, and the
        Bernstein coefficients bound the quadratic form pointwise on the cell.
        Only the Gram changes relative to :mod:`certified_box`; it comes from
        the Woodbury tables, so no cell needs a factorisation.
        """
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        return self.certificate.cell_bound(
            self.augmented_gram(low), low, high, order
        )

    def sweep(self, cells_per_axis=8, order=2):
        """Bracket the box: exact at centres, rigorous per cell."""
        if np.isscalar(cells_per_axis):
            cells_per_axis = (int(cells_per_axis),) * self.dimension
        cells_per_axis = tuple(int(count) for count in cells_per_axis)
        edges = [
            logarithmic_edges(low, high, count)
            for (low, high), count in zip(self.ranges, cells_per_axis)
        ]
        started = time.perf_counter()
        worst_exact = 0.0
        worst_bound = 0.0
        location = None
        for choice in itertools.product(*[range(count) for count in cells_per_axis]):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            centre = 0.5 * (low + high)
            exact = float(np.max(self.absolute_error(centre)))
            bound = float(np.max(self.cell_bound(low, high, order)))
            if exact > worst_exact:
                worst_exact = exact
                location = centre.tolist()
            worst_bound = max(worst_bound, bound)
        return {
            "cells": int(np.prod(cells_per_axis)),
            "order": int(order),
            "exact_at_centres": worst_exact,
            "cell_bound": worst_bound,
            "worst_centre": location,
            "basis_order": self.order,
            "shift": self.shift,
            "m_b": int(self.boundary.size),
            "span_columns": self.span_columns,
            "preparation_seconds": self.preparation_seconds,
            "seconds": time.perf_counter() - started,
            "floating_point_certified": False,
        }
