"""Rigorous whole-box certificate for one delivered BCI basis.

The certificate answers a question that a validation run cannot: how large can
the junction transfer error of a *fixed* reduced basis become at *any* heat
transfer coefficient vector of the continuous box?

Statement proved (exact arithmetic)

For a fixed real frequency shift ``s >= 0`` and every HTC vector ``p`` in the
box ``[p_low, p_high]``, write

    A(p)   = K + s*C + sum_i p_i H_i          (symmetric positive definite)
    X(p)   = A(p)^-1 G
    X_V(p) = V (V^T A(p) V)^-1 V^T G          (Galerkin projection, V delivered)
    Y(p)   = G^T X(p),  Y_V(p) = G^T X_V(p)   (co-located junction transfer)

Two standard facts give an exact, computable bound.

1. Galerkin optimality.  ``X_V(p)`` minimizes the ``A(p)``-energy distance to
   ``X(p)`` inside ``range(V)``.  Hence for *every* trial coefficient matrix
   ``q(p)`` (a polynomial here) with residual ``r_q(p) = G - A(p) V q(p)``,

       Y(p) - Y_V(p) = r(p)^T A(p)^-1 r(p) <= r_q(p)^T A(p)^-1 r_q(p),
       r(p) = G - A(p) V (V^T A(p) V)^-1 V^T G  >= 0   (Loewner order).

   Positive semidefiniteness gives ``|entry_ab| <= sqrt(diag_a diag_b)``.

2. Loewner monotonicity.  Every ``H_i`` is positive semidefinite and ``p``
   stays above its block anchor, so ``A(p) >= A(anchor)`` and

       r_q(p)^T A(p)^-1 r_q(p) <= r_q(p)^T A(anchor)^-1 r_q(p).

The right-hand side is computable without any further full-order solve:

* a Taylor jet of the *small* reduced solve is a polynomial trial, so ``r_q``
  is a polynomial whose coefficient blocks live in the fixed span
  ``[G, (K+sC)V, H_1 V, ..., H_d V]`` with ``(d+2) m + k`` columns;
* one Riesz Gram ``Z^T A(anchor)^-1 Z`` of that span is precomputed per anchor;
* on a cell the polynomial is enclosed by its tensor Bernstein coefficients,
  which form a convex combination, so the Gram quadratic form is bounded by the
  largest coefficient-wise value, pointwise on the whole cell.

Relative statements need the exact same-parameter rise.  The family is an
entrywise nonnegative M-matrix family and ``dY_ab/dp_k = -x_a^T H_k x_b`` is
entrywise nonpositive, so the *exact* transfer at a cell's upper HTC corner is
a positive lower bound of ``Y(p)`` at every point of that cell.  ``sweep``
normalizes by that corner, which costs one direct solve per cell and no
fraction of the reduced transfer.

Any fixed real shift is covered.  The BDF1 step recursion solves with
``A(p) + C/dt``, which is another member of the same affine family, so
``shift = 1/dt`` certifies that operator family as well.

Cost

Preparation is one Riesz Gram per anchor: one factorization plus
``span_columns`` sparse solves each.  Every cell then needs the exact transfer
at its upper corner for the normalization, i.e. one further factorization and
``source_count`` solves.  Nothing here is an AMG-CG extraction solve, and all
remaining cell evaluation is small dense algebra in the delivered reduced
order: refining the jet order is free, and refining the partition costs only
those corner denominators while tightening the bound.

Limits, stated explicitly

* Inequalities hold in exact real arithmetic; sparse factorizations, Gram
  matrices and small dense solves are ordinary floating point
  (``floating_point_certified=False`` in every report).
* The bound is an upper bound and is not claimed to be tight.
* ``sweep`` covers one fixed real shift.  The frequency axis is deliberately
  absent: an anchored Riesz operator has to overestimate the whole HTC range,
  and the measured loss of that weighting is about six orders of magnitude
  (see the playground report).
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import comb


# ---------------------------------------------------------------------------
# Bernstein enclosure helpers
# ---------------------------------------------------------------------------


def multi_indices(dimension, degree):
    """All multi-indices with ``0 <= |alpha| <= degree``."""
    result = [tuple([0] * dimension)]
    for total in range(1, degree + 1):
        result.extend(
            index
            for index in itertools.product(range(total + 1), repeat=dimension)
            if sum(index) == total
        )
    return result


def bernstein_operator(degree):
    """Inverse of the equispaced Bernstein evaluation matrix."""
    nodes = np.arange(degree + 1) / degree
    evaluation = np.column_stack(
        [
            float(comb(degree, index)) * nodes**index * (1 - nodes) ** (degree - index)
            for index in range(degree + 1)
        ]
    )
    return la.inv(evaluation)


def bernstein_coefficients(values, degree, dimension):
    """Tensor Bernstein coefficients of equispaced nodal values.

    ``values`` has shape ``(degree + 1,) * dimension + trailing``; trailing
    axes are transformed independently.
    """
    inverse = bernstein_operator(degree)
    result = values
    for axis in range(dimension):
        moved = np.moveaxis(result, axis, 0)
        result = np.moveaxis(np.tensordot(inverse, moved, axes=(1, 0)), 0, axis)
    return result


def logarithmic_edges(low, high, blocks):
    return 10.0 ** np.linspace(np.log10(low), np.log10(high), int(blocks) + 1)


def sparse_factor(operator):
    return spla.splu(sp.csc_matrix(operator).tocsc(), permc_spec="MMD_AT_PLUS_A")


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------


class BoxCertificate:
    """Parameter-box error certificate for one fixed delivered basis.

    Parameters
    ----------
    kernel, terms:
        ``K`` and the boundary matrices ``H_i`` (sparse, symmetric).
    source:
        ``(n, k)`` co-located source/output shape.
    ranges:
        ``(d, 2)`` effective HTC box.
    basis:
        ``(n, m)`` delivered reduced basis with orthonormal columns.
    shift:
        real frequency shift ``s >= 0`` of the certified operator.
    mass:
        heat capacity matrix; required for a nonzero ``shift``.
    blocks:
        log-uniform Riesz anchor blocks per parameter axis.
    """

    def __init__(
        self,
        kernel,
        terms,
        source,
        ranges,
        basis,
        *,
        shift=0.0,
        mass=None,
        blocks=None,
    ):
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
        if self.source.shape[0] != self.kernel.shape[0]:
            raise ValueError("source must match the kernel size")
        if self.shift and self.mass is None:
            raise ValueError("a nonzero shift requires the mass matrix")

        self.source_count = int(self.source.shape[1])
        self.order = int(self.basis.shape[1])
        self.base_operator = (
            self.kernel if self.mass is None else self.kernel + self.shift * self.mass
        )
        span_blocks = [self.source, self.base_operator @ self.basis]
        span_blocks.extend(term @ self.basis for term in self.terms)
        self.span = np.ascontiguousarray(np.column_stack(span_blocks))
        self.span_columns = int(self.span.shape[1])

        self.projected_base = np.ascontiguousarray(self.basis.T @ (self.kernel @ self.basis))
        self.projected_terms = [
            np.ascontiguousarray(self.basis.T @ (term @ self.basis)) for term in self.terms
        ]
        self.projected_mass = (
            None
            if self.mass is None
            else np.ascontiguousarray(self.basis.T @ (self.mass @ self.basis))
        )
        self.projected_source = np.ascontiguousarray(self.basis.T @ self.source)

        self._grams: dict = {}
        self._denominators: dict = {}
        self.block_edges = None
        self.blocks = None
        self.anchor_seconds = 0.0
        self._prepare_blocks(blocks)

    # ------------------------------------------------------ anchor measures

    def _prepare_blocks(self, blocks):
        blocks = (4,) * self.dimension if blocks is None else (
            (int(blocks),) * self.dimension
            if np.isscalar(blocks)
            else tuple(int(block) for block in blocks)
        )
        if len(blocks) != self.dimension:
            raise ValueError("one block count per parameter is required")
        self.blocks = tuple(blocks)
        self.block_edges = [
            logarithmic_edges(low, high, count)
            for (low, high), count in zip(self.ranges, self.blocks)
        ]

    def block_index(self, point):
        """Index of the block containing ``point``."""
        indices = []
        for axis, value in enumerate(np.asarray(point, dtype=np.float64)):
            edges = self.block_edges[axis]
            position = int(np.searchsorted(edges, value + 1e-12, side="right") - 1)
            indices.append(min(max(position, 0), self.blocks[axis] - 1))
        return int(np.ravel_multi_index(tuple(indices), self.blocks))

    def block_lower(self, index):
        positions = np.unravel_index(index, self.blocks)
        return np.array(
            [self.block_edges[axis][position] for axis, position in enumerate(positions)]
        )

    def block_upper(self, index):
        positions = np.unravel_index(index, self.blocks)
        return np.array(
            [self.block_edges[axis][position + 1] for axis, position in enumerate(positions)]
        )

    def operator(self, parameter, omega=0.0):
        """Sparse operator of the affine family at one parameter."""
        operator = self.base_operator
        if omega:
            operator = operator + omega * self.mass
        for value, term in zip(parameter, self.terms):
            operator = operator + float(value) * term
        return sp.csc_matrix(operator)

    def anchor_gram(self, parameter, omega=0.0):
        """Riesz Gram of the residual span at one anchor operator."""
        key = (float(omega),) + tuple(np.round(np.asarray(parameter, float), 12))
        gram = self._grams.get(key)
        if gram is None:
            factor = sparse_factor(self.operator(parameter, omega))
            gram = self.span.T @ factor.solve(self.span)
            gram = np.ascontiguousarray(0.5 * (gram + gram.T))
            self._grams[key] = gram
        return gram

    def transfer(self, parameter):
        """Exact full-order transfer at one parameter (reference only)."""
        return np.ascontiguousarray(self.source.T @ sparse_factor(
            self.operator(np.asarray(parameter, float))
        ).solve(self.source))

    # ---------------------------------------------------------- small system

    def _reduced_matrix(self, point):
        matrix = self.projected_base.astype(float).copy()
        if self.shift:
            matrix = matrix + self.shift * self.projected_mass
        for value, term in zip(point, self.projected_terms):
            matrix = matrix + float(value) * term
        return matrix

    def _reduced_solve(self, matrix, rhs):
        return la.solve(matrix, rhs, assume_a="pos", check_finite=False)

    def _block_coefficients(self, point):
        return [float(value) for value in point]

    def _trial_jets(self, point, order):
        """Taylor jets of the small reduced solve at one expansion point."""
        matrix = self._reduced_matrix(point)
        directions = list(self.projected_terms)
        base = self._reduced_solve(matrix, self.projected_source)
        dimension = len(directions)
        jets = {tuple([0] * dimension): base}
        for alpha in multi_indices(dimension, order):
            if not any(alpha):
                continue
            rhs = np.zeros_like(base)
            for axis, value in enumerate(alpha):
                if value:
                    previous = list(alpha)
                    previous[axis] -= 1
                    rhs = rhs + directions[axis] @ jets[tuple(previous)]
            jets[alpha] = -self._reduced_solve(matrix, rhs)
        return jets

    def _stacked(self, coefficients, block_coefficients):
        rows = [np.eye(self.source_count, dtype=np.float64)]
        rows.append(-coefficients)
        for value in block_coefficients:
            rows.append(-value * coefficients)
        return np.vstack(rows)

    def _gram_form(self, gram, block):
        return np.ascontiguousarray((block.T @ (gram @ block)).real)

    # ---------------------------------------------------------- cell bound

    def cell_bound(self, gram, low, high, order=2):
        """Absolute entrywise bound of the transfer error on one box cell."""
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        dimension = len(center)
        jets = self._trial_jets(center, order)
        degree = order + 1
        nodes = degree + 1
        shape = (nodes,) * dimension
        values = np.empty(shape + (self.span_columns, self.source_count))
        for node in np.ndindex(shape):
            scaled = np.array([-1.0 + 2.0 * position / degree for position in node])
            trial = np.zeros_like(self.projected_source)
            for alpha, jet in jets.items():
                weight = 1.0
                for axis, value in enumerate(alpha):
                    weight *= (scaled[axis] * half[axis]) ** value
                trial = trial + weight * jet
            values[node] = self._stacked(
                trial, self._block_coefficients(center + scaled * half)
            )
        coefficients = bernstein_coefficients(values, degree, dimension)
        worst = np.full((self.source_count, self.source_count), -np.inf)
        for node in np.ndindex(shape):
            np.maximum(worst, self._gram_form(gram, coefficients[node]), out=worst)
        np.maximum(worst, 0.0, out=worst)
        return np.sqrt(np.outer(np.diag(worst), np.diag(worst)))

    def cell_denominator(self, point):
        """Exact transfer at a lattice point, cached.

        The M-matrix family has entrywise decreasing transfers
        (``dY_ab/dp_k = -x_a^T H_k x_b <= 0``), so the transfer at the *upper*
        corner of a cell is a positive lower bound of every transfer inside it.
        """
        key = tuple(float(f"{value:.14e}") for value in np.asarray(point, float))
        transfer = self._denominators.get(key)
        if transfer is None:
            transfer = self.transfer(point)
            self._denominators[key] = transfer
        return transfer

    # --------------------------------------------------------------- sweeps

    def sweep(self, cells_per_axis, order=2, blocks=None):
        """Bound the entrywise relative steady transfer error on the box.

        Every cell of a uniform log partition is compared with the exact
        transfer at its own upper corner, which is the tightest corner-based
        lower bound available for the same-parameter normalization.
        """
        if blocks is not None:
            self._prepare_blocks(blocks)
        if np.isscalar(cells_per_axis):
            cells_per_axis = (int(cells_per_axis),) * self.dimension
        cells_per_axis = tuple(int(count) for count in cells_per_axis)
        edges = [
            logarithmic_edges(low, high, count)
            for (low, high), count in zip(self.ranges, cells_per_axis)
        ]
        grams = {}
        worst_relative = 0.0
        worst_absolute = 0.0
        worst_diagonal = 0.0
        location = None
        started = time.perf_counter()
        cells = 0
        for choice in itertools.product(*[range(count) for count in cells_per_axis]):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            index = self.block_index(low)
            gram = grams.get(index)
            if gram is None:
                gram = self.anchor_gram(self.block_lower(index))
                grams[index] = gram
            bound = self.cell_bound(gram, low, high, order)
            denominator = self.cell_denominator(high)
            relative = bound / denominator
            diagonal = np.outer(np.diag(denominator), np.diag(denominator)) ** 0.5
            cells += 1
            value = float(np.max(relative))
            if value > worst_relative:
                worst_relative = value
                worst_absolute = float(np.max(bound))
                worst_diagonal = float(np.max(bound / diagonal))
                entry = np.unravel_index(int(np.argmax(relative)), relative.shape)
                location = {
                    "cell_low": low.tolist(),
                    "cell_high": high.tolist(),
                    "anchor": int(index),
                    "entry": [int(entry[0]), int(entry[1])],
                }
        return {
            "cells": int(cells),
            "order": int(order),
            "blocks": list(self.blocks),
            "anchors": int(np.prod(self.blocks)),
            "span_columns": self.span_columns,
            "basis_order": self.order,
            "shift": self.shift,
            "steady_relative_bound": worst_relative,
            "steady_diagonal_bound": worst_diagonal,
            "steady_absolute_bound": worst_absolute,
            "worst_location": location,
            "denominator_points": len(self._denominators),
            "seconds": time.perf_counter() - started,
            "floating_point_certified": False,
        }
