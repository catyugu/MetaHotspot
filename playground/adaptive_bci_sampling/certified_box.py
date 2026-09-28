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

import heapq
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


def generalized_max(matrix, denominator, floor=1e-14):
    """Largest generalized eigenvalue ``lambda_max(W, D)`` for SPD ``D``.

    The quantity is the sharpest scalar ``c`` with ``W <= c*D`` in the Loewner
    order, i.e. the relative error a symmetric matrix statement should report.
    A jitter proportional to the denominator trace is added only if the
    Cholesky factorization of ``D`` fails, which makes the result larger and
    therefore keeps it a valid upper bound.
    """
    form = np.ascontiguousarray(0.5 * (matrix + matrix.T))
    weight = np.ascontiguousarray(0.5 * (denominator + denominator.T))
    jitter = floor * max(float(np.max(np.abs(form))), float(np.trace(weight)))
    for attempt in range(4):
        try:
            return float(
                np.max(la.eigvalsh(form + attempt * jitter * np.eye(form.shape[0]), weight))
            )
        except la.LinAlgError:
            continue
    return float("inf")


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
        span="shift",
        gram_cache=None,
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
        self.span_kind = span
        if span not in ("shift", "common"):
            raise ValueError(f"unknown span kind {span!r}")
        self.base_operator = (
            self.kernel if self.mass is None else self.kernel + self.shift * self.mass
        )
        if span == "common":
            # ``Z = [G, K V, C V, H_1 V, ...]`` carries the residual of *every*
            # shift, because the shift enters only through the coefficient ``s``:
            # ``R_q(h, s) = Z [I; -q; -s q; -h_1 q; ...]``.  With ``A(h, s) >=
            # A(a, 0)`` for every ``s >= 0``, one anchor Gram ``S_a(0) =
            # Z^T A(a,0)^-1 Z`` then serves the whole frequency plan.
            if self.mass is None:
                raise ValueError("a shift-free common span needs the mass matrix")
            span_blocks = [self.source, self.kernel @ self.basis, self.mass @ self.basis]
        else:
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

        # A shift-free common span has shift-independent anchor Grams, so callers
        # that certify a whole frequency plan can hand in one shared cache and pay
        # for each anchor once instead of once per shift.
        self._grams: dict = {} if gram_cache is None else gram_cache
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

    def operator(self, parameter, omega=0.0, shift=None):
        """Sparse operator of the affine family at one parameter.

        ``shift=None`` uses the certificate's own shift; a shift-free common
        anchor span passes ``shift=0.0``, because its Riesz Gram has to come from
        an operator that is below ``A(h, s)`` for every non-negative ``s``.
        """
        value = self.shift if shift is None else float(shift)
        operator = self.kernel if self.mass is None else self.kernel + value * self.mass
        if omega:
            operator = operator + omega * self.mass
        for parameter_value, term in zip(parameter, self.terms):
            operator = operator + float(parameter_value) * term
        return sp.csc_matrix(operator)

    def anchor_gram(self, parameter, omega=0.0):
        """Riesz Gram of the residual span at one anchor operator."""
        key = (float(omega),) + tuple(np.round(np.asarray(parameter, float), 12))
        gram = self._grams.get(key)
        if gram is None:
            anchor_shift = 0.0 if self.span_kind == "common" else None
            factor = sparse_factor(self.operator(parameter, omega, shift=anchor_shift))
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
        """Parameter column of the residual in the certificate's span.

        The ``shift`` span already contains ``(K + sC)V``, so its coefficient is
        just the HTC vector.  The ``common`` span splits that block into ``KV``
        and ``CV``, so the shift enters as the extra leading coefficient ``s``.
        """
        if self.span_kind == "common":
            return [self.shift] + [float(value) for value in point]
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

    def cell_bound(self, gram, low, high, order=2, trial="taylor"):
        """Absolute entrywise bound of the transfer error on one box cell."""
        kinds = (trial,) if isinstance(trial, str) else tuple(trial)
        best = None
        for kind in kinds:
            worst = np.full((self.source_count, self.source_count), -np.inf)
            for form in self._cell_forms(gram, low, high, order, kind):
                np.maximum(worst, form, out=worst)
            np.maximum(worst, 0.0, out=worst)
            value = np.sqrt(np.outer(np.diag(worst), np.diag(worst)))
            best = value if best is None else np.minimum(best, value)
        return best

    def _cell_forms(self, gram, low, high, order=2, trial="taylor"):
        """Bernstein coefficient matrices ``Xi_nu^T S_a Xi_nu`` of one cell.

        The residual polynomial of any polynomial trial is enclosed on the cell
        by its tensor Bernstein coefficients ``beta_nu >= 0`` with
        ``sum_nu beta_nu = 1``, so matrix convexity of ``X -> X^T S_a X`` gives,
        on the whole cell,

            R_q(h)^T A(a)^-1 R_q(h) <= sum_nu beta_nu(h) Xi_nu^T S_a Xi_nu,

        which is what a relative matrix statement needs: every coefficient is
        divided by the same cell denominator, so no coefficient can be replaced
        by its entrywise maximum.
        """
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        center = 0.5 * (low + high)
        half = 0.5 * (high - low)
        dimension = len(center)
        degree = order + 1
        nodes = degree + 1
        shape = (nodes,) * dimension
        coefficients = self._trial_coefficients(center, half, order, trial)
        values = np.empty(shape + (self.span_columns, self.source_count))
        for node in np.ndindex(shape):
            scaled = np.array([-1.0 + 2.0 * position / degree for position in node])
            values[node] = self._stacked(
                self._trial_at(coefficients, half, scaled),
                self._block_coefficients(center + scaled * half),
            )
        bernstein = bernstein_coefficients(values, degree, dimension)
        return [self._gram_form(gram, bernstein[node]) for node in np.ndindex(shape)]

    def cell_matrix_bound(self, gram, low, high, denominator, order=2, trial="taylor"):
        """Relative collocated port-transfer defect ``sup_h lambda_max(E(h), Y(h))``.

        ``E(h) = Y(h) - Y_V(h) = R_V(h)^T A(h)^-1 R_V(h) >= 0`` is the collocated
        port defect and ``denominator`` must be a matrix below the exact transfer
        ``Y(h)`` at every point of the cell (the upper corner serves, since the
        family is entrywise decreasing in the HTC vector); then the returned value
        is an upper bound on the worst relative all-input port defect
        ``sup_w w^T (Y(h) - Y_V(h)) w / w^T Y(h) w`` over the same cell.  No square
        root is taken anywhere: this is the port quantity itself, not a state
        amplitude.
        """
        kinds = (trial,) if isinstance(trial, str) else tuple(trial)
        best = None
        for kind in kinds:
            forms = self._cell_forms(gram, low, high, order, kind)
            value = 0.0
            for form in forms:
                value = max(value, generalized_max(form, denominator))
            best = value if best is None else min(best, value)
        return float(best)

    def point_floor(self, gram, point):
        """``lambda_max(R_V(h)^T A(a)^-1 R_V(h), Y_V(h))`` at one parameter point.

        The exact-parameter value of the cell form, i.e. the certificate bound
        with the parameter enclosure removed.  At a fixed anchor this is the part
        of a cell bound that shrinking the cell cannot lower, so evaluating it at
        the centre and the corners of an unresolved leaf separates a genuine
        anchor-mismatch floor from a partition that is merely too coarse.
        """
        point = np.asarray(point, dtype=np.float64)
        coefficients = self._reduced_solve(self._reduced_matrix(point), self.projected_source)
        block = self._stacked(coefficients, self._block_coefficients(point))
        return float(generalized_max(self._gram_form(gram, block), self.cell_denominator(point)))

    def cell_reduced_transfer(self, point):
        """Reduced (Galerkin) transfer at a lattice point, cached.

        The Galerkin identity gives ``Y(b) - Y_V(b) = R_V(b)^T A(b)^-1 R_V(b) >= 0``
        and the parameter monotonicity gives ``Y(h) >= Y(b)`` on the enclosing
        cell, so ``Y_V(b) <= Y(h)`` holds on the cell and the reduced transfer is a
        valid denominator at no full-order cost.  Replacing the exact ``Y(b)`` by
        it can only inflate the same generalized eigenvalue by at most
        ``1 / (1 - delta_b^2)`` when ``E(b) <= delta_b^2 Y(b)``, which is
        ``1 + O(1e-4)`` at the defects measured here.
        """
        key = ("reduced",) + tuple(
            float(f"{value:.14e}") for value in np.asarray(point, float)
        )
        transfer = self._denominators.get(key)
        if transfer is None:
            projected = self.projected_source
            transfer = np.ascontiguousarray(
                projected.T @ self._reduced_solve(self._reduced_matrix(point), projected)
            )
            transfer = 0.5 * (transfer + transfer.T)
            self._denominators[key] = transfer
        return transfer

    def defect(self, parameter, omega=0.0):
        """Exact relative collocated port defect at one parameter, one solve.

        Returns ``(defect, transfer)`` with ``defect = lambda_max(E(h), Y(h))``
        and the exact transfer matrix used as its denominator, so a caller can
        also report the accuracy statement in transfer units.
        """
        point = np.asarray(parameter, dtype=np.float64)
        operator = self.operator(point, omega)
        factor = sparse_factor(operator)
        reduced = np.asarray(self.basis.T @ (operator @ self.basis))
        coefficients = self._reduced_solve(reduced, self.projected_source)
        residual = self.source - np.asarray(operator @ (self.basis @ coefficients))
        error = np.asarray(residual.T @ factor.solve(residual))
        transfer = np.asarray(self.source.T @ factor.solve(self.source))
        return (
            generalized_max(0.5 * (error + error.T), 0.5 * (transfer + transfer.T)),
            np.ascontiguousarray(0.5 * (transfer + transfer.T)),
        )

    def branch_and_bound(self, threshold, initial_cells=2, orders=(2, 3, 4, 5),
                         trial="taylor", samples=3, max_cells=512,
                         max_rounds=32, upgrade_anchor=False, oracle=False):
        """Certified branch and bound over the box, without any enrichment.

        A leaf is measured once, when it is created: the successive trial orders
        are tried at its fixed cell and anchor (p-refinement, no new Gram matrix)
        and the smallest bound wins, optionally together with the exact witness.
        Leaves then live in a max-heap keyed by that bound, so the loop always
        splits the currently worst leaf and never re-measures a leaf it already
        measured - the reduced-side work is proportional to the number of leaves
        created instead of leaves times rounds.  When the popped leaf is already
        below the threshold every remaining leaf is too, by the heap property,
        and the loop accepts; this is the only place the threshold is consulted.

        The anchor is the shared block anchor, so any number of leaves costs at
        most one Gram matrix per block, and a leaf whose bound is still above the
        threshold may build its own behind ``upgrade_anchor``.  The exact witness
        ``L`` behind ``oracle=True`` is research instrumentation: correctness and
        stopping use the certificate ``U`` alone, and the default certificate mode
        never solves a full-order system beyond the shared anchor Gram matrices.
        """
        orders = tuple(int(value) for value in orders)
        explore = (trial,) if isinstance(trial, str) else tuple(trial)
        start = time.perf_counter()
        counts = {"grams": 0, "witnesses": 0, "measurements": 0}
        block_grams = {}
        placements = itertools.count()

        def block_anchor(low):
            block = self.block_index(low)
            if block not in block_grams:
                block_grams[block] = self.anchor_gram(self.block_lower(block))
                counts["grams"] += 1
            return block_grams[block]

        def measure(low, high, gram):
            """``(bound, order, witness, witness point)`` of one leaf."""
            counts["measurements"] += 1
            weight, _ = self.cell_weight(high, "reduced")
            bound, order_used = None, orders[-1]
            for order in orders:
                value = self.cell_matrix_bound(gram, low, high, weight, order, explore)
                if bound is None or value < bound:
                    bound, order_used = value, order
                if value <= threshold:
                    break
            witness, witness_point = 0.0, None
            if oracle:
                step = (high / low) ** (1.0 / (samples + 1))
                for point in itertools.product(
                    *[low[axis] * step[axis] ** np.arange(samples + 2)
                      for axis in range(self.dimension)]
                ):
                    value, _ = self.defect(np.array(point))
                    counts["witnesses"] += 1
                    if value > witness:
                        witness, witness_point = value, point
            return bound, order_used, witness, witness_point

        def place(low, high, gram):
            bound, order, witness, witness_point = measure(low, high, gram)
            heapq.heappush(
                heap, (-bound, next(placements), low, high, gram, order, witness,
                       None if witness_point is None else np.array(witness_point))
            )
            return bound

        edges = [
            logarithmic_edges(low, high, initial_cells) for low, high in self.ranges
        ]
        heap = []
        for choice in itertools.product(*[range(initial_cells)] * self.dimension):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            place(low, high, block_anchor(low))
        created = len(heap)
        splits = 0

        history = []
        while heap and splits < max_rounds:
            minus, _, low, high, gram, order, witness, witness_point = heapq.heappop(heap)
            bound = -minus
            if bound <= threshold:
                return self._bound_report(
                    True, history, bound, splits, len(heap) + 1, created, orders,
                    explore, threshold, low, high, witness, witness_point, oracle,
                    counts, start,
                )
            own = None
            if upgrade_anchor and gram is block_grams[self.block_index(low)]:
                # only a leaf unresolved after p-refinement earns its own anchor:
                # the enclosing block anchor is valid but loose
                own = self.anchor_gram(low)
                counts["grams"] += 1
                refined = measure(low, high, own)
                if refined[0] < bound:
                    heapq.heappush(
                        heap, (-refined[0], next(placements), low, high, own,
                               refined[1], refined[2],
                               None if refined[3] is None else np.array(refined[3]))
                    )
                    history.append(self._step_entry(
                        splits, len(heap), len(block_grams), refined[0], refined[1],
                        refined[2], low, high, oracle, "anchor",
                    ))
                    continue
                own = None
            if created + 2 > max_cells:
                break
            axis = int(np.argmax(np.log(high) - np.log(low)))
            middle = np.sqrt(low[axis] * high[axis])
            for side in range(2):
                child_low = low.copy()
                child_high = high.copy()
                if side == 0:
                    child_high[axis] = middle
                else:
                    child_low[axis] = middle
                place(child_low, child_high, block_anchor(child_low))
            created += 2
            splits += 1
            root = heap[0]
            history.append(self._step_entry(
                splits, len(heap), len(block_grams), -root[0], root[5], root[6],
                root[2], root[3], oracle, "split",
            ))
        root = heap[0]
        return self._bound_report(
            False, history, -root[0], splits, len(heap), created, orders, explore,
            threshold, root[2], root[3], root[6], root[7], oracle, counts, start,
        )

    @staticmethod
    def _step_entry(step, leaves, blocks, bound, order, witness, low, high, oracle, action):
        return {
            "round": step,
            "action": action,
            "leaves": leaves,
            "anchors": blocks,
            "worst_bound": bound,
            "worst_order": order,
            "worst_witness": witness if oracle else None,
            "worst_cell": [low.tolist(), high.tolist()],
        }

    @staticmethod
    def _bound_report(accepted, history, bound, splits, leaves, created, orders,
                      trial, threshold, low, high, witness, witness_point, oracle,
                      counts, start):
        return {
            "accepted": bool(accepted),
            "splits": splits,
            "leaves": leaves,
            "leaves_created": created,
            "orders": list(orders),
            "trial": trial,
            "threshold": float(threshold),
            "bound": bound,
            "witness": witness if oracle else None,
            "oracle": bool(oracle),
            "worst_cell": [low.tolist(), high.tolist()],
            "bound_over_exact": None,
            "seconds": time.perf_counter() - start,
            "gram_matrices": counts["grams"],
            "witness_solves": counts["witnesses"],
            "measurements": counts["measurements"],
            "history": history,
        }

    def cell_weight(self, point, denominator="reduced"):
        """Cell denominator, reduced unless it is not positive definite."""
        if denominator == "reduced":
            weight = self.cell_reduced_transfer(point)
            if float(np.min(np.linalg.eigvalsh(weight))) > 0.0:
                return weight, 0
            return self.cell_denominator(point), 1
        return self.cell_denominator(point), 0

    def _trial_coefficients(self, center, half, order, trial):
        """Tensor monomial coefficients of a polynomial trial of the reduced solve.

        ``taylor`` uses the Taylor jets of the reduced solve at the cell centre;
        ``chebyshev`` interpolates it at Chebyshev nodes of the cell; ``logfit``
        least-squares fits it at log-spread nodes of the cell.  Every variant is
        a genuine polynomial in the *affine* cell coordinate ``h = center + half
        * xi``, which is what keeps the residual a polynomial of degree
        ``order + 1`` per axis.  Fitting in ``log h`` would break the enclosure,
        because ``h = exp(log h)`` is not polynomial in the fitted variable; the
        log-spread variant only uses log-spaced *nodes*, never a log parameter.

        The fit is performed in the unit coordinate ``xi`` and the coefficients
        are then converted to powers of ``h - center``.  Fitting directly in
        ``h - center`` is the same algebra but badly conditioned, because the
        Vandermonde columns then span ``half^order``.
        """
        count = order + 1
        dimension = self.dimension
        shape = (count,) * dimension + self.projected_source.shape
        if trial == "taylor":
            coefficients = np.zeros(shape)
            for alpha, jet in self._trial_jets(center, order).items():
                coefficients[alpha] = jet
            return coefficients
        if trial == "chebyshev":
            nodes = [np.cos(np.pi * (2.0 * np.arange(count) + 1.0) / (2.0 * count))] * dimension
        elif trial == "logfit":
            nodes = []
            for axis in range(dimension):
                low, high = center[axis] - half[axis], center[axis] + half[axis]
                inside = np.exp(np.linspace(np.log(low), np.log(high), count + 2)[1:-1])
                nodes.append((inside - center[axis]) / half[axis])
        else:
            raise ValueError(f"unknown trial {trial!r}")
        counts = [len(axis_nodes) for axis_nodes in nodes]
        values = np.empty(tuple(counts) + self.projected_source.shape)
        for node in np.ndindex(*counts):
            point = center + np.array(
                [nodes[axis][index] * half[axis] for axis, index in enumerate(node)]
            )
            values[node] = self._reduced_solve(
                self._reduced_matrix(point), self.projected_source
            )
        for axis in range(dimension):
            design = np.vander(nodes[axis], count, increasing=True)
            inverse = la.lstsq(design, np.eye(counts[axis]))[0]
            moved = np.moveaxis(values, axis, 0)
            values = np.moveaxis(np.tensordot(inverse, moved, axes=(1, 0)), 0, axis)
        for alpha in np.ndindex(*([count] * dimension)):
            weight = 1.0
            for axis, value in enumerate(alpha):
                if value:
                    weight *= half[axis] ** value
            if weight != 1.0:
                values[alpha] = values[alpha] / weight
        return values

    def _trial_at(self, coefficients, half, scaled):
        """Evaluate a tensor monomial trial at ``xi = scaled`` of the unit cube."""
        trial = np.zeros_like(self.projected_source)
        for alpha in np.ndindex(*coefficients.shape[: self.dimension]):
            weight = 1.0
            for axis, value in enumerate(alpha):
                if value:
                    weight *= (scaled[axis] * half[axis]) ** value
            if weight:
                trial = trial + weight * coefficients[alpha]
        return trial

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

    def sweep_matrix(self, cells_per_axis, order=2, local_anchor=True,
                     trial="taylor", denominator="reduced"):
        """Bound ``sup_h lambda_max(E(h), Y(h))`` over the whole box.

        This is the quantity that survives the falsification of the cheap
        ``C``-metric scalar: at one fixed shift it is exactly the worst relative
        all-input collocated port defect of the delivered basis over all
        admissible HTC vectors, i.e. ``sup_h sup_w w^T (Y(h) - Y_V(h)) w /
        w^T Y(h) w``.  It is a port quantity, so no square root is taken.

        ``local_anchor`` uses the lower corner of the cell itself as the Riesz
        anchor, which is the tightest anchor valid on that cell but costs one
        Gram per cell; otherwise the enclosing log block anchor of ``blocks`` is
        reused, which is what ``sweep`` does.  ``trial`` is one polynomial trial
        or a tuple whose statements are combined by taking the minimum.
        ``denominator`` selects the exact transfer at the cell's upper corner or
        the reduced transfer, which needs no full-order solve.
        """
        if np.isscalar(cells_per_axis):
            cells_per_axis = (int(cells_per_axis),) * self.dimension
        cells_per_axis = tuple(int(count) for count in cells_per_axis)
        edges = [
            logarithmic_edges(low, high, count)
            for (low, high), count in zip(self.ranges, cells_per_axis)
        ]
        grams = {}
        worst = 0.0
        location = None
        cells = 0
        fallbacks = 0
        started = time.perf_counter()
        for choice in itertools.product(*[range(count) for count in cells_per_axis]):
            low = np.array([edges[axis][index] for axis, index in enumerate(choice)])
            high = np.array([edges[axis][index + 1] for axis, index in enumerate(choice)])
            if local_anchor:
                key = tuple(np.round(low, 14))
                anchor = low
            else:
                block = self.block_index(low)
                key = (block,)
                anchor = self.block_lower(block)
            gram = grams.get(key)
            if gram is None:
                gram = self.anchor_gram(anchor)
                grams[key] = gram
            weight, fallback = self.cell_weight(high, denominator)
            fallbacks += fallback
            value = self.cell_matrix_bound(gram, low, high, weight, order, trial)
            cells += 1
            if value > worst:
                worst = value
                location = {
                    "cell_low": low.tolist(),
                    "cell_high": high.tolist(),
                    "anchor": [float(value) for value in anchor],
                }
        return {
            "cells": int(cells),
            "order": int(order),
            "trial": trial,
            "denominator": denominator,
            "local_anchor": bool(local_anchor),
            "anchors": len(grams),
            "span_columns": self.span_columns,
            "basis_order": self.order,
            "shift": self.shift,
            "worst_relative_port_defect": float(worst),
            "worst_location": location,
            "denominator_points": len(self._denominators),
            "denominator_fallbacks": int(fallbacks),
            "seconds": time.perf_counter() - started,
            "floating_point_certified": False,
        }
