"""Two-space output error bounds for co-located BDF1 thermal step transfers.

For each parameter and discrete time the absolute error of the delivered
Galerkin transfer is at most |dual correction| plus the primal/dual energy
residual product. Both Riesz Grams use the fixed minimum-HTC operator.
Finite candidate evaluation is not a continuous-parameter certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class StepCertificate:
    projected_source: tuple[np.ndarray, np.ndarray]
    projected_base: tuple[np.ndarray, np.ndarray]
    projected_mass: tuple[np.ndarray, np.ndarray]
    projected_terms: tuple[list[np.ndarray], list[np.ndarray]]
    grams: tuple[np.ndarray, np.ndarray]
    dual_test_primal_residual: np.ndarray
    dt: float

    def evaluate(self, parameter, *, steps: int):
        """Return (ROM transfer, dual correction, bound on uncorrected ROM)."""
        parameter = np.asarray(parameter, dtype=np.float64).ravel()
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if parameter.size != len(self.projected_terms[0]):
            raise ValueError("parameter dimension does not match boundary terms")
        cholesky = []
        for base, mass, terms in zip(
            self.projected_base, self.projected_mass, self.projected_terms
        ):
            operator = base + mass
            for value, term in zip(parameter, terms):
                operator = operator + float(value) * term
            cholesky.append(scipy.linalg.cho_factor(operator, check_finite=False))

        count = self.projected_source[0].shape[1]
        coefficients = [np.zeros_like(source) for source in self.projected_source]
        previous_residual = [
            np.zeros((gram.shape[0], count)) for gram in self.grams
        ]
        primal_norms = np.zeros((steps + 1, count))
        dual_norms = np.zeros((steps + 1, count))
        primal_coordinates = [None]
        dual_increments = [None]
        rom = np.zeros((steps + 1, count, count))
        correction = np.zeros_like(rom)
        bound = np.zeros_like(rom)

        for n in range(1, steps + 1):
            current = []
            for index in range(2):
                source = self.projected_source[index]
                mass = self.projected_mass[index]
                previous = coefficients[index]
                updated = scipy.linalg.cho_solve(
                    cholesky[index], source + mass @ previous,
                    check_finite=False,
                )
                coords = np.vstack((
                    np.eye(count), -updated, -(updated - previous),
                    *(-float(value) * updated for value in parameter),
                ))
                current.append((updated, updated - previous, coords))
                coefficients[index] = updated

            rom[n] = self.projected_source[0].T @ current[0][0]
            primal_coordinates.append(current[0][2])
            dual_increments.append(current[1][1])
            dual_residual = current[1][2] - previous_residual[1]
            for index, coords in enumerate((current[0][2], dual_residual)):
                gram = self.grams[index]
                norms_squared = np.einsum(
                    "ij,ij->j", coords, gram @ coords
                )
                previous_norms = primal_norms if index == 0 else dual_norms
                previous_norms[n] = previous_norms[n - 1] + np.maximum(
                    norms_squared, 0.0
                )
            previous_residual[1] = current[1][2]

            for k in range(1, n + 1):
                correction[n] += dual_increments[n - k + 1].T @ (
                    self.dual_test_primal_residual @ primal_coordinates[k]
                )
            bound[n] = np.abs(correction[n]) + np.outer(
                np.sqrt(dual_norms[n]), np.sqrt(primal_norms[n])
            )
        return rom, correction, bound


def prepare_step_certificate(
    K, C, terms, source, ranges, primal_basis, dual_basis, *, dt,
    minimum_factor=None, residual_cache=None,
):
    """Precompute affine `K(p_min)^-1` residual Grams for both spaces."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    K = sp.csc_matrix(K)
    D = sp.csc_matrix(C) / dt
    terms = [sp.csc_matrix(term) for term in terms]
    ranges = np.asarray(ranges, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    V = np.asarray(primal_basis, dtype=np.float64)
    W = np.asarray(dual_basis, dtype=np.float64)
    if ranges.shape != (len(terms), 2):
        raise ValueError("ranges must have shape (n_terms, 2)")
    if any(basis.ndim != 2 or basis.shape[0] != K.shape[0]
           or basis.shape[1] < 1 for basis in (V, W)):
        raise ValueError("both spaces must be nonempty bases of the full field")
    if source.ndim != 2 or source.shape[0] != K.shape[0]:
        raise ValueError("source must have shape (n_cells, n_ports)")

    minimum = K.copy()
    for value, term in zip(ranges[:, 0], terms):
        minimum = minimum + float(value) * term
    factor = minimum_factor or spla.splu(minimum.tocsc())

    cache = {} if residual_cache is None else residual_cache
    spans = []
    gram = []
    for basis in (V, W):
        key = id(basis)
        if key not in cache:
            span = np.column_stack((
                source, K @ basis, D @ basis,
                *(term @ basis for term in terms)
            ))
            current = span.T @ factor.solve(np.ascontiguousarray(span))
            cache[key] = (span, np.ascontiguousarray((current + current.T) / 2))
        span, current = cache[key]
        spans.append(span)
        gram.append(current)
    return StepCertificate(
        projected_source=(V.T @ source, W.T @ source),
        projected_base=(V.T @ (K @ V), W.T @ (K @ W)),
        projected_mass=(V.T @ (D @ V), W.T @ (D @ W)),
        projected_terms=(
            [V.T @ (term @ V) for term in terms],
            [W.T @ (term @ W) for term in terms],
        ),
        grams=(gram[0], gram[1]),
        dual_test_primal_residual=W.T @ spans[0],
        dt=float(dt),
    )
