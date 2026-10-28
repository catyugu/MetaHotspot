"""Static interface lift and positive-real rational enrichment.

Coordinates are [all physical interface temperatures, interior coefficients].
Only the interior dynamics are reduced; patch patterns select enrichment loads,
not a smaller interface. All operators use a fixed, nonparametric component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from assembly import CHIP_RECTS_MM, MM, Model
from scipy import special

# Bound the temporary storage used by multi-RHS sparse solves and projection.
BLOCK_SIZE = 32


def factorize(operator):
    """Use a symmetric graph ordering for the thermal stiffness operators."""
    return spl.splu(operator, permc_spec="MMD_AT_PLUS_A")


def patterns(model: Model, patches: int = 2) -> np.ndarray:
    """Independent bilinear hats on each physical silicon/TIM interface.

    Patch patterns only choose enrichment loads; every interface DOF survives.
    Coordinates are used because active nodes no longer form a full rectangle.
    """
    if patches < 1:
        raise ValueError("patches must be positive")
    xy = model.points[: model.m, :2] / MM
    knots = np.linspace(0.0, 1.0, patches + 1)
    columns = []
    for x0, y0, x1, y1 in CHIP_RECTS_MM:
        local = (xy - [x0, y0]) / [x1 - x0, y1 - y0]
        inside = np.all((local >= -1e-12) & (local <= 1 + 1e-12), axis=1)
        hats = np.maximum(0.0, 1.0 - patches * np.abs(local[:, :, None] - knots))
        for a in range(patches + 1):
            columns.extend(
                inside * hats[:, 0, a] * hats[:, 1, b] for b in range(patches + 1)
            )
    block = np.column_stack(columns)
    scale = np.sqrt(np.sum(model.D_gamma[:, None] * block**2, axis=0))
    if np.any(scale == 0):
        raise ValueError("patch grid is too fine for this interface mesh")
    return block / scale


def port_inputs(model: Model, interface: np.ndarray | None = None) -> sp.csc_matrix:
    """Collocated loads: area-weighted patches (default audit: all flux ports).

    With interface=None, keep the original sqrt(area) flux scaling. Otherwise
    interface contains temperature patterns U and the flux loads are D U.
    A sparse map avoids allocating an N-by-m dense identity-like input block.
    """
    flux = (
        sp.diags(np.sqrt(model.D_gamma))
        if interface is None
        else sp.csc_matrix(model.D_gamma[:, None] * interface)
    )
    flux = sp.vstack([flux, sp.csc_matrix((model.n, flux.shape[1]))])
    return sp.hstack([flux, sp.csc_matrix(model.B_P)], format="csc")


def orthonormalize(block: np.ndarray, against: np.ndarray, tol=1e-9) -> np.ndarray:
    """Rank-revealing QR in capacity-scaled coordinates, with two passes.

    Normalise each incoming column before deflation so the rank decision does
    not compare interface-velocity units with power-source units. Test the
    residual against that original unit scale, not against a vanishing block.
    """
    lengths = la.norm(block, axis=0)
    block = block[:, lengths > 0.0] / lengths[lengths > 0.0]
    if not block.shape[1]:
        return block
    for _ in range(2):
        block -= against @ (against.T @ block)
    q, r, _ = la.qr(block, mode="economic", pivoting=True, check_finite=False)
    rank = int(np.count_nonzero(np.abs(np.diag(r)) > tol))
    return q[:, :rank]


def spectral_bounds(interior, weights, factor) -> tuple[float, float]:
    """Estimate the extreme eigenvalues of C^(-1/2) A C^(-1/2).

    Reuse the lift factorization for inverse iteration; the upper end needs
    only sparse matvecs. These are numerical estimates, not certified bounds.
    """
    root = np.sqrt(weights)
    scale = sp.diags(1.0 / root)
    normalized = (scale @ interior @ scale).tocsc()
    inverse = spl.LinearOperator(
        interior.shape, matvec=lambda x: root * factor.solve(root * x), dtype=float
    )
    start = np.random.default_rng(0).standard_normal(weights.size)
    low = spl.eigsh(
        normalized,
        k=1,
        sigma=0.0,
        which="LM",
        OPinv=inverse,
        v0=start,
        tol=1e-6,
        return_eigenvectors=False,
    )[0]
    high = spl.eigsh(
        normalized, k=1, which="LA", v0=start, tol=1e-6, return_eigenvectors=False
    )[0]
    return float(low) * (1.0 - 1e-5), float(high) * (1.0 + 1e-5)


def elliptic_plan(epsilon: float, lam_min: float, lam_max: float) -> np.ndarray:
    """Zero moment followed by MPMM elliptic positive-real matching points.

    The Zolotarev estimate sets a sampling budget, not a measured ROM error
    tolerance. No frequency band is used to truncate the interior spectrum.
    """
    if not 0.0 < epsilon < 1.0:
        raise ValueError("shift tolerance must lie strictly between zero and one")
    if not 0.0 < lam_min <= lam_max or not np.isfinite(lam_max):
        raise ValueError("the interior spectral interval must be finite and positive")
    log_ratio = math.log(lam_max) - math.log(lam_min)
    count = max(
        1,
        math.ceil(
            (math.log(4.0) - math.log(epsilon))
            * (math.log(4.0) + log_ratio)
            / math.pi**2
        ),
    )
    complement = math.exp(-2.0 * log_ratio)
    parameter = 1.0 - complement
    if parameter == 1.0:
        raise ValueError("spectral ratio exceeds the precision of the elliptic rule")
    theta = (2.0 * np.arange(count) + 1.0) * special.ellipkm1(complement) / (2 * count)
    dn = special.ellipj(theta, parameter)[2]
    return np.r_[0.0, np.sort(lam_max * dn)]


@dataclass
class Basis:
    model: Model
    lift: np.ndarray  # n x m, exact static interface response
    directions: np.ndarray  # n x r, capacity-orthonormal
    ends: list[int]  # nested spaces after each retained shift block
    shifts: np.ndarray
    spectrum: tuple[float, float]
    lift_residual: float
    zero_moment_error: float

    def project_inputs(self, inputs: sp.csc_matrix) -> np.ndarray:
        """Apply T^T without constructing T = [[I, 0], [lift, directions]]."""
        m = self.model.m
        return np.vstack(
            [
                inputs[:m].toarray() + (inputs[m:].T @ self.lift).T,
                (inputs[m:].T @ self.directions).T,
            ]
        )

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        """Block congruence projection; no full transform or weighted lift copy.

        A lift = -K_IG makes the stiffness cross block zero. Use that identity
        instead of multiplying dense lift columns by their roundoff residuals.
        """
        model, lift, w = self.model, self.lift, self.directions
        m, r = model.m, w.shape[1]
        c = model.capacity[m:]
        a, coupling = model.K[m:, m:], model.K[m:, :m]
        capacity = np.empty((m + r, m + r))
        stiffness = np.zeros_like(capacity)
        stiffness[:m, :m] = model.K[:m, :m].toarray() + coupling.T @ lift
        for first in range(0, m, BLOCK_SIZE):
            cols = slice(first, min(first + BLOCK_SIZE, m))
            weighted = c[:, None] * lift[:, cols]
            capacity[:m, cols] = lift.T @ weighted
            capacity[m:, cols] = w.T @ weighted
        capacity[np.arange(m), np.arange(m)] += model.capacity[:m]
        capacity[:m, m:] = capacity[m:, :m].T
        capacity[m:, m:] = w.T @ (c[:, None] * w)
        stiffness[m:, m:] = w.T @ (a @ w)
        return (capacity + capacity.T) / 2, (stiffness + stiffness.T) / 2


def extract(model: Model, patches: int = 2, epsilon: float = 1e-3) -> Basis:
    """Exact static lift plus shifted responses to [C lift U, B_power]."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("shift tolerance must lie strictly between zero and one")
    u = patterns(model, patches)
    m = model.m
    a = model.K[m:, m:]
    c = model.capacity[m:]
    coupling = model.K[m:, :m]
    factor = factorize(a)
    lift = np.empty((model.n, m), order="F")
    residual2, load2 = 0.0, 0.0
    for first in range(0, m, BLOCK_SIZE):
        cols = slice(first, min(first + BLOCK_SIZE, m))
        rhs = -coupling[:, cols].toarray()
        lift[:, cols] = factor.solve(rhs)
        residual2 += la.norm(a @ lift[:, cols] - rhs) ** 2
        load2 += la.norm(rhs) ** 2
    lift_residual = math.sqrt(residual2 / load2)
    spectrum = spectral_bounds(a, c, factor)
    shifts = elliptic_plan(epsilon, *spectrum)
    stimulus = np.column_stack([c[:, None] * (lift @ u), model.B_P[m:]])
    # Capacity-scaled storage permits deflation without repeatedly copying and
    # weighting the accumulated basis. Convert to physical directions once.
    vectors = np.empty((model.n, min(model.n, shifts.size * stimulus.shape[1])))
    root = np.sqrt(c)
    used, ends = 0, []
    zero_moment_error = 0.0
    capacity = sp.diags(c, format="csc")
    for index, shift in enumerate(shifts):
        if shift != 0.0:
            factor = factorize(a + shift * capacity)
        response = factor.solve(stimulus)
        # Release this LU before allocating the next shifted factorization.
        del factor
        block = orthonormalize(root[:, None] * response, vectors[:, :used])
        count = min(block.shape[1], model.n - used)
        vectors[:, used : used + count] = block[:, :count]
        used += count
        if shift == 0.0:
            w = vectors[:, :used] / root[:, None]
            reduced_load = w.T @ stimulus
            exact = stimulus.T @ response
            reduced = reduced_load.T @ la.solve(
                w.T @ (a @ w), reduced_load, assume_a="pos"
            )
            scale = np.sqrt(np.diag(exact))
            normalizer = scale[:, None] * scale
            zero_moment_error = float(
                la.norm((exact - reduced) / normalizer) / la.norm(exact / normalizer)
            )
        if not ends or used != ends[-1]:
            ends.append(used)
        if used == model.n:
            shifts = shifts[: index + 1]
            break
    directions = vectors[:, :used].copy()
    directions /= root[:, None]
    return Basis(
        model,
        lift,
        directions,
        ends,
        shifts,
        spectrum,
        lift_residual,
        zero_moment_error,
    )
