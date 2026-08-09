#!/usr/bin/env python3
"""Shared operator-level utilities for macromodel (MOR) experiments.

This module is deliberately *model-agnostic*: it operates only on the generic
`Operators` interface (K, C, f sparse matrices + rhs) and plain numpy/scipy
data.  No geometry, mesh, material or config type appears here, so any
reduction method (BCI-FANTASTIC, tangential rational Krylov, ...) can be
built on top of whatever model supplies its operators — the method only sees
the K/C/f interface, not the concrete model internals.

Contents
--------
* dense helpers:            symmetric_dense, eigenpairs_descending
* MPMM frequency sampling:  mpmm_elliptic_shift_count, mpmm_elliptic_shifts
* Krylov enrichment:        orthonormalize_block, reduced_response, response_error
* sparse operator helpers:  normalized_operators
* parametric basis:         random_parameter_vectors, build_parametric_basis
* BCI Galerkin projection:  project_bci
* ROM linear solves:        assemble_reduced_k, solve_rom_steady, solve_rom_transient
* accuracy metrics:         temperature_error_metrics, accuracy_summary, format_accuracy
* mesh helpers:             grid_cells, coordinate_map

The accuracy metrics and mesh helpers were folded in from the former
``experiment_setup.py`` when that module was absorbed here; everything remains
model-agnostic (works on ``Compiled`` objects / plain arrays, never on a
concrete model config).
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import special

from metahotspot.compiled import Operators


# ---------------------------------------------------------------------------
# dense linear algebra
# ---------------------------------------------------------------------------


def symmetric_dense(matrix) -> np.ndarray:
    """Symmetric part of a dense matrix (0.5*(M + M^T))."""
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def eigenpairs_descending(matrix):
    """Symmetric eigen-decomposition, eigenpairs sorted by descending value."""
    values, vectors = scipy.linalg.eigh(symmetric_dense(matrix), check_finite=False)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


# ---------------------------------------------------------------------------
# MPMM elliptic-optimal frequency sampling  (FANTASTIC 2014 eqs. 4-5)
# ---------------------------------------------------------------------------


def mpmm_elliptic_shift_count(
    relative_epsilon: float, lambda_min: float, lambda_max: float
):
    """Smallest m satisfying 4*exp(-m*pi^2/log(4/k')) <= eps (Extended FANTASTIC eq.).

    This is the Zolotarev-optimal point count of the MPMM method (Codecasa
    et al.): the relative error of the elliptic-rational approximation decays
    as ``4 exp(-m pi^2 / log(4/k'))`` with the *first* power of m.  The ``m^2``
    exponent in the THERMINIC 2014 preprint rendering is a typesetting
    artefact; the 2021 journal formulation and the exponential-convergence
    rate agree on the linear-in-m exponent.
    """
    kappa = lambda_max / max(lambda_min, np.finfo(float).tiny)
    k_prime = lambda_min / lambda_max
    log_term = max(math.log(4.0 / k_prime), np.finfo(float).tiny)
    for m in range(1, 400):
        if 4.0 * math.exp(-m * math.pi * math.pi / log_term) <= relative_epsilon:
            return m
    raise RuntimeError("MPMM elliptic shift count did not converge")


def mpmm_elliptic_shifts(count: int, lambda_max: float, kappa: float) -> np.ndarray:
    """Elliptic-dn distributed real shifts on [0, lambda_max] (FANTASTIC 2014 eq. 5)."""
    modulus = 1.0 - 1.0 / (kappa * kappa)
    modulus = float(np.clip(modulus, 0.0, 1.0 - 1.0e-12))
    k_complete = special.ellipk(modulus)
    theta = (2.0 * np.arange(1, count + 1) - 1.0) * k_complete / count
    _, _, dn_values, _ = special.ellipj(theta, modulus)
    shifts = lambda_max * np.asarray(dn_values, dtype=np.float64)
    return np.sort(shifts)[::-1]


# ---------------------------------------------------------------------------
# residual-driven Krylov enrichment
# ---------------------------------------------------------------------------


def orthonormalize_block(basis, vectors):
    """Orthonormalize ``vectors`` against the current column-orthonormal ``basis``.

    Returns the new orthonormal columns (a basis for the span of the input that
    is orthogonal to ``basis``).  ``basis`` may be empty (shape (n, 0)).
    """
    block = np.asarray(vectors, dtype=np.float64).copy()
    if not block.size:
        return np.empty((block.shape[0], 0), dtype=np.float64)

    for _ in range(2):
        if basis.shape[1]:
            block -= basis @ (basis.T @ block)

    q, r, _ = scipy.linalg.qr(
        block,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    diagonal = np.abs(np.diag(r))
    if not diagonal.size or diagonal[0] == 0.0:
        return np.empty((block.shape[0], 0), dtype=np.float64)
    keep = diagonal > np.finfo(float).eps * max(block.shape) * diagonal[0]
    return np.ascontiguousarray(q[:, keep])


def reduced_response(basis, A, B_dense):
    """Solve the projected system (basis^T A basis) x = -(basis^T B_dense)."""
    if not basis.shape[1]:
        return np.empty((0, B_dense.shape[1]), dtype=np.float64)
    reduced_A = symmetric_dense(basis.T @ (A @ basis))
    factor = scipy.linalg.cho_factor(
        reduced_A,
        lower=True,
        overwrite_a=False,
        check_finite=False,
    )
    return scipy.linalg.cho_solve(
        factor,
        -(basis.T @ B_dense),
        overwrite_b=False,
        check_finite=False,
    )


def response_error(response, basis, reduced, A, reference):
    """Residual of the current reduced model against the full response.

    Returns ``(error_response, error_eigenvalues, error_tangents, score)`` where
    ``score`` is the A-energy norm of the worst residual direction relative to
    ``reference`` (the response Gramian's leading eigenvalue).
    """
    error_response = response - basis @ reduced if basis.shape[1] else response
    error_gram = symmetric_dense(error_response.T @ (A @ error_response))
    values, tangents = eigenpairs_descending(error_gram)
    score = math.sqrt(float(values[0]) / reference)
    return error_response, values, tangents, score


# ---------------------------------------------------------------------------
# sparse operator helpers
# ---------------------------------------------------------------------------


def normalized_operators(K, C, f) -> Operators:
    """CSC-normalize K, C (eliminate explicit zeros) and copy f."""
    K = sp.csc_matrix(K)
    C = sp.csc_matrix(C)
    K.eliminate_zeros()
    C.eliminate_zeros()
    return Operators(K, C, np.asarray(f, dtype=np.float64).copy())


# ---------------------------------------------------------------------------
# per-port spectral bounds  (FANTASTIC 2014 step 1, Extended FANTASTIC line 2)
# ---------------------------------------------------------------------------


EIGENBOUND_SUBSPACE_CAP = 64  # block-Krylov dimension for per-port bounds


def _global_eigenvalue_bounds(K, C) -> tuple[float, float]:
    """Global ``(lambda_min, lambda_max)`` of the generalized pencil ``K x = λ C x``."""
    vals_high, _ = spla.eigsh(K, k=1, M=C, which="LA")
    vals_low, _ = spla.eigsh(K, k=3, M=C, sigma=0.0, which="LM")
    positive_low = vals_low[vals_low > 1.0e-9]
    if positive_low.size == 0:
        raise RuntimeError("no positive small eigenvalue found")
    return float(positive_low.min()), float(vals_high.max())


def port_eigenvalue_bounds(
    K,
    C,
    g,
    *,
    shift=1.0e-6,
    cap=EIGENBOUND_SUBSPACE_CAP,
) -> tuple[float, float]:
    """Per-port spectral bounds ``(lambda_min, lambda_max)`` for source shape ``g``.

    FANTASTIC 2014 step 1: the min/max eigenvalues are estimated *with
    respect to the power impulse thermal response of the i-th heat source*,
    i.e. only the part of the ``(K, C)`` pencil that the source actually
    excites is considered — a global eigsh over the whole pencil over-
    broadens the shift set.  The pencil is projected onto the Krylov
    subspace the source drives:

    * slow end  ``{(shift*C + K)^-1 C}^n g`` — one factorization at a small
      positive ``shift`` (well-posed despite Neumann-singular K), the rest
      are triangular back-solves; the projected pencil's min positive
      eigenvalue.
    * fast end  ``{C^-1 K}^n g`` — C is the FVM diagonal mass matrix, so
      ``C^-1 K`` is a plain sparse matvec; the projected pencil's max
      eigenvalue.

    Returns ``(lambda_min, lambda_max)``; falls back to the global ``eigsh``
    bounds when the per-port projection yields a degenerate interval.
    """
    K = K.tocsc()
    C = C.tocsc()
    c_diag = np.asarray(C.diagonal()).ravel()
    if np.any(c_diag <= 0.0):
        raise ValueError("port_eigenvalue_bounds: C must have a positive diagonal")
    g = np.asarray(g, dtype=np.float64).ravel()
    n = K.shape[0]
    scale = float(np.median(c_diag))
    g_norm = max(float(np.linalg.norm(g)), np.finfo(float).tiny)

    def project_spectrum(Q, keep):
        if keep < 2:
            return None
        Q = np.ascontiguousarray(Q[:, :keep])
        Kp = Q.T @ (K @ Q)
        Cp = Q.T @ (C @ Q)
        Kp = 0.5 * (Kp + Kp.T)
        Cp = 0.5 * (Cp + Cp.T)
        try:
            return scipy.linalg.eigvalsh(Kp, Cp, check_finite=False)
        except Exception:
            return None

    def krylov(apply_step):
        Q = np.zeros((n, cap))
        Q[:, 0] = g / g_norm
        keep = 1
        for _ in range(1, cap):
            w = apply_step(Q[:, keep - 1])
            for _rep in range(2):
                w = w - Q[:, :keep] @ (Q[:, :keep].T @ w)
            nrm = float(np.linalg.norm(w))
            if nrm < 1.0e-10:
                break
            Q[:, keep] = w / nrm
            keep += 1
        return Q, keep

    # -- slow end: shift-invert block-Krylov, min positive eigenvalue ------
    s0 = max(float(shift), scale * 1.0e-6)
    lu = spla.splu((K + s0 * C).tocsc())
    Qs, ks = krylov(lambda v: lu.solve(C @ v))
    vals = project_spectrum(Qs, ks)
    positive = vals[vals > 1.0e-9] if vals is not None else np.empty(0)
    lambda_min = float(positive.min()) if positive.size else None

    # -- fast end: forward block-Krylov, max eigenvalue --------------------
    Qf, kf = krylov(lambda v: (K @ v) / c_diag)
    vals = project_spectrum(Qf, kf)
    lambda_max = float(vals.max()) if vals is not None and vals.size else None

    if (
        lambda_min is not None
        and lambda_max is not None
        and lambda_max > lambda_min
    ):
        return lambda_min, lambda_max
    return _global_eigenvalue_bounds(K, C)


# ---------------------------------------------------------------------------
# BCI Galerkin projection  (FANTASTIC 2014 + BCI matrix reduction 2015)
# ---------------------------------------------------------------------------


def project_bci(
    operators: Operators,
    source_shape: np.ndarray,
    boundary_terms,
    basis,
    boundary_epsilon=1.0e-3,
):
    """Project the full-domain operators onto ``basis``, exposing BCI ports.

    Builds the reduced (Ĉ, K̂0, F̂, F̂_ϑ, A_k) of a boundary-condition-independent
    DCTM following Extended FANTASTIC (Codecasa et al. 2021, eqs. 9-16 and the
    boundary-DoF definition of Section IV-B):

    * the source ports enter the RHS through ``F̂ = Vᵀ G_src`` and the junction
      temperatures read ``T_j = T0 + F̂ᵀ θ``;
    * the boundary is exposed as an explicit port: the rows of ``V`` on the
      boundary cells form ``V_∂Ω``, whose SVD ``V_∂Ω = U_∂Ω Σ_∂Ω W_∂Ωᵀ`` is
      truncated at singular values below ``boundary_epsilon``; the boundary
      output matrix is ``F̂_ϑ = W_∂Ω Σ_∂Ω`` (M̂ × Θ) so ``T_ϑ = T0 + F̂_ϑᵀ θ``;
    * each boundary group k contributes the affine term ``h_k F̂_ϑ A_k F̂_ϑᵀ``
      with ``A_k = U_∂Ωᵀ H_k U_∂Ω`` (Θ × Θ, H_k the group's exposed-area
      diagonal), so the effective stiffness is ``K̂(h) = K̂0 + Σ_k h_k F̂_ϑ A_k
      F̂_ϑᵀ`` — the projected HTC matrix of the ambient coupling
      ``P_ϑ = -Ĥ(T_ϑ - T_A e_Θ)``, ``Ĥ = Σ_k h_k A_k``.

    ``operators`` is the full-domain h-free ``(K, C, f)``, ``source_shape``
    the (N, n_src) source-shape matrix ``G_src``, and ``boundary_terms`` a
    list of diagonal sparse ``H_k`` (exposed area per cell, one per group).
    Interior modes are *rise* coordinates above ambient.

    Returns ``(C_hat, K_hat0, F_hat, F_bdry, A_bdry)`` where ``F_bdry`` is the
    (M̂, Θ) boundary output matrix and ``A_bdry[k]`` the (Θ, Θ) HTC matrix of
    boundary group k.
    """

    def project(matrix):
        reduced = sp.csc_matrix(basis.T @ matrix @ basis)
        reduced = (0.5 * (reduced + reduced.T)).tocsc()
        reduced.eliminate_zeros()
        return reduced

    C_hat = project(operators.C)
    K_hat0 = project(operators.K)
    F_hat = np.asarray(basis.T @ source_shape, dtype=np.float64)

    # Boundary cells: union of cells carrying any exposed area.
    n_cell = operators.K.shape[0]
    b_diag = np.zeros(n_cell)
    for term in boundary_terms:
        b_diag += np.asarray(term.diagonal()).ravel()
    b_cells = np.flatnonzero(b_diag > 0.0)
    if b_cells.size == 0:
        raise ValueError("no boundary cells found for BCI port extraction")

    # V_∂Ω = rows of V on boundary cells; SVD, truncated at boundary_epsilon.
    V_d = np.asarray(basis[b_cells, :])
    U_d, s_d, Wt_d = np.linalg.svd(V_d, full_matrices=False)
    keep = np.flatnonzero(s_d > boundary_epsilon * s_d[0])
    theta = keep.size
    if theta == 0:
        raise RuntimeError("boundary SVD retained no modes")
    U_t = np.ascontiguousarray(U_d[:, keep])  # (M_∂Ω, Θ)
    S_t = s_d[keep]  # (Θ,)
    W_t = np.ascontiguousarray(Wt_d[keep, :].T)  # (M̂, Θ)
    F_bdry = W_t * S_t  # (M̂, Θ) = W Σ

    # Per-group HTC matrices A_k = U_∂Ωᵀ H_k U_∂Ω (Θ, Θ), in boundary-cell frame.
    A_bdry = []
    for term in boundary_terms:
        Hk_b = np.asarray(term.diagonal()).ravel()[b_cells]
        A_k = U_t.T @ (Hk_b[:, None] * U_t)  # Θ×Θ
        A_bdry.append((0.5 * (A_k + A_k.T)).astype(np.float64))

    return C_hat, K_hat0, F_hat, F_bdry, A_bdry


def assemble_reduced_k(K_hat0, F_bdry, A_bdry, h_vec) -> sp.csc_matrix:
    """``K̂(h) = K̂0 + Σ_k h_k F̂_ϑ A_k F̂_ϑᵀ`` for admissible ``h_vec``."""
    K = K_hat0.tocsc()
    for h, A in zip(h_vec, A_bdry):
        K = K + h * sp.csc_matrix(F_bdry @ A @ F_bdry.T)
    return K.tocsc()


def solve_rom_steady(K_hat, F_hat, power) -> np.ndarray:
    """Steady reduced interior: ``K̂(h) θ = F̂ P`` (K̂ is SPD for h > 0)."""
    rhs = F_hat @ np.asarray(power, dtype=np.float64)
    return np.asarray(sp.linalg.spsolve(K_hat.tocsc(), rhs)).ravel()


def solve_rom_transient(
    C_hat,
    K_hat,
    F_hat,
    power_t,
    dt: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF1 transient of the reduced interior.

    ``(Ĉ/dt + K̂) θ_{n+1} = Ĉ θ_n/dt + F̂ P(t_{n+1})`` on the same output grid
    as the model reference (0, dt, 2·dt, …, duration).  ``K_hat`` is the
    already-assembled ``K̂(h)``; ``power_t`` is a callable ``t -> P(t)``.
    Returns ``(times, theta_history)``.
    """
    n_modes = C_hat.shape[0]
    A = K_hat.tocsc()
    lhs = (C_hat / dt + A).tocsc()
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    history = np.empty((times.size, n_modes), dtype=np.float64)
    theta = np.zeros(n_modes)
    solver = sp.linalg.splu(lhs)
    for i, t in enumerate(times):
        rhs = (C_hat @ theta) / dt + F_hat @ np.asarray(power_t(t), dtype=np.float64)
        theta = solver.solve(rhs)
        history[i] = theta
    return times, history


# ---------------------------------------------------------------------------
# parametric basis construction (FANTASTIC BCI 2015 Algorithm 1)
# ---------------------------------------------------------------------------

TARGET_RELATIVE_EPSILON = 1.0e-3  # elliptic shift-count target (Extended FANTASTIC eq.)
RESIDUAL_TOLERANCE = 1.0e-3  # residual-driven enrichment stop tolerance
MAX_ORDER = 2048
RANDOM_PARAMETER_SAMPLES = 20  # random h-vectors for training (Algorithm 1)
PER_PORT_SAMPLES = 6  # random h-vectors per (port, shift) candidate
RANDOM_SEED = 20260805


def random_parameter_vectors(h_ranges, sample_count, seed, boundaries=None):
    """Random admissible h-vectors (one h per group), log-uniform.  No greedy.

    ``h_ranges`` is an ``(n_groups, 2)`` array of admissible ``(lo, hi)``
    ranges, one row per affine parameter; the log-uniform draws are vectorized
    across groups.  FANTASTIC BCI 2015 Algorithm 1: parameters chosen at
    random to avoid reduced-basis greedy stagnation.  ``boundaries`` (geometric
    holdout) are appended so the certified range is covered at its extremes.
    """
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    lows = np.log10(h_ranges[:, 0])
    highs = np.log10(h_ranges[:, 1])
    rng = np.random.default_rng(seed)
    draws = 10.0 ** rng.uniform(lows, highs, size=(sample_count, h_ranges.shape[0]))
    vectors = [tuple(row) for row in draws]
    for b in boundaries or ():
        vectors.append(tuple(b))
    seen, out = set(), []
    for v in vectors:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_parametric_basis(
    operators,
    source_shape,
    boundary_terms,
    h_ranges,
    *,
    boundaries=None,
    residual_tolerance=RESIDUAL_TOLERANCE,
    max_order=MAX_ORDER,
    target_relative_epsilon=TARGET_RELATIVE_EPSILON,
    sample_count=PER_PORT_SAMPLES,
    seed=RANDOM_SEED,
):
    """Certified full-domain basis by per-port residual-driven enrichment.

    Faithful realization of the Extended FANTASTIC Algorithm 1 (Codecasa
    et al. 2021) — the algorithm behind Simcenter Flotherm's BCI-ROM — as far
    as the model-agnostic ``(K, C, f, G_src, H_k)`` interface allows:

    * **per heat-source port** the spectral bounds ``(λ_i, Λ_i)`` of the
      power-impulse response are estimated (FANTASTIC 2014 step 1, via
      :func:`port_eigenvalue_bounds`), giving its own elliptic shift count
      ``m_i`` (eq. 4) and dn-distributed real shifts (eq. 5);
    * each candidate solves the **single-column** frequency-domain problem
      ``(σM + K + Σ_k h_k H_k) φ = g_k`` for the k-th source shape (paper
      eq. 26), crossed with **fresh random** HTC vectors per (port, shift);
    * residual directions above tolerance are inserted immediately
      (Algorithm 1 line 6: ``||residual|| > ε``), the basis is kept
      column-orthonormal throughout;
    * a final SVD truncates columns whose singular values fall below
      ``ε · σ_max`` (Algorithm 1 closing / FANTASTIC 2014 step 7).

    ``operators`` is the full-domain h-free ``(K, C, f)``, ``source_shape``
    the ``(N, n_src)`` source-shape matrix ``G_src`` whose columns are the
    per-port unit-power shapes, ``boundary_terms`` a list of diagonal sparse
    ``H_k`` (exposed area per cell, one per group), and ``h_ranges`` an
    ``(n_groups, 2)`` array of admissible ``(lo, hi)`` ranges.
    """
    started = time.perf_counter()
    K = operators.K.tocsc()
    C = operators.C.tocsc()
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    G = np.asarray(source_shape, dtype=np.float64)
    n_src = G.shape[1]

    internal_order = K.shape[0]
    order_limit = min(max_order, internal_order)
    # Algorithm 1 line 1: start from the uniform-temperature direction e_M so
    # the basis can represent arbitrary ambient shifts (Extended FANTASTIC).
    basis = np.full((internal_order, 1), 1.0 / math.sqrt(internal_order))
    history = []
    per_port_plans = []
    candidate_total = 0
    processed_count = 0
    pre_svd_order = 1
    worst_score = 0.0
    converged = True

    def candidate_A(h_vec, shift):
        A = K + shift * C
        for h, H in zip(h_vec, boundary_terms):
            A = A + h * H
        return (0.5 * (A + A.T)).tocsc()

    def enrich(g):
        """One candidate: solve, measure residual, insert directions above tol."""
        nonlocal basis, worst_score, converged, processed_count
        A = candidate_A(h_vec, shift)
        response = np.asarray(spla.splu(A).solve(g))
        response_gram = symmetric_dense(response.T @ g)
        response_values, _ = eigenpairs_descending(response_gram)
        reference = max(float(response_values[0]), np.finfo(float).tiny)

        order_before = basis.shape[1]
        reduced = reduced_response(basis, A, -g)
        error_response, error_values, tangents, score_before = response_error(
            response,
            basis,
            reduced,
            A,
            reference,
        )
        requested = int(
            np.count_nonzero(error_values > residual_tolerance**2 * reference)
        )
        available = order_limit - basis.shape[1]
        count = min(requested, available)

        added = 0
        if count:
            block = orthonormalize_block(basis, error_response @ tangents[:, :count])
            if not block.shape[1]:
                raise RuntimeError("rational Krylov enrichment stalled")
            basis = np.column_stack((basis, block))
            added = block.shape[1]

        if count == requested and added == count:
            score_after = (
                math.sqrt(float(error_values[count]) / reference)
                if count < error_values.size
                else 0.0
            )
        else:
            reduced = reduced_response(basis, A, -g)
            _, _, _, score_after = response_error(
                response, basis, reduced, A, reference
            )

        worst_score = max(worst_score, score_after)
        processed_count += 1
        history.append(
            {
                "port": port,
                "order_before": int(order_before),
                "order_after": int(basis.shape[1]),
                "score_before": float(score_before),
                "score_after": float(score_after),
                "h_vec": h_vec,
                "shift": float(shift),
                "requested": int(requested),
                "added": int(added),
            }
        )
        if requested > available or score_after > residual_tolerance:
            converged = False

    for port in range(n_src):
        g = G[:, port : port + 1]
        try:
            lambda_min, lambda_max = port_eigenvalue_bounds(K, C, g)
        except Exception as exc:
            raise RuntimeError(
                "per-port spectrum estimation failed for source port "
                f"{port} ({exc}); falling back would need the global pencil"
            ) from exc
        kappa = lambda_max / max(lambda_min, np.finfo(float).tiny)
        elliptic_count = mpmm_elliptic_shift_count(
            target_relative_epsilon, lambda_min, lambda_max
        )
        shifts = np.r_[0.0, mpmm_elliptic_shifts(elliptic_count, lambda_max, kappa)]
        per_port_plans.append(
            {
                "port": int(port),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "kappa": float(kappa),
                "shift_count": int(elliptic_count) + 1,  # + steady-state shift 0
                "shifts_per_s": shifts.tolist(),
            }
        )

        for shift in shifts:
            if not converged:
                break
            # fresh random h-vectors per (port, shift): deterministic sub-seed
            sub_seed = seed + 100003 * port + int(round(float(shift) * 1.0e6))
            h_vecs = random_parameter_vectors(
                h_ranges, sample_count, sub_seed, boundaries
            )
            candidate_total += len(h_vecs)
            for h_vec in h_vecs:
                if not converged:
                    break
                enrich(g)
        if not converged:
            break

    # Algorithm 1 closing: SVD-truncate the basis at the relative error ε.
    pre_svd_order = int(basis.shape[1])
    U_b, s_b, Vt_b = scipy.linalg.svd(
        np.asarray(basis), full_matrices=False, check_finite=False
    )
    s_cut = residual_tolerance * max(float(s_b[0]), np.finfo(float).tiny)
    keep = np.flatnonzero(s_b > s_cut)
    basis = np.ascontiguousarray(U_b[:, keep])

    if basis.shape[1]:
        orthogonality = basis.T @ basis - np.eye(basis.shape[1])
        orthogonality_error = float(np.max(np.abs(orthogonality)))
    else:
        orthogonality_error = 0.0
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    return basis, {
        "per_port_plans": per_port_plans,
        "target_relative_epsilon": target_relative_epsilon,
        "parameter_sampling": "random per (port, shift) (FANTASTIC BCI Algorithm 1)",
        "candidate_count": candidate_total,
        "processed_candidate_count": processed_count,
        "basis_order": int(basis.shape[1]),
        "pre_svd_order": pre_svd_order,
        "svd_kept_order": int(basis.shape[1]),
        "maximum_order": int(order_limit),
        "orthogonality_error": orthogonality_error,
        "relative_response_error": float(worst_score),
        "residual_tolerance": residual_tolerance,
        "converged": bool(converged),
        "history": history,
        "seconds": time.perf_counter() - started,
        "memory_strategy": (
            "stream one single-column frequency-domain response per candidate; "
            "form the residual Gramian directly; never cache candidates or "
            "repeat global scans; per-port eigenvalue/shift planning"
        ),
    }


# ---------------------------------------------------------------------------
# accuracy metrics and mesh helpers  (folded in from experiment_setup.py)
# ---------------------------------------------------------------------------

MAX_RELATIVE_RISE_ERROR = 0.01


def temperature_error_metrics(reference, approximation, ambient_K: float) -> dict:
    reference = np.asarray(reference)
    approximation = np.asarray(approximation)
    absolute_error = float(np.max(np.abs(approximation - reference)))
    reference_rise = float(np.max(np.abs(reference - ambient_K)))
    relative_error = (
        absolute_error / reference_rise
        if reference_rise
        else float(absolute_error != 0.0)
    )
    return {
        "reference_temperature_range_K": [
            float(reference.min()),
            float(reference.max()),
        ],
        "max_absolute_rise_error_K": absolute_error,
        "max_relative_rise_error": relative_error,
        "passed": relative_error < MAX_RELATIVE_RISE_ERROR,
    }


def accuracy_summary(
    reference_steady,
    reduced_steady,
    reference_history,
    reduced_history,
    ambient_K: float,
) -> dict:
    steady = temperature_error_metrics(reference_steady, reduced_steady, ambient_K)
    transient = temperature_error_metrics(
        reference_history[-1], reduced_history[-1], ambient_K
    )
    return {
        "steady_reference_temperature_range_K": steady["reference_temperature_range_K"],
        "transient_final_reference_temperature_range_K": transient[
            "reference_temperature_range_K"
        ],
        "steady_max_absolute_rise_error_K": steady["max_absolute_rise_error_K"],
        "steady_max_relative_rise_error": steady["max_relative_rise_error"],
        "transient_final_max_absolute_rise_error_K": transient[
            "max_absolute_rise_error_K"
        ],
        "transient_final_max_relative_rise_error": transient["max_relative_rise_error"],
        "accuracy_passed": steady["passed"] and transient["passed"],
    }


def format_accuracy(summary: dict) -> str:
    steady_range = summary["steady_reference_temperature_range_K"]
    transient_range = summary["transient_final_reference_temperature_range_K"]
    return (
        f"reference range steady={steady_range[0]:.3f}..{steady_range[1]:.3f} K, "
        f"transient final={transient_range[0]:.3f}..{transient_range[1]:.3f} K; "
        f"rise error steady={summary['steady_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['steady_max_relative_rise_error']:.3%}, transient final="
        f"{summary['transient_final_max_absolute_rise_error_K']:.5f} K/"
        f"{summary['transient_final_max_relative_rise_error']:.3%}"
    )


def grid_cells(compiled) -> np.ndarray:
    return compiled.grid_to_cell.reshape(compiled.nx, compiled.ny, compiled.nz)


def coordinate_map(source, target, z_offset: int, label: str) -> np.ndarray:
    if source.nx != target.nx or source.ny != target.ny:
        raise RuntimeError(f"{label}: lateral meshes differ")
    source_grid = grid_cells(source)
    target_grid = grid_cells(target)[:, :, z_offset : z_offset + source.nz]
    if target_grid.shape != source_grid.shape:
        raise RuntimeError(f"{label}: z range differs")
    valid = source_grid >= 0
    if not np.array_equal(valid, target_grid >= 0):
        raise RuntimeError(f"{label}: geometry occupancy differs")

    source_ids = source_grid[valid]
    target_ids = target_grid[valid]
    if (
        source_ids.size != source.cell_count
        or np.unique(source_ids).size != source.cell_count
    ):
        raise RuntimeError(f"{label}: source cell IDs are incomplete")
    mapping = np.empty(source.cell_count, dtype=np.int64)
    mapping[source_ids] = target_ids
    if np.unique(mapping).size != mapping.size:
        raise RuntimeError(f"{label}: target mapping is not one-to-one")
    return mapping
