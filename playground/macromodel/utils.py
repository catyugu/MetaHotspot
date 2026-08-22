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
* Krylov enrichment:        orthonormalize_block, response_error
* sparse operator helpers:  normalized_operators
* parametric basis:         build_parametric_basis
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
import pyamg

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


# ---------------------------------------------------------------------------
# SPD linear solve  (Extended FANTASTIC 2021 lines 715-720: iterative solver
# warm-started from the reduced-model estimate; no re-factorization)
# ---------------------------------------------------------------------------


def spd_solve(A, b, x0=None, rtol=1.0e-10, maxiter=2000):
    """Solve the SPD system ``A x = b`` by Ruge-Stueben AMG-preconditioned CG.

    ``A`` is the full-domain operator ``(σM + K + Σ_k h_k H_k)`` — symmetric
    positive definite (h > 0 makes the boundary term positive definite and
    σ ≥ 0).  ``x0`` is the warm-start guess (the reduced-model estimate, or
    the previous shift's response), so CG is warm-started (Extended FANTASTIC
    2021) on top of the AMG preconditioner.
    """
    import pyamg

    b = np.asarray(b, dtype=np.float64).ravel()
    if x0 is None:
        x0 = np.zeros(b.size, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64).ravel()

    ml = pyamg.ruge_stuben_solver(A.tocsr())
    M = ml.aspreconditioner(cycle="V")
    x, _ = spla.cg(A, b, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter, M=M)
    return x


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

_inv_precond_cache: dict = {}


def _shift_invert_preconditioner(K, C, s0):
    A = (K + s0 * C).tocsc()
    key = (id(K.data), int(K.nnz), float(s0))
    entry = _inv_precond_cache.get(key)
    if entry is None:
        entry = pyamg.ruge_stuben_solver(A.tocsr()).aspreconditioner(cycle="V")
        _inv_precond_cache[key] = entry
    return A, entry


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
    _Ashot, _precond = _shift_invert_preconditioner(K, C, s0)

    def _inv_apply(v):
        x, _ = spla.cg(
            _Ashot, v, x0=v, rtol=INNER_RTOL, atol=0.0, maxiter=1000, M=_precond
        )
        return x

    Qs, ks = krylov(_inv_apply)
    vals = project_spectrum(Qs, ks)
    positive = vals[vals > 1.0e-9] if vals is not None else np.empty(0)
    lambda_min = float(positive.min()) if positive.size else None

    # -- fast end: forward block-Krylov, max eigenvalue --------------------
    Qf, kf = krylov(lambda v: (K @ v) / c_diag)
    vals = project_spectrum(Qf, kf)
    lambda_max = float(vals.max()) if vals is not None and vals.size else None
    return lambda_min, lambda_max


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
    """Steady reduced interior solved via Jacobi-preconditioned CG.

    Solves K̂(h) θ = F̂ P, where K̂ is SPD for h > 0.
    """
    rhs = F_hat @ np.asarray(power, dtype=np.float64)
    A = K_hat.tocsc().tocsr()

    # ---- Jacobi preconditioner: M = diag(A)^{-1} ----
    diag = A.diagonal()
    M = sp.diags(1.0 / diag, format="csr")

    theta, info = spla.cg(A, rhs, rtol=1e-8, atol=0.0, maxiter=1000, M=M)

    if info != 0:
        print(f"Warning: CG did not converge, info={info}")

    return theta.ravel()


LARGE_SYSTEM_DOF = 1000
TRANSIENT_RTOL = 1.0e-8
TRANSIENT_MAXITER = 10000


def solve_rom_transient(
    C_hat,
    K_hat,
    F_hat,
    power_t,
    dt: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-step BDF1 transient of the reduced interior.

    Uses Jacobi-preconditioned CG (JPCG) instead of AMG.
    """
    n_modes = C_hat.shape[0]
    A = K_hat.tocsc()
    lhs = (C_hat / dt + A).tocsc()
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    history = np.empty((times.size, n_modes), dtype=np.float64)
    theta = np.zeros(n_modes)

    lhs_csr = lhs.tocsr()
    diag = lhs_csr.diagonal()
    M = sp.diags(1.0 / diag, format="csr")

    for i, t in enumerate(times):
        rhs = (C_hat @ theta) / dt + F_hat @ np.asarray(power_t(t), dtype=np.float64)
        theta, _ = spla.cg(
            lhs_csr,
            rhs,
            x0=theta,
            rtol=TRANSIENT_RTOL,
            atol=0.0,
            maxiter=TRANSIENT_MAXITER,
            M=M,
        )
        history[i] = theta
    return times, history


# ---------------------------------------------------------------------------
# parametric basis construction (FANTASTIC BCI 2015 Algorithm 1)
# ---------------------------------------------------------------------------

TARGET_RELATIVE_EPSILON = 1.0e-3  # elliptic shift-count target (Extended FANTASTIC eq.)
RESIDUAL_TOLERANCE = 1.0e-3  # residual-driven enrichment stop tolerance
MAX_ORDER = 2048
PROBE_ROUNDS = 3  # consecutive random h-vectors that must certify a (port, shift)
RANDOM_SEED = 20260805
INNER_RTOL = 1.0e-3
ENRICH_RTOL = 1.0e-6


def random_h(h_ranges, seed) -> tuple[float, ...]:
    """One random admissible h-vector (one h per group), log-uniform.  No greedy.

    ``h_ranges`` is an ``(n_groups, 2)`` array of admissible ``(lo, hi)`` ranges,
    one row per affine parameter; a single log-uniform draw across all groups.
    FANTASTIC BCI 2015 Algorithm 1: parameters chosen at random to avoid
    reduced-basis greedy stagnation.  A deterministic ``seed`` is advanced per
    call so consecutive draws are independent but reproducible.
    """
    h_ranges = np.asarray(h_ranges, dtype=np.float64)
    lows = np.log10(h_ranges[:, 0])
    highs = np.log10(h_ranges[:, 1])
    rng = np.random.default_rng(seed)
    draws = 10.0 ** rng.uniform(lows, highs, size=h_ranges.shape[0])
    return tuple(float(v) for v in draws)


def build_parametric_basis(
    operators,
    source_shape,
    boundary_terms,
    h_ranges,
    *,
    residual_tolerance=RESIDUAL_TOLERANCE,
    max_order=MAX_ORDER,
    target_relative_epsilon=TARGET_RELATIVE_EPSILON,
    probe_rounds=PROBE_ROUNDS,
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
    * for each (port, shift) a random HTC vector is drawn and the **current**
      basis is probed (small dense reduced solve + one sparse matvec); only
      when a probe fails (``ρ > ε``) is a **single-column** full solve
      ``(σM + K + Σ_k h_k H_k) φ = g_k`` (paper eq. 26) done and its residual
      directions inserted (Algorithm 1 line 6), keeping the basis
      column-orthonormal;
    * the **adaptive stop**: the (port, shift) is certified only when
      ``probe_rounds`` consecutive freshly drawn parameters all satisfy
      ``ρ ≤ ε`` — sampling stops on the error estimate, not a hardcoded count;
    * a final SVD truncates columns whose singular values fall below
      ``ε · σ_max`` (Algorithm 1 closing / FANTASTIC 2014 step 7).

    ``operators`` is the full-domain h-free ``(K, C, f)``, ``source_shape``
    the ``(N, n_src)`` source-shape matrix ``G_src`` whose columns are the
    per-port unit-power shapes, ``boundary_terms`` a list of diagonal sparse
    ``H_k`` (exposed area per cell, one per group), and ``h_ranges`` an
    ``(n_groups, 2)`` array of admissible ``(lo, hi)`` ranges.  ``probe_rounds``
    is the number of consecutive random draws that must certify a (port, shift)
    before the model advances (a small robustness constant, not a sample
    budget).
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
    outer_idx = 0
    pre_svd_order = 1
    worst_score = 0.0
    converged = True

    def candidate_A(h_vec, shift):
        A = K + shift * C
        for h, H in zip(h_vec, boundary_terms):
            A = A + h * H
        return A.tocsc()

    H_diags = [np.asarray(H.diagonal()).ravel() for H in boundary_terms]
    c_diag = np.asarray(C.diagonal()).ravel()
    projected: dict = {}
    g_hat: np.ndarray
    g_vec: np.ndarray
    g_norm: float

    def init_port(g):
        nonlocal projected, g_hat, g_vec, g_norm
        g_vec = np.asarray(g, dtype=np.float64).ravel()
        g_norm = max(float(np.linalg.norm(g_vec)), np.finfo(float).tiny)
        projected = {
            "K": symmetric_dense(basis.T @ (K @ basis)),
            "C": symmetric_dense(basis.T @ (C @ basis)),
            "H": [symmetric_dense(basis.T @ (H @ basis)) for H in boundary_terms],
        }
        g_hat = np.asarray(basis.T @ g_vec, dtype=np.float64).ravel()

    def extend_projected(W, B_old):
        """Append block W (basis before append = B_old) to every M̂ and ĝ."""
        nonlocal g_hat
        for key, M in (("K", K), ("C", C)):
            MW = M @ W
            cross = B_old.T @ MW
            projected[key] = np.block(
                [[projected[key], cross], [cross.T, symmetric_dense(W.T @ MW)]]
            )
        for j, H in enumerate(boundary_terms):
            HW = H @ W
            cross = B_old.T @ HW
            projected["H"][j] = np.block(
                [[projected["H"][j], cross], [cross.T, symmetric_dense(W.T @ HW)]]
            )
        g_hat = np.concatenate([g_hat, np.asarray(W.T @ g_vec).ravel()])

    def reduced_solve(h_vec, shift):
        """Solve the small dense reduced system (K̂+σĈ+Σ h_k Ĥ_k) x̂ = ĝ."""
        A_hat = projected["K"].copy()
        if shift:
            A_hat = A_hat + shift * projected["C"]
        for h, H_hat in zip(h_vec, projected["H"]):
            A_hat = A_hat + h * H_hat
        return np.linalg.solve(symmetric_dense(A_hat), g_hat)

    def probe_residual(h_vec, shift):
        """Relative residual ρ of the current basis at (shift, h_vec)."""
        v = basis @ reduced_solve(h_vec, shift)
        res = K @ v + shift * (c_diag * v) - g_vec
        for h, Hd in zip(h_vec, H_diags):
            res = res + h * (Hd * v)
        return float(np.linalg.norm(res) / g_norm)

    def enrich(h_vec, x0):
        """Full solve at (shift, h_vec); append the response to the basis."""
        nonlocal basis, worst_score, converged, processed_count
        if basis.shape[1] >= order_limit:
            converged = False  # budget exhausted; outer loop will stop
            return x0 if x0 is not None else np.zeros(g_vec.size)
        A = candidate_A(h_vec, shift)
        response = np.asarray(spd_solve(A, g_vec, x0=x0, rtol=ENRICH_RTOL)).reshape(
            -1, 1
        )
        block = orthonormalize_block(basis, response)
        if not block.shape[1]:
            raise RuntimeError("rational Krylov enrichment stalled")
        B_old = basis
        basis = np.column_stack((basis, block))
        extend_projected(block, B_old)
        reference = max(float(response.ravel() @ g_vec), np.finfo(float).tiny)
        _, _, _, score_after = response_error(
            response, basis, reduced_solve(h_vec, shift)[:, None], A, reference
        )
        worst_score = max(worst_score, score_after)
        processed_count += 1
        if score_after > residual_tolerance:
            converged = False
        return response

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

        # per-port reduced-model state: warm-start x0 and the incremental
        # projected matrices are re-seeded for each source port.
        init_port(g)
        x0 = None
        for shift in shifts:
            if not converged:
                break
            passes = 0
            h_samples = 0
            while passes < probe_rounds and converged:
                sub_seed = (
                    seed + 100003 * port + int(round(float(shift) * 1.0e6)) + h_samples
                )
                h_vec = random_h(h_ranges, sub_seed)
                h_samples += 1
                candidate_total += 1
                if probe_residual(h_vec, shift) <= residual_tolerance:
                    passes += 1
                    continue
                passes = 0
                x0 = enrich(h_vec, x0)
            outer_idx += 1
            history.append(
                {
                    "outer_idx": outer_idx,
                    "port": int(port),
                    "shift": float(shift),
                    "h_samples": h_samples,
                }
            )
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
        "parameter_sampling": (
            "random per (port, shift), error-driven probe stop "
            "(FANTASTIC BCI Algorithm 1 / Extended FANTASTIC 2021)"
        ),
        "probe_rounds": int(probe_rounds),
        "candidate_count": candidate_total,
        "processed_candidate_count": processed_count,
        "outer_count": outer_idx,
        "solver": "warm-started Ruge-Stueben AMG-preconditioned CG (spd_solve)",
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
            "solve by warm-started Ruge-Stueben AMG-CG (no re-factorization) with an "
            "incremental reduced model for the probe residual; per-port "
            "eigenvalue/shift planning"
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
