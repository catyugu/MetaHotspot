#!/usr/bin/env python3
"""Interface-DOF (material-decoupled) coupling for BCI-ROM <-> neighbour DOMAIN.

ROM-detailed and ROM-ROM junctions are connected through explicit interface DOFs
``T_Gamma`` (one per shared face pair) allocated ON the junction face.  Each
sub-model couples through its OWN material half-conductance
``g_S = k_S * A / d_S`` (``d_S`` = cell-centre -> face distance), so the macro is
decoupled from the external (neighbour) material properties and can be
interconnected to an arbitrary neighbour without re-extraction::

    [ K_U + R'gU R    -R'gU      0        ] [tU ]   [ F_U P ]
    [ -gU R            gU+gL     -gL R''  ] [Tg ] = [  0    ]
    [ 0               -R''gL     K_L+R''gL R''][tL ]   [  0    ]

with ``R = V[iface_U, :]``, ``R'' = W[iface_L, :]`` (identity for a full-FVM
neighbour), ``K_U = V'(U_block)V``, and ``U_block`` the sub-model's OWN operator
(conduction + own ambient) minus the monolithic *series* face conductance on the
interface-cell diagonal.  Eliminating ``T_Gamma`` recovers the monolithic series
conductance ``g = gU*gL/(gU+gL)`` exactly (verified to machine precision), i.e.
with full (identity) bases this reproduces the monolithic solve.

Only ``utils.build_parametric_basis`` / ``normalized_operators`` from the
macromodel layer are used; everything here is shared by the ROM-detailed
(``experiment_ddcoupling.py``) and ROM-ROM (``experiment_romrom_coupling.py``)
experiments.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from utils import build_parametric_basis, normalized_operators


def partition(
    M_full,
    C_full,
    G,
    crown_diag,
    bottom_diag,
    iface_areas,
    zc,
    split_m,
    kU,
    kL,
):
    """Split monolithic operators at an interior face into U/L sub-models.

    Returns both sub-blocks plus the material-decoupled per-pair interface data:
    series conductance ``g_series`` (the monolithic cross-block face
    conductance = ``gU*gL/(gU+gL)``), the own-material half-conductances
    ``gU``/``gL``, the U/L interface cell locations and the own operator (with
    the series conductance removed from the interface diagonal).

    ``crown_diag`` / ``bottom_diag`` are the full-domain diagonals of the
    U-side (top/crown) and L-side (bottom) ambient boundary groups.
    """
    U = np.sort(np.flatnonzero(zc >= split_m - 1e-12))
    L = np.sort(np.flatnonzero(zc < split_m - 1e-12))
    nU, nL = U.size, L.size

    cross = M_full[U][:, L] != 0
    rU, cL = cross.nonzero()
    g_series = -np.asarray(M_full[U[rU], L[cL]]).ravel()
    assert np.all(g_series > 0)

    # own-material half conductances (physical: g_S = k_S * A / d_S)
    A = np.asarray(iface_areas[U[rU]]).ravel()
    dU = zc[U[rU]] - split_m
    dL = split_m - zc[L[cL]]
    gU = kU * A / dU
    gL = kL * A / dL
    g_from_split = gU * gL / (gU + gL)
    rel = float(np.max(np.abs(g_from_split - g_series) / np.maximum(g_series, 1e-300)))

    def own(M_SS, C_SS, G_S, pair_rows):
        """Sub-model's own operator with the series conductance removed from
        the interface cells' diagonal (interface insulated, neighbour not baked
        in) -- this is what the macro basis is built/projected on."""
        M = M_SS.tocsc().copy()
        d = np.zeros(M.shape[0])
        np.add.at(d, pair_rows, g_series)
        M = M - sp.diags(d).tocsc()
        return M, C_SS, np.asarray(G_S, dtype=np.float64)

    M_UU, C_UU, G_U = own(M_full[U][:, U], C_full[U][:, U], G[U, :], rU)
    M_LL, C_LL, G_L = own(M_full[L][:, L], C_full[L][:, L], G[L, :], cL)

    H_crown_U = sp.diags(np.asarray(crown_diag[U], dtype=np.float64))
    area_if = np.zeros(nU)
    np.add.at(area_if, rU, A)
    H_iface_U = sp.diags(area_if)

    # L-side own boundary skeleton: own ambient (fr4 bottom at z=0) + interface
    iface_L = np.zeros(nL)
    np.add.at(iface_L, cL, A)
    H_iface_L = sp.diags(iface_L)
    H_bottom_L = sp.diags(np.asarray(bottom_diag[L], dtype=np.float64))

    return dict(
        U=U,
        L=L,
        nU=nU,
        nL=nL,
        rU=rU,
        cL=cL,
        g_series=g_series,
        gU=gU,
        gL=gL,
        halfcond_rel_err=rel,
        M_UU=M_UU,
        M_LL=M_LL,
        C_UU=C_UU,
        C_LL=C_LL,
        G_U=G_U,
        G_L=G_L,
        H_crown_U=H_crown_U,
        H_iface_U=H_iface_U,
        H_bottom_L=H_bottom_L,
        H_iface_L=H_iface_L,
        iface_L_indicator=np.asarray(iface_L > 0, dtype=np.float64),
    )


def build_rom_basis(
    M_block,
    C_block,
    G_block,
    boundary_terms,
    h_ranges,
    *,
    tolerance=1e-3,
    max_order=1024,
    seed=20260805,
):
    """BCI-FANTASTIC basis of a sub-model on its OWN operator.

    ``M_block`` is the own (series-removed) operator; the boundary skeleton is
    ``[own ambient groups..., interface-face group]`` so the basis also spans
    interface-driven responses.  Returns ``(basis, summary)``.
    """
    core = normalized_operators(
        M_block, C_block, np.zeros(M_block.shape[0])
    )
    return build_parametric_basis(
        core,
        G_block,
        boundary_terms,
        np.asarray(h_ranges, dtype=np.float64),
        tolerance=tolerance,
        max_order=max_order,
        seed=seed,
    )


def assemble_coupled(
    pbasis_U,
    W_L,  # None -> full FVM neighbour (identity)
    p,
    full_L=True,
):
    """Assemble the interface-DOF coupled K, C, rhs-shape for U reduced + L.

    Block order: ``[ theta_U (m) | T_Gamma (npairs) | theta_L (mL) ]``.
    ``pbasis_U`` is ``(basis_U, summary_U)``; ``W_L`` the lower basis (``None``
    for a full-FVM neighbour); ``p`` the :func:`partition` dict.
    """
    V, _sumU = pbasis_U
    m = V.shape[1]
    npairs = p["rU"].size
    gU = p["gU"]
    gL = p["gL"]

    # U side
    K_U = V.T @ (p["M_UU"] @ V)
    K_U = 0.5 * (K_U + K_U.T)
    C_U = V.T @ (p["C_UU"] @ V)
    C_U = 0.5 * (C_U + C_U.T)
    F_U = V.T @ p["G_U"]
    R = V[p["rU"], :]  # (npairs, m) rows of U interface cells

    gU_diag = sp.diags(gU)
    gL_diag = sp.diags(gL)

    # L side
    if W_L is not None:
        W, _sumL = W_L
        mL = W.shape[1]
        K_L = W.T @ (p["M_LL"] @ W)
        K_L = 0.5 * (K_L + K_L.T)
        C_L = W.T @ (p["C_LL"] @ W)
        C_L = 0.5 * (C_L + C_L.T)
        F_L = W.T @ p["G_L"]
        Rp = W[p["cL"], :]  # (npairs, mL)
        Cb = sp.bmat([[C_U, None, None], [None, None, None], [None, None, C_L]])
    else:
        # full-FVM neighbour: W = identity, L interface rows = cL directly
        mL = p["nL"]
        K_L = p["M_LL"]
        C_L = p["C_LL"]
        F_L = p["G_L"]
        Rp = sp.csc_matrix((np.ones(npairs), (np.arange(npairs), p["cL"])), shape=(npairs, mL))
        Cb = sp.bmat([[C_U, None, None], [None, None, None], [None, None, C_L]])
    Cb = sp.bmat(
        [
            [C_U, None, None],
            [None, sp.csc_matrix((npairs, npairs)), None],
            [None, None, C_L],
        ]
    ).tocsc()

    A00 = K_U + (R.T @ gU_diag @ R)
    A01 = -(R.T @ gU_diag)
    A10 = -(gU_diag @ R)
    A11 = sp.diags(gU + gL)
    A12 = -(gL_diag @ Rp)
    A22 = K_L + (Rp.T @ gL_diag @ Rp)

    Kc = sp.bmat(
        [
            [A00, A01, None],
            [A10, A11, A12],
            [None, A12.T, A22],
        ]
    ).tocsc()
    Kc = 0.5 * (Kc + Kc.T).tocsc()
    Cc = Cb.tocsc()

    # rhs shape: [F_U P | 0 | F_L P_with_gamma?] -- L has no own source in case1,
    # but keep the term for generality (both sub-models may carry sources).
    def rhs(P):
        return np.concatenate([F_U @ np.asarray(P), np.zeros(npairs), F_L @ np.asarray(P)])

    size = m + npairs + (mL if W_L is not None else p["nL"])
    return dict(
        K=Kc,
        C=Cc,
        rhs=rhs,
        size=size,
        m=m,
        npairs=npairs,
        nL=mL if W_L is not None else p["nL"],
        R=R,
        Rp=Rp,
    )

