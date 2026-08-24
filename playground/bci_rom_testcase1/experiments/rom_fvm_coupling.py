r"""Case-1 ROM-FVM coupling: interface-DOF (material-decoupled) method.

Methodology follows the deleted ``iface_coupling.py`` / ``experiment_ddcoupling.py``
(commit  8df8e7a; the code was removed as obsolete but the approach is the
reference).  The upper three layers are reduced to a BCI-ROM, the passive FR4
substrate is kept as FULL FVM, and the two are joined through explicit interface
temperature DOFs, one per shared face pair:

    UPPER sub-model  U : Aluminum + E-10 + 4 silicon dies  (z 10..22 mm)  -> ROM
    LOWER sub-model  L : FR4                                (z  0..10 mm)  -> Full FVM

    [ K_U + R^T gU R    -R^T gU      0        ] [th_U]   [ F_U P ]
    [ -gU R              gU + gL     -gL Rp    ] [Th_G] = [  0   ]
    [ 0                 -Rp^T gL     K_L + ... ] [ y_L ]   [ F_L P ]

The coupling is material-decoupled: each side supplies only its OWN half-cell
conductance ``g_S = k_S A / d_S`` (``k_S`` = per-cell face-normal conductivity,
``d_S`` = cell-centre to face distance).  Unlike the older scalar-``k`` case
the conductivities are read per interface cell here, so the interface, which
spans both the Al/FR4 solid-solid face *and* the air/FR4 solid-air face, is
handled generically.  The U ROM basis is built on its own series-removed
operator with boundary skeleton ``[die-crown ambient group, interface-face
group]`` so it spans interface-driven responses; the passive FR4 side is never
reduced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

PROJECT = Path(__file__).resolve().parents[3]  # repo root
CASE = PROJECT / "playground" / "bci_rom_testcase1"
MACRO = PROJECT / "playground" / "macromodel"
sys.path[:0] = [str(CASE), str(MACRO), str(PROJECT / "python")]
from model_case1 import Case1Config, Case1Model  # noqa: E402
from utils import build_parametric_basis, normalized_operators  # noqa: E402

OUT = PROJECT / "results" / "weekly_0825"
AMB = 308.15
H_CROWN = 5.0e1  # die crowns (Face.ZP)
H_FR4 = 1.0e3  # FR4 bottom (Face.ZM)
POWER = np.array([0.1, 0.2, 0.3, 0.4])
Z_SPLIT = 10.0e-3  # interface face z = 10 mm
ROM_TOL = 1.0e-3
MAX_ORDER = 1024
SEED = 20260825
DURATION_S = 1000.0
DT_S = 5.0


def physical_system():
    """Monolithic affine K(h), C, die-source G, per-cell geometry."""
    model = Case1Model(
        Case1Config(
            max_xy_cell_mm=2.5,
            max_z_cell_mm=2.5,
            duration_s=DURATION_S,
            dt_s=DT_S,
        )
    )
    core = model.core_operators()
    K0 = core.K.tocsc()
    C = core.C.tocsc()
    peff = model.physical_to_effective((H_CROWN, H_FR4))
    H_crown, H_fr4 = model.boundary_terms()
    M_full = K0 + float(peff[0]) * H_crown.tocsc() + float(peff[1]) * H_fr4.tocsc()
    M_full = (0.5 * (M_full + M_full.T)).tocsc()
    G = model.source_shape()
    return model, M_full, C, G


def recover_cell_kz(K_full, centers, half):
    """Per-cell face-normal conductivity recovered from the assembled matrix.

    ``cell_layout.conductivity`` keys off sparse ``layer_ids`` and assigns the
    layer conductivity to air-padded cells too (e.g. the FR4-overhang air above
    z=10), so it is WRONG for air cells.  The assembled stiffness ``K_full`` is
    the single correct ground truth: for each face of a cell the matrix stores
    ``c = A/(d_i/k_i + d_j/k_j)``.  For a cell, take the median ``k`` back out of
    its (same-material) neighbour faces via ``k = c*(d_i+d_n)/A``.  Fully
    general: no reliance on the layer table, works for air and solid alike.
    """
    K = K_full.tocsr()
    n = K.shape[0]
    kz = np.empty(n)
    for c in range(n):
        vals = []
        row = K.indices[K.indptr[c] : K.indptr[c + 1]]
        data = K.data[K.indptr[c] : K.indptr[c + 1]]
        for ni, ci in zip(row, data):
            if ni == c or ci >= 0:
                continue
            g = -ci
            dc = centers[ni] - centers[c]
            axis = int(np.argmax(np.abs(dc)))
            others = [a for a in range(3) if a != axis]
            A = 4.0 * half[c, others[0]] * half[c, others[1]]
            # cell's own half + neighbour half along the face axis
            if g > 0 and A > 0:
                vals.append(g * (half[c, axis] + half[ni, axis]) / A)
        kz[c] = float(np.median(vals)) if vals else 0.0
    return kz


def partition(model, M_full, C, G):
    """Split at z=Z_SPLIT into U (ROM) and L (full-FVM) with per-pair data.

    Returns own (series-removed) sub-operators plus the material-decoupled
    interface half-conductances ``gU``/``gL`` (computed from per-cell k so the
    interface can mix air and solid cells).
    """
    lay = model.cell_layout
    zc = lay.centers[:, 2]
    kz = recover_cell_kz(M_full, lay.centers, lay.half_sizes)  # matrix-truth k
    half = lay.half_sizes[:, 2]  # cell-centre -> face distance along z
    area = (2.0 * lay.half_sizes[:, 0]) * (2.0 * lay.half_sizes[:, 1])

    U = np.sort(np.flatnonzero(zc >= Z_SPLIT - 1e-12))
    L = np.sort(np.flatnonzero(zc < Z_SPLIT - 1e-12))
    nU, nL = U.size, L.size

    cross = M_full[U, :][:, L].tocsc()
    rU, cL = cross.nonzero()
    g_series = -np.asarray(cross[rU, cL]).ravel()
    assert np.all(g_series > 0)

    # own material half conductances per interface cell (general k, incl. air)
    A = area[U[rU]]
    gU = kz[U[rU]] * A / half[U[rU]]
    gL = kz[L[cL]] * A / half[L[cL]]
    g_from_split = gU * gL / (gU + gL)
    halfcond_rel_err = float(
        np.max(np.abs(g_from_split - g_series) / np.maximum(g_series, 1e-300))
    )

    def own(M_SS, C_SS, G_S, pair_rows):
        M = M_SS.tocsc().copy()
        d = np.zeros(M.shape[0])
        np.add.at(d, pair_rows, g_series)
        M = M - sp.diags(d).tocsc()
        return M, C_SS, np.asarray(G_S, dtype=np.float64)

    M_UU, C_UU, G_U = own(M_full[U, :][:, U], C[U, :][:, U], G[U, :], rU)
    M_LL, C_LL, G_L = own(M_full[L, :][:, L], C[L, :][:, L], G[L, :], cL)

    # U boundary skeleton: own ambient (die crowns) + interface-face group.
    crown_diag = np.asarray(model.boundary_terms()[0].diagonal()).ravel()
    H_crown_U = sp.diags(crown_diag[U])
    area_if_U = np.zeros(nU)
    np.add.at(area_if_U, rU, A)
    H_iface_U = sp.diags(area_if_U)
    # L boundary skeleton (kept for generality; FR4 has only bottom ambient by
    # the boundary_terms already folded into M_full, so no extra group needed).
    iface_L = np.zeros(nL)
    np.add.at(iface_L, cL, A)
    H_iface_L = sp.diags(iface_L)

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
        halfcond_rel_err=halfcond_rel_err,
        M_UU=M_UU,
        M_LL=M_LL,
        C_UU=C_UU,
        C_LL=C_LL,
        G_U=G_U,
        G_L=G_L,
        H_crown_U=H_crown_U,
        H_iface_U=H_iface_U,
        H_iface_L=H_iface_L,
    )


def build_u_rom(p):
    """BCI-FANTASTIC basis of the UPPER sub-model on its own operator."""
    core = normalized_operators(p["M_UU"], p["C_UU"], np.zeros(p["nU"]))
    basis, summary = build_parametric_basis(
        core,
        p["G_U"],
        [p["H_crown_U"], p["H_iface_U"]],
        np.asarray([[1.0, 1e4], [1.0, 1e4]], dtype=np.float64),
        tolerance=ROM_TOL,
        max_order=MAX_ORDER,
        probe_rounds=2,
        seed=SEED,
    )
    return basis, summary


def assemble_coupled(p, V):
    """Interface-DOF coupled matrix for U reduced + L full-FVM (identity)."""
    m = V.shape[1]
    npairs = p["rU"].size
    nL = p["nL"]
    gU, gL = p["gU"], p["gL"]

    K_U = V.T @ (p["M_UU"] @ V)
    K_U = 0.5 * (K_U + K_U.T)
    C_U = V.T @ (p["C_UU"] @ V)
    C_U = 0.5 * (C_U + C_U.T)
    F_U = V.T @ p["G_U"]
    F_L = p["G_L"]
    R = V[p["rU"], :]  # (npairs, m) trace
    Rp = sp.csc_matrix(
        (np.ones(npairs), (np.arange(npairs), p["cL"])), shape=(npairs, nL)
    )
    gU_d, gL_d = sp.diags(gU), sp.diags(gL)

    A00 = K_U + (R.T @ gU_d @ R)
    A01 = -(R.T @ gU_d)
    A10 = -(gU_d @ R)
    A11 = sp.diags(gU + gL)
    A12 = -(gL_d @ Rp)
    A22 = p["M_LL"] + (Rp.T @ gL_d @ Rp)

    Kc = sp.bmat([[A00, A01, None], [A10, A11, A12], [None, A12.T, A22]]).tocsc()
    Kc = 0.5 * (Kc + Kc.T).tocsc()
    Cc = sp.block_diag((C_U, sp.csc_matrix((npairs, npairs)), p["C_LL"])).tocsc()

    def rhs(P):
        P = np.asarray(P, dtype=np.float64)
        return np.concatenate([F_U @ P, np.zeros(npairs), F_L @ P])

    return dict(K=Kc, C=Cc, rhs=rhs, m=m, npairs=npairs, nL=nL, size=m + npairs + nL)


def run():
    model, M_full, C, G = physical_system()
    p = partition(model, M_full, C, G)
    nl, nu, ng = p["nL"], p["nU"], p["rU"].size

    full = sla.spsolve(M_full, G @ POWER)  # monolithic steady reference

    print(
        f"split z={Z_SPLIT*1e3:.0f} mm  U cells={nu} (ROM)  "
        f"L cells={nl} (full FVM)  interface pairs={ng}  "
        f"halfcond rel err={p['halfcond_rel_err']:.1e}"
    )

    V, summary = build_u_rom(p)
    m = V.shape[1]
    print(
        f"U BCI-ROM basis order = {m}  (response err "
        f"{summary['relative_response_error']:.2e}, "
        f"extraction {summary['seconds']:.2f} s)"
    )
    print(
        f"  per-port shifts: {[pl['shift_count'] for pl in summary['per_port_plans']]}"
    )

    c = assemble_coupled(p, V)
    print(f"coupled DOFs = {c['size']}  (U {m} + iface {ng} + L {nl})")

    # ---- transient: fixed-step BDF1 on the coupled block system ---------
    nz = c["size"]
    rhs_all = c["rhs"](POWER)
    lhs = (c["C"] / DT_S + c["K"]).tocsc()
    times = np.arange(0.0, DURATION_S + 0.5 * DT_S, DT_S)
    Z = np.empty((times.size, nz))
    Z[0] = np.zeros(nz)  # initial state = ambient (rise 0)
    z = np.zeros(nz)
    t_slv = sp.linalg.splu(lhs.tocsc())
    t0 = __import__("time").perf_counter()
    for i in range(1, times.size):
        z = t_slv.solve((c["C"] @ z) / DT_S + rhs_all)
        Z[i] = z
    coupled_transient_s = __import__("time").perf_counter() - t0

    # steady
    ss = np.asarray(sla.spsolve(c["K"].tocsc(), c["rhs"](POWER))).ravel()
    junc_coup = AMB + (p["G_U"].T @ (V @ ss[:m]))
    junc_ref = AMB + (G.T @ full)
    peak_coup = AMB + float(max(np.max(V @ ss[:m]), np.max(ss[m + ng :])))
    peak_ref = AMB + float(np.max(full))
    print("\n=== STEADY junction (K):  coupled (iface-DOF) vs monolithic reference ===")
    print(f"{'die':<5}{'coupled':>10}{'reference':>10}{'diff K':>10}{'rel%':>8}")
    for i, nm in enumerate(["S0", "S1", "S2", "S3"]):
        d = junc_coup[i] - junc_ref[i]
        print(
            f"{nm:<5}{junc_coup[i]:>10.3f}{junc_ref[i]:>10.3f}{d:>+10.4f}"
            f"{100*d/(junc_ref[i]-AMB):>8.3f}"
        )
    print(f"peak field = {peak_coup:.3f} K  reference = {peak_ref:.3f} K")
    print(f"steady junction max err = {np.max(np.abs(junc_coup-junc_ref)):.5f} K")

    # full-field relative error on the recovered upper + retained full lower
    rec = np.r_[V @ ss[:m], ss[m + ng :]]
    full_part = np.r_[full[p["U"]], full[p["L"]]]
    field_rel = float(np.linalg.norm(rec - full_part) / np.linalg.norm(full))
    print(f"full-field relative error (recovered U + full L) = {field_rel:.3e}")

    # transient junction comparison vs monolithic reference
    junc_coup_hist = AMB + (Z[:, :m] @ V.T) @ p["G_U"]  # (nt,4)
    ref_tr = model.full_reference((H_CROWN, H_FR4))
    junc_ref_hist = model.junction_temperature(ref_tr.history)
    rise = np.maximum(np.abs(junc_ref_hist[-1] - AMB), 1e-12)
    tr_err_K = float(np.max(np.abs(junc_coup_hist - junc_ref_hist)))
    tr_err_pct = 100.0 * tr_err_K / float(np.max(rise))
    final_err_K = float(np.max(np.abs(junc_coup_hist[-1] - junc_ref_hist[-1])))
    print("\n=== TRANSIENT junction (K): coupled vs monolithic ===")
    for i, nm in enumerate(["S0", "S1", "S2", "S3"]):
        d = np.max(np.abs(junc_coup_hist[:, i] - junc_ref_hist[:, i]))
        print(
            f"{nm:<5} max err {d:>10.4f} K   " f"{100*d/rise[i]:>8.3f}% of steady rise"
        )
    print(f"overall max transient junction err = {tr_err_K:.5f} K ({tr_err_pct:.4f}%)")
    print(f"final-time junction err = {final_err_K:.5f} K")
    print(f"coupled transient solve ({times.size} steps) = {coupled_transient_s:.2f} s")

    result = dict(
        split_mm=Z_SPLIT * 1e3,
        upper_cells=nu,
        lower_cells=nl,
        interface_pairs=ng,
        upper_rom_order=m,
        halfcond_rel_err=float(p["halfcond_rel_err"]),
        extraction_seconds=float(summary["seconds"]),
        full_residual=float(np.linalg.norm(M_full @ full - G @ POWER)),
        steady_junction_coupled_K=junc_coup.tolist(),
        steady_junction_reference_K=junc_ref.tolist(),
        steady_junction_max_err_K=float(np.max(np.abs(junc_coup - junc_ref))),
        peak_coupled_K=float(peak_coup),
        peak_reference_K=float(peak_ref),
        full_field_relative_error=field_rel,
        transient_duration_s=DURATION_S,
        transient_dt_s=DT_S,
        transient_steps=int(times.size),
        transient_max_junction_err_K=tr_err_K,
        transient_max_junction_err_pct=tr_err_pct,
        transient_final_junction_err_K=final_err_K,
        coupled_transient_solve_s=coupled_transient_s,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rom_fvm_coupling.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(
        "\n"
        + json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("steady_junction_coupled_K", "steady_junction_reference_K")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
