#!/usr/bin/env python3
"""Experimental: BCI-ROM (upper 3 layers) coupled to full FVM (lower FR4).

Domain split of ``case1.ecxml`` at the horizontal interface face z = 10 mm
(the aluminum bottom face = FR4 top face):

    UPPER sub-model  U  : Aluminum + E-10 + 4 silicon dies   (z 10..22 mm)
                          -> BCI-FANTASTIC reduced order model (our ROM)
    LOWER sub-model  L  : FR4                                 (z  0..10 mm)
                          -> FULL FVM (unreduced)

Coupling via static substructuring / Galerkin condensation on the shared
interface face.  The full (monolithic) operator is partitioned into U/L
blocks; the cross blocks M_UL / M_LU (exactly the Al-FR4 face conductances)
are retained as the interface coupling, so with an exact basis the coupled
result coincides with the monolithic reference.  Reducing U with the ROM
basis V gives the coupled reduced system::

    [ Ĉ_U   0   ] [dθ/dt ]   [ K̂_U      Vᵀ M_UL ] [θ  ]   [ F̂_U P ]
    [ 0    C_LL ] [dy_L/dt] + [ M_LU V     M_LL   ] [y_L] = [  0   ]

    K̂_U = Vᵀ M_UU V,   F̂_U = Vᵀ G_U,   Ĉ_U = Vᵀ C_UU V

Boundary scenario identical to reproduce_case1.py: die crowns h=5e1, FR4
bottom h=1e3, sides adiabatic, ambient 308.15 K, sources 0.1..0.4 W.  Max
cell size 2.5 mm.  Steady + transient, compared against the monolithic
full-FVM reference.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT.parent / "macromodel", _ROOT.parent.parent.parent / "python"):
    sys.path.insert(0, str(p))

from model_case1 import Case1Config, Case1Model, DIES  # noqa: E402
from utils import build_parametric_basis, normalized_operators  # noqa: E402

AMB = 308.15
H_CROWN = 5.0e1  # die crowns (Face.ZP)
H_FR4 = 1.0e3  # FR4 bottom (Face.ZM)
SOURCES = np.array([0.1, 0.2, 0.3, 0.4])
DIE_NAMES = ["S0", "S1", "S2", "S3"]
Z_SPLIT = 10.0  # mm  (interface face: Al bottom / FR4 top)
DURATION_S = 1000.0
DT_S = 5.0
OUT = _ROOT / "results" / "experiment_ddcoupling"


def build():
    """Monolithic model -> affine K(h), C, G, per-cell z-centre, crown diag,
    and interface (z=10) exposed cells/areas."""
    mm = Case1Model(Case1Config())
    core = mm.core_operators()
    G = mm.source_shape()
    H_crown, H_fr4 = mm.boundary_terms()
    M_full = core.K.tocsc() + H_CROWN * H_crown.tocsc() + H_FR4 * H_fr4.tocsc()
    M_full = (0.5 * (M_full + M_full.T)).tocsc()
    C_full = core.C.tocsc()
    full = mm._full
    zc = mm._cell_z_centers()

    # per-cell in-plane face area (m^2) to weight the interface basis group.
    grid = full.grid_to_cell.reshape(full.nx, full.ny, full.nz)
    x = mm.config.axis_vertices_mm * 1e-3
    dx = np.diff(x)
    face_areas = np.zeros(full.cell_count)
    for ix in range(full.nx):
        for iy in range(full.ny):
            slab = grid[ix, iy, :]
            v = slab >= 0
            face_areas[slab[v]] = dx[ix] * dx[iy]  # square cells

    interfaces = {
        "areas": face_areas,
        "zc": zc,
    }
    crown_diag = np.asarray(H_crown.diagonal()).ravel()
    return mm, M_full, C_full, G, crown_diag, interfaces


def partition(M_full, C_full, G, crown_diag, if_areas, zc, split_m):
    n = M_full.shape[0]
    U = np.sort(np.flatnonzero(zc >= split_m - 1e-12))
    L = np.sort(np.flatnonzero(zc < split_m - 1e-12))
    nU, nL = U.size, L.size

    # interface pairs = cross-block nonzero off-diagonals
    cross = M_full[U][:, L] != 0
    rU, cL = cross.nonzero()
    g = -np.asarray(M_full[U[rU], L[cL]]).ravel()
    assert np.all(g > 0)

    M_UU = M_full[U][:, U].tocsc()
    M_UL = M_full[U][:, L].tocsc()
    M_LU = M_full[L][:, U].tocsc()
    M_LL = M_full[L][:, L].tocsc()
    C_UU = C_full[U][:, U].tocsc()
    C_LL = C_full[L][:, L].tocsc()
    G_U = np.asarray(G[U, :])

    H_crown_U = sp.diags(crown_diag[U])
    area_if = np.zeros(nU)
    area_if[rU] = if_areas[U[rU]]  # full-order areas -> U-local rows
    H_iface_U = sp.diags(area_if)

    return dict(
        rU=rU,
        cL=cL,
        nU=nU,
        nL=nL,
        M_UU=M_UU,
        M_UL=M_UL,
        M_LU=M_LU,
        M_LL=M_LL,
        C_UU=C_UU,
        C_LL=C_LL,
        G_U=G_U,
        H_crown_U=H_crown_U,
        H_iface_U=H_iface_U,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mm, M_full, C_full, G, crown_diag, iface = build()
    p = partition(
        M_full, C_full, G, crown_diag, iface["areas"], iface["zc"], Z_SPLIT * 1e-3
    )
    print(
        f"split z={Z_SPLIT}mm   U cells={p['nU']}  L cells={p['nL']}  "
        f"interface pairs={p['rU'].size}"
    )

    # ---- U BCI basis on its h-free core, spanned over [crown, interface] ----
    core_U = normalized_operators(p["M_UU"], p["C_UU"], np.zeros(p["nU"]))
    basis, summary = build_parametric_basis(
        core_U,
        p["G_U"],
        [p["H_crown_U"], p["H_iface_U"]],
        np.array([[1.0, 1e4], [1.0, 1e4]]),
        residual_tolerance=1e-3,
        max_order=1024,
        target_relative_epsilon=1e-3,
        probe_rounds=3,
        seed=20260805,
    )
    m = basis.shape[1]
    t_basis = time.perf_counter() - t0
    print(
        f"U BCI-ROM basis order = {m}  ({t_basis:.1f}s, response err "
        f"{summary['relative_response_error']:.2e})"
    )

    # ---- coupled reduced operators ----
    K_hatU = basis.T @ (p["M_UU"] @ basis)
    K_hatU = 0.5 * (K_hatU + K_hatU.T)
    C_hatU = basis.T @ (p["C_UU"] @ basis)
    C_hatU = 0.5 * (C_hatU + C_hatU.T)
    F_hatU = basis.T @ p["G_U"]
    Vt_UL = basis.T @ p["M_UL"]
    M_LU_V = p["M_LU"] @ basis

    Kc = sp.bmat([[K_hatU, Vt_UL], [M_LU_V, p["M_LL"]]]).tocsc()
    Cc = sp.bmat([[C_hatU, None], [None, p["C_LL"]]]).tocsc()
    Kc = 0.5 * (Kc + Kc.T).tocsc()

    def coupled_rhs(P):
        return np.concatenate([F_hatU @ np.asarray(P), np.zeros(p["nL"])])

    # ---- steady ----
    z_ss = np.asarray(sp.linalg.spsolve(Kc.tocsc(), coupled_rhs(SOURCES))).ravel()
    # ---- transient (fixed-step BDF1 on the coupled block system) ----
    nz = m + p["nL"]
    lhs = (Cc / DT_S + Kc).tocsc()
    times = np.arange(0.0, DURATION_S + 0.5 * DT_S, DT_S)
    Z = np.empty((times.size, nz))
    z = np.zeros(nz)
    solver = sp.linalg.splu(lhs.tocsc())
    for i, t in enumerate(times):
        z = solver.solve((Cc @ z) / DT_S + coupled_rhs(SOURCES))
        Z[i] = z

    def junction(theta):
        # junction rise = G_U^T (y_U_rise), y_U ~ V theta
        return AMB + (p["G_U"].T @ (basis @ theta))

    junc_coup_ss = junction(z_ss[:m])
    junc_coup_hist = AMB + (Z[:, :m] @ basis.T) @ p["G_U"]  # (nt,4)

    # reference monolithic
    ref = mm.full_reference((H_CROWN, H_FR4))
    junc_ref_ss = mm.junction_temperature(ref.steady_temperature)
    junc_ref_hist = mm.junction_temperature(ref.history)

    rise_U_ss = float(np.max(basis @ z_ss[:m]))
    rise_L_ss = float(np.max(z_ss[m:]))
    peak_coup = AMB + max(rise_U_ss, rise_L_ss)

    print("\n=== STEADY junction (K):  coupled vs monolithic reference ===")
    print(f"{'':6}{'coupled':>10}{'reference':>10}{'diff':>10}")
    for i, nm in enumerate(DIE_NAMES):
        print(
            f"{nm:<6}{junc_coup_ss[i]:>10.3f}{junc_ref_ss[i]:>10.3f}"
            f"{junc_coup_ss[i]-junc_ref_ss[i]:>+10.4f}"
        )
    print(
        f"coupled peak field = {peak_coup:.3f} K   "
        f"reference peak = {ref.steady_temperature.max():.3f} K  "
        f"(reported ~331 K)"
    )
    print(
        f"steady junction max err = "
        f"{np.max(np.abs(junc_coup_ss-junc_ref_ss)):.5f} K"
    )

    denom = np.abs(junc_ref_ss - AMB)
    err = np.max(np.abs(junc_coup_hist - junc_ref_hist), axis=0)
    print("\n=== TRANSIENT junction max err (K / % of steady rise) ===")
    for i, nm in enumerate(DIE_NAMES):
        print(f"{nm:<6}{err[i]:>10.4f} K   {100*err[i]/denom[i]:>8.3f}%")

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(
            ref.times,
            junc_ref_hist[:, i],
            "-",
            color="tab:blue",
            label="reference (monolithic full FVM)",
            lw=2,
        )
        ax.plot(
            times,
            junc_coup_hist[:, i],
            "--",
            color="tab:red",
            label="coupled (BCI-ROM upper + full FVM lower)",
            lw=2,
        )
        ax.set_title(DIE_NAMES[i])
        ax.set_ylabel("[K]")
        ax.grid(alpha=0.3)
    axes[-1, 1].set_xlabel("time [s]")
    axes[-1, 0].set_xlabel("time [s]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Junction transient: coupled BCI-ROM vs monolithic reference", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "coupled_transient.png", dpi=150)
    plt.close(fig)
    print(f"\nplots -> {OUT}")

    with open(OUT / "summary.txt", "w") as f:
        f.write(
            f"z_split={Z_SPLIT}mm  U basis order={m}  U cells={p['nU']}  "
            f"L cells={p['nL']}  iface pairs={p['rU'].size}\n"
        )
        f.write("steady junction: coupled,reference\n")
        for i, nm in enumerate(DIE_NAMES):
            f.write(f"{nm},{junc_coup_ss[i]:.5f},{junc_ref_ss[i]:.5f}\n")
        f.write(
            f"peak coupled={peak_coup:.5f} ref={ref.steady_temperature.max():.5f}\n"
        )


if __name__ == "__main__":
    main()
