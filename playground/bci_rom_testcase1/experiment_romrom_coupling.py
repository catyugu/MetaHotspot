#!/usr/bin/env python3
"""Experimental: BCI-ROM (upper 3 layers) <-> BCI-ROM (lower FR4), coupled via
explicit interface DOFs (material-decoupled).

Both sub-domains of ``case1.ecxml`` are reduced BCI-FANTASTIC ROMs, connected
only through explicit interface DOFs ``T_Gamma`` (one per shared face pair) on
the junction face z = 10 mm::

    [ K_U + R'gU R   -R'gU      0        ] [tU ]   [ F_U P ]
    [ -gU R           gU+gL     -gL R''  ] [Tg ] = [  0    ]
    [ 0              -R''gL     K_L+R''gL R''][tL ]   [  0    ]

Each ROM is extracted on its OWN (series-removed) operator with the interface
face as an explicit enrichment boundary group, so neither macro depends on the
other's material.  The lower FR4 slab is passive (no heat source), so its basis
is driven by the interface-face profile rather than a source port.

Reference is the monolithic full-FVM solve; boundary scenario identical to
reproduce_case1.py (crowns h=5e1, FR4 bottom h=1e3, ambient 308.15 K,
sources 0.1..0.4 W).  Steady + transient.
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
from iface_coupling import partition, build_rom_basis, assemble_coupled  # noqa: E402

AMB = 308.15
H_CROWN = 5.0e1  # die crowns (Face.ZP)
H_FR4 = 1.0e3  # FR4 bottom (Face.ZM)
SOURCES = np.array([0.1, 0.2, 0.3, 0.4])
DIE_NAMES = ["S0", "S1", "S2", "S3"]
Z_SPLIT = 10.0  # mm  (interface face: Al bottom / FR4 top)
DURATION_S = 1000.0
DT_S = 5.0
OUT = _ROOT / "results" / "experiment_romrom_coupling"
K_U_IFACE = 201.0
K_L_IFACE = 0.3


def build():
    mm = Case1Model(Case1Config())
    core = mm.core_operators()
    G = mm.source_shape()
    H_crown, H_fr4 = mm.boundary_terms()
    M_full = core.K.tocsc() + H_CROWN * H_crown.tocsc() + H_FR4 * H_fr4.tocsc()
    M_full = (0.5 * (M_full + M_full.T)).tocsc()
    C_full = core.C.tocsc()
    full = mm._full
    zc = mm._cell_z_centers()

    grid = full.grid_to_cell.reshape(full.nx, full.ny, full.nz)
    x = mm.config.axis_vertices_mm * 1e-3
    dx = np.diff(x)
    face_areas = np.zeros(full.cell_count)
    for ix in range(full.nx):
        for iy in range(full.ny):
            slab = grid[ix, iy, :]
            v = slab >= 0
            face_areas[slab[v]] = dx[ix] * dx[iy]

    crown_diag = np.asarray(H_crown.diagonal()).ravel()
    bottom_diag = np.asarray(H_fr4.diagonal()).ravel()
    return mm, M_full, C_full, G, crown_diag, bottom_diag, face_areas, zc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    mm, M_full, C_full, G, crown_diag, bottom_diag, face_areas, zc = build()
    p = partition(
        M_full, C_full, G, crown_diag, bottom_diag, face_areas, zc,
        Z_SPLIT * 1e-3, K_U_IFACE, K_L_IFACE,
    )
    print(
        f"split z={Z_SPLIT}mm   U cells={p['nU']}  L cells={p['nL']}  "
        f"interface pairs={p['rU'].size}   halfcond rel err={p['halfcond_rel_err']:.1e}"
    )

    # ---- U basis: driven by its 4 heat-source ports --------------------
    pbasis_U = build_rom_basis(
        p["M_UU"], p["C_UU"], p["G_U"],
        [p["H_crown_U"], p["H_iface_U"]],
        [[1.0, 1e4], [1.0, 1e4]],
    )
    basis_U, sumU = pbasis_U

    # ---- L basis: passive slab driven by the interface profile ---------
    # source shape = a unit load spread over the L interface cells (both a
    # single column and one column per interface cell group are spanned via
    # the interface boundary group).  A single dense col is enough here.
    G_L_drive = p["iface_L_indicator"][:, None]
    pbasis_L = build_rom_basis(
        p["M_LL"], p["C_LL"], G_L_drive,
        [p["H_bottom_L"], p["H_iface_L"]],
        [[1.0, 1e4], [1.0, 1e4]],
    )
    basis_L, sumL = pbasis_L
    print(
        f"U basis order = {basis_U.shape[1]}  ({sumU['seconds']:.1f}s, err "
        f"{sumU['relative_response_error']:.1e})   "
        f"L basis order = {basis_L.shape[1]}  ({sumL['seconds']:.1f}s, err "
        f"{sumL['relative_response_error']:.1e})"
    )

    # ---- coupled operators (both reduced, interface DOFs) --------------
    c = assemble_coupled(pbasis_U, W_L=pbasis_L, p=p)
    print(
        f"coupled DOFs = {c['size']}  (U {c['m']} + iface {c['npairs']} + L {c['nL']})"
    )

    def coupled_rhs(P):
        return c["rhs"](P)

    # steady
    z_ss = np.asarray(sp.linalg.spsolve(c["K"].tocsc(), coupled_rhs(SOURCES))).ravel()
    # transient BDF1
    nz = c["size"]
    lhs = (c["C"] / DT_S + c["K"]).tocsc()
    times = np.arange(0.0, DURATION_S + 0.5 * DT_S, DT_S)
    Z = np.empty((times.size, nz))
    z = np.zeros(nz)
    solver = sp.linalg.splu(lhs.tocsc())
    for i, t in enumerate(times):
        z = solver.solve((c["C"] @ z) / DT_S + coupled_rhs(SOURCES))
        Z[i] = z

    def junction(theta):
        return AMB + (p["G_U"].T @ (basis_U @ theta))

    junc_coup_ss = junction(z_ss[:c["m"]])
    junc_coup_hist = AMB + (Z[:, :c["m"]] @ basis_U.T) @ p["G_U"]

    ref = mm.full_reference((H_CROWN, H_FR4))
    junc_ref_ss = mm.junction_temperature(ref.steady_temperature)
    junc_ref_hist = mm.junction_temperature(ref.history)

    print("\n=== STEADY junction (K):  ROM-ROM (iface-DOF) vs monolithic ===")
    print(f"{'':6}{'coupled':>10}{'reference':>10}{'diff':>10}")
    for i, nm in enumerate(DIE_NAMES):
        print(
            f"{nm:<6}{junc_coup_ss[i]:>10.3f}{junc_ref_ss[i]:>10.3f}"
            f"{junc_coup_ss[i]-junc_ref_ss[i]:>+10.4f}"
        )
    print(f"steady junction max err = {np.max(np.abs(junc_coup_ss-junc_ref_ss)):.5f} K")

    denom = np.abs(junc_ref_ss - AMB)
    err = np.max(np.abs(junc_coup_hist - junc_ref_hist), axis=0)
    print("\n=== TRANSIENT junction max err (K / % of steady rise) ===")
    for i, nm in enumerate(DIE_NAMES):
        print(f"{nm:<6}{err[i]:>10.4f} K   {100*err[i]/denom[i]:>8.3f}%")

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(ref.times, junc_ref_hist[:, i], "-", color="tab:blue",
                label="reference (monolithic full FVM)", lw=2)
        ax.plot(times, junc_coup_hist[:, i], "--", color="tab:red",
                label="ROM upper <-> ROM lower (iface DOFs)", lw=2)
        ax.set_title(DIE_NAMES[i])
        ax.set_ylabel("[K]")
        ax.grid(alpha=0.3)
    axes[-1, 1].set_xlabel("time [s]")
    axes[-1, 0].set_xlabel("time [s]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Junction transient: ROM<->ROM interface-DOF vs monolithic", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "coupled_transient.png", dpi=150)
    plt.close(fig)
    print(f"\nplots -> {OUT}")

    with open(OUT / "summary.txt", "w") as f:
        f.write(
            f"z_split={Z_SPLIT}mm iface-DOF  U order={c['m']} L order={c['nL']} "
            f"U cells={p['nU']} L cells={p['nL']} pairs={c['npairs']} "
            f"halfcond rel err={p['halfcond_rel_err']:.1e}\n"
        )
        f.write("steady junction: coupled,reference\n")
        for i, nm in enumerate(DIE_NAMES):
            f.write(f"{nm},{junc_coup_ss[i]:.5f},{junc_ref_ss[i]:.5f}\n")
        f.write(f"steady junction max err = {np.max(np.abs(junc_coup_ss-junc_ref_ss)):.5f} K\n")


if __name__ == "__main__":
    main()
