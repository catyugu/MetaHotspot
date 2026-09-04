#!/usr/bin/env python3
"""BCI-ROM Case 1 reproduction — 3-way comparison of reduced-order results.

Reproduces ``playground/bci_rom_testcase1/case1.ecxml`` (a 3-layer stack +
4 silicon dies) and compares, for both STEADY and TRANSIENT, the reduced-order
results from three sources:

    1. FULL FVM   : the full finite-volume model (operators assembled by the
                    MetaHotspot C++ engine), solved directly at the effective
                    (series-condensed) coefficient ``p = k·h/(k + h·half)``.
    2. FLOTHERM ROM: the BCI-ROM exported by Simcenter FloTHERM
                    (``MATRICES/*.mtx``, ROM size 36) — reduced solve +
                    junction (COG probe) recovery.
    3. OUR ROM    : the BCI-FANTASTIC reduced-order model built here (per-port
                    spectral bounds -> elliptic shift count -> residual-driven
                    Krylov enrichment -> SVD truncation -> Galerkin
                    projection), reduced solve + full-field recovery.

Boundary scenario (FloTHERM condition-independent pairing, see model_case1):::

    Face.ZP (die crowns, area 4e-4 m2) : h = 5e1   (FloTHERM "Ambient:0")
    Face.ZM (FR4  bottom, area 6e-3 m2) : h = 1e3   (FloTHERM "Ambient:1")
    side faces                         : no BC (adiabatic)
    ambient / initial temperature      : 35 C = 308.15 K

The physical HTC vector is the public boundary parameter: the native
reference (:meth:`full_reference`) takes it verbatim and maps it internally
to the surface-consistent effective coefficient
``p_c = k_c·h / (k_c + h·half_c)``, so the full FVM reproduces FloTHERM's
surface-consistent junction temperatures with no caller-side mapping.  The
ROM training consumes the *effective* coefficient
``model.physical_to_effective(h)`` (the same space
:meth:`~AffineParametricModel.h_ranges` returns for the basis).

Sources S0..S3 = 0.1 / 0.2 / 0.3 / 0.4 W (constant).
Outputs: PNG comparisons (steady bars + transient time series) and a printed
numeric comparison table.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from model_case1 import Case1Config, Case1Model, DIES  # noqa: E402
from metahotspot.macromodel.utils import (  # noqa: E402
    build_parametric_basis,
    project_bci,
    assemble_reduced_k,
    solve_rom_steady,
    solve_rom_transient,
    accuracy_summary,
)

# ---------------------------------------------------------------- scenario ----

AMB = 308.15  # 35 C ambient / initial
H_CROWN = 5.0e1  # die crowns   (Face.ZP, area 4e-4) -> "Ambient:0"
H_FR4 = 1.0e3  # FR4 bottom   (Face.ZM, area 6e-3) -> "Ambient:1"
H_VEC_MODEL = (H_CROWN, H_FR4)  # physical HTC vector [ZP crowns, ZM FR4]
SOURCES = np.array([0.1, 0.2, 0.3, 0.4])
DIE_NAMES = ["S0", "S1", "S2", "S3"]

DURATION_S = 2000.0
DT_S = 50.0

# FANTASTIC-BCI extraction options
PROBE_ROUNDS = 3
ROM_TOLERANCE = 1.0e-3
MAX_ORDER = 1024
SEED = 20260805

OUT = _ROOT / "results" / "reproduce_case1"


# --------------------------------------------------------- FloTHERM ROM ------


def load_flotherm():
    """Load FloTHERM BCI-ROM matrices (n=36 modal)."""
    M_dir = _ROOT / "MATRICES"

    def load(name):
        return sio.mmread(M_dir / name)

    K = load("K_bci_hat.mtx").tocsc()
    Mm = load("M_bci_hat.mtx").tocsc()
    g = np.asarray(load("g_bci_hat.mtx").todense())  # (36, 4) source input
    dH0 = load("delta_H_bci_hat[0].mtx").tocsc()  # Areas[0]=6e-3 -> h=1e3
    dH1 = load("delta_H_bci_hat[1].mtx").tocsc()  # Areas[1]=4e-4 -> h=5e1
    return K, Mm, g, dH0, dH1


def flotherm_steady(K_eff, g):
    x = np.asarray(sp.linalg.spsolve(K_eff, g @ SOURCES)).ravel()
    return AMB + g.T @ x  # junction (COG-probe) recovery


def flotherm_transient(Mm, K_eff, g, dt, duration):
    times, theta = solve_rom_transient(Mm, K_eff, g, lambda t: SOURCES, dt, duration)
    return times, AMB + theta @ g


# ----------------------------------------------------------- run everything ---


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    model = Case1Model(
        Case1Config(
            max_xy_cell_mm=1.0, max_z_cell_mm=1.0, dt_s=DT_S, duration_s=DURATION_S
        )
    )
    print(
        f"model: {model.name}  full cells={model.full_cell_count}  "
        f"ports={len(model.source_ports())}  groups={len(model.boundary_groups())}"
    )

    core = model.core
    G = model.source_shape
    terms = model.boundary_terms
    h_ranges = model.h_ranges()

    # ---- full FVM (affine full-domain solve; model maps physical h internally) --
    t0 = time.perf_counter()
    full = model.full_reference(H_VEC_MODEL)
    t_full_ref = time.perf_counter() - t0
    print("Finished full reference solving")
    Tf_ss = full.steady_temperature
    junc_full_ss = model.junction_temperature(Tf_ss)
    junc_full_hist = model.junction_temperature(full.history)  # (nt, 4)

    # ---- our BCI-FANTASTIC ROM -----------------------------------------
    print("building our BCI-FANTASTIC basis ...")
    t0 = time.perf_counter()
    basis, summary = build_parametric_basis(
        core,
        G,
        terms,
        h_ranges,
        tolerance=ROM_TOLERANCE,
        max_order=MAX_ORDER,
        probe_rounds=PROBE_ROUNDS,
        seed=SEED,
    )
    t_basis = time.perf_counter() - t0
    print(
        f"  basis order = {basis.shape[1]}  ({t_basis:.1f}s)  "
        f"response err={summary['relative_response_error']:.2e}"
    )

    C_hat, K0, F_hat, F_bdry, A_bdry = project_bci(core, G, terms, basis)
    p_vec = model.physical_to_effective(H_VEC_MODEL)
    K_hat = assemble_reduced_k(K0, F_bdry, A_bdry, p_vec)

    theta_ss = solve_rom_steady(K_hat, F_hat, SOURCES)
    junc_rom_ss = AMB + F_hat.T @ theta_ss
    rec_ss = AMB + basis @ theta_ss  # recovered full field

    r_times, theta_hist = solve_rom_transient(
        C_hat, K_hat, F_hat, lambda t: SOURCES, DT_S, DURATION_S
    )
    junc_rom_hist = AMB + theta_hist @ F_hat
    rec_hist = AMB + theta_hist @ basis.T

    # ---- FloTHERM ROM ---------------------------------------------------
    Kf, Mf, g, dH0, dH1 = load_flotherm()
    K_eff_fl = Kf + H_FR4 * dH0 + H_CROWN * dH1
    junc_fl_ss = flotherm_steady(K_eff_fl, g)
    fl_times, junc_fl_hist = flotherm_transient(Mf, K_eff_fl, g, DT_S, DURATION_S)

    # ---- numeric comparison --------------------------------------------
    print("\n=== STEADY junction temperatures (K) ===")
    print(
        f"{'port':<6}{'full FVM':>10}{'our ROM':>10}{'Flotherm':>10}"
        f"{'our-FVM':>10}{'fl-FVM':>10}"
    )
    for i, name in enumerate(DIE_NAMES):
        print(
            f"{name:<6}{junc_full_ss[i]:>10.3f}{junc_rom_ss[i]:>10.3f}"
            f"{junc_fl_ss[i]:>10.3f}{junc_rom_ss[i]-junc_full_ss[i]:>+10.3f}"
            f"{junc_fl_ss[i]-junc_full_ss[i]:>+10.3f}"
        )
    print(f"\nsteady full-FVM field peak = {Tf_ss.max():.3f} K")

    # steady full-field accuracy of our ROM vs full FVM
    rec_acc = accuracy_summary(Tf_ss, rec_ss, full.history, rec_hist, AMB)

    # transient max junction error (% of steady rise), per scenario
    def pct_err(a, b):
        denom = np.abs(junc_full_ss - AMB)
        return 100.0 * np.max(np.abs(a - b), axis=0) / np.maximum(denom, 1e-9)

    err_rom = pct_err(junc_rom_hist, junc_full_hist)
    err_fl = pct_err(junc_fl_hist, junc_full_hist)
    print("\n=== TRANSIENT max junction error vs full FVM (% of steady rise) ===")
    print(f"{'port':<6}{'our ROM':>10}{'Flotherm':>10}")
    for i, name in enumerate(DIE_NAMES):
        print(f"{name:<6}{err_rom[i]:>10.4f}{err_fl[i]:>10.4f}")

    # ---- plot: steady bars ---------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(4)
    w = 0.25
    ax.bar(idx - w, junc_full_ss, w, label="full FVM", color="tab:blue")
    ax.bar(idx, junc_rom_ss, w, label="our ROM (BCI-FANTASTIC)", color="tab:orange")
    ax.bar(idx + w, junc_fl_ss, w, label="FloTHERM ROM", color="tab:green")
    ax.set_xticks(idx)
    ax.set_xticklabels(DIE_NAMES)
    ax.axhline(AMB, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("junction temperature [K]")
    ax.set_title("Steady-state junction temperature, case1 " "(ambient 308.15 K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "steady_junction.png", dpi=150)
    plt.close(fig)

    # ---- plot: transient time series (per die) -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(
            full.times,
            junc_full_hist[:, i],
            "-",
            color="tab:blue",
            label="full FVM",
            lw=2,
        )
        ax.plot(
            r_times,
            junc_rom_hist[:, i],
            "--",
            color="tab:orange",
            label="our ROM",
            lw=2,
        )
        ax.plot(
            fl_times,
            junc_fl_hist[:, i],
            ":",
            color="tab:green",
            label="FloTHERM ROM",
            lw=2,
        )
        ax.set_title(DIE_NAMES[i])
        ax.set_ylabel("[K]")
        ax.grid(alpha=0.3)
    axes[-1, -1].set_xlabel("time [s]")
    axes[-1, 0].set_xlabel("time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Transient junction temperature — case1 " "(power step at t=0)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "transient_junction.png", dpi=150)
    plt.close(fig)

    print(f"\nplots -> {OUT}")

    # ---- save numeric summary ------------------------------------------
    with open(OUT / "summary.txt", "w") as f:
        f.write(
            f"ambient={AMB}  h_crown={H_CROWN}  h_fr4={H_FR4}  "
            f"dt={DT_S}  duration={DURATION_S}\n"
        )
        f.write(f"sources={SOURCES.tolist()}\n")
        f.write(
            f"full cells={model.full_cell_count}  our basis order="
            f"{basis.shape[1]}\n"
        )
        f.write("steady junction:\n")
        f.write("port,full_fvm,our_rom,flotherm\n")
        for i, n in enumerate(DIE_NAMES):
            f.write(
                f"{n},{junc_full_ss[i]:.4f},{junc_rom_ss[i]:.4f},"
                f"{junc_fl_ss[i]:.4f}\n"
            )
        f.write("steady full-field peak: " f"{Tf_ss.max():.4f}\n")
        f.write("transient max junction error (% of steady rise):\n")
        f.write("port,our_rom,flotherm\n")
        for i, n in enumerate(DIE_NAMES):
            f.write(f"{n},{err_rom[i]:.4f},{err_fl[i]:.4f}\n")
        f.write("full-field recovery (our ROM vs full FVM):\n")
        for k in (
            "steady_max_absolute_rise_error_K",
            "steady_max_relative_rise_error",
            "transient_final_max_absolute_rise_error_K",
        ):
            f.write(f"  {k} = {rec_acc[k]:.6g}\n")

    return dict(
        junc_full_ss=junc_full_ss,
        junc_rom_ss=junc_rom_ss,
        junc_fl_ss=junc_fl_ss,
        junc_full_hist=junc_full_hist,
        junc_rom_hist=junc_rom_hist,
        junc_fl_hist=junc_fl_hist,
        times=full.times,
        basis=basis,
        summary=summary,
    )


if __name__ == "__main__":
    run()
