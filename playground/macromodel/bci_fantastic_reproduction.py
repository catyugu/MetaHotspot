#!/usr/bin/env python3
"""Faithful BCI-FANTASTIC reproduction with a Flotherm-shaped validation.

Implements the BCI-FANTASTIC pipeline (Codecasa, THERMINIC 2015, extending
FANTASTIC THERMINIC 2014) as faithfully as the MetaHotspot macromodel
scaffold allows, and validates it the way Simcenter Flotherm's BCI-ROM
Validation (v2020.2, Sec. 3-4) validates its ROMs:

  * Parametric MOR with boundary-condition independence:
      - boundary faces partitioned into groups (top / side), each with an
        independent heat-exchange coefficient h_k drawn from an admissible
        range   (BCI 2015 Sec. 2, eq. 5);
      - the Robin terms are NOT in the reduced operators: each boundary group
        is exposed as boundary ports and eliminated through the exact
        saturating closure g*h*A/(g+h*A)   (BCI 2015 Sec. 3-4);
      - Algorithm 1: parameters sampled at random (not greedy), residual-driven
        enrichment one step per candidate   (BCI 2015 Algorithm 1);
      - complex-frequency shifts are the FANTASTIC-2014 elliptic-optimal
        points with per-problem shift count from the eigenvalue bounds.
  * Flotherm-style validation:
      - power step at t=0 (exercises all frequencies), transient to steady;
      - independent holdout HTC scenarios drawn in-range (NOT the training
        samples), so BCI (any BC in range) is actually tested;
      - percent error = max_t |Theta_full - Theta_rom| / Theta_full,ss * 100
        per monitor point (die-top junction temperatures), then max/mean/std
        over scenarios   (Flotherm v2020.2 Sec. 3 eq. 6).

This script is *model-agnostic*: it obtains its model from the
:mod:`affine_parametric_models` factory (``create``) and drives it through the
abstract :class:`AffineParametricModel` contract.  It never names a concrete
model or a config field, and it is *parameter-count-agnostic*: the number of
affine parameters is whatever ``boundary_groups()`` reports, and the shared
:func:`build_parametric_basis` enrichment is driven unchanged.  The validation
plots are shaped for the two-group case (transient comparison at
``(10000, 10000)``, error scatter over ``(h_top, h_side)``), so when the model
does not have exactly two boundary groups the validation still runs but the
report plots are skipped.

Outputs curves (PNG) instead of a JSON report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metahotspot.compiled import Operators
from affine_parametric_models import create
from utils import (
    build_parametric_basis,
    project_exact_ports,
    project_closure_group,
)

OUT_DIR = Path("results/bci_fantastic_reproduction")
RANDOM_PARAMETER_SAMPLES = 24  # random h-vectors for training (Algorithm 1)
RANDOM_SEED = 20260805
RESIDUAL_TOLERANCE = 5.0e-3  # residual-driven enrichment stop tolerance
TARGET_RELATIVE_EPSILON = 5.0e-3  # elliptic shift-count target (FANTASTIC 2014 eq. 4)
MAX_ORDER = 2048
GRID_PER_AXIS = 8  # holdout points per sweep axis (two-group: 8x8 = 64 combos)


def plot_results(cfg, summary, basis, scenario_results, curves, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)

    hist = summary["history"]
    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(1, len(hist) + 1)
        ax.plot(
            x,
            [h["order_after"] for h in hist],
            "o-",
            color="tab:blue",
            label="basis order",
        )
        ax.set_xlabel("candidate")
        ax.set_ylabel("basis order", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(
            x,
            [h["score_after"] for h in hist],
            "s--",
            color="tab:red",
            label="response error",
        )
        ax2.set_ylabel("response error", color="tab:red")
        ax.set_title("Residual-driven enrichment (Algorithm 1)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "enrichment.png", dpi=150)
        plt.close(fig)

    errors = np.asarray([s["max_err_pct"] for s in scenario_results])

    fig, ax = plt.subplots(figsize=(8, 5))
    # always show the transient comparison at h = (1000, 1000) W/m2K
    target = (10000.0, 10000.0)
    ambient_K = cfg.get("ambient_K", 300.0)
    for i, (h_vec, t, ref, rom) in enumerate(curves):
        if tuple(h_vec) != target:
            continue
        for m in range(ref.shape[1]):
            ax.plot(t, ref[:, m] - ambient_K, "-", color=f"C{m}", label=f"full mon{m}")
            ax.plot(
                t,
                rom[:, m] - ambient_K,
                "o--",
                color=f"C{m}",
                mfc="none",
                label=f"ROM mon{m}",
            )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("rise over ambient [K]")
        ax.set_title(f"Transient response, h=({h_vec[0]:.2g}, {h_vec[1]:.2g})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "transient_comparison.png", dpi=150)
    plt.close(fig)

    h0 = np.asarray([s["h_vec"][0] for s in scenario_results])
    h1 = np.asarray([s["h_vec"][1] for s in scenario_results])
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(h0, h1, c=errors, cmap="viridis", s=80)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("h_top [W/m2K]")
    ax.set_ylabel("h_side [W/m2K]")
    fig.colorbar(sc, label="max err [%]")
    ax.set_title("Holdout error vs boundary coefficients")
    fig.tight_layout()
    fig.savefig(plot_dir / "error_vs_h.png", dpi=150)
    plt.close(fig)

    print(f"plots -> {plot_dir}")


# ------------------------------------------------------------- main ----


def run(model, plot_dir: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ambient_K = model.ambient_K

    # -- macro DtN core + boundary groups -------------------------------
    core = model.core_operators()
    ports = model.port_count
    groups = model.boundary_groups()
    boundary_groups = [(g.cells, g.g) for g in groups]
    group_areas = [g.areas for g in groups]

    # -- extraction -----------------------------------------------------
    h_ranges = np.asarray([g.h_range for g in groups], dtype=np.float64)
    basis, summary = build_parametric_basis(
        core,
        ports,
        boundary_groups,
        group_areas,
        h_ranges=h_ranges,
        boundaries=None,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_ORDER,
        target_relative_epsilon=TARGET_RELATIVE_EPSILON,
        sample_count=RANDOM_PARAMETER_SAMPLES,
        seed=RANDOM_SEED,
    )
    cfg = model.report_dict()
    print(
        f"macro grid {model.macro_nx}x{model.macro_nx}x{cfg.get('macro_nz', cfg.get('nz'))}; "
        f"interface ports={ports}; "
        f"basis order {summary['basis_order']}; "
        f"worst response err {summary['relative_response_error']:.3e}"
    )

    reduced_core = project_exact_ports(core, ports, basis, ambient_K)
    n_modes = basis.shape[1]
    n_cell = basis.shape[0]

    proj_closure = [
        project_closure_group(g.cells, g.g, g.areas, n_cell, basis) for g in groups
    ]

    def online_operators(h_vec):
        delta = sum(cm(h).toarray() for cm, h in zip(proj_closure, h_vec))
        D = sp.bmat(
            (
                (sp.csc_matrix((ports, ports)), sp.csc_matrix((ports, n_modes))),
                (sp.csc_matrix((n_modes, ports)), sp.csc_matrix(delta)),
            ),
            format="csc",
        )
        return Operators((reduced_core.K + D).tocsc(), reduced_core.C, reduced_core.f)

    mon_detail = model.monitor_cells()
    mon_full = model.monitor_full(mon_detail)

    detail_count = model.detail_cell_count
    initial = model.initial_state(n_modes)

    # -- validation with independent holdout ----------------------------
    # The model lays out its own parameter space (for the two-group package
    # this is the dense (h_top, h_side) product grid); the experiment only
    # iterates the points, so the BCI claim (any BC in range) is exercised
    # across the parameter space, and error_vs_h has enough points.
    holdout = model.parameter_points(count=GRID_PER_AXIS)

    scenario_results = []
    curves = []
    for h_vec in holdout:
        ref = model.full_reference(h_vec)
        ref_ss = ref.steady_temperature[mon_full]
        ref_curves = ref.history[:, mon_full]

        reduced = online_operators(h_vec)
        rom_ss, _ = model.solve_reduced(reduced, initial, False)
        times, rom_states, _ = model.solve_reduced(reduced, initial, True)
        assert np.allclose(times, ref.times, atol=1e-9, rtol=0.0)

        rom_curves = rom_states[:, mon_detail]

        per_point = []
        for m in range(mon_detail.size):
            denom = abs(ref_ss[m] - ambient_K)
            err = (
                100.0 * np.max(np.abs(ref_curves[:, m] - rom_curves[:, m])) / denom
                if denom
                else 0.0
            )
            per_point.append(err)
        scenario_results.append(
            {"h_vec": h_vec, "max_err_pct": max(per_point), "per_point": per_point}
        )
        curves.append((h_vec, ref.times, ref_curves, rom_curves))
        print(
            f"  holdout h={tuple(round(x,2) for x in h_vec)}: max err {max(per_point):.4f}%"
        )

    errors = np.asarray([s["max_err_pct"] for s in scenario_results])
    print(
        f"holdout max {errors.max():.4f}% mean {errors.mean():.4f}% std {errors.std():.4f}%"
    )

    # The report plots are shaped for the two-group parameterization (the
    # transient comparison pins h=(10000, 10000) and the error scatter maps
    # (h_top, h_side)); for any other number of affine parameters skip them.
    if len(groups) == 2:
        plot_results(
            model.report_dict(), summary, basis, scenario_results, curves, plot_dir
        )
    else:
        print(
            f"{len(groups)} affine parameter(s) — report plots are two-group "
            "shaped, skipping plot generation"
        )
    return summary, scenario_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--model",
        default="bci_pkg",
        help="registered affine parametric model name (default: bci_pkg)",
    )
    args = parser.parse_args(argv)

    # quick/strict is a factory toggle: each model applies its own quick-mode
    # config recipe when asked, the experiment never names a config field.
    model = create(args.model, quick=args.quick)
    t0 = time.perf_counter()
    run(model, OUT_DIR)
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
