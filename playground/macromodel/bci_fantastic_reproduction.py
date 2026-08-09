#!/usr/bin/env python3
"""Faithful BCI-FANTASTIC reproduction with a Flotherm-shaped validation.

Implements the BCI-FANTASTIC pipeline (Codecasa, THERMINIC 2015, extending
FANTASTIC THERMINIC 2014) as faithfully as the MetaHotspot macromodel
scaffold allows, and validates it the way Simcenter Flotherm's BCI-ROM
Validation (v2020.2, Sec. 3-4) validates its ROMs:

  * Parametric MOR with boundary-condition independence:
      - the *whole package* is one FEM domain; heat sources are the ports
        (FANTASTIC 2014: ``(σM + K)X = g_i`` for every source port i), each
        with an independent shape in ``G_src``;
      - **per heat-source port** the spectral bounds ``(λ_i, Λ_i)`` of the
        power-impulse response are estimated and a per-port elliptic shift
        count ``m_i`` is computed (Extended FANTASTIC 2021 Algorithm 1 — the
        method behind Simcenter Flotherm's BCI-ROM), so ports that excite
        narrower spectra get fewer shifts;
      - boundary faces are partitioned into groups (top / sides / bottom),
        each with an independent heat-exchange coefficient h_k drawn from an
        admissible range   (BCI 2015 Sec. 2, eq. 5);
      - the Robin terms enter linearly as ``Σ_k h_k Ĥ_k`` (BCI 2015 eq. 7),
        each group exposed as an area-averaged boundary port ``Ĝ_k``;
      - Algorithm 1: parameters sampled at random (not greedy); after each
        enrichment the reduced model is probed at freshly drawn random HTC
        vectors until the residual satisfies the tolerance — sampling stops on
        the error estimate, not on a hardcoded count  (BCI 2015 Algorithm 1 /
        Extended FANTASTIC 2021 lines 5-6);
      - the final basis is SVD-truncated at the relative error ε.
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
:func:`build_parametric_basis` enrichment is driven unchanged.  The default
model is the Package-on-Package ``bci_pop`` (Flotherm BCI-ROM Validation §4.1:
two stacked wire-bonded BGAs, three ambient groups).  The report plots the
per-(port, shift) inner-loop sample count (enrichment cost) and a transient
comparison at the first holdout point.

Outputs curves (PNG) instead of a JSON report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from affine_parametric_models import create
from utils import (
    accuracy_summary,
    assemble_reduced_k,
    build_parametric_basis,
    project_bci,
    solve_rom_steady,
    solve_rom_transient,
)

OUT_DIR = Path("results/bci_fantastic_reproduction")
PROBE_ROUNDS = 3  # consecutive random h-vectors certifying each (port, shift)
RANDOM_SEED = 20260805
RESIDUAL_TOLERANCE = 1.0e-3  # residual-driven enrichment stop tolerance
TARGET_RELATIVE_EPSILON = 1.0e-3  # elliptic shift-count target (Extended FANTASTIC eq.)
MAX_ORDER = 2048
GRID_PER_AXIS = 8  # holdout points (2-group: 8x8=64 grid; 3-group: 8 scenarios)


def plot_results(cfg, summary, basis, scenario_results, curves, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)

    hist = summary["history"]
    if hist:
        # One bar per (port, shift): the number of h-vector samples the inner
        # loop needed (enrichment rounds plus certified probes) before the
        # reduced model met the residual tolerance.  x is the outer-loop
        # sequence number.
        ports = [h["port"] for h in hist]
        shifts = [h["shift"] for h in hist]
        samples = np.asarray([h["h_samples"] for h in hist])
        x = np.asarray([h["outer_idx"] for h in hist])
        colors = plt.cm.tab10(np.asarray(ports) % 10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x, samples, color=colors, width=0.8, edgecolor="none", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xlabel("outer-loop sequence (port, shift)")
        ax.set_ylabel("h-vector samples in inner loop")
        ax.grid(axis="y", alpha=0.3)

        # label each bar with (port, shift)
        ticks = ax.get_xticks()
        labels = [f"({p},{s:.3g})" for p, s in zip(ports, shifts)]
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)

        # legend: distinct ports
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=plt.cm.tab10(p % 10),
                markersize=8,
                label=f"port {p}",
            )
            for p in sorted(set(ports))
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

        ax.set_title(
            "Inner-loop sample count per (port, shift)\n"
            f"(certified after {summary['probe_rounds']} consecutive probes "
            f"below $\\varepsilon$ = {summary['residual_tolerance']:.0e})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / "enrichment.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    # transient comparison at the first holdout point (any group count)
    target = scenario_results[0]["h_vec"] if scenario_results else None
    ambient_K = cfg.get("ambient_K", 300.0)
    for i, (h_vec, t, ref, rom) in enumerate(curves):
        if target is not None and tuple(h_vec) != tuple(target):
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
        ax.set_title(f"Transient response, h=({', '.join(f'{v:.2g}' for v in h_vec)})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "transient_comparison.png", dpi=150)
    plt.close(fig)

    print(f"plots -> {plot_dir}")


# ------------------------------------------------------------- main ----


def run(model, plot_dir: Path, quick: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ambient_K = model.ambient_K

    # -- full-domain core + source ports + boundary groups ---------------
    core = model.core_operators()
    source_shape = model.source_shape()
    groups = model.boundary_groups()
    boundary_terms = model.boundary_terms()
    h_ranges = model.h_ranges()

    # -- extraction (driven by the real power inputs) -------------------
    basis, summary = build_parametric_basis(
        core,
        source_shape,
        boundary_terms,
        h_ranges,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_ORDER,
        target_relative_epsilon=TARGET_RELATIVE_EPSILON,
        probe_rounds=PROBE_ROUNDS,
        seed=RANDOM_SEED,
    )
    cfg = model.report_dict()
    n_modes = basis.shape[1]
    print(
        f"macro grid {model.config.nx}x{model.config.nx}x{cfg.get('nz')}; "
        f"source ports={len(model.source_ports())}; "
        f"basis order {n_modes}; "
        f"worst response err {summary['relative_response_error']:.3e}"
    )
    for plan in summary["per_port_plans"]:
        print(
            f"  port {plan['port']}: lambda in [{plan['lambda_min']:.3e}, "
            f"{plan['lambda_max']:.3e}] 1/s, kappa={plan['kappa']:.2e}, "
            f"{plan['shift_count']} shifts"
        )

    C_hat, K_hat0, F_hat, F_bdry, A_bdry = project_bci(
        core, source_shape, boundary_terms, basis
    )

    def online_operators(h_vec):
        K_hat = assemble_reduced_k(K_hat0, F_bdry, A_bdry, h_vec)
        return C_hat, K_hat, F_hat

    # -- validation with independent holdout ----------------------------
    # The model lays out its own parameter space (e.g. bci_pop: DoE-style
    # random scenarios); the experiment only iterates the points, so the BCI
    # claim (any BC in range) is exercised across the parameter space.  Quick
    # mode keeps the holdout light.
    holdout = model.parameter_points(count=GRID_PER_AXIS if not quick else 4)

    scenario_results = []
    curves = []
    for h_vec in holdout:
        ref = model.full_reference(h_vec)
        ref_ss = ref.steady_temperature
        ref_junc = model.junction_temperature(ref_ss)
        ref_junc_history = model.junction_temperature(ref.history)

        C_hat, K_hat, F_hat = online_operators(h_vec)
        power_nominal = model.nominal_power()
        reduced_started = time.perf_counter()
        theta_ss = solve_rom_steady(K_hat, F_hat, power_nominal)
        rom_junc_ss = ambient_K + F_hat.T @ theta_ss
        rom_steady_s = time.perf_counter() - reduced_started

        reduced_started = time.perf_counter()
        times, theta_history = solve_rom_transient(
            C_hat,
            K_hat,
            F_hat,
            model.source_power,
            dt=model.dt,
            duration=model.config.duration_s,
        )
        rom_transient_s = time.perf_counter() - reduced_started
        assert np.allclose(times, ref.times, atol=1e-9, rtol=0.0)
        rom_junc_history = ambient_K + theta_history @ F_hat

        per_point = []
        for m in range(F_hat.shape[1]):
            denom = abs(ref_junc[m] - ambient_K)
            err = (
                100.0
                * np.max(np.abs(ref_junc_history[:, m] - rom_junc_history[:, m]))
                / denom
                if denom
                else 0.0
            )
            per_point.append(err)
        scenario_results.append(
            {
                "h_vec": h_vec,
                "max_err_pct": max(per_point),
                "per_point": per_point,
                "rom_steady_s": rom_steady_s,
                "rom_transient_s": rom_transient_s,
                "full_transient_s": ref.transient_s,
            }
        )
        curves.append((h_vec, ref.times, ref_junc_history, rom_junc_history))

        # full-field context: temperature ranges and max rise errors.
        recovered_steady = ambient_K + basis @ theta_ss
        recovered_history = ambient_K + theta_history @ basis.T
        full_acc = accuracy_summary(
            ref_ss,
            np.asarray(recovered_steady),
            ref.history,
            np.asarray(recovered_history),
            ambient_K,
        )
        s_range = full_acc["steady_reference_temperature_range_K"]
        t_range = full_acc["transient_final_reference_temperature_range_K"]
        speedup = ref.transient_s / max(rom_transient_s, np.finfo(float).tiny)
        print(
            f"  holdout h={tuple(round(x,2) for x in h_vec)}: "
            f"ref range steady={s_range[0]:.3f}..{s_range[1]:.3f} K, "
            f"transient final={t_range[0]:.3f}..{t_range[1]:.3f} K; "
            f"rise error steady={full_acc['steady_max_absolute_rise_error_K']:.5f} K/"
            f"{full_acc['steady_max_relative_rise_error']:.3%}, transient final="
            f"{full_acc['transient_final_max_absolute_rise_error_K']:.5f} K/"
            f"{full_acc['transient_final_max_relative_rise_error']:.3%}; "
            f"junction max err {max(per_point):.4f}%; "
            f"full/ROM={ref.transient_s:.3f}/{rom_transient_s:.4f}s, "
            f"speedup={speedup:.2f}x"
        )

    errors = np.asarray([s["max_err_pct"] for s in scenario_results])
    print(
        f"holdout max {errors.max():.4f}% mean {errors.mean():.4f}% std {errors.std():.4f}%"
    )

    # report plots: transient comparison at the first holdout point (any group
    # count); the error-vs-h scatter for the two-group case, a max-error
    # histogram otherwise.  Always safe to plot.
    plot_results(
        model.report_dict(), summary, basis, scenario_results, curves, plot_dir
    )
    return summary, scenario_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--model",
        default="bci_pop",
        help="registered affine parametric model name (default: bci_pop)",
    )
    args = parser.parse_args(argv)

    # quick/strict is a factory toggle: each model applies its own quick-mode
    # config recipe when asked, the experiment never names a config field.
    model = create(args.model, quick=args.quick)
    t0 = time.perf_counter()
    run(model, OUT_DIR, quick=args.quick)
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
