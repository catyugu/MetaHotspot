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
model or a config field, so it runs unchanged against any registered
multi-group implementation (``--model bci_pkg`` by default).

Outputs curves (PNG) instead of a JSON report.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metahotspot.compiled import Operators  # noqa: E402
from affine_parametric_models import create  # noqa: E402
from utils import (  # noqa: E402
    closure_diagonal_multi,
    eigenpairs_descending,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    orthonormalize_block,
    project_exact_ports,
    project_closure_group,
    reduced_response,
    response_error,
    symmetric_dense,
)

OUT_DIR = Path("results/bci_fantastic_reproduction")
H_RANGE = (1.0, 1.0e6)  # Flotherm default 1..10,000 W/m2K
RANDOM_PARAMETER_SAMPLES = 24  # random h-vectors for training (Algorithm 1)
RANDOM_SEED = 20260805
RESIDUAL_TOLERANCE = 5.0e-3  # residual-driven enrichment stop tolerance
TARGET_RELATIVE_EPSILON = 5.0e-3  # elliptic shift-count target (FANTASTIC 2014 eq. 4)
MAX_ORDER = 2048


def random_parameter_vectors(h_ranges, sample_count, seed, boundaries=None):
    """Random admissible h-vectors (one h per group), log-uniform.  No greedy.

    FANTASTIC BCI 2015 Algorithm 1: parameters chosen at random to avoid
    reduced-basis greedy stagnation.  ``boundaries`` (geometric holdout) are
    appended so the certified range is covered at its extremes.
    """
    rng = np.random.default_rng(seed)
    vectors = [
        tuple(
            10.0 ** rng.uniform(math.log10(lo), math.log10(hi)) for lo, hi in h_ranges
        )
        for _ in range(sample_count)
    ]
    for b in boundaries or ():
        vectors.append(tuple(b))
    seen, out = set(), []
    for v in vectors:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_bci_basis(
    core,
    ports,
    boundary_groups,
    boundary_areas,
    *,
    h_ranges,
    boundaries,
    residual_tolerance,
    max_order,
):
    """Multi-group BCI-FANTASTIC extraction (Algorithm 1).

    Candidates are ``(h_vec, shift)``: random admissible boundary-coefficient
    vectors crossed with the FANTASTIC-2014 elliptic-optimal complex shifts.
    The candidate operator is
        A(h_vec, shift) = K_ii + shift*C_ii + diag(closure_multi(h_vec))
    with the exact saturating per-group closure.  Every candidate streams one
    frequency-domain solve; residual directions above tolerance are inserted
    immediately.  The basis is kept column-orthonormal throughout (real-time
    modified Gram-Schmidt).
    """
    started = time.perf_counter()
    K0, C0, B0, D0 = (
        core.K[ports:, ports:].tocsc(),
        core.C[ports:, ports:].tocsc(),
        core.K[ports:, :ports].tocsc(),
        core.C[ports:, :ports].tocsc(),
    )
    h_vectors = random_parameter_vectors(
        h_ranges, RANDOM_PARAMETER_SAMPLES, RANDOM_SEED, boundaries
    )

    eigenvalue_scale = max(float(np.max(np.abs(C0.diagonal()))), np.finfo(float).tiny)
    eigenvalue_ratio = max(
        math.sqrt(np.linalg.cond(K0.todense().astype(np.float64))), 1.0
    )
    kappa = eigenvalue_ratio**2
    lambda_min = float(eigenvalue_scale / kappa)
    lambda_max = float(eigenvalue_scale)
    if kappa > 1.0e6:
        lambda_min = max(lambda_min, lambda_max / 1.0e6)
        kappa = lambda_max / lambda_min
    elliptic_count = mpmm_elliptic_shift_count(
        TARGET_RELATIVE_EPSILON, lambda_min, lambda_max
    )
    shifts = np.r_[0.0, mpmm_elliptic_shifts(elliptic_count, lambda_max, kappa)]

    raw_points = [(hv, float(shift)) for hv in h_vectors for shift in shifts]
    internal_order = K0.shape[0]
    order_limit = min(max_order, internal_order)
    basis = np.empty((internal_order, 0), dtype=np.float64)
    history = []
    worst_score = 0.0
    converged = True

    for h_vec, shift in raw_points:
        closure = closure_diagonal_multi(
            h_vec, boundary_groups, boundary_areas, internal_order
        )
        A = (K0 + shift * C0 + sp.diags(closure)).tocsc()
        A = (0.5 * (A + A.T)).tocsc()
        B_dense = (B0 + shift * D0).toarray()

        response = np.asarray(spla.splu(A).solve(-B_dense))
        response_gram = symmetric_dense(-response.T @ B_dense)
        response_values, _ = eigenpairs_descending(response_gram)
        reference = max(float(response_values[0]), np.finfo(float).tiny)

        order_before = basis.shape[1]
        reduced = reduced_response(basis, A, B_dense)
        error_response, error_values, tangents, score_before = response_error(
            response, basis, reduced, A, reference
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
            reduced = reduced_response(basis, A, B_dense)
            _, _, _, score_after = response_error(
                response, basis, reduced, A, reference
            )

        worst_score = max(worst_score, score_after)
        history.append(
            {
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
            break

    if basis.shape[1]:
        orthogonality = basis.T @ basis - np.eye(basis.shape[1])
        orthogonality_error = float(np.max(np.abs(orthogonality)))
    else:
        orthogonality_error = 0.0
    if orthogonality_error > 1.0e-10:
        raise RuntimeError("rational Krylov basis lost orthogonality")

    return basis, {
        "parameter_vectors": h_vectors,
        "elliptic_shift_count": elliptic_count,
        "elliptic_shifts": shifts[1:].tolist(),
        "eigenvalue_ratio_kappa": kappa,
        "target_relative_epsilon": TARGET_RELATIVE_EPSILON,
        "candidate_count": len(raw_points),
        "basis_order": int(basis.shape[1]),
        "relative_response_error": float(worst_score),
        "residual_tolerance": residual_tolerance,
        "converged": bool(converged and len(history) == len(raw_points)),
        "history": history,
        "seconds": time.perf_counter() - started,
    }


# -------------------------------------------------------------- validation ----


# ------------------------------------------------------------- plots ----


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
    h_ranges = tuple(g.h_range for g in groups)
    basis, summary = build_bci_basis(
        core,
        ports,
        boundary_groups,
        group_areas,
        h_ranges=h_ranges,
        boundaries=None,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_ORDER,
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
    # Log-uniform grid over (h_top, h_side) in the admissible range: a dense
    # independent holdout so the BCI claim (any BC in range) is exercised
    # across the parameter space, and error_vs_h has enough points.
    grid_per_axis = 8  # 64 combos
    axis = np.geomspace(h_ranges[0][0], h_ranges[0][1], grid_per_axis)
    holdout = [(float(a), float(b)) for a in axis for b in axis]

    scenario_results = []
    curves = []
    for h_vec in holdout:
        h_top, h_side = h_vec
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

    plot_results(
        model.report_dict(), summary, basis, scenario_results, curves, plot_dir
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

    overrides = (
        {"max_xy_cell_mm": 2.0, "duration_s": 300.0, "dt_s": 30.0}
        if args.quick
        else None
    )
    model = create(args.model, overrides=overrides)
    t0 = time.perf_counter()
    run(model, OUT_DIR)
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
