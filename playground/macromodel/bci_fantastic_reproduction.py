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

Layout (z from 0 up): die (detail, heat source) at the bottom, then
substrate + cap (macro, solid stack) on top.  The macro block bottom face is
the interface (= die top); its top face and side walls are the parametric
boundary groups {top, side}.  Monitor points are the die-top junction cells.

Outputs curves (PNG) instead of a JSON report.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metahotspot  # noqa: E402
from metahotspot.compiled import Operators, SolveOptions  # noqa: E402
from metahotspot.enums import Axis, Face, GeometryOp, LengthUnit, Study  # noqa: E402
from metahotspot.macromodel import (
    PortMap,
    PortPatch,
    solve as solve_macro,
)  # noqa: E402

from utils import (  # noqa: E402
    closure_diagonal_multi,
    eigenpairs_descending,
    extract_boundary_groups,
    mpmm_elliptic_shift_count,
    mpmm_elliptic_shifts,
    normalized_operators,
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

MATERIALS = (
    ("organic", ".65", ".65", ".55", "1900", "1100"),
    ("silicon", "130", "130", "115", "2330", "700"),
)


# ---------------------------------------------------------------- geometry ----


@dataclass(frozen=True)
class PkgConfig:
    """Flotherm-flavoured package: die (detail, hot) under substrate+cap (macro)."""

    ambient_K: float = 300.0
    size_mm: float = 14.0  # all layers share lateral extent
    die_h_mm: float = 0.6
    substrate_h_mm: float = 1.0
    cap_h_mm: float = 1.0
    die_cells: int = 2
    substrate_cells: int = 3
    cap_cells: int = 3
    max_xy_cell_mm: float = 1.0
    source_power_W: float = 1.0
    # fraction of the total power in the left heat zone (right = 1 - left)
    left_power_frac: float = 0.6
    duration_s: float = 600.0
    dt_s: float = 30.0
    h_ranges: tuple = (H_RANGE, H_RANGE)  # [top, side]

    @property
    def macro_layers(self):
        return (
            (self.substrate_h_mm, self.substrate_cells),
            (self.cap_h_mm, self.cap_cells),
        )

    @property
    def detail_layers(self):
        return ((self.die_h_mm, self.die_cells),)

    @property
    def detail_nz(self) -> int:
        return sum(cells for _, cells in self.detail_layers)

    @property
    def macro_nz(self) -> int:
        return sum(cells for _, cells in self.macro_layers)

    @property
    def nz(self) -> int:
        return self.detail_nz + self.macro_nz

    @property
    def macro_height_mm(self) -> float:
        return sum(thickness for thickness, _ in self.macro_layers)

    @property
    def total_height_mm(self) -> float:
        return self.die_h_mm + self.macro_height_mm

    @cached_property
    def axis_vertices_mm(self) -> np.ndarray:
        half = self.size_mm / 2.0
        fixed = np.unique([-half, 0.0, half])
        vertices = [float(fixed[0])]
        for left, right in zip(fixed[:-1], fixed[1:]):
            pieces = max(1, math.ceil((right - left) / self.max_xy_cell_mm))
            vertices.extend(np.linspace(left, right, pieces + 1)[1:])
        return np.asarray(vertices)

    @property
    def nx(self) -> int:
        return self.axis_vertices_mm.size - 1

    def report_dict(self) -> dict:
        return {
            **{k: v for k, v in self.__dict__.items() if not k.startswith("_")},
            "nx": self.nx,
            "nz": self.nz,
        }


def solve_options(cfg: PkgConfig, transient: bool) -> SolveOptions:
    """Solve options tuned for the package geometry (fixed-step BDF1)."""
    dt = cfg.dt_s if transient else 1.0
    return SolveOptions(
        linear_solver="EigenSparseLU",
        linear_tolerance=1.0e-12,
        linear_max_iterations=5000,
        nonlinear_max_iterations=30,
        nonlinear_relative_tolerance=1.0e-11,
        nonlinear_absolute_tolerance=1.0e-11,
        integrator="Bdf1",
        step_strategy="Fixed",
        error_rel_tol=1.0e-3,
        min_dt=dt,
        max_dt=dt,
        fixed_dt=dt,
    )


def z_vertices(layers) -> np.ndarray:
    vertices = [0.0]
    for thickness, cells in layers:
        vertices.extend(vertices[-1] + thickness * np.arange(1, cells + 1) / cells)
    return np.asarray(vertices)


def add_square(model, block: int, size_mm: float) -> None:
    half = size_mm / 2.0
    model.add_rect(
        block,
        GeometryOp.ADD,
        f"{-half:.17g}",
        f"{-half:.17g}",
        f"{size_mm:.17g}",
        f"{size_mm:.17g}",
    )


def build_geometry(
    cfg: PkgConfig,
    study: Study,
    *,
    detail: bool,
    macro: bool,
    convection: dict[int, float] | None = None,
):
    """Assemble geometry.  Layers bottom-up: die (detail), then substrate+cap (macro).

    The macro model (independent compile) spans z in [0, macro_h]; its bottom
    face z=0 is the interface (= die top).  ``convection`` maps Face -> h for
    the native full reference: top on the cap top, side on the macro side
    walls, bottom on the die bottom.
    """
    if not detail and not macro:
        raise ValueError("at least one domain must be enabled")
    model = metahotspot.Model()
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=cfg.duration_s if study == Study.TRANSIENT else 0.0,
        output_interval=cfg.dt_s if study == Study.TRANSIENT else 0.0,
    )
    # Physical z from bottom to top: [die, substrate, cap].  add_layer pushes
    # each new layer to the bottom (last added sits at z=0), so layers are
    # added top-first: cap, then substrate, then die.
    zv = [0.0]
    if detail:
        zv.extend(z_vertices(cfg.detail_layers)[1:])
    if macro:
        macro_z = z_vertices(cfg.macro_layers)[1:] + (cfg.die_h_mm if detail else 0.0)
        zv.extend(macro_z)
    model.set_mesh(cfg.axis_vertices_mm, cfg.axis_vertices_mm, np.asarray(zv))
    for material in MATERIALS:
        model.add_material(*material)

    # macro: cap (top) then substrate (bottom); bottom-up call order inside the
    # "add top-first" scheme means cap is added first, substrate second.
    if macro:
        for thickness, name in (
            (cfg.cap_h_mm, "silicon"),
            (cfg.substrate_h_mm, "organic"),
        ):
            layer = model.add_layer(str(thickness))
            add_square(model, model.add_block(layer, name), cfg.size_mm)

    # detail: die, added last so it sits at the bottom.
    if detail:
        die = model.add_layer(str(cfg.die_h_mm))
        add_square(model, model.add_block(die, "silicon"), cfg.size_mm)
        half = cfg.size_mm / 2.0
        zone_volume = half * cfg.size_mm * cfg.die_h_mm * 1.0e-9  # m^3
        for x0, frac in (
            (-half, cfg.left_power_frac),
            (0.0, 1.0 - cfg.left_power_frac),
        ):
            block = model.add_block(
                die,
                "silicon",
                heat_source=f"{frac * cfg.source_power_W / zone_volume:.17g}",
            )
            model.add_rect(
                block,
                GeometryOp.ADD,
                f"{x0:.17g}",
                f"{-half:.17g}",
                f"{half:.17g}",
                f"{cfg.size_mm:.17g}",
            )

    model.set_default_neumann("0")

    if convection:
        half = cfg.size_mm / 2.0
        die_h = cfg.die_h_mm
        macro_h = cfg.macro_height_mm
        total_h = cfg.total_height_mm
        for face, h in convection.items():
            if not h:
                continue
            f = Face(face)
            if f == Face.ZP:  # cap top (global top)
                regions = [(Axis.Z, total_h, -half, half, -half, half)]
            elif f == Face.ZM:  # die bottom
                regions = [(Axis.Z, 0.0, -half, half, -half, half)]
            else:  # macro side walls, z in [die_h, total_h]
                z0, z1 = die_h, total_h
                regions = []
                for axis, coord in (
                    (Axis.X, -half),
                    (Axis.X, half),
                    (Axis.Y, -half),
                    (Axis.Y, half),
                ):
                    regions.append((axis, coord, -half, half, z0, z1))
            model.add_convection(str(h), str(cfg.ambient_K), regions)
    return model


def macro_interface_patches(cfg: PkgConfig) -> list[PortPatch]:
    """Interface ports on the macro block bottom (= die top, macro z=0)."""
    verts = cfg.axis_vertices_mm * 1.0e-3
    return [
        PortPatch(
            int(Face.ZM), 0.0, (verts[ix], verts[ix + 1], verts[iy], verts[iy + 1])
        )
        for ix in range(verts.size - 1)
        for iy in range(verts.size - 1)
    ]


def macro_boundary_groups(cfg: PkgConfig):
    """Boundary port groups [top, side] with per-group patch areas.

    Top = cap top face (macro z = macro_h), one patch per exposed cell.
    Side = the four side walls over the full macro height, one patch per cell.
    """
    verts = cfg.axis_vertices_mm * 1.0e-3
    zv = z_vertices(cfg.macro_layers) * 1.0e-3
    top_z = zv[-1]

    top = [
        PortPatch(
            int(Face.ZP), top_z, (verts[ix], verts[ix + 1], verts[iy], verts[iy + 1])
        )
        for ix in range(verts.size - 1)
        for iy in range(verts.size - 1)
    ]
    top_areas = np.asarray(
        [
            (verts[ix + 1] - verts[ix]) * (verts[iy + 1] - verts[iy])
            for ix in range(verts.size - 1)
            for iy in range(verts.size - 1)
        ]
    )

    half = cfg.size_mm / 2.0 * 1e-3
    side, side_areas = [], []
    for axis_face, coord, a_verts in (
        (Face.XM, -half, verts),
        (Face.XP, half, verts),
        (Face.YM, -half, verts),
        (Face.YP, half, verts),
    ):
        for a0, a1 in zip(a_verts[:-1], a_verts[1:]):
            for b0, b1 in zip(zv[:-1], zv[1:]):
                side.append(PortPatch(int(axis_face), float(coord), (a0, a1, b0, b1)))
                side_areas.append((a1 - a0) * (b1 - b0))
    return [top, side], [top_areas, np.asarray(side_areas)]


def detail_monitor_cells(cfg: PkgConfig, detail_compiled) -> np.ndarray:
    """Die-top cell indices (in the detail model) above each heat zone centre."""
    grid = detail_compiled.grid_to_cell.reshape(
        detail_compiled.nx, detail_compiled.ny, detail_compiled.nz
    )
    top = grid[:, :, -1]
    verts = cfg.axis_vertices_mm * 1e-3
    centres = (verts[:-1] + verts[1:]) / 2.0
    quarter = cfg.size_mm / 4.0 * 1e-3
    cells = []
    for xc in (-quarter, quarter):
        ix = int(np.argmin(np.abs(centres - xc)))
        iy = int(np.argmin(np.abs(centres - 0.0)))
        cells.append(int(top[ix, iy]))
    return np.asarray(cells)


# ------------------------------------------------------------- extraction ----


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
    cfg: PkgConfig,
    core: Operators,
    ports: int,
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


def full_reference(cfg: PkgConfig, h_top, h_side):
    """Native steady+transient reference for the full (detail+macro) model."""
    convection = {
        int(Face.ZP): h_top,
        int(Face.ZM): 0.0,
        int(Face.XM): h_side,
        int(Face.XP): h_side,
        int(Face.YM): h_side,
        int(Face.YP): h_side,
    }
    steady = build_geometry(
        cfg, Study.STEADY, detail=True, macro=True, convection=convection
    ).compile()
    transient = build_geometry(
        cfg, Study.TRANSIENT, detail=True, macro=True, convection=convection
    ).compile()
    with steady.solve(opts=solve_options(cfg, False)) as sol:
        steady_temperature = sol.temperature
    with transient.solve(opts=solve_options(cfg, True)) as sol:
        times = sol.history_times
        history = sol.temperature_history
    return steady_temperature, times, history


def solve_rom(cfg, detail_compiled, detail_ports, reduced, initial, transient):
    started = time.perf_counter()
    with solve_macro(
        reduced, detail_ports, initial, solve_options(cfg, transient)
    ) as sol:
        elapsed = time.perf_counter() - started
        if transient:
            return sol.history_times, sol.state_history, elapsed
        return sol.state, elapsed


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
    for i, (h_vec, t, ref, rom) in enumerate(curves):
        if tuple(h_vec) != target:
            continue
        for m in range(ref.shape[1]):
            ax.plot(
                t, ref[:, m] - cfg.ambient_K, "-", color=f"C{m}", label=f"full mon{m}"
            )
            ax.plot(
                t,
                rom[:, m] - cfg.ambient_K,
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


def run(cfg: PkgConfig, plot_dir: Path, strict: bool):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- macro DtN core + boundary groups -------------------------------
    macro = build_geometry(cfg, Study.STEADY, detail=False, macro=True).compile()
    interface = macro_interface_patches(cfg)
    groups, group_areas = macro_boundary_groups(cfg)
    group_sizes = [len(g) for g in groups]
    all_boundary = [p for g in groups for p in g]

    pm_merged = PortMap(macro, interface + all_boundary)
    merged = normalized_operators(*pm_merged.assemble())
    boundary_groups = extract_boundary_groups(merged, len(interface), group_sizes)

    pm_core = PortMap(macro, interface)
    core = normalized_operators(*pm_core.assemble())
    ports = pm_core.port_count

    # -- extraction -----------------------------------------------------
    basis, summary = build_bci_basis(
        cfg,
        core,
        ports,
        boundary_groups,
        group_areas,
        h_ranges=cfg.h_ranges,
        boundaries=None,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_order=MAX_ORDER,
    )
    print(
        f"macro grid {cfg.nx}x{cfg.nx}x{cfg.macro_nz}; interface ports={ports}; "
        f"basis order {summary['basis_order']}; "
        f"worst response err {summary['relative_response_error']:.3e}"
    )

    reduced_core = project_exact_ports(core, ports, basis, cfg.ambient_K)
    n_modes = basis.shape[1]
    n_cell = basis.shape[0]

    proj_closure = [
        project_closure_group(cells, g, areas, n_cell, basis)
        for (cells, g), areas in zip(boundary_groups, group_areas)
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

    # -- detail (die) model for the coupled solve -----------------------
    detail_steady = build_geometry(
        cfg, Study.STEADY, detail=True, macro=False
    ).compile()
    detail_transient = build_geometry(
        cfg, Study.TRANSIENT, detail=True, macro=False
    ).compile()
    full_layout = build_geometry(cfg, Study.STEADY, detail=True, macro=True).compile()

    # interface ports on the detail model (die top) = macro interface patches
    detail_interface = [
        PortPatch(int(Face.ZP), cfg.die_h_mm * 1e-3, p.rectangle) for p in interface
    ]
    detail_ports_steady = PortMap(detail_steady, detail_interface)
    detail_ports_transient = PortMap(detail_transient, detail_interface)

    # map detail cells -> full layout cells
    dg = detail_steady.grid_to_cell.reshape(
        detail_steady.nx, detail_steady.ny, detail_steady.nz
    )
    fg = full_layout.grid_to_cell.reshape(
        full_layout.nx, full_layout.ny, full_layout.nz
    )
    dz = cfg.detail_nz
    valid = dg >= 0
    assert np.array_equal(valid, fg[:, :, :dz] >= 0)
    detail_to_full = np.empty(detail_steady.cell_count, dtype=np.int64)
    detail_to_full[dg[valid]] = fg[:, :, :dz][valid]
    assert np.unique(detail_to_full).size == detail_to_full.size

    mon_detail = detail_monitor_cells(cfg, detail_steady)
    mon_full = detail_to_full[mon_detail]

    detail_count = detail_steady.cell_count
    initial = np.r_[np.full(detail_count + ports, cfg.ambient_K), np.zeros(n_modes)]

    # -- validation with independent holdout ----------------------------
    # Log-uniform grid over (h_top, h_side) in the admissible range: a dense
    # independent holdout so the BCI claim (any BC in range) is exercised
    # across the parameter space, and error_vs_h has enough points.
    grid_per_axis = 8  # 64 combos
    axis = np.geomspace(cfg.h_ranges[0][0], cfg.h_ranges[0][1], grid_per_axis)
    holdout = [(float(a), float(b)) for a in axis for b in axis]

    scenario_results = []
    curves = []
    for h_vec in holdout:
        h_top, h_side = h_vec
        ref_steady, ref_times, ref_history = full_reference(cfg, h_top, h_side)
        ref_ss = ref_steady[mon_full]
        ref_curves = ref_history[:, mon_full]

        reduced = online_operators(h_vec)
        rom_ss, _ = solve_rom(
            cfg, detail_steady, detail_ports_steady, reduced, initial, False
        )
        times, rom_states, _ = solve_rom(
            cfg, detail_transient, detail_ports_transient, reduced, initial, True
        )
        assert np.allclose(times, ref_times, atol=1e-9, rtol=0.0)

        rom_curves = rom_states[:, mon_detail]

        per_point = []
        for m in range(mon_detail.size):
            denom = abs(ref_ss[m] - cfg.ambient_K)
            err = (
                100.0 * np.max(np.abs(ref_curves[:, m] - rom_curves[:, m])) / denom
                if denom
                else 0.0
            )
            per_point.append(err)
        scenario_results.append(
            {"h_vec": h_vec, "max_err_pct": max(per_point), "per_point": per_point}
        )
        curves.append((h_vec, ref_times, ref_curves, rom_curves))
        print(
            f"  holdout h={tuple(round(x,2) for x in h_vec)}: max err {max(per_point):.4f}%"
        )

    errors = np.asarray([s["max_err_pct"] for s in scenario_results])
    print(
        f"holdout max {errors.max():.4f}% mean {errors.mean():.4f}% std {errors.std():.4f}%"
    )

    plot_results(cfg, summary, basis, scenario_results, curves, plot_dir)
    return summary, scenario_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    cfg = (
        replace(PkgConfig(), max_xy_cell_mm=2.0, duration_s=300.0, dt_s=30.0)
        if args.quick
        else PkgConfig()
    )
    t0 = time.perf_counter()
    run(cfg, OUT_DIR, strict=args.strict)
    print(f"total {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
