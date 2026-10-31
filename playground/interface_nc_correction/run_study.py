#!/usr/bin/env python3
"""Measure the two interface schemes on the lid-on-die package.

For every interface-grid variant and both bondline conditions the script solves the same
physical problem three ways --- the repository patch condensation (``correctors = 0``), the
fully non-orthogonally corrected scheme (iterated to convergence) and a fine conforming
monolithic reference --- and records:

* the interface flux density distribution (area-weighted relative L2 and max),
* the interface plane temperature (RMS and max over patches),
* the die-mean and hotspot-mean temperature,
* the whole-field max and relative RMS error,
* the corrector count and the *cost* of the correction (solves, CG iterations, wall time),
* the error trajectory against the corrector index.

A separate refinement mode (``--refine``) repeats one mismatch family at three grid
resolutions to measure the order of the residual error.

    python run_study.py               # all variants, both bondlines
    python run_study.py --refine      # grid-refinement ladder only

Artifacts under ``results/``: ``summary.csv``, ``iterations.csv``, ``mesh_info.csv``,
``cost.csv``, ``refinement.csv``, ``reference_error.csv``, ``patches_*.csv``,
``report.json``, ``vtu/interface_*.vtu`` and ``vtu/fields_*.vtu``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyamg
import scipy.sparse.linalg as spla

import model as mdl
from nc_interface import FvmSide, InterfaceProblem

OUT = Path(__file__).resolve().parent / "results"
VTU = OUT / "vtu"

REF_H = 0.0625e-3
REF_COARSE_H = 0.125e-3
BONDLINES = (("ideal", None), ("tim_0p05", 2.0e5))
PRIMARY = "die0p5_lid1p0_aligned"
FIELD_VTU_VARIANTS = (PRIMARY, "die2p0_lid0p25_fine")


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------


def solve_steady(system, power) -> tuple[np.ndarray, int]:
    """Conforming solve by AMG-preconditioned CG; returns (field, cg iterations)."""
    A = system.K.tocsr()
    hierarchy = pyamg.ruge_stuben_solver(A, interpolation="direct")
    counter = [0]

    def count(_xk):
        counter[0] += 1

    x, info = spla.cg(
        A,
        system.source @ np.asarray(power, dtype=np.float64),
        rtol=1.0e-12,
        atol=0.0,
        maxiter=5000,
        M=hierarchy.aspreconditioner(cycle="V"),
        callback=count,
    )
    if info != 0:
        raise RuntimeError(f"reference solve failed: info={info}")
    return np.asarray(x, dtype=np.float64), counter[0]


def power_vector(phys: mdl.Physics) -> np.ndarray:
    return np.array([phys.hotspot_W, phys.background_W])


def sample_cells(system, values, points) -> np.ndarray:
    grid = system.grid
    out = np.full(points.shape[0], np.nan)
    for i, p in enumerate(points):
        ix = int(np.clip(np.searchsorted(grid.x, p[0]) - 1, 0, system.grid.n[0] - 1))
        iy = int(np.clip(np.searchsorted(grid.y, p[1]) - 1, 0, system.grid.n[1] - 1))
        iz = int(np.clip(np.searchsorted(grid.z, p[2]) - 1, 0, system.grid.n[2] - 1))
        q = system.ids[ix, iy, iz]
        if q >= 0:
            out[i] = values[q]
    return out


# ---------------------------------------------------------------------------
# interface sampling on the reference
# ---------------------------------------------------------------------------


def reference_interface(system, values, phys, patches, contact_h):
    """Patch-averaged reference flux density and interface plane temperature.

    The comparison quantity of a discrete patch is the area average over the patch (the
    flux under the hotspot varies by two orders of magnitude across one coarse patch).
    The plane temperature is the conductance-weighted combination of the two adjacent
    rows with the bondline film on the attachment (above) side, matching the two-block
    coupling.
    """
    n_below = phys.component_z_cells
    grid = system.grid
    halves = system.halves()
    k = system.conductivity()
    spacing = grid.spacings()
    ok = (system.ids[:, :, n_below - 1] >= 0) & (system.ids[:, :, n_below] >= 0)
    below = np.where(ok, system.ids[:, :, n_below - 1], 0)
    above = np.where(ok, system.ids[:, :, n_below], 0)
    film = 0.0 if contact_h is None else 1.0 / contact_h
    g_below = k[below] / halves[below, 2]
    g_above = k[above] / (halves[above, 2] + k[above] * film)
    # positive = upward, i.e. from the component (below) into the lid (above)
    flux = (values[below] - values[above]) / (
        halves[below, 2] / k[below] + film + halves[above, 2] / k[above]
    )
    plane = (g_below * values[below] + g_above * values[above]) / (g_below + g_above)

    ix, iy = np.meshgrid(np.arange(grid.n[0]), np.arange(grid.n[1]), indexing="ij")
    xc = 0.5 * (grid.x[:-1] + grid.x[1:])
    yc = 0.5 * (grid.y[:-1] + grid.y[1:])
    fx = xc[ix].ravel()[ok.ravel()]
    fy = yc[iy].ravel()[ok.ravel()]
    fine_area = (spacing[0][ix] * spacing[1][iy]).ravel()[ok.ravel()]
    qf, pf = flux.ravel()[ok.ravel()], plane.ravel()[ok.ravel()]

    n = patches.count
    out_flux = np.empty(n)
    out_plane = np.empty(n)
    rects = patches.rects
    for p in range(n):
        inside = (
            (fx >= rects[p, 0])
            & (fx <= rects[p, 1])
            & (fy >= rects[p, 2])
            & (fy <= rects[p, 3])
        )
        weight = fine_area[inside]
        total = float(np.sum(weight))
        if total > 0.0:
            out_flux[p] = float(np.sum(weight * qf[inside]) / total)
            out_plane[p] = float(np.sum(weight * pf[inside]) / total)
        else:
            centre = 0.5 * np.array(
                [rects[p, 0] + rects[p, 1], rects[p, 2] + rects[p, 3]]
            )
            nearest = int(np.argmin((fx - centre[0]) ** 2 + (fy - centre[1]) ** 2))
            out_flux[p] = qf[nearest]
            out_plane[p] = pf[nearest]
    return out_flux, out_plane


def reference_pair(phys, h_xy, contact_h):
    system = mdl.build_reference(phys, h_xy)
    mdl.apply_contact_film(system, phys, contact_h)
    field, iterations = solve_steady(system, power_vector(phys))
    return system, field, iterations


def reference_payload(phys, reference, comp, env) -> dict:
    """Sample the reference solution onto one variant's blocks."""
    system, field, iterations = reference
    die_points = comp.centers()
    lid_points = env.centers()
    die_ref = sample_cells(system, field, die_points)
    lid_ref = sample_cells(system, field, lid_points)
    weight = phys.hotspot_W * comp.source[:, 0] + phys.background_W * comp.source[:, 1]
    weight = weight / weight.sum()
    hotspot = comp.source[:, 0]
    return {
        "system": system,
        "field": field,
        "cg_iterations": iterations,
        "die_ref": die_ref,
        "lid_ref": lid_ref,
        "volume": np.r_[comp.volumes(), env.volumes()],
        "reference": np.r_[die_ref, lid_ref],
        "die_weight": weight,
        "hotspot_weight": hotspot,
        "t_die": float(weight @ die_ref),
        "t_hotspot": float(hotspot @ die_ref),
    }


def reference_discretisation_error(phys, contact_h) -> dict:
    """Two-grid estimate of the reference's own error."""
    fine, t_fine, it_fine = reference_pair(phys, REF_H, contact_h)
    coarse, t_coarse, it_coarse = reference_pair(phys, REF_COARSE_H, contact_h)
    comp, _env, _dp, _sp = mdl.build_blocks(phys, "die0p5_lid0p5_same")
    die_points = comp.centers()
    weight = phys.hotspot_W * comp.source[:, 0] + phys.background_W * comp.source[:, 1]
    weight = weight / weight.sum()

    def t_die(system, values):
        return float(weight @ sample_cells(system, values, die_points))

    diff = np.abs(sample_cells(fine, t_fine, coarse.centers()) - t_coarse)
    return {
        "reference_h_m": REF_H,
        "coarse_h_m": REF_COARSE_H,
        "reference_cells": int(fine.n),
        "coarse_cells": int(coarse.n),
        "estimated_reference_error_K": float(np.nanmax(diff)),
        "estimated_reference_die_error_K": abs(t_die(fine, t_fine) - t_die(coarse, t_coarse)),
        "max_rise_K": float(t_fine.max()),
        "reference_cg_iterations": int(it_fine),
        "coarse_cg_iterations": int(it_coarse),
    }


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------


def _blocked_groups(problem, block: int = 4):
    cents = problem.patches.centroids()
    ux = np.unique(cents[:, 0])
    uy = np.unique(cents[:, 1])
    if ux.size * uy.size != cents.shape[0]:
        return None
    ix = np.searchsorted(ux, cents[:, 0]) // block
    iy = np.searchsorted(uy, cents[:, 1]) // block
    _, group = np.unique(np.column_stack((ix, iy)), axis=0, return_inverse=True)
    return group.ravel()


def blocked_flux_error(problem, qdens, q_ref, area, block: int = 4) -> float:
    """Relative L2 of the patch flux density averaged onto a 4x4 coarser grid.

    The raw per-patch flux under a discontinuous source carries a corner singularity whose
    amplitude is set by the mesh, not by the interface scheme; the blocked metric measures
    the smooth part a coupling actually transports.
    """
    group = _blocked_groups(problem, block)
    if group is None:
        return float("nan")
    weight = np.bincount(group, weights=area)
    a_mean = np.bincount(group, weights=area * qdens) / weight
    r_mean = np.bincount(group, weights=area * q_ref) / weight
    return float(np.sqrt(np.sum(weight * (a_mean - r_mean) ** 2) / np.sum(weight * r_mean**2)))


def observables(problem, state, ref) -> dict:
    node = np.asarray(state[problem.node_slice])
    area = problem.patches.area
    q = problem.patch_flux(state)
    qdens = q / area
    skew = problem._skew(state)
    q_ref, t_ref = ref["flux"], ref["plane"]
    num = float(np.sum(area * (qdens - q_ref) ** 2))
    den = float(np.sum(area * q_ref**2))

    die_state = np.asarray(state[: problem.ldof])
    lid_state = np.asarray(state[problem.right_slice])
    err = np.abs(np.r_[die_state - ref["die_ref"], lid_state - ref["lid_ref"]])
    t_die = float(ref["die_weight"] @ die_state)
    t_hot = float(ref["hotspot_weight"] @ die_state)
    return {
        "if_q_l2_rel": float(np.sqrt(num / den)),
        "if_q_l2_rel_blocked": blocked_flux_error(problem, qdens, q_ref, area),
        "if_q_max_rel": float(np.max(np.abs(qdens - q_ref)) / np.max(np.abs(q_ref))),
        "if_T_rms_K": float(np.sqrt(np.sum(area * (node - t_ref) ** 2) / np.sum(area))),
        "if_T_max_K": float(np.max(np.abs(node - t_ref))),
        "t_die_K": t_die,
        "t_die_err_mK": 1e3 * (t_die - ref["t_die"]),
        "t_die_pct": 100.0 * (t_die - ref["t_die"]) / ref["t_die"],
        "t_hotspot_K": t_hot,
        "t_hotspot_err_mK": 1e3 * (t_hot - ref["t_hotspot"]),
        "t_hotspot_pct": 100.0 * (t_hot - ref["t_hotspot"]) / ref["t_hotspot"],
        "field_max_K": float(err.max()),
        "field_rms_rel": float(
            np.sqrt(np.sum(ref["volume"] * err**2) / np.sum(ref["volume"] * ref["reference"] ** 2))
        ),
        "net_flux_W": float(q.sum()),
        "skew_flux_max_W": float(np.max(np.abs(problem.interface_conductance_series() * skew))),
    }


# ---------------------------------------------------------------------------
# one (bondline, variant) case
# ---------------------------------------------------------------------------


def timed_solve(problem, **kwargs):
    before = (problem._solver.solves, problem._solver.iterations)
    start = time.perf_counter()
    result = problem.solve(**kwargs)
    cost = {
        "solves": problem._solver.solves - before[0],
        "cg_iterations": problem._solver.iterations - before[1],
        "wall_s": time.perf_counter() - start,
    }
    return result, cost


def run_variant(phys, variant, contact_h, correctors=14, reference=None):
    comp, env, comp_port, env_port = mdl.build_blocks(phys, variant, contact_h=contact_h)
    start = time.perf_counter()
    problem = InterfaceProblem(
        FvmSide(comp, {"z+": comp_port}),
        FvmSide(env, {"z-": env_port}),
        comp_port,
        env_port,
        power=power_vector(phys),
    )
    setup_s = time.perf_counter() - start

    if reference is None:
        reference = reference_pair(phys, REF_H, contact_h)
    ref = reference_payload(phys, reference, comp, env)
    ref["flux"], ref["plane"] = reference_interface(
        ref["system"], ref["field"], phys, problem.patches, contact_h
    )

    repo, cost_repo = timed_solve(problem, correctors=0)
    trajectory, _ = timed_solve(problem, correctors=correctors)
    converged, cost_corrected = timed_solve(problem, tol=1.0e-11)
    relaxation = 1.0
    if converged["steps"][-1].max_dT > 1.0e-9:
        relaxation = 0.5
        converged, cost_corrected = timed_solve(problem, tol=1.0e-11,
                                                max_correctors=120,
                                                relaxation=relaxation)
        if converged["steps"][-1].max_dT > 1.0e-9:
            raise RuntimeError(
                f"{variant}: corrector loop did not converge even with relaxation"
            )

    rows = []
    for index, state in enumerate(trajectory["states"]):
        row = observables(problem, state, ref)
        row["corrector"] = index
        rows.append(row)
    repo_row = observables(problem, repo["state"], ref)
    corrected_row = observables(problem, converged["state"], ref)

    q_repo = problem.patch_flux(repo["state"], corrected=False)
    q_nc = problem.patch_flux(converged["state"])
    node_repo = np.asarray(repo["state"][problem.node_slice])
    node_nc = np.asarray(converged["state"][problem.node_slice])
    area = problem.patches.area
    q_scale = float(np.sqrt(np.sum(area * (ref["flux"] * area) ** 2)))
    scheme = {
        "dq_l2_rel": float(np.sqrt(np.sum(area * (q_repo - q_nc) ** 2)) / max(q_scale, 1e-300)),
        "dq_max_W": float(np.max(np.abs(q_repo - q_nc))),
        "dT_rms_mK": float(
            1e3 * np.sqrt(np.sum(area * (node_repo - node_nc) ** 2) / np.sum(area))
        ),
        "dT_max_mK": float(1e3 * np.max(np.abs(node_repo - node_nc))),
        "dt_die_mK": float(1e3 * (repo_row["t_die_K"] - corrected_row["t_die_K"])),
        "dflux_total_mW": float(1e3 * (q_repo.sum() - q_nc.sum())),
    }

    meta = {
        "variant": variant,
        "bondline": "ideal" if contact_h is None else f"{contact_h:.3e}",
        "patches": int(problem.n_patch),
        "component_faces": int(comp_port.cells.size),
        "envelope_faces": int(env_port.cells.size),
        "component_cells": int(comp.n),
        "envelope_cells": int(env.n),
        "coupled_size": int(problem.K.shape[0]),
        "component_h_mm": float(mdl.VARIANTS[variant][0] * 1e3),
        "correctors_to_1e-11": int(converged["correctors"]),
        "relaxation": relaxation,
        "skew_max_mm": float(np.abs(np.r_[problem.delta_l, problem.delta_r]).max() * 1e3),
        "skew_over_half_max": float(
            np.max(
                np.abs(
                    np.r_[
                        problem.delta_l
                        / comp.halves()[comp_port.cells][:, 2][problem.patches.li][:, None],
                        problem.delta_r
                        / env.halves()[env_port.cells][:, 2][problem.patches.ri][:, None],
                    ]
                )
            )
        ),
        "partial_patches": int(
            np.sum(problem.patches.xi_r < 1.0 - 1e-12)
            + np.sum(problem.patches.xi_l < 1.0 - 1e-12)
        ),
        "setup_s": setup_s,
        "repo": repo_row,
        "corrected": corrected_row,
        "cost_repo": cost_repo,
        "cost_corrected": cost_corrected,
        "scheme": scheme,
    }
    return meta, rows, {
        "problem": problem, "trajectory": trajectory, "repo": repo,
        "converged": converged, "comp": comp, "env": env, "ref": ref,
    }


# ---------------------------------------------------------------------------
# VTU writers
# ---------------------------------------------------------------------------


def write_interface_vtu(path: Path, problem, ref, repo_state, nc_state) -> None:
    """The common-patch surface as a quad mesh with flux and temperature fields."""
    rects = problem.patches.rects
    n = problem.n_patch
    z0 = float(problem.interface_z)
    points = np.empty((4 * n, 3))
    for i, (x0, x1, y0, y1) in enumerate(rects):
        points[4 * i : 4 * i + 4] = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    connectivity = np.arange(4 * n, dtype=np.int64).reshape(-1, 4)
    area = problem.patches.area
    q_repo = problem.patch_flux(repo_state, corrected=False)
    q_nc = problem.patch_flux(nc_state)
    q_ref = ref["flux"] * area
    fields = {
        "q_repo_W_m2": q_repo / area,
        "q_corrected_W_m2": q_nc / area,
        "q_reference_W_m2": q_ref / area,
        "dq_repo_W_m2": (q_repo - q_ref) / area,
        "dq_corrected_W_m2": (q_nc - q_ref) / area,
        "T_rise_repo_K": np.asarray(repo_state[problem.node_slice]),
        "T_rise_corrected_K": np.asarray(nc_state[problem.node_slice]),
        "T_rise_reference_K": ref["plane"],
        "delta_attachment_m": np.linalg.norm(problem.delta_r, axis=1),
        "xi_attachment": problem.patches.xi_r,
        "area_m2": area,
    }
    _write_quads(path, points, connectivity, fields)


def _write_quads(path: Path, points, connectivity, fields) -> None:
    n_cells = connectivity.shape[0]
    flat = connectivity.ravel()
    offsets = 4 * np.arange(1, n_cells + 1)
    types = np.full(n_cells, 9, dtype=np.uint8)  # VTK_QUAD
    arrays = []
    for name, data in fields.items():
        arrays.append(
            f'<DataArray type="Float64" Name="{name}" format="ascii">'
            + " ".join(f"{v:.9e}" for v in np.asarray(data, dtype=np.float64).ravel())
            + "</DataArray>"
        )
    pts = " ".join(f"{x:.9e} {y:.9e} {z:.9e}" for x, y, z in points)
    Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">\n'
        '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        f"{pts}</DataArray></Points>\n"
        '<Cells><DataArray type="Int64" Name="connectivity" format="ascii">'
        + " ".join(str(int(v)) for v in flat)
        + '</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'
        + " ".join(str(int(v)) for v in offsets)
        + '</DataArray><DataArray type="UInt8" Name="types" format="ascii">'
        + " ".join(str(int(v)) for v in types)
        + "</DataArray></Cells>\n"
        f"<CellData>{''.join(arrays)}</CellData>\n"
        "</Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )


def write_fields_vtu(path: Path, blocks, fields) -> None:
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    points, cells = [], []
    offset = 0
    for system in blocks:
        ijk = system.cells_ijk
        grid = system.grid
        lo = np.column_stack((grid.x[ijk[:, 0]], grid.y[ijk[:, 1]], grid.z[ijk[:, 2]]))
        hi = np.column_stack(
            (grid.x[ijk[:, 0] + 1], grid.y[ijk[:, 1] + 1], grid.z[ijk[:, 2] + 1])
        )
        points.append(
            (lo[:, None, :] + corners[None, :, :] * (hi - lo)[:, None, :]).reshape(-1, 3)
        )
        cells.append(np.arange(8 * ijk.shape[0], dtype=np.int64).reshape(-1, 8) + offset)
        offset += 8 * ijk.shape[0]
    pts = np.concatenate(points)
    cls = np.concatenate(cells)
    arrays = []
    for name, per_block in fields.items():
        data = np.concatenate(per_block)
        arrays.append(
            f'<DataArray type="Float64" Name="{name}" format="ascii">'
            + " ".join(f"{v:.9e}" for v in data)
            + "</DataArray>"
        )
    Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{pts.shape[0]}" NumberOfCells="{cls.shape[0]}">\n'
        '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        + " ".join(f"{x:.9e} {y:.9e} {z:.9e}" for x, y, z in pts)
        + '</DataArray></Points>\n'
        '<Cells><DataArray type="Int64" Name="connectivity" format="ascii">'
        + " ".join(str(int(v)) for v in cls.ravel())
        + '</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'
        + " ".join(str(int(v)) for v in 8 * np.arange(1, cls.shape[0] + 1))
        + '</DataArray><DataArray type="UInt8" Name="types" format="ascii">'
        + " ".join(["12"] * cls.shape[0])
        + "</DataArray></Cells>\n"
        f"<CellData>{''.join(arrays)}</CellData>\n"
        "</Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# csv helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()}
            )


def export_patches(path: Path, problem, ref, repo_state, nc_state) -> None:
    cents = problem.patches.centroids()
    area = problem.patches.area
    q_repo = problem.patch_flux(repo_state, corrected=False)
    q_nc = problem.patch_flux(nc_state)
    node_repo = np.asarray(repo_state[problem.node_slice])
    node_nc = np.asarray(nc_state[problem.node_slice])
    skew = problem._skew(nc_state)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["x_m", "y_m", "area_m2", "q_repo_W", "q_nc_W", "q_ref_W",
             "T_rise_repo_K", "T_rise_corrected_K", "T_rise_reference_K", "skew_K", "delta_r_m"]
        )
        for i in range(problem.n_patch):
            writer.writerow(
                [
                    f"{cents[i, 0]:.10g}", f"{cents[i, 1]:.10g}", f"{area[i]:.10g}",
                    f"{q_repo[i]:.10g}", f"{q_nc[i]:.10g}",
                    f"{ref['flux'][i] * area[i]:.10g}",
                    f"{node_repo[i]:.10g}", f"{node_nc[i]:.10g}",
                    f"{ref['plane'][i]:.10g}", f"{skew[i]:.10g}",
                    f"{np.linalg.norm(problem.delta_r[i]):.10g}",
                ]
            )


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def run_study(phys: mdl.Physics, variants: list[str]) -> None:
    summary, iterations, meshes, costs, ref_rows = [], [], [], [], []
    for bondline, contact_h in BONDLINES:
        error = reference_discretisation_error(phys, contact_h)
        error["bondline"] = bondline
        ref_rows.append(error)
        print(
            f"[{bondline}] reference {error['reference_cells']} cells at {REF_H*1e3:.3f} mm, "
            f"own error {error['estimated_reference_error_K']*1e3:.1f} mK field / "
            f"{error['estimated_reference_die_error_K']*1e3:.2f} mK die "
            f"({error['reference_cg_iterations']} CG iterations)"
        )
        reference = reference_pair(phys, REF_H, contact_h)
        for variant in variants:
            meta, rows, extra = run_variant(phys, variant, contact_h, reference=reference)
            for scheme in ("repo", "corrected"):
                line = {"bondline": bondline, "variant": variant, "scheme": scheme}
                line.update(meta[scheme])
                summary.append(line)
                cost = meta["cost_repo"] if scheme == "repo" else meta["cost_corrected"]
                costs.append(
                    {
                        "bondline": bondline, "variant": variant, "scheme": scheme,
                        "patches": meta["patches"],
                        "correctors": meta["correctors_to_1e-11"],
                        "solves": cost["solves"], "cg_iterations": cost["cg_iterations"],
                        "wall_s": cost["wall_s"], "setup_s": meta["setup_s"],
                        "if_q_l2_rel": meta[scheme]["if_q_l2_rel"],
                        "if_T_rms_K": meta[scheme]["if_T_rms_K"],
                        "t_die_err_mK": meta[scheme]["t_die_err_mK"],
                    }
                )
            for row in rows:
                iterations.append({"bondline": bondline, "variant": variant, **row})
            meshes.append(
                {
                    "bondline": bondline, "variant": variant,
                    "patches": meta["patches"],
                    "component_cells": meta["component_cells"],
                    "envelope_cells": meta["envelope_cells"],
                    "coupled_size": meta["coupled_size"],
                    "partial_patches": meta["partial_patches"],
                    "skew_max_mm": meta["skew_max_mm"],
                    "skew_over_half_max": meta["skew_over_half_max"],
                    "correctors_to_1e-11": meta["correctors_to_1e-11"],
                    "relaxation": meta["relaxation"],
                    "scheme_dq_l2_rel": meta["scheme"]["dq_l2_rel"],
                    "scheme_dq_max_W": meta["scheme"]["dq_max_W"],
                    "scheme_dT_rms_mK": meta["scheme"]["dT_rms_mK"],
                    "scheme_dT_max_mK": meta["scheme"]["dT_max_mK"],
                    "scheme_dt_die_mK": meta["scheme"]["dt_die_mK"],
                    "scheme_dflux_total_mW": meta["scheme"]["dflux_total_mW"],
                    "setup_s": meta["setup_s"],
                }
            )

            problem = extra["problem"]
            problem.interface_z = phys.interface_z
            write_interface_vtu(
                VTU / f"interface_{bondline}_{variant}.vtu",
                problem, extra["ref"], extra["repo"]["state"], extra["converged"]["state"],
            )
            if bondline == "ideal" and variant in FIELD_VTU_VARIANTS:
                write_fields_vtu(
                    VTU / f"fields_{bondline}_{variant}.vtu",
                    [extra["comp"], extra["env"]],
                    {
                        "T_rise_reference": [extra["ref"]["die_ref"], extra["ref"]["lid_ref"]],
                        "T_rise_repo": [np.asarray(extra["repo"]["state"][: problem.ldof]),
                                   np.asarray(extra["repo"]["state"][problem.right_slice])],
                        "T_rise_corrected": [np.asarray(extra["converged"]["state"][: problem.ldof]),
                                        np.asarray(extra["converged"]["state"][problem.right_slice])],
                    },
                )
                export_patches(
                    OUT / f"patches_{bondline}_{variant}.csv", problem, extra["ref"],
                    extra["repo"]["state"], extra["converged"]["state"],
                )
            repo_obs, corr_obs = meta["repo"], meta["corrected"]
            print(
                f"  {variant:15s} patches {meta['patches']:5d} skew/half "
                f"{meta['skew_over_half_max']:6.2f} | repo flux "
                f"{repo_obs['if_q_l2_rel']*100:6.2f}% ifT {repo_obs['if_T_rms_K']*1e3:6.1f} mK "
                f"die {repo_obs['t_die_err_mK']:+7.1f} mK | corr flux "
                f"{corr_obs['if_q_l2_rel']*100:6.2f}% ifT {corr_obs['if_T_rms_K']*1e3:6.1f} mK "
                f"die {corr_obs['t_die_err_mK']:+7.1f} mK | "
                f"{meta['correctors_to_1e-11']} corr, "
                f"{meta['cost_corrected']['solves']} solves, "
                f"{meta['cost_corrected']['cg_iterations']} cg"
            )

    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "iterations.csv", iterations)
    write_csv(OUT / "mesh_info.csv", meshes)
    write_csv(OUT / "cost.csv", costs)
    write_csv(OUT / "reference_error.csv", ref_rows)
    write_report(phys, ref_rows, meshes)


REFINEMENT = (
    ("refine_1p00_mm", 1.00, "env_1p0_offset"),
    ("refine_0p50_mm", 0.50, "env_0p5_offset"),
    ("refine_0p25_mm", 0.25, "env_0p25_offset"),
)


def run_refinement(phys: mdl.Physics) -> None:
    """One mismatch family at three resolutions: measures the residual error order."""
    rows = []
    for name, h_mm, pattern in REFINEMENT:
        mdl.VARIANTS[name] = (h_mm * mdl.MM, pattern)
        meta, _rows, _extra = run_variant(phys, name, None)
        rows.append(
            {
                "family": name, "h_mm": h_mm, "patches": meta["patches"],
                "repo_flux_l2": meta["repo"]["if_q_l2_rel"],
                "corr_flux_l2": meta["corrected"]["if_q_l2_rel"],
                "repo_ifT_rms": meta["repo"]["if_T_rms_K"],
                "corr_ifT_rms": meta["corrected"]["if_T_rms_K"],
                "repo_die_mK": meta["repo"]["t_die_err_mK"],
                "corr_die_mK": meta["corrected"]["t_die_err_mK"],
                "correctors": meta["correctors_to_1e-11"],
                "corr_solves": meta["cost_corrected"]["solves"],
                "corr_cg": meta["cost_corrected"]["cg_iterations"],
            }
        )
        print(
            f"  h={h_mm:4.2f} mm patches {meta['patches']:5d} | repo flux "
            f"{meta['repo']['if_q_l2_rel']*100:7.3f}% | corr flux "
            f"{meta['corrected']['if_q_l2_rel']*100:7.3f}% | repo ifT "
            f"{meta['repo']['if_T_rms_K']*1e3:7.2f} mK | corr ifT "
            f"{meta['corrected']['if_T_rms_K']*1e3:7.2f} mK"
        )
    write_csv(OUT / "refinement.csv", rows)


def write_report(phys, ref_rows, meshes) -> None:
    (OUT / "report.json").write_text(
        json.dumps(
            {
                "physics": {
                    "base_mm": [phys.base_xy * 1e3, phys.base_xy * 1e3, phys.base_t * 1e3],
                    "die_mm": [phys.die_xy * 1e3, phys.die_xy * 1e3, phys.die_t * 1e3],
                    "lid_mm": [phys.lid_xy * 1e3, phys.lid_xy * 1e3, phys.lid_t * 1e3],
                    "interface_z_mm": phys.interface_z * 1e3,
                    "top_z_mm": phys.top_z * 1e3,
                    "k": {"base": phys.k_base, "die": phys.k_die, "lid": phys.k_lid},
                    "power_W": phys.power_W,
                    "hotspot_W": phys.hotspot_W,
                    "source_mm": phys.source_xy * 1e3,
                    "htc": {
                        "board": phys.h_board, "lid": phys.h_lid,
                        "side": phys.h_side, "cavity": phys.h_cavity,
                    },
                    "bondlines": {name: h for name, h in BONDLINES},
                },
                "reference": ref_rows,
                "meshes": meshes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    VTU.mkdir(parents=True, exist_ok=True)
    phys = mdl.Physics()
    if args.refine:
        run_refinement(phys)
        return
    variants = args.variants or (
        ["die0p5_lid0p5_same", PRIMARY, "die2p0_lid0p25_fine"]
        if args.quick else list(mdl.VARIANTS)
    )
    run_study(phys, variants)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
