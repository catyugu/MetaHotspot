#!/usr/bin/env python3
r"""E5 — Nonconforming-grid EROM coupling on the Case-1 physical stack.

The BCI/FloTHERM EROM contract (Codecasa et al., "Connecting MOR-based BCI
compact thermal models", THERMINIC 2017, Sec. 5) couples nonconforming
Cartesian grids through every non-empty face intersection (common-patch set
F_if) with area fractions xi and binary incidences E.  This experiment
validates that machinery on a physically layered stack (4 dies /
interconnect / E-10 over FR4) with the interface at z = 10 mm — the die side
on a fine XY grid, the FR4 side on a different (nonconforming) XY grid.

Three couplings solve the SAME physical scenario:
  (1) fine/detailed  + fine/detailed    — conforming identity baseline
  (2) fine/detailed  + coarse/detailed  — nonconforming detailed/detailed
                                          (algebraic reference for (3))
  (3) fine EROM      + coarse/detailed  — nonconforming ROM/FVM (the product)

The monolithic fine-model solve is the physical reference; (2) vs (1) shows
the pure nonconforming-grid discretization cost; (3) vs (2) isolates the ROM
error at a nonconforming attachment (same patches, same interface nodes).

Reported per case (separate observables, never inferred from closure alone):
patch count, area conservation, xi ranges, matrix symmetry/PD, junction
error, interface-trace error, interface flux balance, side-field errors.

Outputs: results/optimization/nonconforming_erom.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[2]  # playground/bci_rom_testcase1
ROOT = CASE.parents[1]
sys.path[:0] = [str(CASE)]

from model_case1 import Case1Config, Case1Model  # noqa: E402
from metahotspot.macromodel import embeddable as er  # noqa: E402

OUT = ROOT / "playground" / "bci_rom_testcase1" / "results" / "optimization"

AMBIENT = 308.15
BOUNDARY_H = (5.0e1, 1.0e3)  # physical HTC (die crowns, FR4 bottom)
POWER_W = np.array([0.1, 0.2, 0.3, 0.4])
INTERFACE_Z = 10.0e-3  # E-10 (upper) / FR4 (lower) boundary
DT_S = 5.0
DURATION_S = 200.0


def build_model(xy_mm, z_mm=2.5):
    return Case1Model(
        Case1Config(
            max_xy_cell_mm=xy_mm,
            max_z_cell_mm=z_mm,
            dt_s=DT_S,
            duration_s=DURATION_S,
        )
    )


def side_split(model, keep_upper):
    z = model.cell_layout.centers[:, 2]
    return np.flatnonzero(
        z >= INTERFACE_Z - 1.0e-12 if keep_upper else z < INTERFACE_Z - 1.0e-12
    )


def junction_rise_coupled(state, side, offset):
    """Source-port temperature rise (K) of a coupled side (both side types)."""
    q = np.asarray(state[offset : offset + side.dof_order])
    if hasattr(side, "source_matrix"):
        return np.asarray(side.source_matrix.T @ q).ravel()
    return np.asarray(side.F_hat.T @ q).ravel()


def sample_ref_interface_history(
    ref_history_rise, model, patch_centres, times, step_times
):
    """Monolithic *transient* rise at patch centres, conductance-weighted in z.

    ref_history_rise: (n_ref_times, n_cells) rise history of the monolithic
    solve; times: its time grid; step_times: the coupled output grid.  For
    each coupled output time, interpolate the two adjacent cell rows in time,
    then combine them conductance-weighted at the plane (see
    sample_ref_interface).
    """
    c = model.cell_layout.centers
    hi = np.flatnonzero((c[:, 2] > INTERFACE_Z) & (c[:, 2] < INTERFACE_Z + 2.0e-3))
    lo = np.flatnonzero((c[:, 2] < INTERFACE_Z) & (c[:, 2] > INTERFACE_Z - 2.0e-3))
    half = model.cell_layout.half_sizes
    k = model.cell_layout.conductivity
    g_hi = k[hi, 2] / half[hi, 2]
    g_lo = k[lo, 2] / half[lo, 2]
    # per-patch nearest cells
    ih = np.empty(patch_centres.shape[0], dtype=np.int64)
    il = np.empty(patch_centres.shape[0], dtype=np.int64)
    for i, (px, py) in enumerate(patch_centres):
        d2h = (c[hi, 0] - px) ** 2 + (c[hi, 1] - py) ** 2
        d2l = (c[lo, 0] - px) ** 2 + (c[lo, 1] - py) ** 2
        ih[i] = int(np.argmin(d2h))
        il[i] = int(np.argmin(d2l))
    w_hi = g_hi[ih] / (g_hi[ih] + g_lo[il])
    w_lo = g_lo[il] / (g_hi[ih] + g_lo[il])
    out = np.empty((step_times.size, patch_centres.shape[0]))
    for t, st in enumerate(step_times):
        # interp requires scalar fp per call; loop patches
        row = np.empty(patch_centres.shape[0])
        for i in range(patch_centres.shape[0]):
            r_hi = np.interp(st, times, ref_history_rise[:, hi[ih[i]]])
            r_lo = np.interp(st, times, ref_history_rise[:, lo[il[i]]])
            row[i] = w_hi[i] * r_hi + w_lo[i] * r_lo
        out[t] = row
    return out


def sample_ref_interface(ref_rise, model, patch_centres):
    """Monolithic rise at the interface plane, conductance-weighted in z.

    The interface plane at z=INTERFACE_Z sits between the lower cell row
    (z < INTERFACE_Z) and the upper cell row (z > INTERFACE_Z) of the
    reference model.  At a massless interface node the steady balance is

        g_up (T_up - T_plane) + g_dn (T_dn - T_plane) = 0
        T_plane = (g_up T_up + g_dn T_dn) / (g_up + g_dn)

    with each cell's own one-sided conductance to the plane g = k/half.
    This is what the independent interface node of the coupled solve measures;
    a naive row-centre mean, or a cross-weighted (swapped-g) combination, is
    off by up to ~1 K across the material/conductivity contrast.
    """
    c = model.cell_layout.centers
    hi = np.flatnonzero((c[:, 2] > INTERFACE_Z) & (c[:, 2] < INTERFACE_Z + 2.0e-3))
    lo = np.flatnonzero((c[:, 2] < INTERFACE_Z) & (c[:, 2] > INTERFACE_Z - 2.0e-3))
    half = model.cell_layout.half_sizes
    k = model.cell_layout.conductivity
    g_hi = k[hi, 2] / half[hi, 2]
    g_lo = k[lo, 2] / half[lo, 2]
    out = np.empty(patch_centres.shape[0])
    for i, (px, py) in enumerate(patch_centres):
        d2h = (c[hi, 0] - px) ** 2 + (c[hi, 1] - py) ** 2
        d2l = (c[lo, 0] - px) ** 2 + (c[lo, 1] - py) ** 2
        ih = int(np.argmin(d2h))
        il = int(np.argmin(d2l))
        out[i] = (ref_rise[hi[ih]] * g_hi[ih] + ref_rise[lo[il]] * g_lo[il]) / (
            g_hi[ih] + g_lo[il]
        )
    return out


def patch_centres_from_ports(lport, rport, areas=None):
    """Centres of the common-patch set F_if (boundary-edge union grid).

    Mirrors er.common_patches: every non-empty intersection of a left face
    with a right face.  The patch centre is the midpoint of its rectangle.

    When ``areas`` is given, patches with (near-)zero area are numerical
    slivers produced by float-inexact duplicate edges between two grids that
    nominally share a coordinate (e.g. die edges of a conforming interface);
    they are filtered out so the returned centres match only physical patches.
    """
    rl = lport.rects
    rr = rport.rects
    x_edges = np.unique(np.r_[rl[:, (0, 1)], rr[:, (0, 1)]])
    y_edges = np.unique(np.r_[rl[:, (2, 3)], rr[:, (2, 3)]])
    centres = []
    for xl, xr in zip(x_edges[:-1], x_edges[1:]):
        for yl, yr in zip(y_edges[:-1], y_edges[1:]):
            if xr <= xl or yr <= yl:
                continue
            lm = np.flatnonzero(
                (rl[:, 0] <= xl + 1e-9)
                & (rl[:, 1] >= xr - 1e-9)
                & (rl[:, 2] <= yl + 1e-9)
                & (rl[:, 3] >= yr - 1e-9)
            )
            rm = np.flatnonzero(
                (rr[:, 0] <= xl + 1e-9)
                & (rr[:, 1] >= xr - 1e-9)
                & (rr[:, 2] <= yl + 1e-9)
                & (rr[:, 3] >= yr - 1e-9)
            )
            if lm.size and rm.size:
                centres.append((0.5 * (xl + xr), 0.5 * (yl + yr)))
    centres = np.asarray(centres, dtype=np.float64)
    if areas is not None:
        keep = np.asarray(areas) >= 1.0e-12
        if keep.size != centres.shape[0]:
            raise RuntimeError("patch-centre reconstruction count mismatch")
        centres = centres[keep]
    return centres


def solve_and_report(
    tag, left, right, lport, rport, ref_junction, ref_rise, ref_model, ref_full
):
    """Couple left/right through lport/rport; report all observables.

    ``ref_full`` is the monolithic AffineSolveResult (steady + transient used
    for the interface trace comparisons).  Returns the report dict plus the
    coupled steady state and block layout so the caller can compute fine-side
    field errors without re-solving.
    """
    t0 = time.perf_counter()
    areas, E_l, E_r, xi_l, xi_r, _li, _ri = er.common_patches(lport, rport)
    npatch = areas.size
    K, C, rhs, ldof, rdof, npatch2 = er.connect(
        left, right, lport, rport, power=POWER_W
    )
    assert npatch2 == npatch
    connect_s = time.perf_counter() - t0
    sym = float(np.max(np.abs(K - K.T)))
    eigmin = float(np.min(np.linalg.eigvalsh(K.toarray())))
    steady, history = er.solve_system(K, C, rhs, DT_S, DURATION_S)
    solve_s = time.perf_counter() - t0 - connect_s

    # junction error (all 4 ports live on the left/upper side)
    j = AMBIENT + junction_rise_coupled(steady, left, 0)
    junc_err = float(np.max(np.abs(j - AMBIENT - ref_junction)))

    # interface: node rises vs monolithic field at physical patch centres.
    # (Near-zero-area sliver patches from float-inexact duplicate edges carry
    # no conductance and meaningless nodes; exclude them from the comparison.)
    T_if_all = np.asarray(steady[ldof : ldof + npatch])
    phys = np.asarray(areas) >= 1.0e-12
    T_if = T_if_all[phys]
    centres = patch_centres_from_ports(lport, rport, areas=areas)
    assert centres.shape[0] == T_if.size, (centres.shape[0], T_if.size)
    ref_if = sample_ref_interface(ref_rise, ref_model, centres)
    iface_err = float(np.max(np.abs(T_if - ref_if)))

    # per-patch flux balance (physical patches)
    Vl, hl = left.interface_trace(lport, E_l, xi_l)
    Vr, hr = right.interface_trace(rport, E_r, xi_r)
    q_l = np.asarray(steady[:ldof])
    q_r = np.asarray(steady[ldof + npatch :])
    flux_l = float(np.sum(hl[phys] * (np.asarray(Vl @ q_l)[phys] - T_if)))
    flux_r = float(np.sum(hr[phys] * (np.asarray(Vr @ q_r)[phys] - T_if)))

    # fine-side (left) field error vs monolithic — left cells carry full-model
    # indices when left is a subdomain of the fine model or an EROM of it.
    left_full_cells = np.asarray(left.cells, dtype=np.int64)
    if hasattr(left, "basis"):
        left_rise = np.asarray(left.basis @ q_l)
    else:
        left_rise = q_l
    field_err = float(
        np.max(np.abs(left_rise - ref_rise[left_full_cells]))
        if left_full_cells.size
        else 0.0
    )

    # transient interface trace error (peak over time, physical patches) vs
    # the monolithic *transient* plane history, sampled at the coupled steps
    traj = np.asarray(history)  # (n_steps+1, total); row 0 = t=0
    T_if_traj = traj[:, ldof : ldof + npatch][:, phys]
    ref_hist_rise = np.asarray(ref_full.history) - AMBIENT  # (n_ref_times, cells)
    step_times = np.arange(traj.shape[0]) * DT_S
    ref_if_traj = sample_ref_interface_history(
        ref_hist_rise, ref_model, centres, np.asarray(ref_full.times), step_times
    )
    iface_traj_err = float(
        np.max(np.abs(T_if_traj - ref_if_traj)) if traj.shape[0] else 0.0
    )
    # final-time trace error (transient converged toward steady)
    iface_final_err = float(np.max(np.abs(T_if_traj[-1] - ref_if_traj[-1])))

    report = {
        "tag": tag,
        "interface_patches": int(npatch),
        "physical_interface_patches": int(phys.sum()),
        "sliver_patches": int((~phys).sum()),
        "patch_area_total_m2": float(areas.sum()),
        "left_face_area_m2": float(lport.areas.sum()),
        "right_face_area_m2": float(rport.areas.sum()),
        "left_fraction_max": float(xi_l.max()),
        "right_fraction_max": float(xi_r.max()),
        "matrix_symmetry": sym,
        "matrix_PD_min_eig": eigmin,
        "junction_max_err_K": junc_err,
        "junction_max_rel_err_pct": 100.0
        * junc_err
        / float(np.max(ref_junction) + 1e-12),
        "interface_trace_maxerr_K": iface_err,
        "interface_trace_traj_maxerr_K": iface_traj_err,
        "interface_trace_final_maxerr_K": iface_final_err,
        "interface_flux_l_W": flux_l,
        "interface_flux_r_W": flux_r,
        "interface_flux_balance_W": flux_l + flux_r,
        "upper_field_maxerr_K": field_err,
        "steady_max_rise_ref_K": float(ref_rise.max()),
        "n_transient_steps": int(history.shape[0]),
        "connect_s": round(connect_s, 4),
        "solve_s": round(solve_s, 2),
    }
    return report, steady, ldof, npatch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fine = build_model(2.5)  # die side + reference mesh
    coarse = build_model(4.0)  # FR4 side on a nonconforming XY grid
    print(f"fine cells={fine.full_cell_count}  coarse cells={coarse.full_cell_count}")

    t0 = time.perf_counter()
    full = fine.full_reference(BOUNDARY_H)
    ref_mono_s = time.perf_counter() - t0
    ref_rise = np.asarray(full.steady_temperature) - AMBIENT
    ref_junction = fine.junction_temperature(full.steady_temperature) - AMBIENT
    print(
        f"monolithic reference: {ref_mono_s:.1f}s  junc rise={np.round(ref_junction, 4)}"
    )

    fine_up = er.build_subdomain(
        fine, side_split(fine, True), name="fine_upper", physical_h=BOUNDARY_H
    )
    fine_lo = er.build_subdomain(
        fine, side_split(fine, False), name="fine_lower", physical_h=BOUNDARY_H
    )
    coarse_lo = er.build_subdomain(
        coarse, side_split(coarse, False), name="coarse_lower", physical_h=BOUNDARY_H
    )

    c1, _st1, _ld1, _np1 = solve_and_report(
        "conforming_dd",
        fine_up,
        fine_lo,
        fine_up.port("z-"),
        fine_lo.port("z+"),
        ref_junction,
        ref_rise,
        fine,
        full,
    )
    c2, _st2, _ld2, _np2 = solve_and_report(
        "nonconforming_dd",
        fine_up,
        coarse_lo,
        fine_up.port("z-"),
        coarse_lo.port("z+"),
        ref_junction,
        ref_rise,
        fine,
        full,
    )

    t0 = time.perf_counter()
    rom = er.extract_rom(
        fine_up, tolerance=1.0e-3, max_order=512, probe_rounds=2, seed=20260825
    )
    extract_s = time.perf_counter() - t0
    print(f"EROM extraction: order={rom.m}  {extract_s:.1f}s")

    c3, _st3, _ld3, _np3 = solve_and_report(
        "nonconforming_rom",
        rom,
        coarse_lo,
        rom.port("z-"),
        coarse_lo.port("z+"),
        ref_junction,
        ref_rise,
        fine,
        full,
    )

    payload = {
        "scenario": "Case-1 stack cut at z=10mm; crown h=50 / FR4 h=1000 W/m2K; "
        "P=[0.1,0.2,0.3,0.4]W; ambient 308.15K; dt=5s, 200s",
        "meshes": {
            "fine_xy_mm": 2.5,
            "coarse_xy_mm": 4.0,
            "fine_cells": int(fine.full_cell_count),
            "coarse_cells": int(coarse.full_cell_count),
            "interface_z_m": INTERFACE_Z,
            "z_mm": 2.5,
        },
        "monolithic_reference": {
            "steady_s": round(ref_mono_s, 2),
            "junction_rise_K": ref_junction.tolist(),
            "max_rise_K": float(ref_rise.max()),
        },
        "rom_extraction": {
            "basis_order": int(rom.m),
            "extract_s": round(extract_s, 2),
            "relative_response_error": float(
                rom.summary.get("relative_response_error", 0.0)
            ),
        },
        "cases": {c["tag"]: c for c in (c1, c2, c3)},
    }
    (OUT / "nonconforming_erom.json").write_text(
        json.dumps(payload, indent=1, default=float), encoding="utf-8"
    )
    print(json.dumps(payload, indent=1, default=float))


if __name__ == "__main__":
    main()
