#!/usr/bin/env python3
"""One representative mismatched interface, six panels.

Case: the lid (0.75 mm in-plane cells, shifted by 1/8 mm off the die's grid lines) bonded
to the die's 10 x 10 mm top face, while the die itself is finely meshed (0.25 mm).  The
die side therefore resolves the interface field, and what is left to see is the interface
scheme itself.

    python showcase.py

Writes ``results/figures/showcase_flux_temperature.png`` (2 rows x 3 columns):
    top     reference heat flux | corrected error | not-corrected error
    bottom  reference temperature | corrected error | not-corrected error
and ``results/vtu/interface_showcase.vtu`` plus ``results/showcase.json`` with the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import model as mdl
import run_study as rs
from nc_interface import FvmSide, InterfaceProblem, assemble, common_patches_with_rects, window_port

MM = 1.0e-3
OUT = Path(__file__).resolve().parent / "results"
FIGURES = OUT / "figures"
VTU = OUT / "vtu"

DIE_H = 0.25 * MM  # fine component mesh
LID_H = 0.75 * MM  # coarse attachment mesh
LID_SHIFT = 0.125 * MM  # half a die cell, so no lid line meets a die line
REF_H = 0.0625 * MM
CONTACT_H = None  # ideal bond; set 2e5 for a 0.05 K cm^2/W bondline


def blocks(phys: mdl.Physics):
    """Component (base + die on a 0.25 mm grid) and shifted lid block."""
    xs = mdl.graded_axis(phys.base_xy / 2, DIE_H, phys.die_xy / 2)
    comp_grid, comp_active = mdl.grid_of_layers(
        "component", xs, xs, phys.component_layers(), z0=0.0
    )
    comp = assemble(
        comp_grid, active=comp_active, ambient=phys.component_ambient,
        sources=phys.sources(), name="component",
    )

    half = phys.lid_xy / 2
    count = int(round((phys.lid_xy - 2 * LID_SHIFT) / LID_H))
    edges = np.r_[half * -1.0,
                  -half + LID_SHIFT + LID_H * np.arange(count),
                  half]
    env_grid, env_active = mdl.grid_of_layers(
        "envelope", edges, edges, phys.envelope_layers(), z0=phys.interface_z
    )
    env = assemble(
        env_grid, active=env_active, ambient=phys.envelope_ambient, name="envelope"
    )

    die_half = phys.die_xy / 2
    window = (-die_half, die_half, -die_half, die_half)
    comp_port = window_port(comp, window, axis=2, direction=+1, label="z+")
    env_port = window_port(env, window, axis=2, direction=-1, label="z-")
    if CONTACT_H is not None:
        env_port = mdl.with_contact(env_port, CONTACT_H)
    from nc_interface import pad_sources

    pad_sources(comp, env)
    return comp, env, comp_port, env_port


def panel(axis, rects, values, title, cmap, vmin, vmax):
    boxes = PatchCollection(
        [
            Rectangle((x0 * 1e3, y0 * 1e3), (x1 - x0) * 1e3, (y1 - y0) * 1e3)
            for x0, x1, y0, y1 in rects
        ],
        cmap=cmap, norm=Normalize(vmin, vmax), edgecolor="none",
    )
    boxes.set_array(values)
    axis.add_collection(boxes)
    half = np.max(np.abs(np.r_[rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]]))
    axis.set_xlim(-half * 1e3, half * 1e3)
    axis.set_ylim(-half * 1e3, half * 1e3)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=8)
    axis.tick_params(labelsize=6)
    axis.set_xlabel("x [mm]", fontsize=7)
    return boxes


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    VTU.mkdir(parents=True, exist_ok=True)
    phys = mdl.Physics()

    comp, env, comp_port, env_port = blocks(phys)
    problem = InterfaceProblem(
        FvmSide(comp, {"z+": comp_port}),
        FvmSide(env, {"z-": env_port}),
        comp_port,
        env_port,
        power=rs.power_vector(phys),
    )
    patches = common_patches_with_rects(comp_port, env_port)
    rects = patches.rects
    area = patches.area

    reference_system, reference_field, ref_cg = rs.reference_pair(phys, REF_H, CONTACT_H)
    q_ref, t_ref = rs.reference_interface(
        reference_system, reference_field, phys, patches, CONTACT_H
    )

    repo = problem.solve(correctors=0)
    corrected = problem.solve(tol=1.0e-11)
    if corrected["steps"][-1].max_dT > 1.0e-9:
        raise RuntimeError("corrector loop did not converge")

    # flux densities: the patch flux divided by its area, comparable to the reference
    q_repo = problem.patch_flux(repo["state"], corrected=False) / area
    q_nc = problem.patch_flux(corrected["state"]) / area
    t_repo = np.asarray(repo["state"][problem.node_slice])
    t_nc = np.asarray(corrected["state"][problem.node_slice])

    def l2(values):
        return float(np.sqrt(np.sum(area * (values - q_ref) ** 2) / np.sum(area * q_ref**2)))

    def rms(values):
        return float(np.sqrt(np.sum(area * (values - t_ref) ** 2) / np.sum(area)))

    numbers = {
        "case": {
            "ambient_K": 0.0,
            "temperature_convention": "all temperatures are rises above the ambient, which is 0 K",
            "absolute_K_at_298K_ambient": [
                298.15 + float(t_ref.min()),
                298.15 + float(t_ref.max()),
            ],
            "die_h_mm": DIE_H / MM,
            "lid_h_mm": LID_H / MM,
            "lid_shift_mm": LID_SHIFT / MM,
            "patches": int(patches.count),
            "partial_patches": int(
                np.sum(patches.xi_l < 1 - 1e-12) + np.sum(patches.xi_r < 1 - 1e-12)
            ),
            "reference_h_mm": REF_H / MM,
            "reference_cells": int(reference_system.n),
            "reference_cg_iterations": int(ref_cg),
            "delta_max_mm": float(np.abs(np.r_[problem.delta_l, problem.delta_r]).max() / MM),
            "skew_over_half_max": float(
                np.max(
                    np.abs(
                        np.r_[
                            problem.delta_l
                            / comp.halves()[comp_port.cells][:, 2][patches.li][:, None],
                            problem.delta_r
                            / env.halves()[env_port.cells][:, 2][patches.ri][:, None],
                        ]
                    )
                )
            ),
            "correctors": int(corrected["correctors"]),
        },
        "flux": {
            "reference_max_kW_m2": float(q_ref.max() / 1e3),
            "repo_l2": l2(q_repo),
            "corrected_l2": l2(q_nc),
            "repo_max_error_kW_m2": float(np.abs(q_repo - q_ref).max() / 1e3),
            "corrected_max_error_kW_m2": float(np.abs(q_nc - q_ref).max() / 1e3),
        },
        "temperature": {
            "reference_min_max_K": [float(t_ref.min()), float(t_ref.max())],
            "repo_rms_mK": 1e3 * rms(t_repo),
            "corrected_rms_mK": 1e3 * rms(t_nc),
            "repo_max_mK": 1e3 * float(np.abs(t_repo - t_ref).max()),
            "corrected_max_mK": 1e3 * float(np.abs(t_nc - t_ref).max()),
        },
    }
    (OUT / "showcase.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- figure
    q_ref_kw, q_repo_err, q_nc_err = q_ref / 1e3, (q_repo - q_ref) / 1e3, (q_nc - q_ref) / 1e3
    t_ref_mk, t_repo_err, t_nc_err = t_ref, 1e3 * (t_repo - t_ref), 1e3 * (t_nc - t_ref)
    flux_err = float(max(np.abs(q_repo_err).max(), np.abs(q_nc_err).max()))
    temp_err = float(max(np.abs(t_repo_err).max(), np.abs(t_nc_err).max()))

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.6), constrained_layout=True)
    layout = (
        (axes[0, 0], q_ref_kw, f"reference heat flux [kW/m$^2$]",
         "inferno", 0.0, float(q_ref_kw.max())),
        (axes[0, 1], q_nc_err,
         f"corrected error  (L2 {100*numbers['flux']['corrected_l2']:.1f}%, "
         f"max {numbers['flux']['corrected_max_error_kW_m2']:.0f} kW/m$^2$)",
         "coolwarm", -flux_err, flux_err),
        (axes[0, 2], q_repo_err,
         f"not corrected error  (L2 {100*numbers['flux']['repo_l2']:.1f}%, "
         f"max {numbers['flux']['repo_max_error_kW_m2']:.0f} kW/m$^2$)",
         "coolwarm", -flux_err, flux_err),
        (axes[1, 0], t_ref_mk, "reference interface temperature rise (ambient = 0) [K]",
         "viridis", float(t_ref_mk.min()), float(t_ref_mk.max())),
        (axes[1, 1], t_nc_err,
         f"corrected error  (RMS {numbers['temperature']['corrected_rms_mK']:.1f} mK, "
         f"max {numbers['temperature']['corrected_max_mK']:.0f} mK)",
         "coolwarm", -temp_err, temp_err),
        (axes[1, 2], t_repo_err,
         f"not corrected error  (RMS {numbers['temperature']['repo_rms_mK']:.1f} mK, "
         f"max {numbers['temperature']['repo_max_mK']:.0f} mK)",
         "coolwarm", -temp_err, temp_err),
    )
    for axis, values, title, cmap, vmin, vmax in layout:
        image = panel(axis, rects, values, title, cmap, vmin, vmax)
        unit = "kW/m$^2$" if axis in (axes[0, :]).tolist() else (
            "K" if axis is axes[1, 0] else "mK")
        fig.colorbar(image, ax=axis, shrink=0.86, pad=0.02, label=unit)
    fig.suptitle(
        f"die {DIE_H/MM:g} mm / lid {LID_H/MM:g} mm shifted {LID_SHIFT/MM:g} mm, "
        f"{patches.count} patches, reference {REF_H/MM:g} mm "
        f"({reference_system.n} cells); {corrected['correctors']} corrector steps",
        fontsize=10,
    )
    fig.savefig(FIGURES / "showcase_flux_temperature.png", dpi=170)
    plt.close(fig)

    # ---------------------------------------------------------------- VTU
    quads = np.empty((4 * patches.count, 3))
    for i, (x0, x1, y0, y1) in enumerate(rects):
        quads[4 * i : 4 * i + 4] = [
            (x0, y0, phys.interface_z), (x1, y0, phys.interface_z),
            (x1, y1, phys.interface_z), (x0, y1, phys.interface_z),
        ]
    fields = {
        "q_reference_W_m2": q_ref,
        "q_corrected_W_m2": q_nc,
        "q_repo_W_m2": q_repo,
        "dq_corrected_W_m2": q_nc - q_ref,
        "dq_repo_W_m2": q_repo - q_ref,

        "T_rise_reference_K": t_ref,
        "T_rise_corrected_K": t_nc,
        "T_rise_repo_K": t_repo,
        "delta_l_m": np.linalg.norm(problem.delta_l, axis=1),
        "delta_r_m": np.linalg.norm(problem.delta_r, axis=1),
        "area_m2": area,
    }
    _write_quads(VTU / "interface_showcase.vtu", quads, fields)

    # ------------------------------------------------- three-dimensional blocks
    die_points, lid_points = comp.centers(), env.centers()
    ref_die = rs.sample_cells(reference_system, reference_field, die_points)
    ref_lid = rs.sample_cells(reference_system, reference_field, lid_points)
    repo_die = np.asarray(repo["state"][: problem.ldof])
    repo_lid = np.asarray(repo["state"][problem.right_slice])
    nc_die = np.asarray(corrected["state"][: problem.ldof])
    nc_lid = np.asarray(corrected["state"][problem.right_slice])
    rs.write_fields_vtu(
        VTU / "fields_showcase.vtu",
        [comp, env],
        {
            "T_rise_reference_K": [ref_die, ref_lid],
            "T_rise_repo_K": [repo_die, repo_lid],
            "T_rise_corrected_K": [nc_die, nc_lid],
            "dT_rise_repo_mK": [1e3 * (repo_die - ref_die), 1e3 * (repo_lid - ref_lid)],
            "dT_rise_corrected_mK": [1e3 * (nc_die - ref_die), 1e3 * (nc_lid - ref_lid)],
            "k_W_mK": [comp.conductivity(), env.conductivity()],
            "block": [np.zeros(comp.n), np.ones(env.n)],
        },
    )

    print(json.dumps(numbers, indent=2))
    print(f"\nwrote {FIGURES / 'showcase_flux_temperature.png'}")
    print(f"wrote {VTU / 'interface_showcase.vtu'}")
    print(f"wrote {VTU / 'fields_showcase.vtu'}")


def _write_quads(path: Path, points, fields) -> None:
    n_cells = points.shape[0] // 4
    connectivity = np.arange(4 * n_cells, dtype=np.int64)
    arrays = [
        f'<DataArray type="Float64" Name="{name}" format="ascii">'
        + " ".join(f"{v:.9e}" for v in np.asarray(data, dtype=np.float64).ravel())
        + "</DataArray>"
        for name, data in fields.items()
    ]
    Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">\n'
        '<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        + " ".join(f"{x:.9e} {y:.9e} {z:.9e}" for x, y, z in points)
        + '</DataArray></Points>\n'
        '<Cells><DataArray type="Int64" Name="connectivity" format="ascii">'
        + " ".join(str(int(v)) for v in connectivity)
        + '</DataArray><DataArray type="Int64" Name="offsets" format="ascii">'
        + " ".join(str(int(v)) for v in 4 * np.arange(1, n_cells + 1))
        + '</DataArray><DataArray type="UInt8" Name="types" format="ascii">'
        + " ".join(["9"] * n_cells)
        + "</DataArray></Cells>\n"
        f"<CellData>{''.join(arrays)}</CellData>\n"
        "</Piece></UnstructuredGrid></VTKFile>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
