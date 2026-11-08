#!/usr/bin/env python3
"""Demonstrate loading and validating the current simple_case1 FloTHERM EROM export.

This script is intentionally independent of historical case scripts.  It reconstructs the current ecxml model, reads the current binary
payload, and reports source scaling, closure, and detailed-vs-EROM errors.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import metahotspot
from metahotspot.enums import Face
from metahotspot.macromodel.geometry import CellGeometry
from metahotspot.macromodel.utils import normalized_operators

CASE = Path(__file__).resolve().parent
PAYLOAD = CASE / "simple_case1_EROM"
AMBIENT_K = 308.15
HTC = 1000.0


def matrix_file(name: str) -> np.ndarray:
    data = (PAYLOAD / name).read_bytes()
    rows, cols = struct.unpack_from("<2I", data)
    return (
        np.frombuffer(data, dtype="<f8", offset=8, count=rows * cols)
        .copy()
        .reshape(rows, cols)
    )


def vector_file(name: str) -> np.ndarray:
    data = (PAYLOAD / name).read_bytes()
    n = struct.unpack_from("<I", data)[0]
    return np.frombuffer(data, dtype="<f8", offset=4, count=n).copy()


def links() -> np.ndarray:
    data = (PAYLOAD / "XresLink").read_bytes()
    n = struct.unpack_from("<I", data)[0]
    return np.asarray(
        [struct.unpack_from("<5I6d", data, 4 + 68 * i) for i in range(n)],
        dtype=np.float64,
    )


def source_power() -> float:
    text = (PAYLOAD / "xData.txt").read_text(encoding="utf-8")
    match = re.search(r"SourceValues:\s*([0-9.eE+-]+)", text)
    if not match:
        raise RuntimeError("SourceValues missing from xData.txt")
    return float(match.group(1))


def current_grid_axis_mm() -> np.ndarray:
    """Return the fixed 15-cell FloTHERM mesh used by this demonstration."""
    return np.r_[
        np.linspace(0.0, 25.0, 5),
        np.linspace(25.0, 75.0, 8)[1:],
        np.linspace(75.0, 100.0, 5)[1:],
    ]


def build_current_detailed(x: np.ndarray, y: np.ndarray):
    root = ET.parse(CASE / "simple_case1.ecxml").getroot()
    solids = [e for e in root.find("geometry") if e.tag.endswith("solid3dBlock")]
    base = solids[0]
    source = next(e for e in root.find("geometry") if e.tag.endswith("sourceBlock"))
    material = root.find("materials")[0]
    name = material.findtext("name")
    k = material.find("thermalConductivity/isotropic/conductivity").text
    rho = material.findtext("density")
    c = material.findtext("specific_heat")

    def xyz(parent, tag):
        node = parent.find(tag)
        return tuple(float(node.findtext(a)) for a in "xyz")

    base_pos, base_size = xyz(base, "location"), xyz(base, "size")
    src_pos, src_size = xyz(source, "location"), xyz(source, "size")
    model = metahotspot.Model()
    model.add_fom_domain()
    grid_mm = current_grid_axis_mm()
    model.set_cartesian_mesh(grid_mm * 1e-3, grid_mm * 1e-3, grid_mm * 1e-3)
    material_id = model.add_material(name, k, k, k, rho, c)
    body = model.add_volume(material_id)
    model.add_box(body, base_pos, np.array(base_pos) + base_size)
    heat = 100.0 / np.prod(src_size)
    source = model.add_volume(material_id, model.add_source(f"{heat:.17g}"))
    model.add_box(source, src_pos, np.array(src_pos) + src_size)
    model.set_default_neumann("0")
    compiled = model.compile()
    return normalized_operators(*compiled.assemble()), compiled


def main():
    link = links()
    x = current_grid_axis_mm() / 1000.0
    y = current_grid_axis_mm() / 1000.0
    Vb = matrix_file("Vb")
    g = matrix_file("g_bci_hat")[:, 0]
    h = matrix_file("h_bci_hat")[:, 0]
    K = vector_file("K_bci_hat")
    M = vector_file("M_bci_hat")
    R = vector_file("Xresistances")
    area = (link[:, 6] - link[:, 5]) * (link[:, 8] - link[:, 7])
    ops, compiled = build_current_detailed(x, y)
    cells = compiled.fields
    geometry = CellGeometry(cells)
    top = geometry.surface(Face.ZP).cell_ids
    centers = geometry.centers
    top = top[np.lexsort((centers[top, 1], centers[top, 0]))]
    top_area = geometry.sizes[top, 0] * geometry.sizes[top, 1]
    source = ops.f.reshape(-1)
    k_top = compiled.eval_materials().conductivity_z[top]
    half = geometry.half_sizes[top, 2]
    p = k_top * HTC / (k_top + HTC * half)
    detailed = spla.spsolve(
        ops.K + sp.csc_matrix((p * top_area, (top, top)), shape=ops.K.shape), source
    )
    detailed_junction = float((source / source.sum()) @ detailed)
    if top.size != Vb.shape[0]:
        raise RuntimeError(
            f"current ecxml reconstruction has {top.size} top cells, export has {Vb.shape[0]} faces; mesh=({x.size - 1},{y.size - 1})"
        )
    closure = area / (R + 1.0 / HTC)
    Hrom = sp.csc_matrix(Vb.T @ (closure[:, None] * Vb))
    rom_K = sp.diags(K) + Hrom
    scale = source_power()
    # FloTHERM's xData SourceValues supplies the physical source power.  The
    # exported g-vector is the unit-power modal source shape.
    q = spla.spsolve(rom_K, scale * g)
    rom_source_rise = float(g @ q)
    rom_trace = Vb @ q
    source_error = rom_source_rise - detailed_junction
    trace_error = float(np.max(np.abs(rom_trace - detailed[top])))
    closure_error = float(np.max(np.abs(closure - p * top_area)))

    print("simple_case1 EROM validation")
    print(f"  source power                 {scale:.6g} W")
    print(
        f"  detailed vs ROM source rise  {detailed_junction:.6f} / {rom_source_rise:.6f} K"
    )
    print(
        f"  source rise error             {source_error:+.6f} K ({100.0 * source_error / detailed_junction:+.3f} %)"
    )
    print(f"  interface trace max error     {trace_error:.6f} K")
    print(f"  closure max error             {closure_error:.3e} W/K")


if __name__ == "__main__":
    main()
