#!/usr/bin/env python3
"""
train_macromodel.py — Train a POD-based SmartMacro model

Given a case XML with a SmartMacro block:
  1. Parse the global mesh and block geometry
  2. Identify interior vs interface cells for the smart block
  3. Build the local FVM stiffness matrix (same stencil as C++ Assembler)
  4. Partition into interior (i) and interface (f) DOFs
  5. Compute K_port = K_ff - K_fi * K_ii^(-1) * K_if  (Schur complement)
  6. Perform POD (eigendecomposition) on K_port → retain n_modes dominant modes
  7. Write trained model XML + binary with phi_basis, K_modal, f_modal

Usage:
  python scripts/train_macromodel.py cases/macromodel_tests/case1.xml \\
      --output cases/macromodel_tests/trained_copper.xml --n-modes 10
"""

import argparse
import math
import os
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Namespaces used in the XML
NS = "http://schemas.datacontract.org/2004/07/ThermalSim.Models"
NS_A = "http://schemas.microsoft.com/2003/10/Serialization/Arrays"
NS_MESH = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh"
NS_BC = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"


def xfind(root, tag):
    """Find child with or without namespace."""
    x = root.find(f"{{{NS}}}{tag}")
    return x if x is not None else root.find(tag)


def xfindall(root, tag):
    """Find all children with or without namespace."""
    x = root.findall(f"{{{NS}}}{tag}")
    return x if x else root.findall(tag)


def afind(root, tag):
    """Find child in the a: serialization namespace."""
    x = root.find(f"{{{NS_A}}}{tag}")
    return x if x is not None else root.find(tag)


def bfind(root, tag):
    """Find child in the b: mesh namespace."""
    return root.find(f"{{{NS_MESH}}}{tag}")


def get_text(elem, default=""):
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def parse_double(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def find_smartmacro_block(root):
    """Find the first SmartMacro block."""
    layers = xfind(root, "Layers")
    if layers is None:
        return None, None, None, None, None

    for layer_elem in xfindall(layers, "Layer"):
        blocks_elem = xfind(layer_elem, "Blocks")
        if blocks_elem is None:
            continue
        for block_elem in xfindall(blocks_elem, "Block"):
            btype_elem = xfind(block_elem, "BlockType")
            if btype_elem is not None:
                bt = get_text(btype_elem).lower()
                if bt == "smartmacro":
                    material_name = get_text(xfind(block_elem, "MaterialName"))
                    model_file = get_text(xfind(block_elem, "ModelFile"))
                    rects_elem = xfind(block_elem, "AllRects")
                    rect_elem = (
                        xfind(rects_elem, "Rect") if rects_elem is not None else None
                    )
                    return layer_elem, block_elem, rect_elem, material_name, model_file

    return None, None, None, None, None


def compute_layer_stacking(root, si_scale):
    """Compute z_start/z_end for each layer (same algorithm as layer_processor.cpp).

    Returns list of (z_start, z_end) in SI meters, index 0 = first XML layer.
    """
    import math

    layers = xfind(root, "Layers")
    if layers is None:
        return []

    layer_elems = xfindall(layers, "Layer")
    n = len(layer_elems)
    thickness = [0.0] * n
    z_cursor = 0.0

    for l, layer_elem in enumerate(layer_elems):
        if l == 0:
            # Layer 0: thickness = max(block thicknesses, layer thickness)
            max_t = 0.0
            blocks_elem = xfind(layer_elem, "Blocks")
            if blocks_elem is not None:
                for b in xfindall(blocks_elem, "Block"):
                    bthick_text = get_text(xfind(b, "ThicknessExpression"))
                    if bthick_text:
                        t = parse_double(bthick_text) * si_scale
                        if t > max_t:
                            max_t = t
            layer_thick_text = get_text(xfind(layer_elem, "ThicknessExpression"))
            layer_t = (
                parse_double(layer_thick_text) * si_scale if layer_thick_text else 0.0
            )
            thickness[l] = max(max_t, layer_t)
        else:
            layer_t = (
                parse_double(get_text(xfind(layer_elem, "ThicknessExpression")))
                * si_scale
            )
            thickness[l] = layer_t
        z_cursor += thickness[l]

    result = []
    for l in range(n):
        z_start = z_cursor - thickness[l]
        z_end = z_cursor
        z_cursor -= thickness[l]
        result.append((z_start, z_end))

    return result


def eval_block_geometry(root, layer_elem, block_elem, rect_elem, si_scale, layer_index):
    """Evaluate geometry expressions for the smart block, including correct Z stacking.

    Returns (x_min, x_max, y_min, y_max, z_min, z_max) in SI.
    """

    def get_offset(elem, tag):
        return parse_double(get_text(xfind(elem, tag))) * si_scale

    layer_x_off = get_offset(layer_elem, "XOffsetExpression")
    layer_y_off = get_offset(layer_elem, "YOffsetExpression")
    block_x_off = get_offset(block_elem, "XOffsetExpression")
    block_y_off = get_offset(block_elem, "YOffsetExpression")

    x_expr = parse_double(get_text(xfind(rect_elem, "XExpression")))
    y_expr = parse_double(get_text(xfind(rect_elem, "YExpression")))
    w_expr = parse_double(get_text(xfind(rect_elem, "WidthExpression")))
    h_expr = parse_double(get_text(xfind(rect_elem, "HeightExpression")))

    if w_expr < 0:
        x_expr += w_expr
        w_expr = -w_expr
    if h_expr < 0:
        y_expr += h_expr
        h_expr = -h_expr

    x_min = x_expr * si_scale + block_x_off + layer_x_off
    y_min = y_expr * si_scale + block_y_off + layer_y_off
    x_max = x_min + w_expr * si_scale
    y_max = y_min + h_expr * si_scale

    # Get layer stacking for correct Z range
    stacking = compute_layer_stacking(root, si_scale)
    l_start, l_end = stacking[layer_index]

    block_thick_text = get_text(xfind(block_elem, "ThicknessExpression"))
    if block_thick_text:
        b_thick = parse_double(block_thick_text) * si_scale
        z_min = l_start
        z_max = l_start + b_thick
    else:
        z_min = l_start
        z_max = l_end

    return x_min, x_max, y_min, y_max, z_min, z_max


def build_mesh_from_xml(root):
    """Extract vertex arrays from XML, return (nx, ny, nz, cx, cy, cz, dx, dy, dz)."""
    results = xfind(root, "Results")
    if results is None:
        raise ValueError("No Results element")

    result3d = afind(results, "anyType")
    if result3d is None:
        # Try first child
        for c in results:
            if "Result" in c.tag:
                result3d = c
                break
    if result3d is None:
        raise ValueError("No Result3D element")

    mesh_elem = None
    for c in result3d:
        if "Mesh" in c.tag:
            mesh_elem = c
            break
    if mesh_elem is None:
        raise ValueError("No Mesh element")

    def read_array(parent, tag):
        arr = []
        # Try with b: namespace first, then plain
        el = bfind(parent, tag)
        if el is None:
            el = xfind(parent, tag)
        if el is None:
            el = parent.find(tag)
        if el is not None:
            for d in el:
                val = parse_double(get_text(d))
                arr.append(val)
        return arr

    xarr = read_array(mesh_elem, "XArray")
    yarr = read_array(mesh_elem, "YArray")
    zarr = read_array(mesh_elem, "ZArray")

    if not xarr or not yarr or not zarr:
        raise ValueError(f"Empty mesh: X={len(xarr)}, Y={len(yarr)}, Z={len(zarr)}")

    nx = len(xarr) - 1
    ny = len(yarr) - 1
    nz = len(zarr) - 1

    cx = np.array([(xarr[i] + xarr[i + 1]) * 0.5 for i in range(nx)])
    cy = np.array([(yarr[i] + yarr[i + 1]) * 0.5 for i in range(ny)])
    cz = np.array([(zarr[i] + zarr[i + 1]) * 0.5 for i in range(nz)])

    dx = np.array([xarr[i + 1] - xarr[i] for i in range(nx)])
    dy = np.array([yarr[i + 1] - yarr[i] for i in range(ny)])
    dz = np.array([zarr[i + 1] - zarr[i] for i in range(nz)])

    return nx, ny, nz, cx, cy, cz, dx, dy, dz


def scale_mesh_to_si(nx, ny, nz, cx, cy, cz, dx, dy, dz, si_scale):
    """Scale mesh coordinates from XML coordinate system to SI meters."""
    cx_si = cx * si_scale
    cy_si = cy * si_scale
    cz_si = cz * si_scale
    dx_si = dx * si_scale
    dy_si = dy * si_scale
    dz_si = dz * si_scale
    return nx, ny, nz, cx_si, cy_si, cz_si, dx_si, dy_si, dz_si


def get_material_props(root, mat_name):
    """Get thermal conductivity kx, ky, kz from XML Materials section."""
    materials = xfind(root, "Materials")
    if materials is None:
        raise ValueError(f"Materials section not found")

    for kv in materials.findall(f"{{{NS_A}}}KeyValueOfstringMaterialGyu7GfTz"):
        key_elem = afind(kv, "Key")
        if key_elem is not None and get_text(key_elem) == mat_name:
            val_elem = afind(kv, "Value")
            if val_elem is not None:
                kx = parse_double(get_text(xfind(val_elem, "DaoreXishu")))
                return kx, kx, kx

    raise ValueError(f"Material '{mat_name}' not found")


def main():
    register_namespaces_ns()
    parser = argparse.ArgumentParser(description="Train a POD-based SmartMacro model")
    parser.add_argument("input_xml", help="Path to the case XML file")
    parser.add_argument(
        "--output",
        "-o",
        default="trained_model.xml",
        help="Output path for trained SmartMacro model XML",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=10,
        help="Number of POD modes to retain (default: 10)",
    )
    args = parser.parse_args()

    input_path = args.input_xml
    output_path = args.output
    n_modes = args.n_modes

    # 1. Parse
    print(f"Parsing {input_path}...")
    tree = ET.parse(input_path)
    root = tree.getroot()

    lu_text = get_text(xfind(root, "LengthUnit")).lower()
    si_scale_map = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "inch": 0.0254,
        "mil": 2.54e-5,
    }
    si_scale = si_scale_map.get(lu_text, 1e-3)
    print(f"Length unit: '{lu_text}', SI scale: {si_scale}")

    # 2. Find SmartMacro block
    layer_elem, block_elem, rect_elem, mat_name, model_file = find_smartmacro_block(
        root
    )
    if rect_elem is None:
        print("ERROR: No SmartMacro block found")
        sys.exit(1)
    print(f"SmartMacro block material: {mat_name}, model file: {model_file}")

    # 3. Build mesh (XML coords are in case's length unit, e.g. Mm=mm)
    nx, ny, nz, cx, cy, cz, dx, dy, dz = build_mesh_from_xml(root)
    # Convert mesh coordinates to SI meters for geometry matching
    cx_si = cx * si_scale
    cy_si = cy * si_scale
    cz_si = cz * si_scale
    dx_si = dx * si_scale
    dy_si = dy * si_scale
    dz_si = dz * si_scale
    print(f"Mesh: {nx} x {ny} x {nz} = {nx * ny * nz} cells")
    print(
        f"  Cell center range: x=[{cx_si[0]:.4g},{cx_si[-1]:.4g}] "
        f"y=[{cy_si[0]:.4g},{cy_si[-1]:.4g}] z=[{cz_si[0]:.4g},{cz_si[-1]:.4g}]"
    )

    # 4. Evaluate block geometry with correct Z stacking
    # Find the smart block's layer index (0-based in XML layer order)
    layers_root = xfind(root, "Layers")
    layer_index = -1
    for li, le in enumerate(xfindall(layers_root, "Layer")):
        if le == layer_elem:
            layer_index = li
            break
    x_min, x_max, y_min, y_max, z_min, z_max = eval_block_geometry(
        root, layer_elem, block_elem, rect_elem, si_scale, layer_index
    )
    print(
        f"Block: x=[{x_min:.4g},{x_max:.4g}] y=[{y_min:.4g},{y_max:.4g}] z=[{z_min:.4g},{z_max:.4g}]"
    )

    # 5. Identify block cells
    block_old_indices = []
    block_ix, block_iy, block_iz = [], [], []
    eps = 1e-9
    for ix in range(nx):
        for iy in range(ny):
            for iz_ in range(nz):
                if (
                    x_min - eps <= cx_si[ix] <= x_max + eps
                    and y_min - eps <= cy_si[iy] <= y_max + eps
                    and z_min - eps <= cz_si[iz_] <= z_max + eps
                ):
                    old = ix * ny * nz + iy * nz + iz_
                    block_old_indices.append(old)
                    block_ix.append(ix)
                    block_iy.append(iy)
                    block_iz.append(iz_)

    n_block = len(block_old_indices)
    print(f"Block cells: {n_block}")

    if n_block == 0:
        print("ERROR: No cells inside block geometry")
        sys.exit(1)

    block_set = set(block_old_indices)
    old_to_local = {o: i for i, o in enumerate(block_old_indices)}

    # 6. Classify interface vs interior
    is_port = [False] * n_block
    dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    for i in range(n_block):
        ix, iy, iz = block_ix[i], block_iy[i], block_iz[i]
        for d_ix, d_iy, d_iz in dirs:
            nix, niy, niz = ix + d_ix, iy + d_iy, iz + d_iz
            if 0 <= nix < nx and 0 <= niy < ny and 0 <= niz < nz:
                n_old = nix * ny * nz + niy * nz + niz
                if n_old not in block_set:
                    is_port[i] = True
                    break

    port_local = sorted(
        [i for i, p in enumerate(is_port) if p],
        key=lambda i: (block_ix[i], block_iy[i], block_iz[i]),
    )
    interior_local = [i for i, p in enumerate(is_port) if not p]

    n_ports = len(port_local)
    n_int = len(interior_local)
    print(f"Interior: {n_int}, Ports: {n_ports}")

    if n_ports == 0:
        print("ERROR: No ports found")
        sys.exit(1)

    # 7. Material
    kx, ky, kz = get_material_props(root, mat_name)
    print(f"Material k: {kx} W/(m·K)")

    # 8. Build local stiffness matrix
    axis_of_dir = [0, 0, 1, 1, 2, 2]
    dir_vecs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    def face_area(f, ddx, ddy, ddz):
        a = axis_of_dir[f]
        d = [ddx, ddy, ddz]
        return d[(a + 1) % 3] * d[(a + 2) % 3]

    def k_along(f):
        return [kx, ky, kz][axis_of_dir[f]]

    def half_len(f, ddx, ddy, ddz):
        return [ddx, ddy, ddz][axis_of_dir[f]] * 0.5

    n_total = n_int + n_ports
    K_local = lil_matrix((n_total, n_total))

    loc_map = {}
    for r, li in enumerate(interior_local):
        loc_map[li] = r
    for r, li in enumerate(port_local):
        loc_map[li] = n_int + r

    print("Building local stiffness matrix...")
    # Also build local RHS vector for BCs with non-zero RHS terms
    f_local = np.zeros(n_total)

    # Identify the ZP convection BC from the case XML
    # The boundary is Z|E|30|-50,150,-50,150 with ThirdType h=50, T_inf=300
    # This applies to the ZP face (dir 5) of the smart block's top cells
    boundary_h = 0.0
    boundary_Tinf = 0.0
    boundaries_elem = xfind(root, "Boundaries")
    if boundaries_elem is not None:
        NS_BC = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"
        for bdry in boundaries_elem:
            for child in bdry:
                if "ThermalBoundary" in child.tag:
                    thermal = child
                    for sub in thermal:
                        if "ConvectionCoefficient" in sub.tag:
                            h_val = parse_double(get_text(sub))
                        if "EnvironmentTemperature" in sub.tag:
                            t_val = parse_double(get_text(sub))
                    if h_val > 0:
                        boundary_h = h_val
                        boundary_Tinf = t_val
                        print(f"Found convection BC: h={boundary_h}, T_inf={t_val}")
                        break

    for i in range(n_block):
        ri = loc_map[i]
        ix, iy, iz = block_ix[i], block_iy[i], block_iz[i]
        ddx, ddy, ddz = dx_si[ix], dy_si[iy], dz_si[iz]

        for f in range(6):
            d_ix, d_iy, d_iz = dir_vecs[f]
            nix, niy, niz = ix + d_ix, iy + d_iy, iz + d_iz

            if not (0 <= nix < nx and 0 <= niy < ny and 0 <= niz < nz):
                # Domain boundary face — apply BC
                if f == 5 and boundary_h > 0:  # ZP face
                    a_f = face_area(f, ddx, ddy, ddz)
                    h_i = half_len(f, ddx, ddy, ddz)
                    k_i = k_along(f)
                    # ThirdType BC: same as in assembler.cpp
                    coeff = k_i * boundary_h * a_f / (k_i + boundary_h * h_i)
                    K_local[ri, ri] += coeff
                    f_local[ri] += coeff * boundary_Tinf
                continue

            n_old = nix * ny * nz + niy * nz + niz

            if n_old in block_set:
                j = old_to_local[n_old]
                rj = loc_map[j]

                a_f = face_area(f, ddx, ddy, ddz)
                h_i = half_len(f, ddx, ddy, ddz)
                k_i = k_along(f)

                ndx, ndy, ndz = dx_si[nix], dy_si[niy], dz_si[niz]
                h_j = half_len(f, ndx, ndy, ndz)
                k_j = k_along(f)

                denom = h_i / k_i + h_j / k_j
                if denom > 0:
                    cond = a_f / denom
                    K_local[ri, ri] += cond
                    K_local[ri, rj] -= cond

    # Add mass matrix for flux-free steady — this is purely conductive.
    # Partition
    K_csr = K_local.tocsr()
    K_ii = K_csr[:n_int, :n_int].tocsc()
    K_if = K_csr[:n_int, n_int:].tocsc()
    K_fi = K_csr[n_int:, :n_int].tocsc()
    K_ff = K_csr[n_int:, n_int:].tocsc()
    f_i = f_local[:n_int]
    f_f = f_local[n_int:]

    # 9. Schur complement
    print("Computing Schur complement K_port...")
    K_if_dense = K_if.toarray()
    X = spsolve(K_ii, K_if_dense)
    K_port = K_ff.toarray() - K_fi.toarray() @ X

    # RHS contribution from BCs: f_port_reduced = f_f - K_fi * K_ii^(-1) * f_i
    f_port = f_f - K_fi.toarray() @ spsolve(K_ii, f_i)

    print(
        f"K_port: [{K_port.shape[0]}x{K_port.shape[1]}], "
        f"min={K_port.min():.4e}, max={K_port.max():.4e}"
    )
    print(f"f_port: [{len(f_port)}], min={f_port.min():.4e}, max={f_port.max():.4e}")

    sym_err = np.max(np.abs(K_port - K_port.T))
    if sym_err > 1e-10:
        print(f"Symmetry error: {sym_err:.4e}, symmetrizing")
        K_port = 0.5 * (K_port + K_port.T)

    # 10. POD: eigendecomposition of K_port → extract n_modes smoothest modes
    # For thermal diffusion, the physically dominant modes are those with the
    # smallest eigenvalues (smooth spatial variations), not the largest.
    print(f"Computing POD basis (n_modes={n_modes})...")
    eigvals, eigvecs = np.linalg.eigh(K_port)
    # eigh returns ascending order — smallest eigenvalues first = smoothest modes
    # Take the first n_modes as our POD basis
    n_modes = min(n_modes, n_ports)
    n_modes = max(n_modes, 1)
    Phi = np.ascontiguousarray(eigvecs[:, :n_modes])  # [N_ports x n_modes]

    K_modal = Phi.T @ K_port @ Phi  # [n_modes x n_modes]
    f_modal = Phi.T @ f_port        # [n_modes]

    print(f"  Smallest eigenvalue: {eigvals[0]:.6e}, largest among selected: {eigvals[n_modes-1]:.6e}")
    print(f"  Eigenvalue ratio (selected/total): {np.sum(eigvals[:n_modes]) / np.sum(np.abs(eigvals)):.8f}")

    # 11. Write trained model
    base = os.path.splitext(output_path)[0]
    data_path = base + ".data"
    print(f"Writing {output_path} (XML) + {data_path} (binary)...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Write binary data file:
    #   [f_modal: M doubles][K_modal: M*M doubles row-major][phi_basis: N*M doubles row-major]
    K_modal_flat = np.ascontiguousarray(K_modal, dtype=np.float64).ravel(order="C")
    Phi_flat = np.ascontiguousarray(Phi, dtype=np.float64).ravel(order="C")
    with open(data_path, "wb") as f:
        f.write(f_modal.astype(np.float64).tobytes())
        f.write(K_modal_flat.tobytes())
        f.write(Phi_flat.tobytes())

    # Write tiny XML with pointer to data file
    ET.register_namespace("", "SmartMacroModel")
    root_elem = ET.Element("SmartMacroModel")

    ET.SubElement(root_elem, "Name").text = f"{mat_name}_block"
    ET.SubElement(root_elem, "NPorts").text = str(n_ports)
    ET.SubElement(root_elem, "NModes").text = str(n_modes)
    ET.SubElement(root_elem, "DataFile").text = os.path.basename(data_path)

    po = ET.SubElement(root_elem, "PortOrder")
    for li in port_local:
        pe = ET.SubElement(po, "Port")
        ET.SubElement(pe, "IX").text = str(block_ix[li])
        ET.SubElement(pe, "IY").text = str(block_iy[li])
        ET.SubElement(pe, "IZ").text = str(block_iz[li])

    pretty = (
        minidom.parseString(ET.tostring(root_elem, encoding="utf-8"))
        .toprettyxml(indent="  ", encoding="utf-8")
        .decode("utf-8")
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    data_size_kb = os.path.getsize(data_path) / 1024
    print(
        f"Done!  Ports: {n_ports}, Modes: {n_modes}, "
        f"data file: {data_size_kb:.1f} KB"
    )
    return 0


def register_namespaces_ns():
    """Register namespaces so etree preserves them."""
    ET.register_namespace("", NS)
    ET.register_namespace("a", NS_A)
    ET.register_namespace("b", NS_MESH)


if __name__ == "__main__":
    raise SystemExit(main())
