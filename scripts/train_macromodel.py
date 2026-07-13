#!/usr/bin/env python3
"""
train_macromodel.py — Train a face-level POD-based SmartMacro model

Given a case XML and a (layer, block) position:
  1. Parse the global mesh and block geometry
  2. Identify ALL boundary FACES of the block (not cells)
  3. Build cell-cell FVM matrix K_cc + scatter matrix S (cell-face coupling)
  4. Factor K_cc = sparse LU, use implicit matvec for eigsh:
       K_port_face @ v = C_DtN ∘ v - S^T @ (K_cc^{-1} @ (S @ v))
  5. POD (eigsh, which='SA') → retain n_modes dominant modes
  6. Write trained model XML + binary with phi_basis_face, K_modal

Usage:
  python scripts/train_macromodel.py cases/macromodel_tests/case1.xml \
      --output-dir cases/macromodel_tests/ --n-modes 10

Differences from old version:
  - Ports are boundary FACES, not boundary cells
  - No BC handling — fully boundary independent
  - Uses scipy.sparse.linalg.eigsh with implicit matvec (no dense K_port_face)
  - f_modal is always zero and eliminated from the binary format
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu, eigsh, LinearOperator

# Namespaces used in the XML
NS = "http://schemas.datacontract.org/2004/07/ThermalSim.Models"
NS_A = "http://schemas.microsoft.com/2003/10/Serialization/Arrays"
NS_MESH = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh"


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


def get_text(elem, default=""):
    if elem is not None and elem.text:
        return elem.text.strip()
    return default


def parse_double(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def parse_variables(root):
    """Extract variable definitions (name → value) from <Variables> section."""
    variables = {}
    vars_elem = xfind(root, "Variables")
    if vars_elem is None:
        return variables
    for kv in vars_elem:
        key_elem = afind(kv, "Key")
        val_elem = afind(kv, "Value")
        if (
            key_elem is not None
            and val_elem is not None
            and key_elem.text
            and val_elem.text
        ):
            variables[key_elem.text.strip()] = parse_double(val_elem.text.strip())
    return variables


def eval_expression(expr_str, variables):
    """Evaluate a simple arithmetic expression that may reference variables.

    Handles: identifiers, numbers, +, -, *, /, unary negation.
    Expressions like 't_top', '-w_bottom/2', 'h_middle*3/2' are resolved
    by substituting known variable values first, then evaluating.
    """
    s = expr_str.strip() if expr_str else ""
    if not s:
        return 0.0
    # Fast path: already a plain number
    try:
        return float(s)
    except ValueError:
        pass
    # Substitute variable names (longest first to avoid partial-name clashes)
    for name in sorted(variables.keys(), key=len, reverse=True):
        s = s.replace(name, str(variables[name]))
    # Evaluate the resulting arithmetic expression
    try:
        return float(eval(s))
    except Exception as e:
        print(
            f"  Warning: cannot evaluate expression '{expr_str}': {e}", file=sys.stderr
        )
        return 0.0


def find_block_by_position(root, layer_index, block_index):
    """Find a block at the given layer index and block index within that layer."""
    layers = xfind(root, "Layers")
    if layers is None:
        return None, None, None, None, None, None
    layer_elems = xfindall(layers, "Layer")
    if layer_index < 0 or layer_index >= len(layer_elems):
        print(f"ERROR: Layer index {layer_index} out of range (0-{len(layer_elems)-1})")
        return None, None, None, None, None, None
    layer_elem = layer_elems[layer_index]
    layer_name = get_text(xfind(layer_elem, "Name"))
    blocks_elem = xfind(layer_elem, "Blocks")
    if blocks_elem is None:
        print(f"ERROR: No Blocks element in layer {layer_index}")
        return None, None, None, None, None, None
    block_elems = xfindall(blocks_elem, "Block")
    if block_index < 0 or block_index >= len(block_elems):
        print(f"ERROR: Block index {block_index} out of range (0-{len(block_elems)-1})")
        return None, None, None, None, None, None
    block_elem = block_elems[block_index]
    material_name = get_text(xfind(block_elem, "MaterialName"))
    block_name = get_text(xfind(block_elem, "Name"))
    rects_elem = xfind(block_elem, "AllRects")
    rect_elem = xfind(rects_elem, "Rect") if rects_elem is not None else None
    return layer_elem, block_elem, rect_elem, material_name, layer_name, block_name


def compute_layer_stacking(root, si_scale, variables):
    """Compute z_start/z_end for each layer (same algorithm as layer_processor.cpp)."""
    layers = xfind(root, "Layers")
    if layers is None:
        return []
    layer_elems = xfindall(layers, "Layer")
    n = len(layer_elems)
    thickness = [0.0] * n
    z_cursor = 0.0
    for l, layer_elem in enumerate(layer_elems):
        if l == 0:
            max_t = 0.0
            blocks_elem = xfind(layer_elem, "Blocks")
            if blocks_elem is not None:
                for b in xfindall(blocks_elem, "Block"):
                    bthick_text = get_text(xfind(b, "ThicknessExpression"))
                    if bthick_text:
                        t = eval_expression(bthick_text, variables) * si_scale
                        if t > max_t:
                            max_t = t
            layer_thick_text = get_text(xfind(layer_elem, "ThicknessExpression"))
            layer_t = (
                eval_expression(layer_thick_text, variables) * si_scale
                if layer_thick_text
                else 0.0
            )
            thickness[l] = max(max_t, layer_t)
        else:
            layer_t = (
                eval_expression(
                    get_text(xfind(layer_elem, "ThicknessExpression")), variables
                )
                * si_scale
            )
            thickness[l] = layer_t
        z_cursor += thickness[l]
    result = []
    for l in range(n):
        z_start = z_cursor - thickness[l]
        result.append((z_start, z_start + thickness[l]))
        z_cursor -= thickness[l]
    return result


def eval_block_geometry(
    root, layer_elem, block_elem, rect_elem, si_scale, layer_index, variables
):
    """Evaluate geometry expressions for the smart block, including correct Z stacking."""

    def get_offset(elem, tag):
        return eval_expression(get_text(xfind(elem, tag)), variables) * si_scale

    layer_x_off = get_offset(layer_elem, "XOffsetExpression")
    layer_y_off = get_offset(layer_elem, "YOffsetExpression")
    block_x_off = get_offset(block_elem, "XOffsetExpression")
    block_y_off = get_offset(block_elem, "YOffsetExpression")

    x_expr = eval_expression(get_text(xfind(rect_elem, "XExpression")), variables)
    y_expr = eval_expression(get_text(xfind(rect_elem, "YExpression")), variables)
    w_expr = eval_expression(get_text(xfind(rect_elem, "WidthExpression")), variables)
    h_expr = eval_expression(get_text(xfind(rect_elem, "HeightExpression")), variables)

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

    stacking = compute_layer_stacking(root, si_scale, variables)
    l_start, l_end = stacking[layer_index]

    block_thick_text = get_text(xfind(block_elem, "ThicknessExpression"))
    if layer_index == 0 and block_thick_text:
        # Top layer only: clip block to its own thickness (matches C++ layer_processor.cpp)
        b_thick = eval_expression(block_thick_text, variables) * si_scale
        z_min = l_start
        z_max = l_start + b_thick
    else:
        # Non-top layers: block fills the entire layer height
        z_min = l_start
        z_max = l_end

    return x_min, x_max, y_min, y_max, z_min, z_max


def build_mesh_from_xml(root):
    """Extract vertex arrays from XML."""
    results = xfind(root, "Results")
    if results is None:
        raise ValueError("No Results element")
    result3d = afind(results, "anyType")
    if result3d is None:
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
        el = parent.find(f"{{{NS_MESH}}}{tag}")
        if el is None:
            el = xfind(parent, tag)
        return [parse_double(get_text(d)) for d in el] if el is not None else []

    xarr = read_array(mesh_elem, "XArray")
    yarr = read_array(mesh_elem, "YArray")
    zarr = read_array(mesh_elem, "ZArray")

    nx, ny, nz = len(xarr) - 1, len(yarr) - 1, len(zarr) - 1
    cx = np.array([(xarr[i] + xarr[i + 1]) * 0.5 for i in range(nx)])
    cy = np.array([(yarr[i] + yarr[i + 1]) * 0.5 for i in range(ny)])
    cz = np.array([(zarr[i] + zarr[i + 1]) * 0.5 for i in range(nz)])
    dx = np.array([xarr[i + 1] - xarr[i] for i in range(nx)])
    dy = np.array([yarr[i + 1] - yarr[i] for i in range(ny)])
    dz = np.array([zarr[i + 1] - zarr[i] for i in range(nz)])

    return nx, ny, nz, cx, cy, cz, dx, dy, dz


# Axis convention for face directions (same as mesh_utils.hpp)
# 0=X|E, 1=X|W, 2=Y|N, 3=Y|S, 4=Z|P, 5=Z|B
AXIS_OF_DIR = [0, 0, 1, 1, 2, 2]  # which axis each dir corresponds to
DIR_VECS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]


def get_material_props(root, mat_name):
    """Get thermal conductivity kx, ky, kz from XML Materials section."""
    materials = xfind(root, "Materials")
    for kv in materials.findall(f"{{{NS_A}}}KeyValueOfstringMaterialGyu7GfTz"):
        key_elem = afind(kv, "Key")
        if key_elem is not None and get_text(key_elem) == mat_name:
            val_elem = afind(kv, "Value")
            if val_elem is not None:
                kx = parse_double(get_text(xfind(val_elem, "DaoreXishu")))
                return kx, kx, kx
    raise ValueError(f"Material '{mat_name}' not found")


def main():
    parser = argparse.ArgumentParser(
        description="Train a FACE-LEVEL POD-based SmartMacro model on any block in any case XML"
    )
    parser.add_argument("input_xml", help="Path to the case XML file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (0-based)")
    parser.add_argument("--block", type=int, default=0, help="Block index (0-based)")
    parser.add_argument(
        "--n-modes", type=int, default=15, help="Number of POD modes to retain"
    )
    parser.add_argument(
        "--material-k", type=float, default=None, help="Override material k [W/mK]"
    )
    args = parser.parse_args()

    input_path = args.input_xml
    n_modes = args.n_modes
    output_dir = args.output_dir

    # ── 1. Parse ───────────────────────────────────────────────────────────
    print("Parsing", input_path)
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

    layer_elem, block_elem, rect_elem, mat_name, layer_name, block_name = (
        find_block_by_position(root, args.layer, args.block)
    )
    print(
        f"Training layer {args.layer} ('{layer_name}') block {args.block} "
        f"(material: {mat_name}) — face-level, BC-free DtN"
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = f"trained_layer{args.layer}_block{args.block}"
    output_path = os.path.join(output_dir, base_name + ".xml")
    input_basename = os.path.splitext(os.path.basename(input_path))[0]
    modified_case_path = os.path.join(output_dir, f"modified_{input_basename}.xml")

    # ── 2. Build mesh ──────────────────────────────────────────────────────
    nx, ny, nz, cx, cy, cz, dx, dy, dz = build_mesh_from_xml(root)
    cx_si, cy_si, cz_si = cx * si_scale, cy * si_scale, cz * si_scale
    dx_si, dy_si, dz_si = dx * si_scale, dy * si_scale, dz * si_scale
    print(f"Mesh: {nx} x {ny} x {nz} = {nx*ny*nz} cells")

    # ── 3. Parse variable definitions and evaluate block geometry ─────────
    variables = parse_variables(root)
    if variables:
        print(f"Variables: { {k: v for k, v in sorted(variables.items())} }")
    x_min, x_max, y_min, y_max, z_min, z_max = eval_block_geometry(
        root, layer_elem, block_elem, rect_elem, si_scale, args.layer, variables
    )
    print(
        f"Block: x=[{x_min:.4g},{x_max:.4g}] y=[{y_min:.4g},{y_max:.4g}] z=[{z_min:.4g},{z_max:.4g}]"
    )

    # ── 4. Identify block cells ────────────────────────────────────────────
    eps = 1e-9
    block_old_indices = []
    block_ix, block_iy, block_iz = [], [], []
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

    # Cache material properties (invariant across all cells)
    kx_mat, ky_mat, kz_mat = get_material_props(root, mat_name)

    # ── 5. Enumerate boundary FACES and build port_faces ───────────────────
    # A port face is a face where the neighbor is OUTSIDE the block
    # (either another layer's cell or the domain boundary).
    port_faces = []
    port_face_count = [0] * n_block  # how many port faces per cell
    n_int = 0  # interior cells counter

    for i in range(n_block):
        ix, iy, iz = block_ix[i], block_iy[i], block_iz[i]
        dx_c = dx_si[ix]
        dy_c = dy_si[iy]
        dz_c = dz_si[iz]
        is_boundary = False
        for f in range(6):
            d_ix, d_iy, d_iz = DIR_VECS[f]
            nix, niy, niz = ix + d_ix, iy + d_iy, iz + d_iz
            if 0 <= nix < nx and 0 <= niy < ny and 0 <= niz < nz:
                n_old = nix * ny * nz + niy * nz + niz
                if n_old in block_set:
                    continue  # neighbor is also in the block — internal face
            # Face is on boundary: no neighbor outside block
            a = AXIS_OF_DIR[f]
            A_f = [dy_c * dz_c, dx_c * dz_c, dx_c * dy_c][a]
            half_dist_block = [dx_c, dy_c, dz_c][a] * 0.5
            k_along = [kx_mat, ky_mat, kz_mat][a]
            C_DtN_f = k_along * A_f / half_dist_block

            port_faces.append(
                {
                    "cell_local": i,
                    "ix": ix,
                    "iy": iy,
                    "iz": iz,
                    "dir": f,
                    "C_DtN": C_DtN_f,
                }
            )
            port_face_count[i] += 1
            is_boundary = True

        if not is_boundary:
            n_int += 1

    n_ports = len(port_faces)
    print(f"Interior cells: {n_int}, Port faces: {n_ports}")
    if n_ports == 0:
        print("ERROR: No port faces found")
        sys.exit(1)

    C_DtN_arr = np.array([pf["C_DtN"] for pf in port_faces], dtype=np.float64)

    # ── 6. Build K_cc (cell-cell FVM matrix) ───────────────────────────────
    # K_cc [n_block × n_block]: cell-to-cell internal conduction
    print("Building K_cc (cell-cell)...")
    rows_cc, cols_cc, data_cc = [], [], []

    for i in range(n_block):
        ix, iy, iz = block_ix[i], block_iy[i], block_iz[i]
        ddx = dx_si[ix]
        ddy = dy_si[iy]
        ddz = dz_si[iz]

        for f in range(6):
            d_ix, d_iy, d_iz = DIR_VECS[f]
            nix, niy, niz = ix + d_ix, iy + d_iy, iz + d_iz
            if not (0 <= nix < nx and 0 <= niy < ny and 0 <= niz < nz):
                continue
            n_old = nix * ny * nz + niy * nz + niz
            if n_old not in block_set:
                continue

            # Neighbor is also in the block — internal face
            j = old_to_local[n_old]
            a = AXIS_OF_DIR[f]
            A_f = [[ddy * ddz, ddx * ddz, ddx * ddy][a]][0]
            h_i = [ddx, ddy, ddz][a] * 0.5
            k_i = [kx_mat, ky_mat, kz_mat][a]

            ndx, ndy, ndz = dx_si[nix], dy_si[niy], dz_si[niz]
            h_j = [ndx, ndy, ndz][a] * 0.5
            k_j = k_i  # same material for now

            denom = h_i / k_i + h_j / k_j
            if denom > 0:
                cond = A_f / denom
                rows_cc.append(i)
                cols_cc.append(i)
                data_cc.append(cond)
                rows_cc.append(i)
                cols_cc.append(j)
                data_cc.append(-cond)

    # Add C_DtN contributions to K_cc diagonal for boundary cells
    for f, pf in enumerate(port_faces):
        c = pf["cell_local"]
        rows_cc.append(c)
        cols_cc.append(c)
        data_cc.append(C_DtN_arr[f])

    K_cc_csr = csr_matrix((data_cc, (rows_cc, cols_cc)), shape=(n_block, n_block))

    # ── 7. Build S matrix (cell-face coupling) ─────────────────────────────
    # S[c, f] = C_DtN_f if port face f belongs to cell c
    # Used in implicit matvec: K_port_face @ v = C_DtN ∘ v - S^T @ (K_cc^{-1} @ (S @ v))
    print("Building cell-face coupling system...")

    row_indices_s, col_indices_s, data_s = [], [], []
    for f, pf in enumerate(port_faces):
        c = pf["cell_local"]
        row_indices_s.append(c)
        col_indices_s.append(f)
        data_s.append(C_DtN_arr[f])

    S = csr_matrix((data_s, (row_indices_s, col_indices_s)), shape=(n_block, n_ports))

    # Factor K_cc once (used for implicit matvec in eigsh)
    print("  Factorizing K_cc (sparse LU)...")
    lu = splu(K_cc_csr)

    # ── 8. POD: eigendecomposition of K_port_face via implicit matvec ──────
    # K_port_face @ v = C_DtN ∘ v - S^T @ (K_cc^{-1} @ (S @ v))
    #                  = C_DtN ∘ v - S^T @ (lu.solve(S @ v))

    def matvec(v):
        """K_port_face @ v using implicit scheme."""
        result = C_DtN_arr * v
        t_cell = S @ v
        u_cell = lu.solve(t_cell)
        result -= S.T @ u_cell
        return result

    print(f"Computing POD via eigsh (n_modes={n_modes}, n_ports={n_ports})...")
    n_modes = min(n_modes, n_ports - 1, n_block)
    n_modes = max(n_modes, 1)

    A_op = LinearOperator((n_ports, n_ports), matvec=matvec, dtype=np.float64)
    Phi_face = None
    eigvals_orig = None

    # Try eigsh with progressive tolerance loosening
    config = {
        "k": n_modes,
        "which": "SA",
        "tol": 1e-6,
        "maxiter": 5000,
        "ncv": 2 * n_modes + 1,
    }
    try:
        eigvals, eigvecs = eigsh(A_op, **config)
        nk = min(n_modes, len(eigvals))
        eigvals_orig = eigvals[:nk]
        Phi_face = np.ascontiguousarray(eigvecs[:, :nk], dtype=np.float64)
        print(f"  eigsh succeeded: k={config['k']}, tol={config['tol']}")
    except Exception as e:
        print(f"  eigsh(k={config['k']}, tol={config['tol']}) failed: {e}")

    if Phi_face is None:
        print("ERROR: Could not compute POD basis")
        sys.exit(1)

    # Ensure Phi is properly oriented
    Phi_face = np.ascontiguousarray(Phi_face, dtype=np.float64)
    print(f"  Eigenvalue range: {eigvals_orig[0]:.6e} to {eigvals_orig[-1]:.6e}")

    # ── 9. Project K_modal = Φ^T @ K_port_face @ Φ ─────────────────────────
    K_modal = np.zeros((n_modes, n_modes))
    for j in range(n_modes):
        phi_j = Phi_face[:, j]
        K_port_phi_j = matvec(phi_j)
        for i in range(n_modes):
            K_modal[i, j] = np.dot(Phi_face[:, i], K_port_phi_j)

    print(f"  K_modal: min={K_modal.min():.4e}, max={K_modal.max():.4e}")

    # ── 10. Write trained model (binary: [K_modal: M×M][phi_basis: N_ports×M]) ─
    base = os.path.splitext(output_path)[0]
    data_path = base + ".data"
    print(f"Writing {output_path} (XML) + {data_path} (binary)...")

    K_modal_flat = np.ascontiguousarray(K_modal, dtype=np.float64).ravel(order="C")
    Phi_flat = np.ascontiguousarray(Phi_face, dtype=np.float64).ravel(order="C")
    with open(data_path, "wb") as f:
        f.write(K_modal_flat.tobytes())
        f.write(Phi_flat.tobytes())

    # Write XML with face-level port info (including <Dir>) and block material k
    ET.register_namespace("", "SmartMacroModel")
    root_elem = ET.Element("SmartMacroModel")

    ET.SubElement(root_elem, "Name").text = (
        f"trained_layer{args.layer}_block{args.block}"
    )
    ET.SubElement(root_elem, "NPorts").text = str(n_ports)
    ET.SubElement(root_elem, "NModes").text = str(n_modes)
    ET.SubElement(root_elem, "DataFile").text = os.path.basename(data_path)

    po = ET.SubElement(root_elem, "PortOrder")
    for pf in port_faces:
        pe = ET.SubElement(po, "Port")
        ET.SubElement(pe, "IX").text = str(pf["ix"])
        ET.SubElement(pe, "IY").text = str(pf["iy"])
        ET.SubElement(pe, "IZ").text = str(pf["iz"])
        ET.SubElement(pe, "Dir").text = str(pf["dir"])

    pretty = ET.tostring(root_elem, encoding="unicode")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    data_size_kb = os.path.getsize(data_path) / 1024
    print(
        f"Done!  Port faces: {n_ports}, Modes: {n_modes}, data file: {data_size_kb:.1f} KB"
    )

    # ── 11. Write modified case XML ─────────────────────────────────────────
    trained_basename = os.path.basename(output_path)

    # ElementTree.write() cannot reproduce the original default-namespace format
    # (<Structure xmlns="...">) — it emits prefixed roots (<ns0:Structure>) which
    # the C++ XML parser rejects.  Use line-based text patching instead.
    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    block_depth = 0
    layer_count = -1  # 0-based counter for <Layer> tags
    block_count = -1  # 0-based counter for <Block> tags within current layer
    target_layer = args.layer
    target_block = args.block
    target_block_depth = -1
    in_target_layer = False
    block_type_inserted = False
    model_file_inserted = False
    out_lines = []

    for line in lines:
        stripped = line.strip()

        # Track layer depth to reset block counter per layer
        if stripped == "<Layer>":
            layer_count += 1
            block_count = -1
            in_target_layer = layer_count == target_layer

        # Only count <Block> opening tags (not <Blocks>, <ParentBlock>, etc.)
        if stripped == "<Block>" and in_target_layer:
            block_depth += 1
            block_count += 1
            if block_count == target_block:
                target_block_depth = block_depth

        if target_block_depth > 0 and block_depth >= target_block_depth:
            if stripped.startswith("<BlockType>"):
                line = line.replace(stripped, "<BlockType>SmartMacro</BlockType>")
                block_type_inserted = True
            if stripped.startswith("<ModelFile>"):
                line = line.replace(
                    stripped, f"<ModelFile>{trained_basename}</ModelFile>"
                )
                model_file_inserted = True

        if stripped == "</Block>" and block_depth > 0:
            if block_depth == target_block_depth:
                indent = line[: len(line) - len(line.lstrip())]
                if not block_type_inserted:
                    out_lines.append(f"{indent}  <BlockType>SmartMacro</BlockType>\n")
                if not model_file_inserted:
                    out_lines.append(
                        f"{indent}  <ModelFile>{trained_basename}</ModelFile>\n"
                    )
                target_block_depth = -1
            block_depth -= 1

        out_lines.append(line)

    with open(modified_case_path, "w", encoding="utf-8") as f:
        f.write("".join(out_lines))

    print(f"Modified case XML: {modified_case_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
