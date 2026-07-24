#!/usr/bin/env python3
"""
BCI-ROM: Block-Canonical-Insertion ROM for parametric heat sink.

Reduces only the copper (heat-sink) layer across the parametric convection
coefficient h (1 to 1e6 W/m2K) on the top surface.  Uses MPMM:

  - port face patches (copper-TIM interface) are treated as power ports
  - for each training h, extract the interface temperature snapshot
  - merge subspaces via column-wise concatenation -> SVD -> reduced basis
  - online: project the (external+port) Schur complement, solve, recover
    internal copper DOFs, export the full field to VTU.

Only copper is reduced. TIM, silicon, and the recovered copper are exported.

Requires: metahotspot, numpy, scipy
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums

# ===========================================================================
#  Configuration
# ===========================================================================

OUT_DIR = Path(__file__).resolve().parent / "bci_rom_output"

# Training: 20 log-spaced samples  (covers 6 decades)
H_TRAIN = np.geomspace(1.0, 1e6, 20)
H_TEST = np.array([2.5, 15.0, 1.0e2, 3.0e3, 8.0e4, 2.5e5])

MACRO_LAYER = 0  # copper
MACRO_BLOCK = 0  # copper block (only block in layer 0)

# SVD energy threshold
ENERGY_THR = 0.9999

# ===========================================================================
#  Model building
# ===========================================================================


def build_model(h_value: float) -> metahotspot.Model:
    """Build the 3-layer stack model with top-surface convection = *h*."""
    m = metahotspot.Model()
    m.set_settings(
        study=enums.Study.STEADY,
        length_unit=enums.LengthUnit.MILLIMETER,
        initial_temperature_K=300.0,
    )

    # Mesh (same layout as simple_steady_case2.xml)
    m.set_mesh(
        x=np.linspace(-50, 150, 51),
        y=np.linspace(-50, 150, 65),
        z=np.linspace(0, 30, 10),
    )

    # Materials
    m.add_material("copper", kx="400", ky="400", kz="400",
                   rho="8920", c="385")
    m.add_material("TIM", kx="5", ky="5", kz="5",
                   rho="1200", c="1000")
    m.add_material("silicon", kx="130", ky="130", kz="130",
                   rho="2330", c="0.71")

    # Layers
    lid0 = m.add_layer(thickness="20")  # copper   (Z 0 ... 20 mm)
    lid1 = m.add_layer(thickness="5")   # TIM      (Z 20 ... 25 mm)
    lid2 = m.add_layer(thickness="5")   # silicon  (Z 25 ... 30 mm)

    # Blocks
    b0 = m.add_block(lid0, material_name="copper",
                     x_offset="0", y_offset="0")
    b1 = m.add_block(lid1, material_name="TIM",
                     x_offset="0", y_offset="0")
    b2 = m.add_block(lid2, material_name="silicon",
                     heat_source="1e6", x_offset="10", y_offset="5")
    b3 = m.add_block(lid2, material_name="silicon",
                     heat_source="2e6", x_offset="10", y_offset="55")

    # Rectangular geometry ops
    m.add_rect(b0, op=enums.GeometryOp.ADD,
               x="-50", y="-50", width="200", height="200")
    m.add_rect(b1, op=enums.GeometryOp.ADD,
               x="0", y="0", width="100", height="100")
    m.add_rect(b2, op=enums.GeometryOp.ADD,
               x="0", y="0", width="80", height="40")
    m.add_rect(b3, op=enums.GeometryOp.ADD,
               x="0", y="0", width="80", height="40")

    # ---- Top-surface convection (param) --------------------------------
    bc_id = m.add_boundary()
    m.set_convection(bc_id, coefficient=str(h_value),
                     ambient_temperature="300")
    m.add_face_region(bc_id, axis=enums.Axis.Z, coordinate=30.0,
                      a_min=-50, a_max=150, b_min=-50, b_max=150)

    return m


def compile_assemble(
    h_value: float,
) -> tuple:
    """Build -> compile -> assemble; return (K, f, layer_ids, block_ids, mesh_z)."""
    m = build_model(h_value)
    compiled = m.compile()
    nc = compiled.cell_count()
    layer_ids = compiled.layer_ids().copy()
    block_ids = compiled.block_ids().copy()

    assembly = compiled.assemble()
    K = assembly.stiffness_matrix()
    f = assembly.rhs().copy()
    assembly.close()
    m.close()
    return compiled, K, f, layer_ids, block_ids


# ===========================================================================
#  Partitioning  (identical logic to macromodel_demo.py)
# ===========================================================================


def partition(K, nc, layer_ids, block_ids, macro_layer, macro_block):
    """Split mesh into external (e), port (p), and internal (i) sets.

    Port  = macro cells adjacent to at least one non-macro cell.
    Internal = macro cells *not* adjacent to any non-macro cell.
    External = all non-macro cells.
    """
    is_macro = (layer_ids == macro_layer) & (block_ids == macro_block)
    macro_set = set(np.where(is_macro)[0])

    port_set = set()
    for i in macro_set:
        rs, re = K.indptr[i], K.indptr[i + 1]
        for j in K.indices[rs:re]:
            if j not in macro_set:
                port_set.add(i)
                break

    is_port = np.zeros(nc, dtype=bool)
    for p in port_set:
        is_port[p] = True

    e_idx = np.sort(np.where(~is_macro)[0])
    p_idx = np.sort(np.where(is_port)[0])
    i_idx = np.sort(np.where(is_macro & ~is_port)[0])
    return e_idx, p_idx, i_idx


# ===========================================================================
#  Guyan-like Schur complement  (for the copper block)
# ===========================================================================


def build_schur(K, f, e_idx, p_idx, i_idx):
    """Compute K_ii LU and the condensed (e+p) RHS f' (without building K_s).

    Returns (K_ii_lu, f_s, K_ip, K_pi, K_ee, K_ep, K_pe, K_pp, f_e, f_p, f_i).
    """
    K_ee = K[e_idx, :][:, e_idx].tocsc()
    K_ep = K[e_idx, :][:, p_idx].tocsc()
    K_pe = K[p_idx, :][:, e_idx].tocsr()
    K_pp = K[p_idx, :][:, p_idx]
    K_pi = K[p_idx, :][:, i_idx].tocsr()
    K_ip = K[i_idx, :][:, p_idx].tocsc()
    K_ii = K[i_idx, :][:, i_idx].tocsc()

    f_e = f[e_idx]
    f_p = f[p_idx]
    f_i = f[i_idx]

    # Factor K_ii
    K_ii_lu = splu(K_ii)

    # Condensed RHS  f_s = [f_e;  f_p - K_pi * inv(K_ii) * f_i]
    f_s = np.concatenate([f_e, f_p - K_pi.dot(K_ii_lu.solve(f_i))])

    return K_ii_lu, f_s, K_ip, K_pi, K_ee, K_ep, K_pe, K_pp, f_e, f_p, f_i


# ===========================================================================
#  BCI-ROM solve
# ===========================================================================


def bci_rom_solve(K, f, e_idx, p_idx, i_idx, U_r):
    """Solve the BCI-reduced system for one *h* value.

    Parameters
    ----------
    U_r : ndarray  (n_port x r)
        Reduced basis from SVD.

    Returns
    -------
    u_e   : (n_external,)      external DOF temperatures
    u_p   : (n_port,)          recovered port temperatures
    u_i   : (n_internal,)      recovered internal-copper temperatures
    """
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    r = U_r.shape[1]

    # ---- fetch submatrices --------------------------------------------
    K_ee = K[e_idx, :][:, e_idx].tocsc()
    K_ep = K[e_idx, :][:, p_idx].tocsc()
    K_pe = K[p_idx, :][:, e_idx].tocsr()
    K_pp = K[p_idx, :][:, p_idx]
    K_pi = K[p_idx, :][:, i_idx].tocsr()
    K_ip = K[i_idx, :][:, p_idx].tocsc()
    K_ii = K[i_idx, :][:, i_idx].tocsc()
    f_e = f[e_idx]
    f_p = f[p_idx]
    f_i = f[i_idx]

    # ---- factor K_ii --------------------------------------------------
    K_ii_lu = splu(K_ii)

    # ---- build projected (e + r) system -------------------------------

    # 1. K_rom_pp = U_r^T * K_pp * U_r  -  U_r^T * K_pi * inv(K_ii) * K_ip * U_r
    K_pp_proj = U_r.T @ (K_pp @ U_r)  # (r, r)  dense

    # Correction term  --  r solves with K_ii  (cheap since r << n_port)
    #   T = K_ii \ (K_ip * U_r)        (ni x r)
    K_ip_U = K_ip @ U_r  # CSC @ dense  ->  dense (ni x r)
    T = np.column_stack([K_ii_lu.solve(K_ip_U[:, k]) for k in range(r)])
    correction = U_r.T @ (K_pi @ T)  # (r, r)
    K_rom_pp = K_pp_proj - correction

    # 2. coupling blocks
    K_ep_proj = K_ep @ U_r  # (ne, r)  dense
    K_pe_proj = (U_r.T @ K_pe).toarray() if hasattr(U_r.T @ K_pe, 'toarray') else U_r.T @ K_pe  # (r, ne)

    # 3. build K_rom  (dense,  (ne+r) x (ne+r) )
    K_rom = np.zeros((ne + r, ne + r))
    K_rom[:ne, :ne] = K_ee.toarray()
    K_rom[:ne, ne:] = K_ep_proj
    K_rom[ne:, :ne] = K_pe_proj
    K_rom[ne:, ne:] = K_rom_pp

    # 4. condensed RHS projected onto U_r
    #    f_rom_p = U_r^T * f_p  -  U_r^T * K_pi * inv(K_ii) * f_i
    f_i_schur = K_ii_lu.solve(f_i)
    f_rom_p = U_r.T @ (f_p - K_pi.dot(f_i_schur))
    f_rom = np.concatenate([f_e, f_rom_p])

    # 5. solve the reduced system
    sol = np.linalg.solve(K_rom, f_rom)
    u_e = sol[:ne]
    y_r = sol[ne:]

    # 6. reconstruct port & internal
    u_p = U_r @ y_r
    u_i = K_ii_lu.solve(f_i - (K_ip @ u_p).ravel())

    return u_e, u_p, u_i


# ===========================================================================
#  VTU writer  (matches the C++ vtu_writer.cpp order and structure)
#
#  Sweeps ix-iy-iz, only writes active cells, compacts nodes to only
#  those referenced by active cells, body-centered temperature data.
# ===========================================================================


def write_vtu(filename, mesh_x, mesh_y, mesh_z, active_mask, cell_data_dict):
    """Write VTU with hexahedral cells, matching the C++ order and format.

    Only active cells are written. The node set is compacted to only those
    referenced by active cells.  Temperature fields are body-centered CellData.

    Parameters
    ----------
    mesh_x, mesh_y, mesh_z : ndarray
        Vertex coordinates along each axis.
    active_mask : ndarray (nx, ny, nz), bool
        True for cells that belong to the model.
    cell_data_dict : dict[str, ndarray]
        Named arrays of length ``n_active`` (body-centered values).
    """
    nx = len(mesh_x) - 1
    ny = len(mesh_y) - 1
    nz = len(mesh_z) - 1
    node_nx = nx + 1
    node_ny = ny + 1
    node_nz = nz + 1
    total_nodes = node_nx * node_ny * node_nz

    def node_idx(vx, vy, vz):
        """Match C++ formula: vx * nny * nnz + vy * nnz + vz."""
        return vx * node_ny * node_nz + vy * node_nz + vz

    # ---- Pass 1: mark referenced nodes ---------------------------------
    node_remap = np.full(total_nodes, -1_000_000_000, dtype=np.int32)
    INVALID = -1_000_000_000

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not active_mask[ix, iy, iz]:
                    continue
                n = [node_idx(ix, iy, iz),
                     node_idx(ix + 1, iy, iz),
                     node_idx(ix + 1, iy + 1, iz),
                     node_idx(ix, iy + 1, iz),
                     node_idx(ix, iy, iz + 1),
                     node_idx(ix + 1, iy, iz + 1),
                     node_idx(ix + 1, iy + 1, iz + 1),
                     node_idx(ix, iy + 1, iz + 1)]
                for k in range(8):
                    if node_remap[n[k]] == INVALID:
                        node_remap[n[k]] = 0  # referenced

    # ---- Pass 2: compact node remap ------------------------------------
    compact_count = 0
    for i in range(total_nodes):
        if node_remap[i] != INVALID:
            node_remap[i] = compact_count
            compact_count += 1

    num_points = int(compact_count)

    # ---- Node coordinates (points only, ix-iy-iz order) ----------------
    # Node at (vx, vy, vz) is at vertex (mesh_x[vx], mesh_y[vy], mesh_z[vz]).
    # This is equivalent to the C++ reconstruction:
    #   vx==0 → cx[0] - 0.5*dx[0]; vx>0 → cx[vx-1] + 0.5*dx[vx-1]
    # which simplifies to mesh_x[vx].
    coords_lines = []
    for vx in range(node_nx):
        for vy in range(node_ny):
            for vz in range(node_nz):
                i = node_idx(vx, vy, vz)
                if node_remap[i] == INVALID:
                    continue
                coords_lines.append(f"{mesh_x[vx]:.8g} {mesh_y[vy]:.8g} {mesh_z[vz]:.8g}")
    coords_str = "\n".join(coords_lines) + "\n"

    # ---- Cells: connectivity, offsets, types, cell DataArrays ----------
    # All built during the same sweep, matching C++ order.
    conn_lines = []
    off_lines = []
    type_lines = []
    n_active = 0

    # Pre-allocate per-attribute string builders
    cell_arr_strs = {name: [] for name in cell_data_dict}
    # Build active-cell-index -> sequential-counter mapping for data lookup
    # active cell counter in the assembly matches ix-iy-iz sweep order
    n_accum = 0

    # We need a grid_to_active mapping (opposite of active_to_grid)
    # active_mask[ix,iy,iz] True -> compact counter in ix-iy-iz sweep
    grid_to_active = np.full((nx, ny, nz), -1, dtype=np.int32)
    pos = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if active_mask[ix, iy, iz]:
                    grid_to_active[ix, iy, iz] = pos
                    pos += 1

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if not active_mask[ix, iy, iz]:
                    continue

                compact_cell = grid_to_active[ix, iy, iz]

                # Connectivity (8 node indices in VTK hex order)
                n = [node_remap[node_idx(ix, iy, iz)],
                     node_remap[node_idx(ix + 1, iy, iz)],
                     node_remap[node_idx(ix + 1, iy + 1, iz)],
                     node_remap[node_idx(ix, iy + 1, iz)],
                     node_remap[node_idx(ix, iy, iz + 1)],
                     node_remap[node_idx(ix + 1, iy, iz + 1)],
                     node_remap[node_idx(ix + 1, iy + 1, iz + 1)],
                     node_remap[node_idx(ix, iy + 1, iz + 1)]]
                conn_lines.append(f"{n[0]} {n[1]} {n[2]} {n[3]} {n[4]} {n[5]} {n[6]} {n[7]}")

                n_active += 1
                off_lines.append(f"{n_active * 8}")
                type_lines.append("12")

                # CellData attributes
                for name, arr in cell_data_dict.items():
                    cell_arr_strs[name].append(f"{arr[compact_cell]:.8g}")

    conn_str = "\n".join(conn_lines) + "\n"
    off_str = "\n".join(off_lines) + "\n"
    type_str = "\n".join(type_lines) + "\n"

    # ---- Build XML (ascii format, matching C++) -------------------------
    lines = ['<?xml version="1.0"?>']
    lines.append('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">')
    lines.append('  <UnstructuredGrid>')
    lines.append(f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{n_active}">')

    # Points
    lines.append('      <Points>')
    lines.append('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    lines.append(coords_str)
    lines.append('        </DataArray>')
    lines.append('      </Points>')

    # CellData (body-centered)
    lines.append('      <CellData>')
    for name, arr in cell_data_dict.items():
        val_str = "\n".join(cell_arr_strs[name]) + "\n"
        lines.append(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="ascii">')
        lines.append(val_str)
        lines.append('        </DataArray>')
    lines.append('      </CellData>')

    # Cells
    lines.append('      <Cells>')
    lines.append('        <DataArray type="Int32" Name="connectivity" format="ascii">')
    lines.append(conn_str)
    lines.append('        </DataArray>')
    lines.append('        <DataArray type="Int32" Name="offsets" format="ascii">')
    lines.append(off_str)
    lines.append('        </DataArray>')
    lines.append('        <DataArray type="UInt8" Name="types" format="ascii">')
    lines.append(type_str)
    lines.append('        </DataArray>')
    lines.append('      </Cells>')

    lines.append('    </Piece>')
    lines.append('  </UnstructuredGrid>')
    lines.append('</VTKFile>')

    filename.parent.mkdir(parents=True, exist_ok=True)
    raw = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(raw)

    return filename


# ===========================================================================
#  Main
# ===========================================================================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("  BCI-ROM -- Block-Canonical-Insertion ROM for parametric heat sink")
    print("=" * 72)

    # ==================================================================
    #  Phase 0 -- reference model (just to get mesh / grid topology)
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 0: Reference model  (h = 50 W/m^2*K)")
    print("-" * 72)

    ref_model = build_model(50.0)
    ref_compiled = ref_model.compile()
    nc = ref_compiled.cell_count()
    nn = ref_compiled.node_count()
    print(f"  Active cells: {nc}")
    print(f"  Nodes:        {nn}")

    # Mesh from the model
    mesh_x = np.linspace(-50, 150, 51)
    mesh_y = np.linspace(-50, 150, 65)
    mesh_z = np.linspace(0, 30, 10)
    nx, ny, nz = 50, 64, 9

    # Build active mask (geometry follows build_model)
    active_mask = np.zeros((nx, ny, nz), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                xc = (mesh_x[ix] + mesh_x[ix + 1]) / 2.0
                yc = (mesh_y[iy] + mesh_y[iy + 1]) / 2.0
                zc = (mesh_z[iz] + mesh_z[iz + 1]) / 2.0
                # C++ geometry_compiler.cpp: layer 0 (copper) Z=[10,30] (TOP),
                # layer 1 (TIM) Z=[5,10], layer 2 (silicon) Z=[0,5] (BOTTOM)
                if zc >= 10.0:               # copper
                    active_mask[ix, iy, iz] = True
                elif zc >= 5.0:              # TIM
                    active_mask[ix, iy, iz] = (0.0 <= xc <= 100.0 and 0.0 <= yc <= 100.0)
                else:                        # silicon (bottom)
                    r1 = (10.0 <= xc <= 90.0 and 5.0 <= yc <= 45.0)
                    r2 = (10.0 <= xc <= 90.0 and 55.0 <= yc <= 95.0)
                    active_mask[ix, iy, iz] = r1 or r2

    n_active = int(np.sum(active_mask))
    print(f"  Grid:  {nx} x {ny} x {nz} = {nx * ny * nz} cells")
    print(f"  Active: {n_active} cells")

    # Reference assembly
    ref_assembly = ref_compiled.assemble()
    K_ref = ref_assembly.stiffness_matrix()
    f_ref = ref_assembly.rhs().copy()
    ref_assembly.close()

    ref_layer = ref_compiled.layer_ids().copy()
    ref_block = ref_compiled.block_ids().copy()
    e_idx, p_idx, i_idx = partition(K_ref, nc, ref_layer, ref_block,
                                    MACRO_LAYER, MACRO_BLOCK)
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    print(f"\n  Partition:")
    print(f"    External (TIM+silicon):     {ne:>6} DOFs")
    print(f"    Port     (Cu-TIM interface): {np_:>6} DOFs")
    print(f"    Internal (copper interior):  {ni:>6} DOFs")
    print(f"    Copper total:                {np_ + ni:>6} DOFs")

    ref_model.close()

    # ==================================================================
    #  Phase 1 -- Training: collect port snapshots across H_TRAIN
    # ==================================================================
    print("\n" + "-" * 72)
    print(f"Phase 1: Training sweep  ({len(H_TRAIN)} h values)")
    print("-" * 72)

    snapshots = []
    training_times = []
    first = True

    for i, h in enumerate(H_TRAIN):
        t0 = _time.perf_counter()
        print(f"  [{i + 1:2d}/{len(H_TRAIN)}]  h = {h:8.2e} ...", end=" ", flush=True)

        comp, K, f, layer_ids, block_ids = compile_assemble(h)

        if first:
            # Verify partition consistency
            _e, _p, _i = partition(K, comp.cell_count(), layer_ids, block_ids,
                                    MACRO_LAYER, MACRO_BLOCK)
            if len(_e) != ne or len(_p) != np_ or len(_i) != ni:
                print(f"PARTITION MISMATCH at h={h:.2e}!", file=sys.stderr)
                return 1
            first = False

        # Full solve
        sol = comp.solve()
        T_full = sol.cell_temperatures().copy()
        sol.close()
        comp.close()

        # Port snapshot
        u_p = T_full[p_idx].copy()
        snapshots.append(u_p)

        dt = _time.perf_counter() - t0
        training_times.append(dt)
        T_range = (T_full.min(), T_full.max())
        print(f"  [{T_range[0]:.2f}, {T_range[1]:.2f}] K  ({dt:.2f}s)")

    avg_t = np.mean(training_times)
    print(f"\n  Average training time per h: {avg_t:.2f}s")
    print(f"  Total training time:          {np.sum(training_times):.2f}s")

    # ==================================================================
    #  Phase 2 -- SVD of port-snapshot matrix
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 2: SVD on port-snapshot matrix")
    print("-" * 72)

    S = np.column_stack(snapshots)  # (np_ x N_train)
    print(f"  Snapshot matrix: {S.shape[0]} x {S.shape[1]}")

    U, s, Vt = np.linalg.svd(S, full_matrices=False)

    cum_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    r = int(np.searchsorted(cum_energy, ENERGY_THR) + 1)
    r = min(max(r, 2), len(s))

    print(f"\n  Singular values:")
    for k in range(min(r + 3, len(s))):
        label = " <- r" if k == r else ""
        if k < r:
            label = "  v"
        if k < len(s):
            print(f"    s[{k}] = {s[k]:.6e}  cum. energy = {cum_energy[k] * 100:.4f}%{label}")

    print(f"\n  Selected rank r = {r}  (energy > {ENERGY_THR * 100:.2f}%)")
    U_r = U[:, :r]

    # ==================================================================
    #  Phase 3 -- ROM evaluation on test set
    # ==================================================================
    print("\n" + "-" * 72)
    print(f"Phase 3: ROM evaluation  ({len(H_TEST)} test h values)")
    print("-" * 72)

    eval_results_full = {}
    eval_results_rom = {}
    eval_errors = {}

    for j, h_test in enumerate(H_TEST):
        t0 = _time.perf_counter()
        print(f"  [{j + 1}/{len(H_TEST)}]  h = {h_test:10.2e} ...", end=" ", flush=True)

        # Build & assemble for this h
        comp, K, f, layer_ids, block_ids = compile_assemble(h_test)

        # Full solve (reference)
        sol = comp.solve()
        T_full = sol.cell_temperatures().copy()
        sol.close()

        # ROM solve
        u_e, u_p, u_i = bci_rom_solve(K, f, e_idx, p_idx, i_idx, U_r)

        # Reconstruct full cell field
        T_rom = np.zeros(nc, dtype=np.float64)
        T_rom[e_idx] = u_e
        T_rom[p_idx] = u_p
        T_rom[i_idx] = u_i

        # Cell-based error
        diff = T_rom - T_full
        rel_err = np.linalg.norm(diff) / np.linalg.norm(T_full)
        max_err = np.max(np.abs(diff))

        dt = _time.perf_counter() - t0
        print(f"  rel. err = {rel_err:.2e}, max|err| = {max_err:.4e}  ({dt:.2f}s)")

        # Store
        key = f"h_{h_test}"
        eval_results_full[key] = T_full
        eval_results_rom[key] = T_rom
        eval_errors[key] = (rel_err, max_err)

        comp.close()

    # Summary table
    print(f"\n  {'h':>14s}  {'rel. error':>12s}  {'max|err|':>14s}  {'time':>8s}")
    print("  " + "-" * 54)
    for key, (rel, mx) in eval_errors.items():
        h_val = float(key.split("_")[1])
        print(f"  {h_val:>14.2e}  {rel:>12.2e}  {mx:>14.4e}  {'---':>8s}")

    # Print effective dimension reduction
    n_total = ne + np_ + ni
    n_rom = ne + r
    print(f"\n  Dimension reduction (online phase):")
    print(f"    Full system:   {n_total:>6} DOFs")
    print(f"    ROM system:    {n_rom:>6} DOFs  (ne={ne} + r={r})")
    print(f"    Reduction:     {n_rom / n_total * 100:.1f}% of full")

    # ==================================================================
    #  Phase 4 -- VTU export  (body-centered CellData, matching C++ order)
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 4: VTU export  (body-centered cell fields)")
    print("-" * 72)

    # Build mapping from SoA order (ix-iy-iz) to VTU order (ix-iy-iz).
    #
    # The geometry compiler assigns SoA indices by sweeping the 3-D grid
    #   ix ← 0..nx-1, iy ← 0..ny-1, iz ← 0..nz-1
    # and numbering active cells consecutively (C++ geometry_compiler.cpp
    # line 175-177).
    #
    # The C++ VTU writer sweeps the same order (vtu_writer.cpp line 84-86)
    # and uses grid_to_cell[old_idx] which returns the SoA index directly.
    #
    # Therefore the active-cell arrays returned by compiled.solve() et al.
    # are ALREADY in ix-iy-iz order and no reordering is needed.

    # Write full-solution VTU  (CellData, one attribute per h)
    print('  Writing full solutions...')
    full_cell_data = {}
    for key, T_cell in eval_results_full.items():
        h_val = key.replace('h_', '')
        full_cell_data['T_full_' + h_val] = T_cell

    if full_cell_data:
        out = write_vtu(
            OUT_DIR / 'bci_full_solutions.vtu',
            mesh_x, mesh_y, mesh_z, active_mask,
            full_cell_data,
        )
        print(f'  Written: {out}  ({len(full_cell_data)} attributes)')

    # Write ROM results VTU  (CellData: ROM + error + one reference)
    print('  Writing ROM results...')
    rom_cell_data = {}
    for key, T_cell in eval_results_rom.items():
        h_val = key.replace('h_', '')
        rom_cell_data['T_rom_' + h_val] = T_cell

    for key in eval_results_rom:
        T_full_cell = eval_results_full[key]
        T_rom_cell = eval_results_rom[key]
        h_val = key.replace('h_', '')
        rom_cell_data['T_err_' + h_val] = np.abs(T_rom_cell - T_full_cell)

    first_full_key = list(eval_results_full.keys())[0]
    rom_cell_data['T_full_reference'] = eval_results_full[first_full_key]

    if rom_cell_data:
        out = write_vtu(
            OUT_DIR / 'bci_rom_results.vtu',
            mesh_x, mesh_y, mesh_z, active_mask,
            rom_cell_data,
        )
        print(f'  Written: {out}  ({len(rom_cell_data)} attributes)')

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  BCI-ROM demo complete!")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Files:")
    for f in sorted(OUT_DIR.iterdir()):
        sz_mb = f.stat().st_size / 1e6
        print(f"    {f.name}  ({sz_mb:.2f} MB)")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
