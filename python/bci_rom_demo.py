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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import metahotspot
from metahotspot import enums

# ===========================================================================
#  Configuration
# ===========================================================================

OUT_DIR = Path(__file__).resolve().parent / "bci_rom_output"

H_TRAIN = np.geomspace(1.0, 1e6, 20)
H_TEST = np.array([2.5, 15.0, 1.0e2, 3.0e3, 8.0e4, 2.5e5])

MACRO_LAYER = 0  # copper
MACRO_BLOCK = 0  # only block in layer 0

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
    m.set_mesh(
        x=np.linspace(-50, 150, 51),
        y=np.linspace(-50, 150, 65),
        z=np.linspace(0, 30, 10),
    )

    for name, k in [("copper", "400"), ("TIM", "5"), ("silicon", "130")]:
        m.add_material(
            name,
            kx=k,
            ky=k,
            kz=k,
            rho="8920" if name == "copper" else "2330" if name == "silicon" else "1200",
            c="385" if name == "copper" else "0.71" if name == "silicon" else "1000",
        )

    lid0 = m.add_layer(thickness="20")  # copper  Z=[10,30] (top)
    lid1 = m.add_layer(thickness="5")  # TIM     Z=[5,10]
    lid2 = m.add_layer(thickness="5")  # silicon Z=[0,5]  (bottom with heat sources)

    b0 = m.add_block(lid0, "copper")
    b1 = m.add_block(lid1, "TIM")
    b2 = m.add_block(lid2, "silicon", heat_source="1e6", x_offset="10", y_offset="5")
    b3 = m.add_block(lid2, "silicon", heat_source="2e6", x_offset="10", y_offset="55")

    m.add_rect(b0, x="-50", y="-50", width="200", height="200")
    m.add_rect(b1, x="0", y="0", width="100", height="100")
    m.add_rect(b2, x="0", y="0", width="80", height="40")
    m.add_rect(b3, x="0", y="0", width="80", height="40")

    _regions = [
        (enums.Axis.Z, 30.0, -50, 150, -50, 150),
    ]
    m.add_convection(
        coefficient=str(h_value), ambient_temperature="300", regions=_regions
    )
    return m


# ===========================================================================
#  Partitioning  —  cell sets via grid_to_cell
# ===========================================================================


def partition(compiled, nc, macro_layer, macro_block):
    """Split mesh into external (e), port (p), and internal (i) sets.

    Port  = macro cells adjacent to at least one non-macro cell.
    Internal = macro cells *not* adjacent to any non-macro cell.
    External = all non-macro cells.

    Uses grid_to_cell() topology directly — no need to replicate the
    geometry compiler's mask logic.
    """
    K = compiled.assemble(compiled.default_state()).K
    meta = compiled.metadata()
    layer_ids = meta.layer_ids
    block_ids = meta.block_ids

    is_macro = (layer_ids == macro_layer) & (block_ids == macro_block)
    macro_set = set(np.where(is_macro)[0])

    port_set = set()
    for i in macro_set:
        for j in K.indices[K.indptr[i] : K.indptr[i + 1]]:
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
#  BCI-ROM solve
# ===========================================================================


def bci_rom_solve(K, f, e_idx, p_idx, i_idx, U_r):
    """Solve the BCI-reduced system for a given *h* value.

    Parameters
    ----------
    U_r : ndarray  (n_port x r)
        Reduced basis from SVD.
    """
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)
    r = U_r.shape[1]

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

    K_ii_lu = splu(K_ii)

    # Projected (e + r) system
    K_pp_proj = U_r.T @ (K_pp @ U_r)  # (r,r)
    K_ip_U = K_ip @ U_r  # (ni, r)
    T = np.column_stack([K_ii_lu.solve(K_ip_U[:, k]) for k in range(r)])
    correction = U_r.T @ (K_pi @ T)  # (r,r)
    K_rom_pp = K_pp_proj - correction

    K_ep_proj = K_ep @ U_r  # (ne, r)
    K_pe_proj = U_r.T @ K_pe  # (r, ne)

    K_rom = np.zeros((ne + r, ne + r))
    K_rom[:ne, :ne] = K_ee.toarray()
    K_rom[:ne, ne:] = K_ep_proj
    K_rom[ne:, :ne] = K_pe_proj
    K_rom[ne:, ne:] = K_rom_pp

    f_i_schur = K_ii_lu.solve(f_i)
    f_rom_p = U_r.T @ (f_p - K_pi.dot(f_i_schur))
    f_rom = np.concatenate([f_e, f_rom_p])

    sol = np.linalg.solve(K_rom, f_rom)
    u_e = sol[:ne]
    u_p = U_r @ sol[ne:]
    u_i = K_ii_lu.solve(f_i - (K_ip @ u_p).ravel())

    return u_e, u_p, u_i


# ===========================================================================
#  VTU writer  (matches C++ : ix-iy-iz sweep, active cells only)
# ===========================================================================


def write_vtu(filename, mesh_x, mesh_y, mesh_z, grid_to_cell, cell_data_dict):
    """Write VTU with hexahedral cells, matching the C++ order and format.

    ``grid_to_cell`` is ``Compiled.grid_to_cell().reshape(nx, ny, nz)``
    — a 3-D array where ``grid_to_cell[ix, iy, iz]`` is the SoA index
    or ``SIZE_MAX`` for inactive cells.
    ``cell_data_dict`` values are already in SoA (ix-iy-iz) order.
    """
    nx, ny, nz = len(mesh_x) - 1, len(mesh_y) - 1, len(mesh_z) - 1
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    total_nodes = nnx * nny * nnz

    def nidx(vx, vy, vz):
        return vx * nny * nnz + vy * nnz + vz

    # Pass 1: mark referenced nodes
    INVALID = -1_000_000_000
    remap = np.full(total_nodes, INVALID, dtype=np.int32)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if grid_to_cell[ix, iy, iz] >= np.iinfo(np.uintp).max:
                    continue
                for k in [
                    nidx(ix, iy, iz),
                    nidx(ix + 1, iy, iz),
                    nidx(ix + 1, iy + 1, iz),
                    nidx(ix, iy + 1, iz),
                    nidx(ix, iy, iz + 1),
                    nidx(ix + 1, iy, iz + 1),
                    nidx(ix + 1, iy + 1, iz + 1),
                    nidx(ix, iy + 1, iz + 1),
                ]:
                    if remap[k] == INVALID:
                        remap[k] = 0

    # Pass 2: compact
    compact = 0
    for i in range(total_nodes):
        if remap[i] != INVALID:
            remap[i] = compact
            compact += 1
    num_points = compact

    # Node coordinates (ix-iy-iz)
    coords_lines = []
    for vx in range(nnx):
        for vy in range(nny):
            for vz in range(nnz):
                if remap[nidx(vx, vy, vz)] != INVALID:
                    coords_lines.append(
                        f"{mesh_x[vx]:.8g} {mesh_y[vy]:.8g} {mesh_z[vz]:.8g}"
                    )

    # Cells + CellData (same ix-iy-iz sweep)
    conn_lines, off_lines, type_lines = [], [], []
    cell_vals = {name: [] for name in cell_data_dict}
    n_active = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                soa = grid_to_cell[ix, iy, iz]
                if soa >= np.iinfo(np.uintp).max:
                    continue
                n = [
                    remap[nidx(ix, iy, iz)],
                    remap[nidx(ix + 1, iy, iz)],
                    remap[nidx(ix + 1, iy + 1, iz)],
                    remap[nidx(ix, iy + 1, iz)],
                    remap[nidx(ix, iy, iz + 1)],
                    remap[nidx(ix + 1, iy, iz + 1)],
                    remap[nidx(ix + 1, iy + 1, iz + 1)],
                    remap[nidx(ix, iy + 1, iz + 1)],
                ]
                conn_lines.append(
                    f"{n[0]} {n[1]} {n[2]} {n[3]} {n[4]} {n[5]} {n[6]} {n[7]}"
                )
                n_active += 1
                off_lines.append(f"{n_active * 8}")
                type_lines.append("12")
                for name, arr in cell_data_dict.items():
                    cell_vals[name].append(f"{arr[soa]:.8g}")

    # XML
    lines = ['<?xml version="1.0"?>']
    lines.append(
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
    )
    lines.append("  <UnstructuredGrid>")
    lines.append(
        f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{n_active}">'
    )
    lines.append(
        '      <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
    )
    lines.append("\n".join(coords_lines) + "\n")
    lines.append("        </DataArray></Points>")
    lines.append("      <CellData>")
    for name, vals in cell_vals.items():
        lines.append(
            f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="ascii">'
        )
        lines.append("\n".join(vals) + "\n")
        lines.append("        </DataArray>")
    lines.append("      </CellData>")
    lines.append("      <Cells>")
    lines.append(
        f'        <DataArray type="Int32" Name="connectivity" format="ascii">{" ".join(conn_lines)}\n</DataArray>'
    )
    lines.append(
        f'        <DataArray type="Int32" Name="offsets" format="ascii">{" ".join(off_lines)}\n</DataArray>'
    )
    lines.append(
        f'        <DataArray type="UInt8" Name="types" format="ascii">{" ".join(type_lines)}\n</DataArray>'
    )
    lines.append("      </Cells>")
    lines.append("    </Piece>")
    lines.append("  </UnstructuredGrid>")
    lines.append("</VTKFile>")

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
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
    #  Phase 0 — reference model
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 0: Reference model  (h = 50 W/m2K)")
    print("-" * 72)

    ref = build_model(50.0)
    comp_ref = ref.compile()
    meta = comp_ref.metadata()
    nc = meta.cell_count

    nx, ny, nz = 50, 64, 9
    mesh_x = np.linspace(-50, 150, 51)
    mesh_y = np.linspace(-50, 150, 65)
    mesh_z = np.linspace(0, 30, 10)

    # grid_to_cell is flat in ix-iy-iz order:
    #   idx = ix * ny * nz + iy * nz + iz
    # Reshape to (nx, ny, nz) gives g2c[ix, iy, iz] = SoA index or SIZE_MAX
    g2c_3d = meta.grid_to_cell.reshape(nx, ny, nz)

    e_idx, p_idx, i_idx = partition(comp_ref, nc, MACRO_LAYER, MACRO_BLOCK)
    ne, np_, ni = len(e_idx), len(p_idx), len(i_idx)

    print(f"  Active cells: {nc}  (grid {nx}x{ny}x{nz} = {nx*ny*nz})")
    print(f"  External (TIM+silicon):     {ne:>6} DOFs")
    print(f"  Port     (Cu-TIM interface): {np_:>6} DOFs")
    print(f"  Internal (copper interior):  {ni:>6} DOFs")

    ref.close()

    # ==================================================================
    #  Phase 1 — Training sweep
    # ==================================================================
    print("\n" + "-" * 72)
    print(f"Phase 1: Training sweep  ({len(H_TRAIN)} h values)")
    print("-" * 72)

    snapshots = []
    times = []

    for i, h in enumerate(H_TRAIN):
        t0 = _time.perf_counter()
        print(f"  [{i + 1:2d}/{len(H_TRAIN)}]  h = {h:8.2e} ...", end=" ", flush=True)

        m = build_model(h)
        c = m.compile()
        sol = c.solve()
        T = sol.temperature.copy()
        sol.close()
        c.close()
        m.close()

        snapshots.append(T[p_idx].copy())
        dt = _time.perf_counter() - t0
        times.append(dt)
        print(f"  [{T.min():.2f}, {T.max():.2f}] K  ({dt:.2f}s)")

    print(f"\n  Total training: {np.sum(times):.2f}s  ({np.mean(times):.2f}s/h)")

    # ==================================================================
    #  Phase 2 — SVD
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 2: SVD on port-snapshot matrix")
    print("-" * 72)

    S = np.column_stack(snapshots)
    print(f"  Snapshot matrix: {S.shape[0]} x {S.shape[1]}")

    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    cum = np.cumsum(s**2) / np.sum(s**2)
    r = min(max(int(np.searchsorted(cum, ENERGY_THR) + 1), 2), len(s))

    for k in range(min(len(s), r + 3)):
        print(
            f"    s[{k}] = {s[k]:.6e}  cum. energy = {cum[k] * 100:.4f}%{'  v' if k < r else ' <- r' if k == r else ''}"
        )

    print(f"\n  Selected rank r = {r}  (energy > {ENERGY_THR * 100:.1f}%)")
    U_r = U[:, :r]

    # ==================================================================
    #  Phase 3 — ROM evaluation
    # ==================================================================
    print("\n" + "-" * 72)
    print(f"Phase 3: ROM evaluation  ({len(H_TEST)} test h values)")
    print("-" * 72)

    T_full, T_rom, errors = {}, {}, {}

    for j, h_test in enumerate(H_TEST):
        t0 = _time.perf_counter()
        print(f"  [{j + 1}/{len(H_TEST)}]  h = {h_test:10.2e} ...", end=" ", flush=True)

        m = build_model(h_test)
        c = m.compile()
        assembly = c.assemble(c.default_state())
        K, C, f = assembly.K, assembly.C, assembly.f
        assembly.close()
        sol = c.solve()
        Tf = sol.temperature.copy()
        sol.close()

        u_e, u_p, u_i = bci_rom_solve(K, f, e_idx, p_idx, i_idx, U_r)
        Tr = np.zeros(nc)
        Tr[e_idx], Tr[p_idx], Tr[i_idx] = u_e, u_p, u_i

        diff = Tr - Tf
        rel = np.linalg.norm(diff) / np.linalg.norm(Tf)
        mx = np.max(np.abs(diff))
        dt = _time.perf_counter() - t0
        print(f"  rel. err = {rel:.2e}, max|err| = {mx:.4e}  ({dt:.2f}s)")

        key = f"h_{h_test}"
        T_full[key] = Tf
        T_rom[key] = Tr
        errors[key] = (rel, mx)
        c.close()
        m.close()

    # Summary
    print(f"\n  {'h':>14s}  {'rel. error':>12s}  {'max|err|':>14s}")
    print("  " + "-" * 42)
    for k, (rel, mx) in errors.items():
        print(f"  {float(k.split('_')[1]):>14.2e}  {rel:>12.2e}  {mx:>14.4e}")

    n_total = ne + np_ + ni
    print(
        f"\n  Reduction: {ne + r} / {n_total} DOFs  ({100 * (ne + r) / n_total:.1f}%)"
    )

    # ==================================================================
    #  Phase 4 — VTU export
    # ==================================================================
    print("\n" + "-" * 72)
    print("Phase 4: VTU export")
    print("-" * 72)

    print("  Writing full solutions...")
    write_vtu(
        OUT_DIR / "bci_full_solutions.vtu",
        mesh_x,
        mesh_y,
        mesh_z,
        g2c_3d,
        {f"T_full_{k.replace('h_', '')}": v for k, v in T_full.items()},
    )

    cd = {f"T_rom_{k.replace('h_', '')}": v for k, v in T_rom.items()}
    for k in T_rom:
        cd[f"T_err_{k.replace('h_', '')}"] = np.abs(T_rom[k] - T_full[k])
    cd["T_full_reference"] = T_full[list(T_full.keys())[0]]

    print("  Writing ROM results...")
    write_vtu(OUT_DIR / "bci_rom_results.vtu", mesh_x, mesh_y, mesh_z, g2c_3d, cd)

    # ==================================================================
    #  Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("  BCI-ROM demo complete!")
    print(f"  Output: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
