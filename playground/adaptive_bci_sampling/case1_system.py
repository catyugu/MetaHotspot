"""Shared Case1 matrix reconstruction; no experiment or certificate imports.

Matches the previously archived assembler formulas. Native validation remains
explicit metadata; extracting this helper does not change the physical model.
"""
import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]/'python'), str(HERE.parent/'bci_rom_testcase1')]


def case1_reconstruction(mesh_mm):
    """Matrix-only reconstruction from Case1Config, matching assembler formulas.

    Native C API is not required. This is NOT claimed to be independently
    validated against the native assembled matrices in this environment.
    Air-filled background, anisotropy, harmonic faces and source volumes follow
    model_case1.py/assembler.cpp. The native builder should be used when available.
    """
    from model_case1 import Case1Config, MATERIALS, DIES, LAYERS
    cfg = Case1Config(max_xy_cell_mm=mesh_mm, max_z_cell_mm=mesh_mm)
    axes = [cfg.x_vertices_mm*1e-3, cfg.y_vertices_mm*1e-3, cfg.z_vertices_mm*1e-3]
    widths = [np.diff(a) for a in axes]
    centers = [(a[:-1]+a[1:])/2 for a in axes]
    x, y, z = np.meshgrid(*centers, indexing='ij')
    shape = x.shape
    ids = np.arange(x.size).reshape(shape)
    dx, dy, dz = np.meshgrid(*widths, indexing='ij')
    volume = dx*dy*dz
    conductivity = np.full((*shape, 3), .02643)
    rho_cp = np.full(shape, 1.149*1007)
    bottom = 0.
    for thickness, material, sx, sy, cx, cy, _ in LAYERS:
        mask = ((z >= bottom) & (z < bottom+thickness*1e-3)
                & (abs(x-cx*1e-3) < sx*5e-4) & (abs(y-cy*1e-3) < sy*5e-4))
        kx, ky, kz, rho, cp = map(float, MATERIALS[material])
        conductivity[mask] = [kx, ky, kz]
        rho_cp[mask] = rho*cp
        bottom += thickness*1e-3
    G = np.zeros((x.size, 4))
    top_mask = np.zeros(shape, bool)
    for j, (_, (xlo, xhi), (ylo, yhi)) in enumerate(DIES):
        mask = ((z >= .018) & (x >= xlo*1e-3) & (x <= xhi*1e-3)
                & (y >= ylo*1e-3) & (y <= yhi*1e-3))
        kx, ky, kz, rho, cp = map(float, MATERIALS['Silicon'])
        conductivity[mask] = [kx, ky, kz]
        rho_cp[mask] = rho*cp
        G[mask.ravel(), j] = volume[mask]/volume[mask].sum()
        top_mask |= mask
    rows, cols, data = [], [], []
    for axis in range(3):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(0, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        sizes = [dx, dy, dz]
        area = volume[left]/sizes[axis][left]
        g = area/(sizes[axis][left]/(2*conductivity[left+(axis,)])
                  + sizes[axis][right]/(2*conductivity[right+(axis,)]))
        i, j, g = ids[left].ravel(), ids[right].ravel(), g.ravel()
        rows.extend([i, j, i, j]); cols.extend([i, j, j, i]); data.extend([g, g, -g, -g])
    K = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(x.size, x.size)).tocsc()
    C = sp.diags((rho_cp*volume).ravel()).tocsc()
    top = top_mask & (z == centers[2][-1])
    bot = z == centers[2][0]
    area = dx*dy
    H = [sp.diags(np.where(mask, area, 0).ravel()).tocsc() for mask in [top, bot]]
    ranges = []
    for mask in [top, bot]:
        k, half = conductivity[mask, 2], dz[mask]/2
        weights = area[mask]
        ranges.append([float(np.dot(weights, k*h/(k+h*half))/weights.sum())
                       for h in [1., 1e4]])
    return K, C, G, H, np.asarray(ranges), {
        'model': 'Case1 matrix-only reconstruction', 'native_validated': False,
        'mesh_mm': mesh_mm, 'shape': list(shape),
        'parameter': 'effective Robin coefficient p; physical HTC endpoints [1,1e4]',
        'source_column_sums': G.sum(axis=0).tolist(),
        'K_constant_defect': float(la.norm(K @ np.ones(x.size))),
    }
