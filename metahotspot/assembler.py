"""组装器：构建一维热阻网格系统矩阵 (A x = b)。"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

from metahotspot.metahotspot_types import *
from metahotspot.config import TOL
from metahotspot.units import UnitConverter
from metahotspot.logger import get_logger

logger = get_logger()

def find_bc(axis: str, coord_m: float, u_m: float, v_m: float, bcs: List[ParsedFaceKey], uc: UnitConverter) -> dict | None:
    """精确匹配暴露面所属的热边界条件。"""
    coord_origin = uc.from_m(coord_m)
    u_origin = uc.from_m(u_m)
    v_origin = uc.from_m(v_m)

    for bc in bcs:
        if bc.axis == axis and abs(bc.coord - coord_origin) <= TOL.geom_tol:
            for r in bc.rects:
                if (r[0] - TOL.geom_tol <= u_origin <= r[1] + TOL.geom_tol) and \
                   (r[2] - TOL.geom_tol <= v_origin <= r[3] + TOL.geom_tol):
                    return bc.params
    return None

def apply_boundary_condition(bc: dict, A_face: float, d_half: float, k_cell: float) -> Tuple[float, float]:
    """独立边界解析，返回(对角线电导累加量, 右端项 b 累加量)"""
    if bc['type'] == 'first':
        r_bc = d_half / (k_cell * A_face)
        cond = 1.0 / r_bc
        return cond, bc['temperature'] * cond
    elif bc['type'] == 'second':
        return 0.0, bc['heat_flux'] * A_face
    elif bc['type'] == 'third':
        r_bc = d_half / (k_cell * A_face) + 1.0 / (bc['h_conv'] * A_face)
        cond = 1.0 / r_bc
        return cond, bc['t_inf'] * cond
    return 0.0, 0.0

def assemble_system(topo: MeshTopology, fields: PhysicalFields, bcs: List[ParsedFaceKey], config: ModelConfig) -> SystemMatrix:
    """构建 CTM 稀疏系统矩阵。"""
    logger.info("Assembling system matrix...")
    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    uc = UnitConverter(config.length_unit)

    x_m = uc.to_m(topo.coords.x)
    y_m = uc.to_m(topo.coords.y)
    z_m = uc.to_m(topo.coords.z)
    cx = (x_m[:-1] + x_m[1:]) / 2.0
    cy = (y_m[:-1] + y_m[1:]) / 2.0
    cz = (z_m[:-1] + z_m[1:]) / 2.0
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    dz = np.diff(z_m)

    k_3d = fields.k.reshape((nx, ny, nz))
    hs_3d = fields.heat_source.reshape((nx, ny, nz))
    solid_mask = k_3d > 0.0

    reduced_idx_map = -np.ones((nx, ny, nz), dtype=np.int32)
    solid_indices = np.where(solid_mask)
    n_solid = len(solid_indices[0])
    reduced_idx_map[solid_mask] = np.arange(n_solid)

    A_diag = np.zeros(n_solid, dtype=np.float64)
    b = np.zeros(n_solid, dtype=np.float64)
    rows, cols, data = [], [], []

    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        vol = dx[i] * dy[j] * dz[k]
        b[seq] += hs_3d[i, j, k] * vol

        k_c = k_3d[i, j, k]

        # X 轴向
        A_x = dy[j] * dz[k]
        d_x_i = dx[i] / 2.0
        if i + 1 < nx and solid_mask[i+1, j, k]:
            seq_n = reduced_idx_map[i+1, j, k]
            r_ij = d_x_i / (k_c * A_x) + (dx[i+1]/2.0) / (k_3d[i+1, j, k] * A_x)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if i == 0 or not solid_mask[i-1, j, k]:
            if bc := find_bc('X', x_m[i], cy[j], cz[k], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_x, d_x_i, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b
        if i == nx - 1 or not solid_mask[i+1, j, k]:
            if bc := find_bc('X', x_m[i+1], cy[j], cz[k], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_x, d_x_i, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b

        # Y 轴向
        A_y = dx[i] * dz[k]
        d_y_j = dy[j] / 2.0
        if j + 1 < ny and solid_mask[i, j+1, k]:
            seq_n = reduced_idx_map[i, j+1, k]
            r_ij = d_y_j / (k_c * A_y) + (dy[j+1]/2.0) / (k_3d[i, j+1, k] * A_y)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if j == 0 or not solid_mask[i, j-1, k]:
            if bc := find_bc('Y', y_m[j], cx[i], cz[k], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_y, d_y_j, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b
        if j == ny - 1 or not solid_mask[i, j+1, k]:
            if bc := find_bc('Y', y_m[j+1], cx[i], cz[k], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_y, d_y_j, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b

        # Z 轴向
        A_z = dx[i] * dy[j]
        d_z_k = dz[k] / 2.0
        if k + 1 < nz and solid_mask[i, j, k+1]:
            seq_n = reduced_idx_map[i, j, k+1]
            r_ij = d_z_k / (k_c * A_z) + (dz[k+1]/2.0) / (k_3d[i, j, k+1] * A_z)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if k == 0 or not solid_mask[i, j, k-1]:
            if bc := find_bc('Z', z_m[k], cx[i], cy[j], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_z, d_z_k, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b
        if k == nz - 1 or not solid_mask[i, j, k+1]:
            if bc := find_bc('Z', z_m[k+1], cx[i], cy[j], bcs, uc):
                add_cond, add_b = apply_boundary_condition(bc, A_z, d_z_k, k_c)
                A_diag[seq] += add_cond; b[seq] += add_b

    rows.extend(range(n_solid))
    cols.extend(range(n_solid))
    data.extend(A_diag)

    A_csr = sp.coo_matrix((data, (rows, cols)), shape=(n_solid, n_solid)).tocsr()

    return SystemMatrix(
        n_rows=n_solid,
        n_cols=n_solid,
        row_ptr=A_csr.indptr,
        col_idx=A_csr.indices,
        data=A_csr.data,
        b=b,
        power_row_ptr=np.array([]),
        power_col_idx=np.array([]),
        power_data=np.array([])
    )