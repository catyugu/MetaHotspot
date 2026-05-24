"""组装器：构建一维热阻网格系统矩阵 (A x = b)。"""

import numpy as np
import scipy.sparse as sp
from typing import List

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

def assemble_system(topo: MeshTopology, fields: PhysicalFields, bcs: List[ParsedFaceKey], config: ModelConfig) -> SystemMatrix:
    """构建 CTM 稀疏系统矩阵。"""
    logger.info("Assembling system matrix...")
    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    uc = UnitConverter(config.length_unit)
    
    # 三维坐标和间距恢复 (单位: 米)
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
    
    # 建立活动网格的压缩索引映射 (跳过空气槽/空洞)
    reduced_idx_map = -np.ones((nx, ny, nz), dtype=np.int32)
    solid_indices = np.where(solid_mask)
    n_solid = len(solid_indices[0])
    
    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        reduced_idx_map[i, j, k] = seq
        
    A_diag = np.zeros(n_solid, dtype=np.float64)
    b = np.zeros(n_solid, dtype=np.float64)
    rows, cols, data = [], [], []
    
    def process_boundary(axis: str, coord_m: float, u_m: float, v_m: float, 
                         A_face: float, d_half: float, k_cell: float, seq: int):
        bc = find_bc(axis, coord_m, u_m, v_m, bcs, uc)
        if not bc: return # 绝热默认

        if bc['type'] == 'first':
            r_bc = d_half / (k_cell * A_face)
            cond = 1.0 / r_bc
            A_diag[seq] += cond
            b[seq] += bc['temperature'] * cond
        elif bc['type'] == 'second':
            b[seq] += bc['heat_flux'] * A_face
        elif bc['type'] == 'third':
            r_bc = d_half / (k_cell * A_face) + 1.0 / (bc['h_conv'] * A_face)
            cond = 1.0 / r_bc
            A_diag[seq] += cond
            b[seq] += bc['t_inf'] * cond

    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        vol = dx[i] * dy[j] * dz[k]
        b[seq] += hs_3d[i, j, k] * vol
        
        # 内部方向连结与界面检查
        # X 轴向
        A_x = dy[j] * dz[k]
        d_x_i = dx[i] / 2.0
        if i + 1 < nx and solid_mask[i+1, j, k]:
            seq_n = reduced_idx_map[i+1, j, k]
            r_ij = d_x_i / (k_3d[i,j,k] * A_x) + (dx[i+1]/2.0) / (k_3d[i+1,j,k] * A_x)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if i == 0 or not solid_mask[i-1, j, k]:
            process_boundary('X', x_m[i], cy[j], cz[k], A_x, d_x_i, k_3d[i,j,k], seq)
        if i == nx - 1 or not solid_mask[i+1, j, k]:
            process_boundary('X', x_m[i+1], cy[j], cz[k], A_x, d_x_i, k_3d[i,j,k], seq)

        # Y 轴向
        A_y = dx[i] * dz[k]
        d_y_i = dy[j] / 2.0
        if j + 1 < ny and solid_mask[i, j+1, k]:
            seq_n = reduced_idx_map[i, j+1, k]
            r_ij = d_y_i / (k_3d[i,j,k] * A_y) + (dy[j+1]/2.0) / (k_3d[i,j+1,k] * A_y)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if j == 0 or not solid_mask[i, j-1, k]:
            process_boundary('Y', y_m[j], cx[i], cz[k], A_y, d_y_i, k_3d[i,j,k], seq)
        if j == ny - 1 or not solid_mask[i, j+1, k]:
            process_boundary('Y', y_m[j+1], cx[i], cz[k], A_y, d_y_i, k_3d[i,j,k], seq)

        # Z 轴向
        A_z = dx[i] * dy[j]
        d_z_i = dz[k] / 2.0
        if k + 1 < nz and solid_mask[i, j, k+1]:
            seq_n = reduced_idx_map[i, j, k+1]
            r_ij = d_z_i / (k_3d[i,j,k] * A_z) + (dz[k+1]/2.0) / (k_3d[i,j,k+1] * A_z)
            cond = 1.0 / r_ij
            rows.extend([seq, seq_n]); cols.extend([seq_n, seq]); data.extend([-cond, -cond])
            A_diag[seq] += cond; A_diag[seq_n] += cond
        if k == 0 or not solid_mask[i, j, k-1]:
            process_boundary('Z', z_m[k], cx[i], cy[j], A_z, d_z_i, k_3d[i,j,k], seq)
        if k == nz - 1 or not solid_mask[i, j, k+1]:
            process_boundary('Z', z_m[k+1], cx[i], cy[j], A_z, d_z_i, k_3d[i,j,k], seq)
            
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