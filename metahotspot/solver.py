"""求解器核心：稳态与节点插值后处理。"""

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from metahotspot.metahotspot_types import (
    SystemMatrix,
    MeshTopology,
    PhysicalFields,
    SimulationResult,
    ModelConfig,
)
from metahotspot.assembler import find_bc
from metahotspot.units import UnitConverter
from metahotspot.logger import get_logger

logger = get_logger()


def solve_system(
    sys_mat: SystemMatrix,
    topo: MeshTopology,
    fields: PhysicalFields,
    config: ModelConfig,
    bcs,
) -> SimulationResult:
    logger.info("Solving linear system...")

    A = sp.csr_matrix(
        (sys_mat.data, sys_mat.col_idx, sys_mat.row_ptr),
        shape=(sys_mat.n_rows, sys_mat.n_cols),
    )
    T_reduced = spla.spsolve(A, sys_mat.b)

    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    k_3d = fields.k.reshape((nx, ny, nz))
    solid_mask = k_3d > 0.0
    solid_indices = np.where(solid_mask)

    T_cell_3d = np.full((nx, ny, nz), np.nan, dtype=np.float64)
    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        T_cell_3d[i, j, k] = T_reduced[seq]

    logger.info("Interpolating cell-centered field to nodal mesh (Vectorized)...")

    T_nodes_3d = np.full((topo.n_x, topo.n_y, topo.n_z), np.nan, dtype=np.float64)

    for i in range(topo.n_x):
        for j in range(topo.n_y):
            for k in range(topo.n_z):
                vals, cnt = [], 0
                for di in [-1, 0]:
                    for dj in [-1, 0]:
                        for dk in [-1, 0]:
                            ci, cj, ck = i + di, j + dj, k + dk
                            if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
                                if solid_mask[ci, cj, ck]:
                                    vals.append(T_cell_3d[ci, cj, ck])
                if vals:
                    T_nodes_3d[i, j, k] = np.mean(vals)

    uc = UnitConverter(config.length_unit)
    x_m = uc.to_m(topo.coords.x)
    y_m = uc.to_m(topo.coords.y)
    z_m = uc.to_m(topo.coords.z)

    for i in range(topo.n_x):
        for j in range(topo.n_y):
            for k in range(topo.n_z):
                if np.isnan(T_nodes_3d[i, j, k]):
                    continue

                if i == 0 or i == topo.n_x - 1:
                    bc = find_bc("X", x_m[i], y_m[j], z_m[k], bcs, uc)
                    if bc and bc["type"] == "first":
                        T_nodes_3d[i, j, k] = bc["temperature"]
                if j == 0 or j == topo.n_y - 1:
                    bc = find_bc("Y", y_m[j], x_m[i], z_m[k], bcs, uc)
                    if bc and bc["type"] == "first":
                        T_nodes_3d[i, j, k] = bc["temperature"]
                if k == 0 or k == topo.n_z - 1:
                    bc = find_bc("Z", z_m[k], x_m[i], y_m[j], bcs, uc)
                    if bc and bc["type"] == "first":
                        T_nodes_3d[i, j, k] = bc["temperature"]

    T_flat = T_nodes_3d.ravel(order="C")
    valid_T = T_flat[~np.isnan(T_flat)]

    logger.info(f"Done. Max T: {np.max(valid_T):.2f} K, Min T: {np.min(valid_T):.2f} K")

    return SimulationResult(
        temperatures=T_flat,
        time_series=np.empty(0),
        max_temperature=float(np.max(valid_T)),
        min_temperature=float(np.min(valid_T)),
        max_index=int(np.nanargmax(T_flat)),
        min_index=int(np.nanargmin(T_flat)),
    )
