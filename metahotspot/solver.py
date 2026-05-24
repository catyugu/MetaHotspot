"""求解器核心：稳态 FVM 求解。"""

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
from metahotspot.exporter import export_vtu
from metahotspot.logger import get_logger

logger = get_logger()


def solve_steady_state(
    sys_mat: SystemMatrix,
    topo: MeshTopology,
    fields: PhysicalFields,
    config: ModelConfig,
    bcs,
    output_vtu: str | None = None,
) -> SimulationResult:
    """纯 FVM 求解：直接求解 cell-centered 温度场，无节点插值，无边界回插。

    Parameters
    ----------
    output_vtu : str | None
        如果指定，则将结果导出为 VTU 文件
    """
    logger.info("Solving linear system...")

    A = sp.csr_matrix(
        (sys_mat.data, sys_mat.col_idx, sys_mat.row_ptr),
        shape=(sys_mat.n_rows, sys_mat.n_cols),
    )
    T_reduced = spla.spsolve(A, sys_mat.b)

    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    k_3d = fields.k.reshape((nx, ny, nz))
    solid_mask = k_3d > 0.0

    T_cell_3d = np.full((nx, ny, nz), np.nan, dtype=np.float64)
    for seq, (i, j, k) in enumerate(zip(*np.where(solid_mask))):
        T_cell_3d[i, j, k] = T_reduced[seq]

    T_flat = T_cell_3d.ravel(order="C")
    valid_T = T_flat[~np.isnan(T_flat)]

    logger.info(f"Done. Max T: {np.max(valid_T):.2f} K, Min T: {np.min(valid_T):.2f} K")

    if output_vtu:
        export_vtu(T_reduced, topo, fields, config, output_vtu)

    return SimulationResult(
        temperatures=T_flat,
        time_series=np.empty(0),
        max_temperature=float(np.max(valid_T)),
        min_temperature=float(np.min(valid_T)),
        max_index=int(np.nanargmax(T_flat)),
        min_index=int(np.nanargmin(T_flat)),
    )
