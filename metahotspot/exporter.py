"""VTU 后处理模块：导出体温度数据和网格到 VTU 格式。"""

import os
import numpy as np

from metahotspot.metahotspot_types import MeshTopology, PhysicalFields, ModelConfig
from metahotspot.units import UnitConverter
from metahotspot.logger import get_logger

try:
    import meshio
except ImportError:
    meshio = None

logger = get_logger()


def export_vtu(
    temperatures: np.ndarray,
    topo: MeshTopology,
    fields: PhysicalFields,
    config: ModelConfig,
    output_path: str,
) -> None:
    """将温度场和网格导出为 VTU 格式（可用 Paraview 打开）。

    Parameters
    ----------
    temperatures : np.ndarray
        shape: (n_cells,) - 单元格中心温度 (K)
    topo : MeshTopology
        网格拓扑信息
    fields : PhysicalFields
        物理场数据（用于材料可视化和 solid mask）
    config : ModelConfig
        模型配置
    output_path : str
        输出文件路径（.vtu 后缀）
    """
    if meshio is None:
        logger.warning("meshio not installed, skipping VTU export")
        return

    logger.info(f"Exporting VTU to {output_path}")

    uc = UnitConverter(config.length_unit)
    x_m = uc.to_m(topo.coords.x)
    y_m = uc.to_m(topo.coords.y)
    z_m = uc.to_m(topo.coords.z)

    # solid 掩码
    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    k_3d = fields.k.reshape((nx, ny, nz))
    solid_mask = k_3d > 0.0
    solid_indices = np.where(solid_mask)
    n_solid = solid_indices[0].size

    # 构建节点坐标 (n_x * n_y * n_z, 3) via broadcasting
    X, Y, Z = np.meshgrid(x_m, y_m, z_m, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # 六面体节点编号（按 VTK 逆时针环绕约定）— 向量化
    i_idx, j_idx, k_idx = solid_indices
    n_x = topo.n_x
    n_xy = n_x * topo.n_y
    n0 = i_idx + j_idx * n_x + k_idx * n_xy
    n1 = (i_idx + 1) + j_idx * n_x + k_idx * n_xy
    n2 = (i_idx + 1) + (j_idx + 1) * n_x + k_idx * n_xy
    n3 = i_idx + (j_idx + 1) * n_x + k_idx * n_xy
    n4 = i_idx + j_idx * n_x + (k_idx + 1) * n_xy
    n5 = (i_idx + 1) + j_idx * n_x + (k_idx + 1) * n_xy
    n6 = (i_idx + 1) + (j_idx + 1) * n_x + (k_idx + 1) * n_xy
    n7 = i_idx + (j_idx + 1) * n_x + (k_idx + 1) * n_xy
    hex_cells = np.column_stack([n0, n1, n2, n3, n4, n5, n6, n7]).astype(np.int64)
    # 温度场
    temp_cell = temperatures[:n_solid]

    meshio.Mesh(
        points,
        [("hexahedron", hex_cells)],
        cell_data={
            "Temperature_K": [temp_cell]
        },
    ).write(output_path)

    logger.info(
        f"VTU export done: {n_solid} cells, {topo.n_x * topo.n_y * topo.n_z} points"
    )
