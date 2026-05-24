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

    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    k_3d = fields.k.reshape((nx, ny, nz))
    solid_mask = k_3d > 0.0

    # 构建节点坐标 (n_x * n_y * n_z, 3)
    points = np.zeros((topo.n_x * topo.n_y * topo.n_z, 3), dtype=np.float64)
    idx = 0
    for k in range(topo.n_z):
        for j in range(topo.n_y):
            for i in range(topo.n_x):
                points[idx, 0] = x_m[i]
                points[idx, 1] = y_m[j]
                points[idx, 2] = z_m[k]
                idx += 1

    # 构建六面体单元 connectivity (n_solid, 8)
    solid_indices = np.where(solid_mask)
    n_solid = len(solid_indices[0])
    hex_cells = np.zeros((n_solid, 8), dtype=np.int64)

    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        # 8 节点六面体编号（按 VTK 逆时针环绕约定）
        # --- 底面 (Z = k) ---
        n0 = i + j * topo.n_x + k * topo.n_x * topo.n_y
        n1 = (i + 1) + j * topo.n_x + k * topo.n_x * topo.n_y
        n2 = (i + 1) + (j + 1) * topo.n_x + k * topo.n_x * topo.n_y
        n3 = i + (j + 1) * topo.n_x + k * topo.n_x * topo.n_y

        # --- 顶面 (Z = k + 1) ---
        n4 = i + j * topo.n_x + (k + 1) * topo.n_x * topo.n_y
        n5 = (i + 1) + j * topo.n_x + (k + 1) * topo.n_x * topo.n_y
        n6 = (i + 1) + (j + 1) * topo.n_x + (k + 1) * topo.n_x * topo.n_y
        n7 = i + (j + 1) * topo.n_x + (k + 1) * topo.n_x * topo.n_y

        hex_cells[seq] = [n0, n1, n2, n3, n4, n5, n6, n7]

    # 材料 ID（用于 Paraview 可视化）
    material_id = np.zeros(n_solid, dtype=np.int32)
    for seq, (i, j, k) in enumerate(zip(*solid_indices)):
        material_id[seq] = int(fields.k[seq] * 1e6)  # 用热导率指纹作为材料 ID

    # 温度场
    temp_cell = temperatures[:n_solid]

    meshio.Mesh(
        points,
        [("hexahedron", hex_cells)],
        cell_data={
            "Temperature_K": [temp_cell],
            "MaterialID": [material_id],
        },
    ).write(output_path)

    logger.info(
        f"VTU export done: {n_solid} cells, {topo.n_x * topo.n_y * topo.n_z} points"
    )
