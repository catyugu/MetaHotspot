"""烘焙器：将物理和几何信息映射到离散网格生成掩码数组和物理场。"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

from metahotspot.metahotspot_types import (
    ModelConfig,
    MeshTopology,
    PhysicalFields,
    ParsedFaceKey,
)
from metahotspot.config import TOL
from metahotspot.units import UnitConverter
from metahotspot.logger import get_logger

logger = get_logger()


def parse_face_key(fk_str: str, params: dict) -> ParsedFaceKey:
    """解析 FaceKey 字符串。"""
    parts = fk_str.split("|")
    axis = parts[0]
    coord = float(parts[2])

    rects = []
    if len(parts) == 4:
        # 分号分隔的矩形列表
        for rs in parts[3].split(";"):
            if not rs.strip():
                continue
            u1, u2, v1, v2 = map(float, rs.split(","))
            rects.append((u1, u2, v1, v2))
    elif len(parts) == 7:
        # 竖线分隔的单一矩形
        u1, u2, v1, v2 = map(float, parts[3:7])
        rects.append((u1, u2, v1, v2))

    return ParsedFaceKey(axis, coord, rects, params)


def bake_model(
    config: ModelConfig, topo: MeshTopology
) -> Tuple[PhysicalFields, List[ParsedFaceKey]]:
    """烘焙模型属性到单元格级别。"""
    logger.info("Baking geometry properties...")
    n_cells = topo.n_cells
    uc = UnitConverter(config.length_unit)

    # 物理属性 SoA 数组 (shape: n_cells,)
    k_arr = np.zeros(n_cells, dtype=np.float64)
    cp_arr = np.zeros(n_cells, dtype=np.float64)
    rho_arr = np.zeros(n_cells, dtype=np.float64)
    hs_arr = np.zeros(n_cells, dtype=np.float64)

    # 重塑为 3D 以便于基于坐标的分配 (shape: nx, ny, nz)
    nx, ny, nz = topo.n_x - 1, topo.n_y - 1, topo.n_z - 1
    k_3d = k_arr.reshape((nx, ny, nz))
    cp_3d = cp_arr.reshape((nx, ny, nz))
    rho_3d = rho_arr.reshape((nx, ny, nz))
    hs_3d = hs_arr.reshape((nx, ny, nz))

    # 获取单元格中心点还原回原始单位用于和原始几何配置做比较
    centers_3d = uc.from_m(topo.cell_geom.centers.reshape((nx, ny, nz, 3)))

    # 计算层 Z 轴边界 (层按从上到下排列)
    z_max_total = config.mesh_coords.z[-1]
    current_z = z_max_total
    layer_bounds: Dict[str, Tuple[float, float]] = {}
    for layer in config.layers:
        z_top = current_z
        z_bot = current_z - layer.thickness
        layer_bounds[layer.name] = (z_bot, z_top)
        current_z = z_bot

    # 获取单元格中心点还原回原始单位用于和原始几何配置做比较
    X = centers_3d[..., 0]
    Y = centers_3d[..., 1]
    Z = centers_3d[..., 2]

    # 分配属性
    for layer in config.layers:
        z_bot, z_top = layer_bounds[layer.name]
        layer_mask = (Z >= z_bot - TOL.geom_tol) & (Z <= z_top + TOL.geom_tol)

        for block in layer.blocks:
            mat = config.materials.get(block.material_name)
            if not mat:
                continue

            block_mask = np.zeros((nx, ny, nz), dtype=bool)
            for rect in block.rects:
                rect_mask = (
                    (X >= rect.x - TOL.geom_tol)
                    & (X <= rect.x + rect.width + TOL.geom_tol)
                    & (Y >= rect.y - TOL.geom_tol)
                    & (Y <= rect.y + rect.height + TOL.geom_tol)
                )
                if rect.add_sub:
                    block_mask |= rect_mask
                else:
                    block_mask &= ~rect_mask

            final_mask = layer_mask & block_mask
            k_3d[final_mask] = mat.k
            cp_3d[final_mask] = mat.cp
            rho_3d[final_mask] = mat.density
            hs_3d[final_mask] = block.heat_source

    # 解析边界
    parsed_bcs = []
    for bc in config.boundaries:
        for fk_str in bc.face_keys:
            parsed_bcs.append(parse_face_key(fk_str, bc.params))

    fields = PhysicalFields(
        k=k_arr,
        cp=cp_arr,
        density=rho_arr,
        heat_source=hs_arr,
        temperature=np.full(n_cells, config.initial_temperature, dtype=np.float64),
    )

    return fields, parsed_bcs
