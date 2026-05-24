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

    # 物理属性 SoA 数组 (shape: n_cells,) — 初始为 0 便于累加
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
    # 单元格角点坐标 (已在原始单位中，直接使用)
    x_corners = topo.coords.x  # shape: (nx,)
    y_corners = topo.coords.y  # shape: (ny,)
    z_corners = topo.coords.z  # shape: (nz,)

    # 单元格尺寸 (shape: nx, ny, nz) - 与 x_corners 单位一致
    dx = np.diff(x_corners)
    dy = np.diff(y_corners)
    dz = np.diff(z_corners)

    for layer in config.layers:
        z_bot, z_top = layer_bounds[layer.name]
        layer_mask = (Z >= z_bot - TOL.geom_tol) & (Z <= z_top + TOL.geom_tol)

        for block in layer.blocks:
            mat = config.materials.get(block.material_name)
            if not mat:
                continue

            block_mask = np.zeros((nx, ny, nz), dtype=bool)
            intersect_vol = np.zeros((nx, ny, nz), dtype=np.float64)

            for rect in block.rects:
                # 计算每个单元格与热源块矩形的交集体积
                rect_x_min = rect.x
                rect_x_max = rect.x + rect.width
                rect_y_min = rect.y
                rect_y_max = rect.y + rect.height
                rect_z_min = z_bot
                rect_z_max = z_top

                # 交集的 6 个面
                inter_x = np.maximum(
                    0,
                    np.minimum(x_corners[1:], rect_x_max)
                    - np.maximum(x_corners[:-1], rect_x_min),
                )
                inter_y = np.maximum(
                    0,
                    np.minimum(y_corners[1:], rect_y_max)
                    - np.maximum(y_corners[:-1], rect_y_min),
                )
                inter_z = np.maximum(
                    0,
                    np.minimum(z_corners[1:], rect_z_max)
                    - np.maximum(z_corners[:-1], rect_z_min),
                )

                # 展成 (nx, ny, nz) 的交集体积
                inter_vol_rect = (
                    inter_x[:, None, None]
                    * inter_y[None, :, None]
                    * inter_z[None, None, :]
                )

                if rect.add_sub:
                    block_mask |= inter_vol_rect > TOL.geom_tol
                    intersect_vol += inter_vol_rect
                else:
                    block_mask &= ~(inter_vol_rect > TOL.geom_tol)
                    intersect_vol -= inter_vol_rect

            final_mask = layer_mask & block_mask
            vol_3d = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
            hs_weight = intersect_vol / np.maximum(vol_3d, TOL.geom_tol)
            valid = final_mask & (intersect_vol > TOL.geom_tol)
            hs_3d[valid] += block.heat_source * hs_weight[valid]

            # 体积加权热导率 / Cp / 密度 (等效介质模型)
            w = intersect_vol[valid] / np.maximum(vol_3d[valid], TOL.geom_tol)
            k_3d[valid] += mat.k * w
            cp_3d[valid] += mat.cp * w
            rho_3d[valid] += mat.density * w

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
