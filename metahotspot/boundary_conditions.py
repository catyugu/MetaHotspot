import re
import numpy as np
from typing import Tuple, List

from metahotspot.metahotspot_types import MeshTopology, PhysicalFields


def resolve_boundary_cells(
    topo: MeshTopology, fields: PhysicalFields, face_key: str, target_regex: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用边界单元解析器。
    根据指定的面 (face_key) 和 目标正则 (target_regex) 过滤边界单元。
    支持对 layer_name 或 unit_name 的双向匹配。
    """
    if face_key not in topo.boundary_faces:
        return np.array([], dtype=int), np.array([], dtype=float)

    c_ids, _, areas = topo.boundary_faces[face_key]

    if not target_regex:
        return c_ids, areas

    pattern = re.compile(target_regex)

    # 提取边界单元对应的 层名称 和 单元名称
    layer_names = [fields.layer_name_map[fields.layer_ids[cid]] for cid in c_ids]
    unit_names = [fields.unit_name_map[fields.unit_ids[cid]] for cid in c_ids]

    # 如果层名称或单元名称任意一个匹配正则，则保留该单元
    mask = np.array(
        [
            bool(pattern.match(l_name)) or bool(pattern.match(u_name))
            for l_name, u_name in zip(layer_names, unit_names)
        ]
    )

    return c_ids[mask], areas[mask]


# ==========================================
# 状态修改算子 (State Modification Operators)
# ==========================================


def apply_pressure_bc(
    c_ids: np.ndarray,
    params: dict,
    fields: PhysicalFields,
    is_pressure_boundary: np.ndarray,
) -> None:
    """流体压力边界算子"""
    fluid_mask = fields.is_fluid[c_ids]
    valid_c_ids = c_ids[fluid_mask]

    if len(valid_c_ids) > 0:
        is_pressure_boundary[valid_c_ids] = True
        fields.pressure[valid_c_ids] = float(params["pressure"])


def apply_temperature_state_bc(
    c_ids: np.ndarray, params: dict, fields: PhysicalFields
) -> None:
    """通用温度状态设定算子（记录 Dirichlet 边界值）"""
    fields.boundary_temperature[c_ids] = float(params["temperature"])


# ==========================================
# 矩阵装配算子 (Matrix Assembly Operators)
# ==========================================


def apply_convection_matrix_bc(
    c_ids: np.ndarray,
    areas: np.ndarray,
    params: dict,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
    """对流边界(Robin)矩阵算子 (完全向量化)"""
    h, t_inf = float(params["h"]), float(params["T_inf"])
    vols, k = topo.volumes[c_ids], fields.k[c_ids]

    # 向量化计算传热系数
    g = areas / ((0.5 * (vols / areas) / k) + (1.0 / h))

    rows.extend(c_ids.tolist())
    cols.extend(c_ids.tolist())
    data.extend((-g).tolist())
    rhs[c_ids] += g * t_inf


def apply_temperature_matrix_bc(
    c_ids: np.ndarray,
    areas: np.ndarray,
    params: dict,
    topo: MeshTopology,
    fields: PhysicalFields,
    rows: List[int],
    cols: List[int],
    data: List[float],
    rhs: np.ndarray,
) -> None:
    """恒温边界(Dirichlet)矩阵算子 - 罚函数法 (完全向量化)"""
    if len(c_ids) == 0:
        return

    temp = float(params["temperature"])
    h_inf = 1e20  # 使用巨大对流换热系数锁定表面温度
    vols, k = topo.volumes[c_ids], fields.k[c_ids]

    g = areas / ((0.5 * (vols / areas) / k) + (1.0 / h_inf))

    rows.extend(c_ids.tolist())
    cols.extend(c_ids.tolist())
    data.extend((-g).tolist())
    rhs[c_ids] += g * temp
