import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional


# ==========================================
# 强类型配置 (替代原有的散装 dict)
# ==========================================
@dataclass(slots=True)
class BoundaryConditionConfig:
    name: str
    type: str  # "convection", "pressure"
    face: str
    target: str = ""
    h: float = 0.0
    T_inf: float = 0.0
    pressure: float = 0.0
    temperature: float = np.nan


@dataclass(slots=True)
class SimulationConfig:
    simulation_type: str
    ambient: float
    init_temperature: float
    timestep: float
    time: float
    mesh_file_path: str
    ptrace_file_path: str
    init_temperature_file_path: str
    boundary_conditions: List[BoundaryConditionConfig] = field(default_factory=list)
    # ... 其他全局参数可按需扩展


# ==========================================
# 核心数据结构 (纯 SoA 布局)
# ==========================================
@dataclass(slots=True)
class MeshTopology:
    """Pure geometric and topological data (SoA layout)"""

    n_cells: int
    centers: np.ndarray
    dims: np.ndarray
    boxes: np.ndarray
    volumes: np.ndarray
    internal_faces: np.ndarray  # shape (N, 2)，使用 NumPy 数组替代 list[tuple]
    boundary_faces: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]  # c_ids, normals, areas
    sorted_indices: np.ndarray
    orig_to_new_id: np.ndarray


@dataclass(slots=True)
class PhysicalFields:
    """Physical properties and state fields (SoA layout)"""

    k: np.ndarray
    cp: np.ndarray
    density: np.ndarray
    is_fluid: np.ndarray
    dynamic_viscosity: np.ndarray
    hydroC: np.ndarray  # shape (n_cells, 3)
    pressure: np.ndarray
    inlet_temperature: np.ndarray

    # 【工业规范】移除 object 数组，使用 int16 索引替代，极大提升内存连续性与 Numba 兼容性
    layer_ids: np.ndarray
    unit_ids: np.ndarray

    # 映射表（仅供最终输出时查找名称）
    layer_name_map: List[str] = field(default_factory=list)
    unit_name_map: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SystemMatrices:
    """Assembled algebraic equations A * T = b"""

    A_total: sp.csr_matrix
    b_total: np.ndarray
    power_matrix: sp.csr_matrix
    unit_names: List[str]
