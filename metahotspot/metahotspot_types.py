import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass(slots=True)
class BoundaryCondition:
    """强类型边界条件定义"""

    name: str
    type: str
    face: str
    target: str
    parameters: Dict[str, float]


@dataclass(slots=True)
class MaterialProps:
    """强类型材料属性定义"""

    k: float
    cp: float
    density: float
    is_fluid: bool
    dynamic_viscosity: float


@dataclass(slots=True)
class UnitRegion:
    """数值计算单元几何区域"""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    props: MaterialProps


@dataclass(slots=True)
class LayerRegion:
    """数值计算层几何区域"""

    name: str
    lz: float
    dz: float
    props: MaterialProps
    units: List[UnitRegion]


@dataclass(slots=True)
class PowerSource:
    """强类型热源区域"""

    name: str
    lx: float
    ly: float
    lz: float
    dx: float
    dy: float
    dz: float


@dataclass(slots=True)
class SolverConfig:
    """强类型求解器配置 (解耦IO与计算)"""

    simulation_type: str
    timestep: float
    init_temperature: float
    mesh_file_path: str
    ptrace_file_path: str
    init_temperature_file_path: str
    default_solid: MaterialProps
    boundary_conditions: List[BoundaryCondition]


@dataclass(slots=True)
class MeshTopology:
    """纯几何与拓扑数据 (SoA 布局)"""

    n_cells: int
    centers: np.ndarray
    dims: np.ndarray
    boxes: np.ndarray
    volumes: np.ndarray
    internal_faces: np.ndarray
    boundary_faces: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    sorted_indices: np.ndarray
    orig_to_new_id: np.ndarray


@dataclass(slots=True)
class PhysicalFields:
    """物理属性与状态场 (SoA 布局)"""

    k: np.ndarray
    cp: np.ndarray
    density: np.ndarray
    is_fluid: np.ndarray
    dynamic_viscosity: np.ndarray
    hydroC: np.ndarray
    pressure: np.ndarray
    boundary_temperature: np.ndarray

    layer_ids: np.ndarray
    unit_ids: np.ndarray
    layer_name_map: List[str] = field(default_factory=list)
    unit_name_map: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SystemMatrices:
    """装配后的代数方程 A * T = b"""

    A_total: sp.csr_matrix
    b_total: np.ndarray
    power_matrix: sp.csr_matrix
    unit_names: List[str]
