import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


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
