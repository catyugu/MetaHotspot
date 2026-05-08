import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass


@dataclass(slots=True)
class MeshTopology:
    """Pure geometric and topological data (SoA layout)"""

    n_cells: int
    centers: np.ndarray
    dims: np.ndarray
    boxes: np.ndarray
    volumes: np.ndarray
    internal_faces: list[tuple[int, int]]
    boundary_faces: dict[str, list[tuple[int, np.ndarray, float]]]
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
    hydroC: (
        np.ndarray
    )  # hydrodynamic coefficient, shape (n_cells, 3) — anisotropic conductance along [X, Y, Z] axes
    pressure: np.ndarray
    inlet_temperature: np.ndarray
    layer_names: np.ndarray
    unit_names: np.ndarray


@dataclass(slots=True)
class SystemMatrices:
    """Assembled algebraic equations A * T = b"""

    A_total: sp.csr_matrix
    b_total: np.ndarray
    power_matrix: sp.csr_matrix
    unit_names: list[str]
