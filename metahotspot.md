# Project Source Code: metahotspot

## Directory Structure
```text
.
├── legacy
│   ├── __init__.py
│   ├── converter.py
│   └── hotspot_parser.py
├── __init__.py
├── assembler.py
├── assembler_kernels.py
├── fluid_preprocessor.py
├── gmsh_mesher.py
├── mesh_preprocessor.py
├── metahotspot_solver.py
├── metahotspot_types.py
├── model25d.py
└── thermal_solver.py
```

## File Contents

### File: assembler.py
```py
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    SystemMatrices,
)
from metahotspot.assembler_kernels import (
    overlap_area_kernel,
    find_adjacent_pairs_kernel,
)


class FVMAssembler:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(
        self, topo: MeshTopology, fields: PhysicalFields, config: dict, stackup: list
    ):
        self.topo, self.fields, self.config, self.stackup = (
            topo,
            fields,
            config,
            stackup,
        )
        self.flow_axes = np.zeros(self.topo.n_cells, dtype=int)

    def assemble(self) -> SystemMatrices:
        self._precompute_flow_axes()  # Compute flow axes from pressure gradient
        A_cond = self._build_conduction_matrix()
        A_bc, b_bc = self._build_boundary_terms()
        A_adv, b_adv = self._build_advection_matrix()
        power_mat, unit_names = self._build_power_matrix()
        return SystemMatrices(
            A_cond + A_bc + A_adv, b_bc + b_adv, power_mat, unit_names
        )

    def _precompute_flow_axes(self) -> None:
        """Based on solved pressure field, infer dominant flow axis for each fluid cell (axis with largest pressure drop)"""
        if not np.any(self.fields.is_fluid):
            return
        p_drops = np.zeros((self.topo.n_cells, 3))
        for c_a, c_b, axis, _ in self._find_adjacent_pairs():
            if self.fields.is_fluid[c_a] and self.fields.is_fluid[c_b]:
                dp = abs(self.fields.pressure[c_a] - self.fields.pressure[c_b])
                p_drops[c_a, axis] = max(p_drops[c_a, axis], dp)
                p_drops[c_b, axis] = max(p_drops[c_b, axis], dp)

        fluid_mask = self.fields.is_fluid
        self.flow_axes[fluid_mask] = np.argmax(p_drops[fluid_mask], axis=1)

    def _find_adjacent_pairs(self):
        """Generator that yields adjacent cell pairs with their overlap area and normal axis."""
        c_a_arr, c_b_arr, axis_arr, area_arr, count = find_adjacent_pairs_kernel(
            self.topo.boxes
        )
        for i in range(count):
            yield c_a_arr[i], c_b_arr[i], axis_arr[i], area_arr[i]

    @staticmethod
    def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
        return overlap_area_kernel(box_a, box_b, axis)

    def _build_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data, n = [], [], [], self.topo.n_cells
        for c_a, c_b, axis, area in self._find_adjacent_pairs():
            g = 1.0 / self._calc_resistance(c_a, c_b, axis, area)
            rows.extend([c_a, c_b, c_a, c_b])
            cols.extend([c_a, c_b, c_b, c_a])
            data.extend([-g, -g, g, g])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _calc_resistance(self, c_a: int, c_b: int, axis: int, area: float) -> float:
        fluid_a, fluid_b = self.fields.is_fluid[c_a], self.fields.is_fluid[c_b]
        if fluid_a != fluid_b:
            f_id, s_id = (c_a, c_b) if fluid_a else (c_b, c_a)
            flow_axis = self.flow_axes[f_id]
            ax_w = (flow_axis + 1) % 3
            ax_h = (flow_axis + 2) % 3
            Nu = self._compute_nusselt(f_id, flow_axis)
            d_h = (
                2 * self.topo.dims[f_id, ax_w] * self.topo.dims[f_id, ax_h]
                / (self.topo.dims[f_id, ax_w] + self.topo.dims[f_id, ax_h])
            )
            h_f = (Nu * self.fields.k[f_id]) / d_h if d_h > 0 else 1e-6
            return self.topo.dims[s_id, axis] / (
                2.0 * self.fields.k[s_id] * area
            ) + 1.0 / (h_f * area)
        return (self.topo.dims[c_a, axis] / (2.0 * self.fields.k[c_a] * area)) + (
            self.topo.dims[c_b, axis] / (2.0 * self.fields.k[c_b] * area)
        )

    def _compute_nusselt(self, c_id: int, flow_axis: int) -> float:
        ax_w = (flow_axis + 1) % 3
        ax_h = (flow_axis + 2) % 3
        w, h = sorted([self.topo.dims[c_id, ax_w], self.topo.dims[c_id, ax_h]])
        AR = w / h if h > 0 else 1.0
        return 8.235 * (
            1
            - 2.0421 * AR
            + 3.0853 * AR**2
            - 2.4765 * AR**3
            + 1.0578 * AR**4
            - 0.1861 * AR**5
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rhs, rows, cols, data = (
            self.topo.n_cells,
            np.zeros(self.topo.n_cells),
            [],
            [],
            [],
        )
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "convection":
                continue
            h, t_inf, target, face_key = (
                float(bc["h"]),
                float(bc["T_inf"]),
                bc.get("target"),
                bc.get("face", ""),
            )
            for c_id, _, area in self.topo.boundary_faces.get(face_key, []):
                if target and target != self.fields.layer_names[c_id]:
                    continue
                g = area / (
                    (0.5 * (self.topo.volumes[c_id] / area) / self.fields.k[c_id])
                    + (1.0 / h)
                )
                rows.append(c_id)
                cols.append(c_id)
                data.append(-g)
                rhs[c_id] += g * t_inf
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _build_advection_matrix(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rows, cols, data, rhs, tol = (
            self.topo.n_cells,
            [],
            [],
            [],
            np.zeros(self.topo.n_cells),
            self.GEOMETRY_TOLERANCE,
        )
        if not np.any(self.fields.is_fluid):
            return sp.csr_matrix((n, n)), rhs

        net_outflux = np.zeros(n)
        for c_a, c_b, axis, area in self._find_adjacent_pairs():
            if not (self.fields.is_fluid[c_a] and self.fields.is_fluid[c_b]):
                continue

            # Use axis-specific hydroC for fluid-fluid pairs
            hc_a = self.fields.hydroC[c_a, axis]
            hc_b = self.fields.hydroC[c_b, axis]
            sum_hc = hc_a + hc_b
            C_eff = (
                2.0 * hc_a * hc_b / sum_hc
                if sum_hc > 0
                else 0.0
            )
            mass_flux = (
                (self.fields.pressure[c_a] - self.fields.pressure[c_b])
                * C_eff
                * (self.fields.density[c_a] + self.fields.density[c_b])
                * 0.5
            )
            net_outflux[c_a], net_outflux[c_b] = (
                net_outflux[c_a] + mass_flux,
                net_outflux[c_b] - mass_flux,
            )
            if abs(mass_flux) > tol:
                up, dn = (c_a, c_b) if mass_flux > 0 else (c_b, c_a)
                adv = abs(mass_flux) * self.fields.cp[up]
                rows.extend([up, dn])
                cols.extend([up, up])
                data.extend([-adv, adv])

        fluid_ids = np.where(self.fields.is_fluid)[0]
        for c_id in fluid_ids:
            influx = net_outflux[c_id]
            if influx > tol and not np.isnan(self.fields.inlet_temperature[c_id]):
                rhs[c_id] += (
                    influx * self.fields.cp[c_id] * self.fields.inlet_temperature[c_id]
                )
            elif influx < -tol:
                rows.append(c_id)
                cols.append(c_id)
                data.append(influx * self.fields.cp[c_id])
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _build_power_matrix(self) -> Tuple[sp.csr_matrix, List[str]]:
        active_units, z_cursor = [], 0.0
        for l in self.stackup:
            if l.active:
                for u in l.units:
                    active_units.append(
                        {
                            "name": u.name,
                            "lx": u.lx,
                            "ly": u.ly,
                            "lz": z_cursor,
                            "dx": u.dx,
                            "dy": u.dy,
                            "dz": l.thickness,
                        }
                    )
            z_cursor += l.thickness
        n_cells = self.topo.n_cells
        if not active_units:
            return sp.csr_matrix((n_cells, 0)), []
        rows, cols, data, boxes = [], [], [], self.topo.boxes
        for j, u in enumerate(active_units):
            vol_u = u["dx"] * u["dy"] * u["dz"]
            if vol_u <= 0:
                continue
            u_min, u_max = np.array([u["lx"], u["ly"], u["lz"]]), np.array(
                [u["lx"], u["ly"], u["lz"]]
            ) + np.array([u["dx"], u["dy"], u["dz"]])
            intersect = np.prod(
                np.maximum(
                    0, np.minimum(boxes[:, 3:], u_max) - np.maximum(boxes[:, :3], u_min)
                ),
                axis=1,
            )
            valid = np.where(intersect > self.GEOMETRY_TOLERANCE)[0]
            rows.extend(valid)
            cols.extend([j] * len(valid))
            data.extend(intersect[valid] / vol_u)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(n_cells, len(active_units))
        ), [u["name"] for u in active_units]

```

### File: assembler_kernels.py
```py
"""
Numba compute kernels for FVM assembly.
计算与状态分离 - OOP 层负责数据解包，Kernel 负责计算。
"""

from numba import njit
import numpy as np

# ==========================================
# 类型别名 - 确保类型稳定性
# ==========================================
FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32


# ==========================================
# Numba 装饰器工厂 - 统一配置
# ==========================================
def _jit_kernel(func):
    """工业标准装饰器: cache + fastmath + nogil"""
    return njit(cache=True, fastmath=True, nogil=True)(func)


@_jit_kernel
def overlap_area_kernel(
    box_a: FLOAT_DTYPE,
    box_b: FLOAT_DTYPE,
    axis: INT_DTYPE,
) -> FLOAT_DTYPE:
    """
    计算两个包围盒在指定轴法向上的重叠面积。

    Parameters
    ----------
    box_a : np.ndarray, shape=(6,)
        [x_min, y_min, z_min, x_max, y_max, z_max]
    box_b : np.ndarray, shape=(6,)
        同上
    axis : int
        0=X, 1=Y, 2=Z（取垂直于该轴的两个面）

    Returns
    -------
    float
        重叠面积 [m^2]
    """
    # 三个轴对应的面索引
    # axes 格式: (排除轴的最小坐标索引, 排除轴的次小坐标索引, 排除轴的最大坐标索引, 排除轴的最大坐标索引)
    if axis == 0:
        # Y-Z plane: 排除 X 轴
        d1 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    elif axis == 1:
        # X-Z plane: 排除 Y 轴
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    else:
        # X-Y plane: 排除 Z 轴
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


# ==========================================
# 数组 Kernel（邻近单元查找 - Sweep and Prune）
# ==========================================


@_jit_kernel
def find_adjacent_pairs_kernel(boxes):
    """
    Sweep-and-prune 邻近单元查找。

    预分配策略：每个单元最多 6 个相邻面
    max_pairs = n_cells * 6

    Parameters
    ----------
    boxes : np.ndarray, shape=(n_cells, 6)
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    c_a_arr : np.ndarray
    c_b_arr : np.ndarray
    axis_arr : np.ndarray
    area_arr : np.ndarray
    count : int
    """
    n_cells = boxes.shape[0]
    max_pairs = n_cells * 6

    c_a_arr = np.empty(max_pairs, dtype=np.int32)
    c_b_arr = np.empty(max_pairs, dtype=np.int32)
    axis_arr = np.empty(max_pairs, dtype=np.int32)
    area_arr = np.empty(max_pairs, dtype=np.float64)

    ptr = 0
    tol = 1e-15

    sorted_ids = np.argsort(boxes[:, 0])

    for i in range(len(sorted_ids)):
        c_a = sorted_ids[i]
        for j in range(i + 1, len(sorted_ids)):
            c_b = sorted_ids[j]
            if boxes[c_b, 0] > boxes[c_a, 3] + tol:
                break

            b_a = boxes[c_a]
            b_b = boxes[c_b]

            # BBox Y/Z 排斥校验
            if (
                max(b_a[1], b_b[1]) > min(b_a[4], b_b[4]) + tol
                or max(b_a[2], b_b[2]) > min(b_a[5], b_b[5]) + tol
            ):
                continue

            # 面接触检查
            for axis in range(3):
                if (
                    abs(b_a[axis + 3] - b_b[axis]) < tol
                    or abs(b_a[axis] - b_b[axis + 3]) < tol
                ):
                    # 计算重叠面积
                    if axis == 0:
                        d1 = min(b_a[4], b_b[4]) - max(b_a[1], b_b[1])
                        d2 = min(b_a[5], b_b[5]) - max(b_a[2], b_b[2])
                    elif axis == 1:
                        d1 = min(b_a[3], b_b[3]) - max(b_a[0], b_b[0])
                        d2 = min(b_a[5], b_b[5]) - max(b_a[2], b_b[2])
                    else:
                        d1 = min(b_a[3], b_b[3]) - max(b_a[0], b_b[0])
                        d2 = min(b_a[4], b_b[4]) - max(b_a[1], b_b[1])
                    area = d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0
                    if area > tol:
                        c_a_arr[ptr] = c_a
                        c_b_arr[ptr] = c_b
                        axis_arr[ptr] = axis
                        area_arr[ptr] = area
                        ptr += 1

    return c_a_arr[:ptr], c_b_arr[:ptr], axis_arr[:ptr], area_arr[:ptr], ptr

```

### File: fluid_preprocessor.py
```py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from metahotspot.metahotspot_types import MeshTopology, PhysicalFields


class FluidPreprocessor:
    """
    Specialized for calculating and solidifying fluid dynamic fields (pressure, convection coefficients)
    before thermal assembly.
    """

    def __init__(self, config: dict):
        self.config = config

    def solve_flow(self, topo: MeshTopology, fields: PhysicalFields) -> None:
        if not np.any(fields.is_fluid):
            return

        # Temporary state for boundary condition tracking during flow solve
        is_pressure_boundary = np.zeros(topo.n_cells, dtype=bool)

        self._init_cell_hydro_properties(topo, fields)
        self._apply_pressure_boundary_conditions(topo, fields, is_pressure_boundary)
        self._solve_pressure(topo, fields, is_pressure_boundary)

    def _init_cell_hydro_properties(
        self, topo: MeshTopology, fields: PhysicalFields
    ) -> None:
        m = fields.is_fluid & (fields.dynamic_viscosity > 0)
        if not np.any(m):
            return
        v = fields.dynamic_viscosity[m]

        # Compute anisotropic hydroC for X(0), Y(1), Z(2) axes
        for axis in range(3):
            ax_w = (axis + 1) % 3
            ax_h = (axis + 2) % 3

            L = topo.dims[m, axis]
            w = topo.dims[m, ax_w]
            h = topo.dims[m, ax_h]

            hydroC_axis = np.zeros(np.sum(m))
            cond_eq, cond_gt = np.abs(h - w) < 1e-10, h > w
            cond_lt = ~(cond_eq | cond_gt)

            hydroC_axis[cond_eq] = (0.42229 * h[cond_eq] ** 4) / (
                12 * v[cond_eq] * L[cond_eq]
            )
            hydroC_axis[cond_gt] = (
                (1 - 0.63 * (w[cond_gt] / h[cond_gt])) * w[cond_gt] ** 3 * h[cond_gt]
            ) / (12 * v[cond_gt] * L[cond_gt])
            hydroC_axis[cond_lt] = (
                (1 - 0.63 * (h[cond_lt] / w[cond_lt])) * h[cond_lt] ** 3 * w[cond_lt]
            ) / (12 * v[cond_lt] * L[cond_lt])

            fields.hydroC[m, axis] = hydroC_axis

    def _apply_pressure_boundary_conditions(
        self,
        topo: MeshTopology,
        fields: PhysicalFields,
        is_pressure_boundary: np.ndarray,
    ) -> None:
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "pressure":
                continue
            pressure, temp, face, target = (
                float(bc["pressure"]),
                float(bc.get("temperature", np.nan)),
                bc.get("face", ""),
                bc.get("target"),
            )
            for c_id, _, _ in topo.boundary_faces.get(face, []):
                if fields.is_fluid[c_id] and (
                    not target or fields.layer_names[c_id] == target
                ):
                    (
                        is_pressure_boundary[c_id],
                        fields.pressure[c_id],
                        fields.inlet_temperature[c_id],
                    ) = (True, pressure, temp)

    def _solve_pressure(
        self,
        topo: MeshTopology,
        fields: PhysicalFields,
        is_pressure_boundary: np.ndarray,
    ) -> None:
        fluid_ids = np.where(fields.is_fluid)[0]
        if len(fluid_ids) == 0:
            return
        n_fluid, global_to_fluid = len(fluid_ids), np.full(topo.n_cells, -1, dtype=int)
        global_to_fluid[fluid_ids] = np.arange(n_fluid)
        rows, cols, data, b_p, diag_C, is_p_bound = (
            [],
            [],
            [],
            np.zeros(n_fluid),
            np.zeros(n_fluid),
            is_pressure_boundary[fluid_ids],
        )

        bound_idx = np.where(is_p_bound)[0]
        rows.extend(bound_idx)
        cols.extend(bound_idx)
        data.extend(np.ones(len(bound_idx)))
        b_p[bound_idx] = fields.pressure[fluid_ids][bound_idx]

        for c0, c1 in topo.internal_faces:
            i0, i1 = global_to_fluid[c0], global_to_fluid[c1]
            if i0 == -1 or i1 == -1:
                continue

            # Dynamically infer axis from cell center difference
            axis = np.argmax(np.abs(topo.centers[c1] - topo.centers[c0]))
            hc0 = fields.hydroC[c0, axis]
            hc1 = fields.hydroC[c1, axis]

            sum_hc = hc0 + hc1
            C_eff = (2.0 * hc0 * hc1 / sum_hc) if sum_hc > 0 else 0.0

            if not is_pressure_boundary[c0]:
                rows.append(i0)
                cols.append(i1)
                data.append(C_eff)
                diag_C[i0] += C_eff
            if not is_pressure_boundary[c1]:
                rows.append(i1)
                cols.append(i0)
                data.append(C_eff)
                diag_C[i1] += C_eff

        for i in range(n_fluid):
            if not is_p_bound[i]:
                rows.append(i)
                cols.append(i)
                data.append(-diag_C[i])

        try:
            fields.pressure[fluid_ids] = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)), b_p
            )
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")

```

### File: gmsh_mesher.py
```py
import math
from collections import deque
from pathlib import Path

import gmsh
from metahotspot.model25d import load_config, load_stackup


class GmshMesher:
    DEFAULT_MAX_MESH_SIZE = 0.01
    DEFAULT_MIN_MESH_SIZE = 0.0005
    DEFAULT_REFINEMENT_DISTANCE = 0.002

    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)
        self._node_id = 1
        self._elem_id = 1
        self._node_map: dict = {}
        self._global_node_coords: dict = {}

    def generate_mesh(self, config_path: str, mesh_params: dict = None) -> None:
        mesh_params = mesh_params or {}
        base_dir = str(Path(config_path).parent)

        # 换用统一入口加载JSON
        config = load_config(config_path)

        max_mesh_size = mesh_params.get("max_mesh_size", self.DEFAULT_MAX_MESH_SIZE)
        min_mesh_size = mesh_params.get("min_mesh_size", self.DEFAULT_MIN_MESH_SIZE)
        refine_distance = mesh_params.get(
            "refine_distance", self.DEFAULT_REFINEMENT_DISTANCE
        )

        stackup = load_stackup(config, base_dir)

        heat_boxes = [
            (u.lx, u.ly, u.lx + u.dx, u.ly + u.dy)
            for l in stackup
            if l.active
            for u in l.units
        ]
        z_cursor = 0.0

        for layer in stackup:
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], layer.tag)

            lz, dz = z_cursor, layer.thickness
            z_cursor += dz

            leaves = self._subdivide_layer(
                layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
            )
            self._create_hex_elements(discrete_tag, lz, dz, leaves)

    def _subdivide_layer(
        self, layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
    ):
        leaves, queue = [], deque(
            [(u.lx, u.ly, u.lx + u.dx, u.ly + u.dy) for u in layer.units]
        )

        while queue:
            x0, y0, x1, y1 = queue.popleft()
            w, h = x1 - x0, y1 - y0
            needs_split = w > max_mesh_size or h > max_mesh_size

            if not needs_split and (
                w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01
            ):
                for hb in heat_boxes:
                    dist_x, dist_y = max(0.0, x0 - hb[2], hb[0] - x1), max(
                        0.0, y0 - hb[3], hb[1] - y1
                    )
                    if math.hypot(dist_x, dist_y) <= refine_distance:
                        needs_split = True
                        break

            if needs_split:
                if w >= h:
                    mid = (x0 + x1) / 2.0
                    queue.extend([(x0, y0, mid, y1), (mid, y0, x1, y1)])
                else:
                    mid = (y0 + y1) / 2.0
                    queue.extend([(x0, y0, x1, mid), (x0, mid, x1, y1)])
            else:
                leaves.append((x0, y0, x1, y1))

        return leaves

    def _get_node(self, x: float, y: float, z: float) -> int:
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in self._node_map:
            self._node_map[key] = self._node_id
            self._global_node_coords[self._node_id] = (x, y, z)
            self._node_id += 1
        return self._node_map[key]

    def _create_hex_elements(self, discrete_tag, lz, dz, leaves) -> None:
        element_tags, element_nodes, used_node_ids = [], [], set()

        for x0, y0, x1, y1 in leaves:
            nodes = [
                self._get_node(x0, y0, lz),
                self._get_node(x1, y0, lz),
                self._get_node(x1, y1, lz),
                self._get_node(x0, y1, lz),
                self._get_node(x0, y0, lz + dz),
                self._get_node(x1, y0, lz + dz),
                self._get_node(x1, y1, lz + dz),
                self._get_node(x0, y1, lz + dz),
            ]
            element_tags.append(self._elem_id)
            element_nodes.extend(nodes)
            used_node_ids.update(nodes)
            self._elem_id += 1

        if element_tags:
            layer_nodes_tags = sorted(used_node_ids)
            layer_nodes_coords = [
                coord
                for nid in layer_nodes_tags
                for coord in self._global_node_coords[nid]
            ]
            gmsh.model.mesh.addNodes(
                3, discrete_tag, layer_nodes_tags, layer_nodes_coords
            )
            gmsh.model.mesh.addElements(
                3, discrete_tag, [5], [element_tags], [element_nodes]
            )

    def finalize(self, output_path: str) -> None:
        gmsh.write(output_path)
        gmsh.finalize()

```

### File: mesh_preprocessor.py
```py
from typing import Any, Dict, List, Tuple

import meshio
import numpy as np

from metahotspot.metahotspot_types import MeshTopology, PhysicalFields


class MeshPreprocessor:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(self, config: Dict[str, Any], stackup: List[Any]) -> None:
        self.config = config
        self.stackup = stackup

    def process(self, mesh_path: str) -> Tuple[MeshTopology, PhysicalFields]:
        mesh = meshio.read(mesh_path)
        topo = self._extract_geometry(mesh)
        fields = self._map_physical_properties(topo)
        return topo, fields

    def _extract_geometry(self, mesh: meshio.Mesh) -> MeshTopology:
        hex_blocks = [b.data for b in mesh.cells if b.type == "hexahedron"]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")
        hex_data = np.vstack(hex_blocks)
        coords = mesh.points[hex_data]
        lowers, uppers = np.min(coords, axis=1), np.max(coords, axis=1)
        centers, dims = (lowers + uppers) * 0.5, uppers - lowers
        vols = np.prod(dims, axis=1)
        sorted_indices = self._compute_morton_sort(lowers, uppers, centers)
        n_cells = len(centers)
        c_centers, c_dims, c_boxes, c_vols = (
            centers[sorted_indices],
            dims[sorted_indices],
            np.hstack([lowers[sorted_indices], uppers[sorted_indices]]),
            vols[sorted_indices],
        )
        orig_to_new_id = np.empty(n_cells, dtype=int)
        orig_to_new_id[sorted_indices] = np.arange(n_cells)
        internal_faces, boundary_faces = self._build_topology(
            mesh, hex_data, sorted_indices, c_centers
        )
        return MeshTopology(
            n_cells,
            c_centers,
            c_dims,
            c_boxes,
            c_vols,
            internal_faces,
            boundary_faces,
            sorted_indices,
            orig_to_new_id,
        )

    def _compute_morton_sort(self, lowers, uppers, centers) -> np.ndarray:
        b_min = np.min(lowers, axis=0)
        diff = np.where((d := np.max(uppers, axis=0) - b_min) == 0, 1, d)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)
        morton_keys = np.zeros(len(centers), dtype=np.int64)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0].astype(np.int64) >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1].astype(np.int64) >> i) & 1) << (
                3 * i + 1
            )
            morton_keys |= ((norm_centers[:, 2].astype(np.int64) >> i) & 1) << (
                3 * i + 2
            )
        return np.argsort(morton_keys)

    def _build_topology(
        self,
        mesh: meshio.Mesh,
        hex_data: np.ndarray,
        sorted_indices: np.ndarray,
        c_centers: np.ndarray,
    ) -> Tuple[list, dict]:
        face_to_cells = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]
            faces = [
                (nodes[0], nodes[3], nodes[2], nodes[1]),
                (nodes[4], nodes[5], nodes[6], nodes[7]),
                (nodes[0], nodes[1], nodes[5], nodes[4]),
                (nodes[3], nodes[7], nodes[6], nodes[2]),
                (nodes[0], nodes[4], nodes[7], nodes[3]),
                (nodes[1], nodes[2], nodes[6], nodes[5]),
            ]
            for f in faces:
                face_to_cells.setdefault(tuple(sorted(f)), []).append(new_id)
        internal_faces = [tuple(c) for c in face_to_cells.values() if len(c) == 2]
        boundary_faces_raw = {f: c[0] for f, c in face_to_cells.items() if len(c) == 1}
        boundary_faces = {"+X": [], "-X": [], "+Y": [], "-Y": [], "+Z": [], "-Z": []}
        for f, c_id in boundary_faces_raw.items():
            pts = mesh.points[list(f)]
            cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area = np.linalg.norm(cross)
            if area < self.GEOMETRY_TOLERANCE:
                continue
            normal = cross / area
            if np.dot(np.mean(pts, axis=0) - c_centers[c_id], normal) < 0:
                normal = -normal
            abs_n = np.abs(normal)
            if abs_n[2] >= abs_n[0] and abs_n[2] >= abs_n[1]:
                dir_key = "+Z" if normal[2] > 0 else "-Z"
            elif abs_n[0] >= abs_n[1]:
                dir_key = "+X" if normal[0] > 0 else "-X"
            else:
                dir_key = "+Y" if normal[1] > 0 else "-Y"
            boundary_faces[dir_key].append((c_id, normal, area))
        return internal_faces, boundary_faces

    def _map_physical_properties(self, topo: MeshTopology) -> PhysicalFields:
        n, centers, tol = topo.n_cells, topo.centers, self.GEOMETRY_TOLERANCE
        k, cp, density, is_fluid, dynamic_viscosity = (
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n, dtype=bool),
            np.zeros(n),
        )
        layer_names, unit_names = np.empty(n, dtype=object), np.empty(n, dtype=object)
        def_mat = self.config["materials"]["default_solid"]
        (
            k[:],
            cp[:],
            density[:],
            is_fluid[:],
            dynamic_viscosity[:],
            layer_names[:],
            unit_names[:],
        ) = (
            def_mat["k"],
            def_mat["cp"],
            def_mat["density"],
            def_mat.get("fluid", False),
            def_mat.get("dynamic_viscosity", 0.0),
            "default_layer",
            "",
        )
        z_cursor = 0.0
        for layer in self.stackup:
            z_min, z_max = z_cursor, z_cursor + layer.thickness
            z_cursor = z_max
            l_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)
            if not np.any(l_mask):
                continue
            (
                k[l_mask],
                cp[l_mask],
                density[l_mask],
                is_fluid[l_mask],
                dynamic_viscosity[l_mask],
                layer_names[l_mask],
            ) = (
                layer.k,
                layer.cp,
                layer.density,
                layer.is_fluid,
                layer.dynamic_viscosity,
                layer.name,
            )
            for unit in layer.units:
                u_mask = (
                    l_mask
                    & (centers[:, 0] >= unit.lx - tol)
                    & (centers[:, 0] <= unit.lx + unit.dx + tol)
                    & (centers[:, 1] >= unit.ly - tol)
                    & (centers[:, 1] <= unit.ly + unit.dy + tol)
                )
                if np.any(u_mask):
                    (
                        k[u_mask],
                        cp[u_mask],
                        density[u_mask],
                        is_fluid[u_mask],
                        dynamic_viscosity[u_mask],
                        unit_names[u_mask],
                    ) = (
                        unit.k,
                        unit.cp,
                        unit.density,
                        unit.is_fluid,
                        unit.dynamic_viscosity,
                        unit.name,
                    )
        return PhysicalFields(
            k,
            cp,
            density,
            is_fluid,
            dynamic_viscosity,
            np.zeros((n, 3)),
            np.zeros(n),
            np.full(n, np.nan),
            layer_names,
            unit_names,
        )

```

### File: metahotspot_solver.py
```py
import os

import meshio
import numpy as np
import time

from metahotspot.assembler import FVMAssembler
from metahotspot.thermal_solver import ThermalSolver
from metahotspot.mesh_preprocessor import MeshPreprocessor
from metahotspot.fluid_preprocessor import FluidPreprocessor
from metahotspot.metahotspot_types import MeshTopology
from metahotspot.model25d import load_config, load_stackup


class MetaHotspotSolver:
    def __init__(self, config_path: str):
        self.config_path, self.base_dir = config_path, os.path.dirname(config_path)
        self.config, self.stackup = load_config(config_path), load_stackup(
            load_config(config_path), os.path.dirname(config_path)
        )
        self.mesh_path = os.path.join(self.base_dir, self.config["mesh_file_path"])

    def run(self):
        start = time.perf_counter()
        print("[INFO] Preprocessing mesh and properties...")
        topo, fields = MeshPreprocessor(self.config, self.stackup).process(
            self.mesh_path
        )
        print("[INFO] Solving fluid flow (if applicable)...")
        FluidPreprocessor(self.config).solve_flow(topo, fields)
        print("[INFO] Assembling system matrices...")
        matrices = FVMAssembler(topo, fields, self.config, self.stackup).assemble()
        print("[INFO] Solving equations...")
        solver, ptrace = ThermalSolver(matrices, self.config), self._load_ptrace()
        if self.config["simulation_type"] == "steady":
            temperatures = solver.solve_steady(
                np.array(
                    [
                        np.mean([s.get(n, 0.0) for s in ptrace])
                        for n in matrices.unit_names
                    ]
                )
                if ptrace
                else np.zeros(len(matrices.unit_names))
            )
            print("[INFO] Exporting results...")
            self._export_vtu(topo, temperatures, "result.vtu")
        else:
            temperatures = solver.solve_transient(
                self.config["timestep"],
                ptrace,
                self._get_init_temp(topo),
                topo.volumes,
                fields.cp,
            )
            print("[INFO] Exporting results...")
            self._export_vtu(topo, temperatures, "transient_result.vtu")
        end = time.perf_counter()
        print(f"[INFO] Simulation completed in {end - start:.2f} seconds\n\n")

    def _load_ptrace(self) -> list[dict]:
        path = os.path.join(self.base_dir, self.config.get("ptrace_file_path", ""))
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            headers = f.readline().split()
            return [dict(zip(headers, map(float, l.split()))) for l in f if l.strip()]

    def _get_init_temp(self, topo: MeshTopology) -> np.ndarray:
        temp = np.full(topo.n_cells, float(self.config["init_temperature"]))
        init_file = self.config.get("init_temperature_file_path")
        if init_file and os.path.exists(os.path.join(self.base_dir, init_file)):
            init_mesh, offset = meshio.read(os.path.join(self.base_dir, init_file)), 0
            for block, block_temps in zip(
                init_mesh.cells, init_mesh.cell_data.get("Temperature_K", [])
            ):
                if block.type == "hexahedron":
                    count = len(block_temps)
                    valid_ids = np.arange(offset, offset + count)
                    valid_mask = valid_ids < len(topo.orig_to_new_id)
                    temp[topo.orig_to_new_id[valid_ids[valid_mask]]] = block_temps[
                        valid_mask
                    ]
                    offset += count
        return temp

    def _export_vtu(self, topo: MeshTopology, temperatures: np.ndarray, filename: str):
        mapped, orig_mesh, hex_blocks, temp_chunks, offset = (
            np.empty(topo.n_cells),
            meshio.read(self.mesh_path),
            [],
            [],
            0,
        )
        mapped[topo.sorted_indices] = temperatures
        for block in orig_mesh.cells:
            if block.type == "hexahedron":
                count = len(block.data)
                hex_blocks.append(block)
                temp_chunks.append(mapped[offset : offset + count])
                offset += count
        meshio.Mesh(
            orig_mesh.points, hex_blocks, cell_data={"Temperature_K": temp_chunks}
        ).write(os.path.join(self.base_dir, filename))

```

### File: metahotspot_types.py
```py
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

```

### File: model25d.py
```py
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ==========================================
# 单一真相：全局默认配置与标准材料库
# ==========================================
DEFAULT_CONFIG = {
    "simulation_type": "steady",
    "ambient": 318.15,
    "init_temperature": 318.15,
    "t_chip": 0.00015,
    "t_tim": 0.00002,
    "t_spreader": 0.001,
    "t_sink": 0.0069,
    "base_proc_freq": 3.0e9,
    "r_convec": 0.1,
    "material_interface": "tim",
    "material_spreader": "copper",
    "material_sink": "copper",
    "init_file": "",
    "sampling_intvl": 0.01,
    "time": 0.01,
    "timestep": 0.01,
    "mesh_file_path": "mesh.msh",
    "ptrace_file_path": "",
    "init_temperature_file_path": "",
    "pumping_pressure": 52000.0,
    "inlet_temperature": 298.15,
    "boundary_conditions": [],
    "stackup": [],
    "materials": {},
}

STANDARD_MATERIALS = {
    "silicon": {
        "k": 130.0,
        "cp": 1.63e6,
        "fluid": False,
        "density": 2330.0,
        "dynamic_viscosity": 0.0,
    },
    "copper": {
        "k": 400.0,
        "cp": 3.44e6,
        "fluid": False,
        "density": 8960.0,
        "dynamic_viscosity": 0.0,
    },
    "aluminum": {
        "k": 237.0,
        "cp": 2.42e6,
        "fluid": False,
        "density": 2700.0,
        "dynamic_viscosity": 0.0,
    },
    "tim": {
        "k": 4.0,
        "cp": 4.0e6,
        "fluid": False,
        "density": 1000.0,
        "dynamic_viscosity": 0.0,
    },
    "water": {
        "k": 0.6069,
        "cp": 4.17e6,
        "fluid": True,
        "density": 1000.0,
        "dynamic_viscosity": 8.89e-4,
    },
    "default_solid": {
        "k": 1.0,
        "cp": 1.0e6,
        "fluid": False,
        "density": 1000.0,
        "dynamic_viscosity": 0.0,
    },
}


@dataclass(slots=True)
class Unit2D:
    """2D layout unit for FVM mesh generation with full property resolution."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: str
    k: float
    cp: float
    density: float
    dynamic_viscosity: float
    is_fluid: bool


@dataclass(slots=True)
class Layer25D:
    """2.5D layer definition with fully resolved properties."""

    name: str
    tag: int
    thickness: float
    material: str
    k: float
    cp: float
    density: float
    dynamic_viscosity: float
    is_fluid: bool
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return merge_with_defaults(raw_config)


def merge_with_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    for k, v in raw_config.items():
        if k in config and type(config[k]) is not type(v):
            try:
                if v not in {"(null)", "null", "None", ""}:
                    config[k] = type(config[k])(v)
            except ValueError:
                config[k] = v
        else:
            config[k] = v

    config["t_interface"] = raw_config.get("t_interface", config["t_tim"])
    config["time"] = raw_config.get("time", max(config["sampling_intvl"], 0.01))
    config["timestep"] = raw_config.get("timestep", config["sampling_intvl"])

    if "init_temp" in raw_config:
        config["init_temperature"] = float(raw_config["init_temp"])

    for mat_name, mat_props in STANDARD_MATERIALS.items():
        if mat_name not in config["materials"]:
            config["materials"][mat_name] = dict(mat_props)

    return config


def _resolve_prop(
    key: str, unit_data: dict, unit_mat: dict, layer_mat: dict, default_mat: dict
) -> Any:
    """单一回退关口：严格执行 局部设定 > 单元材料 > 层材料 > 默认材料 优先级"""
    if key in unit_data and unit_data[key] is not None:
        return unit_data[key]
    if key in unit_mat and unit_mat[key] is not None:
        return unit_mat[key]
    if key in layer_mat and layer_mat[key] is not None:
        return layer_mat[key]
    return default_mat.get(key)


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    layers = []
    stackup_data = config.get("stackup", [])
    materials = config.get("materials", {})
    def_mat = materials.get("default_solid", STANDARD_MATERIALS["default_solid"])

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))

        layer_mat_name = layer_cfg.get("material", "silicon")
        layer_mat = materials.get(layer_mat_name, def_mat)
        layout_file = layer_cfg.get("layout_file", "")
        units = []

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        umat_name = u.get("material", layer_mat_name)
                        umat = materials.get(umat_name, layer_mat)

                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                material=umat_name,
                                k=float(
                                    _resolve_prop("k", u, umat, layer_mat, def_mat)
                                ),
                                cp=float(
                                    _resolve_prop("cp", u, umat, layer_mat, def_mat)
                                ),
                                density=float(
                                    _resolve_prop(
                                        "density", u, umat, layer_mat, def_mat
                                    )
                                ),
                                dynamic_viscosity=float(
                                    _resolve_prop(
                                        "dynamic_viscosity", u, umat, layer_mat, def_mat
                                    )
                                ),
                                is_fluid=bool(
                                    _resolve_prop("fluid", u, umat, layer_mat, def_mat)
                                ),
                            )
                        )

        if not units:
            units.append(
                Unit2D(
                    name=f"{name}_bulk",
                    lx=lx,
                    ly=ly,
                    dx=dx,
                    dy=dy,
                    material=layer_mat_name,
                    k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
                    cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
                    density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
                    dynamic_viscosity=float(
                        _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
                    ),
                    is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                material=layer_mat_name,
                k=float(_resolve_prop("k", {}, {}, layer_mat, def_mat)),
                cp=float(_resolve_prop("cp", {}, {}, layer_mat, def_mat)),
                density=float(_resolve_prop("density", {}, {}, layer_mat, def_mat)),
                dynamic_viscosity=float(
                    _resolve_prop("dynamic_viscosity", {}, {}, layer_mat, def_mat)
                ),
                is_fluid=bool(_resolve_prop("fluid", {}, {}, layer_mat, def_mat)),
                active=bool(layer_cfg.get("active", False)),
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers

```

### File: thermal_solver.py
```py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from metahotspot.metahotspot_types import SystemMatrices


class ThermalSolver:
    def __init__(self, matrices: SystemMatrices, config: dict):
        self.mat, self.config = matrices, config

    def solve_steady(self, mean_powers: np.ndarray) -> np.ndarray:
        temp = splinalg.spsolve(
            -self.mat.A_total, self.mat.b_total + (self.mat.power_matrix @ mean_powers)
        )
        print(f"[RESULT] T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K")
        return temp

    def solve_transient(
        self,
        dt: float,
        ptrace: list[dict],
        init_temp: np.ndarray,
        vols: np.ndarray,
        cp: np.ndarray,
    ) -> np.ndarray:
        c_mat, temp = sp.diags(cp * vols) / dt, init_temp.copy()
        solve_step = splinalg.factorized((c_mat - self.mat.A_total).tocsc())
        for i, step_power in enumerate(ptrace):
            temp = solve_step(
                (c_mat @ temp)
                + self.mat.b_total
                + (
                    self.mat.power_matrix
                    @ np.array([step_power.get(n, 0.0) for n in self.mat.unit_names])
                )
            )
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K"
                )
        return temp

```

### File: __init__.py
```py
"""MetaHotspot Python package."""

```

### File: legacy\converter.py
```py
import os
import json
import shutil
import csv
from typing import Dict, List, Tuple

from metahotspot.legacy.hotspot_parser import HotSpotParser
from metahotspot.model25d import merge_with_defaults, STANDARD_MATERIALS


def _find_first_by_suffix(directory: str, suffix: str) -> str:
    for entry in os.listdir(directory):
        if entry.endswith(suffix):
            return os.path.join(directory, entry)
    return ""


def _layout_bbox(units: List[dict]) -> Tuple[float, float, float, float]:
    if not units:
        return 0.0, 0.0, 0.01, 0.01
    min_x, min_y = min(u["left_x"] for u in units), min(u["bottom_y"] for u in units)
    max_x = max(u["left_x"] + u["width"] for u in units)
    max_y = max(u["bottom_y"] + u["height"] for u in units)
    return min_x, min_y, max_x - min_x, max_y - min_y


class SimulationModelBuilder25D:
    def __init__(self, parser: HotSpotParser, example_dir: str, output_dir: str):
        self.parser = parser
        self.example_dir = example_dir
        self.layouts_dir = os.path.join(output_dir, "layouts")
        os.makedirs(self.layouts_dir, exist_ok=True)

        raw_config = parser.parse_config(os.path.join(example_dir, "example.config"))
        self.config = merge_with_defaults(raw_config)

        self.materials: Dict[str, dict] = dict(STANDARD_MATERIALS)
        self.stackup: List[dict] = []
        self.boundary_conditions: List[dict] = []
        self.global_width, self.global_height = self._calculate_global_size()

    def _calculate_global_size(self) -> Tuple[float, float]:
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        files_to_check = (
            [
                l["flp_file"]
                for l in lcf_layers
                if not l.get("flp_file", "").lower().endswith(".csv")
            ]
            if lcf_layers
            else [f for f in os.listdir(self.example_dir) if f.endswith(".flp")]
        )

        widths, heights = [], []
        for file_name in files_to_check:
            units = self.parser.parse_flp(os.path.join(self.example_dir, file_name))
            if units:
                _, _, w, h = _layout_bbox(units)
                widths.append(w)
                heights.append(h)

        if not widths and any(
            l.get("flp_file", "").endswith(".csv") for l in lcf_layers
        ):
            return 0.03, 0.03
        return (max(widths), max(heights)) if widths else (0.01, 0.01)

    def build_materials(self) -> "SimulationModelBuilder25D":
        mat_path = os.path.join(self.example_dir, "example.materials")
        parsed_mats = self.parser.parse_materials(mat_path)
        self.materials.update(parsed_mats)

        if "coolant_visc" in self.config:
            self.materials["water"]["dynamic_viscosity"] = float(
                self.config["coolant_visc"]
            )
        return self

    def _export_layout_json(
        self,
        name: str,
        flp_units: List[dict],
        layer_k: float = None,
        layer_cp: float = None,
    ) -> str:
        if not flp_units:
            return ""
        min_x, min_y, lw, lh = _layout_bbox(flp_units)
        ox = (self.global_width - lw) / 2.0 - min_x
        oy = (self.global_height - lh) / 2.0 - min_y

        json_units = []
        for u in flp_units:
            unit_data = {
                "name": u["name"],
                "lx": u["left_x"] + ox,
                "ly": u["bottom_y"] + oy,
                "dx": u["width"],
                "dy": u["height"],
            }
            if layer_k is not None:
                unit_data["k"] = float(u.get("k", layer_k))
                unit_data["cp"] = float(u.get("specific_heat", layer_cp))
            json_units.append(unit_data)

        file_path = f"{name}_layout.json"
        with open(
            os.path.join(self.layouts_dir, file_path), "w", encoding="utf-8"
        ) as f:
            json.dump(json_units, f, indent=2)
        return f"layouts/{file_path}"

    def build_chip_layers(self) -> "SimulationModelBuilder25D":
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        if not lcf_layers:
            flp_units = self.parser.parse_flp(
                _find_first_by_suffix(self.example_dir, ".flp")
            )
            layout_ref = self._export_layout_json("layer_1", flp_units)
            self.stackup.append(
                self._create_layer_dict(
                    1,
                    "layer_1",
                    self.config["t_chip"],
                    "silicon",
                    bool(flp_units),
                    layout_ref,
                )
            )
            return self

        for layer in lcf_layers:
            tag = int(layer["id"]) + 1
            name = f"layer_{tag}"
            thickness = float(layer["thickness"])
            flp_file = layer.get("flp_file", "")

            is_numeric = layer["type"] == "numeric"
            mat_name = f"{name}_mat" if is_numeric else str(layer["material"])

            if is_numeric:
                self.materials[mat_name] = {
                    "k": float(layer["k"]),
                    "cp": float(layer["cp"]),
                    "fluid": False,
                }

            if flp_file.lower().endswith(".csv"):
                self._handle_microchannel_layer(
                    name, tag, thickness, os.path.join(self.example_dir, flp_file)
                )
                continue

            flp_units = self.parser.parse_flp(os.path.join(self.example_dir, flp_file))
            layout_ref = self._export_layout_json(
                name,
                flp_units,
                layer.get("k") if is_numeric else None,
                layer.get("cp") if is_numeric else None,
            )
            active = bool(layer.get("power") and flp_units)
            self.stackup.append(
                self._create_layer_dict(
                    tag, name, thickness, mat_name, active, layout_ref
                )
            )

        return self

    def build_package_and_cooling(self) -> "SimulationModelBuilder25D":
        has_lcf = bool(_find_first_by_suffix(self.example_dir, ".lcf"))

        if not has_lcf:
            self._add_pkg_layer(
                "TIM",
                self.config["t_interface"],
                self.global_width,
                self.config["material_interface"],
                1000,
            )

        s_spread = float(
            self.config.get("s_spreader", max(self.global_width, self.global_height))
        )
        self._add_pkg_layer(
            "Spreader",
            self.config["t_spreader"],
            s_spread,
            self.config["material_spreader"],
            1001,
        )

        s_sink = float(
            self.config.get("s_sink", max(self.global_width, self.global_height))
        )
        self._add_pkg_layer(
            "Sink", self.config["t_sink"], s_sink, self.config["material_sink"], 1002
        )

        self.boundary_conditions.append(
            {
                "name": "sink_conv",
                "type": "convection",
                "face": "+Z",
                "target": "Sink",
                "h": 1.0 / (self.config["r_convec"] * s_sink * s_sink),
                "T_inf": self.config["ambient"],
            }
        )

        if os.path.exists(os.path.join(self.example_dir, "horizontal.csv")) and not any(
            "microchannel" in l["name"] for l in self.stackup
        ):
            self._handle_microchannel_layer(
                "microchannel",
                500,
                0.0001,
                os.path.join(self.example_dir, "horizontal.csv"),
            )

        return self

    def _create_layer_dict(
        self,
        tag: int,
        name: str,
        thickness: float,
        material: str,
        active: bool,
        layout_file: str = "",
    ) -> dict:
        return {
            "tag": tag,
            "name": name,
            "thickness": thickness,
            "material": material,
            "active": active,
            "layout_file": layout_file,
            "lx": 0.0,
            "ly": 0.0,
            "dx": self.global_width,
            "dy": self.global_height,
        }

    def _add_pkg_layer(
        self, name: str, thick: float, side: float, mat_candidate: str, tag: int
    ):
        lx, ly = (self.global_width - side) / 2.0, (self.global_height - side) / 2.0
        mat_key = mat_candidate.strip().lower()

        layer = self._create_layer_dict(tag, name, thick, mat_key, False)
        layer.update({"lx": lx, "ly": ly, "dx": side, "dy": side})
        self.stackup.append(layer)

    def _handle_microchannel_layer(
        self, name: str, tag: int, thickness: float, csv_path: str
    ):
        mc_units = self._parse_microchannel_csv(csv_path)
        if mc_units:
            layout_path = f"{name}_microchannel_layout.json"
            with open(
                os.path.join(self.layouts_dir, layout_path), "w", encoding="utf-8"
            ) as f:
                json.dump(mc_units, f, indent=2)

            self.stackup.append(
                self._create_layer_dict(
                    tag, name, thickness, "silicon", True, f"layouts/{layout_path}"
                )
            )

            self.boundary_conditions.extend(
                [
                    {
                        "name": "mc_inlet",
                        "type": "pressure",
                        "face": "-X",
                        "target": name,
                        "pressure": self.config["pumping_pressure"],
                        "temperature": self.config["inlet_temperature"],
                    },
                    {
                        "name": "mc_outlet",
                        "type": "pressure",
                        "face": "+X",
                        "target": name,
                        "pressure": 0.0,
                    },
                ]
            )

    def _parse_microchannel_csv(self, csv_path: str) -> List[dict]:
        with open(csv_path, "r", encoding="utf-8") as f:
            grid = [
                [1 if int(x.strip()) > 0 else 0 for x in row if x.strip()]
                for row in csv.reader(f)
                if row
            ]

        if not grid:
            return []
        rows, cols = len(grid), len(grid[0])
        dx, dy = self.global_width / cols, self.global_height / rows
        visited, units = [[False] * cols for _ in range(rows)], []

        for r in range(rows):
            for c in range(cols):
                if visited[r][c]:
                    continue
                val, w, h = grid[r][c], 0, 1
                while c + w < cols and grid[r][c + w] == val and not visited[r][c + w]:
                    w += 1
                while r + h < rows:
                    if not all(
                        grid[r + h][c + i] == val and not visited[r + h][c + i]
                        for i in range(w)
                    ):
                        break
                    h += 1
                for i in range(h):
                    for j in range(w):
                        visited[r + i][c + j] = True

                is_fluid = val == 1
                mat = "water" if is_fluid else "silicon"

                units.append(
                    {
                        "name": f"mc_{'fluid' if is_fluid else 'solid'}_{len(units)}",
                        "lx": c * dx,
                        "ly": (rows - r - h) * dy,
                        "dx": w * dx,
                        "dy": h * dy,
                        "is_fluid": is_fluid,
                        "material": mat,
                    }
                )
        return units

    def get_result(self) -> dict:
        return {
            "config": self.config,
            "materials": self.materials,
            "stackup": self.stackup,
            "boundary_conditions": self.boundary_conditions,
        }


def convert_hotspot_to_metahotspot(
    example_dir: str,
    output_dir: str,
    simulation_type: str = "steady",
    config_name: str = "solver_config.json",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model = (
        SimulationModelBuilder25D(HotSpotParser(), example_dir, output_dir)
        .build_materials()
        .build_chip_layers()
        .build_package_and_cooling()
        .get_result()
    )

    cfg = model["config"]
    ptrace_path = _find_first_by_suffix(example_dir, ".ptrace")
    ptrace_name = os.path.basename(ptrace_path) if ptrace_path else ""
    if ptrace_path:
        shutil.copy(ptrace_path, os.path.join(output_dir, ptrace_name))

    json_data = {
        "simulation_type": simulation_type,
        "time": cfg["time"],
        "timestep": cfg["timestep"],
        "sampling_intvl": cfg["sampling_intvl"],
        "proc_freq": cfg["base_proc_freq"],
        "ambient": cfg["ambient"],
        "init_temperature": cfg["init_temperature"],
        "mesh_file_path": cfg["mesh_file_path"],
        "ptrace_file_path": ptrace_name,
        "materials": model["materials"],
        "stackup": model["stackup"],
        "boundary_conditions": model["boundary_conditions"],
    }

    if cfg["init_file"]:
        json_data["init_temperature_file_path"] = cfg["init_file"]

    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)
    return config_path


def convert_hotspot_with_modes(
    example_dir: str, output_dir: str, mode: str = "both"
) -> List[str]:
    mode = mode.lower().strip()
    res = []
    if mode in ("steady", "both"):
        res.append(
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "steady", "solver_config_steady.json"
            )
        )
    if mode in ("transient", "both"):
        res.append(
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "transient", "solver_config_transient.json"
            )
        )
    return res

```

### File: legacy\hotspot_parser.py
```py
import os
import re
from typing import Dict, Generator, List, Any


def _read_valid_lines(file_path: str) -> Generator[str, None, None]:
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


class HotSpotParser:
    @staticmethod
    def parse_flp(file_path: str) -> List[dict]:
        units: List[dict] = []
        for line in _read_valid_lines(file_path):
            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue

            unit = {
                "name": parts[0],
                "width": float(parts[1]),
                "height": float(parts[2]),
                "left_x": float(parts[3]),
                "bottom_y": float(parts[4]),
            }

            if len(parts) >= 7:
                try:
                    unit["specific_heat"] = float(parts[5])
                    resistivity = float(parts[6])
                    unit["k"] = 1.0 / resistivity if resistivity != 0 else 0.0
                except ValueError:
                    pass
            units.append(unit)
        return units

    @staticmethod
    def parse_config(file_path: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for line in _read_valid_lines(file_path):
            match = re.match(r"^-(\w+)\s+([^#]+)", line)
            if match:
                key, value = match.groups()
                try:
                    config[key] = float(value.strip())
                except ValueError:
                    config[key] = value.strip()
        return config

    @staticmethod
    def parse_materials(file_path: str) -> Dict[str, dict]:
        materials: Dict[str, dict] = {}
        lines = list(_read_valid_lines(file_path))
        index = 0
        while index < len(lines):
            name = lines[index]
            is_fluid = lines[index + 1].lower() == "fluid"
            materials[name] = {
                "k": float(lines[index + 2]),
                "cp": float(lines[index + 3]),
                "fluid": is_fluid,
            }
            if is_fluid:
                materials[name]["dynamic_viscosity"] = float(lines[index + 4])
                index += 5
            else:
                index += 4
        return materials

    @staticmethod
    def parse_lcf(file_path: str) -> List[dict]:
        layers: List[dict] = []
        lines = list(_read_valid_lines(file_path))
        index = 0
        while index < len(lines):
            layer_id = int(lines[index])
            active = lines[index + 2].upper() == "Y"
            field = lines[index + 3]
            try:
                cp = float(field)
                resistivity = float(lines[index + 4])
                layers.append(
                    {
                        "id": layer_id,
                        "power": active,
                        "cp": cp,
                        "k": 1.0 / resistivity if resistivity != 0 else 0.0,
                        "thickness": float(lines[index + 5]),
                        "flp_file": lines[index + 6],
                        "type": "numeric",
                    }
                )
                index += 7
            except ValueError:
                layers.append(
                    {
                        "id": layer_id,
                        "power": active,
                        "material": field,
                        "thickness": float(lines[index + 4]),
                        "flp_file": lines[index + 5],
                        "type": "named",
                    }
                )
                index += 6
        return layers

```

### File: legacy\__init__.py
```py
"""metahotspot.legacy submodule"""

```

