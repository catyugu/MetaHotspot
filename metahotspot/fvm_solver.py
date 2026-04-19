import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml


@dataclass(slots=True)
class Cell:
    original_id: int
    id: int
    center: np.ndarray
    dims: np.ndarray
    box: np.ndarray  # [xmin, ymin, zmin, xmax, ymax, zmax]
    k: float
    cp: float
    tag: int
    vol: float


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    """计算两个 3D 包围盒在指定法向平面上的重叠面积"""
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]  # 取另外两个轴的索引
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    GEOMETRY_TOLERANCE = 1e-12
    DEFAULT_INITIAL_TEMPERATURE = 318.15

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        self.mesh_path = os.path.join(
            self.base_dir, self.config.get("mesh_file_path", "mesh.msh")
        )
        self.mesh = meshio.read(self.mesh_path)

        self._init_materials()
        self.cells: List[Cell] = []

        self._prepare_mesh()
        self._precompute_power_matrix()

    def _init_materials(self) -> None:
        """解耦材质加载，构建 tag 到材质属性的扁平映射表"""
        self.materials = self.config["materials"]
        self.tag_to_material = {}
        for mat_name, tags in self.config.get("domain_material_assignment", {}).items():
            for tag in tags:
                self.tag_to_material[tag] = self.materials[mat_name]

    def _prepare_mesh(self) -> None:
        print("[INFO] Preparing mesh data...")
        hex_blocks = [b.data for b in self.mesh.cells if b.type == "hexahedron"]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")

        hex_data = np.vstack(hex_blocks)
        physical_tags = self.mesh.cell_data_dict.get("gmsh:physical", {}).get(
            "hexahedron", np.full(len(hex_data), -1)
        )

        coords = self.mesh.points[hex_data]
        lowers, uppers = np.min(coords, axis=1), np.max(coords, axis=1)
        centers = (lowers + uppers) * 0.5
        dims = uppers - lowers
        vols = np.prod(dims, axis=1)

        # 向量化 Morton Key 排序以加速内存局部性访问
        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        sorted_indices = np.argsort(morton_keys)

        # 依据空间排序构建 Cells
        for new_id, orig_id in enumerate(sorted_indices):
            tag = int(physical_tags[orig_id])
            mat = self.tag_to_material.get(tag, self.materials["silicon"])
            box = np.array([*lowers[orig_id], *uppers[orig_id]])

            self.cells.append(
                Cell(
                    original_id=orig_id,
                    id=new_id,
                    center=centers[orig_id],
                    dims=dims[orig_id],
                    box=box,
                    k=float(mat["k"]),
                    cp=float(mat["cp"]),
                    tag=tag,
                    vol=float(vols[orig_id]),
                )
            )

        # 建立原 ID 到新 ID 的映射表，用于初始温度和结果保存
        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}

    def _precompute_power_matrix(self) -> None:
        """预计算各热源对网格的映射矩阵 (Cells × PowerUnits)"""
        power_units = self.config.get("power_units", [])
        self.unit_names = [u["name"] for u in power_units]

        rows, cols, data = [], [], []

        for unit_idx, unit in enumerate(power_units):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            box_b = np.array(
                [
                    unit["lx"],
                    unit["ly"],
                    unit["lz"],
                    unit["lx"] + unit["dx"],
                    unit["ly"] + unit["dy"],
                    unit["lz"] + unit["dz"],
                ]
            )

            for cell in self.cells:
                # 快速包围盒相交体积计算
                box_a = cell.box
                overlap = np.maximum(
                    0,
                    np.minimum(box_a[3:], box_b[3:]) - np.maximum(box_a[:3], box_b[:3]),
                )
                intersect_vol = np.prod(overlap)

                if intersect_vol > 1e-15:
                    rows.append(cell.id)
                    cols.append(unit_idx)
                    data.append(intersect_vol / vol)

        n_cells = len(self.cells)
        n_units = len(power_units)
        self.power_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_cells, n_units)
        )

    def _get_initial_temperatures(self, n_cells: int) -> np.ndarray:
        """加载初始温度场（支持稳态结果向瞬态的完美继承）"""
        init_file = self.config.get("init_temperature_file_path")
        if not init_file or init_file in {"(null)", "None", ""}:
            return np.full(
                n_cells,
                float(
                    self.config.get(
                        "init_temperature", self.DEFAULT_INITIAL_TEMPERATURE
                    )
                ),
            )

        init_path = os.path.join(self.base_dir, init_file)
        if not os.path.exists(init_path):
            print(
                f"[WARNING] Init file {init_path} not found. Using default {self.DEFAULT_INITIAL_TEMPERATURE} K."
            )
            return np.full(
                n_cells,
                float(
                    self.config.get(
                        "init_temperature", self.DEFAULT_INITIAL_TEMPERATURE
                    )
                ),
            )

        print(f"[INFO] Loading initial state from {init_path}")
        init_mesh = meshio.read(init_path)
        temps = np.zeros(n_cells)

        offset = 0
        hex_data = init_mesh.cell_data.get("Temperature_K", [])

        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type == "hexahedron":
                for i, t in enumerate(block_temps):
                    orig_id = offset + i
                    new_id = self.orig_to_new_id.get(orig_id)
                    if new_id is not None:
                        temps[new_id] = t
                offset += len(block_temps)

        return temps

    def _add_pairwise_conductance(
        self,
        rows: list,
        cols: list,
        data: list,
        c_a: Cell,
        c_b: Cell,
        axis: int,
        area: float,
    ) -> None:
        if area <= 1e-15:
            return
        res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
            c_b.dims[axis] / (2.0 * c_b.k * area)
        )
        if res <= 1e-20:
            return

        g = 1.0 / res
        rows.extend([c_a.id, c_b.id, c_a.id, c_b.id])
        cols.extend([c_a.id, c_b.id, c_b.id, c_a.id])
        data.extend([-g, -g, g, g])

    def assemble_g_matrix(self) -> sp.csr_matrix:
        """利用 Sweep-and-Prune 算法构建三维热导矩阵"""
        print(
            f"[INFO] Building full 3D non-conformal G matrix ({len(self.cells)} cells)..."
        )
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE

        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            # X轴剪枝
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]

            for c_b in active_list:
                # 检查 Y 和 Z 轴重叠
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
                    continue

                # 提取接触面并计算热导
                for axis in range(3):
                    if (
                        abs(c_a.box[axis + 3] - c_b.box[axis]) < tol
                        or abs(c_a.box[axis] - c_b.box[axis + 3]) < tol
                    ):
                        area = _overlap_area(c_a.box, c_b.box, axis)
                        self._add_pairwise_conductance(
                            rows, cols, data, c_a, c_b, axis, area
                        )

            active_list.append(c_a)

        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(self.cells))
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n = len(self.cells)
        rhs, rows, cols, data = np.zeros(n), [], [], []
        z_max = max(c.box[5] for c in self.cells)

        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "convection":
                continue
            h = float(bc["h"])
            t_inf = float(bc["T_inf"])
            selection = set(bc.get("selection", []))

            for c in self.cells:
                if c.tag in selection and abs(c.box[5] - z_max) < 1e-6:
                    area = c.dims[0] * c.dims[1]
                    g = 1.0 / ((0.5 * c.dims[2] / (c.k * area)) + (1.0 / (h * area)))
                    rows.append(c.id)
                    cols.append(c.id)
                    data.append(-g)
                    rhs[c.id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def solve(self) -> None:
        ptrace_path = os.path.join(
            self.base_dir, self.config.get("ptrace_file_path", "")
        )
        ptrace_steps = []
        if os.path.exists(ptrace_path):
            with open(ptrace_path, "r", encoding="utf-8") as f:
                headers = f.readline().split()
                ptrace_steps = [
                    dict(zip(headers, map(float, line.split())))
                    for line in f
                    if line.strip()
                ]

        g_matrix = self.assemble_g_matrix()
        g_bc, boundary_rhs = self._build_boundary_terms()
        g_total = g_matrix + g_bc

        if self.config.get("simulation_type", "steady") == "steady":
            print("[SIM] Solving steady state...")
            mean_powers = np.array(
                [
                    (
                        np.mean([s.get(name, 0.0) for s in ptrace_steps])
                        if ptrace_steps
                        else 0.0
                    )
                    for name in self.unit_names
                ]
            )

            power_rhs = self.power_matrix @ mean_powers
            temperatures = splinalg.spsolve(-g_total, boundary_rhs + power_rhs)

            print(
                f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
            )
            self.save(temperatures, "result.vtu")
            return

        print("[SIM] Solving transient...")
        dt = float(self.config.get("timestep", 0.1))
        total_time = float(self.config.get("time", 0.0))
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)

        if not ptrace_steps:
            ptrace_steps = [{}] * n_steps

        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - g_total).tocsc())

        temperatures = self._get_initial_temperatures(len(self.cells))

        for i, step_power in enumerate(ptrace_steps):
            power_vec = np.array(
                [step_power.get(name, 0.0) for name in self.unit_names]
            )
            power_rhs = self.power_matrix @ power_vec

            rhs = (c_mat @ temperatures) + boundary_rhs + power_rhs
            temperatures = solve_step(rhs)

            if i % 10 == 0 or i == len(ptrace_steps) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )

        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        # 将排序后的结果映射回原始 Mesh 顺序
        mapped = np.zeros(len(self.cells))
        for c in self.cells:
            mapped[c.original_id] = temperatures[c.id]

        offset, temp_chunks = 0, []
        for block in self.mesh.cells:
            count = len(block.data)
            if block.type == "hexahedron":
                temp_chunks.append(mapped[offset : offset + count])
                offset += count
            else:
                temp_chunks.append(np.full(count, np.nan))

        self.mesh.cell_sets.clear()
        self.mesh.cell_data = {"Temperature_K": temp_chunks}
        self.mesh.write(os.path.join(self.base_dir, output_name))
        print(f"[FILE] Results saved to {output_name}")
