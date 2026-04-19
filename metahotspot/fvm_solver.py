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
    box: Tuple[float, float, float, float, float, float]
    k: float
    cp: float
    tag: int
    vol: float


def _overlap_area(
    box_a: Tuple[float, ...], box_b: Tuple[float, ...], axis: int
) -> float:
    """计算两个 3D 包围盒在指定法向平面上的重叠面积"""
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]  # 取另外两个轴的索引
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


def _intersection_volume(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """计算两个 3D 包围盒的相交体积"""
    dx = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
    dy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    dz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    return dx * dy * dz if dx > 0.0 and dy > 0.0 and dz > 0.0 else 0.0


class FVMSolver:
    GEOMETRY_TOLERANCE = 1e-12

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        self.mesh = meshio.read(
            os.path.join(self.base_dir, self.config.get("mesh_file_path", "mesh.msh"))
        )
        self.materials = self.config["materials"]

        self.tag_to_material = {
            tag: mat
            for mat, tags in self.config.get("domain_material_assignment", {}).items()
            for tag in tags
        }

        self.cells: List[Cell] = []
        self._prepare_mesh()
        self._precompute_power_maps()

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
        centers, dims = (lowers + uppers) * 0.5, uppers - lowers
        vols = np.prod(dims, axis=1)

        # 1. 提取所需的所有数据并创建 Cell 实例
        for i in range(len(hex_data)):
            tag = int(physical_tags[i])
            mat = self.materials[self.tag_to_material.get(tag, "silicon")]
            self.cells.append(
                Cell(
                    original_id=i,
                    id=i,
                    center=centers[i],
                    dims=dims[i],
                    box=(*lowers[i], *uppers[i]),
                    k=float(mat["k"]),
                    cp=float(mat["cp"]),
                    tag=tag,
                    vol=float(vols[i]),
                )
            )

        # 2. NumPy 向量化计算 Morton Key 以大幅加速排序
        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        # 3. 依据 Morton Key 进行内存局部性重排序
        self.cells = [self.cells[i] for i in np.argsort(morton_keys)]
        for new_id, cell in enumerate(self.cells):
            cell.id = new_id

    def _precompute_power_maps(self) -> None:
        """预计算各热源对网格的映射向量，消除瞬态计算中的 O(N^2) 复杂度"""
        self.power_maps: Dict[str, np.ndarray] = {}
        for unit in self.config.get("power_units", []):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            box = (
                unit["lx"],
                unit["ly"],
                unit["lz"],
                unit["lx"] + unit["dx"],
                unit["ly"] + unit["dy"],
                unit["lz"] + unit["dz"],
            )
            vec = np.zeros(len(self.cells))
            for cell in self.cells:
                intersect = _intersection_volume(cell.box, box)
                if intersect > 1e-15:
                    vec[cell.id] = intersect / vol
            self.power_maps[unit["name"]] = vec

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
        print(
            f"[INFO] Building full 3D non-conformal G matrix ({len(self.cells)} cells) via Sweep-and-Prune..."
        )
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE

        # 沿 X 轴排序用于 Sweep and Prune
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            # 剪枝：剔除 X 轴已经不重叠的单元
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]

            for c_b in active_list:
                # 检查 Y 和 Z 轴重叠
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
                    continue

                # 提取接触面进行热导计算
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
            h, t_inf, selection = (
                float(bc["h"]),
                float(bc["T_inf"]),
                set(bc.get("selection", [])),
            )

            for c in self.cells:
                if c.tag in selection and abs(c.box[5] - z_max) < 1e-6:
                    area = c.dims[0] * c.dims[1]
                    g = 1.0 / ((0.5 * c.dims[2] / (c.k * area)) + (1.0 / (h * area)))
                    rows.append(c.id)
                    cols.append(c.id)
                    data.append(-g)
                    rhs[c.id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _get_power_rhs(self, power_step: Dict[str, float]) -> np.ndarray:
        """从预计算的映射中快速生成当前步的 rhs 向量"""
        rhs = np.zeros(len(self.cells))
        for name, val in power_step.items():
            if val != 0 and name in self.power_maps:
                rhs += val * self.power_maps[name]
        return rhs

    def solve(self) -> None:
        # 读取功率 trace
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

        g_total = self.assemble_g_matrix() + self._build_boundary_terms()[0]
        boundary_rhs = self._build_boundary_terms()[1]

        if self.config.get("simulation_type", "steady") == "steady":
            print("[SIM] Solving steady state...")
            steady_step = {
                u["name"]: (
                    float(np.mean([s.get(u["name"], 0.0) for s in ptrace_steps]))
                    if ptrace_steps
                    else 0.0
                )
                for u in self.config.get("power_units", [])
            }
            temperatures = splinalg.spsolve(
                -g_total, boundary_rhs + self._get_power_rhs(steady_step)
            )
            print(
                f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
            )
            self.save(temperatures, "result.vtu")
            return

        print("[SIM] Solving transient...")
        dt, total_time = float(self.config.get("timestep", 0.1)), float(
            self.config.get("time", 0.0)
        )
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)
        if not ptrace_steps:
            ptrace_steps = [{}] * n_steps

        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - g_total).tocsc())

        # 初始温度处理
        temperatures = np.full(
            len(self.cells), float(self.config.get("init_temperature", 318.15))
        )
        # 如果需要加载自定义温度场，可以在此添加之前那段 _load_initial_temperature 的逻辑

        for i, step_power in enumerate(ptrace_steps):
            rhs = (
                (c_mat @ temperatures) + boundary_rhs + self._get_power_rhs(step_power)
            )
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
