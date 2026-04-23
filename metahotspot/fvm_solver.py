import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml

from metahotspot.model25d import load_stackup


@dataclass(slots=True)
class Cell:
    original_id: int
    id: int
    center: np.ndarray
    dims: np.ndarray
    box: np.ndarray
    k: float
    cp: float
    tag: int
    vol: float


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
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

        self._sanitize_config()
        self._init_materials_and_stackup()

        self.cells: List[Cell] = []
        self._prepare_mesh()
        self._precompute_power_matrix()

    def _sanitize_config(self) -> None:
        self.config["init_temperature"] = float(
            self.config.get("init_temperature", self.DEFAULT_INITIAL_TEMPERATURE)
        )
        self.config["timestep"] = float(self.config.get("timestep", 0.1))
        self.config["time"] = float(self.config.get("time", 0.0))
        self.config["simulation_type"] = str(
            self.config.get("simulation_type", "steady")
        )
        self.config["ptrace_file_path"] = str(self.config.get("ptrace_file_path", ""))
        self.config.setdefault("stackup", [])
        self.config.setdefault("boundary_conditions", [])
        self.config.setdefault("init_temperature_file_path", None)

    def _init_materials_and_stackup(self) -> None:
        self.materials = self.config.get("materials", {})
        self.stackup = load_stackup(self.config, self.base_dir)

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

        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        sorted_indices = np.argsort(morton_keys)

        mat_k_array, mat_cp_array = np.zeros(len(centers)), np.zeros(len(centers))

        # 核心改动：利用 2.5D Stackup 为 3D 网格中心点映射材料属性
        tol = self.GEOMETRY_TOLERANCE
        z_cursor = 0.0
        for layer in self.stackup:
            z_min = z_cursor
            z_max = z_cursor + layer.thickness
            z_cursor = z_max

            layer_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)
            if not np.any(layer_mask):
                continue

            def_mat = self.materials.get(
                layer.default_material, {"k": 1.0, "cp": 1.0e6}
            )
            mat_k_array[layer_mask] = float(def_mat["k"])
            mat_cp_array[layer_mask] = float(def_mat["cp"])

            # 覆盖异构材料单元
            for u in layer.units:
                u_mask = (
                    layer_mask
                    & (centers[:, 0] >= u.lx - tol)
                    & (centers[:, 0] <= u.lx + u.dx + tol)
                    & (centers[:, 1] >= u.ly - tol)
                    & (centers[:, 1] <= u.ly + u.dy + tol)
                )
                if np.any(u_mask):
                    if u.k is not None:
                        mat_k_array[u_mask] = u.k
                        mat_cp_array[u_mask] = u.cp
                    elif u.material and u.material in self.materials:
                        mat_k_array[u_mask] = float(self.materials[u.material]["k"])
                        mat_cp_array[u_mask] = float(self.materials[u.material]["cp"])

        self.face_to_cell = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]
            self.cells.append(
                Cell(
                    original_id=orig_id,
                    id=new_id,
                    center=centers[orig_id],
                    dims=dims[orig_id],
                    box=np.array([*lowers[orig_id], *uppers[orig_id]]),
                    k=mat_k_array[orig_id],
                    cp=mat_cp_array[orig_id],
                    tag=int(physical_tags[orig_id]),
                    vol=float(vols[orig_id]),
                )
            )

            fs = [
                tuple(sorted([nodes[0], nodes[3], nodes[2], nodes[1]])),
                tuple(sorted([nodes[4], nodes[5], nodes[6], nodes[7]])),
                tuple(sorted([nodes[0], nodes[1], nodes[5], nodes[4]])),
                tuple(sorted([nodes[3], nodes[7], nodes[6], nodes[2]])),
                tuple(sorted([nodes[0], nodes[4], nodes[7], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[6], nodes[5]])),
            ]
            for f in fs:
                self.face_to_cell[f] = new_id

        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}
        self._extract_boundary_faces()

    def _extract_boundary_faces(self) -> None:
        self.boundary_faces = {}
        if "quad" not in self.mesh.cells_dict:
            return
        quad_data = self.mesh.cells_dict["quad"]
        quad_tags = self.mesh.cell_data_dict.get("gmsh:physical", {}).get("quad", [])

        for i, nodes in enumerate(quad_data):
            f = tuple(sorted(nodes))
            if f not in self.face_to_cell:
                continue
            tag = int(quad_tags[i]) if len(quad_tags) > i else -1
            if tag == -1:
                continue

            p = self.mesh.points[nodes]
            area = np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
            if area > self.GEOMETRY_TOLERANCE:
                self.boundary_faces.setdefault(tag, []).append(
                    (self.face_to_cell[f], area)
                )

    def _precompute_power_matrix(self) -> None:
        # 核心改动：从 2.5D Stackup 收集 active_units 的 3D 信息
        active_units_3d = []
        z_cursor = 0.0
        for layer in self.stackup:
            if layer.active:
                for u in layer.units:
                    active_units_3d.append(
                        {
                            "name": u.name,
                            "lx": u.lx,
                            "ly": u.ly,
                            "lz": z_cursor,
                            "dx": u.dx,
                            "dy": u.dy,
                            "dz": layer.thickness,
                        }
                    )
            z_cursor += layer.thickness

        self.unit_names = [u["name"] for u in active_units_3d]

        if not active_units_3d or not self.cells:
            self.power_matrix = sp.csr_matrix((len(self.cells), 0))
            return

        cell_boxes = np.array([c.box for c in self.cells])
        cell_lowers, cell_uppers = cell_boxes[:, :3], cell_boxes[:, 3:]
        rows, cols, data = [], [], []

        for unit_idx, unit in enumerate(active_units_3d):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            u_lower = np.array([unit["lx"], unit["ly"], unit["lz"]])
            u_upper = u_lower + np.array([unit["dx"], unit["dy"], unit["dz"]])

            overlap_lowers = np.maximum(cell_lowers, u_lower)
            overlap_uppers = np.minimum(cell_uppers, u_upper)
            overlap_dims = np.maximum(0, overlap_uppers - overlap_lowers)

            intersect_vols = np.prod(overlap_dims, axis=1)
            valid_mask = intersect_vols > self.GEOMETRY_TOLERANCE

            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 0:
                rows.extend(valid_indices)
                cols.extend([unit_idx] * len(valid_indices))
                data.extend(intersect_vols[valid_mask] / vol)

        self.power_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(active_units_3d))
        )

    def _get_initial_temperatures(self, n_cells: int) -> np.ndarray:
        default_temp = self.config["init_temperature"]
        init_file = self.config["init_temperature_file_path"]
        if not init_file or init_file in {"(null)", "None", ""}:
            return np.full(n_cells, default_temp)
        init_path = os.path.join(self.base_dir, init_file)
        if not os.path.exists(init_path):
            return np.full(n_cells, default_temp)

        init_mesh = meshio.read(init_path)
        temps = np.zeros(n_cells)
        offset = 0
        hex_data = init_mesh.cell_data.get("Temperature_K", [])
        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type != "hexahedron":
                continue
            for i, t in enumerate(block_temps):
                new_id = self.orig_to_new_id.get(offset + i)
                if new_id is not None:
                    temps[new_id] = t
            offset += len(block_temps)
        return temps

    def assemble_g_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]
            for c_b in active_list:
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
                    continue
                for axis in range(3):
                    if not (
                        abs(c_a.box[axis + 3] - c_b.box[axis]) < tol
                        or abs(c_a.box[axis] - c_b.box[axis + 3]) < tol
                    ):
                        continue
                    area = _overlap_area(c_a.box, c_b.box, axis)
                    if area <= tol:
                        continue
                    res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                        c_b.dims[axis] / (2.0 * c_b.k * area)
                    )
                    if res <= tol:
                        continue
                    g = 1.0 / res
                    rows.extend([c_a.id, c_b.id, c_a.id, c_b.id])
                    cols.extend([c_a.id, c_b.id, c_b.id, c_a.id])
                    data.extend([-g, -g, g, g])
            active_list.append(c_a)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(self.cells))
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n = len(self.cells)
        rhs, rows, cols, data = np.zeros(n), [], [], []
        for bc in self.config["boundary_conditions"]:
            if bc.get("type") != "convection":
                continue
            h, t_inf = float(bc["h"]), float(bc["T_inf"])
            for tag in bc.get("selection", []):
                for cell_id, area in self.boundary_faces.get(tag, []):
                    c = self.cells[cell_id]
                    dist = c.vol / area
                    g = area / ((0.5 * dist / c.k) + (1.0 / h))
                    rows.append(c.id)
                    cols.append(c.id)
                    data.append(-g)
                    rhs[c.id] += g * t_inf
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _load_ptrace(self) -> List[dict]:
        ptrace_path = os.path.join(self.base_dir, self.config["ptrace_file_path"])
        if not os.path.exists(ptrace_path):
            return []
        with open(ptrace_path, "r", encoding="utf-8") as f:
            headers = f.readline().split()
            return [
                dict(zip(headers, map(float, line.split())))
                for line in f
                if line.strip()
            ]

    def solve(self) -> None:
        self.g_total = self.assemble_g_matrix() + self._build_boundary_terms()[0]
        self.boundary_rhs = self._build_boundary_terms()[1]
        self.ptrace_steps = self._load_ptrace()
        if self.config["simulation_type"] == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
        mean_powers = np.array(
            [
                (
                    np.mean([s.get(name, 0.0) for s in self.ptrace_steps])
                    if self.ptrace_steps
                    else 0.0
                )
                for name in self.unit_names
            ]
        )
        power_rhs = self.power_matrix @ mean_powers
        temperatures = splinalg.spsolve(-self.g_total, self.boundary_rhs + power_rhs)
        print(
            f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
        )
        self.save(temperatures, "result.vtu")

    def _solve_transient(self) -> None:
        dt, total_time = self.config["timestep"], self.config["time"]
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)
        ptrace = self.ptrace_steps or [{}] * n_steps
        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - self.g_total).tocsc())
        temperatures = self._get_initial_temperatures(len(self.cells))

        for i, step_power in enumerate(ptrace):
            power_vec = np.array(
                [step_power.get(name, 0.0) for name in self.unit_names]
            )
            rhs = (
                (c_mat @ temperatures)
                + self.boundary_rhs
                + (self.power_matrix @ power_vec)
            )
            temperatures = solve_step(rhs)
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )
        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        import meshio

        mapped = np.zeros(len(self.cells))
        for c in self.cells:
            mapped[c.original_id] = temperatures[c.id]
        hex_blocks, temp_chunks, offset = [], [], 0
        for block in self.mesh.cells:
            if block.type == "hexahedron":
                count = len(block.data)
                hex_blocks.append(block)
                temp_chunks.append(mapped[offset : offset + count])
                offset += count
        out_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells=hex_blocks,
            cell_data={"Temperature_K": temp_chunks},
        )
        out_mesh.write(os.path.join(self.base_dir, output_name))
