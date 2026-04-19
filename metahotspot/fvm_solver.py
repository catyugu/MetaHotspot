import math
import os
from typing import Dict, List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml


def _compute_morton_key(
    x: float,
    y: float,
    z: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    bits: int = 10,
) -> int:
    def _float_to_int(val: float, min_val: float, max_val: float, bits: int) -> int:
        if max_val == min_val:
            return 0
        t = int((val - min_val) / (max_val - min_val) * ((1 << bits) - 1))
        return max(0, min(t, (1 << bits) - 1))

    ix = _float_to_int(x, x_min, x_max, bits)
    iy = _float_to_int(y, y_min, y_max, bits)
    iz = _float_to_int(z, z_min, z_max, bits)

    morton = 0
    for i in range(bits):
        morton |= ((ix >> i) & 1) << (3 * i)
        morton |= ((iy >> i) & 1) << (3 * i + 1)
        morton |= ((iz >> i) & 1) << (3 * i + 2)
    return morton


def _overlap_area(
    box_a: Tuple[float, ...], box_b: Tuple[float, ...], axis: int
) -> float:
    if axis == 0:
        dy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
        dz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
        return dy * dz if dy > 0.0 and dz > 0.0 else 0.0
    elif axis == 1:
        dx = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        dz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
        return dx * dz if dx > 0.0 and dz > 0.0 else 0.0
    else:
        dx = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        dy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
        return dx * dy if dx > 0.0 and dy > 0.0 else 0.0


def _intersection_volume(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    dx = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
    if dx <= 0.0:
        return 0.0
    dy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    if dy <= 0.0:
        return 0.0
    dz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    if dz <= 0.0:
        return 0.0
    return dx * dy * dz


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
            tag: material
            for material, tags in self.config.get(
                "domain_material_assignment", {}
            ).items()
            for tag in tags
        }

        self.cells: List[dict] = []
        self._prepare_mesh()

    def _prepare_mesh(self) -> None:
        print("[INFO] Preparing mesh data...")

        hex_blocks = [
            block.data for block in self.mesh.cells if block.type == "hexahedron"
        ]
        if not hex_blocks:
            raise ValueError("No hexahedron cells found in mesh")

        hex_data = np.vstack(hex_blocks)
        physical_tags = self.mesh.cell_data_dict.get("gmsh:physical", {}).get(
            "hexahedron"
        )
        if physical_tags is None:
            physical_tags = np.full(len(hex_data), -1, dtype=int)

        points = self.mesh.points
        coords = points[hex_data]
        lowers = np.min(coords, axis=1)
        uppers = np.max(coords, axis=1)
        centers = (lowers + uppers) / 2.0
        dims = uppers - lowers
        vols = np.prod(dims, axis=1)

        for cell_id in range(len(hex_data)):
            tag = int(physical_tags[cell_id])
            material_name = self.tag_to_material.get(tag, "silicon")
            material = self.materials[material_name]

            self.cells.append(
                {
                    "original_id": cell_id,  # 记录原始 ID，防止保存时乱码
                    "id": cell_id,  # 之后会被排序更新
                    "center": centers[cell_id],
                    "dims": dims[cell_id],
                    "box": (*lowers[cell_id], *uppers[cell_id]),
                    "k": float(material["k"]),
                    "cp": float(material["cp"]),
                    "tag": tag,
                    "vol": float(vols[cell_id]),
                }
            )

        x_min, y_min, z_min = np.min(lowers, axis=0)
        x_max, y_max, z_max = np.max(uppers, axis=0)

        # Morton 重排序以加速缓存局部性
        self.cells.sort(
            key=lambda c: _compute_morton_key(
                c["center"][0],
                c["center"][1],
                c["center"][2],
                x_min,
                x_max,
                y_min,
                y_max,
                z_min,
                z_max,
            )
        )

        for new_id, cell in enumerate(self.cells):
            cell["id"] = new_id

    def _add_pairwise_conductance(
        self,
        rows: List[int],
        cols: List[int],
        data: List[float],
        cell_a: dict,
        cell_b: dict,
        axis: int,
        area: float,
    ) -> None:
        if area <= 1e-15:
            return

        dist_a = cell_a["dims"][axis] / 2.0
        dist_b = cell_b["dims"][axis] / 2.0
        resistance = (dist_a / (cell_a["k"] * area)) + (dist_b / (cell_b["k"] * area))

        if resistance <= 1e-20:
            return

        conductance = 1.0 / resistance
        rows.extend([cell_a["id"], cell_b["id"], cell_a["id"], cell_b["id"]])
        cols.extend([cell_a["id"], cell_b["id"], cell_b["id"], cell_a["id"]])
        data.extend([-conductance, -conductance, conductance, conductance])

    def assemble_g_matrix(self) -> sp.csr_matrix:
        n_cells = len(self.cells)
        rows, cols, data = [], [], []

        print(
            f"[INFO] Building full 3D non-conformal G matrix ({n_cells} cells) via Sweep-and-Prune..."
        )
        tol = self.GEOMETRY_TOLERANCE

        # Sweep and Prune (SAP) 算法：基于 X 轴排序，完美解决任何多尺度下的接触面探测
        sorted_cells = sorted(self.cells, key=lambda c: c["box"][0])
        active_list = []

        for cell_a in sorted_cells:
            # 剔除 X 轴已经完全脱离接触范围的网格
            active_list = [
                c for c in active_list if c["box"][3] >= cell_a["box"][0] - tol
            ]

            for cell_b in active_list:
                # 检查 Y 轴重叠
                if (
                    max(cell_a["box"][1], cell_b["box"][1])
                    > min(cell_a["box"][4], cell_b["box"][4]) + tol
                ):
                    continue
                # 检查 Z 轴重叠
                if (
                    max(cell_a["box"][2], cell_b["box"][2])
                    > min(cell_a["box"][5], cell_b["box"][5]) + tol
                ):
                    continue

                # 若到达此步，说明两个网格在 3D 空间存在相互接触，精确计算接触面积
                # X 面接触
                if (
                    abs(cell_a["box"][3] - cell_b["box"][0]) < tol
                    or abs(cell_a["box"][0] - cell_b["box"][3]) < tol
                ):
                    area = _overlap_area(cell_a["box"], cell_b["box"], 0)
                    self._add_pairwise_conductance(
                        rows, cols, data, cell_a, cell_b, 0, area
                    )

                # Y 面接触
                if (
                    abs(cell_a["box"][4] - cell_b["box"][1]) < tol
                    or abs(cell_a["box"][1] - cell_b["box"][4]) < tol
                ):
                    area = _overlap_area(cell_a["box"], cell_b["box"], 1)
                    self._add_pairwise_conductance(
                        rows, cols, data, cell_a, cell_b, 1, area
                    )

                # Z 面接触
                if (
                    abs(cell_a["box"][5] - cell_b["box"][2]) < tol
                    or abs(cell_a["box"][2] - cell_b["box"][5]) < tol
                ):
                    area = _overlap_area(cell_a["box"], cell_b["box"], 2)
                    self._add_pairwise_conductance(
                        rows, cols, data, cell_a, cell_b, 2, area
                    )

            active_list.append(cell_a)

        return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    def _read_ptrace_steps(self) -> List[Dict[str, float]]:
        ptrace_path = os.path.join(
            self.base_dir, self.config.get("ptrace_file_path", "")
        )
        if not ptrace_path or not os.path.exists(ptrace_path):
            return []

        steps = []
        with open(ptrace_path, "r", encoding="utf-8") as handle:
            headers = handle.readline().split()
            for line in handle:
                if not line.strip():
                    continue
                steps.append(dict(zip(headers, [float(v) for v in line.split()])))
        return steps

    def _load_temperature_field_from_mesh(self, mesh_path: str) -> np.ndarray:
        mesh = meshio.read(mesh_path)
        if "Temperature_K" not in mesh.cell_data:
            raise KeyError(f"No Temperature_K cell data found in {mesh_path}")

        values = []
        for block, block_values in zip(mesh.cells, mesh.cell_data["Temperature_K"]):
            if block.type == "hexahedron":
                values.extend(np.asarray(block_values, dtype=float).tolist())
        return np.asarray(values, dtype=float)

    def _load_initial_temperature(self, n_cells: int) -> np.ndarray:
        init_file = str(self.config.get("init_temperature_file_path", "")).strip()
        if init_file:
            candidate_path = os.path.join(self.base_dir, init_file)
            if os.path.exists(candidate_path):
                raw_values = self._load_temperature_field_from_mesh(candidate_path)
                if raw_values.size == n_cells:
                    print(f"[INFO] Using initial temperature from {init_file}")
                    # 将加载的原始排列重新映射为当前排序好的 ID 顺序
                    mapped_init = np.zeros(n_cells, dtype=float)
                    for cell in self.cells:
                        mapped_init[cell["id"]] = raw_values[cell["original_id"]]
                    return mapped_init
                print(
                    "[WARN] Cell count mismatch; fallback to uniform init_temperature."
                )

        return np.full(n_cells, float(self.config.get("init_temperature", 318.15)))

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n_cells = len(self.cells)
        rhs = np.zeros(n_cells)
        rows, cols, data = [], [], []
        z_max = max(cell["box"][5] for cell in self.cells)

        for boundary in self.config.get("boundary_conditions", []):
            if boundary.get("type") != "convection":
                continue

            h_coeff = float(boundary["h"])
            t_inf = float(boundary["T_inf"])
            selection = set(boundary.get("selection", []))

            for cell in self.cells:
                if cell["tag"] not in selection or abs(cell["box"][5] - z_max) > 1e-6:
                    continue

                area = cell["dims"][0] * cell["dims"][1]
                half_thickness = 0.5 * cell["dims"][2]
                conductance = 1.0 / (
                    (half_thickness / (cell["k"] * area)) + (1.0 / (h_coeff * area))
                )

                rows.append(cell["id"])
                cols.append(cell["id"])
                data.append(-conductance)
                rhs[cell["id"]] += conductance * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells)), rhs

    def _build_power_vector(
        self, power_step: Dict[str, float], cached_units: List[dict] = None
    ) -> np.ndarray:
        rhs = np.zeros(len(self.cells))
        power_units = cached_units or self.config.get("power_units", [])

        if not cached_units:
            parsed_units = []
            for unit in power_units:
                vol = unit["dx"] * unit["dy"] * unit["dz"]
                if vol > 0:
                    parsed_units.append(
                        {
                            "name": unit["name"],
                            "box": (
                                unit["lx"],
                                unit["ly"],
                                unit["lz"],
                                unit["lx"] + unit["dx"],
                                unit["ly"] + unit["dy"],
                                unit["lz"] + unit["dz"],
                            ),
                            "vol": vol,
                        }
                    )
            power_units = parsed_units

        for cell in self.cells:
            for unit in power_units:
                val = power_step.get(unit["name"], 0.0)
                if val == 0.0:
                    continue

                intersection = _intersection_volume(cell["box"], unit["box"])
                if intersection > 1e-15:
                    rhs[cell["id"]] += (intersection / unit["vol"]) * val

        return rhs

    def solve(self) -> None:
        ptrace_steps = self._read_ptrace_steps()
        conductance_matrix = self.assemble_g_matrix()
        boundary_matrix, boundary_rhs = self._build_boundary_terms()

        g_total = conductance_matrix + boundary_matrix

        simulation_type = self.config.get("simulation_type", "steady")
        if simulation_type == "steady":
            print("[SIM] Solving steady state...")
            steady_step: Dict[str, float] = {}
            if ptrace_steps:
                for power_unit in self.config.get("power_units", []):
                    unit_name = power_unit["name"]
                    values = [step.get(unit_name, 0.0) for step in ptrace_steps]
                    steady_step[unit_name] = float(np.mean(values)) if values else 0.0

            rhs = boundary_rhs + self._build_power_vector(steady_step)
            temperatures = splinalg.spsolve(-g_total, rhs)

            print(
                f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
            )
            self.save(temperatures, "result.vtu")
            return

        print("[SIM] Solving transient...")
        dt = float(self.config.get("timestep", 0.1))
        total_time = float(self.config.get("time", 0.0))

        if not ptrace_steps:
            n_steps = max(1, int(math.ceil(total_time / dt)) if total_time > 0 else 1)
            ptrace_steps = [{} for _ in range(n_steps)]

        capacity_matrix = sp.diags([cell["cp"] * cell["vol"] for cell in self.cells])
        system_matrix = capacity_matrix / dt - g_total

        print("[INFO] Pre-factorizing system matrix for fast transient stepping...")
        system_matrix_csc = system_matrix.tocsc()
        solve_step = splinalg.factorized(system_matrix_csc)

        temperatures = self._load_initial_temperature(len(self.cells))

        parsed_power_units = []
        for unit in self.config.get("power_units", []):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol > 0:
                parsed_power_units.append(
                    {
                        "name": unit["name"],
                        "box": (
                            unit["lx"],
                            unit["ly"],
                            unit["lz"],
                            unit["lx"] + unit["dx"],
                            unit["ly"] + unit["dy"],
                            unit["lz"] + unit["dz"],
                        ),
                        "vol": vol,
                    }
                )

        for step_index, step_power in enumerate(ptrace_steps):
            rhs = (capacity_matrix / dt) @ temperatures
            rhs += boundary_rhs
            rhs += self._build_power_vector(step_power, cached_units=parsed_power_units)

            temperatures = solve_step(rhs)

            if step_index % 10 == 0 or step_index == len(ptrace_steps) - 1:
                print(
                    f"[STEP {step_index:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )

        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        # 将 Morton 排序计算的结果，恢复为 VTU 文件原有的网格顺序，以修复输出图像错乱
        mapped_values = np.zeros(len(self.cells), dtype=float)
        for cell in self.cells:
            mapped_values[cell["original_id"]] = temperatures[cell["id"]]

        offset = 0
        temperature_chunks = []

        for block in self.mesh.cells:
            count = len(block.data)
            if block.type == "hexahedron":
                chunk = mapped_values[offset : offset + count]
                offset += count
            else:
                chunk = np.full(count, np.nan, dtype=float)
            temperature_chunks.append(chunk)

        self.mesh.cell_sets = {}
        self.mesh.cell_data = {"Temperature_K": temperature_chunks}
        self.mesh.write(os.path.join(self.base_dir, output_name))
        print(f"[FILE] Results saved to {output_name}")
