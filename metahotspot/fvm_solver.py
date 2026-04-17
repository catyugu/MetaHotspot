import math
import os
from typing import Dict, List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    other_axes = [idx for idx in range(3) if idx != axis]
    lower = np.maximum(box_a[other_axes], box_b[other_axes])
    upper = np.minimum(
        box_a[[idx + 3 for idx in other_axes]],
        box_b[[idx + 3 for idx in other_axes]],
    )
    dims = upper - lower
    return float(np.prod(dims)) if np.all(dims > 0.0) else 0.0


def _intersection_volume(box_a: np.ndarray, box_b: np.ndarray) -> float:
    lower = np.maximum(box_a[:3], box_b[:3])
    upper = np.minimum(box_a[3:], box_b[3:])
    dims = upper - lower
    return float(np.prod(dims)) if np.all(dims > 0.0) else 0.0


class FVMSolver:
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
        for cell_id, nodes in enumerate(hex_data):
            coords = points[nodes]
            lower = np.min(coords, axis=0)
            upper = np.max(coords, axis=0)

            tag = int(physical_tags[cell_id])
            material_name = self.tag_to_material.get(tag, "silicon")
            material = self.materials[material_name]

            self.cells.append(
                {
                    "id": cell_id,
                    "center": (lower + upper) / 2.0,
                    "dims": upper - lower,
                    "box": np.concatenate([lower, upper]),
                    "k": float(material["k"]),
                    "cp": float(material["cp"]),
                    "tag": tag,
                    "vol": float(np.prod(upper - lower)),
                }
            )

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
        if area <= 0.0:
            return

        dim_a = cell_a["dims"][axis] / 2.0
        dim_b = cell_b["dims"][axis] / 2.0
        resistance = (dim_a / (cell_a["k"] * area)) + (dim_b / (cell_b["k"] * area))
        if resistance <= 0.0:
            return

        conductance = 1.0 / resistance

        rows.extend([cell_a["id"], cell_b["id"], cell_a["id"], cell_b["id"]])
        cols.extend([cell_a["id"], cell_b["id"], cell_b["id"], cell_a["id"]])
        data.extend([-conductance, -conductance, conductance, conductance])

    def assemble_g_matrix(self) -> sp.csr_matrix:
        n_cells = len(self.cells)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        tolerance = 1e-10

        print(f"[INFO] Building non-conformal G matrix ({n_cells} cells)...")

        z_levels = sorted({round(cell["center"][2], 10) for cell in self.cells})
        layer_cells: Dict[float, List[dict]] = {
            z_level: [
                cell for cell in self.cells if round(cell["center"][2], 10) == z_level
            ]
            for z_level in z_levels
        }

        for level_index, z_level in enumerate(z_levels):
            current_layer = layer_cells[z_level]

            # Intra-layer conduction (x and y directions).
            for i, cell_a in enumerate(current_layer):
                for j in range(i + 1, len(current_layer)):
                    cell_b = current_layer[j]

                    x_touch = (
                        abs(cell_a["box"][3] - cell_b["box"][0]) < tolerance
                        or abs(cell_b["box"][3] - cell_a["box"][0]) < tolerance
                    )
                    if x_touch:
                        area = _overlap_area(cell_a["box"], cell_b["box"], axis=0)
                        self._add_pairwise_conductance(
                            rows, cols, data, cell_a, cell_b, 0, area
                        )

                    y_touch = (
                        abs(cell_a["box"][4] - cell_b["box"][1]) < tolerance
                        or abs(cell_b["box"][4] - cell_a["box"][1]) < tolerance
                    )
                    if y_touch:
                        area = _overlap_area(cell_a["box"], cell_b["box"], axis=1)
                        self._add_pairwise_conductance(
                            rows, cols, data, cell_a, cell_b, 1, area
                        )

            # Inter-layer conduction (z direction).
            if level_index >= len(z_levels) - 1:
                continue

            upper_layer = layer_cells[z_levels[level_index + 1]]
            for cell_a in current_layer:
                for cell_b in upper_layer:
                    z_touch = (
                        abs(cell_a["box"][5] - cell_b["box"][2]) < tolerance
                        or abs(cell_b["box"][5] - cell_a["box"][2]) < tolerance
                    )
                    if not z_touch:
                        continue

                    area = _overlap_area(cell_a["box"], cell_b["box"], axis=2)
                    self._add_pairwise_conductance(
                        rows, cols, data, cell_a, cell_b, 2, area
                    )

        return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    def _read_ptrace_steps(self) -> List[Dict[str, float]]:
        ptrace_path = os.path.join(
            self.base_dir, self.config.get("ptrace_file_path", "")
        )
        if not ptrace_path or not os.path.exists(ptrace_path):
            return []

        steps: List[Dict[str, float]] = []
        with open(ptrace_path, "r", encoding="utf-8") as handle:
            headers = handle.readline().split()
            for line in handle:
                if not line.strip():
                    continue
                values = [float(value) for value in line.split()]
                steps.append(dict(zip(headers, values)))

        return steps

    def _load_temperature_field_from_mesh(self, mesh_path: str) -> np.ndarray:
        mesh = meshio.read(mesh_path)

        field_name = None
        for candidate in ("Temperature_K", "Temperature"):
            if candidate in mesh.cell_data:
                field_name = candidate
                break

        if field_name is None:
            raise KeyError(
                f"No Temperature_K or Temperature cell data found in {mesh_path}"
            )

        values: List[float] = []
        for block, block_values in zip(mesh.cells, mesh.cell_data[field_name]):
            if block.type != "hexahedron":
                continue
            values.extend(np.asarray(block_values, dtype=float).tolist())

        return np.asarray(values, dtype=float)

    def _load_initial_temperature(self, n_cells: int) -> np.ndarray:
        init_file = str(self.config.get("init_temperature_file_path", "")).strip()
        if init_file:
            candidate_path = os.path.join(self.base_dir, init_file)
            if os.path.exists(candidate_path):
                loaded = self._load_temperature_field_from_mesh(candidate_path)
                if loaded.size == n_cells:
                    print(f"[INFO] Using initial temperature from {init_file}")
                    return loaded
                print(
                    "[WARN] init_temperature_file_path cell count mismatch "
                    f"({loaded.size} != {n_cells}); fallback to uniform init_temperature."
                )

        return np.full(n_cells, float(self.config.get("init_temperature", 318.15)))

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n_cells = len(self.cells)
        rhs = np.zeros(n_cells)
        row_indices: List[int] = []
        col_indices: List[int] = []
        diagonal_values: List[float] = []

        z_max = max(cell["box"][5] for cell in self.cells)
        for boundary in self.config.get("boundary_conditions", []):
            if boundary.get("type") != "convection":
                continue

            h_coeff = float(boundary["h"])
            t_inf = float(boundary["T_inf"])
            selection = set(boundary.get("selection", []))

            for cell in self.cells:
                if cell["tag"] not in selection:
                    continue
                if abs(cell["box"][5] - z_max) > 1e-8:
                    continue

                area = cell["dims"][0] * cell["dims"][1]
                # FVM ghost-node treatment for Robin BC:
                # R_eq = (dz/2)/(k*A) + 1/(h*A), G_eq = 1 / R_eq
                # This avoids over-cooling by accounting for half-cell conduction.
                half_thickness = 0.5 * cell["dims"][2]
                conduction_resistance = half_thickness / (cell["k"] * area)
                convection_resistance = 1.0 / (h_coeff * area)
                conductance = 1.0 / (conduction_resistance + convection_resistance)

                row_indices.append(cell["id"])
                col_indices.append(cell["id"])
                diagonal_values.append(-conductance)
                rhs[cell["id"]] += conductance * t_inf

        matrix = sp.csr_matrix(
            (diagonal_values, (row_indices, col_indices)), shape=(n_cells, n_cells)
        )
        return matrix, rhs

    def _build_power_vector(self, power_step: Dict[str, float]) -> np.ndarray:
        rhs = np.zeros(len(self.cells))
        power_units = self.config.get("power_units", [])

        for cell in self.cells:
            for power_unit in power_units:
                box = np.array(
                    [
                        power_unit["lx"],
                        power_unit["ly"],
                        power_unit["lz"],
                        power_unit["lx"] + power_unit["dx"],
                        power_unit["ly"] + power_unit["dy"],
                        power_unit["lz"] + power_unit["dz"],
                    ],
                    dtype=float,
                )

                intersection = _intersection_volume(cell["box"], box)
                if intersection <= 1e-15:
                    continue

                unit_volume = power_unit["dx"] * power_unit["dy"] * power_unit["dz"]
                if unit_volume <= 0.0:
                    continue

                rhs[cell["id"]] += (intersection / unit_volume) * power_step.get(
                    power_unit["name"], 0.0
                )

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
                f"[RESULT] T_min={np.min(temperatures):.2f} K, "
                f"T_max={np.max(temperatures):.2f} K"
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

        temperatures = self._load_initial_temperature(len(self.cells))

        for step_index, step_power in enumerate(ptrace_steps):
            rhs = (capacity_matrix / dt) @ temperatures
            rhs += boundary_rhs
            rhs += self._build_power_vector(step_power)

            temperatures = splinalg.spsolve(system_matrix, rhs)

            if step_index % 10 == 0 or step_index == len(ptrace_steps) - 1:
                print(
                    f"[STEP {step_index:4d}] "
                    f"T_min={np.min(temperatures):.2f} K, "
                    f"T_max={np.max(temperatures):.2f} K"
                )

        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        values = np.asarray(temperatures, dtype=float)
        offset = 0
        temperature_chunks: List[np.ndarray] = []
        legacy_chunks: List[np.ndarray] = []

        for block in self.mesh.cells:
            count = len(block.data)
            if block.type == "hexahedron":
                chunk = values[offset : offset + count]
                offset += count
            else:
                chunk = np.full(count, np.nan, dtype=float)

            temperature_chunks.append(chunk)
            legacy_chunks.append(chunk.copy())

        self.mesh.cell_sets = {}
        self.mesh.cell_data = {
            "Temperature_K": temperature_chunks,
            "Temperature": legacy_chunks,
        }
        self.mesh.write(os.path.join(self.base_dir, output_name))
        print(f"[FILE] Results saved to {output_name}")
