import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from metahotspot.model25d import load_config, load_stackup


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
    name: str = ""
    layer_name: str = ""
    is_fluid: bool = False
    pressure: float = 0.0


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        # 单一真相入口：加载彻底清洗过的 config
        self.config = load_config(config_path)
        self.mesh_path = os.path.join(self.base_dir, self.config["mesh_file_path"])
        self.mesh = meshio.read(self.mesh_path)

        self.materials = self.config["materials"]
        self.stackup = load_stackup(self.config, self.base_dir)
        self.cells: List[Cell] = []

        # 因为全局配置已清洗，这里可以直接取值而不需要Fallback
        self.water_density = 1000.0
        self.water_visc = self.materials["water"]["dynamic_viscosity"]

        self._prepare_mesh()
        self._precompute_power_matrix()

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
        dims, vols = uppers - lowers, np.prod(uppers - lowers, axis=1)

        b_min, diff = np.min(lowers, axis=0), np.max(uppers, axis=0) - np.min(
            lowers, axis=0
        )
        diff = np.where(diff == 0, 1, diff)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)
        sorted_indices = np.argsort(morton_keys)

        self._map_materials_and_build_cells(
            sorted_indices, centers, lowers, uppers, dims, vols, physical_tags
        )
        self._extract_faces(hex_data, sorted_indices)

    def _map_materials_and_build_cells(
        self, sorted_indices, centers, lowers, uppers, dims, vols, tags
    ):
        tol = self.GEOMETRY_TOLERANCE
        z_cursor = 0.0
        layer_bounds = [
            (l, z_cursor, (z_cursor := z_cursor + l.thickness)) for l in self.stackup
        ]

        for new_id, orig_id in enumerate(sorted_indices):
            c_center = centers[orig_id]
            k, cp, name, layer_name, is_fluid = 1.0, 1.0e6, "", "", False

            for layer, z_min, z_max in layer_bounds:
                if z_min - tol <= c_center[2] <= z_max + tol:
                    layer_name = layer.name
                    def_mat = self.materials.get(
                        layer.default_material, self.materials["default_solid"]
                    )
                    k, cp = float(def_mat["k"]), float(def_mat["cp"])

                    for u in layer.units:
                        if (
                            u.lx - tol <= c_center[0] <= u.lx + u.dx + tol
                            and u.ly - tol <= c_center[1] <= u.ly + u.dy + tol
                        ):
                            name, is_fluid = u.name, u.is_fluid
                            if u.k is not None:
                                k, cp = u.k, u.cp
                            else:
                                mat = self.materials.get(u.material, def_mat)
                                k, cp = float(mat["k"]), float(mat["cp"])
                            break
                    break

            self.cells.append(
                Cell(
                    orig_id,
                    new_id,
                    c_center,
                    dims[orig_id],
                    np.array([*lowers[orig_id], *uppers[orig_id]]),
                    k,
                    cp,
                    int(tags[orig_id]),
                    float(vols[orig_id]),
                    name,
                    layer_name,
                    is_fluid,
                )
            )

    def _extract_faces(self, hex_data, sorted_indices):
        self.face_to_cells = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]
            faces = [
                tuple(sorted([nodes[0], nodes[3], nodes[2], nodes[1]])),
                tuple(sorted([nodes[4], nodes[5], nodes[6], nodes[7]])),
                tuple(sorted([nodes[0], nodes[1], nodes[5], nodes[4]])),
                tuple(sorted([nodes[3], nodes[7], nodes[6], nodes[2]])),
                tuple(sorted([nodes[0], nodes[4], nodes[7], nodes[3]])),
                tuple(sorted([nodes[1], nodes[2], nodes[6], nodes[5]])),
            ]
            for f in faces:
                self.face_to_cells.setdefault(f, []).append(new_id)

        self.internal_faces = {
            f: tuple(c) for f, c in self.face_to_cells.items() if len(c) == 2
        }
        boundary_faces_all = {
            f: tuple(c) for f, c in self.face_to_cells.items() if len(c) == 1
        }
        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}

        self.boundary_faces_by_direction = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }
        for f, (c_id,) in boundary_faces_all.items():
            pts = self.mesh.points[list(f)]
            cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area = np.linalg.norm(cross)
            if area < self.GEOMETRY_TOLERANCE:
                continue

            normal = cross / area
            if np.dot(np.mean(pts, axis=0) - self.cells[c_id].center, normal) < 0:
                normal = -normal

            abs_n = np.abs(normal)
            if abs_n[2] >= abs_n[0] and abs_n[2] >= abs_n[1]:
                dir_key = "+Z" if normal[2] > 0 else "-Z"
            elif abs_n[0] >= abs_n[1]:
                dir_key = "+X" if normal[0] > 0 else "-X"
            else:
                dir_key = "+Y" if normal[1] > 0 else "-Y"

            self.boundary_faces_by_direction[dir_key].append((c_id, normal, area))

    def _solve_pressure_field(self) -> None:
        fluid_cells = [c for c in self.cells if c.is_fluid]
        if not fluid_cells:
            return

        cell_to_idx = {c.id: i for i, c in enumerate(fluid_cells)}
        n_fluid = len(fluid_cells)
        bc_pressures, self.inlet_temps = {}, {}

        # 动态处理具有不同参数结构的BC
        for bc in self.config["boundary_conditions"]:
            if bc.get("type") == "pressure":
                for c_id, _, _ in self.boundary_faces_by_direction.get(
                    bc.get("face", ""), []
                ):
                    c = self.cells[c_id]
                    if c.is_fluid and c.layer_name == bc.get("target"):
                        bc_pressures[c_id] = float(bc["pressure"])
                        if "temperature" in bc:
                            self.inlet_temps[c_id] = float(bc["temperature"])

        avg_dims = np.mean([c.dims for c in fluid_cells], axis=0)
        h, w, L = avg_dims[2], avg_dims[0], avg_dims[1]

        if abs(h - w) < 1e-10:
            hydroC = (0.42229 * h**4) / (12 * self.water_visc * L)
        elif h > w:
            hydroC = ((1 - 0.63 * (w / h)) * w**3 * h) / (12 * self.water_visc * L)
        else:
            hydroC = ((1 - 0.63 * (h / w)) * h**3 * w) / (12 * self.water_visc * L)
        self.hydroC = hydroC

        rows, cols, data = [], [], []
        b_pressure = np.zeros(n_fluid)

        for c in fluid_cells:
            i = cell_to_idx[c.id]
            if c.id in bc_pressures:
                rows.append(i)
                cols.append(i)
                data.append(1.0)
                b_pressure[i] = bc_pressures[c.id]
                continue

            neighbors = [
                c1_id if c0_id == c.id else c0_id
                for f, (c0_id, c1_id) in self.internal_faces.items()
                if (c0_id == c.id or c1_id == c.id)
                and (c1_id in cell_to_idx and c0_id in cell_to_idx)
            ]
            rows.extend([i] * (len(neighbors) + 1))
            cols.append(i)
            data.append(-len(neighbors) * hydroC)
            cols.extend([cell_to_idx[n] for n in neighbors])
            data.extend([hydroC] * len(neighbors))

        try:
            pressure = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)),
                b_pressure,
            )
            for c in fluid_cells:
                c.pressure = pressure[cell_to_idx[c.id]]
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")

    def _assemble_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]
            for c_b in active_list:
                if (
                    max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol
                    or max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol
                ):
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

                    is_a_fluid, is_b_fluid = c_a.is_fluid, c_b.is_fluid

                    if is_a_fluid != is_b_fluid:
                        fluid_c = c_a if is_a_fluid else c_b
                        solid_c = c_b if is_a_fluid else c_a

                        f_dims = sorted(fluid_c.dims)
                        w, h = f_dims[0], f_dims[1]

                        # 水力直径
                        d_h = 2 * w * h / (w + h)

                        AR = min(w, h) / max(w, h)

                        # London and Shah Nu 经验公式
                        Nu = 8.235 * (
                            1
                            - 2.0421 * AR
                            + 3.0853 * AR**2
                            - 2.4765 * AR**3
                            + 1.0578 * AR**4
                            - 0.1861 * AR**5
                        )

                        # 对流换热系数
                        h_f = (Nu * fluid_c.k) / d_h

                        # R_total = R_solid + R_conv
                        R_solid = solid_c.dims[axis] / (2.0 * solid_c.k * area)
                        R_conv = 1.0 / (h_f * area)
                        res = R_solid + R_conv
                    else:
                        res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                            c_b.dims[axis] / (2.0 * c_b.k * area)
                        )

                    if res > tol:
                        g = 1.0 / res
                        rows.extend([c_a.id, c_b.id, c_a.id, c_b.id])
                        cols.extend([c_a.id, c_b.id, c_b.id, c_a.id])
                        data.extend([-g, -g, g, g])
            active_list.append(c_a)

        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(self.cells))
        )

    def _assemble_advection_matrix(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n = len(self.cells)
        rows, cols, data, rhs = [], [], [], np.zeros(n)
        fluid_cells = [c for c in self.cells if c.is_fluid]
        if not fluid_cells or not hasattr(self, "hydroC"):
            return sp.csr_matrix((n, n)), rhs

        net_internal_outflux = {c.id: 0.0 for c in fluid_cells}
        for f, (c0_id, c1_id) in self.internal_faces.items():
            c0, c1 = self.cells[c0_id], self.cells[c1_id]
            if not (c0.is_fluid and c1.is_fluid):
                continue
            mass_flux = (c0.pressure - c1.pressure) * self.hydroC * self.water_density
            net_internal_outflux[c0_id] += mass_flux
            net_internal_outflux[c1_id] -= mass_flux

            if abs(mass_flux) > self.GEOMETRY_TOLERANCE:
                up_id, dn_id = (c0_id, c1_id) if mass_flux > 0 else (c1_id, c0_id)
                adv_term = abs(mass_flux) * self.cells[up_id].cp
                rows.extend([up_id, dn_id])
                cols.extend([up_id, up_id])
                data.extend([-adv_term, adv_term])

        for c in fluid_cells:
            influx = net_internal_outflux[c.id]
            if influx > self.GEOMETRY_TOLERANCE and c.id in getattr(
                self, "inlet_temps", {}
            ):
                rhs[c.id] += influx * c.cp * self.inlet_temps[c.id]
            elif influx < -self.GEOMETRY_TOLERANCE:
                rows.append(c.id)
                cols.append(c.id)
                data.append(influx * c.cp)

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _precompute_power_matrix(self) -> None:
        active_units = [
            {
                "name": u.name,
                "lx": u.lx,
                "ly": u.ly,
                "lz": z_cursor,
                "dx": u.dx,
                "dy": u.dy,
                "dz": l.thickness,
            }
            for z_cursor, l in zip(
                np.cumsum([0] + [l.thickness for l in self.stackup[:-1]]), self.stackup
            )
            if l.active
            for u in l.units
        ]
        self.unit_names = [u["name"] for u in active_units]
        if not active_units or not self.cells:
            self.power_matrix = sp.csr_matrix((len(self.cells), 0))
            return

        c_boxes = np.array([c.box for c in self.cells])
        rows, cols, data = [], [], []
        for j, u in enumerate(active_units):
            vol = u["dx"] * u["dy"] * u["dz"]
            if vol <= 0:
                continue
            u_min, u_max = np.array([u["lx"], u["ly"], u["lz"]]), np.array(
                [u["lx"] + u["dx"], u["ly"] + u["dy"], u["lz"] + u["dz"]]
            )
            intersect = np.prod(
                np.maximum(
                    0,
                    np.minimum(c_boxes[:, 3:], u_max)
                    - np.maximum(c_boxes[:, :3], u_min),
                ),
                axis=1,
            )
            valid = np.where(intersect > self.GEOMETRY_TOLERANCE)[0]
            rows.extend(valid)
            cols.extend([j] * len(valid))
            data.extend(intersect[valid] / vol)
        self.power_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(active_units))
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        n, rhs, rows, cols, data = (
            len(self.cells),
            np.zeros(len(self.cells)),
            [],
            [],
            [],
        )
        # JSON列表可以直接遍历
        for bc in [
            b
            for b in self.config["boundary_conditions"]
            if b.get("type") == "convection"
        ]:
            for c_id, _, area in self.boundary_faces_by_direction.get(
                bc.get("face", ""), []
            ):
                c = self.cells[c_id]
                if bc.get("target") and bc.get("target") != c.layer_name:
                    continue
                h, t_inf = float(bc["h"]), float(bc["T_inf"])
                g = area / ((0.5 * (c.vol / area) / c.k) + (1.0 / h))
                rows.append(c.id)
                cols.append(c.id)
                data.append(-g)
                rhs[c.id] += g * t_inf
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def solve(self) -> None:
        self.g_total = self._assemble_conduction_matrix()
        bc_mat, self.boundary_rhs = self._build_boundary_terms()
        self.g_total += bc_mat

        if any(c.is_fluid for c in self.cells):
            self._solve_pressure_field()
            adv_mat, adv_rhs = self._assemble_advection_matrix()
            self.g_total += adv_mat
            self.boundary_rhs += adv_rhs

        ptrace_path = os.path.join(self.base_dir, self.config["ptrace_file_path"])
        self.ptrace_steps = []
        if os.path.exists(ptrace_path):
            with open(ptrace_path, "r") as f:
                headers = f.readline().split()
                self.ptrace_steps = [
                    dict(zip(headers, map(float, l.split()))) for l in f if l.strip()
                ]

        if self.config["simulation_type"] == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
        mean_powers = np.array(
            [
                (
                    np.mean([s.get(n, 0.0) for s in self.ptrace_steps])
                    if self.ptrace_steps
                    else 0.0
                )
                for n in self.unit_names
            ]
        )
        temperatures = splinalg.spsolve(
            -self.g_total, self.boundary_rhs + (self.power_matrix @ mean_powers)
        )
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

        temperatures = np.full(len(self.cells), self.config["init_temperature"])
        init_file = self.config["init_temperature_file_path"]
        if init_file and os.path.exists(os.path.join(self.base_dir, init_file)):
            init_mesh = meshio.read(os.path.join(self.base_dir, init_file))
            hex_data = init_mesh.cell_data.get("Temperature_K", [])
            offset = 0
            for block, block_temps in zip(init_mesh.cells, hex_data):
                if block.type == "hexahedron":
                    for i, t in enumerate(block_temps):
                        if (nid := self.orig_to_new_id.get(offset + i)) is not None:
                            temperatures[nid] = t
                    offset += len(block_temps)

        for i, step_power in enumerate(ptrace):
            power_vec = np.array([step_power.get(n, 0.0) for n in self.unit_names])
            temperatures = solve_step(
                (c_mat @ temperatures)
                + self.boundary_rhs
                + (self.power_matrix @ power_vec)
            )
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )
        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
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

        meshio.Mesh(
            points=self.mesh.points,
            cells=hex_blocks,
            cell_data={"Temperature_K": temp_chunks},
        ).write(os.path.join(self.base_dir, output_name))
