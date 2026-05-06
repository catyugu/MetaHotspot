import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
    tag: int
    vol: float
    name: str = ""
    layer_name: str = ""

    # Material / Physical Properties
    k: float = 1.0
    cp: float = 1.0e6
    density: float = 1000.0
    dynamic_viscosity: float = 0.0
    is_fluid: bool = False

    # Computed / Transient State Properties
    hydroC: float = 0.0
    pressure: float = 0.0
    temperature: float = 0.0
    inlet_temperature: Optional[float] = None
    is_pressure_boundary: bool = False


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    GEOMETRY_TOLERANCE = 1e-15

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = load_config(config_path)
        self.mesh_path = os.path.join(self.base_dir, self.config["mesh_file_path"])
        self.mesh = meshio.read(self.mesh_path)

        self.stackup = load_stackup(self.config, self.base_dir)
        self.cells: List[Cell] = []

        self._prepare_mesh()
        self._precompute_power_matrix()

    def _get_cell_properties(self, center: np.ndarray) -> tuple:
        tol = self.GEOMETRY_TOLERANCE
        for layer, z_min, z_max in self.layer_bounds:
            if not (z_min - tol <= center[2] <= z_max + tol):
                continue
            for u in layer.units:
                if (
                    u.lx - tol <= center[0] <= u.lx + u.dx + tol
                    and u.ly - tol <= center[1] <= u.ly + u.dy + tol
                ):
                    return (
                        u.name,
                        layer.name,
                        u.k,
                        u.cp,
                        u.density,
                        u.dynamic_viscosity,
                        u.is_fluid,
                    )
            return (
                "",
                layer.name,
                layer.k,
                layer.cp,
                layer.density,
                layer.dynamic_viscosity,
                layer.is_fluid,
            )
        return "", "", 1.0, 1e6, 1000.0, 0.0, False

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

        b_min = np.min(lowers, axis=0)
        diff = np.max(uppers, axis=0) - b_min
        diff = np.where(diff == 0, 1, diff)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)
        sorted_indices = np.argsort(morton_keys)

        z_cursor = 0.0
        self.layer_bounds = []
        for l in self.stackup:
            self.layer_bounds.append((l, z_cursor, z_cursor + l.thickness))
            z_cursor += l.thickness

        for new_id, orig_id in enumerate(sorted_indices):
            name, layer_name, k, cp, den, visc, is_fluid = self._get_cell_properties(
                centers[orig_id]
            )
            self.cells.append(
                Cell(
                    original_id=orig_id,
                    id=new_id,
                    center=centers[orig_id],
                    dims=dims[orig_id],
                    box=np.array([*lowers[orig_id], *uppers[orig_id]]),
                    tag=int(physical_tags[orig_id]),
                    vol=float(vols[orig_id]),
                    name=name,
                    layer_name=layer_name,
                    k=k,
                    cp=cp,
                    density=den,
                    dynamic_viscosity=visc,
                    is_fluid=is_fluid,
                )
            )

        self._extract_faces(hex_data, sorted_indices)

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

    def _init_cell_hydro_properties(self):
        for c in self.cells:
            if not c.is_fluid or c.dynamic_viscosity <= 0:
                continue
            w, L, h = c.dims[0], c.dims[1], c.dims[2]
            v = c.dynamic_viscosity
            if abs(h - w) < 1e-10:
                c.hydroC = (0.42229 * h**4) / (12 * v * L)
            elif h > w:
                c.hydroC = ((1 - 0.63 * (w / h)) * w**3 * h) / (12 * v * L)
            else:
                c.hydroC = ((1 - 0.63 * (h / w)) * h**3 * w) / (12 * v * L)

    def _apply_boundary_conditions(self):
        for bc in self.config["boundary_conditions"]:
            if bc.get("type") == "pressure":
                for c_id, _, _ in self.boundary_faces_by_direction.get(
                    bc.get("face", ""), []
                ):
                    c = self.cells[c_id]
                    if c.is_fluid and c.layer_name == bc.get("target"):
                        c.is_pressure_boundary = True
                        c.pressure = float(bc["pressure"])
                        if "temperature" in bc:
                            c.inlet_temperature = float(bc["temperature"])

    def _solve_pressure_field(self) -> None:
        fluid_cells = [c for c in self.cells if c.is_fluid]
        if not fluid_cells:
            return

        n_fluid = len(fluid_cells)
        cell_to_idx = {c.id: i for i, c in enumerate(fluid_cells)}
        rows, cols, data = [], [], []
        b_pressure = np.zeros(n_fluid)
        diag_C = np.zeros(n_fluid)

        for i, c in enumerate(fluid_cells):
            if c.is_pressure_boundary:
                rows.append(i), cols.append(i), data.append(1.0)
                b_pressure[i] = c.pressure

        for f, (c0_id, c1_id) in self.internal_faces.items():
            if c0_id not in cell_to_idx or c1_id not in cell_to_idx:
                continue
            c0, c1 = self.cells[c0_id], self.cells[c1_id]
            i0, i1 = cell_to_idx[c0_id], cell_to_idx[c1_id]
            C_eff = (
                2.0 * c0.hydroC * c1.hydroC / (c0.hydroC + c1.hydroC)
                if (c0.hydroC + c1.hydroC) > 0
                else 0.0
            )

            if not c0.is_pressure_boundary:
                rows.extend([i0]), cols.extend([i1]), data.extend([C_eff])
                diag_C[i0] += C_eff
            if not c1.is_pressure_boundary:
                rows.extend([i1]), cols.extend([i0]), data.extend([C_eff])
                diag_C[i1] += C_eff

        for i, c in enumerate(fluid_cells):
            if not c.is_pressure_boundary:
                rows.append(i), cols.append(i), data.append(-diag_C[i])

        try:
            solved_p = splinalg.spsolve(
                sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid)),
                b_pressure,
            )
            for i, c in enumerate(fluid_cells):
                c.pressure = solved_p[i]
        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")

    def _compute_nusselt(self, fluid_c: Cell) -> float:
        w, h = sorted([fluid_c.dims[0], fluid_c.dims[1]])
        AR = w / h if h > 0 else 1.0
        return 8.235 * (
            1
            - 2.0421 * AR
            + 3.0853 * AR**2
            - 2.4765 * AR**3
            + 1.0578 * AR**4
            - 0.1861 * AR**5
        )

    def _get_face_conduction(self, c_a: Cell, c_b: Cell, tol: float) -> list:
        if (
            max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol
            or max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol
        ):
            return []

        res_list = []
        for axis in range(3):
            if not (
                abs(c_a.box[axis + 3] - c_b.box[axis]) < tol
                or abs(c_a.box[axis] - c_b.box[axis + 3]) < tol
            ):
                continue
            area = _overlap_area(c_a.box, c_b.box, axis)
            if area <= tol:
                continue

            if c_a.is_fluid != c_b.is_fluid:
                fluid_c, solid_c = (c_a, c_b) if c_a.is_fluid else (c_b, c_a)
                Nu = self._compute_nusselt(fluid_c)
                d_h = (
                    2
                    * fluid_c.dims[0]
                    * fluid_c.dims[1]
                    / (fluid_c.dims[0] + fluid_c.dims[1])
                )
                h_f = (Nu * fluid_c.k) / d_h if d_h > 0 else 1e-6
                res = solid_c.dims[axis] / (2.0 * solid_c.k * area) + 1.0 / (h_f * area)
            else:
                res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                    c_b.dims[axis] / (2.0 * c_b.k * area)
                )

            if res > tol:
                res_list.append(1.0 / res)
        return res_list

    def _assemble_conduction_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]
            for c_b in active_list:
                for g in self._get_face_conduction(c_a, c_b, tol):
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

        if not fluid_cells:
            return sp.csr_matrix((n, n)), rhs

        net_internal_outflux = {c.id: 0.0 for c in fluid_cells}

        for f, (c0_id, c1_id) in self.internal_faces.items():
            c0, c1 = self.cells[c0_id], self.cells[c1_id]
            if not (c0.is_fluid and c1.is_fluid):
                continue

            C_eff = (
                2.0 * c0.hydroC * c1.hydroC / (c0.hydroC + c1.hydroC)
                if (c0.hydroC + c1.hydroC) > 0
                else 0.0
            )
            density_eff = (c0.density + c1.density) / 2.0
            mass_flux = (c0.pressure - c1.pressure) * C_eff * density_eff

            net_internal_outflux[c0_id] += mass_flux
            net_internal_outflux[c1_id] -= mass_flux

            if abs(mass_flux) > self.GEOMETRY_TOLERANCE:
                up_id, dn_id = (c0_id, c1_id) if mass_flux > 0 else (c1_id, c0_id)
                adv_term = abs(mass_flux) * self.cells[up_id].cp
                rows.extend([up_id, dn_id]), cols.extend([up_id, up_id]), data.extend(
                    [-adv_term, adv_term]
                )

        for c in fluid_cells:
            influx = net_internal_outflux[c.id]
            if influx > self.GEOMETRY_TOLERANCE and c.inlet_temperature is not None:
                rhs[c.id] += influx * c.cp * c.inlet_temperature
            elif influx < -self.GEOMETRY_TOLERANCE:
                rows.append(c.id), cols.append(c.id), data.append(influx * c.cp)

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _precompute_power_matrix(self) -> None:
        z_cursor = 0.0
        active_units = []
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
                rows.extend([c.id]), cols.extend([c.id]), data.append(-g)
                rhs[c.id] += g * t_inf
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def solve(self) -> None:
        self._init_cell_hydro_properties()
        self._apply_boundary_conditions()

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

        for c in self.cells:
            c.temperature = temperatures[c.id]

        print(
            f"[RESULT] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
        )
        self.save("result.vtu")

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

        for c in self.cells:
            c.temperature = temperatures[c.id]

        self.save("transient_result.vtu")

    def save(self, output_name: str) -> None:
        mapped = np.zeros(len(self.cells))
        for c in self.cells:
            mapped[c.original_id] = c.temperature

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
