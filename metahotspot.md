# Project Source Code: metahotspot

## Directory Structure
```text
.
├── __init__.py
├── converter.py
├── fvm_solver.py
├── gmsh_mesher.py
├── hotspot_parser.py
└── model25d.py
```

## File Contents

### File: converter.py
```py
import os
import json
import shutil
import csv
from typing import Dict, List, Tuple

from metahotspot.hotspot_parser import HotSpotParser
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

        # JSON 格式中边界条件本身就是灵活字典
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
        if mat_key not in self.materials:
            self.materials[mat_key] = self.materials["copper"]

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
                k = self.materials[mat]["k"]
                cp = self.materials[mat]["cp"]

                units.append(
                    {
                        "name": f"mc_{'fluid' if is_fluid else 'solid'}_{len(units)}",
                        "lx": c * dx,
                        "ly": (rows - r - h) * dy,
                        "dx": w * dx,
                        "dy": h * dy,
                        "is_fluid": is_fluid,
                        "material": mat,
                        "k": k,
                        "cp": cp,
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

### File: fvm_solver.py
```py
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

```

### File: gmsh_mesher.py
```py
import math
from collections import deque
from pathlib import Path
from typing import List

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

### File: hotspot_parser.py
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

### File: model25d.py
```py
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    "water": {"k": 0.6069, "cp": 4.17e6, "fluid": True, "dynamic_viscosity": 8.89e-4},
    "default_solid": {"k": 1.0, "cp": 1.0e6, "fluid": False},
}


@dataclass
class Unit2D:
    """2D layout unit for FVM mesh generation."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: Optional[str] = None
    k: Optional[float] = None
    cp: Optional[float] = None
    is_fluid: bool = False


@dataclass
class Layer25D:
    name: str
    tag: int
    thickness: float
    default_material: str
    active: bool
    units: List[Unit2D] = field(default_factory=list)
    lx: float = 0.0
    ly: float = 0.0
    dx: float = 0.01
    dy: float = 0.01


def load_config(config_path: str) -> Dict[str, Any]:
    """统一配置加载入口：确保下游直接读取到清洗完毕、具有绝对信任度的配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return merge_with_defaults(raw_config)


def merge_with_defaults(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """配置关口：将原始配置与默认值合并。一处更改，处处有效。"""
    config = dict(DEFAULT_CONFIG)

    # 注入用户配置并处理隐式类型转换
    for k, v in raw_config.items():
        if k in config and type(config[k]) is not type(v):
            try:
                # 忽略 null 占位符
                if v not in {"(null)", "null", "None", ""}:
                    config[k] = type(config[k])(v)
            except ValueError:
                config[k] = v
        else:
            config[k] = v

    # 消除零散的Fallback逻辑：处理强依赖关系
    config["t_interface"] = raw_config.get("t_interface", config["t_tim"])
    config["time"] = raw_config.get("time", max(config["sampling_intvl"], 0.01))
    config["timestep"] = raw_config.get("timestep", config["sampling_intvl"])

    if "init_temp" in raw_config:
        config["init_temperature"] = float(raw_config["init_temp"])

    # 确保基础材料始终存在，防止下游 key error
    for mat_name, mat_props in STANDARD_MATERIALS.items():
        if mat_name not in config["materials"]:
            config["materials"][mat_name] = mat_props

    return config


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    """Load 2.5D stackup model from strict config."""
    layers = []
    stackup_data = config.get("stackup", [])

    for i, layer_cfg in enumerate(stackup_data):
        tag = int(layer_cfg.get("tag", i + 100))
        name = str(layer_cfg.get("name", f"layer_{tag}"))
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))
        material = layer_cfg.get("material", "silicon")

        units = []
        layout_file = layer_cfg.get("layout_file", "")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
                        units.append(
                            Unit2D(
                                name=u["name"],
                                lx=float(u["lx"]),
                                ly=float(u["ly"]),
                                dx=float(u["dx"]),
                                dy=float(u["dy"]),
                                material=u.get("material"),
                                k=u.get("k"),
                                cp=u.get("cp"),
                                is_fluid=bool(u.get("is_fluid", False)),
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
                    material=material,
                    is_fluid=False,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                default_material=material,
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

### File: __init__.py
```py
"""MetaHotspot Python package."""

```

