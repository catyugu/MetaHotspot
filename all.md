# Project Source Code: metahotspot

## Directory Structure
```text
.
├── __init__.py
├── converter.py
├── fvm_solver.py
├── gmsh_mesher.py
└── hotspot_parser.py
```

## File Contents

### File: converter.py
```py
import os
import shutil
from typing import Dict, List, Tuple

import toml

from metahotspot.gmsh_mesher import GmshMesher
from metahotspot.hotspot_parser import HotSpotParser


def _find_first_by_suffix(directory: str, suffix: str) -> str:
    for entry in os.listdir(directory):
        if entry.endswith(suffix):
            return os.path.join(directory, entry)
    return ""


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _layout_bbox_from_flp_units(units: List[dict]) -> Tuple[float, float, float, float]:
    min_x = min(unit["left_x"] for unit in units)
    min_y = min(unit["bottom_y"] for unit in units)
    max_x = max(unit["left_x"] + unit["width"] for unit in units)
    max_y = max(unit["bottom_y"] + unit["height"] for unit in units)
    return min_x, min_y, max_x - min_x, max_y - min_y


def _collect_global_xy_size(
    parser: HotSpotParser, example_dir: str, lcf_layers: List[dict]
) -> Tuple[float, float]:
    widths: List[float] = []
    heights: List[float] = []

    for layer in lcf_layers:
        geometry_file = os.path.join(example_dir, layer["flp_file"])

        units = parser.parse_flp(geometry_file)
        if not units:
            continue

        _, _, width, height = _layout_bbox_from_flp_units(units)
        widths.append(width)
        heights.append(height)

    if widths and heights:
        return max(widths), max(heights)

    for _, _, files in os.walk(example_dir):
        for file_name in files:
            if not file_name.endswith(".flp"):
                continue
            units = parser.parse_flp(os.path.join(example_dir, file_name))
            if not units:
                continue
            _, _, width, height = _layout_bbox_from_flp_units(units)
            widths.append(width)
            heights.append(height)

    if widths and heights:
        return max(widths), max(heights)

    return 0.01, 0.01


def _estimate_total_time(ptrace_path: str, sampling_interval: float) -> float:
    if not ptrace_path or not os.path.exists(ptrace_path):
        return max(sampling_interval, 0.01)

    count = 0
    with open(ptrace_path, "r", encoding="utf-8") as handle:
        _ = handle.readline()
        for line in handle:
            if line.strip():
                count += 1

    if count <= 0:
        return max(sampling_interval, 0.01)
    return count * sampling_interval


def _build_base_model_data(
    parser: HotSpotParser, example_dir: str
) -> Tuple[dict, Dict[int, dict], str]:
    config = parser.parse_config(os.path.join(example_dir, "example.config"))
    materials = parser.parse_materials(os.path.join(example_dir, "example.materials"))

    standard_materials = {
        "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
        "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
        "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
        "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
        "water": {
            "k": 0.6,
            "cp": 4.2e6,
            "fluid": True,
            "dynamic_viscosity": float(config.get("coolant_visc", 8.89e-4)),
        },
    }
    for name, props in standard_materials.items():
        if name not in materials:
            materials[name] = props

    ptrace_source = _find_first_by_suffix(example_dir, ".ptrace")
    ptrace_name = os.path.basename(ptrace_source) if ptrace_source else ""

    lcf_path = _find_first_by_suffix(example_dir, ".lcf")
    lcf_layers = parser.parse_lcf(lcf_path) if lcf_path else []

    global_width, global_height = _collect_global_xy_size(
        parser, example_dir, lcf_layers
    )

    layers_entities: Dict[int, dict] = {}
    power_units: List[dict] = []
    boundary_conditions: List[dict] = []
    domain_assignment: Dict[str, List[int]] = {}

    z_cursor = 0.0

    def ensure_material(
        material_name: str,
        fallback_name: str,
        k_key: str,
        cp_key: str,
    ) -> str:
        chosen = str(material_name or "").strip().lower()
        if not chosen:
            chosen = fallback_name

        if chosen not in materials:
            fallback = materials.get(fallback_name, {"k": 1.0, "cp": 1.0e6})
            materials[chosen] = {
                "k": float(config.get(k_key, fallback["k"])),
                "cp": float(config.get(cp_key, fallback["cp"])),
                "fluid": False,
            }

        return chosen

    if lcf_layers:
        for layer in lcf_layers:
            layer_tag = int(layer["id"]) + 1
            geometry_file = os.path.join(example_dir, layer["flp_file"])
            thickness = float(layer["thickness"])
            mesh_size = 0.0005

            if layer["type"] == "numeric":
                material_name = f"layer_{layer['id']}_mat"
                materials[material_name] = {
                    "k": float(layer["k"]),
                    "cp": float(layer["cp"]),
                    "fluid": False,
                }
            else:
                material_name = str(layer["material"])

            domain_assignment.setdefault(material_name, []).append(layer_tag)

            flp_units = parser.parse_flp(geometry_file)
            if not flp_units:
                layers_entities[layer_tag] = {
                    "units": [
                        {
                            "name": f"layer_{layer['id']}_extent",
                            "lx": 0.0,
                            "ly": 0.0,
                            "lz": z_cursor,
                            "dx": global_width,
                            "dy": global_height,
                            "dz": thickness,
                        }
                    ],
                }
                z_cursor += thickness
                continue

            min_x, min_y, layer_width, layer_height = _layout_bbox_from_flp_units(
                flp_units
            )
            offset_x = (global_width - layer_width) / 2.0 - min_x
            offset_y = (global_height - layer_height) / 2.0 - min_y

            layers_entities[layer_tag] = {
                "units": [
                    {
                        "name": f"layer_{layer['id']}_extent",
                        "lx": min_x + offset_x,
                        "ly": min_y + offset_y,
                        "lz": z_cursor,
                        "dx": layer_width,
                        "dy": layer_height,
                        "dz": thickness,
                    }
                ],
            }

            for unit in flp_units:
                entity = {
                    "name": unit["name"],
                    "lx": unit["left_x"] + offset_x,
                    "ly": unit["bottom_y"] + offset_y,
                    "lz": z_cursor,
                    "dx": unit["width"],
                    "dy": unit["height"],
                    "dz": thickness,
                }
                if layer["power"]:
                    power_units.append(entity)

            z_cursor += thickness

    else:
        flp_path = _find_first_by_suffix(example_dir, ".flp")
        if flp_path:
            flp_units = parser.parse_flp(flp_path)
            chip_thickness = float(config.get("t_chip", 0.00015))
            layer_tag = 1
            domain_assignment.setdefault("silicon", []).append(layer_tag)

            if flp_units:
                min_x, min_y, layer_width, layer_height = _layout_bbox_from_flp_units(
                    flp_units
                )
                offset_x = (global_width - layer_width) / 2.0 - min_x
                offset_y = (global_height - layer_height) / 2.0 - min_y

                layers_entities[layer_tag] = {
                    "units": [
                        {
                            "name": "chip_extent",
                            "lx": min_x + offset_x,
                            "ly": min_y + offset_y,
                            "lz": z_cursor,
                            "dx": layer_width,
                            "dy": layer_height,
                            "dz": chip_thickness,
                        }
                    ],
                }

                for unit in flp_units:
                    power_units.append(
                        {
                            "name": unit["name"],
                            "lx": unit["left_x"] + offset_x,
                            "ly": unit["bottom_y"] + offset_y,
                            "lz": z_cursor,
                            "dx": unit["width"],
                            "dy": unit["height"],
                            "dz": chip_thickness,
                        }
                    )

            z_cursor += chip_thickness

    def add_package_layer(
        name: str,
        thickness: float,
        side_length: float,
        material_name: str,
        tag: int,
    ) -> None:
        nonlocal z_cursor
        lx = (global_width - side_length) / 2.0
        ly = (global_height - side_length) / 2.0

        layers_entities[tag] = {
            "units": [
                {
                    "name": name,
                    "lx": lx,
                    "ly": ly,
                    "lz": z_cursor,
                    "dx": side_length,
                    "dy": side_length,
                    "dz": thickness,
                }
            ],
        }
        domain_assignment.setdefault(material_name, []).append(tag)
        z_cursor += thickness

    interface_material = ensure_material(
        str(config.get("material_interface", "tim")),
        "tim",
        "k_interface",
        "p_interface",
    )
    spreader_material = ensure_material(
        str(config.get("material_spreader", "copper")),
        "copper",
        "k_spreader",
        "p_spreader",
    )
    sink_material = ensure_material(
        str(config.get("material_sink", "copper")),
        "copper",
        "k_sink",
        "p_sink",
    )

    if not lcf_layers:
        add_package_layer(
            "TIM",
            float(config.get("t_interface", config.get("t_tim", 0.00002))),
            global_width,
            interface_material,
            1000,
        )

    add_package_layer(
        "Spreader",
        float(config.get("t_spreader", 0.001)),
        float(config.get("s_spreader", max(global_width, global_height))),
        spreader_material,
        1001,
    )
    add_package_layer(
        "Sink",
        float(config.get("t_sink", 0.0069)),
        float(config.get("s_sink", max(global_width, global_height))),
        sink_material,
        1002,
    )

    sink_side = float(config.get("s_sink", max(global_width, global_height)))
    sink_area = sink_side * sink_side
    r_convec = float(config.get("r_convec", 0.1))

    boundary_conditions.append(
        {
            "name": "sink_conv",
            "type": "convection",
            "h": 1.0 / (r_convec * sink_area),
            "T_inf": float(config.get("ambient", 318.15)),
            "selection": [1002],
        }
    )

    base_data = {
        "config": config,
        "materials": materials,
        "domain_assignment": domain_assignment,
        "layers_entities": layers_entities,
        "power_units": power_units,
        "boundary_conditions": boundary_conditions,
        "global_width": global_width,
        "global_height": global_height,
    }

    return base_data, layers_entities, ptrace_name


def convert_hotspot_to_metahotspot(
    example_dir: str,
    output_dir: str,
    simulation_type: str = "steady",
    output_config_name: str = "solver_config.toml",
    generate_mesh: bool = True,
) -> str:
    _ensure_dir(output_dir)

    parser = HotSpotParser()
    base_data, layers_entities, ptrace_name = _build_base_model_data(
        parser, example_dir
    )
    config = base_data["config"]

    ptrace_source = _find_first_by_suffix(example_dir, ".ptrace")
    if ptrace_source:
        shutil.copy(
            ptrace_source, os.path.join(output_dir, os.path.basename(ptrace_source))
        )

    sampling_intvl = float(config.get("sampling_intvl", 0.01))
    timestep = float(config.get("timestep", sampling_intvl))
    total_time = float(
        config.get("time", _estimate_total_time(ptrace_source, sampling_intvl))
    )

    init_file = str(config.get("init_file", "(null)"))
    init_file = "" if init_file in {"(null)", "null", "None"} else init_file

    toml_data = {
        "simulation_type": simulation_type,
        "time": total_time,
        "timestep": timestep,
        "sampling_intvl": sampling_intvl,
        "proc_freq": float(config.get("base_proc_freq", 3.0e9)),
        "materials": base_data["materials"],
        "domain_material_assignment": base_data["domain_assignment"],
        "mesh_file_path": "mesh.msh",
        "ptrace_file_path": ptrace_name,
        "power_units": base_data["power_units"],
        "ambient": float(config.get("ambient", 318.15)),
        "init_temperature": float(
            config.get("init_temp", config.get("ambient", 318.15))
        ),
        "boundary_conditions": base_data["boundary_conditions"],
    }

    if init_file:
        toml_data["init_temperature_file_path"] = init_file

    config_path = os.path.join(output_dir, output_config_name)
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(toml_data, handle)

    if generate_mesh:
        mesher = GmshMesher()

        # 提取网格控制参数
        base_size = max(float(config.get("s_sink", 0.06)) / 10.0, 0.006)
        min_size = 0.001
        # 热扩散半径：影响热源正下方的细化面积，默认 10mm
        refine_dist = 0.010

        # 使用全新的 2.5D Quadtree 生成器
        mesher.generate_2_5D_mesh(
            layers_entities=layers_entities,
            power_units=base_data["power_units"],
            base_mesh_size=base_size,
            min_mesh_size=min_size,
            refine_distance=refine_dist,
        )

        mesher.finalize(os.path.join(output_dir, "mesh.msh"))

    return config_path


def convert_hotspot_with_modes(
    example_dir: str, output_dir: str, mode: str = "both"
) -> List[str]:
    normalized_mode = mode.lower().strip()
    if normalized_mode not in {"steady", "transient", "both"}:
        raise ValueError("mode must be one of: steady, transient, both")

    if normalized_mode == "steady":
        return [
            convert_hotspot_to_metahotspot(
                example_dir,
                output_dir,
                simulation_type="steady",
                output_config_name="solver_config_steady.toml",
                generate_mesh=True,
            )
        ]

    if normalized_mode == "transient":
        return [
            convert_hotspot_to_metahotspot(
                example_dir,
                output_dir,
                simulation_type="transient",
                output_config_name="solver_config_transient.toml",
                generate_mesh=True,
            )
        ]

    created = [
        convert_hotspot_to_metahotspot(
            example_dir,
            output_dir,
            simulation_type="steady",
            output_config_name="solver_config_steady.toml",
            generate_mesh=True,
        ),
        convert_hotspot_to_metahotspot(
            example_dir,
            output_dir,
            simulation_type="transient",
            output_config_name="solver_config_transient.toml",
            generate_mesh=False,
        ),
    ]

    return created

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

```

### File: gmsh_mesher.py
```py
import math
from typing import Dict, List

import gmsh
import numpy as np


class GmshMesher:
    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)

    def generate_2_5D_mesh(
        self,
        layers_entities: Dict[int, dict],
        power_units: List[dict],
        base_mesh_size: float = 0.006,
        min_mesh_size: float = 0.0005,
        refine_distance: float = 0.010,
    ) -> None:
        """
        生成 2.5D 挤压网格：层内 Quadtree 自适应（非共形），层间严格拉伸（共形）
        """
        # 1. 计算全局 2D 包围盒
        x_min = min(u["lx"] for l in layers_entities.values() for u in l["units"])
        x_max = max(
            u["lx"] + u["dx"] for l in layers_entities.values() for u in l["units"]
        )
        y_min = min(u["ly"] for l in layers_entities.values() for u in l["units"])
        y_max = max(
            u["ly"] + u["dy"] for l in layers_entities.values() for u in l["units"]
        )

        # 2. 收集所有物理边界和热源边界，用于触发网格细化
        geometry_boxes = []
        for l in layers_entities.values():
            for u in l["units"]:
                geometry_boxes.append(
                    (u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"])
                )

        heat_boxes = []
        for u in power_units:
            heat_boxes.append((u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"]))

        leaves = []

        def refine(x0: float, y0: float, x1: float, y1: float) -> None:
            """递归生成 2D Quadtree"""
            dx = x1 - x0
            dy = y1 - y0

            # 停止条件 1：达到最小网格尺寸
            if dx <= min_mesh_size * 1.01 and dy <= min_mesh_size * 1.01:
                leaves.append((x0, y0, x1, y1))
                return

            needs_refinement = False

            # 触发条件 1：距离热源较近 (捕捉热流扩散)
            for hb in heat_boxes:
                dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                if math.sqrt(dist_x**2 + dist_y**2) <= refine_distance:
                    needs_refinement = True
                    break

            # 触发条件 2：网格跨越了物理边界 (确保网格完美贴合所有层级模块的边缘)
            if not needs_refinement:
                for gb in geometry_boxes:
                    # 如果垂直边界穿过当前网格，并且 Y 方向有交集
                    if (x0 < gb[0] < x1 or x0 < gb[2] < x1) and (
                        max(y0, gb[1]) < min(y1, gb[3])
                    ):
                        needs_refinement = True
                        break
                    # 如果水平边界穿过当前网格，并且 X 方向有交集
                    if (y0 < gb[1] < y1 or y0 < gb[3] < y1) and (
                        max(x0, gb[0]) < min(x1, gb[2])
                    ):
                        needs_refinement = True
                        break

            if needs_refinement:
                mid_x = (x0 + x1) / 2.0
                mid_y = (y0 + y1) / 2.0
                split_x = dx > min_mesh_size * 1.01
                split_y = dy > min_mesh_size * 1.01

                xs = [x0, mid_x, x1] if split_x else [x0, x1]
                ys = [y0, mid_y, y1] if split_y else [y0, y1]

                for i in range(len(xs) - 1):
                    for j in range(len(ys) - 1):
                        refine(xs[i], ys[j], xs[i + 1], ys[j + 1])
            else:
                leaves.append((x0, y0, x1, y1))

        # 3. 初始化基础粗网格并启动递归划分
        nx = max(1, int(round((x_max - x_min) / base_mesh_size)))
        ny = max(1, int(round((y_max - y_min) / base_mesh_size)))
        xs = np.linspace(x_min, x_max, nx + 1)
        ys = np.linspace(y_min, y_max, ny + 1)

        for i in range(nx):
            for j in range(ny):
                refine(xs[i], ys[j], xs[i + 1], ys[j + 1])

        # 4. 将 2D Quadtree 向上拉伸 (Extrude) 到 3D 的每一层
        node_id = 1
        elem_id = 1

        for tag, layer_data in layers_entities.items():
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

            # 提取当前层的 Z 轴高度和厚度
            lz = layer_data["units"][0]["lz"]
            dz = layer_data["units"][0]["dz"]

            layer_node_tags = []
            layer_node_coords = []
            node_map = {}

            def get_node(x: float, y: float, z: float) -> int:
                nonlocal node_id
                key = (round(x, 6), round(y, 6), round(z, 6))
                if key not in node_map:
                    node_map[key] = node_id
                    layer_node_tags.append(node_id)
                    layer_node_coords.extend([x, y, z])
                    node_id += 1
                return node_map[key]

            element_tags = []
            element_nodes = []

            for leaf in leaves:
                x0, y0, x1, y1 = leaf
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

                # 只有当 2D 叶子节点落在当前物理层的有效区域内时，才生成 3D 实体
                inside = False
                for u in layer_data["units"]:
                    if (
                        u["lx"] <= cx <= u["lx"] + u["dx"]
                        and u["ly"] <= cy <= u["ly"] + u["dy"]
                    ):
                        inside = True
                        break

                if inside:
                    n0 = get_node(x0, y0, lz)
                    n1 = get_node(x1, y0, lz)
                    n2 = get_node(x1, y1, lz)
                    n3 = get_node(x0, y1, lz)
                    n4 = get_node(x0, y0, lz + dz)
                    n5 = get_node(x1, y0, lz + dz)
                    n6 = get_node(x1, y1, lz + dz)
                    n7 = get_node(x0, y1, lz + dz)

                    element_tags.append(elem_id)
                    element_nodes.extend([n0, n1, n2, n3, n4, n5, n6, n7])
                    elem_id += 1

            if element_tags:
                gmsh.model.mesh.addNodes(
                    3, discrete_tag, layer_node_tags, layer_node_coords
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
from typing import Dict, List


class HotSpotParser:
    @staticmethod
    def parse_flp(file_path: str) -> List[dict]:
        units: List[dict] = []
        if not os.path.exists(file_path):
            return units

        with open(file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

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

                # Optional extra fields for heterogeneous materials (Hotspot 6.0+)
                if len(parts) >= 7:
                    try:
                        unit["specific_heat"] = float(parts[5])
                        unit["resistivity"] = float(parts[6])
                        unit["k"] = (
                            1.0 / unit["resistivity"]
                            if unit["resistivity"] != 0
                            else 0.0
                        )
                    except ValueError:
                        pass

                units.append(unit)

        return units

    @staticmethod
    def parse_config(file_path: str) -> Dict[str, object]:
        config: Dict[str, object] = {}
        if not os.path.exists(file_path):
            return config

        with open(file_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                match = re.match(r"^-(\w+)\s+([^#]+)", line)
                if not match:
                    continue

                key, value = match.groups()
                value = value.strip()
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value

        return config

    @staticmethod
    def parse_materials(file_path: str) -> Dict[str, dict]:
        materials: Dict[str, dict] = {}
        if not os.path.exists(file_path):
            return materials

        with open(file_path, "r", encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]

        index = 0
        while index < len(lines):
            name = lines[index]
            material_type = lines[index + 1]
            conductivity = float(lines[index + 2])
            heat_capacity = float(lines[index + 3])

            if material_type.lower() == "fluid":
                dynamic_viscosity = float(lines[index + 4])
                materials[name] = {
                    "k": conductivity,
                    "cp": heat_capacity,
                    "fluid": True,
                    "dynamic_viscosity": dynamic_viscosity,
                }
                index += 5
            else:
                materials[name] = {
                    "k": conductivity,
                    "cp": heat_capacity,
                    "fluid": False,
                }
                index += 4

        return materials

    @staticmethod
    def parse_lcf(file_path: str) -> List[dict]:
        layers: List[dict] = []
        if not os.path.exists(file_path):
            return layers

        with open(file_path, "r", encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]

        index = 0
        while index < len(lines):
            layer_id = int(lines[index])
            has_power = lines[index + 2].upper() == "Y"
            field = lines[index + 3]

            try:
                cp = float(field)
                resistivity = float(lines[index + 4])
                thickness = float(lines[index + 5])
                flp_file = lines[index + 6]

                layers.append(
                    {
                        "id": layer_id,
                        "power": has_power,
                        "cp": cp,
                        "k": 1.0 / resistivity if resistivity != 0 else 0.0,
                        "thickness": thickness,
                        "flp_file": flp_file,
                        "type": "numeric",
                    }
                )
                index += 7
            except ValueError:
                thickness = float(lines[index + 4])
                flp_file = lines[index + 5]

                layers.append(
                    {
                        "id": layer_id,
                        "power": has_power,
                        "material": field,
                        "thickness": thickness,
                        "flp_file": flp_file,
                        "type": "named",
                    }
                )
                index += 6

        return layers

```

### File: __init__.py
```py
"""MetaHotspot Python package."""

```

