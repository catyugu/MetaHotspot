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
from typing import Dict, List, Tuple, Optional

import toml

from metahotspot.gmsh_mesher import GmshMesher
from metahotspot.hotspot_parser import HotSpotParser


# 常量定义
DEFAULT_AMBIENT = 318.15
DEFAULT_T_CHIP = 0.00015
DEFAULT_T_TIM = 0.00002
DEFAULT_T_SPREADER = 0.001
DEFAULT_T_SINK = 0.0069
DEFAULT_PROC_FREQ = 3.0e9
DEFAULT_R_CONVEC = 0.1

STANDARD_MATERIALS = {
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    "water": {
        "k": 0.6,
        "cp": 4.2e6,
        "fluid": True,
        "dynamic_viscosity": 8.89e-4,
    },
}


def _find_first_by_suffix(directory: str, suffix: str) -> str:
    for entry in os.listdir(directory):
        if entry.endswith(suffix):
            return os.path.join(directory, entry)
    return ""


def _layout_bbox_from_flp(units: List[dict]) -> Tuple[float, float, float, float]:
    if not units:
        return 0.0, 0.0, 0.01, 0.01
    min_x = min(u["left_x"] for u in units)
    min_y = min(u["bottom_y"] for u in units)
    max_x = max(u["left_x"] + u["width"] for u in units)
    max_y = max(u["bottom_y"] + u["height"] for u in units)
    return min_x, min_y, max_x - min_x, max_y - min_y


class SimulationModelBuilder:
    """Builder 模式：负责解耦并逐步构建仿真模型数据"""

    def __init__(self, parser: HotSpotParser, example_dir: str):
        self.parser = parser
        self.example_dir = example_dir
        self.config = parser.parse_config(os.path.join(example_dir, "example.config"))

        self.materials: Dict[str, dict] = {}
        self.domain_assignment: Dict[str, List[int]] = {}
        self.heterogeneous_overrides: List[dict] = []
        self.layers_entities: Dict[int, dict] = {}
        self.power_units: List[dict] = []
        self.boundary_conditions: List[dict] = []

        self.z_cursor = 0.0
        self.global_width, self.global_height = self._calculate_global_size()

    def _calculate_global_size(self) -> Tuple[float, float]:
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        widths, heights = [], []
        files_to_check = (
            [layer["flp_file"] for layer in lcf_layers]
            if lcf_layers
            else [f for f in os.listdir(self.example_dir) if f.endswith(".flp")]
        )

        for file_name in files_to_check:
            units = self.parser.parse_flp(os.path.join(self.example_dir, file_name))
            if units:
                _, _, w, h = _layout_bbox_from_flp(units)
                widths.append(w)
                heights.append(h)

        return (max(widths), max(heights)) if widths else (0.01, 0.01)

    def build_materials(self) -> "SimulationModelBuilder":
        mat_path = os.path.join(self.example_dir, "example.materials")
        self.materials = self.parser.parse_materials(mat_path)

        for name, props in STANDARD_MATERIALS.items():
            if name not in self.materials:
                self.materials[name] = dict(props)
                if name == "water" and "coolant_visc" in self.config:
                    self.materials[name]["dynamic_viscosity"] = float(
                        self.config["coolant_visc"]
                    )
        return self

    def _ensure_material(
        self, mat_name: str, fallback_name: str, k_key: str, cp_key: str
    ) -> str:
        chosen = str(mat_name or "").strip().lower() or fallback_name
        if chosen not in self.materials:
            fallback = self.materials.get(fallback_name, {"k": 1.0, "cp": 1.0e6})
            self.materials[chosen] = {
                "k": float(self.config.get(k_key, fallback["k"])),
                "cp": float(self.config.get(cp_key, fallback["cp"])),
                "fluid": False,
            }
        return chosen

    def _add_layer_entities(
        self,
        tag: int,
        thickness: float,
        flp_units: List[dict],
        layer_k: float = None,
        layer_cp: float = None,
        is_numeric: bool = False,
    ):
        if not flp_units:
            self.layers_entities[tag] = {
                "units": [
                    {
                        "name": f"layer_{tag}_extent",
                        "lx": 0.0,
                        "ly": 0.0,
                        "lz": self.z_cursor,
                        "dx": self.global_width,
                        "dy": self.global_height,
                        "dz": thickness,
                    }
                ]
            }
            return

        min_x, min_y, lw, lh = _layout_bbox_from_flp(flp_units)
        ox, oy = (self.global_width - lw) / 2.0 - min_x, (
            self.global_height - lh
        ) / 2.0 - min_y

        layer_units = []
        for u in flp_units:
            entity = {
                "name": u["name"],
                "lx": u["left_x"] + ox,
                "ly": u["bottom_y"] + oy,
                "lz": self.z_cursor,
                "dx": u["width"],
                "dy": u["height"],
                "dz": thickness,
            }
            layer_units.append(entity)

            if is_numeric and ("k" in u or "specific_heat" in u):
                k_val = float(u.get("k", layer_k))
                cp_val = float(u.get("specific_heat", layer_cp))

                self.heterogeneous_overrides.append(
                    {**entity, "k": k_val, "cp": cp_val}
                )

        self.layers_entities[tag] = {"units": layer_units}

    def build_chip_layers(self) -> "SimulationModelBuilder":
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        if lcf_layers:
            for layer in lcf_layers:
                tag = int(layer["id"]) + 1
                thickness = float(layer["thickness"])
                is_numeric = layer["type"] == "numeric"
                mat_name = (
                    f"layer_{layer['id']}_mat" if is_numeric else str(layer["material"])
                )

                if is_numeric:
                    self.materials[mat_name] = {
                        "k": float(layer["k"]),
                        "cp": float(layer["cp"]),
                        "fluid": False,
                    }

                self.domain_assignment.setdefault(mat_name, []).append(tag)
                flp_units = self.parser.parse_flp(
                    os.path.join(self.example_dir, layer["flp_file"])
                )

                self._add_layer_entities(
                    tag,
                    thickness,
                    flp_units,
                    layer.get("k"),
                    layer.get("cp"),
                    is_numeric,
                )

                if layer.get("power") and flp_units:
                    self.power_units.extend(self.layers_entities[tag]["units"])
                self.z_cursor += thickness
        else:
            # Fallback for pure FLP without LCF
            flp_units = self.parser.parse_flp(
                _find_first_by_suffix(self.example_dir, ".flp")
            )
            thickness = float(self.config.get("t_chip", DEFAULT_T_CHIP))
            tag = 1

            self.domain_assignment.setdefault("silicon", []).append(tag)
            self._add_layer_entities(tag, thickness, flp_units)
            if flp_units:
                self.power_units.extend(self.layers_entities[tag]["units"])
            self.z_cursor += thickness

        return self

    def _add_pkg_layer(
        self, name: str, thick: float, side: float, mat: str, tag: int
    ) -> None:
        lx, ly = (self.global_width - side) / 2.0, (self.global_height - side) / 2.0
        self.layers_entities[tag] = {
            "units": [
                {
                    "name": name,
                    "lx": lx,
                    "ly": ly,
                    "lz": self.z_cursor,
                    "dx": side,
                    "dy": side,
                    "dz": thick,
                }
            ]
        }
        self.domain_assignment.setdefault(mat, []).append(tag)
        self.z_cursor += thick

    def build_package_and_cooling(self) -> "SimulationModelBuilder":
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        has_lcf = bool(lcf_path)

        mat_tim = self._ensure_material(
            str(self.config.get("material_interface", "tim")),
            "tim",
            "k_interface",
            "p_interface",
        )
        mat_spread = self._ensure_material(
            str(self.config.get("material_spreader", "copper")),
            "copper",
            "k_spreader",
            "p_spreader",
        )
        mat_sink = self._ensure_material(
            str(self.config.get("material_sink", "copper")),
            "copper",
            "k_sink",
            "p_sink",
        )

        if not has_lcf:
            t_tim = float(
                self.config.get("t_interface", self.config.get("t_tim", DEFAULT_T_TIM))
            )
            self._add_pkg_layer("TIM", t_tim, self.global_width, mat_tim, 1000)

        s_spread = float(
            self.config.get("s_spreader", max(self.global_width, self.global_height))
        )
        self._add_pkg_layer(
            "Spreader",
            float(self.config.get("t_spreader", DEFAULT_T_SPREADER)),
            s_spread,
            mat_spread,
            1001,
        )

        s_sink = float(
            self.config.get("s_sink", max(self.global_width, self.global_height))
        )
        self._add_pkg_layer(
            "Sink",
            float(self.config.get("t_sink", DEFAULT_T_SINK)),
            s_sink,
            mat_sink,
            1002,
        )

        # 添加边界条件
        r_convec = float(self.config.get("r_convec", DEFAULT_R_CONVEC))
        self.boundary_conditions.append(
            {
                "name": "sink_conv",
                "type": "convection",
                "h": 1.0 / (r_convec * s_sink * s_sink),
                "T_inf": float(self.config.get("ambient", DEFAULT_AMBIENT)),
                "selection": [1002],
            }
        )

        return self

    def get_result(self) -> dict:
        return {
            "config": self.config,
            "materials": self.materials,
            "domain_assignment": self.domain_assignment,
            "heterogeneous_overrides": self.heterogeneous_overrides,
            "layers_entities": self.layers_entities,
            "power_units": self.power_units,
            "boundary_conditions": self.boundary_conditions,
        }


def convert_hotspot_to_metahotspot(
    example_dir: str,
    output_dir: str,
    simulation_type: str = "steady",
    output_config_name: str = "solver_config.toml",
    generate_mesh: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    parser = HotSpotParser()

    # 使用 Builder 模式构建模型
    builder = SimulationModelBuilder(parser, example_dir)
    model = (
        builder.build_materials()
        .build_chip_layers()
        .build_package_and_cooling()
        .get_result()
    )
    config = model["config"]

    ptrace_path = _find_first_by_suffix(example_dir, ".ptrace")
    ptrace_name = os.path.basename(ptrace_path) if ptrace_path else ""
    if ptrace_path:
        shutil.copy(ptrace_path, os.path.join(output_dir, ptrace_name))

    sampling_intvl = float(config.get("sampling_intvl", 0.01))

    toml_data = {
        "simulation_type": simulation_type,
        "time": float(config.get("time", max(sampling_intvl, 0.01))),
        "timestep": float(config.get("timestep", sampling_intvl)),
        "sampling_intvl": sampling_intvl,
        "proc_freq": float(config.get("base_proc_freq", DEFAULT_PROC_FREQ)),
        "materials": model["materials"],
        "domain_material_assignment": model["domain_assignment"],
        "heterogeneous_material_overrides": model["heterogeneous_overrides"],
        "mesh_file_path": "mesh.msh",
        "ptrace_file_path": ptrace_name,
        "power_units": model["power_units"],
        "ambient": float(config.get("ambient", DEFAULT_AMBIENT)),
        "init_temperature": float(
            config.get("init_temp", config.get("ambient", DEFAULT_AMBIENT))
        ),
        "boundary_conditions": model["boundary_conditions"],
    }

    init_file = str(config.get("init_file", ""))
    if init_file and init_file not in {"(null)", "null", "None"}:
        toml_data["init_temperature_file_path"] = init_file

    config_path = os.path.join(output_dir, output_config_name)
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(toml_data, handle)

    if generate_mesh:
        mesher = GmshMesher()
        mesher.generate_2_5D_mesh(
            layers_entities=model["layers_entities"],
            power_units=model["power_units"],
            max_mesh_size=0.003,
            min_mesh_size=0.0005,
            refine_distance=0.001,
        )
        mesher.finalize(os.path.join(output_dir, "mesh.msh"))

    return config_path


def convert_hotspot_with_modes(
    example_dir: str, output_dir: str, mode: str = "both"
) -> List[str]:
    mode = mode.lower().strip()
    if mode == "steady":
        return [
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "steady", "solver_config_steady.toml"
            )
        ]
    elif mode == "transient":
        return [
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "transient", "solver_config_transient.toml"
            )
        ]

    return [
        convert_hotspot_to_metahotspot(
            example_dir, output_dir, "steady", "solver_config_steady.toml", True
        ),
        convert_hotspot_to_metahotspot(
            example_dir, output_dir, "transient", "solver_config_transient.toml", False
        ),
    ]

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

        self._init_materials()
        self.cells: List[Cell] = []

        self._prepare_mesh()
        self._precompute_power_matrix()

    def _init_materials(self) -> None:
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

        # 基于 Morton Key 进行空间排序，加速后续查询与内存局部性
        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        sorted_indices = np.argsort(morton_keys)

        # [性能优化]: 向量化处理异构材料覆盖映射，替代原有的双层 for 循环
        mat_k_array = np.zeros(len(centers))
        mat_cp_array = np.zeros(len(centers))

        # 1. 初始化基础材料
        for i, tag in enumerate(physical_tags):
            mat = self.tag_to_material.get(tag, self.materials["silicon"])
            mat_k_array[i] = float(mat["k"])
            mat_cp_array[i] = float(mat["cp"])

        # 2. 向量化应用覆盖层 (Overrides)
        overrides = self.config.get("heterogeneous_material_overrides", [])
        for ov in overrides:
            # 优先应用内联属性，覆盖材料库中的值
            if "k" in ov and "cp" in ov:
                ov_k = float(ov["k"])
                ov_cp = float(ov["cp"])

            x0, y0, z0 = float(ov["lx"]), float(ov["ly"]), float(ov["lz"])
            x1, y1, z1 = (
                x0 + float(ov["dx"]),
                y0 + float(ov["dy"]),
                z0 + float(ov["dz"]),
            )

            mask = (
                (centers[:, 0] >= x0 - self.GEOMETRY_TOLERANCE)
                & (centers[:, 0] <= x1 + self.GEOMETRY_TOLERANCE)
                & (centers[:, 1] >= y0 - self.GEOMETRY_TOLERANCE)
                & (centers[:, 1] <= y1 + self.GEOMETRY_TOLERANCE)
                & (centers[:, 2] >= z0 - self.GEOMETRY_TOLERANCE)
                & (centers[:, 2] <= z1 + self.GEOMETRY_TOLERANCE)
            )
            mat_k_array[mask] = ov_k
            mat_cp_array[mask] = ov_cp
        # 构建最终的 Cells 列表
        for new_id, orig_id in enumerate(sorted_indices):
            tag = int(physical_tags[orig_id])
            box = np.array([*lowers[orig_id], *uppers[orig_id]])

            self.cells.append(
                Cell(
                    original_id=orig_id,
                    id=new_id,
                    center=centers[orig_id],
                    dims=dims[orig_id],
                    box=box,
                    k=mat_k_array[orig_id],
                    cp=mat_cp_array[orig_id],
                    tag=tag,
                    vol=float(vols[orig_id]),
                )
            )

        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}

    def _precompute_power_matrix(self) -> None:
        """预计算映射矩阵：使用 NumPy 广播加速包围盒相交计算"""
        power_units = self.config.get("power_units", [])
        self.unit_names = [u["name"] for u in power_units]

        if not power_units or not self.cells:
            self.power_matrix = sp.csr_matrix((len(self.cells), 0))
            return

        # 将所有 cell 的 box 提取为 numpy array: shape (N, 6)
        cell_boxes = np.array([c.box for c in self.cells])
        cell_lowers, cell_uppers = cell_boxes[:, :3], cell_boxes[:, 3:]

        rows, cols, data = [], [], []

        for unit_idx, unit in enumerate(power_units):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            u_lower = np.array([unit["lx"], unit["ly"], unit["lz"]])
            u_upper = u_lower + np.array([unit["dx"], unit["dy"], unit["dz"]])

            # 向量化计算所有 Cell 与当前 unit 的相交包围盒
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
            (data, (rows, cols)), shape=(len(self.cells), len(power_units))
        )

    def _get_initial_temperatures(self, n_cells: int) -> np.ndarray:
        default_temp = float(
            self.config.get("init_temperature", self.DEFAULT_INITIAL_TEMPERATURE)
        )
        init_file = self.config.get("init_temperature_file_path")

        if not init_file or init_file in {"(null)", "None", ""}:
            return np.full(n_cells, default_temp)

        init_path = os.path.join(self.base_dir, init_file)
        if not os.path.exists(init_path):
            print(
                f"[WARNING] Init file {init_path} not found. Using default {default_temp} K."
            )
            return np.full(n_cells, default_temp)

        print(f"[INFO] Loading initial state from {init_path}")
        init_mesh = meshio.read(init_path)
        temps = np.zeros(n_cells)
        offset = 0
        hex_data = init_mesh.cell_data.get("Temperature_K", [])

        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type == "hexahedron":
                for i, t in enumerate(block_temps):
                    new_id = self.orig_to_new_id.get(offset + i)
                    if new_id is not None:
                        temps[new_id] = t
                offset += len(block_temps)

        return temps

    def assemble_g_matrix(self) -> sp.csr_matrix:
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
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
                    continue

                for axis in range(3):
                    if (
                        abs(c_a.box[axis + 3] - c_b.box[axis]) < tol
                        or abs(c_a.box[axis] - c_b.box[axis + 3]) < tol
                    ):
                        area = _overlap_area(c_a.box, c_b.box, axis)
                        if area > self.GEOMETRY_TOLERANCE:
                            res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                                c_b.dims[axis] / (2.0 * c_b.k * area)
                            )
                            if res > self.GEOMETRY_TOLERANCE:
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
        z_max = max(c.box[5] for c in self.cells)

        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") != "convection":
                continue
            h, t_inf = float(bc["h"]), float(bc["T_inf"])
            selection = set(bc.get("selection", []))

            for c in self.cells:
                if (
                    c.tag in selection
                    and abs(c.box[5] - z_max) < self.GEOMETRY_TOLERANCE
                ):
                    area = c.dims[0] * c.dims[1]
                    g = 1.0 / ((0.5 * c.dims[2] / (c.k * area)) + (1.0 / (h * area)))
                    rows.append(c.id)
                    cols.append(c.id)
                    data.append(-g)
                    rhs[c.id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _load_ptrace(self) -> List[dict]:
        ptrace_path = os.path.join(
            self.base_dir, self.config.get("ptrace_file_path", "")
        )
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
        g_matrix = self.assemble_g_matrix()
        g_bc, boundary_rhs = self._build_boundary_terms()
        self.g_total = g_matrix + g_bc
        self.boundary_rhs = boundary_rhs
        self.ptrace_steps = self._load_ptrace()

        if self.config.get("simulation_type", "steady") == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
        print("[SIM] Solving steady state...")
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
        print("[SIM] Solving transient...")
        dt = float(self.config.get("timestep", 0.1))
        total_time = float(self.config.get("time", 0.0))
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)

        ptrace = self.ptrace_steps or [{}] * n_steps
        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - self.g_total).tocsc())

        temperatures = self._get_initial_temperatures(len(self.cells))

        for i, step_power in enumerate(ptrace):
            power_vec = np.array(
                [step_power.get(name, 0.0) for name in self.unit_names]
            )
            power_rhs = self.power_matrix @ power_vec

            rhs = (c_mat @ temperatures) + self.boundary_rhs + power_rhs
            temperatures = solve_step(rhs)

            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )

        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
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
from collections import deque

import gmsh


class GmshMesher:
    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)

    def generate_2_5D_mesh(
        self,
        layers_entities: Dict[int, dict],
        power_units: List[dict],
        max_mesh_size: float = 0.006,
        min_mesh_size: float = 0.0005,
        refine_distance: float = 0.010,
    ) -> None:
        """
        局部剖分策略：
        1. 以每一层的实际 functional units 作为初始网格节点（完美贴合 unit 边界，绝不外延拉伸）。
        2. 若单元过大 (w or h > max_mesh_size)，对其长边进行中点切分。
        3. 若单元处于热源附近，继续对长边进行细化，直至逼近 min_mesh_size。
        """
        # 提前收集热源框，用于局部加密判定
        heat_boxes = [
            (u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"])
            for u in power_units
        ]

        node_id = 1
        elem_id = 1

        for tag, layer_data in layers_entities.items():
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], tag)

            lz = layer_data["units"][0]["lz"]
            dz = layer_data["units"][0]["dz"]

            leaves = []
            queue = deque()

            # 初始化：直接以功能单元的物理边界框作为待细化的基础几何网格
            for u in layer_data["units"]:
                queue.append((u["lx"], u["ly"], u["lx"] + u["dx"], u["ly"] + u["dy"]))

            # 递归细分
            while queue:
                x0, y0, x1, y1 = queue.popleft()
                w = x1 - x0
                h = y1 - y0

                needs_split = False

                # 判定条件 1: 网格尺寸大于允许的最大尺寸限制
                if w > max_mesh_size or h > max_mesh_size:
                    needs_split = True
                # 判定条件 2: 网格在热源的影响范围内，且长边仍大于最小尺寸限制
                elif w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01:
                    for hb in heat_boxes:
                        dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                        dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                        if math.hypot(dist_x, dist_y) <= refine_distance:
                            needs_split = True
                            break

                # 执行切分：永远沿着最长的边切分一刀
                if needs_split:
                    if w >= h:
                        mid = (x0 + x1) / 2.0
                        queue.append((x0, y0, mid, y1))
                        queue.append((mid, y0, x1, y1))
                    else:
                        mid = (y0 + y1) / 2.0
                        queue.append((x0, y0, x1, mid))
                        queue.append((x0, mid, x1, y1))
                else:
                    leaves.append((x0, y0, x1, y1))

            # 根据最终的 leaves 构建当前层独立的 3D Hexahedrons (完全抛弃全层间的强行共形)
            layer_nodes_tags = []
            layer_nodes_coords = []
            node_map = {}

            def get_node(x: float, y: float, z: float) -> int:
                nonlocal node_id
                # 保持坐标精度位以防止浮点数误差产生冗余节点
                key = (round(x, 12), round(y, 12), round(z, 12))
                if key not in node_map:
                    node_map[key] = node_id
                    layer_nodes_tags.append(node_id)
                    layer_nodes_coords.extend([x, y, z])
                    node_id += 1
                return node_map[key]

            element_tags = []
            element_nodes = []

            for x0, y0, x1, y1 in leaves:
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

