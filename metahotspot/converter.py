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
