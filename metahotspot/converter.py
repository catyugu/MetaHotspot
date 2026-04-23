import os
import shutil
from typing import Dict, List, Tuple

import toml

from metahotspot.gmsh_mesher import GmshMesher
from metahotspot.hotspot_parser import HotSpotParser

DEFAULT_CONFIG_SCHEMA = {
    "ambient": 318.15,
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
}

STANDARD_MATERIALS = {
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    "water": {"k": 0.6, "cp": 4.2e6, "fluid": True, "dynamic_viscosity": 8.89e-4},
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


class SimulationModelBuilder25D:
    def __init__(self, parser: HotSpotParser, example_dir: str, output_dir: str):
        self.parser = parser
        self.example_dir = example_dir
        self.output_dir = output_dir

        raw_config = parser.parse_config(os.path.join(example_dir, "example.config"))
        self.config = {**DEFAULT_CONFIG_SCHEMA, **raw_config}
        self._finalize_config_logic()

        self.materials: Dict[str, dict] = {}
        self.stackup: List[dict] = []
        self.boundary_conditions: List[dict] = []

        self.global_width, self.global_height = self._calculate_global_size()

    def _finalize_config_logic(self) -> None:
        for k, v in DEFAULT_CONFIG_SCHEMA.items():
            self.config[k] = type(v)(self.config.get(k, v))
        self.config["t_interface"] = float(
            self.config.get("t_interface", self.config["t_tim"])
        )
        self.config["time"] = float(
            self.config.get("time", max(self.config["sampling_intvl"], 0.01))
        )
        self.config["timestep"] = float(
            self.config.get("timestep", self.config["sampling_intvl"])
        )
        self.config["init_temp"] = float(
            self.config.get("init_temp", self.config["ambient"])
        )

    def _calculate_global_size(self) -> Tuple[float, float]:
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        files_to_check = (
            [layer["flp_file"] for layer in lcf_layers]
            if lcf_layers
            else [f for f in os.listdir(self.example_dir) if f.endswith(".flp")]
        )

        widths, heights = [], []
        for file_name in files_to_check:
            units = self.parser.parse_flp(os.path.join(self.example_dir, file_name))
            if units:
                _, _, w, h = _layout_bbox_from_flp(units)
                widths.append(w)
                heights.append(h)
        return (max(widths), max(heights)) if widths else (0.01, 0.01)

    def build_materials(self) -> "SimulationModelBuilder25D":
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
            fallback = self.materials.get(
                fallback_name,
                STANDARD_MATERIALS.get(fallback_name, {"k": 1.0, "cp": 1.0e6}),
            )
            self.materials[chosen] = {
                "k": float(self.config.get(k_key, fallback["k"])),
                "cp": float(self.config.get(cp_key, fallback["cp"])),
                "fluid": False,
            }
        return chosen

    def _copy_flp_to_output(self, flp_source_name: str, fallback_name: str) -> str:
        if not flp_source_name:
            return ""
        src_path = os.path.join(self.example_dir, flp_source_name)
        if not os.path.exists(src_path):
            return ""

        target_name = os.path.basename(flp_source_name) or f"{fallback_name}.flp"
        dst_path = os.path.join(self.output_dir, target_name)
        if os.path.abspath(src_path) != os.path.abspath(dst_path):
            shutil.copy2(src_path, dst_path)
        return target_name

    def build_chip_layers(self) -> "SimulationModelBuilder25D":
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        if not lcf_layers:
            flp_path = _find_first_by_suffix(self.example_dir, ".flp")
            flp_units = self.parser.parse_flp(flp_path)
            flp_ref = self._copy_flp_to_output(os.path.basename(flp_path), "layer_1")
            self.stackup.append(
                {
                    "tag": 1,
                    "name": "layer_1",
                    "thickness": self.config["t_chip"],
                    "material": "silicon",
                    "active": bool(flp_units),
                    "flp_file": flp_ref,
                    "lx": 0.0,
                    "ly": 0.0,
                    "dx": self.global_width,
                    "dy": self.global_height,
                }
            )
            return self

        for layer in lcf_layers:
            tag = int(layer["id"]) + 1
            name = f"layer_{tag}"
            thickness = float(layer["thickness"])
            is_numeric = layer["type"] == "numeric"
            mat_name = f"{name}_mat" if is_numeric else str(layer["material"])

            if is_numeric:
                self.materials[mat_name] = {
                    "k": float(layer["k"]),
                    "cp": float(layer["cp"]),
                    "fluid": False,
                }

            flp_name = layer.get("flp_file", "")
            flp_units = self.parser.parse_flp(os.path.join(self.example_dir, flp_name))
            flp_ref = self._copy_flp_to_output(flp_name, name)

            self.stackup.append(
                {
                    "tag": tag,
                    "name": name,
                    "thickness": thickness,
                    "material": mat_name,
                    "active": bool(layer.get("power") and flp_units),
                    "flp_file": flp_ref,
                    "lx": 0.0,
                    "ly": 0.0,
                    "dx": self.global_width,
                    "dy": self.global_height,
                }
            )

        return self

    def build_package_and_cooling(self) -> "SimulationModelBuilder25D":
        has_lcf = bool(_find_first_by_suffix(self.example_dir, ".lcf"))

        mat_tim = self._ensure_material(
            self.config["material_interface"], "tim", "k_interface", "p_interface"
        )
        mat_spread = self._ensure_material(
            self.config["material_spreader"], "copper", "k_spreader", "p_spreader"
        )
        mat_sink = self._ensure_material(
            self.config["material_sink"], "copper", "k_sink", "p_sink"
        )

        def _add_pkg_layer(name, thick, side, mat, tag):
            lx, ly = (self.global_width - side) / 2.0, (self.global_height - side) / 2.0
            self.stackup.append(
                {
                    "tag": tag,
                    "name": name,
                    "thickness": thick,
                    "material": mat,
                    "active": False,
                    "lx": lx,
                    "ly": ly,
                    "dx": side,
                    "dy": side,
                }
            )

        if not has_lcf:
            _add_pkg_layer(
                "TIM", self.config["t_interface"], self.global_width, mat_tim, 1000
            )

        s_spread = float(
            self.config.get("s_spreader", max(self.global_width, self.global_height))
        )
        _add_pkg_layer(
            "Spreader", self.config["t_spreader"], s_spread, mat_spread, 1001
        )

        s_sink = float(
            self.config.get("s_sink", max(self.global_width, self.global_height))
        )
        _add_pkg_layer("Sink", self.config["t_sink"], s_sink, mat_sink, 1002)

        self.boundary_conditions.append(
            {
                "name": "sink_conv",
                "type": "convection",
                "h": 1.0 / (self.config["r_convec"] * s_sink * s_sink),
                "T_inf": self.config["ambient"],
                "target_geometry": "top_surface",
                "selection": [],
            }
        )
        return self

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
    output_config_name: str = "solver_config.toml",
    generate_mesh: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    builder = SimulationModelBuilder25D(HotSpotParser(), example_dir, output_dir)
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

    toml_data = {
        "simulation_type": simulation_type,
        "time": config["time"],
        "timestep": config["timestep"],
        "sampling_intvl": config["sampling_intvl"],
        "proc_freq": config["base_proc_freq"],
        "ambient": config["ambient"],
        "init_temperature": config["init_temp"],
        "mesh_file_path": "mesh.msh",
        "ptrace_file_path": ptrace_name,
        "materials": model["materials"],
        "stackup": model["stackup"],
        "boundary_conditions": model["boundary_conditions"],
    }

    if config["init_file"] and config["init_file"] not in {"(null)", "null", "None"}:
        toml_data["init_temperature_file_path"] = config["init_file"]

    if generate_mesh:
        from metahotspot.model25d import load_stackup

        loaded_stackup = load_stackup(toml_data, output_dir)
        mesher = GmshMesher()
        boundary_info = mesher.generate_2_5D_mesh(
            stackup=loaded_stackup,
            max_mesh_size=0.003,
            min_mesh_size=0.0005,
            refine_distance=0.001,
        )
        mesher.finalize(os.path.join(output_dir, "mesh.msh"))

        if boundary_info:
            z_max_val = max(
                info["val"] for info in boundary_info.values() if info["axis"] == "Z"
            )
            top_tags = [
                tag
                for tag, info in boundary_info.items()
                if info["axis"] == "Z" and abs(info["val"] - z_max_val) < 1e-12
            ]
            for bc in toml_data["boundary_conditions"]:
                if bc.pop("target_geometry", None) == "top_surface":
                    bc["selection"] = top_tags
    else:
        steady_config = os.path.join(output_dir, "solver_config_steady.toml")
        if os.path.exists(steady_config):
            try:
                prev_data = toml.load(steady_config)
                for bc_new, bc_old in zip(
                    toml_data["boundary_conditions"],
                    prev_data.get("boundary_conditions", []),
                ):
                    bc_new["selection"] = bc_old.get("selection", [])
                    bc_new.pop("target_geometry", None)
            except Exception:
                pass

    config_path = os.path.join(output_dir, output_config_name)
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(toml_data, handle)

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
    if mode == "transient":
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
