import os
import shutil
from typing import Dict, List, Tuple, Any

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
DEFAULT_COOLANT_VISC = 8.89e-4

STANDARD_MATERIALS = {
    "silicon": {"k": 130.0, "cp": 1.63e6, "fluid": False},
    "copper": {"k": 400.0, "cp": 3.44e6, "fluid": False},
    "aluminum": {"k": 237.0, "cp": 2.42e6, "fluid": False},
    "tim": {"k": 4.0, "cp": 4.0e6, "fluid": False},
    "water": {
        "k": 0.6,
        "cp": 4.2e6,
        "fluid": True,
        "dynamic_viscosity": DEFAULT_COOLANT_VISC,
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


def _collect_global_xy_size(
    parser: HotSpotParser, example_dir: str, lcf_layers: List[dict]
) -> Tuple[float, float]:
    widths, heights = [], []
    for layer in lcf_layers:
        units = parser.parse_flp(os.path.join(example_dir, layer["flp_file"]))
        if units:
            _, _, w, h = _layout_bbox_from_flp(units)
            widths.append(w)
            heights.append(h)

    if not widths:
        for file_name in os.listdir(example_dir):
            if file_name.endswith(".flp"):
                units = parser.parse_flp(os.path.join(example_dir, file_name))
                if units:
                    _, _, w, h = _layout_bbox_from_flp(units)
                    widths.append(w)
                    heights.append(h)

    return (max(widths), max(heights)) if widths else (0.01, 0.01)


def _init_materials(parser: HotSpotParser, example_dir: str, config: dict) -> dict:
    materials = parser.parse_materials(os.path.join(example_dir, "example.materials"))
    for name, props in STANDARD_MATERIALS.items():
        if name not in materials:
            materials[name] = dict(props)
            if name == "water" and "coolant_visc" in config:
                materials[name]["dynamic_viscosity"] = float(config["coolant_visc"])
    return materials


def _ensure_material_exists(
    mat_name: str,
    fallback_name: str,
    k_key: str,
    cp_key: str,
    materials: dict,
    config: dict,
) -> str:
    chosen = str(mat_name or "").strip().lower() or fallback_name
    if chosen not in materials:
        fallback = materials.get(fallback_name, {"k": 1.0, "cp": 1.0e6})
        materials[chosen] = {
            "k": float(config.get(k_key, fallback["k"])),
            "cp": float(config.get(cp_key, fallback["cp"])),
            "fluid": False,
        }
    return chosen


def _build_base_model_data(
    parser: HotSpotParser, example_dir: str
) -> Tuple[Dict, Dict, str]:
    config = parser.parse_config(os.path.join(example_dir, "example.config"))
    materials = _init_materials(parser, example_dir, config)

    ptrace_path = _find_first_by_suffix(example_dir, ".ptrace")
    lcf_path = _find_first_by_suffix(example_dir, ".lcf")
    lcf_layers = parser.parse_lcf(lcf_path) if lcf_path else []

    g_width, g_height = _collect_global_xy_size(parser, example_dir, lcf_layers)

    model = {
        "config": config,
        "materials": materials,
        "domain_assignment": {},
        "heterogeneous_overrides": [],
        "layers_entities": {},
        "power_units": [],
        "boundary_conditions": [],
        "global_width": g_width,
        "global_height": g_height,
    }

    z_cursor = 0.0

    if lcf_layers:
        for layer in lcf_layers:
            tag = int(layer["id"]) + 1
            thickness = float(layer["thickness"])
            mat_name = (
                f"layer_{layer['id']}_mat"
                if layer["type"] == "numeric"
                else str(layer["material"])
            )

            if layer["type"] == "numeric":
                materials[mat_name] = {
                    "k": float(layer["k"]),
                    "cp": float(layer["cp"]),
                    "fluid": False,
                }

            model["domain_assignment"].setdefault(mat_name, []).append(tag)

            flp_units = parser.parse_flp(os.path.join(example_dir, layer["flp_file"]))

            # 重要改动：将所有的 flp 单元都注入到 layers_entities 中，以此作为网格划分的基底
            if not flp_units:
                model["layers_entities"][tag] = {
                    "units": [
                        {
                            "name": f"layer_{layer['id']}_extent",
                            "lx": 0.0,
                            "ly": 0.0,
                            "lz": z_cursor,
                            "dx": g_width,
                            "dy": g_height,
                            "dz": thickness,
                        }
                    ]
                }
            else:
                min_x, min_y, lw, lh = _layout_bbox_from_flp(flp_units)
                ox, oy = (g_width - lw) / 2.0 - min_x, (g_height - lh) / 2.0 - min_y

                layer_units = []
                for u in flp_units:
                    entity = {
                        "name": u["name"],
                        "lx": u["left_x"] + ox,
                        "ly": u["bottom_y"] + oy,
                        "lz": z_cursor,
                        "dx": u["width"],
                        "dy": u["height"],
                        "dz": thickness,
                    }
                    layer_units.append(entity)

                    if layer["type"] == "numeric" and (
                        "k" in u or "specific_heat" in u
                    ):
                        override_mat = f"layer_{layer['id']}_unit_{len(model['heterogeneous_overrides'])}_mat"
                        materials[override_mat] = {
                            "k": float(u.get("k", layer["k"])),
                            "cp": float(u.get("specific_heat", layer["cp"])),
                            "fluid": False,
                        }
                        model["heterogeneous_overrides"].append(
                            {**entity, "material": override_mat}
                        )

                    if layer["power"]:
                        model["power_units"].append(entity)

                model["layers_entities"][tag] = {"units": layer_units}
            z_cursor += thickness
    else:
        # Fallback to single FLP
        flp_units = parser.parse_flp(_find_first_by_suffix(example_dir, ".flp"))
        thickness = float(config.get("t_chip", DEFAULT_T_CHIP))
        tag = 1
        model["domain_assignment"].setdefault("silicon", []).append(tag)

        if flp_units:
            min_x, min_y, lw, lh = _layout_bbox_from_flp(flp_units)
            ox, oy = (g_width - lw) / 2.0 - min_x, (g_height - lh) / 2.0 - min_y

            layer_units = []
            for u in flp_units:
                entity = {
                    "name": u["name"],
                    "lx": u["left_x"] + ox,
                    "ly": u["bottom_y"] + oy,
                    "lz": z_cursor,
                    "dx": u["width"],
                    "dy": u["height"],
                    "dz": thickness,
                }
                layer_units.append(entity)
                model["power_units"].append(entity)

            model["layers_entities"][tag] = {"units": layer_units}
        z_cursor += thickness

    def _add_pkg(name: str, thick: float, side: float, mat: str, tag: int) -> None:
        nonlocal z_cursor
        lx, ly = (g_width - side) / 2.0, (g_height - side) / 2.0
        model["layers_entities"][tag] = {
            "units": [
                {
                    "name": name,
                    "lx": lx,
                    "ly": ly,
                    "lz": z_cursor,
                    "dx": side,
                    "dy": side,
                    "dz": thick,
                }
            ]
        }
        model["domain_assignment"].setdefault(mat, []).append(tag)
        z_cursor += thick

    mat_tim = _ensure_material_exists(
        str(config.get("material_interface", "tim")),
        "tim",
        "k_interface",
        "p_interface",
        materials,
        config,
    )
    mat_spread = _ensure_material_exists(
        str(config.get("material_spreader", "copper")),
        "copper",
        "k_spreader",
        "p_spreader",
        materials,
        config,
    )
    mat_sink = _ensure_material_exists(
        str(config.get("material_sink", "copper")),
        "copper",
        "k_sink",
        "p_sink",
        materials,
        config,
    )

    if not lcf_layers:
        _add_pkg(
            "TIM",
            float(config.get("t_interface", config.get("t_tim", DEFAULT_T_TIM))),
            g_width,
            mat_tim,
            1000,
        )

    s_spread = float(config.get("s_spreader", max(g_width, g_height)))
    _add_pkg(
        "Spreader",
        float(config.get("t_spreader", DEFAULT_T_SPREADER)),
        s_spread,
        mat_spread,
        1001,
    )

    s_sink = float(config.get("s_sink", max(g_width, g_height)))
    _add_pkg(
        "Sink", float(config.get("t_sink", DEFAULT_T_SINK)), s_sink, mat_sink, 1002
    )

    r_convec = float(config.get("r_convec", DEFAULT_R_CONVEC))
    model["boundary_conditions"].append(
        {
            "name": "sink_conv",
            "type": "convection",
            "h": 1.0 / (r_convec * s_sink * s_sink),
            "T_inf": float(config.get("ambient", DEFAULT_AMBIENT)),
            "selection": [1002],
        }
    )

    return (
        model,
        model["layers_entities"],
        os.path.basename(ptrace_path) if ptrace_path else "",
    )


def convert_hotspot_to_metahotspot(
    example_dir: str,
    output_dir: str,
    simulation_type: str = "steady",
    output_config_name: str = "solver_config.toml",
    generate_mesh: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    parser = HotSpotParser()
    model, layers_entities, ptrace_name = _build_base_model_data(parser, example_dir)
    config = model["config"]

    if ptrace_name:
        shutil.copy(
            os.path.join(example_dir, ptrace_name),
            os.path.join(output_dir, ptrace_name),
        )

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
            layers_entities=layers_entities,
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
