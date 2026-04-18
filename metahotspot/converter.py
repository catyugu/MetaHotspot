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


def _is_csv_geometry(path: str) -> bool:
    return path.lower().endswith(".csv")


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
        if _is_csv_geometry(geometry_file):
            continue

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
    microchannel_cells: List[dict] = []
    boundary_conditions: List[dict] = []
    domain_assignment: Dict[str, List[int]] = {}
    microchannel_group_map: Dict[str, int] = {}

    z_cursor = 0.0
    next_boundary_group_id = 20000

    if lcf_layers:
        for layer in lcf_layers:
            layer_tag = int(layer["id"]) + 1
            geometry_file = os.path.join(example_dir, layer["flp_file"])
            thickness = float(layer["thickness"])
            mesh_size = 0.0005 if layer["power"] else 0.001

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

            if _is_csv_geometry(geometry_file):
                codes = parser.parse_csv_grid(geometry_file)
                rows = len(codes)
                cols = max((len(row) for row in codes), default=0)

                if rows > 0 and cols > 0:
                    cell_dx = global_width / cols
                    cell_dy = global_height / rows
                    code_counts: Dict[int, int] = {}

                    for row_index, row_values in enumerate(codes):
                        for col_index, code in enumerate(row_values):
                            if col_index >= cols:
                                continue

                            lx = col_index * cell_dx
                            ly = (rows - 1 - row_index) * cell_dy
                            cell_name = f"MC_L{layer['id']}_R{row_index}_C{col_index}_code{code}"
                            microchannel_cells.append(
                                {
                                    "name": cell_name,
                                    "layer_tag": layer_tag,
                                    "code": int(code),
                                    "lx": lx,
                                    "ly": ly,
                                    "lz": z_cursor,
                                    "dx": cell_dx,
                                    "dy": cell_dy,
                                    "dz": thickness,
                                }
                            )
                            code_counts[int(code)] = code_counts.get(int(code), 0) + 1

                    # Map CSV semantic codes to boundary entity group IDs.
                    # Typical convention in Hotspot microchannel CSVs:
                    # 0=solid/wall, 1=fluid interior, 2=outlet, 3=inlet.
                    for code_value in sorted(code_counts):
                        group_id = next_boundary_group_id
                        next_boundary_group_id += 1
                        key = f"layer_{layer['id']}_code_{code_value}"
                        microchannel_group_map[key] = group_id

                    inlet_key = f"layer_{layer['id']}_code_3"
                    outlet_key = f"layer_{layer['id']}_code_2"
                    fluid_key = f"layer_{layer['id']}_code_1"
                    inlet_temp = float(
                        config.get("inlet_temperature", config.get("ambient", 318.15))
                    )

                    if fluid_key in microchannel_group_map:
                        boundary_conditions.append(
                            {
                                "name": f"fluid_temp_layer_{layer['id']}",
                                "type": "temperature",
                                "T": inlet_temp,
                                "selection": [microchannel_group_map[fluid_key]],
                            }
                        )

                    if inlet_key in microchannel_group_map:
                        boundary_conditions.append(
                            {
                                "name": f"inlet_microchannel_layer_{layer['id']}",
                                "type": "inlet_velocity",
                                "v": float(config.get("inlet_velocity", 1.0)),
                                "selection": [microchannel_group_map[inlet_key]],
                            }
                        )

                    if outlet_key in microchannel_group_map:
                        boundary_conditions.append(
                            {
                                "name": f"outlet_microchannel_layer_{layer['id']}",
                                "type": "outlet_pressure",
                                "p": 0.0,
                                "selection": [microchannel_group_map[outlet_key]],
                            }
                        )

                layers_entities[layer_tag] = {
                    "mesh_size": mesh_size,
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

            flp_units = parser.parse_flp(geometry_file)
            if not flp_units:
                layers_entities[layer_tag] = {
                    "mesh_size": mesh_size,
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
                "mesh_size": mesh_size,
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
                    "mesh_size": 0.0005,
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
            else:
                layers_entities[layer_tag] = {
                    "mesh_size": 0.0005,
                    "units": [
                        {
                            "name": "chip_extent",
                            "lx": 0.0,
                            "ly": 0.0,
                            "lz": z_cursor,
                            "dx": global_width,
                            "dy": global_height,
                            "dz": chip_thickness,
                        }
                    ],
                }

            z_cursor += chip_thickness

    def add_package_layer(
        name: str,
        thickness: float,
        side_length: float,
        material_name: str,
        tag: int,
        mesh_size: float,
    ) -> None:
        nonlocal z_cursor
        lx = (global_width - side_length) / 2.0
        ly = (global_height - side_length) / 2.0

        layers_entities[tag] = {
            "mesh_size": mesh_size,
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

    add_package_layer(
        "TIM",
        float(config.get("t_interface", config.get("t_tim", 0.00002))),
        global_width,
        "tim",
        1000,
        0.001,
    )
    add_package_layer(
        "Spreader",
        float(config.get("t_spreader", 0.001)),
        float(config.get("s_spreader", max(global_width, global_height))),
        "copper",
        1001,
        0.003,
    )
    add_package_layer(
        "Sink",
        float(config.get("t_sink", 0.0069)),
        float(config.get("s_sink", max(global_width, global_height))),
        "aluminum",
        1002,
        0.006,
    )

    if not boundary_conditions:
        boundary_conditions.append(
            {
                "name": "sink_conv",
                "type": "convection",
                "h": 1.0
                / (
                    float(config.get("r_convec", 0.1))
                    * (
                        float(config.get("s_sink", max(global_width, global_height)))
                        ** 2
                    )
                ),
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
        "microchannel_cells": microchannel_cells,
        "microchannel_group_map": microchannel_group_map,
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

    if base_data["microchannel_cells"]:
        toml_data["microchannel_cells"] = base_data["microchannel_cells"]
    if base_data["microchannel_group_map"]:
        toml_data["microchannel_group_map"] = base_data["microchannel_group_map"]

    config_path = os.path.join(output_dir, output_config_name)
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(toml_data, handle)

    if generate_mesh:
        mesher = GmshMesher()
        node_id = 1
        elem_id = 1
        for tag, layer_data in layers_entities.items():
            node_id, elem_id = mesher.generate_layer_mesh_unified(
                tag,
                layer_data["units"],
                layer_data["mesh_size"],
                node_id,
                elem_id,
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

    legacy_template = os.path.join(output_dir, "solver_config.toml")
    if os.path.exists(legacy_template):
        os.remove(legacy_template)

    return created
