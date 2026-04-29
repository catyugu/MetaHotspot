import os
import json
import shutil
import csv
from typing import Dict, List, Tuple, Any

import toml
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
    "water": {"k": 0.6069, "cp": 4.17e6, "fluid": True, "dynamic_viscosity": 8.89e-4},
}


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
        self.config = self._normalize_config(raw_config)

        self.materials: Dict[str, dict] = {}
        self.stackup: List[dict] = []
        self.boundary_conditions: List[dict] = []
        self.global_width, self.global_height = self._calculate_global_size()

    def _normalize_config(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Boundary validation: clean config once, no fallbacks elsewhere."""
        cfg = dict(DEFAULT_CONFIG_SCHEMA)
        for k, v in raw.items():
            if k in cfg:
                cfg[k] = type(cfg[k])(v)
            else:
                cfg[k] = v

        cfg["t_interface"] = float(cfg.get("t_interface", cfg["t_tim"]))
        cfg["time"] = float(cfg.get("time", max(cfg["sampling_intvl"], 0.01)))
        cfg["timestep"] = float(cfg.get("timestep", cfg["sampling_intvl"]))
        cfg["init_temp"] = float(cfg.get("init_temp", cfg["ambient"]))
        return cfg

    def _calculate_global_size(self) -> Tuple[float, float]:
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        files_to_check = (
            [
                layer["flp_file"]
                for layer in lcf_layers
                if not layer.get("flp_file", "").lower().endswith(".csv")
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
            return 0.03, 0.03  # Default fallback for CSV-only microchannel grids

        return (max(widths), max(heights)) if widths else (0.01, 0.01)

    def build_materials(self) -> "SimulationModelBuilder25D":
        mat_path = os.path.join(self.example_dir, "example.materials")
        self.materials = self.parser.parse_materials(mat_path)
        for name, props in STANDARD_MATERIALS.items():
            if name not in self.materials:
                self.materials[name] = dict(props)
        if "coolant_visc" in self.config and "water" in self.materials:
            self.materials["water"]["dynamic_viscosity"] = float(
                self.config["coolant_visc"]
            )
        return self

    def _get_material_props(self, name: str, default_name: str) -> dict:
        chosen = str(name or "").strip().lower() or default_name
        return self.materials.get(
            chosen, STANDARD_MATERIALS.get(default_name, {"k": 1.0, "cp": 1.0e6})
        )

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
        ox, oy = (self.global_width - lw) / 2.0 - min_x, (
            self.global_height - lh
        ) / 2.0 - min_y

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

        if self._has_unprocessed_microchannel():
            mc_csv = os.path.join(self.example_dir, "horizontal.csv")
            self._handle_microchannel_layer("microchannel", 500, 0.0001, mc_csv)

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
            self.materials[mat_key] = STANDARD_MATERIALS.get(
                "copper", {"k": 400.0, "cp": 3.44e6}
            )

        layer = self._create_layer_dict(tag, name, thick, mat_key, False)
        layer.update({"lx": lx, "ly": ly, "dx": side, "dy": side})
        self.stackup.append(layer)

    def _has_unprocessed_microchannel(self) -> bool:
        if os.path.exists(os.path.join(self.example_dir, "horizontal.csv")):
            return not any("microchannel" in l.get("name", "") for l in self.stackup)
        return False

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

            self.materials.setdefault("water", STANDARD_MATERIALS["water"])
            self.stackup.append(
                self._create_layer_dict(
                    tag, name, thickness, "water", True, f"layouts/{layout_path}"
                )
            )

            self.boundary_conditions.extend(
                [
                    {
                        "name": "mc_inlet",
                        "type": "pressure",
                        "face": "-X",
                        "target": name,
                        "pressure": float(self.config.get("pumping_pressure", 52000)),
                        "temperature": float(
                            self.config.get("inlet_temperature", 298.15)
                        ),
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
        """Merge contiguous fluid cells into unified channels."""
        if not os.path.exists(csv_path):
            return []

        with open(csv_path, "r", encoding="utf-8") as f:
            grid = [
                [int(x.strip()) for x in row if x.strip()]
                for row in csv.reader(f)
                if row
            ]

        if not grid:
            return []
        rows, cols = len(grid), len(grid[0])
        dx, dy = 0.03 / cols, 0.03 / rows
        visited = [[False] * cols for _ in range(rows)]

        units = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and not visited[r][c]:
                    # Flood fill to find channel bounds
                    q, cells = [(r, c)], []
                    while q:
                        cr, cc = q.pop()
                        if visited[cr][cc] or grid[cr][cc] != 1:
                            continue
                        visited[cr][cc] = True
                        cells.append((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            if 0 <= cr + dr < rows and 0 <= cc + dc < cols:
                                q.append((cr + dr, cc + dc))

                    min_r, max_r = min(x[0] for x in cells), max(x[0] for x in cells)
                    min_c, max_c = min(x[1] for x in cells), max(x[1] for x in cells)

                    units.append(
                        {
                            "name": f"mc_channel_{len(units)}",
                            "lx": min_c * dx,
                            "ly": (rows - 1 - max_r) * dy,
                            "dx": (max_c - min_c + 1) * dx,
                            "dy": (max_r - min_r + 1) * dy,
                            "is_fluid": True,
                            "material": "water",
                            "k": 0.6069,
                            "cp": 4.17e6,
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
    config_name: str = "solver_config.toml",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model = (
        SimulationModelBuilder25D(HotSpotParser(), example_dir, output_dir)
        .build_materials()
        .build_chip_layers()
        .build_package_and_cooling()
        .get_result()
    )

    ptrace_path = _find_first_by_suffix(example_dir, ".ptrace")
    ptrace_name = os.path.basename(ptrace_path) if ptrace_path else ""
    if ptrace_path:
        shutil.copy(ptrace_path, os.path.join(output_dir, ptrace_name))

    toml_data = {
        "simulation_type": simulation_type,
        "time": model["config"]["time"],
        "timestep": model["config"]["timestep"],
        "sampling_intvl": model["config"]["sampling_intvl"],
        "proc_freq": model["config"]["base_proc_freq"],
        "ambient": model["config"]["ambient"],
        "init_temperature": model["config"]["init_temp"],
        "mesh_file_path": "mesh.msh",
        "ptrace_file_path": ptrace_name,
        "materials": model["materials"],
        "stackup": model["stackup"],
        "boundary_conditions": model["boundary_conditions"],
    }

    if model["config"]["init_file"] and model["config"]["init_file"] not in {
        "(null)",
        "null",
        "None",
    }:
        toml_data["init_temperature_file_path"] = model["config"]["init_file"]

    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(toml_data, f)
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
            example_dir, output_dir, "steady", "solver_config_steady.toml"
        ),
        convert_hotspot_to_metahotspot(
            example_dir, output_dir, "transient", "solver_config_transient.toml"
        ),
    ]
