import os
import json
import shutil
import csv
from typing import Dict, List, Tuple

import toml
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
        # 统一注入所有默认值，后续逻辑绝对信任 config
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

        # 边界条件持有多样参数列表（灵活字典）
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

            # 多样化的边界条件参数列表
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

    cfg = model["config"]
    ptrace_path = _find_first_by_suffix(example_dir, ".ptrace")
    ptrace_name = os.path.basename(ptrace_path) if ptrace_path else ""
    if ptrace_path:
        shutil.copy(ptrace_path, os.path.join(output_dir, ptrace_name))

    toml_data = {
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
        toml_data["init_temperature_file_path"] = cfg["init_file"]

    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w", encoding="utf-8") as f:
        toml.dump(toml_data, f)
    return config_path


def convert_hotspot_with_modes(
    example_dir: str, output_dir: str, mode: str = "both"
) -> List[str]:
    mode = mode.lower().strip()
    res = []
    if mode in ("steady", "both"):
        res.append(
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "steady", "solver_config_steady.toml"
            )
        )
    if mode in ("transient", "both"):
        res.append(
            convert_hotspot_to_metahotspot(
                example_dir, output_dir, "transient", "solver_config_transient.toml"
            )
        )
    return res
