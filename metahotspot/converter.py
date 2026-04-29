import os
import json
import shutil
from typing import Dict, List, Tuple

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
        self.layouts_dir = os.path.join(output_dir, "layouts")
        os.makedirs(self.layouts_dir, exist_ok=True)

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
                _, _, w, h = _layout_bbox_from_flp(units)
                widths.append(w)
                heights.append(h)

        # If no FLP files found (e.g., only CSV), use default or calculate from CSV
        if not widths and lcf_layers:
            for layer in lcf_layers:
                flp_file = layer.get("flp_file", "")
                if flp_file.lower().endswith(".csv"):
                    # Calculate size from CSV grid dimensions
                    csv_path = os.path.join(self.example_dir, flp_file)
                    if os.path.exists(csv_path):
                        import csv

                        with open(csv_path, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            rows = sum(1 for _ in reader)
                        # Assuming 0.03m chip
                        widths.append(0.03)
                        heights.append(0.03)
                        break

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

    def _export_layout_json(
        self,
        name: str,
        flp_units: List[dict],
        layer_k: float = None,
        layer_cp: float = None,
        is_numeric: bool = False,
    ) -> str:
        if not flp_units:
            return ""

        min_x, min_y, lw, lh = _layout_bbox_from_flp(flp_units)
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
            if is_numeric and ("k" in u or "specific_heat" in u):
                unit_data["k"] = float(u.get("k", layer_k))
                unit_data["cp"] = float(u.get("specific_heat", layer_cp))
            json_units.append(unit_data)

        file_name = f"{name}_layout.json"
        with open(
            os.path.join(self.layouts_dir, file_name), "w", encoding="utf-8"
        ) as f:
            json.dump(json_units, f, indent=2)
        return f"layouts/{file_name}"

    def build_chip_layers(self) -> "SimulationModelBuilder25D":
        lcf_path = _find_first_by_suffix(self.example_dir, ".lcf")
        lcf_layers = self.parser.parse_lcf(lcf_path) if lcf_path else []

        if not lcf_layers:
            flp_units = self.parser.parse_flp(
                _find_first_by_suffix(self.example_dir, ".flp")
            )
            layout_ref = self._export_layout_json("layer_1", flp_units)
            self.stackup.append(
                {
                    "tag": 1,
                    "name": "layer_1",
                    "thickness": self.config["t_chip"],
                    "material": "silicon",
                    "active": bool(flp_units),
                    "layout_file": layout_ref,
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
            flp_file = layer.get("flp_file", "")

            if is_numeric:
                self.materials[mat_name] = {
                    "k": float(layer["k"]),
                    "cp": float(layer["cp"]),
                    "fluid": False,
                }

            # Check if floorplan file is actually a microchannel CSV grid
            if flp_file.lower().endswith(".csv"):
                # Parse as microchannel grid instead of floorplan
                csv_path = os.path.join(self.example_dir, flp_file)
                mc_layer_cfg = {
                    "dx": self.global_width / 40.0,  # Approximate grid resolution
                    "dy": self.global_height / 39.0,
                    "thickness": thickness,
                }
                mc_units = self._build_microchannel_layer(csv_path, mc_layer_cfg)
                if mc_units:
                    mc_layout_path = os.path.join(
                        self.layouts_dir, f"{name}_microchannel_layout.json"
                    )
                    with open(mc_layout_path, "w", encoding="utf-8") as f:
                        json.dump(mc_units, f)

                    # Add water material if not exists
                    if "water" not in self.materials:
                        self.materials["water"] = {
                            "k": 0.6,
                            "cp": 4.17e6,
                            "fluid": True,
                        }

                    self.stackup.append(
                        {
                            "tag": tag,
                            "name": name,
                            "thickness": thickness,
                            "material": "water",
                            "active": True,
                            "layout_file": f"layouts/{name}_microchannel_layout.json",
                            "lx": 0.0,
                            "ly": 0.0,
                            "dx": self.global_width,
                            "dy": self.global_height,
                        }
                    )
                continue

            flp_units = self.parser.parse_flp(os.path.join(self.example_dir, flp_file))
            layout_ref = self._export_layout_json(
                name, flp_units, layer.get("k"), layer.get("cp"), is_numeric
            )

            self.stackup.append(
                {
                    "tag": tag,
                    "name": name,
                    "thickness": thickness,
                    "material": mat_name,
                    "active": bool(layer.get("power") and flp_units),
                    "layout_file": layout_ref,
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
                "face": "+Z",  # Top surface of the sink (outer boundary)
                "target": "Sink",  # Applies only to Sink layer
                "h": 1.0 / (self.config["r_convec"] * s_sink * s_sink),
                "T_inf": self.config["ambient"],
            }
        )

        # Check for microchannel layer (horizontal.csv) in various locations
        # Could be at root level or in subdirectory like microchannel_geometries/
        mc_csv = None

        # Check root level first
        root_csv = os.path.join(self.example_dir, "horizontal.csv")
        if os.path.exists(root_csv):
            mc_csv = root_csv

        # Check subdirectories
        if mc_csv is None:
            for entry in os.listdir(self.example_dir):
                full_path = os.path.join(self.example_dir, entry)
                if os.path.isdir(full_path):
                    sub_csv = os.path.join(full_path, "horizontal.csv")
                    if os.path.exists(sub_csv):
                        mc_csv = sub_csv
                        break

        # Check if LCF already created a microchannel layer (water layer with CSV-based layout)
        mc_layer = None
        mc_layer_idx = None
        for i, layer in enumerate(self.stackup):
            layout_file = layer.get("layout_file", "")
            if (
                layer.get("material") == "water"
                and "microchannel" in layout_file.lower()
            ):
                mc_layer = layer
                mc_layer_idx = i
                break

        if mc_csv is not None or mc_layer is not None:
            # Use existing microchannel layer from LCF, or create new one
            if mc_layer is None:
                mc_units = self._build_microchannel_layer(
                    mc_csv,
                    {
                        "dx": self.global_width / 100.0,
                        "dy": self.global_height / 100.0,
                    },
                )
                if mc_units:
                    mc_layout_path = os.path.join(
                        self.layouts_dir, "microchannel_layout.json"
                    )
                    with open(mc_layout_path, "w", encoding="utf-8") as f:
                        json.dump(mc_units, f)

                mc_layer = {
                    "tag": 500,
                    "name": "microchannel",
                    "thickness": 0.0001,
                    "material": "water",
                    "active": True,
                    "layout_file": "layouts/microchannel_layout.json",
                    "lx": 0.0,
                    "ly": 0.0,
                    "dx": self.global_width,
                    "dy": self.global_height,
                }
                self.stackup.append(mc_layer)
                mc_layer_idx = len(self.stackup) - 1
            else:
                # Use existing layer, check if we need to build layout from CSV
                if mc_csv is not None and mc_layer.get("layout_file") == "":
                    mc_units = self._build_microchannel_layer(
                        mc_csv,
                        {
                            "dx": self.global_width / 100.0,
                            "dy": self.global_height / 100.0,
                        },
                    )
                    if mc_units:
                        mc_layout_path = os.path.join(
                            self.layouts_dir, "microchannel_layout.json"
                        )
                        with open(mc_layout_path, "w", encoding="utf-8") as f:
                            json.dump(mc_units, f)
                        mc_layer["layout_file"] = "layouts/microchannel_layout.json"

            if "water" not in self.materials:
                self.materials["water"] = {
                    "k": 0.6069,
                    "cp": 4.172638e6,
                    "fluid": True,
                }

            # Add microchannel pressure boundary conditions
            inlet_temp = float(self.config.get("inlet_temperature", 298.15))
            pumping_pressure = float(self.config.get("pumping_pressure", 52000))

            # Use the actual layer name from the microchannel layer
            layer_name = mc_layer["name"]

            self.boundary_conditions.append(
                {
                    "name": "mc_inlet",
                    "type": "pressure",
                    "face": "-X",
                    "target": layer_name,
                    "pressure": pumping_pressure,
                    "temperature": inlet_temp,
                }
            )

            self.boundary_conditions.append(
                {
                    "name": "mc_outlet",
                    "type": "pressure",
                    "face": "+X",
                    "target": layer_name,
                    "pressure": 0.0,
                }
            )

        return self

    def _build_microchannel_layer(self, csv_path: str, layer_cfg: dict) -> List[dict]:
        """Parse horizontal.csv and create merged microchannel channel entities.

        Merges contiguous fluid cells (value 1) into full channel rectangles.
        Inlet/outlet walls (values 2/3) are detected geometrically by the solver
        using boundary face direction matching, so we just mark all as cell_type=1.

        CSV format:
            0 = solid (skip)
            1 = fluid (active channel)
            2 = inlet wall
            3 = outlet wall
        """
        if not os.path.exists(csv_path):
            return []

        import csv

        grid = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                row_vals = [int(x.strip()) for x in row if x.strip()]
                if row_vals:
                    grid.append(row_vals)

        if not grid:
            return []

        rows, cols = len(grid), len(grid[0])

        # Cell size based on chip dimensions
        dx = 0.03 / cols
        dy = 0.03 / rows

        # Find all fluid regions (value 1) - merge contiguous cells into channels
        visited = [[False] * cols for _ in range(rows)]
        channels = []

        def flood_fill(start_row, start_col):
            """Find all contiguous fluid cells (value 1) connected to start."""
            if visited[start_row][start_col]:
                return None
            if grid[start_row][start_col] != 1:
                return None

            queue = [(start_row, start_col)]
            cells = []
            while queue:
                r, c = queue.pop()
                if visited[r][c]:
                    continue
                if grid[r][c] != 1:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                if r > 0:
                    queue.append((r - 1, c))
                if r < rows - 1:
                    queue.append((r + 1, c))
                if c > 0:
                    queue.append((r, c - 1))
                if c < cols - 1:
                    queue.append((r, c + 1))
            return cells

        # Find all fluid regions
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1 and not visited[row][col]:
                    cells = flood_fill(row, col)
                    if cells:
                        min_r = min(r for r, c in cells)
                        max_r = max(r for r, c in cells)
                        min_c = min(c for r, c in cells)
                        max_c = max(c for r, c in cells)
                        channels.append(
                            {
                                "row_range": (min_r, max_r),
                                "col_range": (min_c, max_c),
                            }
                        )

        # Build units from channels - each channel becomes one entity
        units = []
        for idx, ch in enumerate(channels):
            min_r, max_r = ch["row_range"]
            min_c, max_c = ch["col_range"]

            # Y: flip row index (CSV row 0 = top)
            ly = (rows - 1 - max_r) * dy
            lx = min_c * dx
            unit_dy = (max_r - min_r + 1) * dy
            unit_dx = (max_c - min_c + 1) * dx

            unit = {
                "name": f"mc_channel_{idx}",
                "lx": lx,
                "ly": ly,
                "dx": unit_dx,
                "dy": unit_dy,
                "cell_type": 1,  # All merged channels are fluid
                "material": "water",
                "k": 0.6069,
                "cp": 4.172638e6,
            }
            units.append(unit)

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
    output_config_name: str = "solver_config.toml",
) -> str:
    """Convert HotSpot example to MetaHotspot config TOML.

    Note: Meshing is decoupled. Use GmshMesher separately with the config path.
    """
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

    config_path = os.path.join(output_dir, output_config_name)
    with open(config_path, "w", encoding="utf-8") as handle:
        toml.dump(toml_data, handle)

    return config_path


def convert_hotspot_with_modes(
    example_dir: str, output_dir: str, mode: str = "both"
) -> List[str]:
    """Convert HotSpot example to MetaHotspot configs (steady + transient).

    Meshing is decoupled - call GmshMesher separately with config path.
    """
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
