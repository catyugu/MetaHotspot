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

```

### File: fvm_solver.py
```py
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import toml

from metahotspot.model25d import load_stackup


@dataclass(slots=True)
class Cell:
    """FVM cell representing a hexahedral mesh element.

    Cell types for microchannel:
        0 = SOLID (non-fluid)
        1 = FLUID (active fluid cell)
        2 = INLET (fluid cell with pressure BC)
        3 = OUTLET (fluid cell with pressure BC)
    """

    original_id: int
    id: int
    center: np.ndarray
    dims: np.ndarray
    box: np.ndarray
    k: float
    cp: float
    tag: int
    vol: float
    name: str = ""  # Unit name for BC matching
    layer_name: str = ""  # Layer name for BC matching
    cell_type: int = 0  # 0=solid, 1=fluid, 2=inlet, 3=outlet
    # Computed from pressure solve
    pressure: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Inlet temperature for advective BCs (set from pressure BC config)
    inlet_temp: float = 298.15


def _overlap_area(box_a: np.ndarray, box_b: np.ndarray, axis: int) -> float:
    axes = [(1, 2, 4, 5), (0, 2, 3, 5), (0, 1, 3, 4)][axis]
    d1 = min(box_a[axes[2]], box_b[axes[2]]) - max(box_a[axes[0]], box_b[axes[0]])
    d2 = min(box_a[axes[3]], box_b[axes[3]]) - max(box_a[axes[1]], box_b[axes[1]])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


class FVMSolver:
    """Finite Volume Method solver for 2.5D thermal simulation.

    Supports microchannel cooling with pressure-driven flow:
    - Build pressure matrix from hydraulic network
    - Solve for pressure at each fluid cell
    - Compute velocity from pressure gradient
    - Apply upwind advection scheme
    """

    GEOMETRY_TOLERANCE = 1e-12
    DEFAULT_INITIAL_TEMPERATURE = 318.15
    # Water properties for microchannel
    WATER_DENSITY = 1000.0  # kg/m^3
    WATER_VISCOSITY = 8.89e-4  # Pa·s

    def __init__(self, config_path: str) -> None:
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        self.mesh_path = os.path.join(
            self.base_dir, self.config.get("mesh_file_path", "mesh.msh")
        )
        self.mesh = meshio.read(self.mesh_path)

        self._sanitize_config()
        self._init_materials_and_stackup()

        self.cells: List[Cell] = []
        self._prepare_mesh()
        self._precompute_power_matrix()

    def _sanitize_config(self) -> None:
        self.config["init_temperature"] = float(
            self.config.get("init_temperature", self.DEFAULT_INITIAL_TEMPERATURE)
        )
        self.config["timestep"] = float(self.config.get("timestep", 0.1))
        self.config["time"] = float(self.config.get("time", 0.0))
        self.config["simulation_type"] = str(
            self.config.get("simulation_type", "steady")
        )
        self.config["ptrace_file_path"] = str(self.config.get("ptrace_file_path", ""))
        self.config.setdefault("stackup", [])
        self.config.setdefault("boundary_conditions", [])
        self.config.setdefault("init_temperature_file_path", None)

    def _init_materials_and_stackup(self) -> None:
        self.materials = self.config.get("materials", {})
        self.stackup = load_stackup(self.config, self.base_dir)

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

        b_min, b_max = np.min(lowers, axis=0), np.max(uppers, axis=0)
        diff = np.where((b_max - b_min) == 0, 1, b_max - b_min)
        norm_centers = np.clip(((centers - b_min) / diff * 1023).astype(int), 0, 1023)

        morton_keys = np.zeros(len(centers), dtype=int)
        for i in range(10):
            morton_keys |= ((norm_centers[:, 0] >> i) & 1) << (3 * i)
            morton_keys |= ((norm_centers[:, 1] >> i) & 1) << (3 * i + 1)
            morton_keys |= ((norm_centers[:, 2] >> i) & 1) << (3 * i + 2)

        sorted_indices = np.argsort(morton_keys)

        mat_k_array, mat_cp_array = np.zeros(len(centers)), np.zeros(len(centers))
        cell_layer_names = np.array([""] * len(centers), dtype=object)

        # 核心改动：利用 2.5D Stackup 为 3D 网格中心点映射材料属性
        tol = self.GEOMETRY_TOLERANCE
        z_cursor = 0.0
        for layer in self.stackup:
            z_min = z_cursor
            z_max = z_cursor + layer.thickness
            z_cursor = z_max

            layer_mask = (centers[:, 2] >= z_min - tol) & (centers[:, 2] <= z_max + tol)
            if not np.any(layer_mask):
                continue

            def_mat = self.materials.get(
                layer.default_material, {"k": 1.0, "cp": 1.0e6}
            )
            mat_k_array[layer_mask] = float(def_mat["k"])
            mat_cp_array[layer_mask] = float(def_mat["cp"])
            cell_layer_names[layer_mask] = layer.name

            # 覆盖异构材料单元
            for u in layer.units:
                u_mask = (
                    layer_mask
                    & (centers[:, 0] >= u.lx - tol)
                    & (centers[:, 0] <= u.lx + u.dx + tol)
                    & (centers[:, 1] >= u.ly - tol)
                    & (centers[:, 1] <= u.ly + u.dy + tol)
                )
                if np.any(u_mask):
                    if u.k is not None:
                        mat_k_array[u_mask] = u.k
                        mat_cp_array[u_mask] = u.cp
                    elif u.material and u.material in self.materials:
                        mat_k_array[u_mask] = float(self.materials[u.material]["k"])
                        mat_cp_array[u_mask] = float(self.materials[u.material]["cp"])

        # Pre-compute layer z-bounds for cell matching
        layer_z_min = {}
        z_cursor = 0.0
        for layer in self.stackup:
            layer_z_min[layer.name] = z_cursor
            z_cursor += layer.thickness

        self.face_to_cells: Dict[tuple, List[int]] = {}
        for new_id, orig_id in enumerate(sorted_indices):
            nodes = hex_data[orig_id]

            # Initialize cell properties
            cell_type = 0  # Default: solid
            unit_name = ""
            layer_name = cell_layer_names[orig_id]

            # Find matching Unit2D from stackup
            # Use center-based matching to handle mesh refinement
            # Only search in the cell's own layer (determined by z-position)
            c_center = centers[orig_id]
            for layer in self.stackup:
                # Check if cell's z-center is within this layer's z-range
                z_min = layer_z_min[layer.name]
                z_max = z_min + layer.thickness
                if c_center[2] >= z_min - tol and c_center[2] <= z_max + tol:
                    # Cell belongs to this layer, search its units
                    for u in layer.units:
                        if (
                            c_center[0] >= u.lx - tol
                            and c_center[0] <= u.lx + u.dx + tol
                            and c_center[1] >= u.ly - tol
                            and c_center[1] <= u.ly + u.dy + tol
                        ):
                            cell_type = getattr(
                                u, "cell_type", 0
                            )  # 0=solid, 1=fluid, 2=inlet, 3=outlet
                            unit_name = u.name
                            break
                    break

            c = Cell(
                original_id=orig_id,
                id=new_id,
                center=centers[orig_id],
                dims=dims[orig_id],
                box=np.array([*lowers[orig_id], *uppers[orig_id]]),
                k=float(mat_k_array[orig_id]),
                cp=float(mat_cp_array[orig_id]),
                tag=int(physical_tags[orig_id]),
                vol=float(vols[orig_id]),
                name=unit_name,
                layer_name=layer_name,
                cell_type=cell_type,
            )
            self.cells.append(c)

            fs = [  # 6 faces of hexahedron
                tuple(sorted([nodes[0], nodes[3], nodes[2], nodes[1]])),  # -Z face
                tuple(sorted([nodes[4], nodes[5], nodes[6], nodes[7]])),  # +Z face
                tuple(sorted([nodes[0], nodes[1], nodes[5], nodes[4]])),  # -Y face
                tuple(sorted([nodes[3], nodes[7], nodes[6], nodes[2]])),  # +Y face
                tuple(sorted([nodes[0], nodes[4], nodes[7], nodes[3]])),  # -X face
                tuple(sorted([nodes[1], nodes[2], nodes[6], nodes[5]])),  # +X face
            ]
            for f in fs:
                if f not in self.face_to_cells:
                    self.face_to_cells[f] = []
                self.face_to_cells[f].append(new_id)

        # Build internal and boundary face maps
        self.internal_faces = {
            f: tuple(c_ids)
            for f, c_ids in self.face_to_cells.items()
            if len(c_ids) == 2
        }
        self.boundary_faces_all = {
            f: tuple(c_ids)
            for f, c_ids in self.face_to_cells.items()
            if len(c_ids) == 1
        }

        self.orig_to_new_id = {c.original_id: c.id for c in self.cells}
        self._extract_boundary_faces()

    def _extract_boundary_faces(self) -> None:
        """Extract boundary faces and compute their outward normal direction.

        Creates self.boundary_faces_by_direction:
            { "+Z": [(cell_id, face_normal, area), ...],
              "-Z": [...],
              "+X": [...],
              "-X": [...],
              "+Y": [...],
              "-Y": [...] }
        """
        self.boundary_faces_by_direction: Dict[str, List[tuple]] = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }

        if not self.boundary_faces_all:
            return

        tol = self.GEOMETRY_TOLERANCE

        for f, (c_id,) in self.boundary_faces_all.items():
            pts = self.mesh.points[list(f)]

            # Calculate face normal (pointing outward from cell)
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            normal = cross_prod / area

            # Get cell center to determine outward direction
            c = self.cells[c_id]
            face_center = np.mean(pts, axis=0)

            # Vector from cell center to face center
            vec = face_center - c.center

            # If vector points same direction as normal, normal is outward
            # Otherwise flip it
            if np.dot(vec, normal) < 0:
                normal = -normal

            # Determine direction label
            abs_normal = np.abs(normal)
            if abs_normal[2] >= abs_normal[0] and abs_normal[2] >= abs_normal[1]:
                direction = "+Z" if normal[2] > 0 else "-Z"
            elif abs_normal[0] >= abs_normal[1]:
                direction = "+X" if normal[0] > 0 else "-X"
            else:
                direction = "+Y" if normal[1] > 0 else "-Y"

            self.boundary_faces_by_direction[direction].append((c_id, normal, area))

    def _solve_pressure(self) -> None:
        """Build and solve the hydraulic pressure matrix for microchannel.

        Uses Hagen-Poiseuille equation for hydraulic conductance:
            hydroC = (1 - 0.63*(min/max)) * min^3 * max / (12 * viscosity * L)

        The pressure matrix is a Laplacian-like system where:
        - Each fluid cell is a node
        - Edges between adjacent fluid cells have conductance hydroC
        - Inlet cells have Dirichlet BC: pressure = pumping_pressure
        - Outlet cells have Dirichlet BC: pressure = 0
        """
        # Get microchannel pressure BCs
        pressure_bcs = []
        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") == "pressure":
                pressure_bcs.append(
                    {
                        "face": bc.get("face", ""),
                        "target": bc.get("target", ""),
                        "pressure": float(bc.get("pressure", 0.0)),
                        "temperature": bc.get("temperature"),
                    }
                )

        # Build fluid connectivity graph
        fluid_cells = [c for c in self.cells if c.cell_type in (1, 2, 3)]
        if not fluid_cells:
            print("[INFO] No fluid cells found, skipping pressure solve")
            return

        # Create mapping from cell id to pressure matrix index
        cell_to_idx = {c.id: i for i, c in enumerate(fluid_cells)}
        n_fluid = len(fluid_cells)

        # Compute hydraulic conductance for each cell
        avg_dims = np.mean([c.dims for c in fluid_cells], axis=0)
        h = avg_dims[2]  # thickness (Z direction - height of channel)
        w = avg_dims[0]  # width (X direction)
        L = avg_dims[1]  # length (Y direction)

        viscosity = self.WATER_VISCOSITY

        # Hagen-Poiseuille for rectangular channel
        if abs(h - w) < 1e-10:  # Square
            hydroC = (0.42229 * h**4) / (12 * viscosity * L)
        elif h > w:
            hydroC = ((1 - 0.63 * (w / h)) * w**3 * h) / (12 * viscosity * L)
        else:
            hydroC = ((1 - 0.63 * (h / w)) * h**3 * w) / (12 * viscosity * L)

        print(f"[INFO] Hydraulic conductance: {hydroC:.6e} m^3/(Pa·s)")
        print(f"[INFO] Fluid cells: {n_fluid}")

        # Build pressure matrix (Laplacian-like)
        rows, cols, data = [], [], []

        for c in fluid_cells:
            i = cell_to_idx[c.id]

            # Find neighboring fluid cells
            neighbors = []
            for f, (c0_id, c1_id) in self.internal_faces.items():
                if c0_id == c.id and c1_id in cell_to_idx:
                    neighbors.append(c1_id)
                elif c1_id == c.id and c0_id in cell_to_idx:
                    neighbors.append(c0_id)

            # Diagonal entry: negative sum of all conductances
            rows.append(i)
            cols.append(i)
            data.append(-len(neighbors) * hydroC)

            # Off-diagonal entries
            for neighbor_id in neighbors:
                j = cell_to_idx[neighbor_id]
                rows.append(i)
                cols.append(j)
                data.append(hydroC)

        A_pressure = sp.csr_matrix((data, (rows, cols)), shape=(n_fluid, n_fluid))
        b_pressure = np.zeros(n_fluid)

        # Apply boundary conditions based on geometric location, not cell_type
        # For each pressure BC, find cells on the specified face direction
        for bc in pressure_bcs:
            face = bc["face"]
            target = bc["target"]
            pressure = bc["pressure"]
            temperature = bc.get("temperature")

            # Find boundary faces matching this BC
            bc_faces = self.boundary_faces_by_direction.get(face, [])
            for c_id, normal, area in bc_faces:
                c = self.cells[c_id]
                # Only apply to cells in the target layer
                if c.layer_name != target:
                    continue
                # Only apply to fluid cells
                if c.cell_type not in (1, 2, 3):
                    continue

                i = cell_to_idx.get(c.id)
                if i is None:
                    continue

                # Fix pressure at this cell
                A_pressure[i, :] = 0
                A_pressure[i, i] = 1
                b_pressure[i] = pressure

                # Set inlet temperature if provided
                if temperature is not None:
                    c.inlet_temp = temperature

        # Solve pressure system
        try:
            pressure = splinalg.spsolve(A_pressure, b_pressure)

            # Store pressure in cells
            for c in fluid_cells:
                c.pressure = pressure[cell_to_idx[c.id]]

            print(
                f"[INFO] Pressure solved. Range: {pressure.min():.2f} to {pressure.max():.2f} Pa"
            )

        except Exception as e:
            print(f"[WARNING] Pressure solve failed: {e}")
            # Set zero pressure as fallback
            for c in fluid_cells:
                c.pressure = 0.0

    def _compute_face_normal(self, pts: np.ndarray) -> np.ndarray:
        """Compute outward normal for a face given its points."""
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        cross_prod = np.cross(v1, v2)
        area = np.linalg.norm(cross_prod)
        if area > self.GEOMETRY_TOLERANCE:
            return cross_prod / area
        return np.array([0.0, 0.0, 1.0])

    def _compute_velocity_from_pressure(self) -> None:
        """Compute velocity at each fluid cell face from pressure gradient.

        Uses Darcy's law: v = -K * grad(P) / mu
        For simplicity, assumes velocity is proportional to pressure difference.
        """
        viscosity = self.WATER_VISCOSITY

        for c in self.cells:
            if c.cell_type not in (1, 2, 3):
                c.velocity = np.zeros(3)
                continue

            # Find pressure gradient from neighbors
            pressure_grad = np.zeros(3)
            count = 0

            for f, (c0_id, c1_id) in self.internal_faces.items():
                if c0_id == c.id:
                    c1 = self.cells[c1_id]
                    if c1.cell_type in (1, 2, 3):
                        # Vector from c to neighbor
                        dvec = c1.center - c.center
                        dist = np.linalg.norm(dvec)
                        if dist > self.GEOMETRY_TOLERANCE:
                            # Pressure difference in direction of neighbor
                            dP = c1.pressure - c.pressure
                            pressure_grad += dP * dvec / (dist * dist)
                            count += 1
                elif c1_id == c.id:
                    c0 = self.cells[c0_id]
                    if c0.cell_type in (1, 2, 3):
                        dvec = c0.center - c.center
                        dist = np.linalg.norm(dvec)
                        if dist > self.GEOMETRY_TOLERANCE:
                            dP = c0.pressure - c.pressure
                            pressure_grad += dP * dvec / (dist * dist)
                            count += 1

            if count > 0:
                pressure_grad /= count

            # Darcy's law: v = -k/mu * grad(P) where k is permeability
            # For a channel: k = hydroC * L / A
            # Simplified: velocity proportional to negative pressure gradient
            perm = 1e-10  # Approximate permeability
            c.velocity = -perm / viscosity * pressure_grad

            # Also check boundary faces for direction
            for f, (c_id,) in self.boundary_faces_all.items():
                if c_id != c.id:
                    continue
                pts = self.mesh.points[list(f)]
                normal = self._compute_face_normal(pts)
                area = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))

                if c.cell_type == 2:  # INLET
                    # Flow enters from boundary, velocity points inward
                    c.velocity = -normal * np.abs(c.pressure) * 0.001
                elif c.cell_type == 3:  # OUTLET
                    # Flow exits to boundary, velocity points outward
                    c.velocity = normal * np.abs(c.pressure) * 0.001

    def _add_fluid_advection_generic(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Assemble fluid advection matrix using upwind scheme and computed velocity.

        Returns:
            Tuple of (advection_matrix, advection_rhs)
        """
        n = len(self.cells)
        rows, cols, data = [], [], []
        rhs = np.zeros(n)
        tol = self.GEOMETRY_TOLERANCE

        # 1. Compute internal fluid face fluxes using velocity from pressure
        for f, (c0_id, c1_id) in self.internal_faces.items():
            c0, c1 = self.cells[c0_id], self.cells[c1_id]

            # Skip if not both fluid (cell_type 1, 2, or 3)
            if c0.cell_type == 0 or c1.cell_type == 0:
                continue

            # Get face points from mesh
            pts = self.mesh.points[list(f)]

            # Calculate face normal and area
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            n_vec = cross_prod / area

            # Ensure normal points from c0 to c1 (c1 - c0 direction)
            vec_c0_c1 = c1.center - c0.center
            if np.dot(n_vec, vec_c0_c1) < 0:
                n_vec = -n_vec

            # Use velocity from pressure solve (stored in cell.velocity)
            v_avg = 0.5 * (c0.velocity + c1.velocity)

            # Volume flux: Q = dot(v_avg, n_vec) * area
            vol_flux = np.dot(v_avg, n_vec) * area

            # Mass flux: m_dot = vol_flux * density
            density = self.WATER_DENSITY

            # Determine upstream cell based on velocity direction
            if np.dot(v_avg, n_vec) > 0:
                # Flow from c0 to c1, c0 is upstream
                upstream, downstream = c0, c1
                upstream_id, downstream_id = c0_id, c1_id
            else:
                # Flow from c1 to c0, c1 is upstream
                upstream, downstream = c1, c0
                upstream_id, downstream_id = c1_id, c0_id

            mass_flux = vol_flux * density
            cp = upstream.cp
            advection_term = mass_flux * cp

            # Only add significant terms
            if abs(advection_term) > tol:
                # Donor cell loses energy (negative coefficient)
                rows.append(upstream_id)
                cols.append(upstream_id)
                data.append(-advection_term)

                # Receiver cell gains energy (positive coefficient)
                rows.append(downstream_id)
                cols.append(downstream_id)
                data.append(advection_term)

        # 2. Handle fluid boundary faces (inlet/outlet) with temperature
        for f, (c0_id,) in self.boundary_faces_all.items():
            c0 = self.cells[c0_id]

            # Skip if not fluid (cell_type 0 = solid) or no inlet temperature
            if c0.cell_type == 0 or c0.inlet_temp is None:
                continue

            # Get face points
            pts = self.mesh.points[list(f)]

            # Calculate face area
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            cross_prod = np.cross(v1, v2)
            area = np.linalg.norm(cross_prod)

            if area < tol:
                continue

            # Velocity at boundary
            vel_mag = np.linalg.norm(c0.velocity)
            if vel_mag < tol:
                continue

            # Mass flux at inlet
            density = self.WATER_DENSITY
            mass_flux = vel_mag * area * density

            # Inlet: energy enters system from inlet_temp
            rhs[c0_id] += mass_flux * c0.cp * c0.inlet_temp

            # Boundary cell loses energy via outflow
            rows.append(c0_id)
            cols.append(c0_id)
            data.append(-mass_flux * c0.cp)

        G_adv = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        return G_adv, rhs

    def _precompute_power_matrix(self) -> None:
        # 核心改动：从 2.5D Stackup 收集 active_units 的 3D 信息
        active_units_3d = []
        z_cursor = 0.0
        for layer in self.stackup:
            if layer.active:
                for u in layer.units:
                    active_units_3d.append(
                        {
                            "name": u.name,
                            "lx": u.lx,
                            "ly": u.ly,
                            "lz": z_cursor,
                            "dx": u.dx,
                            "dy": u.dy,
                            "dz": layer.thickness,
                        }
                    )
            z_cursor += layer.thickness

        self.unit_names = [u["name"] for u in active_units_3d]

        if not active_units_3d or not self.cells:
            self.power_matrix = sp.csr_matrix((len(self.cells), 0))
            return

        cell_boxes = np.array([c.box for c in self.cells])
        cell_lowers, cell_uppers = cell_boxes[:, :3], cell_boxes[:, 3:]
        rows, cols, data = [], [], []

        for unit_idx, unit in enumerate(active_units_3d):
            vol = unit["dx"] * unit["dy"] * unit["dz"]
            if vol <= 0:
                continue

            u_lower = np.array([unit["lx"], unit["ly"], unit["lz"]])
            u_upper = u_lower + np.array([unit["dx"], unit["dy"], unit["dz"]])

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
            (data, (rows, cols)), shape=(len(self.cells), len(active_units_3d))
        )

    def _get_initial_temperatures(self, n_cells: int) -> np.ndarray:
        default_temp = self.config["init_temperature"]
        init_file = self.config["init_temperature_file_path"]
        if not init_file or init_file in {"(null)", "None", ""}:
            return np.full(n_cells, default_temp)
        init_path = os.path.join(self.base_dir, init_file)
        if not os.path.exists(init_path):
            return np.full(n_cells, default_temp)

        init_mesh = meshio.read(init_path)
        temps = np.zeros(n_cells)
        offset = 0
        hex_data = init_mesh.cell_data.get("Temperature_K", [])
        for block, block_temps in zip(init_mesh.cells, hex_data):
            if block.type != "hexahedron":
                continue
            for i, t in enumerate(block_temps):
                new_id = self.orig_to_new_id.get(offset + i)
                if new_id is not None:
                    temps[new_id] = t
            offset += len(block_temps)
        return temps

    def assemble_g_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        tol = self.GEOMETRY_TOLERANCE
        sorted_cells = sorted(self.cells, key=lambda c: c.box[0])
        active_list: List[Cell] = []

        for c_a in sorted_cells:
            active_list = [c for c in active_list if c.box[3] >= c_a.box[0] - tol]
            for c_b in active_list:
                if max(c_a.box[1], c_b.box[1]) > min(c_a.box[4], c_b.box[4]) + tol:
                    continue
                if max(c_a.box[2], c_b.box[2]) > min(c_a.box[5], c_b.box[5]) + tol:
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
                    res = (c_a.dims[axis] / (2.0 * c_a.k * area)) + (
                        c_b.dims[axis] / (2.0 * c_b.k * area)
                    )
                    if res <= tol:
                        continue
                    g = 1.0 / res
                    rows.extend([c_a.id, c_b.id, c_a.id, c_b.id])
                    cols.extend([c_a.id, c_b.id, c_b.id, c_a.id])
                    data.extend([-g, -g, g, g])
            active_list.append(c_a)
        return sp.csr_matrix(
            (data, (rows, cols)), shape=(len(self.cells), len(self.cells))
        )

    def _build_boundary_terms(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build boundary condition terms using direction and target-based selection.

        BC format with layer targeting:
            [[boundary_conditions]]
            name = "sink_conv"
            type = "convection"
            face = "+Z"
            target = "Sink"  # Layer name to apply this BC
            h = 2777.78
            T_inf = 318.15

        Cell-level override:
            [[boundary_conditions]]
            name = "cell_inlet"
            type = "inlet"
            unit_name = "microchannel_0"  # Specific unit
            face = "-X"
            temperature = 298.15
        """
        n = len(self.cells)
        rhs, rows, cols, data = np.zeros(n), [], [], []

        # Group BCs by direction
        bcs_by_direction: Dict[str, list] = {
            "+X": [],
            "-X": [],
            "+Y": [],
            "-Y": [],
            "+Z": [],
            "-Z": [],
        }

        for bc in self.config.get("boundary_conditions", []):
            if bc.get("type") == "convection":
                face = bc.get("face", "")
                if face in bcs_by_direction:
                    bcs_by_direction[face].append(bc)

        # Apply boundary conditions by direction
        for direction, bcs in bcs_by_direction.items():
            if not bcs:
                continue

            faces = self.boundary_faces_by_direction.get(direction, [])
            for cell_id, normal, area in faces:
                c = self.cells[cell_id]

                # Check for cell-level override first (unit_name match)
                cell_bc = None
                if c.name:  # Only check if cell has a name
                    for bc in self.config.get("boundary_conditions", []):
                        if bc.get("unit_name") and bc.get("unit_name") == c.name:
                            cell_bc = bc
                            break

                if cell_bc and cell_bc.get("type") == "convection":
                    h = float(cell_bc["h"])
                    t_inf = float(cell_bc["T_inf"])
                else:
                    # Find layer-level BC that matches this cell's layer
                    layer_bc = None
                    for bc in bcs:
                        target = bc.get("target", "")  # Layer name
                        if not target:
                            # No target specified, applies to all layers
                            layer_bc = bc
                        elif target == c.layer_name:
                            # Target matches this cell's layer
                            layer_bc = bc
                            break

                    if layer_bc is None:
                        continue

                    h = float(layer_bc["h"])
                    t_inf = float(layer_bc["T_inf"])

                dist = c.vol / area
                g = area / ((0.5 * dist / c.k) + (1.0 / h))
                rows.append(c.id)
                cols.append(c.id)
                data.append(-g)
                rhs[c.id] += g * t_inf

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n)), rhs

    def _load_ptrace(self) -> List[dict]:
        ptrace_path = os.path.join(self.base_dir, self.config["ptrace_file_path"])
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
        self.g_total = self.assemble_g_matrix() + self._build_boundary_terms()[0]
        self.boundary_rhs = self._build_boundary_terms()[1]

        # Check for fluid cells and solve pressure if present
        fluid_cells = [c for c in self.cells if c.cell_type in (1, 2, 3)]
        if fluid_cells:
            print(
                f"[INFO] Found {len(fluid_cells)} fluid cells, solving pressure-driven flow..."
            )
            # Solve pressure field first, then compute velocities
            self._solve_pressure()
            self._compute_velocity_from_pressure()

            # Assemble advection with computed velocities
            advection_mat, advection_rhs = self._add_fluid_advection_generic()
            self.g_total = self.g_total + advection_mat
            self.boundary_rhs = self.boundary_rhs + advection_rhs

        self.ptrace_steps = self._load_ptrace()
        if self.config["simulation_type"] == "steady":
            self._solve_steady_state()
        else:
            self._solve_transient()

    def _solve_steady_state(self) -> None:
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
        dt, total_time = self.config["timestep"], self.config["time"]
        n_steps = max(1, math.ceil(total_time / dt) if total_time > 0 else 1)
        ptrace = self.ptrace_steps or [{}] * n_steps
        c_mat = sp.diags([c.cp * c.vol for c in self.cells]) / dt
        solve_step = splinalg.factorized((c_mat - self.g_total).tocsc())
        temperatures = self._get_initial_temperatures(len(self.cells))

        for i, step_power in enumerate(ptrace):
            power_vec = np.array(
                [step_power.get(name, 0.0) for name in self.unit_names]
            )
            rhs = (
                (c_mat @ temperatures)
                + self.boundary_rhs
                + (self.power_matrix @ power_vec)
            )
            temperatures = solve_step(rhs)
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temperatures):.2f} K, T_max={np.max(temperatures):.2f} K"
                )
        self.save(temperatures, "transient_result.vtu")

    def save(self, temperatures: np.ndarray, output_name: str) -> None:
        import meshio

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
        out_mesh = meshio.Mesh(
            points=self.mesh.points,
            cells=hex_blocks,
            cell_data={"Temperature_K": temp_chunks},
        )
        out_mesh.write(os.path.join(self.base_dir, output_name))

```

### File: gmsh_mesher.py
```py
import math
from collections import deque
from pathlib import Path
from typing import List

import gmsh
import toml
from metahotspot.model25d import load_stackup


class GmshMesher:
    """Mesher that takes a config TOML path and produces a .msh file.

    Decoupled from converter - call with config path after conversion.
    """

    DEFAULT_MAX_MESH_SIZE = 0.003
    DEFAULT_MIN_MESH_SIZE = 0.0005
    DEFAULT_REFINEMENT_DISTANCE = 0.001

    def __init__(self, model_name: str = "MetaHotspotMesh") -> None:
        gmsh.initialize()
        gmsh.model.add(model_name)
        self._node_id = 1
        self._elem_id = 1
        self._node_map: dict = {}
        self._global_node_coords: dict = {}

    def generate_mesh(self, config_path: str, mesh_params: dict = None) -> None:
        """Generate mesh from config TOML path.

        Args:
            config_path: Path to solver_config.toml
            mesh_params: Optional dict with max_mesh_size, min_mesh_size, refine_distance.
                        Defaults to GmshMesher.DEFAULT_* values.
        """
        if mesh_params is None:
            mesh_params = {}

        base_dir = str(Path(config_path).parent)
        config = toml.load(config_path)

        max_mesh_size = mesh_params.get("max_mesh_size", self.DEFAULT_MAX_MESH_SIZE)
        min_mesh_size = mesh_params.get("min_mesh_size", self.DEFAULT_MIN_MESH_SIZE)
        refine_distance = mesh_params.get(
            "refine_distance", self.DEFAULT_REFINEMENT_DISTANCE
        )

        stackup = load_stackup(config, base_dir)
        self._generate_2_5D_mesh(stackup, max_mesh_size, min_mesh_size, refine_distance)

    def _generate_2_5D_mesh(
        self,
        stackup,
        max_mesh_size: float,
        min_mesh_size: float,
        refine_distance: float,
    ) -> None:
        """Internal mesh generation logic."""

        # Collect heat source boxes for mesh refinement
        heat_boxes = []
        for layer in stackup:
            if layer.active:
                for u in layer.units:
                    heat_boxes.append((u.lx, u.ly, u.lx + u.dx, u.ly + u.dy))

        z_cursor = 0.0

        for layer in stackup:
            discrete_tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.addPhysicalGroup(3, [discrete_tag], layer.tag)

            lz = z_cursor
            dz = layer.thickness
            z_cursor += dz

            leaves = self._subdivide_layer(
                layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
            )
            self._create_hex_elements(layer, discrete_tag, lz, dz, leaves)

    def _subdivide_layer(
        self, layer, max_mesh_size, min_mesh_size, refine_distance, heat_boxes
    ):
        """Subdivide layer into quad leaves for hex mesh generation."""
        leaves = []
        queue = deque()

        for u in layer.units:
            queue.append((u.lx, u.ly, u.lx + u.dx, u.ly + u.dy))

        while queue:
            x0, y0, x1, y1 = queue.popleft()
            w = x1 - x0
            h = y1 - y0

            needs_split = False

            if w > max_mesh_size or h > max_mesh_size:
                needs_split = True
            elif w > min_mesh_size * 1.01 or h > min_mesh_size * 1.01:
                for hb in heat_boxes:
                    dist_x = max(0.0, x0 - hb[2], hb[0] - x1)
                    dist_y = max(0.0, y0 - hb[3], hb[1] - y1)
                    if math.hypot(dist_x, dist_y) <= refine_distance:
                        needs_split = True
                        break

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

        return leaves

    def _get_node(self, x: float, y: float, z: float) -> int:
        """Get or create a node at (x, y, z)."""
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in self._node_map:
            self._node_map[key] = self._node_id
            self._global_node_coords[self._node_id] = (x, y, z)
            self._node_id += 1
        return self._node_map[key]

    def _create_hex_elements(self, layer, discrete_tag, lz, dz, leaves) -> None:
        """Create hex elements for a layer's quad leaves."""
        element_tags: List[int] = []
        element_nodes: List[int] = []
        used_node_ids = set()

        for x0, y0, x1, y1 in leaves:
            # Collect nodes for bottom face (-Z)
            n0 = self._get_node(x0, y0, lz)
            n1 = self._get_node(x1, y0, lz)
            n2 = self._get_node(x1, y1, lz)
            n3 = self._get_node(x0, y1, lz)

            # Collect nodes for top face (+Z)
            n4 = self._get_node(x0, y0, lz + dz)
            n5 = self._get_node(x1, y0, lz + dz)
            n6 = self._get_node(x1, y1, lz + dz)
            n7 = self._get_node(x0, y1, lz + dz)

            element_tags.append(self._elem_id)
            element_nodes.extend([n0, n1, n2, n3, n4, n5, n6, n7])
            used_node_ids.update([n0, n1, n2, n3, n4, n5, n6, n7])
            self._elem_id += 1

        if element_tags:
            # Build ordered node lists for addNodes
            layer_nodes_tags = sorted(used_node_ids)
            layer_nodes_coords = []
            for nid in layer_nodes_tags:
                x, y, z = self._global_node_coords[nid]
                layer_nodes_coords.extend([x, y, z])

            gmsh.model.mesh.addNodes(
                3, discrete_tag, layer_nodes_tags, layer_nodes_coords
            )
            gmsh.model.mesh.addElements(
                3, discrete_tag, [5], [element_tags], [element_nodes]
            )

    def finalize(self, output_path: str) -> None:
        """Write mesh file and cleanup gmsh."""
        gmsh.write(output_path)
        gmsh.finalize()

```

### File: hotspot_parser.py
```py
import os
import re
from typing import Dict, Generator, List


def _read_valid_lines(file_path: str) -> Generator[str, None, None]:
    """Generator: yields non-empty, non-comment lines from file."""
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

            # Optional extra fields for heterogeneous materials (Hotspot 6.0+)
            if len(parts) >= 7:
                try:
                    unit["specific_heat"] = float(parts[5])
                    unit["resistivity"] = float(parts[6])
                    unit["k"] = (
                        1.0 / unit["resistivity"] if unit["resistivity"] != 0 else 0.0
                    )
                except ValueError:
                    pass

            units.append(unit)

        return units

    @staticmethod
    def parse_config(file_path: str) -> Dict[str, object]:
        config: Dict[str, object] = {}

        for line in _read_valid_lines(file_path):
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
        lines = list(_read_valid_lines(file_path))

        index = 0
        while index < len(lines):
            name = lines[index]
            material_type = lines[index + 1]
            conductivity = float(lines[index + 2])
            heat_capacity = float(lines[index + 3])

            materials[name] = {
                "k": conductivity,
                "cp": heat_capacity,
                "fluid": material_type.lower() == "fluid",
            }

            if materials[name]["fluid"]:
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
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class Unit2D:
    """2D layout unit for FVM mesh generation.

    Cell types (from horizontal.csv):
        0 = SOLID (non-fluid)
        1 = FLUID (active fluid cell)
        2 = INLET (fluid cell with pressure BC)
        3 = OUTLET (fluid cell with pressure BC)
    """

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: Optional[str] = None
    k: Optional[float] = None
    cp: Optional[float] = None
    # Cell type for microchannel: 0=solid, 1=fluid, 2=inlet, 3=outlet
    cell_type: int = 0  # Default: solid


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


def load_stackup(config: Dict[str, Any], base_dir: str) -> List[Layer25D]:
    """Load 2.5D stackup model from config and layout files."""
    layers = []
    stackup_cfg = config.get("stackup", [])

    for i, layer_cfg in enumerate(stackup_cfg):
        tag = layer_cfg.get("tag", i + 100)
        name = layer_cfg.get("name", f"layer_{tag}")
        thickness = float(layer_cfg["thickness"])
        default_material = layer_cfg.get("material", "silicon")
        active = bool(layer_cfg.get("active", False))

        lx = float(layer_cfg.get("lx", 0.0))
        ly = float(layer_cfg.get("ly", 0.0))
        dx = float(layer_cfg.get("dx", 0.01))
        dy = float(layer_cfg.get("dy", 0.01))

        units = []
        layout_file = layer_cfg.get("layout_file")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    layout_data = json.load(f)
                    for u in layout_data:
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
                                cell_type=u.get("cell_type", 0),
                            )
                        )
            else:
                print(
                    f"[WARNING] Layout file {full_path} not found. Falling back to bulk layer."
                )

        # If no valid layout units, create a bulk unit
        if not units:
            units.append(
                Unit2D(
                    name=f"{name}_bulk",
                    lx=lx,
                    ly=ly,
                    dx=dx,
                    dy=dy,
                    material=default_material,
                    cell_type=0,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=thickness,
                default_material=default_material,
                active=active,
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

