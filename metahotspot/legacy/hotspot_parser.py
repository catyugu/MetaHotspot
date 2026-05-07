import os
import re
from typing import Dict, Generator, List, Any


def _read_valid_lines(file_path: str) -> Generator[str, None, None]:
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

            if len(parts) >= 7:
                try:
                    unit["specific_heat"] = float(parts[5])
                    resistivity = float(parts[6])
                    unit["k"] = 1.0 / resistivity if resistivity != 0 else 0.0
                except ValueError:
                    pass
            units.append(unit)
        return units

    @staticmethod
    def parse_config(file_path: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for line in _read_valid_lines(file_path):
            match = re.match(r"^-(\w+)\s+([^#]+)", line)
            if match:
                key, value = match.groups()
                try:
                    config[key] = float(value.strip())
                except ValueError:
                    config[key] = value.strip()
        return config

    @staticmethod
    def parse_materials(file_path: str) -> Dict[str, dict]:
        materials: Dict[str, dict] = {}
        lines = list(_read_valid_lines(file_path))
        index = 0
        while index < len(lines):
            name = lines[index]
            is_fluid = lines[index + 1].lower() == "fluid"
            materials[name] = {
                "k": float(lines[index + 2]),
                "cp": float(lines[index + 3]),
                "fluid": is_fluid,
            }
            if is_fluid:
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
