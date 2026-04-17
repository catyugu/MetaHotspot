import os
import re
from csv import reader
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

                units.append(
                    {
                        "name": parts[0],
                        "width": float(parts[1]),
                        "height": float(parts[2]),
                        "left_x": float(parts[3]),
                        "bottom_y": float(parts[4]),
                    }
                )

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

    @staticmethod
    def parse_csv_grid(file_path: str) -> List[List[int]]:
        grid: List[List[int]] = []
        if not os.path.exists(file_path):
            return grid

        with open(file_path, "r", encoding="utf-8") as handle:
            csv_reader = reader(handle)
            for row in csv_reader:
                cleaned = [cell.strip() for cell in row if cell.strip() != ""]
                if not cleaned:
                    continue
                try:
                    grid.append([int(cell) for cell in cleaned])
                except ValueError:
                    continue

        return grid
