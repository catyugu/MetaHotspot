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
