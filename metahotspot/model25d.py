import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Unit2D:
    """2D layout unit for FVM mesh generation."""

    name: str
    lx: float
    ly: float
    dx: float
    dy: float
    material: Optional[str] = None
    k: Optional[float] = None
    cp: Optional[float] = None
    is_fluid: bool = False


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

    for i, layer_cfg in enumerate(config.get("stackup", [])):
        tag = layer_cfg.get("tag", i + 100)
        name = layer_cfg.get("name", f"layer_{tag}")
        lx, ly = float(layer_cfg.get("lx", 0.0)), float(layer_cfg.get("ly", 0.0))
        dx, dy = float(layer_cfg.get("dx", 0.01)), float(layer_cfg.get("dy", 0.01))

        units = []
        layout_file = layer_cfg.get("layout_file")

        if layout_file and layout_file.lower() not in {"none", "(null)", ""}:
            full_path = os.path.join(base_dir, layout_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for u in json.load(f):
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
                                is_fluid=bool(u.get("is_fluid", False)),
                            )
                        )

        if not units:
            units.append(
                Unit2D(
                    name=f"{name}_bulk",
                    lx=lx,
                    ly=ly,
                    dx=dx,
                    dy=dy,
                    material=layer_cfg.get("material", "silicon"),
                    is_fluid=False,
                )
            )

        layers.append(
            Layer25D(
                name=name,
                tag=tag,
                thickness=float(layer_cfg["thickness"]),
                default_material=layer_cfg.get("material", "silicon"),
                active=bool(layer_cfg.get("active", False)),
                units=units,
                lx=lx,
                ly=ly,
                dx=dx,
                dy=dy,
            )
        )

    return layers
