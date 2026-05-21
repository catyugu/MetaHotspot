"""MetaHotspot XML 配置文件解析器。

解析 ThermalSim 导出的 XML 格式，生成 ModelConfig 等数据结构。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from metahotspot.metahotspot_types import (
    BlockGeometry,
    LayerConfig,
    MaterialModel,
    ModelConfig,
    Rect,
    ThermalBoundary,
)

# ============================================================================
# XML 命名空间
# ============================================================================

NS = "http://schemas.datacontract.org/2004/07/ThermalSim.Models"
NS_A = "http://schemas.microsoft.com/2003/10/Serialization/Arrays"
NS_B = "http://schemas.datacontract.org/2004/07/ThermalSim.Dialogs"
NS_BC = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.BoundaryConditions"
NS_I = "http://www.w3.org/2001/XMLSchema-instance"
NS_MESH = "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh"

# 简化命名空间前缀映射
NS_MAP = {
    "Structure": NS,
    "a": NS_A,
    "b": NS_B,
    "bc": NS_BC,
}


def _tag(local: str) -> str:
    """生成带命名空间的标签。"""
    return f"{{{NS}}}" + local


def _tag_a(local: str) -> str:
    """生成带命名空间的标签 (Arrays)。"""
    return f"{{{NS_A}}}" + local


def _tag_bc(local: str) -> str:
    """生成带命名空间的标签 (BoundaryConditions)。"""
    return f"{{{NS_BC}}}" + local


def _tag_mesh(local: str) -> str:
    """生成带命名空间的标签 (Mesh)。"""
    return f"{{{NS_MESH}}}" + local


def _find_text(elem: ET.Element, tag: str, default: str = "") -> str:
    """查找子元素文本，失败返回默认值。"""
    child = elem.find(_tag(tag))
    return child.text.strip() if child is not None and child.text else default


def _find_float(elem: ET.Element, tag: str, default: float = 0.0) -> float:
    """查找子元素并转为 float。"""
    text = _find_text(elem, tag)
    try:
        return float(text)
    except ValueError:
        return default


def _find_bool(elem: ET.Element, tag: str, default: bool = False) -> bool:
    """查找子元素并转为 bool。"""
    text = _find_text(elem, tag).lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return default


def _find_int(elem: ET.Element, tag: str, default: int = 0) -> int:
    """查找子元素并转为 int。"""
    text = _find_text(elem, tag)
    try:
        return int(text)
    except ValueError:
        return default


# ============================================================================
# 热边界条件解析
# ============================================================================


def parse_thermal_boundary(bc_elem: Optional[ET.Element]) -> dict:
    """解析热边界条件元素。

    返回:
        dict with keys:
        - "first":  {"temperature": float}  # 恒温边界温度 (K)
        - "second": {"heat_flux": float}     # 热流密度 (W/m²)
        - "third":  {"h_conv": float, "t_inf": float}  # 对流系数和环境温度 (K)
    """
    if bc_elem is None:
        return {"type": "second", "heat_flux": 0.0}

    bc_type = bc_elem.get(f"{{{NS_I}}}type", "")

    if "FirstType" in bc_type:
        temp_elem = bc_elem.find(_tag_bc("Temperature"))
        temp = (
            float(temp_elem.text) if temp_elem is not None and temp_elem.text else 0.0
        )
        return {"type": "first", "temperature": temp}

    elif "SecondType" in bc_type:
        flux_elem = bc_elem.find(_tag_bc("HeatFlux"))
        flux = (
            float(flux_elem.text) if flux_elem is not None and flux_elem.text else 0.0
        )
        return {"type": "second", "heat_flux": flux}

    elif "ThirdType" in bc_type:
        h_elem = bc_elem.find(_tag_bc("ConvectionCoefficient"))
        h_conv = float(h_elem.text) if h_elem is not None and h_elem.text else 0.0
        t_elem = bc_elem.find(_tag_bc("EnvironmentTemperature"))
        t_inf = float(t_elem.text) if t_elem is not None and t_elem.text else 0.0
        return {"type": "third", "h_conv": h_conv, "t_inf": t_inf}

    return {"type": "second", "heat_flux": 0.0}


# ============================================================================
# Rect 解析
# ============================================================================


def parse_rect(rect_elem: ET.Element) -> Rect:
    """解析 Rect 元素。"""
    name = _find_text(rect_elem, "Name")
    add_sub = _find_bool(rect_elem, "Add_sub", True)
    x_raw = _find_float(rect_elem, "XExpression")
    y_raw = _find_float(rect_elem, "YExpression")
    width_raw = _find_float(rect_elem, "WidthExpression")
    height_raw = _find_float(rect_elem, "HeightExpression")
    x_interval = _find_float(rect_elem, "XIntervalExpression")
    y_interval = _find_float(rect_elem, "YIntervalExpression")

    return Rect(
        name=name,
        add_sub=add_sub,
        x=x_raw,
        y=y_raw,
        width=width_raw,
        height=height_raw,
        x_interval=x_interval,
        y_interval=y_interval,
    )


# ============================================================================
# BlockGeometry 解析
# ============================================================================


def parse_block(block_elem: ET.Element) -> BlockGeometry:
    """解析 Block 元素。"""
    name = _find_text(block_elem, "Name")
    material_name = _find_text(block_elem, "MaterialName")
    thickness = _find_float(block_elem, "ThicknessExpression")
    x_offset = _find_float(block_elem, "XOffsetExpression")
    y_offset = _find_float(block_elem, "YOffsetExpression")
    z_offset = _find_float(block_elem, "ZOffsetExpression")
    heat_source = _find_float(block_elem, "TiReyuan")

    # 解析 AllRects
    rects: List[Rect] = []
    all_rects_elem = block_elem.find(_tag("AllRects"))
    if all_rects_elem is not None:
        for rect_child in all_rects_elem:
            if rect_child.tag == _tag("Rect"):
                rects.append(parse_rect(rect_child))

    return BlockGeometry(
        name=name,
        material_name=material_name,
        thickness=thickness,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        heat_source=heat_source,
        rects=rects,
    )


# ============================================================================
# LayerConfig 解析
# ============================================================================


def parse_layer(layer_elem: ET.Element) -> LayerConfig:
    """解析 Layer 元素。"""
    name = _find_text(layer_elem, "Name")
    thickness_raw = _find_float(layer_elem, "ThicknessExpression")
    mesh_size_x = _find_float(layer_elem, "MeshSizeXExpression")
    mesh_size_y = _find_float(layer_elem, "MeshSizeYExpression")
    mesh_size_z = _find_float(layer_elem, "MeshSizeZExpression")
    is_top = _find_bool(layer_elem, "IsTopLayer", False)
    is_die = _find_bool(layer_elem, "IsDie", False)
    is_tim = _find_bool(layer_elem, "IsTIM", False)
    is_substrate = _find_bool(layer_elem, "IsSubstrate", False)

    # 解析 Blocks
    blocks: List[BlockGeometry] = []
    blocks_elem = layer_elem.find(_tag("Blocks"))
    if blocks_elem is not None:
        for block_child in blocks_elem:
            if block_child.tag == _tag("Block"):
                blocks.append(parse_block(block_child))

    return LayerConfig(
        name=name,
        thickness=thickness_raw,
        mesh_size_x=mesh_size_x,
        mesh_size_y=mesh_size_y,
        mesh_size_z=mesh_size_z,
        is_top_layer=is_top,
        is_die=is_die,
        is_tim=is_tim,
        is_substrate=is_substrate,
        blocks=blocks,
    )


# ============================================================================
# MaterialModel 解析
# ============================================================================


def parse_material(key: str, value_elem: ET.Element) -> MaterialModel:
    """解析 Material 元素。

    参数:
        key: 材料名称 (来自 Key 元素)
        value_elem: Value 子元素 (包含材料属性)
    """
    k = _find_float(value_elem, "DaoreXishu")
    density = _find_float(value_elem, "Midu")
    cp_elem = value_elem.find(_tag("BiRerong"))
    cp = float(cp_elem.text) if cp_elem is not None and cp_elem.text else 0.0

    # 处理 nil 值
    if cp_elem is not None and cp_elem.get(f"{{{NS_A}}}nil") == "true":
        cp = 0.0

    return MaterialModel(
        name=key,
        k=k,
        cp=cp,
        density=density,
    )


# ============================================================================
# ThermalBoundary 解析
# ============================================================================


def parse_boundary(boundary_elem: ET.Element) -> ThermalBoundary:
    """解析 Boundary 元素。"""
    name = _find_text(boundary_elem, "Name")

    # 解析 FaceKeys
    face_keys_elem = boundary_elem.find(_tag("FaceKeys"))
    if face_keys_elem is None:
        raise ValueError(f"Boundary {name} has no FaceKeys")

    face_key_strs: List[str] = []
    for key_child in face_keys_elem:
        if key_child.tag == _tag_a("string") and key_child.text:
            face_key_strs.append(key_child.text)

    if not face_key_strs:
        raise ValueError(f"Boundary {name} has no FaceKeys strings")

    # 获取热边界条件
    bc_elem = boundary_elem.find(_tag("ThermalBoundary"))
    bc = parse_thermal_boundary(bc_elem)

    return ThermalBoundary(
        name=name,
        boundary_type=bc["type"],
        face_keys=face_key_strs,
        params=bc,
    )


# ============================================================================
# 网格坐标解析 (来自 Results/Mesh)
# ============================================================================


@dataclass(slots=True)
class MeshCoordinates:
    """网格坐标数据。"""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def parse_mesh(mesh_elem: ET.Element) -> MeshCoordinates:
    """解析 Mesh 元素。"""
    x_arr: List[float] = []
    y_arr: List[float] = []
    z_arr: List[float] = []

    # Try Mesh namespace first, fall back to base namespace
    x_elem = mesh_elem.find(_tag_mesh("XArray"))
    if x_elem is None:
        x_elem = mesh_elem.find(_tag("XArray"))
    if x_elem is not None:
        for child in x_elem:
            if child.tag == _tag_a("double") and child.text:
                x_arr.append(float(child.text))

    y_elem = mesh_elem.find(_tag_mesh("YArray"))
    if y_elem is None:
        y_elem = mesh_elem.find(_tag("YArray"))
    if y_elem is not None:
        for child in y_elem:
            if child.tag == _tag_a("double") and child.text:
                y_arr.append(float(child.text))

    z_elem = mesh_elem.find(_tag_mesh("ZArray"))
    if z_elem is None:
        z_elem = mesh_elem.find(_tag("ZArray"))
    if z_elem is not None:
        for child in z_elem:
            if child.tag == _tag_a("double") and child.text:
                z_arr.append(float(child.text))

    return MeshCoordinates(
        x=np.array(x_arr, dtype=np.float64),
        y=np.array(y_arr, dtype=np.float64),
        z=np.array(z_arr, dtype=np.float64),
    )


# ============================================================================
# 主解析函数
# ============================================================================


def parse_xml(xml_path: str | Path) -> Tuple[ModelConfig, MeshCoordinates]:
    """解析 XML 配置文件。

    参数:
        xml_path: XML 文件路径

    返回:
        Tuple[ModelConfig, MeshCoordinates]: 模型配置和网格坐标
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    if root.tag != _tag("Structure"):
        raise ValueError(f"Root element must be Structure, got {root.tag}")

    # 全局配置
    study_type = _find_text(root, "StudyType", "Steady")
    ambient_temp = _find_float(root, "AmbientTemperature", 300.0)
    initial_temp = _find_float(root, "InitialTemperature", 300.0)
    length_unit = _find_text(root, "LengthUnit", "Mm")

    # 瞬态配置
    transient_duration = _find_float(root, "TransientStudyDuration", 0.0)
    transient_timestep = _find_float(root, "TransientStudyTimeStep", 0.0)

    # 解析 Layers
    layers: List[LayerConfig] = []
    layers_elem = root.find(_tag("Layers"))
    if layers_elem is not None:
        for layer_child in layers_elem:
            if layer_child.tag == _tag("Layer"):
                layers.append(parse_layer(layer_child))

    # 解析 Materials
    materials: Dict[str, MaterialModel] = {}
    materials_elem = root.find(_tag("Materials"))
    if materials_elem is not None:
        for kv_elem in materials_elem:
            if "KeyValueOf" in kv_elem.tag:
                key_elem = kv_elem.find(_tag_a("Key"))
                value_elem = kv_elem.find(_tag_a("Value"))
                if key_elem is not None and key_elem.text and value_elem is not None:
                    mat = parse_material(key_elem.text, value_elem)
                    materials[mat.name] = mat

    # 解析 Boundaries
    boundaries: List[ThermalBoundary] = []
    boundaries_elem = root.find(_tag("Boundaries"))
    if boundaries_elem is not None:
        for bc_child in boundaries_elem:
            if bc_child.tag == _tag("Boundary"):
                boundaries.append(parse_boundary(bc_child))

    # 解析 Mesh 坐标 (从 Results/anyType/Result3D/Mesh 中获取)
    mesh_coords = MeshCoordinates(
        x=np.array([], dtype=np.float64),
        y=np.array([], dtype=np.float64),
        z=np.array([], dtype=np.float64),
    )
    results_elem = root.find(_tag("Results"))
    if results_elem is not None:
        any_type_elem = results_elem.find(_tag_a("anyType"))
        if any_type_elem is not None:
            mesh_elem = any_type_elem.find(_tag("Mesh"))
            if mesh_elem is not None:
                mesh_coords = parse_mesh(mesh_elem)

    model_config = ModelConfig(
        study_type=study_type,
        ambient_temperature=ambient_temp,
        initial_temperature=initial_temp,
        length_unit=length_unit,
        transient_duration=transient_duration,
        transient_timestep=transient_timestep,
        layers=layers,
        materials=materials,
        boundaries=boundaries,
        mesh_coords=mesh_coords
    )
    return model_config
