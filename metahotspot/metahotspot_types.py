"""MetaHotspot data types.

All data classes use @dataclass(slots=True) and SoA (Structure of Arrays) design.
 ndarray dimensions are annotated in comments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(slots=True)
class Rect:
    """矩形操作基元，定义块中的加/减操作区域。

    Add_sub=True 表示添加区域，Add_sub=False 表示减去区域。
    """
    name: str
    add_sub: bool  # True=加操作, False=减操作
    x: float       # 左下角 X 坐标
    y: float       # 左下角 Y 坐标
    width: float   # 矩形宽度
    height: float  # 矩形高度
    x_interval: float = 0.0  # X 方向阵列间隔
    y_interval: float = 0.0  # Y 方向阵列间隔


@dataclass(slots=True)
class BlockGeometry:
    """块几何信息，包含多个矩形操作。

    一个块由多个 Rect 组成（加/减操作），通过 TiReyuan 定义体热源。
    """
    name: str
    material_name: str
    thickness: float           # 厚度 (m)
    x_offset: float = 0.0      # X 偏移
    y_offset: float = 0.0      # Y 偏移
    z_offset: float = 0.0      # Z 偏移
    heat_source: float = 0.0   # 体热源密度 TiReyuan (W/m³)
    rects: List[Rect] = field(default_factory=list)


@dataclass(slots=True)
class LayerConfig:
    """层配置信息，对应 XML 中的 Layer 元素。

    包含层的几何信息和其中的块。
    """
    name: str
    thickness: float           # 厚度 (m)
    mesh_size_x: float = 0.0   # X 方向网格尺寸，0=自动
    mesh_size_y: float = 0.0   # Y 方向网格尺寸，0=自动
    mesh_size_z: float = 0.0   # Z 方向网格尺寸，0=自动
    is_top_layer: bool = False
    is_die: bool = False
    is_tim: bool = False
    is_substrate: bool = False
    blocks: List[BlockGeometry] = field(default_factory=list)


@dataclass(slots=True)
class ThermalBoundary:
    """热边界条件定义，支持三类边界条件。

    type: "first"  = 恒温边界 (Dirichlet)
          "second" = 热流边界 (Neumann, HeatFlux)
          "third"  = 对流边界 (Convection)

    params: dict 包含边界参数:
        - "first":  {"temperature": float}  # 恒温边界温度 (K)
        - "second": {"heat_flux": float}     # 热流密度 (W/m²)
        - "third":  {"h_conv": float, "t_inf": float}  # 对流系数和环境温度

    face_keys: List[str] 原始 FaceKey 字符串列表，用于定义边界几何
    """
    name: str
    boundary_type: str          # "first", "second", "third"
    face_keys: List[str] = field(default_factory=list)  # FaceKey 字符串列表
    params: dict = field(default_factory=dict)  # 边界条件参数


@dataclass(slots=True)
class MaterialModel:
    """材料热物性模型。

    用于稳态和瞬态热分析。
    """
    name: str
    k: float          # 热导率 (W/(m·K))
    cp: float         # 比热容 (J/(kg·K))
    density: float    # 密度 (kg/m³)


@dataclass(slots=True)
class MeshCoordinates:
    """网格坐标数组 (SoA format)。

    用于存储 X, Y, Z 三个方向的网格坐标。
    """
    x: np.ndarray  # shape: (nx,) - X 方向网格点坐标
    y: np.ndarray  # shape: (ny,) - Y 方向网格点坐标
    z: np.ndarray  # shape: (nz,) - Z 方向网格点坐标

@dataclass(slots=True)
class ModelConfig:
    """模型全局配置，对应 XML 中的 Structure 根元素。

    包含仿真类型、初始条件、环境参数等。
    """
    study_type: str                # "Steady" 或 "Transient"
    ambient_temperature: float     # 环境温度 (K)
    initial_temperature: float      # 初始温度 (K)
    length_unit: str = "Mm"         # 长度单位，默认毫米
    transient_duration: float = 0.0  # 瞬态总时长 (s)
    transient_timestep: float = 0.0  # 瞬态时间步长 (s)

    layers: List[LayerConfig] = field(default_factory=list)
    materials: Dict[str, MaterialModel] = field(default_factory=dict)
    boundaries: List[ThermalBoundary] = field(default_factory=list)
    mesh_coords: MeshCoordinates | None = None

# ============================================================================
# SoA 格式数据结构（用于计算内核）
# ============================================================================

@dataclass(slots=True)
class CellProperties:
    """单元格物性数据 (SoA format)。

    所有数组按单元格索引排列。
    """
    k: np.ndarray          # shape: (n_cells,) - 热导率
    cp: np.ndarray         # shape: (n_cells,) - 比热容
    density: np.ndarray    # shape: (n_cells,) - 密度
    heat_source: np.ndarray  # shape: (n_cells,) - 体热源密度 (W/m³)

@dataclass(slots=True)
class CellGeometry:
    """单元格几何信息 (SoA format)。
    """
    centers: np.ndarray    # shape: (n_cells, 3) - 单元格中心坐标 (x,y,z)
    volumes: np.ndarray    # shape: (n_cells,) - 单元格体积
    layer_ids: np.ndarray  # shape: (n_cells,) - 所属层 ID
    block_ids: np.ndarray  # shape: (n_cells,) - 所属块 ID

@dataclass(slots=True)
class BoundaryCondition:
    """边界条件数据结构 (SoA format)。

    存储边界条件的索引和数值。
    """
    # shape 维度说明:
    # c_ids: (n_boundary_cells,) - 边界条件对应的单元格索引
    # areas: (n_boundary_cells,) - 边界面面积
    # params: Dict[string, float] - 参数
    c_ids: np.ndarray
    areas: np.ndarray
    params: Dict[str, float]


@dataclass(slots=True)
class MeshTopology:
    """网格拓扑信息 (SoA format)。

    整合网格几何和连接关系。
    """
    n_cells: int
    n_x: int                # X 方向网格点数
    n_y: int                # Y 方向网格点数
    n_z: int                # Z 方向网格点数

    coords: MeshCoordinates
    cell_geom: CellGeometry

    # 层级映射
    layer_names: List[str]  # layer_id -> layer_name
    material_names: List[str]  # material_id -> material_name


@dataclass(slots=True)
class PhysicalFields:
    """物理场数据 (SoA format)。

    存储温度等场量。
    """
    k: np.ndarray            # shape: (n_cells,) - 热导率
    cp: np.ndarray           # shape: (n_cells,) - 比热容
    density: np.ndarray      # shape: (n_cells,) - 密度
    heat_source: np.ndarray  # shape: (n_cells,) - 体热源密度 (W/m³)
    temperature: np.ndarray   # shape: (n_cells,) - 温度 (K)

@dataclass(slots=True)
class SolverConfig:
    """求解器配置。
    """
    study_type: str           # "Steady" 或 "Transient"
    init_temperature: float   # 初始温度 (K)
    ambient_temperature: float # 环境温度 (K)
    timestep: float = 0.0     # 时间步长 (s)
    transient_duration: float = 0.0  # 瞬态总时长 (s)

    tolerance_abs: float = 1e-6   # 绝对容差
    tolerance_rel: float = 1e-6   # 相对容差


@dataclass(slots=True)
class SystemMatrix:
    """系统矩阵和向量 (SoA format)。

    存储线性方程组 Ax = b 的系数矩阵和右端项。
    """
    # A: 稀疏矩阵 (CSR format)
    n_rows: int
    n_cols: int
    row_ptr: np.ndarray   # shape: (n_rows + 1,)
    col_idx: np.ndarray   # shape: (n_nz,) - 非零元素列索引
    data: np.ndarray      # shape: (n_nz,) - 非零元素值

    # b: 右端项
    b: np.ndarray         # shape: (n_rows,)

    # 功率矩阵 (体热源)
    power_row_ptr: np.ndarray
    power_col_idx: np.ndarray
    power_data: np.ndarray


@dataclass(slots=True)
class SimulationResult:
    """仿真结果数据。
    """
    temperatures: np.ndarray  # shape: (n_cells,) - 最终温度
    time_series: np.ndarray   # shape: (n_steps, n_cells) - 瞬态温度时间序列
    max_temperature: float
    min_temperature: float
    max_index: int
    min_index: int