### 一、 核心架构设计

将整个流程解耦为 4 个独立的核心模块，并通过纯数据类（Data Classes）进行信息传递：

1.  **数据层 (Data Containers)**：严格保持 SoA（Structure of Arrays）风格。这是未来 C++ 和 Python 之间零拷贝（Zero-copy）交互的基石。
2.  **网格与属性预处理器 (Preprocessor)**：负责读取网格、构建空间索引、提取拓扑（面、体积）、并映射物理属性（$k, c_p, \rho$）。
3.  **方程组装器 (Assembler)**：根据拓扑和属性，计算热导率、对流、平流等，最终生成标准的稀疏矩阵 $A$ 和右端项 $b$。
4.  **数值求解器 (Numerical Solver)**：完全不知道“网格”和“材料”的存在，只负责解算纯粹的线性代数方程组 $A x = b$（稳态）或执行时间积分（瞬态）。

---

### 二、 数据传输对象 (SoA 风格)

使用 `@dataclass` 隔离状态，摒弃将其挂载在 `self` 上的做法。这避免了状态突变，也明确了 C++ 需要返回的数据格式。

```python
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

@dataclass(slots=True)
class MeshTopology:
    """纯几何与拓扑数据 (SoA布局)"""
    n_cells: int
    centers: np.ndarray      # shape (N, 3)
    dims: np.ndarray         # shape (N, 3)
    boxes: np.ndarray        # shape (N, 6) [xmin, ymin, zmin, xmax, ymax, zmax]
    volumes: np.ndarray      # shape (N,)
    
    # 拓扑连接
    internal_faces: list[tuple[int, int]] 
    boundary_faces: dict[str, list[tuple[int, np.ndarray, float]]] # face_dir -> (cell_id, normal, area)

@dataclass(slots=True)
class PhysicalFields:
    """物理属性与状态场 (SoA布局)"""
    k: np.ndarray
    cp: np.ndarray
    density: np.ndarray
    is_fluid: np.ndarray
    
    # 流体相关
    dynamic_viscosity: np.ndarray
    hydroC: np.ndarray
    pressure: np.ndarray
    inlet_temperature: np.ndarray

@dataclass(slots=True)
class SystemMatrices:
    """组装好的代数方程组 A * T = b"""
    A_total: sp.csr_matrix
    b_total: np.ndarray
    power_matrix: sp.csr_matrix # shape (N, n_units)
    unit_names: list[str]
```

---

### 三、 模块化拆分与降维重构

保持代码层级不超过三层，多用 numpy 向量化掩码（Masking）替代深层 `for` 循环。

#### 1. 预处理器 (Preprocessor)
只负责“准备数据”。未来这一块非常适合用 C++ 重写（尤其是 Morton 排序和非共形网格的 Bounding Box 相交检测）。

```python
class MeshPreprocessor:
    def __init__(self, config: dict, stackup: list):
        self.config = config
        self.stackup = stackup

    def process(self, mesh_path: str) -> tuple[MeshTopology, PhysicalFields]:
        # 1. 提取网格几何特征 (降低原代码的嵌套深度)
        centers, dims, boxes, vols, sorted_indices = self._extract_geometry(mesh_path)
        
        # 2. 构建拓扑 (提取面)
        internal_faces, boundary_faces = self._build_topology()
        
        topo = MeshTopology(
            n_cells=len(centers), centers=centers, dims=dims, 
            boxes=boxes, volumes=vols, internal_faces=internal_faces, 
            boundary_faces=boundary_faces
        )
        
        # 3. 属性映射 (建议将原先的逐个cell循环，改为基于 bounding box 的 numpy 掩码批量赋值)
        fields = self._map_physical_properties(topo)
        
        return topo, fields

    def _map_physical_properties(self, topo: MeshTopology) -> PhysicalFields:
        # TODO: 使用 numpy 向量化重写 _find_cell_props，避免 O(N) 级别的 for 循环
        pass
```

#### 2. 矩阵组装器 (Assembler)
负责物理方程的离散化。它接收 `MeshTopology` 和 `PhysicalFields`，输出纯数学的 `SystemMatrices`。

```python
class FVMAssembler:
    def __init__(self, topo: MeshTopology, fields: PhysicalFields, config: dict):
        self.topo = topo
        self.fields = fields
        self.config = config

    def assemble(self) -> SystemMatrices:
        n = self.topo.n_cells
        
        # 1. 流场求解 (原 _solve_pressure_field)
        self._solve_fluid_pressure()
        
        # 2. 组装传导矩阵 (原 _assemble_conduction_matrix)
        A_cond = self._build_conduction_matrix()
        
        # 3. 组装边界与对流 (原 _build_boundary_terms)
        A_bc, b_bc = self._build_boundary_terms()
        
        # 4. 组装平流矩阵 (原 _assemble_advection_matrix)
        A_adv, b_adv = self._build_advection_matrix()
        
        # 5. 组装热源映射矩阵 (原 _precompute_power_matrix)
        power_mat, unit_names = self._build_power_matrix()

        # 合并矩阵
        A_total = A_cond + A_bc + A_adv
        b_total = b_bc + b_adv

        return SystemMatrices(A_total, b_total, power_mat, unit_names)
    
    def _build_conduction_matrix(self) -> sp.csr_matrix:
        # 保持单层或双层循环，提取内部逻辑为独立函数
        pass
```

#### 3. 数值求解器 (Equation Solver)
求解器应该变得非常“薄”，它不需要知道任何网格或材料知识。

```python
import scipy.sparse.linalg as splinalg
import numpy as np

class ThermalSolver:
    def __init__(self, matrices: SystemMatrices, config: dict):
        self.mat = matrices
        self.config = config

    def solve_steady(self, mean_powers: np.ndarray) -> np.ndarray:
        """纯粹的线性代数求解: A * T = b + P"""
        rhs = self.mat.b_total + (self.mat.power_matrix @ mean_powers)
        # 注意：原代码是 -g_total，这里假设 A_total 的符号已处理正确
        temperature = splinalg.spsolve(-self.mat.A_total, rhs) 
        return temperature

    def solve_transient(self, dt: float, ptrace: list[dict], init_temp: np.ndarray, vols: np.ndarray, cp: np.ndarray) -> np.ndarray:
        """纯粹的向后欧拉时间积分"""
        c_mat = sp.diags(cp * vols) / dt
        solve_step = splinalg.factorized((c_mat - self.mat.A_total).tocsc())
        
        temperature = init_temp.copy()
        
        for i, step_power in enumerate(ptrace):
            power_vec = self._extract_power_vec(step_power)
            rhs = (c_mat @ temperature) + self.mat.b_total + (self.mat.power_matrix @ power_vec)
            temperature = solve_step(rhs)
            
        return temperature
```

---

### 四、 顶层控制器 (Pipeline Orchestrator)

最后，原先的 `solver.py` 脚本或 `FVMSolver` 类的主干，将变成一个清晰的 Pipeline 调度器。

```python
class MetaHotspotSolver:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.stackup = load_stackup(self.config, os.path.dirname(config_path))
        self.mesh_path = os.path.join(os.path.dirname(config_path), self.config["mesh_file_path"])

    def run(self):
        print("[INFO] Preprocessing mesh and properties...")
        preprocessor = MeshPreprocessor(self.config, self.stackup)
        topo, fields = preprocessor.process(self.mesh_path)

        print("[INFO] Assembling system matrices...")
        assembler = FVMAssembler(topo, fields, self.config)
        matrices = assembler.assemble()

        print("[INFO] Solving equations...")
        solver = ThermalSolver(matrices, self.config)
        
        if self.config["simulation_type"] == "steady":
            # 伪代码：解析 mean_powers
            mean_powers = self._get_mean_powers(matrices.unit_names) 
            temperatures = solver.solve_steady(mean_powers)
        else:
            # 伪代码：瞬态逻辑
            temperatures = solver.solve_transient(...)

        print("[INFO] Exporting results...")
        self._export_vtu(topo, temperatures, "result.vtu")
```