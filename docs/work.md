引入 `numba.njit` 是将纯 Python 数值计算向高性能（C/C++ 级别）迈进的绝佳选择，非常符合你最终迁移到 C++20 的长期目标。你的 `MetaHotspot` 已经做得非常好的一点是**采用了 SoA（Structure of Arrays）模式**（如 `MeshTopology` 和 `PhysicalFields` 中的扁平化 `np.ndarray`）。Numba 最喜欢的正是这种内存连续的 NumPy 数组。

为了确保代码符合工业规范、保持简洁优雅，并顺利接入 Numba，你需要遵循 **“计算与状态分离”（Data-Oriented Design）** 的架构原则。

以下是针对 `MetaHotspot` 引入 Numba 的架构指导与重构方案：

### 一、 核心架构规范：解耦 OOP 与计算内核

Numba 的 `nopython` 模式（`@njit`）对 Python 的面向对象特性、动态数据结构（`list.append`, `dict`）支持很差。在工业界，标准的做法是**将类（Class）作为数据的管理者，将 Numba 纯函数作为计算核心（Kernels）**。

建议新建专门的 kernel 模块（例如 `assembler_kernels.py`, `topology_kernels.py`），将繁重的循环提取出来。

**重构前（目前的代码）：**
```python
class FVMAssembler:
    def _build_conduction_matrix(self):
        # 混杂了 self 访问、生成器 yield 和 Python list
        rows, cols, data = [], [], []
        for c_a, c_b, axis, area in self._find_adjacent_pairs():
            ...
            rows.extend([c_a, c_b, c_a, c_b])
```

**重构后（工业化 SoA + JIT 风格）：**
```python
# assembler_kernels.py
import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def build_conduction_coo(n_cells: int, boxes: np.ndarray, is_fluid: np.ndarray, dims: np.ndarray, k: np.ndarray):
    # 纯数据输入，返回扁平数组
    ...
    return rows[:ptr], cols[:ptr], data[:ptr]

# assembler.py
class FVMAssembler:
    def _build_conduction_matrix(self):
        # OOP 层只负责数据解包和组装
        rows, cols, data = build_conduction_coo(
            self.topo.n_cells, self.topo.boxes, 
            self.fields.is_fluid, self.topo.dims, self.fields.k
        )
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))
```

---

### 二、 关键性能瓶颈改造指南

#### 1. 组装阶段（Assembler）：消灭动态 List 与 Generator
`assembler.py` 中的 `_find_adjacent_pairs` 使用了 `yield` 和动态的 `active_list`。在 Numba 中，动态分配内存代价高昂。

**SoA 优雅写法：预分配上限策略**
对于六面体网格，一个面最多对应有限个邻居（通常内部最多 6 个面）。我们可以预先分配一个足够大的 NumPy 数组，用指针记录实际写入数量，最后截断。这在 C++ 也是极佳的实践。

```python
@njit(cache=True, fastmath=True)
def compute_conduction_coo_kernel(n_cells, boxes, is_fluid, dims, k, tol=1e-15):
    # 预分配 COO 数组（每个 cell 最多 6 个相邻面，每个面产生 4 个矩阵元素）
    max_elements = n_cells * 6 * 4  
    rows = np.empty(max_elements, dtype=np.int32)
    cols = np.empty(max_elements, dtype=np.int32)
    data = np.empty(max_elements, dtype=np.float64)
    
    ptr = 0
    # 注意：Numba 会将这里的循环编译为等效的 C 循环，效率极高
    for c_a in range(n_cells):
        for c_b in range(c_a + 1, n_cells):
            # 用内联逻辑替代对象方法，实现包围盒快速排斥校验
            if boxes[c_a, 1] > boxes[c_b, 4] + tol or ... :
                continue
                
            # 计算重叠、热阻...
            # 记录数据
            rows[ptr] = c_a; cols[ptr] = c_a; data[ptr] = g; ptr += 1
            rows[ptr] = c_b; cols[ptr] = c_b; data[ptr] = g; ptr += 1
            rows[ptr] = c_a; cols[ptr] = c_b; data[ptr] = -g; ptr += 1
            rows[ptr] = c_b; cols[ptr] = c_a; data[ptr] = -g; ptr += 1

    return rows[:ptr], cols[:ptr], data[:ptr]
```

#### 2. 网格预处理（Mesh Preprocessor）：Hash 表的替代方案
`mesh_preprocessor.py` 中的 `_build_topology` 用 Python 的 `dict`（`face_to_cells`）来匹配内部面。Numba 无法高效处理元组作为 key 的字典。

**优雅的数组化（Array-based）算法：“排序与扫描”**
将面转化为唯一的 ID 序列，然后排序。相邻的相同面即为内部面（Internal Faces）。这也是并行化（如以后使用 CUDA/C++ `std::sort`）的标准做法。

```python
@njit(cache=True)
def build_topology_kernel(hex_data):
    n_cells = hex_data.shape[0]
    faces_per_cell = 6
    nodes_per_face = 4
    
    # 扁平化存储所有面: [总面数, 4个节点ID]
    total_faces = n_cells * faces_per_cell
    face_nodes = np.empty((total_faces, nodes_per_face), dtype=np.int32)
    face_cell_ids = np.empty(total_faces, dtype=np.int32)
    
    # 填充面数据，并将节点 ID 排序以便后续整体比较
    for i in range(n_cells):
        nodes = hex_data[i]
        # ... 构建 faces 数组，注意对每个 face 内部的节点 id 从小到大排序
        # 记录属于哪个 cell
    
    # 利用 Numba 支持的 argsort/Lexsort 找到重复面
    # 重复 2 次的归为 internal_faces，出现 1 次的归为 boundary_faces
    ...
```

---

### 三、 Numba 使用最佳实践与装饰器规范

为了代码库统一管理，建议统一定义一套 Numba 装饰器配置。

1. **类型一致性 (Type Stability)**
   确保传入 Numba 的 NumPy 数组类型高度一致。比如不要混用 `float32` 和 `float64`。建议在 `model25d.py` 或者顶层定义类型别名常量：
   `FLOAT_DTYPE = np.float64`
   `INT_DTYPE = np.int32`
   
2. **强制编译选项**
   永远使用 `njit`（严格拒绝对象模式），并开启 `cache` 和释放 GIL，这样以后可以很方便地换成 `@njit(parallel=True)` 并用 `prange` 进行多线程加速。

```python
# 建议在 utils 或者 config 中统一定义
import numba

# 工业标准装饰器配置
jit_kernel = numba.njit(
    cache=True,         # AOT 缓存，避免每次启动都重新编译，对 CLI 工具至关重要
    fastmath=True,      # 允许浮点运算重排，对热力学非病态矩阵无影响，能加速 20%
    nogil=True          # 释放 Python 全局解释器锁，为以后并行化铺路
)

@jit_kernel
def some_heavy_math_loop(k_array, cp_array):
    ...
```

### 四、 演进路线建议

1. **第一步（微创手术）：** 不要大改现有类的结构，只把 `_calc_resistance`, `_compute_nusselt`, `_overlap_area` 提取为文件级别的 `@njit` 函数。
2. **第二步（结构重组）：** 把整个 `A_cond` 和 `A_adv` 的构建过程转入 Kernel，彻底告别 Python 的 for 循环。此时你应该能看到数量级的性能提升。