# 工作指南

为了平滑过渡到 C++，你需要对当前的 `FVMSolver` 进一步解耦。以下是详细的架构指引：

### 1. 架构解耦：职责分离 (Separation of Concerns)

目前的 `FVMSolver` 承担了太多职责（读配置、读网格、计算拓扑、物理量分配、矩阵组装、调用求解器、写 VTU）。你需要将其拆分为四个核心模块，形成流水线（Pipeline）：

*   **`Context` (数据总线/状态机)**：只负责持有 SoA 数据（即那些 `np.ndarray`）。
*   **`Preprocessor` (预处理器)**：负责读取 Mesh，计算中心点，提取面拓扑，执行 Morton 排序，并根据 Config 将物理量映射到单元上。
*   **`Assembler` (组装器)**：根据拓扑关系和物理场数据，计算系数（导热、对流、热源），并输出标准的 CSR 稀疏矩阵的三元组（rows, cols, data）。
*   **`Solver` (求解引擎)**：接收 CSR 矩阵和 RHS（右端项）向量，执行求解，并更新 `Context` 中的温度场。

### 2. 给 Python 代码的重构指引 (Step-by-Step)s

**步骤一：把拓扑提取剥离出去**
将 `_extract_faces` 和 `_classify_boundary_face` 移出 Solver。这些纯几何操作不需要知道任何物理属性（k, cp, T），只依赖网格节点坐标。将来这部分可以直接用 C++ 重写，利用多线程或更高效的哈希表提取面。

**步骤二：将组装逻辑改为纯函数 (Pure Functions)**
目前的 `_assemble_conduction_matrix` 依赖 `self` 里的很多状态。将其重构为一个接收纯数组的静态函数或独立类的函数：
```python
# 现在的写法：高度耦合 self
def _assemble_conduction_matrix(self) -> sp.csr_matrix: ...

# 未来的写法（为 C++ 绑定做准备）：
@staticmethod
def assemble_conduction(
    c_box: np.ndarray, c_k: np.ndarray, c_dims: np.ndarray, 
    c_is_fluid: np.ndarray, n_cells: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (rows, cols, data)，外层再去调用 sp.csr_matrix"""
    # 这里的逻辑将来直接替换为调用 C++ 模块：
    # return _cpp_core.assemble_conduction(c_box, c_k, ...)
```

**步骤三：优化热源映射 (Power Matrix)**
目前的 `_precompute_power_matrix` 用了 numpy 的广播计算交集。当网格规模达到百万，热源块达到上千个时，这里的交叉对比 $O(N \times M)$ 会非常慢。
*   **指引**：将这一步也圈定为 C++ 加速的重点。C++ 中可以引入 **BVH (Bounding Volume Hierarchy)** 或 **R-Tree** 空间索引算法，将复杂度降到 $O(M \log N)$。