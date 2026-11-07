# 数据流与设计原则

## 核心对象与生命周期

| 对象 | 职责 | 生命周期 |
| --- | --- | --- |
| `ModelDefinition` | 几何、材料、热源、边界、探针和 study 设置 | 可变输入；由 XML reader 或 C/Python 构造接口填充 |
| `mhs::model::Model` | 显式网格、材料与积分数据、编译表达式和冻结流场 | `build_model` 的独立结果；供重复装配和求解使用 |
| `Operators {K, C, f}` | 当前状态和时刻的线性化系统 `C dx/dt + K x = f` | 热求解工作区持有；重复装配只刷新数值，单次导出独立拥有 |
| `Solution` | 最终单元状态、状态历史和探针记录 | 独立拥有求解结果 |

`compile` 不修改输入；`assemble` 和 `solve` 不修改编译模型。C opaque handle 和 Python wrapper 负责所有权与转换。Python 返回的 NumPy/SciPy 数据在 native handle 销毁后仍有效。

## 编译与装配

| 阶段 | 输入 | 输出与规则 |
| --- | --- | --- |
| 几何解析 | Layer / Block / Rect 和几何表达式 | SI 米下的区域；同层后出现的 Block 覆盖前者 |
| 网格输入 | Cartesian 坐标轴与物理区域 | `DiscreteMesh`：单元边界、唯一有向面片和单元—面片 CSR 关联 |
| 网格契约 | 控制体与面片几何、连接 | 验证坐标轴有限且严格递增；生成控制体、有向面片与 CSR 关联 |
| 体积投影 | 网格与物理区域 | `cell_material` 和 `source_terms`；调用者保证控制体与材料及物理域边界对齐，编译器不校验 |
| 边界投影 | 外露面片与有序 `BoundaryPatch` | 生成连续编号的 `boundary_faces`；每片引用原始 FaceId 和 `BoundaryCondition`，不改变拓扑，后出现的边界覆盖前者，未覆盖部分使用默认边界 |
| 表达式编译 | 材料、源、边界表达式 | `CompiledExpression`；线程模型见 [expr-api.md](expr-api.md) |
| 流体预处理 | 流体材料、连接和流体边界 | 独立按面积投影流体边界；临时压力解转换为内部面体积流量、边界片质量通量与入口温度、流固换热因子 |
| 容量与体热源装配 | 单元几何、材料、体积投影、状态和时间 | `rho c V` 及各投影片上的热源积分 |
| 面片装配 | 面片、两侧物理类型、边界描述 | 内部面只装配一次；外边界迹由统一局部方程求解并凝聚到单元系统 |
| 流体能量装配 | 冻结流量、状态和材料 | 上风对流和入口能量贡献，直接追加到主装配的矩阵条目与 RHS；复用容量装配求值的 rho/cp，无中间增量对象 |

当前 TPFA 适用于轴对齐长方体、相邻中心沿面法向对齐及对角导热张量。数值内核消费已有网格，不生成或加密网格。

`SourceTerm` 仅保存 CellId、源表索引、体积权重和积分点；一个控制体可以接收多个源。常数体热源按体积守恒，空间表达式在各交叠片上采用中点积分。材料归属仅保存每单元一个材料索引，取该单元切分遍历中最后命中的材料；未命中物理区域的片段不贡献热源。这不能代替材料混合离散，调用者必须保证材料对齐。热与流体边界分别采用后出现条件覆盖前者的面积投影，重叠矩形按并集计量；同一单元上的不同入口分别累计质量流量与焓。压力条件施加于投影面积，质量流量按选中面积分配；未指定的入口温度取初始温度。

`FluidBoundaryTerm` 保存 FaceId 和内联的 `FluidBoundaryCondition = variant<FluidInflow, FluidOutflow>`。两种记录均保存非负的质量流率（kg/s），仅 `FluidInflow` 保存流入温度；按压力解得到的实际流向选择类型，而非输入的入口/出口标签，因此压力边界反向流动时仍采用该边界的流入温度。零流量归入 `FluidOutflow`，能量贡献为零。这些记录只有数值和内联判别信息，无动态分配；每片的冻结流量独立，不需要额外条件表。

## 求解与输出

公开 C++ 入口 `mhs::solver::solve(model, initial_state, options)` 位于 `solver/solver.hpp`。内部驱动 `solve_system`（`src/solver/solve.hpp`）通过 `SystemAssembler(state, time)` 刷新调用者持有的 `Operators`，返回引用在下一次装配前有效。热求解的局部 `ThermalAssembler` 绑定只读模型，私有持有算子和缓存，一次建立 `K/C` 的模式和材料求值缓冲区，逐次清零并填充数值。`C` 保留统一稀疏矩阵表示。非线性求解器消费只读 `LinearSystemView`，矩阵及 RHS 由工作区持有。瞬态一次建立 `K/C` 的联合模式，后续通过 `add_scaled_into` 原地更新 `A=K+αC`，支持非对角 `C`；初始联合模式必须覆盖整个求解期间的结构。稳态在 `t=0` 求解；瞬态由私有 AdaptiveTimeStepper 控制自适应 BDF1/2 步长与阶数，`SolveOptions` 仅暴露 `time_tolerance` 与可选 `max_time_step`。输出时刻独立于求解步，observer 接收稠密插值生成的中间状态及真实积分终点。

探针初始化时定位控制体，记录时由局部邻域重建温度；共享面上的点选择下角坐标按字典序最大的控制体。单元状态按 CellId 访问，历史快照只保存单元状态。应用层根据离散网格写出 VTU，并使用输入的 Cartesian 顶点规格采样 XML 输出。

## 输入与数组约定

- C++ 的 `ModelDefinition::mesh`、C 的 `mhs_model_set_mesh` 和 Python 的 `Model.set_mesh` 只接收笛卡尔网格顶点规格，坐标使用 SI 米。控制体与面片连接由编译器生成，不提供外部离散网格输入。
- C/Python 网格导出使用 `face_minus`、`face_plus`、`face_axis` 和 `face_bounds`。前三者为 `(face_count,)`，bounds 为 `(face_count, 4)`，每行为 `[a_min, a_max, b_min, b_max]`，按升序切向轴排列（YZ、XZ、XY）。cell 的 min/max 为 `(cell_count, 3)`。正通量沿 axis 从 minus 流向 plus；外边界恰有一侧为 `MHS_INVALID_INDEX`。
- 面积由切向区间长度之积得到；单元至面的距离为法向半边长，平面位置由 minus 的 max 或 plus 的 min 得到。外法向由缺失的一侧确定。一个 cell side 可有多个积分 patch，支持轴对齐的非匹配界面。cell→faces 仅保留一套 CSR 索引，编译器直接构建。
- 几何、材料、`K/C/f`、`state` 和 `state_history` 按 CellId 排列。Python `Solution.state` 和 `state_history` 返回对应数组。
- `Compiled.source_operator(state, time)` 按物理源返回稀疏功率列。`SourcePort` 必须且只能指定 `source_ids` 或物理 `cells` 选择器之一。
- `BoundaryGroup(surface=geometry.surface(Face.ZP), h_range=...)` 使用实际 FaceId 和投影面积，同一控制体上的面片贡献累加。Robin 参数采用组平均的有效仿射系数，并非物理 HTC 的精确仿射表示。

函数在编译阶段统一构造求值器：分段线性要求至少两个点，按横坐标排序，并拒绝重复横坐标和非有限值；周期分段常数要求非空值和有限正周期。表达式解析失败立即抛出异常，不保存失败求值器。所有探针值共用 `Solution::snapshot_times`；XML 输出显式接收这份时间序列。

Python `Compiled.info` 只获取标量元信息，`Compiled.fields` 在首次访问时复制几何、材料索引与源投影数组。两者独立缓存；`cell_count` 和 `default_state()` 无需复制网格。

## 索引约定

- 数量、循环位置和数组偏移使用 `std::size_t`。
- `core::EntityIndex = uint32_t` 保存实体引用，model 的各类 `Id` 是其语义别名；无效引用显式使用 `core::invalid_id = UINT32_MAX`。
- `core::SparseIndex = int32_t` 用于 CSR 存储，`Triplet` 是聚合类型。第三方索引类型留在实现文件中。
- C ABI 对应 `mhs_index_t`、`mhs_sparse_index_t` 和 `size_t`；Python 索引数组对应 `np.uint32` 和 `np.int32`。

## C ABI revision 1

`MHS_ABI_VERSION` 与 `mhs_abi_version()` 标识 C ABI 修订，独立于软件包版本。
Python 加载动态库时先校验此修订，旧动态库会得到明确错误。C/C++ 调用者须同步更新头文件和动态库。

- 删除外部离散网格输入及 `build_model(definition, mesh)` 重载，模型编译统一通过 `ModelDefinition`。
- `mhs_operators_copy_k/c` 直接复制 CSR，布局为 `row_ptr[state_count+1]`、`col_idx[nnz]`、`values[nnz]`；Python `assemble()` 返回 SciPy CSR，需要 CSC 的算法自行转换。
- `mhs_solution_info_t.converged` 和 Python `Solution.converged` 暴露求解收敛结果。API 成功表示调用完成，收敛状态由该字段判断。
- 流体边界与热边界使用相同的区域数组；入口温度用可空指针表示，NULL 取初始温度。Python 保留原有单区域调用和 `None`。删除无操作的流体边界类型；空区域条件报错。
- 输入数组和字符串由库复制；输出数组由调用者分配。四类句柄独立拥有对象，析构允许 NULL。创建句柄的调用在验证参数前将有效的输出地址清零。
- 成功的 status 调用清空当前线程错误；失败调用替换错误，查询和析构保留错误。初始状态仅允许 NULL/0 或完整的逐单元数组。
- 网格规格原子替换，任何坐标轴校验失败均保留先前规格；Python 同时接受数组与可转换为 float64 的一维序列。
