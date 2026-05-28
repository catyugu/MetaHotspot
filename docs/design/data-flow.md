# 数据流与设计原则

---

## 数据流总览

```text
XML 文件
  └─> io::Reader
        └─> model::IOStructure（IO 模型，仅含字符串，映射 XML schema）
                    └─> preprocessor::ModelBuilder
                          ├─> LayerProcessor::resolve_layer_geometry()
                          │     └─> model::InternalCellFields（SoA：layer_id, material_id, heat_source）
                          ├─> FaceKeyProcessor::resolve_face_keys()
                          │     └─> model::InternalFaceBCFields + BCParamTable（SoA，BC 字符串已解析）
                          ├─> 编译所有表达式 → expr::FieldExpression
                          │     ├─> MaterialProps.k/rho/c（每种材料一个）
                          │     ├─> BCParamTable 参数（每个边界参数一个）
                          │     ├─> 热源（每个单元一个，由 Block.ti_reyuan_expr 编译）
                          │     └─> 用户自定义函数池（exprtk + native）
                          └─> model::InternalModel（SoA，所有表达式已编译，无原始字符串）
                                └─> scheduler::Scheduler
                                      ├─> assembler::Assembler
                                      │     ├─ mat.props.k.eval(ctx)  → 材料导热系数
                                      │     ├─ bc_params.dirichlet_T[idx].eval(ctx) → BC 参数
                                      │     ├─ cells.heat_source[cell_idx].eval(ctx) → 热源
                                      │     └─> solver::SolverBase（Eigen 求解）
                                      │           └─> Eigen 线性代数解
                                      └─> postprocessor::PostProcessor
                                            ├─> VTU 文件（ParaView 可视化）
                                            └─> XML 结果文件（GUI 兼容）
```

---

## 关键设计原则详解

### 1. 内部模型不含原始字符串

所有表达式（材料属性、BC 参数、热源）在预处理阶段编译为 `FieldExpression`。调度器/组装器只调用 `.eval(ctx)`，永远不需要字符串。

**预处理阶段**：

- 接收：`model::IOStructure`（含字符串如 `"1e9"`, `"sin(x)*T"`）
- 输出：`model::InternalModel`（不含任何字符串，全是 `FieldExpression`）

**组装阶段**：

```cpp
// 只需要 eval，不需要知道表达式原本是什么字符串
double k = material.props.k.eval(ctx);
double Q = cells.heat_source[cell_idx].eval(ctx);
```

### 2. 热源为 per-cell

`CellFields.heat_source` 是 `vector<FieldExpression>`，每个单元一个，由 `Block.ti_reyuan_expr` 编译。

即使 `ti_reyuan_expr = "1e9"`（常数），也通过 `FieldExpression::make_constant(1e9)` 存储，保持类型一致。

### 3. 无虚函数

使用模板静态多态。例如 `solver::SolverBase` 不需要虚函数：

```cpp
// solver 工厂直接返回具体类型，不使用虚函数调度
template<SolverType T>
class ConcreteSolver { /* ... */ };

class SolverFactory {
public:
    static std::unique_ptr<SolverBase> create(const SolverConfig& config) {
        if (config.type == SolverType::SparseLU) {
            return std::make_unique<ConcreteSolver<SolverType::SparseLU>>(config);
        }
        return std::make_unique<ConcreteSolver<SolverType::BiCGSTAB>>(config);
    }
};
```

### 4. 无异常，mhs::panic() 退出

```cpp
// 不可恢复错误：记录并退出
MHS_LOG_ERROR("Failed to open file: {}", path);
MHS_LOG_ERROR("Solver diverged at step {}", step);

// 可恢复错误：记录警告并返回回退值
MHS_LOG_WARN_RETURN("Material not found, using default k={}", 400.0);
```

### 5. SoA 贯穿全局

所有热循环中的字段数组按字段连续存储，而非按单元打包为结构体：

```cpp
// GOOD: SoA — 所有 material_id 连续读取，缓存友好
struct CellFields {
    std::vector<MaterialID> material_id;  // 连续访问
    std::vector<LayerID> layer_id;        // 连续访问
    std::vector<FieldExpression> heat_source;  // 连续访问
};

// BAD: AoS — 访问 material_id 时也读入了其他字段，缓存污染
struct Cell { MaterialID mat; LayerID layer; FieldExpression Q; };
std::vector<Cell> cells;  // 缓存不友好
```

### 6. 纯函数优先

`assembler::assemble()` 是纯函数 — 给定相同的 `model` + `state` + `t`，返回相同的 `A` 和 `b`：

```cpp
LinearSystem Assembler::assemble(const GlobalState& state, double t) const {
    // model_ 是成员，但它是 const 引用，不会被修改
    // state 是 const 引用，不会被修改
    // t 是值传递
    // 返回值是新的 LinearSystem，不修改任何成员
}
```

### 7. 无共享可变状态

模块间通过 const 引用和返回值通信：

```cpp
// GOOD: 接收 const 引用，返回新对象
LinearSystem assemble(const InternalModel&, const GlobalState&, double t);

// GOOD: 返回 const&，内部缓存不变
const MaterialProps& get_material(MaterialID id) const { return material_table[id]; }

// BAD: 修改共享状态
void modify_global_state(GlobalState& state);  // 避免
```

---

## 各阶段数据变换

| 阶段              | 输入                                  | 输出                            | 关键操作                                       |
| ----------------- | ------------------------------------- | ------------------------------- | ---------------------------------------------- |
| XML 解析          | XML 文件                              | `model::IOStructure`            | tinyxml2 解析 + io::Reader                     |
| 预处理-几何       | `model::IOStructure` + 变量           | `MeshGeometry`                  | 解析几何表达式，计算顶点坐标                   |
| 预处理-单元归属   | `MeshGeometry` + 层几何               | `CellFields`                    | 判断每个单元属于哪个 Layer/Block               |
| 预处理-面 BC      | `MeshGeometry` + `Boundaries`         | `FaceBCFields` + `BCParamTable` | 解析 face_key，填充面数组                      |
| 预处理-表达式编译 | IO 字符串表达式                       | `FieldExpression`               | exprtk 编译或 `make_constant`                  |
| 组装              | `InternalModel` + `GlobalState` + `t` | `LinearSystem`                  | 对每个单元：求材料属性、热源、BC → 组装 A 和 b |
| 线性求解          | `A * x = b`                           | `x`                             | Eigen `SparseLU` 或 `BiCGSTAB`                 |
| Newton 更新       | `ΔT`                                  | `T_new = T_old + ω·ΔT`          | 状态更新，ω = 欠松弛因子                       |
| 后处理            | `InternalModel` + `T`                 | VTU + XML                       | 插值到顶点，写出文件                           |
