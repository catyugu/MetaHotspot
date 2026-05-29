# 数据流与设计原则

---

## 数据流总览

```text
XML 文件
  └─> io::Reader
        └─> model::IOStructure（IO 模型，仅含字符串，映射 XML schema）
                    └─> preprocessor::ModelBuilder
                          ├─> LayerProcessor::resolve()
                          │     └─> CellFields（valid_mask, index_map, material_id, layer_id）
                          ├─> FaceKeyProcessor::resolve()
                          │     └─> CellFields.cell_bcs（每单元每面独立 BC）
                          │         + BCParamTable（编译后的表达式）
                          ├─> 编译所有表达式 → expr::CompiledExpression
                          │     ├─> MaterialProps.k/rho/c（每种材料一个）
                          │     ├─> BCParamTable 参数（每个边界参数一个）
                          │     └─> 热源（每个活跃单元一个）
                          └─> model::InternalModel
                                └─> scheduler::Scheduler
                                      ├─> assembler::Assembler
                                      │     ├─ mat.props.k.eval(ctx) → 材料导热系数
                                      │     ├─ cell_bcs[].types[].eval(ctx) → BC 参数
                                      │     ├─ heat_source[].eval(ctx) → 热源
                                      │     └─> solver::SolverBase（Eigen 求解）
                                      └─> postprocessor::PostProcessor
                                            ├─> VTU（展开 T，虚拟区域 NaN）
                                            └─> XML（展开 T，虚拟区域 NaN）
```

---

## 各阶段数据变换

| 阶段              | 输入                                  | 输出                       | 关键操作                                   |
| ----------------- | ------------------------------------- | -------------------------- | ------------------------------------------ |
| XML 解析          | XML 文件                              | `model::IOStructure`       | tinyxml2 解析，包含 mesh_vertex_x/y/z      |
| 预处理-几何       | `IOStructure.mesh_vertex_*`           | `MeshGeometry`             | 直接使用 XML 坐标，计算 dx/dy/dz, cx/cy/cz |
| 预处理-虚拟单元   | `MeshGeometry` + 层几何               | `valid_mask` + `index_map` | 标记虚拟单元，生成紧凑化映射               |
| 预处理-单元归属   | `MeshGeometry` + 层几何               | `material_id` + `layer_id` | 判断每个单元属于哪个 Layer/Block           |
| 预处理-面 BC      | `MeshGeometry` + `Boundaries`         | `CellBC` + `BCParamTable`  | 为每个单元每面分配 BC，解决投影重叠        |
| 预处理-表达式编译 | IO 字符串表达式                       | `CompiledExpression`       | exprtk 编译或 `make_constant`              |
| 组装              | `InternalModel` + `GlobalState` + `t` | `LinearSystem`             | 遍历活跃单元，组装 A 和 b                  |
| 线性求解          | `A * x = b`                           | `x`                        | Eigen `SparseLU` 或 `BiCGSTAB`             |
| Newton 更新       | `ΔT`                                  | `T_new = T_old + ω·ΔT`     | 状态更新                                   |
| 后处理            | `InternalModel` + `T`                 | VTU + XML                  | 展开 T 向量，写出文件                      |

---

## 关键设计原则详解

### 1. 内部模型不含原始字符串

所有表达式在预处理阶段编译为 `CompiledExpression`。

### 2. 热源为 per-cell

`CellFields.heat_source` 是 `vector<CompiledExpression>`，每个活跃单元一个。

### 3. Cell 级别 BC

BC 存储在单元级别，解决面投影重叠问题：

- 每个单元的每个面独立 BC（`CellBC { types[6], param_idxs[6] }`）
- `FaceBCFields` 已移除

### 4. 虚拟单元标记与展开

- `valid_mask` + `index_map` 标记虚拟单元
- Assembler 只处理活跃单元
- Postprocessor 展开 T 向量，虚拟区域填充 NaN

### 5. SoA 贯穿全局

```cpp
struct CellFields {
    std::vector<uint8_t> valid_mask;   // 全网格大小
    std::vector<size_t> index_map;     // 全网格大小
    std::vector<size_t> material_id;    // 全网格大小
    std::vector<size_t> layer_id;       // 全网格大小
    std::vector<CellBC> cell_bcs;      // 紧凑（活跃单元）
    std::vector<CompiledExpression> heat_source; // 紧凑（活跃单元）
};
```

### 6. 无共享可变状态

模块间通过 const 引用和返回值通信。

### 7. 无异常，mhs::panic() 退出
