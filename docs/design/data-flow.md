# 数据流与设计原则

---

## 数据流总览

```text
XML 文件
  └─> io::read_xml
        └─> model::IOStructure（IO 模型，仅含字符串，映射 XML schema）
                    └─> Preprocessor::load
                          ├─> expr::clear_registry + register variables/functions
                          ├─> Build MeshGeometry from mesh_vertex_x/y/z (×si_scale)
                          │     └─> compute dx/dy/dz, cx/cy/cz
                          ├─> resolve_geometry() — 预求解层 Z 范围 + Block XY 坐标
                          ├─> Build material_table (parse k/rho/c expressions)
                          ├─> preprocessor::resolve_layers()
                          │     └─> CellFields（valid_mask, index_map, material_id, layer_id）
                          ├─> Compile heat_source per cell (find_block_for_cell again)
                          ├─> preprocessor::resolve_face_keys()
                          │     └─> CellFields.cell_bcs（每单元每面独立 BC）
                          │         + BCParamTable（编译后的表达式）
                          └─> model::InternalModel
                                └─> Scheduler::run()
                                      ├─> Assembler::assemble(state)
                                      │     ├─> MaterialProps.k/rho/c.eval(ctx)
                                      │     ├─> CellBC types + BCParamTable eval(ctx)
                                      │     ├─> heat_source[].eval(ctx)
                                      │     └─> Solver::solve(A, b)
                                      └─> Postprocessor::interpolate_cell_to_node
                                            ├─> io::write_vtu（展开 T，虚拟区域 NaN）
                                            └─> io::write_xml（展开 T，虚拟区域 NaN）
```

---

## 各阶段数据变换

| 阶段              | 输入                                       | 输出                       | 关键操作                                           |
| ----------------- | ------------------------------------------ | -------------------------- | -------------------------------------------------- |
| XML 解析          | XML 文件                                   | `model::IOStructure`       | tinyxml2 解析，包含 mesh_vertex_x/y/z              |
| 预处理-几何       | `IOStructure.mesh_vertex_*`                | `MeshGeometry`             | si_scale 转换，计算 dx/dy/dz, cx/cy/cz             |
| 预处理-层几何     | `IOStructure.layers`                       | `ResolvedLayerGeometry[]`  | 预求解层 Z 范围和 Block XY 坐标（Block 无 Z 维度） |
| 预处理-虚拟单元   | `MeshGeometry` + 层几何                    | `valid_mask` + `index_map` | 标记虚拟单元，生成紧凑化映射                       |
| 预处理-单元归属   | `MeshGeometry` + 层几何                    | `material_id` + `layer_id` | 判断每个单元属于哪个 Layer/Block                   |
| 预处理-面 BC      | `MeshGeometry` + `Boundaries`              | `CellBC` + `BCParamTable`  | 为每个单元每面分配 BC，解决投影重叠                |
| 预处理-表达式编译 | IO 字符串表达式                            | `CompiledExpression`       | exprtk 编译或 `make_constant`                      |
| 组装              | `InternalModel` + `GlobalState`（含 `dt`） | `LinearSystem`             | 遍历活跃单元，组装 A 和 b                          |
| 线性求解          | `A * x = b`                                | `x`                        | Eigen `SparseLU` 或 `BiCGSTAB`                     |
| 非线性 更新       | `ΔT`                                       | `T_new = T_old + ω·ΔT`     | 状态更新                                           |
| 后处理            | `InternalModel` + `T`                      | VTU + XML                  | 展开 T 向量，写出文件                              |

---

## 关键设计原则详解

### 1. 内部模型不含原始字符串

所有表达式在预处理阶段编译为 `CompiledExpression`。

### 2. 热源为 per-cell

`CellFields.heat_source` 是 `vector<CompiledExpression>`，每个活跃单元一个。

### 3. Cell 级别 BC

BC 存储在单元级别，解决面投影重叠问题：

- 每个单元的每个面独立 BC（`CellBC { types[6], param_idxs[6] }`）
- **FaceBCFields 已移除**（ADR-0005）

### 4. 虚拟单元标记与展开

- `valid_mask` + `index_map` 标记虚拟单元
- Assembler 只处理活跃单元
- Postprocessor 展开 T 向量，虚拟区域填充 NaN

### 5. SoA 贯穿全局

```cpp
struct CellFields {
    int cell_count = 0;  // = N_active

    // Full-grid size (nx*ny*nz): virtual + active
    std::vector<size_t> index_map;     // Maps old grid index → compact active index. SIZE_MAX = virtual
    std::vector<uint8_t> valid_mask;   // 1 = active cell, 0 = virtual
    std::vector<size_t> material_id;   // Full grid size
    std::vector<size_t> layer_id;      // Full grid size

    // Compact size (N_active): active cells only
    std::vector<CellBC> cell_bcs;
    std::vector<CompiledExpression> heat_source;
};
```

### 6. 无共享可变状态（expr 除外）

模块间通过 const 引用和返回值通信。expr 模块使用全局注册表（线程安全），需在每次 load 前调用 `clear_registry()`。

### 7. 无异常，mhs::panic() 退出
