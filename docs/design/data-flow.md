# 数据流与设计原则

## 总览

```text
XML
  └─> io::read_xml                          (tinyxml2)
        └─> mhs::IOStructure                (AoS, 含字符串)
                └─> Preprocessor::load
                      ├─> expr::clear_registry + set_variable(几何) + register_native(ios.functions)
                      ├─> MeshGeometry (×si_scale)
                      ├─> resolve_geometry         (几何预求)
                      ├─> material_table           (k/ρ/c 编译)
                      ├─> resolve_layers           (valid_mask, index_map, material_id, layer_id)
                      ├─> heat_source_table        (去重 ti_reyuan_expr)
                      ├─> resolve_face_keys        (CellBC + BCParamTable + other_bc)
                      └─> InternalModel
                              └─> Scheduler::run
                                    ├─> Assembler::assemble(state)
                                    │     ├─> tbb::parallel_for(0, total)   // skip virtual
                                    │     │     ├─> material_table[mat_id].{k,ρ,c}.eval(ctx)
                                    │     │     ├─> cell_bcs.types/param_idxs + bc_params.eval
                                    │     │     ├─> heat_source_table[hs_idx].eval
                                    │     │     └─> thread-local triplets + b
                                    │     ├─> combine_each merge
                                    │     └─> nonlinear::solve → Solver::solve(A,b)
                                    └─> Postprocessor::interpolate_cell_to_node
                                          └─> io::write_vtu + io::write_xml   (virtual → NaN)
```

## 各阶段

| 阶段              | 输入                            | 输出                       | 关键                              |
| ----------------- | ------------------------------- | -------------------------- | --------------------------------- |
| XML 解析          | XML 文件                        | `IOStructure`              | tinyxml2                          |
| 预处理-几何       | `mesh_vertex_*`                 | `MeshGeometry`             | si_scale, dx/dy/dz, cx/cy/cz      |
| 预处理-层几何     | `IOStructure.layers`            | `ResolvedLayerGeometry[]`  | 预求 Z 范围 + Block XY            |
| 预处理-虚拟单元   | mesh + 层几何                   | `valid_mask` + `index_map` | 标记 + 紧凑映射                   |
| 预处理-单元归属   | mesh + 层几何                   | `material_id` + `layer_id` | cell → block 反向遍历（后写优先） |
| 预处理-面 BC      | mesh + `Boundaries`             | `CellBC` + `BCParamTable`  | 6 面独立 + `other_bc` 兜底        |
| 预处理-表达式编译 | IO 字符串                       | `CompiledExpression`       | exprtk 或 `make_constant`         |
| 组装              | `InternalModel` + `GlobalState` | `LinearSystem`             | TBB 并行；`eval()` 锁无关         |
| 线性求解          | `A x = b`                       | `x`                        | SparseLU / BiCGSTAB               |
| 非线性更新        | `ΔT`                            | `T_new = T_old + ω·ΔT`     | 状态更新                          |
| 后处理            | `InternalModel` + `T`           | VTU + XML                  | 展开到全网格，虚拟位置 NaN        |

## 关键设计原则

### 1. 内部模型不含原始字符串

所有表达式预编译为 `CompiledExpression`（`mhs::expr`，`mhs::CompiledExpression` 为别名）。

### 2. 热源字典化

`heat_source_table`（去重 `vector<CompiledExpression>`）+ 每单元 `uint16_t` 索引。重复公式只编译一次；N 个单元 → N 个 2 字节索引 + 唯一公式数个 AST。

### 3. Cell 级别 BC

`CellBC { types[6], param_idxs[6] }`。

### 4. 虚拟单元

`valid_mask` + `index_map` 标记。Assembler 跳过；Postprocessor 展开写 NaN。

### 5. SoA 贯穿内部模型

所有热循环数组按字段连续存储。

### 6. 无共享可变状态

模块间通过 const 引用和返回值通信。`expr` 注册表是唯一全局可变状态，需在 `Preprocessor::load()` 起头 `clear_registry()`。

### 7. 无异常，panic 退出

不可恢复错误经 `MHS_LOG_ERROR` 调 `mhs::logger::panic()`。**唯一例外**：`bin/main.cpp` 用 `try/catch` 捕获 tinyxml2/exprtk 的 `std::exception` 并转 panic — 边界 entry 必需。
