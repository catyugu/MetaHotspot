# 数据流与设计原则

## 总览

```text
XML
  └─> mhs::io::read_xml                          (tinyxml2)
        └─> mhs::core::ModelDefinition               (AoS, 含字符串)
                └─> mhs::sim::build_model
                      ├─> 构造本地 mhs::core::SymbolTable（几何变量 + mhs::sim::register_all_functions(ios.functions) 注入的 native）
                      ├─> mhs::core::MeshGeometry (×si_scale)
                      ├─> mhs::sim::resolve_geometry         (几何预求)
                      ├─> material_table           (kx/ky/kz/ρ/c 编译)
                      ├─> mhs::sim::assign_cell_layers       (index_map [full-grid]; material_id + heat_source_idx [compact])
                      ├─> heat_source_table        (去重 ti_reyuan_expr)
                      ├─> mhs::sim::resolve_boundary_patches (face_bcs [N_active * 6] 扁平数组)
                      ├─> mhs::sim::fluid::build_domain
                      │     └─> pressure scratch → frozen face flux + interface factor
                      └─> mhs::core::Model (含 face_bcs, FluidDomain)
                              └─> mhs::sim::solve
                                    ├─> mhs::sim::time_scheme::StepController::rebuild(duration, output_dt)
                                    │     ├─> StepController::prepare(dt_sug, t, duration) → dt_exec
                                    │     ├─> mhs::sim::Assembler::assemble(ctx)
                                    │     │     ├─> base thermal parallel_for    // no fluid branches
                                    │     │     │     ├─> material_table[mat_id].{kx,ky,kz}.eval(ctx)   @ ctx.T
                                    │     │     │     ├─> material_table[mat_id].{rho,c}.eval(ctx)       @ ctx.T
                                    │     │     │     ├─> k_along(dir) 选用该面法向对应的 k
                                    │     │     │     ├─> face_bcs[c*6 + dir] + bc_params.eval
                                    │     │     │     ├─> heat_source_table[hs_idx].eval
                                    │     │     │     └─> thread-local triplets + b + mass
                                    │     │     ├─> fluid::assemble_increment    // same sparse coordinates
                                    │     │     └─> merge once → AssemblyResult {K, f, M_diag}
                                    │     ├─> mhs::sim::time_scheme::build_system(kind, ops, hist, dt) → LinearSystem
                                    │     ├─> mhs::sim::nonlinear_solve(ls_provider, T, solver) [Anderson 加速定点迭代]
                                    │     └─> mhs::sim::time_scheme::estimate_error(hist, T, dt, cfg) → ErrorEstimate
                                    ├─> StepController::flush_outputs(t + dt) → output times
                                    ├─> mhs::sim::ProbeRecorder::record(time, cell_T)   // 每步 O(n_probes) 局部采样
                                    └─> mhs::post::interpolate_cell_to_node           // run() 结束后一次性展开
                                          ├─> cell 内 k 退化为三轴算术平均（软权重）
                                          ├─> 面中心外推使用该面法向对应的 k
                                          └─> mhs::io::write_vtu + mhs::io::write_xml(probeTraces)   (virtual → NaN)
```

## 各阶段

| 阶段              | 输入                                  | 输出                        | 关键                                                     |
|-------------------|---------------------------------------|-----------------------------|----------------------------------------------------------|
| XML 解析          | XML 文件                              | `ModelDefinition`           | tinyxml2                                                 |
| 预处理-几何       | `mesh_vertex_*`                       | `MeshGeometry`              | si_scale, dx/dy/dz, cx/cy/cz                             |
| 预处理-层几何     | `ModelDefinition.layers`              | `ResolvedLayerGeometry[]`   | 预求 Z 范围 + Block XY                                   |
| 预处理-虚拟单元   | mesh + 层几何                         | `index_map`                 | full-grid；标记 + 紧凑映射                               |
| 预处理-单元归属   | mesh + 层几何                         | `material_id`               | compact（`c_idx` 索引）；cell→block 反向遍历（后写优先） |
| 预处理-面 BC      | mesh + `Boundaries`                   | `face_bcs` + `BCParamTable` | 6 面独立 + `other_bc` 兜底                               |
| 预处理-表达式编译 | IO 字符串                             | `CompiledExpression`        | muparser 或 `make_constant`                              |
| 组装              | `Model` + `AssembleContext`           | `LinearSystem`              | TBB 并行；`eval()` 锁无关                                |
| 线性求解          | `A x = b`                             | `x`                         | EigenSparseLU / EigenBiCGSTAB                            |
| 非线性更新        | `ΔT`                                  | `T_new = T_old + ω·ΔT`      | 状态更新                                                 |
| 后处理            | `Model` + `T`                         | VTU + XML                   | 展开到全网格，虚拟位置 NaN                               |
| 探针记录          | `cell_T` + `model.observation_points` | `ProbeTrace[]`              | 每步 O(n_probes) 局部采样；trace 作为 `Solution` 返回    |

## 关键设计原则

### 1. 内部模型不含原始字符串

所有表达式预编译为 `mhs::core::CompiledExpression`。

### 2. 热源字典化

`heat_source_table`（去重 `vector<CompiledExpression>`）+ 每单元 `uint16_t` 索引。重复公式只编译一次；N 个单元 → N 个 2 字节索引 + 唯一公式数个 AST。

### 3. 面级别 BC

`Model::face_bcs[N_active * 6]` 扁平数组。`face_bcs[c * 6 + dir]` 直接索引。

### 4. 虚拟单元

`index_map` + `face_bcs` 标记。Assembler 跳过；Postprocessor 展开写 NaN。

### 5. SoA 贯穿内部模型

所有热循环数组按字段连续存储。

### 6. 无共享可变状态

模块间通过 const 引用和返回值通信。`expr` 模块**没有**任何全局注册表或互斥锁：所有 setup 阶段的符号（几何变量 + native 闭包）通过显式 `mhs::core::SymbolTable` 按值传递，`CompiledExpression` 在构造时捕获其副本，运行时 `eval()` 零同步。多个 `build_model()` 调用互不共享构建状态。

### 7. 无异常，panic 退出

不可恢复错误经 `MHS_FATAL` 调 `mhs::logger::panic()`。**唯一例外**：`bin/main.cpp` 用 `try/catch` 捕获 tinyxml2/muparser 的 `std::exception` 并转 panic — 边界 entry 必需。

### 8. 各向异性热导率 — 面法向匹配与后处理距离权重

`MaterialProps` 按三轴拆分 `kx / ky / kz`。装配器通过 `k_along(dir)` 根据面法向选取对应的分量：X 面用 `kx`，Y 面用 `ky`，Z 面用 `kz`。后处理器在节点插值和面中心外推权重中使用各向异性逆距离

```text
w = 1 / (dx²/kx + dy²/ky + dz²/kz)
```

以使 k 大的方向传播得快的影响更显著，与扩散方程的各向异性一致。

`k_along` / `half_length_along` / `face_area` / `neighbor_grid_index` 等面法向查表助手统一定义在 `src/common/mesh_utils.hpp`（`mhs::utils` 命名空间），由装配器和预处理器共享，避免两处分叉的 `switch (FaceDir)` 分支。
