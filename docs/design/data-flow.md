# 数据流与设计原则

## 总览

```text
XML
  └─> mhs::io::read_xml                          (tinyxml2)
        └─> mhs::model::ModelDefinition              (有序 Authoring Model)
                └─> mhs::sim::build_model
                      ├─> 构造本地 mhs::core::SymbolTable（几何变量 + mhs::sim::register_all_functions(ios.functions) 注入的 native）
                      ├─> mhs::core::MeshGeometry (×si_scale)
                      ├─> mhs::sim::resolve_geometry         (几何预求)
                      ├─> material_table           (kx/ky/kz/ρ/c 编译)
                      ├─> mhs::sim::assign_cell_layers       (grid_to_cell [full-grid]; cell_to_grid + fields [compact])
                      ├─> heat_source_table        (按 Block 编译 volumetric_heat_source)
                      ├─> mhs::sim::resolve_boundary_patches (face_bcs [N_active * 6] 扁平数组)
                      ├─> mhs::sim::fluid::build_domain
                      │     └─> pressure scratch → frozen face flux + interface factor
                      └─> mhs::core::Model (含 face_bcs, FluidDomain)
                              └─> mhs::sim::solve
                                    ├─> mhs::sim::time_scheme::StepController::prepare(dt_sug, t, duration) → dt_exec
                                    ├─> mhs::sim::Assembler::assemble(ctx)
                                    │     ├─> assemble_cells parallel_for // no fluid branches
                                    │     │     ├─> material_table[mat_id].{kx,ky,kz}.eval(ctx)   @ cell state
                                    │     │     ├─> material_table[mat_id].{rho,c}.eval(ctx)       @ cell state
                                    │     │     ├─> k_along(dir) 选用该面法向对应的 k
                                    │     │     ├─> face_bcs[c*6 + dir] + bc_params.eval
                                    │     │     ├─> heat_source_table[hs_idx].eval
                                    │     │     └─> thread-local K/C triplets + f
                                    │     ├─> assemble_fluid               // same sparse coordinates
                                    │     └─> merge once → AssemblyResult {K, C, f}
                                    ├─> mhs::sim::time_scheme::build_system(kind, ops, hist, dt) → LinearSystem
                                    ├─> mhs::sim::nonlinear_solve(ls_provider, state, solver) [Anderson 加速定点迭代]
                                    └─> mhs::sim::time_scheme::estimate_error(hist, state, dt, cfg) → ErrorEstimate
                                    ├─> StepController::flush_outputs(t + dt) → output times
                                    ├─> mhs::sim::ProbeRecorder::record(time, cell_T)   // 每步 O(n_probes) 局部采样
                                    └─> mhs::post::interpolate_cell_to_node           // solve() 结束后一次性展开
                                          ├─> cell 内 k 退化为三轴算术平均（软权重）
                                          ├─> 面中心外推使用该面法向对应的 k
                                          └─> mhs::io::write_vtu + mhs::io::write_xml(probeTraces)   (virtual → NaN)
```

## 各阶段

| 阶段              | 输入                                  | 输出                            | 关键                                                     |
|-------------------|---------------------------------------|---------------------------------|----------------------------------------------------------|
| XML 解析          | XML 文件                              | `ModelDefinition`               | tinyxml2                                                 |
| 预处理-几何       | `mesh.{x,y,z}_vertices`               | `MeshGeometry`                  | si_scale, dx/dy/dz, cx/cy/cz                             |
| 预处理-层几何     | `ModelDefinition.layers`              | `ResolvedLayerGeometry[]`       | 预求 Z 范围 + Block XY                                   |
| 预处理-单元拓扑   | mesh + 层几何                         | `grid_to_cell` + `cell_to_grid` | 精确双向映射；虚拟网格标记                               |
| 预处理-单元归属   | mesh + 层几何                         | `material_id`                   | compact（`c_idx` 索引）；cell→block 反向遍历（后写优先） |
| 预处理-面 BC      | mesh + `BoundaryPatch[]`              | `face_bcs` + `BCParamTable`     | 6 面独立 + `default_boundary` 兜底                       |
| 预处理-表达式编译 | IO 字符串                             | `CompiledExpression`            | muparser 或 `make_constant`                              |
| 组装              | `Model` + `AssembleContext`           | `AssemblyResult {K,C,f}`        | TBB 并行；`eval()` 锁无关                                |
| 线性求解          | `A x = b`                             | `x`                             | EigenSparseLU / EigenBiCGSTAB                            |
| 非线性更新        | 线性解 `G(x)`                         | 下一状态迭代值                  | Anderson 加速或欠松弛                                    |
| 后处理            | `Model` + `cell_temperature`          | VTU + XML                       | 展开到全网格，虚拟位置 NaN                               |
| 探针记录          | `cell_T` + `model.observation_points` | `ProbeTrace[]`                  | 每步 O(n_probes) 局部采样；trace 作为 `Solution` 返回    |

运行期 SoA 和面 BC 约定见代码注释；表达式线程模型见 [expr-api.md](expr-api.md)。
