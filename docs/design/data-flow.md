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
                                    └─> mhs::sim::solve_system(Study, SystemAssembler, state)
                                    ├─> mhs::sim::time_scheme::StepController::prepare(dt_sug, t) → dt_exec
                                    ├─> mhs::sim::assemble_thermal(model, fvm_state, time)
                                    │     ├─> assemble_cells parallel_for // no fluid branches
                                    │     │     ├─> material_table[mat_id].{kx,ky,kz}.eval(ctx)   @ cell state
                                    │     │     ├─> material_table[mat_id].{rho,c}.eval(ctx)       @ cell state
                                    │     │     ├─> k_along(dir) 选用该面法向对应的 k
                                    │     │     ├─> face_bcs[c*6 + dir] + bc_params.eval
                                    │     │     ├─> heat_source_table[hs_idx].eval
                                    │     │     └─> thread-local K/C triplets + f
                                    │     ├─> assemble_fluid               // same sparse coordinates
                                    │     └─> merge once → Operators {K, C, f}
                                    ├─> mhs::sim::time_scheme::build_system(kind, ops, hist, dt) → LinearSystem
                                    ├─> mhs::sim::nonlinear_solve(ls_provider, state, solver) [Anderson 加速定点迭代]
                                    └─> mhs::sim::time_scheme::estimate_error(hist, state, dt, cfg) → ErrorEstimate
                                    ├─> StepController::output_due(t + dt) → exact solved output state
                                    ├─> mhs::sim::ProbeRecorder::record(time, cell_T)   // 输出时刻 O(n_probes) 局部采样
                                    └─> mhs::post::interpolate_cell_to_node           // solve_thermal() 结束后一次性展开
                                          ├─> cell 内 k 退化为三轴算术平均（软权重）
                                          ├─> 面中心外推使用该面法向对应的 k
                                          └─> mhs::io::write_vtu + mhs::io::write_xml(probeTraces)   (virtual → NaN)
```

## 核心对象与生命周期

公共模型流程只保留四个对象。它们分别对应一个明确的数学或运行期语义，
不把 C ABI handle、Python wrapper 或中间 buffer 当成新的领域对象。

```text
ModelDefinition
    可变的 authoring model；保存几何、材料、边界、源和 study 设置
    由 XML reader 或 Model builder 创建；compile 不修改它
        │ compile
        ▼
CompiledModel (mhs::core::Model)
    不可变的离散运行期模型；保存网格、cell mapping、已编译表达式、face BC
    和流体预处理结果；独立拥有编译结果
        │ assemble(state, time)
        ▼
Operators {K, C, f}
    某一状态和时刻的线性化离散系统：C · dx/dt + K · x = f
    一次 assembly 的值对象；不拥有 CompiledModel，也不改变它
        │ solve(initial_state, options)
        ▼
Solution
    一次求解的最终状态、accepted/output history 和 probe traces
    独立拥有结果；不依赖调用方传入的 state buffer
```

生命周期约定：

- `ModelDefinition` 只负责 setup；`compile` 是从定义到冻结离散模型的一次转换。
- `CompiledModel` 是重复 `assemble` 和 `solve` 的唯一运行期入口。
- `Operators` 是 assembly 的结果，不是第二种模型定义，也不是求解器的全局状态。
- `Solution` 是一次 solve 的不可变快照；关闭或修改输入对象不改变它。
- C API 的 opaque handle 和 Python 对象只负责 ownership 与转换，不增加上述对象的语义。
- Python 返回的 NumPy/SciPy 数据属于 Python；native handle 销毁后仍然有效。

## 各阶段

| 阶段              | 输入                                  | 输出                            | 关键                                                     |
| ----------------- | ------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| XML 解析          | XML 文件                              | `ModelDefinition`               | `mhs_io` / tinyxml2                                      |
| 预处理-几何       | `mesh.{x,y,z}_vertices`               | `MeshGeometry`                  | si_scale, dx/dy/dz, cx/cy/cz                             |
| 预处理-层几何     | `ModelDefinition.layers`              | `ResolvedLayerGeometry[]`       | 预求 Z 范围 + Block XY                                   |
| 预处理-单元拓扑   | mesh + 层几何                         | `grid_to_cell` + `cell_to_grid` | 精确双向映射；虚拟网格标记                               |
| 预处理-单元归属   | mesh + 层几何                         | `material_id`                   | compact（`c_idx` 索引）；cell→block 反向遍历（后写优先） |
| 预处理-面 BC      | mesh + `BoundaryPatch[]`              | `face_bcs` + `BCParamTable`     | 6 面独立 + `default_boundary` 兜底                       |
| 预处理-表达式编译 | `ModelDefinition` 中的表达式字符串    | `CompiledExpression`            | `mhs_expression`（muparser + TBB）                       |
| 组装              | 当前完整状态 + 时间                   | `Operators {K,C,f}`             | `SystemAssembler` 拥有全部当前热/流体耦合物理            |
| 时间输出          | accepted state + output grid          | observer state                  | Adaptive/Fixed 均截短到网格；禁止插值全局/模态状态       |
| 线性求解          | `A x = b`                             | `x`                             | 默认 AmgCg（AMGCL + Eigen）；可选 Pardiso（MKL）         |
| 非线性更新        | 线性解 `G(x)`                         | 下一状态迭代值                  | Anderson 加速或欠松弛                                    |
| 后处理            | `Model` + `cell_temperature`          | VTU + XML                       | 展开到全网格，虚拟位置 NaN                               |
| 探针记录          | `cell_T` + `model.observation_points` | `ProbeTrace[]`                  | 每步 O(n_probes) 局部采样；trace 作为 `Solution` 返回    |

运行期 SoA 和面 BC 约定见代码注释；表达式线程模型见 [expr-api.md](expr-api.md)。
