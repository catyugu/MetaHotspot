# MetaHotspot 接口设计

本文档描述 MetaHotspot 热仿真框架的核心数据结构与模块接口，基于 `docs/adr/` 中记录的建筑决策。

---

## 文档结构

| 文件                                             | 内容                                                                          |
| ------------------------------------------------ | :---------------------------------------------------------------------------- |
| **[io-model.md](io-model.md)**                   | IO 模型结构（直接映射 XML schema）                                            |
| **[internal-model.md](internal-model.md)**       | 内部模型结构（SoA 布局，扁平化）                                              |
| **[module-interfaces.md](module-interfaces.md)** | 模块接口定义（io、preprocessor、assembler、solver、scheduler、postprocessor） |
| **[expr-api.md](expr-api.md)**                   | expr 模块接口（表达式解析与求值、native function）                            |
| **[project-structure.md](project-structure.md)** | 项目结构（目录、CMake、Logger 接口）                                          |

---

## 核心决策摘要

以下决策分散在各 ADR 文档中，此处汇总为快速参考：

| #        | 决策                                                                                                                                 |
| -------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| ADR-0001 | 全系统按瞬态设计；稳态 = t=0 时的单次非线性迭代                                                                                      |
| ADR-0002 | Cell-centered DOF；边界条件通过面积分施加，无需面 DOF                                                                                |
| ADR-0003 | 全局 SoA（Structure of Arrays）布局                                                                                                  |
| ADR-0004 | 几何表达式与场/BC 表达式求值分离；preprocessor 编译所有表达式为 `CompiledExpression`；TBB ETS 包装保证 `eval()` 零锁，热源去重为字典 |
| ADR-0005 | Cell-level BC 存储；每个单元的每个面独立 BC，解决面投影重叠                                                                          |

> **注**：Block 仅在 XY 平面通过 add/sub Rect 定义几何，Z 维度完全由 Layer 控制（`ResolvedBlock` 无 Z 范围字段）。FaceKey 第3字段为空间坐标（CoordValue），非层索引。

### 关键原则

1. **内部模型不含原始字符串** — 所有表达式在 preprocess 阶段编译为 `CompiledExpression`
2. **热源为字典化 per-cell 索引** — `InternalModel::heat_source_table`（去重 `vector<CompiledExpression>`，索引 0 为 `make_constant(0.0)`）+ `CellFields::heat_source_idx`（`vector<uint16_t>`）
3. **无虚函数（solver 除外）** — Solver 使用虚接口（工厂模式），其余模块均使用模板静态多态
4. **无异常** — 错误通过 `mhs::logger` 记录，程序通过 `mhs::logger::panic()` 退出
5. **POD 类型优先** — 所有内部模型结构均为 POD 兼容
6. **纯函数优先** — `assembler::assemble()` 在给定 model + state 下无状态
7. **SoA 贯穿全局** — 所有热循环数组按字段连续存储
8. **表达式预编译** — exprtk 表达式编译一次，求值多次
9. **无共享可变状态** — 模块间通过 const 引用和返回值通信（`eval()` 路径除外，其线程隔离由 ETS 层保证）
10. **复杂形式用 native function** — 通过 `register_native()` 注册 C++ 函数到 expr 模块
11. **不支持 2D** — `Dimension::Dimension2D` 在预处理阶段触发 panic
12. **Lockless expr eval** — `CompiledExpression` 包装 `tbb::enumerable_thread_specific<ExprTKCompiled>`；每个 TBB 工作线程在首次 `eval()` 时懒构造自己的 ExprTK AST，无需任何 mutex 或原子操作。代价：每线程一份 AST（公式字符串按值捕获在 ETS 构造器中）
13. **TBB parallel assembly** — `tbb::parallel_for` 扫描全网格索引范围，跳过虚拟单元；`tbb::enumerable_thread_specific<ThreadLocalData>` 持有每线程 triplet 列表 + RHS 向量，并行结束后一次性合并到全局矩阵与 RHS
