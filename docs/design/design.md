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

| #        | 决策                                                                                 |
| -------- | :----------------------------------------------------------------------------------- |
| ADR-0001 | 全系统按瞬态设计；稳态 = t=0 时的单次非线性迭代                                      |
| ADR-0002 | Cell-centered DOF；边界条件通过面积分施加，无需面 DOF                                |
| ADR-0003 | 全局 SoA（Structure of Arrays）布局                                                  |
| ADR-0004 | 几何表达式与场/BC 表达式求值分离；preprocessor 编译所有表达式为 `CompiledExpression` |
| ADR-0005 | Cell-level BC 存储；每个单元的每个面独立 BC，解决面投影重叠                          |

> **注**：Block 仅在 XY 平面通过 add/sub Rect 定义几何，Z 维度完全由 Layer 控制（`ResolvedBlock` 无 Z 范围字段）。FaceKey 第3字段为空间坐标（CoordValue），非层索引。

### 关键原则

1. **内部模型不含原始字符串** — 所有表达式在 preprocess 阶段编译为 `CompiledExpression`
2. **热源为 per-cell** — `CellFields.heat_source` 是 `vector<CompiledExpression>`，由 `Block.ti_reyuan_expr` 编译
4. **无虚函数（solver 除外）** — Solver 使用虚接口（工厂模式），其余模块均使用模板静态多态
4. **无异常** — 错误通过 `mhs::logger` 记录，程序通过 `mhs::logger::panic()` 退出
5. **POD 类型优先** — 所有内部模型结构均为 POD 兼容
6. **纯函数优先** — `assembler::assemble()` 在给定 model + state 下无状态
7. **SoA 贯穿全局** — 所有热循环数组按字段连续存储
8. **表达式预编译** — exprtk 表达式编译一次，求值多次
9. **无共享可变状态** — 模块间通过 const 引用和返回值通信
10. **复杂形式用 native function** — 通过 `register_native()` 注册 C++ 函数到 expr 模块
11. **不支持 2D** — `Dimension::Dimension2D` 在预处理阶段触发 panic
