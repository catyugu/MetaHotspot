# 接口设计索引

本文档是模块接口的入口。各小节给出**当前代码**对应的接口契约（不含历史）。

| 文件                                             | 内容                                                                                                     |
|--------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| **[data-flow.md](data-flow.md)**                 | 数据流与流程（总览、各阶段表、关键设计原则）                                                             |
| **[expr-api.md](expr-api.md)**                   | expr 模块接口                                                                                            |
| **[io-structure.md](io-structure.md)**           | IO 模型结构（直接映射 XML schema）                                                                       |
| **[internal-model.md](internal-model.md)**       | 内部模型结构（SoA 布局，扁平化）                                                                         |
| **[module-interfaces.md](module-interfaces.md)** | 模块接口（io、preprocessor、assembler、time_scheme、linear_solver、scheduler、nonlinear、postprocessor） |
| **[project-structure.md](project-structure.md)** | 项目结构、CMake、Logger                                                                                  |

## 决策摘要（详见 `docs/adr/`）

| ADR  | 决策                                                                               |
|------|:-----------------------------------------------------------------------------------|
| 0001 | 全系统按瞬态设计；稳态 = t=0 时的单次非线性迭代                                    |
| 0002 | Cell-centered DOF；BC 走面积分，不存面 DOF；`face_bcs` 扁平数组存储             |
| 0003 | 内部模型全部 SoA                                                                   |
| 0004 | 几何 vs 场/BC 表达式分离；TBB ETS 锁无关求值；热源索引表                           |

## 关键原则

1. 内部模型不含原始字符串 — 所有表达式预编译为 `CompiledExpression`
2. 热源索引表 — `Model::heat_source_table` + 每单元 `uint16_t` 索引
3. CLI 入口统一捕获 `std::exception`；不可恢复错误通过 `MHS_FATAL` 记录后退出
4. POD 优先；纯函数优先（`Assembler::assemble` 在 `(model, ctx)` 下无状态）
5. SoA 贯穿内部模型
6. expr 预编译，`eval()` 锁无关
7. 复杂形式用 native function — `mhs::sim::register_all_functions(symbols, ...)` 将 `ModelDefinition::Function` 写入本地 `mhs::core::SymbolTable::natives`，由 `parse(formula, symbols)` 在编译时绑定
8. **不支持 2D** — `ModelDefinition` 只描述当前实现支持的 3D 网格，不保留未生效的维度枚举。
