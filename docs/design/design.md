# 接口设计索引

本文档是模块接口的入口。各小节给出**当前代码**对应的接口契约（不含历史）。

| 文件                                             | 内容                                                                                                     |
|--------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| **[data-flow.md](data-flow.md)**                 | 数据流与流程（总览、各阶段表、关键设计原则）                                                             |
| **[expr-api.md](expr-api.md)**                   | expr 模块接口                                                                                            |
| **[io-structure.md](io-structure.md)**                   | IO 模型结构（直接映射 XML schema）                                                                       |
| **[model.md](model.md)**       | 内部模型结构（SoA 布局，扁平化）                                                                         |
| **[module-interfaces.md](module-interfaces.md)** | 模块接口（io、preprocessor、assembler、time_scheme、linear_solver、scheduler、nonlinear、postprocessor） |
| **[project-structure.md](project-structure.md)** | 项目结构、CMake、Logger                                                                                  |

## 决策摘要（详见 `docs/adr/`）

| ADR  | 决策                                                                 |
|------|:---------------------------------------------------------------------|
| 0001 | 全系统按瞬态设计；稳态 = t=0 时的单次非线性迭代                      |
| 0002 | Cell-centered DOF；BC 走面积分，不存面 DOF（已被 0005 替代存储形式） |
| 0003 | 内部模型全部 SoA                                                     |
| 0004 | 几何 vs 场/BC 表达式分离；TBB ETS 锁无关求值；热源字典化             |
| 0005 | Cell-level BC：每单元存 6 面 BC，无面投影歧义                        |

## 关键原则

1. 内部模型不含原始字符串 — 所有表达式预编译为 `CompiledExpression`
2. 热源字典化 — `Model::heat_source_table` + 每单元 `uint16_t` 索引
3. 无虚函数（仅 `LinearSolver` 保留虚接口；旧 `TimeScheme` 虚层次已拆除，替换为纯函数 + `StepController` 策略类）
4. 无异常 — 错误走 `mhs::logger::panic()`，**程序入口 `bin/main.cpp` 的 `try/catch` 是唯一例外**（捕获 tinyxml2/muparser 抛出的 std::exception 并转 panic 退出）
5. POD 优先；纯函数优先（`Assembler::assemble` 在 `(model, ctx)` 下无状态）
6. SoA 贯穿内部模型
7. expr 预编译，`eval()` 锁无关
8. 复杂形式用 native function — `mhs::sim::register_all_functions(symbols, ...)` 将 `IOStructure::Function` 写入本地 `mhs::core::SymbolTable::natives`，由 `parse(formula, symbols)` 在编译时绑定
9. **不支持 2D** — `Dimension::Dimension2D` 在 IO 解析阶段会被赋值，但预处理阶段未实现 2D 网格构建。
