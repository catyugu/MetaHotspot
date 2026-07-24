# 设计文档索引

`CONTEXT.md` 保存跨模块稳定约束；本目录保存当前实现契约；`docs/adr/` 只记录重要决策及其理由。接口、数据流和布局分别只有一个事实源，其他文档应链接引用而不复制。

| 文件                                         | 唯一负责的内容                                |
| -------------------------------------------- | --------------------------------------------- |
| [data-flow.md](data-flow.md)                 | 端到端数据流、阶段输入输出和热循环行为        |
| [expr-api.md](expr-api.md)                   | 表达式编译、求值上下文和线程模型              |
| [io-structure.md](io-structure.md)           | `ModelDefinition` authoring model 及 XML 边界 |
| [internal-model.md](internal-model.md)       | `mhs::core::Model` 的运行期 SoA 布局          |
| [module-interfaces.md](module-interfaces.md) | 模块公开接口与调用约定                        |
| [project-structure.md](project-structure.md) | 构建目标、源码归属和命名空间规则              |

## 决策记录

| ADR                                      | 决策                                                     |
| ---------------------------------------- | -------------------------------------------------------- |
| [0001](../adr/0001-transient-first.md)   | 瞬态优先；稳态为 `t = 0` 的单次非线性求解                |
| [0002](../adr/0002-cell-centered-dof.md) | Cell-centered DOF；边界走面积分；每单元六面存储已解析 BC |
| [0003](../adr/0003-soa-layout.md)        | 运行期内部模型采用 SoA                                   |
| [0004](../adr/0004-expr-split.md)        | 几何表达式与场表达式分离；场表达式锁无关求值             |
