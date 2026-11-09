# MetaHotspot Context

MetaHotspot 是面向电子封装多层堆叠结构的三维有限体积热仿真引擎。消费显式控制体与面片网格，采用单元中心主状态和外边界温度迹，支持稳态、瞬态，以及可选的单向流热耦合。

本文只保存跨模块工作所需的稳定上下文。数据结构、调用链和接口签名分别以对应设计文档为准，不在此重复。

## 架构边界

构建目标划分、目录归属和命名空间规则以[项目结构](docs/design/project-structure.md)为准，不在此重复。

依赖方向：

```text
IO → ModelDefinition → Compiler → Model → Solver → Solution → IO
```

`mhs::core` 是编译后的公共内核，提供容器、内存池与低级 kernel；`model` 保存输入定义和运行期领域数据。依赖为 `model → expression → core`、`linear → core`，compiler/solver 在其上工作。Eigen 仅用于内部小规模 QR 与可选 PARDISO 实现，不属于公开头或 SDK 链接依赖。

## 求解约定

- 稳态是在 `t = 0` 执行一次非线性求解；瞬态由 AdaptiveTimeStepper 控制 BDF1/2 步长、阶数与误差，各次非线性迭代组装并求解 `(K + cj C) T = f - C history`，中间输出由稠密插值生成，终点强制落步。
- 全局算子统一写为 `C * dx/dt + K * x = f`。
- `solve_system` 只负责非线性迭代、时间推进和输出时刻；它通过
  `SystemAssembler(state, time)` 请求整个系统的当前线性化，不理解 FVM、
  端口或耦合拓扑。
- Python 宏模型原型位于 `python/metahotspot/macromodel` 与 `playground/`，分别处理源端口、边界参数和可嵌入接口；其降阶误差与接口模型假设由各实验独立验证。
- 核心瞬态采用 AdaptiveTimeStepper 自适应 BDF1/2；输出时刻可由稠密插值采样，终点为真实积分状态。
- 模态瞬态宏模型必须在同一基底下提供完整的 `K_r`、`C_r`、`f_r` 和一致
  初态。静态端口柔度的 SVD 只验证稳态端口响应，不能单独证明瞬态降阶精度；
  瞬态 ROM 需要动态模态或时域快照及时间步收敛验证。
- Model FVM 详细离散始终由求解器内部组装。
- `solve_system` 要求显式完整初状态；`solve(state=)` 为空时才从
  `initial_temperature` 构建均匀温度。
- 每个外边界积分片具有温度迹；局部静态凝聚保持全局状态和算子为单元规模。
- 流体预处理只持久化热组装所需的冻结面流量和换热数据。

## 数据约定

- 内部几何一律使用 SI 米。
- 控制体、唯一有向面片与 CSR 关联构成离散网格；CellId 同时标识控制体与温度状态下标。
- `DiscreteMesh::facets` 是唯一的积分面片集合，热学与流体边界矩形在几何编译阶段共同切分外表面；`facet_thermal_condition` 按 FacetId 绑定类型化 `BoundaryCondition`。边界表达式中的 `T` 指物理表面温度迹。
- 当前 TPFA 要求轴对齐控制体的相邻中心沿面法向对齐，并采用对角导热张量；网格由调用方提供。
- 材料、边界和热源表达式在 setup 阶段编译，热循环中不解析字符串。
- Layer / Block / Rect / Boundary 的输入顺序属于模型语义，后出现的覆盖先出现的。
- 当前只支持三维网格。

完整数据流以 [data-flow.md](docs/design/data-flow.md) 为唯一事实源。

## 设计铁律

1. POD 与纯函数优先，不建立多层继承体系。
2. `mhs` 只作品牌命名空间，不重导出或定义类型。
3. 公共领域接口使用两层命名空间；内部实现包括 `mhs::compiler::fluid`；时间积分通过私有 AdaptiveTimeStepper 适配器完成。
4. 头文件禁止 `using namespace`。
5. 匿名命名空间用于单文件私有实现。
6. 外部格式字符串止于 IO；运行期模型不保存 XML 表示。
7. setup 与 solve 明确分离，临时预处理状态不进入 `Model`。

## 文档导航

- [设计文档索引](docs/design/design.md)
- [ADR](docs/adr/)
- [命令行与构建目标](README.md)
