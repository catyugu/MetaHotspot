# MetaHotspot Context

MetaHotspot 是面向电子封装多层堆叠结构的三维有限体积热仿真引擎。当前采用结构化网格和单元中心温度自由度，支持稳态、瞬态，以及可选的单向流热耦合。

本文只保存跨模块工作所需的稳定上下文。数据结构、调用链和接口签名分别以对应设计文档为准，不在此重复。

## 架构边界

| 目标             | 职责                                          |
| ---------------- | --------------------------------------------- |
| `mhs_core`       | header-only 运行期数据契约和网格助手          |
| `mhs_compiler`   | `ModelDefinition` → 运行期 SoA 模型及冻结流场 |
| `mhs_solver`     | 算子组装、线性/非线性求解、时间推进和后处理   |
| `mhs_expression` | muparser 与 TBB 表达式封装                    |
| `mhs_linear`     | AMGCL/Eigen 迭代求解及可选 MKL/Pardiso 封装   |
| `mhs_io`         | XML 输入、XML/VTU 输出及外部格式适配          |
| `mhs_logging`    | spdlog 封装                                   |

依赖方向：

```text
IO → ModelDefinition → Compiler → Model → Solver → Solution → IO
```

`mhs::core` 不依赖 `mhs::sim`、`mhs::io`、`mhs::post` 或 `mhs::logger`。命名空间规则和目录归属见 [项目结构](docs/design/project-structure.md)。

## 求解约定

- 稳态是在 `t = 0` 执行一次非线性求解；瞬态执行 `assemble → build_system → nonlinear_solve → estimate_error`。
- 全局算子统一写为 `C * dx/dt + K * x = f`。
- `solve_system` 只负责非线性迭代、时间推进和输出时刻；它通过
  `SystemAssembler(state, time)` 请求整个系统的当前线性化，不理解 FVM、
  端口或耦合拓扑。
- 模型降阶（BCI-FANTASTIC 热源即端口）在 `playground/macromodel` 以纯
  Python 实现：热源区作为端口、边界组作为仿射 Robin 项，降阶基由功率
  输入驱动，在线用 scipy 固定步 BDF1 求解，不依赖 C++ 端口耦合。
- 瞬态只保留 `Adaptive` 与 `Fixed` 两种步进。二者都会在输出时刻和终止
  时刻截短当前步，observer 只接收真实积分状态；禁止对包含模态系数的
  全局状态做时间插值。
- 模态瞬态宏模型必须在同一基底下提供完整的 `K_r`、`C_r`、`f_r` 和一致
  初态。静态端口柔度的 SVD 只验证稳态端口响应，不能单独证明瞬态降阶精度；
  瞬态 ROM 需要动态模态或时域快照及时间步收敛验证。
- Model FVM 详细离散始终由求解器内部组装。
- `solve_system` 要求显式完整初状态；`solve(state=)` 为空时才从
  `initial_temperature` 构建均匀温度。
- 热边界作用于单元面，不引入面自由度。
- 流体预处理只持久化热组装所需的冻结面流量和换热数据。

完整数据流以 [data-flow.md](docs/design/data-flow.md) 为唯一事实源。

## 数据约定

- 内部几何一律使用 SI 米。
- 运行期使用扁平 SoA；活跃单元由 `cell_to_grid` 紧凑遍历。
- `face_bcs[cell * 6 + dir]` 保存单元面的已解析热边界。
- 材料、边界和热源表达式在 setup 阶段编译，热循环中不解析字符串。
- Layer / Block / Rect / Boundary 的输入顺序属于模型语义，后出现的覆盖先出现的。
- 当前只支持三维网格。

完整数据流以 [data-flow.md](docs/design/data-flow.md) 为唯一事实源。

## 设计铁律

1. POD 与纯函数优先，不建立多层继承体系。
2. `mhs` 只作品牌命名空间，不重导出或定义类型。
3. 公共 API 最多两层命名空间；`detail` 只保存跨文件私有实现。
4. 头文件禁止 `using namespace`。
5. 匿名命名空间用于单文件私有实现。
6. 外部格式字符串止于 IO；运行期模型不保存 XML 表示。
7. setup 与 solve 明确分离，临时预处理状态不进入 `Model`。

## 文档导航

- [设计文档索引](docs/design/design.md)
- [ADR](docs/adr/)
- [命令行与构建目标](README.md)
