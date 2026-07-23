# 项目结构

本文只定义源码、构建目标和命名空间的归属。模块接口见 [module-interfaces.md](module-interfaces.md)，运行流程见 [data-flow.md](data-flow.md)。

## 构建目标

| 目录 | 目标 | 职责 |
|---|---|---|
| `src/model/` | `mhs_model` | 轻量 authoring model 与 `ModelBuilder`；无第三方依赖 |
| `src/runtime/` | `mhs_runtime` | header-only 运行期契约和网格助手 |
| `src/compiler/` | `mhs_compiler` | 几何解析、SoA 编译和冻结流场构建 |
| `src/solver/` | `mhs_solver` | 组装、迭代、时间推进、探针和后处理 |
| `src/numerics/expression/` | `mhs_expression` | muparser 与 TBB 表达式封装 |
| `src/numerics/linear/` | `mhs_linear` | Eigen / MKL 线性求解封装 |
| `src/io/` | `mhs_io` | tinyxml2 适配及 XML/VTU 输出 |
| `src/logging/` | `mhs_logging` | spdlog 封装 |
| `src/api/` | `metahotspot` C API | opaque handle 与 C ABI 适配 |
| `bin/` | `metahotspot` CLI | 参数解析、日志初始化和顶层错误处理 |
| `tests/` | `mhs_tests` | 单元测试和模块行为验证 |

模块内按职责拆分 `.cpp`，但 assembler、scheduler、fluid 等实现细节不单独建库。第三方依赖或编译成本边界才构成独立目标。

## 命名空间

| 命名空间 | 角色 |
|---|---|
| `mhs` | 品牌前缀；不定义或重导出类型 |
| `mhs::model` | authoring model 与 builder |
| `mhs::core` | 运行期数据契约、表达式句柄和共享枚举 |
| `mhs::utils` | 网格、采样和物理辅助函数 |
| `mhs::sim` | 编译、组装、数值求解和调度 |
| `mhs::sim::fluid` | 冻结流场构建及流热装配 |
| `mhs::io` | 输入输出适配 |
| `mhs::post` | 结果插值和派生量 |
| `mhs::logger` | 日志服务 |

命名空间表达领域边界，不要求与目录一一对应。公共 API 最多两层；`detail` 只用于跨文件私有实现，匿名命名空间用于单文件私有实现。

## 构建规则

- C++17，禁用编译器扩展。
- 项目源码启用严格告警并视为错误；第三方库除外。
- Pardiso 代码只能在 `MHS_ENABLE_PARDISO` 边界内出现。
- `mhs_runtime` 和 `mhs_model` 不得引入 Eigen、tinyxml2、spdlog 或 TBB 实现依赖。
- C API 公共头只暴露 C 类型、枚举、POD view 和 opaque handle。

具体选项和依赖声明以根 `CMakeLists.txt`、`cmake/Dependencies.cmake` 与各目标 `CMakeLists.txt` 为准，不在本文复制。
