# 项目结构

本文只定义源码、构建目标和命名空间的归属。模块接口在各目标头文件中自描述，运行流程见 [data-flow.md](data-flow.md)。

## 构建目标

| 目录                       | 目标               | 职责                                                        |
| -------------------------- | ------------------ | ----------------------------------------------------------- |
| `include/metahotspot/`     | 各组件的公开头文件 | 对外 API，按组件目录组织                                    |
| `src/compiler/`            | `mhs_compiler`     | 给定网格适配、几何投影、表达式编译和冻结流场构建            |
| `src/solver/`              | `mhs_solver`       | 组装、非线性迭代、线性求解、时间推进、探针和后处理          |
| `src/numerics/expression/` | `mhs_expression`   | muparser 与 TBB 表达式封装；header-only 的 `mhs::core` 契约也归它 |
| `src/numerics/linear/`     | `mhs_linear`       | AMGCL/Eigen AMG 迭代求解、求解器工厂和可选 MKL/Pardiso 封装 |
| `src/io/`                  | `mhs_io`           | tinyxml2 适配及 XML/VTU 输出                                |
| `src/logging/`             | `mhs_logging`      | spdlog 封装                                                 |
| `src/api/`                 | `mhs_c_api`        | opaque handle 与 C ABI 适配                                 |
| `apps/metahotspot/`        | `metahotspot` CLI  | 参数解析、日志初始化和顶层错误处理                          |
| `tests/`                   | `mhs_tests`        | 单元测试和模块行为验证                                      |

`mhs_c_api` 是共享库，导出别名为 `mhs::c_api`；CLI 的目标名为 `metahotspot`。其他 C++ 组件为静态库，统一入口为 INTERFACE 目标 `mhs::metahotspot`。
`mhs::core` 契约是 header-only 的，归 `mhs_expression` 目标，不单独建目标。

模块内按职责拆分 `.cpp`，但 assembler、time scheme、fluid 等实现细节不单独建库。第三方依赖或编译成本边界才构成独立目标。

头文件按可见性分家：`include/metahotspot/<组件目录>/` 下的一律是公开 API，整棵树原样安装；`src/<组件目录>/` 下的只在构建树内可见，是实现细节或模块间内部接口，不安装。
两侧的包含路径同名，因此 `#include "core/model.hpp"` 在源码树和安装树中都指向同一个公开头文件。
C 接口头文件是 `include/metahotspot/metahotspot_c_api.h`。

## 命名空间

| 命名空间          | 角色                                 |
| ----------------- | ------------------------------------ |
| `mhs`             | 品牌前缀；不定义或重导出类型         |
| `mhs::model`      | authoring model 数据类型             |
| `mhs::core`       | 运行期数据契约、表达式句柄和共享枚举 |
| `mhs::sim`        | 编译、组装、数值求解和调度           |
| `mhs::sim::fluid` | 冻结流场构建及流热装配               |
| `mhs::io`         | 输入输出适配                         |
| `mhs::post`       | 结果插值和派生量                     |
| `mhs::logger`     | 日志服务                             |

命名空间表达领域边界，不要求与目录一一对应。公共领域接口使用两层命名空间；`mhs::sim::fluid` 和 `mhs::sim::time_scheme` 保存内部实现。匿名命名空间用于单文件私有实现。

## 构建规则

- C++20，禁用编译器扩展。
- `mhs_options` 为项目目标配置严格告警；`MHS_WERROR=ON` 时视为错误（顶层默认 ON）。第三方头文件按 SYSTEM 引入，其告警不进入本项目策略。
- Pardiso 代码只能在 `MHS_ENABLE_PARDISO` 边界内出现。
- `mhs::core` 契约是 header-only 的，随 `mhs_expression` 一起被链接；`mhs::model` 类型不依赖 tinyxml2、muparser、Eigen、TBB 或 spdlog。
- `mhs::core` 的运行期类型保存编译后表达式，因此依赖 muparser/TBB；`mhs_solver` 和 `mhs_linear` 使用 Eigen，`mhs_io` 使用 tinyxml2，`mhs_logging` 使用 spdlog。
- C API 公共头只暴露 C 类型、枚举和 opaque handle；批量结果通过 copy-out API 写入调用方缓冲区。

具体选项和依赖声明以根 `CMakeLists.txt`、`cmake/Dependencies.cmake` 与各目标 `CMakeLists.txt` 为准，不在本文复制。
