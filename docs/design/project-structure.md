# 项目结构

## 目录

```bash
MetaHotspot/
├── CMakeLists.txt
├── cmake/
│   ├── CompilerOptions.cmake    # /W4 /WX 或 -Wall -Wextra -Wpedantic -Werror
│   ├── CPM.cmake                # CPM package manager
│   ├── Dependencies.cmake       # CPM: Eigen, spdlog, muparser, tinyxml2, oneTBB
│   ├── Utilities.cmake          # 辅助脚本（copy libomp.dll 等）
│   ├── config.h.in
│   └── Deps/                    # 各第三方依赖的 CPM 配置
│       ├── mkl.cmake
│       ├── muparser.cmake
│       ├── other.cmake
│       └── tbb.cmake
├── src/
│   ├── data/                    # mhs::core               数据契约（types, io_model, model）
│   ├── io/                      # mhs::io                XML 读 + VTU/XML 写
│   ├── expr/                    # mhs::core (子组织)     muparser 封装, CompiledExpression
│   ├── common/                  # mhs::logger, mhs::utils (mesh_utils)
│   ├── preprocessor/            # mhs::sim (子组织)      Preprocessor + 自由函数
│   ├── assembler/               # mhs::sim (子组织)      TBB 并行组装
│   ├── linear_solver/          # mhs::sim (子组织)      LinearSolver + 求解器实现
│   ├── nonlinear/               # mhs::sim (子组织)      Anderson 加速
│   ├── scheduler/               # mhs::sim (子组织)      时间 + 非线性调度，ProbeRecorder
│   ├── time_scheme/            # mhs::sim::time_scheme   纯函数积分器 + StepController
│   └── postprocessor/           # mhs::post (子组织)     单元→节点插值、局部采样
├── tests/                       # GTest, 每模块一个套件
└── bin/                         # 主程序入口
```

## CMake 顶层

```cmake
cmake_minimum_required(VERSION 3.16)
project(MetaHotspot VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VERBOSE "Enable DEBUG level logging" OFF)
option(USE_MKL "Enable Intel MKL-backed Pardiso solver (default ON)" ON)

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_BINARY_DIR}/config.h)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)
include(cmake/Utilities.cmake)

add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
add_subdirectory(bin)
```

## Logger

`mhs::logger` 自由函数（封装 spdlog）：

```cpp
namespace mhs::logger {
    void init(std::string_view log_file = {}, bool console_output = true);
    void flush();
    [[noreturn]] void panic();   // log + flush + std::exit(1)，不抛

    template <typename... Args>
    void debug / info / warn / error(spdlog::format_string_t<Args...> fmt, Args&&... args);
}
```

宏：

| 宏              | 含义                            |
|-----------------|---------------------------------|
| `MHS_LOG_DEBUG` | `VERBOSE=ON` 时启用，否则空展开 |
| `MHS_LOG_INFO`  | 始终启用                        |
| `MHS_LOG_WARN`  | 记录警告 + 报告回退值           |
| `MHS_FATAL`     | 记录后 `panic()` 退出           |

`spdlog::flush_on(spdlog::level::warn)` — 警告及以上自动 flush，保证 panic 前不丢日志。

## 2D 支持

**不支持。** `mhs::core::Dimension::Dimension2D` 在 IO 解析阶段会被赋值，但预处理阶段未实现 2D 网格构建，会导致路径异常。当前只支持 `Dimension3D`。

## 命名空间

| 命名空间      | 源目录                                                                  | 角色                                     |
|---------------|-------------------------------------------------------------------------|------------------------------------------|
| `mhs`         | —                                                                       | 库品牌前缀（壳，不含类型定义）           |
| `mhs::core`   | `data/` + `expr/`                                                       | 数据模型、表达式、POD 枚举、共享基础设施 |
| `mhs::utils`  | `common/mesh_utils.hpp`                                                 | 面法向查表                               |
| `mhs::sim`    | `assembler/` `linear_solver/` `scheduler/` `nonlinear/` `preprocessor/` | 数值引擎：组装、线性/非线性求解、调度    |
| `mhs::io`     | `io/`                                                                   | XML I/O、VTU 输出                        |
| `mhs::post`   | `postprocessor/*`                                                       | 单元→节点插值、局部采样辅助、导出场      |
| `mhs::logger` | `common/logger.hpp`、`common/logger.cpp`                                | 独立日志服务（不并入 core）              |

公共 API 最多两层 `mhs::领域`；第三层 `mhs::领域::detail` 仅隐藏跨文件实现。命名空间与目录解耦。
