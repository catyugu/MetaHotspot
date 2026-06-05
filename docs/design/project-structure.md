# 项目结构

## 目录

```bash
MetaHotspot/
├── CMakeLists.txt
├── cmake/
│   ├── CompilerOptions.cmake    # /W4 /WX 或 -Wall -Wextra -Wpedantic -Werror
│   ├── Dependencies.cmake       # CPM: Eigen, spdlog, exprtk, tinyxml2, oneTBB
│   └── config.h.in
├── src/
│   ├── io/                      # mhs::io        XML 读 + VTU/XML 写
│   ├── expr/                    # mhs::expr      exprtk 封装, CompiledExpression
│   ├── common/                  # mhs, mhs::logger  types, io_model, internal_model, face_dir_tables
│   ├── preprocessor/            # mhs::preprocessor (free fns) + mhs::Preprocessor
│   ├── assembler/               # mhs::assembler TBB 并行组装
│   ├── solver/                  # mhs           Eigen 求解器工厂
│   ├── nonlinear/               # mhs::nonlinear Anderson 加速
│   ├── scheduler/               # mhs           时间 + 非线性调度
│   └── postprocessor/           # mhs           单元→节点插值
├── tests/                       # GTest, 每模块一个套件
└── bin/                         # mhs           主程序入口
```

## CMake 顶层

```cmake
cmake_minimum_required(VERSION 3.16)
project(MetaHotspot VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VERBOSE "Enable DEBUG level logging" OFF)
configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_BINARY_DIR}/config.h)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

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

| 宏                | 含义                                                        |
| ----------------- | ----------------------------------------------------------- |
| `MHS_LOG_DEBUG`   | `VERBOSE=ON` 时启用，否则空展开                             |
| `MHS_LOG_INFO`    | 始终启用                                                    |
| `MHS_LOG_WARN`    | 记录警告 + 报告回退值                                       |
| `MHS_LOG_ERROR`   | 记录后 `panic()` 退出                                       |

`spdlog::flush_on(spdlog::level::warn)` — 警告及以上自动 flush，保证 panic 前不丢日志。

## 2D 支持

**不支持。** `Dimension::Dimension2D` 在预处理阶段 `panic`。简化面 DOF 分支。

## 命名空间

| 命名空间            | 内容                                                                                            |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `mhs`               | 域类型、IO/内部模型、Preprocessor、Solver、Scheduler、Postprocessor、`face_dir_tables` 查表助手 |
| `mhs::io`           | XML/VTU 序列化                                                                                  |
| `mhs::expr`         | exprtk 封装、CompiledExpression、注册表                                                         |
| `mhs::preprocessor` | 自由函数：resolve_*, parse_face_key, …                                                          |
| `mhs::assembler`    | Assembler、LinearSystem、ThreadLocalData                                                        |
| `mhs::nonlinear`    | `solve()` 自由函数                                                                              |
| `mhs::logger`       | spdlog 封装 + `panic()`                                                                         |

`solver` / `scheduler` / `postprocessor` **没有**独立子命名空间 — 类型直接在 `mhs::`。
