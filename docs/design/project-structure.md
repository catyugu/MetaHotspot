# 项目结构

## 目录

```bash
MetaHotspot/
├── CMakeLists.txt
├── cmake/
│   ├── CPM.cmake                # CPM package manager
│   ├── Dependencies.cmake       # CPM: Eigen, spdlog, muparser, tinyxml2, oneTBB
│   ├── Utilities.cmake          # library helper、严格告警、运行库复制
│   ├── config.h.in
│   └── Deps/                    # 第三方依赖配置
│       ├── mkl.cmake
│       ├── muparser.cmake
│       ├── other.cmake
│       └── tbb.cmake
├── src/
│   ├── model/                   # mhs_model       纯建模契约与 ModelBuilder（无第三方依赖）
│   ├── compiler/                # mhs_compiler    ModelDefinition → 运行期 SoA、冻结流场
│   ├── solver/                  # mhs_solver      组装、迭代、时间推进、探针与后处理
│   ├── numerics/
│   │   ├── expression/          # mhs_expression  muparser + TBB 表达式封装
│   │   └── linear/              # mhs_linear      Eigen / MKL 线性求解封装
│   ├── io/                      # mhs_io          tinyxml2 适配、XML / VTU 写出
│   └── logging/                 # mhs_logging     spdlog 封装
├── tests/                       # GTest, 每模块一个套件
└── bin/                         # 主程序入口
```

## CMake 顶层

```cmake
cmake_minimum_required(VERSION 3.16)
project(MetaHotspot VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VERBOSE "Enable DEBUG level logging" OFF)
option(USE_MKL "Enable Intel MKL-backed Pardiso solver (default ON)" ON)

configure_file(${CMAKE_SOURCE_DIR}/cmake/config.h.in ${CMAKE_BINARY_DIR}/config.h)

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

| 宏              | 含义                                            |
| --------------- | ----------------------------------------------- |
| `MHS_LOG_DEBUG` | 记录 debug；`VERBOSE=ON` 时 logger 级别允许输出 |
| `MHS_LOG_INFO`  | 始终启用                                        |
| `MHS_LOG_WARN`  | 记录警告 + 报告回退值                           |

`spdlog::flush_on(spdlog::level::warn)` — 警告及以上自动 flush，保证 panic 前不丢日志。

## 2D 支持

**不支持。** `ModelDefinition` 只描述当前实现支持的 3D 网格，不再保留未生效的 `Dimension` 字段。

## 命名空间

| 命名空间          | 源目录                                           | 角色                                     |
| ----------------- | ------------------------------------------------ | ---------------------------------------- |
| `mhs`             | —                                                | 库品牌前缀（壳，不含类型定义）           |
| `mhs::model`      | `model/`                                         | 建模契约与顺序式 ModelBuilder            |
| `mhs::core`       | `compiler/` + `solver/` + `numerics/expression/` | 求解模型、表达式、POD 枚举、共享基础设施 |
| `mhs::utils`      | `compiler/` + `solver/`                          | 网格、采样和物理助手                     |
| `mhs::sim`        | `compiler/` + `solver/` + `numerics/linear/`     | 模型编译、组装、线性/非线性求解与调度    |
| `mhs::sim::fluid` | `compiler/` + `solver/`                          | 冻结流场构建与不改变稀疏模式的热装配增量 |
| `mhs::io`         | `io/`                                            | XML I/O、VTU 输出                        |
| `mhs::post`       | `solver/`                                        | 单元→节点插值、温度范围                  |
| `mhs::logger`     | `logging/`                                       | 独立日志服务                             |

公共 API 最多两层 `mhs::领域`；第三层 `mhs::领域::detail` 仅隐藏跨文件实现。命名空间与目录解耦。
