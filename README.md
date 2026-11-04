# MetaHotspot

三维有限体积热仿真库，提供 C++ 模型与求解接口、C/Python 接口及 XML 命令行应用。
使用 CMake 构建与安装，通过 CPM 获取第三方依赖。

## C++ 构建环境要求

- Git
- CMake >= 3.26
- 任意一个可用的生成器（如果要开发，推荐 Makefile, Ninja, NMake Makefile 等可以生成 compile_commands.json 的生成器）
- 支持 C++20 标准（不用 module 特性）的编译器
- 任意 OpenMP 运行时

## 下载与构建

克隆，设立起基本 Python 开发环境（如果只开发 C++ 侧则不必建立 Python 环境）：

```sh
git clone https://github.com/catyugu/MetaHotspot.git
cd MetaHotspot
uv venv
uv sync
```

Windows 使用 MSVC 工具链，在开发者命令行中构建；Linux 使用 GCC 或 Clang；macOS 使用 AppleClang 并安装 `libomp`（`brew install libomp`）。
CPM 按固定 Git 标签自动获取依赖，无需手动下载各个 C++ 库。
MKL 默认关闭；启用 `MHS_USE_MKL` 时必须能找到其 CMake 配置，否则配置失败。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja # 推荐用 Ninja 或者 NMake Makefile 以生成 compile_commands.json 供 clangd 等 llvm 工具使用 
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `MHS_BUILD_TESTS` | 顶层 ON，子项目 OFF | 构建现有单元测试 |
| `MHS_BUILD_APPS` | 顶层 ON，子项目 OFF | 构建 `metahotspot` 命令行应用 |
| `MHS_BUILD_C_API` | ON | 构建 `mhs_c_api` 动态库，供 C/Python 调用 |
| `MHS_INSTALL` | 顶层 ON，子项目 OFF | 安装及导出 CMake 包 |
| `MHS_WERROR` | 顶层 ON，子项目 OFF | 本项目编译警告视为错误 |
| `MHS_SANITIZE` | OFF | 用 ASan/UBSan 编译并链接全部本项目目标（仅 GCC/Clang） |
| `MHS_VERBOSE` | OFF | 日志级别改为 `DEBUG` |
| `MHS_USE_MKL` | OFF | 启用可选的 MKL/Pardiso 支持 |

`mhs_metahotspot` 为 INTERFACE 目标，其余 C++ 组件为静态库，C API 为动态库。只构建 C++ 库时可关闭测试、应用和 C API。
`CPM_SOURCE_CACHE` 可指定共享缓存以复用下载；未指定时沿用仓库的 `.cache/cpm`。

## 安装与打包

```sh
cmake --install build --config Release --prefix ./build/package
cpack --config build/CPackConfig.cmake -C Release -B build
```

ZIP 包包含本库及 CPM 构建的依赖的安装产物：头文件、静态库、C API 动态库、TBB 运行库、
CMake 配置和可选命令行应用。包对应构建时的平台、架构、编译器及配置。
Windows 预构建 SDK 提供 MSVC 和 clang-cl Release 版本，随包携带 TBB 和对应的 OpenMP 运行库；目标机器需安装兼容的 Visual C++ Redistributable。对于启用 MKL 的构建，需要使用者自行处理对应 MKL 运行环境。

实验项目通过 `add_subdirectory(path/to/MetaHotspot)` 引入源码，或把安装目录加入
`CMAKE_PREFIX_PATH` 后调用 `find_package(MetaHotspot 1 CONFIG REQUIRED)`。
两种方式均使用 `target_link_libraries(your_target PRIVATE mhs::metahotspot)` 接入 C++ 库，
通过 `<model/model_definition.hpp>`、`<compiler/model_compiler.hpp>`、`<solver/solver.hpp>` 使用现有接口。
C 接口使用 `mhs::c_api` 目标与 `<metahotspot_c_api.h>`。

各 C++ 组件也可分别链接，源码接入与安装后的目标名称相同。每个目标会传递自身所需的
头文件路径及公开依赖；调用方无需手动补充 `src/` 或第三方头文件路径。
每个 C++ 组件均通过目标接口传递 C++20 要求，调用方无需重复设置。
可链接目标及其职责以[项目结构](docs/design/project-structure.md)为准，此处不重复。

`include/metahotspot/` 下的头文件都是公开 API，安装时整棵树原样复制；`src/` 下的头文件是实现细节，
不安装。`mhs::metahotspot` 是主要 C++ 组件的统一入口。

## Python 包

Python 包是对 C API 的 `ctypes` 封装，源码在 `python/metahotspot/`，构建由根目录的
`pyproject.toml`（scikit-build-core）驱动：轮子里包含本项目编译出的 C API 动态库及其运行时依赖，
`pip install` 之后即可直接使用，不需要另外构建 C++。

`MHS_CAPI_PATH` 可指定别处的 C API 动态库，覆盖随包加载的那份。
库版本只在 `include/metahotspot/metahotspot_c_api.h` 的 `MHS_VERSION_*` 宏里定义一次：
CMake 的 `project(VERSION)`、安装的 CMake 包版本、Python 包元数据和 `metahotspot.__version__`
都由它派生，不要另建版本号副本。

瞬态求解采用自适应 BDF1/2，仅需两个时间控制参数：

```python
from metahotspot import SolveOptions

solution = compiled.solve(opts=SolveOptions(
    time_tolerance=1e-4,
    max_time_step=None,
))
```

`time_tolerance` 控制时间离散误差；`max_time_step` 可设为正数（秒），用于解析窄脉冲等快速激励，
`None` 表示自动。模型的输出间隔控制结果采样，求解器独立选择步长并插值生成中间输出；
终点始终由实际求解步给出。C++ 对应 `SolveOptions` 的同名字段，C API 以 `max_time_step = 0`
表示自动。更多数值策略见 [ADR-0001](docs/adr/0001-transient-first.md)。

## 规范

- 语言：C++20
- 构建系统：CMake
- 命名空间：`mhs::core`、`mhs::expression`、`mhs::model`、`mhs::linear`、`mhs::compiler`、`mhs::solver`、`mhs::io` 和 `mhs::logger` 按模块组织。单文件私有实现使用匿名命名空间。
- 编译选项：项目目标通过 `mhs_options` 使用 `/W4` 或 `-Wall -Wextra -Wpedantic`；`MHS_WERROR=ON` 时加 `/WX` 或 `-Werror`。第三方头文件按 SYSTEM 引入，其告警不进入本项目策略。
- 通用设计模式：面向数据设计。尽可能使用 POD 和纯函数。无多层继承。
- 多线程：尽可能使用无锁数据结构。利用 tbb 进行多线程。
- 单元测试：`tests/*.cpp` 由 CMake 自动收集，统一链接到单个 `mhs_tests` 可执行文件，用例由 CTest 发现并运行；新增测试文件无需修改 `tests/CMakeLists.txt`。
- 案例回归：`python run_cases.py` 运行 `simple_steady_cases`、`nonlinear_steady_cases` 和 `simple_transient_cases` 三组案例，通过 `scripts/compare_lib.py` 与 `cases/` 中的参考输出比较。该脚本使用默认单配置构建目录 `build/apps/metahotspot/` 中的 CLI。
- 要求完整工作流：从 XML 到 XML（写出结果）。但允许将结果导出为 VTU/VTK 等格式以进行验证。
- 日志：`mhs_logging` 封装 spdlog，默认级别为 `INFO`；`MHS_VERBOSE=ON` 时使用 `DEBUG`。热循环诊断使用 `DEBUG`。
- 错误处理：模块通过标准异常报告错误；CLI 的仿真流程捕获 `std::exception`，打印失败原因并返回 1。

## 架构

构建目标、目录归属和命名空间规则见[项目结构](docs/design/project-structure.md)；本节只说明运行期数据契约。

运行期以 `DiscreteMesh`（控制体、唯一有向面片和 CSR 关联）和独立的材料、体积与边界积分数据为契约；主温度状态按 CellId 编号，每个外边界积分片具有局部温度迹。迹在局部边界方程中求解并凝聚，因此全局算子仍是单元规模。装配、流体、采样、VTU 和 Python 宏模型使用显式面片关联。`compiler/cartesian_mesh.cpp` 仅将已有坐标轴转换为这个契约；也可通过 C++ / C / Python 直接输入外部网格。网格构建时校验实体容量、面片单元引用范围和轴编号；坐标、拓扑、覆盖、材料对齐与 TPFA 正交性由调用者保证。

当前通量方法适用于正交、轴对齐控制体与对角导热张量；不生成或加密网格。接口和数据顺序见[数据流与设计原则](docs/design/data-flow.md)。

## 运行

`metahotspot` 的命令行参数解析统一由 `mhs::cli`（`apps/metahotspot/cli.hpp`）完成，
所有标志都是命名、顺序无关的；不接受位置参数。

| Flag                     | 默认值                       | 说明                                         |
| ------------------------ | ---------------------------- | -------------------------------------------- |
| `--input <file>`         | —（必填）                    | 输入 XML                                     |
| `--output-vtu <file>`    | `./output.vtu`               | VTU 输出路径                                 |
| `--output-xml <file>`    | `./output.xml`               | XML 输出路径                                 |
| `--log-file <file>`      | `metahotspot.log`            | 日志文件路径                                 |
| `--no-console-log`       | —                            | 关闭控制台日志                               |
| `--help`                 | —                            | 打印帮助并以 0 退出                          |

示例：

```bash
# 最常见的调用：指定主输入文件
metahotspot --input cases/simple_steady_cases/simple_steady_case1.xml

# 顺序无关：把 output-vtu 放最前也行
metahotspot --output-vtu /tmp/out.vtu \
            --output-xml /tmp/out.xml \
            --input   cases/simple_steady_cases/simple_steady_case1.xml

```

## 第三方依赖

- **CPM**：用于引入其余依赖项。
- **tinyxml2**：XML 解析和轻量级 DOM 操作。由 `io` 模块用于读取/写入 XML，作为静态库的私有链接依赖，安装包仍通过 `find_dependency` 查找。
- **spdlog**：结构化日志，由 `mhs_logging`（`mhs::logger`）封装。
- **muparser**：数学表达式解析和即时编译。由 `mhs_expression` 用于评估用户定义的函数、材料律、边界条件。
- **Eigen**：仅供内部小规模 QR 与可选 MKL/Pardiso 实现使用，不随 SDK 导出。基础容器是 `core::Vector`、`core::Matrix` 和 `core::CSRMatrix`；AMGCL 直接读取 CSR。
- **AMGCL**：代数多重网格预条件与 CG 求解，编入 `mhs_linear`。
- **OpenMP**：AMGCL 的线程后端。muparser 关闭其 OpenMP 后端，表达式求值由 TBB 并行调度。
- **GTest**：单元测试框架，用例由 CTest 发现并运行。
- **tbb**: 用于并行化和 CPU 资源调度。

## 编号与数据顺序

`CellId` 标识控制体及主温度未知量，`FaceId` 标识唯一有向面片。`BoundaryFace` 标识外边界积分片及其温度迹；物理边界投影不改变面片拓扑。

几何、材料、`K`、`C`、`f`、`state` 和 `state_history` 按 CellId 排列。Python 暴露 `Solution.state` 与 `Solution.state_history` 两组结果。

层从上（高 Z）到下（低 Z）排列：Layer 0 在顶部，Layer N-1 在底部。
