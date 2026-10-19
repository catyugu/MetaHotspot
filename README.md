# MetaHotspot

三维有限体积热仿真库，提供 C++ 模型与求解接口、C/Python 接口及 XML 命令行工具。
构建与安装布局参照 [modern-cpp-template](https://github.com/catyugu/modern-cpp-template)，保留现有模块和 CPM 依赖获取方式。

## 下载与构建

需要 Git、CMake 3.26+ 和支持 C++20 的编译器。Windows 使用 MSVC 或 clang-cl，
在开发者命令行中构建；Linux 使用 GCC 或 Clang；macOS 使用 AppleClang。
CPM 按固定 Git 标签自动获取依赖，无需手动下载各个 C++ 库。
muparser 默认启用 OpenMP；macOS 需安装 `libomp`（`brew install libomp`），
也可通过 `-DENABLE_OPENMP=OFF` 关闭。MKL 默认关闭，仅使用 Pardiso 时需要安装并启用。

```sh
git clone https://github.com/catyugu/MetaHotspot.git
cd MetaHotspot
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

使用 Ninja 时，在配置命令中加入 `-G Ninja`。`run_tests.py` 仍可运行默认 `build/` 下的单配置测试；
Visual Studio 等多配置生成器使用上面的 `ctest -C Release`。

| 选项 | 默认值 | 用途 |
| --- | --- | --- |
| `MHS_BUILD_TESTS` | 顶层 ON，子项目 OFF | 构建现有单元测试 |
| `MHS_BUILD_TOOLS` | 顶层 ON，子项目 OFF | 构建 `metahotspot` XML 命令行工具 |
| `MHS_BUILD_C_API` | ON | 构建 `mhs_c_api` 动态库，供 C/Python 调用 |
| `MHS_INSTALL` | 顶层 ON，子项目 OFF | 安装及导出 CMake 包 |
| `MHS_WERROR` | 顶层 ON，子项目 OFF | 本项目编译警告视为错误 |
| `USE_MKL` | OFF | 启用可选的 MKL/Pardiso 支持 |

C++ 模块保持静态库，C API 保持动态库。只构建 C++ 库时可关闭测试、工具和 C API。
`CPM_SOURCE_CACHE` 可指定共享缓存以复用下载；未指定时沿用仓库的 `.cache/cpm`。

## 安装与打包

```sh
cmake --install build --config Release --prefix ./build/package
cpack --config build/CPackConfig.cmake -C Release -B build
```

ZIP 包包含本库及 CPM 构建的依赖的安装产物：头文件、静态库、C API 动态库、TBB 运行库、
CMake 配置和可选命令行工具。包对应构建时的平台、架构、编译器及配置；
运行环境仍需相应的 C++/OpenMP 运行库。启用 MKL 的包还需要对应 MKL 运行环境。
GitHub Actions 为 Linux、macOS 和 Windows 构建、测试、安装并上传 ZIP artifact，下载时选择对应任务。

实验项目通过 `add_subdirectory(path/to/MetaHotspot)` 引入源码，或把安装目录加入
`CMAKE_PREFIX_PATH` 后调用 `find_package(MetaHotspot 1 CONFIG REQUIRED)`。
两种方式均使用 `target_link_libraries(your_target PRIVATE mhs::metahotspot)` 接入 C++ 库，
通过 `<core/model_definition.hpp>`、`<compiler/model_compiler.hpp>`、`<core/solver.hpp>` 使用现有接口。
C 接口使用 `mhs::c_api` 目标与 `<metahotspot.h>`。Python 包仍位于 `python/`，
通过 `MHS_CAPI_PATH` 指定构建或安装后的 C API 动态库。

各 C++ 组件也可分别链接，源码接入与安装后的目标名称相同。每个目标会传递自身所需的
头文件路径及公开依赖；调用方无需手动补充 `src/` 或第三方头文件路径。
仓库在顶层统一设置 C++20，单独使用组件的外部项目也应采用 C++20。

| 目标 | 组件 |
| --- | --- |
| `mhs::core` | 模型定义、离散网格和运行期数据类型 |
| `mhs::expression` | 表达式编译与求值 |
| `mhs::amg` | AMG 迭代求解器实现 |
| `mhs::linear` | 线性求解器接口与工厂、可选 Pardiso 实现 |
| `mhs::compiler` | 模型与几何编译 |
| `mhs::solver` | 装配、求解、时间推进和后处理 |
| `mhs::io` | XML 读取及 XML/VTU 输出 |
| `mhs::logging` | 日志接口 |

所有已安装的组件头文件继续开放，`mhs::metahotspot` 保留为主要 C++ 组件的统一入口。

`python ci/check_build.py` 执行构建、现有单元测试、安装和打包；
可用 `--generator`、`--config` 和 `--build-dir` 选择构建方式。

## 规范

- 语言：C++20
- 构建系统：CMake
- 命名空间：主命名空间为 `mhs`, 领域子命名空间为 `mhs::model` / `mhs::core` / `mhs::sim` / `mhs::io` / `mhs::post` / `mhs::logger`。命名空间与目录解耦。模块内部，无需暴露的实现应该用匿名空间。
- 编译选项：严格（`/W4 /WX` 或 `-Wall -Wextra -Wpedantic -Werror`），第三方库除外。
- 通用设计模式：面向数据设计。尽可能使用 POD 和纯函数。无多层继承。
- 多线程：尽可能使用无锁数据结构。利用 tbb 进行多线程。
- 单元测试：每个模块一个测试套件，不要过于零碎。
- 回归测试：应包括单元测试和案例行为验证。回归测试的结果会与 `cases/` 目录中的参考输出文件进行核对。可以使用 `pytest` 方便地进行回归测试。
- 要求完整工作流：从 XML 到 XML（写出结果）。但允许将结果导出为 VTU/VTK 等格式以进行验证。
- 日志：放弃原生流或格式化输入。应封装一个成熟的日志库（如 spdlog），以实现结构化、高性能的日志。默认级别应为 `INFO`。热循环中如需打印，则应该使用 `DEBUG`。
- 错误处理：模块可通过 `std::exception` 报告错误；CLI 入口统一捕获、记录并退出。可恢复问题应记录警告和回退值。

## 架构

### 构建目标

- **`src/core/model_definition.hpp`**：header-only authoring model 类型（`mhs::model` 命名空间），不依赖第三方库；Layer / Block / Rect / Boundary 的输入顺序是模型语义。
- **`mhs_core`**：header-only 的显式控制体、面片与输出视图契约，作为 compiler、solver 和结果 IO 的稳定依赖边界。
- **`mhs_compiler`**：消费有序 `ModelDefinition` 和给定网格，完成几何区域/热源的体积投影、边界积分投影、表达式编译和冻结流场构建。
- **`mhs_solver`**：消费运行期模型，负责热与流体组装、线性/非线性迭代、时间推进、探针和后处理。不包含宏模型逻辑。
- **`mhs_expression`**：muparser 与 TBB 封装；只在表达式实现变化时重编。
- **`mhs_linear`**：AMGCL/Eigen 迭代求解和可选 MKL/Pardiso 封装；与建模代码的增量编译隔离。
- **`mhs_io`**：tinyxml2 适配以及 XML / VTU 输出；外部 FaceKey 格式不会进入建模或引擎层。
- **`mhs_logging`**：spdlog 封装。

运行期以 `DiscreteMesh`（控制体、唯一有向面片和 CSR 关联）和独立的材料、体积与边界积分数据为契约；温度状态直接按 CellId 编号。装配、流体、采样、VTU 和 CTM 使用显式面片关联。`compiler/cartesian_mesh.cpp` 仅将已有坐标轴转换为这个契约；也可通过 C++ / C / Python 直接输入外部网格。网格有效性（坐标、拓扑、覆盖、材料对齐与 TPFA 正交性）由调用者保证，程序不做校验。

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
- **tinyxml2**：XML 解析和轻量级 DOM 操作。由 `io` 模块用于读取/写入 XML。
- **spdlog**：结构化日志，由 `mhs_logging`（`mhs::logger`）封装。
- **muparser**：数学表达式解析和即时编译。由 `mhs_expression` 用于评估用户定义的函数、材料律、边界条件。
- **Eigen**：稠密向量、稀疏矩阵及 AMGCL 的 Eigen 后端；可选提供 MKL/Pardiso 直接求解器。
- **GTest**：测试框架。每个模块一个测试套件。
- **tbb**: 用于并行化和 CPU 资源调度。

## 编号与数据顺序

`CellId` 标识控制体及对应温度未知量，`FaceId` 标识唯一有向面片。编号不表示空间位置或邻接关系；物理边界投影不改变面片拓扑。

几何、材料、`K`、`C`、`f`、`state` 和 `state_history` 统一按 CellId 排列。Python 的 `Solution.temperature` 和 `temperature_history` 直接共享状态数组。

层从上（高 Z）到下（低 Z）排列：Layer 0 在顶部，Layer N-1 在底部。
