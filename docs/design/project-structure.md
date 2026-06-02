# 项目结构

---

## 目录布局

```bash
MetaHotspot/
├── CMakeLists.txt            # 顶层入口，定义项目名、版本、C++ 标准
├── cmake/
│   ├── Dependencies.cmake     # CPM 依赖声明（Eigen、spdlog、exprtk、tinyxml2 等）
│   └── CompilerOptions.cmake  # 严格编译选项（/W4 /WX 或 -Wall -Wextra -Wpedantic -Werror）
├── src/
│   ├── CMakeLists.txt         # 所有模块的源文件、include 目录、链接库
│   ├── model/                 # 类型、IO 模型、内部模型数据结构
│   ├── io/                    # XML 序列化/反序列化
│   ├── expr/                  # exprtk 封装、CompiledExpression、native function 注册
│   ├── preprocessor/          # 网格生成、BC 解析、表达式编译
│   ├── assembler/             # Jacobian 和 RHS 组装
│   ├── nonlinear/             # Anderson 加速非线性迭代（namespace mhs::nonlinear）
│   ├── solver/                # Eigen 稀疏求解器工厂（namespace mhs, not mhs::solver）
│   ├── scheduler/             # 仿真循环调度（namespace mhs, not mhs::scheduler）
│   ├── postprocessor/         # VTU/XML 输出（namespace mhs, not mhs::postprocessor）
│   └── logger/                # spdlog 封装、free function API、mhs::panic()
├── tests/
│   ├── CMakeLists.txt         # GTest 配置、测试发现
│   ├── test_expr.cpp          # 表达式求值测试
│   ├── test_preprocessor.cpp  # 网格生成、BC 解析测试
│   ├── test_assembler.cpp     # 组装测试
│   ├── test_scheduler.cpp     # 仿真循环集成测试
│   ├── test_postprocessor.cpp # 后处理测试
│   └── test_logger.cpp        # 日志测试
├── bin/                       # 可执行目标构建输出目录
│   └── CMakeLists.txt         # 主程序入口 target
```

---

## CMake 层次结构

```cmake
# 顶层 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MetaHotspot VERSION 1.0.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# VERBOSE=ON 时启用 DEBUG 日志，否则默认 INFO
option(VERBOSE "Enable DEBUG level logging" OFF)

include(cmake/CompilerOptions.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bin)
```

```cmake
# cmake/CompilerOptions.cmake
# MSVC
add_compile_options(/W4 /WX)
# GCC/Clang
add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# 第三方库除外（通过 target_compile_options 传递）
```

---

## Logger 接口

### Free Function API

```cpp
namespace mhs::logger {

// 初始化日志系统（仅需在程序入口调用一次）
void init(std::string_view log_file = {}, bool console_output = true);

// 手动刷新日志缓冲
void flush();

// 记录错误并退出进程（无格式化参数，由 MHS_LOG_ERROR 宏调用）
[[noreturn]] void panic();

// 模板化日志记录函数（直接转发到 spdlog）
template <typename... Args>
void debug(spdlog::format_string_t<Args...> fmt, Args&&... args);

template <typename... Args>
void info(spdlog::format_string_t<Args...> fmt, Args&&... args);

template <typename... Args>
void warn(spdlog::format_string_t<Args...> fmt, Args&&... args);

template <typename... Args>
void error(spdlog::format_string_t<Args...> fmt, Args&&... args);

} // namespace mhs::logger
```

### 日志宏

```cpp
// DEBUG 级别日志（VERBOSE=ON 时启用，否则为空宏）
#ifdef VERBOSE
#define MHS_LOG_DEBUG(...) ::mhs::logger::debug(__VA_ARGS__)
#else
#define MHS_LOG_DEBUG(...) (void)0
#endif

// INFO 级别日志（始终启用）
#define MHS_LOG_INFO(...) ::mhs::logger::info(__VA_ARGS__)

// WARN 级别日志，记录警告并报告默认值
#define MHS_LOG_WARN(...) ::mhs::logger::warn(__VA_ARGS__)

// ERROR 级别日志，记录后触发 panic（程序终止）
// 实际行为：先调用 error(fmt, args)，再调用 panic()
#define MHS_LOG_ERROR(...) (::mhs::logger::error(__VA_ARGS__), ::mhs::logger::panic())
```

### mhs::logger::panic()

```cpp
namespace mhs::logger {

// 无格式化参数。仅记录标记并调用 std::exit(1)。
// 不抛出异常，不触发栈展开。
// 由 MHS_LOG_ERROR 宏调用——宏先调用 error(fmt, args) 记录消息，再调用 panic() 退出。
[[noreturn]] void panic();

} // namespace mhs::logger
```

### 使用示例

```cpp
// 正常日志
MHS_LOG_INFO("Starting step {} of {}", step, total_steps);

// DEBUG 日志（热循环中安全使用）
MHS_LOG_DEBUG("Cell {}: k={}, Q={}", cell_idx, k, Q);

// 不可恢复错误
MHS_LOG_ERROR("Failed to parse XML at line {}: {}", line_num, what);

// 可恢复错误，明确报告回退
MHS_LOG_WARN("Material not found, using default k={}", 400.0);
```

---

## 2D 支持

**不支持 2D**。

在 IO 模型中 `Dimension::Dimension2D` 被接受，但在预处理阶段会触发 panic：

```cpp
// preprocessor/model_builder.cpp
if (io_model.dimension == Dimension::Dimension2D) {
    MHS_LOG_ERROR("Dimension2D is not supported. Only Dimension3D is implemented.");
}
```

这是刻意的简化 — 避免在面 DOF 处理（Z-/Z+ vs Y-/Y+ vs X-/X+）上写分支逻辑。

---

## 命名空间总结

| 命名空间             | 模块                                                  |
| -------------------- | ----------------------------------------------------- |
| `mhs::model`         | 类型、IO 模型、内部模型数据结构                       |
| `mhs::io`            | XML 序列化/反序列化                                   |
| `mhs::expr`          | exprtk 封装、CompiledExpression、native function 注册 |
| `mhs::preprocessor`  | 网格生成、BC 解析、表达式编译（free functions）       |
| `mhs`                | Preprocessor 类、Solver、Scheduler、Postprocessor     |
| `mhs::assembler`     | Jacobian 和 RHS 组装                                  |
| `mhs::nonlinear`     | Anderson 加速非线性迭代（free function solve()）      |
| `mhs::logger`        | spdlog 封装、free function API、mhs::panic()          |
