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
│   ├── solver/                # Eigen 稀疏求解器工厂
│   ├── scheduler/            # 仿真循环调度
│   ├── postprocessor/         # VTU/XML 输出
│   ├── logger/                # spdlog 封装、全局单例、mhs::panic()
│   └── utils/                 # 通用工具函数
├── tests/
│   ├── CMakeLists.txt         # GTest 配置、测试发现
│   ├── model/                 # 模型结构测试
│   ├── expr/                  # 表达式求值测试
│   ├── preprocessor/          # 网格生成、BC 解析测试
│   ├── assembler/             # 组装测试
│   └── scheduler/             # 仿真循环集成测试
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

### 全局单例

```cpp
namespace mhs::logger {

// 全局日志单例，程序启动时初始化。
// 日志级别由 CMake VERBOSE 选项控制：
//   VERBOSE=OFF  → 默认级别 INFO
//   VERBOSE=ON   → 默认级别 DEBUG
// 用户也可在运行时通过 MHS_LOG_LEVEL env var 覆盖。
Logger& instance();

// 初始化（通常在 main() 开头调用）
void init(const std::string& log_file = "", bool console_output = true);

} // namespace mhs::logger
```

### 日志宏

```cpp
// INFO 级别日志（始终启用）
#define MHS_LOG_INFO(...) ...

// DEBUG 级别日志（VERBOSE=ON 时启用，否则为空宏）
#define MHS_LOG_DEBUG(...) ...

// ERROR 级别日志，记录后触发 panic（程序终止）
#define MHS_LOG_ERROR(...) mhs::logger::instance().panic(__VA_ARGS__)

// WARN 级别日志，记录警告并报告默认值
#define MHS_LOG_WARN(...)
```

### mhs::panic()

```cpp
namespace mhs::logger {

// 记录错误信息到日志（ERROR 级别），然后 std::exit(1)。
// 不抛出异常，不触发栈展开。
[[noreturn]] void panic(const char* fmt, auto&&... args);

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
| `mhs::preprocessor`  | 网格生成、BC 解析、表达式编译                         |
| `mhs::assembler`     | Jacobian 和 RHS 组装                                  |
| `mhs::solver`        | Eigen 稀疏求解器工厂                                  |
| `mhs::scheduler`     | 仿真循环调度                                          |
| `mhs::postprocessor` | VTU/XML 输出                                          |
| `mhs::logger`        | spdlog 封装、全局单例、mhs::panic()                   |
| `mhs::utils`         | 通用工具函数                                          |
