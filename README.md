# MetaHotspot

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

运行期以 `DiscreteMesh`（控制体、唯一有向面片和 CSR 关联）和独立的材料、体积与边界积分数据为契约；温度状态直接按 CellId 编号。装配、流体、采样、VTU 和 CTM 使用显式面片关联。`compiler/cartesian_mesh.cpp` 仅将已有坐标轴转换为这个契约；也可通过 C++ / C / Python 直接输入外部网格。

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
