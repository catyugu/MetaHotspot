# 项目结构

公开头位于 `include/metahotspot/<模块>/`，整棵树随 SDK 安装；实现及内部头位于 `src/<模块>/`，不安装。两者的包含路径一致。`model` 是仅含头文件的 INTERFACE 目标，其余 C++20 组件使用静态库，C API 使用共享库；统一入口为 `mhs::metahotspot`。

## 模块与依赖

表中列出直接依赖。PUBLIC 表示公开头或接口依赖；PRIVATE 依赖仍可能通过静态库的链接闭包传递给最终程序。

| 目录 / 目标 | 命名空间与职责 | PUBLIC 依赖 | PRIVATE 依赖 |
| --- | --- | --- | --- |
| `core` / `mhs_core` | `mhs::core`：容器、内存池、CSR/vector kernels | 无 | TBB、项目编译选项 |
| `expression` / `mhs_expression` | `mhs::expression`：表达式编译与求值 | 无 | core、muparser、TBB |
| `model` / `mhs_model` | `mhs::model`：输入定义、运行期模型、网格与领域类型 | core、expression（INTERFACE） | 无 |
| `linear` / `mhs_linear` | `mhs::linear`：求解接口、AMGCL、可选 PARDISO | core | AMGCL、OpenMP、可选 Eigen/MKL |
| `compiler` / `mhs_compiler` | `mhs::compiler`：几何投影、表达式编译、冻结流场 | model | linear |
| `solver` / `mhs_solver` | `mhs::solver`：组装、非线性迭代、时间积分、解与物理场重建 | model | expression、linear、logging、Eigen、TBB、AdaptiveTimeStepper |
| `logging` / `mhs_logging` | `mhs::logger`：日志服务 | spdlog | 项目编译选项 |
| `api` / `mhs_c_api` | C ABI 与 opaque handle | 无 | compiler、solver、logging |

每个组件都有 `mhs::<组件>` 别名和同名 SDK 导出目标。流体编译代码属于 compiler 组件；AdaptiveTimeStepper 是 solver 的私有头文件依赖。依赖由各目标的 `CMakeLists.txt` 声明，安装与消费者验证覆盖各组件。

## 公共内核

`core` 不含热仿真领域类型、表达式或求解策略。只有基础数据表示、内存管理，以及由表示决定的无状态低级操作归入此层。`build_csr`、`spmv`、`add_scaled`、`add_scaled_into`、向量范数和有限值检查属于 core；AMG、迭代收敛判断和时间积分属于上层模块。

容器保留模板与轻量 inline 访问器，固定 `double` 的重 kernel 在 `.cpp` 中实现。`Vector<T>` 是采用池化分配器的 `std::vector` 别名；唯一资源实现位于 `src/core/memory.cpp`，线程安全的小块复用和大块上游分配策略保持不变。分配器保存资源指针，移动、交换、释放沿用原资源；资源及其所在动态库必须存活到容器释放之后。

`Matrix<T>` 是连续行主序矩阵。`CSRMatrix<T>` 组合 `shared_ptr<const CSRPattern>` 和独立 `Vector<T>` 数值数组，由创建矩阵的代码显式传递共享模式。复制矩阵共享模式、复制数值；模式构造后不可修改。使用零基、有符号 32 位稀疏索引；每行列号严格递增、无重复，显式零保留。公开模式数组构造检查不变量，矩阵构造检查数值长度；仅 `build_csr` 和 `add_scaled` 两个友元构造 kernel 可使用私有 canonical 构造，消除重复扫描。`build_csr` 串行分桶，小输入逐行排序并追加输出；大输入按行并行排序及原地合并，再前缀求和并按最终 nnz 分配、并行写出。显式零与每行累加顺序保持一致。`add_scaled` 构造联合模式，联合结构与输入相同时复用输入模式；`add_scaled_into` 只刷新现有联合模式或超集模式的数值，不分配存储。向量操作接受 `std::span`，SpMV 接受只读 `CSRMatrixView`。第三方类型不出现在这些公开头中。

## 数据模型与计算流程

`model::ModelDefinition` 保存声明式输入，`compiler::build_model` 产生 `model::Model`。后者持有 `DiscreteMesh`、`FluidModel`、材料、边界和编译后表达式。PTA 语法、XML 兼容、节点采样和 VTU 输出全部位于 `apps/metahotspot`，使用 `mhs::app::*` 命名空间，不进入 SDK。`model::StudyType` 在定义和运行期共用，领域表索引与 CellId/FacetId 归 model；实体索引表示与稀疏索引归 core。

求解入口为 `solver/solver.hpp`，配置为 `solver::SolveOptions`，结果为 `solver::Solution`。内部组装、物理场重建、历史状态与时间积分都归 solver。表达式模块不依赖 model，从而保持 `model → expression → core` 的单向关系。

`linear/solver.hpp` 只公开 Solver、配置、结果与工厂。`compute(CSRMatrixView)` 后允许调用方释放输入；`solve(b, x)` 以 x 为初始猜测并原地写回，直接后端忽略初值。b 和 x 不得重叠。AMGCL 直接读取 CSR；Eigen 只用于内部 QR 和可选 PARDISO，不属于 SDK 公开头或查找依赖。

`apps/metahotspot` 只在 `MHS_BUILD_APPS=ON` 时进入构建；内部静态目标 `metahotspot_frontend` 仅通过公开聚合目标 `mhs::metahotspot` 消费主库。PTA 兼容、XML/VTU 输出、节点采样及真实 XML 流体案例测试位于 `apps/metahotspot/tests`，使用私有 `metahotspot_frontend_tests`；这些测试仅在 APP 与测试同时启用时构建。`tests` 下的 `mhs_tests` 直接构造 SI 通用模型，因此 `MHS_BUILD_APPS=OFF, MHS_BUILD_TESTS=ON` 不会拉入 tinyxml2 或任何 APP 代码。
